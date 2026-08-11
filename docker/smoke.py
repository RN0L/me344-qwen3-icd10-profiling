#!/usr/bin/env python3
"""Image self-test: prove this image can see its accelerator before a real run is queued.

This is the default CMD of all three benchmark images. It touches no model, no dataset and no
bucket, so it runs in seconds and can be submitted as a throwaway Job on either GKE cluster —
which matters because RBAC on the GH200 cluster forbids `kubectl exec`, making pod logs the only
channel through which an image can be inspected at all.

The failure it exists to catch is the quiet one. An amd64 image on the ARM64 Grace host dies
loudly with `exec format error` and never reaches this file. A CUDA plugin that fails to load
does not: JAX falls back to the CPU backend, training "works", and the GPU row of the benchmark
silently measures a 72-core ARM CPU instead of a Hopper GPU. ME344_EXPECT_BACKEND is baked into
each Dockerfile so that fallback is an exit code rather than a plausible-looking number.

Set ME344_EXPECT_DEVICES on the smoke Job to also assert the device COUNT. A v5e 2x4 slice that
comes up with 4 chips instead of 8 is a valid JAX process — it just silently halves the machine
the benchmark claims to be measuring, and hardware.chips in the results file would then be a
statement about the manifest rather than about the hardware.

    kubectl create job smoke-tpu-$TEAM --image=$IMAGE_URI_TPU ...   # ME344_EXPECT_DEVICES=8

Exit status:
  0  jax.default_backend() matched ME344_EXPECT_BACKEND and the expected devices were found
  1  wrong backend, wrong device count, no devices, or JAX failed to import
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

# tpu -> jax reports "tpu"; cuda -> jax reports "gpu"; cpu -> jax reports "cpu".
BACKEND_ALIASES = {"tpu": "tpu", "cuda": "gpu", "gpu": "gpu", "cpu": "cpu"}

REPORTED_DISTRIBUTIONS = (
    "jax",
    "jaxlib",
    "libtpu",
    "jax-cuda12-pjrt",
    "jax-cuda12-plugin",
    "google-tunix",
    "qwix",
    "flax",
    "optax",
    "orbax-checkpoint",
    "transformers",
    "datasets",
    "numpy",
)


def distribution_version(name: str) -> str | None:
    """Installed version of `name`, or None when the image does not carry it."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def nvidia_smi_report() -> dict:
    """Query the driver the way docs/metrics-contract.md says the GPU row must.

    memory.peak_bytes for the GPU comes from `nvidia-smi --query-gpu=memory.used`, so if this
    call does not work inside the image the utilization and memory blocks cannot be filled and
    the run is worthless. Better to find that out from a 10-second smoke Job.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "available": False,
            "reason": "nvidia-smi not on PATH — GKE mounts the host driver at /usr/local/nvidia "
                      "and the image must add /usr/local/nvidia/bin itself",
        }

    fields = ["name", "memory.total", "memory.used", "driver_version", "compute_cap"]
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "reason": repr(exc)}

    def mib_to_bytes(cell: str) -> int | None:
        """MiB as an int, or None. nvidia-smi prints `[N/A]` for a field a driver withholds.

        int("[N/A]") would raise, and an uncaught traceback here would replace the whole report
        with a stack trace — on a cluster where the pod log is the only channel, that would cost
        an entire submit/queue/boot cycle to diagnose. Report the unparsed cell instead.
        """
        try:
            return int(cell) * 1024 * 1024
        except ValueError:
            return None

    gpus = []
    for line in completed.stdout.strip().splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(fields):
            gpus.append({"unparsed": line})
            continue
        name, total_mib, used_mib, driver_version, compute_capability = cells
        gpus.append(
            {
                "name": name,
                "memory_total_bytes": mib_to_bytes(total_mib),
                "memory_total_raw": total_mib,
                "memory_used_bytes": mib_to_bytes(used_mib),
                "memory_used_raw": used_mib,
                "driver_version": driver_version,
                "compute_capability": compute_capability,
            }
        )
    return {"available": True, "gpus": gpus}


def describe_device(device) -> dict:
    """A JSON-safe summary of one JAX device. `coords` only exists on TPU."""
    return {
        "id": device.id,
        "process_index": device.process_index,
        "platform": device.platform,
        "device_kind": device.device_kind,
        "coords": list(getattr(device, "coords", None) or []),
    }


def main() -> int:
    expected = os.environ.get("ME344_EXPECT_BACKEND", "").strip().lower()
    expect_devices_raw = os.environ.get("ME344_EXPECT_DEVICES", "").strip()

    report: dict = {
        "expect_backend": expected or None,
        "expect_devices": expect_devices_raw or None,
        "host_arch": platform.machine(),
        "host_platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: distribution_version(name) for name in REPORTED_DISTRIBUTIONS},
    }

    # Imported here, after the version block is already assembled, so that a broken plugin still
    # prints something diagnostic instead of an empty log.
    try:
        import jax
    except Exception as exc:  # noqa: BLE001 - any import failure is equally fatal and worth showing
        report["jax_import_error"] = repr(exc)
        print(json.dumps(report, indent=2, default=str), flush=True)
        print("SMOKE FAILED: could not import jax", file=sys.stderr, flush=True)
        return 1

    try:
        devices = jax.devices()
        report["jax_backend"] = jax.default_backend()
        report["device_count"] = len(devices)
        report["devices"] = [describe_device(device) for device in devices]
    except Exception as exc:  # noqa: BLE001
        report["jax_device_error"] = repr(exc)
        devices = []
        report["jax_backend"] = None
        report["device_count"] = 0

    # Same call the TPU/GPU rows use for memory.peak_bytes; returns None on the CPU backend.
    try:
        report["memory_stats"] = jax.local_devices()[0].memory_stats() if devices else None
    except Exception as exc:  # noqa: BLE001
        report["memory_stats"] = {"error": repr(exc)}

    if expected == "cuda" or report.get("jax_backend") == "gpu":
        report["nvidia_smi"] = nvidia_smi_report()

    print(json.dumps(report, indent=2, default=str), flush=True)

    ok = True
    if not devices:
        print("FAIL: jax.devices() returned nothing", file=sys.stderr, flush=True)
        ok = False
    if expected:
        wanted = BACKEND_ALIASES.get(expected)
        if wanted is None:
            print(
                f"FAIL: ME344_EXPECT_BACKEND={expected!r} is not one of {sorted(BACKEND_ALIASES)}",
                file=sys.stderr,
                flush=True,
            )
            ok = False
        elif report.get("jax_backend") != wanted:
            print(
                f"FAIL: expected JAX backend {wanted!r} (ME344_EXPECT_BACKEND={expected!r}) but "
                f"got {report.get('jax_backend')!r} — the accelerator plugin did not load and "
                f"this image would have benchmarked the wrong hardware",
                file=sys.stderr,
                flush=True,
            )
            ok = False

    if expect_devices_raw:
        try:
            want_devices = int(expect_devices_raw)
        except ValueError:
            print(
                f"FAIL: ME344_EXPECT_DEVICES={expect_devices_raw!r} is not an integer",
                file=sys.stderr,
                flush=True,
            )
            ok = False
        else:
            if report.get("device_count") != want_devices:
                print(
                    f"FAIL: expected {want_devices} device(s) but JAX sees "
                    f"{report.get('device_count')} — a partially attached slice would report "
                    f"hardware.chips from the manifest while running on fewer chips",
                    file=sys.stderr,
                    flush=True,
                )
                ok = False

    print("SMOKE OK" if ok else "SMOKE FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
