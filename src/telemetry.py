#!/usr/bin/env python3
"""Telemetry for the ME344 final project: Qwen3-4B LoRA across CPU / GPU / TPU.

Every benchmark run — bare-metal CPU node, GH200 GKE pod, TPU v5e Kueue Job — imports this
module and emits exactly one JSON file matching ``docs/metrics-contract.md``. Charts, tables
and the report read those files and nothing else, so this module is the single point where a
number can enter the project.

Design rules baked in here:

* **Nothing is invented.** Where a backend cannot report a quantity (JAX exposes no TPU core
  utilization counter, for example) the field is ``null`` and a sentence explaining why lands
  in ``notes``. There is no code path that guesses a plausible number.
* **Nothing is silently dropped.** ``phases.other_s`` is the residual between the attributed
  phases and ``total_wall_s``, so the phase breakdown always sums to the wall clock and
  ``emit`` refuses to write a record where it does not.
* **Nothing is rounded on the way in.** Rounding is the chart layer's job.
* **Telemetry never kills a run.** The utilization sampler swallows every probe failure and
  records the error instead. The only function that raises by design is ``emit``, and only on a
  contract violation — it validates *before* touching the filesystem, so a partial record can
  never exist. A failure of the file write itself is reported in ``notes`` and the record still
  goes out over the log channel, because losing a finished benchmark to a read-only mount is
  the worse outcome.
* **Capacities are the container's, not the host's.** ``memory.capacity_bytes`` and
  ``hardware.chips`` come from the cgroup limit and CPU affinity, because ``/proc/meminfo`` and
  ``os.cpu_count()`` both report the whole machine from inside a pod — and ``memory.peak_pct``
  is what places the OOM boundary on the sweep chart.

Dependencies: stdlib, plus ``psutil`` when it happens to be installed (there are working
stdlib fallbacks for every psutil call, because the Tunix TPU image does not ship it). ``jax``
is imported lazily inside the TPU branches so this module stays importable on the CPU node and
on a laptop.

Two output channels are supported, because the GPU cluster's RBAC persona cannot exec into
pods and **pod logs are the only way data leaves that container**:

    results/<run_id>.json        <- emit(path, **blocks)
    stdout, sentinel-delimited   <- automatic, harvested with extract_records(kubectl_logs)

Typical use::

    import telemetry as tel

    timer = tel.PhaseTimer(backend="tpu")
    sampler = tel.UtilizationSampler("tpu", interval_s=1.0)
    rec = tel.StepRecorder(batch_size=8, seq_len=512, warmup_steps=10)

    with timer.phase("model_load_s"):
        model = load_model()
    sampler.start()
    try:
        with timer.phase("steady_state_s"):
            for batch in batches:
                with rec.step() as s:
                    s.loss = train_step(batch)       # coerced (and thus synced) on exit
        with timer.phase("checkpoint_write_s"):
            save_adapter()
        status, err = "ok", None
    except Exception as exc:                          # OOM boundary lands in the sweep chart
        status, err = tel.classify_exception(exc), exc
    finally:
        sampler.stop()
        timer.stop()

    tel.emit(
        "results/tpu-v5e8-bs8-seq512.json",
        run_id="tpu-v5e8-bs8-seq512", backend="tpu",
        hardware=tel.hardware_block("tpu", label="TPU v5e 2x4"),
        config={...},
        phases=timer.snapshot(),
        steps=rec.steps,
        steady=rec.steady() if status == "ok" else None,
        memory=tel.peak_memory("tpu", sampler=sampler) if status == "ok" else None,
        utilization=sampler.block(),
        status=status, error=err,
        notes=timer.notes + rec.notes + sampler.notes,
    )

Run ``python3 src/telemetry.py`` to execute the self-test.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

try:  # POSIX only; present on every machine this project runs on, guarded anyway.
    import resource as _resource
except ImportError:  # pragma: no cover - Windows
    _resource = None

try:
    import psutil as _psutil
except ImportError:  # the Tunix TPU image does not ship psutil; stdlib fallbacks below
    _psutil = None

__all__ = [
    "PhaseTimer",
    "StepRecorder",
    "UtilizationSampler",
    "peak_memory",
    "emit",
    "build_record",
    "validate_record",
    "merge_external_phases",
    "hardware_block",
    "make_run_id",
    "classify_exception",
    "log_record",
    "extract_records",
    "utc_timestamp",
    "ContractError",
    "BACKENDS",
    "LOG_BEGIN",
    "LOG_END",
]

# --------------------------------------------------------------------------------------
# Contract constants — these mirror docs/metrics-contract.md exactly. Changing one of these
# is a contract change and must be made in the markdown first.
# --------------------------------------------------------------------------------------

BACKENDS = ("cpu", "gpu", "tpu")
STATUSES = ("ok", "oom", "error", "timeout")

#: Phases measured outside the training process (by the launcher, from kubectl timestamps).
EXTERNAL_PHASES = ("submit_to_running_s", "image_pull_s")
#: Phases measured inside the training process by :class:`PhaseTimer`.
IN_PROCESS_PHASES = ("model_load_s", "compile_s", "steady_state_s", "checkpoint_write_s")
#: Every phase key except ``total_wall_s``. ``other_s`` is always the residual.
PHASE_KEYS = EXTERNAL_PHASES + IN_PROCESS_PHASES + ("other_s",)

REQUIRED_TOP_LEVEL = (
    "run_id", "backend", "timestamp_utc", "hardware", "config", "phases",
    "steps", "steady", "memory", "utilization", "eval", "status", "error", "notes",
)
REQUIRED_HARDWARE = ("label", "chips", "chip_model", "host_arch", "hbm_per_chip_bytes")
REQUIRED_CONFIG = (
    "model", "batch_size", "seq_len", "lora_rank", "lora_alpha",
    "dtype", "max_steps", "remat", "file_cache_capacity",
)
REQUIRED_PHASES = PHASE_KEYS + ("total_wall_s",)
REQUIRED_STEP = ("step", "wall_s", "loss")
REQUIRED_STEADY = (
    "median_step_s", "p10_step_s", "p90_step_s",
    "tokens_per_s", "docs_per_s", "warmup_steps_excluded",
)
REQUIRED_MEMORY = ("peak_bytes", "capacity_bytes", "peak_pct", "source")
REQUIRED_UTILIZATION = ("mean_pct", "max_pct", "sample_interval_s", "n_samples", "source")
REQUIRED_EVAL = ("micro_f1", "n_examples", "note")

#: Sentinels that make a record recoverable from ``kubectl logs`` on the GPU cluster, where
#: the RBAC persona cannot exec into the pod or read its filesystem.
LOG_BEGIN = "===TELEMETRY-JSON-BEGIN==="
LOG_END = "===TELEMETRY-JSON-END==="

_MIB = 1024 * 1024
_PHASE_SUM_ABS_TOL_S = 1e-3  # plus a 1e-6 relative term; see _check_phase_sum

#: A record that quotes its own sentinels inside a string field would truncate itself when the
#: launcher harvests it from the log — the non-greedy match would end at the quoted marker and
#: the block would fail to parse. Any occurrence in text is neutralised on the way in.
_SENTINEL_SUBSTITUTIONS = {
    LOG_BEGIN: "[telemetry BEGIN marker removed from text]",
    LOG_END: "[telemetry END marker removed from text]",
}


def _defang_sentinels(text: str) -> str:
    for marker, replacement in _SENTINEL_SUBSTITUTIONS.items():
        if marker in text:
            text = text.replace(marker, replacement)
    return text


class ContractError(ValueError):
    """Raised when a record does not satisfy docs/metrics-contract.md.

    Always raised *before* anything is written, so a rejected record never leaves a partial
    file behind.
    """


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def utc_timestamp(when: datetime | None = None) -> str:
    """Second-precision UTC timestamp in the exact shape the contract shows."""
    when = when or datetime.now(timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(backend: str, hw_tag: str, batch_size: int, seq_len: int,
                variant: str | None = None) -> str:
    """``<backend>-<hardware>-bs<batch>-seq<seqlen>[-<variant>]`` per the contract."""
    if backend not in BACKENDS:
        raise ContractError(f"backend must be one of {BACKENDS}, got {backend!r}")
    rid = f"{backend}-{hw_tag}-bs{int(batch_size)}-seq{int(seq_len)}"
    if variant:
        rid = f"{rid}-{variant}"
    return rid


def _percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (identical to numpy.percentile's default method).

    Implemented here rather than imported so this module keeps its stdlib-only footprint.
    """
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _to_float(value: Any) -> float | None:
    """Coerce a scalar to a Python float.

    Handles numpy scalars and 0-d JAX arrays via ``.item()``. Note that calling ``.item()`` on
    a JAX array **blocks until the value is materialised** — that is deliberate and is what
    makes per-step wall times honest under JAX's async dispatch.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_float(item())
        except Exception:
            pass
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _run_cmd(cmd: Sequence[str], timeout_s: float = 10.0) -> str:
    """Run a command and return stdout. Raises on non-zero exit or missing binary."""
    proc = subprocess.run(
        list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout_s, check=True,
    )
    return proc.stdout


def _host_memory_bytes() -> int | None:
    """Host DRAM capacity, from whichever source this machine actually offers."""
    if _psutil is not None:
        try:
            return int(_psutil.virtual_memory().total)
        except Exception:
            pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        return int(_run_cmd(["sysctl", "-n", "hw.memsize"], timeout_s=5.0).strip())
    except Exception:
        return None


def _cgroup_memory_limit_bytes() -> int | None:
    """The container's memory ceiling, or ``None`` when there is no finite one.

    Every backend in this project runs inside a container with an explicit limit —
    ``podman run --memory=28g`` on the CPU node, ``resources.limits.memory`` in the Kubernetes
    manifests. ``/proc/meminfo`` and ``psutil`` both report the *host's* DRAM inside that
    container, so using them for ``memory.capacity_bytes`` would compute ``memory.peak_pct``
    against the wrong denominator and put the OOM boundary in the wrong place on the sweep
    chart. cgroup v2 is checked first, then v1.
    """
    for name, unlimited_token in (
        ("/sys/fs/cgroup/memory.max", "max"),                     # cgroup v2 (unified)
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None),    # cgroup v1
    ):
        try:
            with open(name, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if not raw or (unlimited_token is not None and raw == unlimited_token):
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 spells "unlimited" as a near-2^63 sentinel rather than a token.
        if 0 < value < (1 << 62):
            return value
    return None


def _total_system_memory_bytes() -> int | None:
    """Memory this process may actually use: the cgroup limit if there is one, else host DRAM."""
    host = _host_memory_bytes()
    limit = _cgroup_memory_limit_bytes()
    if limit is None:
        return host
    if host is None:
        return limit
    return min(limit, host)


def _cgroup_cpu_quota() -> int | None:
    """Whole CPUs the cgroup CFS quota allows, or ``None`` when the quota is unlimited."""
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as fh:
            parts = fh.read().split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = float(parts[0]), float(parts[1])
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r", encoding="utf-8") as fh:
            quota = float(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r", encoding="utf-8") as fh:
            period = float(fh.read().strip())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def _effective_cpu_count() -> int | None:
    """CPUs this process may actually use.

    ``os.cpu_count()`` reports the *host's* logical cores even inside a container pinned with
    ``--cpuset-cpus 0-7`` or capped by ``resources.limits.cpu: 8``. That number becomes
    ``hardware.chips`` and the ``x86-<n>c`` slug of the run id, so it has to be the count the
    workload was actually given: the tightest of CPU affinity, cgroup quota and host cores.
    """
    counts: list[int] = []
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            counts.append(len(getaffinity(0)))
        except OSError:
            pass
    host = os.cpu_count()
    if host:
        counts.append(int(host))
    quota = _cgroup_cpu_quota()
    if quota is not None:
        counts.append(quota)
    return min(counts) if counts else None


def _peak_rss_bytes() -> int | None:
    """Peak resident set size of this process and its reaped children.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS/BSD — the one portability trap in
    this function.
    """
    if _resource is None:
        return None
    scale = 1 if sys.platform == "darwin" else 1024
    try:
        peak = max(
            _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss,
            _resource.getrusage(_resource.RUSAGE_CHILDREN).ru_maxrss,
        )
    except (OSError, ValueError):
        return None
    return int(peak) * scale


def classify_exception(exc: BaseException | None) -> str:
    """Map an exception to a contract ``status``: ``ok`` / ``oom`` / ``timeout`` / ``error``.

    Recognises the three shapes an out-of-memory failure actually takes in this project:
    Python's ``MemoryError``, XLA's ``RESOURCE_EXHAUSTED`` (TPU and GPU under JAX), and
    CUDA's ``out of memory`` (PyTorch on the GH200).
    """
    if exc is None:
        return "ok"
    if isinstance(exc, MemoryError):
        return "oom"
    name = type(exc).__name__
    text = f"{name}: {exc}".lower()
    oom_pat = (
        r"resource[_ ]exhausted|out of memory|out-of-memory|\boom\b|bad_alloc|"
        r"outofmemory|failed to allocate|allocation .*failed|"
        r"ran out of (memory|hbm)|memory allocator"
    )
    if re.search(oom_pat, text):
        return "oom"
    if isinstance(exc, TimeoutError) or re.search(r"deadline exceeded|timed out|timeout", text):
        return "timeout"
    return "error"


# --------------------------------------------------------------------------------------
# 1. PhaseTimer
# --------------------------------------------------------------------------------------

class PhaseTimer:
    """Wall-clock phase accounting whose parts always sum to the whole.

    In-process phases (``model_load_s``, ``compile_s``, ``steady_state_s``,
    ``checkpoint_write_s``) are measured with ``with timer.phase(name):`` blocks and
    accumulate across repeated entries — a checkpoint written five times contributes five
    intervals to ``checkpoint_write_s``.

    External phases (``submit_to_running_s``, ``image_pull_s``) happen before this process
    exists. They come from three places, in priority order: :meth:`set_external`, the
    environment variables ``TELEMETRY_SUBMIT_TO_RUNNING_S`` / ``TELEMETRY_IMAGE_PULL_S``, or
    (``backend="cpu"`` only) an implicit ``0.0``, because the bare-metal node has neither a
    scheduler queue nor an image pull. Otherwise they stay ``null`` and the launcher fills
    them in afterwards with :func:`merge_external_phases`.

    ``total_wall_s`` is the known external time plus this process's own wall clock, and
    ``other_s`` is whatever that total minus the attributed phases leaves over. The residual
    is reported, never dropped: unattributed time is exactly the thing the bottleneck
    analysis is looking for.

    Nesting phases is rejected — overlapping intervals would double-count and quietly break
    the sum invariant.
    """

    _EXTERNAL_ENV = {
        "submit_to_running_s": "TELEMETRY_SUBMIT_TO_RUNNING_S",
        "image_pull_s": "TELEMETRY_IMAGE_PULL_S",
    }

    def __init__(
        self,
        backend: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if backend is not None and backend not in BACKENDS:
            raise ContractError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        self._clock = clock
        self._t0 = clock()
        self._t_end: float | None = None
        self._acc: dict[str, float] = {k: 0.0 for k in IN_PROCESS_PHASES}
        self._external: dict[str, float | None] = {k: None for k in EXTERNAL_PHASES}
        self._active: tuple[str, float] | None = None
        self._notes: list[str] = []
        self._lock = threading.Lock()

        env = os.environ if env is None else env
        for key, var in self._EXTERNAL_ENV.items():
            raw = env.get(var)
            if raw is None or raw == "":
                continue
            try:
                self._external[key] = float(raw)
            except ValueError:
                self._note(f"ignored {var}={raw!r}: not a float")
        # The bare-metal CPU row is started by `podman run` on a node with no scheduler and a
        # locally built image, so both external phases really are zero there. Under Kubernetes
        # they are not — even the CPU manifest goes through a scheduler and an image pull — so
        # the zeroing is conditional on not being in a pod. Claiming 0.0 in a pod would be a
        # measurement this process never made.
        if backend == "cpu" and not env.get("KUBERNETES_SERVICE_HOST"):
            for key in EXTERNAL_PHASES:
                if self._external[key] is None:
                    self._external[key] = 0.0
            self._note(
                "phases.submit_to_running_s and phases.image_pull_s are 0.0 on the bare-metal "
                "CPU backend: the container is started directly on the node with `podman run` "
                "from a locally built image, so there is no scheduler queue and no registry "
                "pull. Both are measured, not assumed, on the GPU and TPU rows."
            )

    # -- notes ---------------------------------------------------------------------------
    @property
    def notes(self) -> list[str]:
        """Human-readable caveats about this timer's numbers, for the record's ``notes``."""
        out = list(self._notes)
        for key in EXTERNAL_PHASES:
            if self._external[key] is None:
                out.append(
                    f"phases.{key} is null: it is measured by the launcher from kubectl "
                    f"timestamps, not from inside the pod. Merge it with "
                    f"telemetry.merge_external_phases()."
                )
        return out

    def _note(self, msg: str) -> None:
        if msg not in self._notes:
            self._notes.append(msg)

    # -- measurement ---------------------------------------------------------------------
    @contextmanager
    def phase(self, name: str) -> Iterator["PhaseTimer"]:
        """Time a block and accumulate it into ``name``.

        Time is attributed even if the block raises, so the phases of a run that OOMs during
        model load still show where it got to.
        """
        if name not in IN_PROCESS_PHASES:
            raise ContractError(
                f"unknown phase {name!r}; in-process phases are {IN_PROCESS_PHASES}. "
                "Unattributed time belongs in other_s, which is computed, not set."
            )
        with self._lock:
            if self._active is not None:
                raise RuntimeError(
                    f"cannot start phase {name!r} inside active phase {self._active[0]!r}: "
                    "overlapping phases would double-count and break the sum invariant"
                )
            self._active = (name, self._clock())
        try:
            yield self
        finally:
            with self._lock:
                active = self._active
                if active is not None:
                    # Never unpack unconditionally: if the block that is unwinding already
                    # cleared the slot, a TypeError here would replace the real exception with
                    # a telemetry one, and telemetry must never be what kills a run.
                    self._acc[active[0]] += self._clock() - active[1]
                    self._active = None

    def add(self, name: str, seconds: float) -> None:
        """Add pre-measured seconds to an in-process phase (e.g. a library-reported time)."""
        if name not in IN_PROCESS_PHASES:
            raise ContractError(f"unknown phase {name!r}; expected one of {IN_PROCESS_PHASES}")
        value = _to_float(seconds)
        if value is None:
            raise ContractError(f"phase {name!r} needs a finite number, got {seconds!r}")
        with self._lock:
            self._acc[name] += value

    def set_external(self, name: str, seconds: float | None) -> None:
        """Set a phase measured outside this process (``submit_to_running_s``/``image_pull_s``)."""
        if name not in EXTERNAL_PHASES:
            raise ContractError(f"unknown external phase {name!r}; expected {EXTERNAL_PHASES}")
        with self._lock:
            self._external[name] = None if seconds is None else _to_float(seconds)

    def stop(self) -> float:
        """Freeze the total wall clock. Idempotent; returns in-process elapsed seconds."""
        with self._lock:
            if self._t_end is None:
                self._t_end = self._clock()
            return self._t_end - self._t0

    @property
    def elapsed_s(self) -> float:
        """In-process wall seconds so far (or at :meth:`stop` if it has been called)."""
        end = self._t_end if self._t_end is not None else self._clock()
        return end - self._t0

    def snapshot(self) -> dict[str, float | None]:
        """Build the contract's ``phases`` block. Safe to call mid-run and repeatedly."""
        with self._lock:
            end = self._t_end if self._t_end is not None else self._clock()
            in_process_wall = end - self._t0
            acc = dict(self._acc)
            external = dict(self._external)
            active = self._active
        if active is not None:
            # A snapshot taken from inside a phase (or from an except/finally block whose
            # phase context has not unwound yet) still attributes the open interval. The
            # max() covers snapshot-after-stop, where the frozen end predates the phase.
            acc[active[0]] += max(0.0, end - active[1])

        external_sum = sum(v for v in external.values() if v is not None)
        total_wall_s = external_sum + in_process_wall
        attributed = external_sum + sum(acc.values())
        other = total_wall_s - attributed
        if -1e-6 < other < 0.0:
            other = 0.0  # float noise, not a real negative
        elif other < 0.0:
            self._note(
                f"phases.other_s is negative ({other!r}s): attributed phases exceed the "
                "wall clock, which means a phase was added twice or an external phase is "
                "too large. The number is reported as measured rather than clamped."
            )

        block: dict[str, float | None] = {}
        for key in EXTERNAL_PHASES:
            block[key] = external[key]
        for key in IN_PROCESS_PHASES:
            block[key] = acc[key]
        block["other_s"] = other
        block["total_wall_s"] = total_wall_s
        return block


# --------------------------------------------------------------------------------------
# 2. StepRecorder
# --------------------------------------------------------------------------------------

@dataclass
class _StepScope:
    """Mutable handle yielded by :meth:`StepRecorder.step`. Set ``.loss`` inside the block."""
    step: int
    loss: Any = None


class StepRecorder:
    """Per-step wall times and losses, plus the contract's ``steady`` block.

    ``steps[0]`` is the XLA/torch.compile step and is always an outlier; ``steady`` therefore
    drops the first ``warmup_steps`` entries (default 10, per the contract's example) before
    computing median / p10 / p90. Throughput is derived from the *median* step:
    ``tokens_per_s = batch_size * seq_len / median_step_s`` and ``docs_per_s =
    batch_size / median_step_s``.

    Under JAX, dispatch is async: stopping a timer right after ``train_step(...)`` returns
    would measure enqueue latency rather than compute. Two mechanisms prevent that. The
    ``sync_fn`` constructor argument is invoked inside the timed region if given (pass
    ``lambda: jax.block_until_ready(state)``), and assigning a device scalar to ``scope.loss``
    forces a materialisation on exit — also inside the timed region.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        warmup_steps: int = 10,
        clock: Callable[[], float] = time.monotonic,
        sync_fn: Callable[[], Any] | None = None,
    ) -> None:
        if batch_size <= 0 or seq_len <= 0:
            raise ContractError("batch_size and seq_len must be positive")
        if warmup_steps < 0:
            raise ContractError("warmup_steps must be >= 0")
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.warmup_steps = int(warmup_steps)
        self._clock = clock
        self._sync_fn = sync_fn
        self._steps: list[dict[str, Any]] = []
        self._notes: list[str] = []
        self._lock = threading.Lock()

    # -- recording -----------------------------------------------------------------------
    def record(self, wall_s: float, loss: Any = None, step: int | None = None) -> dict[str, Any]:
        """Record one step. ``step`` defaults to a 1-based counter."""
        value = _to_float(wall_s)
        if value is None:
            raise ContractError(f"step wall_s must be a finite number, got {wall_s!r}")
        with self._lock:
            index = len(self._steps) + 1 if step is None else int(step)
            entry = {"step": index, "wall_s": value, "loss": _to_float(loss)}
            self._steps.append(entry)
        if loss is not None and entry["loss"] is None:
            self._note(
                f"step {index}: loss was not finite (or not a number) and is recorded as "
                "null rather than as a fabricated value."
            )
        return entry

    @contextmanager
    def step(self, step: int | None = None) -> Iterator[_StepScope]:
        """Time one training step. Assign the loss to the yielded scope inside the block.

        A step that raises is **not** recorded — a half-executed step has no meaningful wall
        time, and the OOM record should show the steps that actually completed.
        """
        index = (len(self._steps) + 1) if step is None else int(step)
        scope = _StepScope(step=index)
        start = self._clock()
        yield scope
        # Both of these block until the device has actually finished the step, and both run
        # before the clock stops, which is the whole point.
        if self._sync_fn is not None:
            try:
                self._sync_fn()
            except Exception as exc:  # a broken sync must not kill training
                self._note(f"sync_fn raised {type(exc).__name__}: {exc}; step times may be "
                           "measured against async dispatch rather than completion")
        loss = _to_float(scope.loss)
        elapsed = self._clock() - start
        with self._lock:
            self._steps.append({"step": index, "wall_s": elapsed, "loss": loss})
        if scope.loss is not None and loss is None:
            self._note(
                f"step {index}: loss was not finite (or not a number) and is recorded as "
                "null rather than as a fabricated value."
            )

    # -- results -------------------------------------------------------------------------
    @property
    def steps(self) -> list[dict[str, Any]]:
        """The contract's ``steps`` array (a copy; safe to mutate)."""
        with self._lock:
            return [dict(s) for s in self._steps]

    @property
    def n_steps(self) -> int:
        with self._lock:
            return len(self._steps)

    @property
    def total_step_wall_s(self) -> float:
        with self._lock:
            return sum(s["wall_s"] for s in self._steps)

    @property
    def notes(self) -> list[str]:
        return list(self._notes)

    def _note(self, msg: str) -> None:
        if msg not in self._notes:
            self._notes.append(msg)

    def steady(self) -> dict[str, Any] | None:
        """The contract's ``steady`` block, or ``None`` if no step ever completed.

        If the run produced fewer steps than ``warmup_steps``, the number actually excluded
        is reduced (down to 1, so the compile step is still dropped, or 0 if only one step
        exists) and reported truthfully in ``warmup_steps_excluded``, with a note. The field
        always describes what was really done.
        """
        with self._lock:
            steps = list(self._steps)
        n = len(steps)
        if n == 0:
            self._note("steady is null: no training step completed.")
            return None

        excluded = self.warmup_steps
        if excluded >= n:
            excluded = 1 if n > 1 else 0
            self._note(
                f"only {n} step(s) completed, fewer than the {self.warmup_steps} warmup "
                f"steps normally excluded; steady is computed over the last {n - excluded} "
                f"step(s) with warmup_steps_excluded={excluded}. Treat it as indicative "
                "only — it is dominated by compile and cache-warming effects."
            )

        walls = [s["wall_s"] for s in steps[excluded:]]
        median = _percentile(walls, 50)
        p10 = _percentile(walls, 10)
        p90 = _percentile(walls, 90)

        if median and median > 0.0:
            tokens_per_s: float | None = (self.batch_size * self.seq_len) / median
            docs_per_s: float | None = self.batch_size / median
        else:
            tokens_per_s = docs_per_s = None
            self._note(
                "steady.tokens_per_s and steady.docs_per_s are null: the median step time "
                "was zero or unmeasurable, so throughput cannot be derived."
            )

        return {
            "median_step_s": median,
            "p10_step_s": p10,
            "p90_step_s": p90,
            "tokens_per_s": tokens_per_s,
            "docs_per_s": docs_per_s,
            "warmup_steps_excluded": excluded,
        }


# --------------------------------------------------------------------------------------
# 3. UtilizationSampler
# --------------------------------------------------------------------------------------

@dataclass
class _Sample:
    t: float
    util_pct: float | None = None
    mem_bytes: float | None = None
    capacity_bytes: float | None = None


class _Probe:
    """Base class for a backend's one-shot utilization/memory read."""

    source = "unknown"

    def sample(self) -> _Sample:  # pragma: no cover - abstract
        raise NotImplementedError


class _GpuProbe(_Probe):
    """``nvidia-smi`` one-shot query. Memory comes back in MiB thanks to ``nounits``."""

    source = ("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
              "--format=csv,noheader,nounits")

    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = int(gpu_index)

    @staticmethod
    def _num(field_text: str) -> float | None:
        text = field_text.strip()
        if not text or text.upper().strip("[]") in ("N/A", "NOT SUPPORTED", "UNKNOWN"):
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def sample(self) -> _Sample:
        out = _run_cmd([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        rows = [r for r in out.splitlines() if r.strip()]
        if not rows:
            raise RuntimeError("nvidia-smi returned no rows")
        if self.gpu_index >= len(rows):
            raise RuntimeError(f"gpu_index {self.gpu_index} out of range ({len(rows)} GPUs)")
        parts = rows[self.gpu_index].split(",")
        if len(parts) < 3:
            raise RuntimeError(f"unparseable nvidia-smi row: {rows[self.gpu_index]!r}")
        util = self._num(parts[0])
        used = self._num(parts[1])
        total = self._num(parts[2])
        return _Sample(
            t=time.time(),
            util_pct=util,
            mem_bytes=None if used is None else used * _MIB,
            capacity_bytes=None if total is None else total * _MIB,
        )


class _TpuProbe(_Probe):
    """JAX device memory statistics. JAX exposes no core-utilization counter.

    ``memory_stats()`` gives ``bytes_in_use`` / ``peak_bytes_in_use`` / ``bytes_limit`` for
    the local device's HBM. There is deliberately no utilization number here: reporting HBM
    occupancy as if it were compute utilization would be a fabrication.
    """

    source = "jax.local_devices()[0].memory_stats() (HBM occupancy; no utilization counter)"

    def __init__(self) -> None:
        self._device = None

    def _dev(self):
        if self._device is None:
            import jax  # lazy: keeps this module importable without JAX
            devices = jax.local_devices()
            if not devices:
                raise RuntimeError("jax.local_devices() is empty")
            self._device = devices[0]
        return self._device

    def stats(self) -> dict[str, Any]:
        stats = self._dev().memory_stats()
        if not stats:
            raise RuntimeError("memory_stats() returned nothing for this device")
        return dict(stats)

    def sample(self) -> _Sample:
        stats = self.stats()
        return _Sample(
            t=time.time(),
            util_pct=None,
            mem_bytes=_to_float(stats.get("bytes_in_use")),
            capacity_bytes=_to_float(stats.get("bytes_limit")),
        )


class _CpuProbe(_Probe):
    """System CPU percentage and process RSS, via psutil when present.

    The stdlib fallbacks exist because the Tunix image has no psutil: ``/proc/stat`` deltas
    on Linux, and ``os.times()`` deltas normalised by core count elsewhere.
    """

    def __init__(self) -> None:
        self._mode = "psutil"
        self._proc = None
        self._prev_stat: tuple[float, float] | None = None
        self._prev_times: tuple[float, float] | None = None
        self._capacity = _total_system_memory_bytes()
        self._ncpu = _effective_cpu_count() or 1
        if _psutil is not None:
            try:
                self._proc = _psutil.Process()
                _psutil.cpu_percent(interval=None)  # prime the delta
            except Exception:
                self._proc = None
                self._mode = self._pick_fallback()
        else:
            self._mode = self._pick_fallback()
        self._prime_fallback()

    def _pick_fallback(self) -> str:
        return "proc" if os.path.exists("/proc/stat") else "times"

    def _prime_fallback(self) -> None:
        if self._mode == "proc":
            try:
                self._prev_stat = self._read_proc_stat()
            except Exception:
                self._mode = "times"
        if self._mode == "times":
            self._prev_times = (self._cpu_time(), time.monotonic())

    @property
    def source(self) -> str:  # type: ignore[override]
        return {
            "psutil": ("psutil.cpu_percent() [host-wide, all cores] + "
                       "psutil.Process().memory_info().rss"),
            "proc": ("/proc/stat deltas [host-wide, all cores] + /proc/self/statm "
                     "(psutil unavailable)"),
            "times": (f"os.times() deltas for this process tree / {self._ncpu} usable CPU(s) + "
                      "resource.getrusage (psutil unavailable)"),
        }[self._mode]

    @staticmethod
    def _read_proc_stat() -> tuple[float, float]:
        with open("/proc/stat", "r", encoding="utf-8") as fh:
            fields = fh.readline().split()
        vals = [float(v) for v in fields[1:9]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
        return idle, sum(vals)

    @staticmethod
    def _cpu_time() -> float:
        t = os.times()
        return t.user + t.system + t.children_user + t.children_system

    @staticmethod
    def _proc_rss() -> int | None:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as fh:
                resident_pages = int(fh.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return None

    def sample(self) -> _Sample:
        util: float | None
        rss: float | None
        if self._mode == "psutil" and self._proc is not None:
            util = float(_psutil.cpu_percent(interval=None))
            rss = float(self._proc.memory_info().rss)
        elif self._mode == "proc":
            idle, total = self._read_proc_stat()
            prev = self._prev_stat
            self._prev_stat = (idle, total)
            if prev is None or total <= prev[1]:
                util = None
            else:
                d_idle, d_total = idle - prev[0], total - prev[1]
                util = 100.0 * (1.0 - d_idle / d_total)
            rss = self._proc_rss()
            if rss is None:
                rss = _peak_rss_bytes()
        else:
            now_cpu, now_wall = self._cpu_time(), time.monotonic()
            prev = self._prev_times
            self._prev_times = (now_cpu, now_wall)
            if prev is None or now_wall <= prev[1]:
                util = None
            else:
                util = 100.0 * ((now_cpu - prev[0]) / (now_wall - prev[1])) / self._ncpu
            rss = _peak_rss_bytes()
        return _Sample(t=time.time(), util_pct=util, mem_bytes=rss,
                       capacity_bytes=self._capacity)


def _make_probe(backend: str, gpu_index: int = 0) -> _Probe:
    if backend == "gpu":
        return _GpuProbe(gpu_index=gpu_index)
    if backend == "tpu":
        return _TpuProbe()
    if backend == "cpu":
        return _CpuProbe()
    raise ContractError(f"backend must be one of {BACKENDS}, got {backend!r}")


class UtilizationSampler:
    """Background accelerator-utilization sampler producing the ``utilization`` block.

    One daemon thread wakes on a fixed deadline schedule (so a slow ``nvidia-smi`` call does
    not make the interval drift) and takes one backend-specific reading:

    ``gpu``
        ``nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total``.
    ``tpu``
        ``jax.local_devices()[0].memory_stats()``. JAX exposes no utilization percentage, so
        ``mean_pct`` and ``max_pct`` are ``null`` and HBM occupancy is reported instead in
        ``mean_memory_pct`` / ``max_memory_pct``, with an explanatory note. No utilization
        number is invented.
    ``cpu``
        ``psutil`` cpu_percent + RSS, with stdlib fallbacks.

    **This class never raises into the training loop.** Every probe failure is caught and
    counted; after ``max_consecutive_errors`` failures the sampler stops probing (so a
    missing ``nvidia-smi`` does not spawn a subprocess every second for an hour) and says so
    in ``notes``. Starting twice, or stopping without starting, is a no-op.
    """

    def __init__(
        self,
        backend: str,
        interval_s: float = 1.0,
        gpu_index: int = 0,
        max_consecutive_errors: int = 5,
    ) -> None:
        if backend not in BACKENDS:
            raise ContractError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if interval_s <= 0:
            raise ContractError("interval_s must be positive")
        self.backend = backend
        self.interval_s = float(interval_s)
        self.gpu_index = int(gpu_index)
        self.max_consecutive_errors = int(max_consecutive_errors)

        self._samples: list[_Sample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._probe: _Probe | None = None
        self._probe_source = f"{backend}: sampler never started"
        self._n_errors = 0
        self._consecutive = 0
        self._last_error: str | None = None
        self._disabled = False
        self._notes: list[str] = []

    # -- lifecycle -----------------------------------------------------------------------
    def start(self) -> "UtilizationSampler":
        """Start sampling. Idempotent; failures to construct the probe are recorded, not raised."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        try:
            self._probe = _make_probe(self.backend, self.gpu_index)
            self._probe_source = self._probe.source
        except Exception as exc:
            self._probe = None
            self._disabled = True
            self._record_error(exc)
            self._note(
                f"utilization sampling is unavailable on this {self.backend} host: "
                f"{type(exc).__name__}: {exc}. utilization.mean_pct/max_pct are null."
            )
            self._probe_source = f"{self.backend}: probe unavailable ({type(exc).__name__})"
            return self
        self._thread = threading.Thread(
            target=self._loop, name=f"telemetry-{self.backend}-sampler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> "UtilizationSampler":
        """Stop sampling and join the thread. Safe to call twice or without :meth:`start`."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                self._note("utilization sampler thread did not join within "
                           f"{timeout_s}s; it is a daemon and will not block exit.")
        self._thread = None
        return self

    def __enter__(self) -> "UtilizationSampler":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -- sampling ------------------------------------------------------------------------
    def _loop(self) -> None:
        next_at = time.monotonic()
        while not self._stop_event.is_set():
            if not self._disabled:
                self.sample_once()
            next_at += self.interval_s
            delay = next_at - time.monotonic()
            if delay <= 0:  # a slow probe overran the interval; resync rather than spin
                next_at = time.monotonic()
                delay = min(self.interval_s, 0.01)
            if self._stop_event.wait(delay):
                break

    def sample_once(self) -> _Sample | None:
        """Take one reading. Returns ``None`` (never raises) if the probe failed."""
        probe = self._probe
        if probe is None:
            try:
                probe = self._probe = _make_probe(self.backend, self.gpu_index)
                self._probe_source = probe.source
            except Exception as exc:
                self._record_error(exc)
                return None
        try:
            sample = probe.sample()
        except Exception as exc:
            self._record_error(exc)
            return None
        with self._lock:
            self._samples.append(sample)
            self._consecutive = 0
        return sample

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._n_errors += 1
            last = self._last_error = f"{type(exc).__name__}: {exc}"
            self._consecutive += 1
            consecutive = self._consecutive
        if consecutive >= self.max_consecutive_errors and not self._disabled:
            self._disabled = True
            self._note(
                f"utilization sampling stopped after {consecutive} consecutive failures "
                f"({last}); the samples collected before that point are kept."
            )

    # -- results -------------------------------------------------------------------------
    @property
    def samples(self) -> list[dict[str, float | None]]:
        with self._lock:
            return [
                {"t": s.t, "util_pct": s.util_pct, "mem_bytes": s.mem_bytes,
                 "capacity_bytes": s.capacity_bytes}
                for s in self._samples
            ]

    @property
    def n_samples(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def peak_memory_bytes(self) -> float | None:
        """Highest memory reading observed, or ``None`` if nothing was sampled."""
        with self._lock:
            values = [s.mem_bytes for s in self._samples if s.mem_bytes is not None]
        return max(values) if values else None

    @property
    def capacity_bytes(self) -> float | None:
        """Most recent reported capacity (HBM limit / GPU total / host DRAM)."""
        with self._lock:
            values = [s.capacity_bytes for s in self._samples if s.capacity_bytes is not None]
        return values[-1] if values else None

    @property
    def notes(self) -> list[str]:
        out = list(self._notes)
        if self.backend == "tpu":
            out.append(
                "utilization.mean_pct/max_pct are null on TPU by design: JAX exposes device "
                "memory statistics but no core-utilization counter, and this project does not "
                "fabricate one. HBM occupancy is reported in utilization.mean_memory_pct and "
                "utilization.max_memory_pct instead."
            )
        if self._n_errors and not any("stopped after" in n for n in out):
            out.append(
                f"utilization sampler recorded {self._n_errors} probe failure(s); "
                f"last was {self._last_error}."
            )
        return out

    def _note(self, msg: str) -> None:
        if msg not in self._notes:
            self._notes.append(msg)

    def block(self) -> dict[str, Any]:
        """The contract's ``utilization`` block, plus memory-occupancy extras."""
        with self._lock:
            samples = list(self._samples)
            n_errors = self._n_errors
            source = self._probe_source
        utils = [s.util_pct for s in samples if s.util_pct is not None]
        mem_pcts = [
            100.0 * s.mem_bytes / s.capacity_bytes
            for s in samples
            if s.mem_bytes is not None and s.capacity_bytes
        ]
        return {
            "mean_pct": (sum(utils) / len(utils)) if utils else None,
            "max_pct": max(utils) if utils else None,
            "sample_interval_s": self.interval_s,
            "n_samples": len(samples),
            "source": source,
            "mean_memory_pct": (sum(mem_pcts) / len(mem_pcts)) if mem_pcts else None,
            "max_memory_pct": max(mem_pcts) if mem_pcts else None,
            "n_probe_errors": n_errors,
        }


# --------------------------------------------------------------------------------------
# 4. peak_memory
# --------------------------------------------------------------------------------------

def peak_memory(
    backend: str,
    sampler: UtilizationSampler | None = None,
    capacity_bytes: float | None = None,
    gpu_index: int = 0,
) -> dict[str, Any]:
    """The contract's ``memory`` block, from the source the contract names for each backend.

    ``tpu``
        ``jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]``, capacity
        ``bytes_limit``. This is a true high-water mark maintained by the runtime.
    ``gpu``
        ``nvidia-smi --query-gpu=memory.used``. That is an instantaneous reading, so the peak
        comes from the sampler's observations when a sampler is supplied — pass one, or the
        block will only reflect memory at the moment of the call, which for a finished run is
        after teardown. The ``source`` string always says which of the two it was.
    ``cpu``
        peak RSS from ``resource.getrusage`` (self and reaped children), capacity = host DRAM.

    Never raises: if no source is reachable the values are ``null`` and ``source`` explains.
    """
    if backend not in BACKENDS:
        raise ContractError(f"backend must be one of {BACKENDS}, got {backend!r}")

    peak: float | None = None
    capacity: float | None = capacity_bytes
    source = "unavailable"

    if backend == "tpu":
        try:
            probe = _TpuProbe()
            stats = probe.stats()
            peak = _to_float(stats.get("peak_bytes_in_use"))
            if peak is None:
                peak = _to_float(stats.get("bytes_in_use"))
                source = ('jax.local_devices()[0].memory_stats()["bytes_in_use"] '
                          "(peak_bytes_in_use not reported by this runtime)")
            else:
                source = 'jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]'
            if capacity is None:
                capacity = _to_float(stats.get("bytes_limit"))
        except Exception as exc:
            if sampler is not None and sampler.peak_memory_bytes is not None:
                peak = sampler.peak_memory_bytes
                capacity = capacity if capacity is not None else sampler.capacity_bytes
                source = ("max of sampled jax memory_stats()['bytes_in_use'] "
                          f"(direct read failed: {type(exc).__name__})")
            else:
                source = f"unavailable: jax memory_stats() failed ({type(exc).__name__}: {exc})"

    elif backend == "gpu":
        sampled = sampler.peak_memory_bytes if sampler is not None else None
        if sampled is not None:
            peak = sampled
            capacity = capacity if capacity is not None else sampler.capacity_bytes
            source = ("max over sampled nvidia-smi --query-gpu=memory.used "
                      f"({sampler.n_samples} samples at {sampler.interval_s}s)")
        else:
            try:
                one = _GpuProbe(gpu_index=gpu_index).sample()
                peak = one.mem_bytes
                capacity = capacity if capacity is not None else one.capacity_bytes
                source = ("nvidia-smi --query-gpu=memory.used, single instantaneous read "
                          "(no sampler supplied, so this is not a high-water mark)")
            except Exception as exc:
                source = f"unavailable: nvidia-smi failed ({type(exc).__name__}: {exc})"

    else:  # cpu
        peak = _peak_rss_bytes()
        if peak is None:
            sampled = sampler.peak_memory_bytes if sampler is not None else None
            if sampled is not None:
                peak = sampled
                source = "max of sampled process RSS (getrusage unavailable)"
            else:
                source = "unavailable: resource.getrusage() is not available on this host"
        else:
            unit = "bytes on darwin" if sys.platform == "darwin" else "KiB on linux"
            source = ("max(resource.getrusage(RUSAGE_SELF), RUSAGE_CHILDREN).ru_maxrss "
                      f"[{unit}] — peak RSS")
        if capacity is None:
            capacity = _total_system_memory_bytes()
            # peak_pct is only meaningful against the ceiling the container was actually given.
            cgroup_limit = _cgroup_memory_limit_bytes()
            if cgroup_limit is not None and capacity == cgroup_limit:
                source += "; capacity_bytes is the cgroup memory limit of this container"
            elif capacity is not None:
                source += "; capacity_bytes is host DRAM (no cgroup memory limit is set)"

    peak_pct = (100.0 * peak / capacity) if (peak is not None and capacity) else None
    return {
        "peak_bytes": peak,
        "capacity_bytes": capacity,
        "peak_pct": peak_pct,
        "source": source,
    }


# --------------------------------------------------------------------------------------
# Hardware detection (real queries only — every field is measured or explicitly passed in)
# --------------------------------------------------------------------------------------

def _cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"], timeout_s=5.0).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def hardware_block(backend: str, label: str, **overrides: Any) -> dict[str, Any]:
    """The contract's ``hardware`` block, detected from the live machine.

    Any of ``chips``, ``chip_model``, ``host_arch``, ``hbm_per_chip_bytes`` passed as keyword
    arguments wins over detection — use that for facts the machine cannot report (the TPU
    slice topology name, for instance). Detection failures leave the field ``None`` rather
    than guessing.

    For the CPU backend, ``chips`` is the logical core count and ``hbm_per_chip_bytes`` is
    host DRAM: the columns keep their meaning of "parallel units" and "memory available to
    one unit's worth of the job".
    """
    if backend not in BACKENDS:
        raise ContractError(f"backend must be one of {BACKENDS}, got {backend!r}")
    unexpected = set(overrides) - set(REQUIRED_HARDWARE)
    if unexpected:
        raise ContractError(f"unknown hardware override(s): {sorted(unexpected)}")

    block: dict[str, Any] = {
        "label": label,
        "chips": None,
        "chip_model": None,
        "host_arch": platform.machine(),
        "hbm_per_chip_bytes": None,
    }

    try:
        if backend == "tpu":
            import jax  # lazy
            devices = jax.local_devices()
            block["chips"] = len(devices)
            block["chip_model"] = getattr(devices[0], "device_kind", None)
            stats = devices[0].memory_stats() or {}
            block["hbm_per_chip_bytes"] = _to_float(stats.get("bytes_limit"))
        elif backend == "gpu":
            out = _run_cmd([
                "nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            rows = [r for r in out.splitlines() if r.strip()]
            block["chips"] = len(rows)
            if rows:
                name, total = (p.strip() for p in rows[0].split(",")[:2])
                block["chip_model"] = name
                block["hbm_per_chip_bytes"] = float(total) * _MIB
        else:
            block["chips"] = _effective_cpu_count()
            block["chip_model"] = _cpu_model_name()
            block["hbm_per_chip_bytes"] = _total_system_memory_bytes()
    except Exception:
        pass  # detection is best-effort; overrides and nulls carry the truth

    block.update(overrides)

    # Overrides usually originate in pod environment variables, where everything is a string.
    # Coerce them here rather than letting validate_record reject the record at the very end of
    # a forty-minute run: an unparseable value becomes null (honest) instead of fatal.
    chips = _to_float(block["chips"])
    block["chips"] = int(chips) if chips is not None else None
    block["hbm_per_chip_bytes"] = _to_float(block["hbm_per_chip_bytes"])
    for key in ("label", "chip_model", "host_arch"):
        if block[key] is not None and not isinstance(block[key], str):
            block[key] = str(block[key])
    return block


# --------------------------------------------------------------------------------------
# 5. emit / validation
# --------------------------------------------------------------------------------------

def _sanitize(obj: Any, issues: list[str], path: str = "$") -> Any:
    """Make ``obj`` JSON-safe without rounding anything.

    Non-finite floats become ``null`` (JSON has no NaN and a bare ``NaN`` token would break
    any non-Python reader), numpy/JAX scalars are materialised via ``.item()``, and anything
    else unserialisable is stringified. Every substitution is appended to ``issues`` so it can
    surface in ``notes`` instead of vanishing.
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return _defang_sentinels(obj)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        issues.append(f"{path} was {obj!r} (non-finite) and is written as null")
        return None
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v, issues, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v, issues, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return utc_timestamp(obj)
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _sanitize(item(), issues, path)
        except Exception:
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return _sanitize(tolist(), issues, path)
        except Exception:
            pass
    issues.append(f"{path} of type {type(obj).__name__} was coerced to a string")
    return str(obj)


def _require_keys(block: Any, keys: Sequence[str], where: str) -> None:
    if not isinstance(block, Mapping):
        raise ContractError(f"{where} must be an object, got {type(block).__name__}")
    missing = [k for k in keys if k not in block]
    if missing:
        raise ContractError(f"{where} is missing required field(s): {missing}")


def _require_number(
    value: Any, where: str, allow_null: bool = True, integral: bool = False
) -> None:
    """Type-check one leaf of the contract.

    Presence is not enough: ``_sanitize`` stringifies anything it cannot serialise, so a value
    that arrived as an unexpected object would reach ``results/`` as ``"Traced<float32[]>"``
    and only fail much later, inside the chart layer, on a machine that no longer has the run.
    Numbers are checked where the contract shows numbers.
    """
    if value is None:
        if allow_null:
            return
        raise ContractError(f"{where} must not be null")
    if isinstance(value, bool):
        raise ContractError(f"{where} must be a number, got the boolean {value!r}")
    if integral:
        if not isinstance(value, int):
            raise ContractError(f"{where} must be an integer, got {value!r}")
        return
    if not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be a number or null, got {value!r}")
    if not math.isfinite(float(value)):
        raise ContractError(f"{where} must be finite, got {value!r}")


def _require_text(value: Any, where: str, allow_null: bool = False) -> None:
    if value is None and allow_null:
        return
    if not isinstance(value, str):
        raise ContractError(f"{where} must be a string, got {type(value).__name__}")


def _check_phase_sum(phases: Mapping[str, Any]) -> None:
    total = _to_float(phases.get("total_wall_s"))
    if total is None:
        raise ContractError(
            f"phases.total_wall_s must be a finite number, got {phases.get('total_wall_s')!r}"
        )
    parts = 0.0
    for key in PHASE_KEYS:
        value = phases[key]
        if value is None:
            continue
        seconds = _to_float(value)
        if seconds is None:
            raise ContractError(f"phases.{key} must be a finite number or null, got {value!r}")
        parts += seconds
    tol = _PHASE_SUM_ABS_TOL_S + 1e-6 * abs(total)
    if abs(parts - total) > tol:
        raise ContractError(
            f"phases do not sum to total_wall_s: sum(non-null phases)={parts!r} vs "
            f"total_wall_s={total!r} (delta {parts - total!r}s, tolerance {tol!r}s). "
            "Unattributed time belongs in other_s; it must not be dropped."
        )


def validate_record(record: Mapping[str, Any]) -> None:
    """Raise :class:`ContractError` unless ``record`` satisfies docs/metrics-contract.md."""
    _require_keys(record, REQUIRED_TOP_LEVEL, "record")

    if not isinstance(record["run_id"], str) or not record["run_id"]:
        raise ContractError("run_id must be a non-empty string")
    if record["backend"] not in BACKENDS:
        raise ContractError(f"backend must be one of {BACKENDS}, got {record['backend']!r}")
    if record["status"] not in STATUSES:
        raise ContractError(f"status must be one of {STATUSES}, got {record['status']!r}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(record["timestamp_utc"])):
        raise ContractError(
            f"timestamp_utc must look like 2026-08-10T21:04:11Z, got {record['timestamp_utc']!r}"
        )

    hardware = record["hardware"]
    _require_keys(hardware, REQUIRED_HARDWARE, "hardware")
    _require_text(hardware["label"], "hardware.label")
    _require_number(hardware["chips"], "hardware.chips", integral=True)
    _require_text(hardware["chip_model"], "hardware.chip_model", allow_null=True)
    _require_text(hardware["host_arch"], "hardware.host_arch", allow_null=True)
    _require_number(hardware["hbm_per_chip_bytes"], "hardware.hbm_per_chip_bytes")

    config = record["config"]
    _require_keys(config, REQUIRED_CONFIG, "config")
    _require_text(config["model"], "config.model")
    _require_text(config["dtype"], "config.dtype")
    _require_text(config["remat"], "config.remat", allow_null=True)
    for key in ("batch_size", "seq_len", "lora_rank", "max_steps"):
        _require_number(config[key], f"config.{key}", allow_null=False, integral=True)
    _require_number(config["lora_alpha"], "config.lora_alpha", allow_null=False)

    _require_keys(record["phases"], REQUIRED_PHASES, "phases")
    _check_phase_sum(record["phases"])

    utilization = record["utilization"]
    _require_keys(utilization, REQUIRED_UTILIZATION, "utilization")
    _require_number(utilization["mean_pct"], "utilization.mean_pct")
    _require_number(utilization["max_pct"], "utilization.max_pct")
    _require_number(utilization["sample_interval_s"], "utilization.sample_interval_s",
                    allow_null=False)
    _require_number(utilization["n_samples"], "utilization.n_samples",
                    allow_null=False, integral=True)
    _require_text(utilization["source"], "utilization.source")

    evaluation = record["eval"]
    _require_keys(evaluation, REQUIRED_EVAL, "eval")
    _require_number(evaluation["micro_f1"], "eval.micro_f1")
    _require_number(evaluation["n_examples"], "eval.n_examples", allow_null=False, integral=True)
    _require_text(evaluation["note"], "eval.note")

    steps = record["steps"]
    if not isinstance(steps, list):
        raise ContractError(f"steps must be a list, got {type(steps).__name__}")
    for i, step in enumerate(steps):
        _require_keys(step, REQUIRED_STEP, f"steps[{i}]")
        _require_number(step["step"], f"steps[{i}].step", allow_null=False, integral=True)
        _require_number(step["wall_s"], f"steps[{i}].wall_s", allow_null=False)
        _require_number(step["loss"], f"steps[{i}].loss")

    status = record["status"]
    if status == "ok":
        if record["steady"] is None:
            raise ContractError(
                "steady must not be null when status == 'ok'. If no step completed, the run "
                "did not succeed — use status 'error' or 'oom'."
            )
        if record["memory"] is None:
            raise ContractError("memory must not be null when status == 'ok'")
        if record["error"] is not None:
            raise ContractError("error must be null when status == 'ok'")
    else:
        if not record["error"]:
            raise ContractError(f"status {status!r} requires a non-empty error field")

    if record["steady"] is not None:
        steady = record["steady"]
        _require_keys(steady, REQUIRED_STEADY, "steady")
        for key in ("median_step_s", "p10_step_s", "p90_step_s", "tokens_per_s", "docs_per_s"):
            _require_number(steady[key], f"steady.{key}")
        _require_number(steady["warmup_steps_excluded"], "steady.warmup_steps_excluded",
                        allow_null=False, integral=True)
    if record["memory"] is not None:
        memory = record["memory"]
        _require_keys(memory, REQUIRED_MEMORY, "memory")
        for key in ("peak_bytes", "capacity_bytes", "peak_pct"):
            _require_number(memory[key], f"memory.{key}")
        _require_text(memory["source"], "memory.source")
    if record["error"] is not None:
        _require_text(record["error"], "error")
    if not isinstance(record["notes"], str):
        raise ContractError(f"notes must be a string, got {type(record['notes']).__name__}")


def _format_error(error: Any) -> tuple[str | None, str | None]:
    """Normalise ``error`` to ``(message, traceback_text)``."""
    if error is None:
        return None, None
    if isinstance(error, BaseException):
        message = f"{type(error).__name__}: {error}"
        tb = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ) if error.__traceback__ else None
        return message[:4000], (tb[-8000:] if tb else None)
    return str(error)[:4000], None


def build_record(**blocks: Any) -> dict[str, Any]:
    """Assemble and validate a contract record without writing anything.

    Required keyword arguments: ``run_id``, ``backend``, ``hardware``, ``config``, ``phases``,
    ``steps``, ``utilization``. Defaulted: ``timestamp_utc`` (now), ``status`` (derived from
    ``error`` if one is given, else ``"ok"``), ``steady``/``memory`` (``None``), ``eval``
    (a not-evaluated placeholder with a ``null`` score — never a fabricated 0.0), ``notes``
    (``""``; a list is joined with ``" | "``).

    ``error`` may be an exception object; its type, message and traceback are captured and
    the status is inferred with :func:`classify_exception` unless ``status`` is explicit.
    Raises :class:`ContractError` on any violation.
    """
    unknown = set(blocks) - set(REQUIRED_TOP_LEVEL) - {"error_traceback"}
    if unknown:
        raise ContractError(
            f"unknown field(s) for a contract record: {sorted(unknown)}. "
            f"Allowed: {sorted(REQUIRED_TOP_LEVEL)}"
        )

    error_obj = blocks.get("error")
    error_msg, error_tb = _format_error(error_obj)
    status = blocks.get("status")
    if status is None:
        status = classify_exception(error_obj) if isinstance(error_obj, BaseException) else (
            "ok" if error_msg is None else "error"
        )

    raw_notes = blocks.get("notes", "")
    if isinstance(raw_notes, str):
        note_list = [raw_notes] if raw_notes else []
    else:
        note_list = [str(n) for n in raw_notes if str(n)]

    record: dict[str, Any] = {
        "run_id": blocks.get("run_id"),
        "backend": blocks.get("backend"),
        "timestamp_utc": blocks.get("timestamp_utc") or utc_timestamp(),
        "hardware": blocks.get("hardware"),
        "config": blocks.get("config"),
        "phases": blocks.get("phases"),
        "steps": blocks.get("steps", []),
        "steady": blocks.get("steady"),
        "memory": blocks.get("memory"),
        "utilization": blocks.get("utilization"),
        "eval": blocks.get("eval") or {
            "micro_f1": None,
            "n_examples": 0,
            "note": "not evaluated in this run",
        },
        "status": status,
        "error": error_msg,
        "notes": "",
    }
    if error_tb:
        record["error_traceback"] = error_tb
    elif blocks.get("error_traceback"):
        record["error_traceback"] = str(blocks["error_traceback"])[-8000:]

    missing = [k for k in REQUIRED_TOP_LEVEL if record.get(k) is None
               and k not in ("steady", "memory", "error", "notes")]
    if missing:
        raise ContractError(
            f"missing required field(s): {missing}. emit() refuses to write a partial record."
        )

    if status == "oom":
        note_list.append(
            "Run ended in an out-of-memory failure; steady and memory are null by contract "
            "and the phases below cover the run up to the failure point."
        )
    if isinstance(record["memory"], Mapping) and record["memory"].get("peak_bytes") is None:
        note_list.append(
            f"memory.peak_bytes is null: {record['memory'].get('source')}"
        )

    issues: list[str] = []
    record = _sanitize(record, issues)
    note_list.extend(issues)
    record["notes"] = _defang_sentinels(
        " | ".join(dict.fromkeys(n.strip() for n in note_list if n.strip()))
    )

    validate_record(record)
    return record


def emit(path: str | os.PathLike | None, **blocks: Any) -> dict[str, Any]:
    """Validate the blocks, write ``results/<run_id>.json``, and return the record.

    ``path`` may be a file path, a directory (the file is named ``<run_id>.json`` inside it),
    or ``None`` to skip the file entirely and rely on the log channel. The write is atomic —
    a temporary file in the same directory is ``os.replace``-d into position — and validation
    happens first, so **a rejected record never produces a partial file**.

    The record is also printed to stdout between :data:`LOG_BEGIN` / :data:`LOG_END`
    sentinels, which is how results escape the GPU cluster, where pod logs are the only
    output channel available to this RBAC persona. Pass ``quiet=True`` (or set
    ``TELEMETRY_QUIET=1``) to suppress that.

    If the file write fails (read-only mount, full ephemeral disk, gcsfuse error) the record is
    **still logged**, with the failure appended to ``notes``, because losing a completed
    benchmark to a filesystem problem is the worse outcome. The ``OSError`` is only re-raised
    when ``quiet`` has also disabled the log channel, in which case nothing else would report it.

    OOM handling: pass the exception as ``error=exc`` with ``steady=None, memory=None`` and
    keep the ``phases`` snapshot taken at the failure point. The status is inferred as
    ``"oom"``, so the sweep chart's OOM boundary is built from a real record rather than a
    missing file.
    """
    quiet = blocks.pop("quiet", None)
    if quiet is None:
        quiet = os.environ.get("TELEMETRY_QUIET") == "1"
    indent = blocks.pop("indent", 2)

    record = build_record(**blocks)
    text = json.dumps(record, indent=indent, sort_keys=False, allow_nan=False)

    write_error: OSError | None = None
    if path is not None:
        target = Path(path)
        tmp: Path | None = None
        try:
            if str(path).endswith(os.sep) or target.is_dir():
                target = target / f"{record['run_id']}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            tmp.write_text(text + "\n", encoding="utf-8")
            os.replace(tmp, target)
        except OSError as exc:
            # A read-only mount, a full ephemeral disk or a gcsfuse hiccup must not destroy the
            # measurement: on the GH200 cluster the pod log is the ONLY channel out of the
            # container, so the record still has to reach stdout. The failure is recorded in
            # notes rather than swallowed, and the run keeps its own exit status.
            write_error = exc
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    if write_error is not None:
        failure = (
            f"the results file could not be written to {path}: "
            f"{type(write_error).__name__}: {write_error}. This record exists only in the "
            "process log, between the telemetry harvest sentinels."
        )
        record["notes"] = _defang_sentinels(
            f"{record['notes']} | {failure}" if record["notes"] else failure
        )
        text = json.dumps(record, indent=indent, sort_keys=False, allow_nan=False)
        print(f"[TELEMETRY] WARNING: {failure}", file=sys.stderr, flush=True)

    if not quiet:
        log_record(record, text=text)
    if write_error is not None and quiet:
        # Nothing was written and nothing was logged — the caller silenced the only remaining
        # channel, so the loss has to surface as an exception rather than as silence.
        raise write_error
    return record


def log_record(record: Mapping[str, Any], stream=None, text: str | None = None) -> None:
    """Print a record to a stream between the harvest sentinels, and flush."""
    stream = stream if stream is not None else sys.stdout
    payload = text if text is not None else json.dumps(record, indent=2, allow_nan=False)
    # Last line of defence for records that did not come through build_record: a sentinel
    # quoted inside the payload would truncate the block the launcher parses.
    payload = _defang_sentinels(payload)
    stream.write(f"\n{LOG_BEGIN}\n{payload}\n{LOG_END}\n")
    stream.flush()


def extract_records(text: str) -> list[dict[str, Any]]:
    """Recover every sentinel-delimited record from a blob of ``kubectl logs`` output.

    Kubernetes may interleave or prefix log lines, so each captured block is parsed
    independently and unparseable blocks are skipped rather than aborting the harvest.
    """
    out: list[dict[str, Any]] = []
    pattern = re.compile(
        re.escape(LOG_BEGIN) + r"\s*(.*?)\s*" + re.escape(LOG_END), re.DOTALL
    )
    for match in pattern.finditer(text):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def merge_external_phases(
    target: str | os.PathLike | dict[str, Any],
    submit_to_running_s: float | None = None,
    image_pull_s: float | None = None,
) -> dict[str, Any]:
    """Fill in phases the pod could not measure and repair ``total_wall_s``.

    The launcher knows how long the Job waited for Kueue admission and how long the image
    pull took (from kubectl event timestamps); the container does not. This adds those
    numbers after the fact, adjusts ``total_wall_s`` by the same delta so the sum invariant
    still holds, re-validates, and rewrites the file when ``target`` is a path.
    """
    if isinstance(target, dict):
        record, path = target, None
    else:
        path = Path(target)
        record = json.loads(path.read_text(encoding="utf-8"))

    phases = record.get("phases")
    if not isinstance(phases, dict):
        raise ContractError("record has no phases block to merge into")
    base_total = _to_float(phases.get("total_wall_s"))
    if base_total is None:
        raise ContractError(
            f"phases.total_wall_s must be a finite number, got {phases.get('total_wall_s')!r}"
        )

    updates = {"submit_to_running_s": submit_to_running_s, "image_pull_s": image_pull_s}
    delta = 0.0
    for key, value in updates.items():
        if value is None:
            continue
        new = _to_float(value)
        if new is None:
            raise ContractError(f"{key} must be a finite number, got {value!r}")
        old = _to_float(phases.get(key))
        delta += new - (old if old is not None else 0.0)
        phases[key] = new
    phases["total_wall_s"] = base_total + delta

    validate_record(record)
    if path is not None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return record


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------

def _selftest() -> int:  # noqa: C901 - a linear script of assertions reads better flat
    import io
    import shutil
    import tempfile

    passed: list[str] = []

    def ok(name: str) -> None:
        passed.append(name)
        print(f"  pass  {name}")

    print("telemetry.py self-test")
    print(f"  python {platform.python_version()} on {platform.system()} "
          f"{platform.machine()}; psutil={'yes' if _psutil else 'no'}")

    # -- _percentile against hand-computed numpy-equivalent values -----------------------
    vals = [0.40, 0.41, 0.42, 0.43, 0.47]
    assert _percentile(vals, 50) == 0.42
    assert abs(_percentile(vals, 10) - 0.404) < 1e-12
    assert abs(_percentile(vals, 90) - 0.454) < 1e-12
    assert _percentile([], 50) is None
    assert _percentile([1.5], 90) == 1.5
    ok("_percentile matches numpy linear interpolation")

    # -- PhaseTimer sum invariant with a deterministic fake clock -------------------------
    now = [1000.0]
    timer = PhaseTimer(backend="tpu", clock=lambda: now[0], env={})
    with timer.phase("model_load_s"):
        now[0] += 1620.0
    with timer.phase("compile_s"):
        now[0] += 180.0
    with timer.phase("steady_state_s"):
        now[0] += 300.0
    with timer.phase("checkpoint_write_s"):
        now[0] += 30.0
    with timer.phase("checkpoint_write_s"):  # accumulates, does not overwrite
        now[0] += 30.0
    now[0] += 12.5  # unattributed
    timer.set_external("submit_to_running_s", 180.2)
    timer.set_external("image_pull_s", 45.1)
    timer.stop()
    ph = timer.snapshot()
    assert ph["checkpoint_write_s"] == 60.0, ph
    assert abs(ph["other_s"] - 12.5) < 1e-9, ph
    assert abs(ph["total_wall_s"] - (180.2 + 45.1 + 2172.5)) < 1e-9, ph
    assert abs(sum(ph[k] for k in PHASE_KEYS) - ph["total_wall_s"]) < 1e-9
    ok("PhaseTimer phases sum to total_wall_s and other_s carries the remainder")

    # nesting is rejected; snapshot from inside a phase still attributes the open interval
    nested = PhaseTimer(clock=lambda: now[0], env={})
    try:
        with nested.phase("compile_s"):
            with nested.phase("model_load_s"):
                pass
        raise AssertionError("nesting should have raised")
    except RuntimeError:
        pass
    inside = PhaseTimer(clock=lambda: now[0], env={})
    with inside.phase("model_load_s"):
        now[0] += 5.0
        snap = inside.snapshot()
    assert abs(snap["model_load_s"] - 5.0) < 1e-9, snap
    assert abs(snap["other_s"]) < 1e-9, snap
    ok("PhaseTimer rejects nesting and snapshots correctly from inside a phase")

    # externals from the environment, and the cpu-backend zero default
    env_timer = PhaseTimer(clock=lambda: now[0],
                           env={"TELEMETRY_SUBMIT_TO_RUNNING_S": "12.5"})
    assert env_timer.snapshot()["submit_to_running_s"] == 12.5
    assert env_timer.snapshot()["image_pull_s"] is None
    assert any("image_pull_s is null" in n for n in env_timer.notes)
    cpu_timer = PhaseTimer(backend="cpu", clock=lambda: now[0], env={})
    assert cpu_timer.snapshot()["submit_to_running_s"] == 0.0
    ok("PhaseTimer reads external phases from env and zeroes them on the CPU backend")

    # -- StepRecorder --------------------------------------------------------------------
    tick = [0.0]
    rec = StepRecorder(batch_size=8, seq_len=512, warmup_steps=2, clock=lambda: tick[0])
    for i, (dt, loss) in enumerate([(182.4, 8.31), (0.44, 7.90), (0.40, 7.5),
                                    (0.42, 7.2), (0.47, 7.1), (0.41, 7.0),
                                    (0.43, 6.9)]):
        with rec.step() as scope:
            tick[0] += dt
            scope.loss = loss
        assert rec.steps[i]["step"] == i + 1
    steps = rec.steps
    assert len(steps) == 7 and abs(steps[0]["wall_s"] - 182.4) < 1e-9
    st = rec.steady()
    assert st["warmup_steps_excluded"] == 2
    assert abs(st["median_step_s"] - 0.42) < 1e-9, st
    assert abs(st["tokens_per_s"] - (8 * 512) / st["median_step_s"]) < 1e-9, st
    assert abs(st["tokens_per_s"] - (8 * 512) / 0.42) < 1e-3, st
    assert abs(st["docs_per_s"] - 8 / 0.42) < 1e-6, st
    assert st["p10_step_s"] <= st["median_step_s"] <= st["p90_step_s"]
    ok("StepRecorder excludes warmup and derives throughput from the median step")

    short = StepRecorder(batch_size=1, seq_len=512, warmup_steps=10, clock=lambda: tick[0])
    for dt in (30.0, 2.0, 2.1):
        with short.step() as scope:
            tick[0] += dt
            scope.loss = 1.0
    st_short = short.steady()
    assert st_short["warmup_steps_excluded"] == 1, st_short
    assert any("fewer than the 10 warmup steps" in n for n in short.notes)
    assert StepRecorder(batch_size=1, seq_len=8).steady() is None
    ok("StepRecorder degrades honestly when fewer steps ran than the warmup window")

    nan_rec = StepRecorder(batch_size=2, seq_len=16, warmup_steps=0, clock=lambda: tick[0])
    nan_rec.record(wall_s=0.5, loss=float("nan"))
    assert nan_rec.steps[0]["loss"] is None
    assert any("not finite" in n for n in nan_rec.notes)

    class _FakeDeviceScalar:  # stands in for a jax/numpy 0-d array
        def __init__(self, v): self._v = v
        def item(self): return self._v

    nan_rec.record(wall_s=0.6, loss=_FakeDeviceScalar(3.25))
    assert nan_rec.steps[1]["loss"] == 3.25
    ok("StepRecorder nulls non-finite losses and materialises device scalars")

    # -- UtilizationSampler --------------------------------------------------------------
    with UtilizationSampler("cpu", interval_s=0.05) as sampler:
        total = 0.0
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            total += math.sqrt(total + 1.0)
    ub = sampler.block()
    assert set(REQUIRED_UTILIZATION).issubset(ub), ub
    assert ub["n_samples"] >= 2, ub
    assert ub["sample_interval_s"] == 0.05
    assert "psutil" in ub["source"] or "proc" in ub["source"] or "os.times" in ub["source"]
    sampler.stop()  # idempotent
    ok(f"UtilizationSampler(cpu) collected {ub['n_samples']} samples and stops cleanly")

    broken = UtilizationSampler("cpu", interval_s=0.01, max_consecutive_errors=3)

    class _BrokenProbe(_Probe):
        source = "deliberately broken probe"
        def sample(self): raise RuntimeError("probe exploded")

    broken._probe = _BrokenProbe()
    broken._probe_source = _BrokenProbe.source
    broken._thread = threading.Thread(target=broken._loop, daemon=True)
    broken._thread.start()
    time.sleep(0.15)
    broken.stop()
    bb = broken.block()
    assert bb["n_samples"] == 0 and bb["n_probe_errors"] >= 3, bb
    assert bb["mean_pct"] is None and bb["max_pct"] is None
    assert any("stopped after" in n for n in broken.notes)
    ok("UtilizationSampler survives a probe that always raises and disables itself")

    tpu_sampler = UtilizationSampler("tpu", interval_s=0.05).start()
    time.sleep(0.1)
    tpu_sampler.stop()
    tb = tpu_sampler.block()
    assert tb["mean_pct"] is None and tb["max_pct"] is None, tb
    assert any("null on TPU by design" in n for n in tpu_sampler.notes)
    ok("UtilizationSampler(tpu) reports null utilization with an explicit note")

    # -- peak_memory ---------------------------------------------------------------------
    mem = peak_memory("cpu")
    assert set(REQUIRED_MEMORY) == set(mem)
    assert mem["peak_bytes"] is None or mem["peak_bytes"] > 1_000_000, mem
    gpu_mem = peak_memory("gpu")  # no nvidia-smi on this host: must not raise
    assert set(REQUIRED_MEMORY) == set(gpu_mem)
    ok(f"peak_memory(cpu) -> {mem['peak_bytes']} bytes; peak_memory(gpu) degrades to nulls")

    # -- emit round trip -----------------------------------------------------------------
    tmpdir = Path(tempfile.mkdtemp(prefix="telemetry-selftest-"))
    try:
        hw = hardware_block("cpu", label="self-test host")
        assert set(hw) == set(REQUIRED_HARDWARE)
        cfg = {
            "model": "Qwen/Qwen3-4B", "batch_size": 8, "seq_len": 512,
            "lora_rank": 32, "lora_alpha": 64.0, "dtype": "bfloat16",
            "max_steps": 100, "remat": "DECODER", "file_cache_capacity": "100Gi",
        }
        run_id = make_run_id("tpu", "v5e8", 8, 512)
        assert run_id == "tpu-v5e8-bs8-seq512"
        path = tmpdir / f"{run_id}.json"
        written = emit(
            path,
            run_id=run_id, backend="tpu",
            hardware=hardware_block("tpu", label="TPU v5e 2x4", chips=8,
                                    chip_model="tpu-v5-lite-podslice",
                                    host_arch="x86_64",
                                    hbm_per_chip_bytes=17179869184),
            config=cfg,
            phases=ph,
            steps=steps,
            steady=st,
            memory={"peak_bytes": 12884901888, "capacity_bytes": 17179869184,
                    "peak_pct": 75.0, "source": "jax memory_stats()"},
            utilization=tb,
            eval={"micro_f1": 0.0, "n_examples": 0,
                  "note": "functional sanity check only, not a graded metric"},
            notes=timer.notes + rec.notes + tpu_sampler.notes,
            quiet=True,
        )
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        assert reloaded == written, "round trip changed the record"
        validate_record(reloaded)
        assert set(REQUIRED_TOP_LEVEL).issubset(reloaded)
        assert "null on TPU by design" in reloaded["notes"]
        ok(f"emit -> {path.name} round-trips and validates ({path.stat().st_size} bytes)")

        # directory target names the file from run_id
        dir_written = emit(tmpdir, run_id="cpu-x86-32c-bs1-seq512", backend="cpu",
                           hardware=hw, config=cfg, phases=cpu_timer.snapshot(),
                           steps=[], steady=st, memory=mem, utilization=ub, quiet=True)
        assert (tmpdir / "cpu-x86-32c-bs1-seq512.json").is_file()
        assert dir_written["eval"]["micro_f1"] is None, "eval must not fabricate a score"
        ok("emit accepts a directory target and defaults eval to a null score, not 0.0")

        # missing required field -> raises, writes nothing
        missing_path = tmpdir / "should-not-exist.json"
        try:
            emit(missing_path, run_id="x-y-bs1-seq1", backend="cpu", hardware=hw,
                 config=cfg, phases=ph, steps=[], utilization=ub, quiet=True)
            raise AssertionError("emit should have raised on the missing utilization/steady")
        except ContractError as exc:
            assert "missing required field" in str(exc) or "steady" in str(exc), exc
        assert not missing_path.exists(), "emit wrote a partial file on a validation failure"
        ok("emit raises on a missing required field and writes nothing")

        # broken phase sum -> raises
        bad_phases = dict(ph)
        bad_phases["total_wall_s"] = bad_phases["total_wall_s"] + 100.0
        try:
            emit(None, run_id=run_id, backend="tpu", hardware=hw, config=cfg,
                 phases=bad_phases, steps=steps, steady=st, memory=mem,
                 utilization=ub, quiet=True)
            raise AssertionError("emit should have rejected the phase sum")
        except ContractError as exc:
            assert "do not sum to total_wall_s" in str(exc), exc
        ok("emit rejects a phase breakdown that does not sum to total_wall_s")

        # unknown field -> raises (typo protection)
        try:
            emit(None, run_id=run_id, backend="tpu", hardware=hw, config=cfg, phases=ph,
                 steps=steps, steady=st, memory=mem, utilization=ub,
                 utilisation=ub, quiet=True)
            raise AssertionError("emit should have rejected the misspelled field")
        except ContractError as exc:
            assert "unknown field" in str(exc), exc
        ok("emit rejects unknown/misspelled top-level fields")

        # typed leaves: a value that _sanitize stringified must not reach results/
        for bad_kwargs, fragment in (
            ({"steps": [{"step": 1, "wall_s": "0.4", "loss": None}]}, "wall_s must be a number"),
            ({"steps": [{"step": 1, "wall_s": 0.4, "loss": "Traced<f32[]>"}]},
             "loss must be a number"),
            ({"config": {**cfg, "batch_size": "8"}}, "batch_size must be an integer"),
            ({"steady": {**st, "warmup_steps_excluded": None}},
             "warmup_steps_excluded must not be null"),
        ):
            base = dict(run_id=run_id, backend="tpu", hardware=hw, config=cfg, phases=ph,
                        steps=steps, steady=st, memory=mem, utilization=ub, quiet=True)
            base.update(bad_kwargs)
            try:
                emit(None, **base)
                raise AssertionError(f"emit should have rejected: {fragment}")
            except ContractError as exc:
                assert fragment in str(exc), (fragment, exc)
        ok("emit type-checks every numeric leaf instead of writing a stringified value")

        # a record that quotes the harvest sentinels must not truncate its own log block
        sentinel_buf = io.StringIO()
        real_stdout, sys.stdout = sys.stdout, sentinel_buf
        try:
            emit(None, run_id=run_id, backend="tpu", hardware=hw, config=cfg, phases=ph,
                 steps=[], steady=None, memory=None, utilization=ub, status="error",
                 error=RuntimeError(f"crash {LOG_END} tail {LOG_BEGIN}"))
        finally:
            sys.stdout = real_stdout
        recovered = extract_records(sentinel_buf.getvalue())
        assert len(recovered) == 1, sentinel_buf.getvalue()
        assert LOG_END not in json.dumps(recovered[0])
        assert "marker removed from text" in recovered[0]["error"]
        ok("a record quoting the harvest sentinels is defanged and stays parseable")

        # a failed file write must not destroy the record: the log copy is the GPU cluster's
        # only channel, so it is written even when results/ is unwritable.
        readonly = tmpdir / "readonly"
        readonly.mkdir()
        os.chmod(readonly, 0o500)
        try:
            ro_buf = io.StringIO()
            real_stdout, sys.stdout = sys.stdout, ro_buf
            try:
                emit(readonly / "x.json", run_id=run_id, backend="tpu", hardware=hw, config=cfg,
                     phases=ph, steps=steps, steady=st, memory=mem, utilization=ub)
            finally:
                sys.stdout = real_stdout
            rescued = extract_records(ro_buf.getvalue())
            assert len(rescued) == 1, ro_buf.getvalue()[-400:]
            assert "could not be written" in rescued[0]["notes"], rescued[0]["notes"]
            validate_record(rescued[0])
            assert not (readonly / "x.json").exists()
        finally:
            os.chmod(readonly, 0o700)
        ok("emit still logs the record when the results file cannot be written")

        # externals are only zeroed off-cluster; inside a pod they stay null and get merged
        pod_timer = PhaseTimer(backend="cpu", clock=lambda: now[0],
                               env={"KUBERNETES_SERVICE_HOST": "10.0.0.1"})
        pod_phases = pod_timer.snapshot()
        assert pod_phases["submit_to_running_s"] is None, pod_phases
        assert pod_phases["image_pull_s"] is None, pod_phases
        ok("PhaseTimer does not claim 0.0 externals for a CPU run inside Kubernetes")

        # OOM path
        oom_partial = PhaseTimer(clock=lambda: now[0], env={})
        with oom_partial.phase("model_load_s"):
            now[0] += 900.0
        oom_partial.stop()
        try:
            raise RuntimeError(
                "RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to "
                "allocate 15.20G. That was not possible."
            )
        except RuntimeError as exc:
            assert classify_exception(exc) == "oom"
            oom_path = tmpdir / "tpu-v5e8-bs32-seq512.json"
            oom_rec = emit(
                oom_path, run_id="tpu-v5e8-bs32-seq512", backend="tpu",
                hardware=hw, config={**cfg, "batch_size": 32},
                phases=oom_partial.snapshot(), steps=[], steady=None, memory=None,
                utilization=tb, error=exc, quiet=True,
            )
        assert oom_rec["status"] == "oom"
        assert oom_rec["steady"] is None and oom_rec["memory"] is None
        assert oom_rec["phases"]["model_load_s"] == 900.0
        assert "RESOURCE_EXHAUSTED" in oom_rec["error"]
        assert "error_traceback" in oom_rec
        validate_record(json.loads(oom_path.read_text(encoding="utf-8")))
        assert classify_exception(MemoryError("cannot allocate")) == "oom"
        assert classify_exception(RuntimeError("CUDA out of memory. Tried to allocate")) == "oom"
        assert classify_exception(TimeoutError("deadline exceeded")) == "timeout"
        assert classify_exception(ValueError("bad shape")) == "error"
        assert classify_exception(None) == "ok"
        ok("emit writes a valid status='oom' record with phases preserved")

        # merge_external_phases keeps the invariant
        merged = merge_external_phases(path, submit_to_running_s=222.0, image_pull_s=11.0)
        assert merged["phases"]["submit_to_running_s"] == 222.0
        assert abs(sum(merged["phases"][k] for k in PHASE_KEYS)
                   - merged["phases"]["total_wall_s"]) < 1e-6
        validate_record(json.loads(path.read_text(encoding="utf-8")))
        ok("merge_external_phases backfills launcher-side phases and repairs the total")

        # log channel round trip (the only output path on the GPU cluster)
        buf = io.StringIO()
        buf.write("some kubectl noise\n")
        log_record(written, stream=buf)
        buf.write("I0810 21:04:11 more noise\n")
        harvested = extract_records(buf.getvalue())
        assert len(harvested) == 1 and harvested[0] == written
        assert extract_records("no sentinels here") == []
        ok("log_record/extract_records survive a round trip through interleaved log noise")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\nOK — {len(passed)} checks passed. Records conform to docs/metrics-contract.md.")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
