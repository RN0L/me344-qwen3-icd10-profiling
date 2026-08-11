#!/usr/bin/env python3
"""Environment -> argv shim for ``src/train_qwen3_icd.py``.

WHY THIS FILE EXISTS
--------------------
The three Job definitions run the trainer with **no arguments at all**::

    python3 -u ${TPU_ENTRYPOINT}          # manifests/tpu-train.yaml
    python3 -u ${GPU_ENTRYPOINT}          # manifests/gpu-train.yaml
    python3 -u ${CPU_ENTRYPOINT}          # manifests/cpu-run.sh

and ``manifests/env.sh`` resolves those to ``/app/train_tpu.py``, ``/app/train_gpu.py`` and
``/app/train_cpu.py`` with the comment "copied to /app by each Dockerfile". Everything the run
needs is passed as pod environment instead: MODEL_PATH, DATA_PATH, RESULTS_DIR, RUN_ID,
BATCH_SIZE, SEQ_LEN and the rest.

``src/train_qwen3_icd.py`` is a normal argparse program: ``--backend`` and ``--out`` are
*required* and everything else is a flag. Something has to translate one interface into the
other, and the image is where that belongs — the image is what decides what ``/app`` contains.
So each Dockerfile installs this file as ``/app/train_<its backend>.py``.

It is deliberately dependency-free (stdlib only) and it ``exec``s rather than spawns, so the
trainer inherits the pod's PID 1 slot, its signals and its exit code unchanged. It prints the
command line it resolved before handing over, because on the GH200 cluster the RBAC persona
cannot exec into pods and the pod log is the only channel through which a run can be inspected.

WHAT IT DOES NOT DO
-------------------
It invents nothing. Every value comes from an environment variable that a manifest actually
sets; anything unset is simply left off the command line so the trainer's own default applies.
No measurement is computed here except ``--submit-to-running-s``, which is a subtraction of two
clocks (``SUBMIT_EPOCH``, stamped by ``me344_submit`` at ``kubectl create`` time, and now) and is
exactly the number the manifests document themselves as producing.

Trailing arguments are passed through, so the shim never gets in the way::

    python3 /app/train_tpu.py --max-steps 5 --eval-examples 0

Because they are appended after the derived flags, argparse's last-wins behaviour makes them
overrides.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
TRAINER = APP_DIR / "src" / "train_qwen3_icd.py"

# The trainer's --backend vocabulary is telemetry.BACKENDS = ("cpu", "gpu", "tpu"). "cuda" is the
# Dockerfile/plugin name for the same row and is accepted as an alias here so that
# ME344_EXPECT_BACKEND (which is "cuda" in Dockerfile.cuda, because that is what JAX calls the
# plugin) can be used as a fallback source.
BACKEND_ALIASES = {"cpu": "cpu", "gpu": "gpu", "cuda": "gpu", "tpu": "tpu"}

# env var -> trainer flag, for values that are passed through verbatim when non-empty.
# The left column is the manifests' vocabulary; the right column is the trainer's.
VALUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("BATCH_SIZE", "--batch-size"),
    ("SEQ_LEN", "--seq-len"),
    ("MAX_STEPS", "--max-steps"),
    ("WARMUP_STEPS", "--warmup-steps"),
    ("LORA_RANK", "--lora-rank"),
    ("LORA_ALPHA", "--lora-alpha"),
    ("LEARNING_RATE", "--lr"),
    ("DTYPE", "--dtype"),
    ("REMAT", "--remat"),
    ("LORA_MODULES", "--lora-modules"),
    ("SEED", "--seed"),
    ("RUN_ID", "--run-id"),
    ("RUN_VARIANT", "--variant"),
    ("HW_TAG", "--hw-tag"),
    ("HW_LABEL", "--hw-label"),
    ("HW_CHIP_MODEL", "--chip-model"),
    ("NOTES", "--notes"),
    ("TASKS", "--tasks"),
    ("TRAIN_SPLIT", "--train-split"),
    ("EVAL_SPLIT", "--eval-split"),
    ("MAX_DOC_CHARS", "--max-doc-chars"),
    ("EVAL_EVERY_N_STEPS", "--eval-every"),
    ("EVAL_EXAMPLES", "--eval-examples"),
    ("EVAL_MAX_NEW_TOKENS", "--eval-max-new-tokens"),
    ("CKPT_DIR", "--ckpt-dir"),
    ("SAVE_INTERVAL_STEPS", "--save-interval-steps"),
    ("ADAPTER_DIR", "--adapter-out"),
    ("FILE_CACHE_CAPACITY", "--file-cache-capacity"),
    ("JAX_PLATFORMS_OVERRIDE", "--jax-platforms"),
    ("WANDB_PROJECT", "--wandb-project"),
    ("WANDB_RUN", "--wandb-run"),
    ("UTIL_INTERVAL_S", "--util-interval"),
)

# env var -> trainer flag, for switches. Set the variable to 1/true/yes to pass the flag.
BOOL_FLAGS: tuple[tuple[str, str], ...] = (
    ("SKIP_EXPORT", "--skip-export"),
    ("NO_STEP_SYNC", "--no-step-sync"),
    ("EXIT_ZERO_ON_ERROR", "--exit-zero-on-error"),
)

TRUTHY = {"1", "true", "yes", "on"}


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


def first_env(*names: str) -> str:
    for name in names:
        value = env(name)
        if value:
            return value
    return ""


def die(message: str) -> None:
    """Exit 2 with one line on stderr. Never returns; exit 2 distinguishes a launcher
    configuration error from the trainer's own exit codes (0 ok/oom/timeout, 1 error)."""
    print(f"ENTRYPOINT FATAL: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def resolve_backend() -> str:
    """cpu / gpu / tpu, from this file's installed name first.

    Each image installs this file under exactly one name, so the filename is the most reliable
    statement of which row the image is: it cannot be overridden by a stray environment variable
    inherited from the launcher, which is precisely the mistake that would make the GPU row
    report itself as a CPU run.
    """
    stem = Path(sys.argv[0]).stem
    if stem.startswith("train_"):
        candidate = stem[len("train_"):].lower()
        if candidate in BACKEND_ALIASES:
            return BACKEND_ALIASES[candidate]
    for var in ("BACKEND", "ME344_EXPECT_BACKEND"):
        value = env(var).lower()
        if value in BACKEND_ALIASES:
            return BACKEND_ALIASES[value]
    die(
        "cannot tell which backend this is. Install this file as /app/train_{cpu,gpu,tpu}.py "
        f"(argv[0] was {sys.argv[0]!r}) or set BACKEND to one of cpu, gpu, tpu."
    )


def resolve_out(run_id: str) -> str:
    """``--out``: the exact path the Job wrapper then cats back out of the pod.

    All three manifests print ``${RESULTS_DIR}/${RUN_ID}.json`` between the harvest sentinels, so
    the record has to land on that exact path and not on the trainer's own derived name.
    """
    results_dir = first_env("RESULTS_DIR", "OUT_DIR")
    if not results_dir:
        die("RESULTS_DIR is not set — there is nowhere to write the contract JSON record.")
    # On gcsfuse this is a no-op prefix creation; on the GPU/CPU rows /results is an emptyDir or
    # a bind mount that already exists. Either way the trainer must not fail on a missing dir.
    try:
        os.makedirs(results_dir, exist_ok=True)
    except OSError as exc:
        print(f"ENTRYPOINT WARN: could not create {results_dir}: {exc!r}", file=sys.stderr, flush=True)
    if run_id:
        return os.path.join(results_dir, f"{run_id}.json")
    # No RUN_ID: hand the trainer the directory and let it name the file from the run_id it
    # derives itself. Never fabricate a name here — a results file whose name does not match the
    # run_id inside it is worse than no file.
    return results_dir + os.sep


def submit_to_running_s() -> str | None:
    """Seconds between ``kubectl create`` and this process starting.

    SUBMIT_EPOCH is stamped by ``me344_submit`` (manifests/env.sh) immediately before the Job is
    created, and it covers Kueue admission plus node boot plus image pull. It is a difference of
    two real clocks, not an estimate. If the launcher later measures the split more precisely from
    kubectl event timestamps, ``telemetry.merge_external_phases()`` adjusts the record by the
    delta, so filling it in here is safe.
    """
    raw = env("SUBMIT_EPOCH")
    if not raw:
        return None
    try:
        submitted = float(raw)
    except ValueError:
        print(f"ENTRYPOINT WARN: ignoring SUBMIT_EPOCH={raw!r}: not a number", file=sys.stderr, flush=True)
        return None
    elapsed = time.time() - submitted
    if elapsed < 0:
        print(
            f"ENTRYPOINT WARN: ignoring SUBMIT_EPOCH={raw!r}: it is in the future "
            "(pod clock vs submitter clock)",
            file=sys.stderr,
            flush=True,
        )
        return None
    return repr(elapsed)


def build_argv(backend: str) -> list[str]:
    model_path = first_env("MODEL_PATH")
    if not model_path:
        die("MODEL_PATH is not set — the trainer needs the Qwen3-4B safetensors directory.")
    # DATA_ROOT is the trainer's own variable name. DATA_PATH is what all three manifests set,
    # and it points at the prepared CodiEsp tree, which prepare_data.py leaves the raw
    # <split>/text_files/ + *_annotations_task*_processed.tsv layout inside — that is the layout
    # the trainer reads. RAW_DATA_PATH (cpu-run.sh mounts the untouched drop there) is the last
    # resort, so a run is still possible before prepare_data.py has been executed.
    data_root = first_env("DATA_ROOT", "DATA_PATH", "RAW_DATA_PATH")
    if not data_root:
        die("none of DATA_ROOT / DATA_PATH / RAW_DATA_PATH is set — there is no CodiEsp corpus to train on.")

    run_id = env("RUN_ID")
    argv = [
        "--backend", backend,
        "--model-path", model_path,
        "--data-root", data_root,
        "--out", resolve_out(run_id),
    ]

    for var, flag in VALUE_FLAGS:
        value = env(var)
        if value:
            argv += [flag, value]

    # utilization.sample_interval_s: the GPU Job states it as NVIDIA_SMI_SAMPLE_INTERVAL_S. Only
    # consulted when UTIL_INTERVAL_S did not already supply one.
    if not env("UTIL_INTERVAL_S"):
        nvidia_interval = env("NVIDIA_SMI_SAMPLE_INTERVAL_S")
        if nvidia_interval:
            argv += ["--util-interval", nvidia_interval]

    for var, flag in BOOL_FLAGS:
        if env(var).lower() in TRUTHY:
            argv.append(flag)

    external = submit_to_running_s()
    if external is not None:
        argv += ["--submit-to-running-s", external]

    return argv


def main() -> int:
    backend = resolve_backend()
    if not TRAINER.is_file():
        die(f"{TRAINER} is missing — the image did not COPY src/ in.")

    argv = build_argv(backend) + sys.argv[1:]
    command = [sys.executable, "-u", str(TRAINER), *argv]

    # Print before exec: this line is the only record of how the run was parameterised on a
    # cluster where pod logs are the sole output channel.
    print("[ENTRYPOINT] backend=%s trainer=%s" % (backend, TRAINER), flush=True)
    print("[ENTRYPOINT] exec: " + " ".join(command), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    os.execv(sys.executable, command)
    return 127  # unreachable: execv only returns by raising


if __name__ == "__main__":
    raise SystemExit(main())
