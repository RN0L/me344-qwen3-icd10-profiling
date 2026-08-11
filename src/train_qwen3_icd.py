#!/usr/bin/env python3
"""LoRA fine-tune Qwen3-4B on CodiEsp (Spanish clinical cases -> ICD-10 codes).

One code path, three backends. ``--backend {cpu,gpu,tpu}`` selects the JAX platform; the mesh is
built from ``jax.device_count()`` so an 8-chip TPU v5e slice, a single GH200 and a 32-core x86 node
all run the identical workload. Every phase is timed with :mod:`telemetry` and the run emits exactly
one JSON record matching ``docs/metrics-contract.md`` to ``--out``.

Derived from the verified-working Lab 2 Gemma-3-4B Tunix script
(``10-lab-2-gemma/1-build-image.sh``), adapted to Qwen3's API:

  * ``tunix.models.qwen3.model.ModelConfig.qwen3_4b()`` (not ``gemma3_4b_it``)
  * ``Qwen3.__call__`` takes ``input_tokens`` first, not Gemma3's ``last_tokens``
  * the qwix ``module_path`` regex targets HF projection names (``q_proj|k_proj|…``), not einsums

Where the Qwen3 Tunix surface could differ from what we were told, this script introspects the real
object and asserts with expected-vs-actual rather than guessing silently.

The dataset contract belongs to ``src/prepare_data.py`` and is imported, not re-implemented: the
training text is its ``prompt + completion + tokenizer.eos_token`` recipe, right-truncated, which
is exactly what ``dataset_stats.json``'s truncation table was measured over and exactly what
``src/evaluate.py`` prompts the merged adapter with afterwards. ``--data-root`` accepts either the
prepared ``<root>/<split>.jsonl`` or the raw CodiEsp bundle, and converts the latter with
``prepare_data.build_split`` so both routes produce byte-identical prompts.

Every flag is env-defaulted (``BACKEND``, ``MODEL_PATH``, ``DATA_ROOT``/``DATA_PATH``,
``RESULTS_DIR``, ``RUN_ID``, ``BATCH_SIZE``, ``SEQ_LEN``, ``MAX_STEPS``, ``HW_*``, ``CKPT_DIR``,
``ADAPTER_DIR``, ``FILE_CACHE_CAPACITY``, ``SUBMIT_EPOCH``, …), because the Job manifests
configure the container through ``env:`` and some invoke the entrypoint with no arguments at all.
A flag on the command line always wins over the environment.

Phase mapping onto the contract's four in-process phases (``telemetry.IN_PROCESS_PHASES``)::

    model_load_s        create_model_from_safe_tensors over the FUSE mount
    compile_s           step 1, which is the XLA compile
    steady_state_s      steps 2..N
    checkpoint_write_s  LoRA adapter safetensors export
    other_s             JAX init, LoRA wrap + trace, tokenisation, trainer build, eval

``other_s`` is the residual telemetry computes, never a number this script sets. Its composition is
itemised in ``notes`` so the bottleneck analysis keeps the detail without inventing phase keys.

Every phase boundary prints a ``[PHASE-*]`` marker and the final record is echoed between
``telemetry.LOG_BEGIN``/``LOG_END`` sentinels, because the GH200 cluster's RBAC persona cannot exec
into pods and **pod logs are the only channel results can leave that container through**.

Example::

    python3 src/train_qwen3_icd.py --backend tpu \\
        --model-path /gcs/teams/team-lharnold/models/Qwen3-4B \\
        --data-root  /gcs/teams/team-lharnold/data/codiesp \\
        --batch-size 8 --seq-len 512 --max-steps 100 \\
        --out results/tpu-v5e8-bs8-seq512.json
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import io
import json
import logging
import os
import re
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import telemetry as tel  # sibling module: src/telemetry.py
except ImportError as _exc:  # pragma: no cover - a broken deployment, not a degraded one
    raise SystemExit(
        f"FATAL: cannot import the sibling module 'telemetry' from {SCRIPT_DIR}. "
        f"Expected src/telemetry.py next to this file. Underlying error: {_exc}"
    ) from _exc

try:
    # src/prepare_data.py owns the dataset contract: the prompt template, the code
    # normalisation and the `prompt + completion + tokenizer.eos_token` training recipe that
    # dataset_stats.json's truncation table is computed against. It is imported rather than
    # re-implemented so the trained adapter is evaluable by src/evaluate.py, which reads the
    # same prompts back. It is stdlib-only at import time, so this costs nothing.
    import prepare_data as prep
except ImportError as _exc:  # pragma: no cover
    raise SystemExit(
        f"FATAL: cannot import the sibling module 'prepare_data' from {SCRIPT_DIR}. "
        f"Expected src/prepare_data.py next to this file. Underlying error: {_exc}"
    ) from _exc


# ---------------------------------------------------------------------------------------------
# stdout markers — pod logs are the only output channel on the GH200 cluster.
# ---------------------------------------------------------------------------------------------

_REAL_STDOUT = sys.stdout


def say(msg: str, *args: Any) -> None:
    print(msg % args if args else msg, file=_REAL_STDOUT, flush=True)


def marker(kind: str, msg: str, *args: Any) -> None:
    say("[%s] %s", kind, msg % args if args else msg)


class SubTimer:
    """Wall-clock timing for the work that lands in ``other_s``.

    ``telemetry.PhaseTimer`` deliberately accepts only the contract's four in-process phase keys,
    so these finer-grained timings are reported in ``notes`` instead of inventing phase names.
    """

    def __init__(self) -> None:
        self.spans: dict[str, float] = {}

    def __call__(self, name: str, note: str = ""):
        return _SubSpan(self, name, note)

    def summary(self) -> str:
        if not self.spans:
            return ""
        parts = " ".join(f"{k}={v:.2f}s" for k, v in self.spans.items())
        return f"phases.other_s is composed of: {parts}"


class _SubSpan:
    def __init__(self, owner: SubTimer, name: str, note: str) -> None:
        self.owner, self.name, self.note = owner, name, note

    def __enter__(self):
        marker("PHASE-START", "%s%s", self.name, f" — {self.note}" if self.note else "")
        self.t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.monotonic() - self.t0
        self.owner.spans[self.name] = self.owner.spans.get(self.name, 0.0) + elapsed
        marker("PHASE-END", "%s %.3fs", self.name, elapsed)


# ---------------------------------------------------------------------------------------------
# Per-step loss capture
# ---------------------------------------------------------------------------------------------

_LOSS_LINE_RE = re.compile(
    r"Train step\s+(\d+)\s+training loss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"
)


class LossCollector:
    """Collects per-step training loss from three independent channels.

    1. A transparent proxy around ``trainer.metrics_logger`` when Tunix exposes one.
    2. A ``logging`` handler on the root logger.
    3. A tee on stdout/stderr matching the exact line Tunix printed in the Lab 2 run:
       ``Train step 100 training loss: 0.399344  - training perplexity: 1.490846``

    First channel to report a given step wins. If none report, the loss stays ``null`` — a
    fabricated number would be worse than a missing one.
    """

    def __init__(self) -> None:
        self.by_step: dict[int, float] = {}
        self.ordered: list[float] = []
        self.channels: set[str] = set()
        self._lock = threading.Lock()

    def observe_line(self, line: str, channel: str) -> None:
        for m in _LOSS_LINE_RE.finditer(line):
            try:
                step, loss = int(m.group(1)), float(m.group(2))
            except ValueError:
                continue
            with self._lock:
                self.by_step.setdefault(step, loss)
                self.channels.add(channel)

    def observe_metric(self, name: str, value: float, step: int | None) -> None:
        low = name.lower()
        if "loss" not in low or "eval" in low or "val" in low or "perplexity" in low:
            return
        with self._lock:
            self.channels.add("metrics_logger")
            if step is not None:
                self.by_step.setdefault(int(step), float(value))
            else:
                self.ordered.append(float(value))

    def loss_for(self, index_1based: int) -> float | None:
        with self._lock:
            if index_1based in self.by_step:
                return self.by_step[index_1based]
            if (index_1based - 1) in self.by_step:  # some loggers count from zero
                return self.by_step[index_1based - 1]
            if 0 < index_1based <= len(self.ordered):
                return self.ordered[index_1based - 1]
        return None


class _TeeStream(io.TextIOBase):
    """Pass-through stdout/stderr wrapper that scans for Tunix's per-step loss line.

    Splits on both ``\\n`` and ``\\r`` so tqdm-style progress lines are still scanned. Everything
    else delegates to the wrapped stream, so libraries that reach for the real file descriptor
    keep working.
    """

    _LINE_BREAK = re.compile(r"[\r\n]")

    def __init__(self, inner: Any, collector: LossCollector, channel: str) -> None:
        self._inner = inner
        self._collector = collector
        self._channel = channel
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        n = self._inner.write(s)
        self._buf += s
        if "\n" in self._buf or "\r" in self._buf:
            *lines, self._buf = self._LINE_BREAK.split(self._buf)
            for line in lines:
                if "training loss" in line:
                    self._collector.observe_line(line, self._channel)
        if len(self._buf) > 65536:  # a stream without line breaks must not grow unbounded
            self._buf = self._buf[-4096:]
        return n

    def writelines(self, lines: Iterable[str]) -> None:  # type: ignore[override]
        for line in lines:
            self.write(line)

    def flush(self) -> None:  # type: ignore[override]
        self._inner.flush()

    def fileno(self) -> int:  # type: ignore[override]
        return self._inner.fileno()

    def writable(self) -> bool:  # type: ignore[override]
        return True

    def isatty(self) -> bool:  # type: ignore[override]
        return bool(getattr(self._inner, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._inner, "encoding", "utf-8")


class _LossLogHandler(logging.Handler):
    def __init__(self, collector: LossCollector) -> None:
        super().__init__(level=logging.NOTSET)
        self._collector = collector

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - a logging handler must never raise
            return
        if "training loss" in msg:
            self._collector.observe_line(msg, "logging")


class _MetricsLoggerProxy:
    """Transparent proxy around Tunix's MetricsLogger that sniffs (name, value, step) on log()."""

    def __init__(self, inner: Any, collector: LossCollector) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_collector", collector)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def log(self, *args: Any, **kwargs: Any) -> Any:
        try:
            self._sniff(args, kwargs)
        except Exception:  # noqa: BLE001 - telemetry must never kill training
            pass
        return object.__getattribute__(self, "_inner").log(*args, **kwargs)

    def _sniff(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        collector: LossCollector = object.__getattribute__(self, "_collector")
        name = kwargs.get("metric_name") or kwargs.get("name")
        value = kwargs.get("value")
        step = kwargs.get("step")
        mode = kwargs.get("mode")
        rest = list(args)
        if name is None:
            for item in rest:
                if isinstance(item, str):
                    name = item
                    rest.remove(item)
                    break
        if name is None:
            return
        if value is None:
            for item in rest:
                scalar = _as_scalar(item)
                if scalar is not None:
                    value = scalar
                    rest.remove(item)
                    break
        if value is None:
            return
        if step is None:
            for item in rest:
                if isinstance(item, int) and not isinstance(item, bool):
                    step = item
                    break
        if mode is not None and "train" not in str(mode).lower():
            return
        collector.observe_metric(str(name), float(value), int(step) if step is not None else None)


def _as_scalar(obj: Any) -> float | None:
    if isinstance(obj, (bool, str)):
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    size = getattr(obj, "size", None)
    if size is not None and size != 1:
        return None
    for attr in ("item", "__float__"):
        if hasattr(obj, attr):
            try:
                return float(obj.item()) if attr == "item" else float(obj)
            except Exception:  # noqa: BLE001
                return None
    return None


# ---------------------------------------------------------------------------------------------
# CodiEsp dataset
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Example:
    """One CodiEsp document in the exact shape ``src/prepare_data.py`` writes to JSONL."""

    doc_id: str
    prompt: str
    completion: str
    codes: list[str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            assert isinstance(obj, dict), f"{path}:{lineno} is not a JSON object"
            rows.append(obj)
    return rows


def load_split(data_root: Path, split: str, limit: int = 0, seed: int = 0) -> list[Example]:
    """Load one split, preferring the prepared JSONL and falling back to the raw corpus.

    ``src/prepare_data.py`` writes ``<data_root>/<split>.jsonl`` with the keys
    ``doc_id, text, codes, prompt, completion``; that is what the Job manifests mount at
    ``$DATA_PATH`` and what ``src/evaluate.py`` scores against, so it is the primary path. If it
    is absent the raw ``train_dev`` bundle is converted in-process by ``prepare_data.build_split``
    — the same function, so the prompts are byte-identical either way rather than a second,
    divergent parser guessing at the TSV column layout.
    """
    assert data_root.is_dir(), f"--data-root is not a directory: {data_root}"
    # Lesson from Lab 2: touch the mount before globbing it, or the first lookup under a gcsfuse
    # prefix pays full first-touch latency (and, without implicit-dirs, finds nothing at all).
    entries = sorted(p.name for p in data_root.iterdir())

    jsonl = data_root / f"{split}.jsonl"
    if jsonl.is_file():
        rows = _read_jsonl(jsonl)
        source = str(jsonl)
    else:
        split_dir = data_root / split
        assert (split_dir / prep.TEXT_SUBDIR).is_dir(), (
            f"found neither the prepared {jsonl} nor the raw layout "
            f"{split_dir / prep.TEXT_SUBDIR}/*.txt. Run src/prepare_data.py first, or point "
            f"--data-root at the raw train_dev bundle. Entries under {data_root}: {entries[:20]}"
        )
        rows, provenance = prep.build_split(split_dir, split, limit=0, seed=seed)
        source = f"{split_dir} (raw CodiEsp, {provenance['annotation_file']})"

    assert rows, f"split {split!r} loaded from {source} is empty"

    examples: list[Example] = []
    for i, row in enumerate(rows):
        codes = [str(c) for c in (row.get("codes") or [])]
        text = row.get("text")
        prompt = row.get("prompt")
        if not prompt:
            assert isinstance(text, str) and text, (
                f"record {i} of {source} has neither a 'prompt' nor a 'text' field; keys: "
                f"{sorted(row)}"
            )
            prompt = prep.build_prompt(text)
        completion = row.get("completion") or prep.build_completion(codes)
        examples.append(Example(
            doc_id=str(row.get("doc_id", i)), prompt=prompt,
            completion=completion, codes=codes,
        ))
    if limit > 0:
        examples = examples[:limit]

    marker("DATA", "%s: %d example(s) from %s, mean %.1f codes/doc", split, len(examples), source,
           statistics.fmean(len(e.codes) for e in examples))
    return examples


def training_text(tokenizer: Any, ex: Example) -> str:
    """``prompt + completion + tokenizer.eos_token`` — prepare_data.TRAINING_TEXT_RECIPE.

    That string is exactly what ``dataset_stats.json``'s truncation table was computed over, so
    the reported truncation fractions describe the sequences this script really trains on.
    """
    eos = getattr(tokenizer, "eos_token", None) or ""
    return f"{ex.prompt}{ex.completion}{eos}"


# ---------------------------------------------------------------------------------------------
# Batching and per-step timing
# ---------------------------------------------------------------------------------------------


class StepSync:
    """Blocks until the accelerator has really finished the step that was just dispatched.

    ``jax.effects_barrier()`` is **not** sufficient on its own: it waits on the runtime tokens of
    *effectful* computations, and a pure Tunix train step produces no token, so on a pure step it
    returns immediately and every "step time" would be an enqueue latency. The real sync is
    ``jax.block_until_ready()`` on LoRA parameters, which are outputs of the optimiser update and
    therefore are not ready until the step is.

    Only a handful of parameters are probed (they are all outputs of the same executable, so any
    one of them is enough) and the ``nnx.Variable`` objects are captured once, because walking the
    whole graph every step would add its own cost to the number being measured. If Tunix ever
    rebuilds the model object instead of updating the variables in place the captured probes would
    go stale — that is detected by watching whether the underlying arrays change identity, and
    reported in ``notes`` rather than silently producing dispatch-latency numbers.
    """

    def __init__(self, jax_mod: Any, nnx_mod: Any, model: Any, n_probes: int = 4) -> None:
        self._jax = jax_mod
        self._barrier = getattr(jax_mod, "effects_barrier", None)
        variables = [v for _, v in nnx_mod.iter_graph(model)
                     if isinstance(v, nnx_mod.LoRAParam)]
        if variables and n_probes > 0:
            stride = max(1, len(variables) // n_probes)
            self._probes = variables[::stride][:n_probes]
        else:
            self._probes = variables
        self._last_ids: tuple[int, ...] = ()
        self.n_calls = 0
        self.n_unchanged = 0
        self.failures = 0

    @property
    def n_probes(self) -> int:
        return len(self._probes)

    def __call__(self) -> None:
        try:
            values = [p.value for p in self._probes]
            ids = tuple(id(v) for v in values)
            if values:
                self._jax.block_until_ready(values)
            if self._barrier is not None:
                self._barrier()
        except Exception:  # noqa: BLE001 - a broken barrier must never kill training
            self.failures += 1
            return
        if self._last_ids and ids == self._last_ids:
            self.n_unchanged += 1
        self._last_ids = ids
        self.n_calls += 1

    def notes(self) -> list[str]:
        out: list[str] = []
        if not self._probes:
            out.append("per-step sync found no nnx.LoRAParam to block on; step times may reflect "
                       "async dispatch rather than device completion")
        if self.failures:
            out.append(f"the per-step device sync raised on {self.failures} step boundary/ies; "
                       "those step times may reflect async dispatch rather than completion")
        if self.n_calls > 2 and self.n_unchanged >= self.n_calls - 1:
            out.append("the probed LoRA parameter arrays never changed identity across steps, so "
                       "jax.block_until_ready() may have been blocking on a stale buffer; treat "
                       "the per-step distribution (p10/p90) as indicative and steady.median_step_s "
                       "as the reliable number")
        return out


class TimedDataset:
    """A materialised list of Tunix batches that timestamps every pull, after a device sync.

    The trainer's loop is ``for batch in ds: train_step(batch)``, so the interval between
    successive pulls is one full step — no dependency on Tunix internals. JAX dispatch is async,
    so ``sync_fn`` (see :class:`StepSync`) runs *before* each timestamp; without it the loop would
    race ahead of the accelerator and the numbers would measure enqueue latency instead of
    compute. ``__len__``/``__getitem__`` are provided so the trainer can treat this as a sequence;
    multi-epoch iteration simply appends more timestamps.

    The closing mark is taken in the generator's ``finally``, which CPython runs as soon as the
    trainer breaks out of the loop. That matters: taking it after ``trainer.train()`` returns
    would fold the trainer's teardown — an Orbax checkpoint save, for one — into the last step.
    """

    def __init__(self, batches: list[dict[str, Any]], sync_fn: Any | None = None) -> None:
        self._batches = batches
        self._sync_fn = sync_fn
        self.pull_times: list[float] = []
        self.end_time: float | None = None
        self.sync_failures = 0

    def _sync(self) -> None:
        if self._sync_fn is None:
            return
        try:
            self._sync_fn()
        except Exception:  # noqa: BLE001 - a broken barrier must not kill training
            self.sync_failures += 1

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._batches[idx]

    def __iter__(self):
        self.end_time = None  # a fresh epoch reopens the interval
        try:
            for batch in self._batches:
                self._sync()
                self.pull_times.append(time.monotonic())
                yield batch
        finally:
            self.close()

    def close(self) -> None:
        """Take the closing mark. Idempotent — only the first call after a pull counts."""
        if self.end_time is not None:
            return
        self._sync()
        self.end_time = time.monotonic()

    def step_durations(self) -> list[float]:
        marks = list(self.pull_times)
        if marks and self.end_time is not None and self.end_time > marks[-1]:
            marks.append(self.end_time)
        return [marks[i + 1] - marks[i] for i in range(len(marks) - 1)]


def build_batches(
    tokenizer: Any,
    examples: list[Example],
    batch_size: int,
    seq_len: int,
    max_batches: int,
) -> list[dict[str, Any]]:
    import jax.numpy as jnp
    from tunix.sft import utils as sft_utils

    texts = [training_text(tokenizer, ex) for ex in examples]
    enc = tokenizer(
        texts,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    tokens, mask = enc["input_ids"], enc["attention_mask"].astype(bool)
    n_full = len(tokens) // batch_size
    if max_batches > 0:
        n_full = min(n_full, max_batches)
    assert n_full > 0, (
        f"not enough examples for a single batch: {len(tokens)} examples, batch_size={batch_size}"
    )

    batches = []
    for i in range(n_full):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        tok = jnp.asarray(tokens[sl], dtype=jnp.int32)
        m = jnp.asarray(mask[sl], dtype=jnp.bool_)
        batches.append({
            "input_tokens": tok,
            "input_mask": m,
            "positions": sft_utils.build_positions_from_mask(m),
            "attention_mask": sft_utils.make_causal_attn_mask(m),
        })
    return batches


# ---------------------------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------------------------


def _import_qwen3():
    try:
        from tunix.models.qwen3 import model as qwen3_lib
    except ImportError as exc:
        import pkgutil

        try:
            import tunix.models as tm

            available = sorted(m.name for m in pkgutil.iter_modules(tm.__path__))
        except Exception:  # noqa: BLE001
            available = ["<tunix.models not importable>"]
        raise AssertionError(
            f"expected module 'tunix.models.qwen3.model'; actual tunix.models submodules: "
            f"{available}. Underlying import error: {exc}"
        ) from exc
    # Qwen3 does NOT mirror gemma3's layout: there is no `params_safetensors` submodule.
    # Verified against tunix @ c00f424 on 2026-08-10 — `tunix.models.qwen3` exposes exactly
    # ['mapping_sglang_jax', 'mapping_vllm_jax', 'model', 'params'], and the loader
    # `create_model_from_safe_tensors` lives in `params`. Prefer that; fall back to the gemma3
    # spelling only in case a later tunix adds it.
    try:
        from tunix.models.qwen3 import params as qwen3_params
    except ImportError as exc:
        raise AssertionError(
            "expected module 'tunix.models.qwen3.params' (holds create_model_from_safe_tensors "
            f"for Qwen3); underlying import error: {exc}"
        ) from exc

    qwen3_params_st = qwen3_params
    if not hasattr(qwen3_params_st, "create_model_from_safe_tensors"):
        try:
            from tunix.models.qwen3 import params_safetensors as _st  # gemma3-style fallback

            qwen3_params_st = _st
        except ImportError:
            pass
    assert hasattr(qwen3_params_st, "create_model_from_safe_tensors"), (
        "expected create_model_from_safe_tensors in tunix.models.qwen3.params (or "
        f".params_safetensors); actual params attributes: "
        f"{sorted(a for a in dir(qwen3_params) if not a.startswith('_'))}"
    )
    return qwen3_lib, qwen3_params_st, qwen3_params


def _field_names(obj: Any) -> set[str]:
    return {f.name for f in dataclasses.fields(obj)} if dataclasses.is_dataclass(obj) else set()


def _dtype_from_name(name: str):
    import jax.numpy as jnp

    table = {
        "bfloat16": jnp.bfloat16, "bf16": jnp.bfloat16,
        "float32": jnp.float32, "fp32": jnp.float32, "f32": jnp.float32,
        "float16": jnp.float16, "fp16": jnp.float16, "f16": jnp.float16,
    }
    assert name in table, f"unsupported --dtype {name!r}; expected one of {sorted(table)}"
    return table[name]


def apply_dtype(config: Any, dtype_name: str) -> tuple[Any, list[str]]:
    """Set every dtype-ish field the Tunix ModelConfig actually declares. Never guesses a name."""
    dtype = _dtype_from_name(dtype_name)
    fields = _field_names(config)
    updates = {
        c: dtype
        for c in ("dtype", "param_dtype", "compute_dtype", "weight_dtype", "activation_dtype")
        if c in fields
    }
    if updates:
        return dataclasses.replace(config, **updates), sorted(updates)
    marker("WARN", "ModelConfig declares no dtype field (fields=%s) — --dtype %s is recorded in the "
                   "run record but not applied to the model config", sorted(fields), dtype_name)
    return config, []


def apply_vocab_override(config: Any, model_path: Path) -> Any:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(str(model_path))
    hf_vocab = getattr(getattr(hf_config, "text_config", hf_config), "vocab_size", None)
    if hf_vocab is None:
        marker("WARN", "HF config reports no vocab_size — leaving the Tunix vocab as-is")
        return config
    fields = _field_names(config)
    name = next((c for c in ("num_embed", "vocab_size", "n_vocab") if c in fields), None)
    assert name is not None, (
        "expected the Qwen3 ModelConfig to declare one of num_embed/vocab_size/n_vocab; actual "
        f"fields: {sorted(fields)}"
    )
    current = getattr(config, name)
    if current != hf_vocab:
        marker("MODEL", "overriding %s: %d -> checkpoint vocab_size %d", name, current, hf_vocab)
        config = dataclasses.replace(config, **{name: hf_vocab})
    return config


def apply_remat(config: Any, model: Any, qwen3_lib: Any, remat_name: str) -> Any:
    remat_enum = getattr(qwen3_lib, "RematConfig", None)
    assert remat_enum is not None, (
        "expected tunix.models.qwen3.model.RematConfig; actual module attributes: "
        f"{sorted(a for a in dir(qwen3_lib) if not a.startswith('_'))}"
    )
    member = getattr(remat_enum, remat_name, None)
    assert member is not None, (
        f"expected RematConfig.{remat_name}; actual members: "
        f"{[m for m in dir(remat_enum) if m.isupper()]}"
    )
    fields = _field_names(config)
    field = next((c for c in ("remat_config", "remat", "rematerialization") if c in fields), None)
    assert field is not None, (
        f"expected the Qwen3 ModelConfig to declare a remat field; actual fields: {sorted(fields)}"
    )
    new_config = dataclasses.replace(config, **{field: member})
    model.config = new_config
    layers_attr = next((a for a in ("layers", "blocks", "decoder_layers", "h") if hasattr(model, a)), None)
    assert layers_attr is not None, (
        "expected the LoRA-wrapped Qwen3 model to expose its decoder stack as one of "
        "layers/blocks/decoder_layers/h; actual attributes: "
        f"{sorted(a for a in dir(model) if not a.startswith('_'))[:40]}"
    )
    n = 0
    for layer in getattr(model, layers_attr):
        layer.config = new_config
        n += 1
    marker("MODEL", "RematConfig.%s applied to model.%s (%d layers)", remat_name, layers_attr, n)
    return new_config


def token_kwarg_name(model_cls: Any) -> str:
    """Qwen3.__call__ takes ``input_tokens`` first; Gemma3 took ``last_tokens``. Read, don't guess."""
    sig = inspect.signature(model_cls.__call__)
    params = [p for p in sig.parameters if p != "self"]
    assert params, f"{model_cls.__name__}.__call__ has no parameters besides self"
    first = params[0]
    assert first in ("input_tokens", "last_tokens", "tokens", "input_ids"), (
        f"expected {model_cls.__name__}.__call__ first argument to be 'input_tokens' (Qwen3) or "
        f"'last_tokens' (Gemma3); actual signature: {sig}"
    )
    if first != "input_tokens":
        marker("WARN", "%s.__call__ first arg is %r, not the documented 'input_tokens' — adapting. "
                       "Full signature: %s", model_cls.__name__, first, sig)
    for required in ("positions", "cache", "attention_mask"):
        assert required in sig.parameters, (
            f"expected {model_cls.__name__}.__call__ to accept {required!r}; actual signature: {sig}"
        )
    return first


# ---------------------------------------------------------------------------------------------
# LoRA adapter export
# ---------------------------------------------------------------------------------------------

_PROJ_TO_BLOCK = {
    "q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
    "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp",
}
_LAYER_IDX_RE = re.compile(r"(?:^|[./_])(\d+)(?:[./_]|$)")


def _nnx_path_to_hf_key(path_str: str) -> str:
    """Map a Tunix/nnx module path onto a PEFT key without relying on Tunix's internal naming.

    Only the seven HF projections are ever wrapped and each one determines its parent block
    unambiguously, so ``…/layers/7/attn/q_proj`` -> ``…layers.7.self_attn.q_proj``.
    """
    proj = next((p for p in _PROJ_TO_BLOCK if p in path_str), None)
    assert proj is not None, (
        f"LoRA path {path_str!r} contains none of {sorted(_PROJ_TO_BLOCK)} — the module_path regex "
        "matched something unexpected and the adapter cannot be written in PEFT layout"
    )
    indices = _LAYER_IDX_RE.findall(path_str.split(proj)[0])
    assert indices, f"no layer index found in LoRA path {path_str!r}"
    return f"base_model.model.model.layers.{int(indices[-1])}.{_PROJ_TO_BLOCK[proj]}.{proj}"


def export_lora_adapter(
    model: Any, out_dir: Path, rank: int, alpha: float, base_model_name: str, qwen3_params: Any
) -> int:
    import numpy as np
    from flax import nnx
    from safetensors.numpy import save_file

    try:
        from tunix.models.safetensors_saver import join_path
    except ImportError:
        def join_path(parts):  # type: ignore[misc]
            return "/".join(str(p) for p in parts)

    lora_layers: dict[str, list[Any]] = {}
    for path, value in nnx.iter_graph(model):
        if isinstance(value, nnx.LoRAParam):
            path_str, leaf = join_path(path[:-1]), str(path[-1])
            if path_str in lora_layers:
                assert "lora_b" in leaf.lower(), f"expected lora_b second at {path_str}, got {leaf}"
                lora_layers[path_str].append(np.asarray(value.value))
            else:
                assert "lora_a" in leaf.lower(), f"expected lora_a first at {path_str}, got {leaf}"
                lora_layers[path_str] = [np.asarray(value.value)]
    assert lora_layers, "no nnx.LoRAParam found on the trained model — nothing to export"

    def to_peft(kind: str, tensor: Any):
        arr = np.asarray(tensor)
        return arr.reshape(-1, rank).T if kind == "lora_A" else arr.reshape(rank, -1).T

    extractor = getattr(qwen3_params, "_extract_qwen3_lora_layers", None) if qwen3_params else None
    key_mapper = getattr(qwen3_params, "_qwen3_state_key_to_safetensors_key", None) if qwen3_params else None

    adapter: dict[str, Any] = {}
    if extractor is not None and key_mapper is not None:
        marker("EXPORT", "using the tunix.models.qwen3.params key mapper")
        for key, (lora_a, lora_b) in extractor(lora_layers).items():
            hf = key_mapper(key).removesuffix(".weight")
            adapter[f"{hf}.lora_A.weight"] = np.ascontiguousarray(to_peft("lora_A", lora_a))
            adapter[f"{hf}.lora_B.weight"] = np.ascontiguousarray(to_peft("lora_B", lora_b))
    else:
        marker("EXPORT", "tunix.models.qwen3.params exposes no key mapper — deriving PEFT keys "
                         "from projection names")
        for path_str, tensors in lora_layers.items():
            assert len(tensors) == 2, (
                f"expected exactly lora_a and lora_b at {path_str}, got {len(tensors)} tensors"
            )
            hf = _nnx_path_to_hf_key(path_str)
            adapter[f"{hf}.lora_A.weight"] = np.ascontiguousarray(to_peft("lora_A", tensors[0]))
            adapter[f"{hf}.lora_B.weight"] = np.ascontiguousarray(to_peft("lora_B", tensors[1]))

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(adapter, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(
        json.dumps({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "fan_in_fan_out": False,
            "base_model_name_or_path": base_model_name,
            "target_modules": sorted(_PROJ_TO_BLOCK),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(adapter)


# ---------------------------------------------------------------------------------------------
# Optional functional eval (micro-F1 over predicted ICD-10 code sets)
# ---------------------------------------------------------------------------------------------

_CODE_SPLIT_RE = re.compile(r"[,\s;]+")
#: A CodiEsp code cell is whitespace-free (prepare_data.normalize_code enforces exactly that) and
#: may contain '/' — the CIE-O morphology entries look like ``8550/3``.
_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,17}$")


def _parse_codes(text: str) -> set[str]:
    out = set()
    for tok in _CODE_SPLIT_RE.split(text.strip().lower()):
        tok = tok.strip(" .;:•-")
        # prepare_data.EMPTY_COMPLETION: the model saying "none" is an empty code set, not a code.
        if not tok or tok == prep.EMPTY_COMPLETION:
            continue
        if _CODE_RE.match(tok):
            out.add(tok)
    return out


def greedy_eval(
    model: Any, mesh_ctx: Any, mesh: Any, tokenizer: Any, examples: list[Example],
    seq_len: int, max_new_tokens: int, token_kw: str,
) -> dict[str, Any]:
    """Fixed-shape greedy decode: one XLA compile, then one full forward per generated token.

    Deliberately cache-free — the Tunix KV-cache API is not among the things we verified for
    Qwen3, and this is a functional sanity check, not a graded metric. Costly, hence opt-in.
    """
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from tunix.sft import utils as sft_utils

    def _forward(m, tokens, positions, attn_mask, idx):
        out = m(**{token_kw: tokens}, positions=positions, cache=None, attention_mask=attn_mask)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return jnp.take(logits[0], idx, axis=0)

    forward = nnx.jit(_forward)

    eos_ids = set()
    for name in ("<|endoftext|>", "<|im_end|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(name)
        except Exception:  # noqa: BLE001
            tid = None
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    tp = fp = fn = 0
    n_done = 0
    budget = seq_len - max_new_tokens
    assert budget > 0, f"--seq-len {seq_len} must exceed --eval-max-new-tokens {max_new_tokens}"

    with mesh_ctx(mesh):
        for ex in examples:
            # Tail of the prompt, not the head: prepare_data's prompt ends with the
            # "### ICD-10 diagnosis codes" delimiter, and dropping that would ask the model to
            # continue the case report instead of answering it.
            ids = tokenizer(ex.prompt)["input_ids"][-budget:]
            buf = np.full((1, seq_len), pad_id, dtype=np.int32)
            msk = np.zeros((1, seq_len), dtype=bool)
            buf[0, : len(ids)] = np.asarray(ids, dtype=np.int32)
            msk[0, : len(ids)] = True
            cur = len(ids)
            generated: list[int] = []
            for _ in range(max_new_tokens):
                jm = jnp.asarray(msk)
                row = forward(
                    model, jnp.asarray(buf),
                    sft_utils.build_positions_from_mask(jm),
                    sft_utils.make_causal_attn_mask(jm),
                    jnp.asarray(cur - 1, dtype=jnp.int32),
                )
                nxt = int(np.asarray(jnp.argmax(row)))
                if nxt in eos_ids or cur >= seq_len:
                    break
                generated.append(nxt)
                buf[0, cur], msk[0, cur] = nxt, True
                cur += 1
            pred = _parse_codes(tokenizer.decode(generated, skip_special_tokens=True))
            gold = set(ex.codes)
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
            n_done += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "micro_f1": f1,
        "n_examples": n_done,
        "note": (f"functional sanity check only, not a graded metric; greedy decode of "
                 f"{max_new_tokens} tokens, no KV cache"),
        "micro_precision": precision,
        "micro_recall": recall,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


# ---------------------------------------------------------------------------------------------
# Hardware labelling
# ---------------------------------------------------------------------------------------------


def derive_hw_tag(backend: str, hw: dict[str, Any]) -> str:
    """The ``<hardware>`` slug of the run-id convention: v5e8, gh200, x86-32c."""
    chips = int(hw.get("chips") or 1)
    model = str(hw.get("chip_model") or "").lower()
    if backend == "tpu":
        gen = "v5e" if ("v5" in model and "lite" in model) else None
        if gen is None:
            m = re.search(r"v(\d+)\s*([a-z]*)", model)
            gen = f"v{m.group(1)}{m.group(2)[:1]}" if m else re.sub(r"[^a-z0-9]+", "", model) or "tpu"
        return f"{gen}{chips}"
    if backend == "gpu":
        m = re.search(r"(gh\d+|gb\d+|h\d{3}|a\d{3}|l\d+|v\d{3}|t\d+)", model)
        tag = m.group(1) if m else (re.sub(r"[^a-z0-9]+", "", model.split()[-1]) if model else "gpu")
        return f"{tag}x{chips}" if chips > 1 else tag
    arch = str(hw.get("host_arch") or "cpu").lower()
    arch = "x86" if arch in ("x86_64", "amd64") else ("arm" if "arm" in arch or "aarch" in arch else arch)
    return f"{arch}-{chips}c"


def detect_hardware(args: argparse.Namespace) -> dict[str, Any]:
    """The contract's ``hardware`` block: live detection, with the launcher's ``HW_*`` on top.

    Used on both the normal path and the crash path in :func:`main`, so a run that dies during
    JAX init still reports the hardware the manifest said it asked for.
    """
    overrides: dict[str, Any] = {}
    if args.chip_model:
        overrides["chip_model"] = args.chip_model
    if args.chips > 0:
        overrides["chips"] = args.chips
    if args.host_arch:
        overrides["host_arch"] = args.host_arch
    block = tel.hardware_block(args.backend, label=args.hw_label or "pending", **overrides)
    if not args.hw_label:
        block["label"] = derive_hw_label(args.backend, block)
    return block


def derive_hw_label(backend: str, hw: dict[str, Any]) -> str:
    chips = hw.get("chips")
    model = hw.get("chip_model") or "unknown"
    if backend == "tpu":
        return f"TPU {model} x{chips}"
    if backend == "gpu":
        return f"{model}" if chips == 1 else f"{model} x{chips}"
    return f"CPU {chips} cores {hw.get('host_arch')} ({model})"


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def env_str(*names: str, default: str = "") -> str:
    """First non-empty environment variable among ``names``, else ``default``.

    Every flag below is env-defaulted because the three Job manifests configure the container
    entirely through ``env:`` and two of them still invoke the entrypoint with no arguments at
    all. A flag on the command line always wins; the environment is only the default.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def env_num(kind: Any, *names: str, default: Any) -> Any:
    raw = env_str(*names)
    if not raw:
        return default
    try:
        return kind(raw)
    except (TypeError, ValueError):
        marker("WARN", "ignoring %s=%r: not a %s", names[0], raw, kind.__name__)
        return default


def default_results_dir() -> str:
    """``$RESULTS_DIR`` if the launcher set one, else the contract's ``results/`` beside src/."""
    return env_str("RESULTS_DIR", default=str(SCRIPT_DIR.parent / "results"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_qwen3_icd.py",
        description="LoRA fine-tune Qwen3-4B on CodiEsp across CPU / GPU / TPU with one code path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", default=env_str("BACKEND", "ME344_EXPECT_BACKEND"),
                   choices=tel.BACKENDS)
    p.add_argument("--jax-platforms", default="",
                   help="override JAX_PLATFORMS; empty derives it from --backend")

    p.add_argument("--model-path", default=env_str("MODEL_PATH"),
                   help="Qwen3 HF safetensors dir, e.g. /gcs/teams/$TEAM/models/Qwen3-4B")
    p.add_argument("--model-config", default=env_str("MODEL_CONFIG") or "qwen3_4b",
                   help="tunix ModelConfig factory name; must match the checkpoint at "
                        "--model-path. qwen3_4b for Qwen3-4B, qwen3_0p6b for Qwen3-0.6B.")
    p.add_argument("--data-root", default=env_str("DATA_ROOT", "DATA_PATH"),
                   help="dataset root: <root>/<split>.jsonl from src/prepare_data.py, or the raw "
                        "CodiEsp bundle (<root>/<split>/text_files/*.txt + *_processed.tsv)")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="dev")
    p.add_argument("--max-train-examples", type=int, default=0,
                   help="head-slice the train split to this many documents; 0 uses all of them")

    p.add_argument("--batch-size", type=int,
                   default=env_num(int, "BATCH_SIZE", "BATCH_SIZE_OVERRIDE", default=8))
    p.add_argument("--seq-len", type=int, default=env_num(int, "SEQ_LEN", default=512))
    p.add_argument("--lora-rank", type=int, default=env_num(int, "LORA_RANK", default=32))
    p.add_argument("--lora-alpha", type=float, default=env_num(float, "LORA_ALPHA", default=64.0))
    p.add_argument("--lr", type=float, default=env_num(float, "LEARNING_RATE", default=1e-3))
    p.add_argument("--max-steps", type=int, default=env_num(int, "MAX_STEPS", default=100))
    p.add_argument("--dtype", default=env_str("DTYPE", default="bfloat16"),
                   choices=("bfloat16", "bf16", "float32", "fp32", "f32", "float16", "fp16", "f16"))
    p.add_argument("--remat", default=env_str("REMAT", default="DECODER"),
                   choices=("NONE", "BLOCK", "DECODER"))
    p.add_argument("--lora-modules",
                   default=".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj",
                   help="qwix LoraProvider module_path regex (Qwen3 uses HF projection names)")
    p.add_argument("--seed", type=int, default=env_num(int, "SEED", default=0))

    p.add_argument("--out", default=default_results_dir(),
                   help="path of the contract JSON record; a directory becomes <dir>/<run_id>.json")
    p.add_argument("--run-id", default=env_str("RUN_ID"),
                   help="empty derives it from backend/hardware/bs/seq")
    p.add_argument("--variant", default=env_str("RUN_VARIANT"),
                   help="run-id suffix, e.g. 'filecache'")
    p.add_argument("--hw-tag", default=env_str("HW_TAG"),
                   help="override the run-id hardware slug, e.g. v5e8")
    p.add_argument("--hw-label", default=env_str("HW_LABEL"),
                   help="override hardware.label, e.g. 'TPU v5e 2x4'")
    p.add_argument("--chip-model", default=env_str("HW_CHIP_MODEL"),
                   help="override hardware.chip_model, e.g. tpu-v5-lite-podslice")
    p.add_argument("--chips", type=int, default=env_num(int, "HW_CHIPS", default=0),
                   help="override hardware.chips; 0 keeps the detected value")
    p.add_argument("--host-arch", default=env_str("HW_HOST_ARCH"),
                   help="override hardware.host_arch, e.g. aarch64 on the GH200's Grace host")
    p.add_argument("--model-name", default=env_str("MODEL_NAME", default="Qwen/Qwen3-4B"),
                   help="recorded verbatim in config.model")
    p.add_argument("--notes", default="")

    p.add_argument("--warmup-steps", type=int, default=env_num(int, "WARMUP_STEPS", default=10),
                   help="steps excluded from the steady-state statistics")
    p.add_argument("--util-interval", type=float,
                   default=env_num(float, "UTIL_INTERVAL_S", default=1.0))
    p.add_argument("--no-step-sync", action="store_true",
                   help="do not block on the device between steps (faster, but step times then "
                        "measure async dispatch rather than completion)")
    p.add_argument("--eval-every", type=int, default=env_num(int, "EVAL_EVERY_N_STEPS", default=0),
                   help="Tunix eval_every_n_steps; 0 disables in-loop eval so step times stay clean")
    p.add_argument("--eval-examples", type=int, default=env_num(int, "EVAL_EXAMPLES", default=0),
                   help="greedy-decode this many dev cases for a micro-F1 sanity check; 0 skips")
    p.add_argument("--eval-max-new-tokens", type=int,
                   default=env_num(int, "EVAL_MAX_NEW_TOKENS", default=24))

    p.add_argument("--ckpt-dir", default=env_str("CKPT_DIR"),
                   help="Orbax checkpoint root; empty disables it")
    p.add_argument("--save-interval-steps", type=int, default=0, help="0 means only at the end")
    p.add_argument("--adapter-out", default=env_str("ADAPTER_DIR"),
                   help="adapter dir; empty means <out dir>/adapters/<run_id>")
    p.add_argument("--skip-export", action="store_true", help="skip the safetensors adapter export")

    p.add_argument("--file-cache-capacity", default=env_str("FILE_CACHE_CAPACITY", "FILE_CACHE",
                                                            default="none"),
                   help="recorded in config.file_cache_capacity; the CSI attribute lives in the manifest")
    p.add_argument("--submit-to-running-s", type=float, default=None,
                   help="externally measured kubectl submit->Running seconds")
    p.add_argument("--image-pull-s", type=float, default=None,
                   help="externally measured image pull seconds")

    p.add_argument("--wandb-project", default=env_str("WANDB_PROJECT",
                                                      default="me344-final-project"))
    p.add_argument("--wandb-run", default="", help="empty uses the run_id")
    p.add_argument("--exit-zero-on-error", action="store_true",
                   help="exit 0 even on status=error (oom and timeout always exit 0)")

    args = p.parse_args(argv)
    if not args.backend:
        p.error("--backend is required (or set BACKEND in the environment)")
    if not args.model_path:
        p.error("--model-path is required (or set MODEL_PATH in the environment)")
    if not args.data_root:
        p.error("--data-root is required (or set DATA_ROOT / DATA_PATH in the environment)")
    for name in ("batch_size", "seq_len", "max_steps", "lora_rank"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive, got {getattr(args, name)}")
    if args.warmup_steps < 0:
        p.error("--warmup-steps must be >= 0")

    # phases.submit_to_running_s measured from inside the container when the launcher stamped
    # SUBMIT_EPOCH at `kubectl create` time and did not pass the number explicitly. It is
    # inclusive of the image pull, which scripts/collect-phases.sh later splits out via
    # telemetry.merge_external_phases(); until then a real number beats a null.
    if args.submit_to_running_s is None:
        epoch = env_num(float, "SUBMIT_EPOCH", default=None)
        if epoch is not None:
            args.submit_to_running_s = max(0.0, time.time() - epoch)
            marker("ENV", "submit_to_running_s=%.1fs derived from SUBMIT_EPOCH (inclusive of the "
                          "image pull until collect-phases.sh splits it out)",
                   args.submit_to_running_s)
    return args


def resolve_out(out: str) -> str:
    """Normalise ``--out`` into something ``telemetry.emit`` cannot misread, and create it.

    ``emit`` names the file ``<run_id>.json`` only when the path already **is** a directory or
    ends in a separator. ``--out /gcs/teams/$TEAM/results`` on a fresh mount is neither, so
    without this it would silently write a *file* called ``results``. Anything without a ``.json``
    suffix is therefore treated as a directory and created up front.
    """
    path = Path(out)
    if path.suffix.lower() == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path) + os.sep


def out_dir_for(out: str) -> Path:
    """``--out`` may name a file or a directory; adapters and W&B logs go beside it either way."""
    path = Path(out)
    return path if (str(out).endswith(os.sep) or path.is_dir()) else path.parent


def configure_backend_env(backend: str, override: str) -> None:
    platforms = override or {"cpu": "cpu", "gpu": "cuda", "tpu": "tpu"}[backend]
    os.environ["JAX_PLATFORMS"] = platforms
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    marker("ENV", "JAX_PLATFORMS=%s backend=%s", platforms, backend)


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------


class Run:
    """Mutable state shared between :func:`train` and the emit path in :func:`main`."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.timer = tel.PhaseTimer(backend=args.backend)
        self.sub = SubTimer()
        self.sampler = tel.UtilizationSampler(args.backend, interval_s=args.util_interval)
        self.recorder = tel.StepRecorder(
            batch_size=args.batch_size, seq_len=args.seq_len, warmup_steps=args.warmup_steps
        )
        self.hardware: dict[str, Any] = {}
        # A usable run_id before JAX has been imported, so a crash during init still emits a
        # record with a meaningful name. Replaced once the hardware has actually been detected.
        self.run_id: str = args.run_id or tel.make_run_id(
            args.backend, args.hw_tag or "unknownhw", args.batch_size, args.seq_len,
            args.variant or None,
        )
        self.eval: dict[str, Any] | None = None
        self.notes: list[str] = [args.notes] if args.notes else []
        self.trainable_params: int | None = None
        self.dtype_fields: list[str] = []
        self.n_chips: int | None = None

        if args.submit_to_running_s is not None:
            self.timer.set_external("submit_to_running_s", args.submit_to_running_s)
        if args.image_pull_s is not None:
            self.timer.set_external("image_pull_s", args.image_pull_s)

    def note(self, msg: str) -> None:
        marker("NOTE", "%s", msg)
        self.notes.append(msg)

    def config_block(self) -> dict[str, Any]:
        a = self.args
        return {
            "model": a.model_name,
            "batch_size": a.batch_size,
            "seq_len": a.seq_len,
            "lora_rank": a.lora_rank,
            "lora_alpha": a.lora_alpha,
            "dtype": a.dtype,
            "max_steps": a.max_steps,
            "remat": a.remat,
            "file_cache_capacity": a.file_cache_capacity,
            "lr": a.lr,
            "seed": a.seed,
            "lora_modules": a.lora_modules,
            "warmup_steps": a.warmup_steps,
            "model_path": a.model_path,
            "data_root": a.data_root,
            "dtype_fields_applied": self.dtype_fields,
            "trainable_lora_params": self.trainable_params,
            "devices": self.n_chips,
            "step_sync": not a.no_step_sync,
        }


def train(run: Run) -> None:  # noqa: C901 - a linear pipeline reads better than split fragments
    a = run.args
    sub = run.sub

    # -- JAX init and mesh --------------------------------------------------------------------
    with sub("jax_init", "importing JAX/Tunix and building the device mesh"):
        import jax
        import jax.numpy as jnp
        import optax
        import qwix
        from flax import nnx
        from transformers import AutoTokenizer
        from tunix.sft import peft_trainer
        from tunix.sft.metrics_logger import MetricsLoggerOptions

        qwen3_lib, qwen3_params_st, qwen3_params = _import_qwen3()

        devices = jax.devices()
        assert devices, "JAX reports zero devices"
        run.n_chips = jax.device_count()
        marker("DEVICES", "%d device(s): platform=%s kind=%s coords=%s", run.n_chips,
               devices[0].platform, getattr(devices[0], "device_kind", "?"),
               [getattr(d, "coords", None) for d in devices])

        run.hardware = detect_hardware(a)
        # The record must state which model actually ran. --model-name defaults to Qwen3-4B, so a
        # 0.6B run launched with only --model-config would otherwise be filed as a 4B result and
        # silently corrupt every downstream comparison. Derive it from the config factory whenever
        # the two disagree, and carry the size into the run_id so filenames stay distinguishable.
        _SIZE_FROM_CONFIG = {"qwen3_0p6b": "0.6B", "qwen3_1p7b": "1.7B", "qwen3_4b": "4B",
                             "qwen3_8b": "8B", "qwen3_14b": "14B", "qwen3_32b": "32B"}
        size = _SIZE_FROM_CONFIG.get(a.model_config)
        if size and size.lower() not in a.model_name.lower():
            corrected = f"Qwen/Qwen3-{size}"
            marker("RUN", "--model-name %r disagrees with --model-config %r; recording %r",
                   a.model_name, a.model_config, corrected)
            a.model_name = corrected
        hw_tag = a.hw_tag or derive_hw_tag(a.backend, run.hardware)
        if size and size != "4B":
            hw_tag = f"{hw_tag}-{size.replace('.', 'p').lower()}"
        run.run_id = a.run_id or tel.make_run_id(
            a.backend, hw_tag, a.batch_size, a.seq_len, a.variant or None,
        )
        marker("RUN", "run_id=%s hardware=%s", run.run_id, run.hardware["label"])

        mesh_shape, axis_names = (jax.device_count(), 1), ("fsdp", "tp")
        mesh = jax.make_mesh(
            mesh_shape, axis_names, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_shape)
        )
        mesh_ctx = getattr(jax, "set_mesh", None) or getattr(jax.sharding, "use_mesh", None)
        assert mesh_ctx is not None, (
            "expected jax.set_mesh or jax.sharding.use_mesh; actual jax version "
            f"{getattr(jax, '__version__', 'unknown')}"
        )
        marker("MESH", "%s", dict(zip(axis_names, mesh_shape)))

    run.sampler.start()
    marker("TELEMETRY", "utilization sampler started (interval %.2fs)", a.util_interval)

    # -- model load ---------------------------------------------------------------------------
    model_path = Path(a.model_path)
    marker("PHASE-START", "model_load_s — reading Qwen3-4B safetensors from %s", model_path)
    load_t0 = time.monotonic()
    with run.timer.phase("model_load_s"):
        assert model_path.is_dir(), f"--model-path is not a directory: {model_path}"
        # Lesson from Lab 2: prime the gcsfuse implicit-dirs cache before Tunix globs the path.
        entries = list(model_path.iterdir())
        marker("MODEL", "primed the FUSE cache: %d entries under %s", len(entries), model_path)

        # Model size is a knob, not a constant: the 4B fine-tune exceeds the RAM of the 31 GiB
        # CPU node under every configuration tried (docs/backend-feasibility.md), so the
        # cross-backend comparison is run a second time at qwen3_0p6b, which fits everywhere.
        factory = getattr(qwen3_lib.ModelConfig, a.model_config, None)
        assert factory is not None, (
            f"expected tunix.models.qwen3.model.ModelConfig.{a.model_config}(); actual factories: "
            f"{[m for m in dir(qwen3_lib.ModelConfig) if not m.startswith('_')]}"
        )
        config = factory()
        marker("MODEL", "config factory %s() — %d layers, embed_dim %s",
               a.model_config, getattr(config, "num_layers", -1), getattr(config, "embed_dim", "?"))
        config = apply_vocab_override(config, model_path)
        config, run.dtype_fields = apply_dtype(config, a.dtype)
        marker("MODEL", "--dtype %s applied to config field(s) %s", a.dtype, run.dtype_fields)

        loader = qwen3_params_st.create_model_from_safe_tensors
        loader_kwargs: dict[str, Any] = {}
        try:
            if "dtype" in inspect.signature(loader).parameters:
                loader_kwargs["dtype"] = _dtype_from_name(a.dtype)
        except (TypeError, ValueError):
            pass
        with mesh_ctx(mesh):
            base_model = loader(str(model_path), config, mesh, **loader_kwargs)
    marker("PHASE-END", "model_load_s %.3fs", time.monotonic() - load_t0)

    model_cls = getattr(qwen3_lib, "Qwen3", None)
    assert model_cls is not None, (
        "expected class tunix.models.qwen3.model.Qwen3; actual module attributes: "
        f"{sorted(x for x in dir(qwen3_lib) if not x.startswith('_'))}"
    )
    assert isinstance(base_model, model_cls), (
        f"expected the loader to return a {model_cls.__name__}; actual {type(base_model).__name__}"
    )
    token_kw = token_kwarg_name(model_cls)
    marker("MODEL", "Qwen3.__call__ token argument is %r", token_kw)

    # -- LoRA wrap ----------------------------------------------------------------------------
    with sub("lora_wrap", f"qwix rank={a.lora_rank} alpha={a.lora_alpha} + forward trace"):
        provider = qwix.LoraProvider(module_path=a.lora_modules, rank=a.lora_rank, alpha=a.lora_alpha)
        trace_b, trace_t = max(1, jax.device_count()), 8
        trace_input = {
            token_kw: jnp.ones((trace_b, trace_t), dtype=jnp.int32),
            "positions": jnp.tile(jnp.arange(trace_t)[None, :], (trace_b, 1)),
            "cache": None,
            "attention_mask": jnp.tril(jnp.ones((trace_b, trace_t, trace_t), dtype=jnp.bool_)),
        }
        with mesh_ctx(mesh):
            lora_model = qwix.apply_lora_to_model(
                base_model, provider, rngs=nnx.Rngs(a.seed), **trace_input
            )
        del base_model

        try:
            from tunix.rl import reshard

            lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)
            marker("MODEL", "resharded the LoRA model onto the mesh")
        except ImportError:
            marker("WARN", "tunix.rl.reshard unavailable — skipping the explicit reshard")

        n_lora = sum(p.size for p in jax.tree.leaves(nnx.state(lora_model, nnx.LoRAParam)))
        assert n_lora > 0, (
            f"qwix matched zero modules with module_path={a.lora_modules!r} — Qwen3 uses HF "
            "projection names (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj)"
        )
        run.trainable_params = int(n_lora)
        marker("MODEL", "LoRA injected: %.2fM trainable params", n_lora / 1e6)
        apply_remat(config, lora_model, qwen3_lib, a.remat)

    # -- data ---------------------------------------------------------------------------------
    with sub("data_prep", f"CodiEsp from {a.data_root}"):
        data_root = Path(a.data_root)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Right truncation, matching dataset_stats.json's truncation table
        # ("truncation_side": "right (the completion is lost first)"). Overflow eats the label,
        # which is what prepare_data.py measured and reported; flipping the side here would make
        # that table describe a different workload from the one being benchmarked.
        tokenizer.truncation_side = "right"
        tokenizer.padding_side = "right"

        train_examples = load_split(data_root, a.train_split, a.max_train_examples, a.seed)
        eval_examples: list[Example] = []
        try:
            eval_examples = load_split(data_root, a.eval_split, 0, a.seed)
        except AssertionError as exc:
            run.note(f"eval split {a.eval_split!r} could not be loaded from {data_root} "
                     f"({exc}); training without one")

        # Build only the batches the trainer can consume. attention_mask is (B, T, T) bool, so an
        # unbounded prebuild is how a large --seq-len OOMs the host instead of the accelerator.
        train_batches = build_batches(tokenizer, train_examples, a.batch_size, a.seq_len,
                                      max_batches=a.max_steps + 1)
        eval_batches: list[dict[str, Any]] = []
        if eval_examples and a.eval_every > 0:
            eval_batches = build_batches(tokenizer, eval_examples, a.batch_size, a.seq_len,
                                         max_batches=8)
        per_batch_mib = (a.batch_size * (a.seq_len * 9 + a.seq_len ** 2)) / 1024 / 1024
        marker("DATA", "%d train batches (+%d eval) of shape (%d, %d), ~%.1f MiB each",
               len(train_batches), len(eval_batches), a.batch_size, a.seq_len, per_batch_mib)
        if len(train_batches) < a.max_steps:
            run.note(f"only {len(train_batches)} batches exist for max_steps={a.max_steps}; the "
                     "trainer loops over multiple epochs")

    # -- trainer ------------------------------------------------------------------------------
    collector = LossCollector()
    handler = _LossLogHandler(collector)
    step_sync: StepSync | None = None
    if a.no_step_sync:
        run.note("--no-step-sync: nothing blocks on the device between steps, so steps[].wall_s "
                 "measures async dispatch rather than device completion")
    else:
        step_sync = StepSync(jax, nnx, lora_model)
        marker("TELEMETRY", "per-step device sync: jax.block_until_ready() on %d LoRA parameter(s)"
                            " + jax.effects_barrier()", step_sync.n_probes)
    timed_train = TimedDataset(train_batches, sync_fn=step_sync)

    with sub("trainer_build", "constructing the Tunix PeftTrainer"):
        cfg: dict[str, Any] = {"max_steps": a.max_steps}
        if a.eval_every > 0 and eval_batches:
            cfg["eval_every_n_steps"] = a.eval_every
        if a.ckpt_dir:
            from orbax import checkpoint as ocp

            cfg["checkpoint_root_directory"] = a.ckpt_dir
            cfg["checkpointing_options"] = ocp.CheckpointManagerOptions(
                save_interval_steps=a.save_interval_steps or a.max_steps, max_to_keep=1
            )
        if os.environ.get("WANDB_API_KEY"):
            cfg["metrics_logging_options"] = MetricsLoggerOptions(
                log_dir=str(Path(a.ckpt_dir or out_dir_for(a.out)) / "logs" / run.run_id),
                project_name=a.wandb_project,
                run_name=a.wandb_run or run.run_id,
                backend_kwargs={"wandb": {"tags": [
                    "tunix", "qwen3-4b", "lora", "codiesp", a.backend, f"chips-{jax.device_count()}",
                ]}},
            )
            marker("WANDB", "enabled: project=%s run=%s", a.wandb_project, a.wandb_run or run.run_id)
        else:
            marker("WANDB", "WANDB_API_KEY is unset — metrics logging disabled")

        accepted = _field_names(peft_trainer.TrainingConfig) or set(
            inspect.signature(peft_trainer.TrainingConfig).parameters
        )
        assert "max_steps" in accepted, (
            f"expected TrainingConfig to accept max_steps; actual fields: {sorted(accepted)}"
        )
        for key in sorted(set(cfg) - accepted):
            marker("WARN", "TrainingConfig does not accept %r — dropping it", key)
            cfg.pop(key)
        if dataclasses.is_dataclass(peft_trainer.TrainingConfig):
            required = [
                f.name for f in dataclasses.fields(peft_trainer.TrainingConfig)
                if f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
                and f.name not in cfg
            ]
            assert not required, (
                f"TrainingConfig requires {required}, which this script does not set — pass "
                "--ckpt-dir if checkpoint_root_directory is among them"
            )
        trainer = peft_trainer.PeftTrainer(
            lora_model, optax.adamw(a.lr), peft_trainer.TrainingConfig(**cfg)
        )
        inner = getattr(trainer, "metrics_logger", None)
        if inner is not None and hasattr(inner, "log"):
            try:
                trainer.metrics_logger = _MetricsLoggerProxy(inner, collector)
                marker("TELEMETRY", "wrapped trainer.metrics_logger for per-step loss capture")
            except Exception as exc:  # noqa: BLE001 - a read-only property must not fail the run
                marker("WARN", "could not wrap trainer.metrics_logger (%s: %s) — falling back to "
                               "stdout/logging scraping for per-step loss", type(exc).__name__, exc)
        else:
            marker("TELEMETRY", "trainer exposes no metrics_logger.log — relying on stdout/logging "
                                "scraping for per-step loss")
        train_params = [p for p in inspect.signature(trainer.train).parameters.values()
                        if p.name != "self"]

    # -- the training loop ---------------------------------------------------------------------
    marker("PHASE-START", "compile_s + steady_state_s — %d steps, bs=%d seq=%d. The FIRST step is "
                          "an XLA compile (silence is normal).", a.max_steps, a.batch_size, a.seq_len)
    logging.getLogger().addHandler(handler)
    prev_out, prev_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(prev_out, collector, "stdout")
    sys.stderr = _TeeStream(prev_err, collector, "stderr")
    train_t0 = time.monotonic()
    try:
        with mesh_ctx(mesh):
            if eval_batches:
                trainer.train(timed_train, eval_batches)
            elif len(train_params) >= 2 and train_params[1].default is inspect.Parameter.empty:
                trainer.train(timed_train, None)  # eval_ds is positional-required on this build
            else:
                trainer.train(timed_train)
        timed_train.close()
    finally:
        sys.stdout, sys.stderr = prev_out, prev_err
        logging.getLogger().removeHandler(handler)
        train_wall = time.monotonic() - train_t0
        step_times = timed_train.step_durations()[: a.max_steps]
        if step_times:
            run.timer.add("compile_s", step_times[0])
            run.timer.add("steady_state_s", sum(step_times[1:]))
            for i, wall in enumerate(step_times):
                run.recorder.record(wall_s=wall, loss=collector.loss_for(i + 1), step=i + 1)
        marker("PHASE-END", "compile_s + steady_state_s %.3fs over %d recorded step(s)",
               train_wall, len(step_times))

    if timed_train.sync_failures:
        run.note(f"the per-step device sync raised on {timed_train.sync_failures} step "
                 "boundary/ies; those step times may measure async dispatch rather than completion")
    if step_sync is not None:
        for msg in step_sync.notes():
            run.note(msg)
    if len(step_times) < 2:
        run.note(f"only {len(step_times)} step boundary/ies observed in a {train_wall:.1f}s train "
                 "phase — the trainer probably materialised the dataset before iterating, so "
                 "per-step timings are unreliable")
    else:
        covered = sum(step_times)
        if covered < 0.5 * train_wall:
            run.note(f"observed step intervals cover only {covered:.1f}s of a {train_wall:.1f}s "
                     "train phase; the remainder is trainer overhead inside other_s")
    n_loss = sum(1 for s in run.recorder.steps if s["loss"] is not None)
    marker("TRAIN", "recorded %d step(s), %d with a loss (channels: %s)",
           run.recorder.n_steps, n_loss, sorted(collector.channels) or ["none"])
    if run.recorder.n_steps and n_loss == 0:
        run.note("no per-step loss could be captured from Tunix; steps[].loss is null rather than "
                 "a fabricated value")
    run.note("steady.tokens_per_s counts padded tokens (batch_size * seq_len), which is the "
             "compute actually performed")

    # -- eval ----------------------------------------------------------------------------------
    if a.eval_examples > 0 and eval_examples:
        with sub("eval", f"greedy decode over {a.eval_examples} dev case(s)"):
            try:
                run.eval = greedy_eval(lora_model, mesh_ctx, mesh, tokenizer,
                                       eval_examples[: a.eval_examples], a.seq_len,
                                       a.eval_max_new_tokens, token_kw)
                marker("EVAL", "micro_f1=%.4f over %d example(s)",
                       run.eval["micro_f1"], run.eval["n_examples"])
            except Exception as exc:  # noqa: BLE001 - eval must never fail a benchmark run
                run.eval = {"micro_f1": None, "n_examples": 0,
                            "note": f"functional sanity check failed: {type(exc).__name__}: {exc}"}
                marker("WARN", "eval failed: %s: %s", type(exc).__name__, exc)
    elif a.eval_examples > 0:
        run.note("--eval-examples was set but no dev split loaded; eval skipped")

    # -- adapter export -------------------------------------------------------------------------
    if a.skip_export:
        marker("EXPORT", "--skip-export is set — no adapter written")
        run.note("checkpoint_write_s is 0.0: --skip-export suppressed the adapter export")
    else:
        adapter_dir = (Path(a.adapter_out) if a.adapter_out
                       else out_dir_for(a.out) / "adapters" / run.run_id)
        marker("PHASE-START", "checkpoint_write_s — exporting the LoRA adapter to %s", adapter_dir)
        export_t0 = time.monotonic()
        with run.timer.phase("checkpoint_write_s"):
            n_tensors = export_lora_adapter(lora_model, adapter_dir, a.lora_rank, a.lora_alpha,
                                            a.model_name, qwen3_params)
        marker("PHASE-END", "checkpoint_write_s %.3fs — %d tensors to %s",
               time.monotonic() - export_t0, n_tensors, adapter_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out = resolve_out(args.out)
    configure_backend_env(args.backend, args.jax_platforms)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    for noisy in ("orbax", "absl", "jax._src"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    run = Run(args)  # starts the wall clock before JAX is imported, so init lands in other_s
    marker("RUN-START", "%s", " ".join(sys.argv))

    error: BaseException | None = None
    try:
        train(run)
    except Exception as exc:  # noqa: BLE001 - the sweep driver walks the ladder into OOM
        error = exc
        marker("FAILURE", "%s: %s", type(exc).__name__, exc)
        say(traceback.format_exc())
    finally:
        run.sampler.stop()
        run.timer.stop()

    status = tel.classify_exception(error)
    if status == "ok" and run.recorder.n_steps == 0:
        status = "error"
        error = RuntimeError(
            "training finished without recording a single step; steady statistics would be "
            "meaningless, so this run is reported as an error rather than as a success"
        )
        marker("FAILURE", "%s", error)

    steady = run.recorder.steady() if status == "ok" else None
    memory = tel.peak_memory(args.backend, sampler=run.sampler) if status == "ok" else None
    summary = run.sub.summary()

    record = tel.emit(
        args.out,
        run_id=run.run_id,
        backend=args.backend,
        hardware=run.hardware or detect_hardware(args),
        config=run.config_block(),
        phases=run.timer.snapshot(),
        steps=run.recorder.steps,
        steady=steady,
        memory=memory,
        utilization=run.sampler.block(),
        eval=run.eval,
        status=status,
        error=error,
        notes=run.notes + ([summary] if summary else [])
        + run.timer.notes + run.recorder.notes + run.sampler.notes,
    )

    marker("RUN-END", "status=%s total_wall_s=%.2f record=%s",
           record["status"], record["phases"]["total_wall_s"], args.out)
    if status == "ok":
        return 0
    if status in ("oom", "timeout"):
        # Exit 0 so a batch-size sweep can walk into the OOM boundary without dying.
        return 0
    return 0 if args.exit_zero_on_error else 1


if __name__ == "__main__":
    sys.exit(main())
