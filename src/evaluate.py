#!/usr/bin/env python3
"""Functional sanity check for the CodiEsp ICD-10 LoRA fine-tune.

WHAT THIS IS, HONESTLY
----------------------
This is *not* a graded metric and it is not a quality claim. The benchmark this project
is actually being marked on is the infrastructure telemetry in ``results/*.json`` — the
phase breakdown, the steady-state step time, the OOM boundary. This script exists for
exactly one reason: so the report can say "the workload learned something" and point at
a number instead of asserting it.

Treat the number with suspicion, on purpose:

* The CodiEsp corpus is small and its CIE-10-ES label space is large relative to it, so
  most codes are seen once or never during training and micro-F1 is dominated by the few
  high-frequency codes. The exact document count, distinct-code count and codes-per-doc
  distribution for the corpus this repo actually prepared are measured by
  ``src/prepare_data.py`` and written to ``dataset_stats.json``; this script never
  restates them from memory, and the only corpus numbers it emits are the ones it
  counted itself on the slice it just scored (see ``detail.*`` in the output).
* The benchmark trains for ``--max-steps`` steps, a throughput step count, not a
  convergence run.
* A small but non-zero micro-F1 is the pass condition: it means the model emits
  well-formed CIE-10-ES codes instead of Spanish prose. A micro-F1 of 0.00 with
  ``n_unparseable == n_examples`` is the actual failure signal — the fine-tune did not
  take and the training loop was measuring noise.

``n_unparseable`` is reported precisely because it is the interesting number for a base
model. ``--adapter none`` will happily answer a Spanish clinical coding prompt with a
paragraph of Spanish clinical prose containing no codes at all. The delta in
``n_unparseable`` between base and fine-tuned is usually a louder signal that LoRA did
something than the delta in F1.

USAGE
-----
    # base model, 50 dev docs, quick
    python src/evaluate.py --data data/codiesp-sft --adapter none \
        --limit 50 --out results/eval-base.json

    # fine-tuned, same slice, single flag changed
    python src/evaluate.py --data data/codiesp-sft \
        --adapter /gcs/teams/team-lharnold/codiesp-lora-adapter-4b \
        --limit 50 --out results/eval-lora.json

    # parser + metric unit tests, no model, no GPU, ~50 ms
    python src/evaluate.py --self-test

The emitted JSON drops straight into the ``eval`` block of docs/metrics-contract.md:

    rec = json.load(open("results/tpu-v5e8-bs8-seq512.json"))
    rec["eval"] = json.load(open("results/eval-lora.json"))

DEPENDENCIES (exact pins — this script is deliberately outside the JAX/TPU stack)
--------------------------------------------------------------------------------
    uv pip install \
        'torch==2.9.1' \
        'transformers==5.13.1' \
        'safetensors==0.7.0' \
        'datasets==5.0.0'      # only needed if prepare_data.py used save_to_disk

``transformers==5.13.1`` is the same pin as the training image, so tokenizer behaviour
(chat template, special tokens) is identical between train and eval. ``datasets`` is an
optional import and is only touched for a HuggingFace ``save_to_disk`` directory.
``--self-test`` needs none of them.

RUNTIME NOTES
-------------
* Runs on plain PyTorch + transformers, deliberately: it must produce the same number on
  the CPU node, on the GH200 and on a laptop, so the sanity check is never entangled with
  the JAX/TPU stack being benchmarked. It is *not* on any timed path.
* **The prompt is the trainer's prompt, byte for byte.** ``src/prepare_data.py`` owns the
  dataset contract; ``src/train_qwen3_icd.py`` imports it and trains on
  ``prompt + completion + tokenizer.eos_token``. This script imports the same module and
  uses the same ``prompt`` column (falling back to ``prepare_data.build_prompt`` for a
  record that carries only raw case text), so there is one template in the repo rather
  than a copy per script. Scoring a LoRA fine-tune under differently-worded instructions
  measures the distribution shift, not the fine-tune. Two guards back this up: the
  self-test asserts ``build_prompt`` is literally ``prepare_data``'s, and every run
  re-derives the first 32 prompts from their case text and warns if the JSONL disagrees
  with the ``prepare_data`` importable next to it.
* **No chat template by default.** Training is plain-text completion, so ``--prompt-style``
  defaults to ``raw``. ``chat`` exists only for a checkpoint trained elsewhere.
* The LoRA adapter is merged into the base weights by hand rather than through ``peft``.
  The trainer's exporter emits either ``base_model.model.model.layers.N…`` (its own key
  derivation) or bare ``model.layers.N…`` (when Tunix supplies a safetensors key mapper),
  depending on the Tunix version present at export time. ``peft`` silently loads
  *nothing* from the second shape — it logs an "unexpected keys" warning and moves on —
  and the eval would then score the base model twice while the report claimed the
  fine-tune did not work. The manual merge accepts both shapes, asserts that every
  adapter tensor landed on a real ``nn.Linear``, and fails loudly otherwise.
* Progress is logged line-by-line to stdout with flush, because on the GH200 cluster the
  RBAC persona cannot exec into pods and pod logs are the only output channel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    # src/prepare_data.py owns the dataset contract — the prompt template, the delimiter and
    # the `prompt + completion + eos` training recipe. src/train_qwen3_icd.py imports it for
    # exactly the same reason, so importing it here (rather than restating the template) is
    # what makes the prompt this script scores byte-identical to the one the adapter was
    # trained on. It is stdlib-only at import time, so this costs nothing.
    import prepare_data as prep
except ImportError as _exc:  # pragma: no cover - a broken deployment, not a degraded one
    raise SystemExit(
        "FATAL: cannot import the sibling module 'prepare_data' from %s. Expected "
        "src/prepare_data.py next to this file. Underlying error: %s" % (SCRIPT_DIR, _exc)
    )

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

_T0 = time.time()


def log(msg: str, *args: Any) -> None:
    """Timestamped stdout log. Flushed, because pod logs are the only channel on GKE."""
    text = msg % args if args else msg
    print("[%7.1fs] %s" % (time.time() - _T0, text), flush=True)


# --------------------------------------------------------------------------------------
# ICD-10 / CIE-10-ES code parsing
# --------------------------------------------------------------------------------------
#
# CodiEsp gold codes are lowercase and come in two shapes:
#
#   task 1 (diagnostico, ICD-10-CM / CIE-10-ES-Diagnosticos):
#       letter + digit + alphanumeric, optional dot + 1..4 alphanumerics
#       e.g. r51, c50.9, n39.0, z01.89, s72.001a
#
#   task 2 (procedimiento, ICD-10-PCS / CIE-10-ES-Procedimientos):
#       exactly 7 alphanumerics, no dot
#       e.g. 0dtj4zz, bw40zzz, 3e0g76z, b244zz3
#
# The whole point of this parser is to be *tolerant*: base-model output is Spanish prose
# with codes sprinkled through it, so codes are extracted from anywhere in the string
# rather than requiring a clean comma-separated list. That tolerance has a cost, and the
# two constraints below are what keep it from exploding:
#
#   1. ICD-10-PCS never uses the letters I or O (they are excluded to avoid confusion
#      with the digits 1 and 0), and its first character is restricted to the section
#      axis 0-9, B, C, D, F, G, H, X. That kills most Spanish 7-letter words outright
#      ("paciente" is 8 chars; "codigos" contains an o).
#   2. A PCS candidate must contain at least one digit. Without this, the common Spanish
#      word "durante" (7 chars, starts with d, no i/o) parses as a procedure code. The
#      trade-off is that a hypothetical all-letters-after-the-section PCS code would be
#      missed; in the CodiEsp gold files every procedure code contains a digit, so this
#      costs nothing measurable and prevents a systematic false-positive.
#
# Negative lookarounds on both patterns mean a code is only recognised as a whole token,
# which stops the diagnosis pattern from chewing "j4z" out of the middle of "0dtj4zz".

_DIAG_RE = re.compile(r"(?<![0-9a-z])[a-z][0-9][0-9a-z](?:\.[0-9a-z]{1,4})?(?![0-9a-z])")
_PCS_RE = re.compile(
    r"(?<![0-9a-z])(?=[0-9a-z]*[0-9])[0-9bcdfghx][0-9a-hj-np-z]{6}(?![0-9a-z])"
)

# Qwen3 is a hybrid reasoning model: with thinking enabled it wraps its scratchpad in
# <think>...</think>. We disable thinking in the chat template, but a base model asked a
# hard question sometimes emits the block anyway, so strip it before parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK_RE = re.compile(r"</?think>", re.IGNORECASE)

# ChatML / Gemma-style special tokens, e.g. <|im_end|>, <|endoftext|>, <start_of_turn>.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>]*\|>|<(?:/?)(?:start|end)_of_turn>")


def normalize_code(code: str) -> str:
    """Canonical form of a single code: lowercase, no surrounding junk."""
    c = code.strip().lower()
    c = c.strip("\"'`()[]{}<>,;:")
    c = c.rstrip(".")
    return c


def parse_codes(text: str) -> Set[str]:
    """Extract the set of CIE-10-ES codes mentioned anywhere in ``text``.

    Returns an empty set when nothing code-shaped is present — that is what makes an
    example count towards ``n_unparseable``.
    """
    if not text:
        return set()
    cleaned = _THINK_RE.sub(" ", text)
    cleaned = _ORPHAN_THINK_RE.sub(" ", cleaned)
    cleaned = _SPECIAL_TOKEN_RE.sub(" ", cleaned).lower()
    codes: Set[str] = set()
    codes.update(m.group(0) for m in _DIAG_RE.finditer(cleaned))
    codes.update(m.group(0) for m in _PCS_RE.finditer(cleaned))
    return {normalize_code(c) for c in codes if normalize_code(c)}


def is_code(token: str) -> bool:
    """True if ``token`` on its own is a well-formed code."""
    t = normalize_code(token)
    if not t:
        return False
    return bool(_DIAG_RE.fullmatch(t) or _PCS_RE.fullmatch(t))


# Sentinels a label field may carry to mean "no codes". prepare_data.py writes the literal
# string "none" for a document with an empty code list; without this filter that sentinel
# would be admitted as a gold code by the verbatim path below and count as a permanent
# false negative for every such document.
_EMPTY_LABEL_TOKENS = frozenset(
    {"none", "ninguno", "ninguna", "ningun", "n/a", "na", "null", "nil", "-", "--"}
)

# The union of every character CodiEsp gold codes actually use: CIE-10 diagnoses (r51,
# c50.9), CIE-10-PCS procedures (0dtj4zz) and CIE-O morphology entries (8550/3).
_LABEL_CHARSET_RE = re.compile(r"[0-9a-z./_+-]+")


def _verbatim_label_tokens(text: str) -> Optional[Set[str]]:
    """Split an explicit label string into codes, or return None if it is prose.

    Splitting on ``, ; |`` only — never on whitespace — is what makes the prose test
    reliable: a genuine code never contains whitespace, so a fragment that does is a
    sentence and belongs on the regex-scraping path instead.

    Tokens are kept **verbatim**, not filtered through :func:`is_code`. The CodiEsp gold
    files contain entries the tolerant prediction parser deliberately does not recognise
    (CIE-O morphology codes such as ``8550/3``); dropping them here would remove them from
    the denominator and silently inflate recall. Kept verbatim they score as honest misses.
    """
    pieces = [p.strip() for p in re.split(r"[,;|]+", text)]
    pieces = [p for p in pieces if p]
    if not pieces:
        return set()
    out: Set[str] = set()
    for piece in pieces:
        if re.search(r"\s", piece):
            return None  # a fragment with whitespace is prose, not a label list
        token = normalize_code(piece)
        if not token:
            continue
        # Anything left holding a character no CodiEsp code uses (chat special tokens such
        # as `<|im_end|>` survive the answer half of a fused training string) means this is
        # not a clean label list after all; hand the whole string to the scraper.
        if not _LABEL_CHARSET_RE.fullmatch(token):
            return None
        if token not in _EMPTY_LABEL_TOKENS:
            out.add(token)
    return out


def coerce_gold(value: Any) -> Set[str]:
    """Turn whatever the dev record carries as its label into a set of codes.

    Handles a real list of codes, a delimited string ("c50.9, n39.0"), and a free-text
    completion. Splitting is preferred over regex-scraping for strings so that a gold
    label the parser would not recognise still shows up as a miss rather than silently
    vanishing from the denominator.
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        out: Set[str] = set()
        for item in value:
            if isinstance(item, str):
                out.update(coerce_gold(item))
            elif isinstance(item, dict):
                for key in ("code", "codigo", "label"):
                    if key in item:
                        out.update(coerce_gold(item[key]))
                        break
        return out
    if not isinstance(value, str):
        return set()
    # `|` is one of the accepted label delimiters, which means a ChatML token such as
    # `<|im_end|>` would otherwise be shredded into the pseudo-code `im_end`. Remove chat
    # special tokens before any splitting happens.
    value = _SPECIAL_TOKEN_RE.sub(" ", value)
    verbatim = _verbatim_label_tokens(value)
    if verbatim is not None:
        return verbatim
    # Prose — fall back to scraping, e.g. a natural-language completion.
    return parse_codes(value)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def micro_prf(
    preds: Sequence[Set[str]], golds: Sequence[Set[str]]
) -> Tuple[float, float, float, int, int, int]:
    """Micro-averaged precision / recall / F1 over multi-label code sets.

    Micro rather than macro on purpose: the CodiEsp label space is far larger than its
    document count, so a macro average is dominated by the codes that appear once or never
    and would report ~0.00 regardless of what the model learned. Micro pools the per-document
    confusion counts, which is the standard CodiEsp shared-task metric family and is at
    least sensitive to the frequent codes the model could plausibly have learned.
    """
    assert len(preds) == len(golds)
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def exact_match_rate(preds: Sequence[Set[str]], golds: Sequence[Set[str]]) -> float:
    """Fraction of documents where the predicted code set equals the gold set exactly.

    Expected to be ~0.0 for anything but documents whose gold set is a single frequent
    code. It is reported because a suspiciously high value is a bug signal (leakage, or
    the dev split accidentally being the train split), not because it is a target.
    """
    if not preds:
        return 0.0
    hits = sum(1 for p, g in zip(preds, golds) if p == g)
    return hits / len(preds)


# --------------------------------------------------------------------------------------
# Dev split loading
# --------------------------------------------------------------------------------------

_PROMPT_FIELDS = ("prompt", "input", "instruction", "source", "question", "query")
_DOC_FIELDS = ("document", "doc", "text", "note", "case", "content", "full_text", "body")
_CODES_FIELDS = (
    "codes",
    "labels",
    "gold",
    "gold_codes",
    "icd10",
    "targets",
    "target",
    "completion",
    "output",
    "answer",
    "response",
)
_ID_FIELDS = ("doc_id", "id", "document_id", "name", "file", "filename")

# Markers a training-time template may have used to separate prompt from answer. Only
# consulted when a record carries a single fused `text` field and no separate label.
_ANSWER_MARKERS = (
    "<|im_start|>assistant",
    "<start_of_turn>model",
    "### ICD-10 diagnosis codes",  # src/prepare_data.py's PROMPT_DELIMITER
    "Códigos:",
    "Codigos:",
    "CODES:",
    "### Response",
    "### Codes",
)


@dataclass
class Example:
    doc_id: str
    prompt: Optional[str]  # ready-to-use prompt text, if the dataset supplied one
    document: Optional[str]  # raw clinical case text, if the dataset supplied that
    gold: Set[str]


def _first_present(row: Dict[str, Any], names: Sequence[str]) -> Optional[str]:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return n
    return None


def _split_on_marker(text: str) -> Optional[Tuple[str, str]]:
    """Split a fused train string into (prompt half, answer half).

    Returning both halves matters: the gold codes must come from the answer half alone.
    Clinical narratives routinely name codes that are not gold labels (a ruled-out
    differential, a prior admission), so scraping the whole fused string for gold silently
    injects false negatives and depresses recall for reasons that have nothing to do with
    the model.
    """
    for marker in _ANSWER_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            cut = idx + len(marker)
            return text[:cut], text[cut:]
    return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit("%s:%d is not valid JSON: %s" % (path, lineno, exc))
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _load_rows(data_path: Path, split: str) -> List[Dict[str, Any]]:
    """Load raw dicts for ``split`` from whatever prepare_data.py produced.

    Supported, in probe order:
      1. a .jsonl / .json file given directly,
      2. a directory holding <split>.jsonl / <split>.json (or validation/test aliases),
      3. a HuggingFace ``save_to_disk`` directory (DatasetDict or single Dataset).
    """
    candidates = [split, "dev", "validation", "valid", "test", "eval"]
    # dedupe, preserve order
    seen: Set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    if data_path.is_file():
        if data_path.suffix == ".jsonl":
            return _read_jsonl(data_path)
        obj = json.loads(data_path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            for name in candidates:
                if name in obj and isinstance(obj[name], list):
                    return [r for r in obj[name] if isinstance(r, dict)]
        raise SystemExit("Could not find a list of records in %s" % data_path)

    if not data_path.is_dir():
        raise SystemExit("--data path does not exist: %s" % data_path)

    # gcsfuse first-touch: prime the implicit-dirs cache before globbing. Cheap on a real
    # filesystem, the difference between working and "file not found" on a FUSE mount.
    try:
        list(data_path.iterdir())
    except OSError as exc:
        raise SystemExit("Cannot list --data dir %s: %s" % (data_path, exc))

    for name in candidates:
        for suffix in (".jsonl", ".json"):
            candidate = data_path / (name + suffix)
            if candidate.is_file():
                log("Reading dev split from %s", candidate)
                if suffix == ".jsonl":
                    return _read_jsonl(candidate)
                obj = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(obj, list):
                    return [r for r in obj if isinstance(r, dict)]
                raise SystemExit("%s does not contain a JSON list" % candidate)
        nested = data_path / name
        if nested.is_dir():
            for suffix in (".jsonl", ".json"):
                shards = sorted(nested.glob("*" + suffix))
                if not shards:
                    continue
                # Every shard, not just the first: returning early on shard 0 would score a
                # silently truncated dev split and the run would look successful.
                collected: List[Dict[str, Any]] = []
                for inner in shards:
                    log("Reading dev split from %s", inner)
                    if suffix == ".jsonl":
                        collected.extend(_read_jsonl(inner))
                        continue
                    obj = json.loads(inner.read_text(encoding="utf-8"))
                    if isinstance(obj, list):
                        collected.extend(r for r in obj if isinstance(r, dict))
                if collected:
                    return collected

    # HuggingFace datasets on-disk format.
    if (data_path / "dataset_dict.json").is_file() or (
        data_path / "dataset_info.json"
    ).is_file():
        try:
            from datasets import load_from_disk  # type: ignore
        except ImportError:
            raise SystemExit(
                "%s looks like a HuggingFace save_to_disk directory but the `datasets` "
                "package is not installed. `uv pip install datasets==5.0.0`." % data_path
            )
        log("Loading HuggingFace dataset from %s", data_path)
        ds = load_from_disk(str(data_path))
        if hasattr(ds, "keys"):
            for name in candidates:
                if name in ds:
                    log("Using split '%s' (%d rows)", name, len(ds[name]))
                    return [dict(r) for r in ds[name]]
            raise SystemExit(
                "No dev-like split in %s. Available: %s" % (data_path, list(ds.keys()))
            )
        return [dict(r) for r in ds]

    raise SystemExit(
        "Could not locate a '%s' split under %s. Expected %s.jsonl, %s.json, or a "
        "HuggingFace save_to_disk directory." % (split, data_path, split, split)
    )


def load_examples(
    data_path: Path,
    split: str,
    limit: Optional[int],
    prompt_field: Optional[str],
    doc_field: Optional[str],
    codes_field: Optional[str],
) -> List[Example]:
    rows = _load_rows(data_path, split)
    if not rows:
        raise SystemExit("Dev split at %s is empty." % data_path)
    if limit is not None:
        # Deterministic head slice, not a random sample: base and fine-tuned runs must
        # score the identical documents or the comparison is meaningless.
        rows = rows[:limit]

    sample_keys = sorted(rows[0].keys())
    pf = prompt_field or _first_present(rows[0], _PROMPT_FIELDS)
    if prompt_field is not None and prompt_field not in rows[0]:
        raise SystemExit(
            "--prompt-field %r is not a column in the dev records. Keys present: %s"
            % (prompt_field, sample_keys)
        )
    df = doc_field or _first_present(rows[0], _DOC_FIELDS)
    cf = codes_field or _first_present(rows[0], _CODES_FIELDS)
    idf = _first_present(rows[0], _ID_FIELDS)

    if cf is None:
        raise SystemExit(
            "No gold-label field found in the dev records. Keys present: %s. "
            "Pass --codes-field explicitly." % sample_keys
        )
    if pf is None and df is None:
        raise SystemExit(
            "No prompt or document field found in the dev records. Keys present: %s. "
            "Pass --prompt-field or --doc-field explicitly." % sample_keys
        )
    if pf is not None and pf == cf:
        pf = None  # same column cannot be both
    log("Field mapping: prompt=%s document=%s codes=%s id=%s", pf, df, cf, idf)
    log(
        "Prompt source: %s",
        ("dev column %r, used verbatim (this is what train_qwen3_icd.py trains on)" % pf)
        if pf else "prepare_data.build_prompt() applied to the document column",
    )
    # Drift guard. The adapter was trained on prompt+completion+eos built from
    # prepare_data.PROMPT_TEMPLATE. If a JSONL was written by an older prepare_data with a
    # different template, the column is a *different* string from the one the trainer would
    # build today, and the score silently becomes a measure of prompt mismatch.
    if pf and df:
        drifted = 0
        for row in rows[: min(len(rows), 32)]:
            column = row.get(pf)
            text = row.get(df)
            if isinstance(column, str) and isinstance(text, str):
                if column != prep.build_prompt(text):
                    drifted += 1
        if drifted:
            log("WARNING: %d of the first %d records' %r column differs from "
                "prepare_data.build_prompt(%r). The dataset was prepared with a different "
                "template than the one importable here. The column is still used verbatim "
                "(it is what training saw if the trainer read this same file), but "
                "prepare_data.py and this JSONL are out of sync — regenerate the dataset "
                "if the adapter was trained from a different JSONL.",
                drifted, min(len(rows), 32), pf, df)

    examples: List[Example] = []
    for i, row in enumerate(rows):
        prompt = row.get(pf) if pf else None
        document = row.get(df) if df else None
        if isinstance(prompt, list):  # chat-style records
            prompt = "\n".join(
                str(m.get("content", "")) for m in prompt if isinstance(m, dict)
            )
        gold_source = row.get(cf)
        if prompt is None and isinstance(document, str) and cf == df:
            # Fused train string: prompt and answer live in one field. Take the gold from
            # the answer half only, never the case narrative.
            halves = _split_on_marker(document)
            if halves is not None:
                prompt, document = halves[0], None
                gold_source = halves[1]
        gold = coerce_gold(gold_source)
        doc_id = str(row.get(idf, i)) if idf else str(i)
        examples.append(
            Example(
                doc_id=doc_id,
                prompt=prompt if isinstance(prompt, str) and prompt.strip() else None,
                document=(
                    document if isinstance(document, str) and document.strip() else None
                ),
                gold=gold,
            )
        )

    n_gold = sum(len(e.gold) for e in examples)
    n_empty = sum(1 for e in examples if not e.gold)
    log(
        "Loaded %d dev examples, %d gold codes total (%.1f/doc), %d with an empty gold set",
        len(examples),
        n_gold,
        n_gold / max(1, len(examples)),
        n_empty,
    )
    if n_gold == 0:
        raise SystemExit(
            "Every gold label parsed to the empty set — the --codes-field mapping is "
            "almost certainly wrong. Field used: %r." % cf
        )
    return examples


# --------------------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------------------

# Only used by the non-default `--prompt-style chat` escape hatch. Training applies NO chat
# template — src/train_qwen3_icd.py's training_text() is literally
# `prompt + completion + tokenizer.eos_token` over prepare_data's plain-text template — so
# wrapping the prompt in ChatML at eval time is a distribution shift and is off by default.
SYSTEM_PROMPT = "You are a medical coder."


def build_prompt(example: Example) -> str:
    """Prompt text for one dev document, exactly as the trainer assembled it.

    ``src/prepare_data.py`` writes a ``prompt`` column and ``src/train_qwen3_icd.py`` trains
    on ``prompt + completion + eos``, so the column *is* the training prompt and is used
    verbatim. When a record carries only raw case text, the prompt is rebuilt through
    ``prepare_data.build_prompt`` — the same function the trainer falls back to — rather
    than through a second, hand-written template that could drift from it.
    """
    if example.prompt:
        return example.prompt
    return prep.build_prompt(example.document or "")


# --------------------------------------------------------------------------------------
# Model loading, LoRA merge, generation
# --------------------------------------------------------------------------------------


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(requested: str, device: str):
    import torch

    if requested != "auto":
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            requested
        ]
    # bfloat16 everywhere, including CPU: a 4B model is ~8 GB in bf16 and ~16 GB in fp32,
    # and the CPU benchmark node has 31 GiB total. fp32 would fit but leaves no headroom
    # for the KV cache and the transient load-time copy.
    return torch.bfloat16


def _adapter_scaling(
    adapter_path: Path, rank_override: Optional[int], alpha_override: Optional[float]
) -> Tuple[float, Dict[str, Any]]:
    """Read r / lora_alpha from adapter_config.json and return the merge scaling factor."""
    cfg: Dict[str, Any] = {}
    # A --adapter pointing straight at adapter_model.safetensors still has its config
    # sitting next to it; the trainer's exporter writes both into the same directory.
    cfg_dir = adapter_path if adapter_path.is_dir() else adapter_path.parent
    cfg_file = cfg_dir / "adapter_config.json"
    if cfg_file.is_file():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        log("Read LoRA config from %s", cfg_file)
    rank = rank_override if rank_override is not None else cfg.get("r")
    alpha = alpha_override if alpha_override is not None else cfg.get("lora_alpha")
    if rank is None or alpha is None:
        raise SystemExit(
            "Cannot determine the LoRA scaling factor: %s has no readable "
            "adapter_config.json with `r` and `lora_alpha`. Pass --lora-rank and "
            "--lora-alpha explicitly." % adapter_path
        )
    rank = int(rank)
    alpha = float(alpha)
    if rank <= 0:
        raise SystemExit("LoRA rank must be positive, got %d" % rank)
    scaling = alpha / math.sqrt(rank) if cfg.get("use_rslora") else alpha / rank
    return scaling, {"r": rank, "lora_alpha": alpha, "use_rslora": bool(cfg.get("use_rslora"))}


def _normalize_adapter_key(key: str) -> str:
    """Map a saved adapter tensor name onto a plain module path on the HF model."""
    for prefix in ("base_model.model.", "base_model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    # peft writes lora_A.<adapter_name>.weight; our exporter writes lora_A.weight.
    key = re.sub(r"\.lora_([AB])\.[^.]+\.weight$", r".lora_\1.weight", key)
    return key


def _load_adapter_tensors(adapter_path: Path) -> Dict[str, Any]:
    from safetensors.torch import load_file

    if adapter_path.is_file():
        files = [adapter_path]
    else:
        list(adapter_path.iterdir())  # prime gcsfuse before globbing
        files = sorted(adapter_path.glob("*.safetensors"))
    if not files:
        raise SystemExit("No .safetensors found under %s" % adapter_path)
    tensors: Dict[str, Any] = {}
    for f in files:
        tensors.update(load_file(str(f)))
    log("Loaded %d adapter tensors from %s", len(tensors), adapter_path)
    return tensors


def merge_lora_(model, adapter_path: Path, scaling: float) -> int:
    """Fold the LoRA delta into the base weights in place. Returns modules merged.

    PEFT layout: lora_A is (r, in_features), lora_B is (out_features, r), and the
    effective update is ``W += scaling * (B @ A)`` against an nn.Linear weight of shape
    (out_features, in_features). This matches the exporter in this repo, which reshapes
    the qwix LoRAParams to exactly that convention before saving.
    """
    import torch
    from torch import nn

    tensors = _load_adapter_tensors(adapter_path)
    pairs: Dict[str, Dict[str, Any]] = {}
    skipped: List[str] = []
    for raw_key, value in tensors.items():
        key = _normalize_adapter_key(raw_key)
        if key.endswith(".lora_A.weight"):
            pairs.setdefault(key[: -len(".lora_A.weight")], {})["A"] = value
        elif key.endswith(".lora_B.weight"):
            pairs.setdefault(key[: -len(".lora_B.weight")], {})["B"] = value
        else:
            skipped.append(raw_key)
    if skipped:
        log("WARNING: %d adapter tensors are not lora_A/lora_B and were skipped: %s",
            len(skipped), skipped[:5])
    if not pairs:
        raise SystemExit(
            "The adapter at %s contains no lora_A/lora_B tensor pairs. Refusing to "
            "evaluate, because a silent no-op merge would score the base model and be "
            "reported as the fine-tune." % adapter_path
        )

    merged = 0
    for module_name, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            raise SystemExit("Adapter module %s is missing its %s half"
                             % (module_name, "B" if "A" in ab else "A"))
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            raise SystemExit(
                "Adapter targets module %r which does not exist on the loaded model. "
                "Wrong base checkpoint for this adapter?" % module_name
            )
        if not isinstance(module, nn.Linear):
            raise SystemExit(
                "Adapter targets %r which is a %s, not nn.Linear."
                % (module_name, type(module).__name__)
            )
        a = ab["A"].to(torch.float32)
        b = ab["B"].to(torch.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise SystemExit(
                "Bad LoRA shapes for %s: A=%s B=%s (expected A=(r,in), B=(out,r))"
                % (module_name, tuple(a.shape), tuple(b.shape))
            )
        delta = (b @ a) * scaling  # (out, in), computed in fp32 then cast down
        if tuple(delta.shape) != tuple(module.weight.shape):
            raise SystemExit(
                "LoRA delta %s does not match %s weight %s"
                % (tuple(delta.shape), module_name, tuple(module.weight.shape))
            )
        with torch.no_grad():
            module.weight.add_(delta.to(module.weight.dtype).to(module.weight.device))
        merged += 1

    assert merged == len(pairs), "merged %d of %d modules" % (merged, len(pairs))
    log("Merged LoRA into %d linear modules (scaling=%.4f)", merged, scaling)
    return merged


def load_model_and_tokenizer(model_path: str, adapter: Optional[Path], device: str,
                             dtype_name: str, rank_override: Optional[int],
                             alpha_override: Optional[float]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = Path(model_path)
    if p.is_dir():
        # Lesson from Lab 2: touch the mount before any library globs it, or the first
        # safetensors shard lookup pays full FUSE first-touch latency (or finds nothing).
        list(p.iterdir())

    device = _resolve_device(device)
    dtype = _resolve_dtype(dtype_name, device)
    log("Device=%s dtype=%s model=%s", device, str(dtype).replace("torch.", ""), model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batch generation requires left padding, otherwise the pads sit between
    # the prompt and the first generated token and the output is garbage.
    tokenizer.padding_side = "left"

    t0 = time.time()
    # transformers renamed `torch_dtype` to `dtype` in v5 (this repo pins 5.13.1). On v4
    # an unknown kwarg is swallowed into the config rather than raising, so the model
    # would quietly load in fp32 — 16 GB instead of 8 GB for a 4B model, on a 31 GiB
    # node. Pick the right name, then verify it actually took.
    import transformers

    major = int(str(transformers.__version__).split(".")[0])
    dtype_kwarg = "dtype" if major >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_path, **{dtype_kwarg: dtype})
    if model.dtype != dtype:
        log("WARNING: requested %s but the model loaded as %s (transformers %s). "
            "Memory use will be higher than expected.",
            dtype, model.dtype, transformers.__version__)
    log("Base weights loaded in %.1f s (dtype=%s)", time.time() - t0, model.dtype)

    adapter_meta: Dict[str, Any] = {}
    if adapter is not None:
        scaling, adapter_meta = _adapter_scaling(adapter, rank_override, alpha_override)
        adapter_meta["merged_modules"] = merge_lora_(model, adapter, scaling)
        adapter_meta["scaling"] = scaling

    model.to(device)
    model.eval()
    return model, tokenizer, device, adapter_meta


def apply_chat(tokenizer, prompt: str, style: str) -> str:
    """Optionally wrap the prompt in Qwen3's chat template. OFF by default, and it should
    stay off.

    ``src/train_qwen3_icd.py`` trains on ``prompt + completion + tokenizer.eos_token`` —
    plain text, no ChatML — so ``--prompt-style chat`` measures the distribution shift, not
    the fine-tune. It exists only so a chat-formatted checkpoint from somewhere else can be
    scored with the same parser.
    """
    if style != "chat":
        return prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is not None and getattr(tokenizer, "chat_template", None):
        # Qwen3 is a hybrid reasoning model; enable_thinking=False makes the template emit
        # an empty <think></think> pair so the model answers directly instead of burning
        # max_new_tokens on a reasoning trace. Older templates reject the kwarg.
        for extra in ({"enable_thinking": False}, {}):
            try:
                return apply(
                    messages, tokenize=False, add_generation_prompt=True, **extra
                )
            except TypeError:
                continue
            except Exception:  # noqa: BLE001 - fall through to the manual ChatML below
                break
    parts = ["<|im_start|>%s\n%s<|im_end|>\n" % (m["role"], m["content"]) for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def truncate_document(tokenizer, example: Example, max_doc_tokens: int) -> Example:
    """Clip the case text to a token budget, keeping the head.

    Two things this must not get wrong:

    * **Which end is kept.** ``build_batches`` in ``src/train_qwen3_icd.py`` tokenises with
      ``truncation=True`` at transformers' default ``truncation_side="right"``, so training
      sequences keep the head of the document. Keeping the tail here would condition the
      model on a region of the document it never trained on.
    * **What gets clipped.** Only the document is clipped, never the assembled prompt, so
      prepare_data's trailing ``### ICD-10 diagnosis codes`` delimiter always survives —
      without it the model has no cue to start emitting codes. When the prompt column is in
      use the clipped text is spliced back into it in place of the original, which keeps
      both the leading instruction block and the trailing delimiter intact; a prompt whose
      document text cannot be located is left alone rather than blindly cut.
    """
    if example.document is None or max_doc_tokens <= 0:
        return example
    ids = tokenizer(example.document, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_doc_tokens:
        return example
    clipped = tokenizer.decode(ids[:max_doc_tokens], skip_special_tokens=True)
    prompt = example.prompt
    if prompt:
        idx = prompt.find(example.document)
        if idx < 0:
            return example  # cannot splice safely; leave the caller's prompt untouched
        prompt = prompt[:idx] + clipped + prompt[idx + len(example.document) :]
    return Example(example.doc_id, prompt, clipped, example.gold)


def generate_predictions(model, tokenizer, examples: List[Example], device: str,
                         batch_size: int, max_new_tokens: int, max_doc_tokens: int,
                         prompt_style: str) -> List[str]:
    import torch

    n_unclippable = sum(
        1 for e in examples if e.prompt and e.document is None
    ) if max_doc_tokens > 0 else 0
    if n_unclippable:
        log("WARNING: %d/%d examples carry a prompt with no separate document column, so "
            "--max-doc-tokens cannot be applied to them; pass --doc-field so the case text "
            "can be clipped, or expect long prompts.", n_unclippable, len(examples))

    texts = [
        apply_chat(tokenizer, build_prompt(truncate_document(tokenizer, e, max_doc_tokens)),
                   prompt_style)
        for e in examples
    ]
    outputs: List[str] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        chunk = texts[bi * batch_size : (bi + 1) * batch_size]
        # add_special_tokens is left at transformers' default (True), which is what
        # src/train_qwen3_icd.py's build_batches uses and what prepare_data.py measured its
        # truncation table against. For Qwen3 the overhead is zero tokens either way, but
        # matching the trainer's call is the point.
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        t0 = time.time()
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy: the sanity check must be reproducible
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(decoded)
        log("batch %d/%d  (%d prompts, %d prompt tokens, %.1f s)",
            bi + 1, n_batches, len(chunk), enc["input_ids"].shape[1], time.time() - t0)
    return outputs


# --------------------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------------------


def self_test() -> int:
    """Unit tests for the parser and the metrics. No model, no accelerator, no network."""
    failures: List[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append("%s: got %r want %r" % (name, got, want))

    # --- diagnosis codes -------------------------------------------------------------
    check("plain diag", parse_codes("c50.9"), {"c50.9"})
    check("3-char diag", parse_codes("El paciente presenta R51 desde ayer."), {"r51"})
    check("uppercase", parse_codes("Dx: N39.0, E11.9"), {"n39.0", "e11.9"})
    check("4-char ext", parse_codes("s72.001a"), {"s72.001a"})
    check("trailing period", parse_codes("Se codifica como C50.9."), {"c50.9"})
    check("in list", parse_codes("c50.9, n39.0; z01.89"), {"c50.9", "n39.0", "z01.89"})

    # --- procedure codes -------------------------------------------------------------
    check("pcs", parse_codes("0dtj4zz"), {"0dtj4zz"})
    check("pcs imaging", parse_codes("BW40ZZZ"), {"bw40zzz"})
    check("pcs mixed", parse_codes("3e0g76z y b244zz3"), {"3e0g76z", "b244zz3"})
    # A PCS code must not be shredded into a bogus diagnosis code by the diag pattern.
    check("no substring bleed", parse_codes("0dtj4zz"), {"0dtj4zz"})

    # --- false positives the tolerant parser must reject -----------------------------
    check("spanish word durante", parse_codes("durante la intervención"), set())
    check("spanish word paciente", parse_codes("el paciente refiere dolor"), set())
    check("covid19", parse_codes("sospecha de covid19"), set())
    check("bare year", parse_codes("en 2019 ingresó"), set())
    check("empty", parse_codes(""), set())
    check("prose only", parse_codes("Carcinoma ductal infiltrante de mama izquierda."), set())

    # --- thinking-block stripping ----------------------------------------------------
    check(
        "think stripped",
        parse_codes("<think>quizás c50.9 o no</think> Códigos: n39.0"),
        {"n39.0"},
    )

    # --- gold coercion ---------------------------------------------------------------
    check("gold list", coerce_gold(["C50.9", "n39.0"]), {"c50.9", "n39.0"})
    check("gold csv", coerce_gold("c50.9, n39.0"), {"c50.9", "n39.0"})
    check("gold none", coerce_gold(None), set())
    check("gold dicts", coerce_gold([{"code": "C50.9"}, {"code": "r51"}]), {"c50.9", "r51"})
    check("gold prose", coerce_gold("Los códigos son c50.9 y n39.0"), {"c50.9", "n39.0"})
    # Regression: a gold code the *prediction* parser does not recognise must still reach
    # the denominator. prepare_data.py keeps CIE-O morphology entries such as 8550/3, and
    # dropping them here would delete real misses and inflate recall.
    check("gold keeps morphology code",
          coerce_gold(["c50.9", "8550/3", "0dtj4zz"]), {"c50.9", "8550/3", "0dtj4zz"})
    check("gold csv keeps morphology code",
          coerce_gold("c50.9, 8550/3"), {"c50.9", "8550/3"})
    # prepare_data.py writes the literal string "none" for a document with no codes; it is
    # a sentinel, not a code, and must never enter the gold set.
    check("gold sentinel none", coerce_gold("none"), set())
    check("gold sentinel in list", coerce_gold(["none"]), set())
    # Chat special tokens leaking out of a fused answer half must not become gold codes.
    check("gold with chatml tail", coerce_gold("c50.9, r51<|im_end|>"), {"c50.9", "r51"})

    # --- metrics ---------------------------------------------------------------------
    preds = [{"a01", "b02"}, {"c03"}, set()]
    golds = [{"a01", "z99"}, {"c03"}, {"d04"}]
    # tp = 1 + 1 + 0 = 2 ; fp = 1 + 0 + 0 = 1 ; fn = 1 + 0 + 1 = 2
    p, r, f1, tp, fp, fn = micro_prf(preds, golds)
    check("tp", tp, 2)
    check("fp", fp, 1)
    check("fn", fn, 2)
    check("precision", round(p, 6), round(2 / 3, 6))
    check("recall", round(r, 6), 0.5)
    check("micro f1", round(f1, 6), round(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5), 6))
    check("exact match", round(exact_match_rate(preds, golds), 6), round(1 / 3, 6))

    # degenerate cases must not raise
    check("all empty prf", micro_prf([set()], [set()])[:3], (0.0, 0.0, 0.0))
    check("all empty em", exact_match_rate([set()], [set()]), 1.0)
    check("no preds", micro_prf([set()], [{"a01"}])[:3], (0.0, 0.0, 0.0))

    # --- fused prompt/answer splitting -----------------------------------------------
    # Regression: a code named in the case narrative ("se descarta c34.9") must NOT be
    # counted as a gold label just because it shares a field with the answer.
    fused = "Caso clínico:\nSe descarta c34.9 previo.\n\nCódigos: c50.9, r51"
    halves = _split_on_marker(fused)
    check("fused splits", halves is not None, True)
    if halves:
        check("fused prompt keeps instruction", halves[0].endswith("Códigos:"), True)
        check("fused gold is answer-only", coerce_gold(halves[1]), {"c50.9", "r51"})
    check("no marker -> None", _split_on_marker("Paciente con dolor abdominal."), None)

    # --- adapter key normalisation ---------------------------------------------------
    check(
        "peft prefix stripped",
        _normalize_adapter_key(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        ),
        "model.layers.0.self_attn.q_proj.lora_A.weight",
    )
    check(
        "bare key untouched",
        _normalize_adapter_key("model.layers.3.mlp.down_proj.lora_B.weight"),
        "model.layers.3.mlp.down_proj.lora_B.weight",
    )
    # The trainer's exporter falls back to `base_model.model.model.layers.N…` when Tunix
    # exposes no safetensors key mapper; that shape must normalise too.
    check(
        "trainer fallback prefix stripped",
        _normalize_adapter_key(
            "base_model.model.model.layers.7.mlp.gate_proj.lora_A.weight"
        ),
        "model.layers.7.mlp.gate_proj.lora_A.weight",
    )

    # --- prompt parity with the training pipeline -------------------------------------
    # The highest-value assertions in this file. The adapter is trained on
    # `prompt + completion + eos` where `prompt` comes from prepare_data.build_prompt; if
    # this script ever builds a different string, the score silently becomes a measure of
    # prompt mismatch rather than of the fine-tune.
    doc = "Varón de 54 años con dolor torácico."
    check("prompt is prepare_data's", build_prompt(Example("d", None, doc, set())),
          prep.build_prompt(doc))
    check("prompt column wins verbatim",
          build_prompt(Example("d", "READY-MADE PROMPT", doc, set())), "READY-MADE PROMPT")
    # The generation cue: the prompt must end with the delimiter the completion follows,
    # or the model has nothing telling it to start emitting codes.
    check("prompt ends on the delimiter",
          prep.build_prompt(doc).endswith(prep.PROMPT_DELIMITER), True)
    check("document survives into the prompt", doc in prep.build_prompt(doc), True)
    # raw is the default and must be a no-op; `chat` must actually change the string.
    check("raw style is a no-op", apply_chat(None, "PROMPT", "raw"), "PROMPT")
    # prepare_data's empty-completion sentinel must never be scored as a gold code.
    check("prepare_data empty completion is not a code",
          coerce_gold(prep.EMPTY_COMPLETION), set())
    # The delimiter is what _split_on_marker keys on for a fused record.
    check("prepare_data delimiter is a known answer marker",
          any(m in prep.PROMPT_DELIMITER or prep.PROMPT_DELIMITER.startswith(m)
              for m in _ANSWER_MARKERS), True)

    if failures:
        print("SELF-TEST FAILED (%d):" % len(failures), file=sys.stderr)
        for f in failures:
            print("  - " + f, file=sys.stderr)
        return 1
    print("self-test OK")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Functional sanity check: micro-F1 of CIE-10-ES code prediction on the "
            "CodiEsp dev split. NOT a graded metric."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", type=Path,
                    help="dev split produced by src/prepare_data.py (dir or .jsonl)")
    ap.add_argument("--split", default="dev", help="split name to evaluate")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B"),
                    help="base checkpoint dir or HF id (env MODEL_PATH honoured)")
    ap.add_argument("--adapter", default="none",
                    help="'none' for the base model, else the LoRA adapter dir/file")
    ap.add_argument("--out", type=Path, help="where to write the eval JSON")
    ap.add_argument("--pred-out", type=Path, default=None,
                    help="optional JSONL dump of raw generations, for the report")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N dev docs (deterministic head slice)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-doc-tokens", type=int, default=400,
                    help="clip the case text to this many tokens, keeping the tail as the "
                         "trainer does (train seq_len is 512); 0 disables clipping")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--prompt-style", default="raw", choices=["raw", "chat"],
                    help="'raw' matches training (prompt + completion + eos, no ChatML); "
                         "'chat' wraps the prompt in Qwen3's chat template and is a "
                         "distribution shift for adapters trained by this repo")
    ap.add_argument("--prompt-field", default=None,
                    help="use this dev column verbatim as the prompt instead of the "
                         "trainer's template. NOT auto-detected: only correct if that "
                         "column holds the exact string training saw")
    ap.add_argument("--doc-field", default=None, help="override document column name")
    ap.add_argument("--codes-field", default=None, help="override gold-label column name")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="override r when adapter_config.json is missing")
    ap.add_argument("--lora-alpha", type=float, default=None,
                    help="override lora_alpha when adapter_config.json is missing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true",
                    help="run parser/metric unit tests and exit (no model needed)")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.data is None or args.out is None:
        raise SystemExit("--data and --out are required (or use --self-test)")

    adapter: Optional[Path] = None
    if args.adapter and args.adapter.lower() not in ("none", "base", ""):
        adapter = Path(args.adapter)
        if not adapter.exists():
            raise SystemExit("--adapter path does not exist: %s" % adapter)

    examples = load_examples(args.data, args.split, args.limit, args.prompt_field,
                             args.doc_field, args.codes_field)

    import torch

    torch.manual_seed(args.seed)

    model, tokenizer, device, adapter_meta = load_model_and_tokenizer(
        args.model, adapter, args.device, args.dtype, args.lora_rank, args.lora_alpha
    )

    t_gen = time.time()
    generations = generate_predictions(
        model, tokenizer, examples, device, args.batch_size, args.max_new_tokens,
        args.max_doc_tokens, args.prompt_style,
    )
    gen_s = time.time() - t_gen

    preds = [parse_codes(g) for g in generations]
    golds = [e.gold for e in examples]
    n_unparseable = sum(1 for p in preds if not p)

    precision, recall, f1, tp, fp, fn = micro_prf(preds, golds)
    em = exact_match_rate(preds, golds)

    kind = "fine-tuned (LoRA merged)" if adapter is not None else "base (no adapter)"
    distinct_gold = sorted({c for g in golds for c in g})
    # Every number in `note` is counted on the slice that was just scored. Corpus-wide
    # statistics are NOT restated here — they are measured by src/prepare_data.py and live
    # in dataset_stats.json, and a remembered constant in this string would end up quoted
    # in the report as if it had been measured.
    result: Dict[str, Any] = {
        # --- the six fields the metrics contract's `eval` block consumes --------------
        "micro_f1": f1,
        "precision": precision,
        "recall": recall,
        "exact_match": em,
        "n_unparseable": n_unparseable,
        "n_examples": len(examples),
        "note": (
            "functional sanity check only, not a graded metric; %s, greedy decode of %d "
            "tokens over %d %s docs carrying %d distinct gold codes; %d/%d generations "
            "contained no parseable code"
            % (kind, args.max_new_tokens, len(examples), args.split, len(distinct_gold),
               n_unparseable, len(examples))
        ),
        # --- everything else, so the eval block stays contract-shaped -----------------
        "detail": {
            "model": args.model,
            "adapter": str(adapter) if adapter else None,
            "adapter_config": adapter_meta or None,
            "variant": kind,
            "split": args.split,
            "limit": args.limit,
            "device": device,
            # The dtype the weights actually loaded as, not the flag. --dtype auto is the
            # default, and recording the literal string "auto" in a results file would make
            # the record unreproducible.
            "dtype": str(getattr(model, "dtype", args.dtype)).replace("torch.", ""),
            "dtype_requested": args.dtype,
            "prompt_style": args.prompt_style,
            "max_new_tokens": args.max_new_tokens,
            "max_doc_tokens": args.max_doc_tokens,
            "batch_size": args.batch_size,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "n_gold_codes": sum(len(g) for g in golds),
            "n_distinct_gold_codes": len(distinct_gold),
            "n_pred_codes": sum(len(p) for p in preds),
            "n_distinct_pred_codes": len({c for p in preds for c in p}),
            # Derived from the examples, not from the flags: --prompt-field is usually
            # unset because the column is auto-detected, so keying this off the flag would
            # mislabel every default run.
            "prompt_source": (
                "dev prompt column (verbatim)" if any(e.prompt for e in examples)
                else "prepare_data.build_prompt(document)"
            ),
            "generation_wall_s": gen_s,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    if args.pred_out is not None:
        args.pred_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pred_out.open("w", encoding="utf-8") as fh:
            for ex, raw, pred in zip(examples, generations, preds):
                fh.write(json.dumps({
                    "doc_id": ex.doc_id,
                    "gold": sorted(ex.gold),
                    "pred": sorted(pred),
                    "raw_generation": raw,
                }, ensure_ascii=False) + "\n")
        log("Wrote raw generations to %s", args.pred_out)

    log("=" * 72)
    log("%s | %d docs | %.1f s generation", kind, len(examples), gen_s)
    log("micro-F1 %.4f  (P %.4f  R %.4f)   tp=%d fp=%d fn=%d", f1, precision, recall, tp, fp, fn)
    log("exact-match %.4f   unparseable %d/%d", em, n_unparseable, len(examples))
    log("Wrote %s", args.out)
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
