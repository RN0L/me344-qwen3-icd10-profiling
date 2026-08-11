#!/usr/bin/env python3
"""Convert the raw CodiEsp corpus into a tokenizer-ready SFT dataset.

CodiEsp ships Spanish clinical case reports as one plain-text file per document plus a
TAB-separated annotation file per split.  This script keeps **task 1 only** (DIAGNOSTICO,
i.e. ICD-10 diagnosis codes); the task2 (PROCEDIMIENTO) and task3 files that sit next to it
are deliberately ignored.

Outputs, all written under ``--out``:

  train.jsonl        one JSON object per line: doc_id, text, codes, prompt, completion
  dev.jsonl          same shape
  dataset_stats.json doc counts, token-length percentiles under the Qwen3 tokenizer,
                     unique-code counts, codes-per-doc statistics and the truncation
                     table at seq_len 256/512/1024/2048 that the report consumes.

Expected layout under ``--data-root`` (this is the shape of the shipped train_dev bundle)::

    <data-root>/train/text_files/<doc_id>.txt
    <data-root>/train/train_annotations_task1_processed.tsv
    <data-root>/dev/text_files/<doc_id>.txt
    <data-root>/dev/dev_annotations_task1_processed.tsv

Example::

    python src/prepare_data.py \
        --data-root ~/final-project/data/train_dev \
        --out data/codiesp-sft \
        --tokenizer ~/final-project/models/Qwen3-4B

``scripts/run-all.sh`` (stage ``data``) runs this script with **no arguments at all**, handing it
``CODIESP_RAW_DIR`` / ``DATA_LOCAL_DIR`` / ``DATA_GCS_URI`` / ``MODEL_GCS_URI`` in the environment,
so every required argument also has an environment fallback (see ``ENV_FALLBACKS`` below) and the
upload target defaults to ``$DATA_GCS_URI``.

WHO CONSUMES WHAT (read out of the sibling scripts, not assumed)
----------------------------------------------------------------
* ``src/train_qwen3_icd.py`` ``import prepare_data as prep``: it reads ``<data_root>/<split>.jsonl``
  when present, falls back to calling :func:`build_split` on the raw bundle when it is not, and
  builds its training text as ``prompt + completion + tokenizer.eos_token`` — i.e. exactly
  :data:`TRAINING_TEXT_RECIPE`, which is what the truncation table below is computed over. It also
  uses :func:`build_prompt`, :func:`build_completion`, :data:`TEXT_SUBDIR` and
  :data:`EMPTY_COMPLETION`. **This module must therefore stay stdlib-only at import time** (the
  tokenizer import is deliberately inside :func:`load_tokenizer`) and those four names are a public
  interface, not internal details.
* ``src/evaluate.py`` reads ``dev.jsonl`` for ``doc_id``/``text``/``codes`` (and ``prompt`` when it
  is told to with ``--prompt-field``).
* Both of them are pointed at ``$DATA_PATH``, one gcsfuse prefix, which is why ``--upload-to``
  publishes the raw split directories *and* the prepared JSONL under the same prefix: the
  fallback path in the trainer needs ``<split>/text_files/*.txt`` to still be there.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Prompt format
# --------------------------------------------------------------------------------------
# The instruction block is English on purpose: Qwen3's instruction following is most
# reliable in English even when the payload is Spanish, and the graded object here is the
# infrastructure rather than the coding accuracy.  The output contract is stated
# explicitly (lowercase, comma+space separated, alphabetically sorted) so that a *base*
# model has a fair chance of emitting something parseable before any fine-tuning, which is
# what makes the pre/post sanity check meaningful.
#
# The final line is the delimiter.  The prompt always ends with a trailing newline right
# after it, so the completion starts at column 0 of the next line with no leading space.
PROMPT_TEMPLATE = """You are a medical coder. Read the clinical case report below, which is written in Spanish, and list every ICD-10 diagnosis code that applies to it.

Answer format (follow it exactly):
- output lowercase ICD-10 codes only, for example: a23.9, e04.9, r50.9
- separate the codes with a comma and a space
- sort the codes alphabetically
- output nothing else: no explanations, no code descriptions, no translation of the case
- if the case contains no codable diagnosis, output: none

### Clinical case
{case_text}

### ICD-10 diagnosis codes
"""

PROMPT_DELIMITER = "### ICD-10 diagnosis codes\n"
COMPLETION_SEPARATOR = ", "
EMPTY_COMPLETION = "none"

# Full text the trainer is expected to build for each example.  Recorded in the stats file
# so the truncation table can be audited against whatever the training script actually does.
TRAINING_TEXT_RECIPE = "prompt + completion + tokenizer.eos_token"

ANNOTATION_GLOB = "*_annotations_task1_processed.tsv"
TEXT_SUBDIR = "text_files"
KEEP_LABEL = "DIAGNOSTICO"
AUTOMATIC_SUGGESTION_RE = re.compile(r"\s*\(automatic suggestion\)\s*$", re.IGNORECASE)

DEFAULT_SPLITS = ("train", "dev")
DEFAULT_SEQ_LENS = (256, 512, 1024, 2048)

# Splits whose JSONL order is shuffled (with --seed) rather than left in doc_id order.
# Sorting by doc_id groups documents by source journal, which would make every training
# batch topically homogeneous; dev stays sorted so eval output is easy to diff.
SHUFFLED_SPLITS = frozenset({"train"})

# Environment fallbacks for the three otherwise-required arguments, in probe order.  These names
# are the ones scripts/common.sh and manifests/env.sh already export, so `python3
# src/prepare_data.py` with no arguments works from inside scripts/run-all.sh stage `data`.
ENV_FALLBACKS = {
    "data_root": ("CODIESP_RAW_DIR", "DATA_HOST_DIR"),
    "out": ("DATA_LOCAL_DIR", "PREPARED_DATA_HOST_DIR"),
    # MODEL_PATH is the *in-container* gcsfuse mount and does not exist on the node, so it is
    # probed last and only accepted when it really is a directory here.
    "tokenizer": ("TOKENIZER_PATH", "MODEL_HOST_DIR", "MODEL_PATH"),
}
# Where the prepared corpus is published for the containerised backends. Empty disables the upload.
UPLOAD_ENV = "DATA_GCS_URI"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def log(msg: str, *args: object) -> None:
    if args:
        msg = msg % args
    sys.stderr.write("[prepare_data] " + msg + "\n")
    sys.stderr.flush()


def percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile, matching numpy's default method.

    ``sorted_values`` must already be sorted ascending. ``q`` is in [0, 100].
    """
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    pos = (n - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[int(pos)])
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """min / mean / p50 / p90 / p99 / max plus the count, for a numeric sample."""
    if not values:
        return {
            "n": 0, "min": None, "mean": None,
            "p50": None, "p90": None, "p99": None, "max": None,
        }
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "mean": float(sum(ordered)) / len(ordered),
        "p50": percentile(ordered, 50.0),
        "p90": percentile(ordered, 90.0),
        "p99": percentile(ordered, 99.0),
        "max": float(ordered[-1]),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def env_fallback(kind: str, must_be_dir: bool) -> Tuple[Optional[str], Optional[str]]:
    """First usable environment fallback for ``kind``: returns (value, variable name).

    ``must_be_dir`` rejects a variable that points somewhere that does not exist on *this* host.
    That is what keeps ``MODEL_PATH=/gcs/teams/...`` — correct inside the training container, absent
    on the node — from being silently handed to the tokenizer loader, which would then treat it as
    a Hugging Face hub id and fail with a confusing network error.
    """
    for name in ENV_FALLBACKS[kind]:
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        if must_be_dir and not Path(value).expanduser().is_dir():
            continue
        return value, name
    return None, None


def gcs_publish(data_root: Path, out_dir: Path, file_names: Sequence[str], target: str) -> None:
    """Publish the corpus to ``target`` so every containerised backend reads identical bytes.

    Two consumers, one prefix, so both are written:

      <target>/train/…  <target>/dev/…    the RAW split directories, because
                                          src/train_qwen3_icd.py re-reads text_files/*.txt and the
                                          annotation TSVs itself
      <target>/train.jsonl, dev.jsonl, dataset_stats.json
                                          the prepared SFT files, which src/evaluate.py reads

    ``rsync`` is deliberately run WITHOUT any delete flag: the prepared JSONL lives in the same
    prefix and a mirroring delete would remove it on the next run.
    """
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise SystemExit(
            "--upload-to %s was requested but `gcloud` is not on PATH. Re-run with --no-upload to "
            "skip publishing, or install/authenticate the Cloud SDK." % target
        )
    target = target.rstrip("/")
    commands = [[gcloud, "storage", "rsync", "--recursive", str(data_root), target]]
    commands += [
        [gcloud, "storage", "cp", str(out_dir / name), target + "/"] for name in file_names
    ]
    for cmd in commands:
        log("$ %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(
                "gcloud exited %d for: %s\nThe local files under %s are complete; only the upload "
                "failed. Fix the credentials and re-run, or pass --no-upload."
                % (result.returncode, " ".join(cmd), out_dir)
            )
    log("published raw splits + prepared files to %s", target)


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
def normalize_case_text(raw: str) -> str:
    """Clean a raw clinical case file without destroying its paragraph structure."""
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")  # LINE/PARA SEP
    text = text.replace("\u00a0", " ")  # NBSP -> plain space, tokenises far more cleanly
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_code(raw: str) -> Optional[str]:
    """Normalise one annotation code cell.

    Strips the ``(automatic suggestion)`` suffix that some rows carry, NFC-normalises,
    lowercases and trims.  Returns ``None`` for cells that are not codes at all: the dev
    annotations contain one row whose code column holds free text
    (``hematuria macroscópica``), which is an upstream data error.  Whitespace inside the
    cell is the discriminator -- every genuine code in this corpus is whitespace-free,
    including the ICD-10-PCS entries (``08j0xzz``) and the CIE-O morphology entries
    (``8550/3``), both of which are kept.
    """
    code = unicodedata.normalize("NFC", raw)
    code = AUTOMATIC_SUGGESTION_RE.sub("", code)
    code = code.strip().lower()
    if not code:
        return None
    if re.search(r"\s", code):
        return None
    return code


def build_prompt(case_text: str) -> str:
    return PROMPT_TEMPLATE.format(case_text=case_text)


def build_completion(codes: Sequence[str]) -> str:
    if not codes:
        return EMPTY_COMPLETION
    return COMPLETION_SEPARATOR.join(codes)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
class AnnotationCounters:
    def __init__(self) -> None:
        self.rows_total = 0
        self.rows_kept = 0
        self.rows_header_skipped = 0
        self.rows_too_few_fields = 0
        self.rows_wrong_task_label = 0
        self.rows_malformed_code = 0
        self.rows_automatic_suggestion = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "total": self.rows_total,
            "kept": self.rows_kept,
            "header_skipped": self.rows_header_skipped,
            "dropped_too_few_fields": self.rows_too_few_fields,
            "dropped_wrong_task_label": self.rows_wrong_task_label,
            "dropped_malformed_code": self.rows_malformed_code,
            "carried_automatic_suggestion_suffix": self.rows_automatic_suggestion,
        }


def find_annotation_file(split_dir: Path) -> Path:
    matches = sorted(split_dir.glob(ANNOTATION_GLOB))
    if not matches:
        raise FileNotFoundError(
            "no task1 annotation file matching %r under %s" % (ANNOTATION_GLOB, split_dir)
        )
    if len(matches) > 1:
        raise RuntimeError(
            "ambiguous task1 annotation files under %s: %s"
            % (split_dir, ", ".join(p.name for p in matches))
        )
    return matches[0]


def read_annotations(tsv_path: Path) -> Tuple[Dict[str, List[str]], AnnotationCounters]:
    """Return {doc_id: sorted unique codes} for the DIAGNOSTICO rows of one split."""
    counters = AnnotationCounters()
    by_doc: Dict[str, set] = {}

    with tsv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            counters.rows_total += 1
            fields = line.split("\t")
            if lineno == 1 and fields[0].strip().lower() in {"doc_id", "docid", "file"}:
                counters.rows_total -= 1
                counters.rows_header_skipped += 1
                continue
            if len(fields) < 3:
                counters.rows_too_few_fields += 1
                continue
            doc_id = fields[0].strip()
            label = fields[1].strip().upper()
            raw_code = fields[2]
            if label != KEEP_LABEL:
                counters.rows_wrong_task_label += 1
                continue
            if AUTOMATIC_SUGGESTION_RE.search(raw_code):
                counters.rows_automatic_suggestion += 1
            code = normalize_code(raw_code)
            if code is None:
                counters.rows_malformed_code += 1
                log("WARNING: %s:%d dropped non-code annotation %r (doc %s)",
                    tsv_path.name, lineno, raw_code.strip(), doc_id)
                continue
            if not doc_id:
                counters.rows_too_few_fields += 1
                continue
            counters.rows_kept += 1
            by_doc.setdefault(doc_id, set()).add(code)

    return {doc: sorted(codes) for doc, codes in by_doc.items()}, counters


def read_case_text(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as fh:
        return normalize_case_text(fh.read())


def build_split(
    split_dir: Path,
    split_name: str,
    limit: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Build the record list for one split plus its raw (pre-tokenizer) provenance."""
    text_dir = split_dir / TEXT_SUBDIR
    if not text_dir.is_dir():
        raise FileNotFoundError("missing text directory: %s" % text_dir)

    # Lesson from Lab 2: on a gcsfuse mount, prime the implicit-dirs cache with an explicit
    # listing before anything globs the prefix. Without it a glob can come back empty and the
    # failure presents as "the corpus is missing" rather than as a FUSE first-touch problem.
    # This is a no-op on a real filesystem, and --data-root is allowed to be /gcs/... .
    for primed in (split_dir, text_dir):
        log("  primed %s: %d entries", primed, len(list(primed.iterdir())))

    ann_path = find_annotation_file(split_dir)
    codes_by_doc, counters = read_annotations(ann_path)

    text_paths = sorted(text_dir.glob("*.txt"))
    if not text_paths:
        raise FileNotFoundError("no *.txt case files under %s" % text_dir)

    doc_ids_on_disk = {p.stem for p in text_paths}
    annotated_without_text = sorted(set(codes_by_doc) - doc_ids_on_disk)
    if annotated_without_text:
        raise FileNotFoundError(
            "%d annotated doc_ids have no %s/*.txt file, e.g. %s"
            % (len(annotated_without_text), TEXT_SUBDIR, annotated_without_text[:10])
        )

    records: List[Dict[str, object]] = []
    n_no_annotations = 0
    n_empty_text = 0
    for path in text_paths:
        doc_id = path.stem
        codes = codes_by_doc.get(doc_id)
        if not codes:
            n_no_annotations += 1
            log("WARNING: %s/%s has no task1 diagnosis annotations, skipped",
                split_name, path.name)
            continue
        text = read_case_text(path)
        if not text:
            n_empty_text += 1
            log("WARNING: %s/%s is empty after normalisation, skipped", split_name, path.name)
            continue
        prompt = build_prompt(text)
        records.append({
            "doc_id": doc_id,
            "text": text,
            "codes": codes,
            "prompt": prompt,
            "completion": build_completion(codes),
        })

    if not records:
        raise RuntimeError("split %s produced zero usable records" % split_name)

    shuffled = split_name in SHUFFLED_SPLITS
    if shuffled:
        random.Random(seed).shuffle(records)
    if limit > 0:
        records = records[:limit]

    provenance = {
        "split_dir": str(split_dir),
        "annotation_file": ann_path.name,
        "n_text_files": len(text_paths),
        "n_docs": len(records),
        "n_docs_skipped_no_annotations": n_no_annotations,
        "n_docs_skipped_empty_text": n_empty_text,
        "annotation_rows": counters.as_dict(),
        "order": ("shuffled(seed=%d)" % seed) if shuffled else "sorted_by_doc_id",
        "limit_applied": limit if limit > 0 else None,
    }
    return records, provenance


# --------------------------------------------------------------------------------------
# Tokenizer-driven statistics
# --------------------------------------------------------------------------------------
def load_tokenizer(spec: str):
    try:
        from transformers import AutoTokenizer  # imported lazily: heavy dependency
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            "transformers is required for --tokenizer statistics. "
            "Install it (e.g. `uv pip install 'transformers==4.57.1'`) and re-run.\n"
            "Underlying import error: %s" % exc
        )
    path = Path(spec).expanduser()
    kwargs: Dict[str, object] = {}
    if path.is_dir():
        spec_resolved = str(path)
        kwargs["local_files_only"] = True
        # Same gcsfuse first-touch lesson: touch the mount before transformers globs it for
        # tokenizer.json / vocab.json.
        log("  primed %s: %d entries", path, len(list(path.iterdir())))
    else:
        spec_resolved = spec  # treat as a Hugging Face hub id
    log("Loading tokenizer from %s …", spec_resolved)
    return AutoTokenizer.from_pretrained(spec_resolved, use_fast=True, **kwargs)


def token_lengths(tokenizer, texts: Sequence[str], add_special_tokens: bool) -> List[int]:
    """Token count per text, batched through the fast tokenizer."""
    if not texts:
        return []
    lengths: List[int] = []
    batch = 128
    for start in range(0, len(texts), batch):
        enc = tokenizer(
            list(texts[start:start + batch]),
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


def truncation_table(
    prompt_tokens: Sequence[int],
    total_tokens: Sequence[int],
    seq_lens: Sequence[int],
) -> List[Dict[str, object]]:
    """Per-seq_len truncation behaviour under right-truncation to a fixed window.

    The completion sits at the END of the training sequence, so any overflow eats the
    label first.  Two distinct failure modes are reported:

      truncated                -> total > seq_len, some of the label is lost
      completion_fully_lost    -> prompt alone already fills the window, the example
                                  carries no supervision signal at all
    """
    n = len(total_tokens)
    rows: List[Dict[str, object]] = []
    for seq_len in seq_lens:
        n_trunc = sum(1 for t in total_tokens if t > seq_len)
        n_lost = sum(1 for p in prompt_tokens if p >= seq_len)
        kept = [min(t, seq_len) for t in total_tokens]
        mean_kept = float(sum(kept)) / n if n else 0.0
        rows.append({
            "seq_len": seq_len,
            "n_docs": n,
            "n_truncated": n_trunc,
            "frac_truncated": (n_trunc / n) if n else None,
            "n_completion_fully_lost": n_lost,
            "frac_completion_fully_lost": (n_lost / n) if n else None,
            "mean_kept_tokens": mean_kept,
            "mean_padding_fraction": (1.0 - mean_kept / seq_len) if seq_len else None,
        })
    return rows


def split_stats(
    records: Sequence[Dict[str, object]],
    tokenizer,
    seq_lens: Sequence[int],
    add_special_tokens: bool,
    eos_token_count: int,
    special_tokens_overhead: int,
) -> Dict[str, object]:
    texts = [str(r["text"]) for r in records]
    prompts = [str(r["prompt"]) for r in records]
    completions = [str(r["completion"]) for r in records]

    log("  tokenizing %d case texts / prompts / completions …", len(records))
    text_tok = token_lengths(tokenizer, texts, add_special_tokens)
    prompt_tok = token_lengths(tokenizer, prompts, add_special_tokens)
    completion_tok = token_lengths(tokenizer, completions, add_special_tokens)
    # The prompt and the completion are encoded separately, so a tokenizer that adds N special
    # tokens per sequence contributes them twice; the concatenated training text pays them once.
    # Qwen3 measures N=0, but subtracting the one duplicated copy keeps the table honest for any
    # tokenizer someone points this at.
    total_tok = [
        p + c + eos_token_count - special_tokens_overhead
        for p, c in zip(prompt_tok, completion_tok)
    ]

    words = [len(str(r["text"]).split()) for r in records]
    per_doc_codes = [len(r["codes"]) for r in records]  # type: ignore[arg-type]
    unique_codes = sorted({c for r in records for c in r["codes"]})  # type: ignore[union-attr]

    return {
        "n_docs": len(records),
        "codes": {
            "unique": len(unique_codes),
            "total_assignments": sum(per_doc_codes),
            "per_doc_mean": (float(sum(per_doc_codes)) / len(per_doc_codes))
            if per_doc_codes else None,
            "per_doc": describe(per_doc_codes),
        },
        "case_text_words": describe(words),
        "tokens": {
            "case_text": describe(text_tok),
            "prompt": describe(prompt_tok),
            "completion": describe(completion_tok),
            "total": describe(total_tok),
        },
        "truncation": truncation_table(prompt_tok, total_tok, seq_lens),
        "_unique_codes": unique_codes,  # popped before serialisation
    }


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------
JSONL_KEY_ORDER = ("doc_id", "text", "codes", "prompt", "completion")


def write_jsonl(records: Sequence[Dict[str, object]], path: Path) -> Dict[str, object]:
    # ensure_ascii=True on purpose: the JSONL is read inside containers whose locale may be
    # C/POSIX, where Python's default text encoding is ASCII and a raw UTF-8 accent would
    # raise UnicodeDecodeError.  Pure-ASCII output loads identically everywhere.
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            ordered = {k: rec[k] for k in JSONL_KEY_ORDER}
            fh.write(json.dumps(ordered, ensure_ascii=True, sort_keys=False))
            fh.write("\n")
    return {
        "path": str(path),
        "lines": len(records),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_seq_lens(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("seq_len must be positive, got %d" % value)
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--seq-lens must list at least one value")
    return sorted(set(values))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_data.py",
        description="Convert the raw CodiEsp corpus into a tokenizer-ready SFT dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # None of the three is `required=True`: scripts/run-all.sh invokes this script with no
    # arguments and only environment variables, so the fallback chain in ENV_FALLBACKS is what
    # makes that call work. main() raises a pointed SystemExit when neither is present.
    parser.add_argument(
        "--data-root", default=None,
        help="Directory containing the per-split subdirectories (e.g. .../data/train_dev). "
             "Falls back to $%s, then $%s." % ENV_FALLBACKS["data_root"],
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for train.jsonl, dev.jsonl and dataset_stats.json. "
             "Falls back to $%s, then $%s." % ENV_FALLBACKS["out"],
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="Path to a local Qwen3 tokenizer directory (the staged model dir works) or a "
             "Hugging Face hub id such as Qwen/Qwen3-4B. Falls back to the first of $%s, $%s, $%s "
             "that names a directory on this host." % ENV_FALLBACKS["tokenizer"],
    )
    parser.add_argument(
        "--upload-to", default=os.environ.get(UPLOAD_ENV, "").strip(),
        help="gs:// prefix to publish the raw splits and the prepared files to, so the "
             "containerised backends all read identical bytes. Defaults to $%s; empty disables "
             "the upload." % UPLOAD_ENV,
    )
    parser.add_argument(
        "--no-upload", dest="upload", action="store_false",
        help="Never publish to GCS, whatever --upload-to or $%s says." % UPLOAD_ENV,
    )
    parser.set_defaults(upload=True)
    parser.add_argument(
        "--splits", default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split directory names to process.",
    )
    parser.add_argument(
        "--seq-lens", type=parse_seq_lens, default=list(DEFAULT_SEQ_LENS),
        help="Comma-separated sequence lengths for the truncation table.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the deterministic shuffle of the train split.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Cap each split to N documents (0 = no cap). Smoke-test aid only.",
    )
    parser.add_argument(
        "--no-special-tokens", dest="add_special_tokens", action="store_false",
        help="Count tokens with add_special_tokens=False. The default matches the training "
             "script, which calls the tokenizer with transformers' default of True.",
    )
    parser.set_defaults(add_special_tokens=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    sources: Dict[str, str] = {}
    for kind, must_be_dir in (("data_root", True), ("out", False), ("tokenizer", True)):
        if getattr(args, kind):
            sources[kind] = "--" + kind.replace("_", "-")
            continue
        value, var = env_fallback(kind, must_be_dir)
        if value is None:
            raise SystemExit(
                "--%s was not given and none of the environment fallbacks %s %s. Pass the flag "
                "explicitly." % (
                    kind.replace("_", "-"),
                    "/".join("$" + v for v in ENV_FALLBACKS[kind]),
                    "names a directory on this host" if must_be_dir else "is set",
                )
            )
        setattr(args, kind, value)
        sources[kind] = "$" + var
        log("--%s not given, using %s=%s", kind.replace("_", "-"), var, value)

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    upload_target = args.upload_to.strip() if args.upload else ""
    if upload_target and not upload_target.startswith("gs://"):
        raise SystemExit("--upload-to must be a gs:// URI, got %r" % upload_target)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not split_names:
        raise SystemExit("--splits resolved to an empty list")
    if not data_root.is_dir():
        raise SystemExit("--data-root is not a directory: %s" % data_root)
    # Validate every split before the (slow) tokenizer load so bad input fails immediately.
    missing = [s for s in split_names if not (data_root / s).is_dir()]
    if missing:
        raise SystemExit(
            "split director%s not found under %s: %s"
            % ("y" if len(missing) == 1 else "ies", data_root, ", ".join(missing))
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)
    eos_token = getattr(tokenizer, "eos_token", None)
    # How many tokens the trainer's trailing eos_token contributes. Computed rather than
    # assumed to be 1, so the truncation table stays correct for any tokenizer.
    eos_token_count = (
        len(tokenizer(eos_token, add_special_tokens=False)["input_ids"]) if eos_token else 0
    )
    # Qwen tokenizers add nothing for add_special_tokens=True; measured, not assumed.
    special_tokens_overhead = len(
        tokenizer("", add_special_tokens=args.add_special_tokens)["input_ids"]
    )

    per_split_stats: Dict[str, object] = {}
    per_split_files: Dict[str, object] = {}
    unique_codes_by_split: Dict[str, List[str]] = {}

    for split in split_names:
        split_dir = data_root / split
        log("Building split %s from %s …", split, split_dir)
        records, provenance = build_split(split_dir, split, args.limit, args.seed)
        log("  %d documents kept", len(records))

        stats = split_stats(
            records, tokenizer, args.seq_lens, args.add_special_tokens, eos_token_count,
            special_tokens_overhead,
        )
        unique_codes_by_split[split] = stats.pop("_unique_codes")  # type: ignore[arg-type]
        stats.update(provenance)
        per_split_stats[split] = stats

        out_path = out_dir / ("%s.jsonl" % split)
        per_split_files[out_path.name] = write_jsonl(records, out_path)
        log("  wrote %s (%d lines, %d bytes)",
            out_path, per_split_files[out_path.name]["lines"],
            per_split_files[out_path.name]["bytes"])

    # Cross-split code coverage: how much of dev the model has never seen a label for.
    train_codes = set(unique_codes_by_split.get("train", []))
    union_codes = set()
    for codes in unique_codes_by_split.values():
        union_codes |= set(codes)
    code_coverage: Dict[str, object] = {
        "unique_per_split": {s: len(c) for s, c in unique_codes_by_split.items()},
        "unique_union": len(union_codes),
    }
    if train_codes:
        for split, codes in unique_codes_by_split.items():
            if split == "train":
                continue
            unseen = sorted(set(codes) - train_codes)
            code_coverage["%s_codes_unseen_in_train" % split] = len(unseen)
            code_coverage["%s_frac_codes_unseen_in_train" % split] = (
                len(unseen) / len(set(codes)) if codes else None
            )

    stats_doc = {
        "generated_utc": utc_now_iso(),
        "generator": "src/prepare_data.py",
        "argv": list(sys.argv[1:] if argv is None else argv),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "argument_sources": sources,
        "upload_target": upload_target or None,
        "task": "CodiEsp task1 (DIAGNOSTICO) only; task2/task3 files present but ignored",
        "seed": args.seed,
        "tokenizer": {
            "spec": args.tokenizer,
            "name_or_path": getattr(tokenizer, "name_or_path", args.tokenizer),
            "class": type(tokenizer).__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)) or None,
            "eos_token": eos_token,
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        },
        "token_accounting": {
            "training_text": TRAINING_TEXT_RECIPE,
            "add_special_tokens": bool(args.add_special_tokens),
            "special_tokens_overhead_per_sequence": special_tokens_overhead,
            "eos_tokens_appended": eos_token_count,
            "total_tokens": (
                "prompt_tokens + completion_tokens + eos_tokens_appended "
                "- special_tokens_overhead_per_sequence (counted twice by encoding the prompt "
                "and the completion separately)"
            ),
            "truncation_side": "right (the completion is lost first)",
        },
        # Scope of the numbers below, so nothing here can be quoted as a fact about a different
        # tokenisation than the one that produced it.
        "applies_to": {
            "tokens_and_truncation": (
                "the prompt/completion pair in these JSONL files, tokenised as "
                + TRAINING_TEXT_RECIPE
                + ", right-truncated to seq_len"
            ),
            "consumers": {
                "src/train_qwen3_icd.py": (
                    "imports this module, reads <data_root>/<split>.jsonl (falling back to "
                    "prepare_data.build_split on the raw bundle) and trains on "
                    + TRAINING_TEXT_RECIPE
                    + " — the truncation table is valid for it only while that stays true"
                ),
                "src/evaluate.py": "reads dev.jsonl (doc_id, text, codes; prompt on request)",
            },
        },
        "prompt": {
            "template": PROMPT_TEMPLATE,
            "template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
            "delimiter": PROMPT_DELIMITER,
            "completion_separator": COMPLETION_SEPARATOR,
            "empty_completion": EMPTY_COMPLETION,
            "code_order": "lexicographic ascending, deduplicated, lowercase",
        },
        "seq_lens_evaluated": list(args.seq_lens),
        "splits": per_split_stats,
        "code_coverage": code_coverage,
        "files": per_split_files,
    }

    stats_path = out_dir / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats_doc, fh, indent=2, ensure_ascii=True, sort_keys=False)
        fh.write("\n")
    log("Wrote %s", stats_path)

    # Console summary: the truncation table is what the report quotes, so print it.
    for split, stats in per_split_stats.items():
        tokens = stats["tokens"]["total"]  # type: ignore[index]
        log("%s: %d docs | total tokens p50=%.0f p90=%.0f p99=%.0f max=%.0f",
            split, stats["n_docs"], tokens["p50"], tokens["p90"],  # type: ignore[index]
            tokens["p99"], tokens["max"])  # type: ignore[index]
        for row in stats["truncation"]:  # type: ignore[index]
            log("    seq_len=%-5d truncated %5.1f%%  label fully lost %5.1f%%  padding %5.1f%%",
                row["seq_len"], 100.0 * row["frac_truncated"],
                100.0 * row["frac_completion_fully_lost"],
                100.0 * row["mean_padding_fraction"])
        # A workload where most examples carry no label at all is still a valid *throughput*
        # benchmark, but it must be a knowing choice rather than a silent one, so say it out loud
        # for every seq_len the campaign actually uses.
        for row in stats["truncation"]:  # type: ignore[index]
            if row["frac_completion_fully_lost"] and row["frac_completion_fully_lost"] > 0.5:
                log("WARNING: at seq_len=%d, %.1f%% of %s examples lose the ENTIRE label — at that "
                    "window the run measures throughput only, not learning. Zero label loss needs "
                    "seq_len > %.0f (max prompt tokens); zero truncation needs seq_len >= %.0f.",
                    row["seq_len"], 100.0 * row["frac_completion_fully_lost"], split,
                    stats["tokens"]["prompt"]["max"],  # type: ignore[index]
                    stats["tokens"]["total"]["max"])  # type: ignore[index]

    if upload_target:
        log("Publishing to %s (raw splits + prepared files) …", upload_target)
        gcs_publish(
            data_root, out_dir,
            sorted(per_split_files) + [stats_path.name], upload_target,
        )
    else:
        log("No upload target (--no-upload or empty $%s) — local files only. The containerised "
            "backends read $DATA_GCS_URI, so publish before submitting any Job.", UPLOAD_ENV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
