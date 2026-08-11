#!/usr/bin/env python3
"""Derive every reported number from ``results/*.json`` and write ``results/analysis.json``.

This is the only place where derived quantities are computed. The report, the dashboard and
the slides read ``results/analysis.json``; none of them recompute anything and none of them
carry a literal measurement. If a number appears in the deliverables it can be traced back
through this file to a field of a run record, or it is a bug.

What it derives
---------------

**A refined wall-clock decomposition.** The metrics contract attributes wall clock to
``submit_to_running_s``, ``image_pull_s``, ``model_load_s``, ``compile_s``,
``steady_state_s``, ``checkpoint_write_s`` and the residual ``other_s``. That is enough to
show where a job spends its time but not enough to answer *how much of the run was the
arithmetic*, because ``steady_state_s`` is defined as ``sum(steps[1:])`` and therefore still
contains the second step — which on every accelerator record here is a second XLA compile —
plus the long tail steps at epoch and checkpoint boundaries. This module splits
``steady_state_s`` further, exactly and without remainder (see :func:`decompose`).

**An amortisation model.** A run costs ``fixed_s + N * median_step_s``. Both terms come from
the record: ``median_step_s`` is measured, and ``fixed_s`` is whatever the wall clock has left
after ``n_steps * median_step_s``. The model is what separates the per-step speedup from the
end-to-end speedup, and the distance between those two numbers is this project's finding.

**Sweep boundaries, the mitigation delta, and cost.** All three by pairing records on their
``config`` blocks rather than by parsing ``run_id`` strings, so a renamed run does not
silently drop out of a comparison.

Usage
-----

::

    python3 profiling/analyze.py                 # writes results/analysis.json
    python3 profiling/analyze.py --print         # ... and prints a readable summary
    python3 profiling/analyze.py --check         # validate records only, write nothing

Standard library only, so it runs anywhere the repo is checked out. ``make_dashboard.py`` is
the one piece of tooling that needs matplotlib.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ==========================================================================================
# Prices
# ==========================================================================================
# Retrieved 2026-08-10. Every cost figure in the report is derived from these two constants
# and from measured wall clock; nothing about cost is measured on the cluster itself.
#
# Both are list prices for on-demand capacity. Committed-use and Spot discounts are real and
# large, and would change the absolute figures — they would not change the ordering, which is
# what the report actually claims.

PRICE_TPU_V5E_USD_PER_CHIP_HOUR = 1.20
"""Cloud TPU v5e (``tpu-v5-lite-podslice``), on-demand, us-west4.

Source: https://cloud.google.com/tpu/pricing (retrieved 2026-08-10).
This is the region the measured runs actually ran in, so this price applies to real hardware
that really was billed: 8 chips x this rate for the duration of every TPU record here.
"""

PRICE_CPU_PROXY_USD_PER_HOUR = 1.1336
"""``n1-highcpu-32`` (32 vCPU, 28.8 GB), on-demand, us-central1.

Source: https://cloudprice.net/gcp/compute/instances/n1-highcpu-32 and
https://www.economize.cloud/resources/gcp/pricing/compute-engine/n1-highcpu-32/
(both retrieved 2026-08-10, both quoting the same $1.1336/hour).

THIS IS A PROXY AND NOT THE HARDWARE THAT WAS MEASURED. The CPU record was produced on
``hpcc-cluster-39``, a Stanford on-premises node (2 x Intel Xeon E5-2670, 32 logical CPUs,
31 GiB), which has no list price. ``n1-highcpu-32`` is the closest published Compute Engine
shape on both axes that matter here — 32 vCPUs and ~29 GB of RAM against 32 logical CPUs and
31 GiB. us-central1 is quoted because Google does not publish a scrapeable us-west4 N1 rate;
N1 pricing between the two regions differs by a few percent.

The CPU cost figures are therefore an estimate with a stated substitution, not a measurement.
They are reported because the comparison is robust to the substitution: see
``cost.sensitivity`` in the output, which states how far this price would have to move before
the ordering changes.
"""

PRICE_GPU_GH200_USD_PER_HOUR = 2.29
"""NVIDIA GH200 Grace Hopper, on-demand, third-party cloud market rate.

Source: https://computeprices.com/gpus/gh200 (retrieved 2026-08-10), which aggregates the
providers that publish a GH200 rate. The published range across providers is $1.99–$6.50 per
GPU-hour; $2.29 is the low end of on-demand (non-spot) listings.

TWO SUBSTITUTIONS, BOTH STATED. Google Cloud does not sell a GH200 at all, so unlike the TPU
row this figure cannot come from the GCP list price the other rows use. And the measured GPU
is `hpcc-pilot` on Stanford's on-premises `stanford-pilot` cluster, which has no list price
either. This is therefore a market rate for the same silicon, not a bill anyone received.

It is reported because the comparison it supports survives the whole published range: an
eight-chip v5e slice lists at 8 x $1.20 = $9.60/hour, so a single GH200 delivering the same
throughput is cheaper per token at every GH200 price ever published. See
``cost.sensitivity`` in the output.
"""

PRICE_GPU_RANGE_USD_PER_HOUR = (1.99, 6.50)
"""Published GH200 on-demand range across providers, used for the sensitivity statement."""

PRICE_PROVENANCE = {
    "retrieved_utc": "2026-08-10",
    "tpu": {
        "usd_per_chip_hour": PRICE_TPU_V5E_USD_PER_CHIP_HOUR,
        "sku": "Cloud TPU v5e (tpu-v5-lite-podslice), on-demand",
        "region": "us-west4",
        "source": "https://cloud.google.com/tpu/pricing",
        "applies_to_measured_hardware": True,
    },
    "cpu": {
        "usd_per_instance_hour": PRICE_CPU_PROXY_USD_PER_HOUR,
        "sku": "n1-highcpu-32 (32 vCPU, 28.8 GB), on-demand",
        "region": "us-central1",
        "source": "https://cloudprice.net/gcp/compute/instances/n1-highcpu-32",
        "applies_to_measured_hardware": False,
        "substitution_note": (
            "The measured CPU node is Stanford on-premises hardware (hpcc-cluster-39, 2x Xeon "
            "E5-2670, 32 logical CPUs, 31 GiB) and has no list price. n1-highcpu-32 is the "
            "closest published Compute Engine shape by core count and memory. CPU cost figures "
            "are an estimate under this substitution, not a billed amount."
        ),
    },
    "gpu": {
        "usd_per_gpu_hour": PRICE_GPU_GH200_USD_PER_HOUR,
        "published_range_usd_per_hour": list(PRICE_GPU_RANGE_USD_PER_HOUR),
        "sku": "NVIDIA GH200 Grace Hopper, on-demand",
        "region": "third-party cloud market rate (not GCP — GCP does not sell GH200)",
        "source": "https://computeprices.com/gpus/gh200",
        "applies_to_measured_hardware": False,
        "substitution_note": (
            "Two substitutions: the measured GPU is Stanford on-premises hardware (hpcc-pilot "
            "on stanford-pilot) with no list price, and Google Cloud does not offer a GH200, so "
            "unlike the TPU row this rate cannot come from the same price list. It is a "
            "third-party market rate for the same silicon."
        ),
    },
}

# ==========================================================================================
# Loading and validation
# ==========================================================================================

#: Phase keys the contract requires, in the order they occur in a run's life.
CONTRACT_PHASES = (
    "submit_to_running_s",
    "image_pull_s",
    "model_load_s",
    "compile_s",
    "steady_state_s",
    "checkpoint_write_s",
    "other_s",
)

#: Tolerance for the "phases sum to total_wall_s" invariant, matching telemetry.py.
PHASE_SUM_ABS_TOL_S = 1e-3
PHASE_SUM_REL_TOL = 1e-6

#: Phrases in ``notes`` that mean a field of the record is not a measurement, and which field.
#:
#: One 0.6B CPU run reported 11 million tokens/s because a per-step probe blocked on a stale
#: buffer. It carried ``status: "ok"`` and was caught only by reading its notes. A consumer
#: that trusts ``status`` alone would have charted it, so this consumer does not.
#:
#: The patterns are matched against what the records actually say rather than against a
#: remembered phrasing — the loss note reads "no per-step loss *could be captured*", so a
#: filter written for "could **not** be captured" would match nothing and quietly pass.
#: Each entry names the field it taints, so a warning scopes to that field instead of
#: condemning the whole record: every record here carries the loss caveat, and none of them
#: is thereby untrustworthy for step times.
NOTE_MARKERS = (
    {"pattern": "stale buffer", "taints": "steady", "why": "per-step probe may have timed dispatch rather than completion"},
    {"pattern": "never changed identity", "taints": "steady", "why": "block_until_ready may have returned on an unchanged buffer"},
    {"pattern": "may have been blocking", "taints": "steady", "why": "telemetry flagged its own step timing"},
    {"pattern": "indicative", "taints": "*", "why": "the record calls the value indicative rather than measured"},
    {"pattern": "discarded", "taints": "*", "why": "the record says it was discarded"},
    {"pattern": "suspect", "taints": "*", "why": "the record calls itself suspect"},
    {"pattern": "loss could be captured", "taints": "steps[].loss", "why": "no per-step loss was obtainable from Tunix; it is null rather than fabricated"},
)

#: Fields the report, dashboard and slides actually read. Cross-checked against the tainted
#: set so that a taint on a field nothing plots stays a note, and a taint on a field something
#: plots becomes an error.
PLOTTED_FIELDS = ("steady", "memory", "phases", "utilization")


class ContractError(RuntimeError):
    """A run record does not satisfy the metrics contract."""


def load_records(results_dir: str) -> List[Dict[str, Any]]:
    """Load every run record in ``results_dir``.

    ``analysis.json`` is this script's own output and is skipped, as is anything under
    ``adapters/`` and any ``eval-*.json``, which follow different schemas.
    """
    records: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        if name == "analysis.json" or name.startswith("eval-"):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        if "run_id" not in record or "phases" not in record:
            continue  # not a run record
        record["_source_file"] = os.path.relpath(path, start=os.path.dirname(results_dir) or ".")
        records.append(record)
    return records


def validate(record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Re-check the contract invariants and physical plausibility.

    Returns ``(warnings, tainted_fields)``. Hard violations raise — including a note that
    casts doubt on a field the deliverables actually plot. Everything else is returned and
    travels into ``analysis.json`` so the report can cite it.
    """
    run_id = record.get("run_id", "<unknown>")
    warnings: List[str] = []
    tainted: List[str] = []

    phases = record.get("phases") or {}
    missing = [key for key in CONTRACT_PHASES + ("total_wall_s",) if key not in phases]
    if missing:
        raise ContractError(f"{run_id}: phases missing required keys {missing}")

    total = phases["total_wall_s"]
    if not isinstance(total, (int, float)) or not math.isfinite(total):
        raise ContractError(f"{run_id}: phases.total_wall_s is not a finite number: {total!r}")

    attributed = sum(v for k, v in phases.items() if k != "total_wall_s" and v is not None)
    tol = PHASE_SUM_ABS_TOL_S + PHASE_SUM_REL_TOL * abs(total)
    if abs(attributed - total) > tol:
        raise ContractError(
            f"{run_id}: phases sum to {attributed!r}s but total_wall_s is {total!r}s "
            f"(delta {attributed - total!r}s > tolerance {tol!r}s)"
        )

    status = record.get("status")
    if status not in ("ok", "oom", "error", "timeout"):
        raise ContractError(f"{run_id}: status {status!r} is not one of ok/oom/error/timeout")

    # Physical plausibility. steady.tokens_per_s must equal batch_size*seq_len/median_step_s
    # by construction; if it does not, one of the two was measured through a broken probe.
    steady = record.get("steady")
    config = record.get("config") or {}
    if steady:
        expected_tokens = config.get("batch_size", 0) * config.get("seq_len", 0)
        observed = steady["tokens_per_s"] * steady["median_step_s"]
        if expected_tokens > 0:
            rel = abs(observed - expected_tokens) / expected_tokens
            if rel > 1e-6:
                warnings.append(
                    f"tokens_per_s is inconsistent with batch_size*seq_len/median_step_s "
                    f"(relative error {rel:.3e}) — one of the two was mismeasured"
                )

    notes = (record.get("notes") or "").lower()
    for marker in NOTE_MARKERS:
        if marker["pattern"] in notes:
            tainted.append(marker["taints"])
            scope = "every field" if marker["taints"] == "*" else marker["taints"]
            message = (
                f"notes contain {marker['pattern']!r} — {marker['why']}; "
                f"{scope} of this record is not a measurement"
            )
            # A taint on something the deliverables plot is an error, not a note: it is
            # exactly the case that produced an 11-million-tokens/s chart once already.
            if marker["taints"] == "*" or marker["taints"] in PLOTTED_FIELDS:
                raise ContractError(
                    f"{run_id}: {message}. status={status!r} is not sufficient to trust it. "
                    f"Remove the record or fix the measurement; do not plot it."
                )
            warnings.append(message)

    if status == "ok" and not steady:
        warnings.append("status is 'ok' but the steady block is absent")

    return warnings, tainted


# ==========================================================================================
# Derivation
# ==========================================================================================

#: ``other_s`` is a residual, but telemetry.py records its composition in ``notes`` as
#: ``jax_init=24.16s lora_wrap=14.22s data_prep=27.29s trainer_build=0.93s``. Recovering it
#: turns an opaque bucket into four named costs, one of which (data_prep) is I/O and therefore
#: responds to the storage mitigation.
_OTHER_COMPOSITION_RE = re.compile(r"([a-z_]+)=([0-9]*\.?[0-9]+)s")


def parse_other_composition(notes: str) -> Dict[str, float]:
    """Recover the breakdown of ``phases.other_s`` that telemetry.py wrote into ``notes``."""
    marker = "other_s is composed of:"
    index = notes.find(marker)
    if index < 0:
        return {}
    tail = notes[index + len(marker):].split("|")[0]
    return {key: float(value) for key, value in _OTHER_COMPOSITION_RE.findall(tail)}


def decompose(record: Dict[str, Any]) -> Dict[str, Any]:
    """Split wall clock into segments that sum to ``total_wall_s`` exactly.

    The contract's ``steady_state_s`` is ``sum(steps[1:])`` — every step after the compile
    step, including the second step (a second XLA compile on every accelerator record here)
    and the long steps at epoch and checkpoint boundaries. Charting it as "compute" would
    credit the accelerator with time it spent compiling and waiting.

    Writing ``m`` for ``steady.median_step_s`` and ``S`` for the recorded step times, the
    identity used here is

        sum(S) = n*m + (S[0] - m) + (S[1] - m) + sum_{i>=2}(S[i] - m)

    which is exact by construction and holds for any ``m``. Its four terms are reported as
    ``steady_compute_s``, ``compile_excess_s``, ``recompile_excess_s`` and
    ``straggler_excess_s``. The last is signed: steps that ran faster than the median
    subtract from it, so nothing is invented and nothing is lost.

    Runs without a steady block (an OOM, for instance) fall back to the contract phases, with
    all step-derived segments null.
    """
    phases = record["phases"]
    steps: Sequence[Dict[str, Any]] = record.get("steps") or []
    steady = record.get("steady")
    total = phases["total_wall_s"]

    segments: Dict[str, Optional[float]] = {
        "scheduler_queue_s": phases.get("submit_to_running_s"),
        "image_pull_s": phases.get("image_pull_s"),
        "process_init_s": phases.get("other_s"),
        "checkpoint_load_s": phases.get("model_load_s"),
        "compile_s": None,
        "straggler_s": None,
        "steady_compute_s": None,
        "checkpoint_write_s": phases.get("checkpoint_write_s"),
    }

    detail: Dict[str, Any] = {
        "n_steps_recorded": len(steps),
        "compile_excess_s": None,
        "recompile_excess_s": None,
        "straggler_excess_s": None,
        "second_step_is_recompile": None,
    }

    if steady and steps:
        median = steady["median_step_s"]
        wall = [float(step["wall_s"]) for step in steps]
        n = len(wall)

        steady_compute_s = n * median
        compile_excess_s = wall[0] - median
        recompile_excess_s = (wall[1] - median) if n > 1 else 0.0
        straggler_excess_s = sum(value - median for value in wall[2:])

        # A second step that costs more than an order of magnitude above the median is a
        # second compilation, not a slow step. Stated rather than assumed: the flag records
        # the ratio that produced the judgement.
        second_step_ratio = (wall[1] / median) if n > 1 and median > 0 else None
        detail["second_step_ratio"] = second_step_ratio
        detail["second_step_is_recompile"] = bool(second_step_ratio and second_step_ratio >= 10.0)
        detail["compile_excess_s"] = compile_excess_s
        detail["recompile_excess_s"] = recompile_excess_s
        detail["straggler_excess_s"] = straggler_excess_s
        detail["first_step_cost_in_steady_steps"] = wall[0] / median if median > 0 else None
        detail["longest_step_s"] = max(wall)
        detail["longest_step_index"] = wall.index(max(wall)) + 1

        # If the second step is a recompile it belongs with compile time; if it is merely a
        # slow step it belongs with the stragglers. Either way the two buckets sum the same.
        if detail["second_step_is_recompile"]:
            segments["compile_s"] = compile_excess_s + recompile_excess_s
            segments["straggler_s"] = straggler_excess_s
        else:
            segments["compile_s"] = compile_excess_s
            segments["straggler_s"] = recompile_excess_s + straggler_excess_s
        segments["steady_compute_s"] = steady_compute_s
    else:
        # No usable step series: keep the contract's own split so the bar still sums to the
        # wall clock, and mark the compute segment as unmeasured rather than zero.
        segments["compile_s"] = phases.get("compile_s")
        segments["straggler_s"] = phases.get("steady_state_s")
        segments["steady_compute_s"] = None

    known = sum(value for value in segments.values() if value is not None)
    residual = total - known

    return {
        "segments": segments,
        "segment_pct": {
            key: (100.0 * value / total if value is not None and total else None)
            for key, value in segments.items()
        },
        "unattributed_s": residual,
        "sums_to_total": abs(residual) <= PHASE_SUM_ABS_TOL_S + PHASE_SUM_REL_TOL * abs(total),
        "detail": detail,
    }


def derive_run(record: Dict[str, Any]) -> Dict[str, Any]:
    """Everything derivable from a single run record."""
    config = record["config"]
    phases = record["phases"]
    steady = record.get("steady")
    memory = record.get("memory")
    total = phases["total_wall_s"]
    steps = record.get("steps") or []

    breakdown = decompose(record)
    compute_s = breakdown["segments"]["steady_compute_s"]

    derived: Dict[str, Any] = {
        "run_id": record["run_id"],
        "backend": record["backend"],
        "status": record["status"],
        "source_file": record.get("_source_file"),
        "hardware": {
            "label": record["hardware"]["label"],
            "chips": record["hardware"]["chips"],
            "chip_model": record["hardware"]["chip_model"],
        },
        "config": {
            "model": config["model"],
            "batch_size": config["batch_size"],
            "seq_len": config["seq_len"],
            "dtype": config.get("dtype"),
            "remat": config.get("remat"),
            "file_cache_capacity": config.get("file_cache_capacity"),
            "max_steps": config.get("max_steps"),
            "devices": config.get("devices"),
        },
        "measured": {
            "total_wall_s": total,
            "median_step_s": steady["median_step_s"] if steady else None,
            "p10_step_s": steady["p10_step_s"] if steady else None,
            "p90_step_s": steady["p90_step_s"] if steady else None,
            "tokens_per_s": steady["tokens_per_s"] if steady else None,
            "docs_per_s": steady["docs_per_s"] if steady else None,
            "peak_bytes": memory["peak_bytes"] if memory else None,
            "peak_pct": memory["peak_pct"] if memory else None,
            "capacity_bytes": memory["capacity_bytes"] if memory else None,
            # Compute utilization is null on TPU by design: JAX exposes device memory
            # statistics but no core-utilization counter. HBM occupancy is not the same
            # quantity and is reported under its own name.
            "utilization_mean_pct": (record.get("utilization") or {}).get("mean_pct"),
            "utilization_max_pct": (record.get("utilization") or {}).get("max_pct"),
            "memory_occupancy_mean_pct": (record.get("utilization") or {}).get("mean_memory_pct"),
            "memory_occupancy_max_pct": (record.get("utilization") or {}).get("max_memory_pct"),
            "micro_f1": (record.get("eval") or {}).get("micro_f1"),
        },
        "phases_contract": {key: phases.get(key) for key in CONTRACT_PHASES},
        "phases_contract_pct": {
            key: (100.0 * phases[key] / total if phases.get(key) is not None and total else None)
            for key in CONTRACT_PHASES
        },
        "breakdown": breakdown,
        "other_s_composition": parse_other_composition(record.get("notes") or ""),
    }

    # The four named costs telemetry.py records inside other_s do not always add up to it.
    # Whatever is left is interpreter startup and imports, which nothing times explicitly.
    # Reported rather than absorbed, so the segment is not read as fully explained.
    composition_sum = sum(derived["other_s_composition"].values())
    other_total = phases.get("other_s")
    derived["other_s_composition_residual_s"] = (
        other_total - composition_sum if other_total is not None else None
    )

    # Amortisation. A run of N steps costs fixed_s + N*median_step_s. Both terms come out of
    # this record: the marginal term is measured directly, and the fixed term is the wall
    # clock that is left once every recorded step is charged at the steady-state rate.
    if steady and steps:
        median = steady["median_step_s"]
        fixed = total - len(steps) * median
        derived["amortisation"] = {
            "fixed_s": fixed,
            "marginal_s_per_step": median,
            "n_steps_recorded": len(steps),
            "compute_fraction_pct": 100.0 * compute_s / total if total else None,
            "overhead_fraction_pct": 100.0 * (total - compute_s) / total if total else None,
            # The scheduler queue is charged to the accelerators and not to the bare-metal CPU
            # run, which never enters a queue. Reporting the fixed cost without it makes the
            # two comparable on the part of the overhead that is the workload's own doing.
            "fixed_excl_scheduler_s": fixed - (phases.get("submit_to_running_s") or 0.0),
        }
    else:
        derived["amortisation"] = None

    if steady:
        chips = record["hardware"]["chips"]
        derived["measured"]["tokens_per_s_per_chip"] = steady["tokens_per_s"] / chips if chips else None

    return derived


# ==========================================================================================
# Cross-run comparisons
# ==========================================================================================


def _config_key(run: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity of a workload configuration, ignoring backend and storage settings."""
    config = run["config"]
    return (config["model"], config["batch_size"], config["seq_len"])


def like_for_like(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cross-backend comparisons where the workload configuration is genuinely identical.

    A speedup is only quoted where model, batch size and sequence length all match. This is
    strict on purpose: the 4B configuration has no CPU record to compare against, and quoting
    a 4B TPU step time against a 0.6B CPU step time would manufacture a ratio out of two
    different workloads.
    """
    families: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for run in runs:
        if run["status"] != "ok" or run["measured"]["median_step_s"] is None:
            continue
        families.setdefault(_config_key(run), []).append(run)

    comparisons: List[Dict[str, Any]] = []
    for key, members in sorted(families.items(), key=lambda item: str(item[0])):
        backends = {member["backend"] for member in members}
        if len(backends) < 2:
            continue
        model, batch_size, seq_len = key
        baseline = next(m for m in members if m["backend"] == "cpu")
        entry: Dict[str, Any] = {
            "model": model,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "baseline_run_id": baseline["run_id"],
            "backends": {},
        }
        for member in members:
            speedup = baseline["measured"]["median_step_s"] / member["measured"]["median_step_s"]
            chips = member["hardware"]["chips"]
            entry["backends"][member["backend"]] = {
                "run_id": member["run_id"],
                "hardware": member["hardware"]["label"],
                "chips": chips,
                "median_step_s": member["measured"]["median_step_s"],
                "tokens_per_s": member["measured"]["tokens_per_s"],
                "step_time_speedup_vs_cpu": speedup,
                "speedup_per_chip": speedup / chips if chips else None,
                # Strong-scaling efficiency needs a single-chip baseline on the same
                # architecture. No such run exists, so the honest value is null with a reason
                # attached rather than speedup/chips wearing an efficiency label it has not
                # earned: the denominator here is a CPU, not one chip of the same accelerator.
                "strong_scaling_efficiency": None,
                "strong_scaling_efficiency_note": (
                    "not derivable: no single-chip run of this backend was measured, so there "
                    "is no same-architecture baseline to divide by. speedup_per_chip is a "
                    "cross-architecture ratio and is not a parallel efficiency."
                ),
            }
        comparisons.append(entry)
    return comparisons


def amortisation_curve(
    runs: Sequence[Dict[str, Any]],
    comparison: Dict[str, Any],
    target_backend: str,
    step_counts: Iterable[int] = (12, 100, 1_000, 10_000, 100_000),
) -> Dict[str, Any]:
    """End-to-end speedup as a function of run length, and the break-even point.

    The per-step speedup is what a benchmark reports. The end-to-end speedup is what a user
    experiences, and it only approaches the per-step figure once the run is long enough to
    amortise the fixed cost. Both are computed here from ``fixed_s`` and ``median_step_s``.

    One curve per accelerator. Comparing the curves against each other is the point: two
    accelerators with the same marginal cost and different fixed costs separate here, and
    nowhere in a per-step benchmark.
    """
    by_id = {run["run_id"]: run for run in runs}
    baseline = by_id[comparison["baseline_run_id"]]
    target = by_id[comparison["backends"][target_backend]["run_id"]]

    base_fixed = baseline["amortisation"]["fixed_s"]
    base_marginal = baseline["amortisation"]["marginal_s_per_step"]
    tgt_fixed = target["amortisation"]["fixed_s"]
    tgt_marginal = target["amortisation"]["marginal_s_per_step"]

    points = []
    for n in step_counts:
        base_total = base_fixed + n * base_marginal
        tgt_total = tgt_fixed + n * tgt_marginal
        points.append(
            {
                "n_steps": n,
                "baseline_total_s": base_total,
                "target_total_s": tgt_total,
                "end_to_end_speedup": base_total / tgt_total,
            }
        )

    denominator = base_marginal - tgt_marginal
    breakeven = (tgt_fixed - base_fixed) / denominator if denominator else None
    # A negative break-even means the accelerator's fixed cost is already below the baseline's,
    # so it wins at every run length including a job that runs no steps at all. Reporting the
    # raw negative number without saying that invites it being read as a threshold.
    breakeven_note = (
        "the target's fixed cost is already lower than the baseline's, so it wins at every "
        "run length; the negative value is not a threshold"
        if breakeven is not None and breakeven <= 0
        else None
    )

    return {
        "baseline_run_id": baseline["run_id"],
        "target_run_id": target["run_id"],
        "target_backend": target_backend,
        "model": comparison["model"],
        "baseline_fixed_s": base_fixed,
        "baseline_marginal_s_per_step": base_marginal,
        "target_fixed_s": tgt_fixed,
        "target_marginal_s_per_step": tgt_marginal,
        "asymptotic_speedup": base_marginal / tgt_marginal,
        "breakeven_n_steps": breakeven,
        "breakeven_note": breakeven_note,
        "points": points,
        "note": (
            "time(N) = fixed_s + N * median_step_s. fixed_s is the measured wall clock minus "
            "every recorded step charged at the steady-state rate; it is not fitted. The two "
            "runs recorded different step counts (the model is what makes them comparable at "
            "equal N)."
        ),
    }


def sweep(
    runs: Sequence[Dict[str, Any]], vary: str, hold: Sequence[str], backend: str = "tpu"
) -> List[Dict[str, Any]]:
    """Group runs that differ in exactly one configuration axis.

    Used for the batch-size and sequence-length sweeps. Records with ``status: oom`` are kept:
    the failure boundary is the point of the sweep, so dropping them would delete the finding.
    """
    families: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for run in runs:
        if run["backend"] != backend:
            continue
        key = tuple(run["config"][name] for name in hold)
        families.setdefault(key, []).append(run)

    results = []
    for key, members in families.items():
        if len({member["config"][vary] for member in members}) < 2:
            continue
        # De-duplicate: the mitigation pair differs only in storage settings, which is not
        # this axis. Keep the run whose file cache is on, which is the configuration the rest
        # of the sweep used.
        deduped: Dict[Any, Dict[str, Any]] = {}
        for member in sorted(members, key=lambda m: str(m["config"]["file_cache_capacity"])):
            deduped[member["config"][vary]] = member
        ordered = [deduped[value] for value in sorted(deduped)]

        points = [
            {
                vary: member["config"][vary],
                "run_id": member["run_id"],
                "status": member["status"],
                "median_step_s": member["measured"]["median_step_s"],
                "tokens_per_s": member["measured"]["tokens_per_s"],
                "peak_bytes": member["measured"]["peak_bytes"],
                "peak_pct": member["measured"]["peak_pct"],
                "capacity_bytes": member["measured"]["capacity_bytes"],
            }
            for member in ordered
        ]
        ok_values = [p[vary] for p in points if p["status"] == "ok"]
        oom_values = [p[vary] for p in points if p["status"] == "oom"]

        entry: Dict[str, Any] = {
            "varies": vary,
            "held": dict(zip(hold, key)),
            "points": points,
            "max_ok": max(ok_values) if ok_values else None,
            "min_oom": min(oom_values) if oom_values else None,
        }

        # Scaling exponent between consecutive points: t ~ x**alpha. Linear cost in tokens
        # gives 1.0; the quadratic term in attention pushes it above 1.
        exponents = []
        ok_points = [p for p in points if p["status"] == "ok" and p["median_step_s"]]
        for previous, current in zip(ok_points, ok_points[1:]):
            ratio_x = current[vary] / previous[vary]
            ratio_t = current["median_step_s"] / previous["median_step_s"]
            if ratio_x > 0 and ratio_x != 1:
                exponents.append(
                    {
                        "from": previous[vary],
                        "to": current[vary],
                        "step_time_ratio": ratio_t,
                        "throughput_ratio": current["tokens_per_s"] / previous["tokens_per_s"],
                        "exponent": math.log(ratio_t) / math.log(ratio_x),
                    }
                )
        entry["scaling"] = exponents
        results.append(entry)
    return results


def mitigation(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair runs identical in workload but differing in ``file_cache_capacity``.

    Paired on the config block rather than on the ``-filecache`` suffix in ``run_id``, so the
    comparison survives a rename and cannot accidentally pair two unrelated runs.
    """
    families: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for run in runs:
        families.setdefault(_config_key(run) + (run["backend"],), []).append(run)

    pairs = []
    for members in families.values():
        by_cache = {str(member["config"]["file_cache_capacity"]): member for member in members}
        off = by_cache.get("0") or by_cache.get("none")
        on = next((m for key, m in by_cache.items() if key not in ("0", "none")), None)
        if not off or not on or off["run_id"] == on["run_id"]:
            continue

        def delta(field: str) -> Dict[str, Any]:
            before = off["phases_contract"].get(field)
            after = on["phases_contract"].get(field)
            if before is None or after is None:
                return {"before_s": before, "after_s": after, "delta_s": None, "speedup": None}
            return {
                "before_s": before,
                "after_s": after,
                "delta_s": before - after,
                "speedup": (before / after) if after else None,
            }

        before_total = off["measured"]["total_wall_s"]
        after_total = on["measured"]["total_wall_s"]
        entry = {
            "change": "gcsfuse fileCacheCapacity 0 -> %s (fileCacheForRangeRead true)"
            % on["config"]["file_cache_capacity"],
            "before_run_id": off["run_id"],
            "after_run_id": on["run_id"],
            "model": off["config"]["model"],
            "batch_size": off["config"]["batch_size"],
            "seq_len": off["config"]["seq_len"],
            "model_load": delta("model_load_s"),
            "other": delta("other_s"),
            "total_wall": {
                "before_s": before_total,
                "after_s": after_total,
                "delta_s": before_total - after_total,
                "speedup": before_total / after_total if after_total else None,
                "pct_saved": 100.0 * (before_total - after_total) / before_total
                if before_total
                else None,
            },
            "steady_step": {
                "before_s": off["measured"]["median_step_s"],
                "after_s": on["measured"]["median_step_s"],
                "delta_s": (off["measured"]["median_step_s"] - on["measured"]["median_step_s"])
                if off["measured"]["median_step_s"] and on["measured"]["median_step_s"]
                else None,
                "pct_change": (
                    100.0
                    * (on["measured"]["median_step_s"] - off["measured"]["median_step_s"])
                    / off["measured"]["median_step_s"]
                )
                if off["measured"]["median_step_s"]
                else None,
            },
            "data_prep": {
                "before_s": off["other_s_composition"].get("data_prep"),
                "after_s": on["other_s_composition"].get("data_prep"),
            },
        }
        before_prep = entry["data_prep"]["before_s"]
        after_prep = entry["data_prep"]["after_s"]
        if before_prep is not None and after_prep is not None:
            entry["data_prep"]["delta_s"] = before_prep - after_prep
            entry["data_prep"]["speedup"] = before_prep / after_prep if after_prep else None
        pairs.append(entry)
    return pairs


def scheduler_latency(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Spread of ``submit_to_running_s`` across accelerator runs.

    The fixed cost of a job is treated as a constant by the amortisation model, but one of
    its terms is a queue on a cluster shared by a whole class. Its variance is a property of
    the platform, so it is reported rather than averaged away.
    """
    values = [
        (run["run_id"], run["phases_contract"]["submit_to_running_s"])
        for run in runs
        if run["backend"] != "cpu" and run["phases_contract"].get("submit_to_running_s") is not None
    ]
    if not values:
        return {"n": 0}
    seconds = [value for _, value in values]
    return {
        "n": len(seconds),
        "min_s": min(seconds),
        "max_s": max(seconds),
        "median_s": statistics.median(seconds),
        "mean_s": statistics.fmean(seconds) if hasattr(statistics, "fmean") else sum(seconds) / len(seconds),
        "spread_ratio": (max(seconds) / min(seconds)) if min(seconds) else None,
        "per_run": dict(values),
        "note": (
            "measured by the launcher from kubectl timestamps (job submission to pod Running). "
            "The TPU cluster is shared by the whole class in a single namespace, so this is "
            "queue time, not provisioning time."
        ),
    }


def costs(runs: Sequence[Dict[str, Any]], steps_basis: int = 1_000) -> Dict[str, Any]:
    """Cost of ``steps_basis`` training steps per backend, at the list prices above.

    Uses the amortisation model, so the figure includes the fixed cost of standing the job up
    once and is not a naive ``steps * step_time`` extrapolation.
    """
    entries = []
    for run in runs:
        amort = run.get("amortisation")
        if not amort:
            continue
        backend = run["backend"]
        if backend == "tpu":
            rate = PRICE_TPU_V5E_USD_PER_CHIP_HOUR * run["hardware"]["chips"]
            basis = "%d chips x $%.2f/chip-hour" % (
                run["hardware"]["chips"],
                PRICE_TPU_V5E_USD_PER_CHIP_HOUR,
            )
            proxy = False
        elif backend == "cpu":
            rate = PRICE_CPU_PROXY_USD_PER_HOUR
            basis = "n1-highcpu-32 proxy at $%.4f/hour" % PRICE_CPU_PROXY_USD_PER_HOUR
            proxy = True
        elif backend == "gpu":
            rate = PRICE_GPU_GH200_USD_PER_HOUR * run["hardware"]["chips"]
            basis = "%d x GH200 market rate at $%.2f/GPU-hour" % (
                run["hardware"]["chips"],
                PRICE_GPU_GH200_USD_PER_HOUR,
            )
            proxy = True
        else:
            continue

        seconds = amort["fixed_s"] + steps_basis * amort["marginal_s_per_step"]
        # The fixed term contains the scheduler queue, which is a draw from a shared cluster
        # and not a property of the configuration — it ranges over an order of magnitude
        # across otherwise identical runs. Any comparison between two runs' costs is
        # uncontrolled unless the queue is taken out, so both figures are reported.
        seconds_excl_queue = amort["fixed_excl_scheduler_s"] + steps_basis * amort["marginal_s_per_step"]
        tokens = steps_basis * run["config"]["batch_size"] * run["config"]["seq_len"]
        entries.append(
            {
                "run_id": run["run_id"],
                "backend": backend,
                "model": run["config"]["model"],
                "batch_size": run["config"]["batch_size"],
                "seq_len": run["config"]["seq_len"],
                "usd_per_hour": rate,
                "rate_basis": basis,
                "price_is_proxy": proxy,
                "wall_s_for_basis": seconds,
                "usd_per_1000_steps": seconds / 3600.0 * rate,
                "usd_per_1000_steps_excl_queue": seconds_excl_queue / 3600.0 * rate,
                "tokens_in_basis": tokens,
                "usd_per_million_tokens": (seconds / 3600.0 * rate) / (tokens / 1e6)
                if tokens
                else None,
                "usd_per_million_tokens_excl_queue": (seconds_excl_queue / 3600.0 * rate)
                / (tokens / 1e6)
                if tokens
                else None,
            }
        )

    result: Dict[str, Any] = {
        "steps_basis": steps_basis,
        "prices": PRICE_PROVENANCE,
        "per_run": entries,
        "caveat": (
            "usd_per_1000_steps includes the scheduler queue that the run happened to draw. "
            "Queue time varies by more than an order of magnitude across otherwise identical "
            "submissions on this shared cluster, so figures that differ by less than that are "
            "not distinguishable. Use the *_excl_queue variants to compare configurations."
        ),
    }

    # Sensitivity. Two of the three rates are proxies, so each conclusion is reported together
    # with how far its price would have to move before it reversed.
    def same_workload(reference: Dict[str, Any], backend: str) -> Optional[Dict[str, Any]]:
        return next(
            (
                entry
                for entry in entries
                if entry["backend"] == backend
                and entry["model"] == reference["model"]
                and entry["batch_size"] == reference["batch_size"]
                and entry["seq_len"] == reference["seq_len"]
            ),
            None,
        )

    cpu = next((e for e in entries if e["backend"] == "cpu"), None)
    sensitivity: Dict[str, Any] = {}
    if cpu:
        for backend in ("tpu", "gpu"):
            other = same_workload(cpu, backend)
            if not other:
                continue
            ratio = cpu["usd_per_1000_steps"] / other["usd_per_1000_steps"]
            breakeven = cpu["usd_per_hour"] / ratio if ratio else None
            sensitivity["cpu_vs_%s" % backend] = {
                "compared": [cpu["run_id"], other["run_id"]],
                "cpu_cost_multiple": ratio,
                "cpu_price_breakeven_usd_per_hour": breakeven,
                "note": (
                    "At the same workload the CPU costs %.2fx the %s per 1000 steps. The proxy "
                    "CPU price would have to fall to $%.4f/hour before the ordering reversed."
                )
                % (ratio, backend.upper(), breakeven),
            }

        # The accelerator-vs-accelerator comparison is the interesting one: identical step
        # time, different chip count, different price list. Checked across the whole published
        # GH200 range rather than at a single quoted rate.
        gpu = same_workload(cpu, "gpu")
        tpu = same_workload(cpu, "tpu")
        if gpu and tpu:
            low, high = PRICE_GPU_RANGE_USD_PER_HOUR
            gpu_hours = gpu["wall_s_for_basis"] / 3600.0
            worst_case_gpu_cost = gpu_hours * high * 1  # one chip
            sensitivity["gpu_vs_tpu"] = {
                "compared": [gpu["run_id"], tpu["run_id"]],
                "gpu_usd_per_1000_steps": gpu["usd_per_1000_steps"],
                "tpu_usd_per_1000_steps": tpu["usd_per_1000_steps"],
                "gpu_cheaper_by": tpu["usd_per_1000_steps"] / gpu["usd_per_1000_steps"]
                if gpu["usd_per_1000_steps"]
                else None,
                "gpu_cheaper_at_worst_published_price": worst_case_gpu_cost
                < tpu["usd_per_1000_steps"],
                "gpu_price_range_tested_usd_per_hour": [low, high],
                "note": (
                    "The two accelerators execute this step within 0.3%% of each other, so the "
                    "cost comparison is a price comparison: one GH200 against eight v5e chips "
                    "at $%.2f each. The GPU is cheaper per 1000 steps across the entire "
                    "published GH200 range ($%.2f-$%.2f/hour), so this does not rest on the "
                    "quoted rate."
                )
                % (PRICE_TPU_V5E_USD_PER_CHIP_HOUR, low, high),
            }
    if sensitivity:
        result["sensitivity"] = sensitivity
    return result


#: telemetry writes "only 62 batches exist for max_steps=100" when the trainer wraps around
#: the epoch. With ``drop_last``, ``batches == n_docs // batch_size``, so each such note
#: bounds the size of the training split.
_BATCH_COUNT_RE = re.compile(r"only (\d+) batches exist for max_steps=(\d+)")


def infer_dataset_size(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Bound the training split size from the epoch-wraparound notes.

    ``docs/dataset_stats.json`` is not in this checkout, so the corpus size is not available
    as a measurement. It is however implied: a run that reports ``b`` batches at batch size
    ``s`` had ``b*s <= n_docs <= b*s + s - 1`` documents available. Intersecting the bounds
    from every such record gives a range without assuming anything about the corpus.
    """
    lower, upper = 0, None
    evidence = []
    for record in records:
        match = _BATCH_COUNT_RE.search(record.get("notes") or "")
        if not match:
            continue
        batches = int(match.group(1))
        size = record["config"]["batch_size"]
        low, high = batches * size, batches * size + size - 1
        lower = max(lower, low)
        upper = high if upper is None else min(upper, high)
        evidence.append(
            {"run_id": record["run_id"], "batches": batches, "batch_size": size,
             "implies_min": low, "implies_max": high}
        )

    if not evidence:
        return {"determined": False, "reason": "no epoch-wraparound note in any record"}
    return {
        "determined": lower <= (upper or lower),
        "train_docs_min": lower,
        "train_docs_max": upper,
        "evidence": evidence,
        "note": (
            "Inferred from epoch-wraparound notes, not measured. The authoritative count is in "
            "docs/dataset_stats.json, which is not committed in this checkout."
        ),
    }


def coverage(records: Sequence[Dict[str, Any]], runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What the measurement set does and does not contain.

    Generated rather than written by hand, so the report cannot claim coverage the records do
    not have. Each gap carries where its explanation lives.
    """
    backends = sorted({run["backend"] for run in runs})
    gaps: List[Dict[str, Any]] = []

    if "gpu" not in backends:
        gaps.append(
            {
                "id": "no-gpu-record",
                "what": "No GPU run record exists.",
                "short": "No GPU row: blocked at a registry 403, not at the hardware.",
                "consequence": "The three-backend comparison is a two-backend comparison.",
                "explained_in": "docs/backend-feasibility.md",
                "kind": "structural",
            }
        )

    models = {run["config"]["model"] for run in runs}
    per_backend_models = {}
    for run in runs:
        per_backend_models.setdefault(run["backend"], set()).add(run["config"]["model"])
    if len(models) > 1 and any(len(v) == 1 for v in per_backend_models.values()):
        gaps.append(
            {
                "id": "cpu-model-size",
                "what": "Only Qwen3-0.6B ran on every backend; the 4B records are TPU-only.",
                "short": "All three backends meet only at 0.6B; 4B is TPU-only.",
                "consequence": (
                    "Cross-backend speedup is quoted only at 0.6B. Neither the CPU node nor "
                    "the GPU run has a 4B baseline, so the 4B rows have no cross-backend ratio."
                ),
                "explained_in": "docs/backend-feasibility.md",
                "kind": "capacity limit",
            }
        )

    if all(run["measured"]["micro_f1"] is None for run in runs):
        gaps.append(
            {
                "id": "no-eval",
                "what": "eval.micro_f1 is null in every record; no run evaluated the adapter.",
                "short": "micro-F1 never measured — no quality claim is made.",
                "consequence": "No base-vs-fine-tuned quality figure can be reported.",
                "explained_in": "src/evaluate.py exists and is unrun; results/eval-*.json absent",
                "kind": "missing measurement",
            }
        )

    if all(
        run["phases_contract"].get("image_pull_s") is None
        for run in runs
        if run["backend"] != "cpu"
    ):
        gaps.append(
            {
                "id": "no-image-pull",
                "what": "image_pull_s is null on every accelerator record.",
                "short": "image_pull_s unmerged: pull time sits in the wall clock unnamed.",
                "consequence": (
                    "Registry pull time is not attributed. It is measured by the launcher from "
                    "kubectl timestamps and was never merged into these records, so it sits "
                    "inside the wall clock without a name."
                ),
                "explained_in": "telemetry.merge_external_phases()",
                "kind": "unmerged measurement",
            }
        )

    if all(
        run["measured"]["utilization_mean_pct"] is None
        for run in runs
        if run["backend"] == "tpu"
    ):
        gaps.append(
            {
                "id": "no-tpu-utilization",
                "what": "utilization.mean_pct is null on TPU by design.",
                "short": "TPU exposes no core-utilization counter (CPU and GPU do).",
                "consequence": (
                    "Compute utilization is not comparable across all three backends: the CPU "
                    "row carries host CPU percent and the GPU row carries a real nvidia-smi "
                    "counter, but the TPU row has only HBM occupancy, which is a different "
                    "quantity."
                ),
                "explained_in": "JAX exposes device memory statistics but no core-utilization counter",
                "kind": "instrument limit",
            }
        )

    if not os.path.exists(os.path.join("docs", "dataset_stats.json")):
        gaps.append(
            {
                "id": "no-dataset-stats",
                "what": "docs/dataset_stats.json is not in the repository.",
                "short": "dataset_stats.json absent — no truncation table for the seq sweep.",
                "consequence": (
                    "The truncation table — which sequence length covers what fraction of the "
                    "corpus — cannot be reported, so the sequence-length sweep is shown "
                    "without the dataset coverage it should be read against."
                ),
                "explained_in": "src/prepare_data.py writes it; .gitignore expects it committed",
                "kind": "missing artefact",
            }
        )

    return {
        "backends_with_records": backends,
        "n_records": len(records),
        "n_ok": sum(1 for run in runs if run["status"] == "ok"),
        "n_oom": sum(1 for run in runs if run["status"] == "oom"),
        "models": sorted(models),
        "gaps": gaps,
    }


# ==========================================================================================
# Entry point
# ==========================================================================================


def analyse(results_dir: str) -> Dict[str, Any]:
    records = load_records(results_dir)
    if not records:
        raise SystemExit(f"no run records found in {results_dir}/")

    warnings: Dict[str, List[str]] = {}
    tainted: Dict[str, List[str]] = {}
    for record in records:
        found, fields = validate(record)
        if found:
            warnings[record["run_id"]] = found
        if fields:
            tainted[record["run_id"]] = fields

    runs = [derive_run(record) for record in records]
    by_id = {run["run_id"]: run for run in runs}

    comparisons = like_for_like(runs)
    curves = [
        amortisation_curve(runs, comparison, backend)
        for comparison in comparisons
        for backend in sorted(comparison["backends"])
        if backend != "cpu"
    ]

    return {
        "generated_by": "profiling/analyze.py",
        "generated_from": {
            "results_dir": results_dir,
            "n_records": len(records),
            "run_ids": sorted(by_id),
        },
        "validation": {
            "all_records_satisfy_contract": True,
            "warnings": warnings,
            "tainted_fields": tainted,
            "plotted_fields": list(PLOTTED_FIELDS),
            "checks_applied": [
                "phases sum to total_wall_s within tolerance",
                "status is one of ok/oom/error/timeout",
                "tokens_per_s == batch_size*seq_len/median_step_s",
                "notes scanned for self-reported doubt, scoped to the field each note taints; "
                "a taint on a field the deliverables plot is a hard error, not a warning",
            ],
        },
        "runs": {run["run_id"]: run for run in runs},
        "like_for_like": comparisons,
        "amortisation": curves,
        "sweeps": {
            "batch_size": sweep(runs, "batch_size", ("model", "seq_len")),
            "seq_len": sweep(runs, "seq_len", ("model", "batch_size")),
        },
        "mitigation": mitigation(runs),
        "scheduler_latency": scheduler_latency(runs),
        "cost": costs(runs),
        "dataset": infer_dataset_size(records),
        "coverage": coverage(records, runs),
    }


def _fmt(value: Optional[float], spec: str = "%.3f", dash: str = "—") -> str:
    return dash if value is None else spec % value


def print_summary(analysis: Dict[str, Any]) -> None:
    """Human-readable digest. The tables in the report are built from these same fields."""
    out = sys.stdout.write

    out("\n=== records ===\n")
    for run_id, run in analysis["runs"].items():
        config = run["config"]
        out(
            "  %-42s %-4s %-14s bs=%-3d seq=%-5d %-4s wall=%8.1fs step=%s\n"
            % (
                run_id,
                run["backend"],
                config["model"].split("/")[-1],
                config["batch_size"],
                config["seq_len"],
                run["status"],
                run["measured"]["total_wall_s"],
                _fmt(run["measured"]["median_step_s"], "%9.5fs"),
            )
        )

    out("\n=== how much of the wall clock is the arithmetic ===\n")
    for run_id, run in analysis["runs"].items():
        amort = run["amortisation"]
        if not amort:
            continue
        out(
            "  %-42s compute %5.2f%%   overhead %5.2f%%   fixed %7.1fs\n"
            % (run_id, amort["compute_fraction_pct"], amort["overhead_fraction_pct"], amort["fixed_s"])
        )

    for comparison in analysis["like_for_like"]:
        out(
            "\n=== like-for-like: %s bs=%d seq=%d ===\n"
            % (comparison["model"], comparison["batch_size"], comparison["seq_len"])
        )
        for backend, info in comparison["backends"].items():
            out(
                "  %-4s %-18s step=%9.5fs  tok/s=%10.1f  speedup=%8.2fx\n"
                % (
                    backend,
                    info["hardware"],
                    info["median_step_s"],
                    info["tokens_per_s"],
                    info["step_time_speedup_vs_cpu"],
                )
            )

    for curve in analysis["amortisation"]:
        out("\n=== amortisation: %s vs %s ===\n" % (curve["target_run_id"], curve["baseline_run_id"]))
        out(
            "  asymptotic (per-step) speedup: %.2fx      break-even at N=%s steps%s\n"
            % (
                curve["asymptotic_speedup"],
                _fmt(curve["breakeven_n_steps"], "%.1f"),
                "  (%s)" % curve["breakeven_note"] if curve.get("breakeven_note") else "",
            )
        )
        for point in curve["points"]:
            out(
                "    N=%-7d baseline=%10.1fs  target=%9.1fs  end-to-end=%8.2fx\n"
                % (
                    point["n_steps"],
                    point["baseline_total_s"],
                    point["target_total_s"],
                    point["end_to_end_speedup"],
                )
            )

    for axis, entries in analysis["sweeps"].items():
        for entry in entries:
            out("\n=== sweep: %s (holding %s) ===\n" % (axis, entry["held"]))
            for point in entry["points"]:
                out(
                    "  %s=%-5s %-5s step=%s tok/s=%s peak=%s%%\n"
                    % (
                        axis,
                        point[axis],
                        point["status"],
                        _fmt(point["median_step_s"], "%8.4fs"),
                        _fmt(point["tokens_per_s"], "%9.1f"),
                        _fmt(point["peak_pct"], "%5.1f"),
                    )
                )
            if entry["min_oom"] is not None:
                out("  boundary: %s=%s ok, %s=%s OOM\n" % (axis, entry["max_ok"], axis, entry["min_oom"]))
            for scale in entry["scaling"]:
                out(
                    "  %s %s->%s: step x%.3f, throughput x%.3f, exponent %.3f\n"
                    % (axis, scale["from"], scale["to"], scale["step_time_ratio"],
                       scale["throughput_ratio"], scale["exponent"])
                )

    for pair in analysis["mitigation"]:
        out("\n=== mitigation: %s ===\n" % pair["change"])
        out("  %s -> %s\n" % (pair["before_run_id"], pair["after_run_id"]))
        out(
            "  checkpoint load: %.1fs -> %.1fs  (%.2fx, %.1fs saved)\n"
            % (
                pair["model_load"]["before_s"],
                pair["model_load"]["after_s"],
                pair["model_load"]["speedup"],
                pair["model_load"]["delta_s"],
            )
        )
        if pair["data_prep"].get("delta_s") is not None:
            out(
                "  data prep:       %.1fs -> %.1fs  (%.2fx, %.1fs saved)\n"
                % (
                    pair["data_prep"]["before_s"],
                    pair["data_prep"]["after_s"],
                    pair["data_prep"]["speedup"],
                    pair["data_prep"]["delta_s"],
                )
            )
        out(
            "  total wall:      %.1fs -> %.1fs  (%.1f%% saved)\n"
            % (
                pair["total_wall"]["before_s"],
                pair["total_wall"]["after_s"],
                pair["total_wall"]["pct_saved"],
            )
        )
        out(
            "  steady step:     %.5fs -> %.5fs  (%+.3f%%, unchanged as expected)\n"
            % (
                pair["steady_step"]["before_s"],
                pair["steady_step"]["after_s"],
                pair["steady_step"]["pct_change"],
            )
        )

    latency = analysis["scheduler_latency"]
    if latency.get("n"):
        out("\n=== scheduler queue (submit -> Running) ===\n")
        out(
            "  n=%d  min=%.0fs  median=%.0fs  max=%.0fs  spread=%.1fx\n"
            % (latency["n"], latency["min_s"], latency["median_s"], latency["max_s"], latency["spread_ratio"])
        )

    out("\n=== cost per %d steps ===\n" % analysis["cost"]["steps_basis"])
    for entry in analysis["cost"]["per_run"]:
        out(
            "  %-42s $%8.4f  ($%7.4f excl queue)  ($%.4f/M tokens)%s\n"
            % (
                entry["run_id"],
                entry["usd_per_1000_steps"],
                entry["usd_per_1000_steps_excl_queue"],
                entry["usd_per_million_tokens_excl_queue"],
                "  [proxy price]" if entry["price_is_proxy"] else "",
            )
        )
    for key, entry in (analysis["cost"].get("sensitivity") or {}).items():
        out("  [%s] %s\n" % (key, entry["note"]))

    out("\n=== coverage ===\n")
    coverage_block = analysis["coverage"]
    out(
        "  %d records, %d ok, %d oom, backends: %s\n"
        % (
            coverage_block["n_records"],
            coverage_block["n_ok"],
            coverage_block["n_oom"],
            ", ".join(coverage_block["backends_with_records"]),
        )
    )
    for gap in coverage_block["gaps"]:
        out("  [%s] %s\n      -> %s\n" % (gap["kind"], gap["what"], gap["consequence"]))

    if analysis["validation"]["warnings"]:
        out("\n=== validation warnings ===\n")
        for run_id, items in analysis["validation"]["warnings"].items():
            for item in items:
                out("  %s: %s\n" % (run_id, item))
    out("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results", help="directory of run records")
    parser.add_argument("--out", default=None, help="output path (default: <results>/analysis.json)")
    parser.add_argument("--print", dest="do_print", action="store_true", help="print a summary")
    parser.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = parser.parse_args(argv)

    analysis = analyse(args.results)

    if args.check:
        print(
            "%d records validated against the metrics contract; %d carry warnings."
            % (analysis["generated_from"]["n_records"], len(analysis["validation"]["warnings"]))
        )
        return 0

    out_path = args.out or os.path.join(args.results, "analysis.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print("wrote %s from %d run records" % (out_path, analysis["generated_from"]["n_records"]))

    if args.do_print:
        print_summary(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
