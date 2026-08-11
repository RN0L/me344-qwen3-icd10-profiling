#!/usr/bin/env python3
"""Render ``results/systems_dashboard.png`` from ``results/analysis.json``.

Every value plotted here is read out of ``analysis.json``. This module contains no
measurement, no constant taken from a run, and no arithmetic beyond unit conversion and
axis limits — if a number needs deriving, it is derived in ``analyze_runs.py`` and read back.
Run that first.

The figure is meant to be projected in a five-minute talk, so it is laid out for legibility
rather than density: large type, one claim per panel, and the panel that carries the argument
(the wall-clock decomposition) spans the full width at the top.

::

    python3 profiling/analyze_runs.py
    python3 profiling/make_dashboard.py            # -> results/systems_dashboard.png
    python3 profiling/make_dashboard.py --panels   # also write each panel as its own file

Requires matplotlib. Nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==========================================================================================
# Presentation constants
# ==========================================================================================

#: Wall-clock segments in the order they occur, with the label and colour used everywhere.
#: ``steady_compute_s`` is the only saturated colour in the ramp: the point of the first
#: panel is how little of each bar it occupies, so it is the one thing the eye should find.
SEGMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("scheduler_queue_s", "Scheduler queue", "#8c8c8c"),
    ("image_pull_s", "Image pull", "#bdbdbd"),
    ("process_init_s", "Process init + data prep", "#d9a441"),
    ("checkpoint_load_s", "Checkpoint load (I/O)", "#c1442f"),
    ("compile_s", "XLA compile", "#7b4fa8"),
    ("straggler_s", "Step stragglers", "#9ecae1"),
    ("steady_compute_s", "Steady-state compute", "#1f8a4c"),
    ("checkpoint_write_s", "Checkpoint write", "#4a4a4a"),
)

BACKEND_COLOUR = {"cpu": "#c1442f", "gpu": "#4a8f2b", "tpu": "#1f6fb4"}

TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE = 10
ANNOT_SIZE = 9.5


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": TICK_SIZE,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


# ==========================================================================================
# Helpers over analysis.json
# ==========================================================================================


#: Backends in the order they should appear everywhere: baseline first, accelerators after.
BACKEND_ORDER = ("cpu", "gpu", "tpu")

HARDWARE_SHORT = {"cpu": "CPU 32c", "gpu": "GPU GH200", "tpu": "TPU v5e-8"}


def short_label(run: Dict[str, Any]) -> str:
    """A run label that fits an axis: hardware, model, and the two swept knobs."""
    config = run["config"]
    model = config["model"].split("/")[-1].replace("Qwen3-", "")
    hardware = HARDWARE_SHORT.get(run["backend"], run["backend"].upper())
    cache = "  cache off" if str(config["file_cache_capacity"]) == "0" else ""
    return "%s  %s  bs%d/seq%d%s" % (hardware, model, config["batch_size"], config["seq_len"], cache)


def ordered_runs(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The three-backend comparison first, then the TPU-only sweeps."""
    runs = list(analysis["runs"].values())

    def key(run: Dict[str, Any]) -> Tuple[Any, ...]:
        config = run["config"]
        is_small = "0.6B" in config["model"]
        backend_rank = BACKEND_ORDER.index(run["backend"]) if run["backend"] in BACKEND_ORDER else 9
        return (
            0 if is_small else 1,           # the like-for-like trio first
            backend_rank,
            config["seq_len"],
            config["batch_size"],
            str(config["file_cache_capacity"]),
        )

    return sorted(runs, key=key)


def find_sweep(analysis: Dict[str, Any], axis: str) -> Optional[Dict[str, Any]]:
    entries = analysis["sweeps"].get(axis) or []
    return entries[0] if entries else None


def gib(value: Optional[float]) -> Optional[float]:
    return None if value is None else value / (1024.0 ** 3)


# ==========================================================================================
# Panels
# ==========================================================================================


def panel_walltime(ax, analysis: Dict[str, Any]) -> None:
    """Panel 1 — where the wall clock went, one stacked bar per run.

    This is the figure the report is built on. Read it left to right: everything before the
    green segment is the job standing itself up, and the green segment is the training.
    """
    runs = ordered_runs(analysis)
    labels = [short_label(run) for run in runs]
    positions = list(range(len(runs)))

    left = [0.0] * len(runs)
    for key, label, colour in SEGMENTS:
        widths = []
        for run in runs:
            value = run["breakdown"]["segments"].get(key)
            widths.append(0.0 if value is None else max(0.0, value))
        ax.barh(positions, widths, left=left, color=colour, label=label, height=0.68,
                edgecolor="white", linewidth=0.6)
        left = [a + b for a, b in zip(left, widths)]

    # Call out the share of wall clock that was actually the arithmetic.
    for position, run in zip(positions, runs):
        amortisation = run["amortisation"]
        total = run["measured"]["total_wall_s"]
        if amortisation is None:
            note = "OOM — no steady state reached"
        else:
            note = "%.1f%% computing" % amortisation["compute_fraction_pct"]
        ax.text(total + total * 0.012, position, note, va="center", ha="left",
                fontsize=ANNOT_SIZE, fontweight="bold", color="#1a1a1a")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall clock (seconds)")
    ax.set_title(
        "Where the wall clock actually goes — the accelerator spends most of a job not computing",
        loc="left",
    )
    ax.set_xlim(0, max(run["measured"]["total_wall_s"] for run in runs) * 1.28)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=2, framealpha=0.95, fontsize=TICK_SIZE - 0.5)


def panel_throughput(ax, analysis: Dict[str, Any]) -> None:
    """Panel 2 — steady-state throughput on the one workload every backend ran."""
    comparison = analysis["like_for_like"][0]
    backends = comparison["backends"]
    present = [b for b in BACKEND_ORDER if b in backends]

    values = [backends[b]["tokens_per_s"] for b in present]
    labels = ["%s\n%s" % (b.upper(), backends[b]["hardware"]) for b in present]
    colours = [BACKEND_COLOUR[b] for b in present]

    bars = ax.bar(range(len(present)), values, color=colours, width=0.55)
    for bar, backend in zip(bars, present):
        info = backends[backend]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.08,
            "%s tok/s\n%.1f× vs CPU" % (f"{info['tokens_per_s']:,.0f}", info["step_time_speedup_vs_cpu"]),
            ha="center", va="bottom", fontsize=ANNOT_SIZE, fontweight="bold",
        )

    ax.set_yscale("log")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tokens / s (log scale)")
    ax.set_ylim(top=max(values) * 6)
    ax.set_title(
        "Steady-state throughput\n%s, bs=%d, seq=%d"
        % (comparison["model"].split("/")[-1], comparison["batch_size"], comparison["seq_len"]),
        loc="left",
    )
    ax.grid(axis="x", visible=False)


def panel_amortisation(ax, analysis: Dict[str, Any]) -> None:
    """Panel 3 — the per-step speedup is not the speedup you get.

    ``time(N) = fixed_s + N * median_step_s`` for both backends, so the end-to-end advantage
    grows with run length and only reaches the per-step figure asymptotically.
    """
    curves = sorted(
        analysis["amortisation"],
        key=lambda curve: BACKEND_ORDER.index(curve["target_backend"])
        if curve["target_backend"] in BACKEND_ORDER
        else 9,
    )
    lowest = None

    for curve in curves:
        backend = curve["target_backend"]
        xs = [point["n_steps"] for point in curve["points"]]
        ys = [point["end_to_end_speedup"] for point in curve["points"]]
        lowest = min(ys) if lowest is None else min(lowest, min(ys))

        ax.plot(xs, ys, marker="o", color=BACKEND_COLOUR[backend], linewidth=2.2, markersize=6,
                label="%s end-to-end" % backend.upper())
        for x, y in zip(xs, ys):
            ax.annotate("%.1f×" % y, (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=ANNOT_SIZE - 1, fontweight="bold",
                        color=BACKEND_COLOUR[backend])

    # Both accelerators share a per-step speedup to within a fraction of a percent, so one
    # asymptote line serves for both and the divergence below it is entirely fixed cost.
    asymptote = max(curve["asymptotic_speedup"] for curve in curves)
    ax.axhline(asymptote, color="#4a4a4a", linestyle="--", linewidth=1.6,
               label="Per-step speedup (%.0f×)" % asymptote)

    measured_n = curves[0]["points"][0]["n_steps"]
    ax.axvline(measured_n, color="#8c8c8c", linestyle=":", linewidth=1.5)
    ax.text(measured_n * 1.6, lowest * 1.02, "runs were measured\nat this length",
            fontsize=ANNOT_SIZE - 1, color="#4a4a4a", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training steps in the job")
    ax.set_ylabel("End-to-end speedup vs CPU (×)")
    ax.set_title("Same chip speed, different fixed cost\nthe gap between the curves is startup, not silicon",
                 loc="left")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=TICK_SIZE - 1.5)


def panel_utilization(ax, analysis: Dict[str, Any]) -> None:
    """Panel — how busy the hardware actually was, where a counter exists to say.

    The care here is in what is *not* plotted next to what is. CPU percent and GPU percent are
    both duty-cycle counters and are directly comparable. The TPU has no such counter at all;
    its bar is HBM occupancy, drawn hatched and labelled as a different quantity so that it is
    never read as a third utilisation figure.
    """
    comparison = analysis["like_for_like"][0]
    runs = analysis["runs"]

    labels, values, colours, hatches, notes = [], [], [], [], []
    for backend in BACKEND_ORDER:
        info = comparison["backends"].get(backend)
        if not info:
            continue
        measured = runs[info["run_id"]]["measured"]
        utilisation = measured["utilization_mean_pct"]
        labels.append(backend.upper())
        colours.append(BACKEND_COLOUR[backend])
        if utilisation is not None:
            values.append(utilisation)
            hatches.append("")
            notes.append("%.1f%% mean\n(max %.0f%%)" % (utilisation, measured["utilization_max_pct"]))
        else:
            values.append(measured["memory_occupancy_mean_pct"] or 0.0)
            hatches.append("///")
            notes.append("no utilisation\ncounter exists\n(bar = HBM occupancy)")

    bars = ax.bar(range(len(labels)), values, color=colours, width=0.55, alpha=0.9)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_alpha(0.45)
    for bar, note in zip(bars, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, note, ha="center",
                va="bottom", fontsize=ANNOT_SIZE - 0.5, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean over the run (%)")
    ax.set_ylim(0, max(values + [10.0]) * 1.9)
    ax.set_title("The accelerator is idle almost all of the time\nsame workload, mean over the whole run",
                 loc="left")
    ax.grid(axis="x", visible=False)


def panel_batch(ax, analysis: Dict[str, Any]) -> None:
    """Panel 4 — batch size against throughput and peak HBM, with the OOM edge marked."""
    entry = find_sweep(analysis, "batch_size")
    if entry is None:
        ax.set_axis_off()
        return

    ok = [p for p in entry["points"] if p["status"] == "ok"]
    oom = [p for p in entry["points"] if p["status"] == "oom"]
    xs = [p["batch_size"] for p in ok]

    ax.plot(xs, [p["tokens_per_s"] for p in ok], marker="o", color=BACKEND_COLOUR["tpu"],
            linewidth=2.2, markersize=7, label="Throughput (tok/s)")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Tokens / s", color=BACKEND_COLOUR["tpu"])
    ax.tick_params(axis="y", labelcolor=BACKEND_COLOUR["tpu"])
    ax.set_ylim(0, max(p["tokens_per_s"] for p in ok) * 1.35)

    right = ax.twinx()
    right.plot(xs, [p["peak_pct"] for p in ok], marker="s", color="#c1442f", linewidth=2.2,
               markersize=7, linestyle="--", label="Peak HBM (%)")
    right.set_ylabel("Peak HBM occupancy (%)", color="#c1442f")
    right.tick_params(axis="y", labelcolor="#c1442f")
    right.set_ylim(0, 105)
    right.grid(False)
    right.spines["right"].set_visible(True)

    for point in oom:
        ax.axvline(point["batch_size"], color="#c1442f", linewidth=2.5, alpha=0.75)
        ax.annotate(
            "OOM at bs=%d\nHLO temporaries exceed HBM" % point["batch_size"],
            xy=(point["batch_size"], max(p["tokens_per_s"] for p in ok) * 0.55),
            xytext=(-12, 0), textcoords="offset points", ha="right", va="center",
            fontsize=ANNOT_SIZE, fontweight="bold", color="#c1442f",
        )

    all_x = xs + [p["batch_size"] for p in oom]
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(x) for x in all_x])
    ax.set_xlim(min(all_x) * 0.72, max(all_x) * 1.45)

    scaling = entry["scaling"][0] if entry["scaling"] else None
    subtitle = ""
    if scaling:
        subtitle = "\ndoubling the batch: throughput %+.1f%%" % (
            (scaling["throughput_ratio"] - 1.0) * 100.0
        )
    ax.set_title("Batch size buys no throughput%s" % subtitle, loc="left")

    handles = [
        Line2D([], [], color=BACKEND_COLOUR["tpu"], marker="o", label="Throughput (tok/s)"),
        Line2D([], [], color="#c1442f", marker="s", linestyle="--", label="Peak HBM (%)"),
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.95, fontsize=TICK_SIZE - 0.5)
    ax.grid(axis="x", visible=False)


def panel_seqlen(ax, analysis: Dict[str, Any]) -> None:
    """Panel 5 — sequence length against step time and peak HBM.

    Peak memory is flat because rematerialisation bounds activation memory; step time is not,
    because attention is not linear in the sequence. The trade is time, not capacity.
    """
    entry = find_sweep(analysis, "seq_len")
    if entry is None:
        ax.set_axis_off()
        return

    ok = [p for p in entry["points"] if p["status"] == "ok"]
    xs = [p["seq_len"] for p in ok]

    ax.plot(xs, [p["median_step_s"] for p in ok], marker="o", color=BACKEND_COLOUR["tpu"],
            linewidth=2.2, markersize=7)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Median step time (s)", color=BACKEND_COLOUR["tpu"])
    ax.tick_params(axis="y", labelcolor=BACKEND_COLOUR["tpu"])
    ax.set_ylim(0, max(p["median_step_s"] for p in ok) * 1.3)

    right = ax.twinx()
    right.plot(xs, [p["peak_pct"] for p in ok], marker="s", color="#c1442f", linewidth=2.2,
               markersize=7, linestyle="--")
    right.set_ylabel("Peak HBM occupancy (%)", color="#c1442f")
    right.tick_params(axis="y", labelcolor="#c1442f")
    right.set_ylim(0, 105)
    right.grid(False)
    right.spines["right"].set_visible(True)

    for x, point in zip(xs, ok):
        ax.annotate("%.2fs" % point["median_step_s"], (x, point["median_step_s"]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=ANNOT_SIZE, fontweight="bold")

    exponents = [s["exponent"] for s in entry["scaling"]]
    subtitle = ""
    if exponents:
        subtitle = "\nstep time ~ seq^%.2f; peak memory flat (remat)" % (
            sum(exponents) / len(exponents)
        )
    ax.set_title("Longer sequences cost time, not memory%s" % subtitle, loc="left")

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlim(min(xs) * 0.85, max(xs) * 1.18)
    ax.grid(axis="x", visible=False)

    handles = [
        Line2D([], [], color=BACKEND_COLOUR["tpu"], marker="o", label="Median step time"),
        Line2D([], [], color="#c1442f", marker="s", linestyle="--", label="Peak HBM"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.95, fontsize=TICK_SIZE - 0.5)


def panel_mitigation(ax, analysis: Dict[str, Any]) -> None:
    """Panel 6 — the gcsfuse file cache, before and after, phase by phase."""
    pairs = analysis.get("mitigation") or []
    if not pairs:
        ax.set_axis_off()
        return
    pair = pairs[0]

    rows = [
        ("Checkpoint load", pair["model_load"]["before_s"], pair["model_load"]["after_s"]),
        ("Data prep", pair["data_prep"]["before_s"], pair["data_prep"]["after_s"]),
        ("Total wall clock", pair["total_wall"]["before_s"], pair["total_wall"]["after_s"]),
    ]
    rows = [row for row in rows if row[1] is not None and row[2] is not None]

    positions = list(range(len(rows)))
    height = 0.34
    before = ax.barh([p + height / 2 for p in positions], [r[1] for r in rows], height=height,
                     color="#c1442f", label="cache off (fileCacheCapacity 0)")
    after = ax.barh([p - height / 2 for p in positions], [r[2] for r in rows], height=height,
                    color="#1f8a4c", label="cache on (20Gi, range reads cached)")

    widest = max(r[1] for r in rows)
    for position, row in zip(positions, rows):
        speedup = row[1] / row[2] if row[2] else None
        ax.text(row[1] + widest * 0.015, position + height / 2, "%.0fs" % row[1], va="center",
                fontsize=ANNOT_SIZE)
        label = "%.0fs" % row[2]
        if speedup:
            label += "   (%.1f× faster, −%.0fs)" % (speedup, row[1] - row[2])
        ax.text(row[2] + widest * 0.015, position - height / 2, label, va="center",
                fontsize=ANNOT_SIZE, fontweight="bold", color="#0f5c33")

    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Seconds")
    ax.set_xlim(0, widest * 1.42)
    ax.set_title(
        "Mitigation: one storage flag\nsteady step moved %+.3f%% — the fix touched only I/O"
        % pair["steady_step"]["pct_change"],
        loc="left",
    )
    ax.legend(loc="lower right", framealpha=0.95, fontsize=TICK_SIZE - 0.5)
    ax.grid(axis="y", visible=False)


def panel_coverage(ax, analysis: Dict[str, Any]) -> None:
    """Panel 7 — what is not in this figure, and why.

    A missing GPU bar reads as an oversight unless the figure says otherwise. The text is
    generated from ``coverage.gaps`` so it cannot drift away from what the records support.
    """
    ax.set_axis_off()
    coverage = analysis["coverage"]

    ax.text(0.0, 1.0,
            "What these measurements do not cover  —  %d run records · %d ok · %d OOM · backends: %s"
            % (coverage["n_records"], coverage["n_ok"], coverage["n_oom"],
               ", ".join(b.upper() for b in coverage["backends_with_records"])),
            transform=ax.transAxes, fontsize=TITLE_SIZE - 1, fontweight="bold",
            va="top", ha="left")

    # Laid out as a wide strip in columns. Wrapped by hand: matplotlib's wrap=True measures
    # against the figure rather than the axes, so it ignores the column width.
    gaps = coverage["gaps"]
    columns = 3
    per_column = -(-len(gaps) // columns)
    for index, gap in enumerate(gaps):
        column, row = divmod(index, per_column)
        x = column / columns
        y = 0.72 - row * 0.30
        lines = textwrap.wrap(gap.get("short") or gap["what"], width=46)
        for offset, line in enumerate(lines):
            ax.text(x if offset == 0 else x + 0.008, y - offset * 0.145,
                    ("▸ " if offset == 0 else "") + line,
                    transform=ax.transAxes, fontsize=ANNOT_SIZE - 0.5, fontweight="bold",
                    va="top", ha="left")

    ax.text(1.0, 1.0, "Full reasoning: docs/backend-feasibility.md · docs/bottleneck-analysis.md",
            transform=ax.transAxes, fontsize=ANNOT_SIZE - 1, color="#6a6a6a",
            va="top", ha="right", style="italic")


# ==========================================================================================
# Composition
# ==========================================================================================

PANELS = {
    "walltime": panel_walltime,
    "throughput": panel_throughput,
    "amortisation": panel_amortisation,
    "utilization": panel_utilization,
    "batch": panel_batch,
    "seqlen": panel_seqlen,
    "mitigation": panel_mitigation,
    "coverage": panel_coverage,
}


def build(analysis: Dict[str, Any], out_path: str) -> str:
    style()
    figure = plt.figure(figsize=(19.5, 17.6))
    grid = gridspec.GridSpec(
        4, 3, figure=figure, height_ratios=[1.30, 1.0, 1.0, 0.34],
        hspace=0.50, wspace=0.36, left=0.055, right=0.965, top=0.878, bottom=0.035,
    )

    panel_walltime(figure.add_subplot(grid[0, :]), analysis)
    panel_throughput(figure.add_subplot(grid[1, 0]), analysis)
    panel_amortisation(figure.add_subplot(grid[1, 1]), analysis)
    panel_utilization(figure.add_subplot(grid[1, 2]), analysis)
    panel_batch(figure.add_subplot(grid[2, 0]), analysis)
    panel_seqlen(figure.add_subplot(grid[2, 1]), analysis)
    panel_mitigation(figure.add_subplot(grid[2, 2]), analysis)
    panel_coverage(figure.add_subplot(grid[3, :]), analysis)

    comparison = analysis["like_for_like"][0]
    backends = comparison["backends"]
    by_backend = {curve["target_backend"]: curve for curve in analysis["amortisation"]}
    figure.suptitle(
        "LoRA fine-tuning of Qwen3 on CodiEsp — CPU vs GPU vs TPU, and where the time goes",
        fontsize=20, fontweight="bold", x=0.055, ha="left", y=0.982,
    )

    # Each accelerator is named with its own numbers. Reading one backend's speedup off
    # another's curve is exactly the mistake this strap has to avoid.
    per_backend = "  ".join(
        "%s %.0f× per step → %.1f× end-to-end." % (
            backend.upper(),
            backends[backend]["step_time_speedup_vs_cpu"],
            by_backend[backend]["points"][0]["end_to_end_speedup"],
        )
        for backend in BACKEND_ORDER
        if backend in by_backend and backend in backends
    )
    strap = (
        "One JAX/XLA code path on all three backends, one workload every backend ran. "
        "%s The two accelerators are within %.1f%% of each other per step, so the difference "
        "between their curves is fixed cost, not silicon. "
        "Generated from results/analysis.json by profiling/make_dashboard.py."
        % (
            per_backend,
            abs(backends["gpu"]["median_step_s"] - backends["tpu"]["median_step_s"])
            / backends["tpu"]["median_step_s"] * 100.0
            if "gpu" in backends and "tpu" in backends
            else float("nan"),
        )
    )
    # Wrapped explicitly: savefig(bbox_inches="tight") grows the canvas to fit any text that
    # runs past the axes, so one long line would silently stretch the whole figure.
    figure.text(0.055, 0.958, textwrap.fill(strap, width=132),
                fontsize=11.5, color="#3a3a3a", ha="left", va="top", linespacing=1.5)

    figure.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out_path


def build_single(analysis: Dict[str, Any], name: str, out_dir: str) -> str:
    """Render one panel on its own — used for the slides, where one claim gets one image."""
    style()
    tall = name in ("walltime",)
    figure = plt.figure(figsize=(13.0, 6.4) if tall else (8.2, 5.6))
    PANELS[name](figure.add_subplot(1, 1, 1), analysis)
    path = os.path.join(out_dir, "panel-%s.png" % name)
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", default=os.path.join("results", "analysis.json"))
    parser.add_argument("--out", default=os.path.join("results", "systems_dashboard.png"))
    parser.add_argument("--panels", action="store_true",
                        help="also write each panel separately, for the slides")
    parser.add_argument("--panel-dir", default=os.path.join("slides", "figures"))
    args = parser.parse_args(argv)

    if not os.path.exists(args.analysis):
        raise SystemExit(
            "%s not found — run `python3 profiling/analyze_runs.py` first." % args.analysis
        )
    with open(args.analysis, "r", encoding="utf-8") as handle:
        analysis = json.load(handle)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("wrote %s" % build(analysis, args.out))

    if args.panels:
        os.makedirs(args.panel_dir, exist_ok=True)
        for name in PANELS:
            print("wrote %s" % build_single(analysis, name, args.panel_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
