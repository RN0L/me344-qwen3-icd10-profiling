#!/usr/bin/env python3
"""Purpose-built figures for the slide deck, generated from the measured records.

`profiling/make_dashboard.py` produces the report panels. This produces the two
figures the *deck* needs, at slide proportions and with the projector in mind:
larger type, fewer gridlines, one message each.

    uv run --with matplotlib --with numpy python slides/make_slide_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ORANGE, TEAL, GREY, INK = "#E8590C", "#0EA5E9", "#94A3B8", "#15171A"

recs = {}
for f in sorted(RESULTS.glob("*.json")):
    if f.stem != "analysis":
        d = json.loads(f.read_text())
        recs[d["run_id"]] = d

plt.rcParams.update({
    "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#CBD5E1", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#64748B", "ytick.color": "#64748B",
    "axes.grid": True, "grid.color": "#EFF2F5", "grid.linewidth": 0.9,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# ---------------------------------------------------------------- slide 4 ----
TRIO = [
    ("CPU\n32-core", "cpu-x86-32c-0p6b-bs1-seq1024", GREY),
    ("GPU\nGH200 · 1 chip", "gpu-gh200-0p6b-bs1-seq1024", ORANGE),
    ("TPU\nv5e · 8 chips", "tpu-v5e8-bs1-seq1024-filecache-0p6b", TEAL),
]
labels = [t[0] for t in TRIO]
steps = [recs[t[1]]["steady"]["median_step_s"] for t in TRIO]
mem = [recs[t[1]]["memory"]["peak_pct"] for t in TRIO]
colors = [t[2] for t in TRIO]
speed = [steps[0] / s for s in steps]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.1),
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.30})

b = a1.bar(labels, speed, color=colors, width=0.6)
a1.set_yscale("log")
a1.set_ylabel("speedup vs CPU  (log)")
a1.set_title("Speedup vs CPU", loc="left", fontweight="bold", fontsize=15, pad=14)
for rect, v, s in zip(b, speed, steps):
    a1.text(rect.get_x() + rect.get_width()/2, v * 1.18, f"{v:.0f}×",
            ha="center", fontweight="bold", fontsize=15, color=INK)
    inside = max(v * 0.42, 0.62)          # never below the axis floor (ylim starts at 0.5)
    a1.text(rect.get_x() + rect.get_width()/2, inside,
            f"{s*1000:.0f} ms" if s < 1 else f"{s:.1f} s",
            ha="center", fontsize=12, color="white", fontweight="bold")
a1.set_ylim(0.5, max(speed) * 3)

b2 = a2.bar(labels, mem, color=colors, width=0.6)
a2.set_ylabel("peak memory  (% of capacity)")
a2.set_ylim(0, 100)
a2.set_title("Peak memory occupancy", loc="left", fontweight="bold", fontsize=15, pad=14)
for rect, v in zip(b2, mem):
    a2.text(rect.get_x() + rect.get_width()/2, v + 3.5, f"{v:.1f} %",
            ha="center", fontweight="bold", fontsize=14, color=INK)
a2.axhspan(0, 10, color="#FEF3EA", zorder=0)
a2.annotate("both accelerators below 10 %\nnothing to parallelise",
            xy=(1.0, 5.0), xytext=(-0.35, 36), fontsize=10.5, color=ORANGE,
            style="italic", ha="left",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4,
                            connectionstyle="arc3,rad=-0.25"))

fig.suptitle("One Hopper chip keeps pace with eight TPU chips — because neither is busy",
             fontsize=16, fontweight="bold", x=0.008, ha="left", y=1.045)
plt.tight_layout()
fig.savefig(OUT / "slide4-comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- slide 3 ----
base = recs["tpu-v5e8-bs8-seq1024"]
p = base["phases"]
order = ["submit_to_running_s", "model_load_s", "compile_s",
         "steady_state_s", "checkpoint_write_s", "other_s"]
name = {"submit_to_running_s": "scheduling", "model_load_s": "checkpoint load",
        "compile_s": "XLA compile", "steady_state_s": "steady-state steps",
        "checkpoint_write_s": "ckpt write", "other_s": "other"}
col = {"steady_state_s": ORANGE}
total = p["total_wall_s"]

fig, ax = plt.subplots(figsize=(11.0, 2.5))
left = 0.0
for k in order:
    v = p.get(k) or 0.0
    if v <= 0:
        continue
    c = col.get(k, "#CBD5E1" if k != "compile_s" else "#94A3B8")
    ax.barh(0, v, left=left, color=c, edgecolor="white", linewidth=1.6, height=0.55)
    if v / total > 0.07:
        ax.text(left + v/2, 0, f"{name[k]}\n{100*v/total:.0f} %", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if k == "steady_state_s" else INK)
    left += v
ax.set_xlim(0, total); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
ax.set_xlabel("seconds of wall clock")
ax.set_title(f"Only {100*p['steady_state_s']/total:.0f} % of {total:.0f} s is arithmetic",
             loc="left", fontweight="bold", fontsize=15, pad=10)
ax.grid(False)
plt.tight_layout()
fig.savefig(OUT / "slide3-walltime.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"wrote {OUT.relative_to(REPO)}/slide4-comparison.png and slide3-walltime.png")
