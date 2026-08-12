#!/usr/bin/env python3
"""Purpose-built figures for the slide deck, generated from the measured records.

`profiling/make_dashboard.py` produces the report panels. This produces the three
figures the *deck* needs, at slide proportions and with the projector in mind.

The rule that governs the type sizes below: a figure is authored at `figsize`
inches and placed at `SLOT` inches on the slide, so every point size in it is
scaled by `SLOT / figsize`. Authoring an 11 in figure into a 6.3 in slot turns
11 pt labels into 6 pt, which is unreadable from the back of a room. Each figure
here is authored close to the width it is placed at, so the scale factor stays
above 0.85 and the smallest label lands at 10 pt or more on the slide.

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

stats = json.loads((REPO / "docs" / "dataset_stats.json").read_text())

plt.rcParams.update({
    "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#CBD5E1", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#64748B", "ytick.color": "#64748B",
    "axes.grid": True, "grid.color": "#EFF2F5", "grid.linewidth": 0.9,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


# ------------------------------------------------- slide 1: what the data costs
# Placed at 6.20 in, authored at 6.60 in -> scale 0.94.
trunc = {e["seq_len"]: e for e in stats["splits"]["train"]["truncation"]}
windows = [512, 1024, 2048]
kept = [100 * (1 - trunc[w]["frac_truncated"]) for w in windows]

fig, ax = plt.subplots(figsize=(6.6, 2.05))
y = np.arange(len(windows))
cols = [ORANGE if w == 1024 else "#CBD5E1" for w in windows]
bars = ax.barh(y, kept, color=cols, height=0.62)
for rect, v, w in zip(bars, kept, windows):
    ax.text(v + 2.2, rect.get_y() + rect.get_height() / 2, f"{v:.0f} %",
            va="center", fontsize=13, fontweight="bold",
            color=ORANGE if w == 1024 else INK)
ax.set_yticks(y)
ax.set_yticklabels([f"{w} tok" for w in windows], fontsize=13)
ax.invert_yaxis()
ax.set_xlim(0, 118)
ax.set_xticks([0, 50, 100])
ax.set_xticklabels(["0", "50", "100 %"], fontsize=11)
ax.set_title("Share of cases that fit the context window, uncut",
             loc="left", fontweight="bold", fontsize=13.5, pad=9)
ax.text(3.0, 1.0, "we benchmark here", fontsize=12, color="white",
        style="italic", va="center", fontweight="bold")
ax.grid(axis="y", visible=False)
plt.tight_layout()
fig.savefig(OUT / "slide1-coverage.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ----------------------------------------- slide 3: where the wall clock went
# Placed at 6.30 in, authored at 7.00 in -> scale 0.90.
base = recs["tpu-v5e8-bs8-seq1024"]
p = base["phases"]
order = ["submit_to_running_s", "model_load_s", "compile_s",
         "steady_state_s", "checkpoint_write_s", "other_s"]
name = {"submit_to_running_s": "scheduling", "model_load_s": "checkpoint load",
        "compile_s": "XLA compile", "steady_state_s": "steady-state steps",
        "checkpoint_write_s": "ckpt write", "other_s": "other"}
total = p["total_wall_s"]

fig, ax = plt.subplots(figsize=(7.0, 2.35))
left = 0.0
for k in order:
    v = p.get(k) or 0.0
    if v <= 0:
        continue
    on = k == "steady_state_s"
    c = ORANGE if on else ("#94A3B8" if k == "compile_s" else "#CBD5E1")
    ax.barh(0, v, left=left, color=c, edgecolor="white", linewidth=1.8, height=0.5)
    # Percentage inside the segment, name outside it. Wide segments take the name
    # below; the one narrow segment takes it above, so no two labels can collide
    # and nothing has to be squeezed into 70 px.
    share = 100 * v / total
    if share >= 3:
        ax.text(left + v / 2, 0, f"{share:.0f} %", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white" if on else INK)
        above = share < 12
        ax.text(left + v / 2, 0.40 if above else -0.42, name[k], ha="center",
                va="bottom" if above else "top",
                fontsize=11.5, color=ORANGE if on else "#475569",
                fontweight="bold" if on else "normal")
    left += v
ax.set_xlim(0, total)
ax.set_ylim(-1.0, 0.95)
ax.set_yticks([])
ax.set_xlabel("seconds of wall clock", fontsize=11.5)
ax.set_title(f"Only {100 * p['steady_state_s'] / total:.0f} % of {total:.0f} s is arithmetic",
             loc="left", fontweight="bold", fontsize=14.5, pad=10)
ax.grid(False)
plt.tight_layout()
fig.savefig(OUT / "slide3-walltime.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ------------------------------------- slide 4: the tie breaks under real load
# Placed at 7.50 in, authored at 8.20 in -> scale 0.91.
g4 = {r["config"]["batch_size"]: r for r in recs.values()
      if r["backend"] == "gpu" and r["config"]["model"].endswith("4B")}
t4 = {r["config"]["batch_size"]: r for r in recs.values()
      if r["backend"] == "tpu" and r["config"]["model"].endswith("4B")
      and r["config"]["seq_len"] == 1024}

fig, ax = plt.subplots(figsize=(8.2, 2.45))
bs = [8, 16, 32]
gv = [g4[b]["steady"]["tokens_per_s"] if g4.get(b) and g4[b]["status"] == "ok" else None
      for b in bs]
tv = [t4[b]["steady"]["tokens_per_s"] if t4.get(b) and t4[b]["status"] == "ok" else None
      for b in bs]
x = np.arange(len(bs))
ax.plot(x, gv, "o-", color=ORANGE, lw=3.2, ms=11, label="GH200, 1 chip")
ax.plot([i for i, v in zip(x, tv) if v], [v for v in tv if v], "s-", color=TEAL,
        lw=3.2, ms=11, label="v5e, 8 chips")
for i, v in enumerate(gv):
    if v:
        ax.annotate(f"{v:,.0f}", (i, v), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=12.5, fontweight="bold", color=ORANGE)
for i, v in enumerate(tv):
    if v:
        ax.annotate(f"{v:,.0f}", (i, v), textcoords="offset points", xytext=(0, -22),
                    ha="center", fontsize=12.5, fontweight="bold", color=TEAL)
ax.annotate("v5e out of memory", (2, 4620), xytext=(1.38, 7900), fontsize=12,
            color=TEAL, style="italic",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
ax.set_xticks(x)
ax.set_xticklabels([f"batch {b}" for b in bs], fontsize=12.5)
ax.set_xlim(-0.28, 2.28)
ax.set_ylabel("tokens / s", fontsize=12)
ax.set_ylim(2500, 15800)
ax.legend(frameon=False, loc="center left", fontsize=12)
ax.set_title("Qwen3-4B: the Hopper keeps scaling, the slice saturates at batch 8",
             loc="left", fontweight="bold", fontsize=13.5, pad=10)
plt.tight_layout()
fig.savefig(OUT / "slide4-sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig)

for f in ("slide1-coverage", "slide3-walltime", "slide4-sweep"):
    print(f"wrote {(OUT / (f + '.png')).relative_to(REPO)}")
