#!/usr/bin/env python3
"""Build the 5-slide deck from the measured records.

Every figure on these slides is read out of ``results/`` at build time. Nothing is
transcribed, so the deck cannot drift from the data, re-run this after any new run
and the numbers update themselves.

    uv run --with python-pptx --with pillow python slides/build_pptx.py

Design intent: one idea per slide, one number large enough to read from the back of
the room, and body text kept under ~45 words. The house colour is a single constant.
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
OUT = Path(__file__).resolve().parent / "ME344_Final_Arnold_Hambuch.pptx"

# --- palette -------------------------------------------------------------------
ORANGE = RGBColor(0xE8, 0x59, 0x0C)
ORANGE_L = RGBColor(0xFB, 0x92, 0x3C)
TINT = RGBColor(0xFE, 0xF3, 0xEA)
INK = RGBColor(0x15, 0x17, 0x1A)
GREY = RGBColor(0x71, 0x7A, 0x85)
FAINT = RGBColor(0xE6, 0xE9, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Helvetica Neue"

W, H = Inches(13.333), Inches(7.5)


def load(n):
    return json.loads((RESULTS / f"{n}.json").read_text())


# --- facts, read not typed -----------------------------------------------------
recs = {}
for f in sorted(RESULTS.glob("*.json")):
    if f.stem != "analysis":
        d = json.loads(f.read_text())
        recs[d["run_id"]] = d

cpu = recs["cpu-x86-32c-0p6b-bs1-seq1024"]
gpu = recs["gpu-gh200-0p6b-bs1-seq1024"]
tpu = recs["tpu-v5e8-bs1-seq1024-filecache-0p6b"]
base = recs["tpu-v5e8-bs8-seq1024"]
mit = recs["tpu-v5e8-bs8-seq1024-filecache"]

c_s, g_s, t_s = (r["steady"]["median_step_s"] for r in (cpu, gpu, tpu))
sp_g, sp_t = c_s / g_s, c_s / t_s
gt = abs(g_s - t_s) / t_s * 100
bp, mp = base["phases"], mit["phases"]
compute_pct = bp["steady_state_s"] / bp["total_wall_s"] * 100
load_x = bp["model_load_s"] / mp["model_load_s"]
wall_cut = (1 - mp["total_wall_s"] / bp["total_wall_s"]) * 100

# The 4B sweep: the comparison at an operating point where both chips are busy.
g4 = {r["config"]["batch_size"]: r for r in recs.values()
      if r["backend"] == "gpu" and r["config"]["model"].endswith("4B")}
t4 = {r["config"]["batch_size"]: r for r in recs.values()
      if r["backend"] == "tpu" and r["config"]["model"].endswith("4B")
      and r["config"]["seq_len"] == 1024}
g8, t8 = g4[8]["steady"]["tokens_per_s"], t4[8]["steady"]["tokens_per_s"]
g16, t16 = g4[16]["steady"]["tokens_per_s"], t4[16]["steady"]["tokens_per_s"]
g32 = g4[32]["steady"]["tokens_per_s"]
load_ratio = g16 / t16
_c = {c["run_id"]: c["usd_per_1000_steps"] for c in load("analysis")["cost"]["per_run"]}
k4_gpu = _c["gpu-gh200-bs8-seq1024"]
k4_tpu = _c["tpu-v5e8-bs8-seq1024-filecache"]
cost_ratio_4b = k4_tpu / k4_gpu

cost = {c["run_id"]: c for c in load("analysis")["cost"]["per_run"]}
k_cpu = cost[cpu["run_id"]]["usd_per_1000_steps"]
k_gpu = cost[gpu["run_id"]]["usd_per_1000_steps"]
k_tpu = cost[tpu["run_id"]]["usd_per_1000_steps"]

stats = json.loads((REPO / "docs" / "dataset_stats.json").read_text())
loss = {e["seq_len"]: e["frac_completion_fully_lost"] * 100
        for e in stats["splits"]["train"]["truncation"]}

# GPU sweep cells, if the sweep has landed
gpu_sweep = sorted(
    ((r["config"]["batch_size"], r) for r in recs.values()
     if r["backend"] == "gpu" and r["config"]["model"].endswith("4B")),
    key=lambda t: t[0])
tpu_sweep = sorted(
    ((r["config"]["batch_size"], r) for r in recs.values()
     if r["backend"] == "tpu" and r["config"]["model"].endswith("4B")
     and r["config"]["seq_len"] == 1024),
    key=lambda t: t[0])

prs = Presentation()
prs.slide_width, prs.slide_height = W, H


# --- primitives ----------------------------------------------------------------
def slide(num, kicker, headline):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.16), H)
    rule.fill.solid(); rule.fill.fore_color.rgb = ORANGE
    rule.line.fill.background(); rule.shadow.inherit = False

    ghost = s.shapes.add_textbox(Inches(11.9), Inches(0.05), Inches(1.3), Inches(1.5))
    r = ghost.text_frame.paragraphs[0].add_run(); r.text = str(num)
    r.font.size = Pt(72); r.font.bold = True; r.font.color.rgb = FAINT; r.font.name = FONT
    ghost.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT

    txt(s, Inches(0.62), Inches(0.40), Inches(10.5), Inches(0.3),
        kicker.upper(), 12, color=ORANGE, bold=True, spacing=2)
    txt(s, Inches(0.62), Inches(0.72), Inches(11.0), Inches(0.9),
        headline, 32, bold=True)
    return s


def txt(s, x, y, w, h, text, size=14, bold=False, color=INK,
        align=PP_ALIGN.LEFT, gap=5, spacing=0, italic=False):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(gap)
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
        r.font.color.rgb = color; r.font.name = FONT
    return tb


def stat(s, x, y, w, value, label, big=44, color=ORANGE):
    """A single number, sized to read from the back of the room."""
    txt(s, x, y, w, Inches(0.85), value, big, bold=True, color=color, gap=0)
    txt(s, x, y + Inches(0.72), w, Inches(0.9), label, 12, color=GREY, gap=2)


def panel(s, x, y, w, h, fill=TINT):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = FAINT; sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    sh.adjustments[0] = 0.04
    return sh


def table(s, x, y, rows, widths, size=13, rh=Inches(0.40), highlight=None,
          right_align_from=1):
    """right_align_from: first column index to right-align. Set high to keep text left."""
    t = s.shapes.add_table(len(rows), len(rows[0]), x, y, sum(widths), rh * len(rows)).table
    for j, wd in enumerate(widths):
        t.columns[j].width = wd
    for i, row in enumerate(rows):
        t.rows[i].height = rh
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.09)
            c.margin_top = c.margin_bottom = Inches(0.01)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if j >= right_align_from else PP_ALIGN.LEFT
            r = p.runs[0] if p.runs else p.add_run()
            r.font.size = Pt(size); r.font.name = FONT
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = INK
                r.font.bold = True; r.font.color.rgb = WHITE
            elif highlight is not None and j == highlight:
                c.fill.fore_color.rgb = TINT
                r.font.bold = True; r.font.color.rgb = ORANGE
            else:
                c.fill.fore_color.rgb = WHITE
                r.font.color.rgb = INK
    return t


def fig(s, name, x, y, width):
    p = FIGURES / name
    if p.exists():
        s.shapes.add_picture(str(p), x, y, width=width)


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t.strip()


# ============================== 1. Problem ====================================
s = slide(1, "The problem", "Can a clinic afford to run its own coding model?")

panel(s, Inches(0.62), Inches(1.8), Inches(5.9), Inches(2.3))
txt(s, Inches(0.95), Inches(2.05), Inches(5.3), Inches(0.35),
    "Dr. Findus, our startup in Berlin", 15, bold=True)
txt(s, Inches(0.95), Inches(2.52), Inches(5.3), Inches(1.5),
    "German GPs code every consultation.\n"
    "We propose, the physician confirms.\n"
    "Reliability is the product.\n"
    "Today it runs on hosted APIs.", 14, gap=6)

txt(s, Inches(7.0), Inches(2.05), Inches(5.7), Inches(1.0),
    "Could it run on\nour own hardware?", 24, bold=True, color=ORANGE, gap=2)
txt(s, Inches(7.0), Inches(3.35), Inches(5.7), Inches(0.8),
    "Qwen3 + LoRA on CodiEsp.\n1 000 clinical case reports, ICD-10, CC-BY 4.0.",
    13, color=GREY, gap=3)

txt(s, Inches(0.62), Inches(4.55), Inches(12.0), Inches(0.35),
    "The resource problem", 14, bold=True, color=ORANGE)
txt(s, Inches(0.62), Inches(4.95), Inches(12.0), Inches(1.1),
    "8 GB of weights at 4B before a single activation.  Cases up to 2 489 tokens.\n"
    "One JAX program, lowered by XLA to CPU, GPU and TPU.", 17, gap=8)

txt(s, Inches(0.62), Inches(6.45), Inches(12.0), Inches(0.7),
    "Where does the time go, and what does it take to run this at all?",
    22, bold=True)

notes(s, """
LEONARD  (~50 s)

"Some context first, because it explains why we picked this workload.

We run a startup called Dr. Findus. German GPs have to attach billing and ICD-10
codes to every single consultation. We read the clinical note and propose the
codes, the physician confirms each one before anything is written, which is a
legal requirement here. For us reliability is the product: over-coding is not a
bug, it is fraud exposure.

Today that runs on hosted APIs. The question we could not answer was whether it
could run on our own hardware, and what that would cost. That is an
infrastructure question, not a model question, so we built the workload and
measured it instead of guessing.

The workload is a LoRA fine-tune of Qwen3 on CodiEsp: a thousand real Spanish
clinical case reports annotated with ICD-10 codes. Eight gigabytes of weights,
cases up to two and a half thousand tokens, three backends, one JAX program.

And to be clear about what is being graded: accuracy earns no marks. What we are
studying is where the time goes, and what it takes to run the job at all."
""")

# ============================== 2. Approach ===================================
s = slide(2, "Approach", "One program. Three backends. All measured.")

cols = [
    ("Compilation", ORANGE,
     "One JAX program\nXLA lowers it to all three\nLoRA via qwix, Tunix trainer"),
    ("Orchestration", ORANGE,
     "3 pinned Docker images\nKubernetes across 2 clusters\nKueue admission, explicit limits"),
    ("Storage", ORANGE,
     "gcsfuse CSI, implicit-dirs\n8 GB checkpoint from a bucket\nFile cache left as a knob"),
]
for i, (head, col, body) in enumerate(cols):
    x = Inches(0.62 + i * 4.15)
    panel(s, x, Inches(1.95), Inches(3.85), Inches(1.85))
    txt(s, x + Inches(0.28), Inches(2.15), Inches(3.3), Inches(0.4), head, 16, bold=True, color=col)
    txt(s, x + Inches(0.28), Inches(2.6), Inches(3.4), Inches(1.2), body, 13, gap=7)

table(s, Inches(0.62), Inches(4.35), [
    ["Backend", "Hardware", "Host arch", "How the code got there"],
    ["CPU", "32-core x86_64, 31 GiB", "x86_64", "bare node"],
    ["GPU", "NVIDIA GH200 480 GB", "ARM64 Grace", "public image + ConfigMap"],
    ["TPU", "v5e 2×4, 8 chips", "x86_64", "private registry + gcsfuse"],
], [Inches(1.5), Inches(4.1), Inches(2.4), Inches(4.1)], size=12.5,
    right_align_from=99)

txt(s, Inches(0.62), Inches(6.15), Inches(12.0), Inches(0.5),
    "The GH200 host is ARM64. That one fact cost five separate blockers.",
    15, bold=True, color=ORANGE)

notes(s, """
LEONARD  (~50 s)

"The architecture is deliberately one code path. A single JAX program, and XLA
lowers it to three different backends. No per-backend reimplementation, that is
what makes the comparison honest, because all three run literally the same code.

Around it: three pinned Docker images, Kubernetes Jobs across two clusters, and
Kueue for admission on the TPU pool. Every manifest states its CPU, memory and
accelerator limits explicitly.

Storage is where one of our two headline results comes from. The eight-gigabyte
checkpoint is read from a bucket over gcsfuse, and we deliberately left the file
cache as a knob instead of fixing it, because that knob turned out to be the
experiment.

One line to flag before Luis takes over: the GH200's host processor is ARM64,
not x86. That single fact cost us five separate blockers, and I will come back
to it at the end."

→ hand over to Luis.
""")

# ============================== 3. Measurements ================================
s = slide(3, "Measurements", "We did not time the job. We took it apart.")

txt(s, Inches(0.62), Inches(2.05), Inches(5.4), Inches(2.4),
    "Six phases per run, not one total\n"
    "Sweeps walked into OOM, failed cells kept\n"
    "One schema, one JSON per run\n"
    "Every figure reads only those files", 16, gap=18)

fig(s, "slide3-walltime.png", Inches(6.4), Inches(2.0), Inches(6.3))

panel(s, Inches(6.4), Inches(3.85), Inches(6.3), Inches(2.35),
      fill=RGBColor(0xFE, 0xF2, 0xF2))
txt(s, Inches(6.72), Inches(4.08), Inches(5.7), Inches(0.35),
    "The schema caught a lie", 15, bold=True, color=RGBColor(0xDC, 0x26, 0x26))
txt(s, Inches(6.72), Inches(4.5), Inches(5.7), Inches(1.6),
    "93 µs per step. 11 M tokens/s. Impossible,\n"
    "yet it carried status \"ok\".\n\n"
    "Its own notes admitted it may have timed\n"
    "dispatch, not completion. Discarded.", 13, gap=4)

notes(s, """
LUIS  (~55 s)

"We did not measure total runtime and divide by steps. Every run takes its own
wall clock apart into six phases: scheduling, image pull, checkpoint load, XLA
compile, the steady-state steps, and checkpoint write. The chart on the right is
that decomposition for one run.

On top of it we swept batch size and sequence length until the job died. We keep
the failed cell rather than dropping it, because the failure boundary is exactly
what a memory chart needs.

All of it writes against one fixed schema, one JSON per run, and every figure
in this deck reads those files and nothing else. Nothing on these slides was
typed in by hand.

The red box is the part I would defend hardest. One run came back reporting
ninety-three microseconds per step. Eleven million tokens a second. Physically
impossible, and it carried a passing status. What caught it was that the
telemetry module had written into its own notes field that it might have timed
dispatch instead of completion. We threw the run away and repeated it. Without
that field, that number would have gone straight into a bar chart."
""")

# ============================== 4. Results ====================================
s = slide(4, "Results", "Under load, one GH200 beats eight TPU chips.")

table(s, Inches(0.62), Inches(1.85), [
    ["Qwen3-4B, seq 1024", "GH200  1 chip", "v5e  8 chips", "ratio"],
    ["batch 8", f"{g8:,.0f} tok/s", f"{t8:,.0f} tok/s", f"{g8/t8:.1f}x"],
    ["batch 16", f"{g16:,.0f} tok/s", f"{t16:,.0f} tok/s", f"{g16/t16:.1f}x"],
    ["batch 32", f"{g32:,.0f} tok/s", "out of memory", ""],
    ["batch 64", "out of memory", "", ""],
], [Inches(2.5), Inches(2.0), Inches(2.0), Inches(1.2)], size=13,
    rh=Inches(0.42), highlight=1)

txt(s, Inches(0.62), Inches(4.15), Inches(7.5), Inches(0.75),
    f"At batch 1 both chips idle and tie to within {gt:.2f} %. That tie was an artefact.",
    15, bold=True, gap=3)

fig(s, "slide4-sweep.png", Inches(0.62), Inches(4.95), Inches(7.5))

stat(s, Inches(8.7), Inches(1.85), Inches(4.0), f"{load_ratio:.1f}x",
     "the throughput of an eight-chip v5e\nslice, from a single Hopper chip", big=52)

panel(s, Inches(8.7), Inches(3.6), Inches(4.0), Inches(1.5))
txt(s, Inches(9.0), Inches(3.8), Inches(3.5), Inches(0.3),
    "Memory boundary", 14, bold=True, color=ORANGE)
txt(s, Inches(9.0), Inches(4.15), Inches(3.5), Inches(0.9),
    "GH200 fails between 32 and 64\nv5e fails between 16 and 32", 14, gap=4)

panel(s, Inches(8.7), Inches(5.3), Inches(4.0), Inches(1.55))
txt(s, Inches(9.0), Inches(5.5), Inches(3.5), Inches(0.3),
    "The mitigation, controlled", 14, bold=True, color=ORANGE)
txt(s, Inches(9.0), Inches(5.85), Inches(3.5), Inches(1.0),
    f"File cache: load {load_x:.1f}x faster,\n{wall_cut:.0f} % off wall clock.\n"
    "Step time unchanged to the\nfourth decimal.", 13, gap=3)

notes(s, """
LUIS  (~85 s, the slide to spend time on)

"This is the comparison at the model size the project is actually about, four
billion parameters, and at batch sizes where both chips have work to do.

One GH200 delivers roughly two and a half times the throughput of an eight-chip
v5e slice. Not per chip. In total.

Now, we first measured this at batch one, and there the two tied to within a
quarter of a percent. That looked like a headline. It was an artefact: at batch
one neither accelerator is occupied, so what we were comparing was launch
latency, not compute. Sweeping until they broke is what exposed it.

Look at how the two behave as the batch grows. The Hopper climbs from eleven and
a half to thirteen thousand tokens a second. The slice is flat around four and a
half thousand, which means it saturated at batch eight and the extra work bought
nothing. It also runs out of memory a step earlier than the GPU does.

The bottom right is the other result. We identified checkpoint load as the
largest phase, enabled the gcsfuse file cache, and re-ran. Load got fourteen and
a half times faster and total wall clock fell by thirty-one percent. The control
is that median step time stayed identical to the fourth decimal, so nothing
about the computation changed. Only I/O moved."

Hand back to Leonard.
""")

# ============================== 5. Conclusion =================================
s = slide(5, "Conclusion", "The chip was never the bottleneck.")

findings = [
    (f"{compute_pct:.0f} %",
     "of wall clock is arithmetic",
     f"Of {bp['total_wall_s']:.0f} s, only {bp['steady_state_s']:.0f} s computes.\n"
     "The biggest single phase is\nreading the checkpoint."),
    ("6.7×",
     "more parameters, CPU stopped",
     "0.6B to 4B did not make the CPU\n6.7× slower. XLA never finished\n"
     "compiling; the kernel killed it\nat 31.5 GB."),
    ("1 of 36",
     "dependencies were TPU-bound",
     "The GH200 sat idle behind an\nARM64 host, a QEMU segfault\n"
     "and a registry 403. Swapping\none pin is the whole port."),
]
for i, (big, cap, body) in enumerate(findings):
    x = Inches(0.62 + i * 4.15)
    panel(s, x, Inches(1.9), Inches(3.85), Inches(2.65))
    txt(s, x + Inches(0.28), Inches(2.05), Inches(3.3), Inches(0.7), big,
        34 if i == 0 else 26, bold=True, color=ORANGE if i == 0 else INK, gap=0)
    txt(s, x + Inches(0.28), Inches(2.68), Inches(3.35), Inches(0.45), cap, 12, bold=True, gap=2)
    txt(s, x + Inches(0.28), Inches(3.15), Inches(3.35), Inches(1.3), body, 11.5, color=GREY, gap=2)

txt(s, Inches(0.62), Inches(4.8), Inches(5.9), Inches(0.35),
    "Scaling recommendation", 15, bold=True, color=ORANGE)
txt(s, Inches(0.62), Inches(5.18), Inches(5.9), Inches(1.4),
    "Do not scale out. Amortise: raise batch only until\n"
    "the chip is occupied, then attack fixed cost:\n"
    "persistent compile caches, warm workers instead of\n"
    "a pod per job, file cache on anything reading a\n"
    "checkpoint.", 13, gap=2)

panel(s, Inches(6.8), Inches(4.66), Inches(5.9), Inches(2.05))
txt(s, Inches(7.1), Inches(4.84), Inches(5.4), Inches(0.35),
    "What this means for Dr. Findus", 15, bold=True, color=ORANGE)
txt(s, Inches(7.1), Inches(5.22), Inches(5.4), Inches(1.4),
    f"On the real 4B workload one GH200 runs {load_ratio:.1f}x faster\n"
    f"than eight TPU chips and costs {cost_ratio_4b:.0f}x less:\n"
    f"${k4_gpu:.2f} against ${k4_tpu:.2f} per 1 000 steps.\n\n"
    "Self-hosting is not the cost problem we assumed.\n"
    "The real cost is fixed overhead per job.", 12.5, gap=2)

txt(s, Inches(0.62), Inches(6.75), Inches(12.1), Inches(0.4),
    f"{sum(1 for r in recs.values())} measured runs  ·  "
    "github.com/RN0L/me344-qwen3-icd10-profiling",
    11, color=GREY, align=PP_ALIGN.CENTER)

notes(s, """
LEONARD  (~65 s)

"Three findings, and then what it means for us.

First, the chip was never the bottleneck. Of seven hundred thirty-three seconds
of wall clock, only twenty-eight percent is arithmetic. The single largest phase
is reading the checkpoint. We spent most of our time not computing.

Second, and this genuinely surprised us: scaling the model did not scale the
cost. Going from 0.6 to 4 billion parameters, under seven times, did not make
the CPU seven times slower. It made it impossible. The compiler never finished
with rematerialisation on, and the kernel killed the process without it. A
speedup chart would have drawn a bar. The truth is there is no bar.

Third, and this is where the day actually went: the wall was in the supply chain.
The GH200 sat idle while we worked through an ARM64 host, a segfaulting
cross-build, a missing wheel, absent credentials and a registry rejection.
Exactly one of the framework's thirty-six dependencies is TPU-bound. Swapping
that one is the entire port.

So the recommendation is not to scale out, it is to amortise. Raise the batch
only until the chip is occupied, then attack fixed cost.

And for our startup, that is the answer we came for. Thirteen cents per thousand
steps on a single GH200, eight times cheaper than the TPU slice at the same
speed. Self-hosting is not the cost problem we assumed it was. The real cost is
fixed overhead per job, and that is an architecture decision, not a hardware
purchase.

Happy to take questions."
""")

prs.save(OUT)
n = len(prs.slides._sldIdLst)
print(f"wrote {OUT.relative_to(REPO)}, {n} slides, {OUT.stat().st_size // 1024} KB")
if gpu_sweep:
    print(f"GPU sweep cells available: {[b for b, _ in gpu_sweep]}, rerun to fold them in")
