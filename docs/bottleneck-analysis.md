# Bottleneck analysis — the derivation

The README states the diagnosis in a paragraph. This file shows the work: how the wall clock
was split, what the split implies, and — the part that makes it a diagnosis rather than an
observation — which competing explanations the same records rule out.

Every figure below is a field of `results/analysis.json`, produced by
`profiling/analyze_runs.py` from the run records. None of it is retyped, and none of it is
rounded on the way in.

---

## The diagnosis in one sentence

**This workload is fixed-cost bound, not compute bound: every job pays a near-constant
125–460 s to stand itself up, that cost does not shrink when the accelerator gets faster, and
its largest controllable component is XLA compilation — which on the reference run costs more
wall-clock time (138.5 s) than the training itself (108.6 s).**

The sharpest single piece of evidence is not a phase breakdown but a pair of accelerators. A
GH200 and an eight-chip v5e slice execute this training step within **0.23 %** of each other
and finish the same job **2.44× apart** end to end, in exactly the ratio of their fixed costs
(125.7 s against 307.1 s). And on the one backend that exposes a duty-cycle counter, mean
utilisation over the whole run was **1.59 %**.

---

## 1. How the wall clock was split

The metrics contract already attributes wall clock to seven phases. That is enough to see
where a job spends its time, and not enough to answer the question this project asks — *what
fraction of the run was the arithmetic?* — because `steady_state_s` is defined as
`sum(steps[1:])`, and on every accelerator record here `steps[1]` is a second XLA compile, not
a training step.

`analyze_runs.py` splits it further using an identity that is exact for any `m`:

```
sum(S) = n·m  +  (S[0] − m)  +  (S[1] − m)  +  Σ_{i≥2}(S[i] − m)
         │        │              │              │
         │        │              │              └── stragglers: epoch and checkpoint
         │        │              │                  boundaries, signed so faster-than-median
         │        │              │                  steps subtract
         │        │              └── the second compile
         │        └── the first compile
         └── steady-state compute: every step charged at the measured median
```

with `S` the recorded step times and `m = steady.median_step_s`. Nothing is fitted and nothing
is estimated. The residual after reassembling all eight segments is ≤ 1.1 × 10⁻¹³ s on every
record, i.e. floating-point noise — `breakdown.sums_to_total` is `true` for all eight.

**Calling `steps[1]` a compile is a judgement, and it is recorded as one.** The flag is set
when the second step costs ≥ 10× the median. The observed ratios make it uncontroversial on
both accelerators — and the fact that a CUDA backend and a TPU backend both compile twice is
itself evidence that the second compilation belongs to the program, not to one platform:

| Run | `steps[1]` / median |
|---|---|
| `tpu-v5e8-bs1-seq1024-filecache-0p6b` | 495.7× |
| `gpu-gh200-0p6b-bs1-seq1024` | 451.2× |
| `tpu-v5e8-bs8-seq512-filecache` | 93.1× |
| `tpu-v5e8-bs8-seq1024` | 39.4× |
| `tpu-v5e8-bs8-seq1024-filecache` | 39.3× |
| `tpu-v5e8-bs16-seq1024-filecache` | 16.0× |
| `tpu-v5e8-bs8-seq2048-filecache` | 13.1× |
| `cpu-x86-32c-0p6b-bs1-seq1024` | **3.8×** |

The CPU row falls below the threshold and its second step is therefore charged to stragglers,
not to compilation. That is the conservative choice — it makes the CPU's compile segment
smaller and the comparison *less* flattering to the accelerator — and it is stated here
because a reader who disagrees can move 52.5 s between two segments of one bar and change no
conclusion in this document.

---

## 2. The evidence

### 2.1 The share of wall clock that is the arithmetic

| Run | Steady-state compute | of wall clock |
|---|---|---|
| `cpu-x86-32c-0p6b-bs1-seq1024` | 227.9 s | **48.9 %** |
| `tpu-v5e8-bs8-seq2048-filecache` | 171.0 s | 41.3 % |
| `tpu-v5e8-bs16-seq1024-filecache` | 110.5 s | 31.1 % |
| `tpu-v5e8-bs8-seq1024-filecache` | 108.6 s | 21.6 % |
| `tpu-v5e8-bs8-seq1024` (cache off) | 108.6 s | 14.8 % |
| `tpu-v5e8-bs8-seq512-filecache` | 28.1 s | 5.8 % |
| `tpu-v5e8-bs1-seq1024-filecache-0p6b` | 8.0 s | **2.5 %** |
| `gpu-gh200-0p6b-bs1-seq1024` | 3.2 s | **2.5 %** |

Read the first and last two rows together. They are the *same workload* — Qwen3-0.6B, batch 1,
sequence 1024, same script, same flags, one code path lowered by XLA to three backends. The CPU
spends about half its wall clock computing. Both accelerators spend one fortieth, and they
arrive at that number independently, on different silicon, in different data centres, through
different storage systems. The accelerators did not fail to make the arithmetic fast; they made
the arithmetic so cheap that everything else became the job.

### 2.2 The per-step speedup is not the speedup anyone experiences

On that like-for-like trio the GPU executes a step **238.01×** and the TPU **237.45×** faster
than the CPU (18.9899 s → 0.07979 s and 0.07997 s; the throughput ratios are identical to seven
digits, because both derive from the same step times).

Modelling a run as `time(N) = fixed_s + N · median_step_s`, with `fixed_s` taken as the
measured wall clock minus every recorded step charged at the steady rate:

| | fixed | marginal |
|---|---|---|
| CPU | 238.1 s | 18.9899 s/step |
| GPU | **125.7 s** | 0.07979 s/step |
| TPU | **307.1 s** | 0.07997 s/step |

| Run length | CPU | GPU | TPU | GPU speedup | TPU speedup |
|---|---|---|---|---|---|
| 12 steps (the CPU run's length) | 466.0 s | 126.7 s | 308.0 s | **3.68×** | **1.51×** |
| 40 steps (the GPU run's length) | 997.7 s | 128.9 s | 310.3 s | 7.74× | 3.21× |
| 100 steps (the TPU run's length) | 2 137.1 s | 133.7 s | 315.1 s | 15.99× | 6.78× |
| 1 000 steps | 19 228.0 s | 205.5 s | 387.1 s | 93.57× | 49.68× |
| 10 000 steps | 190 136.8 s | 923.6 s | 1 106.8 s | 205.87× | 171.79× |
| 100 000 steps | 1 899 225.1 s | 8 104.3 s | 8 304.5 s | 234.35× | 228.70× |

The TPU breaks even against the CPU at **N = 3.6 steps**. The GPU's fixed cost is already below
the CPU's, so its break-even is negative — it wins at every run length, and the negative number
is not a threshold. In both cases the *advertised* 238× arrives only asymptotically. At the
length these benchmark jobs actually ran, a reader who timed the wall clocks with a stopwatch
would have measured 1.5× for the TPU, and would have been right.

**The gap between 238× and 1.5× is the bottleneck, quantified.** Every engineering decision
below is about closing it.

### 2.2a The controlled comparison — the same evidence without the CPU

The CPU comparison invites an objection: the CPU is a different class of machine, so of course
the ratios move around. The two accelerators answer it, because they hold the thing under
suspicion constant.

| | GPU GH200 (1 chip) | TPU v5e 2x4 (8 chips) | Difference |
|---|---|---|---|
| Median step time | 0.07979 s | 0.07997 s | **0.23 %** |
| Fixed cost | 125.7 s | 307.1 s | **2.44×** |
| End-to-end at 12 steps | 3.68× | 1.51× | **2.44×** |

Their marginal costs are the same measurement twice. Their end-to-end results differ by exactly
the ratio of their fixed costs. **If any property of the silicon were the limit, these two rows
would not separate.**

And the 181.4 s of fixed-cost difference is itself attributable, which matters because it
prevents the comparison being read as "Hopper beats v5e":

| Component | GPU | TPU | Difference |
|---|---|---|---|
| Scheduler queue | 11.00 s | 143.00 s | **132.00 s** |
| Process init + data prep | 32.82 s | 71.83 s | **39.00 s** |
| XLA compile | 76.00 s | 81.64 s | 5.64 s |
| Checkpoint load | 1.75 s | 4.39 s | 2.65 s |
| Step stragglers | 3.69 s | 5.19 s | 1.50 s |
| Checkpoint write | 0.44 s | 1.04 s | 0.60 s |
| **Total** | | | **181.39 s** |

The six rows sum to the fixed-cost difference exactly, because they are every segment of the
wall clock except the steady-state compute the two runs share.

**72.8 % of the difference is queue time** on a cluster the GPU does not share with a
whole class — and the TPU's queue ranges 4–207 s across runs, so this particular draw was
unlucky rather than typical. Most of the remainder is storage: the GPU job's corpus arrives with
the pod, the TPU's arrives through a FUSE mount over Cloud Storage (30.57 s of data preparation
against 2.09 s). Compilation, the one component that is genuinely about the compiler and the
chip, differs by 5.6 s out of 181.4.

So the correct reading is not that one accelerator is faster. It is that **an identical
workload, on chips of identical speed, took 2.4× longer end to end because of scheduling and
storage.** That is the thesis of this document, established without reference to the CPU at all.

### 2.3 What the fixed cost is made of

For the reference run `tpu-v5e8-bs8-seq1024-filecache` (Qwen3-4B, bs 8, seq 1024, cache on),
502.8 s of wall clock:

| Segment | Seconds | % of wall |
|---|---|---|
| Scheduler queue | 116.0 | 23.1 % |
| Image pull | *unattributed* | — |
| Process init + data prep | 86.1 | 17.1 % |
| Checkpoint load (I/O) | 16.1 | 3.2 % |
| **XLA compile (two compilations)** | **138.5** | **27.6 %** |
| Step stragglers | 35.3 | 7.0 % |
| **Steady-state compute** | **108.6** | **21.6 %** |
| Checkpoint write | 2.1 | 0.4 % |

Compilation is the single largest segment, larger than the training it exists to enable, and
it is paid on every job:

| Run | Compile (both) | Compute | Ratio |
|---|---|---|---|
| `gpu-gh200-0p6b-bs1-seq1024` | 76.0 s | 3.2 s | **23.83×** |
| `tpu-v5e8-bs1-seq1024-filecache-0p6b` | 81.6 s | 8.0 s | 10.21× |
| `tpu-v5e8-bs8-seq512-filecache` | 133.7 s | 28.1 s | 4.76× |
| `tpu-v5e8-bs8-seq1024-filecache` | 138.5 s | 108.6 s | 1.28× |
| `tpu-v5e8-bs8-seq1024` | 136.2 s | 108.6 s | 1.25× |
| `tpu-v5e8-bs16-seq1024-filecache` | 112.8 s | 110.5 s | 1.02× |
| `tpu-v5e8-bs8-seq2048-filecache` | 109.1 s | 171.0 s | 0.64× |

Compile time sits in a band of 76.0–138.5 s across **three architectures** and configurations
that differ by 6.7× in model size and 4× in sequence length. It is close to a constant, and a
constant paid per job is precisely the kind of cost that a faster chip cannot amortise and a
longer run can. That a Hopper GPU and an eight-chip TPU slice compile the same graph in 76.0 s
and 81.6 s — a 7 % spread across entirely different compiler backends — is the clearest sign
that this cost belongs to the program rather than to the hardware.

That the two `bs8/seq1024` runs — separated by a storage flag that changes nothing about the
graph — compiled in 73.29 s and 70.72 s, a spread of 3.6 %, is the direct evidence that **the
same program is recompiled from scratch on every job.**

### 2.4 The fixed cost is not even stable

`submit_to_running_s` across the seven accelerator records:

| Run | Queue |
|---|---|
| `tpu-v5e8-bs8-seq512-filecache` | 207 s |
| `tpu-v5e8-bs1-seq1024-filecache-0p6b` | 143 s |
| `tpu-v5e8-bs32-seq1024-filecache` | 128 s |
| `tpu-v5e8-bs8-seq1024-filecache` | 116 s |
| `tpu-v5e8-bs8-seq1024` | 114 s |
| `tpu-v5e8-bs16-seq1024-filecache` | 5 s |
| `tpu-v5e8-bs8-seq2048-filecache` | 4 s |

Median 116 s, range 4–207 s, **spread 51.8×**. The TPU cluster runs the whole class in one
`default` namespace behind a single Kueue LocalQueue, so this is contention, not provisioning.

Two consequences are carried through the rest of the analysis. First, any comparison between
two runs whose costs differ by less than this is not a measurement of anything. Second,
`analysis.json` reports `usd_per_1000_steps_excl_queue` alongside the inclusive figure, and
the report uses the exclusive one whenever it compares configurations to each other.

---

## 3. What the data rules out

A diagnosis is only worth as much as the alternatives it excludes. Four were live.

### "It is compute bound."

Ruled out directly. Steady-state compute is 2.5 %–48.9 % of wall clock and is **21.6 %** on the
reference run and 2.5 % on both accelerators at the like-for-like configuration. All 8
completed runs spend more wall clock not computing than computing. A workload cannot be limited by the resource it uses for a fifth of its life.

### "It is I/O bound."

It *was*, and it is not any more — which is a stronger statement than either alternative,
because the transition was measured rather than argued.

With the gcsfuse file cache off, checkpoint load was 235.1 s, **32.1 % of a 732.9 s run**, and
the single largest segment. That is a textbook I/O bottleneck. Turning the cache on cut it to
16.1 s (3.2 %). If I/O had been the binding constraint, removing it should have made the job
fast. It did not: total wall clock fell 31.4 %, from 732.9 s to 502.8 s, and the job remained
78.4 % non-compute. **The bottleneck moved rather than disappeared** — from storage to
compilation — which is what identifies compilation as the constraint rather than storage.

### "It is memory bound."

Memory is a *capacity* constraint here, not a throughput constraint. The distinction is
measurable and the records make it.

HBM does bound the configuration: batch 32 fails with `RESOURCE_EXHAUSTED: the total memory
required for HLO temporaries (21.67G) exceeds available HBM (15.75G)`, so batch 16 is the
largest that runs. But raising the batch would not have bought throughput even if the memory
were there:

| Batch | Median step | Tokens/s | Peak HBM |
|---|---|---|---|
| 8 | 1.7512 s | 4 677.9 | 69.4 % |
| 16 | 3.5650 s | 4 595.8 | 65.5 % |
| 32 | — | — | **OOM** |

Doubling the batch multiplies step time by **2.036** and changes throughput by **−1.8 %**. Per
token, the step costs what it costs; the batch dimension buys nothing back. So the memory
ceiling forbids a configuration that was not worth having, and cannot be the reason the job is
slow.

Two further observations belong here because they are counterintuitive and are in the data.
Peak HBM at batch 16 (65.5 %) is *lower* than at batch 8 (69.4 %) even though the OOM at batch
32 is a memory failure: peak occupancy is set by the XLA scheduler's buffer assignment, not by
a simple function of batch size. And across sequence lengths 512 → 1024 → 2048 peak HBM is
flat (73.8 %, 69.4 %, 70.1 %) while step time rises as `seq^1.30`. That flatness is
`remat=DECODER` doing its job: rematerialisation bounds activation memory, so longer sequences
are paid for in time and not in capacity.

Where memory *is* decisive is feasibility, and there it is decisive completely: the 4B
fine-tune does not run on the 31 GiB CPU node in any configuration tried
([`backend-feasibility.md`](backend-feasibility.md)). That is a capacity result, and it is why
the CPU column is a 0.6B column.

### "The accelerator is underutilised at the chip level."

This is the one alternative that turned out to be **true**, and it is worth separating what is
now measured from what still is not.

**On the GPU it is measured, and it is decisive.** `nvidia-smi` exposes a duty-cycle counter
that JAX does not, and over 111 samples at 1 Hz across the whole run:

| | |
|---|---|
| Mean GPU utilisation | **1.59 %** |
| Maximum GPU utilisation | 48.0 % |
| Peak memory | 5.15 GB of 102.6 GB (**5.02 %**) |

The accelerator is not working hard and slowly. It is **idle for 98.4 % of the run**, on the
same workload that gives a 238× per-step speedup. That is the wall-clock decomposition
confirmed by a completely independent instrument: the phase timers say 2.5 % of the run was
compute, and the GPU's own counter says it was busy 1.6 % of the time. Two different
measurement paths, the same conclusion.

**On the TPU it remains not claimable, and is not claimed.** `utilization.mean_pct` is null on
every TPU record by design: JAX exposes device memory statistics but no core-utilization
counter, and the project does not fabricate one. The 0.6B run used **7.26 %** of HBM while
driving 8 chips with a global batch of 1 — that is evidence of *low occupancy*, and occupancy
is not utilisation. Establishing MXU efficiency would need a profile this project did not take.

What both accelerators do license is that the 238× measurement is **conservative**: it was
obtained from hardware that was demonstrably barely loaded.

---

## 4. The mitigation

One change was made and re-measured under the metrics contract: `fileCacheCapacity` `0` →
`20Gi` with `fileCacheForRangeRead: true`, on an otherwise identical configuration
(`tpu-v5e8-bs8-seq1024` → `tpu-v5e8-bs8-seq1024-filecache`).

| | Before | After | Delta |
|---|---|---|---|
| Checkpoint load | 235.09 s | 16.13 s | **14.57× faster, −219.0 s** |
| Data prep (inside `other_s`) | 40.32 s | 27.29 s | 1.48× faster, −13.0 s |
| Total wall clock | 732.90 s | 502.78 s | **−230.1 s, −31.4 %** |
| Steady-state step time | 1.75117 s | 1.75123 s | **+0.003 %** |

The last row is the one that makes the first row believable. A storage change that also moved
the compute time would have meant the two runs differed in something else. It did not: the
step time moved by three thousandths of a percent, well inside the p10–p90 band
(1.75006–1.75185 s). **The fix moved exactly the phase it targeted and nothing else.**

The data-prep row is a bonus that was not designed for: the corpus is read through the same
mount, so it benefited too, at a smaller ratio because it is read once rather than repeatedly.

---

## 5. What follows, and what does not

**Follows.** The next intervention worth making is a persistent XLA compilation cache. The
manifest already plumbs `JAX_COMPILATION_CACHE_DIR` and `manifests/env.sh` defaults it to
empty — deliberately, so that `compile_s` would be visible rather than hidden during
measurement. Turning it on targets a segment measured at 81.6–138.5 s per job, 26–32 % of wall
clock, whose near-constancy across runs is the evidence that it is recomputed work.

**Follows.** Fewer, longer jobs. Every number in §2.2 says the same thing: the fixed cost is
amortised only by run length, and this project's own benchmark runs (12, 40 and 100 steps) sit
in the worst part of that curve.

**Follows.** Put the data where the compute is. §2.2a attributes 39.0 s of the two
accelerators' fixed-cost difference to process init and data preparation, of which the corpus
read is the bulk: 30.57 s through a gcsfuse mount against 2.09 s from inside the pod. For a
corpus of this size, shipping it with the image is strictly better than mounting a bucket.

**Does not follow.** That the second compilation can be removed. It is present on every
accelerator record at 13×–496× the median step, and it is roughly half the compile budget, but
these records do not say *why* a second graph is compiled. What the GPU row adds is that it is
not a TPU quirk: a CUDA backend does it too, so the cause is in the program rather than in one
XLA target. Establishing what the cause actually is needs a trace this project did not take.
It is listed in the README as a diagnosis to run, not as a fix to apply.

**Does not follow.** Anything about model quality. `eval.micro_f1` is null in every record; no
run evaluated an adapter. `src/evaluate.py` exists and is unrun. The report makes no claim
about whether the fine-tune learned anything, and the corpus is small enough that it should
not: the records imply a training split of **496–503 documents** (four runs report their epoch
wraparound, and `batches = n_docs // batch_size` bounds it), against an ICD-10 label space
several times larger. Even a completed evaluation would have been a functional check.
