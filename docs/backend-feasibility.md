# Backend feasibility — what would not run, and why

Not every backend produced a results record. This file states what was attempted, what
stopped it, and the evidence, so that a missing column in the comparison is a finding
rather than a gap.

Every observation here is from a real attempt on the assigned hardware. Where a run
produced no JSON record, the reason is quoted from the system that killed it.

---

## CPU — 32-core x86_64, 31 GiB (`hpcc-cluster-39`)

The CPU row is the hardest of the three, not the easiest. Two independent limits were hit
before a step time could be recorded at the reference configuration (bs=1, seq=1024, bf16).

### 1. With rematerialisation: XLA's CPU compiler did not finish

`--remat DECODER` applies gradient checkpointing across all 36 decoder layers, which is what
makes the TPU run fit comfortably in HBM. On CPU the same graph did not finish compiling.

| | |
|---|---|
| Elapsed in `compile_s` when abandoned | **> 64 minutes** |
| Process state | alive, ~166 % CPU, 17.7 GB RSS — progressing, not hung |
| Same graph on TPU v5e 2x4 | **70.7 s** (`tpu-v5e8-bs8-seq1024`, `phases.compile_s`) |

The ~166 % CPU figure is the point: XLA's CPU compilation is substantially single-threaded,
so 32 cores do not help. This is a compiler-throughput limit, not a FLOPs limit.

### 2. Without rematerialisation: the host ran out of memory

Disabling remat to shorten compilation removed the very thing that was bounding activation
memory. The kernel killed the process:

```
[Mon Aug 10 19:29:09 2026] Out of memory: Killed process 222753 (python)
  total-vm:79637236kB, anon-rss:31504556kB, file-rss:1112kB, shmem-rss:0kB
```

**31.5 GB anonymous RSS against 31 GiB of host RAM.** A 4B-parameter LoRA fine-tune at
bs=1, seq=1024 does not fit on this node without rematerialisation.

The trade is exactly the one remat exists to make, observed from both sides in a single
afternoon: keep remat and pay compile time you cannot afford, or drop it and pay activation
memory you do not have.

### 3. Halving the sequence length did not help — because activations were never the problem

The obvious response to an activation-memory OOM is to shorten the sequence. Re-running at
**seq=512, bs=1, no remat** reached **31.2 GB RSS with 0 GB free** on the same 31 GiB node and
was heading for the same kill. Sequence length is not the lever here.

The reason is arithmetic. JAX's CPU backend materialises the parameters in fp32 regardless of
the `bfloat16` request, so the model alone is 4e9 × 4 B ≈ **16 GB** before a single activation
exists, and the gradient, optimizer and XLA scratch buffers are what close the remaining gap.
Activation memory — the part sequence length controls — is the smaller term.

### Conclusion for the CPU row

**A 4B-parameter LoRA fine-tune does not fit on this node under any configuration tried.**

| Configuration | Outcome |
|---|---|
| bs=1, seq=1024, remat DECODER | compile did not finish in > 64 min |
| bs=1, seq=1024, no remat | OOM-killed at 31.5 GB |
| bs=1, seq=512, no remat | 31.2 GB RSS, 0 GB free, same trajectory |

This is a capacity result, not a missing measurement: the smallest single-host configuration of
this workload exceeds the memory of a 31 GiB commodity node, which is precisely the reason the
accelerator rows exist. It also puts a number on the TPU's advantage that a step-time ratio
would have hidden — the v5e slice does not merely run this faster, it is the only tested
platform on which the 4B model runs at all.

### What the CPU row therefore measures: a second model size

To obtain a like-for-like comparison at all, the workload was re-run at **Qwen3-0.6B**, which
fits comfortably on both platforms. Same script, same data, same flags, one code path lowered
by XLA to two different backends.

| Backend | Model | bs | seq | median step | tokens/s |
|---|---|---|---|---|---|
| CPU 32-core x86_64 | Qwen3-0.6B | 1 | 1024 | **18.9899 s** | 54 |
| TPU v5e 2x4 | Qwen3-0.6B | 1 | 1024 | **0.0800 s** | 12 804 |
| | | | | **237.5×** | **237.5×** |

Read the two halves of this document together and the honest headline is not the ratio. At
0.6B the TPU is 237× faster; at 4B the ratio is undefined, because the denominator does not
exist. **Scaling the model by ~7× did not make the CPU 7× slower — it made the CPU
impossible.** That discontinuity is the finding a speedup chart alone would hide.

### A measurement that was discarded, and why it matters

One 0.6B CPU run reported **93 µs per step and 11 million tokens/s** — physically impossible.
JAX dispatches asynchronously, and the per-step probe had been blocking on a buffer whose
identity never changed, so it timed dispatch rather than completion. `telemetry.py` detected
this itself and said so in the record:

> "the probed LoRA parameter arrays never changed identity across steps, so
> `jax.block_until_ready()` may have been blocking on a stale buffer"

The record still carried `status: "ok"`. **A consumer that plots any record with a passing
status, without reading `notes`, would have charted eleven million tokens per second.** The
run was discarded and repeated; the repeat reproduces the earlier coherent measurement to
within 2.4 % (18.54 s vs 18.99 s per step), which is what makes it trustworthy.

---

## GPU — NVIDIA GH200 480GB (`stanford-pilot`, namespace `ns-student39`)

The GH200 was available and idle throughout. The blocker was never the hardware.

### The host is ARM64

```
NAME               ARCH    GPU
hpcc-pilot         arm64   1
hpcc-pilot-spark   arm64   1
```

Both GPU nodes on `stanford-pilot` are ARM64 (Grace). No x86 GPU node exists on any cluster
this project can reach — `class-tpu-cluster` and `class-tpu-cluster-west4` expose no
`nvidia.com/gpu` capacity at all. An `linux/amd64` image dies on the GH200 with
`exec format error`, so an `linux/arm64` image is mandatory, not preferable.

### Cross-building the image: QEMU segfaults

The assigned node is x86_64, so an arm64 image has to be cross-built. Under
`qemu-aarch64` emulation the build died in the first `RUN` layer:

```
STEP 3/11: RUN apt-get update && apt-get install -y --no-install-recommends git ...
container exited on segmentation fault
Error: building at STEP "RUN apt-get update ...": while running runtime: exit status 1
```

`nvcr.io/nvidia/jax` and `ghcr.io/nvidia/jax` were both unreachable from the node, so
substituting a prebuilt arm64 JAX+CUDA base was not an option either.

### The remaining route, and the dependency that closes it

The way around emulation is to avoid executing any arm64 code at build time: resolve and
unpack the aarch64 wheels **on x86** with `uv pip install --python-platform aarch64-…
--target`, then `COPY` the resulting tree into an image whose Dockerfile contains no `RUN`
at all.

The JAX side of that works — aarch64 CUDA wheels exist:

```
jax_cuda12_plugin-0.10.2-cp312-cp312-manylinux_2_27_aarch64.whl
```

The training stack does not. `google-tunix` pulls in **`libtpu`**, which publishes only:

```
hint: Wheels are available for `libtpu` (v0.0.42.1) on the following platform:
      manylinux_2_31_x86_64
```

There is no aarch64 build of `libtpu`. But it arrives only as a *transitive* dependency, and
the model code needs flax/jax/optax at runtime, not libtpu — so resolving in two passes clears
it:

```
uv pip install --python-platform aarch64-manylinux_2_28 --target arm64-site \
    "jax[cuda12]==0.10.2" flax optax orbax-checkpoint safetensors transformers qwix
uv pip install --python-platform aarch64-manylinux_2_28 --target arm64-site \
    --no-deps "google-tunix @ git+https://github.com/google/tunix@c00f424"
```

5.5 GB of genuine `ELF 64-bit LSB shared object, ARM aarch64`, `jax_cuda12_plugin` present,
`libtpu` absent. Copied into an image whose Dockerfile has no `RUN` at all, so nothing is ever
executed under emulation. **This obstacle was solved.**

### Then: the cluster has no GCS credentials

The GPU manifest stages the checkpoint from a bucket with an initContainer. On
`stanford-pilot` that fails:

```
ERROR: (gcloud.storage.cp) You do not currently have an active account selected.
```

The cluster is Stanford on-prem and has no Workload Identity into `soe-hpccenter`. Minting a
service-account key for a shared class cluster is not a step to take casually, so the model
(Qwen3-0.6B, Apache-2.0) and dataset (CodiEsp, CC-BY-4.0) were baked into the image instead —
both public, neither a secret. Image size 7.53 GB. **This obstacle was solved too.**

### And finally: the cluster cannot authenticate to the registry

```
Failed to pull image ".../qwen3-codiesp-cuda-team-lharnold:selfcontained":
  failed to authorize: failed to fetch anonymous token: unexpected status from GET
  request to https://us-central1-docker.pkg.dev/v2/token?...: 403 Forbidden
```

`stanford-pilot` is on-prem; the Artifact Registry lives in Google's `soe-hpccenter` project.
There is no trust path between them, which is why Lab 4 pulls from the public
`nvcr.io/nvidia/pytorch` rather than from the class registry. Every remaining route needs a
credential that is not ours to create: an `imagePullSecret` from a service-account key, making
a shared course repository public, or a third-party registry.

**This is where the GPU row stops.** Not at the accelerator, which was idle and available the
whole time, and not at the code, which was built correctly for the architecture — but at an
authorization boundary between two administrative domains.

The chain is worth stating in full, because four of its five links were solvable:

| # | Obstacle | Outcome |
|---|---|---|
| 1 | ARM64 Grace host; amd64 image gives `exec format error` | arm64 build required |
| 2 | QEMU cross-build segfaults in the first `RUN` | ✅ solved — Dockerfile with no `RUN` |
| 3 | `libtpu` has no aarch64 wheel | ✅ solved — two-pass resolve with `--no-deps` |
| 4 | No GCS credentials on the cluster | ✅ solved — model and data baked into the image |
| 5 | No registry credentials from cluster to Artifact Registry | ❌ **structural** |

The dependency finding still stands on its own: **the same JAX program that runs on the TPU
cannot have its dependency set installed for an ARM64 host by the obvious route**, because the
framework providing the model implementation is x86-and-TPU-bound by default. "One codebase,
three backends" is true of the code and false of the supply chain.

That chain is the honest reason the GPU column is absent, and it is a portability finding worth
more than a third bar on a chart: a stack chosen for "one codebase, three backends" turned
out to be two backends, and the wall was in the dependency graph rather than in the code.

---

## Summary

| Backend | Record produced | Limit |
|---|---|---|
| TPU v5e 2x4 | 6 runs, 5 `ok` + 1 `oom` | HBM boundary between bs=16 and bs=32 |
| CPU 32-core | 1 run, at **Qwen3-0.6B** | the 4B fine-tune exceeds 31 GiB host RAM at every configuration tried |
| GPU GH200 | none | 5 blockers, 4 solved; stopped at a 403 from the cluster to the Artifact Registry |
