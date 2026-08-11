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

### What was measured instead

The CPU row was re-run at **seq=512**, where activation memory is roughly halved. There is a
TPU run at the same sequence length (`tpu-v5e8-bs8-seq512-filecache`) so the comparison is
still like-for-like on tokens, with batch size accounted for.

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

There is no aarch64 build of `libtpu`. The dependency is a hard constraint of the training
framework, not of the accelerator: **the same JAX program that runs on the TPU cannot have
its dependency set installed for an ARM64 host, because the framework that provides the
model implementation is itself x86-and-TPU-bound.**

That is the honest reason the GPU column is absent, and it is a portability finding worth
more than a third bar on a chart: a stack chosen for "one codebase, three backends" turned
out to be two backends, and the wall was in the dependency graph rather than in the code.

---

## Summary

| Backend | Record produced | Limit |
|---|---|---|
| TPU v5e 2x4 | 6 runs, 5 `ok` + 1 `oom` | HBM boundary between bs=16 and bs=32 |
| CPU 32-core | seq=512 only | compile time (with remat) / host RAM (without) |
| GPU GH200 | none | `libtpu` has no aarch64 wheel; QEMU cross-build segfaults |
