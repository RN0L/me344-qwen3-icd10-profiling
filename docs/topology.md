# System topology — what ran where, and where each timestamp was taken

The README carries the short version of this diagram. This file is the long form: the
concrete names of every resource, the path a byte takes from the bucket to a TPU chip, and —
the part that matters for the measurements — exactly which component observed each phase
boundary in the metrics contract.

Everything named here is read out of `manifests/env.sh`, `manifests/tpu-train.yaml` and
`manifests/cpu-run.sh`, which are the source of truth. Nothing is retyped from memory.

---

## 1. The three planes

The project runs one JAX program against three different backends. They do not share a
cluster, a storage system, or even a CPU architecture. That is the first thing the diagram
has to show, because it is the reason the three columns of the report are not symmetric.

```
                    GCP project  soe-hpccenter                      Stanford on-prem
    ┌───────────────────────────────────────────────────────┐   ┌────────────────────────────┐
    │                                                       │   │  stanford-pilot            │
    │  Artifact Registry (us-central1)                      │   │  ctx student39-context     │
    │  us-central1-docker.pkg.dev/soe-hpccenter/tpu-images   │   │  ns  ns-student39          │
    │    ├── qwen3-codiesp-tpu-team-lharnold  linux/amd64    │   │                            │
    │    ├── qwen3-codiesp-cpu-team-lharnold  linux/amd64    │   │  hpcc-pilot        arm64   │
    │    └── qwen3-codiesp-gpu-team-lharnold  linux/arm64 ─┐ │   │  hpcc-pilot-spark  arm64   │
    │                          │                          ✗ │   │  1x GH200 480GB each       │
    │                          │ image pull       403 ──────┼───┼──►  ✗ never pulled          │
    │                          ▼                            │   │                            │
    │  GKE  class-tpu-cluster-west4   (region us-west4)      │   │  Job qwen3-gpu (public)    │
    │  namespace: default  ← shared by the whole class       │   │   public base image        │
    │  admission: Kueue LocalQueue "student-queue"           │   │   + src from ConfigMap     │
    │                                                       │   │   + PyPI / HF / Zenodo     │
    │   ┌─────────────────────────────────────────────┐     │   │   no registry credential   │
    │   │ Job qwen3-tpu-team-lharnold                 │     │   │   ✅ this is the route      │
    │   │  nodeSelector                               │     │   │      that produced the row │
    │   │    gke-tpu-accelerator: tpu-v5-lite-podslice│     │   └────────────────────────────┘
    │   │    gke-tpu-topology:    2x4                 │     │
    │   │  limits  google.com/tpu: 8                  │     │
    │   │  serviceAccountName: jax-sa  (Workload Id.) │     │
    │   │                                             │     │
    │   │  ┌───────────────┐   ┌────────────────────┐ │     │
    │   │  │ training      │   │ gcsfuse sidecar    │ │     │
    │   │  │ container     │◄──┤ (CSI-injected)     │ │     │
    │   │  │ /app/src/...  │   │ fileCacheCapacity  │ │     │
    │   │  └───────────────┘   └─────────┬──────────┘ │     │
    │   │        mount /gcs               │           │     │
    │   └─────────────────────────────────┼───────────┘     │
    │                                     │                 │
    │      gcsfuse.csi.storage.gke.io     │                 │
    │                                     ▼                 │
    │   Cloud Storage   gs://me344-tpu-labs-west4           │
    │     teams/team-lharnold/models/Qwen3-4B               │
    │     teams/team-lharnold/models/Qwen3-0.6B             │
    │     teams/team-lharnold/data/codiesp                  │
    │     teams/team-lharnold/ckpt/<run_id>                 │
    │     teams/team-lharnold/adapters/<run_id>             │
    │     teams/team-lharnold/results/<run_id>.json         │
    └───────────────────────────────────────────────────────┘

                    Stanford on-prem, no Kubernetes at all
    ┌───────────────────────────────────────────────────────┐
    │  hpcc-cluster-39     2x Xeon E5-2670, 32 logical CPUs │
    │                      31 GiB DRAM                      │
    │                                                       │
    │   podman run --cpus=32 --memory=28g --memory-swap=28g │
    │     ┌─────────────────────────────────────┐           │
    │     │ qwen3-codiesp-cpu-team-lharnold     │           │
    │     │   /model  ← ~/final-project/models  │ ro, local │
    │     │   /data   ← ~/final-project/data    │ ro, local │
    │     │   /results ← repo results/          │ rw, local │
    │     └─────────────────────────────────────┘           │
    │   no scheduler, no registry pull, no network storage  │
    └───────────────────────────────────────────────────────┘
```

### Why the CPU plane is shaped differently, and what it costs the comparison

The CPU run is not a Kubernetes workload. It is a container started directly on the node with
`podman run`, reading the model and the corpus from **local disk**, not from GCS. This is
deliberate — the node is not part of any cluster that can mount the bucket — but it has two
consequences the report must carry rather than hide:

1. `phases.submit_to_running_s` and `phases.image_pull_s` are **0.0 on the CPU row, measured
   rather than assumed**: there is no scheduler queue to wait in and the image is already
   local. The accelerator rows pay both. Comparing end-to-end wall clock therefore charges
   the TPU for a queue the CPU never enters, which is why `analysis.json` also reports
   `amortisation.fixed_excl_scheduler_s`.
2. The CPU run's I/O is a local read. The storage mitigation that matters on the TPU — the
   gcsfuse file cache — has no analogue on the CPU row and no effect on it.

The container runtime flags are the CPU row's equivalent of Kubernetes resource limits, and
they were chosen to match: `--cpus` is the CFS quota that `resources.limits.cpu` sets,
`--memory`/`--memory-swap` at equal values disable swap so that an over-allocation is an OOM
kill and not a slow crawl into swap — which is exactly how the 4B attempts failed.

---

## 2. The storage path, and the one flag that changes it

Every byte the TPU training container reads travels:

```
Cloud Storage object
  → gcsfuse sidecar          (CSI-injected into the pod, given unlimited CPU and memory
                              via gke-gcsfuse/cpu-limit: "0" and memory-limit: "0")
  → FUSE mount at /gcs       (mountOptions: implicit-dirs)
  → open()/read() in the training container
```

Two `volumeAttributes` on the CSI volume decide whether that path has a cache in it:

| Attribute | Baseline | Mitigation | What it does |
|---|---|---|---|
| `fileCacheCapacity` | `"0"` | `"20Gi"` | Size of the sidecar's local file cache. `"0"` disables it. |
| `fileCacheForRangeRead` | `false` | `true` | Whether a *ranged* read populates the cache. |

The second flag is not optional decoration. `safetensors` reads a checkpoint through ranged
reads, so with `fileCacheForRangeRead: false` the capacity alone would have looked like it did
nothing. Both were flipped together, and the measured effect is in
[`bottleneck-analysis.md`](bottleneck-analysis.md#the-mitigation).

The gcsfuse sidecar is deliberately left uncapped on CPU and memory. During checkpoint load
the sidecar, not the training process, is the busy component; a CPU limit there would have
throttled the very path being measured.

---

## 3. Where each phase boundary was observed

This is the part of the topology that determines whether the numbers mean anything. The
metrics contract splits a run into phases, but the phases are not all observed by the same
component, and two of them cannot be seen from inside the pod at all.

| Phase | Observed by | Boundary |
|---|---|---|
| `submit_to_running_s` | the **launcher**, from `kubectl` timestamps | Job creation → pod `Running` |
| `image_pull_s` | the **launcher**, from `kubectl` pod events | pull start → pull complete |
| `model_load_s` | the **training process**, `with timer.phase(...)` | around the checkpoint read |
| `compile_s` | the **training process** | step 1, which is the XLA compile |
| `steady_state_s` | the **training process** | sum of steps 2..N |
| `checkpoint_write_s` | the **training process** | around the adapter save |
| `other_s` | **computed**, never set | `total_wall_s` minus everything attributed |

Three consequences follow, and all three shape the report:

**`other_s` is a residual, not a measurement.** `telemetry.py` computes it so the phase
breakdown always sums to the wall clock — nothing is silently dropped. It records its own
composition in `notes` (`jax_init=…s lora_wrap=…s data_prep=…s trainer_build=…s`), which
`analyze.py` parses back out. Those four named costs do **not** account for all of
`other_s`: the remainder, reported as `other_s_composition_residual_s`, is interpreter startup
and imports, which nothing times explicitly. It runs from 19.5 s to 69.5 s depending on the
run and is real time that the report attributes to process startup rather than pretending it
is explained.

**`image_pull_s` is null on every accelerator record in this checkout.** It is measured
outside the pod and merged in afterwards with `telemetry.merge_external_phases()`. That merge
was never run for these records, so registry pull time sits inside `total_wall_s` without a
name — most likely inside `submit_to_running_s`, which is measured to pod `Running` and
therefore already contains it. The bars in the dashboard show an empty `Image pull` segment
for this reason and not because the pull was instantaneous.

**The CPU row's zeros are measured, not missing.** `telemetry.py` sets both external phases to
`0.0` on the bare-metal backend and says so in `notes`. A null and a zero mean different
things in these records and the analysis treats them differently.

---

## 4. Container images

Three images, one code path — plus a fourth delivery mechanism that carries no image of ours
at all. The split is forced by the hardware, not by the code.

| Image | Platform | Base | Where it ran |
|---|---|---|---|
| `qwen3-codiesp-tpu-team-lharnold` | `linux/amd64` | `jax[tpu]` | GKE `class-tpu-cluster-west4` |
| `qwen3-codiesp-cpu-team-lharnold` | `linux/amd64` | `jax` (CPU) | `hpcc-cluster-39` via podman |
| `qwen3-codiesp-gpu-team-lharnold` | `linux/arm64` | `jax[cuda12]` | built, correct, **never pulled** |
| *(no image of ours)* | `linux/arm64` | public base + ConfigMap | `stanford-pilot` — **this produced the GPU row** |

The arm64 image exists and is correct — 5.5 GB of genuine `ELF 64-bit LSB shared object, ARM
aarch64` with `jax_cuda12_plugin` present and `libtpu` absent, built without executing a single
instruction under emulation. It was never pulled, because `stanford-pilot` has no trust path to
an Artifact Registry in `soe-hpccenter` and the RBAC persona cannot create the `imagePullSecret`
that would give it one (`kubectl auth can-i create secrets` → `no`).

**The route that worked removes the registry from the topology entirely.** It can create
ConfigMaps and Jobs, so: a public base image, the source tree (273 KB) mounted from a
ConfigMap, dependencies resolved from PyPI at start-up, model from Hugging Face, corpus from
Zenodo. `manifests/gpu-train-public.yaml`.

This matters for the topology diagram because it changes which arrows are load-bearing. The
GPU plane does not depend on the registry, on Workload Identity, or on any cross-domain trust —
only on the pod having egress to the public internet, which a 30-second probe Job confirmed it
has. The earlier conclusion that it did not came from testing reachability **from the node
rather than from inside the cluster**; those are different networks, and Lab 4 pulls from
`nvcr.io` on this very cluster. The full chain is in
[`backend-feasibility.md`](backend-feasibility.md).

---

## 5. Identity and isolation

| Plane | Namespace | Isolation |
|---|---|---|
| TPU | `default` | **none** — shared by the whole class; run names are the only separation |
| GPU | `ns-student39` | real per-student namespace isolation |
| CPU | n/a | a single node, one user |

The TPU cluster's shared `default` namespace is worth stating in a systems report because it
is the mechanism behind the most variable number in the dataset: `submit_to_running_s` ranges
from 4 s to 207 s across otherwise identical submissions, a spread of 51.8×. That is Kueue
admitting a Job into a queue that other students are also using. It is a property of the
platform, not of the workload, and the cost analysis reports figures with and without it for
that reason.

The GPU plane's per-student namespace is the visible contrast: its run queued for **11 s**.
When the report attributes part of the GPU's end-to-end advantage to a shorter queue rather
than to Hopper, this table is the reason.

---

## 6. The three storage paths, side by side

The same data-preparation code runs on all three planes and costs three different amounts,
because it is reading through three different things. This is the single most under-appreciated
row in the whole comparison, so it is stated on its own:

| Plane | How the corpus and checkpoint are reached | `data_prep` | `model_load_s` |
|---|---|---|---|
| TPU | Cloud Storage → gcsfuse CSI sidecar → FUSE mount at `/gcs` | **30.57 s** | 4.39 s |
| GPU | baked into the image / fetched at start-up, then local | **2.09 s** | 1.75 s |
| CPU | node-local disk, bind-mounted read-only | **2.62 s** | 2.70 s |

*(0.6B configuration on all three, so the file sizes are identical.)*

A **14.6× difference in data preparation**, from storage architecture alone, with the code held
constant.

The gcsfuse file-cache mitigation recovered 14.57× on checkpoint load, which is arithmetically
close — but the two are not the same measurement and should not be presented as confirming each
other. The mitigation compares *cache off against cache on* for a checkpoint read on the 4B
runs; this table compares *a FUSE mount against local disk* for a corpus read, and the TPU row
here already has the cache enabled. Two different comparisons of two different files that
happen to land on a similar ratio. What they share is only the direction: reaching a bucket
through a FUSE mount is roughly an order of magnitude more expensive than reading a local file,
whichever end of it you measure.

