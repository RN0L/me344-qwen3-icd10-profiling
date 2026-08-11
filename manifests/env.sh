#!/usr/bin/env bash
# ME344 final project — session variables for all three backends. SOURCE this, do not run it.
#
#   source manifests/env.sh
#   me344_submit tpu && me344_wait tpu && me344_harvest tpu
#
# Everything here is session-scoped and does not survive a logout, so re-source after every fresh
# login on hpcc-cluster-39. A lost $TEAM presents as "a resource looks like it belongs to someone
# else", because the TPU cluster shares one `default` namespace and names are the only isolation.
#
# Every variable uses the `: "${X:=default}"` form, so anything you export BEFORE sourcing (or
# after, since nothing is re-assigned) wins. That is the supported way to sweep a knob:
#
#   BATCH_SIZE_OVERRIDE=16 me344_submit tpu        # batch sweep, run_id follows automatically
#   FILE_CACHE=20Gi FILE_CACHE_RANGE_READ=true me344_submit tpu   # the headline mitigation
#
# Requires: bash 4+, kubectl, gcloud, envsubst (gettext), python3. All present on the node after
# `sudo dnf -y install podman-docker gettext`.

if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
  echo "env.sh must be sourced, not executed:  source ${0}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------------------------
# Repo layout, derived from this file's own location so it works from any cwd. Computed first
# because the image URIs below read docker/images.env out of it.
# ---------------------------------------------------------------------------------------------
: "${ME344_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
: "${ME344_MANIFESTS:=${ME344_REPO}/manifests}"
: "${ME344_RESULTS:=${ME344_REPO}/results}"
export ME344_REPO ME344_MANIFESTS ME344_RESULTS

# ---------------------------------------------------------------------------------------------
# Identity, project, registry
# ---------------------------------------------------------------------------------------------
: "${TEAM:=team-lharnold}"
: "${PROJECT:=soe-hpccenter}"
: "${REGION:=us-west4}"
: "${WORKSHOP_BUCKET:=me344-tpu-labs-west4}"
# The Artifact Registry stays in us-central1 even though the TPU cluster is us-west4. Not a typo.
: "${REGISTRY:=us-central1-docker.pkg.dev/soe-hpccenter/tpu-images}"

# docker/build.sh writes this file after every build: TEAM, REGISTRY, IMAGE_TAG (the immutable
# UTC-timestamp tag of that build) and IMAGE_URI_TPU / IMAGE_URI_GPU / IMAGE_URI_CPU.
# Sourcing it is what keeps the manifests from ever naming an image the build did not produce.
# Anything already exported wins, because every assignment below is `: "${X:=...}"`.
if [ -f "${ME344_REPO}/docker/images.env" ]; then
  # shellcheck disable=SC1091
  . "${ME344_REPO}/docker/images.env"
fi

# Image naming is docker/build.sh's, not this file's: it builds repositories
# `qwen3-codiesp-<tpu|gpu|cpu>-${TEAM}` (repository_for() there) and pushes each one twice, as
# :latest and as an immutable :<UTC timestamp>. The GH200 image is BUILT from Dockerfile.cuda but
# PUSHED under `gpu`, because `gpu` is the backend name every consumer uses — build.sh maps the
# two with backend_for(). These fallbacks only apply before the first build, since sourcing
# docker/images.env above already sets IMAGE_URI_* outright; they still have to match, or the
# very first submit 404s on a repository nothing ever pushed. scripts/common.sh constructs the
# same string a third time — all three must stay in step.
#
# :latest is the default because it is the only tag build.sh guarantees exists for all three
# images at once (a partial rebuild advances only one timestamp). A moving tag does make a
# benchmark irreproducible, so pin the run that goes in the report:
#   export IMAGE_TAG=20260810-214412   # the IMAGE_TAG printed in docker/images.env
: "${IMAGE_TAG:=latest}"
: "${IMAGE_URI_TPU:=${REGISTRY}/qwen3-codiesp-tpu-${TEAM}:${IMAGE_TAG}}"   # linux/amd64, jax[tpu]
: "${IMAGE_URI_GPU:=${REGISTRY}/qwen3-codiesp-gpu-${TEAM}:${IMAGE_TAG}}"   # linux/arm64, CUDA
: "${IMAGE_URI_CPU:=${REGISTRY}/qwen3-codiesp-cpu-${TEAM}:${IMAGE_TAG}}"   # linux/amd64, jax cpu
: "${IMAGE_PULL_POLICY:=Always}"   # phases.image_pull_s is measured; Always still hits the local
                                   # layer cache when the digest matches, it only re-checks.
export TEAM PROJECT REGION WORKSHOP_BUCKET REGISTRY IMAGE_TAG
export IMAGE_URI_TPU IMAGE_URI_GPU IMAGE_URI_CPU IMAGE_PULL_POLICY

# ---------------------------------------------------------------------------------------------
# Clusters, contexts, namespaces
# ---------------------------------------------------------------------------------------------
: "${TPU_CLUSTER:=class-tpu-cluster-west4}"
: "${TPU_CONTEXT:=gke_soe-hpccenter_us-west4_class-tpu-cluster-west4}"
: "${TPU_NAMESPACE:=default}"          # shared by the whole class — names are the only isolation
: "${STUDENT_NUMBER:=39}"
: "${GPU_CONTEXT:=student${STUDENT_NUMBER}-context}"
: "${GPU_NAMESPACE:=ns-student${STUDENT_NUMBER}}"   # real per-student isolation on stanford-pilot
export TPU_CLUSTER TPU_CONTEXT TPU_NAMESPACE STUDENT_NUMBER GPU_CONTEXT GPU_NAMESPACE

# ---------------------------------------------------------------------------------------------
# Where the data and the model live
# ---------------------------------------------------------------------------------------------
# Staged once, read by every backend. The GPU cluster cannot mount GCS, so its Job copies the
# model out of the bucket in an initContainer instead (see gpu-train.yaml).
#
# DATA_GCS_URI must hold the RAW CodiEsp bundle, with its shipped layout preserved:
#
#     <data-root>/train/text_files/*.txt   +  <data-root>/train/*_annotations_task*_processed.tsv
#     <data-root>/dev/text_files/*.txt     +  <data-root>/dev/*_annotations_task*_processed.tsv
#
# That is what src/train_qwen3_icd.py --data-root reads (load_codiesp_split globs text_files/ and
# the *_processed.tsv next to it); it does NOT read the jsonl that src/prepare_data.py emits, so
# do not stage that here. Upload it with:
#     gcloud storage cp -r "${DATA_HOST_DIR}"/* "${DATA_GCS_URI}/"
: "${MODEL_GCS_URI:=gs://${WORKSHOP_BUCKET}/teams/${TEAM}/models/Qwen3-4B}"
: "${DATA_GCS_URI:=gs://${WORKSHOP_BUCKET}/teams/${TEAM}/data/codiesp}"
: "${RESULTS_GCS_URI:=gs://${WORKSHOP_BUCKET}/teams/${TEAM}/results}"
# On the node itself (CPU row): the raw CodiEsp drop and the model cache.
: "${DATA_HOST_DIR:=${HOME}/final-project/data/train_dev}"
: "${MODEL_HOST_DIR:=${HOME}/final-project/models/Qwen3-4B}"
export MODEL_GCS_URI DATA_GCS_URI RESULTS_GCS_URI DATA_HOST_DIR MODEL_HOST_DIR

# ---------------------------------------------------------------------------------------------
# The workload — identical across all three backends, that is the whole point of the benchmark
# ---------------------------------------------------------------------------------------------
: "${MODEL_NAME:=Qwen/Qwen3-4B}"
: "${SEQ_LEN:=512}"
: "${MAX_STEPS:=100}"
: "${WARMUP_STEPS:=10}"        # steady.warmup_steps_excluded in the metrics contract
: "${LORA_RANK:=32}"
: "${LORA_ALPHA:=64.0}"
: "${LEARNING_RATE:=1e-3}"
: "${DTYPE:=bfloat16}"
: "${REMAT:=DECODER}"          # tunix.models.qwen3.model.RematConfig: NONE | BLOCK | DECODER
# In-loop eval is OFF by default and that is a measurement decision, not laziness: an eval pass
# every N steps lands inside the timed step region and would contaminate steady.median_step_s,
# which is the headline number of the whole benchmark. The functional sanity check runs once,
# after training, over EVAL_EXAMPLES dev cases (0 = skip, and eval.micro_f1 is then null rather
# than a made-up number).
: "${EVAL_EVERY_N_STEPS:=20}"
: "${EVAL_EXAMPLES:=0}"
: "${EVAL_MAX_NEW_TOKENS:=24}"
: "${UTIL_INTERVAL_S:=1.0}"    # utilization.sample_interval_s in the metrics contract
# Per-backend batch size. The accelerators run one example per chip; 32 CPU cores do not.
: "${TPU_BATCH_SIZE:=8}"
: "${GPU_BATCH_SIZE:=8}"
: "${CPU_BATCH_SIZE:=1}"
# Sweep knob: set this and every backend uses it, and the run_id changes with it.
: "${BATCH_SIZE_OVERRIDE:=}"
: "${RUN_VARIANT:=}"           # free-form run_id suffix, e.g. RUN_VARIANT=jaxcache
export MODEL_NAME SEQ_LEN MAX_STEPS WARMUP_STEPS LORA_RANK LORA_ALPHA LEARNING_RATE DTYPE REMAT
export EVAL_EVERY_N_STEPS EVAL_EXAMPLES EVAL_MAX_NEW_TOKENS UTIL_INTERVAL_S
export TPU_BATCH_SIZE GPU_BATCH_SIZE CPU_BATCH_SIZE BATCH_SIZE_OVERRIDE RUN_VARIANT

# ONE entrypoint for all three backends — that identity is the premise of the benchmark, so
# there is no per-backend script to drift. `--backend {cpu,gpu,tpu}` selects the JAX platform.
# The path is /app/src/... because every Dockerfile does `COPY src/ /app/src/` with WORKDIR /app
# and PYTHONPATH=/app; there is no /app/train_*.py.
: "${TRAIN_ENTRYPOINT:=/app/src/train_qwen3_icd.py}"
export TRAIN_ENTRYPOINT

# ---------------------------------------------------------------------------------------------
# Hardware descriptors. The manifests are the single source of truth for what the run_id and the
# metrics-contract `hardware` block say; the training script reads these out of the environment
# rather than guessing. Capacity (hbm_per_chip_bytes) is deliberately NOT here — the contract
# says it comes from a runtime probe (jax memory_stats / nvidia-smi / psutil).
# ---------------------------------------------------------------------------------------------
: "${TPU_HW_TAG:=v5e8}"
: "${TPU_HW_LABEL:=TPU v5e 2x4}"
: "${TPU_HW_CHIPS:=8}"
: "${TPU_HW_CHIP_MODEL:=tpu-v5-lite-podslice}"
: "${TPU_HW_HOST_ARCH:=x86_64}"

: "${GPU_HW_TAG:=gh200}"
: "${GPU_HW_LABEL:=NVIDIA GH200 480GB}"
: "${GPU_HW_CHIPS:=1}"
: "${GPU_HW_CHIP_MODEL:=nvidia-gh200-480gb}"
: "${GPU_HW_HOST_ARCH:=aarch64}"       # Grace. An amd64 image dies here with exec format error.

: "${CPU_CORES:=32}"                   # hpcc-cluster-39: 32 cores, 31 GiB
: "${CPU_MEM_GB:=28}"                  # 3 GiB left for the OS and the SSH session
: "${CPU_SHM_GB:=8}"
: "${CPU_HW_TAG:=x86-${CPU_CORES}c}"
: "${CPU_HW_LABEL:=hpcc-cluster-39 ${CPU_CORES}-core x86_64}"
: "${CPU_HW_CHIPS:=${CPU_CORES}}"      # `chips` = units of parallel hardware: 8 TPU chips,
                                       # 1 GPU, 32 CPU cores.
: "${CPU_HW_CHIP_MODEL:=cpu-x86_64}"
: "${CPU_HW_HOST_ARCH:=x86_64}"
export TPU_HW_TAG TPU_HW_LABEL TPU_HW_CHIPS TPU_HW_CHIP_MODEL TPU_HW_HOST_ARCH
export GPU_HW_TAG GPU_HW_LABEL GPU_HW_CHIPS GPU_HW_CHIP_MODEL GPU_HW_HOST_ARCH
export CPU_CORES CPU_MEM_GB CPU_SHM_GB CPU_HW_TAG CPU_HW_LABEL CPU_HW_CHIPS CPU_HW_CHIP_MODEL CPU_HW_HOST_ARCH

# ---------------------------------------------------------------------------------------------
# gcsfuse — the headline experiment
# ---------------------------------------------------------------------------------------------
# Lab 2's training Job had NO fileCacheCapacity and lost roughly 27 minutes streaming the
# checkpoint from GCS with compute idle. That defect is what this project measures, so the cache is
# a knob and not a constant: run once with FILE_CACHE=0, once with the cache on, diff the phases.
# The report's numbers come from those two results/ records, never from this comment.
#
#   FILE_CACHE=0                    cache disabled (the defect, the baseline measurement)
#   FILE_CACHE=20Gi                 enough for the ~8 GiB Qwen3-4B checkpoint plus headroom
#   FILE_CACHE=-1                   unbounded, bounded only by the sidecar's ephemeral storage
#
# fileCacheForRangeRead matters as much as the capacity: safetensors loads with ranged reads, and
# with this false a ranged read bypasses the cache entirely, so the cache would look like it did
# nothing. Turn both on together — me344_filecache does that for you.
: "${FILE_CACHE:=0}"
: "${FILE_CACHE_RANGE_READ:=false}"
: "${GCSFUSE_EPHEMERAL_LIMIT:=0}"      # "0" = unlimited; must exceed FILE_CACHE when the cache is on
export FILE_CACHE FILE_CACHE_RANGE_READ GCSFUSE_EPHEMERAL_LIMIT

# XLA persistent compile cache — the second mitigation lever. Empty means disabled, which is the
# right default for the baseline because a warm cache would hide phases.compile_s.
: "${JAX_CACHE_DIR:=}"
export JAX_CACHE_DIR

me344_filecache() {   # me344_filecache on|off
  case "${1:-}" in
    on)  export FILE_CACHE="${2:-20Gi}" FILE_CACHE_RANGE_READ=true GCSFUSE_EPHEMERAL_LIMIT="${3:-30Gi}" ;;
    off) export FILE_CACHE=0 FILE_CACHE_RANGE_READ=false GCSFUSE_EPHEMERAL_LIMIT=0 ;;
    *)   echo "usage: me344_filecache on [capacity] [ephemeral-limit] | me344_filecache off" >&2; return 2 ;;
  esac
  echo "FILE_CACHE=${FILE_CACHE} FILE_CACHE_RANGE_READ=${FILE_CACHE_RANGE_READ} GCSFUSE_EPHEMERAL_LIMIT=${GCSFUSE_EPHEMERAL_LIMIT}"
}

# ---------------------------------------------------------------------------------------------
# Scheduling and resource bounds (the rubric's "explicit hardware constraints")
# ---------------------------------------------------------------------------------------------
: "${BACKOFF_LIMIT:=0}"          # a silent retry would splice two runs into one set of phase timings
: "${ACTIVE_DEADLINE_S:=7200}"   # worst measured end-to-end was 38 min; 2 h is a generous ceiling
: "${TPU_MEM:=300Gi}"            # proven to schedule on ct5lp-hightpu-8t in Lab 2
: "${TPU_CPU_REQUEST:=8}"        # request only — a CPU *limit* would throttle the gcsfuse read path
: "${TPU_SHM:=16Gi}"
: "${GPU_CPU_REQUEST:=8}"
: "${GPU_CPU_LIMIT:=16}"
: "${GPU_MEM_REQUEST:=48Gi}"
: "${GPU_MEM_LIMIT:=96Gi}"
: "${GPU_EPHEMERAL_REQUEST:=20Gi}"
: "${GPU_EPHEMERAL_LIMIT:=40Gi}"
: "${GPU_MODEL_VOLUME_SIZE:=20Gi}"   # emptyDir that holds the checkpoint copied out of GCS
: "${GPU_SHM:=16Gi}"
: "${GPU_TTL_S:=3600}"           # pod logs are the only output channel there — harvest inside the hour
: "${CPU_K8S_CPU_REQUEST:=8}"
: "${CPU_K8S_CPU_LIMIT:=8}"
: "${CPU_K8S_MEM:=16Gi}"
: "${CPU_K8S_SHM:=8Gi}"
: "${GCP_KEY_SECRET:=gcp-sa-key}"    # optional Secret on stanford-pilot, see gpu-train.yaml
# MUST resolve to an arm64 child manifest: the GH200 node's host half is Grace, and an amd64-only
# image dies there with `exec format error`. gcr.io/google.com/cloudsdktool/google-cloud-cli is
# Google's own registry for the CLI image and publishes linux/amd64 + linux/arm64 under one tag;
# it also avoids Docker Hub's anonymous pull rate limit, which a shared class cluster hits.
: "${GCLOUD_IMAGE:=gcr.io/google.com/cloudsdktool/google-cloud-cli:490.0.0-slim}"
export BACKOFF_LIMIT ACTIVE_DEADLINE_S TPU_MEM TPU_CPU_REQUEST TPU_SHM
export GPU_CPU_REQUEST GPU_CPU_LIMIT GPU_MEM_REQUEST GPU_MEM_LIMIT
export GPU_EPHEMERAL_REQUEST GPU_EPHEMERAL_LIMIT GPU_MODEL_VOLUME_SIZE GPU_SHM GPU_TTL_S
export CPU_K8S_CPU_REQUEST CPU_K8S_CPU_LIMIT CPU_K8S_MEM CPU_K8S_SHM GCP_KEY_SECRET GCLOUD_IMAGE

: "${WAIT_INTERVAL_S:=15}"
: "${WAIT_MAX_POLLS:=480}"       # 480 * 15 s = 2 h, matching ACTIVE_DEADLINE_S
export WAIT_INTERVAL_S WAIT_MAX_POLLS

# ---------------------------------------------------------------------------------------------
# Every name the manifests template in. envsubst is given this list explicitly (SHELL-FORMAT) so
# that it substitutes ONLY these and leaves the shell variables inside the container scripts alone
# — without the list, envsubst would rewrite `$rc` and `$(date)`-adjacent names to empty strings.
# ---------------------------------------------------------------------------------------------
ME344_SUBST_VARS='TEAM PROJECT REGION WORKSHOP_BUCKET REGISTRY IMAGE_TAG
IMAGE_URI_TPU IMAGE_URI_GPU IMAGE_URI_CPU IMAGE_PULL_POLICY GCLOUD_IMAGE
TPU_NAMESPACE GPU_NAMESPACE MODEL_GCS_URI DATA_GCS_URI GCP_KEY_SECRET
RUN_ID BACKEND SUBMIT_EPOCH BATCH_SIZE SEQ_LEN MAX_STEPS WARMUP_STEPS LORA_RANK LORA_ALPHA
LEARNING_RATE DTYPE REMAT EVAL_EVERY_N_STEPS EVAL_EXAMPLES EVAL_MAX_NEW_TOKENS UTIL_INTERVAL_S
MODEL_NAME TRAIN_ENTRYPOINT
HW_TAG HW_LABEL HW_CHIPS HW_CHIP_MODEL HW_HOST_ARCH
FILE_CACHE FILE_CACHE_RANGE_READ GCSFUSE_EPHEMERAL_LIMIT JAX_CACHE_DIR
BACKOFF_LIMIT ACTIVE_DEADLINE_S TPU_MEM TPU_CPU_REQUEST TPU_SHM
GPU_CPU_REQUEST GPU_CPU_LIMIT GPU_MEM_REQUEST GPU_MEM_LIMIT GPU_EPHEMERAL_REQUEST
GPU_EPHEMERAL_LIMIT GPU_MODEL_VOLUME_SIZE GPU_SHM GPU_TTL_S
CPU_K8S_CPU_REQUEST CPU_K8S_CPU_LIMIT CPU_K8S_MEM CPU_K8S_SHM CPU_CORES'
export ME344_SUBST_VARS

me344_envsubst() {   # me344_envsubst < template.yaml > rendered.yaml
  local fmt="" v
  for v in ${ME344_SUBST_VARS}; do fmt="${fmt}\${${v}} "; done
  envsubst "${fmt}"
}

# ---------------------------------------------------------------------------------------------
# Resolution: turn a backend name into the run_id and the hardware block the manifests expect
# ---------------------------------------------------------------------------------------------
me344_resolve() {   # me344_resolve tpu|gpu|cpu
  local backend="${1:?usage: me344_resolve tpu|gpu|cpu}" hw_tag variant=""
  case "${backend}" in
    tpu) BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-${TPU_BATCH_SIZE}}"; hw_tag="${TPU_HW_TAG}"
         HW_LABEL="${TPU_HW_LABEL}"; HW_CHIPS="${TPU_HW_CHIPS}"
         HW_CHIP_MODEL="${TPU_HW_CHIP_MODEL}"; HW_HOST_ARCH="${TPU_HW_HOST_ARCH}" ;;
    gpu) BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-${GPU_BATCH_SIZE}}"; hw_tag="${GPU_HW_TAG}"
         HW_LABEL="${GPU_HW_LABEL}"; HW_CHIPS="${GPU_HW_CHIPS}"
         HW_CHIP_MODEL="${GPU_HW_CHIP_MODEL}"; HW_HOST_ARCH="${GPU_HW_HOST_ARCH}" ;;
    cpu) BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-${CPU_BATCH_SIZE}}"; hw_tag="${CPU_HW_TAG}"
         HW_LABEL="${CPU_HW_LABEL}"; HW_CHIPS="${CPU_HW_CHIPS}"
         HW_CHIP_MODEL="${CPU_HW_CHIP_MODEL}"; HW_HOST_ARCH="${CPU_HW_HOST_ARCH}" ;;
    *)   echo "unknown backend '${backend}' (expected tpu|gpu|cpu)" >&2; return 2 ;;
  esac
  # run_id convention from docs/metrics-contract.md:
  #   <backend>-<hardware>-bs<batch>-seq<seqlen>[-<variant>]
  [ "${backend}" = tpu ] && [ "${FILE_CACHE}" != "0" ] && variant="-filecache"
  [ -n "${RUN_VARIANT}" ] && variant="${variant}-${RUN_VARIANT}"
  BACKEND="${backend}"
  HW_TAG="${hw_tag}"
  RUN_ID="${backend}-${hw_tag}-bs${BATCH_SIZE}-seq${SEQ_LEN}${variant}"
  export BACKEND BATCH_SIZE RUN_ID HW_TAG HW_LABEL HW_CHIPS HW_CHIP_MODEL HW_HOST_ARCH
  return 0
}

me344_job_name() {  # me344_job_name tpu|gpu|cpu
  printf 'qwen3-%s-%s\n' "${1:?}" "${TEAM:?TEAM is not set — source manifests/env.sh}"
}

me344_namespace() {
  case "${1:?}" in
    gpu) printf '%s\n' "${GPU_NAMESPACE}" ;;
    *)   printf '%s\n' "${TPU_NAMESPACE}" ;;
  esac
}

# ---------------------------------------------------------------------------------------------
# Context guard. Three clusters, three contexts, three architectures — `kubectl config
# current-context` before every submit is the only thing preventing a job landing on the wrong
# hardware, and on the shared TPU cluster, on top of someone else's.
# ---------------------------------------------------------------------------------------------
me344_ctx() {   # me344_ctx tpu|gpu|cpu
  local backend="${1:?usage: me344_ctx tpu|gpu|cpu}" want current
  case "${backend}" in
    tpu|cpu) want="${TPU_CONTEXT}" ;;
    gpu)     want="${GPU_CONTEXT}" ;;
    *) echo "unknown backend '${backend}'" >&2; return 2 ;;
  esac
  current="$(kubectl config current-context 2>/dev/null || true)"
  if [ "${current}" != "${want}" ]; then
    echo "context is '${current:-<none>}', want '${want}' — switching"
    if ! kubectl config use-context "${want}" 2>/dev/null; then
      case "${backend}" in
        tpu|cpu)
          echo "no such context yet — fetching credentials for ${TPU_CLUSTER}"
          gcloud container clusters get-credentials "${TPU_CLUSTER}" \
            --region="${REGION}" --project="${PROJECT}"
          kubectl config use-context "${want}" ;;
        gpu)
          echo "context '${want}' is missing from this kubeconfig. The stanford-pilot entry is" >&2
          echo "installed by the Lab 4 setup, it is not fetchable with gcloud." >&2
          return 1 ;;
      esac
    fi
  fi
  printf 'context=%s\n' "$(kubectl config current-context)"
}

# ---------------------------------------------------------------------------------------------
# Submit / wait / harvest
# ---------------------------------------------------------------------------------------------
me344_submit() {   # me344_submit tpu|gpu|cpu
  local backend="${1:?usage: me344_submit tpu|gpu|cpu}" manifest name ns rendered leftover
  : "${TEAM:?TEAM is not set — source manifests/env.sh first}"
  case "${backend}" in
    tpu) manifest="${ME344_MANIFESTS}/tpu-train.yaml" ;;
    gpu) manifest="${ME344_MANIFESTS}/gpu-train.yaml" ;;
    cpu) manifest="${ME344_MANIFESTS}/cpu-train.yaml"
         echo "note: the reported CPU row is measured on hpcc-cluster-39 with manifests/cpu-run.sh."
         echo "      cpu-train.yaml runs the same container on the GKE CPU pool — different silicon"
         echo "      and a different core count, so export CPU_CORES to match before you compare." ;;
    *)   echo "unknown backend '${backend}' (expected tpu|gpu|cpu)" >&2; return 2 ;;
  esac
  [ -f "${manifest}" ] || { echo "missing manifest ${manifest}" >&2; return 1; }

  me344_ctx "${backend}" || return 1
  me344_resolve "${backend}" || return 1
  # phases.submit_to_running_s is measured from here: the container subtracts SUBMIT_EPOCH from
  # its own start time. Refresh it on every submit or the number is meaningless.
  SUBMIT_EPOCH="$(date -u +%s)"; export SUBMIT_EPOCH

  name="$(me344_job_name "${backend}")"
  ns="$(me344_namespace "${backend}")"
  # X's must be the final characters of a mktemp template, so the .yaml is appended afterwards.
  rendered="$(mktemp "${TMPDIR:-/tmp}/${name}.XXXXXX")"
  mv "${rendered}" "${rendered}.yaml"; rendered="${rendered}.yaml"
  me344_envsubst < "${manifest}" > "${rendered}"

  # A variable missing from ME344_SUBST_VARS would survive as a literal `${FOO}` and fail late with
  # a confusing API error. Catch it here instead.
  leftover="$( { grep -o '\${[A-Za-z_][A-Za-z0-9_]*}' "${rendered}" || true; } | sort -u | tr '\n' ' ')"
  if [ -n "${leftover}" ]; then
    echo "unsubstituted variables left in ${rendered}: ${leftover}" >&2
    echo "add them to ME344_SUBST_VARS in manifests/env.sh" >&2
    return 1
  fi

  # `kubectl apply` patches, and the stanford-pilot RBAC persona is create/delete-only, so apply
  # over an existing Job is forbidden. Delete first, create second — everywhere, for symmetry.
  kubectl delete job "${name}" -n "${ns}" --ignore-not-found
  kubectl create -f "${rendered}" || return 1
  printf 'submitted %s (run_id=%s, namespace=%s, manifest=%s)\n' \
    "${name}" "${RUN_ID}" "${ns}" "${rendered}"
  kubectl get pods -n "${ns}" -l "job-name=${name}"
}

me344_wait() {   # me344_wait tpu|gpu|cpu — polls PHASE, never `kubectl logs -f`
  local backend="${1:?usage: me344_wait tpu|gpu|cpu}" name ns phase="" i
  name="$(me344_job_name "${backend}")"; ns="$(me344_namespace "${backend}")"
  # `kubectl logs -f job/<job>` on a Pending pod exits 0 with empty output, so a loop that breaks
  # on success breaks immediately having captured nothing. Poll .status.phase instead.
  for i in $(seq 1 "${WAIT_MAX_POLLS}"); do
    phase="$(kubectl get pods -n "${ns}" -l "job-name=${name}" \
             -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)"
    printf '[%6ds] %s phase=%s\n' "$(( i * WAIT_INTERVAL_S ))" "${name}" "${phase:-<no pod yet>}"
    case "${phase}" in Succeeded|Failed) break ;; esac
    sleep "${WAIT_INTERVAL_S}"
  done
  if [ "${phase}" != "Succeeded" ]; then
    echo "job did not succeed (phase=${phase:-<none>}). Events:" >&2
    kubectl describe pod -n "${ns}" -l "job-name=${name}" | sed -n '/Events/,$p' >&2
    return 1
  fi
}

# me344_extract_block <begin> <end> <file> — the LAST block between two sentinels, exclusive of
# the sentinel lines themselves. Last, not first, because a retried container prints twice.
me344_extract_block() {
  awk -v b="${1:?}" -v e="${2:?}" '
    index($0, b) { buf=""; inblk=1; next }
    index($0, e) { if (inblk) { last=buf }; inblk=0; next }
    inblk        { buf = buf $0 "\n" }
    END          { printf "%s", last }
  ' "${3:?}"
}

me344_harvest() {   # me344_harvest tpu|gpu|cpu — pull the results JSON out of the pod logs
  local backend="${1:?usage: me344_harvest tpu|gpu|cpu}" name ns out log
  me344_resolve "${backend}" || return 1
  name="$(me344_job_name "${backend}")"; ns="$(me344_namespace "${backend}")"
  mkdir -p "${ME344_RESULTS}"
  log="${ME344_RESULTS}/${RUN_ID}.log"
  out="${ME344_RESULTS}/${RUN_ID}.json"
  # -c training is explicit because the gcsfuse CSI driver injects a second container into the
  # pod, and `kubectl logs` on a multi-container pod is ambiguous without it.
  kubectl logs -n "${ns}" "job/${name}" -c training --tail=-1 > "${log}" || return 1
  # Two sentinel pairs can be present: the Job wrapper's, and telemetry.py's own around the
  # record it prints itself. Prefer the wrapper block; fall back to the telemetry one, which
  # survives a wrapper that never got to run. Written to a temp file first so a failed parse
  # cannot leave a truncated results file behind for the chart layer to pick up.
  local tmp="${out}.partial"
  me344_extract_block '-----BEGIN RESULTS JSON-----' '-----END RESULTS JSON-----' "${log}" > "${tmp}"
  if ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "${tmp}" 2>/dev/null; then
    me344_extract_block '===TELEMETRY-JSON-BEGIN===' '===TELEMETRY-JSON-END===' "${log}" > "${tmp}"
  fi
  if ! python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["run_id"], d["status"])' "${tmp}"; then
    echo "no valid results JSON in ${log}" >&2
    rm -f "${tmp}"
    return 1
  fi
  mv "${tmp}" "${out}"
  printf 'wrote %s\n' "${out}"
}

echo "TEAM=${TEAM}  IMAGE_TAG=${IMAGE_TAG}  FILE_CACHE=${FILE_CACHE}  SEQ_LEN=${SEQ_LEN}  MAX_STEPS=${MAX_STEPS}"
echo "repo=${ME344_REPO}"
echo "contexts: tpu=${TPU_CONTEXT} gpu=${GPU_CONTEXT} (current=$(kubectl config current-context 2>/dev/null || echo '<none>'))"
echo "helpers: me344_filecache on|off | me344_submit <b> | me344_wait <b> | me344_harvest <b>"
