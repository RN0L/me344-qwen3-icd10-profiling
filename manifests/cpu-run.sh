#!/usr/bin/env bash
# ME344 final project — Qwen3-4B LoRA on CodiEsp, 32-core x86_64 CPU row.
#
#   source manifests/env.sh && bash manifests/cpu-run.sh
#   BATCH_SIZE_OVERRIDE=4 bash manifests/cpu-run.sh      # sweep, run_id follows automatically
#
# WHY THIS IS A SCRIPT AND NOT A KUBERNETES MANIFEST
# --------------------------------------------------
# hpcc-cluster-39 is a plain Rocky 9.5 login node. It has kubectl and gcloud, and it is the machine
# from which the other two clusters are driven, but it is not itself a Kubernetes node: there is no
# kubelet, no API server, no CNI. `kubectl apply` there talks to GKE, not to the local box. So the
# only way to run the CPU row on the 32 cores the rubric cares about is a container runtime
# directly — podman, via the `docker` shim that `podman-docker` installs.
#
# That does not excuse dropping the hardware bounds. This wrapper states them as explicitly as the
# Kubernetes manifests do, and then PROVES they landed by reading them back out of the container
# after start:
#
#   --cpus / --cpuset-cpus     <-> resources.limits.cpu    (CFS quota + core affinity)
#   --memory / --memory-swap   <-> resources.limits.memory (equal values disable swap, so an OOM
#                                                           is a real OOM and not a swap storm)
#   --shm-size                 <-> the /dev/shm emptyDir   (default is 64 MB and kills dataloaders)
#   -v host:container:ro       <-> the volume mounts       (node-local disk, not a network mount)
#
# manifests/cpu-train.yaml carries the same workload as a real Job for a Kubernetes CPU pool. This
# script is the one that produces the reported cpu-* row, because it is the one that runs on the
# 32-core node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${TEAM:-}" ] || ! command -v me344_resolve >/dev/null 2>&1; then
  # shellcheck source=./env.sh
  . "${HERE}/env.sh"
fi

: "${DOCKER:=docker}"          # podman-docker provides this shim on Rocky 9
: "${CPU_PULL:=true}"          # set false to use a locally built image without touching the registry
me344_resolve cpu
SUBMIT_EPOCH="$(date -u +%s)"
CONTAINER="qwen3-cpu-${TEAM}"

say() { printf '\n=== %s ===\n' "$*"; }

command -v "${DOCKER}" >/dev/null || { echo "no '${DOCKER}' on PATH — sudo dnf -y install podman-docker" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------
# Stage the checkpoint onto node-local disk. This mirrors the GPU Job's initContainer: same source
# of truth (the bucket), same timing file, so phases.model_load_s means the same thing on both.
# Skipped when the directory is already populated. The corpus is normally already on the node, so
# it is only fetched when it is not (see below).
# ---------------------------------------------------------------------------------------------
stage() {   # stage <gcs-uri> <host-dir> [sentinel-file]   (no sentinel => "directory is non-empty")
  local uri="$1" dir="$2" sentinel="${3:-}" start end
  populated() {
    if [ -n "${sentinel}" ]; then [ -f "${dir}/${sentinel}" ]
    else [ -d "${dir}" ] && [ -n "$(ls -A "${dir}" 2>/dev/null)" ]; fi
  }
  if populated; then
    printf 'already staged: %s\n' "${dir}"
    return 0
  fi
  mkdir -p "${dir}"
  say "Staging ${uri} -> ${dir}"
  start="$(date +%s)"
  gcloud storage cp -r "${uri}/*" "${dir}/"
  end="$(date +%s)"
  printf 'staged in %s s\n' "$(( end - start ))"
  populated || { echo "FATAL: ${dir} is still empty (or ${sentinel} is missing) after the copy" >&2; exit 1; }
}

stage "${MODEL_GCS_URI}" "${MODEL_HOST_DIR}" "config.json"
# The corpus is already on this node (the class ships it to ~/final-project/data/train_dev), so
# there is nothing to fetch; only pull it from the bucket when that directory is missing. The
# trainer's --data-root wants the RAW layout, <root>/train/text_files/*.txt plus
# <root>/train/*_annotations_task*_processed.tsv — not the jsonl prepare_data.py writes.
if [ ! -d "${DATA_HOST_DIR}/train/text_files" ]; then
  stage "${DATA_GCS_URI}" "${DATA_HOST_DIR}"
fi
[ -d "${DATA_HOST_DIR}/train/text_files" ] || {
  echo "FATAL: ${DATA_HOST_DIR}/train/text_files is missing. --data-root needs the raw CodiEsp" >&2
  echo "       bundle (train/ and dev/, each with text_files/ and *_annotations_task*.tsv)." >&2
  exit 1
}

# python3 rather than `du -sb`: -b is GNU-only and this script also has to be runnable from a
# laptop for a dry check.
MODEL_BYTES="$(python3 -c 'import os,sys
print(sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(sys.argv[1]) for f in fs))' "${MODEL_HOST_DIR}")"
# Provenance for the report, written for symmetry with the GPU Job's initContainer. The checkpoint
# is already on node-local disk by the time the container starts, so there is no cold fetch to
# attribute and fetch_s is a measured zero rather than an absent field. Nothing on this path reads
# it back — the CPU wrapper has no fetch phase to fold in.
printf '{"fetch_s": 0, "model_bytes": %s, "note": "checkpoint pre-staged on node-local disk"}\n' \
  "${MODEL_BYTES}" > "${MODEL_HOST_DIR}/.fetch_metrics.json"

mkdir -p "${ME344_RESULTS}"
RESULT_JSON="${ME344_RESULTS}/${RUN_ID}.json"
RESULT_LOG="${ME344_RESULTS}/${RUN_ID}.log"
rm -f "${RESULT_JSON}"

if [ "${CPU_PULL}" = "true" ]; then
  say "Pulling ${IMAGE_URI_CPU}"
  "${DOCKER}" pull "${IMAGE_URI_CPU}"
fi

"${DOCKER}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true

say "Starting ${CONTAINER} (run_id=${RUN_ID})"
# --cpus is a quota, --cpuset-cpus is affinity; both are stated so the bound is unambiguous whether
# you read the cgroup or /proc/self/status.
CID="$("${DOCKER}" run -d \
  --name "${CONTAINER}" \
  --cpus="${CPU_CORES}" \
  --cpuset-cpus="0-$(( CPU_CORES - 1 ))" \
  --memory="${CPU_MEM_GB}g" \
  --memory-swap="${CPU_MEM_GB}g" \
  --shm-size="${CPU_SHM_GB}g" \
  -e TEAM="${TEAM}" \
  -e BACKEND=cpu \
  -e RUN_ID="${RUN_ID}" \
  -e SUBMIT_EPOCH="${SUBMIT_EPOCH}" \
  -e MODEL_PATH=/model \
  -e DATA_ROOT=/data \
  -e RESULTS_DIR=/results \
  -e FETCH_METRICS_PATH=/model/.fetch_metrics.json \
  -e MODEL_NAME="${MODEL_NAME}" \
  -e JAX_PLATFORMS=cpu \
  -e OMP_NUM_THREADS="${CPU_CORES}" \
  -e MKL_NUM_THREADS="${CPU_CORES}" \
  -v "${MODEL_HOST_DIR}:/model:ro" \
  -v "${DATA_HOST_DIR}:/data:ro" \
  -v "${ME344_RESULTS}:/results" \
  "${IMAGE_URI_CPU}" \
  python3 -u "${TRAIN_ENTRYPOINT}" \
    --backend cpu \
    --model-path /model \
    --data-root /data \
    --out "/results/${RUN_ID}.json" \
    --run-id "${RUN_ID}" \
    --hw-tag "${HW_TAG}" \
    --hw-label "${HW_LABEL}" \
    --chip-model "${HW_CHIP_MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --max-steps "${MAX_STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lr "${LEARNING_RATE}" \
    --dtype "${DTYPE}" \
    --remat "${REMAT}" \
    --file-cache-capacity "n/a" \
    --util-interval "${UTIL_INTERVAL_S}" \
    --eval-every "${EVAL_EVERY_N_STEPS}" \
    --eval-examples "${EVAL_EXAMPLES}" \
    --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
    --adapter-out "/results/adapters/${RUN_ID}" \
    --notes "hardware bounds applied by the container runtime: --cpus=${CPU_CORES} --memory=${CPU_MEM_GB}g --memory-swap=${CPU_MEM_GB}g --shm-size=${CPU_SHM_GB}g")"

# Read the constraints back off the running container. If the runtime silently ignored them —
# rootless podman without cgroup v2 delegation does exactly that — this prints zeros and the
# benchmark is invalid, so it is worth two lines to check.
say "Applied hardware constraints (0 means the runtime ignored the flag)"
"${DOCKER}" inspect -f \
  'NanoCpus={{.HostConfig.NanoCpus}} CpusetCpus={{.HostConfig.CpusetCpus}} Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}} ShmSize={{.HostConfig.ShmSize}}' \
  "${CID}"
if [ "$("${DOCKER}" inspect -f '{{.HostConfig.Memory}}' "${CID}")" = "0" ]; then
  echo "WARNING: the memory limit did not apply. Re-run with DOCKER='sudo docker' (rootful)." >&2
fi

say "Streaming logs (this is the CPU row: expect it to be slow, that is the finding)"
"${DOCKER}" logs -f "${CID}" | tee "${RESULT_LOG}" &
LOGPID=$!
RC="$("${DOCKER}" wait "${CID}")"
wait "${LOGPID}" 2>/dev/null || true

# A crashed run still has to leave a record behind: the OOM boundary in the sweep chart is built
# from status="oom"/"error" files, so a missing file would silently erase a data point.
if [ ! -f "${RESULT_JSON}" ]; then
  # `phases` is present because scripts/sweep.sh only accepts a record that has one; the single
  # number in it is the wall clock this script measured around the container, nothing else.
  printf '{"run_id":"%s","backend":"cpu","status":"error","error":"the training process exited %s without writing %s","phases":{"total_wall_s":%s},"steps":[],"steady":null,"memory":null,"utilization":null,"eval":null,"notes":"synthesised by manifests/cpu-run.sh; the only measured number here is total_wall_s, taken with date(1) around the container"}\n' \
    "${RUN_ID}" "${RC}" "${RESULT_JSON}" "$(( $(date -u +%s) - SUBMIT_EPOCH ))" > "${RESULT_JSON}"
fi

# Same sentinels as the two Kubernetes jobs, so one harvester reads all three backends.
echo "-----BEGIN RESULTS JSON-----"
cat "${RESULT_JSON}"
# Newline first: a file that does not end in one would glue its closing brace onto the
# sentinel line, which the harvester drops as the terminator.
echo
echo "-----END RESULTS JSON-----"

"${DOCKER}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true

python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["run_id"], d["status"])' "${RESULT_JSON}" \
  || { echo "results file is not valid JSON: ${RESULT_JSON}" >&2; exit 1; }

if [ "${RC}" != "0" ]; then
  echo "container exited ${RC} — see ${RESULT_LOG}" >&2
  exit "${RC}"
fi
say "Done — ${RESULT_JSON}"
