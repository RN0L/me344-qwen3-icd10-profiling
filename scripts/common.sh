#!/usr/bin/env bash
# =============================================================================
# scripts/common.sh — shared configuration and helpers for the run orchestration.
# SOURCE this file, never execute it:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Every value below can be overridden from the environment, or from
# configs/env.sh which is sourced first if it exists.
# =============================================================================
#
# ---------------------------------------------------------------------------
# HOW THIS LAYER RELATES TO manifests/env.sh
# ---------------------------------------------------------------------------
# manifests/env.sh owns the vocabulary: image URIs, contexts, the workload knobs,
# the run_id convention (me344_resolve), the Job names (me344_job_name) and the
# envsubst allowlist (ME344_SUBST_VARS / me344_envsubst). It is sourced by this
# file, and these scripts drive its helpers instead of re-implementing them, so a
# run_id produced here is byte-identical to one produced by hand at a shell.
#
# What this layer adds on top of it:
#   wait-job.sh        a terminal-state wait with phase transitions, a timeout
#                      and a real failure dump. me344_wait polls .status.phase
#                      only, and gives up after a fixed number of polls.
#   collect-phases.sh  the Kubernetes-side timings, which no container can see
#                      because it does not exist yet while they happen.
#   sweep.sh           the cell grid, resumability, and OOM-tolerant harvesting.
#   run-all.sh         stage ordering and the campaign.
#
# THE HARVEST CONTRACT
#   Every Job manifest wraps the training process and echoes the results file
#   between -----BEGIN RESULTS JSON----- and -----END RESULTS JSON-----, emitting
#   a synthesized error record if the process died without writing one. Pod logs
#   are the only output channel allowed on the GH200 namespace, so all three
#   backends harvest the same way. src/telemetry.py additionally frames the
#   record with ===TELEMETRY-JSON-BEGIN=== / ===TELEMETRY-JSON-END===, used here
#   as a fallback. If neither block is present and the run died, sweep.sh
#   synthesizes a schema-valid record from driver-side measurements so that the
#   OOM boundary still appears in the sweep chart.
#
# THE DRIVER OWNS FOUR FIELDS and overwrites whatever the container reported:
#   phases.submit_to_running_s, phases.image_pull_s   (from collect-phases.sh)
#   phases.total_wall_s        (Job completionTime minus Job creationTimestamp)
#   phases.other_s             (recomputed remainder)
# The container derives submit_to_running_s by subtracting SUBMIT_EPOCH from its
# own start time, which cannot separate the image pull from the queue wait and
# cannot see anything that happened before the pod existed. The cluster
# timestamps can, so they win. collect-phases.sh defines each interval exactly.
# ---------------------------------------------------------------------------

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "common.sh is a library — source it, do not execute it." >&2
  exit 64
fi

# --- locations -------------------------------------------------------------
_ME344_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SCRIPTS_DIR:=${_ME344_COMMON_DIR}}"
: "${REPO_ROOT:=$(cd "${_ME344_COMMON_DIR}/.." && pwd)}"
: "${RESULTS_DIR:=${REPO_ROOT}/results}"
: "${LOG_DIR:=${REPO_ROOT}/logs}"
# Resumability markers live under logs/, NOT under results/. .gitignore deliberately
# un-ignores results/ ("the measured results ARE the deliverable"), so a marker file
# written there would be committed alongside the records it is not part of.
: "${STATE_DIR:=${LOG_DIR}/.state}"

# manifests/env.sh is the single source of truth for image URIs, contexts, the
# workload knobs and the run_id convention, and it carries the me344_* helpers
# these scripts drive. Source it first so its values win; everything below only
# fills gaps and maps its names onto the ones used here. Its banner goes to
# stderr because collect-phases.sh writes JSON on stdout.
ME344_ENV="${ME344_ENV:-${REPO_ROOT}/manifests/env.sh}"
if [ -f "${ME344_ENV}" ]; then
  # shellcheck disable=SC1090
  . "${ME344_ENV}" >&2
  ME344_ENV_LOADED=1
else
  ME344_ENV_LOADED=0
fi

# Optional extra site config, still supported.
if [ -f "${REPO_ROOT}/configs/env.sh" ]; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/configs/env.sh"
fi

# --- cluster / project facts (verified 2026-08-10) -------------------------
# Each falls back to the manifests/env.sh name first, then to a literal.
: "${TEAM:=team-lharnold}"
: "${REGION:=us-west4}"
: "${PROJECT:=soe-hpccenter}"
: "${WORKSHOP_BUCKET:=me344-tpu-labs-west4}"
: "${IMAGE_REPO:=${REGISTRY:-us-central1-docker.pkg.dev/soe-hpccenter/tpu-images}}"

# TPU: shared `default` namespace, Kueue-gated. Names are the only isolation.
: "${KCTX_TPU:=${TPU_CONTEXT:-gke_soe-hpccenter_us-west4_class-tpu-cluster-west4}}"
: "${KNS_TPU:=${TPU_NAMESPACE:-default}}"
: "${KUEUE_QUEUE:=student-queue}"

# GPU: one GH200 for the class, ARM64 host, RBAC allows logs but not exec.
: "${KCTX_GPU:=${GPU_CONTEXT:-student39-context}}"
: "${KNS_GPU:=${GPU_NAMESPACE:-ns-student39}}"

# CPU: the class node itself, no scheduler in the path.
: "${CPU_NODE_CORES:=${CPU_CORES:-32}}"

# --- storage layout --------------------------------------------------------
: "${MODEL_GCS_URI:=gs://${WORKSHOP_BUCKET}/teams/${TEAM}/models/Qwen3-4B}"
: "${DATA_GCS_URI:=gs://${WORKSHOP_BUCKET}/teams/${TEAM}/data/codiesp}"
: "${RESULTS_GCS_PREFIX:=${RESULTS_GCS_URI:-gs://${WORKSHOP_BUCKET}/teams/${TEAM}/results}}"
: "${MODEL_PATH:=/gcs/teams/${TEAM}/models/Qwen3-4B}"
: "${DATA_PATH:=/gcs/teams/${TEAM}/data/codiesp}"
: "${CODIESP_RAW_DIR:=${HOME}/final-project/data/train_dev}"
: "${DATA_LOCAL_DIR:=${REPO_ROOT}/data/codiesp}"

# --- experiment defaults ---------------------------------------------------
: "${MODEL_NAME:=Qwen/Qwen3-4B}"
: "${BASE_BATCH:=8}"
: "${BASE_SEQ:=512}"
: "${SWEEP_BATCHES:=1,2,4,8,16,32}"
: "${SWEEP_SEQS:=256,512,1024,2048}"
: "${MAX_STEPS:=100}"
: "${MAX_STEPS_CPU:=30}"        # CPU is ~100x slower per step; same shape, fewer steps
: "${LORA_RANK:=32}"
: "${LORA_ALPHA:=64.0}"
: "${DTYPE:=bfloat16}"
: "${REMAT:=DECODER}"
: "${VARIANT:=${RUN_VARIANT:-}}"
# The gcsfuse file cache knobs live in manifests/env.sh as FILE_CACHE and
# FILE_CACHE_RANGE_READ. "0" disables the cache: the TPU baseline deliberately
# runs with it off to reproduce Lab 2's 27-minute cold checkpoint load, and
# me344_resolve then appends `-filecache` to the run_id by itself whenever the
# cache is on, so a variant name must NOT be passed for that measurement.
# Range reads matter as much as capacity — safetensors loads with ranged reads,
# which bypass the cache entirely unless fileCacheForRangeRead is true.
: "${FILE_CACHE:=0}"
: "${FILE_CACHE_RANGE_READ:=false}"
: "${GCSFUSE_EPHEMERAL_LIMIT:=0}"
: "${FILE_CACHE_MITIGATED:=20Gi}"
: "${GCSFUSE_EPHEMERAL_MITIGATED:=30Gi}"

# --- run plumbing ----------------------------------------------------------
: "${CONTAINER_NAME:=training}"
: "${GCSFUSE_SIDECAR_NAME:=gke-gcsfuse-sidecar}"
# The Job wrapper in manifests/*-train.yaml echoes the results file between these
# two markers on every backend, and emits a synthesized error record if the
# training process died without writing one — so this pair is always present.
: "${METRICS_BEGIN:=-----BEGIN RESULTS JSON-----}"
: "${METRICS_END:=-----END RESULTS JSON-----}"
# src/telemetry.py additionally prints the record itself between these. Used as
# a fallback when the wrapper block is missing or truncated.
: "${TELEMETRY_BEGIN:====TELEMETRY-JSON-BEGIN===}"
: "${TELEMETRY_END:====TELEMETRY-JSON-END===}"
: "${TIMEOUT_MIN:=90}"
: "${TIMEOUT_MIN_CPU:=180}"
: "${POLL_S:=10}"
: "${HEARTBEAT_S:=120}"
: "${POD_DRAIN_MAX_S:=120}"     # how long to wait for a deleted Job's pods to go

# --- logging ---------------------------------------------------------------
# Everything goes to stderr so a script can still write JSON on stdout.
_ts() { date -u '+%H:%M:%SZ'; }
say()  { printf '\n\033[1m=== %s ===\033[0m\n' "$*" >&2; }
info() { printf '[%s] %s\n'        "$(_ts)" "$*" >&2; }
skip() { printf '[%s] SKIP  %s\n'  "$(_ts)" "$*" >&2; }
warn() { printf '[%s] WARN  %s\n'  "$(_ts)" "$*" >&2; }
err()  { printf '[%s] ERROR %s\n'  "$(_ts)" "$*" >&2; }
die()  { err "$*"; exit 1; }

# need <cmd>... — fail early and say how to fix it, rather than 40 minutes in.
need() {
  local missing=() c
  for c in "$@"; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
  if [ "${#missing[@]}" -gt 0 ]; then
    err "missing required tool(s): ${missing[*]}"
    err "  Rocky/RHEL node:  sudo dnf -y install jq gettext podman-docker"
    err "  macOS:            brew install jq gettext coreutils"
    exit 3
  fi
}

require_file() {
  local f="$1" what="${2:-file}"
  [ -f "$f" ] || die "${what} not found: ${f}"
}

# first_existing <path>... — echo the first path that exists, non-zero if none.
first_existing() {
  local p
  for p in "$@"; do
    if [ -e "$p" ]; then printf '%s' "$p"; return 0; fi
  done
  return 1
}

mkdirs() { mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${STATE_DIR}"; }

# --- naming ----------------------------------------------------------------
# Hardware slug used in run_id, per docs/metrics-contract.md. Defers to the
# *_HW_TAG names in manifests/env.sh so both layers produce identical run_ids.
hw_slug() {
  case "$1" in
    cpu) printf '%s' "${CPU_HW_TAG:-x86-${CPU_NODE_CORES}c}" ;;
    gpu) printf '%s' "${GPU_HW_TAG:-gh200}" ;;
    tpu) printf '%s' "${TPU_HW_TAG:-v5e8}" ;;
    *)   die "unknown backend: $1" ;;
  esac
}

# make_run_id <backend> <batch> <seqlen> [variant]
#   <backend>-<hardware>-bs<batch>-seq<seqlen>[-<variant>]
# Pure string form, used for the pre-flight plan and the resumability check.
# resolve_cell() below is what a real run uses, because me344_resolve also
# appends `-filecache` on its own whenever the gcsfuse cache is enabled.
make_run_id() {
  local backend="$1" bs="$2" sl="$3" variant="${4:-}" id
  id="${backend}-$(hw_slug "$backend")-bs${bs}-seq${sl}"
  if [ "${backend}" = "tpu" ] && [ "${FILE_CACHE:-0}" != "0" ]; then id="${id}-filecache"; fi
  if [ -n "$variant" ]; then id="${id}-${variant}"; fi
  printf '%s' "$id"
}

# resolve_cell <backend> <batch> <seqlen> — set BATCH_SIZE, SEQ_LEN, RUN_ID and
# the HW_* block the manifests template in. Delegates to me344_resolve from
# manifests/env.sh when it is available so that the run_id this driver writes to
# results/ is byte-identical to the one the container reports about itself.
resolve_cell() {
  local backend="$1"
  BATCH_SIZE_OVERRIDE="$2"; SEQ_LEN="$3"
  export BATCH_SIZE_OVERRIDE SEQ_LEN RUN_VARIANT="${VARIANT}"
  export FILE_CACHE FILE_CACHE_RANGE_READ GCSFUSE_EPHEMERAL_LIMIT
  if command -v me344_resolve >/dev/null 2>&1; then
    me344_resolve "${backend}" >/dev/null || die "me344_resolve failed for ${backend}"
    HW_ARCH="${HW_HOST_ARCH:-${HW_ARCH}}"
    export HW_ARCH
  else
    BATCH_SIZE="$2"
    RUN_ID="$(make_run_id "${backend}" "$2" "$3" "${VARIANT}")"
    export BATCH_SIZE RUN_ID
  fi
}

# The Job name is fixed per backend (qwen3-<backend>-<team>), not per cell: the
# sweep runs cells one at a time and deletes before it creates, and the shared
# `default` namespace makes a predictable, team-suffixed name the safer choice.
job_name_for() {
  if command -v me344_job_name >/dev/null 2>&1; then
    me344_job_name "$1"
  else
    printf 'qwen3-%s-%s' "$1" "${TEAM:?}"
  fi
}

# Kubernetes object names: lowercase alphanumeric and '-', <=63 chars. Pods add
# a ~6 char suffix to the Job name, so keep Job names comfortably shorter.
sanitize_k8s_name() {
  local n
  n="$(printf '%s' "$1" | tr 'A-Z' 'a-z' | sed -e 's/[^a-z0-9-]/-/g' -e 's/--*/-/g' -e 's/^-//' -e 's/-$//')"
  if [ "${#n}" -gt 52 ]; then n="${n:0:52}"; n="${n%-}"; fi
  printf '%s' "$n"
}

# --- backend selection -----------------------------------------------------
# Per-backend variables must be reassigned on every select_backend call, not
# defaulted with `:=` — run-all.sh selects all three backends in one process, and
# a `:=` default would leave the first backend's values in place for the rest.
# Anything the caller set in the environment before this file was sourced is
# remembered here and still wins.
for _v in HW_LABEL HW_CHIPS HW_CHIP_MODEL HW_ARCH HW_MEM_BYTES \
          IMAGE_NAME IMAGE_URI BUILD_PLATFORM; do
  eval "_INIT_${_v}=\"\${${_v}:-}\""
done
unset _v

# _pref <VARNAME> <default> — the caller's override if there was one, else the
# backend default.
_pref() {
  local ref="_INIT_$1"
  if [ -n "${!ref:-}" ]; then printf '%s' "${!ref}"; else printf '%s' "$2"; fi
}

# select_backend <cpu|gpu|tpu> — sets BACKEND, KCTX, KNS, HW_* and IMAGE_URI.
# The HW_* constants are used ONLY to synthesize a record for a run that died
# before it could report its own hardware block. An empty string means "we do
# not know" and is written to the JSON as null — never as a guess. A successful
# run always overrides all of them with what it measured on the device.
select_backend() {
  BACKEND="$1"
  case "$BACKEND" in
    cpu)
      KCTX=""; KNS=""
      HW_LABEL="$(_pref HW_LABEL "${CPU_HW_LABEL:-CPU x86_64 ${CPU_NODE_CORES} cores}")"
      HW_CHIPS="$(_pref HW_CHIPS "${CPU_HW_CHIPS:-${CPU_NODE_CORES}}")"
      HW_CHIP_MODEL="$(_pref HW_CHIP_MODEL "${CPU_HW_CHIP_MODEL:-}")"
      HW_ARCH="$(_pref HW_ARCH "${CPU_HW_HOST_ARCH:-x86_64}")"
      HW_MEM_BYTES="$(_pref HW_MEM_BYTES "")"
      BUILD_PLATFORM="$(_pref BUILD_PLATFORM "linux/amd64")"
      _BACKEND_IMAGE="${IMAGE_URI_CPU:-}"
      ;;
    gpu)
      KCTX="${KCTX_GPU}"; KNS="${KNS_GPU}"
      HW_LABEL="$(_pref HW_LABEL "${GPU_HW_LABEL:-NVIDIA GH200 480GB}")"
      HW_CHIPS="$(_pref HW_CHIPS "${GPU_HW_CHIPS:-1}")"
      HW_CHIP_MODEL="$(_pref HW_CHIP_MODEL "${GPU_HW_CHIP_MODEL:-}")"
      HW_ARCH="$(_pref HW_ARCH "${GPU_HW_HOST_ARCH:-aarch64}")"  # Grace: an amd64 image dies here
      HW_MEM_BYTES="$(_pref HW_MEM_BYTES "")"        # HBM per GPU: measured, never assumed
      BUILD_PLATFORM="$(_pref BUILD_PLATFORM "linux/arm64")"
      _BACKEND_IMAGE="${IMAGE_URI_GPU:-}"
      ;;
    tpu)
      KCTX="${KCTX_TPU}"; KNS="${KNS_TPU}"
      HW_LABEL="$(_pref HW_LABEL "${TPU_HW_LABEL:-TPU v5e 2x4}")"
      HW_CHIPS="$(_pref HW_CHIPS "${TPU_HW_CHIPS:-8}")"
      HW_CHIP_MODEL="$(_pref HW_CHIP_MODEL "${TPU_HW_CHIP_MODEL:-tpu-v5-lite-podslice}")"
      HW_ARCH="$(_pref HW_ARCH "${TPU_HW_HOST_ARCH:-x86_64}")"
      HW_MEM_BYTES="$(_pref HW_MEM_BYTES "17179869184")"   # 16 GiB HBM per v5e chip
      BUILD_PLATFORM="$(_pref BUILD_PLATFORM "linux/amd64")"
      _BACKEND_IMAGE="${IMAGE_URI_TPU:-}"
      ;;
    *) die "unknown backend '${BACKEND}' (expected cpu, gpu or tpu)" ;;
  esac
  HW_SLUG="$(hw_slug "$BACKEND")"

  # Image URI: an explicit override wins, then whatever manifests/env.sh names
  # for this backend (the manifests reference the same variable, so the two can
  # never disagree), then a tag pinned by run-all.sh.
  IMAGE_URI="$(_pref IMAGE_URI "${_BACKEND_IMAGE}")"
  local pinned="${STATE_DIR}/image-${BACKEND}.txt"
  if [ -z "${IMAGE_URI}" ] && [ -f "$pinned" ]; then
    IMAGE_URI="$(tr -d '[:space:]' < "$pinned")"
  fi
  if [ -z "${IMAGE_URI}" ]; then IMAGE_URI="${IMAGE_REPO}/qwen3-codiesp-${BACKEND}-${TEAM}:${IMAGE_TAG:-latest}"; fi
  IMAGE_NAME="${IMAGE_URI##*/}"; IMAGE_NAME="${IMAGE_NAME%%:*}"
  export BACKEND KCTX KNS HW_LABEL HW_CHIPS HW_CHIP_MODEL HW_ARCH HW_MEM_BYTES
  export HW_SLUG IMAGE_NAME IMAGE_URI BUILD_PLATFORM
}

# kube <args...> — kubectl pinned to the selected cluster. Never relies on
# `kubectl config current-context`, which is the only thing keeping a job from
# landing on the wrong hardware when three clusters are in play.
kube() {
  kubectl --context "${KCTX:?select_backend must run first}" \
          --namespace "${KNS:?select_backend must run first}" "$@"
}

# latest_pod_for_job <job> — newest pod of a Job (Jobs with backoffLimit>0 leave
# older attempts behind). Empty output when the Job never produced a pod.
latest_pod_for_job() {
  kube get pods -l "job-name=$1" -o json 2>/dev/null \
    | jq -r '[.items[]] | sort_by(.metadata.creationTimestamp) | last | .metadata.name // empty' 2>/dev/null \
    || true
}

# Extract the last sentinel-framed JSON block from a log file. Tries the Job
# wrapper markers first, then the telemetry.py ones.
_extract_between() {
  awk -v b="$2" -v e="$3" '
    index($0, b) { buf = ""; inblk = 1; next }
    index($0, e) { if (inblk) { last = buf }; inblk = 0; next }
    inblk        { buf = buf $0 "\n" }
    END          { printf "%s", last }
  ' "$1" 2>/dev/null || true
}

is_valid_record() {
  printf '%s' "$1" | jq -e 'type == "object" and has("phases")' >/dev/null 2>&1
}

# extract_metrics_block <logfile>
#
# Prefers the RICHEST block, not merely the first parseable one. That distinction
# is load-bearing: when the training process dies after telemetry.emit() printed
# its record but before/while the results file was written, the Job wrapper's
# `cat ... || printf` fallback emits a minimal stub — valid JSON, but with no
# `phases` — between the wrapper sentinels. Taking that stub because it parses
# would throw away the only partial phase breakdown the run produced, which is
# exactly the data the OOM boundary in the sweep chart is built from.
#
# Order: wrapper block with phases > telemetry block with phases > wrapper block >
# telemetry block > empty.
extract_metrics_block() {
  local wrapper telemetry
  wrapper="$(_extract_between "$1" "${METRICS_BEGIN}" "${METRICS_END}")"
  telemetry="$(_extract_between "$1" "${TELEMETRY_BEGIN}" "${TELEMETRY_END}")"
  if is_valid_record "${wrapper}";   then printf '%s' "${wrapper}";   return 0; fi
  if is_valid_record "${telemetry}"; then printf '%s' "${telemetry}"; return 0; fi
  if printf '%s' "${wrapper}" | jq -e . >/dev/null 2>&1; then printf '%s' "${wrapper}"; return 0; fi
  printf '%s' "${telemetry}"
}

# --- entrypoint sanity -----------------------------------------------------
# The three Dockerfiles make exactly two copies into the image:
#     COPY docker/smoke.py /app/smoke.py
#     COPY src/            /app/src/
# so a *_ENTRYPOINT that is not under /app/src/ (or /app/smoke.py) names a file
# that does not exist in the image, and the Job dies at `python3: can't open
# file`. On the TPU cluster that is discovered only after Kueue admission and a
# node boot, so it is checked here instead.
entrypoint_repo_path() {
  case "$1" in
    /app/src/*)     printf '%s' "${REPO_ROOT}/src/${1#/app/src/}" ;;
    /app/smoke.py)  printf '%s' "${REPO_ROOT}/docker/smoke.py" ;;
    *)              return 1 ;;
  esac
}

# entrypoint_for <cpu|gpu|tpu> — the in-image path the manifests invoke.
# manifests/env.sh exposes one shared TRAIN_ENTRYPOINT (all three backends run the
# same src/train_qwen3_icd.py and differ only by --backend); the per-backend
# *_ENTRYPOINT names are still honoured if they are set, so this keeps working
# whichever of the two vocabularies that file is using.
entrypoint_for() {
  local per_backend=""
  case "$1" in
    cpu) per_backend="${CPU_ENTRYPOINT:-}" ;;
    gpu) per_backend="${GPU_ENTRYPOINT:-}" ;;
    tpu) per_backend="${TPU_ENTRYPOINT:-}" ;;
    *)   return 1 ;;
  esac
  if [ -n "${per_backend}" ]; then printf '%s' "${per_backend}"
  else printf '%s' "${TRAIN_ENTRYPOINT:-}"; fi
}

# required_cli_flags <python-file> — the argparse options declared required=True.
# Printed one per line; empty output when the file declares none (or is not a
# Python argparse script).
required_cli_flags() {
  [ -f "$1" ] || return 0
  python3 - "$1" <<'PY' 2>/dev/null || true
import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
for call in re.findall(r"add_argument\((.*?)\)\n", src, re.DOTALL):
    if "required=True" in call:
        m = re.search(r"""["'](--[A-Za-z0-9][-A-Za-z0-9]*)["']""", call)
        if m:
            print(m.group(1))
PY
}

# verify_entrypoint <backend> <manifest-or-runner-file>
# 0 = the entrypoint exists in the repo and every required CLI flag it declares
# is actually passed by the file that launches it. Non-zero (with a diagnosis on
# stderr) otherwise. Never guesses: an entrypoint outside the image's COPY paths
# is reported as such rather than assumed to be fine.
verify_entrypoint() {
  local backend="$1" launcher="$2" ep src flag rc=0
  ep="$(entrypoint_for "${backend}")" || { err "unknown backend '${backend}'"; return 1; }
  if [ -z "${ep}" ]; then
    err "${backend}: no entrypoint is set (manifests/env.sh defines TRAIN_ENTRYPOINT)"
    return 1
  fi
  if ! src="$(entrypoint_repo_path "${ep}")"; then
    err "${backend}: entrypoint '${ep}' is not a path the images contain."
    err "  The Dockerfiles COPY src/ to /app/src/ and docker/smoke.py to /app/smoke.py."
    err "  Set TRAIN_ENTRYPOINT in manifests/env.sh to /app/src/<file>.py"
    return 1
  fi
  if [ ! -f "${src}" ]; then
    err "${backend}: entrypoint '${ep}' maps to ${src}, which does not exist in this repo."
    err "  Present in src/: $(ls "${REPO_ROOT}/src" 2>/dev/null | tr '\n' ' ')"
    return 1
  fi
  if [ -n "${launcher}" ] && [ -f "${launcher}" ]; then
    while IFS= read -r flag; do
      [ -n "${flag}" ] || continue
      if ! grep -q -- "${flag}" "${launcher}"; then
        err "${backend}: ${src##*/} declares ${flag} as a required argument, but"
        err "  ${launcher#"${REPO_ROOT}/"} never passes it — the process would exit 2 immediately."
        rc=1
      fi
    done <<<"$(required_cli_flags "${src}")"
  fi
  return "${rc}"
}

export TEAM REGION PROJECT WORKSHOP_BUCKET IMAGE_REPO
export MODEL_PATH DATA_PATH MODEL_GCS_URI DATA_GCS_URI RESULTS_GCS_PREFIX
export METRICS_BEGIN METRICS_END CONTAINER_NAME MODEL_NAME
