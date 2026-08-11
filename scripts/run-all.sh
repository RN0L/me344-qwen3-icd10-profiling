#!/usr/bin/env bash
# =============================================================================
# scripts/run-all.sh — top-level driver for the whole benchmark campaign.
#
#   scripts/run-all.sh                       # everything, resumable
#   scripts/run-all.sh --stages rows,summary
#   scripts/run-all.sh --skip build,data --backends tpu
#   scripts/run-all.sh --dry-run
#
# Options:
#   --stages LIST      run only these (comma separated, in the order below)
#   --skip LIST        run everything except these
#   --from STAGE       start at this stage and run the rest
#   --backends LIST    restrict rows and sweeps (default cpu,tpu,gpu)
#   --force            rebuild images and re-run cells that already have results
#   --dry-run          print what would happen
#   --budget-min N     stop starting new stages after N minutes
#
# STAGES, in the order they run and why that order
#   preflight   tools, kube contexts, the staged model. Fails in seconds rather
#               than 40 minutes in.
#   build       drives docker/build.sh for whichever backend images are not
#               already in Artifact Registry, pinning TAG to the IMAGE_TAG that
#               manifests/env.sh puts in the manifests, then verifies each URI
#               actually landed. A build that pushes under a different repository
#               name than the manifests reference is otherwise discovered as an
#               ImagePullBackOff twenty minutes into a queued TPU job.
#   data        CodiEsp -> training-ready dataset -> the bucket.
#   rows        the three-backend comparison at seq=512, batch 8 on the two
#               accelerators and 1 on the CPU node — the three run_ids the
#               metrics contract names by example. This
#               is the graded headline, so it runs before the sweeps: if the
#               campaign is interrupted, the chart that matters already exists.
#               Order is cpu, tpu, gpu — cheapest and most certain first, and
#               the single class GH200 last so a queue there blocks nothing.
#   mitigation  the same TPU cell again with gcsfuse fileCacheCapacity set. The
#               baseline deliberately runs without it, reproducing Lab 2's
#               27-minute cold checkpoint load; this pair is the before/after.
#   sweep-tpu / sweep-gpu / sweep-cpu   the batch and seq_len sweeps.
#   summary     a table of everything in results/.
#
# RESUMABILITY
#   build and data are one-shot and leave a marker in results/.state.
#   rows, mitigation and the sweeps are resumable at cell granularity: sweep.sh
#   skips any cell that already has a results/<run_id>.json, and logs the skip.
#   Nothing here deletes a results file; --force re-runs cells instead.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
. "${SCRIPT_DIR}/common.sh"

ALL_STAGES="preflight build data rows mitigation sweep-tpu sweep-gpu sweep-cpu summary"
STAGES="${ALL_STAGES}"
SKIP_STAGES=""
FROM_STAGE=""
BACKENDS="cpu,tpu,gpu"
FORCE=0
DRY_RUN=0
BUDGET_MIN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --stages)     STAGES="$(printf '%s' "${2:?--stages needs a value}" | tr ',' ' ')"; shift 2 ;;
    --skip)       SKIP_STAGES="$(printf '%s' "${2:?--skip needs a value}" | tr ',' ' ')"; shift 2 ;;
    --from)       FROM_STAGE="${2:?--from needs a value}"; shift 2 ;;
    --backends)   BACKENDS="${2:?--backends needs a value}"; shift 2 ;;
    --force)      FORCE=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --budget-min) BUDGET_MIN="${2:?--budget-min needs a value}"; shift 2 ;;
    -h|--help)    sed -n '2,45p' "${BASH_SOURCE[0]}" >&2; exit 0 ;;
    *)            err "unknown argument: $1"; exit 3 ;;
  esac
done

need jq
mkdirs
START_EPOCH="$(date -u +%s)"
CAMPAIGN_LOG="${LOG_DIR}/run-all-$(date -u '+%Y%m%dT%H%M%SZ').log"

if [ -n "${FROM_STAGE}" ]; then
  _acc=""; _hit=0
  for s in ${ALL_STAGES}; do
    if [ "${s}" = "${FROM_STAGE}" ]; then _hit=1; fi
    if [ "${_hit}" = "1" ]; then _acc="${_acc} ${s}"; fi
  done
  [ "${_hit}" = "1" ] || die "--from: no such stage '${FROM_STAGE}' (have: ${ALL_STAGES})"
  STAGES="${_acc}"
fi

wants_stage() {
  local s="$1" x
  for x in ${SKIP_STAGES}; do if [ "$x" = "$s" ]; then return 1; fi; done
  for x in ${STAGES};      do if [ "$x" = "$s" ]; then return 0; fi; done
  return 1
}
wants_backend() {
  local b="$1" x
  for x in $(printf '%s' "${BACKENDS}" | tr ',' ' '); do
    if [ "$x" = "$b" ]; then return 0; fi
  done
  return 1
}
budget_left() {
  if [ "${BUDGET_MIN}" -le 0 ]; then return 0; fi
  local used=$(( ( $(date -u +%s) - START_EPOCH ) / 60 ))
  if [ "${used}" -ge "${BUDGET_MIN}" ]; then
    warn "budget of ${BUDGET_MIN} min exhausted (${used} min used) — stopping before the next stage"
    return 1
  fi
  return 0
}
marker() { printf '%s' "${STATE_DIR}/$1.done"; }
stage_done() { [ -f "$(marker "$1")" ] && [ "${FORCE}" = "0" ]; }
mark_done()  { date -u '+%Y-%m-%dT%H:%M:%SZ' > "$(marker "$1")"; }

run_or_echo() {
  if [ "${DRY_RUN}" = "1" ]; then
    printf '  would run: %s\n' "$*" >&2
    return 0
  fi
  "$@"
}

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
stage_preflight() {
  need jq
  if wants_backend gpu || wants_backend tpu; then need kubectl envsubst gcloud; fi

  local ok=1 b
  for b in $(printf '%s' "${BACKENDS}" | tr ',' ' '); do
    case "$b" in
      cpu) continue ;;
      gpu|tpu) ;;
      *) err "unknown backend in --backends: ${b}"; ok=0; continue ;;
    esac
    select_backend "$b"
    if kubectl config get-contexts "${KCTX}" >/dev/null 2>&1; then
      info "context ok for ${b}: ${KCTX} (ns ${KNS})"
    else
      err "kube context '${KCTX}' is not configured (needed for the ${b} row)."
      err "  gcloud container clusters get-credentials <cluster> --region <region> --project ${PROJECT}"
      ok=0
    fi
  done

  # The staged base model is the single most common cause of a 40-minute job
  # failing at minute 3, so check it now.
  if command -v gcloud >/dev/null 2>&1; then
    if gcloud storage ls "${MODEL_GCS_URI}/" >/dev/null 2>&1; then
      info "base model present: ${MODEL_GCS_URI}"
    else
      warn "cannot list ${MODEL_GCS_URI} — the TPU/GPU rows will fail at model load if it is really missing"
    fi
  fi

  # Every manifest names its image through manifests/env.sh. If that URI is not
  # in the registry the Job will sit in ImagePullBackOff, which on the TPU
  # cluster is discovered only after Kueue admission and a node boot. Say so now.
  # A missing image is only a warning when the build stage is still going to run;
  # otherwise it is fatal, because nothing later will create it.
  if command -v gcloud >/dev/null 2>&1; then
    local build_will_run=0
    if wants_stage build; then build_will_run=1; fi
    for b in $(printf '%s' "${BACKENDS}" | tr ',' ' '); do
      select_backend "$b"
      if gcloud artifacts docker images describe "${IMAGE_URI}" >/dev/null 2>&1; then
        info "${b} image present: ${IMAGE_URI}"
      elif [ "${build_will_run}" = "1" ]; then
        warn "${b} image NOT found: ${IMAGE_URI} — the build stage has to produce exactly this URI"
      else
        err "${b} image NOT found: ${IMAGE_URI}, and the build stage is not in this run."
        err "  Either include the build stage or fix IMAGE_URI_$(printf '%s' "$b" | tr 'a-z' 'A-Z') in manifests/env.sh."
        ok=0
      fi
    done
  fi

  # The launcher for each row has to be able to actually start the trainer: the
  # in-image entrypoint must exist, and every argument the trainer declares
  # required must be on the command line the launcher builds. Neither failure
  # surfaces before the container is running, which on the TPU cluster is after
  # Kueue admission and a node boot, so both are checked here in milliseconds.
  for b in $(printf '%s' "${BACKENDS}" | tr ',' ' '); do
    local launcher=""
    case "$b" in
      cpu) launcher="${REPO_ROOT}/manifests/cpu-run.sh" ;;
      gpu) launcher="${REPO_ROOT}/manifests/gpu-train.yaml" ;;
      tpu) launcher="${REPO_ROOT}/manifests/tpu-train.yaml" ;;
    esac
    if ! verify_entrypoint "$b" "${launcher}"; then ok=0; fi
  done

  info "team=${TEAM} results=${RESULTS_DIR} campaign log=${CAMPAIGN_LOG}"
  [ "${ok}" = "1" ] || die "preflight failed — fix the above before starting a multi-hour campaign"
}

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
# docker/build.sh is the authoritative builder: it knows the three Dockerfiles,
# the per-image platforms (linux/arm64 for the GH200 row, since Grace is ARM64
# and an amd64 image dies there with exec format error) and the push. Driving it
# keeps one definition of how an image is made.
resolve_builder() {
  first_existing "${REPO_ROOT}/docker/build.sh"
}

# Backend name -> the suffix docker/build.sh uses. The GH200 image is called
# `cuda` there, not `gpu`.
build_target_for() {
  case "$1" in
    gpu) printf 'cuda' ;;
    *)   printf '%s' "$1" ;;
  esac
}

resolve_dockerfile() {
  local b="$1" var v
  var="DOCKERFILE_$(printf '%s' "$b" | tr 'a-z' 'A-Z')"
  v="${!var:-}"
  if [ -n "$v" ]; then printf '%s' "$v"; return 0; fi
  first_existing \
    "${REPO_ROOT}/docker/Dockerfile.$(build_target_for "$b")" \
    "${REPO_ROOT}/docker/Dockerfile.${b}" \
    "${REPO_ROOT}/Dockerfile.${b}"
}

# image_exists <uri> — is this exact tag already in Artifact Registry?
image_exists() {
  gcloud artifacts docker images describe "$1" >/dev/null 2>&1
}

stage_build() {
  if stage_done build; then skip "stage build (marker ${STATE_DIR}/build.done; --force to rebuild)"; return 0; fi

  local builder targets="" b missing=0
  for b in $(printf '%s' "${BACKENDS}" | tr ',' ' '); do
    select_backend "$b"
    if [ "${FORCE}" = "0" ] && image_exists "${IMAGE_URI}"; then
      skip "${b} image already in the registry: ${IMAGE_URI}"
      continue
    fi
    missing=1
    targets="${targets} $(build_target_for "$b")"
  done
  if [ "${missing}" = "0" ]; then
    if [ "${DRY_RUN}" = "0" ]; then mark_done build; fi
    return 0
  fi

  need docker
  if builder="$(resolve_builder)"; then
    # TAG is passed through so the tag docker/build.sh pushes is the one
    # manifests/env.sh references. Left to its own devices it uses a UTC
    # timestamp, which no manifest would ever name.
    #
    # ONE TARGET PER INVOCATION. docker/build.sh's main() reads only "$1" — a
    # second word on the command line is silently ignored, so passing all three
    # at once would build one image and leave the other two missing, which then
    # surfaces as the post-build verification failing for reasons that look
    # nothing like "only one image was requested".
    local t
    for t in ${targets}; do
      say "building ${t} via ${builder} (TAG=${IMAGE_TAG:-latest})"
      run_or_echo env TEAM="${TEAM}" REGISTRY="${IMAGE_REPO}" TAG="${IMAGE_TAG:-latest}" \
        bash "${builder}" "${t}"
    done
  else
    warn "docker/build.sh not found — falling back to an inline build per backend"
    run_or_echo gcloud auth configure-docker "${IMAGE_REPO%%/*}" --quiet
    for b in $(printf '%s' "${BACKENDS}" | tr ',' ' '); do
      local dockerfile
      if ! dockerfile="$(resolve_dockerfile "$b")"; then
        warn "no Dockerfile for '${b}' — skipping its build"
        continue
      fi
      select_backend "$b"
      say "building ${b} image (${BUILD_PLATFORM}) -> ${IMAGE_URI}"
      run_or_echo docker build --platform "${BUILD_PLATFORM}" \
        -t "${IMAGE_URI}" -f "${dockerfile}" "${REPO_ROOT}"
      run_or_echo docker push "${IMAGE_URI}"
    done
  fi

  # Verify what actually landed. A build that pushed under a different repository
  # name than the manifests reference is the failure this catches, and it would
  # otherwise show up as ImagePullBackOff twenty minutes into a queued TPU job.
  if [ "${DRY_RUN}" = "0" ]; then
    for b in $(printf '%s' "${BACKENDS}" | tr ',' ' '); do
      select_backend "$b"
      if image_exists "${IMAGE_URI}"; then
        info "${b} image present: ${IMAGE_URI}"
      else
        err "${b} image is NOT in the registry after the build: ${IMAGE_URI}"
        err "  manifests/env.sh names this URI (IMAGE_URI_$(printf '%s' "$b" | tr 'a-z' 'A-Z'))."
        err "  Check that docker/build.sh pushes the same repository name and tag."
        return 1
      fi
    done
    mark_done build
  fi
}

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
stage_data() {
  if stage_done data; then skip "stage data (marker ${STATE_DIR}/data.done; --force to redo)"; return 0; fi

  local prep
  if ! prep="$(first_existing \
        "${REPO_ROOT}/src/prepare_data.py" \
        "${REPO_ROOT}/src/prepare_codiesp.py" \
        "${REPO_ROOT}/src/data/prepare.py")"; then
    die "no data preparation script found (expected src/prepare_data.py). Set PREPARE_DATA_CMD to override."
  fi

  if [ "${FORCE}" = "0" ] && command -v gcloud >/dev/null 2>&1 \
     && [ -n "$(gcloud storage ls "${DATA_GCS_URI}/**" 2>/dev/null | head -1 || true)" ]; then
    skip "stage data — ${DATA_GCS_URI} is already populated"
    if [ "${DRY_RUN}" = "0" ]; then mark_done data; fi
    return 0
  fi

  if [ ! -d "${CODIESP_RAW_DIR}" ]; then
    die "raw CodiEsp files not found at ${CODIESP_RAW_DIR} (set CODIESP_RAW_DIR). This stage must run on the class node."
  fi

  # src/prepare_data.py takes no arguments from here and resolves everything from
  # the environment (its ENV_FALLBACKS table): CODIESP_RAW_DIR for the raw bundle,
  # DATA_LOCAL_DIR for the output, DATA_GCS_URI as the upload target, and a
  # tokenizer from TOKENIZER_PATH / MODEL_HOST_DIR / MODEL_PATH — the first of
  # those that is a directory ON THIS HOST. The tokenizer is the one that bites:
  # MODEL_PATH is the in-container gcsfuse mount and does not exist on the node,
  # and MODEL_HOST_DIR is only populated later, by the CPU row. Stage it now
  # rather than letting the stage die on a missing tokenizer after the corpus has
  # already been parsed.
  local tokenizer="${TOKENIZER_PATH:-${MODEL_HOST_DIR:-}}"
  if [ -n "${tokenizer}" ] && [ ! -d "${tokenizer}" ] && command -v gcloud >/dev/null 2>&1; then
    say "staging the tokenizer from ${MODEL_GCS_URI} -> ${tokenizer}"
    info "prepare_data.py needs a local Qwen3 tokenizer; the CPU row needs the same directory later"
    run_or_echo mkdir -p "${tokenizer}"
    # Tolerant on purpose: this is a convenience pre-stage, and prepare_data.py
    # raises its own pointed error if the tokenizer is still not there.
    run_or_echo gcloud storage cp \
      "${MODEL_GCS_URI}/tokenizer*" "${MODEL_GCS_URI}/*config*.json" "${tokenizer}/" \
      || warn "could not pre-stage the tokenizer from ${MODEL_GCS_URI} — prepare_data.py will say what it needs"
  fi

  say "preparing CodiEsp from ${CODIESP_RAW_DIR}"
  if [ -n "${PREPARE_DATA_CMD:-}" ]; then
    run_or_echo env \
      CODIESP_RAW_DIR="${CODIESP_RAW_DIR}" DATA_LOCAL_DIR="${DATA_LOCAL_DIR}" \
      DATA_GCS_URI="${DATA_GCS_URI}" MODEL_GCS_URI="${MODEL_GCS_URI}" \
      MODEL_HOST_DIR="${MODEL_HOST_DIR:-}" TOKENIZER_PATH="${TOKENIZER_PATH:-}" TEAM="${TEAM}" \
      bash -c "${PREPARE_DATA_CMD}"
  else
    run_or_echo env \
      CODIESP_RAW_DIR="${CODIESP_RAW_DIR}" DATA_LOCAL_DIR="${DATA_LOCAL_DIR}" \
      DATA_GCS_URI="${DATA_GCS_URI}" MODEL_GCS_URI="${MODEL_GCS_URI}" \
      MODEL_HOST_DIR="${MODEL_HOST_DIR:-}" TOKENIZER_PATH="${TOKENIZER_PATH:-}" TEAM="${TEAM}" \
      python3 "${prep}"
  fi
  if [ "${DRY_RUN}" = "0" ]; then mark_done data; fi
}

# ---------------------------------------------------------------------------
# rows / mitigation / sweeps — all delegate to sweep.sh so that "run one cell"
# has exactly one implementation.
# ---------------------------------------------------------------------------
sweep() {
  local args=("$@")
  if [ "${FORCE}" = "1" ]; then args+=(--force); fi
  if [ "${DRY_RUN}" = "1" ]; then args+=(--dry-run); fi
  set +e
  "${SCRIPT_DIR}/sweep.sh" "${args[@]}"
  local rc=$?
  set -e
  if [ "${rc}" != "0" ]; then warn "sweep.sh ${args[*]} exited ${rc} — continuing"; fi
  return 0
}

# Batch size for the headline three-backend row.
#
# The accelerators run the comparison cell at BASE_BATCH (8). The CPU row does
# NOT: docs/metrics-contract.md's own run-id examples are
#   cpu-x86-32c-bs1-seq512 / gpu-gh200-bs8-seq512 / tpu-v5e8-bs8-seq512
# and manifests/env.sh sets CPU_BATCH_SIZE=1, because a batch of 8 activations
# for a 4B model does not fit in the node's 28 GiB alongside the weights. Forcing
# bs=8 there would produce an OOM where the contract expects a measurement, and a
# run_id no chart is looking for. Per-step throughput stays comparable because
# steady.median_step_s is normalised by batch in tokens_per_s / docs_per_s.
row_batch_for() {
  case "$1" in
    cpu) printf '%s' "${ROW_BATCH_CPU:-${CPU_BATCH_SIZE:-1}}" ;;
    *)   printf '%s' "${BASE_BATCH}" ;;
  esac
}

stage_rows() {
  local b bs
  for b in cpu tpu gpu; do
    if ! wants_backend "$b"; then skip "row ${b} (not in --backends)"; continue; fi
    budget_left || return 0
    bs="$(row_batch_for "$b")"
    say "row: ${b} at bs=${bs} seq=${BASE_SEQ}"
    sweep --backend "$b" --cells "bs=${bs}:seq=${BASE_SEQ}"
  done
}

stage_mitigation() {
  if ! wants_backend tpu; then skip "mitigation (tpu not in --backends)"; return 0; fi
  budget_left || return 0
  local bs; bs="$(row_batch_for tpu)"
  say "mitigation: same TPU cell with gcsfuse fileCacheCapacity=${FILE_CACHE_MITIGATED}"
  info "the baseline ran with fileCacheCapacity=${FILE_CACHE}; the pair is the before/after for the cold-load bottleneck"
  # No --variant here: me344_resolve appends -filecache by itself once the
  # cache is on, so passing it too would produce ...-filecache-filecache.
  sweep --backend tpu --cells "bs=${bs}:seq=${BASE_SEQ}" \
        --file-cache "${FILE_CACHE_MITIGATED}"
}

stage_sweep() {
  local b="$1"
  if ! wants_backend "$b"; then skip "sweep-${b} (not in --backends)"; return 0; fi
  budget_left || return 0
  say "sweep: ${b} (batch ${SWEEP_BATCHES} at seq=${BASE_SEQ}, then seq ${SWEEP_SEQS} at bs=${BASE_BATCH})"
  sweep --backend "$b" --axis both
}

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
stage_summary() {
  say "results in ${RESULTS_DIR}"
  local n
  n="$(find "${RESULTS_DIR}" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${n}" = "0" ]; then
    warn "no results files yet"
    return 0
  fi
  # shellcheck disable=SC2016
  jq -r -s '
    def fmt($x): if $x == null then "-" else ($x | tostring | .[0:9]) end;
    ["run_id","status","median_step_s","tokens_per_s","peak_pct","total_wall_s"],
    ["------","------","-------------","------------","--------","------------"],
    (sort_by(.run_id)[] | [
      (.run_id // "?"), (.status // "?"),
      fmt(.steady.median_step_s), fmt(.steady.tokens_per_s),
      fmt(.memory.peak_pct), fmt(.phases.total_wall_s)
    ]) | @tsv' "${RESULTS_DIR}"/*.json 2>/dev/null \
    | awk -F'\t' '{printf "  %-44s %-8s %14s %14s %10s %14s\n", $1,$2,$3,$4,$5,$6}' >&2 \
    || warn "could not summarize results (a results file may be malformed)"
  info "${n} result file(s)"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
say "ME344 final project campaign — team=${TEAM}"
info "stages: ${STAGES}"
if [ -n "${SKIP_STAGES}" ]; then info "skipping: ${SKIP_STAGES}"; fi
info "backends: ${BACKENDS}"
if [ "${DRY_RUN}" = "1" ]; then info "DRY RUN — nothing will be submitted"; fi

for stage in ${ALL_STAGES}; do
  if ! wants_stage "${stage}"; then
    skip "stage ${stage}"
    continue
  fi
  budget_left || break
  case "${stage}" in
    preflight)  stage_preflight ;;
    build)      stage_build ;;
    data)       stage_data ;;
    rows)       stage_rows ;;
    mitigation) stage_mitigation ;;
    sweep-tpu)  stage_sweep tpu ;;
    sweep-gpu)  stage_sweep gpu ;;
    sweep-cpu)  stage_sweep cpu ;;
    summary)    stage_summary ;;
  esac
done

say "campaign finished in $(( ( $(date -u +%s) - START_EPOCH ) / 60 )) min"
