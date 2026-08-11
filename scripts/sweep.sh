#!/usr/bin/env bash
# =============================================================================
# scripts/sweep.sh — batch-size and sequence-length sweep driver.
#
#   scripts/sweep.sh --backend tpu
#   scripts/sweep.sh --backend gpu --axis batch
#   scripts/sweep.sh --backend tpu --cells "bs=8:seq=512" --variant filecache \
#   scripts/sweep.sh --backend tpu --cells "bs=8:seq=512" --file-cache 20Gi
#
# Options:
#   --backend cpu|gpu|tpu   required
#   --axis batch|seq|both|none      default both
#   --batches "1,2,4,8,16,32"       batch axis, at --fixed-seq
#   --seqs    "256,512,1024,2048"   sequence axis, at --fixed-batch
#   --fixed-seq N / --fixed-batch N
#   --cells "bs=8:seq=512,bs=16:seq=512"   explicit cells, overrides --axis
#   --variant NAME          extra run_id suffix (RUN_VARIANT). Do NOT pass
#                           "filecache" — me344_resolve appends that by itself
#                           whenever the cache is on, and you would get it twice.
#   --file-cache S          gcsfuse fileCacheCapacity ("0" = off, the baseline).
#                           Also turns fileCacheForRangeRead on, without which a
#                           safetensors ranged read bypasses the cache entirely
#                           and the mitigation would appear to do nothing.
#   --max-steps N / --timeout-min N
#   --force                 re-run cells that already have a results file
#   --dry-run               print the plan and exit
#   --abort-axis-on-oom     stop an axis at its first OOM (default: keep going)
#   --fail-on-error         exit non-zero if any cell errored (default: exit 0)
#
# RESUMABILITY
#   A cell whose results/<run_id>.json exists is skipped and logged. Delete that
#   file to re-run one cell; pass --force to re-run everything.
#
# WHY AN OOM DOES NOT ABORT THE SWEEP
#   The OOM boundary is the chart. A cell that runs out of memory is a measured
#   data point, so it is written out with status "oom" and the sweep moves to the
#   next cell. Only a broken precondition (missing manifest, no cluster) stops
#   the run. If the training container dies before it can report anything, the
#   record is synthesized here from driver-side measurements — with nulls, never
#   invented numbers, for everything that was not observed.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
. "${SCRIPT_DIR}/common.sh"

SW_BACKEND=""
AXIS="both"
CELLS_ARG=""
FORCE=0
DRY_RUN=0
ABORT_AXIS_ON_OOM=0
FAIL_ON_ERROR=0
MAX_STEPS_ARG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --backend)              SW_BACKEND="${2:?--backend needs a value}"; shift 2 ;;
    --axis)                 AXIS="${2:?--axis needs a value}"; shift 2 ;;
    --batches)              SWEEP_BATCHES="${2:?--batches needs a value}"; shift 2 ;;
    --seqs)                 SWEEP_SEQS="${2:?--seqs needs a value}"; shift 2 ;;
    --fixed-seq)            BASE_SEQ="${2:?--fixed-seq needs a value}"; shift 2 ;;
    --fixed-batch)          BASE_BATCH="${2:?--fixed-batch needs a value}"; shift 2 ;;
    --cells)                CELLS_ARG="${2:?--cells needs a value}"; shift 2 ;;
    --variant)              VARIANT="${2:?--variant needs a value}"; shift 2 ;;
    --file-cache|--file-cache-capacity)
                            FILE_CACHE="${2:?--file-cache needs a value}"
                            if [ "${FILE_CACHE}" != "0" ]; then
                              FILE_CACHE_RANGE_READ=true
                              : "${GCSFUSE_EPHEMERAL_LIMIT:=${GCSFUSE_EPHEMERAL_MITIGATED}}"
                              if [ "${GCSFUSE_EPHEMERAL_LIMIT}" = "0" ]; then
                                GCSFUSE_EPHEMERAL_LIMIT="${GCSFUSE_EPHEMERAL_MITIGATED}"
                              fi
                            fi
                            shift 2 ;;
    --max-steps)            MAX_STEPS_ARG="${2:?--max-steps needs a value}"; shift 2 ;;
    --timeout-min)          TIMEOUT_MIN="${2:?--timeout-min needs a value}"; shift 2 ;;
    --force)                FORCE=1; shift ;;
    --dry-run)              DRY_RUN=1; shift ;;
    --abort-axis-on-oom)    ABORT_AXIS_ON_OOM=1; shift ;;
    --fail-on-error)        FAIL_ON_ERROR=1; shift ;;
    -h|--help)              sed -n '2,40p' "${BASH_SOURCE[0]}" >&2; exit 0 ;;
    *)                      err "unknown argument: $1"; exit 3 ;;
  esac
done

: "${SW_BACKEND:?--backend is required (cpu, gpu or tpu)}"
need jq
select_backend "${SW_BACKEND}"
mkdirs

# me344_resolve appends `-filecache` on its own whenever the cache is on, so
# passing it as a variant too yields `...-filecache-filecache` and the mitigation
# record silently stops matching the baseline's run_id pattern.
if [ "${VARIANT}" = "filecache" ] && [ "${FILE_CACHE}" != "0" ]; then
  die "--variant filecache with the cache enabled would produce '...-filecache-filecache'. Drop --variant: the suffix is added automatically."
fi

if [ "${BACKEND}" != "cpu" ]; then
  need kubectl envsubst
  : "${TIMEOUT_MIN:?}"
else
  TIMEOUT_MIN="${TIMEOUT_MIN_CPU}"
fi

# Steps per run: CPU is orders of magnitude slower per step, so it runs fewer of
# them. Shape is unchanged and steady.median_step_s stays comparable; the actual
# value used is recorded in config.max_steps of every record.
if [ -n "${MAX_STEPS_ARG}" ]; then
  MAX_STEPS="${MAX_STEPS_ARG}"
elif [ "${BACKEND}" = "cpu" ]; then
  MAX_STEPS="${MAX_STEPS_CPU}"
fi

# --- resolve the things the other components own ---------------------------
resolve_manifest() {
  local b="$1" var v
  var="MANIFEST_$(printf '%s' "$b" | tr 'a-z' 'A-Z')"
  v="${!var:-}"
  if [ -n "$v" ]; then printf '%s' "$v"; return 0; fi
  first_existing \
    "${REPO_ROOT}/manifests/train-${b}.yaml" \
    "${REPO_ROOT}/manifests/${b}-train.yaml" \
    "${REPO_ROOT}/manifests/train-${b}.job.yaml" \
    "${REPO_ROOT}/manifests/${b}.yaml"
}

# The CPU row runs on hpcc-cluster-39, which is a login node and not a
# Kubernetes node: no kubelet, no API server. manifests/cpu-run.sh runs the same
# container under podman with the cgroup bounds stated explicitly, so that is the
# runner, not a bare python process.
# CPU_CMD is an ARRAY, not a string: `${CPU_CMD}` unquoted would word-split on any
# path containing a space, and quoting it would turn `bash /path/x.sh` into a
# single non-existent filename. An array is the only form that is correct for both.
CPU_CMD=()
CPU_RUNNER=""
resolve_cpu_cmd() {
  if [ -n "${CPU_TRAIN_CMD:-}" ]; then
    # Deliberately word-split: CPU_TRAIN_CMD is a command line supplied by a human.
    # shellcheck disable=SC2206
    CPU_CMD=(${CPU_TRAIN_CMD}); CPU_RUNNER="${CPU_CMD[1]:-${CPU_CMD[0]}}"; return 0
  fi
  local f
  if f="$(first_existing "${REPO_ROOT}/manifests/cpu-run.sh")"; then
    CPU_CMD=(bash "$f"); CPU_RUNNER="$f"; return 0
  fi
  if f="$(first_existing \
        "${REPO_ROOT}/src/train_qwen3_icd.py" \
        "${REPO_ROOT}/src/train_cpu.py" \
        "${REPO_ROOT}/src/train.py")"; then
    CPU_CMD=(python3 "$f"); CPU_RUNNER="$f"; return 0
  fi
  return 1
}

MANIFEST=""
if [ "${BACKEND}" = "cpu" ]; then
  if ! resolve_cpu_cmd; then
    die "no CPU runner found. Expected manifests/cpu-run.sh, or set CPU_TRAIN_CMD."
  fi
else
  if ! MANIFEST="$(resolve_manifest "${BACKEND}")"; then
    die "no Job manifest for '${BACKEND}'. Expected manifests/train-${BACKEND}.yaml, or set MANIFEST_$(printf '%s' "${BACKEND}" | tr 'a-z' 'A-Z')."
  fi
  if ! kubectl config get-contexts "${KCTX}" >/dev/null 2>&1; then
    die "kube context '${KCTX}' is not configured. Run gcloud container clusters get-credentials first."
  fi
fi

# The launcher (Job manifest or cpu-run.sh) names an in-image entrypoint and hands
# it a command line. Both halves have to be right, and neither failure surfaces
# until the container is already running — on the TPU cluster, after a queue wait
# and a node boot. Check them here. SKIP_ENTRYPOINT_CHECK=1 escapes it for a
# deliberately unusual runner (e.g. a stub during a dry test).
if [ "${SKIP_ENTRYPOINT_CHECK:-0}" != "1" ] && [ -z "${CPU_TRAIN_CMD:-}" ]; then
  if [ "${BACKEND}" = "cpu" ]; then
    verify_entrypoint cpu "${CPU_RUNNER}" || die "the CPU runner cannot start the trainer — see above"
  else
    verify_entrypoint "${BACKEND}" "${MANIFEST}" || die "the ${BACKEND} Job manifest cannot start the trainer — see above"
  fi
fi

# Rendering goes through me344_envsubst, which uses the ME344_SUBST_VARS
# allowlist from manifests/env.sh. The allowlist matters: the Job manifests embed
# a bash wrapper containing `$rc` and `$?`, and an unrestricted envsubst would
# quietly rewrite those to empty strings. ME344_SUBST_VARS is the manifests' own
# list, so it can never drift from what they actually reference.
render_manifest() {   # render_manifest <template> <out>
  # envsubst substitutes an allowlisted name that is not in the environment with
  # the EMPTY STRING, not with a literal ${NAME} — so the leftover check below
  # cannot see it, and the result is a Job with `image: ""` or `memory: `. Report
  # names the manifests expect but nothing exported, once per render.
  local v unexported=""
  for v in ${ME344_SUBST_VARS:-}; do
    if ! printenv "$v" >/dev/null 2>&1; then unexported="${unexported} ${v}"; fi
  done
  if [ -n "${unexported}" ]; then
    warn "these names are in ME344_SUBST_VARS but are not exported, so they render as empty:${unexported}"
  fi

  if command -v me344_envsubst >/dev/null 2>&1; then
    me344_envsubst < "$1" > "$2"
  else
    local fmt="" v
    for v in ${ME344_SUBST_VARS:-}; do fmt="${fmt}\${${v}} "; done
    if [ -z "${fmt}" ]; then
      err "no ME344_SUBST_VARS and no me344_envsubst — cannot render safely"
      return 1
    fi
    envsubst "${fmt}" < "$1" > "$2"
  fi
  # A name missing from the allowlist survives as a literal ${FOO} and fails
  # later as a confusing API error. Catch it at render time instead.
  local leftover
  leftover="$( { grep -o '\${[A-Za-z_][A-Za-z0-9_]*}' "$2" || true; } | sort -u | tr '\n' ' ')"
  if [ -n "${leftover}" ]; then
    err "unsubstituted variables left in $2: ${leftover}"
    err "add them to ME344_SUBST_VARS in manifests/env.sh"
    return 1
  fi
  return 0
}

# --- build the cell list ---------------------------------------------------
CELLS=()
add_cell() {
  local c="$1" existing
  for existing in ${CELLS[@]+"${CELLS[@]}"}; do
    if [ "$existing" = "$c" ]; then return 0; fi   # batch and seq axes share a cell
  done
  CELLS+=("$c")
}

if [ -n "${CELLS_ARG}" ]; then
  IFS=',' read -r -a _raw <<<"${CELLS_ARG}"
  for spec in "${_raw[@]}"; do
    b="$(printf '%s' "$spec" | sed -n 's/.*bs=\([0-9]*\).*/\1/p')"
    s="$(printf '%s' "$spec" | sed -n 's/.*seq=\([0-9]*\).*/\1/p')"
    [ -n "$b" ] && [ -n "$s" ] || die "cannot parse cell '${spec}' (want bs=N:seq=N)"
    add_cell "${b}:${s}"
  done
else
  case "${AXIS}" in
    batch|both)
      IFS=',' read -r -a _bs <<<"${SWEEP_BATCHES}"
      for b in "${_bs[@]}"; do add_cell "${b}:${BASE_SEQ}"; done
      ;;
  esac
  case "${AXIS}" in
    seq|both)
      IFS=',' read -r -a _sq <<<"${SWEEP_SEQS}"
      for s in "${_sq[@]}"; do add_cell "${BASE_BATCH}:${s}"; done
      ;;
  esac
  case "${AXIS}" in
    batch|seq|both) ;;
    none) info "--axis none: nothing to run"; exit 0 ;;
    *) die "--axis must be batch, seq, both or none" ;;
  esac
fi

[ "${#CELLS[@]}" -gt 0 ] || die "no cells to run (check --axis / --cells)"

say "sweep: backend=${BACKEND} axis=${AXIS} cells=${#CELLS[@]} variant=${VARIANT:-<none>} file_cache=${FILE_CACHE} range_read=${FILE_CACHE_RANGE_READ}"
info "results -> ${RESULTS_DIR}   logs -> ${LOG_DIR}"
if [ "${BACKEND}" = "cpu" ]; then
  info "runner: ${CPU_CMD[*]}"
else
  info "manifest: ${MANIFEST}"
  info "image: ${IMAGE_URI}"
  info "cluster: ${KNS} @ ${KCTX}"
fi

if [ "${DRY_RUN}" = "1" ]; then
  for cell in "${CELLS[@]}"; do
    bs="${cell%%:*}"; sl="${cell##*:}"
    rid="$(make_run_id "${BACKEND}" "${bs}" "${sl}" "${VARIANT}")"
    state="run"
    if [ -f "${RESULTS_DIR}/${rid}.json" ] && [ "${FORCE}" = "0" ]; then state="skip (exists)"; fi
    printf '  %-42s bs=%-3s seq=%-5s steps=%-4s %s\n' "${rid}" "${bs}" "${sl}" "${MAX_STEPS}" "${state}" >&2
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

# classify_failure <logfile> <k8s terminated reason> <exit code> <wait rc>
# -> oom | timeout | error
classify_failure() {
  local log="$1" reason="${2:-}" code="${3:-}" waitrc="${4:-1}"
  if [ "${reason}" = "OOMKilled" ] || [ "${code}" = "137" ]; then echo "oom"; return; fi
  if [ -f "${log}" ] && grep -qEi \
      'RESOURCE_EXHAUSTED|out of memory|OutOfMemoryError|OOMKilled|MemoryError|ran out of memory|Unable to allocate|failed to allocate|CUDA out of memory' \
      "${log}" 2>/dev/null; then
    echo "oom"; return
  fi
  if [ "${waitrc}" = "2" ] || [ "${code}" = "124" ]; then echo "timeout"; return; fi
  echo "error"
}

# One-line reason for the record's `error` field: prefer the last Python
# exception line, else the last non-empty log line.
error_message() {
  local log="$1" fallback="$2" msg=""
  if [ -f "${log}" ]; then
    msg="$(grep -aE '^[A-Za-z_.]*(Error|Exception|Exhausted|Killed)' "${log}" 2>/dev/null | tail -1 || true)"
    if [ -z "${msg}" ]; then
      msg="$(grep -av '^[[:space:]]*$' "${log}" 2>/dev/null | tail -1 || true)"
    fi
  fi
  if [ -z "${msg}" ]; then msg="${fallback}"; fi
  printf '%s' "${msg}" | cut -c1-500
}

# synthesize_record <status> <error> — a schema-valid record for a run that died
# before emitting its own. Everything unobserved is null, never a guess.
synthesize_record() {
  jq -n \
    --arg run_id "${RUN_ID}" --arg backend "${BACKEND}" \
    --arg ts "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    --arg label "${HW_LABEL}" --argjson chips "${HW_CHIPS}" \
    --arg chip_model "${HW_CHIP_MODEL}" --arg arch "${HW_ARCH}" \
    --arg hbm "${HW_MEM_BYTES}" \
    --arg model "${MODEL_NAME}" \
    --argjson bs "${BATCH_SIZE}" --argjson sl "${SEQ_LEN}" \
    --argjson rank "${LORA_RANK}" --argjson alpha "${LORA_ALPHA}" \
    --arg dtype "${DTYPE}" --argjson steps "${MAX_STEPS}" \
    --arg remat "${REMAT}" --arg fcc "${FILE_CACHE}" \
    --arg status "$1" --arg error "$2" '
    def orNull: if . == "" then null else . end;
    {
      run_id: $run_id, backend: $backend, timestamp_utc: $ts,
      hardware: {
        label: $label, chips: $chips,
        chip_model: ($chip_model | orNull), host_arch: $arch,
        hbm_per_chip_bytes: (if $hbm == "" then null else ($hbm | tonumber) end)
      },
      config: {
        model: $model, batch_size: $bs, seq_len: $sl,
        lora_rank: $rank, lora_alpha: $alpha, dtype: $dtype,
        max_steps: $steps, remat: $remat, file_cache_capacity: $fcc
      },
      phases: {}, steps: [], steady: null, memory: null, utilization: null,
      eval: null, status: $status, error: ($error | orNull),
      notes: "record synthesized by scripts/sweep.sh: the run died before emitting its own metrics block, so only driver-side timings are present."
    }'
}

# write_record <record-json> <phases-json> <driver-elapsed-s> [extra-note]
# The driver owns phases.submit_to_running_s, phases.image_pull_s,
# phases.total_wall_s and phases.other_s — see collect-phases.sh for why.
write_record() {
  local rec="$1" ph="${2:-null}" elapsed="${3:-null}" note="${4:-}"
  local out="${RESULTS_DIR}/${RUN_ID}.json" tmp="${RESULTS_DIR}/.${RUN_ID}.json.tmp"
  if ! printf '%s' "${ph}" | jq -e . >/dev/null 2>&1; then ph="null"; fi
  if [ -z "${elapsed}" ]; then elapsed="null"; fi
  if jq \
    --argjson ph "${ph}" \
    --argjson elapsed "${elapsed}" \
    --arg run_id "${RUN_ID}" \
    --arg note "${note}" '
    # Sum of the disjoint phase parts, i.e. everything except the total and the
    # remainder itself.
    def parts_sum:
      [ .phases | to_entries[]
        | select(.key != "total_wall_s" and .key != "other_s")
        | .value | numbers ] | add // 0;

    ($ph // {}) as $p
    | .run_id = $run_id
    | .phases = (.phases // {})
    | (if $p.submit_to_running_s != null then .phases.submit_to_running_s = $p.submit_to_running_s else . end)
    | (if $p.image_pull_s        != null then .phases.image_pull_s        = $p.image_pull_s        else . end)

    # total_wall_s: prefer the Job creation->completion interval reported by the
    # cluster, else the driver stopwatch around submit and wait. Either one is
    # end-to-end and so can never truly be smaller than the parts it contains.
    # If it is, the two clocks disagree (driver resolution is whole seconds,
    # container resolution is float) and the larger value is kept so that the
    # parts still sum to the total. A material disagreement is noted in notes.
    | (($p.detail.job_total_wall_s // $elapsed) as $driver
       | parts_sum as $sum
       | if $driver == null then .
         elif $driver >= $sum then .phases.total_wall_s = $driver
         else .phases.total_wall_s = $sum
              | (if ($sum - $driver) > 2 then
                   .notes = ((.notes // "") +
                     " WARNING: the driver measured \($driver)s end to end but the phase parts sum to \($sum)s; the larger was kept as total_wall_s. The two clocks disagree — check before charting.")
                 else . end)
         end)
    | .phases.other_s = (
        if (.phases.total_wall_s | type) == "number"
        then (.phases.total_wall_s - parts_sum)
        else (.phases.other_s // null) end)
    | .notes = (((.notes // "") + " " + $note) | sub("^ +"; ""))
    ' <<<"${rec}" > "${tmp}"; then
    mv "${tmp}" "${out}"
  else
    err "could not merge driver phases into the record for ${RUN_ID}; writing it unmerged"
    printf '%s\n' "${rec}" > "${out}"
  fi
  rm -f "${tmp}"
  printf '%s' "${out}"
}

DRIVER_NOTE="phases.submit_to_running_s (image pull excluded), phases.image_pull_s, phases.total_wall_s (Job creation to completion) and phases.other_s were measured by scripts/sweep.sh + scripts/collect-phases.sh, not by the training container."

# ---------------------------------------------------------------------------
# Cell runners. Both return 0 when a results file was written.
# ---------------------------------------------------------------------------

run_cell_k8s() {
  local logfile="${LOG_DIR}/${RUN_ID}.log"
  local phasefile="${LOG_DIR}/${RUN_ID}.phases.json"
  local rendered="${LOG_DIR}/${RUN_ID}.job.yaml"
  local errfile="${LOG_DIR}/${RUN_ID}.kubectl.err"

  JOB_NAME="$(sanitize_k8s_name "$(job_name_for "${BACKEND}")")"
  export JOB_NAME
  # The container subtracts SUBMIT_EPOCH from its own start time, so it has to be
  # refreshed on every submit or the number it reports is meaningless.
  SUBMIT_EPOCH="$(date -u +%s)"
  export SUBMIT_EPOCH

  if ! render_manifest "${MANIFEST}" "${rendered}"; then
    write_record "$(synthesize_record error "could not render ${MANIFEST}")" null null "${DRIVER_NOTE}" >/dev/null
    CELL_STATUS="error"
    return 0
  fi

  # Everything downstream — the delete-before-create, the wait, the log harvest,
  # collect-phases — addresses the Job by JOB_NAME. If the manifest declares a
  # different metadata.name the delete is a no-op, the wait reports "not found"
  # and the cell is scored as an error for a reason that has nothing to do with
  # the hardware. Compare the two now, while it is still one line to fix.
  local manifest_name
  manifest_name="$(sed -n '/^kind:[[:space:]]*Job/,/^spec:/p' "${rendered}" \
                   | sed -n 's/^[[:space:]]\{1,\}name:[[:space:]]*\([A-Za-z0-9._-]\{1,\}\).*/\1/p' | head -1)"
  if [ -n "${manifest_name}" ] && [ "${manifest_name}" != "${JOB_NAME}" ]; then
    err "manifest ${MANIFEST} creates job/${manifest_name} but this driver drives job/${JOB_NAME}"
    err "  me344_job_name (manifests/env.sh) and the manifest's metadata.name must agree."
    write_record "$(synthesize_record error "Job name mismatch: manifest says ${manifest_name}, driver expects ${JOB_NAME}")" \
      null null "${DRIVER_NOTE}" >/dev/null
    CELL_STATUS="error"
    return 0
  fi

  # `kubectl apply` patches, which the create/delete-only RBAC persona rejects.
  # Delete first, then create. Background propagation only: foreground cascade
  # sets a finalizer, which needs an update the persona also lacks.
  kube delete job "${JOB_NAME}" --ignore-not-found --wait=true --timeout=180s >/dev/null 2>&1 || true
  # Drain the old attempt's pods so `kubectl logs` later cannot pick up a stale
  # one. Advisory: if they linger we submit anyway rather than stall the sweep.
  local waited=0
  while [ "${waited}" -lt "${POD_DRAIN_MAX_S}" ]; do
    if [ -z "$(latest_pod_for_job "${JOB_NAME}")" ]; then break; fi
    sleep 5; waited=$((waited + 5))
  done
  if [ "${waited}" -ge "${POD_DRAIN_MAX_S}" ]; then
    warn "pods of a previous ${JOB_NAME} were still present after ${POD_DRAIN_MAX_S}s — submitting anyway"
  fi

  local tries=0 created=0
  while [ "${tries}" -lt 4 ]; do
    if kube create -f "${rendered}" >/dev/null 2>"${errfile}"; then created=1; break; fi
    tries=$((tries + 1))
    if grep -q "AlreadyExists" "${errfile}" 2>/dev/null; then
      warn "job/${JOB_NAME} still terminating, retrying create (${tries}/4)"
      sleep 15
    else
      break
    fi
  done
  if [ "${created}" != "1" ]; then
    local msg; msg="$(head -c 400 "${errfile}" 2>/dev/null || true)"
    err "could not create job/${JOB_NAME}: ${msg}"
    write_record "$(synthesize_record error "kubectl create failed: ${msg}")" null null "${DRIVER_NOTE}" >/dev/null
    CELL_STATUS="error"
    return 0
  fi

  local t0 t1 elapsed waitrc
  t0="${SUBMIT_EPOCH}"
  info "submitted job/${JOB_NAME} (run_id=${RUN_ID})"
  set +e
  "${SCRIPT_DIR}/wait-job.sh" "${JOB_NAME}" \
    --context "${KCTX}" --namespace "${KNS}" \
    --container "${CONTAINER_NAME}" --timeout-min "${TIMEOUT_MIN}"
  waitrc=$?
  set -e
  t1="$(date -u +%s)"
  elapsed=$(( t1 - t0 ))

  # Logs are the only output channel on the GH200 namespace, so always harvest
  # them — before anything else deletes the pod.
  local pod; pod="$(latest_pod_for_job "${JOB_NAME}")"
  : > "${logfile}"
  if [ -n "${pod}" ]; then
    kube logs "${pod}" -c "${CONTAINER_NAME}" > "${logfile}" 2>/dev/null \
      || kube logs "${pod}" > "${logfile}" 2>/dev/null \
      || warn "no logs retrievable for pod/${pod}"
  fi

  "${SCRIPT_DIR}/collect-phases.sh" "${JOB_NAME}" \
    --context "${KCTX}" --namespace "${KNS}" \
    --container "${CONTAINER_NAME}" --out "${phasefile}" >/dev/null 2>&1 || true
  local ph="null"
  if [ -f "${phasefile}" ]; then ph="$(cat "${phasefile}")"; fi

  # An OOMKill shows up here and nowhere else: the kernel kills the process
  # before it can log anything. Joined with US (0x1f) rather than a tab, because
  # a tab is IFS whitespace and an empty reason would shift the exit code into
  # its place — silently turning exit 137 into "no OOM detected".
  local reason="" code=""
  if [ -n "${pod}" ]; then
    local rec
    rec="$(kube get pod "${pod}" -o json 2>/dev/null | jq -r --arg c "${CONTAINER_NAME}" '
      (.status.containerStatuses // [] | map(select(.name == $c)) | first) as $cs
      | [ ($cs.state.terminated.reason // $cs.lastState.terminated.reason // ""),
          ($cs.state.terminated.exitCode // $cs.lastState.terminated.exitCode // "") ]
      | map(tostring) | join("\u001f")' 2>/dev/null || true)"
    IFS=$'\037' read -r reason code <<<"${rec}" || true
  fi

  harvest_and_write "${logfile}" "${ph}" "${elapsed}" "${reason}" "${code}" "${waitrc}"
  return 0
}

run_cell_local() {
  local logfile="${LOG_DIR}/${RUN_ID}.log"
  local t0 t1 elapsed rc

  # The cell is already in the environment (resolve_cell exported BATCH_SIZE,
  # BATCH_SIZE_OVERRIDE, SEQ_LEN, RUN_ID, RUN_VARIANT and the HW_* block), and
  # manifests/cpu-run.sh re-resolves from exactly those names.
  info "running locally: ${CPU_CMD[*]}"
  t0="$(date -u +%s)"
  export SUBMIT_EPOCH="${t0}"
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=60 "${TIMEOUT_MIN}m" \
      "${CPU_CMD[@]}" >"${logfile}" 2>&1
  else
    "${CPU_CMD[@]}" >"${logfile}" 2>&1
  fi
  rc=$?
  set -e
  t1="$(date -u +%s)"
  elapsed=$(( t1 - t0 ))

  # There is no cluster scheduler in front of a local run, so submit_to_running_s
  # is a measured 0 rather than missing data. image_pull_s is 0 in the same sense:
  # it is the phase between the pod being scheduled and the container starting,
  # and there is no pod here. manifests/cpu-run.sh does pull the image and stage
  # the checkpoint from GCS before it starts the container, and that time is real
  # — it lands in phases.other_s (total_wall_s here is the driver's stopwatch
  # around the whole runner, not just the training process) rather than being
  # attributed to a Kubernetes phase that did not happen. The note says so, so no
  # reader has to guess what other_s contains on this row.
  local ph
  ph="$(jq -n '{submit_to_running_s: 0, image_pull_s: 0,
                detail: {note: "local process on the class node: no scheduler admission and no pod lifecycle. The image pull and the GCS->node checkpoint staging performed by manifests/cpu-run.sh happen before the container starts and are included in phases.other_s, not in image_pull_s."}}')"

  local waitrc=0
  if [ "${rc}" = "124" ] || [ "${rc}" = "137" ]; then waitrc=2; fi
  harvest_and_write "${logfile}" "${ph}" "${elapsed}" "" "${rc}" "${waitrc}"
  return 0
}

# harvest_and_write <log> <phases> <elapsed> <k8s reason> <exit code> <wait rc>
harvest_and_write() {
  local logfile="$1" ph="$2" elapsed="$3" reason="$4" code="$5" waitrc="$6"
  local record="" status=""

  record="$(extract_metrics_block "${logfile}")"

  # Fallback: the run may have written the same JSON to the bucket. Useful when
  # a log was rotated or truncated mid-block.
  if ! is_valid_record "${record}" && command -v gcloud >/dev/null 2>&1; then
    local from_gcs
    from_gcs="$(gcloud storage cat "${RESULTS_GCS_PREFIX}/${RUN_ID}.json" 2>/dev/null || true)"
    if is_valid_record "${from_gcs}"; then
      record="${from_gcs}"
      info "recovered metrics from ${RESULTS_GCS_PREFIX}/${RUN_ID}.json"
    fi
  fi

  if is_valid_record "${record}"; then
    status="$(printf '%s' "${record}" | jq -r '.status // "ok"')"
    if [ "${waitrc}" != "0" ] && [ "${status}" = "ok" ]; then
      # The container claimed success but the Job did not reach a terminal
      # success. Trust the cluster, and say why.
      status="$(classify_failure "${logfile}" "${reason}" "${code}" "${waitrc}")"
      record="$(printf '%s' "${record}" | jq --arg s "${status}" \
        --arg e "$(error_message "${logfile}" "job did not complete successfully (wait rc=${waitrc})")" \
        '.status = $s | .error = $e')"
      warn "container reported ok but the Job did not succeed — recorded as ${status}"
    fi
  else
    status="$(classify_failure "${logfile}" "${reason}" "${code}" "${waitrc}")"
    local msg; msg="$(error_message "${logfile}" "no metrics block found in the log and the run did not complete")"
    record="$(synthesize_record "${status}" "${msg}")"
    warn "no ${METRICS_BEGIN} block in the log — synthesized a ${status} record"
  fi

  local out; out="$(write_record "${record}" "${ph}" "${elapsed}" "${DRIVER_NOTE}")"
  CELL_STATUS="${status}"
  info "${RUN_ID}: status=${status} wall=${elapsed}s -> ${out}"
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
N_RUN=0; N_SKIP=0; N_OK=0; N_OOM=0; N_ERR=0
SUMMARY=()
OOM_SEEN_SEQ=""

for cell in "${CELLS[@]}"; do
  # resolve_cell sets BATCH_SIZE, SEQ_LEN, RUN_ID and the HW_* block through
  # me344_resolve, so the run_id here is the same string the container reports.
  resolve_cell "${BACKEND}" "${cell%%:*}" "${cell##*:}"
  export MAX_STEPS LORA_RANK LORA_ALPHA DTYPE REMAT KUEUE_QUEUE
  CELL_STATUS="unknown"

  if [ -f "${RESULTS_DIR}/${RUN_ID}.json" ] && [ "${FORCE}" = "0" ]; then
    existing="$(jq -r '.status // "?"' "${RESULTS_DIR}/${RUN_ID}.json" 2>/dev/null || echo '?')"
    skip "${RUN_ID} — results/${RUN_ID}.json exists (status=${existing}); --force to re-run"
    N_SKIP=$((N_SKIP + 1))
    SUMMARY+=("${RUN_ID}	${existing}	skipped")
    continue
  fi

  if [ "${ABORT_AXIS_ON_OOM}" = "1" ] && [ -n "${OOM_SEEN_SEQ}" ] && [ "${SEQ_LEN}" = "${OOM_SEEN_SEQ}" ]; then
    skip "${RUN_ID} — a smaller batch already OOMed at seq=${SEQ_LEN} (--abort-axis-on-oom)"
    N_SKIP=$((N_SKIP + 1))
    SUMMARY+=("${RUN_ID}	not-run	skipped-after-oom")
    continue
  fi

  say "cell ${RUN_ID}  (bs=${BATCH_SIZE} seq=${SEQ_LEN} steps=${MAX_STEPS})"
  N_RUN=$((N_RUN + 1))

  if [ "${BACKEND}" = "cpu" ]; then
    run_cell_local || true
  else
    run_cell_k8s || true
  fi

  case "${CELL_STATUS}" in
    ok)              N_OK=$((N_OK + 1)) ;;
    oom)             N_OOM=$((N_OOM + 1)); OOM_SEEN_SEQ="${SEQ_LEN}" ;;
    error|timeout|*) N_ERR=$((N_ERR + 1)) ;;
  esac
  SUMMARY+=("${RUN_ID}	${CELL_STATUS}	ran")
done

say "sweep complete: ${N_RUN} run (${N_OK} ok, ${N_OOM} oom, ${N_ERR} error/timeout), ${N_SKIP} skipped"
for line in ${SUMMARY[@]+"${SUMMARY[@]}"}; do
  printf '  %s\n' "${line}" >&2
done

if [ "${FAIL_ON_ERROR}" = "1" ] && [ "${N_ERR}" -gt 0 ]; then exit 1; fi
exit 0
