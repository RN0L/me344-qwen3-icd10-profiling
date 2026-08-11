#!/usr/bin/env bash
# =============================================================================
# scripts/wait-job.sh — poll a Kubernetes Job to a terminal state.
#
#   scripts/wait-job.sh <job-name> [options]
#
# Options:
#   --backend cpu|gpu|tpu   pick context+namespace from configs (default: tpu)
#   --context CTX           override the kube context
#   --namespace NS          override the namespace
#   --container NAME        training container name (default: $CONTAINER_NAME)
#   --timeout-min N         give up after N minutes (default: $TIMEOUT_MIN)
#   --poll-s N              poll interval (default: $POLL_S)
#   --heartbeat-s N         print a liveness line this often (default: $HEARTBEAT_S)
#   --tail N                log lines to dump on failure (default: 80)
#   --tail-on-success       also dump the tail when the Job succeeds
#   --stuck-abort-s N       abort if the pod sits in an unrecoverable image /
#                           config error this long (default: 600, 0 disables)
#
# Exit codes: 0 succeeded | 1 failed | 2 timed out | 3 usage or precondition
#
# WHY IT POLLS AND DOES NOT FOLLOW LOGS
# -------------------------------------
# `kubectl logs -f job/<job>` exits 0 with empty output when the pod is still
# Pending, so a wait built on it reports success for a job that never ran. That
# has cost this project time twice. The wait condition here is the Job's own
# .status.succeeded / .status.failed and its Complete/Failed conditions; logs
# are read for diagnostics only, never as a signal.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
. "${SCRIPT_DIR}/common.sh"

JOB=""
WJ_BACKEND="${BACKEND:-tpu}"
OPT_CTX=""; OPT_NS=""
TAIL_N=80
TAIL_ON_SUCCESS=0
STUCK_ABORT_S=600

while [ $# -gt 0 ]; do
  case "$1" in
    --backend)          WJ_BACKEND="${2:?--backend needs a value}"; shift 2 ;;
    --context)          OPT_CTX="${2:?--context needs a value}"; shift 2 ;;
    --namespace|-n)     OPT_NS="${2:?--namespace needs a value}"; shift 2 ;;
    --container|-c)     CONTAINER_NAME="${2:?--container needs a value}"; shift 2 ;;
    --timeout-min)      TIMEOUT_MIN="${2:?--timeout-min needs a value}"; shift 2 ;;
    --poll-s)           POLL_S="${2:?--poll-s needs a value}"; shift 2 ;;
    --heartbeat-s)      HEARTBEAT_S="${2:?--heartbeat-s needs a value}"; shift 2 ;;
    --tail)             TAIL_N="${2:?--tail needs a value}"; shift 2 ;;
    --tail-on-success)  TAIL_ON_SUCCESS=1; shift ;;
    --stuck-abort-s)    STUCK_ABORT_S="${2:?--stuck-abort-s needs a value}"; shift 2 ;;
    -h|--help)          sed -n '2,30p' "${BASH_SOURCE[0]}" >&2; exit 0 ;;
    -*)                 err "unknown option: $1"; exit 3 ;;
    *)                  [ -z "$JOB" ] && JOB="$1" || { err "unexpected argument: $1"; exit 3; }; shift ;;
  esac
done

: "${JOB:?usage: wait-job.sh <job-name> [--backend tpu|gpu|cpu] [--timeout-min N]}"
need kubectl jq

select_backend "${WJ_BACKEND}"
if [ -n "$OPT_CTX" ]; then KCTX="$OPT_CTX"; fi
if [ -n "$OPT_NS"  ]; then KNS="$OPT_NS"; fi
: "${KCTX:?wait-job.sh needs a kube context — backend '${WJ_BACKEND}' has no cluster}"

# ---------------------------------------------------------------------------
# Diagnostics. Every command here is best-effort: the GH200 namespace runs a
# tight RBAC persona where describe or events may be denied, and a denial must
# degrade the dump rather than abort it.
# ---------------------------------------------------------------------------
section() { printf '\n----- %s -----\n' "$*" >&2; }

dump_logs() {
  local pod="$1" restarts="${2:-0}"
  if [ -z "$pod" ]; then
    warn "no pod was ever created for job/${JOB} — nothing to log"
    return 0
  fi
  section "last ${TAIL_N} log lines: ${pod} [${CONTAINER_NAME}]"
  if ! kube logs "$pod" -c "${CONTAINER_NAME}" --tail="${TAIL_N}" >&2 2>/dev/null; then
    kube logs "$pod" --tail="${TAIL_N}" >&2 2>/dev/null || warn "logs unavailable for ${pod}"
  fi
  if [ "${restarts:-0}" != "0" ] && [ -n "${restarts}" ]; then
    section "previous attempt (restartCount=${restarts})"
    kube logs "$pod" -c "${CONTAINER_NAME}" --previous --tail="${TAIL_N}" >&2 2>/dev/null \
      || warn "previous-container logs unavailable"
  fi
}

dump_diagnostics() {
  local pod="$1" uid="$2" restarts="${3:-0}"
  dump_logs "$pod" "$restarts"

  if [ -n "$pod" ]; then
    section "container termination state"
    kube get pod "$pod" -o json 2>/dev/null | jq -r '
      .status.containerStatuses // [] | .[] |
      "  \(.name): restarts=\(.restartCount)" +
      (if .state.terminated then
         " terminated reason=\(.state.terminated.reason) exit=\(.state.terminated.exitCode)"
       elif .state.waiting then
         " waiting reason=\(.state.waiting.reason) msg=\(.state.waiting.message // "")"
       else " running" end)' >&2 || warn "container state unavailable"

    section "kubectl describe pod ${pod}"
    kube describe pod "$pod" >&2 2>/dev/null || warn "describe pod denied or unavailable"

    section "pod events"
    # Events carry the image-pull and scheduling failures that never reach the
    # container log. Select by UID so a recreated pod of the same name cannot
    # contribute stale events.
    if [ -n "$uid" ]; then
      kube get events --field-selector "involvedObject.uid=${uid}" \
        --sort-by=.lastTimestamp >&2 2>/dev/null || warn "events denied or expired"
    else
      kube get events --field-selector "involvedObject.name=${pod}" \
        --sort-by=.lastTimestamp >&2 2>/dev/null || warn "events denied or expired"
    fi
  fi

  section "kubectl describe job ${JOB}"
  kube describe job "${JOB}" >&2 2>/dev/null || warn "describe job denied or unavailable"

  # A Job that never produced a pod on the TPU cluster almost always means Kueue
  # never admitted it — usually a missing kueue.x-k8s.io/queue-name label.
  section "kueue workload"
  if ! kube get workloads -o json 2>/dev/null | jq -r --arg j "${JOB}" '
        [.items[] | select((.metadata.ownerReferences // []) | any(.name == $j))] as $w
        | if ($w | length) == 0 then
            "  no Kueue Workload owns job/\($j) — either Kueue is not in the path, " +
            "or the Job is missing the kueue.x-k8s.io/queue-name label and will stay Pending forever."
          else
            $w[] | "  \(.metadata.name) queue=\(.spec.queueName // "<none>") " +
              "admitted=\((.status.conditions // [] | map(select(.type == "Admitted")) | last | .status) // "no") " +
              "reason=\((.status.conditions // [] | last | .message) // "")"
          end' >&2; then
    warn "workloads API unavailable (expected on the GH200 cluster)"
  fi
  printf '\n' >&2
}

# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------
if ! kube get job "${JOB}" >/dev/null 2>&1; then
  err "job/${JOB} not found in ${KNS} on context ${KCTX}"
  exit 3
fi

START="$(date -u +%s)"
DEADLINE=$(( START + TIMEOUT_MIN * 60 ))
LAST_STATE=""
LAST_HEARTBEAT="${START}"
STUCK_SINCE=""
POD=""; POD_UID=""; RESTARTS=0

info "waiting on job/${JOB} (ns=${KNS}, ctx=${KCTX}, timeout=${TIMEOUT_MIN}m)"

while :; do
  NOW="$(date -u +%s)"
  ELAPSED=$(( NOW - START ))

  JOB_JSON="$(kube get job "${JOB}" -o json 2>/dev/null || true)"
  if [ -z "${JOB_JSON}" ]; then
    warn "job/${JOB} disappeared while waiting (deleted by someone else?)"
    exit 1
  fi

  # Fields are joined with US (0x1f), not a tab: a tab is IFS whitespace, so
  # bash would collapse consecutive tabs and silently shift every field after an
  # empty one — which is exactly what an unscheduled pod produces.
  JOB_REC="$(printf '%s' "${JOB_JSON}" | jq -r '
    [ (.status.succeeded // 0),
      (.status.failed // 0),
      (.spec.backoffLimit // 6),
      ((.status.conditions // [] | map(select(.type == "Complete" and .status == "True")) | length) > 0),
      ((.status.conditions // [] | map(select(.type == "Failed"   and .status == "True")) | length) > 0)
    ] | map(tostring) | join("\u001f")' 2>/dev/null || true)"
  SUCCEEDED=0; FAILED=0; BACKOFF=6; COMPLETE_COND=false; FAILED_COND=false
  IFS=$'\037' read -r SUCCEEDED FAILED BACKOFF COMPLETE_COND FAILED_COND <<<"${JOB_REC}" || true

  POD_REC="$(kube get pods -l "job-name=${JOB}" -o json 2>/dev/null | jq -r --arg c "${CONTAINER_NAME}" '
    ([.items[]] | sort_by(.metadata.creationTimestamp) | last) as $p
    | if $p == null then ["", "", "<nopod>", "-", 0]
      else
        ( ($p.status.containerStatuses // [] | map(select(.name == $c)) | first)
          // ($p.status.containerStatuses // [] | first) ) as $cs
        | [ $p.metadata.name,
            $p.metadata.uid,
            ($p.status.phase // "Unknown"),
            ( $cs.state.waiting.reason // $cs.state.terminated.reason
              // ($p.status.conditions // [] | map(select(.type == "PodScheduled" and .status != "True")) | first | .reason)
              // "-" ),
            ($cs.restartCount // 0) ]
      end | map(tostring) | join("\u001f")' 2>/dev/null || true)"
  POD=""; POD_UID=""; PHASE="<nopod>"; REASON="-"; RESTARTS=0
  IFS=$'\037' read -r POD POD_UID PHASE REASON RESTARTS <<<"${POD_REC}" || true
  if [ -z "${PHASE}" ]; then PHASE="<nopod>"; fi
  if [ -z "${REASON}" ]; then REASON="-"; fi
  if [ -z "${RESTARTS}" ]; then RESTARTS=0; fi

  STATE="${PHASE}/${REASON}"
  if [ "${STATE}" != "${LAST_STATE}" ]; then
    printf '[%s] (+%4ds) %-28s -> %-28s pod=%s\n' \
      "$(_ts)" "${ELAPSED}" "${LAST_STATE:-<start>}" "${STATE}" "${POD:-<none>}" >&2
    LAST_STATE="${STATE}"
    LAST_HEARTBEAT="${NOW}"
    STUCK_SINCE=""
  fi

  # Unrecoverable pull/config states never resolve on their own. Rather than
  # burn the whole timeout on them, abort once they have persisted.
  case "${REASON}" in
    ErrImagePull|ImagePullBackOff|InvalidImageName|CreateContainerConfigError|CreateContainerError)
      if [ "${STUCK_ABORT_S}" -gt 0 ]; then
        if [ -z "${STUCK_SINCE}" ]; then STUCK_SINCE="${NOW}"; fi
        if [ $(( NOW - STUCK_SINCE )) -ge "${STUCK_ABORT_S}" ]; then
          err "pod stuck in ${REASON} for ${STUCK_ABORT_S}s — aborting"
          dump_diagnostics "${POD}" "${POD_UID}" "${RESTARTS}"
          exit 1
        fi
      fi
      ;;
  esac

  if [ "${COMPLETE_COND}" = "true" ] || [ "${SUCCEEDED:-0}" -ge 1 ] 2>/dev/null; then
    info "job/${JOB} SUCCEEDED after ${ELAPSED}s"
    if [ "${TAIL_ON_SUCCESS}" = "1" ]; then dump_logs "${POD}" "${RESTARTS}"; fi
    exit 0
  fi

  if [ "${FAILED_COND}" = "true" ] || { [ "${FAILED:-0}" -gt "${BACKOFF:-6}" ] 2>/dev/null; }; then
    err "job/${JOB} FAILED after ${ELAPSED}s (failed=${FAILED}, backoffLimit=${BACKOFF})"
    dump_diagnostics "${POD}" "${POD_UID}" "${RESTARTS}"
    exit 1
  fi

  if [ "${NOW}" -ge "${DEADLINE}" ]; then
    err "job/${JOB} TIMED OUT after ${TIMEOUT_MIN} min in state ${STATE}"
    dump_diagnostics "${POD}" "${POD_UID}" "${RESTARTS}"
    exit 2
  fi

  if [ $(( NOW - LAST_HEARTBEAT )) -ge "${HEARTBEAT_S}" ]; then
    # `kubectl logs` with an empty pod name reads its usage text as the resource
    # name and errors; guard rather than rely on the redirect swallowing it.
    LINE=""
    if [ -n "${POD}" ]; then
      LINE="$(kube logs "${POD}" -c "${CONTAINER_NAME}" --tail=1 2>/dev/null | tr -d '\r' | tail -1 || true)"
    fi
    printf '[%s] (+%4ds) %-28s %s\n' "$(_ts)" "${ELAPSED}" "${STATE}" "${LINE:-<no log output yet>}" >&2
    LAST_HEARTBEAT="${NOW}"
  fi

  sleep "${POLL_S}"
done
