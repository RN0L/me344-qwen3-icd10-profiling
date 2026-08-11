#!/usr/bin/env bash
# =============================================================================
# scripts/collect-phases.sh — recover the Kubernetes-side wall clock that the
# training container cannot see, because it does not exist yet while it happens.
#
#   scripts/collect-phases.sh <job-name> [options] > phases.json
#
# Options:
#   --backend cpu|gpu|tpu   context+namespace from configs (default: tpu)
#   --context CTX / --namespace NS / --container NAME
#   --pod NAME              pin a specific pod (default: newest pod of the Job)
#   --out FILE              write here instead of stdout
#
# Emits the two contract fields at the top level, plus a `detail` block holding
# every raw timestamp so a reviewer can re-derive them:
#
#   {"submit_to_running_s": 180.0, "image_pull_s": 45.0, "detail": {...}}
#
# WHICH TIMESTAMP IS WHICH
# ------------------------
#   submit          job.metadata.creationTimestamp
#                     the moment kubectl create returned. On the TPU cluster the
#                     Job exists here but is not yet admitted.
#   admitted        workload.status.conditions[type=Admitted].lastTransitionTime
#                     Kueue let it through. Everything before this is queueing
#                     behind the rest of the class, not our job being slow.
#   scheduled       pod.status.conditions[type=PodScheduled].lastTransitionTime
#                     a node was picked. On TPU this waits for a v5e node to boot.
#   pull_start      first "Pulling" event on the pod
#   pull_end        matching "Pulled" event. GKE puts the measured duration in
#                     the event message ("...in 45.123s"), which is finer than the
#                     1-second resolution of the event timestamps, so it is
#                     preferred and the timestamp difference is the fallback.
#                     An image already on the node emits "already present on
#                     machine" and no "Pulling" — that is a real 0, not a gap.
#   running         pod.status.containerStatuses[name=$CONTAINER].state.running.startedAt
#                     the training process actually started. This is the origin
#                     of the container's own internal clock.
#
# THE ONE SUBTRACTION, AND WHY
# ----------------------------
# The image pull happens *inside* the submit->running interval, so reporting both
# raw would double-count. docs/metrics-contract.md requires phases.* to sum to
# total_wall_s (its worked example sums exactly), so the phases must be disjoint.
# Therefore:
#
#     submit_to_running_s = (running - submit) - image_pull_s
#     image_pull_s        = pull_end - pull_start
#
# i.e. submit_to_running_s is the queue + schedule + node-boot + container-create
# portion with the pull carved out. The undivided interval is kept in
# detail.submit_to_running_inclusive_s.
#
# Any value that cannot be established is emitted as null rather than guessed.
# Events have a ~1h TTL on GKE, so run this promptly after the Job finishes; a
# late run legitimately yields null image_pull_s. Every kubectl call is
# best-effort because the GH200 namespace may deny events or describe.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
. "${SCRIPT_DIR}/common.sh"

JOB=""
CP_BACKEND="${BACKEND:-tpu}"
OPT_CTX=""; OPT_NS=""; OPT_POD=""; OUT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --backend)      CP_BACKEND="${2:?--backend needs a value}"; shift 2 ;;
    --context)      OPT_CTX="${2:?--context needs a value}"; shift 2 ;;
    --namespace|-n) OPT_NS="${2:?--namespace needs a value}"; shift 2 ;;
    --container|-c) CONTAINER_NAME="${2:?--container needs a value}"; shift 2 ;;
    --pod)          OPT_POD="${2:?--pod needs a value}"; shift 2 ;;
    --out|-o)       OUT="${2:?--out needs a value}"; shift 2 ;;
    -h|--help)      sed -n '2,60p' "${BASH_SOURCE[0]}" >&2; exit 0 ;;
    -*)             err "unknown option: $1"; exit 3 ;;
    *)              if [ -z "$JOB" ]; then JOB="$1"; else err "unexpected argument: $1"; exit 3; fi; shift ;;
  esac
done

: "${JOB:?usage: collect-phases.sh <job-name> [--backend tpu|gpu] [--out FILE]}"
need kubectl jq

select_backend "${CP_BACKEND}"
if [ -n "$OPT_CTX" ]; then KCTX="$OPT_CTX"; fi
if [ -n "$OPT_NS"  ]; then KNS="$OPT_NS"; fi
: "${KCTX:?collect-phases.sh needs a kube context — backend '${CP_BACKEND}' has no cluster}"

# --- gather, tolerating denials --------------------------------------------
ERRORS=()
JOB_JSON="$(kube get job "${JOB}" -o json 2>/dev/null || true)"
if [ -z "${JOB_JSON}" ]; then
  ERRORS+=("job/${JOB} not readable in ${KNS}@${KCTX}")
  JOB_JSON='{}'
fi

if [ -n "${OPT_POD}" ]; then
  POD="${OPT_POD}"
else
  POD="$(latest_pod_for_job "${JOB}")"
fi

POD_JSON='{}'
if [ -n "${POD}" ]; then
  POD_JSON="$(kube get pod "${POD}" -o json 2>/dev/null || true)"
  if [ -z "${POD_JSON}" ]; then
    ERRORS+=("pod/${POD} not readable")
    POD_JSON='{}'
  fi
else
  ERRORS+=("no pod found for job/${JOB} (never scheduled, or pods already garbage collected)")
fi

POD_UID="$(printf '%s' "${POD_JSON}" | jq -r '.metadata.uid // empty')"
EVENTS_JSON='{"items":[]}'
if [ -n "${POD_UID}" ]; then
  # Select by UID: a Job that retried leaves same-named-prefix pods behind and
  # name selection would mix their events together.
  EVENTS_JSON="$(kube get events --field-selector "involvedObject.uid=${POD_UID}" -o json 2>/dev/null || true)"
  if [ -z "${EVENTS_JSON}" ]; then
    ERRORS+=("events not readable (RBAC denial or 1h TTL expiry) — image_pull_s will be null")
    EVENTS_JSON='{"items":[]}'
  fi
fi

# Kueue is only in the path on the TPU cluster. Its absence is information, not
# an error, so it is recorded rather than warned about.
WORKLOAD_JSON='{"items":[]}'
if kube get workloads -o json >/dev/null 2>&1; then
  WORKLOAD_JSON="$(kube get workloads -o json 2>/dev/null || echo '{"items":[]}')"
fi

ERRORS_JSON="$(printf '%s\n' "${ERRORS[@]+"${ERRORS[@]}"}" | jq -R . | jq -s 'map(select(. != ""))')"

# --- derive ----------------------------------------------------------------
# All date arithmetic happens in jq (fromdateiso8601) rather than date(1), which
# needs different flags on the Rocky node and on macOS.
JQ_ERR="$(mktemp "${TMPDIR:-/tmp}/collect-phases.XXXXXX")"
trap 'rm -f "${JQ_ERR}"' EXIT

RESULT="$(jq -n \
  --argjson job "${JOB_JSON}" \
  --argjson pod "${POD_JSON}" \
  --argjson events "${EVENTS_JSON}" \
  --argjson workloads "${WORKLOAD_JSON}" \
  --argjson errors "${ERRORS_JSON}" \
  --arg jobname "${JOB}" \
  --arg container "${CONTAINER_NAME}" \
  --arg sidecar "${GCSFUSE_SIDECAR_NAME}" '

  # RFC3339 -> epoch seconds. Kubernetes emits second precision on metadata and
  # event timestamps, microseconds on events.k8s.io eventTime; strip the fraction.
  def ts: if . == null or . == "" then null
          else (sub("\\.[0-9]+Z$"; "Z") | fromdateiso8601) end;
  def secs($a; $b): if $a == null or $b == null then null else ($b - $a) end;

  # "Successfully pulled image "x" in 45.123s (46.2s including waiting)"
  def pull_seconds:
    if . == null then null
    else ([ capture("in (?:(?<h>[0-9]+)h)?(?:(?<m>[0-9]+)m)?(?<s>[0-9.]+)s") ] | first) as $c
      | if $c == null then null
        else ((($c.h // "0") | tonumber) * 3600
            + (($c.m // "0") | tonumber) * 60
            + ($c.s | tonumber))
        end
    end;

  # core/v1 Events carry firstTimestamp; events.k8s.io ones only eventTime.
  def when: (.firstTimestamp // .eventTime // .lastTimestamp // .metadata.creationTimestamp);

  ($events.items // []) as $ev
  | ($ev | map(select(.reason == "Pulling"))  | sort_by(when) | first) as $pulling
  | ($ev | map(select(.reason == "Pulled"))   | sort_by(when) | first) as $pulled
  | ($ev | map(select(.reason == "Scheduled"))| sort_by(when) | first) as $sched

  | ($job.metadata.creationTimestamp | ts)                as $t_submit
  | ( ($pod.status.containerStatuses // [])
      | map(select(.name == $container)) | first
      // ( ($pod.status.containerStatuses // [])
           | map(select(.name != $sidecar)) | first )
      // (($pod.status.containerStatuses // []) | first) )                as $cs
  | ($cs.state.running.startedAt // $cs.state.terminated.startedAt | ts)  as $t_running
  | ($pod.metadata.creationTimestamp | ts)                               as $t_podcreate
  | ( ($pod.status.conditions // []) | map(select(.type == "PodScheduled" and .status == "True"))
      | first | .lastTransitionTime | ts )                               as $t_sched_cond
  | ( $sched | when | ts )                                               as $t_sched_ev
  | ( $t_sched_cond // $t_sched_ev )                                     as $t_scheduled
  | ( $pulling | when | ts )                                             as $t_pull_start
  | ( $pulled  | when | ts )                                             as $t_pull_end
  | ( $job.status.completionTime | ts )                                  as $t_complete

  # Kueue admission for this Job, if Kueue is in the path at all.
  | ( ($workloads.items // [])
      | map(select((.metadata.ownerReferences // []) | any(.name == $jobname)))
      | first )                                                          as $wl
  | ( ($wl.status.conditions // []) | map(select(.type == "Admitted" and .status == "True"))
      | last | .lastTransitionTime | ts )                                as $t_admitted

  # Prefer the duration GKE measured and printed; fall back to event timestamps.
  | ( ($pulled.message | pull_seconds) // secs($t_pull_start; $t_pull_end) ) as $pull_measured
  | ( ($ev | map(select(.reason == "Pulled" and (.message // "" | test("already present"))))
            | length) > 0 )                                              as $already_present
  | ( if $pull_measured != null then $pull_measured
      elif $already_present then 0
      else null end )                                                    as $image_pull_s

  | secs($t_submit; $t_running)                                          as $inclusive
  # The pull is carved out of submit->running so the phases stay disjoint. It can
  # legitimately exceed the interval: the "in 45.1s" GKE prints is measured by the
  # kubelet, the interval is derived from second-resolution API timestamps, and a
  # retried pod can attribute a previous pull attempt here. Subtracting anyway
  # would emit a negative phase, so in that case the undivided interval is kept
  # and the disagreement is recorded rather than hidden.
  | ( if $inclusive == null or $image_pull_s == null then null
      elif ($inclusive - $image_pull_s) < 0 then true
      else false end )                                                   as $pull_overruns
  | ( if $inclusive == null then null
      elif $image_pull_s == null then $inclusive
      elif $pull_overruns then $inclusive
      else ($inclusive - $image_pull_s) end )                            as $submit_to_running_s

  | {
      submit_to_running_s: $submit_to_running_s,
      image_pull_s: $image_pull_s,
      detail: {
        job: $jobname,
        pod: ($pod.metadata.name // null),
        node: ($pod.spec.nodeName // null),
        image: ($pod.spec.containers // [] | map(select(.name == $container)) | first | .image
                // ($pod.spec.containers // [] | first | .image) // null),
        submit_to_running_inclusive_s: $inclusive,
        image_pull_exceeded_submit_to_running: $pull_overruns,
        queue_wait_s: secs($t_submit; $t_admitted),
        admitted_to_scheduled_s: secs(($t_admitted // $t_submit); $t_scheduled),
        scheduled_to_running_s: secs($t_scheduled; $t_running),
        job_total_wall_s: secs($t_submit; $t_complete),
        image_already_present: $already_present,
        image_pull_source: (if ($pulled.message | pull_seconds) != null then "event message"
                            elif secs($t_pull_start; $t_pull_end) != null then "event timestamps (1s resolution)"
                            elif $already_present then "image already on node"
                            else "unavailable" end),
        pulled_message: ($pulled.message // null),
        timestamps: {
          submitted:  ($job.metadata.creationTimestamp // null),
          admitted:   (($wl.status.conditions // []) | map(select(.type == "Admitted" and .status == "True")) | last | .lastTransitionTime // null),
          pod_created:($pod.metadata.creationTimestamp // null),
          scheduled:  (($pod.status.conditions // []) | map(select(.type == "PodScheduled" and .status == "True")) | first | .lastTransitionTime // ($sched | when) // null),
          pull_start: ($pulling | when // null),
          pull_end:   ($pulled  | when // null),
          running:    ($cs.state.running.startedAt // $cs.state.terminated.startedAt // null),
          completed:  ($job.status.completionTime // null)
        },
        submit_to_pod_created_s: secs($t_submit; $t_podcreate),
        kueue_workload: ($wl.metadata.name // null),
        kueue_queue: ($wl.spec.queueName // null),
        container: $container,
        note: (if $pull_overruns then
                 "image_pull_s (\($image_pull_s)s, kubelet-measured) exceeds the submit->running interval (\($inclusive)s, derived from second-resolution API timestamps), so submit_to_running_s is reported UNDIVIDED and the two phases overlap for this record."
               else
                 "submit_to_running_s excludes image_pull_s so phases stay disjoint and additive; the undivided interval is submit_to_running_inclusive_s"
               end),
        errors: $errors
      }
    }' 2>"${JQ_ERR}" || true)"

if [ -z "${RESULT}" ]; then
  # Never fail the caller: an unmeasurable startup is a null, not a dead sweep.
  err "could not derive phases for job/${JOB} — emitting nulls"
  sed 's/^/  jq: /' "${JQ_ERR}" >&2 || true
  RESULT="$(jq -n --arg j "${JOB}" --argjson e "${ERRORS_JSON}" \
    '{submit_to_running_s: null, image_pull_s: null,
      detail: {job: $j, errors: ($e + ["jq derivation failed"])}}')"
fi

if [ -n "${OUT}" ]; then
  printf '%s\n' "${RESULT}" > "${OUT}"
  info "phases for job/${JOB} -> ${OUT}"
else
  printf '%s\n' "${RESULT}"
fi
