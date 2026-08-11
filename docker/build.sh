#!/usr/bin/env bash
# =============================================================================================
# ME344 final project — build and push the three benchmark images.
#
#   ssh me344
#   export TEAM=team-lharnold
#   bash ~/final-project/repo/docker/build.sh                 # all three
#   bash ~/final-project/repo/docker/build.sh cuda            # just the GH200 image
#   bash ~/final-project/repo/docker/build.sh cuda tpu        # several, in order
#   PUSH=0 bash ~/final-project/repo/docker/build.sh cpu      # build locally, do not push
#
# RUN THIS ON THE NODE (hpcc-cluster-39), not on a laptop. The node is where podman lives, where
# the gcloud credential helper is already authenticated as lharnold@stanford.edu, where 32 cores
# make the builds bearable, and where the two amd64 images build natively instead of emulated.
#
# What it produces, in us-central1-docker.pkg.dev/soe-hpccenter/tpu-images (which stays in
# us-central1 even though the TPU cluster is us-west4 — that is not a typo):
#
#   qwen3-codiesp-tpu-${TEAM}   linux/amd64   TPU v5e 2x4, x86_64 host
#   qwen3-codiesp-gpu-${TEAM}   linux/arm64   GH200, ARM64 Grace host
#   qwen3-codiesp-cpu-${TEAM}   linux/amd64   32-core x86_64 node
#
# THOSE NAMES ARE NOT FREE. manifests/env.sh derives IMAGE_URI_TPU / IMAGE_URI_GPU /
# IMAGE_URI_CPU as ${REGISTRY}/qwen3-codiesp-<backend>-${TEAM}:${IMAGE_TAG}, and
# scripts/common.sh falls back to exactly the same string. Push under any other repository name
# and every Job sits in ImagePullBackOff — on the TPU cluster that is only discovered after Kueue
# admission and a node boot. The build TARGET is still called `cuda` (it is the Dockerfile
# suffix, and scripts/run-all.sh maps gpu -> cuda when it calls this script), but the REPOSITORY
# it lands in is called `gpu`, because that is the backend name the rest of the repo uses.
#
# Each image is pushed twice: :${TAG} as an immutable handle so a record in results/ can be
# traced back to the exact image that produced it, and :latest for convenience. The default
# namespace on the TPU cluster is shared with the whole class, so ${TEAM} is in every repository
# name — names are the only isolation there is.
# =============================================================================================
set -euo pipefail

# --- configuration ---------------------------------------------------------------------------

# TEAM is required, but the check lives in preflight() rather than here so that `build.sh --help`
# answers instead of erroring.
#
# Note: no apostrophes inside the ${VAR:?message} guards below. Bash parses the message as shell
# words, so a lone quote character turns the rest of the file into one long string literal.
REGISTRY="${REGISTRY:-us-central1-docker.pkg.dev/soe-hpccenter/tpu-images}"
: "${REGISTRY:?REGISTRY resolved empty — unset it to take the default rather than exporting a blank value.}"

TAG="${TAG:-$(date -u +%Y%m%d-%H%M%S)}"
: "${TAG:?TAG resolved empty — leave it unset to get a UTC timestamp.}"

PUSH="${PUSH:-1}"                 # 0 = build only, useful when iterating without network
VERIFY="${VERIFY:-1}"             # 0 = skip the post-build import check on native-arch images
CONTAINER_CLI="${CONTAINER_CLI:-docker}"   # podman-docker provides `docker` as a podman shim
QEMU_IMAGE="${QEMU_IMAGE:-docker.io/tonistiigi/binfmt:qemu-v10.2.3}"
SKIP_QEMU_SETUP="${SKIP_QEMU_SETUP:-0}"

REGISTRY_HOST="${REGISTRY%%/*}"
: "${REGISTRY_HOST:?REGISTRY must look like host/project/repo. Got: ${REGISTRY}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGES_ENV="${SCRIPT_DIR}/images.env"

ALL_BACKENDS="tpu cuda cpu"

# --- helpers ---------------------------------------------------------------------------------

say()  { printf '\n=== %s ===\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

platform_for() {
  case "$1" in
    # TPU v5e hosts and the ME344 node are both x86_64.
    tpu|cpu) printf 'linux/amd64' ;;
    # GH200's host half is ARM64 Grace. An amd64 image here dies with `exec format error`.
    cuda)    printf 'linux/arm64' ;;
    *)       die "unknown backend '$1' — expected one of: ${ALL_BACKENDS}" ;;
  esac
}

# Build target -> the backend name the rest of the repo uses. The GH200 image is built from
# Dockerfile.cuda but every consumer (manifests/env.sh IMAGE_URI_GPU, scripts/common.sh,
# scripts/run-all.sh) calls that backend `gpu`.
backend_for() {
  case "$1" in
    cuda) printf 'gpu' ;;
    *)    printf '%s' "$1" ;;
  esac
}

# The repository name manifests/env.sh and scripts/common.sh both construct independently. If
# this ever stops matching them the Jobs fail with ImagePullBackOff, so keep the three in step.
repository_for() { printf 'qwen3-codiesp-%s-%s' "$(backend_for "$1")" "${TEAM}"; }

host_platform() {
  case "$(uname -m)" in
    x86_64)          printf 'linux/amd64' ;;
    aarch64|arm64)   printf 'linux/arm64' ;;
    *)               printf 'linux/unknown' ;;
  esac
}

# --- preflight -------------------------------------------------------------------------------

preflight() {
  : "${TEAM:?TEAM is not set. Export TEAM=team-lharnold first. The class shares the default namespace on the TPU cluster, so the team suffix is the only thing keeping these images apart from every other team.}"
  say "Preflight"
  note "repo root      ${REPO_ROOT}"
  note "registry       ${REGISTRY}"
  note "team           ${TEAM}"
  note "immutable tag  ${TAG}"
  note "build host     $(uname -m) ($(host_platform))"

  # The Dockerfiles COPY src/ — an empty src/ fails the build deep inside the COPY with a message
  # that reads like a Docker bug rather than "your teammate has not pushed the trainer yet".
  [ -d "${REPO_ROOT}/src" ] || die "${REPO_ROOT}/src does not exist. The images COPY src/ in; there is nothing to build yet."
  if [ -z "$(ls -A "${REPO_ROOT}/src" 2>/dev/null)" ]; then
    die "${REPO_ROOT}/src is empty. The images COPY src/ in, and COPYing an empty directory fails the build. Wait for the training code to land before building."
  fi
  note "src/           $(find "${REPO_ROOT}/src" -type f | wc -l | tr -d ' ') file(s)"

  # The trainer the images actually execute, and the env->argv shim the manifests invoke it
  # through. A missing shim would only surface as `python3: can't open file /app/train_tpu.py`
  # after a Kueue admission and a node boot.
  [ -f "${REPO_ROOT}/src/train_qwen3_icd.py" ] \
    || die "${REPO_ROOT}/src/train_qwen3_icd.py is missing — that is the trainer /app/train_<backend>.py execs."
  [ -f "${SCRIPT_DIR}/entrypoint.py" ] \
    || die "${SCRIPT_DIR}/entrypoint.py is missing — each image installs it as /app/train_<backend>.py, which is what manifests/env.sh names in TPU_ENTRYPOINT / GPU_ENTRYPOINT / CPU_ENTRYPOINT."

  # podman-docker gives us the `docker` shim; gettext gives envsubst, which the manifest step
  # downstream of this script uses to expand ${TEAM} into the Job YAML.
  #
  # Installing is attempted ONLY for what is actually missing, and only where dnf exists. An
  # unconditional `sudo dnf install` makes this script unusable anywhere but the node — it needs
  # sudo, a package manager and a network, none of which a laptop dry-run has — and under
  # `set -e` its failure aborts the build before a single useful message is printed.
  say "Toolchain"
  local missing=""
  command -v "${CONTAINER_CLI}" >/dev/null || missing="${missing} podman-docker"
  command -v envsubst >/dev/null || missing="${missing} gettext"
  if [ -n "${missing}" ]; then
    if command -v dnf >/dev/null 2>&1; then
      note "installing:${missing}"
      sudo dnf -y install ${missing} >/dev/null || note "dnf install failed — checking anyway"
    else
      note "no dnf here; expecting${missing} to be installed already"
    fi
  fi
  command -v "${CONTAINER_CLI}" >/dev/null \
    || die "'${CONTAINER_CLI}' not on PATH. On the node: sudo dnf -y install podman-docker, or set CONTAINER_CLI=podman to bypass the shim."
  command -v envsubst >/dev/null \
    || die "envsubst not on PATH. On the node: sudo dnf -y install gettext. The manifests are envsubst templates, so this is required downstream even though this script does not use it."
  note "$("${CONTAINER_CLI}" --version)"
  note "$(envsubst --version | head -1)"

  # Artifact Registry auth. This writes a gcloud credential helper entry into ~/.docker/config.json,
  # which rootless podman reads as a fallback to its own auth.json. Only needed when pushing —
  # PUSH=0 exists precisely so the images can be built with no network and no credentials.
  if [ "${PUSH}" = "1" ]; then
    say "Authenticating against ${REGISTRY_HOST}"
    gcloud auth configure-docker "${REGISTRY_HOST}" --quiet
  else
    note "PUSH=0 — skipping gcloud auth configure-docker"
  fi
}

# --- cross-architecture support ---------------------------------------------------------------

# The GH200 image is linux/arm64 and the build node is x86_64, so the aarch64 binaries in every
# RUN layer have to be interpreted. binfmt_misc + qemu-user does that at the kernel level. The
# handler registration is host-wide and survives until reboot, so this is usually a no-op.
#
# qemu-user-static is not in Rocky 9's enabled repos here (it lives in EPEL, which is not
# enabled), so registration goes through the pinned tonistiigi/binfmt container instead. That
# avoids adding a repository to a shared class node just to build one image.
ensure_arm64_emulation() {
  if [ "$(host_platform)" = "linux/arm64" ]; then
    note "build host is already arm64 — no emulation needed"
    return 0
  fi
  if [ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]; then
    note "aarch64 binfmt handler already registered"
    return 0
  fi
  if [ "${SKIP_QEMU_SETUP}" = "1" ]; then
    die "SKIP_QEMU_SETUP=1 but /proc/sys/fs/binfmt_misc/qemu-aarch64 is missing. Register it with: sudo ${CONTAINER_CLI} run --rm --privileged ${QEMU_IMAGE} --install arm64"
  fi

  say "Registering aarch64 emulation (needed for the GH200 image)"
  note "this needs sudo and --privileged; it registers a kernel binfmt handler that lasts until reboot"
  sudo "${CONTAINER_CLI}" run --rm --privileged "${QEMU_IMAGE}" --install arm64
  [ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ] \
    || die "binfmt registration did not take — /proc/sys/fs/binfmt_misc/qemu-aarch64 is still missing. Check that binfmt_misc is mounted (mount | grep binfmt_misc) and that sudo ${CONTAINER_CLI} works."
  note "registered"
}

# --- build -------------------------------------------------------------------------------------

build_one() {
  local backend="$1"
  local platform dockerfile repository image_tagged image_latest

  platform="$(platform_for "${backend}")"
  dockerfile="${SCRIPT_DIR}/Dockerfile.${backend}"
  [ -f "${dockerfile}" ] || die "missing ${dockerfile}"

  repository="$(repository_for "${backend}")"
  image_tagged="${REGISTRY}/${repository}:${TAG}"
  image_latest="${REGISTRY}/${repository}:latest"

  if [ "${platform}" != "$(host_platform)" ]; then
    ensure_arm64_emulation
    note "cross-building ${platform} on $(host_platform) under qemu — expect this one to take tens of minutes"
  fi

  say "Building ${backend} (${platform})"
  note "${image_tagged}"
  # Context is the repo root because the Dockerfiles COPY src/. /.dockerignore is an allow-list,
  # so the corpus, results/ and any credentials in the tree cannot reach the image.
  "${CONTAINER_CLI}" build \
    --platform "${platform}" \
    --file "${dockerfile}" \
    --tag "${image_tagged}" \
    --tag "${image_latest}" \
    "${REPO_ROOT}"

  # Cheap sanity check on images we can actually execute here. It deliberately does NOT run
  # smoke.py: this node has no accelerator, so smoke.py would correctly fail the tpu image. What
  # it proves is that the dependency graph imports at all, which is the failure mode worth
  # catching before a 30-minute queue wait.
  if [ "${VERIFY}" = "1" ] && [ "${platform}" = "$(host_platform)" ]; then
    say "Verifying ${backend} imports"
    # No ENTRYPOINT is set in any of the three Dockerfiles, so a trailing command simply replaces
    # the CMD.
    "${CONTAINER_CLI}" run --rm "${image_tagged}" \
      python3 -c 'import importlib.metadata as m, jax, tunix, qwix, transformers; print("jax", jax.__version__, "| tunix", m.version("google-tunix"), "| qwix", m.version("qwix"), "| transformers", transformers.__version__)'
  elif [ "${VERIFY}" = "1" ]; then
    note "skipping import check — ${platform} would run under emulation and is not worth the minutes"
  fi

  if [ "${PUSH}" = "1" ]; then
    say "Pushing ${backend}"
    "${CONTAINER_CLI}" push "${image_tagged}"
    "${CONTAINER_CLI}" push "${image_latest}"
  else
    note "PUSH=0 — built locally, not pushed"
  fi

  BUILT_SUMMARY="${BUILT_SUMMARY}${backend}	${platform}	${image_tagged}
"
  # Only the images built in THIS invocation may be advertised at the immutable ${TAG} — the
  # others have no such tag in the registry and naming one would produce ImagePullBackOff.
  BUILT_BACKENDS="${BUILT_BACKENDS}${backend} "
}

# --- images.env -------------------------------------------------------------------------------

# The variable names here are not decorative: manifests/env.sh reads IMAGE_URI_TPU,
# IMAGE_URI_GPU and IMAGE_URI_CPU (and every one of its `: "${X:=default}"` assignments yields to
# whatever is already exported), and scripts/common.sh reads the same three. Sourcing this file
# BEFORE manifests/env.sh is therefore what pins a campaign to the images that were just built.
#
# Backends rebuilt in this invocation are pinned to the immutable ${TAG}; the rest fall back to
# :latest, because ${TAG} does not exist in the registry for an image that was not just pushed.
write_images_env() {
  {
    printf '# Generated by docker/build.sh on %s.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '#   source %s && source manifests/env.sh\n' "${IMAGES_ENV}"
    printf '# Rebuilt and tagged %s in that run: %s\n' "${TAG}" "${BUILT_BACKENDS:-<none>}"
    printf 'export TEAM=%s\n' "${TEAM}"
    printf 'export REGISTRY=%s\n' "${REGISTRY}"
    printf 'export IMAGE_TAG=%s\n' "${TAG}"
    local backend tag
    for backend in ${ALL_BACKENDS}; do
      case " ${BUILT_BACKENDS} " in
        *" ${backend} "*) tag="${TAG}" ;;
        *)                tag="latest" ;;
      esac
      printf 'export IMAGE_URI_%s=%s/%s:%s\n' \
        "$(backend_for "${backend}" | tr '[:lower:]' '[:upper:]')" \
        "${REGISTRY}" "$(repository_for "${backend}")" "${tag}"
    done
  } > "${IMAGES_ENV}"
}

# --- main ---------------------------------------------------------------------------------------

main() {
  local targets="" backend requested seen

  # Several targets in one call is the normal case, not an edge case: scripts/run-all.sh builds
  # whatever is missing with a single `bash docker/build.sh cuda tpu cpu`. Reading only $1 would
  # have silently built one image and left the campaign to fail its post-build registry check.
  if [ "$#" -eq 0 ]; then
    set -- all
  fi
  for requested in "$@"; do
    case "${requested}" in
      all)          targets="${targets} ${ALL_BACKENDS}" ;;
      tpu|cuda|cpu) targets="${targets} ${requested}" ;;
      # `gpu` is what the rest of the repo calls the GH200 row; `cuda` is the Dockerfile suffix.
      # Accept both so a caller never has to know which vocabulary this script speaks.
      gpu)          targets="${targets} cuda" ;;
      -h|--help|help)
        printf 'usage: %s [tpu|cuda|gpu|cpu|all ...]\n\n' "$0"
        printf 'Targets may be repeated; `gpu` is an alias for `cuda`.\n'
        printf 'env: TEAM (required), REGISTRY, TAG, PUSH=0|1, VERIFY=0|1, CONTAINER_CLI, QEMU_IMAGE, SKIP_QEMU_SETUP=0|1\n'
        exit 0 ;;
      *)            die "unknown target '${requested}' — expected one of: ${ALL_BACKENDS} gpu all" ;;
    esac
  done

  # De-duplicate while preserving order: `build.sh all cuda` must not build the GH200 image twice
  # under qemu.
  seen=""
  for requested in ${targets}; do
    case " ${seen} " in *" ${requested} "*) continue ;; esac
    seen="${seen}${requested} "
  done
  targets="${seen}"

  BUILT_SUMMARY=""
  BUILT_BACKENDS=""
  preflight
  for backend in ${targets}; do
    build_one "${backend}"
  done
  write_images_env

  say "Done"
  printf 'backend\tplatform\timmutable tag\n'
  printf '%s' "${BUILT_SUMMARY}"
  printf '\nimage URIs written to %s\n' "${IMAGES_ENV}"
  printf 'Next:\n'
  printf '  source %s && source %s/manifests/env.sh\n' "${IMAGES_ENV}" "${REPO_ROOT}"
  printf '  (images.env must come FIRST: manifests/env.sh assigns with := and yields to it)\n'
  printf 'Then submit the smoke Job for each backend before any real run — the images default to\n'
  printf 'docker/smoke.py, which fails loudly if the accelerator plugin did not load instead of\n'
  printf 'quietly benchmarking the wrong hardware.\n'
}

main "$@"
