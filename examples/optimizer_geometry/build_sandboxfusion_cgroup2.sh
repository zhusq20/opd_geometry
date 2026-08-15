#!/usr/bin/env bash
# Build the audited cgroup v2 patch and write an immutable image reference.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PATCH_DIR="${SCRIPT_DIR}/sandboxfusion_patch"
BUILD_TAG="${SANDBOXFUSION_BUILD_TAG:-iclr2027/sandboxfusion-cgroup2:local}"
PIN_FILE="${SANDBOXFUSION_PIN_FILE:-${SLIME_DIR}/data/m2rl/sandbox/sandboxfusion-image.env}"
PUSH_REF="${SANDBOXFUSION_PUSH_REF:-}"
SOURCE_DATE_EPOCH="${SANDBOXFUSION_SOURCE_DATE_EPOCH:-0}"
VERIFY_REPRODUCIBLE="${SANDBOXFUSION_VERIFY_REPRODUCIBLE:-1}"
REPRO_TAG="${SANDBOXFUSION_REPRO_TAG:-iclr2027/sandboxfusion-cgroup2:reprocheck}"
EXPECTED_BASE_IMAGE="volcengine/sandbox-fusion@sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9"
RUNTIME_FINGERPRINT_PATHS=(
  /root/sandbox/sandbox/runners
  /root/sandbox/sandbox/server
  /usr/local/libexec
  /root/sandbox/sandbox/runners/cgroup_v2.py
  /root/sandbox/sandbox/runners/file_security.py
  /root/sandbox/sandbox/runners/isolation.py
  /root/sandbox/sandbox/runners/base.py
  /root/sandbox/sandbox/runners/types.py
  /root/sandbox/sandbox/runners/jupyter.py
  /root/sandbox/sandbox/runners/__init__.py
  /root/sandbox/sandbox/server/sandbox_api.py
  /root/sandbox/sandbox/server/online_judge_api.py
  /usr/local/libexec/sandboxfusion-cgroup2-exec
  /usr/local/libexec/sandboxfusion-prepare-cgroup2
  /usr/local/libexec/sandboxfusion-restrict-exec
)

if [[ "${VERIFY_REPRODUCIBLE}" != "0" ]] && [[ "${VERIFY_REPRODUCIBLE}" != "1" ]]; then
  echo "SANDBOXFUSION_VERIFY_REPRODUCIBLE must be 0 or 1." >&2
  exit 2
fi
if [[ ! "${SOURCE_DATE_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "SANDBOXFUSION_SOURCE_DATE_EPOCH must be a non-negative integer." >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is required to build SandboxFusion, but 'docker' was not found in PATH." >&2
  exit 2
fi
if ! DOCKER_INFO_OUTPUT="$(docker info 2>&1)"; then
  echo "Docker CLI is installed, but 'docker info' failed:" >&2
  printf '%s\n' "${DOCKER_INFO_OUTPUT}" >&2
  echo "Check that the host Docker daemon is running and that this login session can access its socket." >&2
  echo "If Docker access was just granted, start a new login shell and retry." >&2
  exit 2
fi
unset DOCKER_INFO_OUTPUT
if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker Buildx with the reproducible image exporter is required." >&2
  exit 2
fi
if docker info --format '{{json .SecurityOptions}}' | grep -q 'name=rootless'; then
  echo "The audited cgroup delegation requires a rootful Docker daemon." >&2
  exit 2
fi

CGROUP_VERSION="$(docker info --format '{{.CgroupVersion}}')"
if [[ "${CGROUP_VERSION}" != "2" ]]; then
  echo "This patched image targets cgroup v2; Docker reports cgroup v${CGROUP_VERSION}." >&2
  exit 2
fi
DOCKER_ARCHITECTURE="$(docker info --format '{{.Architecture}}')"
if [[ "${DOCKER_ARCHITECTURE}" != "x86_64" ]] && [[ "${DOCKER_ARCHITECTURE}" != "amd64" ]]; then
  echo "The audited upstream image requires an amd64 Docker host; Docker reports ${DOCKER_ARCHITECTURE}." >&2
  exit 2
fi
if [[ "${VERIFY_REPRODUCIBLE}" == "1" ]] && [[ "${REPRO_TAG}" == "${BUILD_TAG}" ]]; then
  echo "SANDBOXFUSION_REPRO_TAG must differ from SANDBOXFUSION_BUILD_TAG." >&2
  exit 2
fi

PATCH_SHA256="$(
  cd "${PATCH_DIR}"
  find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name '*.pyo' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}'
)"

BUILD_CONTEXT="$(mktemp -d)"
trap 'rm -rf -- "${BUILD_CONTEXT}"' EXIT
# The source tree can live on a filesystem whose synthetic permissions cannot
# be applied to /tmp. The build context metadata is normalized below anyway.
cp -R --no-preserve=mode,ownership,timestamps -- "${PATCH_DIR}/." "${BUILD_CONTEXT}/"
find "${BUILD_CONTEXT}" -type d -exec chmod 0755 '{}' +
find "${BUILD_CONTEXT}" -type f -exec chmod 0644 '{}' +
find "${BUILD_CONTEXT}" -exec touch -h -d "@${SOURCE_DATE_EPOCH}" '{}' +

build_image() {
  local tag="$1"
  shift
  docker buildx build \
    --target sandboxfusion-final \
    --platform linux/amd64 \
    --network none \
    --output type=docker,rewrite-timestamp=true \
    --build-arg "PATCH_SHA256=${PATCH_SHA256}" \
    --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
    --tag "${tag}" \
    "$@" \
    "${BUILD_CONTEXT}"
}

image_runtime_fingerprint() {
  local image="$1"
  {
    docker image inspect --format '{{.Architecture}}|{{.Os}}|{{json .Config}}' "${image}"
    docker run \
      --rm \
      --pull never \
      --network none \
      --read-only \
      --entrypoint /bin/sh \
      "${image}" \
      -ec '
        for path do
          if [ -f "${path}" ]; then
            stat -c "%n|%f|%a|%u|%g|%s|%Y" "${path}"
            sha256sum "${path}"
          else
            stat -c "%n|%f|%a|%u|%g|%s" "${path}"
          fi
        done
      ' sandboxfusion-fingerprint "${RUNTIME_FINGERPRINT_PATHS[@]}"
  } | sha256sum | awk '{print $1}'
}

docker buildx build \
  --target sandboxfusion-verified \
  --platform linux/amd64 \
  --network none \
  --output type=cacheonly \
  "${BUILD_CONTEXT}"

if [[ "${VERIFY_REPRODUCIBLE}" == "1" ]]; then
  build_image "${BUILD_TAG}" --no-cache
else
  build_image "${BUILD_TAG}"
fi

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${BUILD_TAG}")"
IMAGE_REFERENCE="${IMAGE_ID}"
BUILT_PATCH_ID="$(docker image inspect --format '{{index .Config.Labels "io.iclr2027.sandboxfusion.patch"}}' "${IMAGE_ID}")"
BUILT_PATCH_SHA256="$(docker image inspect --format '{{index .Config.Labels "io.iclr2027.sandboxfusion.patch-sha256"}}' "${IMAGE_ID}")"
BUILT_BASE_IMAGE="$(docker image inspect --format '{{index .Config.Labels "io.iclr2027.sandboxfusion.base"}}' "${IMAGE_ID}")"
if [[ "${BUILT_PATCH_ID}" != "cgroup2-v1" ]] \
  || [[ "${BUILT_PATCH_SHA256}" != "${PATCH_SHA256}" ]] \
  || [[ "${BUILT_BASE_IMAGE}" != "${EXPECTED_BASE_IMAGE}" ]]; then
  echo "Built image labels do not attest the expected base and cgroup2-v1 patch." >&2
  exit 1
fi

if [[ "${VERIFY_REPRODUCIBLE}" == "1" ]]; then
  build_image "${REPRO_TAG}" --no-cache
  REPRO_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${REPRO_TAG}")"
  if [[ "${REPRO_IMAGE_ID}" != "${IMAGE_ID}" ]]; then
    IMAGE_RUNTIME_FINGERPRINT="$(image_runtime_fingerprint "${IMAGE_ID}")"
    REPRO_RUNTIME_FINGERPRINT="$(image_runtime_fingerprint "${REPRO_IMAGE_ID}")"
    if [[ "${REPRO_RUNTIME_FINGERPRINT}" != "${IMAGE_RUNTIME_FINGERPRINT}" ]]; then
      echo "Reproducibility check failed: normalized runtime fingerprints differ." >&2
      echo "  ${IMAGE_ID}: ${IMAGE_RUNTIME_FINGERPRINT}" >&2
      echo "  ${REPRO_IMAGE_ID}: ${REPRO_RUNTIME_FINGERPRINT}" >&2
      exit 1
    fi
    echo "Docker image IDs differ only in ignored atime/ctime or directory-mtime metadata;" >&2
    echo "normalized runtime fingerprint verified: ${IMAGE_RUNTIME_FINGERPRINT}" >&2
  fi
  docker image rm "${REPRO_TAG}" >/dev/null
fi

if [[ -n "${PUSH_REF}" ]]; then
  if [[ "${PUSH_REF}" == *@sha256:* ]]; then
    echo "SANDBOXFUSION_PUSH_REF must be a mutable staging tag, not a digest reference." >&2
    exit 2
  fi
  docker tag "${BUILD_TAG}" "${PUSH_REF}"
  docker push "${PUSH_REF}"
  REPOSITORY="${PUSH_REF%:*}"
  if [[ "${PUSH_REF##*/}" != *:* ]]; then
    REPOSITORY="${PUSH_REF}"
  fi
  IMAGE_REFERENCE="$(
    docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "${PUSH_REF}" \
      | awk -v repository="${REPOSITORY}" 'index($0, repository "@sha256:") == 1 {print; exit}'
  )"
  if [[ ! "${IMAGE_REFERENCE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "Could not resolve a pushed repository digest for ${PUSH_REF}." >&2
    exit 1
  fi
fi

mkdir -p -- "$(dirname -- "${PIN_FILE}")"
TEMP_PIN="$(mktemp "${PIN_FILE}.XXXXXX")"
trap 'rm -rf -- "${BUILD_CONTEXT}"; rm -f -- "${TEMP_PIN}"' EXIT
{
  printf 'SANDBOXFUSION_IMAGE=%s\n' "${IMAGE_REFERENCE}"
  printf 'SANDBOXFUSION_PATCH_SHA256=%s\n' "${PATCH_SHA256}"
} >"${TEMP_PIN}"
chmod 0600 "${TEMP_PIN}"
mv -f -- "${TEMP_PIN}" "${PIN_FILE}"
trap - EXIT
rm -rf -- "${BUILD_CONTEXT}"

echo "Built image ID: ${IMAGE_ID}"
echo "Immutable deployment reference: ${IMAGE_REFERENCE}"
echo "Pin file: ${PIN_FILE}"
