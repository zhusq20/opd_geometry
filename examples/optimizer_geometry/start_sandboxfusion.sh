#!/usr/bin/env bash
# Start the digest-pinned, cgroup-v2 SandboxFusion build and attest it.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${SANDBOXFUSION_COMPOSE_FILE:-${SCRIPT_DIR}/sandboxfusion-compose.yaml}"
PIN_FILE="${SANDBOXFUSION_PIN_FILE:-${SLIME_DIR}/data/m2rl/sandbox/sandboxfusion-image.env}"
PORT="${SANDBOXFUSION_PORT:-8080}"
RUN_CODE_URL="http://127.0.0.1:${PORT}/run_code"
MARKER="${SANDBOX_PREFLIGHT_MARKER:-${M2RL_SANDBOX_PREFLIGHT_MARKER:-${SLIME_DIR}/data/m2rl/sandbox/sandboxfusion_preflight.json}}"
# Keep the delegated host name aligned with the audited runner fallback. The
# execution environment is deliberately scrubbed, so the helper must resolve
# the same path even when SANDBOX_CGROUP2_ROOT is not propagated to user code.
CGROUP_NAME="${SANDBOXFUSION_CGROUP_NAME:-sandboxfusion}"
EXPECTED_BASE_IMAGE="volcengine/sandbox-fusion@sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9"
AGGREGATE_MEMORY_BYTES="${SANDBOXFUSION_AGGREGATE_MEMORY_BYTES:-34359738368}"
AGGREGATE_PIDS="${SANDBOXFUSION_AGGREGATE_PIDS:-4096}"

write_unsafe_marker() {
  local marker_directory temporary_marker
  marker_directory="$(dirname -- "${MARKER}")"
  mkdir -p -- "${marker_directory}"
  temporary_marker="$(mktemp "${MARKER}.invalid.XXXXXX")"
  printf '%s\n' '{"schema_version":2,"safe":false,"reason":"deployment verification incomplete"}' \
    >"${temporary_marker}"
  chmod 0600 "${temporary_marker}"
  mv -f -- "${temporary_marker}" "${MARKER}"
}

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for the secure SandboxFusion profile." >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "The Docker daemon is unavailable to this user." >&2
  exit 2
fi
if docker info --format '{{json .SecurityOptions}}' | grep -q 'name=rootless'; then
  echo "The audited cgroup delegation requires a rootful Docker daemon." >&2
  exit 2
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required for the secure SandboxFusion profile." >&2
  exit 2
fi
if [[ ! "${PORT}" =~ ^[0-9]{1,5}$ ]] || ((10#${PORT} < 1 || 10#${PORT} > 65535)); then
  echo "Invalid SANDBOXFUSION_PORT: ${PORT}" >&2
  exit 2
fi

if [[ -f "${PIN_FILE}" ]]; then
  while IFS='=' read -r key value; do
    case "${key}" in
      SANDBOXFUSION_IMAGE|SANDBOXFUSION_PATCH_SHA256)
        printf -v "${key}" '%s' "${value}"
        export "${key}"
        ;;
    esac
  done <"${PIN_FILE}"
fi

if [[ -z "${SANDBOXFUSION_IMAGE:-}" ]]; then
  echo "No patched image pin was found. Run:" >&2
  echo "  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh" >&2
  exit 2
fi
if [[ ! "${SANDBOXFUSION_IMAGE}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  && [[ ! "${SANDBOXFUSION_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "Refusing mutable SandboxFusion image reference: ${SANDBOXFUSION_IMAGE}" >&2
  exit 2
fi
if [[ ! "${CGROUP_NAME}" =~ ^[A-Za-z0-9_.-]{1,80}$ ]]; then
  echo "Invalid SANDBOXFUSION_CGROUP_NAME: ${CGROUP_NAME}" >&2
  exit 2
fi
if [[ "${CGROUP_NAME}" != "sandboxfusion" ]]; then
  echo "The audited runner requires SANDBOXFUSION_CGROUP_NAME=sandboxfusion." >&2
  exit 2
fi
if [[ ! "${AGGREGATE_MEMORY_BYTES}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${AGGREGATE_PIDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Aggregate cgroup limits must be positive integers." >&2
  exit 2
fi

CGROUP_VERSION="$(docker info --format '{{.CgroupVersion}}')"
if [[ "${CGROUP_VERSION}" != "2" ]]; then
  echo "The patched secure profile requires host cgroup v2; Docker reports v${CGROUP_VERSION}." >&2
  exit 2
fi
DOCKER_ARCHITECTURE="$(docker info --format '{{.Architecture}}')"
if [[ "${DOCKER_ARCHITECTURE}" != "x86_64" ]] && [[ "${DOCKER_ARCHITECTURE}" != "amd64" ]]; then
  echo "The audited SandboxFusion profile requires an amd64 Docker host." >&2
  exit 2
fi

if ! docker image inspect "${SANDBOXFUSION_IMAGE}" >/dev/null 2>&1; then
  if [[ "${SANDBOXFUSION_IMAGE}" == *@sha256:* ]]; then
    docker pull "${SANDBOXFUSION_IMAGE}"
  else
    echo "Pinned local image ${SANDBOXFUSION_IMAGE} is absent; rebuild it on this host." >&2
    exit 2
  fi
fi

EXPECTED_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${SANDBOXFUSION_IMAGE}")"
EXPECTED_PATCH_SHA256="$(
  cd "${SCRIPT_DIR}/sandboxfusion_patch"
  find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name '*.pyo' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}'
)"
if [[ -n "${SANDBOXFUSION_PATCH_SHA256:-}" ]] \
  && [[ "${EXPECTED_PATCH_SHA256}" != "${SANDBOXFUSION_PATCH_SHA256}" ]]; then
  echo "Current patch sources do not match ${PIN_FILE}; rebuild the image." >&2
  exit 1
fi
PATCH_ID="$(docker image inspect --format '{{index .Config.Labels "io.iclr2027.sandboxfusion.patch"}}' "${EXPECTED_IMAGE_ID}")"
PATCH_SHA256="$(docker image inspect --format '{{index .Config.Labels "io.iclr2027.sandboxfusion.patch-sha256"}}' "${EXPECTED_IMAGE_ID}")"
BASE_IMAGE="$(docker image inspect --format '{{index .Config.Labels "io.iclr2027.sandboxfusion.base"}}' "${EXPECTED_IMAGE_ID}")"
if [[ "${PATCH_ID}" != "cgroup2-v1" ]]; then
  echo "Refusing image without the audited cgroup2-v1 patch label." >&2
  exit 1
fi
if [[ "${PATCH_SHA256}" != "${EXPECTED_PATCH_SHA256}" ]]; then
  echo "Pinned image patch hash does not match the current audited sources." >&2
  exit 1
fi
if [[ "${BASE_IMAGE}" != "${EXPECTED_BASE_IMAGE}" ]]; then
  echo "Pinned image does not attest the audited upstream base digest." >&2
  exit 1
fi
if [[ -n "${SANDBOXFUSION_PATCH_SHA256:-}" ]] \
  && [[ "${PATCH_SHA256}" != "${SANDBOXFUSION_PATCH_SHA256}" ]]; then
  echo "Image patch hash does not match ${PIN_FILE}." >&2
  exit 1
fi

SANDBOXFUSION_CGROUP_PATH="/sys/fs/cgroup/${CGROUP_NAME}"
SANDBOXFUSION_PORT="${PORT}"
export SANDBOXFUSION_CGROUP_PATH SANDBOXFUSION_IMAGE SANDBOXFUSION_PORT
docker compose -f "${COMPOSE_FILE}" config --quiet

# A marker from an older container must never authorize the replacement while
# its image, mount, and active probes are still being verified.
write_unsafe_marker

# Stop an older deployment before cleaning its delegated execution subtree.
docker compose -f "${COMPOSE_FILE}" down --remove-orphans

DEPLOYMENT_MAY_EXIST=0
cleanup_failed_deployment() {
  local status=$?
  trap - EXIT
  if ((status != 0)); then
    write_unsafe_marker || true
    if [[ "${DEPLOYMENT_MAY_EXIST}" == "1" ]]; then
      docker compose -f "${COMPOSE_FILE}" down --remove-orphans >/dev/null 2>&1 || true
    fi
  fi
  exit "${status}"
}
trap cleanup_failed_deployment EXIT

# Prepare a dedicated empty subtree. Only that subtree, rather than the entire
# host cgroup hierarchy, is mounted read-write into the service container. The
# host cgroup root can be mode 0555 even when that filesystem is read-write, so
# DAC_OVERRIDE is required to create the single audited delegation directory.
docker run --rm \
  --cgroupns host \
  --network none \
  --read-only \
  --memory 128m \
  --pids-limit 64 \
  --cap-drop ALL \
  --cap-add DAC_OVERRIDE \
  --cap-add KILL \
  --security-opt no-new-privileges=true \
  --env "SANDBOXFUSION_CGROUP_NAME=${CGROUP_NAME}" \
  --env "SANDBOXFUSION_AGGREGATE_MEMORY_BYTES=${AGGREGATE_MEMORY_BYTES}" \
  --env "SANDBOXFUSION_AGGREGATE_PIDS=${AGGREGATE_PIDS}" \
  --mount type=bind,source=/sys/fs/cgroup,target=/host-cgroup \
  --entrypoint /usr/local/libexec/sandboxfusion-prepare-cgroup2 \
  "${SANDBOXFUSION_IMAGE}" >/dev/null

if [[ "$(<"${SANDBOXFUSION_CGROUP_PATH}/memory.max")" != "${AGGREGATE_MEMORY_BYTES}" ]] \
  || [[ "$(<"${SANDBOXFUSION_CGROUP_PATH}/pids.max")" != "${AGGREGATE_PIDS}" ]]; then
  echo "Delegated cgroup aggregate limits were not applied." >&2
  exit 1
fi

DEPLOYMENT_MAY_EXIST=1
docker compose -f "${COMPOSE_FILE}" up -d --pull never
CONTAINER_HANDLE="$(docker compose -f "${COMPOSE_FILE}" ps -q sandboxfusion)"
PROXY_HANDLE="$(docker compose -f "${COMPOSE_FILE}" ps -q loopback_proxy)"
if [[ -z "${CONTAINER_HANDLE}" ]] || [[ -z "${PROXY_HANDLE}" ]]; then
  echo "SandboxFusion or its loopback proxy was not created." >&2
  docker compose -f "${COMPOSE_FILE}" logs --tail 100 >&2 || true
  exit 1
fi
CONTAINER_ID="$(docker inspect --format '{{.Id}}' "${CONTAINER_HANDLE}")"
PROXY_CONTAINER_ID="$(docker inspect --format '{{.Id}}' "${PROXY_HANDLE}")"

SANDBOX_CONFIG_VALUE="$(
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${CONTAINER_ID}" \
    | awk -F= '$1 == "SANDBOX_CONFIG" {print $2}'
)"
if [[ "${SANDBOX_CONFIG_VALUE}" != "ci" ]]; then
  echo "Refusing SANDBOX_CONFIG=${SANDBOX_CONFIG_VALUE:-unset}; expected ci." >&2
  exit 1
fi
IMAGE_ID="$(docker inspect --format '{{.Image}}' "${CONTAINER_ID}")"
IMAGE_REFERENCE="$(docker inspect --format '{{.Config.Image}}' "${CONTAINER_ID}")"
PROXY_IMAGE_ID="$(docker inspect --format '{{.Image}}' "${PROXY_CONTAINER_ID}")"
if [[ "${IMAGE_ID}" != "${EXPECTED_IMAGE_ID}" ]] || [[ "${PROXY_IMAGE_ID}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Running service/proxy images do not both match pinned image ${EXPECTED_IMAGE_ID}." >&2
  exit 1
fi
CGROUP_MODE="$(docker inspect --format '{{.HostConfig.CgroupnsMode}}' "${CONTAINER_ID}")"
if [[ "${CGROUP_MODE}" != "host" ]]; then
  echo "Refusing cgroup namespace mode ${CGROUP_MODE:-unset}; expected host." >&2
  exit 1
fi
PRIVILEGED="$(docker inspect --format '{{.HostConfig.Privileged}}' "${CONTAINER_ID}")"
if [[ "${PRIVILEGED}" != "false" ]]; then
  echo "Refusing privileged SandboxFusion container." >&2
  exit 1
fi
READ_ONLY_ROOT="$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "${CONTAINER_ID}")"
if [[ "${READ_ONLY_ROOT}" != "true" ]]; then
  echo "SandboxFusion service root filesystem is not read-only." >&2
  exit 1
fi
SERVICE_MEMORY_LIMIT="$(docker inspect --format '{{.HostConfig.Memory}}' "${CONTAINER_ID}")"
SERVICE_PIDS_LIMIT="$(docker inspect --format '{{.HostConfig.PidsLimit}}' "${CONTAINER_ID}")"
if [[ "${SERVICE_MEMORY_LIMIT}" != "4294967296" ]] || [[ "${SERVICE_PIDS_LIMIT}" != "2048" ]]; then
  echo "SandboxFusion control-plane memory/PID limits differ from the audited profile." >&2
  exit 1
fi
EXPECTED_CAPABILITIES=$'DAC_OVERRIDE\nFOWNER\nKILL\nMKNOD\nNET_ADMIN\nSETGID\nSETPCAP\nSETUID\nSYS_ADMIN\nSYS_CHROOT'
ACTUAL_CAPABILITIES="$(
  docker inspect --format '{{range .HostConfig.CapAdd}}{{println .}}{{end}}' "${CONTAINER_ID}" \
    | sed -e 's/^CAP_//' -e '/^$/d' \
    | LC_ALL=C sort
)"
if [[ "${ACTUAL_CAPABILITIES}" != "${EXPECTED_CAPABILITIES}" ]]; then
  echo "SandboxFusion capability allowlist differs from the audited profile." >&2
  exit 1
fi
CAP_DROP="$(docker inspect --format '{{join .HostConfig.CapDrop ","}}' "${CONTAINER_ID}")"
if [[ "${CAP_DROP}" != "ALL" ]]; then
  echo "SandboxFusion must drop all capabilities before applying its allowlist." >&2
  exit 1
fi
SECURITY_OPTIONS="$(docker inspect --format '{{join .HostConfig.SecurityOpt "\n"}}' "${CONTAINER_ID}")"
for option_pattern in '^apparmor[:=]unconfined$' '^no-new-privileges[:=]true$' '^seccomp[:=]builtin$'; do
  if ! grep -Eq "${option_pattern}" <<<"${SECURITY_OPTIONS}"; then
    echo "SandboxFusion security options differ from the audited profile." >&2
    exit 1
  fi
done
MOUNT_SOURCE="$(
  docker inspect --format '{{range .Mounts}}{{println .Destination "\t" .Source}}{{end}}' "${CONTAINER_ID}" \
    | awk -v destination="${SANDBOXFUSION_CGROUP_PATH}" '$1 == destination {print $2}'
)"
if [[ "${MOUNT_SOURCE}" != "${SANDBOXFUSION_CGROUP_PATH}" ]]; then
  echo "Delegated cgroup mount mismatch: ${MOUNT_SOURCE:-missing}." >&2
  exit 1
fi
MOUNT_POLICY="$(
  docker inspect --format '{{range .Mounts}}{{println .Destination "\t" .RW "\t" .Propagation}}{{end}}' "${CONTAINER_ID}" \
    | awk -v destination="${SANDBOXFUSION_CGROUP_PATH}" '$1 == destination {printf "%s:%s", $2, $3}'
)"
if [[ "${MOUNT_POLICY}" != "true:rprivate" ]]; then
  echo "Delegated cgroup mount must be rw with rprivate propagation." >&2
  exit 1
fi
MOUNT_DESTINATIONS="$(
  docker inspect --format '{{range .Mounts}}{{if or (eq .Type "bind") (eq .Type "volume")}}{{println .Destination}}{{end}}{{end}}' "${CONTAINER_ID}" \
    | sed '/^$/d' \
    | LC_ALL=C sort
)"
if [[ "${MOUNT_DESTINATIONS}" != "${SANDBOXFUSION_CGROUP_PATH}" ]]; then
  echo "SandboxFusion has an unexpected bind/volume mount: ${MOUNT_DESTINATIONS:-none}." >&2
  exit 1
fi
PARENT_CGROUP_OPTIONS="$(docker exec "${CONTAINER_ID}" findmnt -n -o OPTIONS --target /sys/fs/cgroup)"
DELEGATED_CGROUP_OPTIONS="$(
  docker exec "${CONTAINER_ID}" findmnt -n -o OPTIONS --target "${SANDBOXFUSION_CGROUP_PATH}"
)"
if [[ ",${PARENT_CGROUP_OPTIONS}," != *,ro,* ]]; then
  echo "SandboxFusion can write the parent cgroup hierarchy; refusing deployment." >&2
  exit 1
fi
if [[ ",${DELEGATED_CGROUP_OPTIONS}," != *,rw,* ]]; then
  echo "SandboxFusion delegated cgroup subtree is not writable." >&2
  exit 1
fi
SERVICE_PORT_BINDINGS="$(
  docker inspect --format '{{range $port, $bindings := .HostConfig.PortBindings}}{{println $port}}{{end}}' \
    "${CONTAINER_ID}" | sed '/^$/d'
)"
if [[ -n "${SERVICE_PORT_BINDINGS}" ]]; then
  echo "SandboxFusion service must not publish a host port directly: ${SERVICE_PORT_BINDINGS}." >&2
  exit 1
fi
NETWORK_IDS="$(
  docker inspect --format '{{range $name, $network := .NetworkSettings.Networks}}{{println $network.NetworkID}}{{end}}' \
    "${CONTAINER_ID}" | sed '/^$/d'
)"
if [[ "$(wc -l <<<"${NETWORK_IDS}")" != "1" ]]; then
  echo "SandboxFusion must be attached to exactly one internal bridge network." >&2
  exit 1
fi
CONTROL_PLANE_NETWORK_INTERNAL="$(docker network inspect --format '{{.Internal}}' "${NETWORK_IDS}")"
if [[ "${CONTROL_PLANE_NETWORK_INTERNAL}" != "true" ]]; then
  echo "SandboxFusion control-plane network is not Docker-internal." >&2
  exit 1
fi

PROXY_PRIVILEGED="$(docker inspect --format '{{.HostConfig.Privileged}}' "${PROXY_CONTAINER_ID}")"
PROXY_READ_ONLY_ROOT="$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "${PROXY_CONTAINER_ID}")"
PROXY_MEMORY_LIMIT="$(docker inspect --format '{{.HostConfig.Memory}}' "${PROXY_CONTAINER_ID}")"
PROXY_PIDS_LIMIT="$(docker inspect --format '{{.HostConfig.PidsLimit}}' "${PROXY_CONTAINER_ID}")"
if [[ "${PROXY_PRIVILEGED}" != "false" ]] \
  || [[ "${PROXY_READ_ONLY_ROOT}" != "true" ]] \
  || [[ "${PROXY_MEMORY_LIMIT}" != "134217728" ]] \
  || [[ "${PROXY_PIDS_LIMIT}" != "64" ]]; then
  echo "Loopback proxy privilege or resource settings differ from the audited profile." >&2
  exit 1
fi
PROXY_CAPABILITIES="$(
  docker inspect --format '{{range .HostConfig.CapAdd}}{{println .}}{{end}}' "${PROXY_CONTAINER_ID}" \
    | sed '/^$/d'
)"
PROXY_CAP_DROP="$(docker inspect --format '{{join .HostConfig.CapDrop ","}}' "${PROXY_CONTAINER_ID}")"
if [[ -n "${PROXY_CAPABILITIES}" ]] || [[ "${PROXY_CAP_DROP}" != "ALL" ]]; then
  echo "Loopback proxy must run without capabilities." >&2
  exit 1
fi
PROXY_SECURITY_OPTIONS="$(
  docker inspect --format '{{join .HostConfig.SecurityOpt "\n"}}' "${PROXY_CONTAINER_ID}"
)"
for option_pattern in '^no-new-privileges[:=]true$' '^seccomp[:=]builtin$'; do
  if ! grep -Eq "${option_pattern}" <<<"${PROXY_SECURITY_OPTIONS}"; then
    echo "Loopback proxy security options differ from the audited profile." >&2
    exit 1
  fi
done
PROXY_MOUNT_DESTINATIONS="$(
  docker inspect --format '{{range .Mounts}}{{if or (eq .Type "bind") (eq .Type "volume")}}{{println .Destination}}{{end}}{{end}}' \
    "${PROXY_CONTAINER_ID}" | sed '/^$/d'
)"
if [[ -n "${PROXY_MOUNT_DESTINATIONS}" ]]; then
  echo "Loopback proxy has an unexpected host-backed mount: ${PROXY_MOUNT_DESTINATIONS}." >&2
  exit 1
fi
PROXY_ENTRYPOINT="$(docker inspect --format '{{json .Config.Entrypoint}}' "${PROXY_CONTAINER_ID}")"
PROXY_UPSTREAM="$(
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${PROXY_CONTAINER_ID}" \
    | awk -F= '$1 == "SANDBOXFUSION_PROXY_UPSTREAM_HOST" || $1 == "SANDBOXFUSION_PROXY_UPSTREAM_PORT"' \
    | LC_ALL=C sort
)"
if [[ "${PROXY_ENTRYPOINT}" != '["python3","-c"]' ]] \
  || [[ "${PROXY_UPSTREAM}" != $'SANDBOXFUSION_PROXY_UPSTREAM_HOST=sandboxfusion\nSANDBOXFUSION_PROXY_UPSTREAM_PORT=8080' ]]; then
  echo "Loopback proxy is not configured as the fixed SandboxFusion relay." >&2
  exit 1
fi
CONFIGURED_PORT_BINDING="$(
  docker inspect --format '{{range (index .HostConfig.PortBindings "8080/tcp")}}{{printf "%s:%s\n" .HostIp .HostPort}}{{end}}' \
    "${PROXY_CONTAINER_ID}"
)"
ACTIVE_PORT_BINDING="$(
  docker inspect --format '{{range (index .NetworkSettings.Ports "8080/tcp")}}{{printf "%s:%s\n" .HostIp .HostPort}}{{end}}' \
    "${PROXY_CONTAINER_ID}"
)"
if [[ "${CONFIGURED_PORT_BINDING}" != "127.0.0.1:${PORT}" ]] \
  || [[ "${ACTIVE_PORT_BINDING}" != "127.0.0.1:${PORT}" ]]; then
  echo "Loopback proxy is not bound exclusively to 127.0.0.1:${PORT}." >&2
  exit 1
fi
PROXY_NETWORK_IDS="$(
  docker inspect --format '{{range $name, $network := .NetworkSettings.Networks}}{{println $network.NetworkID}}{{end}}' \
    "${PROXY_CONTAINER_ID}" | sed '/^$/d'
)"
if [[ "$(wc -l <<<"${PROXY_NETWORK_IDS}")" != "2" ]] \
  || ! grep -Fxq "${NETWORK_IDS}" <<<"${PROXY_NETWORK_IDS}"; then
  echo "Loopback proxy must connect only the internal service network to its ingress network." >&2
  exit 1
fi
PROXY_NETWORK_INTERNAL_FLAGS="$(
  while IFS= read -r network_id; do
    docker network inspect --format '{{.Internal}}' "${network_id}"
  done <<<"${PROXY_NETWORK_IDS}" | sed '/^$/d' | LC_ALL=C sort
)"
if [[ "${PROXY_NETWORK_INTERNAL_FLAGS}" != $'false\ntrue' ]]; then
  echo "Loopback proxy requires exactly one internal and one ingress network." >&2
  exit 1
fi

COMPOSE_SHA256="$(sha256sum "${COMPOSE_FILE}" | awk '{print $1}')"
ready=0
for _attempt in $(seq 1 60); do
  if python3 -c 'import sys,urllib.request; sys.exit(0 if urllib.request.urlopen(sys.argv[1], timeout=2).read() else 1)' "http://127.0.0.1:${PORT}/v1/ping" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" != "1" ]]; then
  echo "SandboxFusion did not become ready on port ${PORT}." >&2
  docker compose -f "${COMPOSE_FILE}" logs --tail 100 >&2 || true
  exit 1
fi

SERVICE_CANARY_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
SERVICE_CANARY_PATH="/tmp/sandboxfusion-service-canary-${SERVICE_CANARY_TOKEN}"
docker exec "${CONTAINER_ID}" python3 -c \
  'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_text(sys.argv[2]); p.chmod(0o644)' \
  "${SERVICE_CANARY_PATH}" "${SERVICE_CANARY_TOKEN}"
SERVICE_NAMESPACES="$(
  docker exec "${CONTAINER_ID}" python3 -c \
    'import json,os; print(json.dumps({n:os.readlink("/proc/1/ns/"+n) for n in ("mnt","pid","net","ipc","uts")}))'
)"
CONTROL_PLANE_NETWORK_DENIED="$(
  docker exec "${CONTAINER_ID}" python3 -c \
    'import socket; s=socket.socket(); s.settimeout(1); rc=s.connect_ex(("1.1.1.1",443)); s.close(); print("true" if rc else "false")'
)"
if [[ "${CONTROL_PLANE_NETWORK_DENIED}" != "true" ]]; then
  echo "SandboxFusion control plane unexpectedly reached the public network." >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/sandbox_preflight.py" \
  --url "${RUN_CODE_URL}" \
  --marker "${MARKER}" \
  --container-id "${CONTAINER_ID}" \
  --proxy-container-id "${PROXY_CONTAINER_ID}" \
  --image-id "${IMAGE_ID}" \
  --proxy-image-id "${PROXY_IMAGE_ID}" \
  --image-reference "${IMAGE_REFERENCE}" \
  --compose-sha256 "${COMPOSE_SHA256}" \
  --sandbox-config "${SANDBOX_CONFIG_VALUE}" \
  --cgroup-version "${CGROUP_VERSION}" \
  --cgroup-path "${SANDBOXFUSION_CGROUP_PATH}" \
  --patch-id "${PATCH_ID}" \
  --patch-sha256 "${PATCH_SHA256}" \
  --base-image "${BASE_IMAGE}" \
  --aggregate-memory-max "${AGGREGATE_MEMORY_BYTES}" \
  --aggregate-pids-max "${AGGREGATE_PIDS}" \
  --service-canary-path "${SERVICE_CANARY_PATH}" \
  --service-canary-token "${SERVICE_CANARY_TOKEN}" \
  --service-namespaces "${SERVICE_NAMESPACES}" \
  --control-plane-network-internal "${CONTROL_PLANE_NETWORK_INTERNAL}" \
  --control-plane-network-denied "${CONTROL_PLANE_NETWORK_DENIED}"

docker exec "${CONTAINER_ID}" rm -f -- "${SERVICE_CANARY_PATH}"

# The probes are sequential during startup, so any remaining per-request
# namespace or overlay directory is a teardown failure rather than live work.
OVERLAY_LEAKS="$(
  docker exec "${CONTAINER_ID}" find /tmp -mindepth 1 -maxdepth 1 -type d -name 'overlay_*' -print
)"
if [[ -n "${OVERLAY_LEAKS}" ]]; then
  echo "SandboxFusion leaked overlay directories after preflight: ${OVERLAY_LEAKS}" >&2
  exit 1
fi
NETNS_STATE="$(docker exec "${CONTAINER_ID}" ip netns list)"
if awk '$1 ~ /^sandbox_[0-9a-f]+$/ {found=1} END {exit !found}' <<<"${NETNS_STATE}"; then
  echo "SandboxFusion leaked a network namespace after preflight." >&2
  exit 1
fi

if [[ ! -r "${SANDBOXFUSION_CGROUP_PATH}" ]] || [[ ! -x "${SANDBOXFUSION_CGROUP_PATH}" ]]; then
  echo "Cannot inspect delegated cgroup cleanup state: ${SANDBOXFUSION_CGROUP_PATH}." >&2
  exit 1
fi
if find "${SANDBOXFUSION_CGROUP_PATH}" -mindepth 1 -maxdepth 1 -type d -print -quit | grep -q .; then
  echo "SandboxFusion leaked an execution cgroup after preflight; refusing deployment." >&2
  exit 1
fi

trap - EXIT
echo "SandboxFusion passed cgroup v2 and active isolation checks; marker: ${MARKER}"
