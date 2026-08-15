#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${script_dir}/../.." && pwd)"
state_dir="${SANDBOXFUSION_STATE_DIR:-${repo}/data/m2rl/sandbox}"
pin_file="${SANDBOXFUSION_PIN_FILE:-${state_dir}/sandboxfusion-image.env}"
marker="${SANDBOX_PREFLIGHT_MARKER:-${M2RL_SANDBOX_PREFLIGHT_MARKER:-${state_dir}/sandboxfusion_preflight.json}}"
log_dir="${repo}/outputs/qwen3_1.7b_single_task/.launcher_logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
build_log="${log_dir}/sandboxfusion.rebuild.${timestamp}.log"
start_log="${log_dir}/sandboxfusion.restart.${timestamp}.log"

cd "${repo}"
mkdir -p "$(dirname "${pin_file}")" "$(dirname "${marker}")" "${log_dir}"

for writable_dir in "$(dirname "${pin_file}")" "$(dirname "${marker}")"; do
  if [[ ! -w "${writable_dir}" ]] || [[ ! -x "${writable_dir}" ]]; then
    echo "SandboxFusion 状态目录不可写：${writable_dir}" >&2
    echo "请将 SANDBOXFUSION_STATE_DIR 指向当前用户拥有的持久化目录。" >&2
    exit 2
  fi
done

if [[ -e /.dockerenv ]] || [[ -e /run/.containerenv ]]; then
  echo "本脚本只能在宿主机运行，不能在训练容器内运行。" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "当前宿主机终端无法访问 Docker daemon。" >&2
  exit 2
fi

echo "[1/2] 重建包含 LiveCodeBench 128 MiB/16 MiB 边界修复的固定镜像"
SANDBOXFUSION_PIN_FILE="${pin_file}" \
  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh 2>&1 | tee "${build_log}"

echo "[2/2] 重新启动并执行 SandboxFusion 安全预检"
SANDBOXFUSION_PIN_FILE="${pin_file}" \
  SANDBOX_PREFLIGHT_MARKER="${marker}" \
  bash examples/optimizer_geometry/start_sandboxfusion.sh 2>&1 | tee "${start_log}"

python3 - "${marker}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    marker = json.load(stream)
if marker.get("safe") is not True:
    raise SystemExit("SandboxFusion marker is not safe")
print("SandboxFusion marker safe=true")
PY

echo "SandboxFusion 已在宿主机启动。"
echo "Image pin：${pin_file}"
echo "Preflight marker：${marker}"
echo "下一步请进入使用 --network host 启动的训练容器，然后运行："
echo "  bash scripts/optimizer_geometry/start_code_grpo_in_container.sh"
