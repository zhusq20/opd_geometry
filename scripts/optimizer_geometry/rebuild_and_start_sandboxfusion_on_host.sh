#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${script_dir}/../.." && pwd)"
marker="${repo}/data/m2rl/sandbox/sandboxfusion_preflight.json"
log_dir="${repo}/outputs/qwen3_1.7b_single_task/.launcher_logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
build_log="${log_dir}/sandboxfusion.rebuild.${timestamp}.log"
start_log="${log_dir}/sandboxfusion.restart.${timestamp}.log"

cd "${repo}"
mkdir -p "$(dirname "${marker}")" "${log_dir}"

if [[ -e /.dockerenv ]] || [[ -e /run/.containerenv ]]; then
  echo "本脚本只能在宿主机运行，不能在训练容器内运行。" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "当前宿主机终端无法访问 Docker daemon。" >&2
  exit 2
fi

echo "[1/2] 重建包含 LiveCodeBench 128 MiB/16 MiB 边界修复的固定镜像"
bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh 2>&1 | tee "${build_log}"

echo "[2/2] 重新启动并执行 SandboxFusion 安全预检"
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

echo "SandboxFusion 已在宿主机启动，marker：${marker}"
echo "下一步请进入使用 --network host 启动的训练容器，然后运行："
echo "  bash scripts/optimizer_geometry/start_code_grpo_in_container.sh"
