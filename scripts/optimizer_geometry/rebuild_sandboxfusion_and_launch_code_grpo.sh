#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/disk2_from_server2/siqizhu4/iclr2027/slime_opd_geometry
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
marker=/tmp/slime-sandbox/sandboxfusion_preflight.json
log_dir="${repo}/outputs/qwen3_1.7b_single_task/.launcher_logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
build_log="${log_dir}/sandboxfusion.rebuild.${timestamp}.log"
start_log="${log_dir}/sandboxfusion.restart.${timestamp}.log"
train_log="${log_dir}/qwen3_1.7b_code_grpo_adamw_responsive16_trainr8192_seed42.retry2.${timestamp}.log"

cd "${repo}"
mkdir -p "$(dirname "${marker}")" "${log_dir}"

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "当前终端无法访问 Docker daemon；请在你刚才显示 docker 组权限的宿主机终端运行本脚本。" >&2
  exit 2
fi

echo "[1/3] 重建包含 LiveCodeBench 128 MiB/16 MiB 边界修复的固定镜像"
bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh 2>&1 | tee "${build_log}"

echo "[2/3] 重新启动并执行 SandboxFusion 安全预检"
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

echo "[3/3] 启动 Coding GRPO AdamW（GPU 2,3,5,7；独立 Ray）"
if tmux has-session -t grpo_code_adamw 2>/dev/null; then
  echo "tmux 会话 grpo_code_adamw 已存在，拒绝重复启动。" >&2
  exit 2
fi
tmux new-session -d -s grpo_code_adamw -c "${repo}" \
  "exec bash '${script_dir}/launch_grpo_code_adamw.sh' >'${train_log}' 2>&1"

sleep 8
if ! tmux has-session -t grpo_code_adamw 2>/dev/null; then
  echo "训练会话提前退出，末尾日志如下：" >&2
  tail -n 100 "${train_log}" >&2
  exit 1
fi

echo "Coding GRPO 已启动。"
echo "训练日志：${train_log}"
echo "查看：tail -f '${train_log}'"
echo "进入：tmux attach -t grpo_code_adamw"
