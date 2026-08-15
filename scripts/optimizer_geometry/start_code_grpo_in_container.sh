#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
requested_repo=/mnt/data_from_server2/siqizhu4/iclr2027/slime_opd_geometry
workspace_repo="$(cd -- "${script_dir}/../.." && pwd)"
if [[ -n "${SLIME_DIR:-}" ]]; then
  repo="${SLIME_DIR}"
elif [[ -f "${requested_repo}/examples/optimizer_geometry/run_single_task_rl.sh" ]]; then
  repo="${requested_repo}"
else
  repo="${workspace_repo}"
fi
single_task_rl="${repo}/examples/optimizer_geometry/run_single_task_rl.sh"
marker="${M2RL_SANDBOX_PREFLIGHT_MARKER:-${repo}/data/m2rl/sandbox/sandboxfusion_preflight.json}"
sandbox_url="${SANDBOXFUSION_BASE_URL:-http://127.0.0.1:8080}"
log_dir="${repo}/outputs/qwen3_1.7b_single_task/.launcher_logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
train_log="${log_dir}/qwen3_1.7b_code_grpo_adamw_responsive16_trainr8192_seed42.${timestamp}.log"

if [[ ! -e /.dockerenv ]] && [[ ! -e /run/.containerenv ]]; then
  echo "本脚本只能在训练容器内运行。" >&2
  exit 2
fi
for command_name in python3 ray tmux; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "训练容器缺少命令：${command_name}" >&2
    exit 2
  fi
done
if [[ ! -f "${single_task_rl}" ]]; then
  echo "找不到现有训练入口：${single_task_rl}" >&2
  echo "请将仓库挂载到训练容器，或通过 SLIME_DIR 指定容器内仓库路径。" >&2
  exit 2
fi

python3 - "${marker}" "${sandbox_url%/}/v1/ping" <<'PY'
import json
import sys
import urllib.request

marker_path, ping_url = sys.argv[1:]
with open(marker_path, encoding="utf-8") as stream:
    marker = json.load(stream)
if marker.get("safe") is not True:
    raise SystemExit(f"SandboxFusion marker is not safe: {marker_path}")
with urllib.request.urlopen(ping_url, timeout=5) as response:
    pong = json.load(response)
if pong != "pong":
    raise SystemExit(f"Unexpected SandboxFusion ping response: {pong!r}")
print(f"SandboxFusion ready: {ping_url}; marker safe=true")
PY
echo "沿用现有训练入口：${single_task_rl}"

mkdir -p "${log_dir}"
if tmux has-session -t grpo_code_adamw 2>/dev/null; then
  echo "tmux 会话 grpo_code_adamw 已存在，拒绝重复启动。" >&2
  exit 2
fi

tmux new-session -d -s grpo_code_adamw -c "${repo}" \
  -e "SLIME_DIR=${repo}" \
  -e "SINGLE_TASK_RL_SCRIPT=${single_task_rl}" \
  -e "SANDBOXFUSION_BASE_URL=${sandbox_url%/}" \
  -e "M2RL_SANDBOX_PREFLIGHT_MARKER=${marker}" \
  "exec bash '${script_dir}/launch_grpo_code_adamw.sh' >'${train_log}' 2>&1"

sleep 8
if ! tmux has-session -t grpo_code_adamw 2>/dev/null; then
  echo "训练会话提前退出，末尾日志如下：" >&2
  tail -n 100 "${train_log}" >&2
  exit 1
fi

echo "Coding GRPO 已在训练容器内启动。"
echo "训练日志：${train_log}"
echo "查看：tail -f '${train_log}'"
echo "进入：tmux attach -t grpo_code_adamw"
