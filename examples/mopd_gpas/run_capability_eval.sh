#!/usr/bin/env bash
# Evaluate a final run or a frozen reference model on the four capability suites.
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
MOPD_DATA_ROOT="${MOPD_DATA_ROOT:-${SLIME_ROOT}/data/m2rl}"
export MOPD_DATA_ROOT
export MOPD_HF_CHECKPOINT="${MOPD_HF_CHECKPOINT:-/workspace/dev/checkpoints/Qwen3-1.7B}"
export MOPD_TEACHER_HF_ROOT="${MOPD_TEACHER_HF_ROOT:-${SLIME_ROOT}/local/mopd_assets/models/teachers_hf}"
export MOPD_QWEN3_4B="${MOPD_QWEN3_4B:-${SLIME_ROOT}/local/mopd_assets/models/qwen3-4b}"

TARGET="${1:-}"
case "${TARGET}" in
  uniform|gpas|cost_gpas|raw_noise|loss_gap|std_mopd|d3_mopd|open_mopd|initial_student|teacher_math|teacher_if|teacher_qwen3_4b) ;;
  *) echo "Usage: $0 {initial_student|teacher_math|teacher_if|teacher_qwen3_4b|uniform|gpas|cost_gpas|raw_noise|loss_gap|std_mopd|d3_mopd|open_mopd}" >&2; exit 2 ;;
esac

OUTPUT_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_v4}"
GENERATED="${MOPD_GENERATED_DIR:-${SLIME_ROOT}/local/mopd_generated}"
DATA_MANIFEST="${GENERATED}/train.yaml"
PROTOCOL="${GENERATED}/protocol.json"
EVAL_CONFIG="${EXAMPLE_DIR}/configs/capability_eval.yaml"
REWARD_CONFIG="${EXAMPLE_DIR}/configs/mopd_capability_rewards.yaml"
EVAL_INDICES=(
  "${MOPD_DATA_ROOT}/eval/m2rl_online/eval_data_index.json"
  "${MOPD_DATA_ROOT}/single_task/code/livecodebench_index_v6.json"
)
TARGET_RESPONSES="${MOPD_CAPABILITY_RESPONSE:-32000}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_SCALE=1p7b

case "${TARGET}" in
  uniform|gpas|cost_gpas|raw_noise|loss_gap|std_mopd|d3_mopd|open_mopd)
    RUN_DIR="${OUTPUT_ROOT}/${TARGET}-seed42"
    INDEX="${RUN_DIR}/checkpoints/mopd_checkpoint_index.json"
    if [[ "${DRY_RUN}" == 1 ]]; then
      MODEL_PATH="${MOPD_HF_CHECKPOINT}"
    else
      [[ -f "${INDEX}" ]] || { echo "Missing checkpoint index ${INDEX}." >&2; exit 2; }
      MODEL_PATH="$(python3 - "${INDEX}" "${TARGET_RESPONSES}" <<'PY'
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
target = int(sys.argv[2])
matches = [row for row in rows if int(row["attempted_responses"]) == target]
if len(matches) != 1 or not matches[0].get("hf_checkpoint"):
    raise SystemExit(f"expected one archived response-{target} HF checkpoint, got {matches}")
print(matches[0]["hf_checkpoint"])
PY
      )"
    fi
    OUTPUT_DIR="${RUN_DIR}/capability_eval/response_${TARGET_RESPONSES}"
    EXPERIMENT_NAME="mopd-${TARGET}-capability-r${TARGET_RESPONSES}"
    ;;
  initial_student)
    MODEL_PATH="${MOPD_HF_CHECKPOINT}"
    OUTPUT_DIR="${OUTPUT_ROOT}/capability_references/initial_student"
    EXPERIMENT_NAME=mopd-capability-initial-student
    ;;
  teacher_math)
    MODEL_PATH="${MOPD_TEACHER_HF_ROOT}/math"
    OUTPUT_DIR="${OUTPUT_ROOT}/capability_references/teacher_math"
    EXPERIMENT_NAME=mopd-capability-teacher-math
    ;;
  teacher_if)
    MODEL_PATH="${MOPD_TEACHER_HF_ROOT}/if"
    OUTPUT_DIR="${OUTPUT_ROOT}/capability_references/teacher_if"
    EXPERIMENT_NAME=mopd-capability-teacher-if
    ;;
  teacher_qwen3_4b)
    MODEL_PATH="${MOPD_QWEN3_4B}"
    OUTPUT_DIR="${OUTPUT_ROOT}/capability_references/teacher_qwen3_4b"
    EXPERIMENT_NAME=mopd-capability-teacher-qwen3-4b
    MODEL_SCALE=4b
    ;;
esac

for path in "${EVAL_CONFIG}" "${DATA_MANIFEST}" "${PROTOCOL}" "${REWARD_CONFIG}" "${EVAL_INDICES[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing required capability input: ${path}" >&2; exit 2; }
done
[[ -f "${MODEL_PATH}/config.json" && -f "${MODEL_PATH}/model.safetensors.index.json" ]] || {
  echo "Incomplete HuggingFace model ${MODEL_PATH}." >&2
  exit 2
}
if [[ "${DRY_RUN}" == 0 && -e "${OUTPUT_DIR}" ]]; then
  echo "Capability output already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

export SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL:-http://127.0.0.1:8080}"
export M2RL_SANDBOX_PREFLIGHT_MARKER="${M2RL_SANDBOX_PREFLIGHT_MARKER:-${SLIME_ROOT}/data/m2rl/sandbox/sandboxfusion_preflight.json}"
export CUDA_VISIBLE_DEVICES="${CAPABILITY_CUDA_VISIBLE_DEVICES:-3}"
export SLIME_ROLLOUT_PORT_BASE="${SLIME_ROLLOUT_PORT_BASE:-24000}"
IFS=, read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#GPU_LIST[@]}" == 1 ]] || { echo "Capability evaluation requires exactly one GPU." >&2; exit 2; }
NUM_GPUS=1
export MODEL_ARGS_ROTARY_BASE=1000000
if [[ "${MODEL_SCALE}" == 4b ]]; then
  source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"
  SGLANG_FRACTION="${SGLANG_MEM_FRACTION:-0.45}"
else
  source "${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh"
  SGLANG_FRACTION="${SGLANG_MEM_FRACTION:-0.7}"
fi

TRAIN_CMD=(
  python3 "${SLIME_ROOT}/train.py"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${MODEL_PATH}" --load "${MODEL_PATH}"
  --no-load-optim --no-load-rng --start-rollout-id 0
  --prompt-data "${DATA_MANIFEST}"
  --data-source-path slime_plugins.m2rl.data_source.MultiTaskRolloutDataSource
  --input-key prompt --label-key label --metadata-key metadata --tool-key tools
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}'
  --rollout-global-dataset
  --num-rollout 0 --rollout-batch-size "${NUM_GPUS}" --global-batch-size "${NUM_GPUS}"
  --n-samples-per-prompt 1
  --eval-config "${EVAL_CONFIG}" --eval-interval 1
  --eval-max-response-len 4096 --eval-max-concurrency "${EVAL_MAX_CONCURRENCY:-48}"
  --m2rl-reward-config "${REWARD_CONFIG}"
  --metrics-output-dir "${OUTPUT_DIR}/metrics"
  --eval-artifact-dir "${OUTPUT_DIR}/eval_artifacts"
  --run-manifest-path "${OUTPUT_DIR}/provenance/run_manifest.json"
  --completion-marker-path "${OUTPUT_DIR}/run_complete.json"
  --experiment-task mopd_capability --experiment-teacher "${TARGET}"
  --experiment-condition capability --experiment-name "${EXPERIMENT_NAME}"
  --experiment-optimizer none --experiment-data-index "${PROTOCOL}"
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.0
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
  --rollout-num-gpus "${NUM_GPUS}" --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "${SGLANG_FRACTION}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-44}"
  --actor-num-nodes 1 --actor-num-gpus-per-node "${NUM_GPUS}" --colocate
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash --seed 42 --sglang-enable-deterministic-inference --log-passrate
)

printf 'Capability command:'; printf ' %q' "${TRAIN_CMD[@]}"; printf '\n'
if [[ "${DRY_RUN}" == 1 ]]; then
  python3 -c '
import slime.backends.megatron_utils.arguments as ma
import slime.utils.arguments as sa
ma.validate_args=lambda args: args
sa.sglang_validate_args=lambda args: args
sa.parse_args()
print("static argument validation: OK")
' "${TRAIN_CMD[@]:2}"
  exit 0
fi

[[ -r "${M2RL_SANDBOX_PREFLIGHT_MARKER}" ]] || {
  echo "Missing readable SandboxFusion preflight marker: ${M2RL_SANDBOX_PREFLIGHT_MARKER}" >&2
  echo "Build and start the audited sandbox with examples/optimizer_geometry/{build,start}_sandboxfusion*.sh." >&2
  exit 2
}
python3 - "${REWARD_CONFIG}" <<'PY'
import sys

from slime_plugins.m2rl.rewards import load_reward_config
from slime_plugins.m2rl.sandbox_security import validate_preflight_marker

route = dict((load_reward_config(sys.argv[1]).get("routes") or {}).get("livecodebench") or {})
url = str(route.get("preflight_url") or route.get("url") or "")
validate_preflight_marker(route, url)
print(f"SandboxFusion attestation: OK ({url})")
PY

python3 "${EXAMPLE_DIR}/provenance.py" start --repo "${SLIME_ROOT}" --run-dir "${OUTPUT_DIR}" \
  --input "${PROTOCOL}" --input "${DATA_MANIFEST}" --input "${EVAL_CONFIG}" --input "${REWARD_CONFIG}" \
  --input "${M2RL_SANDBOX_PREFLIGHT_MARKER}" \
  --checkpoint "${MODEL_PATH}" --source "${EXAMPLE_DIR}" \
  --source "${SLIME_ROOT}/slime_plugins/m2rl/rewards.py" \
  --source "${SLIME_ROOT}/slime_plugins/m2rl/sandbox_security.py" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/actor.py" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/data.py" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/model.py" \
  --source "${SLIME_ROOT}/slime/ray/rollout.py" --source "${SLIME_ROOT}/train.py" \
  -- "${TRAIN_CMD[@]}"

RAY_PID=""
cleanup() {
  if [[ -n "${RAY_PID}" && "${KEEP_RAY:-0}" != 1 ]]; then
    kill -TERM "${RAY_PID}" 2>/dev/null || true
    wait "${RAY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
RAY_ADDRESS="${RAY_ADDRESS:-}"
if [[ -z "${RAY_ADDRESS}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
  GCS_PORT="${RAY_GCS_PORT:-6379}"
  AUX_PORT_BASE="${RAY_AUX_PORT_BASE:-12000}"
  WORKER_PORT_MIN="${RAY_WORKER_PORT_MIN:-$((AUX_PORT_BASE + 16))}"
  WORKER_PORT_MAX="${RAY_WORKER_PORT_MAX:-$((AUX_PORT_BASE + 79))}"
  RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/mopd_capability}"
  mkdir -p "${RAY_TEMP_DIR}"
  ray start --head --node-ip-address "${MASTER_ADDR}" --port "${GCS_PORT}" \
    --num-cpus "${RAY_NUM_CPUS:-8}" --num-gpus "${NUM_GPUS}" --disable-usage-stats \
    --dashboard-host 0.0.0.0 --dashboard-port "${DASHBOARD_PORT}" \
    --dashboard-agent-listen-port "$((AUX_PORT_BASE + 0))" \
    --dashboard-agent-grpc-port "$((AUX_PORT_BASE + 1))" \
    --runtime-env-agent-port "$((AUX_PORT_BASE + 2))" \
    --metrics-export-port "$((AUX_PORT_BASE + 3))" \
    --ray-client-server-port "$((AUX_PORT_BASE + 4))" \
    --object-manager-port "$((AUX_PORT_BASE + 5))" \
    --node-manager-port "$((AUX_PORT_BASE + 6))" \
    --min-worker-port "${WORKER_PORT_MIN}" --max-worker-port "${WORKER_PORT_MAX}" \
    --temp-dir "${RAY_TEMP_DIR}" --block &
  RAY_PID=$!
  RAY_ADDRESS="http://${MASTER_ADDR}:${DASHBOARD_PORT}"
  AGENT_LOG="${RAY_TEMP_DIR}/session_latest/logs/dashboard_agent.log"
  RAY_READY=0
  for _ in $(seq 1 120); do
    kill -0 "${RAY_PID}" 2>/dev/null || { echo "Ray head exited." >&2; exit 1; }
    if [[ -f "${AGENT_LOG}" ]] && grep -q "Dashboard agent http address:" "${AGENT_LOG}" \
      && ray job list --address "${RAY_ADDRESS}" >/dev/null 2>&1; then
      RAY_READY=1
      break
    fi
    sleep 1
  done
  [[ "${RAY_READY}" == 1 ]] || { echo "Ray Jobs API did not become ready." >&2; exit 1; }
fi
RUNTIME_ENV_JSON="$(python3 - "${PYTHONPATH}" <<'PY'
import json, os, sys
keys=(
    "WANDB_API_KEY", "WANDB_BASE_URL", "SANDBOXFUSION_BASE_URL", "M2RL_SANDBOX_PREFLIGHT_MARKER",
    "CUDA_VISIBLE_DEVICES", "MOPD_DATA_ROOT", "SLIME_ROLLOUT_PORT_BASE",
)
env={"PYTHONPATH":sys.argv[1],"CUDA_DEVICE_MAX_CONNECTIONS":"1","PYTHONUNBUFFERED":"1"}
env.update({key:os.environ[key] for key in keys if os.environ.get(key)})
print(json.dumps({"env_vars":env}))
PY
)"
set +e
ray job submit --address "${RAY_ADDRESS}" --runtime-env-json "${RUNTIME_ENV_JSON}" -- "${TRAIN_CMD[@]}"
EXIT_CODE=$?
set -e
python3 "${EXAMPLE_DIR}/provenance.py" finish --run-dir "${OUTPUT_DIR}" --exit-code "${EXIT_CODE}"
exit "${EXIT_CODE}"
