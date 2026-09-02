#!/usr/bin/env bash
# Evaluate one final MOPD checkpoint on MATH-500, LCB, IFBench, and GPQA-Diamond.
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
MOPD_DATA_ROOT="${MOPD_DATA_ROOT:-${SLIME_ROOT}/data/m2rl}"
export MOPD_DATA_ROOT

CONFIG_ID="${1:-}"
case "${CONFIG_ID}" in
  uniform_k1_conventional|uniform_k1_taskwise|gpas_k1_taskwise|cost_gpas_k1_taskwise|uniform_k2_taskwise|cost_gpas_k2_taskwise|all_k4_taskwise|all_k4_conventional) ;;
  *) echo "Usage: $0 CONFIG_ID" >&2; exit 2 ;;
esac
OUTPUT_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_64k_v3}"
EVAL_CONFIG="${EXAMPLE_DIR}/configs/capability_eval.yaml"
GENERATED="${MOPD_GENERATED_DIR:-${EXAMPLE_DIR}/generated/mopd}"
DATA_MANIFEST="${GENERATED}/train.yaml"
PROTOCOL="${GENERATED}/protocol.json"
REWARD_CONFIG="${EXAMPLE_DIR}/configs/mopd_capability_rewards.yaml"
EVAL_INDICES=(
  "${MOPD_DATA_ROOT}/eval/m2rl_online/eval_data_index.json"
  "${MOPD_DATA_ROOT}/single_task/code/livecodebench_index_v6.json"
)
TASK_NAME=mopd_capability
TEACHER_NAME=four_frozen_grpo_teachers

RUN_DIR="${OUTPUT_ROOT}/${CONFIG_ID}-seed42"
LOAD_ROOT="${RUN_DIR}/checkpoints"
INDEX="${LOAD_ROOT}/mopd_checkpoint_index.json"
DRY_RUN="${DRY_RUN:-0}"
TARGET_RESPONSES="${MOPD_CAPABILITY_RESPONSE:-64000}"
if [[ "${DRY_RUN}" == 1 ]]; then
  LOAD_STEP=0
else
  [[ -f "${INDEX}" ]] || { echo "Missing checkpoint index ${INDEX}." >&2; exit 2; }
  LOAD_STEP="$(python3 - "${INDEX}" "${TARGET_RESPONSES}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
target = int(sys.argv[2])
rows = json.loads(path.read_text(encoding="utf-8"))
matches = [row for row in rows if int(row["attempted_responses"]) == target]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one response-{target} checkpoint in {path}, got {matches}")
print(int(matches[0]["rollout_id"]))
PY
  )"
fi
OUTPUT_DIR="${RUN_DIR}/capability_eval/response_${TARGET_RESPONSES}"
for path in "${EVAL_CONFIG}" "${DATA_MANIFEST}" "${PROTOCOL}" "${REWARD_CONFIG}" "${EVAL_INDICES[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing required capability input: ${path}" >&2; exit 2; }
done
if [[ "${DRY_RUN}" == 0 ]]; then
  for anchor in common.pt .metadata; do
    [[ -f "${LOAD_ROOT}/iter_$(printf '%07d' "${LOAD_STEP}")/${anchor}" ]] || {
      echo "Missing response-${TARGET_RESPONSES} checkpoint anchor ${anchor} under ${LOAD_ROOT}." >&2; exit 2;
    }
  done
fi

export SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL:-http://127.0.0.1:8080}"
export CUDA_VISIBLE_DEVICES="${CAPABILITY_CUDA_VISIBLE_DEVICES:-3}"
IFS=, read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#GPU_LIST[@]}" == 1 ]] || {
  echo "Capability evaluation requires exactly one GPU, got ${CUDA_VISIBLE_DEVICES}." >&2; exit 2;
}
declare -A GPU_SET=()
for gpu in "${GPU_LIST[@]}"; do
  [[ -n "${gpu}" && -z "${GPU_SET[${gpu}]+x}" ]] || {
    echo "Capability GPU IDs must be nonempty and unique: ${CUDA_VISIBLE_DEVICES}." >&2; exit 2;
  }
  GPU_SET["${gpu}"]=1
done
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${DRY_RUN}" == 0 ]]; then
  python3 "${EXAMPLE_DIR}/verify_hardware.py" --gpu-ids "${CUDA_VISIBLE_DEVICES}"
fi
HF_CHECKPOINT="${MOPD_HF_CHECKPOINT:-/workspace/dev/checkpoints/Qwen3-1.7B}"
MODEL_CONFIG="${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh"
export MODEL_ARGS_ROTARY_BASE=1000000
source "${MODEL_CONFIG}"

TRAIN_CMD=(
  python3 "${SLIME_ROOT}/train.py"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}" --load "${LOAD_ROOT}" --ckpt-step "${LOAD_STEP}"
  --no-load-optim --no-load-rng --start-rollout-id 0
  --prompt-data "${DATA_MANIFEST}"
  --data-source-path slime_plugins.m2rl.data_source.MultiTaskRolloutDataSource
  --input-key prompt --label-key label --metadata-key metadata --tool-key tools
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}'
  --rollout-global-dataset
  --num-rollout 0 --rollout-batch-size "${NUM_GPUS}" --global-batch-size "${NUM_GPUS}"
  --n-samples-per-prompt 1
  --eval-config "${EVAL_CONFIG}" --eval-interval 1
  --eval-max-response-len 8192 --eval-max-concurrency "${EVAL_MAX_CONCURRENCY:-48}"
  --m2rl-reward-config "${REWARD_CONFIG}"
  --metrics-output-dir "${OUTPUT_DIR}/metrics"
  --eval-artifact-dir "${OUTPUT_DIR}/eval_artifacts"
  --run-manifest-path "${OUTPUT_DIR}/provenance/run_manifest.json"
  --completion-marker-path "${OUTPUT_DIR}/run_complete.json"
  --experiment-task "${TASK_NAME}" --experiment-teacher "${TEACHER_NAME}"
  --experiment-condition capability --experiment-name "mopd-${CONFIG_ID}-capability-r${TARGET_RESPONSES}"
  --experiment-optimizer none --experiment-data-index "${PROTOCOL}"
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.0
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
  --rollout-num-gpus "${NUM_GPUS}" --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.7}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-44}"
  --actor-num-nodes 1 --actor-num-gpus-per-node "${NUM_GPUS}" --colocate
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash --seed 42 --sglang-enable-deterministic-inference --log-passrate
)
if [[ "${USE_WANDB:-1}" == 1 ]]; then
  TRAIN_CMD+=(
    --use-wandb --wandb-mode "${WANDB_MODE:-online}" --wandb-dir "${OUTPUT_DIR}/wandb"
    --wandb-team "${WANDB_ENTITY:-zsqzz}" --wandb-project "${WANDB_PROJECT:-iclr2027-mopd-gpas-64k}"
    --wandb-group "mopd-capability-64k" \
    --wandb-run-name "mopd-${CONFIG_ID}-capability-r${TARGET_RESPONSES}"
    --wandb-run-id-file "${OUTPUT_DIR}/wandb_run_id.txt" --disable-wandb-random-suffix
  )
fi

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
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Capability output already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

PROVENANCE=(
  start --repo "${SLIME_ROOT}" --run-dir "${OUTPUT_DIR}"
  --input "${PROTOCOL}" --input "${DATA_MANIFEST}" --input "${EVAL_CONFIG}"
  --input "${REWARD_CONFIG}" --input "${MODEL_CONFIG}"
  --checkpoint "${HF_CHECKPOINT}" --checkpoint "${LOAD_ROOT}/iter_$(printf '%07d' "${LOAD_STEP}")"
  --source "${EXAMPLE_DIR}"
  --source "${SLIME_ROOT}/slime_plugins/m2rl/rewards.py"
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/actor.py"
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/data.py"
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/model.py"
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/optimizer_factory.py"
  --source "${SLIME_ROOT}/slime/ray/rollout.py"
  --source "${SLIME_ROOT}/slime/utils/arguments.py"
  --source "${SLIME_ROOT}/slime/utils/logging_utils.py"
  --source "${SLIME_ROOT}/slime/utils/metric_utils.py"
  --source "${SLIME_ROOT}/slime/utils/wandb_utils.py"
  --source "${SLIME_ROOT}/train.py"
)
for path in "${EVAL_INDICES[@]}"; do PROVENANCE+=(--input "${path}"); done
python3 "${EXAMPLE_DIR}/provenance.py" "${PROVENANCE[@]}" "${TRAIN_CMD[@]}"

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
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host 0.0.0.0 --dashboard-port "${DASHBOARD_PORT}" --block &
  RAY_PID=$!
  RAY_ADDRESS="http://${MASTER_ADDR}:${DASHBOARD_PORT}"
  for _ in $(seq 1 120); do
    ray job list --address "${RAY_ADDRESS}" >/dev/null 2>&1 && break
    kill -0 "${RAY_PID}" 2>/dev/null || { echo "Ray head exited." >&2; exit 1; }
    sleep 1
  done
  ray job list --address "${RAY_ADDRESS}" >/dev/null 2>&1 || { echo "Ray Jobs API not ready." >&2; exit 1; }
fi
RUNTIME_ENV_JSON="$(python3 - "${PYTHONPATH}" <<'PY'
import json, os, sys
keys=("WANDB_API_KEY","WANDB_BASE_URL","SANDBOXFUSION_BASE_URL","CUDA_VISIBLE_DEVICES","MOPD_DATA_ROOT","MOPD_HF_CHECKPOINT")
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
