#!/usr/bin/env bash
# Evaluate one Qwen3-1.7B checkpoint on a prepared math or code eval config.

set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
export PYTHONPATH="${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PROJECT_CHECKPOINT_ROOT="$(cd -- "${SLIME_DIR}/.." && pwd)/checkpoints"
export SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL:-http://127.0.0.1:8080}"
SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL%/}"
export SANDBOXFUSION_BASE_URL
export M2RL_SANDBOX_PREFLIGHT_MARKER="${M2RL_SANDBOX_PREFLIGHT_MARKER:-${SLIME_DIR}/data/m2rl/sandbox/sandboxfusion_preflight.json}"

MODEL_CONFIG="${MODEL_CONFIG:-${SLIME_DIR}/scripts/models/qwen3-1.7B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${PROJECT_CHECKPOINT_ROOT}/Qwen3-1.7B}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:?Set LOAD_CHECKPOINT to the trained torch_dist checkpoint directory}"
DATA_MANIFEST="${DATA_MANIFEST:?Set DATA_MANIFEST to the task on-policy manifest}"
EVAL_CONFIG="${EVAL_CONFIG:?Set EVAL_CONFIG to the task eval YAML}"
REWARD_CONFIG="${REWARD_CONFIG:-${EXAMPLE_DIR}/configs/rewards.example.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOAD_CHECKPOINT%/}/evaluation}"
RUN_NAME="${RUN_NAME:-$(basename -- "${LOAD_CHECKPOINT%/}")_evaluation}"
METRICS_DIR="${METRICS_DIR:-${OUTPUT_DIR}/metrics}"
EVAL_ARTIFACT_DIR="${EVAL_ARTIFACT_DIR:-${OUTPUT_DIR}/eval_artifacts}"
RUN_MANIFEST_PATH="${RUN_MANIFEST_PATH:-${OUTPUT_DIR}/provenance/run_manifest.json}"
COMPLETION_MARKER_PATH="${COMPLETION_MARKER_PATH:-${OUTPUT_DIR}/run_complete.json}"
USE_WANDB="${USE_WANDB:-1}"
FRESH_EVAL="${FRESH_EVAL:-1}"
DRY_RUN="${DRY_RUN:-0}"
case "${USE_WANDB}:${FRESH_EVAL}:${DRY_RUN}" in
  [01]:[01]:[01]) ;;
  *) echo "USE_WANDB, FRESH_EVAL, and DRY_RUN must each be 0 or 1." >&2; exit 2 ;;
esac

NUM_GPUS="${NUM_GPUS:-4}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${NUM_GPUS}}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-44}"
case "${TASK:-evaluation}" in
  math) DEFAULT_EVAL_MAX_RESPONSE_LEN=32768 ;;
  *) DEFAULT_EVAL_MAX_RESPONSE_LEN=16384 ;;
esac
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-${DEFAULT_EVAL_MAX_RESPONSE_LEN}}"
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-48}"
if ! [[ "${SGLANG_MAX_RUNNING_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SGLANG_MAX_RUNNING_REQUESTS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${EVAL_MAX_RESPONSE_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAX_RESPONSE_LEN must be a positive integer." >&2
  exit 2
fi
if ! [[ "${EVAL_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAX_CONCURRENCY must be a positive integer." >&2
  exit 2
fi
case "${GPU_PREFLIGHT:-1}" in
  0|1) ;;
  *) echo "GPU_PREFLIGHT must be 0 or 1." >&2; exit 2 ;;
esac

VALIDATE_ARGS=(
  --optimizer adamw
  --algorithm grpo
  --manifest "${DATA_MANIFEST}"
  --model-config "${MODEL_CONFIG}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load-checkpoint "${LOAD_CHECKPOINT}"
  --reward-config "${REWARD_CONFIG}"
  --eval-config "${EVAL_CONFIG}"
  --expected-eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
)
if [[ "${CHECK_RUNTIME_DEPS:-1}" == "1" ]]; then
  VALIDATE_ARGS+=(--check-runtime-deps)
fi
python3 "${EXAMPLE_DIR}/validate_experiment.py" "${VALIDATE_ARGS[@]}"

if [[ "${GPU_PREFLIGHT:-1}" == "1" && "${DRY_RUN}" != "1" ]]; then
  EVAL_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  if [[ -z "${EVAL_DEVICES}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES (physical nvidia-smi indices), or set GPU_PREFLIGHT=0 explicitly." >&2
    exit 2
  fi
  IFS=',' read -r -a EVAL_GPU_LIST <<< "${EVAL_DEVICES}"
  if [[ "${#EVAL_GPU_LIST[@]}" -ne "${NUM_GPUS}" ]]; then
    echo "NUM_GPUS=${NUM_GPUS} does not match CUDA_VISIBLE_DEVICES=${EVAL_DEVICES}." >&2
    exit 2
  fi
  python3 "${EXAMPLE_DIR}/gpu_preflight.py" \
    --devices "${EVAL_DEVICES}" \
    --role evaluation \
    --min-free-mib "${EVAL_MIN_FREE_MIB:-75000}" \
    --min-free-fraction "${EVAL_MIN_FREE_FRACTION:-0.90}"
fi

source "${MODEL_CONFIG}"
TRAIN_CMD=(
  python3 "${SLIME_DIR}/train.py"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load "${LOAD_CHECKPOINT}"
  --no-load-optim
  --no-load-rng
  --prompt-data "${DATA_MANIFEST}"
  --data-source-path slime_plugins.m2rl.data_source.MultiTaskRolloutDataSource
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --tool-key tools
  --apply-chat-template
  --num-rollout 0
  --rollout-batch-size 1
  --n-samples-per-prompt 1
  --global-batch-size 1
  --eval-config "${EVAL_CONFIG}"
  --eval-interval 1
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --eval-max-concurrency "${EVAL_MAX_CONCURRENCY}"
  --m2rl-reward-config "${REWARD_CONFIG}"
  --custom-eval-rollout-log-function-path slime_plugins.geometry.forgetting.log_eval_and_forgetting
  --forgetting-output-dir "${OUTPUT_DIR}"
  --metrics-output-dir "${METRICS_DIR}"
  --eval-artifact-dir "${EVAL_ARTIFACT_DIR}"
  --run-manifest-path "${RUN_MANIFEST_PATH}"
  --completion-marker-path "${COMPLETION_MARKER_PATH}"
  --experiment-task "${TASK:-evaluation}"
  --experiment-teacher "${TEACHER_NAME:-none}"
  --experiment-condition evaluation
  --experiment-name "${RUN_NAME}"
  --experiment-optimizer none
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.0
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.7}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  --colocate
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --seed "${SEED:-42}"
  --sglang-enable-deterministic-inference
  --log-passrate
)
if [[ "${TP_SIZE}" -gt 1 ]]; then
  TRAIN_CMD+=(--sequence-parallel)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a USER_EXTRA_ARGS <<< "${EXTRA_ARGS}"
  TRAIN_CMD+=("${USER_EXTRA_ARGS[@]}")
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_MODE_VALUE="${WANDB_MODE:-online}"
  TRAIN_CMD+=(
    --use-wandb
    --wandb-mode "${WANDB_MODE_VALUE}"
    --wandb-dir "${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
    --wandb-team "${WANDB_ENTITY:-zsqzz}"
    --wandb-project "${WANDB_PROJECT:-iclr2027-opd-geometry}"
    --wandb-group "${WANDB_GROUP:-${TASK:-evaluation}_final_eval}"
    --wandb-run-name "${WANDB_RUN_NAME:-${RUN_NAME}}"
    --wandb-run-id-file "${WANDB_RUN_ID_FILE:-${OUTPUT_DIR}/wandb_run_id.txt}"
    --disable-wandb-random-suffix
  )
fi

printf 'Evaluation command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi
if [[ "${FRESH_EVAL}" == "1" && -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing a fresh evaluation in non-empty OUTPUT_DIR=${OUTPUT_DIR}; choose another directory or set FRESH_EVAL=0." >&2
  exit 2
fi

PROVENANCE_ARGS=(
  start
  --repo "${SLIME_DIR}"
  --run-dir "${OUTPUT_DIR}"
  --input "${DATA_MANIFEST}"
  --input "${EVAL_CONFIG}"
  --input "${MODEL_CONFIG}"
  --input "${REWARD_CONFIG}"
  --checkpoint "${HF_CHECKPOINT}"
  --checkpoint "${LOAD_CHECKPOINT}"
)
AUTO_EVAL_INDEX="$(dirname -- "${EVAL_CONFIG}")/livecodebench_index.json"
if [[ -f "${AUTO_EVAL_INDEX}" ]]; then
  PROVENANCE_ARGS+=(--input "${AUTO_EVAL_INDEX}")
fi
if [[ -n "${EXPERIMENT_DATA_INDEX:-}" && -f "${EXPERIMENT_DATA_INDEX}" ]]; then
  PROVENANCE_ARGS+=(--input "${EXPERIMENT_DATA_INDEX}")
fi
if [[ -n "${EXPERIMENT_EVAL_INDEX:-}" && -f "${EXPERIMENT_EVAL_INDEX}" ]]; then
  PROVENANCE_ARGS+=(--input "${EXPERIMENT_EVAL_INDEX}")
fi
if [[ -f "${M2RL_SANDBOX_PREFLIGHT_MARKER}" ]]; then
  PROVENANCE_ARGS+=(--input "${M2RL_SANDBOX_PREFLIGHT_MARKER}")
fi
if [[ "${FRESH_EVAL}" == "0" ]]; then
  PROVENANCE_ARGS+=(--resume)
fi
python3 "${EXAMPLE_DIR}/run_provenance.py" "${PROVENANCE_ARGS[@]}" "${TRAIN_CMD[@]}"

RAY_ADDRESS="${RAY_ADDRESS:-}"
RAY_START_PID=""
# `ray stop` scans every local Ray process.  A blocking supervisor owns this
# node's children and its SIGTERM handler shuts down only that node.
cleanup() {
  if [[ -n "${RAY_START_PID}" && "${KEEP_RAY:-0}" != "1" ]]; then
    kill -TERM "${RAY_START_PID}" 2>/dev/null || true
    wait "${RAY_START_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if [[ -z "${RAY_ADDRESS}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
  ray start \
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT}" \
    --block &
  RAY_START_PID=$!
  RAY_ADDRESS="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"
  RAY_READY=0
  for ((attempt = 1; attempt <= 60; attempt++)); do
    if ! kill -0 "${RAY_START_PID}" 2>/dev/null; then
      echo "Ray head process exited before its Jobs API became ready at ${RAY_ADDRESS}." >&2
      wait "${RAY_START_PID}" 2>/dev/null || true
      exit 1
    fi
    if ray job list --address "${RAY_ADDRESS}" >/dev/null 2>&1; then
      RAY_READY=1
      break
    fi
    sleep 1
  done
  if [[ "${RAY_READY}" != "1" ]]; then
    echo "Timed out waiting for the Ray Jobs API at ${RAY_ADDRESS}." >&2
    exit 1
  fi
fi

MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
RUNTIME_PYTHONPATH="${SLIME_DIR}:${MEGATRON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_ENV_JSON="$(python3 -c 'import json,os,sys; env={"PYTHONPATH": sys.argv[1], "CUDA_DEVICE_MAX_CONNECTIONS": "1", "PYTHONUNBUFFERED": "1"}; keys=("M2RL_EVAL_DIR", "SANDBOXFUSION_BASE_URL", "M2RL_SANDBOX_PREFLIGHT_MARKER", "WANDB_API_KEY", "WANDB_BASE_URL"); env.update({key: os.environ[key] for key in keys if os.environ.get(key)}); print(json.dumps({"env_vars": env}))' "${RUNTIME_PYTHONPATH}")"

set +e
ray job submit \
  --address "${RAY_ADDRESS}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_CMD[@]}"
EVAL_EXIT_CODE=$?
set -e
python3 "${EXAMPLE_DIR}/run_provenance.py" finish --run-dir "${OUTPUT_DIR}" --exit-code "${EVAL_EXIT_CODE}"
exit "${EVAL_EXIT_CODE}"
