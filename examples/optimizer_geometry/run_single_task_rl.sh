#!/usr/bin/env bash
# Run one full M2RL task epoch at seed 42 under AdamW, vanilla SGD, and Muon RL.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh}"
TASK="${TASK:-}"
case "${TASK,,}" in
  math) TASK=math ;;
  code|coding) TASK=code ;;
  science) TASK=science ;;
  *) echo "Set TASK to math, code (or coding), or science." >&2; exit 2 ;;
esac

RL_ALGORITHM="${RL_ALGORITHM:-grpo}"
case "${RL_ALGORITHM}" in
  grpo|ppo) ;;
  *) echo "RL_ALGORITHM must be grpo or ppo." >&2; exit 2 ;;
esac

CONFIG_ROOT="${SINGLE_TASK_CONFIG_ROOT:-${SLIME_DIR}/data/m2rl/single_task}"
MANIFEST="${CONFIG_ROOT}/${TASK}/${TASK}_on_policy.yaml"
if [[ "${TASK}" == "math" ]]; then
  DEFAULT_EVAL="${CONFIG_ROOT}/math/math_eval_aime24_math500.yaml"
else
  DEFAULT_EVAL="${CONFIG_ROOT}/${TASK}/${TASK}_eval.yaml"
fi
EVAL="${EVAL_CONFIG:-}"
if [[ -z "${EVAL}" && -s "${DEFAULT_EVAL}" ]]; then
  EVAL="${DEFAULT_EVAL}"
fi
INDEX="${CONFIG_ROOT}/single_task_index.json"
for path in "${MANIFEST}" "${INDEX}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing prepared single-task artifact: ${path}" >&2
    echo "Run: bash ${SCRIPT_DIR}/prepare_single_task_dataset.sh" >&2
    exit 2
  fi
done
if [[ -n "${EVAL}" && ! -s "${EVAL}" ]]; then
  echo "Evaluation config does not exist or is empty: ${EVAL}" >&2
  exit 2
fi
if [[ -z "${REQUIRE_EVAL+x}" ]]; then
  REQUIRE_EVAL=1
fi
case "${REQUIRE_EVAL}" in
  0|1) ;;
  *) echo "REQUIRE_EVAL must be 0 or 1." >&2; exit 2 ;;
esac
if [[ "${REQUIRE_EVAL}" == "1" && -z "${EVAL}" ]]; then
  echo "No independent evaluation config is ready for TASK=${TASK}." >&2
  if [[ "${TASK}" == "code" ]]; then
    echo "Prepare LiveCodeBench with examples/optimizer_geometry/prepare_livecodebench_eval.py." >&2
  else
    echo "For science, accept GPQA access, export HF_TOKEN, then rerun preparation with REQUIRE_GPQA=1." >&2
  fi
  exit 2
fi

read -r -a OPTIMIZER_LIST <<< "${OPTIMIZERS:-adamw sgd muon}"
if [[ -n "${SEEDS+x}" ]]; then
  read -r -a REQUESTED_SEEDS <<< "${SEEDS}"
  if (( ${#REQUESTED_SEEDS[@]} != 1 )) || [[ "${REQUESTED_SEEDS[0]}" != "42" ]]; then
    echo "The single-task RL paper script requires exactly SEEDS=42." >&2
    exit 2
  fi
fi
if [[ -n "${SEED+x}" && "${SEED}" != "42" ]]; then
  echo "The single-task RL paper script requires SEED=42." >&2
  exit 2
fi
if [[ "${NUM_EPOCH:-1}" != "1" ]]; then
  echo "The single-task RL paper script trains exactly one usable dataset epoch (NUM_EPOCH=1)." >&2
  exit 2
fi
if [[ -n "${NUM_ROLLOUT:-}" || -n "${TARGET_PROMPT_BUDGET:-}" ]]; then
  echo "Single-task RL dataset-epoch runs do not accept NUM_ROLLOUT or TARGET_PROMPT_BUDGET." >&2
  exit 2
fi
if [[ "${MAX_RESPONSE_LEN:-8192}" != "8192" ]]; then
  echo "The single-task GRPO/PPO scripts require training MAX_RESPONSE_LEN=8192." >&2
  exit 2
fi
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
if ! [[ "${MAX_PROMPT_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PROMPT_LEN must be a positive integer." >&2
  exit 2
fi
REQUIRED_MAX_TOKENS_PER_GPU=$((MAX_PROMPT_LEN + 8192))
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${REQUIRED_MAX_TOKENS_PER_GPU}}"
if ! [[ "${MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || \
   (( MAX_TOKENS_PER_GPU < REQUIRED_MAX_TOKENS_PER_GPU )); then
  echo "MAX_TOKENS_PER_GPU must be at least prompt+response=${REQUIRED_MAX_TOKENS_PER_GPU}." >&2
  exit 2
fi
if [[ "${EVAL_INTERVAL:-50}" != "50" ]]; then
  echo "The frozen single-task GRPO/PPO scripts require EVAL_INTERVAL=50 optimizer updates." >&2
  exit 2
fi
if [[ "${SAVE_INTERVAL:-100}" != "100" ]]; then
  echo "The frozen single-task GRPO/PPO scripts require SAVE_INTERVAL=100 optimizer updates." >&2
  exit 2
fi
export NUM_EPOCH=1
export MAX_PROMPT_LEN
export MAX_RESPONSE_LEN=8192
export MAX_TOKENS_PER_GPU
export EVAL_INTERVAL=50
export SAVE_INTERVAL=100
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-12}"
if [[ "${TASK}" == "math" ]]; then
  export EVAL_MAX_RESPONSE_LEN=32768
  export EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-48}"
fi
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-zsqzz}"
export WANDB_PROJECT="${WANDB_PROJECT:-iclr2027-opd-geometry}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SLIME_DIR}/outputs/qwen3_1.7b_single_task}"
export REWARD_CONFIG="${REWARD_CONFIG:-${SCRIPT_DIR}/configs/rewards.example.yaml}"
export EXPERIMENT_DATA_INDEX="${INDEX}"

for optimizer in "${OPTIMIZER_LIST[@]}"; do
  if [[ "${TASK}" == "math" ]]; then
    response_suffix="trainr${MAX_RESPONSE_LEN}_evalr${EVAL_MAX_RESPONSE_LEN}"
  else
    response_suffix="trainr${MAX_RESPONSE_LEN}"
  fi
  run_name="qwen3_1.7b_${TASK}_${RL_ALGORITHM}_${optimizer}_${BATCH_PROFILE:-responsive16}_${response_suffix}_seed42"
  echo "Launching RL task=${TASK} algorithm=${RL_ALGORITHM} optimizer=${optimizer} seed=42 run=${run_name}"
  TASK="${TASK}" \
  ALGORITHM="${RL_ALGORITHM}" \
  OPTIMIZER="${optimizer}" \
  SEED=42 \
  DATA_MANIFEST="${MANIFEST}" \
  EVAL_CONFIG="${EVAL}" \
  RUN_NAME="${run_name}" \
  WANDB_GROUP="qwen3_1.7b_${TASK}_${RL_ALGORITHM}" \
    bash "${LAUNCHER}"
done
