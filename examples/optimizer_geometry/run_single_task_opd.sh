#!/usr/bin/env bash
# Run one M2RL task with Qwen3-1.7B student, Qwen3-8B teacher, and each of
# AdamW, vanilla SGD, and Muon for pure on-policy distillation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh}"
TASK="${TASK:-}"
case "${TASK,,}" in
  math) TASK=math; REQUIRED_EVAL_MAX_RESPONSE_LEN=32768 ;;
  code|coding) TASK=code; REQUIRED_EVAL_MAX_RESPONSE_LEN=16384 ;;
  science) TASK=science; REQUIRED_EVAL_MAX_RESPONSE_LEN=16384 ;;
  *) echo "Set TASK to math, code (or coding), or science." >&2; exit 2 ;;
esac

CONFIG_ROOT="${SINGLE_TASK_CONFIG_ROOT:-${SLIME_DIR}/data/m2rl/single_task}"
MANIFEST="${CONFIG_ROOT}/${TASK}/${TASK}_on_policy.yaml"
DEFAULT_EVAL="${CONFIG_ROOT}/${TASK}/${TASK}_eval.yaml"
if [[ -z "${EVAL_CONFIG:-}" && "${TASK}" == "math" && -n "${MATH_EVAL_DATASETS:-}" ]]; then
  read -r -a math_eval_datasets <<< "${MATH_EVAL_DATASETS//,/ }"
  use_aime24=0
  use_math500=0
  for dataset in "${math_eval_datasets[@]}"; do
    case "${dataset,,}" in
      aime24|aime-2024) use_aime24=1 ;;
      math500|math-500) use_math500=1 ;;
      *) echo "MATH_EVAL_DATASETS accepts aime24 and/or math500, got ${dataset}." >&2; exit 2 ;;
    esac
  done
  case "${use_aime24}:${use_math500}" in
    1:0) DEFAULT_EVAL="${CONFIG_ROOT}/math/math_eval_aime24.yaml" ;;
    0:1) DEFAULT_EVAL="${CONFIG_ROOT}/math/math_eval_math500.yaml" ;;
    1:1) DEFAULT_EVAL="${CONFIG_ROOT}/math/math_eval_aime24_math500.yaml" ;;
    *) echo "MATH_EVAL_DATASETS must select aime24, math500, or both." >&2; exit 2 ;;
  esac
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
  case "${TASK}" in
    math) echo "Run examples/optimizer_geometry/prepare_single_task_dataset.sh to prepare AIME'24/MATH-500 configs." >&2 ;;
    code) echo "Prepare LiveCodeBench with examples/optimizer_geometry/prepare_livecodebench_eval.py." >&2 ;;
    science) echo "Accept GPQA access, export HF_TOKEN, then rerun preparation with REQUIRE_GPQA=1." >&2 ;;
  esac
  exit 2
fi

read -r -a OPTIMIZER_LIST <<< "${OPTIMIZERS:-adamw sgd muon}"
if [[ -n "${SEEDS+x}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
elif [[ -n "${SEED+x}" ]]; then
  SEED_LIST=("${SEED}")
else
  SEED_LIST=(42)
fi
if [[ "${BATCH_PROFILE:-opd64x1}" != "opd64x1" ]]; then
  echo "The frozen OPD paper scripts require BATCH_PROFILE=opd64x1 (64 prompts x 1 response)." >&2
  exit 2
fi
if [[ "${ROLLOUT_BATCH_SIZE:-64}" != "64" || "${N_SAMPLES_PER_PROMPT:-1}" != "1" ]]; then
  echo "The frozen OPD paper scripts require ROLLOUT_BATCH_SIZE=64 and N_SAMPLES_PER_PROMPT=1." >&2
  exit 2
fi
if [[ "${NUM_EPOCH:-1}" != "1" ]]; then
  echo "The frozen OPD paper scripts train exactly one usable dataset epoch (NUM_EPOCH=1)." >&2
  exit 2
fi
if [[ -n "${NUM_ROLLOUT:-}" || -n "${TARGET_PROMPT_BUDGET:-}" ]]; then
  echo "Frozen OPD dataset-epoch runs do not accept NUM_ROLLOUT or TARGET_PROMPT_BUDGET." >&2
  exit 2
fi
if [[ "${MAX_RESPONSE_LEN:-4096}" != "4096" ]]; then
  echo "The frozen OPD paper scripts require MAX_RESPONSE_LEN=4096." >&2
  exit 2
fi
if [[ "${MAX_PROMPT_LEN:-2048}" != "2048" || "${MAX_TOKENS_PER_GPU:-10240}" != "10240" ]]; then
  echo "The frozen OPD paper scripts require MAX_PROMPT_LEN=2048 and MAX_TOKENS_PER_GPU=10240." >&2
  exit 2
fi
if [[ "${EVAL_INTERVAL:-50}" != "50" ]]; then
  echo "The frozen OPD paper scripts require EVAL_INTERVAL=50 optimizer updates." >&2
  exit 2
fi
if [[ "${EVAL_MAX_RESPONSE_LEN:-${REQUIRED_EVAL_MAX_RESPONSE_LEN}}" != "${REQUIRED_EVAL_MAX_RESPONSE_LEN}" ]]; then
  echo "The frozen OPD ${TASK} scripts require EVAL_MAX_RESPONSE_LEN=${REQUIRED_EVAL_MAX_RESPONSE_LEN}." >&2
  exit 2
fi
if [[ "${SGLANG_MAX_RUNNING_REQUESTS:-72}" != "72" ]]; then
  echo "The frozen 4096-token OPD cells require SGLANG_MAX_RUNNING_REQUESTS=72." >&2
  exit 2
fi
export BATCH_PROFILE=opd64x1
export ROLLOUT_BATCH_SIZE=64
export N_SAMPLES_PER_PROMPT=1
export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
export NUM_EPOCH=1
export MAX_PROMPT_LEN=2048
export MAX_RESPONSE_LEN=4096
export MAX_TOKENS_PER_GPU=10240
export EVAL_INTERVAL=50
export EVAL_MAX_RESPONSE_LEN="${REQUIRED_EVAL_MAX_RESPONSE_LEN}"
export EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-48}"
export SGLANG_MAX_RUNNING_REQUESTS=72
if ! [[ "${EVAL_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAX_CONCURRENCY must be a positive integer." >&2
  exit 2
fi
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-zsqzz}"
export WANDB_PROJECT="${WANDB_PROJECT:-iclr2027-opd-geometry}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SLIME_DIR}/outputs/qwen3_1.7b_single_task}"
export REWARD_CONFIG="${REWARD_CONFIG:-${SCRIPT_DIR}/configs/rewards.example.yaml}"
export EXPERIMENT_DATA_INDEX="${INDEX}"

for seed in "${SEED_LIST[@]}"; do
  for optimizer in "${OPTIMIZER_LIST[@]}"; do
    run_name="qwen3_1.7b_${TASK}_qwen3_8b_opd_${optimizer}_${BATCH_PROFILE}_r${MAX_RESPONSE_LEN}_seed${seed}"
    echo "Launching OPD task=${TASK} teacher=Qwen3-8B optimizer=${optimizer} seed=${seed} run=${run_name}"
    TASK="${TASK}" \
    TEACHER="qwen3-8b" \
    ALGORITHM="opd" \
    OPTIMIZER="${optimizer}" \
    SEED="${seed}" \
    DATA_MANIFEST="${MANIFEST}" \
    EVAL_CONFIG="${EVAL}" \
    RUN_NAME="${run_name}" \
    WANDB_GROUP="qwen3_1.7b_${TASK}_qwen3_8b_opd" \
      bash "${LAUNCHER}"
  done
done
