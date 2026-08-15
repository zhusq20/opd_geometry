#!/usr/bin/env bash
# Blind optimizer identity while determining a common 51,200/102,400-prompt budget.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh}"
TUNING_DATA_MANIFEST="${TUNING_DATA_MANIFEST:?Set a tuning-only data manifest}"
TUNING_EVAL_CONFIG="${TUNING_EVAL_CONFIG:?Set a tuning-only eval config}"
PILOT_OUTPUT_ROOT="${PILOT_OUTPUT_ROOT:-${SCRIPT_DIR}/../../outputs/budget_pilot}"
read -r -a PILOT_SEEDS <<< "${PILOT_SEEDS:-1044 1045 1046}"
read -r -a PILOT_ALGORITHMS <<< "${PILOT_ALGORITHMS:-grpo ppo opd}"
BATCH_PROFILE="${BATCH_PROFILE:-responsive16}"
case "${BATCH_PROFILE}" in
  responsive16) PROFILE_PROMPT_BATCH=16 ;;
  responsive8) PROFILE_PROMPT_BATCH=8 ;;
  reference256) PROFILE_PROMPT_BATCH=256 ;;
  *) echo "Unsupported BATCH_PROFILE=${BATCH_PROFILE}." >&2; exit 2 ;;
esac
BASE_PROMPT_BUDGET="${BASE_PROMPT_BUDGET:-51200}"
if (( BASE_PROMPT_BUDGET % PROFILE_PROMPT_BATCH != 0 )); then
  echo "BASE_PROMPT_BUDGET must be divisible by profile batch ${PROFILE_PROMPT_BATCH}." >&2
  exit 2
fi
base_steps=$((BASE_PROMPT_BUDGET / PROFILE_PROMPT_BATCH))
extend_budget="${EXTEND_TO_DOUBLE_BUDGET:-${EXTEND_TO_400:-0}}"
case "${extend_budget}" in
  0) target_steps="${base_steps}"; fresh_start=1 ;;
  1) target_steps=$((2 * base_steps)); fresh_start=0 ;;
  *) echo "EXTEND_TO_DOUBLE_BUDGET (legacy EXTEND_TO_400) must be 0 or 1." >&2; exit 2 ;;
esac
pilot_eval_interval=$(((2560 + PROFILE_PROMPT_BATCH - 1) / PROFILE_PROMPT_BATCH))
pilot_save_interval=$(((12800 + PROFILE_PROMPT_BATCH - 1) / PROFILE_PROMPT_BATCH))

case "${TUNING_DATA_MANIFEST} ${TUNING_EVAL_CONFIG}" in
  *aime*|*AIME*|*math500*|*MATH500*|*math-500*|*MATH-500*|*gpqa*|*GPQA*|*livecodebench*|*LiveCodeBench*)
    echo "Refusing to choose the training budget on a paper-reporting benchmark path." >&2
    exit 2
    ;;
esac

for algorithm in "${PILOT_ALGORITHMS[@]}"; do
  case "${algorithm}" in
    grpo|ppo|opd) ;;
    *) echo "PILOT_ALGORITHMS only accepts grpo, ppo, and opd." >&2; exit 2 ;;
  esac
  for seed in "${PILOT_SEEDS[@]}"; do
    run_name="budget_${algorithm}_adamw_${BATCH_PROFILE}_seed${seed}"
    run_dir="${PILOT_OUTPUT_ROOT}/${run_name}"
    if [[ "${fresh_start}" == "0" ]]; then
      if [[ ! -s "${run_dir}/checkpoints/latest_checkpointed_iteration.txt" ]]; then
        echo "Cannot extend missing ${base_steps}-update checkpoint: ${run_dir}" >&2
        exit 2
      fi
      load_checkpoint="${run_dir}/checkpoints"
    else
      load_checkpoint="${LOAD_CHECKPOINT:-}"
    fi
    environment=(
      ALGORITHM="${algorithm}"
      OPTIMIZER=adamw
      BATCH_PROFILE="${BATCH_PROFILE}"
      SEED="${seed}"
      DATA_MANIFEST="${TUNING_DATA_MANIFEST}"
      EVAL_CONFIG="${TUNING_EVAL_CONFIG}"
      NUM_ROLLOUT="${target_steps}"
      EVAL_INTERVAL="${pilot_eval_interval}"
      SAVE_INTERVAL="${pilot_save_interval}"
      SAVE_CHECKPOINTS=1
      GEOMETRY_INTERVAL="${pilot_save_interval}"
      FRESH_START="${fresh_start}"
      OUTPUT_ROOT="${PILOT_OUTPUT_ROOT}"
      RUN_NAME="${run_name}"
      WANDB_GROUP="budget_pilot_${algorithm}"
    )
    if [[ -n "${load_checkpoint}" ]]; then
      environment+=(LOAD_CHECKPOINT="${load_checkpoint}")
    fi
    env "${environment[@]}" bash "${LAUNCHER}"
  done
done
