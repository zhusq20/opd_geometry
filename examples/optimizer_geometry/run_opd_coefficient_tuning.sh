#!/usr/bin/env bash
# Select one OPD KL coefficient on a disjoint split before optimizer LR tuning.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh}"
TUNING_DATA_MANIFEST="${TUNING_DATA_MANIFEST:?Set a tuning-only data manifest}"
TUNING_EVAL_CONFIG="${TUNING_EVAL_CONFIG:?Set a tuning-only eval config}"
TUNING_OUTPUT_ROOT="${TUNING_OUTPUT_ROOT:-${SCRIPT_DIR}/../../outputs/opd_coefficient_tuning}"
BATCH_PROFILE="${BATCH_PROFILE:-responsive16}"
case "${BATCH_PROFILE}" in
  responsive16) PROFILE_PROMPT_BATCH=16; PROFILE_ADAMW_LR=2.5e-7 ;;
  responsive8) PROFILE_PROMPT_BATCH=8; PROFILE_ADAMW_LR=1.8e-7 ;;
  reference256) PROFILE_PROMPT_BATCH=256; PROFILE_ADAMW_LR=1e-6 ;;
  *) echo "Unsupported BATCH_PROFILE=${BATCH_PROFILE}." >&2; exit 2 ;;
esac
TUNING_PROMPT_BUDGET="${TUNING_PROMPT_BUDGET:-12800}"
TUNING_EVAL_PROMPT_INTERVAL="${TUNING_EVAL_PROMPT_INTERVAL:-2560}"
if (( TUNING_PROMPT_BUDGET % PROFILE_PROMPT_BATCH != 0 )); then
  echo "TUNING_PROMPT_BUDGET must be divisible by profile batch ${PROFILE_PROMPT_BATCH}." >&2
  exit 2
fi
TUNING_STEPS="${TUNING_STEPS:-$((TUNING_PROMPT_BUDGET / PROFILE_PROMPT_BATCH))}"
TUNING_EVAL_INTERVAL="${TUNING_EVAL_INTERVAL:-$(((TUNING_EVAL_PROMPT_INTERVAL + PROFILE_PROMPT_BATCH - 1) / PROFILE_PROMPT_BATCH))}"
read -r -a TUNING_SEEDS <<< "${TUNING_SEEDS:-1042 1043}"
read -r -a OPD_COEFFICIENTS <<< "${OPD_COEFFICIENTS:-0.1 0.3 1.0 3.0}"

case "${TUNING_DATA_MANIFEST} ${TUNING_EVAL_CONFIG}" in
  *aime*|*AIME*|*math500*|*MATH500*|*math-500*|*MATH-500*|*gpqa*|*GPQA*|*livecodebench*|*LiveCodeBench*)
    echo "Refusing to tune on a paper-reporting benchmark path." >&2
    exit 2
    ;;
esac

for coefficient in "${OPD_COEFFICIENTS[@]}"; do
  for seed in "${TUNING_SEEDS[@]}"; do
    run_name="tune_opd_adamw_${BATCH_PROFILE}_kl${coefficient}_seed${seed}"
    env \
      ALGORITHM=opd \
      OPTIMIZER=adamw \
      BATCH_PROFILE="${BATCH_PROFILE}" \
      ADAMW_LR="${COEFFICIENT_PILOT_LR:-${PROFILE_ADAMW_LR}}" \
      OPD_KL_COEF="${coefficient}" \
      SEED="${seed}" \
      DATA_MANIFEST="${TUNING_DATA_MANIFEST}" \
      EVAL_CONFIG="${TUNING_EVAL_CONFIG}" \
      NUM_ROLLOUT="${TUNING_STEPS}" \
      EVAL_INTERVAL="${TUNING_EVAL_INTERVAL}" \
      SAVE_INTERVAL="${TUNING_STEPS}" \
      SAVE_CHECKPOINTS=0 \
      GEOMETRY_INTERVAL="${TUNING_STEPS}" \
      OUTPUT_ROOT="${TUNING_OUTPUT_ROOT}" \
      RUN_NAME="${run_name}" \
      WANDB_GROUP=opd_coefficient_tuning \
      bash "${LAUNCHER}"
  done
done
