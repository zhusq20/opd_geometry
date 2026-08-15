#!/usr/bin/env bash
# Equal-budget LR tuning on a disjoint, non-paper validation split.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh}"
TUNING_DATA_MANIFEST="${TUNING_DATA_MANIFEST:?Set a tuning-only data manifest}"
TUNING_EVAL_CONFIG="${TUNING_EVAL_CONFIG:?Set a tuning-only eval config}"
TUNING_OUTPUT_ROOT="${TUNING_OUTPUT_ROOT:-${SCRIPT_DIR}/../../outputs/optimizer_tuning}"
BATCH_PROFILE="${BATCH_PROFILE:-responsive16}"
case "${BATCH_PROFILE}" in
  responsive16)
    PROFILE_PROMPT_BATCH=16
    PROFILE_ADAPTIVE_LRS="1e-7 2.5e-7 5e-7 1e-6"
    PROFILE_GRPO_SGD_LRS="1e-2 2.5e-2 5e-2 1e-1"
    PROFILE_OTHER_SGD_LRS="1e-3 2.5e-3 5e-3 1e-2"
    ;;
  responsive8)
    PROFILE_PROMPT_BATCH=8
    PROFILE_ADAPTIVE_LRS="9e-8 1.8e-7 3.6e-7 7.2e-7"
    PROFILE_GRPO_SGD_LRS="9e-3 1.8e-2 3.6e-2 7.2e-2"
    PROFILE_OTHER_SGD_LRS="9e-4 1.8e-3 3.6e-3 7.2e-3"
    ;;
  reference256)
    PROFILE_PROMPT_BATCH=256
    PROFILE_ADAPTIVE_LRS="4e-7 1e-6 2e-6 4e-6"
    PROFILE_GRPO_SGD_LRS="4e-2 1e-1 2e-1 4e-1"
    PROFILE_OTHER_SGD_LRS="4e-3 1e-2 2e-2 4e-2"
    ;;
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
read -r -a ALGORITHMS_TO_TUNE <<< "${TUNING_ALGORITHMS:-grpo ppo opd}"

case "${TUNING_DATA_MANIFEST} ${TUNING_EVAL_CONFIG}" in
  *aime*|*AIME*|*math500*|*MATH500*|*math-500*|*MATH-500*|*gpqa*|*GPQA*|*livecodebench*|*LiveCodeBench*)
    echo "Refusing to tune on a paper-reporting benchmark path." >&2
    exit 2
    ;;
esac

for algorithm in "${ALGORITHMS_TO_TUNE[@]}"; do
  for optimizer in adamw muon sgd; do
    case "${optimizer}" in
      adamw|muon)
        read -r -a candidates <<< "${ADAPTIVE_LR_CANDIDATES:-${PROFILE_ADAPTIVE_LRS}}"
        ;;
      sgd)
        if [[ "${algorithm}" == "grpo" ]]; then
          read -r -a candidates <<< "${SGD_GRPO_LR_CANDIDATES:-${PROFILE_GRPO_SGD_LRS}}"
        else
          read -r -a candidates <<< "${SGD_OPD_PPO_LR_CANDIDATES:-${PROFILE_OTHER_SGD_LRS}}"
        fi
        ;;
    esac
    for lr in "${candidates[@]}"; do
      for seed in "${TUNING_SEEDS[@]}"; do
        run_name="tune_${algorithm}_${optimizer}_${BATCH_PROFILE}_lr${lr}_seed${seed}"
        common=(
          ALGORITHM="${algorithm}"
          OPTIMIZER="${optimizer}"
          BATCH_PROFILE="${BATCH_PROFILE}"
          SEED="${seed}"
          DATA_MANIFEST="${TUNING_DATA_MANIFEST}"
          EVAL_CONFIG="${TUNING_EVAL_CONFIG}"
          NUM_ROLLOUT="${TUNING_STEPS}"
          EVAL_INTERVAL="${TUNING_EVAL_INTERVAL}"
          SAVE_INTERVAL="${TUNING_STEPS}"
          SAVE_CHECKPOINTS=0
          GEOMETRY_INTERVAL="${TUNING_STEPS}"
          OUTPUT_ROOT="${TUNING_OUTPUT_ROOT}"
          RUN_NAME="${run_name}"
          WANDB_GROUP="optimizer_tuning_${algorithm}"
        )
        case "${optimizer}" in
          adamw) common+=(ADAMW_LR="${lr}") ;;
          muon) common+=(MUON_LR="${lr}") ;;
          sgd) common+=(SGD_LR="${lr}") ;;
        esac
        env "${common[@]}" bash "${LAUNCHER}"
      done
    done
  done
done
