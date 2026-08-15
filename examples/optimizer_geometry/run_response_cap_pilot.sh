#!/usr/bin/env bash
# Calibrate response length on matched initial-policy rollouts before the matrix.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh}"
TUNING_DATA_MANIFEST="${TUNING_DATA_MANIFEST:?Set a disjoint tuning-only data manifest}"
PILOT_OUTPUT_ROOT="${PILOT_OUTPUT_ROOT:-${SCRIPT_DIR}/../../outputs/response_cap_pilot}"
PILOT_ALGORITHM="${PILOT_ALGORITHM:-grpo}"
PILOT_OPTIMIZER="${PILOT_OPTIMIZER:-adamw}"
PILOT_PROMPT_CAP="${PILOT_PROMPT_CAP:-2048}"
# Cap selection needs enough initial-policy samples to estimate a truncation
# fraction.  Keep this one-rollout calibration on the 256-prompt reference
# profile; the actual tuning and paper runs use responsive16 by default.
PILOT_BATCH_PROFILE="${PILOT_BATCH_PROFILE:-reference256}"
read -r -a PILOT_SEEDS <<< "${PILOT_SEEDS:-1042 1043}"
# max-running values are calibrated for Qwen3-1.7B, 96 GiB cards and
# SGLANG_MEM_FRACTION=0.6 (~498k KV tokens). Recalibrate on other hardware.
read -r -a RESPONSE_PROFILES <<< "${RESPONSE_PROFILES:-4096:72 8192:44 12288:31 16384:24}"

case "${TUNING_DATA_MANIFEST}" in
  *aime*|*AIME*|*math500*|*MATH500*|*math-500*|*MATH-500*|*gpqa*|*GPQA*|*livecodebench*|*LiveCodeBench*)
    echo "Refusing to select response length on a paper-reporting benchmark path." >&2
    exit 2
    ;;
esac

for profile in "${RESPONSE_PROFILES[@]}"; do
  response_cap="${profile%%:*}"
  max_running="${profile##*:}"
  if [[ "${profile}" != *:* ]] || \
     ! [[ "${response_cap}" =~ ^[1-9][0-9]*$ && "${max_running}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Each RESPONSE_PROFILES entry must be RESPONSE_TOKENS:MAX_RUNNING, got ${profile}." >&2
    exit 2
  fi
  max_tokens_per_gpu=$((PILOT_PROMPT_CAP + response_cap))
  for seed in "${PILOT_SEEDS[@]}"; do
    run_name="response_cap_${response_cap}_${PILOT_ALGORITHM}_${PILOT_OPTIMIZER}_seed${seed}"
    env \
      ALGORITHM="${PILOT_ALGORITHM}" \
      OPTIMIZER="${PILOT_OPTIMIZER}" \
      BATCH_PROFILE="${PILOT_BATCH_PROFILE}" \
      SEED="${seed}" \
      DATA_MANIFEST="${TUNING_DATA_MANIFEST}" \
      EVAL_CONFIG="${TUNING_EVAL_CONFIG:-}" \
      NUM_ROLLOUT=1 \
      SAVE_INTERVAL=1 \
      SAVE_CHECKPOINTS=0 \
      GEOMETRY_INTERVAL=1 \
      MAX_PROMPT_LEN="${PILOT_PROMPT_CAP}" \
      MAX_RESPONSE_LEN="${response_cap}" \
      MAX_TOKENS_PER_GPU="${max_tokens_per_gpu}" \
      SGLANG_MAX_RUNNING_REQUESTS="${max_running}" \
      OUTPUT_ROOT="${PILOT_OUTPUT_ROOT}" \
      RUN_NAME="${run_name}" \
      WANDB_GROUP=response_cap_pilot \
      bash "${LAUNCHER}"
  done
done
