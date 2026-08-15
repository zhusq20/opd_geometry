#!/usr/bin/env bash
# Sweep the relative OPD/SFT/RL coefficients for one fixed task/teacher/optimizer.
# Each setting is algorithm:opd_kl:sft_coef:hybrid_opd_coef:task_reward_weight.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ROOT="${SINGLE_TASK_CONFIG_ROOT:?Set SINGLE_TASK_CONFIG_ROOT to prepare_single_task_data.py output}"
LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-teacher.sh}"
TASK="${TASK:-math}"
TEACHER="${TEACHER:-qwen3-8b}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SEED="${SEED:-42}"
TEACHER_SLUG="${TEACHER//-/_}"

DEFAULT_SETTINGS="opd:1:0:0:0 sft_opd:1:0:1:0 sft_opd:1:1:0:0 sft_opd:1:1:0.25:0 sft_opd:1:1:1:0 sft_opd:1:1:4:0 grpo:0:0:0:1 grpo_opd:0.25:0:0:1 grpo_opd:1:0:0:1 grpo_opd:4:0:0:1 ppo:0:0:0:1 ppo_opd:0.25:0:0:1 ppo_opd:1:0:0:1 ppo_opd:4:0:0:1"
read -r -a SETTING_LIST <<< "${MIXTURE_SETTINGS:-${DEFAULT_SETTINGS}}"

slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

for setting in "${SETTING_LIST[@]}"; do
  IFS=':' read -r algorithm opd_kl sft_coef hybrid_opd_coef task_reward_weight extra <<< "${setting}"
  if [[ -n "${extra:-}" || -z "${task_reward_weight:-}" ]]; then
    echo "Invalid MIXTURE_SETTINGS entry ${setting@Q}; expected algorithm:opd_kl:sft:hybrid_opd:task_reward." >&2
    exit 2
  fi
  case "${algorithm}" in
    opd|grpo|ppo|grpo_opd|ppo_opd)
      manifest="${CONFIG_ROOT}/${TASK}/${TASK}_on_policy.yaml"
      ;;
    sft_opd)
      manifest="${CONFIG_ROOT}/${TASK}/${TASK}_sft_opd.yaml"
      ;;
    *)
      echo "Unsupported mixture algorithm ${algorithm@Q}." >&2
      exit 2
      ;;
  esac
  eval_config="${CONFIG_ROOT}/${TASK}/${TASK}_eval.yaml"
  response_suffix=""
  if [[ "${TASK}" == "math" && ( "${algorithm}" == "grpo" || "${algorithm}" == "ppo" ) ]]; then
    eval_config="${CONFIG_ROOT}/math/math_eval_aime24_math500.yaml"
  fi
  case "${algorithm}" in
    grpo|ppo)
      response_suffix="_trainr8192"
      if [[ "${TASK}" == "math" ]]; then
        response_suffix+="_evalr32768"
      fi
      ;;
  esac
  if [[ ! -f "${manifest}" ]]; then
    echo "Missing prepared manifest for task=${TASK}, algorithm=${algorithm}." >&2
    exit 2
  fi
  if [[ ! -f "${eval_config}" ]]; then
    if [[ "${TASK}" == "code" ]]; then
      eval_config=""
    else
      echo "Missing independent eval config for task=${TASK}." >&2
      exit 2
    fi
  fi

  run_name="qwen3_1.7b_${TASK}_${TEACHER_SLUG}_${algorithm}_${OPTIMIZER}_${BATCH_PROFILE:-responsive16}${response_suffix}_kl$(slug "${opd_kl}")_sft$(slug "${sft_coef}")_hopd$(slug "${hybrid_opd_coef}")_rw$(slug "${task_reward_weight}")_seed${SEED}"
  echo "Submitting ${setting}; run=${run_name}"
  launch_env=(
    TASK="${TASK}"
    TEACHER="${TEACHER}"
    OPTIMIZER="${OPTIMIZER}"
    ALGORITHM="${algorithm}"
    SEED="${SEED}"
    OPD_KL_COEF="${opd_kl}"
    SFT_LOSS_COEF="${sft_coef}"
    HYBRID_OPD_LOSS_COEF="${hybrid_opd_coef}"
    OPD_TASK_REWARD_WEIGHT="${task_reward_weight}"
    DATA_MANIFEST="${manifest}"
    EVAL_CONFIG="${eval_config}"
    RUN_NAME="${run_name}"
  )
  case "${algorithm}" in
    grpo|ppo)
      launch_env+=(
        MAX_RESPONSE_LEN=8192
        MAX_TOKENS_PER_GPU=10240
      )
      if [[ "${TASK}" == "math" ]]; then
        launch_env+=(EVAL_MAX_RESPONSE_LEN=32768 EVAL_MAX_CONCURRENCY=48 SGLANG_MAX_RUNNING_REQUESTS=12)
      fi
      ;;
  esac
  env "${launch_env[@]}" bash "${LAUNCHER}"
done
