#!/usr/bin/env bash
# Submit the seed-42, one-dataset-epoch Qwen3-1.7B single-task matrix sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ROOT="${SINGLE_TASK_CONFIG_ROOT:?Set SINGLE_TASK_CONFIG_ROOT to prepare_single_task_data.py output}"
LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-teacher.sh}"

read -r -a TASK_LIST <<< "${TASKS:-math code science}"
read -r -a TEACHER_LIST <<< "${TEACHERS:-qwen3-8b}"
read -r -a OPTIMIZER_LIST <<< "${OPTIMIZERS:-adamw sgd muon}"
read -r -a ALGORITHM_LIST <<< "${ALGORITHMS:-opd}"
if [[ -n "${SEEDS+x}" ]]; then
  read -r -a REQUESTED_SEEDS <<< "${SEEDS}"
  if (( ${#REQUESTED_SEEDS[@]} != 1 )) || [[ "${REQUESTED_SEEDS[0]}" != "42" ]]; then
    echo "The single-task paper matrix requires exactly SEEDS=42." >&2
    exit 2
  fi
fi
if [[ -n "${SEED+x}" && "${SEED}" != "42" ]]; then
  echo "The single-task paper matrix requires SEED=42." >&2
  exit 2
fi
if [[ "${NUM_EPOCH:-1}" != "1" ]]; then
  echo "The single-task paper matrix trains exactly one usable dataset epoch (NUM_EPOCH=1)." >&2
  exit 2
fi
if [[ -n "${NUM_ROLLOUT:-}" || -n "${TARGET_PROMPT_BUDGET:-}" ]]; then
  echo "Single-task dataset-epoch runs do not accept NUM_ROLLOUT or TARGET_PROMPT_BUDGET." >&2
  exit 2
fi
export NUM_EPOCH=1

for task in "${TASK_LIST[@]}"; do
    default_eval_config="${CONFIG_ROOT}/${task}/${task}_eval.yaml"
    if [[ ! -s "${default_eval_config}" ]]; then
      echo "Missing independent eval config for task=${task}: ${default_eval_config}" >&2
      if [[ "${task}" == "code" ]]; then
        echo "Prepare LiveCodeBench with ${SCRIPT_DIR}/prepare_livecodebench_eval.py." >&2
      fi
      exit 2
    fi
    for teacher in "${TEACHER_LIST[@]}"; do
      teacher_slug="${teacher//-/_}"
      for algorithm in "${ALGORITHM_LIST[@]}"; do
        eval_config="${default_eval_config}"
        response_suffix=""
        if [[ "${algorithm}" == "sft_opd" ]]; then
          manifest="${CONFIG_ROOT}/${task}/${task}_sft_opd.yaml"
        else
          manifest="${CONFIG_ROOT}/${task}/${task}_on_policy.yaml"
        fi
        if [[ "${task}" == "math" && ( "${algorithm}" == "grpo" || "${algorithm}" == "ppo" ) ]]; then
          eval_config="${CONFIG_ROOT}/math/math_eval_aime24_math500.yaml"
        fi
        case "${algorithm}" in
          grpo|ppo)
            response_suffix="_trainr8192"
            if [[ "${task}" == "math" ]]; then
              response_suffix+="_evalr32768"
            fi
            ;;
        esac
        if [[ ! -s "${eval_config}" ]]; then
          echo "Missing independent eval config for task=${task}, algorithm=${algorithm}: ${eval_config}" >&2
          exit 2
        fi
        if [[ ! -f "${manifest}" ]]; then
          echo "Missing manifest for task=${task}, algorithm=${algorithm}: ${manifest}" >&2
          exit 2
        fi
        for optimizer in "${OPTIMIZER_LIST[@]}"; do
          run_name="qwen3_1.7b_${task}_${teacher_slug}_${algorithm}_${optimizer}_${BATCH_PROFILE:-responsive16}${response_suffix}_seed42"
          echo "Submitting task=${task} teacher=${teacher} algorithm=${algorithm} optimizer=${optimizer} seed=42"
          launch_env=(
            TASK="${task}"
            TEACHER="${teacher}"
            ALGORITHM="${algorithm}"
            OPTIMIZER="${optimizer}"
            SEED=42
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
              if [[ "${task}" == "math" ]]; then
                launch_env+=(EVAL_MAX_RESPONSE_LEN=32768 EVAL_MAX_CONCURRENCY=48 SGLANG_MAX_RUNNING_REQUESTS=12)
              fi
              ;;
          esac
          env "${launch_env[@]}" bash "${LAUNCHER}"
        done
      done
    done
done
