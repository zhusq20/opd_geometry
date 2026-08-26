#!/usr/bin/env bash
# Run the expensive four-domain evaluation once, on the final mixed checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MIXED_DIR="${MIXED_DIR:-${SLIME_DIR}/outputs/raw_gradient_interference/qwen3_1.7b/seed42/mixed_code_math_qa_if}"
PREPARED_DIR="${PREPARED_DIR:-${MIXED_DIR}/prepared_data_usable4800_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MIXED_DIR}}"
RUN_NAME="${RUN_NAME:-qwen3_1.7b_code_math_qa_if_mixed_batch_grpo_adamw_usable4800_seed42}"
TRAIN_RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${TRAIN_RUN_DIR}/checkpoints}"
DATA_MANIFEST="${DATA_MANIFEST:-${PREPARED_DIR}/mixed_on_policy.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-${PREPARED_DIR}/mixed_final_eval.yaml}"
EXPERIMENT_DATA_INDEX="${EXPERIMENT_DATA_INDEX:-${PREPARED_DIR}/mixed_data_index.json}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-${SCRIPT_DIR}/evaluate_single_task.sh}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${TRAIN_RUN_DIR}/final_eval}"
USE_WANDB="${USE_WANDB:-1}"

export SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL:-http://127.0.0.1:8080}"
if [[ -z "${M2RL_SANDBOX_PREFLIGHT_MARKER:-}" && \
      -r /workspace/sandboxfusion-state/sandboxfusion_preflight.json ]]; then
  export M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json
fi

for path in "${LOAD_CHECKPOINT}" "${DATA_MANIFEST}" "${EVAL_CONFIG}" "${EXPERIMENT_DATA_INDEX}" "${EVAL_LAUNCHER}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Final-evaluation input does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ -s "${OUTPUT_DIR}/run_complete.json" ]]; then
  echo "Final mixed evaluation is already complete: ${OUTPUT_DIR}"
  exit 0
fi

mapfile -t model_paths < <(
  python3 - "${EXPERIMENT_DATA_INDEX}" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))["model"]
print(model["hf_checkpoint"])
print(model["model_config"])
PY
)
if (( ${#model_paths[@]} != 2 )); then
  echo "Could not read model paths from ${EXPERIMENT_DATA_INDEX}." >&2
  exit 2
fi

eval_devices="${EVAL_CUDA_VISIBLE_DEVICES:-${TRAIN_CUDA_VISIBLE_DEVICES:-${AVAILABLE_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}}"
if [[ -z "${eval_devices}" ]]; then
  echo "Set EVAL_CUDA_VISIBLE_DEVICES, TRAIN_CUDA_VISIBLE_DEVICES, AVAILABLE_CUDA_DEVICES, or CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi

echo "Evaluating final checkpoint only: ${LOAD_CHECKPOINT}"
echo "Domains: Code/LiveCodeBench, Math/AIME24+MATH500, QA/GPQA, IF/IFEval+IFBench"
echo "Evaluation output: ${OUTPUT_DIR}"

TASK=mixed_code_math_qa_if_final \
HF_CHECKPOINT="${model_paths[0]}" \
MODEL_CONFIG="${model_paths[1]}" \
LOAD_CHECKPOINT="${LOAD_CHECKPOINT}" \
DATA_MANIFEST="${DATA_MANIFEST}" \
EVAL_CONFIG="${EVAL_CONFIG}" \
EXPERIMENT_DATA_INDEX="${EXPERIMENT_DATA_INDEX}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RUN_NAME="${RUN_NAME}_final_eval" \
ALLOW_MIXED_EVAL_RESPONSE_LEN=1 \
EVAL_MAX_RESPONSE_LEN=32768 \
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-48}" \
SGLANG_MAX_RUNNING_REQUESTS="${EVAL_SGLANG_MAX_RUNNING_REQUESTS:-12}" \
NUM_GPUS="${EVAL_NUM_GPUS:-4}" \
CUDA_VISIBLE_DEVICES="${eval_devices}" \
USE_WANDB="${USE_WANDB}" \
WANDB_GROUP="${WANDB_GROUP:-mixed_batch_grpo_final_eval}" \
RAY_OBJECT_SPILLING_DIR="${RAY_OBJECT_SPILLING_DIR:-${OUTPUT_DIR}/.ray_spill}" \
  bash "${EVAL_LAUNCHER}"
