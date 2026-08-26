#!/usr/bin/env bash
# Equal-volume within-batch GRPO: Qwen3-1.7B base -> Code + Math + QA + IF.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_CODE_RUN_DIR="${SLIME_DIR}/outputs/qwen3_1.7b_code_grpo_after_sandbox_20260817T194524Z/qwen3_1.7b_code_grpo_adamw_responsive16_trainr8192_seed42"
DEFAULT_CODE_CHECKPOINT="${DEFAULT_CODE_RUN_DIR}/checkpoints/iter_0000299"
DEFAULT_REFERENCE_PREPARED_DIR="${SLIME_DIR}/outputs/raw_gradient_interference/qwen3_1.7b/seed42/sequential_code_math_knowledge_if/prepared_data"

CODE_CHECKPOINT="${CODE_CHECKPOINT:-${DEFAULT_CODE_CHECKPOINT}}"
if [[ -z "${CODE_ORIGIN_PROVENANCE:-}" ]]; then
  checkpoint_root="${CODE_CHECKPOINT%/}"
  if [[ "$(basename -- "${checkpoint_root}")" =~ ^iter_[0-9]+$ ]]; then
    checkpoint_root="$(dirname -- "${checkpoint_root}")"
  fi
  CODE_ORIGIN_PROVENANCE="$(dirname -- "${checkpoint_root}")/provenance/run_manifest.json"
fi

MIXED_DIR="${MIXED_DIR:-${SLIME_DIR}/outputs/raw_gradient_interference/qwen3_1.7b/seed42/mixed_code_math_qa_if}"
PREPARED_DIR="${PREPARED_DIR:-${MIXED_DIR}/prepared_data_usable4800_v2}"
REFERENCE_PREPARED_DIR="${REFERENCE_PREPARED_DIR:-${DEFAULT_REFERENCE_PREPARED_DIR}}"
SINGLE_TASK_CONFIG_ROOT="${SINGLE_TASK_CONFIG_ROOT:-${SLIME_DIR}/data/m2rl/single_task}"
CODE_MANIFEST="${CODE_MANIFEST:-${SINGLE_TASK_CONFIG_ROOT}/code/code_on_policy.yaml}"
MODEL_CONFIG_REFERENCE="${MODEL_CONFIG_REFERENCE:-${SLIME_DIR}/scripts/models/qwen3-1.7B.sh}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh}"
FINAL_EVAL_LAUNCHER="${FINAL_EVAL_LAUNCHER:-${SCRIPT_DIR}/evaluate_mixed_grpo.sh}"
REWARD_CONFIG="${REWARD_CONFIG:-${SCRIPT_DIR}/configs/rewards.example.yaml}"

SEED="${SEED:-42}"
PROBE_PROMPTS="${PROBE_PROMPTS:-128}"
TRAIN_PROMPTS_PER_TASK="${TRAIN_PROMPTS_PER_TASK:-4800}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-10240}"
TRAIN_GPU_COUNT="${TRAIN_GPU_COUNT:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.6}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-44}"
PLAN_ONLY="${PLAN_ONLY:-0}"
RESUME_INCOMPLETE_RUN="${RESUME_INCOMPLETE_RUN:-0}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-zsqzz}"
WANDB_PROJECT="${WANDB_PROJECT:-iclr2027-opd-geometry}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"

for boolean_name in PLAN_ONLY RESUME_INCOMPLETE_RUN USE_WANDB RUN_FINAL_EVAL; do
  boolean_value="${!boolean_name}"
  case "${boolean_value}" in
    0|1) ;;
    *) echo "${boolean_name} must be 0 or 1." >&2; exit 2 ;;
  esac
done
if [[ "${USE_WANDB}" != "1" ]]; then
  echo "This primary mixed experiment requires USE_WANDB=1." >&2
  exit 2
fi
if [[ "${WANDB_MODE}" != "online" ]]; then
  echo "This primary mixed experiment requires WANDB_MODE=online." >&2
  exit 2
fi
if [[ "${SEED}" != "42" ]]; then
  echo "The Code/Math/QA/IF comparison is frozen to SEED=42." >&2
  exit 2
fi
for integer_name in \
  PROBE_PROMPTS TRAIN_PROMPTS_PER_TASK ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT \
  MAX_TOKENS_PER_GPU TRAIN_GPU_COUNT ROLLOUT_NUM_GPUS ROLLOUT_GPUS_PER_ENGINE \
  SGLANG_MAX_RUNNING_REQUESTS; do
  integer_value="${!integer_name}"
  if ! [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${integer_name} must be a positive integer." >&2
    exit 2
  fi
done
if [[ "${TRAIN_PROMPTS_PER_TASK}:${ROLLOUT_BATCH_SIZE}:${N_SAMPLES_PER_PROMPT}" != "4800:16:16" ]]; then
  echo "The equal-volume comparison is frozen to 4800 prompts/task, rollout batch 16, and GRPO group size 16." >&2
  echo "Those values reproduce the 300-update Code origin and the three sequential stages." >&2
  exit 2
fi
if (( MAX_TOKENS_PER_GPU < 10240 )); then
  echo "MAX_TOKENS_PER_GPU must cover prompt+response=2048+8192=10240." >&2
  exit 2
fi
if [[ "${TRAIN_GPU_COUNT}:${ROLLOUT_NUM_GPUS}:${ROLLOUT_GPUS_PER_ENGINE}" != "4:4:1" ]]; then
  echo "The calibrated mixed-run hardware contract requires 4 actor GPUs and four TP=1 rollout engines." >&2
  exit 2
fi
if [[ "${SGLANG_MEM_FRACTION}" != "0.6" ]]; then
  echo "The calibrated mixed-run hardware contract requires SGLANG_MEM_FRACTION=0.6." >&2
  exit 2
fi
if (( ROLLOUT_BATCH_SIZE % 4 != 0 )); then
  echo "ROLLOUT_BATCH_SIZE must be divisible by four tasks." >&2
  exit 2
fi
prompts_per_task_per_batch=$((ROLLOUT_BATCH_SIZE / 4))
if (( TRAIN_PROMPTS_PER_TASK % prompts_per_task_per_batch != 0 )); then
  echo "TRAIN_PROMPTS_PER_TASK must be divisible by ${prompts_per_task_per_batch}." >&2
  exit 2
fi
if [[ ! -f "${TRAIN_LAUNCHER}" ]]; then
  echo "Training launcher does not exist: ${TRAIN_LAUNCHER}" >&2
  exit 2
fi
if [[ "${RUN_FINAL_EVAL}" == "1" && ! -f "${FINAL_EVAL_LAUNCHER}" ]]; then
  echo "Final-evaluation launcher does not exist: ${FINAL_EVAL_LAUNCHER}" >&2
  exit 2
fi
for path in "${CODE_CHECKPOINT}" "${CODE_ORIGIN_PROVENANCE}" "${CODE_MANIFEST}" "${MODEL_CONFIG_REFERENCE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required comparison input does not exist: ${path}" >&2
    exit 2
  fi
done

if [[ "${PLAN_ONLY}" != "1" ]]; then
  python3 - "${WANDB_ENTITY}" "${WANDB_PROJECT}" <<'PY'
import sys

import wandb

entity, project = sys.argv[1:]
api = wandb.Api(timeout=20)
_ = api.viewer
resolved_project = api.project(project, entity=entity)
if resolved_project.name != project:
    raise RuntimeError(f"W&B project mismatch: expected {entity}/{project}, got {resolved_project.name}")
print(f"W&B online preflight passed: {entity}/{project}")
PY
fi

# Reuse the exact Math/QA/IF train views frozen for the sequential baseline.
python3 "${SCRIPT_DIR}/prepare_sequential_grpo_data.py" \
  --config-root "${SINGLE_TASK_CONFIG_ROOT}" \
  --output-dir "${REFERENCE_PREPARED_DIR}" \
  --tasks math science if \
  --probe-prompts "${PROBE_PROMPTS}" \
  --train-prompts-per-task "${TRAIN_PROMPTS_PER_TASK}" \
  --seed "${SEED}"

python3 "${SCRIPT_DIR}/prepare_mixed_grpo_data.py" \
  --sequential-data-index "${REFERENCE_PREPARED_DIR}/sequential_data_index.json" \
  --code-checkpoint "${CODE_CHECKPOINT}" \
  --code-origin-provenance "${CODE_ORIGIN_PROVENANCE}" \
  --code-manifest "${CODE_MANIFEST}" \
  --model-config "${MODEL_CONFIG_REFERENCE}" \
  --output-dir "${PREPARED_DIR}" \
  --prompts-per-task "${TRAIN_PROMPTS_PER_TASK}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --group-size "${N_SAMPLES_PER_PROMPT}" \
  --max-prompt-len 2048 \
  --apply-chat-template-kwargs '{"enable_thinking":false}' \
  --seed "${SEED}"

MIXED_MANIFEST="${PREPARED_DIR}/mixed_on_policy.yaml"
MIXED_DATA_INDEX="${PREPARED_DIR}/mixed_data_index.json"
mapfile -t model_paths < <(
  python3 - "${MIXED_DATA_INDEX}" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))["model"]
print(model["hf_checkpoint"])
print(model["base_torch_dist_checkpoint"])
print(model["model_config"])
PY
)
if (( ${#model_paths[@]} != 3 )); then
  echo "Could not read the frozen base-model paths from ${MIXED_DATA_INDEX}." >&2
  exit 2
fi
BASE_HF_CHECKPOINT="${model_paths[0]}"
BASE_TORCH_DIST_CHECKPOINT="${model_paths[1]}"
BASE_MODEL_CONFIG="${model_paths[2]}"

total_prompt_groups=$((TRAIN_PROMPTS_PER_TASK * 4))
optimizer_updates=$((total_prompt_groups / ROLLOUT_BATCH_SIZE))
trajectories_per_task_per_update=$((prompts_per_task_per_batch * N_SAMPLES_PER_PROMPT))
global_batch_size=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

echo "Mixed GRPO model: Qwen3-1.7B base (${BASE_TORCH_DIST_CHECKPOINT})"
echo "Mixed domains: Code + Math + QA (science/GPQA source) + IF"
echo "Data budget: ${TRAIN_PROMPTS_PER_TASK} prompt groups/task; ${total_prompt_groups} total"
echo "Within each update: ${prompts_per_task_per_batch} prompts/task x ${N_SAMPLES_PER_PROMPT} responses = ${trajectories_per_task_per_update} trajectories/task"
echo "Training length: ${optimizer_updates} updates; global trajectory batch=${global_batch_size}"
echo "Four-GPU packing: max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}; SGLang mem_fraction=${SGLANG_MEM_FRACTION}; max_running=${SGLANG_MAX_RUNNING_REQUESTS}"
echo "Manifest: ${MIXED_MANIFEST}"
echo "Code reward requires the configured SandboxFusion service during training."
echo "Evaluation policy: disabled during training; one final-checkpoint evaluation after success."

if [[ "${PLAN_ONLY}" == "1" ]]; then
  echo "PLAN_ONLY=1; data contracts were validated and no training was launched."
  exit 0
fi

export OUTPUT_ROOT="${OUTPUT_ROOT:-${MIXED_DIR}}"
export RUN_NAME="${RUN_NAME:-qwen3_1.7b_code_math_qa_if_mixed_batch_grpo_adamw_usable4800_seed42}"
export WANDB_MODE WANDB_ENTITY WANDB_PROJECT
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
if [[ "${RESUME_INCOMPLETE_RUN}" == "1" ]]; then
  export LOAD_CHECKPOINT="${OUTPUT_ROOT}/${RUN_NAME}/checkpoints"
  export FRESH_START=0
else
  export LOAD_CHECKPOINT="${BASE_TORCH_DIST_CHECKPOINT}"
  export FRESH_START=1
fi
unset LOAD_CHECKPOINT_STEP NUM_EPOCH NUM_ROLLOUT

export HF_CHECKPOINT="${BASE_HF_CHECKPOINT}"
export MODEL_CONFIG="${BASE_MODEL_CONFIG}"
export DATA_MANIFEST="${MIXED_MANIFEST}"
export EXPERIMENT_DATA_INDEX="${MIXED_DATA_INDEX}"
export REWARD_CONFIG
export TASK=mixed_code_math_qa_if
export ALGORITHM=grpo
export OPTIMIZER=adamw
export SEED
export BATCH_PROFILE=responsive16
export TARGET_PROMPT_BUDGET="${total_prompt_groups}"
export ROLLOUT_BATCH_SIZE
export N_SAMPLES_PER_PROMPT
export GLOBAL_BATCH_SIZE="${global_batch_size}"
export ADAMW_LR=1e-6
export WEIGHT_DECAY=0.0
export MAX_PROMPT_LEN=2048
export MAX_RESPONSE_LEN=8192
export MAX_TOKENS_PER_GPU
export TRAIN_GPU_COUNT
export ROLLOUT_NUM_GPUS
export ROLLOUT_GPUS_PER_ENGINE
export SGLANG_MEM_FRACTION
export SGLANG_MAX_RUNNING_REQUESTS
export SAVE_INTERVAL=100
export DISABLE_EVAL=1
export USE_WANDB
export WANDB_GROUP="${WANDB_GROUP:-mixed_batch_grpo}"

export SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL:-http://127.0.0.1:8080}"
if [[ -z "${M2RL_SANDBOX_PREFLIGHT_MARKER:-}" && \
      -r /workspace/sandboxfusion-state/sandboxfusion_preflight.json ]]; then
  export M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json
fi

bash "${TRAIN_LAUNCHER}"

if [[ "${RUN_FINAL_EVAL}" == "1" ]]; then
  LOAD_CHECKPOINT="${OUTPUT_ROOT}/${RUN_NAME}/checkpoints" \
  MIXED_DIR="${MIXED_DIR}" \
  PREPARED_DIR="${PREPARED_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_NAME="${RUN_NAME}" \
  USE_WANDB="${USE_WANDB}" \
    bash "${FINAL_EVAL_LAUNCHER}"
else
  echo "RUN_FINAL_EVAL=0; final evaluation was not launched."
  echo "Run later: bash ${FINAL_EVAL_LAUNCHER}"
fi
