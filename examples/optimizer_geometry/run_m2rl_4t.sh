#!/usr/bin/env bash
# Launch the four-domain M2RL optimizer-geometry training preset.

set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${SLIME_DIR}/.." && pwd)"

M2RL_DATA_ROOT="${M2RL_DATA_ROOT:-${SLIME_DIR}/data/m2rl}"
export DATA_MANIFEST="${DATA_MANIFEST:-${M2RL_DATA_ROOT}/train/multitask_manifest.yaml}"
export REWARD_CONFIG="${REWARD_CONFIG:-${EXAMPLE_DIR}/configs/rewards.example.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SLIME_DIR}/outputs/m2rl_4t_geometry}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${CHECKPOINT_ROOT}/Qwen3-4B}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${CHECKPOINT_ROOT}/Qwen3-4B_torch_dist}"
export MODEL_CONFIG="${MODEL_CONFIG:-${SLIME_DIR}/scripts/models/qwen3-4B.sh}"

export OPTIMIZER="${OPTIMIZER:-adamw}"
export ALGORITHM="${ALGORITHM:-grpo}"
export SEED="${SEED:-42}"
export BATCH_PROFILE="${BATCH_PROFILE:-responsive16}"
export TASK="${TASK:-m2rl_4t}"
export RUN_NAME="${RUN_NAME:-m2rl_4t_${ALGORITHM}_${OPTIMIZER}_${BATCH_PROFILE}_seed${SEED}}"

if [[ ! -s "${DATA_MANIFEST}" ]]; then
  echo "Prepared M2RL data manifest not found: ${DATA_MANIFEST}" >&2
  echo "Run: bash ${EXAMPLE_DIR}/prepare_m2rl_dataset.sh" >&2
  exit 2
fi

if grep -Eq '(^|[[:space:]])(name|rm_type):[[:space:]]*(agent|workbench)([[:space:]]|$)' "${DATA_MANIFEST}"; then
  echo "Refusing to launch: the four-task manifest contains Agent/WorkBench: ${DATA_MANIFEST}" >&2
  exit 2
fi

echo "M2RL-4T manifest: ${DATA_MANIFEST}"
echo "Run: ${RUN_NAME}; algorithm=${ALGORITHM}; optimizer=${OPTIMIZER}; seed=${SEED}"
echo "Code rewards expect SandboxFusion at the URL in ${REWARD_CONFIG}."

exec bash "${EXAMPLE_DIR}/run_experiment.sh"
