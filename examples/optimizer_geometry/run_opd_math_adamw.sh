#!/usr/bin/env bash
# Frozen paper cell: math x AdamW x seed 42.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset SEEDS NUM_ROLLOUT TARGET_PROMPT_BUDGET
export TASK=math
export OPTIMIZERS=adamw
export SEED=42
export BATCH_PROFILE=opd64x1
export ROLLOUT_BATCH_SIZE=64
export N_SAMPLES_PER_PROMPT=1
export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
export NUM_EPOCH=1
export MAX_PROMPT_LEN=2048
export MAX_RESPONSE_LEN=4096
export MAX_TOKENS_PER_GPU=10240
export EVAL_INTERVAL=50
export EVAL_MAX_RESPONSE_LEN=32768
export EVAL_MAX_CONCURRENCY=48
export SGLANG_MAX_RUNNING_REQUESTS=72
exec bash "${SCRIPT_DIR}/run_single_task_opd.sh"
