#!/usr/bin/env bash
# Frozen Qwen3-1.7B IF PPO cell: AdamW actor, AdamW critic, seed 42, one dataset epoch.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset SEEDS NUM_ROLLOUT TARGET_PROMPT_BUDGET
export TASK=if
export RL_ALGORITHM=ppo
export OPTIMIZERS=adamw
export SEED=42
export BATCH_PROFILE=responsive16
export ROLLOUT_BATCH_SIZE=16
export N_SAMPLES_PER_PROMPT=4
export GLOBAL_BATCH_SIZE=64
export ADAMW_LR=2.5e-7
export APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
export NUM_EPOCH=1
export MAX_PROMPT_LEN=2048
export MAX_RESPONSE_LEN=8192
export MAX_TOKENS_PER_GPU=10240
export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export ROLLOUT_TOP_K=-1
export EVAL_INTERVAL=50
export SAVE_INTERVAL=100
export EVAL_MAX_RESPONSE_LEN=32768
export EVAL_MAX_CONCURRENCY=48
export SGLANG_MAX_RUNNING_REQUESTS=12
export REQUIRE_EVAL=1
exec bash "${SCRIPT_DIR}/run_single_task_rl.sh"
