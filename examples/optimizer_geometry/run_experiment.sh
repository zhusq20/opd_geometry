#!/usr/bin/env bash
# Run one optimizer x post-training-algorithm cell. External teacher, sandbox,
# and WorkBench services must already be reachable when their tasks are used.

set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
# The host also has a different /root/slime installation.  Make every local
# preflight/provenance Python process resolve this worktree, just like the Ray
# runtime below, so validation cannot pass against stale modules.
export PYTHONPATH="${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL:-http://127.0.0.1:8080}"
SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL%/}"
export SANDBOXFUSION_BASE_URL
export M2RL_SANDBOX_PREFLIGHT_MARKER="${M2RL_SANDBOX_PREFLIGHT_MARKER:-${SLIME_DIR}/data/m2rl/sandbox/sandboxfusion_preflight.json}"

OPTIMIZER="${OPTIMIZER:-adamw}"
ALGORITHM="${ALGORITHM:-grpo}"
SEED="${SEED:-42}"
BATCH_PROFILE="${BATCH_PROFILE:-responsive16}"
export BATCH_PROFILE
MODEL_CONFIG="${MODEL_CONFIG:-${SLIME_DIR}/scripts/models/qwen3-4B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/root/Qwen3-4B}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-/root/Qwen3-4B_torch_dist}"
DATA_MANIFEST="${DATA_MANIFEST:?Set DATA_MANIFEST to the generated multi-task manifest}"
REWARD_CONFIG="${REWARD_CONFIG:-${EXAMPLE_DIR}/configs/rewards.example.yaml}"
WORKPLACE_ASSISTANT_RESOURCES_SERVER_URL="${WORKPLACE_ASSISTANT_RESOURCES_SERVER_URL:-http://127.0.0.1:12000}"
TEACHER_CONFIG="${TEACHER_CONFIG:-}"
PPO_CONFIG="${PPO_CONFIG:-${EXAMPLE_DIR}/configs/ppo_roles.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-}"
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-}"
if [[ "${TASK:-}" == "math" ]]; then
  case "${ALGORITHM}" in
    grpo|ppo)
      EVAL_CONFIG="${EVAL_CONFIG:-${SLIME_DIR}/data/m2rl/single_task/math/math_eval_aime24_math500.yaml}"
      EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-32768}"
      EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-48}"
      ;;
  esac
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${SLIME_DIR}/optimizer_geometry_runs}"
RUN_NAME="${RUN_NAME:-${ALGORITHM}_${OPTIMIZER}_${BATCH_PROFILE}_seed${SEED}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
GEOMETRY_DIR="${RUN_DIR}/geometry"
METRICS_DIR="${METRICS_DIR:-${RUN_DIR}/metrics}"
EVAL_ARTIFACT_DIR="${EVAL_ARTIFACT_DIR:-${RUN_DIR}/eval_artifacts}"
RUN_MANIFEST_PATH="${RUN_MANIFEST_PATH:-${RUN_DIR}/provenance/run_manifest.json}"
COMPLETION_MARKER_PATH="${COMPLETION_MARKER_PATH:-${RUN_DIR}/run_complete.json}"
USE_WANDB="${USE_WANDB:-1}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
FRESH_START="${FRESH_START:-1}"
DRY_RUN="${DRY_RUN:-0}"
case "${SAVE_CHECKPOINTS}" in
  0|1) ;;
  *) echo "SAVE_CHECKPOINTS must be 0 or 1." >&2; exit 2 ;;
esac
case "${FRESH_START}" in
  0|1) ;;
  *) echo "FRESH_START must be 0 (resume) or 1 (new run)." >&2; exit 2 ;;
esac
case "${DRY_RUN}" in
  0|1) ;;
  *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
OPTIMIZER_GEOMETRY_CRITIC_SAVE="${OPTIMIZER_GEOMETRY_CRITIC_SAVE:-${RUN_DIR}/checkpoints/critic}"
if [[ "${FRESH_START}" == "1" ]]; then
  DEFAULT_CRITIC_LOAD="${LOAD_CHECKPOINT}"
else
  DEFAULT_CRITIC_LOAD="${LOAD_CHECKPOINT}/critic"
fi
OPTIMIZER_GEOMETRY_CRITIC_LOAD="${OPTIMIZER_GEOMETRY_CRITIC_LOAD:-${DEFAULT_CRITIC_LOAD}}"
export OPTIMIZER_GEOMETRY_CRITIC_LOAD OPTIMIZER_GEOMETRY_CRITIC_SAVE

NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${NUM_GPUS}}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
# Responsive profiles reduce the latency of one rollout/update.  The LR transfer is the conservative
# square-root rule from the 256-prompt reference recipe.  Adam beta2 instead
# preserves the second-moment half-life in samples:
#   beta2_new = 0.98 ** (prompt_batch / 256).
# See OPTIMIZER_COMPARISON_PROTOCOL_zh.md for evidence, caveats, and tuning.
case "${BATCH_PROFILE}" in
  opd64x1)
    # Pure-OPD paper cells use 64 independent questions and one sampled
    # response each.  This still produces 64 training samples/update, so it
    # retains the responsive16 optimizer settings that were calibrated at the
    # same global response batch size (16 prompts x 4 responses).
    PROFILE_ROLLOUT_BATCH_SIZE=64
    PROFILE_ADAMW_LR=2.5e-7
    PROFILE_GRPO_SGD_LR=2.5e-2
    PROFILE_OTHER_SGD_LR=2.5e-3
    PROFILE_ADAM_BETA2=0.9987381276
    PROFILE_CRITIC_LR=2.5e-6
    PROFILE_CRITIC_WEIGHT_DECAY=0.025
    ;;
  responsive16)
    PROFILE_ROLLOUT_BATCH_SIZE=16
    PROFILE_ADAMW_LR=2.5e-7
    PROFILE_GRPO_SGD_LR=2.5e-2
    PROFILE_OTHER_SGD_LR=2.5e-3
    PROFILE_ADAM_BETA2=0.9987381276
    PROFILE_CRITIC_LR=2.5e-6
    PROFILE_CRITIC_WEIGHT_DECAY=0.025
    ;;
  responsive8)
    PROFILE_ROLLOUT_BATCH_SIZE=8
    PROFILE_ADAMW_LR=1.8e-7
    PROFILE_GRPO_SGD_LR=1.8e-2
    PROFILE_OTHER_SGD_LR=1.8e-3
    PROFILE_ADAM_BETA2=0.9993688646
    PROFILE_CRITIC_LR=1.8e-6
    PROFILE_CRITIC_WEIGHT_DECAY=0.018
    ;;
  reference256)
    PROFILE_ROLLOUT_BATCH_SIZE=256
    PROFILE_ADAMW_LR=1e-6
    PROFILE_GRPO_SGD_LR=1e-1
    PROFILE_OTHER_SGD_LR=1e-2
    PROFILE_ADAM_BETA2=0.98
    PROFILE_CRITIC_LR=1e-5
    PROFILE_CRITIC_WEIGHT_DECAY=0.1
    ;;
  *)
    echo "Unsupported BATCH_PROFILE=${BATCH_PROFILE}; use opd64x1, responsive16, responsive8, or reference256." >&2
    exit 2
    ;;
esac
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-${PROFILE_ROLLOUT_BATCH_SIZE}}"
if ! [[ "${ROLLOUT_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROLLOUT_BATCH_SIZE must be a positive integer." >&2
  exit 2
fi
if [[ "${ROLLOUT_BATCH_SIZE}" -ne "${PROFILE_ROLLOUT_BATCH_SIZE}" ]]; then
  echo "BATCH_PROFILE=${BATCH_PROFILE} requires ROLLOUT_BATCH_SIZE=${PROFILE_ROLLOUT_BATCH_SIZE}, got ${ROLLOUT_BATCH_SIZE}." >&2
  echo "Choose the matching opd64x1/responsive8/responsive16/reference256 profile so optimizer hyperparameters stay coherent." >&2
  exit 2
fi
RUN_LENGTH_ARGS=()
if [[ -n "${NUM_EPOCH:-}" ]]; then
  if ! [[ "${NUM_EPOCH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_EPOCH must be a positive integer." >&2
    exit 2
  fi
  if [[ -n "${NUM_ROLLOUT:-}" || -n "${TARGET_PROMPT_BUDGET:-}" ]]; then
    echo "NUM_EPOCH cannot be combined with NUM_ROLLOUT or TARGET_PROMPT_BUDGET." >&2
    exit 2
  fi
  RUN_LENGTH_ARGS=(--num-epoch "${NUM_EPOCH}" --include-epoch-tail)
  RUN_LENGTH_SUMMARY="dataset_epochs=${NUM_EPOCH}; usable prompt count and final partial batch are derived at runtime"
else
  TARGET_PROMPT_BUDGET="${TARGET_PROMPT_BUDGET:-51200}"
  if ! [[ "${TARGET_PROMPT_BUDGET}" =~ ^[1-9][0-9]*$ ]] || \
     (( TARGET_PROMPT_BUDGET % ROLLOUT_BATCH_SIZE != 0 )); then
    echo "TARGET_PROMPT_BUDGET must be a positive multiple of ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}." >&2
    exit 2
  fi
  NUM_ROLLOUT="${NUM_ROLLOUT:-$((TARGET_PROMPT_BUDGET / ROLLOUT_BATCH_SIZE))}"
  if ! [[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ROLLOUT must be a positive integer." >&2
    exit 2
  fi
  RUN_LENGTH_ARGS=(--num-rollout "${NUM_ROLLOUT}")
  RUN_LENGTH_SUMMARY="updates=${NUM_ROLLOUT}; prompt_budget=$((NUM_ROLLOUT * ROLLOUT_BATCH_SIZE))"
fi
if [[ -z "${N_SAMPLES_PER_PROMPT+x}" ]]; then
  if [[ "${ALGORITHM}" == "sft_opd" ]]; then
    N_SAMPLES_PER_PROMPT=1
  else
    N_SAMPLES_PER_PROMPT=4
  fi
fi
if ! [[ "${N_SAMPLES_PER_PROMPT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_SAMPLES_PER_PROMPT must be a positive integer." >&2
  exit 2
fi
if [[ "${BATCH_PROFILE}" == "opd64x1" && "${N_SAMPLES_PER_PROMPT}" != "1" ]]; then
  echo "BATCH_PROFILE=opd64x1 requires N_SAMPLES_PER_PROMPT=1." >&2
  exit 2
fi
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
ENFORCE_ONE_UPDATE_PER_ROLLOUT="${ENFORCE_ONE_UPDATE_PER_ROLLOUT:-1}"
case "${ENFORCE_ONE_UPDATE_PER_ROLLOUT}" in
  0|1) ;;
  *) echo "ENFORCE_ONE_UPDATE_PER_ROLLOUT must be 0 or 1." >&2; exit 2 ;;
esac
EXPECTED_GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if [[ "${ENFORCE_ONE_UPDATE_PER_ROLLOUT}" == "1" && \
      "${GLOBAL_BATCH_SIZE}" -ne "${EXPECTED_GLOBAL_BATCH_SIZE}" ]]; then
  echo "The fixed comparison profile requires GLOBAL_BATCH_SIZE=ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT=${EXPECTED_GLOBAL_BATCH_SIZE}." >&2
  echo "Set ENFORCE_ONE_UPDATE_PER_ROLLOUT=0 only for an explicitly non-primary custom run." >&2
  exit 2
fi
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
if ! [[ "${MAX_PROMPT_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PROMPT_LEN must be a positive integer." >&2
  exit 2
fi
if ! [[ "${MAX_RESPONSE_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RESPONSE_LEN must be a positive integer." >&2
  exit 2
fi
DEFAULT_MAX_TOKENS_PER_GPU=10240
if (( MAX_PROMPT_LEN + MAX_RESPONSE_LEN > DEFAULT_MAX_TOKENS_PER_GPU )); then
  DEFAULT_MAX_TOKENS_PER_GPU=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
fi
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${DEFAULT_MAX_TOKENS_PER_GPU}}"
if ! [[ "${MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TOKENS_PER_GPU must be a positive integer." >&2
  exit 2
fi
# Match the paper GRPO/PPO cadence to OPD in optimizer-update units. Other
# algorithms retain the prompt-normalized defaults used by custom experiments.
SAVE_PROMPT_INTERVAL="${SAVE_PROMPT_INTERVAL:-5120}"
EVAL_PROMPT_INTERVAL="${EVAL_PROMPT_INTERVAL:-5120}"
GEOMETRY_PROMPT_INTERVAL="${GEOMETRY_PROMPT_INTERVAL:-256}"
case "${ALGORITHM}" in
  grpo|ppo)
    SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
    ;;
  *)
    SAVE_INTERVAL="${SAVE_INTERVAL:-$(((SAVE_PROMPT_INTERVAL + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-$(((EVAL_PROMPT_INTERVAL + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))}"
    ;;
esac
GEOMETRY_INTERVAL="${GEOMETRY_INTERVAL:-$(((GEOMETRY_PROMPT_INTERVAL + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))}"
GEOMETRY_PROJECTION_DIM="${GEOMETRY_PROJECTION_DIM:-256}"
GEOMETRY_SUPPORT_SAMPLE_SIZE="${GEOMETRY_SUPPORT_SAMPLE_SIZE:-1024}"
GEOMETRY_SUPPORT_WINDOW="${GEOMETRY_SUPPORT_WINDOW:-8}"
GEOMETRY_MATRIX_SAMPLE_COUNT="${GEOMETRY_MATRIX_SAMPLE_COUNT:-1}"
GEOMETRY_MATRIX_RANDOMIZED_RANK="${GEOMETRY_MATRIX_RANDOMIZED_RANK:-16}"
GEOMETRY_CAPTURE_ROLLOUT_ENTROPY="${GEOMETRY_CAPTURE_ROLLOUT_ENTROPY:-1}"
GEOMETRY_WANDB_GROUPS="${GEOMETRY_WANDB_GROUPS:-global,optimizer_branch/adam,optimizer_branch/sgd,optimizer_branch/muon_matrix,optimizer_branch/adam_fallback}"
case "${GEOMETRY_CAPTURE_ROLLOUT_ENTROPY}" in
  0|1) ;;
  *) echo "GEOMETRY_CAPTURE_ROLLOUT_ENTROPY must be 0 or 1." >&2; exit 2 ;;
esac

# These are protocol defaults for a launch that has not supplied the frozen
# result of the disjoint tuning sweep. Scalable Muon's 0.2 spectral scaling
# makes AdamW and Muon learning-rate units comparable. Vanilla SGD needs a
# larger candidate range because it has no adaptive normalization.
case "${ALGORITHM}" in
  grpo)
    DEFAULT_ADAMW_LR="${PROFILE_ADAMW_LR}"
    DEFAULT_SGD_LR="${PROFILE_GRPO_SGD_LR}"
    ;;
  ppo)
    DEFAULT_ADAMW_LR="${PROFILE_ADAMW_LR}"
    DEFAULT_SGD_LR="${PROFILE_OTHER_SGD_LR}"
    ;;
  opd|sft_opd|grpo_opd|ppo_opd)
    DEFAULT_ADAMW_LR="${PROFILE_ADAMW_LR}"
    DEFAULT_SGD_LR="${PROFILE_OTHER_SGD_LR}"
    ;;
  *)
    # The algorithm validation below emits the user-facing error.
    DEFAULT_ADAMW_LR="${PROFILE_ADAMW_LR}"
    DEFAULT_SGD_LR="${PROFILE_OTHER_SGD_LR}"
    ;;
esac
DEFAULT_MUON_LR="${DEFAULT_ADAMW_LR}"
PRIMARY_WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
OPTIMIZER_GEOMETRY_CRITIC_LR="${OPTIMIZER_GEOMETRY_CRITIC_LR:-${PROFILE_CRITIC_LR}}"
OPTIMIZER_GEOMETRY_CRITIC_WEIGHT_DECAY="${OPTIMIZER_GEOMETRY_CRITIC_WEIGHT_DECAY:-${PROFILE_CRITIC_WEIGHT_DECAY}}"
OPTIMIZER_GEOMETRY_CRITIC_BETA2="${OPTIMIZER_GEOMETRY_CRITIC_BETA2:-${PROFILE_ADAM_BETA2}}"
export OPTIMIZER_GEOMETRY_CRITIC_LR OPTIMIZER_GEOMETRY_CRITIC_WEIGHT_DECAY OPTIMIZER_GEOMETRY_CRITIC_BETA2

case "${OPTIMIZER}" in
  adamw|adam)
    OPTIMIZER_ARGS=(
      --optimizer adam
      --lr "${ADAMW_LR:-${DEFAULT_ADAMW_LR}}"
      --weight-decay "${PRIMARY_WEIGHT_DECAY}"
      --adam-beta1 "${ADAM_BETA1:-0.9}"
      --adam-beta2 "${ADAM_BETA2:-${PROFILE_ADAM_BETA2}}"
      --adam-eps "${ADAM_EPS:-1e-8}"
    )
    ;;
  sgd)
    OPTIMIZER_ARGS=(
      --optimizer sgd
      --lr "${SGD_LR:-${DEFAULT_SGD_LR}}"
      --weight-decay "${PRIMARY_WEIGHT_DECAY}"
      --sgd-momentum "${SGD_MOMENTUM:-0.0}"
    )
    ;;
  muon|dist_muon)
    OPTIMIZER_ARGS=(
      --optimizer "${OPTIMIZER}"
      --lr "${MUON_LR:-${DEFAULT_MUON_LR}}"
      --weight-decay "${PRIMARY_WEIGHT_DECAY}"
      --adam-beta1 "${ADAM_BETA1:-0.9}"
      --adam-beta2 "${ADAM_BETA2:-${PROFILE_ADAM_BETA2}}"
      --adam-eps "${ADAM_EPS:-1e-8}"
      --muon-momentum "${MUON_MOMENTUM:-0.95}"
      --muon-num-ns-steps "${MUON_NS_STEPS:-5}"
      --muon-scale-mode "${MUON_SCALE_MODE:-spectral}"
      --muon-tp-mode "${MUON_TP_MODE:-blockwise}"
      --muon-extra-scale-factor "${MUON_EXTRA_SCALE_FACTOR:-0.2}"
      --muon-fp32-matmul-prec "${MUON_FP32_MATMUL_PREC:-medium}"
    )
    ;;
  *)
    echo "Unsupported OPTIMIZER=${OPTIMIZER}; use adamw, sgd, muon, or dist_muon." >&2
    exit 2
    ;;
esac
OPTIMIZER_ARGS+=(
  --lr-decay-style constant
  --lr-warmup-iters "${LR_WARMUP_ITERS:-0}"
  --clip-grad "${CLIP_GRAD:-1.0}"
)

ALGORITHM_ARGS=(
  --entropy-coef 0.0
  --kl-coef 0.0
  --kl-loss-coef 0.0
)
RM_ARGS=(
  --custom-rm-path slime_plugins.m2rl.rewards.reward
)
RM_COMMON_ARGS=(--m2rl-reward-config "${REWARD_CONFIG}")
ROLE_ARGS=()
case "${ALGORITHM}" in
  grpo)
    ALGORITHM_ARGS+=(
      --advantage-estimator grpo
      --eps-clip 0.2
      --eps-clip-high 0.28
      --use-tis
      --tis-clip 2.0
      --tis-clip-low 0.0
    )
    ;;
  ppo)
    ALGORITHM_ARGS+=(
      --advantage-estimator ppo
      --eps-clip 0.2
      --value-clip 0.2
      --normalize-advantages
      --use-tis
      --tis-clip 2.0
      --tis-clip-low 0.0
    )
    ROLE_ARGS+=(--megatron-config-path "${PPO_CONFIG}")
    ;;
  opd)
    ALGORITHM_ARGS+=(
      --advantage-estimator grpo
      --use-opd
      --opd-type sglang
      --opd-kl-coef "${OPD_KL_COEF:-1.0}"
      --opd-teacher-router-config "${TEACHER_CONFIG}"
      --opd-task-reward-weight 0.0
      --eps-clip 0.2
      --eps-clip-high 0.2
    )
    RM_ARGS=(
      --custom-rm-path slime_plugins.m2rl.opd.teacher_reward
      --custom-reward-post-process-path slime_plugins.m2rl.opd.post_process_rewards
    )
    ;;
  sft_opd)
    ALGORITHM_ARGS+=(
      --advantage-estimator grpo
      --use-opd
      --opd-type sglang
      --opd-kl-coef "${OPD_KL_COEF:-1.0}"
      --opd-teacher-router-config "${TEACHER_CONFIG}"
      --opd-task-reward-weight 0.0
      --custom-advantage-function-path slime_plugins.m2rl.hybrid.hybrid_advantages
      --loss-type custom_loss
      --custom-loss-function-path slime_plugins.m2rl.hybrid.hybrid_loss_function
      --loss-mask-type qwen3
      --hybrid-sft-loss-coef "${SFT_LOSS_COEF:-1.0}"
      --hybrid-opd-loss-coef "${HYBRID_OPD_LOSS_COEF:-1.0}"
      --eps-clip 0.2
      --eps-clip-high 0.28
    )
    RM_ARGS=(
      --custom-rm-path slime_plugins.m2rl.opd.teacher_reward
      --custom-reward-post-process-path slime_plugins.m2rl.hybrid.post_process_sft_opd_rewards
    )
    ;;
  grpo_opd|ppo_opd)
    BASE_ESTIMATOR="${ALGORITHM%_opd}"
    ALGORITHM_ARGS+=(
      --advantage-estimator "${BASE_ESTIMATOR}"
      --use-opd
      --opd-type sglang
      --opd-kl-coef "${OPD_KL_COEF:-1.0}"
      --opd-task-reward-weight "${OPD_TASK_REWARD_WEIGHT:-1.0}"
      --eps-clip 0.2
      --use-tis
      --tis-clip 2.0
      --tis-clip-low 0.0
    )
    if [[ "${BASE_ESTIMATOR}" == "ppo" ]]; then
      ALGORITHM_ARGS+=(--value-clip 0.2 --normalize-advantages)
      ROLE_ARGS+=(--megatron-config-path "${PPO_CONFIG}")
    else
      ALGORITHM_ARGS+=(--eps-clip-high 0.28)
    fi
    RM_ARGS=(
      --custom-rm-path slime_plugins.m2rl.opd.combined_reward
      --custom-reward-post-process-path slime_plugins.m2rl.opd.post_process_combined_rewards
    )
    ;;
  *)
    echo "Unsupported ALGORITHM=${ALGORITHM}; use grpo, ppo, opd, sft_opd, grpo_opd, or ppo_opd." >&2
    exit 2
    ;;
esac

# Qwen3 enables thinking when enable_thinking is omitted. RL cells use a
# frozen non-thinking policy template; OPD additionally shares the exact
# student-rendered token IDs with the scoring teacher. Keep this an explicit,
# validated experiment invariant instead of relying on tokenizer defaults.
if [[ -z "${APPLY_CHAT_TEMPLATE_KWARGS+x}" ]]; then
  case "${ALGORITHM}" in
    grpo|ppo|opd|sft_opd|grpo_opd|ppo_opd)
      APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
      ;;
    *)
      APPLY_CHAT_TEMPLATE_KWARGS='{}'
      ;;
  esac
fi
if ! python3 - "${APPLY_CHAT_TEMPLATE_KWARGS}" "${ALGORITHM}" <<'PY'
import json
import sys

try:
    kwargs = json.loads(sys.argv[1])
except (TypeError, ValueError):
    raise SystemExit(1)
if not isinstance(kwargs, dict):
    raise SystemExit(1)
if sys.argv[2] in {"grpo", "ppo", "opd", "sft_opd", "grpo_opd", "ppo_opd"} and kwargs.get("enable_thinking") is not False:
    raise SystemExit(1)
PY
then
  echo "APPLY_CHAT_TEMPLATE_KWARGS must be a JSON object; GRPO/PPO and OPD experiments require enable_thinking=false." >&2
  exit 2
fi

VALIDATE_ARGS=(
  --optimizer "${OPTIMIZER}"
  --algorithm "${ALGORITHM}"
  --manifest "${DATA_MANIFEST}"
  --model-config "${MODEL_CONFIG}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load-checkpoint "${LOAD_CHECKPOINT}"
  --ppo-config "${PPO_CONFIG}"
  --reward-config "${REWARD_CONFIG}"
  --workbench-url "${WORKPLACE_ASSISTANT_RESOURCES_SERVER_URL}"
)
echo "Batch profile: ${BATCH_PROFILE}; prompts/full_update=${ROLLOUT_BATCH_SIZE}; responses/full_update=${GLOBAL_BATCH_SIZE}; ${RUN_LENGTH_SUMMARY}"
echo "Cadence: exact geometry + W&B train/rollout every update; low-frequency geometry=${GEOMETRY_INTERVAL}; eval=${EVAL_INTERVAL}; checkpoint=${SAVE_INTERVAL} updates"
if [[ -n "${TEACHER_CONFIG}" ]]; then
  VALIDATE_ARGS+=(--teacher-config "${TEACHER_CONFIG}")
fi
if [[ -n "${EVAL_CONFIG}" ]]; then
  VALIDATE_ARGS+=(--eval-config "${EVAL_CONFIG}")
  if [[ -n "${EVAL_MAX_RESPONSE_LEN}" ]]; then
    if ! [[ "${EVAL_MAX_RESPONSE_LEN}" =~ ^[1-9][0-9]*$ ]]; then
      echo "EVAL_MAX_RESPONSE_LEN must be a positive integer." >&2
      exit 2
    fi
    VALIDATE_ARGS+=(--expected-eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}")
  fi
fi
if [[ "${CHECK_RUNTIME_DEPS:-1}" == "1" ]]; then
  VALIDATE_ARGS+=(--check-runtime-deps)
fi
python3 "${EXAMPLE_DIR}/validate_experiment.py" "${VALIDATE_ARGS[@]}"

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ "${FRESH_START}" == "1" && -d "${RUN_DIR}" && -n "$(find "${RUN_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing a fresh run in non-empty RUN_DIR=${RUN_DIR}; choose a new RUN_NAME or set FRESH_START=0." >&2
    exit 2
  fi
  if [[ "${FRESH_START}" == "0" ]]; then
    if [[ ! -f "${GEOMETRY_DIR}/actor/initial_projection.pt" ]]; then
      echo "Exact resume requires ${GEOMETRY_DIR}/actor/initial_projection.pt from the original run." >&2
      exit 2
    fi
    if ! compgen -G "${GEOMETRY_DIR}/actor/exact_reference/rank_*.pt" >/dev/null; then
      echo "Exact resume requires the per-rank FP32/model-dtype reference tensors under ${GEOMETRY_DIR}/actor/exact_reference/." >&2
      exit 2
    fi
    if ! compgen -G "${LOAD_CHECKPOINT}/rollout/multitask_dataset_state_dict_*.pt" >/dev/null; then
      echo "Exact resume requires a multi-task sampler state under ${LOAD_CHECKPOINT}/rollout/." >&2
      exit 2
    fi
    if [[ "${ALGORITHM}" == "ppo" || "${ALGORITHM}" == "ppo_opd" ]] && \
       [[ ! -f "${OPTIMIZER_GEOMETRY_CRITIC_LOAD}/latest_checkpointed_iteration.txt" ]]; then
      echo "PPO resume requires the separately saved critic checkpoint at ${OPTIMIZER_GEOMETRY_CRITIC_LOAD}." >&2
      exit 2
    fi
  fi
  mkdir -p "${RUN_DIR}" "${GEOMETRY_DIR}"
fi
# MODEL_ARGS is supplied by the selected model definition.
source "${MODEL_CONFIG}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load "${LOAD_CHECKPOINT}"
)
if [[ "${SAVE_CHECKPOINTS}" == "1" ]]; then
  CKPT_ARGS+=(--save "${RUN_DIR}/checkpoints" --save-interval "${SAVE_INTERVAL}")
elif [[ "${FRESH_START}" == "0" ]]; then
  echo "SAVE_CHECKPOINTS=0 is only allowed for disposable fresh feasibility tests." >&2
  exit 2
fi
if [[ "${FRESH_START}" == "1" ]]; then
  # A release checkpoint commonly stores iteration 0. Without this explicit
  # override Slime interprets it as a completed rollout and silently starts at
  # rollout 1, making a one-rollout smoke test do zero optimizer updates.
  CKPT_ARGS+=(--no-load-optim --no-load-rng --start-rollout-id 0)
fi
ROLLOUT_ARGS=(
  --prompt-data "${DATA_MANIFEST}"
  --data-source-path slime_plugins.m2rl.data_source.MultiTaskRolloutDataSource
  --custom-generate-function-path slime_plugins.m2rl.generate.generate
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --tool-key tools
  --apply-chat-template
  --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"
  --rollout-shuffle
  --rollout-seed "${SEED}"
  --m2rl-task-sampling-seed "${SEED}"
  "${RUN_LENGTH_ARGS[@]}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-prompt-len "${MAX_PROMPT_LEN}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
  --rollout-top-k "${ROLLOUT_TOP_K:--1}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)
PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if [[ "${TP_SIZE}" -gt 1 ]]; then
  PERF_ARGS+=(--sequence-parallel)
fi
GEOMETRY_ARGS=(
  --geometry-output-dir "${GEOMETRY_DIR}"
  --geometry-interval "${GEOMETRY_INTERVAL}"
  --geometry-projection-dim "${GEOMETRY_PROJECTION_DIM}"
  --geometry-seed "${SEED}"
  --geometry-support-sample-size "${GEOMETRY_SUPPORT_SAMPLE_SIZE}"
  --geometry-support-window "${GEOMETRY_SUPPORT_WINDOW}"
  --geometry-matrix-sample-count "${GEOMETRY_MATRIX_SAMPLE_COUNT}"
  --geometry-matrix-randomized-rank "${GEOMETRY_MATRIX_RANDOMIZED_RANK}"
  --geometry-group-by layer
  --geometry-roles actor
  --geometry-wandb-groups "${GEOMETRY_WANDB_GROUPS}"
  --experiment-task "${TASK:-multi}"
  --experiment-teacher "${TEACHER_NAME:-unspecified}"
  --experiment-condition "${ALGORITHM}"
  --experiment-name "${RUN_NAME}"
  --experiment-optimizer "${OPTIMIZER}"
  --experiment-data-index "${EXPERIMENT_DATA_INDEX:-}"
  --forgetting-output-dir "${RUN_DIR}"
  --metrics-output-dir "${METRICS_DIR}"
  --eval-artifact-dir "${EVAL_ARTIFACT_DIR}"
  --run-manifest-path "${RUN_MANIFEST_PATH}"
  --completion-marker-path "${COMPLETION_MARKER_PATH}"
  --custom-eval-rollout-log-function-path slime_plugins.geometry.forgetting.log_eval_and_forgetting
)
if [[ "${GEOMETRY_CAPTURE_ROLLOUT_ENTROPY}" == "1" ]]; then
  GEOMETRY_ARGS+=(--use-rollout-entropy)
fi
EVAL_ARGS=()
if [[ -n "${EVAL_CONFIG}" ]]; then
  EVAL_ARGS+=(--eval-config "${EVAL_CONFIG}" --eval-interval "${EVAL_INTERVAL}")
  if [[ -n "${EVAL_MAX_RESPONSE_LEN}" ]]; then
    EVAL_ARGS+=(--eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}")
  fi
  if [[ -n "${EVAL_MAX_CONCURRENCY}" ]]; then
    if ! [[ "${EVAL_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
      echo "EVAL_MAX_CONCURRENCY must be a positive integer." >&2
      exit 2
    fi
    EVAL_ARGS+=(--eval-max-concurrency "${EVAL_MAX_CONCURRENCY}")
  fi
fi
SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.7}"
  --sglang-enable-deterministic-inference
)
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]]; then
  if ! [[ "${SGLANG_MAX_RUNNING_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SGLANG_MAX_RUNNING_REQUESTS must be a positive integer." >&2
    exit 2
  fi
  SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
MISC_ARGS=(
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${NUM_GPUS}"
  --colocate
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --seed "${SEED}"
)

if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_MODE_VALUE="${WANDB_MODE:-online}"
  MISC_ARGS+=(
    --use-wandb
    --wandb-mode "${WANDB_MODE_VALUE}"
    --wandb-dir "${WANDB_DIR:-${RUN_DIR}/wandb}"
    --wandb-team "${WANDB_ENTITY:-zsqzz}"
    --wandb-project "${WANDB_PROJECT:-iclr2027-opd-geometry}"
    --wandb-group "${WANDB_GROUP:-${TASK:-multi}_${ALGORITHM}}"
    --wandb-run-name "${WANDB_RUN_NAME:-${RUN_NAME}}"
    --wandb-run-id-file "${WANDB_RUN_ID_FILE:-${RUN_DIR}/wandb_run_id.txt}"
    --disable-wandb-random-suffix
  )
  if [[ "${WANDB_MODE_VALUE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set; W&B online mode will use an existing wandb login if available." >&2
  fi
fi
MISC_ARGS+=(
  --log-passrate
  --workplace-assistant-resources-server-url "${WORKPLACE_ASSISTANT_RESOURCES_SERVER_URL}"
)
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a USER_EXTRA_ARGS <<< "${EXTRA_ARGS}"
else
  USER_EXTRA_ARGS=()
fi

TRAIN_CMD=(
  python3 "${SLIME_DIR}/train.py"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${ALGORITHM_ARGS[@]}"
  "${RM_ARGS[@]}"
  "${RM_COMMON_ARGS[@]}"
  "${ROLE_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${GEOMETRY_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${USER_EXTRA_ARGS[@]}"
)

printf 'Experiment command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

PROVENANCE_ARGS=(
  start
  --repo "${SLIME_DIR}"
  --run-dir "${RUN_DIR}"
  --input "${DATA_MANIFEST}"
  --input "${MODEL_CONFIG}"
  --input "${REWARD_CONFIG}"
  --checkpoint "${HF_CHECKPOINT}"
  --checkpoint "${LOAD_CHECKPOINT}"
)
if [[ -n "${EVAL_CONFIG}" ]]; then
  PROVENANCE_ARGS+=(--input "${EVAL_CONFIG}")
  AUTO_EVAL_INDEX="$(dirname -- "${EVAL_CONFIG}")/livecodebench_index.json"
  if [[ -f "${AUTO_EVAL_INDEX}" ]]; then
    PROVENANCE_ARGS+=(--input "${AUTO_EVAL_INDEX}")
  fi
fi
if [[ -n "${EXPERIMENT_DATA_INDEX:-}" && -f "${EXPERIMENT_DATA_INDEX}" ]]; then
  PROVENANCE_ARGS+=(--input "${EXPERIMENT_DATA_INDEX}")
fi
if [[ -n "${EXPERIMENT_EVAL_INDEX:-}" && -f "${EXPERIMENT_EVAL_INDEX}" ]]; then
  PROVENANCE_ARGS+=(--input "${EXPERIMENT_EVAL_INDEX}")
fi
if [[ -f "${M2RL_SANDBOX_PREFLIGHT_MARKER}" ]]; then
  PROVENANCE_ARGS+=(--input "${M2RL_SANDBOX_PREFLIGHT_MARKER}")
fi
if [[ -n "${TEACHER_CONFIG}" ]]; then
  PROVENANCE_ARGS+=(--input "${TEACHER_CONFIG}")
fi
if [[ "${ALGORITHM}" == "ppo" || "${ALGORITHM}" == "ppo_opd" ]]; then
  PROVENANCE_ARGS+=(--input "${PPO_CONFIG}")
fi
if [[ "${FRESH_START}" == "0" ]]; then
  PROVENANCE_ARGS+=(--resume)
fi
if [[ "${ALLOW_SOURCE_CHANGE_ON_RESUME:-0}" == "1" ]]; then
  PROVENANCE_ARGS+=(--allow-source-change)
fi
python3 "${EXAMPLE_DIR}/run_provenance.py" "${PROVENANCE_ARGS[@]}" "${TRAIN_CMD[@]}"

RAY_ADDRESS="${RAY_ADDRESS:-}"
RAY_START_PID=""
# `ray stop` scans every local Ray process.  A blocking supervisor owns this
# node's children and its SIGTERM handler shuts down only that node.
cleanup() {
  if [[ -n "${RAY_START_PID}" && "${KEEP_RAY:-0}" != "1" ]]; then
    kill -TERM "${RAY_START_PID}" 2>/dev/null || true
    wait "${RAY_START_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if [[ -z "${RAY_ADDRESS}" ]]; then
  if [[ "${START_RAY:-1}" != "1" ]]; then
    echo "Set RAY_ADDRESS or START_RAY=1." >&2
    exit 2
  fi
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
  ray start \
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT}" \
    --block &
  RAY_START_PID=$!
  RAY_ADDRESS="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"
  RAY_READY=0
  for ((attempt = 1; attempt <= 60; attempt++)); do
    if ! kill -0 "${RAY_START_PID}" 2>/dev/null; then
      echo "Ray head process exited before its Jobs API became ready at ${RAY_ADDRESS}." >&2
      wait "${RAY_START_PID}" 2>/dev/null || true
      exit 1
    fi
    if ray job list --address "${RAY_ADDRESS}" >/dev/null 2>&1; then
      RAY_READY=1
      break
    fi
    sleep 1
  done
  if [[ "${RAY_READY}" != "1" ]]; then
    echo "Timed out waiting for the Ray Jobs API at ${RAY_ADDRESS}." >&2
    exit 1
  fi
fi

MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
RUNTIME_PYTHONPATH="${SLIME_DIR}:${MEGATRON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_ENV_JSON="$(python3 -c 'import json,os,sys; env={"PYTHONPATH": sys.argv[1], "CUDA_DEVICE_MAX_CONNECTIONS": "1", "PYTHONUNBUFFERED": "1"}; keys=("M2RL_EVAL_DIR", "OPD_TEACHER_URL", "QWEN3_8B_TEACHER_URL", "OPTIMIZER_GEOMETRY_CRITIC_LOAD", "OPTIMIZER_GEOMETRY_CRITIC_SAVE", "OPTIMIZER_GEOMETRY_CRITIC_LR", "OPTIMIZER_GEOMETRY_CRITIC_WEIGHT_DECAY", "OPTIMIZER_GEOMETRY_CRITIC_BETA2", "SANDBOXFUSION_BASE_URL", "M2RL_SANDBOX_PREFLIGHT_MARKER", "WANDB_API_KEY", "WANDB_BASE_URL"); env.update({key: os.environ[key] for key in keys if os.environ.get(key)}); print(json.dumps({"env_vars": env}))' "${RUNTIME_PYTHONPATH}")"

set +e
ray job submit \
  --address "${RAY_ADDRESS}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_CMD[@]}"
TRAIN_EXIT_CODE=$?
set -e
python3 "${EXAMPLE_DIR}/run_provenance.py" finish --run-dir "${RUN_DIR}" --exit-code "${TRAIN_EXIT_CODE}"
exit "${TRAIN_EXIT_CODE}"
