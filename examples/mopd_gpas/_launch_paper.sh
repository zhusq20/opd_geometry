#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
source "${EXAMPLE_DIR}/_profile.sh"
CONFIG_ID="${1:-m-tk-dr}"
LOSS=student_topk
TOPK=16
REDUCTION=domain_response
TASKS="$(IFS=,; echo "${MOPD_PROFILE_TASKS[*]}")"
case "${CONFIG_ID}" in
  s-pg) LOSS=sampled_reverse_kl; TASKS="${MOPD_SINGLE_TASK:-math}" ;;
  s-tk) TASKS="${MOPD_SINGLE_TASK:-math}" ;;
  s-tk64) TOPK=64; TASKS="${MOPD_SINGLE_TASK:-math}" ;;
  m-pg) LOSS=sampled_reverse_kl ;;
  m-tk-dr) ;;
  m-tk-dt) REDUCTION=domain_token ;;
  m-tk-gt) REDUCTION=global_token ;;
  m-tk64-dr) TOPK=64 ;;
  m-tk64-dt) TOPK=64; REDUCTION=domain_token ;;
  m-tk64-gt) TOPK=64; REDUCTION=global_token ;;
  m-intersection64-dr) LOSS=topk_intersection; TOPK=64 ;;
  *) echo "Unknown paper condition ${CONFIG_ID}" >&2; exit 2 ;;
esac
TASKS="${MOPD_TASKS:-${TASKS}}"
IFS=, read -r -a ACTIVE_TASKS <<< "${TASKS}"
RESPONSES=$((${#ACTIVE_TASKS[@]} * 16))
[[ "${#ACTIVE_TASKS[@]}" != 1 ]] || RESPONSES=64
RESPONSES="${MOPD_RESPONSES_PER_UPDATE:-${RESPONSES}}"
STEPS="${MOPD_TOTAL_STEPS:-500}"
SEED="${MOPD_SEED:-42}"
RUN_ID="${MOPD_RUN_ID:-${CONFIG_ID}-s${SEED}}"
RUN_DIR="${MOPD_OUTPUT_ROOT}/${RUN_ID}"
TRAIN_GPUS="${MOPD_TRAIN_GPUS:-${MOPD_TRAIN_GPU:-0}}"
IFS=, read -r -a TRAIN_GPU_IDS <<< "${TRAIN_GPUS}"
TRAIN_GPU="${TRAIN_GPU_IDS[0]}"
TP="${#TRAIN_GPU_IDS[@]}"
ROLLOUT_GPU="${MOPD_INFERENCE_GPU:-1}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS},${ROLLOUT_GPU}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH:-/root/Megatron-LM}${PYTHONPATH:+:${PYTHONPATH}}"
source "${SLIME_ROOT}/scripts/models/${MOPD_MODEL_CONFIG}"
PROMPT_ARGS=(--apply-chat-template)
if [[ "${MOPD_PROFILE}" == qwen3 ]]; then
  PROMPT_ARGS+=(--apply-chat-template-kwargs '{"enable_thinking":false}'
    --chat-template-suffix-to-remove $'<think>\n\n</think>\n\n')
else
  PROMPT_ARGS+=(--apply-chat-template-kwargs '{"enable_thinking":true}')
fi
LOAD="${MOPD_LOAD_CHECKPOINT:-${MOPD_HF_CHECKPOINT}}"
LOAD_ARGS=(--no-load-optim --no-load-rng --override-opt-param-scheduler --start-rollout-id 0)
if [[ "${MOPD_RESUME:-0}" == 1 ]]; then
  LOAD="${RUN_DIR}/checkpoints"
  LOAD_ARGS=()
fi
OCCUPIED_GPUS="$(python3 - "${TRAIN_GPUS}" "${ROLLOUT_GPU}" "${TASKS}" <<'PY'
import os, sys
training = sys.argv[1].split(',')
if len(set(training)) != len(training) or sys.argv[2] in training:
    raise SystemExit('Training GPUs and student rollout GPU must be distinct')
ids = {*training, sys.argv[2]}
if any(not value.isdecimal() for value in ids):
    raise SystemExit('Specify physical GPU IDs as comma-separated nonnegative integers')
for task in sys.argv[3].split(','):
    if os.environ.get('MOPD_TEACHER_' + task.upper() + '_GPU', sys.argv[2]) in training:
        raise SystemExit('Resident teachers must not share a training GPU')
ids.update(os.environ.get('MOPD_TEACHER_' + task.upper() + '_GPU', sys.argv[2]) for task in sys.argv[3].split(','))
print(len(ids))
PY
)"
CMD=(python3 "${SLIME_ROOT}/train.py" "${MODEL_ARGS[@]}"
  --hf-checkpoint "${MOPD_HF_CHECKPOINT}" --load "${LOAD}" "${LOAD_ARGS[@]}"
  --save "${RUN_DIR}/checkpoints" --save-interval 1000000
  --save-hf "${RUN_DIR}/weights/iter_{rollout_id:07d}"
  --prompt-data "${MOPD_GENERATED_DIR}/train.yaml"
  --data-source-path slime_plugins.mopd.data_source.MOPDRolloutDataSource
  --rollout-function-path slime_plugins.mopd.rollout.generate_rollout
  --input-key prompt --label-key label --metadata-key metadata --tool-key tools
  "${PROMPT_ARGS[@]}" --rollout-global-dataset --rollout-shuffle --rollout-seed "${SEED}"
  --num-rollout "${STEPS}" --rollout-batch-size "${RESPONSES}" --global-batch-size 4 --micro-batch-size 1
  --n-samples-per-prompt 1 --rollout-max-prompt-len 2048 --rollout-max-response-len "${MOPD_MAX_RESPONSE_LEN:-4096}"
  --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1 --balance-data
  --mopd-enabled --mopd-profile "${MOPD_PROFILE}" --mopd-tasks "${TASKS}"
  --mopd-reduction "${REDUCTION}" --mopd-loss "${LOSS}" --mopd-allocation uniform
  --mopd-topk "${TOPK}"
  --mopd-responses-per-update "${RESPONSES}" --mopd-microbatches-per-step "$((RESPONSES / 4))"
  --mopd-prompts-per-microbatch 4 --mopd-total-steps "${STEPS}"
  --mopd-response-budget "$((STEPS * RESPONSES))" --mopd-seed "${SEED}"
  --mopd-checkpoint-steps "${MOPD_CHECKPOINT_STEPS:-1,50,100,250,500}"
  --mopd-eval-responses "${MOPD_EVAL_RESPONSES:-}"
  --mopd-output-dir "${RUN_DIR}" --mopd-occupied-gpus "${OCCUPIED_GPUS}"
  --disable-compute-advantages-and-returns
  --loss-type custom_loss --custom-loss-function-path slime_plugins.mopd.loss.paper_loss
  --opd-teacher-router-config "${MOPD_TEACHER_ROUTER_CONFIG}" --opd-task-reward-weight 0.0
  --custom-rm-path slime_plugins.m2rl.opd.teacher_reward
  --custom-reward-post-process-path slime_plugins.mopd.loss.post_process_rewards
  --m2rl-reward-config "${MOPD_REWARD_CONFIG:-${MOPD_GENERATED_DIR}/rewards.yaml}"
  --entropy-coef 0.0 --kl-coef 0.0 --kl-loss-coef 0.0
  --optimizer adam --lr "${MOPD_LR:-2.5e-7}" --weight-decay 0.0
  --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8
  --lr-decay-style constant --lr-warmup-iters 0 --lr-decay-iters "${STEPS}" --clip-grad 1.0
  --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size 1 --context-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --metrics-output-dir "${RUN_DIR}/metrics" --run-manifest-path "${RUN_DIR}/provenance/run_manifest.json"
  --completion-marker-path "${RUN_DIR}/run_complete.json"
  --experiment-task "${TASKS}" --experiment-teacher "${MOPD_PROFILE}_rl"
  --experiment-condition "${CONFIG_ID}" --experiment-name "${RUN_ID}"
  --experiment-optimizer adamw --experiment-data-index "${MOPD_GENERATED_DIR}/protocol.json"
  --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --num-gpus-per-node "$((TP + 1))"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.45}"
  --sglang-context-length "${MOPD_CONTEXT_LENGTH}"
  --actor-num-nodes 1 --actor-num-gpus-per-node "${TP}"
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --seed "${SEED}")
if [[ "${MOPD_OBSERVE_TASK_REWARDS:-1}" == 0 ]]; then
  CMD+=(--mopd-skip-task-rewards)
fi
if [[ "${MOPD_PAPER_MEASUREMENTS:-1}" == 0 ]]; then
  CMD+=(--mopd-skip-paper-measurements)
fi
if [[ -n "${MOPD_SGLANG_ATTENTION_BACKEND:-}" ]]; then
  CMD+=(--sglang-attention-backend "${MOPD_SGLANG_ATTENTION_BACKEND}")
fi
if [[ "${MOPD_DETERMINISTIC_INFERENCE:-1}" == 1 ]]; then
  CMD+=(--sglang-enable-deterministic-inference)
else
  # Keep per-request sampling seeds without requiring batch-invariant GEMMs,
  # whose FP32 BMM kernel exceeds the shared-memory limit of RTX A6000.
  CMD+=(--sglang-sampling-backend pytorch)
fi
if [[ "${REDUCTION}" != domain_response ]]; then
  CMD+=(--calculate-per-token-loss)
fi
if [[ "${MOPD_EVAL_DURING_TRAINING:-1}" == 1 ]]; then
  CMD+=(--eval-config "${MOPD_CAPABILITY_EVAL_CONFIG}" --eval-interval "${MOPD_EVAL_INTERVAL:-250}"
    --eval-function-path slime_plugins.mopd.eval.generate_capability_eval
    --eval-max-response-len 32768 --eval-max-concurrency "${EVAL_MAX_CONCURRENCY:-8}"
    --eval-artifact-dir "${RUN_DIR}/capability_eval" --log-passrate)
fi
if [[ "${USE_WANDB}" == 1 ]]; then
  CMD+=(--use-wandb --wandb-mode "${WANDB_MODE:-online}" --wandb-dir "${RUN_DIR}/wandb"
    --wandb-project "${WANDB_PROJECT}" --wandb-group "${MOPD_PROFILE}-paper"
    --wandb-run-name "${RUN_ID}" --wandb-run-id-file "${RUN_DIR}/wandb_run_id.txt" --disable-wandb-random-suffix)
  [[ -z "${WANDB_ENTITY:-}" ]] || CMD+=(--wandb-team "${WANDB_ENTITY}")
fi
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  python3 -c '
import slime.backends.megatron_utils.arguments as ma
import slime.utils.arguments as sa
ma.validate_args=lambda args: args
sa.sglang_validate_args=lambda args: args
sa.parse_args()
print("paper profile argument validation: OK")
' "${CMD[@]:2}"
  exit 0
fi
mkdir -p "${RUN_DIR}"
[[ "${MOPD_RESUME:-0}" == 1 || ! -e "${RUN_DIR}/run_complete.json" ]] || {
  echo "Run already completed: ${RUN_DIR}; choose MOPD_RUN_ID for a new run." >&2; exit 2;
}
ACTION=start
[[ "${MOPD_RESUME:-0}" != 1 ]] || ACTION=resume
python3 "${EXAMPLE_DIR}/provenance.py" "${ACTION}" --repo "${SLIME_ROOT}" --run-dir "${RUN_DIR}" \
  --input "${MOPD_GENERATED_DIR}/protocol.json" --input "${MOPD_GENERATED_DIR}/train.yaml" \
  --input "${MOPD_TEACHER_ROUTER_CONFIG}" --input "${MOPD_REWARD_CONFIG:-${MOPD_GENERATED_DIR}/rewards.yaml}" \
  --checkpoint "${LOAD}" --source "${EXAMPLE_DIR}" --source "${SLIME_ROOT}/slime_plugins/mopd" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils" --source "${SLIME_ROOT}/slime/utils/arguments.py" \
  --source "${SLIME_ROOT}/slime/rollout/rm_hub/gpqa.py" \
  --source "${SLIME_ROOT}/slime/rollout/sglang_rollout.py" \
  --source "${SLIME_ROOT}/slime_plugins/m2rl" --source "${SLIME_ROOT}/train.py" \
  -- "${CMD[@]}" > "${RUN_DIR}/provenance_start.json"
set +e
"${CMD[@]}"
STATUS=$?
set -e
python3 "${EXAMPLE_DIR}/provenance.py" finish --run-dir "${RUN_DIR}" --exit-code "${STATUS}" > "${RUN_DIR}/provenance_finish.json"
exit "${STATUS}"
