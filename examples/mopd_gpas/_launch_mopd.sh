#!/usr/bin/env bash
set -euo pipefail

[[ $# == 2 ]] || { echo "internal usage: $0 RUN_ID ALLOCATION" >&2; exit 2; }
RUN_ID="$1"
ALLOCATION="$2"
case "${ALLOCATION}" in
  uniform|gpas|cost_gpas|raw_noise|loss_gap|std_mopd|d3_mopd|open_mopd) ;;
  *) echo "Unknown allocation: ${ALLOCATION}" >&2; exit 2 ;;
esac

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export MOPD_HF_CHECKPOINT="${MOPD_HF_CHECKPOINT:-/workspace/dev/checkpoints/Qwen3-1.7B}"
export MOPD_BASE_MEGATRON="${MOPD_BASE_MEGATRON:-/workspace/dev/checkpoints/Qwen3-1.7B_torch_dist}"
export MOPD_TEACHER_HF_ROOT="${MOPD_TEACHER_HF_ROOT:-${SLIME_ROOT}/local/mopd_assets/models/teachers_hf}"
export MOPD_QWEN3_4B="${MOPD_QWEN3_4B:-${SLIME_ROOT}/local/mopd_assets/models/qwen3-4b}"
export MOPD_TRAIN_GPU="${MOPD_TRAIN_GPU:-0}"
export MOPD_INFERENCE_GPU="${MOPD_INFERENCE_GPU:-1}"
export MOPD_TEACHER_MATH_PORT="${MOPD_TEACHER_MATH_PORT:-31001}"
export MOPD_TEACHER_CODE_PORT="${MOPD_TEACHER_CODE_PORT:-31002}"
export MOPD_TEACHER_IF_PORT="${MOPD_TEACHER_IF_PORT:-31003}"
export MOPD_TEACHER_SCIENCE_PORT="${MOPD_TEACHER_SCIENCE_PORT:-31004}"
[[ "${MOPD_TRAIN_GPU}" != "${MOPD_INFERENCE_GPU}" ]] || {
  echo "Training and inference GPU IDs must differ." >&2
  exit 2
}

GENERATED="${MOPD_GENERATED_DIR:-${SLIME_ROOT}/local/mopd_generated}"
TRAIN_MANIFEST="${GENERATED}/train.yaml"
EVAL_CONFIG="${GENERATED}/teacher_loss_eval.yaml"
PROTOCOL="${GENERATED}/protocol.json"
TEACHER_ROUTER="${EXAMPLE_DIR}/configs/teacher_router.yaml"
OUTPUT_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_v4}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}-seed42"
MODEL_CONFIG="${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh"
DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-1}"
MOPD_RESUME="${MOPD_RESUME:-0}"
MOPD_SMOKE_TEST="${MOPD_SMOKE_TEST:-0}"
MOPD_QUICK_SMOKE_TEST="${MOPD_QUICK_SMOKE_TEST:-0}"

if [[ "${MOPD_QUICK_SMOKE_TEST}" == 1 ]]; then
  [[ "${ALLOCATION}" == gpas ]] || { echo "The quick smoke test uses GPAS." >&2; exit 2; }
  TOTAL_STEPS=1
  RESPONSE_BUDGET=64
  CHECKPOINT_STEPS=1
  EVAL_RESPONSES=""
  ROLLOUT_MAX_RESPONSE_LEN=256
  MOPD_MODE_ARGS=(--mopd-quick-smoke-test)
  EVAL_ARGS=()
elif [[ "${MOPD_SMOKE_TEST}" == 1 ]]; then
  [[ "${ALLOCATION}" == gpas ]] || { echo "The smoke protocol uses GPAS." >&2; exit 2; }
  TOTAL_STEPS=20
  RESPONSE_BUDGET=1280
  CHECKPOINT_STEPS=10,20
  EVAL_RESPONSES=640,1280
  ROLLOUT_MAX_RESPONSE_LEN=4096
  MOPD_MODE_ARGS=(--mopd-smoke-test)
  EVAL_ARGS=(
    --eval-config "${EVAL_CONFIG}"
    --eval-function-path slime_plugins.mopd.eval.generate_teacher_loss_eval
    --eval-interval 1 --eval-max-concurrency "${EVAL_MAX_CONCURRENCY:-16}"
    --eval-artifact-dir "${RUN_DIR}/teacher_loss_eval"
  )
else
  TOTAL_STEPS=500
  RESPONSE_BUDGET=32000
  CHECKPOINT_STEPS=50,100,150,200,250,300,350,400,450,500
  EVAL_RESPONSES=3200,6400,9600,12800,16000,19200,22400,25600,28800,32000
  ROLLOUT_MAX_RESPONSE_LEN=4096
  MOPD_MODE_ARGS=()
  EVAL_ARGS=(
    --eval-config "${EVAL_CONFIG}"
    --eval-function-path slime_plugins.mopd.eval.generate_teacher_loss_eval
    --eval-interval 1 --eval-max-concurrency "${EVAL_MAX_CONCURRENCY:-16}"
    --eval-artifact-dir "${RUN_DIR}/teacher_loss_eval"
  )
fi

for path in "${TRAIN_MANIFEST}" "${EVAL_CONFIG}" "${PROTOCOL}" "${TEACHER_ROUTER}"; do
  [[ -f "${path}" ]] || { echo "Missing ${path}; run the prepare stages first." >&2; exit 2; }
done
[[ -f "${MOPD_HF_CHECKPOINT}/config.json" ]] || { echo "Missing student HF checkpoint." >&2; exit 2; }

START_ARGS=()
EXTRA_LOAD_ARGS=()
CHECKPOINT_FOR_PROVENANCE="${MOPD_BASE_MEGATRON}"
if [[ "${MOPD_RESUME}" == 1 ]]; then
  [[ "${DRY_RUN}" == 0 ]] || { echo "A dry run cannot resume mutable state." >&2; exit 2; }
  RESUME_JSON="$(python3 "${EXAMPLE_DIR}/prepare_resume.py" --run-dir "${RUN_DIR}")"
  read -r RESUME_CHECKPOINT_ID RESUME_START_ID RESUME_EVAL_ON_START < <(
    python3 - "${RESUME_JSON}" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
print(value["rollout_id"], value["next_rollout_id"], int(value["eval_on_start"]))
PY
  )
  LOAD_CHECKPOINT="${RUN_DIR}/checkpoints"
  START_ARGS=(--start-rollout-id "${RESUME_START_ID}")
  [[ "${RESUME_EVAL_ON_START}" == 0 ]] || START_ARGS+=(--mopd-eval-on-start)
  EXTRA_LOAD_ARGS=(--ckpt-step "${RESUME_CHECKPOINT_ID}")
  CHECKPOINT_FOR_PROVENANCE="${LOAD_CHECKPOINT}/iter_$(printf '%07d' "${RESUME_CHECKPOINT_ID}")"
else
  LOAD_CHECKPOINT="${MOPD_BASE_MEGATRON}"
  START_ARGS=(--start-rollout-id 0 --no-load-optim --no-load-rng --override-opt-param-scheduler)
  if [[ "${DRY_RUN}" == 0 && -d "${RUN_DIR}" && -n "$(find "${RUN_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to mix a new run into non-empty ${RUN_DIR}." >&2
    exit 2
  fi
fi

if [[ "${DRY_RUN}" == 0 ]]; then
  python3 "${EXAMPLE_DIR}/verify_hardware.py" \
    --training-gpu "${MOPD_TRAIN_GPU}" --inference-gpu "${MOPD_INFERENCE_GPU}"
  bash "${EXAMPLE_DIR}/serve_teachers.sh" status
  if [[ "${USE_WANDB}" == 1 && "${WANDB_MODE:-online}" == online ]]; then
    python3 - <<'PY'
import wandb
if not wandb.api.api_key:
    raise SystemExit("W&B online mode requested, but no API key is configured")
PY
  fi
fi

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/mopd_${MOPD_TRAIN_GPU}_${MOPD_INFERENCE_GPU}_$$}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
export RAY_AUX_PORT_STRIDE="${RAY_AUX_PORT_STRIDE:-128}"
# Keep fixed Ray control ports below Linux's default ephemeral range
# (32768-60999) so startup connections cannot steal them.
export RAY_AUX_PORT_BASE="${RAY_AUX_PORT_BASE:-$((12000 + (RAY_DASHBOARD_PORT - 8265) * RAY_AUX_PORT_STRIDE))}"
export RAY_WORKER_PORT_MIN="${RAY_WORKER_PORT_MIN:-$((RAY_AUX_PORT_BASE + 16))}"
export RAY_WORKER_PORT_MAX="${RAY_WORKER_PORT_MAX:-$((RAY_AUX_PORT_BASE + 63))}"
# Two placement bundles reserve two CPUs; RolloutManager and the synchronization
# Lock actor also need schedulable CPUs outside those bundles.
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"
export SLIME_ROLLOUT_PORT_BASE="${SLIME_ROLLOUT_PORT_BASE:-$((20000 + (RAY_DASHBOARD_PORT - 8265) * 128))}"

export MODEL_ARGS_ROTARY_BASE=1000000
source "${MODEL_CONFIG}"
export CUDA_VISIBLE_DEVICES="${MOPD_TRAIN_GPU},${MOPD_INFERENCE_GPU}"

TRAIN_CMD=(
  python3 "${SLIME_ROOT}/train.py"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${MOPD_HF_CHECKPOINT}"
  --load "${LOAD_CHECKPOINT}"
  --save "${RUN_DIR}/checkpoints"
  --save-interval 1000000
  --save-hf "${RUN_DIR}/weights/iter_{rollout_id:07d}"
  "${START_ARGS[@]}"
  "${EXTRA_LOAD_ARGS[@]}"

  --prompt-data "${TRAIN_MANIFEST}"
  --data-source-path slime_plugins.mopd.data_source.MOPDRolloutDataSource
  --rollout-function-path slime_plugins.mopd.rollout.generate_rollout
  --input-key prompt --label-key label --metadata-key metadata --tool-key tools
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}'
  --rollout-global-dataset --rollout-shuffle --rollout-seed 42
  --num-rollout "${TOTAL_STEPS}" --rollout-batch-size 64 --global-batch-size 4 --micro-batch-size 1
  --n-samples-per-prompt 1
  --rollout-max-prompt-len 2048 --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1
  --balance-data

  --mopd-enabled --mopd-allocation "${ALLOCATION}" --mopd-seed 42
  "${MOPD_MODE_ARGS[@]}"
  --mopd-ema-decay 0.9 --mopd-total-steps "${TOTAL_STEPS}"
  --mopd-microbatches-per-step 16 --mopd-prompts-per-microbatch 4
  --mopd-min-microbatches 2 --mopd-max-microbatches 8
  --mopd-response-budget "${RESPONSE_BUDGET}"
  --mopd-checkpoint-steps "${CHECKPOINT_STEPS}"
  --mopd-eval-responses "${EVAL_RESPONSES}"
  --mopd-failure-penalty 10.0 --mopd-score-chunk-size 1048576
  --mopd-output-dir "${RUN_DIR}/allocation"

  --advantage-estimator grpo --use-rollout-logprobs
  --use-opd --opd-type sglang --opd-kl-coef 1.0
  --opd-teacher-router-config "${TEACHER_ROUTER}" --opd-task-reward-weight 0.0
  --custom-rm-path slime_plugins.m2rl.opd.teacher_reward
  --custom-reward-post-process-path slime_plugins.m2rl.opd.post_process_rewards
  --entropy-coef 0.0 --kl-coef 0.0 --kl-loss-coef 0.0
  --eps-clip 0.2 --eps-clip-high 0.2

  --optimizer adam --lr 2.5e-7 --weight-decay 0.0
  --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8
  --lr-decay-style constant --lr-warmup-iters 0 --lr-decay-iters "${TOTAL_STEPS}" --clip-grad 1.0

  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  "${EVAL_ARGS[@]}"

  --metrics-output-dir "${RUN_DIR}/metrics"
  --run-manifest-path "${RUN_DIR}/provenance/run_manifest.json"
  --completion-marker-path "${RUN_DIR}/run_complete.json"
  --experiment-task multi --experiment-teacher math_if_rl_code_science_qwen3_4b_resident
  --experiment-condition "${RUN_ID}" --experiment-name "${RUN_ID}-seed42"
  --experiment-optimizer adamw --experiment-data-index "${PROTOCOL}"

  --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --num-gpus-per-node 2
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.32}"
  --sglang-enable-deterministic-inference
  --actor-num-nodes 1 --actor-num-gpus-per-node 1
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash --seed 42
)
if [[ "${ALLOCATION}" == std_mopd || "${ALLOCATION}" == d3_mopd || "${ALLOCATION}" == open_mopd ]]; then
  TRAIN_CMD+=(--calculate-per-token-loss)
fi
if [[ "${USE_WANDB}" == 1 ]]; then
  TRAIN_CMD+=(
    --use-wandb --wandb-mode "${WANDB_MODE:-online}" --wandb-dir "${RUN_DIR}/wandb"
    --wandb-team "${WANDB_ENTITY:-zsqzz}" --wandb-project "${WANDB_PROJECT:-iclr2027-mopd-gpas-v4}"
    --wandb-group "Qwen3-1.7B-4T-microbatch-MOPD-${TOTAL_STEPS}-step"
    --wandb-run-name "${RUN_ID}-seed42"
    --wandb-run-id-file "${RUN_DIR}/wandb_run_id.txt" --disable-wandb-random-suffix
  )
fi

printf 'Launch command:'; printf ' %q' "${TRAIN_CMD[@]}"; printf '\n'
if [[ "${DRY_RUN}" == 1 ]]; then
  python3 -c '
import slime.utils.arguments as sa
sa.sglang_validate_args=lambda args: args
sa.parse_args()
print("static argument validation: OK")
' "${TRAIN_CMD[@]:2}"
  exit 0
fi

mkdir -p "${RUN_DIR}"
PROVENANCE_ACTION=start
if [[ "${MOPD_RESUME}" == 1 ]]; then
  APPLIED_RESUME_JSON="$(python3 "${EXAMPLE_DIR}/prepare_resume.py" --run-dir "${RUN_DIR}" --apply)"
  python3 - "${RESUME_JSON}" "${APPLIED_RESUME_JSON}" <<'PY'
import json, sys
before, after = map(json.loads, sys.argv[1:])
for key in ("rollout_id", "next_rollout_id", "operation_index", "attempted_responses"):
    if before[key] != after[key]:
        raise SystemExit(f"resume frontier changed during launch: {key}")
PY
  PROVENANCE_ACTION=resume
fi
python3 "${EXAMPLE_DIR}/provenance.py" "${PROVENANCE_ACTION}" --repo "${SLIME_ROOT}" --run-dir "${RUN_DIR}" \
  --input "${PROTOCOL}" --input "${TRAIN_MANIFEST}" --input "${EVAL_CONFIG}" --input "${TEACHER_ROUTER}" \
  --checkpoint "${MOPD_HF_CHECKPOINT}" --checkpoint "${CHECKPOINT_FOR_PROVENANCE}" \
  --source "${EXAMPLE_DIR}" --source "${SLIME_ROOT}/slime_plugins/mopd" \
  --source "${SLIME_ROOT}/slime_plugins/m2rl/opd.py" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/data.py" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/model.py" \
  --source "${SLIME_ROOT}/slime/ray/rollout.py" \
  --source "${SLIME_ROOT}/slime/utils/arguments.py" --source "${SLIME_ROOT}/train.py" -- "${TRAIN_CMD[@]}"

RAY_PID=""
cleanup() {
  if [[ -n "${RAY_PID}" && "${KEEP_RAY:-0}" != 1 ]]; then
    kill -TERM "${RAY_PID}" 2>/dev/null || true
    wait "${RAY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
RAY_ADDRESS="${RAY_ADDRESS:-}"
if [[ -z "${RAY_ADDRESS}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  mkdir -p "${RAY_TEMP_DIR}"
  ray start --head --node-ip-address "${MASTER_ADDR}" --port "${RAY_GCS_PORT}" \
    --num-cpus "${RAY_NUM_CPUS}" --num-gpus 2 --disable-usage-stats \
    --dashboard-host 0.0.0.0 --dashboard-port "${RAY_DASHBOARD_PORT}" \
    --dashboard-agent-listen-port "$((RAY_AUX_PORT_BASE + 0))" \
    --dashboard-agent-grpc-port "$((RAY_AUX_PORT_BASE + 1))" \
    --runtime-env-agent-port "$((RAY_AUX_PORT_BASE + 2))" \
    --metrics-export-port "$((RAY_AUX_PORT_BASE + 3))" \
    --ray-client-server-port "$((RAY_AUX_PORT_BASE + 4))" \
    --object-manager-port "$((RAY_AUX_PORT_BASE + 5))" \
    --node-manager-port "$((RAY_AUX_PORT_BASE + 6))" \
    --min-worker-port "${RAY_WORKER_PORT_MIN}" --max-worker-port "${RAY_WORKER_PORT_MAX}" \
    --temp-dir "${RAY_TEMP_DIR}" --block &
  RAY_PID=$!
  RAY_ADDRESS="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"
  AGENT_LOG="${RAY_TEMP_DIR}/session_latest/logs/dashboard_agent.log"
  RAY_READY=0
  for _ in $(seq 1 120); do
    kill -0 "${RAY_PID}" 2>/dev/null || { echo "Ray head exited." >&2; exit 1; }
    if [[ -f "${AGENT_LOG}" ]] && grep -q "Dashboard agent http address:" "${AGENT_LOG}" \
      && ray job list --address "${RAY_ADDRESS}" >/dev/null 2>&1; then
      RAY_READY=1
      break
    fi
    sleep 1
  done
  [[ "${RAY_READY}" == 1 ]] || { echo "Ray Jobs API did not become ready." >&2; exit 1; }
fi

RUNTIME_ENV_JSON="$(python3 - "${PYTHONPATH}" <<'PY'
import json, os, sys
keys = (
    "WANDB_API_KEY", "WANDB_BASE_URL", "NLTK_DATA", "CUDA_VISIBLE_DEVICES",
    "MOPD_HF_CHECKPOINT", "MOPD_TEACHER_HF_ROOT", "MOPD_QWEN3_4B",
    "MOPD_TRAIN_GPU", "MOPD_INFERENCE_GPU", "MOPD_TEACHER_MATH_PORT",
    "MOPD_TEACHER_CODE_PORT", "MOPD_TEACHER_IF_PORT", "MOPD_TEACHER_SCIENCE_PORT",
    "SLIME_ROLLOUT_PORT_BASE",
)
env = {"PYTHONPATH": sys.argv[1], "CUDA_DEVICE_MAX_CONNECTIONS": "1", "PYTHONUNBUFFERED": "1"}
env.update({key: os.environ[key] for key in keys if os.environ.get(key)})
print(json.dumps({"env_vars": env}))
PY
)"
set +e
ray job submit --address "${RAY_ADDRESS}" --runtime-env-json "${RUNTIME_ENV_JSON}" -- "${TRAIN_CMD[@]}"
EXIT_CODE=$?
set -e
python3 "${EXAMPLE_DIR}/provenance.py" finish --run-dir "${RUN_DIR}" --exit-code "${EXIT_CODE}"
exit "${EXIT_CODE}"
