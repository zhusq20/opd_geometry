#!/usr/bin/env bash
# Collect scalar-only held-out gradient statistics at one Uniform checkpoint.
set -euo pipefail

STEP="${1:-}"
case "${STEP}" in
  50|250|500) ;;
  *) echo "Usage: $0 {50|250|500}" >&2; exit 2 ;;
esac

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export MOPD_HF_CHECKPOINT="${MOPD_HF_CHECKPOINT:-/workspace/dev/checkpoints/Qwen3-1.7B}"
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
MANIFEST="${GENERATED}/heldout_variance.yaml"
PROTOCOL="${GENERATED}/protocol.json"
TEACHER_ROUTER="${EXAMPLE_DIR}/configs/teacher_router.yaml"
OUTPUT_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_v4}"
UNIFORM_ROOT="${OUTPUT_ROOT}/uniform-seed42/checkpoints"
OUTPUT_DIR="${OUTPUT_ROOT}/heldout_variance/step_$(printf '%03d' "${STEP}")"
INDEX="${UNIFORM_ROOT}/mopd_checkpoint_index.json"
DRY_RUN="${DRY_RUN:-0}"

for path in "${MANIFEST}" "${PROTOCOL}" "${TEACHER_ROUTER}"; do
  [[ -f "${path}" ]] || { echo "Missing ${path}; run the prepare stages first." >&2; exit 2; }
done
if [[ "${DRY_RUN}" == 1 ]]; then
  CHECKPOINT_ROOT="${MOPD_BASE_MEGATRON:-/workspace/dev/checkpoints/Qwen3-1.7B_torch_dist}"
  CHECKPOINT_ID=0
  CONTROLLER_STATE="${CHECKPOINT_ROOT}/rollout/mopd_dataset_state_dict_0.pt"
else
  [[ -f "${INDEX}" ]] || { echo "Missing Uniform checkpoint index ${INDEX}." >&2; exit 2; }
  CHECKPOINT_ID="$(python3 - "${INDEX}" "${STEP}" <<'PY'
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(sys.argv[2])
matches = [row for row in rows if int(row["optimizer_step"]) == step]
if len(matches) != 1 or not bool(matches[0].get("optimizer_state_retained")):
    raise SystemExit(f"step {step} does not have one retained optimizer checkpoint: {matches}")
print(int(matches[0]["rollout_id"]))
PY
  )"
  CHECKPOINT_ROOT="${UNIFORM_ROOT}"
  CHECKPOINT_DIR="${CHECKPOINT_ROOT}/iter_$(printf '%07d' "${CHECKPOINT_ID}")"
  CONTROLLER_STATE="${CHECKPOINT_ROOT}/rollout/mopd_dataset_state_dict_${CHECKPOINT_ID}.pt"
  [[ -f "${CHECKPOINT_DIR}/common.pt" && -f "${CHECKPOINT_DIR}/.metadata" ]] || {
    echo "Incomplete Uniform optimizer checkpoint ${CHECKPOINT_DIR}." >&2
    exit 2
  }
  [[ -f "${CONTROLLER_STATE}" ]] || { echo "Missing Uniform controller state ${CONTROLLER_STATE}." >&2; exit 2; }
  python3 "${EXAMPLE_DIR}/verify_hardware.py" \
    --training-gpu "${MOPD_TRAIN_GPU}" --inference-gpu "${MOPD_INFERENCE_GPU}"
  bash "${EXAMPLE_DIR}/serve_teachers.sh" status
  [[ ! -e "${OUTPUT_DIR}" ]] || { echo "Output already exists: ${OUTPUT_DIR}" >&2; exit 2; }
fi

export MODEL_ARGS_ROTARY_BASE=1000000
source "${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh"
export CUDA_VISIBLE_DEVICES="${MOPD_TRAIN_GPU},${MOPD_INFERENCE_GPU}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/mopd_variance_${STEP}_$$}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
export RAY_AUX_PORT_STRIDE="${RAY_AUX_PORT_STRIDE:-128}"
export RAY_AUX_PORT_BASE="${RAY_AUX_PORT_BASE:-$((45000 + (RAY_DASHBOARD_PORT - 8265) * RAY_AUX_PORT_STRIDE))}"
export RAY_WORKER_PORT_MIN="${RAY_WORKER_PORT_MIN:-$((RAY_AUX_PORT_BASE + 16))}"
export RAY_WORKER_PORT_MAX="${RAY_WORKER_PORT_MAX:-$((RAY_AUX_PORT_BASE + 63))}"
export SLIME_ROLLOUT_PORT_BASE="${SLIME_ROLLOUT_PORT_BASE:-$((20000 + (RAY_DASHBOARD_PORT - 8265) * 128))}"

TRAIN_CMD=(
  python3 "${SLIME_ROOT}/train.py"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${MOPD_HF_CHECKPOINT}"
  --load "${CHECKPOINT_ROOT}" --ckpt-step "${CHECKPOINT_ID}"
  --start-rollout-id 0 --no-load-rng --override-opt-param-scheduler
  --prompt-data "${MANIFEST}"
  --data-source-path slime_plugins.mopd.data_source.MOPDVarianceDataSource
  --rollout-function-path slime_plugins.mopd.rollout.generate_rollout
  --input-key prompt --label-key label --metadata-key metadata --tool-key tools
  --apply-chat-template-kwargs '{"enable_thinking":false}'
  --rollout-global-dataset --rollout-seed 42
  --num-rollout 1 --rollout-batch-size 512 --global-batch-size 4 --micro-batch-size 1
  --n-samples-per-prompt 1
  --rollout-max-prompt-len 2048 --rollout-max-response-len 4096
  --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1
  --balance-data
  --mopd-enabled --mopd-heldout-variance --mopd-variance-checkpoint-step "${STEP}"
  --mopd-variance-controller-state "${CONTROLLER_STATE}"
  --mopd-allocation uniform --mopd-seed 42 --mopd-ema-decay 0.9
  --mopd-total-steps 1 --mopd-microbatches-per-step 128 --mopd-prompts-per-microbatch 4
  --mopd-min-microbatches 32 --mopd-max-microbatches 32 --mopd-response-budget 512
  --mopd-checkpoint-steps "" --mopd-eval-responses ""
  --mopd-failure-penalty 10.0 --mopd-score-chunk-size 1048576
  --mopd-output-dir "${OUTPUT_DIR}"
  --advantage-estimator grpo --use-rollout-logprobs
  --use-opd --opd-type sglang --opd-kl-coef 1.0
  --opd-teacher-router-config "${TEACHER_ROUTER}" --opd-task-reward-weight 0.0
  --custom-rm-path slime_plugins.m2rl.opd.teacher_reward
  --custom-reward-post-process-path slime_plugins.m2rl.opd.post_process_rewards
  --entropy-coef 0.0 --kl-coef 0.0 --kl-loss-coef 0.0
  --eps-clip 0.2 --eps-clip-high 0.2
  --optimizer adam --lr 2.5e-7 --weight-decay 0.0
  --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8
  --lr-decay-style constant --lr-warmup-iters 0 --lr-decay-iters 1 --clip-grad 1.0
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --metrics-output-dir "${OUTPUT_DIR}/metrics"
  --run-manifest-path "${OUTPUT_DIR}/provenance/run_manifest.json"
  --completion-marker-path "${OUTPUT_DIR}/run_complete.json"
  --experiment-task multi --experiment-teacher math_if_rl_code_science_qwen3_4b_resident
  --experiment-condition "heldout-variance-step-${STEP}"
  --experiment-name "heldout-variance-step-${STEP}-seed42"
  --experiment-optimizer adamw-frozen --experiment-data-index "${PROTOCOL}"
  --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --num-gpus-per-node 2
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.32}"
  --sglang-enable-deterministic-inference
  --actor-num-nodes 1 --actor-num-gpus-per-node 1
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash --seed 42
)

printf 'Held-out variance command:'; printf ' %q' "${TRAIN_CMD[@]}"; printf '\n'
if [[ "${DRY_RUN}" == 1 ]]; then
  python3 -c '
import slime.utils.arguments as sa
sa.sglang_validate_args=lambda args: args
sa.parse_args()
print("static argument validation: OK")
' "${TRAIN_CMD[@]:2}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
python3 "${EXAMPLE_DIR}/provenance.py" start --repo "${SLIME_ROOT}" --run-dir "${OUTPUT_DIR}" \
  --input "${PROTOCOL}" --input "${MANIFEST}" --input "${TEACHER_ROUTER}" \
  --input "${CONTROLLER_STATE}" \
  --checkpoint "${MOPD_HF_CHECKPOINT}" \
  --checkpoint "${CHECKPOINT_ROOT}/iter_$(printf '%07d' "${CHECKPOINT_ID}")" \
  --source "${EXAMPLE_DIR}" --source "${SLIME_ROOT}/slime_plugins/mopd" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/model.py" \
  --source "${SLIME_ROOT}/slime/ray/rollout.py" --source "${SLIME_ROOT}/train.py" \
  -- "${TRAIN_CMD[@]}"

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
    --num-cpus "${RAY_NUM_CPUS:-3}" --num-gpus 2 --disable-usage-stats \
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
  for _ in $(seq 1 120); do
    ray job list --address "${RAY_ADDRESS}" >/dev/null 2>&1 && break
    kill -0 "${RAY_PID}" 2>/dev/null || { echo "Ray head exited." >&2; exit 1; }
    sleep 1
  done
  ray job list --address "${RAY_ADDRESS}" >/dev/null 2>&1 || { echo "Ray Jobs API not ready." >&2; exit 1; }
fi
RUNTIME_ENV_JSON="$(python3 - "${PYTHONPATH}" <<'PY'
import json, os, sys
keys = (
    "CUDA_VISIBLE_DEVICES", "MOPD_HF_CHECKPOINT", "MOPD_TEACHER_HF_ROOT", "MOPD_QWEN3_4B",
    "MOPD_TRAIN_GPU", "MOPD_INFERENCE_GPU", "MOPD_TEACHER_MATH_PORT", "MOPD_TEACHER_CODE_PORT",
    "MOPD_TEACHER_IF_PORT", "MOPD_TEACHER_SCIENCE_PORT", "SLIME_ROLLOUT_PORT_BASE",
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
python3 "${EXAMPLE_DIR}/provenance.py" finish --run-dir "${OUTPUT_DIR}" --exit-code "${EXIT_CODE}"
exit "${EXIT_CODE}"
