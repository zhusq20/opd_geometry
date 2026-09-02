#!/usr/bin/env bash
set -euo pipefail

[[ $# == 5 ]] || { echo "internal usage: $0 RUN_MODE RUN_ID K ALLOCATION ADAMW_STATE" >&2; exit 2; }
RUN_MODE="$1"
RUN_ID="$2"
TASK_WIDTH="$3"
ALLOCATION="$4"
ADAMW_STATE="$5"
case "${RUN_MODE}" in warm|train|bank) ;; *) echo "Unknown MOPD run mode: ${RUN_MODE}" >&2; exit 2 ;; esac

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export MOPD_TEACHER_HF_ROOT="${MOPD_TEACHER_HF_ROOT:-${EXAMPLE_DIR}/generated/teachers_hf}"
export MOPD_TEACHER_GPU="${MOPD_TEACHER_GPU:-4}"
export MOPD_TEACHER_PORT="${MOPD_TEACHER_PORT:-31001}"
export MOPD_TEACHER_SLOT_STATE="${MOPD_TEACHER_SLOT_STATE:-${EXAMPLE_DIR}/generated/teacher_slot/state.json}"

MOPD_SEED="${MOPD_SEED:-42}"
case "${MOPD_SEED}" in 42|43|44) ;; *) echo "MOPD_SEED must be 42, 43, or 44." >&2; exit 2 ;; esac
if [[ -n "${MOPD_GENERATED_DIR:-}" ]]; then
  GENERATED="${MOPD_GENERATED_DIR}"
elif [[ "${MOPD_SEED}" == 42 ]]; then
  GENERATED="${EXAMPLE_DIR}/generated/mopd"
else
  GENERATED="${EXAMPLE_DIR}/generated/mopd_seed${MOPD_SEED}"
fi
TRAIN_MANIFEST="${GENERATED}/train.yaml"
EVAL_CONFIG="${GENERATED}/teacher_loss_eval.yaml"
PROTOCOL="${GENERATED}/protocol.json"
TEACHER_ROUTER="${EXAMPLE_DIR}/configs/teacher_router.yaml"
OUTPUT_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_64k_v3}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}-seed${MOPD_SEED}"
WARM_DIR="${MOPD_WARM_DIR:-${OUTPUT_ROOT}/warm_start-seed${MOPD_SEED}}"
HF_CHECKPOINT="${MOPD_HF_CHECKPOINT:-/workspace/dev/checkpoints/Qwen3-1.7B}"
BASE_MEGATRON="${MOPD_BASE_MEGATRON:-/workspace/dev/checkpoints/Qwen3-1.7B_torch_dist}"
MODEL_CONFIG="${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh"

for path in "${TRAIN_MANIFEST}" "${EVAL_CONFIG}" "${PROTOCOL}" "${TEACHER_ROUTER}"; do
  [[ -f "${path}" ]] || { echo "Missing ${path}; run prepare_mopd.py." >&2; exit 2; }
done
[[ -f "${HF_CHECKPOINT}/config.json" ]] || { echo "Missing ${HF_CHECKPOINT}." >&2; exit 2; }
if [[ "${RUN_MODE}" == warm ]]; then
  [[ -f "${BASE_MEGATRON}/latest_checkpointed_iteration.txt" ]] || {
    echo "Missing ${BASE_MEGATRON}." >&2; exit 2;
  }
fi

DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-1}"
MOPD_RESUME="${MOPD_RESUME:-0}"
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || { echo "DRY_RUN must be 0 or 1." >&2; exit 2; }
[[ "${USE_WANDB}" == 0 || "${USE_WANDB}" == 1 ]] || { echo "USE_WANDB must be 0 or 1." >&2; exit 2; }
[[ "${MOPD_RESUME}" == 0 || "${MOPD_RESUME}" == 1 ]] || { echo "MOPD_RESUME must be 0 or 1." >&2; exit 2; }
if [[ "${MOPD_RESUME}" == 1 && ( "${RUN_MODE}" != train || "${DRY_RUN}" == 1 ) ]]; then
  echo "Checkpoint resume is only available for a real train run." >&2
  exit 2
fi
if [[ "${MOPD_SEED}" != 42 ]]; then
  RESPONSE_BUDGET="${MOPD_RESPONSE_BUDGET:-16384}"
  CHECKPOINT_RESPONSES="${MOPD_CHECKPOINT_RESPONSES:-16384}"
  EVAL_RESPONSES="${MOPD_EVAL_RESPONSES:-2048,4096,8192,16384}"
else
  RESPONSE_BUDGET="${MOPD_RESPONSE_BUDGET:-64000}"
  CHECKPOINT_RESPONSES="${MOPD_CHECKPOINT_RESPONSES:-16384,32768,64000}"
  EVAL_RESPONSES="${MOPD_EVAL_RESPONSES:-2048,4096,8192,16384,32768,49152,64000}"
fi

EXTRA_START_ARGS=()
BANK_ARGS=()
if [[ "${RUN_MODE}" == warm ]]; then
  LOAD_CHECKPOINT="${BASE_MEGATRON}"
  START_ARGS=(--start-rollout-id 0 --no-load-optim --no-load-rng)
  NUM_ROLLOUT=8
  INITIAL_RESIDENT=math
elif [[ "${RUN_MODE}" == train ]]; then
  if [[ "${MOPD_RESUME}" == 1 ]]; then
    [[ -d "${RUN_DIR}" ]] || { echo "Resume run directory does not exist: ${RUN_DIR}" >&2; exit 2; }
    RESUME_JSON="$(python3 "${EXAMPLE_DIR}/prepare_resume.py" --run-dir "${RUN_DIR}")"
    read -r RESUME_CHECKPOINT_ID RESUME_START_ID INITIAL_RESIDENT RESUME_EVAL_ON_START < <(
      python3 - "${RESUME_JSON}" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
print(value["rollout_id"], value["next_rollout_id"], value["resident_task"], int(value["eval_on_start"]))
PY
    )
    LOAD_CHECKPOINT="${RUN_DIR}/checkpoints"
    START_ARGS=(--start-rollout-id "${RESUME_START_ID}")
    [[ "${RESUME_EVAL_ON_START}" == 0 ]] || START_ARGS+=(--mopd-eval-on-start)
    EXTRA_START_ARGS=(--ckpt-step "${RESUME_CHECKPOINT_ID}")
  else
    LOAD_CHECKPOINT="${WARM_DIR}/checkpoints"
    START_ARGS=(--start-rollout-id 8 --mopd-reset-sampler --override-opt-param-scheduler)
    INITIAL_RESIDENT=science
  fi
  if [[ "${DRY_RUN}" == 0 && "${MOPD_RESUME}" == 0 ]]; then
    [[ -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || {
      echo "Warm checkpoint missing; run run_warm_start.sh first." >&2; exit 2;
    }
    [[ -f "${LOAD_CHECKPOINT}/rollout/mopd_dataset_state_dict_7.pt" ]] || {
      echo "Warm sampler state missing from ${LOAD_CHECKPOINT}." >&2; exit 2;
    }
  fi
  NUM_ROLLOUT=20000
else
  LOAD_CHECKPOINT="${MOPD_BANK_LOAD_CHECKPOINT:?set MOPD_BANK_LOAD_CHECKPOINT for bank collection}"
  BANK_CHECKPOINT_ID="${MOPD_BANK_CHECKPOINT_ID:?set MOPD_BANK_CHECKPOINT_ID for bank collection}"
  BANK_START_ID=$((BANK_CHECKPOINT_ID + 1))
  if [[ "${DRY_RUN}" == 0 ]]; then
    [[ -f "${LOAD_CHECKPOINT}/rollout/mopd_dataset_state_dict_${BANK_CHECKPOINT_ID}.pt" ]] || {
      echo "Frozen-bank sampler state is missing for rollout ${BANK_CHECKPOINT_ID}." >&2; exit 2;
    }
  fi
  START_ARGS=(
    --start-rollout-id "${BANK_START_ID}"
    --mopd-reset-sampler
    --override-opt-param-scheduler
  )
  EXTRA_START_ARGS=(--ckpt-step "${BANK_CHECKPOINT_ID}")
  NUM_ROLLOUT=$((BANK_START_ID + 32))
  INITIAL_RESIDENT="${MOPD_BANK_RESIDENT_TASK:?set MOPD_BANK_RESIDENT_TASK for bank collection}"
  BANK_ARGS=(--mopd-bank-dir "${RUN_DIR}/bank" --mopd-bank-coordinates 65536 --mopd-bank-units-per-task 8)
fi

CHECKPOINT_FOR_PROVENANCE="${LOAD_CHECKPOINT}"
if [[ "${MOPD_RESUME}" == 1 ]]; then
  CHECKPOINT_FOR_PROVENANCE="${LOAD_CHECKPOINT}/iter_$(printf '%07d' "${RESUME_CHECKPOINT_ID}")"
fi

if [[ "${MOPD_RESUME}" == 0 && "${DRY_RUN}" == 0 && -d "${RUN_DIR}" && -n "$(find "${RUN_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to mix a new run into non-empty ${RUN_DIR}." >&2
  exit 2
fi

export MOPD_TRAIN_CUDA_VISIBLE_DEVICES="${MOPD_TRAIN_CUDA_VISIBLE_DEVICES:-3}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/rm_${MOPD_SEED}_${MOPD_TRAIN_CUDA_VISIBLE_DEVICES}_$$}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
# Separate Ray heads started on the same host can otherwise race while picking
# identical ephemeral dashboard-agent and metrics ports.  Each dashboard port
# gets a disjoint, reproducible auxiliary-port block.
export RAY_AUX_PORT_BASE="${RAY_AUX_PORT_BASE:-$((45000 + (RAY_DASHBOARD_PORT - 8265) * 20))}"
IFS=, read -r -a TRAIN_GPUS <<< "${MOPD_TRAIN_CUDA_VISIBLE_DEVICES}"
[[ "${#TRAIN_GPUS[@]}" == 1 ]] || { echo "Exactly one student GPU ID is required." >&2; exit 2; }
NUM_STUDENT_GPUS="${#TRAIN_GPUS[@]}"
for gpu in "${TRAIN_GPUS[@]}"; do
  [[ "${gpu}" != "${MOPD_TEACHER_GPU}" ]] || { echo "Teacher GPU overlaps student GPUs." >&2; exit 2; }
done

if [[ "${DRY_RUN}" == 0 ]]; then
  python3 "${EXAMPLE_DIR}/verify_hardware.py" \
    --gpu-ids "${MOPD_TRAIN_CUDA_VISIBLE_DEVICES},${MOPD_TEACHER_GPU}"
  bash "${EXAMPLE_DIR}/serve_teachers.sh" status
  bash "${EXAMPLE_DIR}/serve_teachers.sh" switch "${INITIAL_RESIDENT}"
  if [[ "${USE_WANDB}" == 1 && "${WANDB_MODE:-online}" == online ]]; then
    python3 - <<'PY'
import wandb
if not wandb.api.api_key:
    raise SystemExit("W&B online mode requested, but no API key is configured")
print("W&B authentication is configured.")
PY
  fi
fi

export MODEL_ARGS_ROTARY_BASE=1000000
source "${MODEL_CONFIG}"
export CUDA_VISIBLE_DEVICES="${MOPD_TRAIN_CUDA_VISIBLE_DEVICES}"

TRAIN_CMD=(
  python3 "${SLIME_ROOT}/train.py"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load "${LOAD_CHECKPOINT}"
  --save "${RUN_DIR}/checkpoints"
  --save-interval 1
  "${START_ARGS[@]}"
  "${EXTRA_START_ARGS[@]}"

  --prompt-data "${TRAIN_MANIFEST}"
  --data-source-path slime_plugins.mopd.data_source.MOPDRolloutDataSource
  --rollout-function-path slime_plugins.mopd.rollout.generate_rollout
  --input-key prompt --label-key label --metadata-key metadata --tool-key tools
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}'
  --rollout-global-dataset --rollout-shuffle --rollout-seed "${MOPD_SEED}"
  --num-rollout "${NUM_ROLLOUT}" --rollout-batch-size 16 --global-batch-size 64
  --n-samples-per-prompt 4
  --rollout-max-prompt-len 2048 --rollout-max-response-len 8192
  --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1
  --balance-data

  --mopd-enabled --mopd-run-mode "${RUN_MODE}"
  --mopd-allocation "${ALLOCATION}" --mopd-task-width "${TASK_WIDTH}"
  --mopd-adamw-state "${ADAMW_STATE}" --mopd-seed "${MOPD_SEED}"
  --mopd-ema-decay 0.95 --mopd-inclusion-floor 0.05 --mopd-score-max-age 50
  --mopd-response-budget "${RESPONSE_BUDGET}"
  --mopd-checkpoint-responses "${CHECKPOINT_RESPONSES}"
  --mopd-eval-responses "${EVAL_RESPONSES}"
  --mopd-failure-penalty 10.0 --mopd-score-chunk-size 1048576
  --mopd-output-dir "${RUN_DIR}/allocation"
  "${BANK_ARGS[@]}"

  --advantage-estimator grpo --use-rollout-logprobs
  --use-opd --opd-type sglang --opd-kl-coef 1.0
  --opd-teacher-router-config "${TEACHER_ROUTER}" --opd-task-reward-weight 0.0
  --custom-rm-path slime_plugins.m2rl.opd.teacher_reward
  --custom-reward-post-process-path slime_plugins.m2rl.opd.post_process_rewards
  --entropy-coef 0.0 --kl-coef 0.0 --kl-loss-coef 0.0
  --eps-clip 0.2 --eps-clip-high 0.2

  --optimizer adam --lr 2.5e-7 --weight-decay 0.0
  --adam-beta1 0.9 --adam-beta2 0.9987381276 --adam-eps 1e-8
  --lr-decay-style constant --lr-warmup-iters 0 --clip-grad 1.0

  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-10240}"

  --eval-config "${EVAL_CONFIG}"
  --eval-function-path slime_plugins.mopd.eval.generate_teacher_loss_eval
  --eval-interval 1 --eval-max-concurrency "${EVAL_MAX_CONCURRENCY:-16}"
  --eval-artifact-dir "${RUN_DIR}/teacher_loss_eval"

  --metrics-output-dir "${RUN_DIR}/metrics"
  --run-manifest-path "${RUN_DIR}/provenance/run_manifest.json"
  --completion-marker-path "${RUN_DIR}/run_complete.json"
  --experiment-task multi --experiment-teacher four_frozen_grpo_teachers_one_slot
  --experiment-condition "${RUN_ID}" --experiment-name "${RUN_ID}-seed${MOPD_SEED}"
  --experiment-optimizer adamw --experiment-data-index "${PROTOCOL}"

  --rollout-num-gpus "${NUM_STUDENT_GPUS}" --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.5}"
  --sglang-enable-deterministic-inference
  --actor-num-nodes 1 --actor-num-gpus-per-node "${NUM_STUDENT_GPUS}"
  --colocate --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash --seed "${MOPD_SEED}"
)
if [[ "${USE_WANDB}" == 1 ]]; then
  TRAIN_CMD+=(
    --use-wandb --wandb-mode "${WANDB_MODE:-online}" --wandb-dir "${RUN_DIR}/wandb"
    --wandb-team "${WANDB_ENTITY:-zsqzz}" --wandb-project "${WANDB_PROJECT:-iclr2027-mopd-gpas-64k}"
    --wandb-group "${WANDB_GROUP:-Qwen3-1.7B-4T-exact-set-MOPD-64K}" \
    --wandb-run-name "${RUN_ID}-seed${MOPD_SEED}"
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
        raise SystemExit(f"resume frontier changed during launch: {key} {before[key]} != {after[key]}")
print(
    f"Resume prepared at response={after['attempted_responses']} rollout={after['rollout_id']}; "
    f"rewind={after['rewind_required']} archive={after['archive']}"
)
PY
  PROVENANCE_ACTION=resume
fi
python3 "${EXAMPLE_DIR}/provenance.py" "${PROVENANCE_ACTION}" --repo "${SLIME_ROOT}" --run-dir "${RUN_DIR}" \
  --input "${PROTOCOL}" --input "${TRAIN_MANIFEST}" --input "${EVAL_CONFIG}" --input "${TEACHER_ROUTER}" \
  --checkpoint "${HF_CHECKPOINT}" --checkpoint "${CHECKPOINT_FOR_PROVENANCE}" \
  --source "${EXAMPLE_DIR}" \
  --source "${SLIME_ROOT}/slime_plugins/mopd" \
  --source "${SLIME_ROOT}/slime_plugins/m2rl/opd.py" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/actor.py" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/data.py" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/model.py" \
  --source "${SLIME_ROOT}/slime/backends/megatron_utils/optimizer_factory.py" \
  --source "${SLIME_ROOT}/slime/ray/rollout.py" \
  --source "${SLIME_ROOT}/slime/utils/arguments.py" \
  --source "${SLIME_ROOT}/slime/utils/logging_utils.py" \
  --source "${SLIME_ROOT}/slime/utils/wandb_utils.py" \
  --source "${SLIME_ROOT}/train.py" -- "${TRAIN_CMD[@]}"

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
  DASHBOARD_PORT="${RAY_DASHBOARD_PORT}"
  GCS_PORT="${RAY_GCS_PORT}"
  AUX_PORT_BASE="${RAY_AUX_PORT_BASE}"
  [[ "${DASHBOARD_PORT}" =~ ^[0-9]+$ && "${GCS_PORT}" =~ ^[0-9]+$ && "${AUX_PORT_BASE}" =~ ^[0-9]+$ ]] || {
    echo "Ray ports must be decimal integers." >&2
    exit 2
  }
  mkdir -p "${RAY_TEMP_DIR}"
  ray start --head --node-ip-address "${MASTER_ADDR}" --port "${GCS_PORT}" \
    --num-gpus "${NUM_STUDENT_GPUS}" --disable-usage-stats \
    --dashboard-host 0.0.0.0 --dashboard-port "${DASHBOARD_PORT}" \
    --dashboard-agent-listen-port "$((AUX_PORT_BASE + 0))" \
    --dashboard-agent-grpc-port "$((AUX_PORT_BASE + 1))" \
    --runtime-env-agent-port "$((AUX_PORT_BASE + 2))" \
    --metrics-export-port "$((AUX_PORT_BASE + 3))" \
    --ray-client-server-port "$((AUX_PORT_BASE + 4))" \
    --object-manager-port "$((AUX_PORT_BASE + 5))" \
    --node-manager-port "$((AUX_PORT_BASE + 6))" \
    --temp-dir "${RAY_TEMP_DIR}" --block &
  RAY_PID=$!
  RAY_ADDRESS="http://${MASTER_ADDR}:${DASHBOARD_PORT}"
  AGENT_LOG="${RAY_TEMP_DIR}/session_latest/logs/dashboard_agent.log"
  RAY_READY=0
  for _ in $(seq 1 120); do
    kill -0 "${RAY_PID}" 2>/dev/null || { echo "Ray head exited." >&2; exit 1; }
    if [[ -f "${AGENT_LOG}" ]] && grep -q "Dashboard agent http address:" "${AGENT_LOG}" \
      && ray job list --address "${RAY_ADDRESS}" >/dev/null 2>&1; then
      RAY_READY=1
      break
    fi
    if [[ -f "${AGENT_LOG}" ]] && grep -q "HTTP service will be disabled" "${AGENT_LOG}"; then
      echo "Ray dashboard agent failed to bind; inspect ${AGENT_LOG}." >&2
      exit 1
    fi
    sleep 1
  done
  [[ "${RAY_READY}" == 1 ]] || { echo "Ray Jobs API or dashboard agent not ready." >&2; exit 1; }
fi
RUNTIME_ENV_JSON="$(python3 - "${PYTHONPATH}" <<'PY'
import json, os, sys
keys=("WANDB_API_KEY","WANDB_BASE_URL","NLTK_DATA","CUDA_VISIBLE_DEVICES","MOPD_DATA_ROOT","MOPD_HF_CHECKPOINT","MOPD_TEACHER_HF_ROOT","MOPD_TEACHER_GPU","MOPD_TEACHER_PORT","MOPD_TEACHER_SLOT_STATE")
env={"PYTHONPATH":sys.argv[1],"CUDA_DEVICE_MAX_CONNECTIONS":"1","PYTHONUNBUFFERED":"1"}
env.update({key:os.environ[key] for key in keys if os.environ.get(key)})
print(json.dumps({"env_vars":env}))
PY
)"
set +e
ray job submit --address "${RAY_ADDRESS}" --runtime-env-json "${RUNTIME_ENV_JSON}" -- "${TRAIN_CMD[@]}"
EXIT_CODE=$?
set -e
python3 "${EXAMPLE_DIR}/provenance.py" finish --run-dir "${RUN_DIR}" --exit-code "${EXIT_CODE}"
exit "${EXIT_CODE}"
