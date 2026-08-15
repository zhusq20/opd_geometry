#!/usr/bin/env bash
# Qwen3-1.7B student preset with a selectable external SGLang teacher.
# The historical filename is retained for compatibility; TEACHER defaults to
# qwen3-8b and also accepts qwen3-4b-thinking-2507.

set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
PROJECT_CHECKPOINT_ROOT="$(cd -- "${SLIME_DIR}/.." && pwd)/checkpoints"
SHARED_CHECKPOINT_ROOT="$(cd -- "${SLIME_DIR}/../.." && pwd)/checkpoints"

TEACHER="${TEACHER:-qwen3-8b}"
case "${TEACHER,,}" in
  qwen3-8b|qwen/qwen3-8b)
    TEACHER_SLUG="qwen3_8b"
    TEACHER_DISPLAY_NAME="Qwen3-8B"
    TEACHER_CHECKPOINT_NAME="Qwen3-8B"
    DEFAULT_TEACHER_PORT=13141
    ;;
  qwen3-4b-thinking-2507|qwen/qwen3-4b-thinking-2507)
    TEACHER_SLUG="qwen3_4b_thinking_2507"
    TEACHER_DISPLAY_NAME="Qwen3-4B-Thinking-2507"
    TEACHER_CHECKPOINT_NAME="Qwen3-4B-Thinking-2507"
    DEFAULT_TEACHER_PORT=13142
    ;;
  *)
    echo "Unsupported TEACHER=${TEACHER}; use qwen3-8b or qwen3-4b-thinking-2507." >&2
    exit 2
    ;;
esac

if [[ -d "${PROJECT_CHECKPOINT_ROOT}/Qwen3-1.7B" && \
      -d "${PROJECT_CHECKPOINT_ROOT}/Qwen3-1.7B_torch_dist" ]]; then
  DEFAULT_STUDENT_ROOT="${PROJECT_CHECKPOINT_ROOT}"
else
  DEFAULT_STUDENT_ROOT="${SHARED_CHECKPOINT_ROOT}"
fi
if [[ -d "${PROJECT_CHECKPOINT_ROOT}/${TEACHER_CHECKPOINT_NAME}" ]]; then
  DEFAULT_TEACHER_MODEL="${PROJECT_CHECKPOINT_ROOT}/${TEACHER_CHECKPOINT_NAME}"
else
  DEFAULT_TEACHER_MODEL="${SHARED_CHECKPOINT_ROOT}/${TEACHER_CHECKPOINT_NAME}"
fi

export MODEL_CONFIG="${MODEL_CONFIG:-${SLIME_DIR}/scripts/models/qwen3-1.7B.sh}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${DEFAULT_STUDENT_ROOT}/Qwen3-1.7B}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${DEFAULT_STUDENT_ROOT}/Qwen3-1.7B_torch_dist}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${DEFAULT_TEACHER_MODEL}}"
export TEACHER_NAME="${TEACHER_DISPLAY_NAME}"
export ALGORITHM="${ALGORITHM:-opd}"
export BATCH_PROFILE="${BATCH_PROFILE:-responsive16}"
export RUN_NAME="${RUN_NAME:-qwen3_1.7b_${TEACHER_SLUG}_${ALGORITHM}_${OPTIMIZER:-adamw}_${BATCH_PROFILE}_seed${SEED:-42}}"

export TEACHER_CONFIG="${TEACHER_CONFIG:-${EXAMPLE_DIR}/configs/single_teacher.yaml}"
TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-${DEFAULT_TEACHER_PORT}}"
if [[ "${TEACHER_SLUG}" == "qwen3_8b" && -n "${QWEN3_8B_TEACHER_URL:-}" ]]; then
  OPD_TEACHER_URL="${OPD_TEACHER_URL:-${QWEN3_8B_TEACHER_URL}}"
fi
export OPD_TEACHER_URL="${OPD_TEACHER_URL:-http://${TEACHER_HOST}:${TEACHER_PORT}/generate}"
if [[ "${TEACHER_SLUG}" == "qwen3_8b" ]]; then
  export QWEN3_8B_TEACHER_URL="${QWEN3_8B_TEACHER_URL:-${OPD_TEACHER_URL}}"
fi
TEACHER_HEALTH_URL="${TEACHER_HEALTH_URL:-${OPD_TEACHER_URL%/generate}/health_generate}"

DRY_RUN="${DRY_RUN:-0}"
START_TEACHER="${START_TEACHER:-1}"
REUSE_TEACHER="${REUSE_TEACHER:-0}"
case "${DRY_RUN}:${START_TEACHER}:${REUSE_TEACHER}" in
  [01]:[01]:[01]) ;;
  *) echo "DRY_RUN, START_TEACHER, and REUSE_TEACHER must each be 0 or 1." >&2; exit 2 ;;
esac
case "${GPU_PREFLIGHT:-1}" in
  0|1) ;;
  *) echo "GPU_PREFLIGHT must be 0 or 1." >&2; exit 2 ;;
esac

is_opd=0
case "${ALGORITHM}" in
  opd|sft_opd|grpo_opd|ppo_opd) is_opd=1 ;;
esac
if [[ "${is_opd}" != "1" ]]; then
  export TEACHER_NAME="none"
fi

# Capture the caller's full device list before restricting the training job.
AVAILABLE_CUDA_DEVICES="${AVAILABLE_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${AVAILABLE_CUDA_DEVICES}" ]]; then
  echo "Set AVAILABLE_CUDA_DEVICES (or CUDA_VISIBLE_DEVICES) explicitly so the student cannot take a teacher or shared GPU." >&2
  exit 2
fi
IFS=',' read -r -a AVAILABLE_GPUS <<< "${AVAILABLE_CUDA_DEVICES}"

TRAIN_GPU_COUNT_WAS_SET="${TRAIN_GPU_COUNT+x}"
TRAIN_GPU_COUNT="${TRAIN_GPU_COUNT:-4}"
if ! [[ "${TRAIN_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_GPU_COUNT must be a positive integer." >&2
  exit 2
fi
if [[ -z "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
  if (( ${#AVAILABLE_GPUS[@]} < TRAIN_GPU_COUNT )); then
    echo "Need ${TRAIN_GPU_COUNT} student GPUs, but only ${#AVAILABLE_GPUS[@]} were provided." >&2
    exit 2
  fi
  TRAIN_GPUS=("${AVAILABLE_GPUS[@]:0:TRAIN_GPU_COUNT}")
  TRAIN_CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${TRAIN_GPUS[*]}")"
else
  IFS=',' read -r -a TRAIN_GPUS <<< "${TRAIN_CUDA_VISIBLE_DEVICES}"
  if [[ -n "${TRAIN_GPU_COUNT_WAS_SET}" && "${TRAIN_GPU_COUNT}" != "${#TRAIN_GPUS[@]}" ]]; then
    echo "TRAIN_GPU_COUNT=${TRAIN_GPU_COUNT} does not match TRAIN_CUDA_VISIBLE_DEVICES." >&2
    exit 2
  fi
  TRAIN_GPU_COUNT="${#TRAIN_GPUS[@]}"
fi

TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
if ! [[ "${TEACHER_TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TEACHER_TP_SIZE must be a positive integer." >&2
  exit 2
fi
ALLOW_TEACHER_STUDENT_GPU_OVERLAP="${ALLOW_TEACHER_STUDENT_GPU_OVERLAP:-0}"
case "${ALLOW_TEACHER_STUDENT_GPU_OVERLAP}" in
  0|1) ;;
  *) echo "ALLOW_TEACHER_STUDENT_GPU_OVERLAP must be 0 or 1." >&2; exit 2 ;;
esac
if [[ "${is_opd}" == "1" && "${START_TEACHER}" == "1" ]]; then
  if [[ -z "${TEACHER_CUDA_VISIBLE_DEVICES:-}" ]]; then
    if (( ${#AVAILABLE_GPUS[@]} < TRAIN_GPU_COUNT + TEACHER_TP_SIZE )); then
      echo "Need ${TRAIN_GPU_COUNT} student GPUs plus ${TEACHER_TP_SIZE} teacher GPUs." >&2
      exit 2
    fi
    TEACHER_GPUS=("${AVAILABLE_GPUS[@]:TRAIN_GPU_COUNT:TEACHER_TP_SIZE}")
    TEACHER_CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${TEACHER_GPUS[*]}")"
  else
    IFS=',' read -r -a TEACHER_GPUS <<< "${TEACHER_CUDA_VISIBLE_DEVICES}"
  fi
  if (( ${#TEACHER_GPUS[@]} != TEACHER_TP_SIZE )); then
    echo "TEACHER_TP_SIZE=${TEACHER_TP_SIZE} does not match TEACHER_CUDA_VISIBLE_DEVICES." >&2
    exit 2
  fi
  for train_gpu in "${TRAIN_GPUS[@]}"; do
    for teacher_gpu in "${TEACHER_GPUS[@]}"; do
      if [[ "${train_gpu}" == "${teacher_gpu}" ]]; then
        if [[ "${ALLOW_TEACHER_STUDENT_GPU_OVERLAP}" != "1" ]]; then
          echo "Student and teacher GPU lists overlap at device ${train_gpu}." >&2
          echo "Set ALLOW_TEACHER_STUDENT_GPU_OVERLAP=1 only when shared capacity has been checked explicitly." >&2
          exit 2
        fi
        echo "Warning: student and teacher explicitly share GPU ${train_gpu}." >&2
      fi
    done
  done
fi

if [[ -n "${NUM_GPUS:-}" && "${NUM_GPUS}" != "${#TRAIN_GPUS[@]}" ]]; then
  echo "NUM_GPUS=${NUM_GPUS} must match the ${#TRAIN_GPUS[@]} devices in TRAIN_CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
export NUM_GPUS="${#TRAIN_GPUS[@]}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
export TP_SIZE="${TP_SIZE:-1}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${NUM_GPUS}}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.6}"
if ! [[ "${ROLLOUT_NUM_GPUS}" =~ ^[1-9][0-9]*$ && "${ROLLOUT_GPUS_PER_ENGINE}" =~ ^[1-9][0-9]*$ ]] || \
   (( ROLLOUT_NUM_GPUS % ROLLOUT_GPUS_PER_ENGINE != 0 )); then
  echo "ROLLOUT_NUM_GPUS must be a positive multiple of ROLLOUT_GPUS_PER_ENGINE." >&2
  exit 2
fi

# Plain Math GRPO/PPO trains with an 8k response cap but evaluates at 32k. The
# same SGLang engines serve both phases, so size their concurrency for the
# longer of the train and eval caps. On the 96 GiB RTX PRO 6000 test host,
# Qwen3-1.7B at mem_fraction=0.6 exposes about 498k KV-cache tokens per engine;
# response caps 4096/8192/32768 use calibrated ceilings 72/44/12.
if [[ -z "${EVAL_MAX_RESPONSE_LEN:-}" && "${TASK:-}" == "math" ]]; then
  case "${ALGORITHM}" in
    grpo|ppo) export EVAL_MAX_RESPONSE_LEN=32768 ;;
  esac
fi
sglang_capacity_response_len="${MAX_RESPONSE_LEN:-8192}"
if [[ "${sglang_capacity_response_len}" =~ ^[1-9][0-9]*$ && \
      "${EVAL_MAX_RESPONSE_LEN:-}" =~ ^[1-9][0-9]*$ ]] && \
   (( 10#${EVAL_MAX_RESPONSE_LEN} > 10#${sglang_capacity_response_len} )); then
  sglang_capacity_response_len="${EVAL_MAX_RESPONSE_LEN}"
fi
if [[ -z "${SGLANG_MAX_RUNNING_REQUESTS:-}" && "${sglang_capacity_response_len}" -ge 4096 ]]; then
  if [[ "${sglang_capacity_response_len}" == "32768" && "${MAX_PROMPT_LEN:-2048}" -le 2048 ]]; then
    export SGLANG_MAX_RUNNING_REQUESTS=12
  elif [[ "${sglang_capacity_response_len}" == "4096" && "${MAX_PROMPT_LEN:-2048}" -le 2048 ]]; then
    export SGLANG_MAX_RUNNING_REQUESTS=72
  else
    export SGLANG_MAX_RUNNING_REQUESTS=44
  fi
fi

if [[ "${GPU_PREFLIGHT:-1}" == "1" && "${DRY_RUN}" != "1" ]]; then
  python3 "${EXAMPLE_DIR}/gpu_preflight.py" \
    --devices "${TRAIN_CUDA_VISIBLE_DEVICES}" \
    --role student \
    --min-free-mib "${STUDENT_MIN_FREE_MIB:-75000}" \
    --min-free-fraction "${STUDENT_MIN_FREE_FRACTION:-0.90}"
  if [[ "${is_opd}" == "1" && "${START_TEACHER}" == "1" ]]; then
    python3 "${EXAMPLE_DIR}/gpu_preflight.py" \
      --devices "${TEACHER_CUDA_VISIBLE_DEVICES}" \
      --role teacher \
      --min-free-mib "${TEACHER_MIN_FREE_MIB:-70000}" \
      --min-free-fraction "${TEACHER_MIN_FREE_FRACTION:-0.90}"
  fi
fi

echo "Student: Qwen3-1.7B (${HF_CHECKPOINT}); training GPUs=${CUDA_VISIBLE_DEVICES}"
if [[ "${is_opd}" == "1" ]]; then
  echo "Teacher: ${TEACHER_DISPLAY_NAME} (${TEACHER_MODEL_PATH}); endpoint=${OPD_TEACHER_URL}"
fi

TEACHER_PID=""
TEACHER_PGID=""
cleanup_teacher() {
  if [[ -n "${TEACHER_PGID}" ]]; then
    kill -- "-${TEACHER_PGID}" 2>/dev/null || true
    wait "${TEACHER_PID}" 2>/dev/null || true
  fi
  TEACHER_PID=""
  TEACHER_PGID=""
}
trap cleanup_teacher EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

teacher_is_ready() {
  curl --noproxy '*' -fsS --max-time 2 "${TEACHER_HEALTH_URL}" >/dev/null 2>&1
}

if [[ "${is_opd}" == "1" && "${DRY_RUN}" != "1" ]]; then
  if [[ "${START_TEACHER}" == "1" ]]; then
    if [[ ! -f "${TEACHER_MODEL_PATH}/config.json" ]] || \
       ! compgen -G "${TEACHER_MODEL_PATH}/*.safetensors" >/dev/null; then
      echo "Incomplete ${TEACHER_DISPLAY_NAME} Hugging Face checkpoint: ${TEACHER_MODEL_PATH}" >&2
      exit 2
    fi
    if [[ "${CHECK_TOKENIZER_COMPAT:-1}" == "1" ]]; then
      python3 "${EXAMPLE_DIR}/validate_tokenizers.py" \
        --student "${HF_CHECKPOINT}" \
        --teacher "${TEACHER_MODEL_PATH}"
    fi
    if teacher_is_ready; then
      if [[ "${REUSE_TEACHER}" != "1" ]]; then
        echo "A server already answers at ${TEACHER_HEALTH_URL}; set REUSE_TEACHER=1 to use it." >&2
        exit 2
      fi
    else
      TEACHER_LOG="${TEACHER_LOG:-/tmp/slime-${TEACHER_SLUG}-teacher-${TEACHER_PORT}.log}"
      TEACHER_CMD=(
        python3 -m sglang.launch_server
        --model-path "${TEACHER_MODEL_PATH}"
        --host "${TEACHER_BIND_HOST:-127.0.0.1}"
        --port "${TEACHER_PORT}"
        --tp "${TEACHER_TP_SIZE}"
        --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
        --mem-fraction-static "${TEACHER_MEM_FRACTION:-0.6}"
      )
      if [[ -n "${TEACHER_EXTRA_ARGS:-}" ]]; then
        read -r -a TEACHER_USER_ARGS <<< "${TEACHER_EXTRA_ARGS}"
        TEACHER_CMD+=("${TEACHER_USER_ARGS[@]}")
      fi
      CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" \
        setsid "${TEACHER_CMD[@]}" >"${TEACHER_LOG}" 2>&1 &
      TEACHER_PID=$!
      TEACHER_PGID="${TEACHER_PID}"
      echo "Starting ${TEACHER_DISPLAY_NAME} teacher on GPUs=${TEACHER_CUDA_VISIBLE_DEVICES}; log=${TEACHER_LOG}"

      TEACHER_START_TIMEOUT="${TEACHER_START_TIMEOUT:-600}"
      deadline=$((SECONDS + TEACHER_START_TIMEOUT))
      until teacher_is_ready; do
        if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
          echo "${TEACHER_DISPLAY_NAME} teacher exited during startup. Last log lines:" >&2
          tail -n 40 "${TEACHER_LOG}" >&2 || true
          exit 1
        fi
        if (( SECONDS >= deadline )); then
          echo "${TEACHER_DISPLAY_NAME} teacher did not become ready within ${TEACHER_START_TIMEOUT}s. Last log lines:" >&2
          tail -n 40 "${TEACHER_LOG}" >&2 || true
          exit 1
        fi
        sleep 5
      done
      echo "${TEACHER_DISPLAY_NAME} teacher is ready."
    fi
  elif ! teacher_is_ready; then
    echo "External teacher is not ready at ${TEACHER_HEALTH_URL}." >&2
    exit 2
  fi
fi

bash "${EXAMPLE_DIR}/run_experiment.sh"
