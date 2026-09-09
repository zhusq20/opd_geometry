#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
source "${EXAMPLE_DIR}/_profile.sh"
export MOPD_TEACHER_HF_ROOT="${MOPD_TEACHER_HF_ROOT:-${SLIME_ROOT}/local/mopd_assets/models/teachers_hf}"
export MOPD_TRAIN_GPU="${MOPD_TRAIN_GPU:-0}"
export MOPD_TRAIN_GPUS="${MOPD_TRAIN_GPUS:-${MOPD_TRAIN_GPU}}"
export MOPD_INFERENCE_GPU="${MOPD_INFERENCE_GPU:-1}"
export MOPD_HARDWARE_PROFILE="${MOPD_HARDWARE_PROFILE:-frozen-96gb-tp1}"
export MOPD_TEACHER_MATH_PORT="${MOPD_TEACHER_MATH_PORT:-31001}"
export MOPD_TEACHER_CODE_PORT="${MOPD_TEACHER_CODE_PORT:-31002}"
export MOPD_TEACHER_IF_PORT="${MOPD_TEACHER_IF_PORT:-31003}"
export MOPD_TEACHER_SCIENCE_PORT="${MOPD_TEACHER_SCIENCE_PORT:-31004}"
export MOPD_TEACHER_MATH_GPU="${MOPD_TEACHER_MATH_GPU:-${MOPD_INFERENCE_GPU}}"
export MOPD_TEACHER_CODE_GPU="${MOPD_TEACHER_CODE_GPU:-${MOPD_INFERENCE_GPU}}"
export MOPD_TEACHER_IF_GPU="${MOPD_TEACHER_IF_GPU:-${MOPD_INFERENCE_GPU}}"
export MOPD_TEACHER_SCIENCE_GPU="${MOPD_TEACHER_SCIENCE_GPU:-${MOPD_INFERENCE_GPU}}"
SERVER_DIR="${MOPD_TEACHER_SERVER_DIR:-${SLIME_ROOT}/local/mopd_teacher_servers}"
ACTION="${1:-status}"
TASKS=("${MOPD_PROFILE_TASKS[@]}")

task_port() {
  case "$1" in
    math) echo "${MOPD_TEACHER_MATH_PORT}" ;;
    code) echo "${MOPD_TEACHER_CODE_PORT}" ;;
    if) echo "${MOPD_TEACHER_IF_PORT}" ;;
    science) echo "${MOPD_TEACHER_SCIENCE_PORT}" ;;
  esac
}

task_model() {
  case "$1" in
    math|code|if|science) echo "${MOPD_TEACHER_HF_ROOT}/$1" ;;
  esac
}

task_gpu() {
  case "$1" in
    math) echo "${MOPD_TEACHER_MATH_GPU}" ;;
    code) echo "${MOPD_TEACHER_CODE_GPU}" ;;
    if) echo "${MOPD_TEACHER_IF_GPU}" ;;
    science) echo "${MOPD_TEACHER_SCIENCE_GPU}" ;;
  esac
}

task_memory_fraction() {
  if [[ "${MOPD_PROFILE}" == smollm3 ]]; then
    echo "${TEACHER_3B_MEM_FRACTION:-0.20}"
    return
  fi
  if [[ "${MOPD_HARDWARE_PROFILE}" == dual-48gb-tp2 ]]; then
    case "$1" in
      math) echo "${TEACHER_MATH_MEM_FRACTION:-${TEACHER_1P7B_MEM_FRACTION:-0.11}}" ;;
      code) echo "${TEACHER_CODE_MEM_FRACTION:-${TEACHER_1P7B_MEM_FRACTION:-0.11}}" ;;
      if) echo "${TEACHER_IF_MEM_FRACTION:-${TEACHER_1P7B_MEM_FRACTION:-0.14}}" ;;
      science) echo "${TEACHER_SCIENCE_MEM_FRACTION:-${TEACHER_1P7B_MEM_FRACTION:-0.14}}" ;;
    esac
    return
  fi
  case "$1" in
    math|code|if|science) echo "${TEACHER_1P7B_MEM_FRACTION:-0.09}" ;;
  esac
}

healthy() {
  curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$(task_port "$1")/health_generate" >/dev/null 2>&1
}

start_teacher() {
  local task="$1" port model fraction pid_file log_file pid deadline
  port="$(task_port "${task}")"
  model="$(task_model "${task}")"
  fraction="$(task_memory_fraction "${task}")"
  pid_file="${SERVER_DIR}/${task}.pid"
  log_file="${SERVER_DIR}/${task}.log"
  if healthy "${task}"; then
    [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null || {
      echo "Port ${port} has an unmanaged healthy server." >&2
      return 1
    }
    echo "${task} teacher already READY on :${port}."
    return
  fi
  if [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
    echo "${task} teacher PID $(<"${pid_file}") is alive but unhealthy; inspect ${log_file}." >&2
    return 1
  fi
  local command=(
    python3 -m sglang.launch_server
    --model-path "${model}"
    --host 0.0.0.0 --port "${port}" --tp 1
    --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
    --mem-fraction-static "${fraction}"
    --max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-8}"
    --max-total-tokens "${TEACHER_MAX_TOTAL_TOKENS:-$((MOPD_CONTEXT_LENGTH < 34816 ? MOPD_CONTEXT_LENGTH : 34816))}"
    --context-length "${MOPD_CONTEXT_LENGTH}"
  )
  if [[ "${TEACHER_DISABLE_CUDA_GRAPH:-1}" == 1 ]]; then command+=(--disable-cuda-graph); fi
  # Dense input log-probabilities otherwise materialize several full-vocabulary
  # tensors per prefill. Each resident teacher keeps its own allocator cache.
  SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1 \
  SGLANG_LOGITS_PROCESSER_CHUNK_SIZE="${TEACHER_LOGPROBS_CHUNK_SIZE:-512}" \
  CUDA_VISIBLE_DEVICES="$(task_gpu "${task}")" setsid "${command[@]}" >"${log_file}" 2>&1 &
  pid=$!
  echo "${pid}" >"${pid_file}"
  deadline=$((SECONDS + ${TEACHER_START_TIMEOUT:-900}))
  until healthy "${task}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 80 "${log_file}" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      tail -n 80 "${log_file}" >&2 || true
      echo "Timed out waiting for ${task} teacher." >&2
      return 1
    fi
    sleep 5
  done
  echo "${task} teacher READY on :${port}."
}

start_pool() {
  local training_gpu_args=() training_gpu
  IFS=',' read -r -a training_gpus <<<"${MOPD_TRAIN_GPUS}"
  for training_gpu in "${training_gpus[@]}"; do
    training_gpu_args+=(--training-gpu "${training_gpu}")
  done
  local service_gpu_args=() task
  for task in "${TASKS[@]}"; do service_gpu_args+=(--service-gpu "$(task_gpu "${task}")"); done
  mkdir -p "${SERVER_DIR}"
  for task in "${TASKS[@]}"; do start_teacher "${task}"; done
}

stop_pool() {
  local task pid_file pid
  for task in "${TASKS[@]}"; do
    pid_file="${SERVER_DIR}/${task}.pid"
    if [[ -f "${pid_file}" ]]; then
      pid="$(<"${pid_file}")"
      if kill -0 "${pid}" 2>/dev/null; then
        kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
      fi
      rm -f "${pid_file}"
    fi
  done
  echo "Resident teacher pool stopped."
}

status_pool() {
  local task failures=0
  for task in "${TASKS[@]}"; do
    if healthy "${task}"; then
      echo "${task}: READY :$(task_port "${task}") gpu=$(task_gpu "${task}") model=$(task_model "${task}")"
    else
      echo "${task}: DOWN :$(task_port "${task}")" >&2
      failures=1
    fi
  done
  return "${failures}"
}

case "${ACTION}" in
  start) start_pool ;;
  stop) stop_pool ;;
  status) status_pool ;;
  *) echo "Usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
