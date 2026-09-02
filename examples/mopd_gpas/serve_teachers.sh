#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MOPD_TEACHER_HF_ROOT="${MOPD_TEACHER_HF_ROOT:-${EXAMPLE_DIR}/generated/teachers_hf}"
export MOPD_TEACHER_GPU="${MOPD_TEACHER_GPU:-4}"
export MOPD_TEACHER_PORT="${MOPD_TEACHER_PORT:-31001}"
export MOPD_TEACHER_SLOT_STATE="${MOPD_TEACHER_SLOT_STATE:-${EXAMPLE_DIR}/generated/teacher_slot/state.json}"
SLOT_DIR="$(dirname -- "${MOPD_TEACHER_SLOT_STATE}")"
PID_FILE="${SLOT_DIR}/server.pid"
LOG_FILE="${SLOT_DIR}/server.log"
PORT="${MOPD_TEACHER_PORT}"
ACTION="${1:-status}"

mkdir -p "${SLOT_DIR}"

healthy() {
  curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1
}

reported_resident() {
  curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:${PORT}/get_weight_version" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["weight_version"])'
}

stored_resident() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["resident"])' "${MOPD_TEACHER_SLOT_STATE}"
}

validate_assets() {
  bash "${EXAMPLE_DIR}/convert_teachers.sh" --verify-only
}

validate_gpu_layout() {
  local train_value="${MOPD_TRAIN_CUDA_VISIBLE_DEVICES:-3}"
  local -a train_gpus
  IFS=, read -r -a train_gpus <<< "${train_value}"
  [[ "${#train_gpus[@]}" == 1 && "${MOPD_TEACHER_GPU}" != *,* ]] || {
    echo "Expected one student GPU and one teacher GPU." >&2; return 2;
  }
  local gpu
  for gpu in "${train_gpus[@]}"; do
    [[ "${gpu}" != "${MOPD_TEACHER_GPU}" ]] || {
      echo "Teacher GPU ${MOPD_TEACHER_GPU} overlaps student GPUs ${train_value}." >&2; return 2;
    }
  done
  python3 "${EXAMPLE_DIR}/verify_hardware.py" \
    --gpu-ids "${train_value},${MOPD_TEACHER_GPU}"
}

write_state() {
  python3 - "${MOPD_TEACHER_SLOT_STATE}" "${1}" "${2}" "${MOPD_TEACHER_HF_ROOT}/${1}" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps({
    "resident": sys.argv[2],
    "server_pid": int(sys.argv[3]),
    "model_path": str(pathlib.Path(sys.argv[4]).resolve()),
    "updated_unix": time.time(),
}, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

start_slot() {
  validate_assets
  validate_gpu_layout
  if healthy; then
    if [[ -f "${PID_FILE}" && -f "${MOPD_TEACHER_SLOT_STATE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
      [[ "$(reported_resident)" == "$(stored_resident)" ]] || {
        echo "Teacher server and residency state disagree." >&2; return 1;
      }
      echo "Teacher slot already healthy on :${PORT}."
      return
    fi
    echo "Port ${PORT} has a healthy but unmanaged server; stop it before starting this experiment." >&2
    return 1
  fi
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
    echo "Teacher-slot PID $(<"${PID_FILE}") is alive but unhealthy; inspect ${LOG_FILE}." >&2
    return 1
  fi
  local command=(
    python3 -m sglang.launch_server
    --model-path "${MOPD_TEACHER_HF_ROOT}/math"
    --weight-version math
    --host 0.0.0.0 --port "${PORT}" --tp 1
    --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
    --mem-fraction-static "${TEACHER_MEM_FRACTION:-0.8}"
    --max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-8}"
    --max-total-tokens "${TEACHER_MAX_TOTAL_TOKENS:-32768}"
  )
  if [[ "${TEACHER_DISABLE_CUDA_GRAPH:-0}" == 1 ]]; then command+=(--disable-cuda-graph); fi
  CUDA_VISIBLE_DEVICES="${MOPD_TEACHER_GPU}" setsid "${command[@]}" >"${LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${PID_FILE}"
  local deadline=$((SECONDS + ${TEACHER_START_TIMEOUT:-900}))
  until healthy; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 80 "${LOG_FILE}" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      tail -n 80 "${LOG_FILE}" >&2 || true
      echo "Timed out waiting for teacher slot." >&2
      return 1
    fi
    sleep 5
  done
  write_state math "${pid}"
  echo "Teacher slot READY: math resident, GPU=${MOPD_TEACHER_GPU}, port=${PORT}."
}

stop_slot() {
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(<"${PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
      for _ in $(seq 1 30); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
    fi
  fi
  rm -f "${PID_FILE}" "${MOPD_TEACHER_SLOT_STATE}"
  echo "Teacher slot stopped."
}

switch_slot() {
  local task="$1"
  case "${task}" in math|code|if|science) ;; *) echo "Unknown task: ${task}" >&2; return 2 ;; esac
  healthy || { echo "Teacher slot must be started before switching." >&2; return 1; }
  curl --noproxy '*' -fsS --max-time "${TEACHER_SWITCH_TIMEOUT:-900}" \
    -H 'Content-Type: application/json' \
    -d "{\"model_path\":\"${MOPD_TEACHER_HF_ROOT}/${task}\",\"load_format\":\"auto\",\"weight_version\":\"${task}\"}" \
    "http://127.0.0.1:${PORT}/update_weights_from_disk" >/dev/null
  [[ "$(reported_resident)" == "${task}" ]] || {
    echo "Teacher server did not activate ${task}." >&2; return 1;
  }
  local pid
  pid="$(<"${PID_FILE}")"
  write_state "${task}" "${pid}"
  echo "Teacher slot switched to ${task}."
}

case "${ACTION}" in
  start) start_slot ;;
  stop) stop_slot ;;
  switch) [[ $# == 2 ]] || { echo "Usage: $0 switch TASK" >&2; exit 2; }; switch_slot "$2" ;;
  status)
    if healthy && [[ -f "${MOPD_TEACHER_SLOT_STATE}" ]]; then
      resident="$(stored_resident)"
      [[ "$(reported_resident)" == "${resident}" ]] || {
        echo "Teacher server and residency state disagree." >&2; exit 1;
      }
      echo "Teacher slot READY on :${PORT}; resident=${resident}; GPU=${MOPD_TEACHER_GPU}."
    else
      echo "Teacher slot DOWN on :${PORT}." >&2
      exit 1
    fi
    ;;
  *) echo "Usage: $0 {start|stop|status|switch TASK}" >&2; exit 2 ;;
esac
