#!/usr/bin/env bash
# Run all eight seed-42 main cells concurrently on two isolated student/teacher pairs.
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
OUTPUT_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_64k_v3}"
WARM_DIR="${MOPD_WARM_DIR:-${OUTPUT_ROOT}/warm_start-seed42}"

[[ "${DRY_RUN:-0}" == 1 || -f "${WARM_DIR}/checkpoints/rollout/mopd_dataset_state_dict_7.pt" ]] || {
  echo "Missing shared seed-42 warm checkpoint: ${WARM_DIR}" >&2
  exit 2
}
unset RAY_ADDRESS
export MOPD_OUTPUT_ROOT="${OUTPUT_ROOT}"
export MOPD_WARM_DIR="${WARM_DIR}"
export MOPD_SEED=42
export USE_WANDB="${USE_WANDB:-1}"

start_teacher() {
  local student="$1" teacher="$2" port="$3" state="$4"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "dry-run: teacher student=${student} teacher=${teacher} port=${port} state=${state}"
    return
  fi
  MOPD_TRAIN_CUDA_VISIBLE_DEVICES="${student}" \
  MOPD_TEACHER_GPU="${teacher}" \
  MOPD_TEACHER_PORT="${port}" \
  MOPD_TEACHER_SLOT_STATE="${state}" \
    bash "${EXAMPLE_DIR}/serve_teachers.sh" start
}

run_pair() {
  local pair="$1" student="$2" teacher="$3" teacher_port="$4" state="$5"
  local dashboard_port="$6" gcs_port="$7"
  shift 7
  local config
  for config in "$@"; do
    echo "[$(date -u +%FT%TZ)] pair=${pair} starting ${config}"
    MOPD_TRAIN_CUDA_VISIBLE_DEVICES="${student}" \
    MOPD_TEACHER_GPU="${teacher}" \
    MOPD_TEACHER_PORT="${teacher_port}" \
    MOPD_TEACHER_SLOT_STATE="${state}" \
    RAY_DASHBOARD_PORT="${dashboard_port}" \
    RAY_GCS_PORT="${gcs_port}" \
      bash "${EXAMPLE_DIR}/run_mopd.sh" "${config}"
    echo "[$(date -u +%FT%TZ)] pair=${pair} completed ${config}"
  done
}

PAIR_A_STATE="${MOPD_PAIR_A_STATE:-${EXAMPLE_DIR}/generated/teacher_slot/state.json}"
PAIR_B_STATE="${MOPD_PAIR_B_STATE:-${EXAMPLE_DIR}/generated/teacher_slot_pair_b/state.json}"
start_teacher "${MOPD_PAIR_A_STUDENT:-3}" "${MOPD_PAIR_A_TEACHER:-4}" \
  "${MOPD_PAIR_A_TEACHER_PORT:-31001}" "${PAIR_A_STATE}"
start_teacher "${MOPD_PAIR_B_STUDENT:-5}" "${MOPD_PAIR_B_TEACHER:-6}" \
  "${MOPD_PAIR_B_TEACHER_PORT:-31002}" "${PAIR_B_STATE}"

run_pair A "${MOPD_PAIR_A_STUDENT:-3}" "${MOPD_PAIR_A_TEACHER:-4}" \
  "${MOPD_PAIR_A_TEACHER_PORT:-31001}" "${PAIR_A_STATE}" 8265 6380 \
  uniform_k1_conventional cost_gpas_k1_taskwise uniform_k2_taskwise all_k4_taskwise &
PID_A=$!
run_pair B "${MOPD_PAIR_B_STUDENT:-5}" "${MOPD_PAIR_B_TEACHER:-6}" \
  "${MOPD_PAIR_B_TEACHER_PORT:-31002}" "${PAIR_B_STATE}" 8266 6381 \
  uniform_k1_taskwise gpas_k1_taskwise cost_gpas_k2_taskwise all_k4_conventional &
PID_B=$!

STATUS=0
wait "${PID_A}" || STATUS=$?
wait "${PID_B}" || STATUS=$?
if [[ "${STATUS}" != 0 ]]; then
  echo "At least one four-GPU worker failed; inspect the run provenance and resume from its latest checkpoint." >&2
  exit "${STATUS}"
fi
echo "All eight seed-42 64k main runs completed."
