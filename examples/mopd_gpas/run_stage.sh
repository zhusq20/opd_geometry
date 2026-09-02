#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-}"
TARGET="${2:-all}"

case "${STAGE}" in
  fetch-assets) bash "${EXAMPLE_DIR}/fetch_assets.sh" ;;
  prepare)
    python3 "${EXAMPLE_DIR}/prepare_mopd.py" --seed 42
    ;;
  preflight)
    python3 "${EXAMPLE_DIR}/prepare_mopd.py" --seed 42
    bash "${EXAMPLE_DIR}/convert_teachers.sh" --verify-only
    DRY_RUN=1 bash "${EXAMPLE_DIR}/run_stage.sh" dry-run
    ;;
  verify-teachers) bash "${EXAMPLE_DIR}/convert_teachers.sh" --verify-only ;;
  convert-teachers) bash "${EXAMPLE_DIR}/convert_teachers.sh" ;;
  start-teacher) bash "${EXAMPLE_DIR}/serve_teachers.sh" start ;;
  status-teacher) bash "${EXAMPLE_DIR}/serve_teachers.sh" status ;;
  stop-teacher) bash "${EXAMPLE_DIR}/serve_teachers.sh" stop ;;
  warm) bash "${EXAMPLE_DIR}/run_warm_start.sh" ;;
  train4) bash "${EXAMPLE_DIR}/run_mopd_4gpu.sh" ;;
  resume) bash "${EXAMPLE_DIR}/resume_mopd.sh" "${TARGET}" ;;
  train)
    if [[ "${TARGET}" == all ]]; then
      bash "${EXAMPLE_DIR}/run_mopd_matrix.sh"
    else
      bash "${EXAMPLE_DIR}/run_mopd.sh" "${TARGET}"
    fi
    ;;
  bank)
    if [[ "${TARGET}" == all ]]; then
      bash "${EXAMPLE_DIR}/run_frozen_bank_matrix.sh"
    else
      bash "${EXAMPLE_DIR}/run_frozen_bank.sh" "${TARGET}"
    fi
    ;;
  capability)
    if [[ "${TARGET}" == all ]]; then
      bash "${EXAMPLE_DIR}/run_capability_matrix.sh"
    else
      bash "${EXAMPLE_DIR}/run_capability_eval.sh" "${TARGET}"
    fi
    ;;
  package) bash "${EXAMPLE_DIR}/validate_and_package_run.sh" "${TARGET}" ;;
  analyze) bash "${EXAMPLE_DIR}/analyze_all.sh" ;;
  dry-run)
    export DRY_RUN=1
    bash "${EXAMPLE_DIR}/run_warm_start.sh"
    for config in \
      uniform_k1_conventional uniform_k1_taskwise gpas_k1_taskwise cost_gpas_k1_taskwise \
      uniform_k2_taskwise cost_gpas_k2_taskwise all_k4_taskwise all_k4_conventional; do
      bash "${EXAMPLE_DIR}/run_mopd.sh" "${config}"
      bash "${EXAMPLE_DIR}/run_capability_eval.sh" "${config}"
    done
    for checkpoint in warm middle late; do
      bash "${EXAMPLE_DIR}/run_frozen_bank.sh" "${checkpoint}"
    done
    ;;
  *)
    echo "Usage: $0 {fetch-assets|prepare|preflight|verify-teachers|convert-teachers|start-teacher|status-teacher|stop-teacher|warm|train [CONFIG|all]|train4|resume CONFIG|bank [warm|middle|late|all]|capability [CONFIG|all]|package CONFIG|analyze|dry-run}" >&2
    exit 2
    ;;
esac
