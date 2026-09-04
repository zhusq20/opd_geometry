#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-}"
TARGET="${2:-all}"

case "${STAGE}" in
  fetch-assets) bash "${EXAMPLE_DIR}/fetch_assets.sh" ;;
  prepare-heldout) python3 "${EXAMPLE_DIR}/prepare_mopd.py" --heldout-only ;;
  measure-initial)
    CUDA_VISIBLE_DEVICES="${MOPD_INFERENCE_GPU:-1}" python3 "${EXAMPLE_DIR}/measure_initial_kl.py"
    ;;
  prepare)
    python3 "${EXAMPLE_DIR}/prepare_mopd.py"
    ;;
  preflight)
    python3 "${EXAMPLE_DIR}/prepare_mopd.py"
    bash "${EXAMPLE_DIR}/convert_teachers.sh" --verify-only
    DRY_RUN=1 bash "${EXAMPLE_DIR}/run_stage.sh" dry-run
    ;;
  verify-teachers) bash "${EXAMPLE_DIR}/convert_teachers.sh" --verify-only ;;
  convert-teachers) bash "${EXAMPLE_DIR}/convert_teachers.sh" ;;
  start-teacher) bash "${EXAMPLE_DIR}/serve_teachers.sh" start ;;
  status-teacher) bash "${EXAMPLE_DIR}/serve_teachers.sh" status ;;
  stop-teacher) bash "${EXAMPLE_DIR}/serve_teachers.sh" stop ;;
  resume) bash "${EXAMPLE_DIR}/resume_mopd.sh" "${TARGET}" ;;
  quick-smoke) bash "${EXAMPLE_DIR}/run_mopd.sh" gpas-quick-smoke ;;
  smoke) bash "${EXAMPLE_DIR}/run_mopd.sh" gpas-smoke ;;
  train)
    if [[ "${TARGET}" == all ]]; then
      bash "${EXAMPLE_DIR}/run_mopd_matrix.sh"
    else
      bash "${EXAMPLE_DIR}/run_mopd.sh" "${TARGET}"
    fi
    ;;
  baselines) bash "${EXAMPLE_DIR}/run_paper_baselines.sh" ;;
  open-full-fetch)
    shift
    bash "${EXAMPLE_DIR}/open_mopd_full/fetch.sh" "$@"
    ;;
  open-full-train)
    shift
    bash "${EXAMPLE_DIR}/open_mopd_full/train.sh" "$@"
    ;;
  open-full-eval)
    shift
    bash "${EXAMPLE_DIR}/open_mopd_full/evaluate.sh" "$@"
    ;;
  capability)
    if [[ "${TARGET}" == all ]]; then
      bash "${EXAMPLE_DIR}/run_capability_matrix.sh"
    else
      bash "${EXAMPLE_DIR}/run_capability_eval.sh" "${TARGET}"
    fi
    ;;
  variance)
    if [[ "${TARGET}" == all ]]; then
      bash "${EXAMPLE_DIR}/run_heldout_variance_matrix.sh"
    else
      bash "${EXAMPLE_DIR}/run_heldout_variance.sh" "${TARGET}"
    fi
    ;;
  package) bash "${EXAMPLE_DIR}/validate_and_package_run.sh" "${TARGET}" ;;
  analyze) bash "${EXAMPLE_DIR}/analyze_all.sh" ;;
  dry-run)
    export DRY_RUN=1
    for config in uniform gpas cost_gpas raw_noise loss_gap std_mopd d3_mopd open_mopd; do
      bash "${EXAMPLE_DIR}/run_mopd.sh" "${config}"
    done
    ;;
  *)
    echo "Usage: $0 {fetch-assets|convert-teachers|prepare-heldout|measure-initial|prepare|preflight|start-teacher|status-teacher|stop-teacher|quick-smoke|smoke|train [CONFIG|all]|baselines|open-full-fetch [source|assets|all]|open-full-train [--dry-run|--run]|open-full-eval [--dry-run|--run]|resume CONFIG|capability [TARGET|all]|variance [50|250|500|all]|package CONFIG|analyze|dry-run}" >&2
    exit 2
    ;;
esac
