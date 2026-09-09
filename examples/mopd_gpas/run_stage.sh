#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${EXAMPLE_DIR}/_profile.sh"
export PYTHONPATH="$(cd -- "${EXAMPLE_DIR}/../.." && pwd):${PYTHONPATH:-}"
STAGE="${1:-}"
TARGET="${2:-all}"

case "${STAGE}" in
  fetch-assets) bash "${EXAMPLE_DIR}/fetch_assets.sh" ;;
  prepare-heldout) python3 "${EXAMPLE_DIR}/prepare_mopd.py" --heldout-only ;;
  measure-initial)
    CUDA_VISIBLE_DEVICES="${MOPD_INFERENCE_GPU:-1}" python3 "${EXAMPLE_DIR}/measure_initial_kl.py"
    ;;
  prepare)
    python3 "${EXAMPLE_DIR}/prepare_paper.py"
    ;;
  preflight)
    python3 "${EXAMPLE_DIR}/prepare_mopd.py"
    bash "${EXAMPLE_DIR}/convert_teachers.sh" --verify-only
    DRY_RUN=1 bash "${EXAMPLE_DIR}/run_stage.sh" dry-run
    ;;
  verify-teachers) bash "${EXAMPLE_DIR}/convert_teachers.sh" --verify-only ;;
  convert-teachers)
    bash "${EXAMPLE_DIR}/convert_teachers.sh" --verify-only
    STUDENT_MEGATRON="${MOPD_BASE_MEGATRON:-/workspace/dev/checkpoints/Qwen3-1.7B-Base_torch_dist}"
    if [[ ! -f "${STUDENT_MEGATRON}/latest_checkpointed_iteration.txt" ]]; then
      SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
      source "${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh"
      CUDA_VISIBLE_DEVICES="${MOPD_TRAIN_GPU:-0}" \
      PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH:-/root/Megatron-LM}${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${MOPD_HF_CHECKPOINT:-/workspace/dev/checkpoints/Qwen3-1.7B-Base}" \
        --save "${STUDENT_MEGATRON}" --bf16
    fi
    ;;
  start-teacher) bash "${EXAMPLE_DIR}/serve_teachers.sh" start ;;
  status-teacher) bash "${EXAMPLE_DIR}/serve_teachers.sh" status ;;
  stop-teacher) bash "${EXAMPLE_DIR}/serve_teachers.sh" stop ;;
  resume) bash "${EXAMPLE_DIR}/resume_mopd.sh" "${TARGET}" ;;
  quick-smoke) bash "${EXAMPLE_DIR}/run_mopd.sh" gpas-quick-smoke ;;
  smoke) bash "${EXAMPLE_DIR}/run_mopd.sh" gpas-smoke ;;
  train)
    if [[ "${TARGET}" == all ]]; then
      for config in s-pg s-tk m-pg m-tk-dr m-tk-dt m-tk-gt; do
        bash "${EXAMPLE_DIR}/run_mopd.sh" "${config}"
      done
    else
      bash "${EXAMPLE_DIR}/run_mopd.sh" "${TARGET}"
    fi
    ;;
  baselines)
    for config in s-pg s-tk m-pg m-tk-dr m-tk-dt m-tk-gt; do
      bash "${EXAMPLE_DIR}/run_mopd.sh" "${config}"
    done ;;
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
      bash "${EXAMPLE_DIR}/_evaluate_paper.sh" initial_student
      for task in "${MOPD_PROFILE_TASKS[@]}"; do bash "${EXAMPLE_DIR}/_evaluate_paper.sh" "teacher_${task}"; done
    else
      bash "${EXAMPLE_DIR}/_evaluate_paper.sh" "${TARGET}"
    fi
    ;;
  variance|mechanism)
    [[ "${TARGET}" == all || "${TARGET}" == 250 ]] || { echo "The core diagnostic uses only Uniform/250." >&2; exit 2; }
    bash "${EXAMPLE_DIR}/run_common_checkpoint.sh"
    ;;
  package) bash "${EXAMPLE_DIR}/validate_and_package_run.sh" "${TARGET}" ;;
  analyze) bash "${EXAMPLE_DIR}/analyze_all.sh" ;;
  dry-run)
    export DRY_RUN=1
    for config in s-pg s-tk m-pg m-tk-dr m-tk-dt m-tk-gt; do
      bash "${EXAMPLE_DIR}/run_mopd.sh" "${config}"
    done
    ;;
  *)
    echo "Usage: $0 {fetch-assets|convert-teachers|prepare|preflight|start-teacher|status-teacher|stop-teacher|train [RUN|all]|baselines|resume RUN|capability [TARGET|all]|mechanism|package RUN|analyze|dry-run}" >&2
    exit 2
    ;;
esac
