#!/usr/bin/env bash
# Source once from each entry point; site-specific environment remains authoritative.
_MOPD_PROFILE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_MOPD_PROFILE_ROOT="$(cd -- "${_MOPD_PROFILE_DIR}/../.." && pwd)"
export MOPD_PROFILE="${MOPD_PROFILE:-qwen3}"
case "${MOPD_PROFILE}" in
  qwen3) MOPD_PROFILE_TASKS=(math code if science); MOPD_MODEL_CONFIG=qwen3-1.7B.sh; export MOPD_CONTEXT_LENGTH=32768 ;;
  smollm3) MOPD_PROFILE_TASKS=(math code if); MOPD_MODEL_CONFIG=smollm3-3B.sh; export MOPD_CONTEXT_LENGTH=65536 ;;
  *) echo "Unknown MOPD_PROFILE: ${MOPD_PROFILE}" >&2; return 2 ;;
esac
_MOPD_DIRECTORY_NAME="${MOPD_PROFILE}"
[[ "${MOPD_PROFILE}" != smollm3 ]] || _MOPD_DIRECTORY_NAME=smollm3_mixsft
export MOPD_ASSET_ROOT="${MOPD_ASSET_ROOT:-${_MOPD_PROFILE_ROOT}/local/mopd_${_MOPD_DIRECTORY_NAME}_assets}"
export MOPD_HF_CHECKPOINT="${MOPD_HF_CHECKPOINT:-${MOPD_ASSET_ROOT}/models/student}"
export MOPD_BASE_MEGATRON="${MOPD_BASE_MEGATRON:-${MOPD_ASSET_ROOT}/models/student_torch_dist}"
export MOPD_TEACHER_HF_ROOT="${MOPD_TEACHER_HF_ROOT:-${MOPD_ASSET_ROOT}/models/teachers_hf}"
if [[ "${MOPD_PROFILE}" == qwen3 ]]; then
  export MOPD_DATA_ROOT="${MOPD_DATA_ROOT:-${MOPD_ASSET_ROOT}/data/m2rl}"
else
  export MOPD_DATA_ROOT="${MOPD_DATA_ROOT:-${MOPD_ASSET_ROOT}/data}"
fi
export MOPD_GENERATED_DIR="${MOPD_GENERATED_DIR:-${_MOPD_PROFILE_ROOT}/local/mopd_${_MOPD_DIRECTORY_NAME}_generated}"
export MOPD_OUTPUT_ROOT="${MOPD_OUTPUT_ROOT:-${_MOPD_PROFILE_ROOT}/outputs/mopd_${_MOPD_DIRECTORY_NAME}}"
export MOPD_TEACHER_SERVER_DIR="${MOPD_TEACHER_SERVER_DIR:-${_MOPD_PROFILE_ROOT}/local/mopd_${MOPD_PROFILE}_teachers}"
export MOPD_TEACHER_ROUTER_CONFIG="${MOPD_TEACHER_ROUTER_CONFIG:-${MOPD_GENERATED_DIR}/teacher_router.yaml}"
export MOPD_CAPABILITY_EVAL_CONFIG="${MOPD_CAPABILITY_EVAL_CONFIG:-${MOPD_GENERATED_DIR}/capability_eval.yaml}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-iclr2027-mopd-dynamics}"
unset _MOPD_PROFILE_DIR _MOPD_PROFILE_ROOT _MOPD_DIRECTORY_NAME
