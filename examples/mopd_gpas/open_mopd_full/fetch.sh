#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

OFFICIAL_URL="https://github.com/BytedTsinghua-SIA/Open-MOPD.git"
OFFICIAL_REVISION="4809a96cf85a869106ff0ff3f37d0a51e12010ae"
FULL_ROOT="${OPEN_MOPD_FULL_ROOT:-${REPO_ROOT}/local/open_mopd_full}"
SOURCE_ROOT="${OPEN_MOPD_SOURCE_ROOT:-${FULL_ROOT}/source}"
ASSET_ROOT="${OPEN_MOPD_ASSET_ROOT:-${FULL_ROOT}/assets}"
ACTION="${1:-source}"

usage() {
  echo "Usage: $0 {source|assets|all}"
  echo "  source  clone and pin the official Open-MOPD implementation (default)"
  echo "  assets  download the released final-stage student, three teachers, final model, and data"
  echo "  all     fetch source and assets"
  echo
  echo "Optional roots: OPEN_MOPD_FULL_ROOT, OPEN_MOPD_SOURCE_ROOT, OPEN_MOPD_ASSET_ROOT"
}

die() {
  echo "[open-mopd-full] error: $*" >&2
  exit 2
}

fetch_source() {
  command -v git >/dev/null 2>&1 || die "git is required"
  mkdir -p "$(dirname -- "${SOURCE_ROOT}")"
  if [[ -d "${SOURCE_ROOT}/.git" ]]; then
    local remote actual
    remote="$(git -C "${SOURCE_ROOT}" remote get-url origin 2>/dev/null || true)"
    [[ "${remote}" == "${OFFICIAL_URL}" ]] || die "existing source has unexpected origin: ${remote:-<missing>}"
    actual="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
    [[ "${actual}" == "${OFFICIAL_REVISION}" ]] || die "existing source is at ${actual}; use an empty OPEN_MOPD_SOURCE_ROOT for the pinned revision"
    echo "[open-mopd-full] source already pinned: ${SOURCE_ROOT} @ ${actual}"
    return
  fi
  if [[ -e "${SOURCE_ROOT}" ]]; then
    [[ -d "${SOURCE_ROOT}" && -z "$(find "${SOURCE_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      || die "source target exists and is not an empty directory: ${SOURCE_ROOT}"
  fi
  git clone --no-checkout "${OFFICIAL_URL}" "${SOURCE_ROOT}"
  git -C "${SOURCE_ROOT}" checkout --detach "${OFFICIAL_REVISION}"
  local actual
  actual="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
  [[ "${actual}" == "${OFFICIAL_REVISION}" ]] || die "revision verification failed: ${actual}"
  echo "[open-mopd-full] source pinned: ${SOURCE_ROOT} @ ${actual}"
}

download_model() {
  local repo_id="$1"
  local destination="$2"
  hf download "${repo_id}" --local-dir "${destination}"
}

fetch_assets() {
  command -v hf >/dev/null 2>&1 || die "Hugging Face CLI 'hf' is required; install huggingface_hub first"
  mkdir -p "${ASSET_ROOT}/models/teachers" "${ASSET_ROOT}/data"
  download_model \
    "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT" \
    "${ASSET_ROOT}/models/mixsft"
  download_model \
    "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math" \
    "${ASSET_ROOT}/models/teachers/math"
  download_model \
    "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Code" \
    "${ASSET_ROOT}/models/teachers/code"
  download_model \
    "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-IF" \
    "${ASSET_ROOT}/models/teachers/if"
  download_model \
    "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-Final" \
    "${ASSET_ROOT}/models/final"
  hf download \
    "BytedTsinghua-SIA/Open-MOPD-Data" \
    --repo-type dataset \
    --include "rl_prompt_mix/train.parquet" "eval/**/*.parquet" \
    --local-dir "${ASSET_ROOT}/data"
  echo "[open-mopd-full] released assets ready: ${ASSET_ROOT}"
}

case "${ACTION}" in
  -h|--help|help) usage ;;
  source) fetch_source ;;
  assets) fetch_assets ;;
  all)
    fetch_source
    fetch_assets
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
