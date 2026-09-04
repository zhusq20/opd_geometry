#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
ASSET_ROOT="${MOPD_ASSET_ROOT:-${SLIME_ROOT}/local/mopd_assets}"
MODEL_REPO="${MOPD_MODEL_REPO:-zsqzz/mopd-gpas-64k-models}"
MODEL_REVISION="${MOPD_MODEL_REVISION:-46c307ad13f274d8db7c1df949e4df8359ae6b32}"
DATA_REPO="${MOPD_DATA_REPO:-zsqzz/mopd-gpas-64k-data}"
DATA_REVISION="${MOPD_DATA_REVISION:-e3ead6f98b7089e0def39516cd06cf711b10b1ec}"
QWEN3_4B_REVISION="${MOPD_QWEN3_4B_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}"

command -v hf >/dev/null || {
  echo "The Hugging Face CLI is required. Install it with: pip install -U huggingface_hub" >&2
  exit 2
}

mkdir -p "${ASSET_ROOT}/models" "${ASSET_ROOT}/data"
hf download "${MODEL_REPO}" \
  --repo-type model \
  --revision "${MODEL_REVISION}" \
  --local-dir "${ASSET_ROOT}/models"
hf download Qwen/Qwen3-4B \
  --repo-type model \
  --revision "${QWEN3_4B_REVISION}" \
  --local-dir "${ASSET_ROOT}/models/qwen3-4b"
hf download "${DATA_REPO}" \
  --repo-type dataset \
  --revision "${DATA_REVISION}" \
  --local-dir "${ASSET_ROOT}/data"

echo "Assets downloaded to ${ASSET_ROOT}."
echo "Copy configs/site.example.env to local/mopd.env, then source local/mopd.env."
