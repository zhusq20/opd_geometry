#!/usr/bin/env bash
# Download the official M2RL/Nemotron blend and materialize only the four
# non-Agent domains used by the optimizer-geometry experiments.

set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"

DATA_ROOT="${M2RL_DATA_ROOT:-${SLIME_DIR}/data/m2rl}"
SOURCE_DIR="${M2RL_SOURCE_DIR:-${DATA_ROOT}/raw/Nemotron-3-Nano-RL-Training-Blend}"
WORK_DIR="${M2RL_WORK_DIR:-${DATA_ROOT}/work}"
OUTPUT_DIR="${M2RL_OUTPUT_DIR:-${DATA_ROOT}/train}"
SOURCE_DATASET="${M2RL_SOURCE_DATASET:-nvidia/Nemotron-3-Nano-RL-Training-Blend}"
SOURCE_JSONL="${SOURCE_DIR}/train.jsonl"
RESTORE_SCRIPT="${SOURCE_DIR}/create_nanov3_jsonl.py"
COMPLETE_JSONL="${WORK_DIR}/train_complete.jsonl"
MANIFEST_NAME="${M2RL_MANIFEST_NAME:-multitask_manifest.yaml}"
MANIFEST_PATH="${OUTPUT_DIR}/${MANIFEST_NAME}"

SEED="${SEED:-42}"
SAMPLING="${M2RL_SAMPLING:-uniform}"
SAMPLING_UNIT="${M2RL_SAMPLING_UNIT:-batch}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
# The source blend is monolithic and temporarily contains Agent rows. Remove it
# after successful conversion so the retained dataset is strictly four-task.
CLEAN_SOURCE_BLEND="${CLEAN_SOURCE_BLEND:-1}"

case "${FORCE_REBUILD}:${CLEAN_SOURCE_BLEND}" in
  [01]:[01]) ;;
  *) echo "FORCE_REBUILD and CLEAN_SOURCE_BLEND must each be 0 or 1." >&2; exit 2 ;;
esac

if [[ "${FORCE_REBUILD}" == "0" && -s "${MANIFEST_PATH}" && -s "${OUTPUT_DIR}/dataset_info.json" ]] &&
   [[ -s "${OUTPUT_DIR}/math.jsonl" && -s "${OUTPUT_DIR}/science.jsonl" ]] &&
   [[ -s "${OUTPUT_DIR}/if.jsonl" && -s "${OUTPUT_DIR}/code.jsonl" ]] &&
   [[ ! -e "${OUTPUT_DIR}/agent.jsonl" ]] &&
   ! grep -Eq '(^|[[:space:]])(name|rm_type):[[:space:]]*(agent|workbench)([[:space:]]|$)' "${MANIFEST_PATH}"; then
  if [[ "${CLEAN_SOURCE_BLEND}" == "1" ]]; then
    rm -f -- "${SOURCE_JSONL}" "${COMPLETE_JSONL}"
  fi
  echo "M2RL four-task data is already prepared: ${MANIFEST_PATH}"
  exit 0
fi

command -v hf >/dev/null 2>&1 || {
  echo "The Hugging Face CLI is required. Install huggingface_hub or activate the Slime environment." >&2
  exit 2
}
python3 - <<'PY'
import importlib.util

missing = [name for name in ("datasets", "yaml") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
PY

mkdir -p "${SOURCE_DIR}" "${WORK_DIR}" "${OUTPUT_DIR}" "${DATA_ROOT}/hf_cache"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"

if [[ ! -s "${SOURCE_JSONL}" || ! -s "${RESTORE_SCRIPT}" ]]; then
  echo "Downloading ${SOURCE_DATASET} to ${SOURCE_DIR}"
  hf download "${SOURCE_DATASET}" --repo-type dataset --local-dir "${SOURCE_DIR}"
fi

if [[ ! -s "${SOURCE_JSONL}" || ! -s "${RESTORE_SCRIPT}" ]]; then
  echo "The source download is incomplete: expected ${SOURCE_JSONL} and ${RESTORE_SCRIPT}." >&2
  exit 1
fi

if [[ "${FORCE_REBUILD}" == "1" || ! -s "${COMPLETE_JSONL}" ]]; then
  COMPLETE_TMP="${COMPLETE_JSONL}.tmp.$$"
  trap 'rm -f -- "${COMPLETE_TMP:-}"' EXIT
  echo "Restoring the DAPO and Skywork math placeholders"
  python3 "${RESTORE_SCRIPT}" --input "${SOURCE_JSONL}" --output "${COMPLETE_TMP}"
  mv -- "${COMPLETE_TMP}" "${COMPLETE_JSONL}"
  trap - EXIT
fi

echo "Converting M2RL domains: math science if code (Agent is excluded)"
python3 "${EXAMPLE_DIR}/prepare_m2rl_data.py" \
  --input "${COMPLETE_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-name "${MANIFEST_NAME}" \
  --tasks math science if code \
  --sampling "${SAMPLING}" \
  --sampling-unit "${SAMPLING_UNIT}" \
  --seed "${SEED}"

# Remove a stale output from an older five-task preparation, if present.
if [[ -e "${OUTPUT_DIR}/agent.jsonl" ]]; then
  rm -f -- "${OUTPUT_DIR}/agent.jsonl"
fi

if [[ "${CLEAN_SOURCE_BLEND}" == "1" ]]; then
  rm -f -- "${SOURCE_JSONL}" "${COMPLETE_JSONL}"
  echo "Removed the temporary mixed source files (including their Agent rows)."
fi

echo "Prepared manifest: ${MANIFEST_PATH}"
echo "Dataset inventory: ${OUTPUT_DIR}/dataset_info.json"
