#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
FULL_ROOT="${OPEN_MOPD_FULL_ROOT:-${REPO_ROOT}/local/open_mopd_full}"
SOURCE_ROOT="${OPEN_MOPD_SOURCE_ROOT:-${FULL_ROOT}/source}"
ASSET_ROOT="${OPEN_MOPD_ASSET_ROOT:-${FULL_ROOT}/assets}"
MODEL="${OPEN_MOPD_EVAL_MODEL:-${ASSET_ROOT}/models/final}"
OUTPUT_ROOT="${OPEN_MOPD_EVAL_OUTPUT_ROOT:-${FULL_ROOT}/runs/eval}"
PYTHON_BIN="${OPEN_MOPD_PYTHON:-python3}"
EXECUTE=0

usage() {
  echo "Usage: $0 [--dry-run|--run]"
  echo "Evaluate the official final model, or OPEN_MOPD_EVAL_MODEL, on all six paper benchmarks."
}

die() {
  echo "[open-mopd-full-eval] error: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --run) EXECUTE=1 ;;
    --dry-run) EXECUTE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

run_command() {
  printf '[open-mopd-full-eval]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${EXECUTE}" == 1 ]]; then
    "$@"
  fi
}

evaluate_one() {
  local dataset="$1"
  local relative_path="$2"
  local temperature="$3"
  local responses="$4"
  local max_tokens="$5"
  local enable_thinking="$6"
  local input="${ASSET_ROOT}/data/${relative_path}"
  local output="${OUTPUT_ROOT}/${dataset}"
  local rollout=(
    "${PYTHON_BIN}" -m evals.rollout_engine.vllm_rollout
    --model "${MODEL}"
    --input "${input}"
    --output-dir "${output}"
    --temperature "${temperature}"
    --top-p 0.95
    --top-k -1
    --n "${responses}"
    --max-tokens "${max_tokens}"
    --max-model-len 32768
    --data-parallel-size 8
    --stop-token-ids 128012
    --trust-remote-code
  )
  if [[ "${enable_thinking}" == true ]]; then
    rollout+=(--enable-thinking true)
  fi
  if [[ "${EXECUTE}" == 1 ]]; then
    [[ -f "${input}" ]] || die "evaluation data missing: ${input}"
    mkdir -p "${output}"
  fi
  run_command "${rollout[@]}"
  run_command \
    "${PYTHON_BIN}" -m evals.score_rollouts \
    --rollout-dir "${output}" \
    --dataset "${dataset}" \
    --out "${output}/scores.json" \
    --max-tokens "${max_tokens}"
}

if [[ "${EXECUTE}" == 1 ]]; then
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
  [[ -d "${SOURCE_ROOT}/evals" ]] || die "official source missing; run fetch.sh source"
  [[ -d "${MODEL}" ]] || die "evaluation model missing: ${MODEL}"
  export PYTHONPATH="${SOURCE_ROOT}/training/verl:${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

cd "${SOURCE_ROOT}" 2>/dev/null || {
  [[ "${EXECUTE}" == 0 ]] || die "official source missing: ${SOURCE_ROOT}"
  cd "${REPO_ROOT}"
}

evaluate_one aime24 eval/math/aime24.parquet 0.6 64 30000 false
evaluate_one aime25 eval/math/aime25.parquet 0.6 64 30000 false
evaluate_one livecodebench_v5 eval/code/livecodebench_v5.parquet 1.0 10 30000 false
evaluate_one livecodebench_v6 eval/code/livecodebench_v6.parquet 1.0 10 30000 false
evaluate_one ifeval eval/if/ifeval_aligned.parquet 1.0 1 2048 true
evaluate_one ifbench_test eval/if/ifbench_test_aligned.parquet 1.0 1 2048 true

if [[ "${EXECUTE}" == 0 ]]; then
  echo "[open-mopd-full-eval] dry-run only; pass --run to execute"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize.py" "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/summary.json"
fi
