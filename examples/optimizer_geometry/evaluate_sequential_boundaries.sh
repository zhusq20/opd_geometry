#!/usr/bin/env bash
# Run the same locked six-dataset evaluation at every sequential task boundary.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SEQUENCE_DIR="${SEQUENCE_DIR:?Set SEQUENCE_DIR to a completed sequential experiment root}"
SEQUENCE_DIR="$(cd -- "${SEQUENCE_DIR}" && pwd)"
EVAL_ROOT="${EVAL_ROOT:-${SEQUENCE_DIR}/evaluations_v2}"
PRE_CODE_CHECKPOINT="${PRE_CODE_CHECKPOINT:-/workspace/dev/checkpoints/Qwen3-1.7B_torch_dist}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${SEQUENCE_DIR}/analysis_v3}"
EVAL_CONFIG="${EVAL_CONFIG:-${SEQUENCE_DIR}/prepared_data/all_tasks_eval_with_code.yaml}"
DATA_MANIFEST="${DATA_MANIFEST:-${SEQUENCE_DIR}/prepared_data/math/math_on_policy.yaml}"
REWARD_CONFIG="${REWARD_CONFIG:-${SCRIPT_DIR}/configs/rewards.example.yaml}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:?Set four physical GPU indices, for example 6,7,8,9}"
SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL:?Set the verified SandboxFusion endpoint}"
M2RL_SANDBOX_PREFLIGHT_MARKER="${M2RL_SANDBOX_PREFLIGHT_MARKER:?Set a readable current safe preflight marker}"
RAY_DASHBOARD_PORT_BASE="${RAY_DASHBOARD_PORT_BASE:-8365}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

case "${RUN_ANALYSIS}" in
  0|1) ;;
  *) echo "RUN_ANALYSIS must be 0 or 1." >&2; exit 2 ;;
esac
if ! [[ "${RAY_DASHBOARD_PORT_BASE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RAY_DASHBOARD_PORT_BASE must be a positive integer." >&2
  exit 2
fi
for path in "${EVAL_CONFIG}" "${DATA_MANIFEST}" "${REWARD_CONFIG}" "${M2RL_SANDBOX_PREFLIGHT_MARKER}"; do
  if [[ ! -r "${path}" ]]; then
    echo "Required evaluation input is not readable: ${path}" >&2
    exit 2
  fi
done

mapfile -t boundary_records < <(
  python3 - "${SEQUENCE_DIR}/sequence_manifest.json" "${PRE_CODE_CHECKPOINT}" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
if manifest.get("status") != "complete":
    raise SystemExit("Sequential manifest is not complete.")
pre_code = Path(sys.argv[2]).expanduser().resolve()
if not (pre_code / ".metadata").is_file():
    marker = pre_code / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise SystemExit(f"Invalid pre-Code checkpoint: {pre_code}")
    value = marker.read_text().strip()
    if value == "release":
        pre_code = pre_code / "release"
    elif value.isdigit():
        pre_code = pre_code / f"iter_{int(value):07d}"
    else:
        raise SystemExit(f"Unsupported pre-Code checkpoint marker {value!r}: {marker}")
if not (pre_code / ".metadata").is_file() or not (pre_code / "common.pt").is_file():
    raise SystemExit(f"Incomplete pre-Code checkpoint: {pre_code}")
records = (
    ("pre_code", pre_code),
    ("00_code_origin", manifest["code_origin"]),
    ("01_after_math", manifest["stage_endpoints"]["math"]),
    ("02_after_knowledge", manifest["stage_endpoints"]["knowledge"]),
    ("03_after_if", manifest["stage_endpoints"]["if"]),
)
for name, checkpoint in records:
    print(f"{name}\t{Path(checkpoint).resolve()}")
PY
)
if (( ${#boundary_records[@]} != 5 )); then
  echo "Expected five boundary records, found ${#boundary_records[@]}." >&2
  exit 2
fi

mkdir -p -- "${EVAL_ROOT}"
for boundary_index in "${!boundary_records[@]}"; do
  IFS=$'\t' read -r boundary checkpoint <<< "${boundary_records[${boundary_index}]}"
  output_dir="${EVAL_ROOT}/${boundary}"
  if [[ -s "${output_dir}/run_complete.json" && -s "${output_dir}/eval_artifacts/index.jsonl" ]]; then
    echo "Skipping completed boundary evaluation: ${boundary}"
    continue
  fi
  if [[ -e "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "Incomplete boundary output already exists: ${output_dir}" >&2
    exit 2
  fi
  if [[ ! -s "${checkpoint}/.metadata" || ! -s "${checkpoint}/common.pt" ]]; then
    echo "Incomplete boundary checkpoint: ${checkpoint}" >&2
    exit 2
  fi

  echo "Evaluating ${boundary}: ${checkpoint}"
  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
  SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL}" \
  M2RL_SANDBOX_PREFLIGHT_MARKER="${M2RL_SANDBOX_PREFLIGHT_MARKER}" \
  TASK=sequential_four_domain \
  LOAD_CHECKPOINT="${checkpoint}" \
  DATA_MANIFEST="${DATA_MANIFEST}" \
  EVAL_CONFIG="${EVAL_CONFIG}" \
  REWARD_CONFIG="${REWARD_CONFIG}" \
  OUTPUT_DIR="${output_dir}" \
  RUN_NAME="sequential_${boundary}_four_domain_eval" \
  USE_WANDB=0 \
  FRESH_EVAL=1 \
  ALLOW_MIXED_EVAL_RESPONSE_LEN=1 \
  EVAL_MAX_RESPONSE_LEN=32768 \
  EVAL_MAX_CONCURRENCY=48 \
  SGLANG_MAX_RUNNING_REQUESTS=12 \
  RAY_DASHBOARD_PORT="$((RAY_DASHBOARD_PORT_BASE + boundary_index))" \
  RAY_OBJECT_SPILLING_DIR="${output_dir}/.ray_spill" \
  EXPERIMENT_DATA_INDEX="${SEQUENCE_DIR}/prepared_data/sequential_data_index.json" \
  EXPERIMENT_EVAL_INDEX="${SEQUENCE_DIR}/prepared_data/livecodebench_index.json" \
    bash "${SCRIPT_DIR}/evaluate_single_task.sh"
done

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
  analysis_args=(
    python3 "${SCRIPT_DIR}/analyze_sequential_experiment.py"
    --sequence-root "${SEQUENCE_DIR}"
    --pre-code-checkpoint "${PRE_CODE_CHECKPOINT}"
    --eval-root "${EVAL_ROOT}"
    --output-dir "${ANALYSIS_DIR}"
    --force
  )
  if [[ -s "${SEQUENCE_DIR}/parameter_geometry_v3/summary.json" ]]; then
    analysis_args+=(--parameter-geometry "${SEQUENCE_DIR}/parameter_geometry_v3/summary.json")
  fi
  "${analysis_args[@]}"
fi

echo "Completed all sequential boundary evaluations under ${EVAL_ROOT}."
