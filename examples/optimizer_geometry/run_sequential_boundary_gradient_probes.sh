#!/usr/bin/env bash
# Measure exact same-checkpoint four-task raw-gradient interference at each boundary.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEQUENCE_DIR="${SEQUENCE_DIR:?Set SEQUENCE_DIR to a completed sequential experiment root}"
SEQUENCE_DIR="$(cd -- "${SEQUENCE_DIR}" && pwd)"
PROBE_CONFIG_ROOT="${PROBE_CONFIG_ROOT:-${SEQUENCE_DIR}/prepared_all_task_probes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SEQUENCE_DIR}/gradient_probes_v2}"
PRE_CODE_CHECKPOINT="${PRE_CODE_CHECKPOINT:-/workspace/dev/checkpoints/Qwen3-1.7B_torch_dist}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${SEQUENCE_DIR}/analysis_v3}"
AVAILABLE_CUDA_DEVICES="${AVAILABLE_CUDA_DEVICES:?Set four free physical GPU indices}"
SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL:?Set the verified SandboxFusion endpoint}"
M2RL_SANDBOX_PREFLIGHT_MARKER="${M2RL_SANDBOX_PREFLIGHT_MARKER:?Set a readable current safe preflight marker}"
PROBE_PROMPTS="${PROBE_PROMPTS:-16}"
RAY_DASHBOARD_PORT_BASE="${RAY_DASHBOARD_PORT_BASE:-8465}"

for value_name in PROBE_PROMPTS RAY_DASHBOARD_PORT_BASE; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer." >&2
    exit 2
  fi
done
if (( PROBE_PROMPTS % 16 != 0 )); then
  echo "PROBE_PROMPTS must be divisible by the fixed 16-prompt rollout batch." >&2
  exit 2
fi
for task in math code science if; do
  if [[ ! -s "${PROBE_CONFIG_ROOT}/${task}/${task}_gradient_probe.yaml" ]]; then
    echo "Missing frozen ${task} probe manifest under ${PROBE_CONFIG_ROOT}." >&2
    exit 2
  fi
done
if [[ ! -r "${M2RL_SANDBOX_PREFLIGHT_MARKER}" ]]; then
  echo "Sandbox preflight marker is not readable: ${M2RL_SANDBOX_PREFLIGHT_MARKER}" >&2
  exit 2
fi

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

mkdir -p -- "${OUTPUT_ROOT}"
for boundary_index in "${!boundary_records[@]}"; do
  IFS=$'\t' read -r boundary checkpoint <<< "${boundary_records[${boundary_index}]}"
  boundary_output="${OUTPUT_ROOT}/${boundary}"
  echo "Raw-gradient probing ${boundary}: ${checkpoint}"
  AVAILABLE_CUDA_DEVICES="${AVAILABLE_CUDA_DEVICES}" \
  SANDBOXFUSION_BASE_URL="${SANDBOXFUSION_BASE_URL}" \
  M2RL_SANDBOX_PREFLIGHT_MARKER="${M2RL_SANDBOX_PREFLIGHT_MARKER}" \
  LOAD_CHECKPOINT="${checkpoint}" \
  PROBE_CONFIG_ROOT="${PROBE_CONFIG_ROOT}" \
  OUTPUT_DIR="${boundary_output}" \
  ANCHOR_NAME="${boundary}" \
  PROBE_TASKS="code math science if" \
  PROBE_PROMPTS="${PROBE_PROMPTS}" \
  ROLLOUT_BATCH_SIZE=16 \
  N_SAMPLES_PER_PROMPT=16 \
  USE_WANDB=0 \
  RAY_DASHBOARD_PORT="$((RAY_DASHBOARD_PORT_BASE + boundary_index))" \
  RAY_OBJECT_SPILLING_DIR="${boundary_output}/.ray_spill" \
    bash "${SCRIPT_DIR}/run_raw_gradient_probe.sh"
done

python3 "${SCRIPT_DIR}/summarize_sequential_raw_gradient_probes.py" \
  --probe-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --force

python3 "${SCRIPT_DIR}/analyze_sequential_experiment.py" \
  --sequence-root "${SEQUENCE_DIR}" \
  --pre-code-checkpoint "${PRE_CODE_CHECKPOINT}" \
  --eval-root "${SEQUENCE_DIR}/evaluations_v2" \
  --output-dir "${ANALYSIS_DIR}" \
  --force

echo "Completed exact four-task raw-gradient probes under ${OUTPUT_ROOT}."
