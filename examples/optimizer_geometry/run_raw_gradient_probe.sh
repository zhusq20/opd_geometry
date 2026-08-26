#!/usr/bin/env bash
# Same-checkpoint exact raw-gradient probe for a configurable task set.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh}"

LOAD_CHECKPOINT_INPUT="${LOAD_CHECKPOINT:?Set LOAD_CHECKPOINT to one shared torch-dist checkpoint root or iter directory}"
PROBE_CONFIG_ROOT="${PROBE_CONFIG_ROOT:?Set PROBE_CONFIG_ROOT to prepare_sequential_grpo_data.py output}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the probe directory for this checkpoint anchor}"
ANCHOR_NAME="${ANCHOR_NAME:-$(basename -- "${LOAD_CHECKPOINT_INPUT%/}")}"
PROBE_PROMPTS="${PROBE_PROMPTS:-128}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
USE_WANDB="${USE_WANDB:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOAD_CHECKPOINT="$(python3 - "${LOAD_CHECKPOINT_INPUT}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
if (path / ".metadata").is_file():
    resolved = path
else:
    marker = path / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise SystemExit(f"Invalid torch-dist checkpoint: {path}")
    value = marker.read_text().strip()
    if value == "release":
        resolved = path / "release"
    elif value.isdigit():
        resolved = path / f"iter_{int(value):07d}"
    elif value.startswith("iter_") and value.removeprefix("iter_").isdigit():
        resolved = path / value
    else:
        raise SystemExit(f"Unsupported checkpoint marker {value!r}: {marker}")
if not (resolved / ".metadata").is_file() or not (resolved / "common.pt").is_file():
    raise SystemExit(f"Incomplete torch-dist checkpoint: {resolved}")
print(resolved)
PY
)"
unset LOAD_CHECKPOINT_STEP

for value_name in PROBE_PROMPTS ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer." >&2
    exit 2
  fi
done
if (( PROBE_PROMPTS % ROLLOUT_BATCH_SIZE != 0 )); then
  echo "PROBE_PROMPTS must be divisible by ROLLOUT_BATCH_SIZE." >&2
  exit 2
fi
case "${USE_WANDB}:${DRY_RUN}" in
  [01]:[01]) ;;
  *) echo "USE_WANDB and DRY_RUN must each be 0 or 1." >&2; exit 2 ;;
esac

if [[ ! -f "${LAUNCHER}" ]]; then
  echo "Experiment launcher does not exist: ${LAUNCHER}" >&2
  exit 2
fi
PROBE_UPDATES=$((PROBE_PROMPTS / ROLLOUT_BATCH_SIZE))
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
# A probe is a fixed number of full rollout updates, not a dataset epoch.  The
# frozen manifests may contain more prompts than this invocation requests.
# Leaving NUM_EPOCH set would make the launcher consume the whole manifest and
# call the finalized accumulator again on the next update.
unset NUM_EPOCH TARGET_PROMPT_BUDGET

read -r -a tasks <<< "${PROBE_TASKS:-math code science if}"
if (( ${#tasks[@]} == 0 )); then
  echo "PROBE_TASKS must name at least one task." >&2
  exit 2
fi
declare -A seen_tasks=()
for task in "${tasks[@]}"; do
  case "${task}" in
    math|code|science|if) ;;
    *) echo "Unsupported probe task ${task}; use math, code, science, or if." >&2; exit 2 ;;
  esac
  if [[ -n "${seen_tasks[${task}]+x}" ]]; then
    echo "PROBE_TASKS cannot contain duplicate task ${task}." >&2
    exit 2
  fi
  seen_tasks["${task}"]=1
done
for task in "${tasks[@]}"; do
  manifest="${PROBE_CONFIG_ROOT}/${task}/${task}_gradient_probe.yaml"
  if [[ ! -s "${manifest}" ]]; then
    echo "Missing fixed probe manifest: ${manifest}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
for task in "${tasks[@]}"; do
  display_task="${task}"
  if [[ "${task}" == "science" ]]; then
    display_task=knowledge
  fi
  task_output_root="${OUTPUT_DIR}/runs"
  run_name="probe_${ANCHOR_NAME}_${display_task}_seed42"
  run_dir="${task_output_root}/${run_name}"
  gradient_dir="${OUTPUT_DIR}/raw_gradients/${display_task}"
  if [[ -s "${gradient_dir}/manifest.json" && -s "${run_dir}/run_complete.json" ]]; then
    echo "Skipping completed ${display_task} probe at ${ANCHOR_NAME}."
    continue
  fi
  if [[ "${DRY_RUN}" != "1" && ( -e "${gradient_dir}" || -e "${run_dir}" ) ]]; then
    echo "Incomplete probe output already exists for ${display_task}: ${gradient_dir} or ${run_dir}" >&2
    exit 2
  fi

  echo "Probing ${display_task} raw gradient at ${ANCHOR_NAME}: ${PROBE_PROMPTS} prompts"
  TASK="${task}" \
  ALGORITHM=grpo \
  OPTIMIZER=adamw \
  SEED=42 \
  DATA_MANIFEST="${PROBE_CONFIG_ROOT}/${task}/${task}_gradient_probe.yaml" \
  EXPERIMENT_DATA_INDEX="${PROBE_CONFIG_ROOT}/sequential_data_index.json" \
  DISABLE_EVAL=1 \
  OUTPUT_ROOT="${task_output_root}" \
  RUN_NAME="${run_name}" \
  LOAD_CHECKPOINT="${LOAD_CHECKPOINT}" \
  FRESH_START=1 \
  SAVE_CHECKPOINTS=0 \
  NUM_ROLLOUT="${PROBE_UPDATES}" \
  BATCH_PROFILE=responsive16 \
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT}" \
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
  ADAMW_LR=0 \
  WEIGHT_DECAY=0 \
  MAX_PROMPT_LEN=2048 \
  MAX_RESPONSE_LEN=8192 \
  MAX_TOKENS_PER_GPU=10240 \
  APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' \
  GEOMETRY_INTERVAL="${PROBE_UPDATES}" \
  GEOMETRY_RAW_GRADIENT_PROBE_DIR="${gradient_dir}" \
  GEOMETRY_RAW_GRADIENT_PROBE_UPDATES="${PROBE_UPDATES}" \
  GEOMETRY_RAW_GRADIENT_PROBE_ONLY=1 \
  USE_WANDB="${USE_WANDB}" \
  DRY_RUN="${DRY_RUN}" \
    bash "${LAUNCHER}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

analysis_dir="${OUTPUT_DIR}/analysis"
analysis_args=(
  python3 "${SCRIPT_DIR}/analyze_raw_gradient_probes.py"
  --output-dir "${analysis_dir}"
  --force
)
for task in "${tasks[@]}"; do
  display_task="${task}"
  if [[ "${task}" == "science" ]]; then
    display_task=knowledge
  fi
  analysis_args+=(--probe "${display_task}=${OUTPUT_DIR}/raw_gradients/${display_task}")
done
"${analysis_args[@]}"

echo "Completed exact raw-gradient probe for ${tasks[*]}: ${analysis_dir}/summary.json"
