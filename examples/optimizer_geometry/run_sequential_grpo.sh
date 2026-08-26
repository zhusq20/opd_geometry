#!/usr/bin/env bash
# Continual GRPO: existing Code checkpoint -> Math -> Knowledge -> IF.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CODE_CHECKPOINT="${SLIME_DIR}/outputs/qwen3_1.7b_code_grpo_after_sandbox_20260817T194524Z/qwen3_1.7b_code_grpo_adamw_responsive16_trainr8192_seed42/checkpoints/iter_0000299"

CODE_CHECKPOINT="${CODE_CHECKPOINT:-${DEFAULT_CODE_CHECKPOINT}}"
SEQUENCE_DIR="${SEQUENCE_DIR:-${SLIME_DIR}/outputs/raw_gradient_interference/qwen3_1.7b/seed42/sequential_code_math_knowledge_if}"
PREPARED_DIR="${PREPARED_DIR:-${SEQUENCE_DIR}/prepared_data}"
SINGLE_TASK_CONFIG_ROOT="${SINGLE_TASK_CONFIG_ROOT:-${SLIME_DIR}/data/m2rl/single_task}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${SCRIPT_DIR}/run_single_task_rl.sh}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-${SCRIPT_DIR}/evaluate_single_task.sh}"
PROBE_LAUNCHER="${PROBE_LAUNCHER:-${SCRIPT_DIR}/run_raw_gradient_probe.sh}"
USE_WANDB="${USE_WANDB:-1}"
RUN_BOUNDARY_EVAL="${RUN_BOUNDARY_EVAL:-0}"
RUN_GRADIENT_PROBES="${RUN_GRADIENT_PROBES:-0}"
RUN_ORIGIN_MEASUREMENTS="${RUN_ORIGIN_MEASUREMENTS:-0}"
RESUME_INCOMPLETE_STAGE="${RESUME_INCOMPLETE_STAGE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
PROBE_PROMPTS="${PROBE_PROMPTS:-128}"
TRAIN_PROMPTS_PER_TASK="${TRAIN_PROMPTS_PER_TASK:-4800}"
unset LOAD_CHECKPOINT_STEP

for boolean_name in USE_WANDB RUN_BOUNDARY_EVAL RUN_GRADIENT_PROBES RUN_ORIGIN_MEASUREMENTS RESUME_INCOMPLETE_STAGE PLAN_ONLY; do
  boolean_value="${!boolean_name}"
  case "${boolean_value}" in
    0|1) ;;
    *) echo "${boolean_name} must be 0 or 1." >&2; exit 2 ;;
  esac
done
if [[ "${SEED:-42}" != "42" ]]; then
  echo "The paper sequence is frozen to SEED=42." >&2
  exit 2
fi
if ! [[ "${PROBE_PROMPTS}" =~ ^[1-9][0-9]*$ ]] || (( PROBE_PROMPTS % 16 != 0 )); then
  echo "PROBE_PROMPTS must be a positive multiple of 16." >&2
  exit 2
fi
if ! [[ "${TRAIN_PROMPTS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || (( TRAIN_PROMPTS_PER_TASK % 16 != 0 )); then
  echo "TRAIN_PROMPTS_PER_TASK must be a positive multiple of 16." >&2
  exit 2
fi
for path in "${TRAIN_LAUNCHER}" "${EVAL_LAUNCHER}" "${PROBE_LAUNCHER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required launcher does not exist: ${path}" >&2
    exit 2
  fi
done

resolve_checkpoint() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
if (path / ".metadata").is_file():
    print(path)
    raise SystemExit
marker = path / "latest_checkpointed_iteration.txt"
if not marker.is_file():
    raise SystemExit(f"Invalid torch-dist checkpoint: {path}")
iteration = int(marker.read_text().strip())
resolved = path / f"iter_{iteration:07d}"
if not (resolved / ".metadata").is_file() or not (resolved / "common.pt").is_file():
    raise SystemExit(f"Incomplete latest torch-dist checkpoint: {resolved}")
print(resolved)
PY
}

CODE_RESOLVED="$(resolve_checkpoint "${CODE_CHECKPOINT}")"
mkdir -p "${SEQUENCE_DIR}"
export RAY_OBJECT_SPILLING_DIR="${RAY_OBJECT_SPILLING_DIR:-${SEQUENCE_DIR}/.ray_spill}"
python3 "${SCRIPT_DIR}/prepare_sequential_grpo_data.py" \
  --config-root "${SINGLE_TASK_CONFIG_ROOT}" \
  --output-dir "${PREPARED_DIR}" \
  --tasks math science if \
  --probe-prompts "${PROBE_PROMPTS}" \
  --train-prompts-per-task "${TRAIN_PROMPTS_PER_TASK}" \
  --seed 42

SANDBOX_FREE_EVAL="${PREPARED_DIR}/all_tasks_eval.yaml"
DATA_INDEX="${PREPARED_DIR}/sequential_data_index.json"

python3 - "${SEQUENCE_DIR}" "${CODE_CHECKPOINT}" "${CODE_RESOLVED}" "${DATA_INDEX}" \
  "${RUN_BOUNDARY_EVAL}" "${RUN_GRADIENT_PROBES}" "${PROBE_PROMPTS}" \
  "${TRAIN_PROMPTS_PER_TASK}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sequence_dir = Path(sys.argv[1]).resolve()
code_root = Path(sys.argv[2]).resolve()
code_resolved = Path(sys.argv[3]).resolve()
data_index = Path(sys.argv[4]).resolve()
run_eval = bool(int(sys.argv[5]))
run_probes = bool(int(sys.argv[6]))
probe_prompts = int(sys.argv[7])
train_prompts_per_task = int(sys.argv[8])

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def command_value(command, flag):
    if not isinstance(command, list) or flag not in command:
        return None
    index = command.index(flag) + 1
    return command[index] if index < len(command) else None

def converted(command, flag, converter):
    value = command_value(command, flag)
    return converter(value) if value is not None else None

def code_origin_training():
    candidates = [code_root.parent / "provenance/run_manifest.json"]
    if code_root.name.startswith("iter_"):
        candidates.append(code_root.parent.parent / "provenance/run_manifest.json")
    candidates.append(code_resolved.parent.parent / "provenance/run_manifest.json")
    provenance = next((path for path in candidates if path.is_file()), None)
    if provenance is None:
        return {"provenance_manifest": None}
    record = json.loads(provenance.read_text())
    command = record.get("command")
    completion_marker = provenance.parent.parent / "run_complete.json"
    return {
        "provenance_manifest": str(provenance.resolve()),
        "provenance_manifest_sha256": digest(provenance),
        "source_run_status": record.get("status"),
        "source_run_completion_marker": str(completion_marker.resolve()),
        "source_run_complete": completion_marker.is_file(),
        "optimizer": command_value(command, "--experiment-optimizer") or command_value(command, "--optimizer"),
        "learning_rate": converted(command, "--lr", float),
        "weight_decay": converted(command, "--weight-decay", float),
        "grpo_group_size": converted(command, "--n-samples-per-prompt", int),
        "rollout_batch_size": converted(command, "--rollout-batch-size", int),
        "global_batch_size": converted(command, "--global-batch-size", int),
        "num_epoch": converted(command, "--num-epoch", int),
        "seed": converted(command, "--seed", int),
    }

origin_training = code_origin_training()
new_stage_training = {
    "optimizer": "AdamW",
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "grpo_group_size": 16,
    "rollout_batch_size": 16,
    "global_batch_size": 256,
    "num_epoch": 1,
    "train_prompts_per_task": train_prompts_per_task,
    "optimizer_updates_per_task": train_prompts_per_task // 16,
    "checkpoint_interval_updates": 100,
    "seed": 42,
}

manifest_path = sequence_dir / "sequence_manifest.json"
expected = {
    "schema_version": 4,
    "code_origin": str(code_resolved),
    "code_origin_metadata_sha256": digest(code_resolved / ".metadata"),
    "code_origin_common_sha256": digest(code_resolved / "common.pt"),
    "data_index_sha256": digest(data_index),
    "probe_prompts_per_task": probe_prompts,
    "train_prompts_per_task": train_prompts_per_task,
}
if manifest_path.is_file():
    current = json.loads(manifest_path.read_text())
    mismatched = {
        key: (current.get(key), value)
        for key, value in expected.items()
        if current.get(key) != value
    }
    if mismatched:
        raise SystemExit(f"Existing sequential manifest is incompatible: {mismatched}")
else:
    current = {
        "schema_version": 4,
        "status": "planned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence": ["code_origin", "math", "knowledge", "if"],
        "measurement_tasks": ["math", "knowledge", "if"],
        "seed": 42,
        "code_origin_root": str(code_root),
        "code_origin": str(code_resolved),
        "code_origin_metadata_sha256": digest(code_resolved / ".metadata"),
        "code_origin_common_sha256": digest(code_resolved / "common.pt"),
        "data_index": str(data_index),
        "data_index_sha256": digest(data_index),
        "probe_prompts_per_task": probe_prompts,
        "train_prompts_per_task": train_prompts_per_task,
        "optimizer_state_at_task_boundaries": "reset",
        "code_origin_training": origin_training,
        "new_stage_training": new_stage_training,
        "task_boundary_evaluation": run_eval,
        "same_checkpoint_raw_gradient_probes": run_probes,
        "sandbox_execution_enabled": False,
        "notes": [
            "The existing Code checkpoint is reused as a weight-only warm start.",
            "AdamW moments and RNG state are reset at every new task stage.",
            "Code-origin hyperparameters are recovered from its provenance when available; they need not match new stages.",
            "Math, Knowledge, and IF training sets exclude the frozen gradient-probe prompts.",
            f"Each new task uses {train_prompts_per_task} prompts for a short efficiency-first run.",
            "Code training, Code evaluation, and Code raw-gradient probes are omitted so this sequence never requires SandboxFusion.",
            "Consequently, this run cannot measure Code forgetting after the warm-start checkpoint.",
        ],
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
if current.get("code_origin_training", {}).get("source_run_complete") is False:
    print(
        "Warning: the reused Code checkpoint is structurally complete, but its source run has no "
        "run_complete.json; this fact is recorded in sequence_manifest.json."
    )
PY

echo "Sequential GRPO origin: ${CODE_RESOLVED}"
echo "Sequence output: ${SEQUENCE_DIR}"
echo "Task order: existing Code -> Math -> Knowledge -> IF"
echo "Boundary policy: load weights; reset AdamW moments and RNG; new-stage group size=16; LR=1e-6"
echo "Training budget: ${TRAIN_PROMPTS_PER_TASK} prompts = $((TRAIN_PROMPTS_PER_TASK / 16)) updates per task"
echo "Checkpoint cadence: every 100 optimizer updates, including the final update"
echo "Sandbox policy: Code train/eval/probes are omitted; SandboxFusion is not required."

if [[ "${PLAN_ONLY}" == "1" ]]; then
  echo "PLAN_ONLY=1; no training, evaluation, or gradient probe was launched."
  echo "Planned stages:"
  echo "  01 Math      <- ${CODE_RESOLVED}"
  echo "  02 Knowledge <- ${SEQUENCE_DIR}/stages/01_math/seq_01_math_from_code_seed42/checkpoints"
  echo "  03 IF        <- ${SEQUENCE_DIR}/stages/02_knowledge/seq_02_knowledge_from_math_seed42/checkpoints"
  exit 0
fi

TRAIN_GPU_COUNT="${TRAIN_GPU_COUNT:-4}"
if ! [[ "${TRAIN_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_GPU_COUNT must be a positive integer." >&2
  exit 2
fi
if [[ -z "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
  sequence_available_devices="${AVAILABLE_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  if [[ -z "${sequence_available_devices}" ]]; then
    echo "Set AVAILABLE_CUDA_DEVICES, CUDA_VISIBLE_DEVICES, or TRAIN_CUDA_VISIBLE_DEVICES." >&2
    exit 2
  fi
  IFS=',' read -r -a sequence_available_gpus <<< "${sequence_available_devices}"
  if (( ${#sequence_available_gpus[@]} < TRAIN_GPU_COUNT )); then
    echo "Need ${TRAIN_GPU_COUNT} training GPUs, but only ${#sequence_available_gpus[@]} were provided." >&2
    exit 2
  fi
  sequence_train_gpus=("${sequence_available_gpus[@]:0:TRAIN_GPU_COUNT}")
  TRAIN_CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${sequence_train_gpus[*]}")"
else
  IFS=',' read -r -a sequence_train_gpus <<< "${TRAIN_CUDA_VISIBLE_DEVICES}"
  if (( ${#sequence_train_gpus[@]} != TRAIN_GPU_COUNT )); then
    echo "TRAIN_GPU_COUNT=${TRAIN_GPU_COUNT} does not match TRAIN_CUDA_VISIBLE_DEVICES." >&2
    exit 2
  fi
fi
export TRAIN_GPU_COUNT TRAIN_CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"

boundary_measurements() {
  local anchor_name="$1"
  local checkpoint_root="$2"

  if [[ "${RUN_BOUNDARY_EVAL}" == "1" ]]; then
    local eval_dir="${SEQUENCE_DIR}/evaluations/${anchor_name}"
    if [[ -s "${eval_dir}/run_complete.json" ]]; then
      echo "Skipping completed sandbox-free three-task evaluation at ${anchor_name}."
    else
      if [[ -e "${eval_dir}" ]]; then
        echo "Incomplete evaluation directory already exists: ${eval_dir}" >&2
        exit 2
      fi
      TASK=sequential_sandbox_free \
      SEED=42 \
      LOAD_CHECKPOINT="${checkpoint_root}" \
      DATA_MANIFEST="${PREPARED_DIR}/math/math_on_policy.yaml" \
      EVAL_CONFIG="${SANDBOX_FREE_EVAL}" \
      EXPERIMENT_DATA_INDEX="${DATA_INDEX}" \
      OUTPUT_DIR="${eval_dir}" \
      RUN_NAME="sequential_${anchor_name}_sandbox_free_eval" \
      ALLOW_MIXED_EVAL_RESPONSE_LEN=1 \
      EVAL_MAX_RESPONSE_LEN=32768 \
      EVAL_MAX_CONCURRENCY=48 \
      SGLANG_MAX_RUNNING_REQUESTS=12 \
      NUM_GPUS="${TRAIN_GPU_COUNT}" \
      CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
      USE_WANDB="${USE_WANDB}" \
      WANDB_GROUP=sequential_grpo_boundary_eval \
        bash "${EVAL_LAUNCHER}"
    fi
  fi

  if [[ "${RUN_GRADIENT_PROBES}" == "1" ]]; then
    local probe_dir="${SEQUENCE_DIR}/gradient_probes/${anchor_name}"
    if [[ -s "${probe_dir}/analysis/summary.json" ]]; then
      echo "Skipping completed exact raw-gradient probe at ${anchor_name}."
    else
      LOAD_CHECKPOINT="${checkpoint_root}" \
      PROBE_CONFIG_ROOT="${PREPARED_DIR}" \
      OUTPUT_DIR="${probe_dir}" \
      ANCHOR_NAME="${anchor_name}" \
      PROBE_PROMPTS="${PROBE_PROMPTS}" \
      PROBE_TASKS="math science if" \
      USE_WANDB=0 \
        bash "${PROBE_LAUNCHER}"
    fi
  fi
}

if [[ "${RUN_ORIGIN_MEASUREMENTS}" == "1" ]]; then
  boundary_measurements 00_code_origin "${CODE_RESOLVED}"
fi

stage_tasks=(math science if)
stage_labels=(math knowledge if)
stage_parents=(code math knowledge)
previous_checkpoint_root="${CODE_RESOLVED}"
previous_checkpoint_resolved="${CODE_RESOLVED}"
stage_resolved_checkpoints=()

for stage_index in "${!stage_tasks[@]}"; do
  task="${stage_tasks[stage_index]}"
  label="${stage_labels[stage_index]}"
  parent="${stage_parents[stage_index]}"
  stage_number="$(printf '%02d' "$((stage_index + 1))")"
  stage_output_root="${SEQUENCE_DIR}/stages/${stage_number}_${label}"
  run_name="seq_${stage_number}_${label}_from_${parent}_seed42"
  run_dir="${stage_output_root}/${run_name}"
  output_checkpoint_root="${run_dir}/checkpoints"

  if [[ -s "${run_dir}/run_complete.json" ]]; then
    echo "Skipping completed stage ${stage_number} ${label}."
  else
    fresh_start=1
    stage_load_checkpoint="${previous_checkpoint_root}"
    if [[ -d "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -print -quit)" ]]; then
      if [[ "${RESUME_INCOMPLETE_STAGE}" != "1" ]]; then
        echo "Incomplete stage exists: ${run_dir}" >&2
        echo "Inspect it, then set RESUME_INCOMPLETE_STAGE=1 to resume its own checkpoint." >&2
        exit 2
      fi
      resolve_checkpoint "${output_checkpoint_root}" >/dev/null
      fresh_start=0
      stage_load_checkpoint="${output_checkpoint_root}"
      echo "Resuming incomplete ${label} stage from ${stage_load_checkpoint}."
    fi

    TASK="${task}" \
    RL_ALGORITHM=grpo \
    OPTIMIZERS=adamw \
    SEED=42 \
    SINGLE_TASK_CONFIG_ROOT="${PREPARED_DIR}" \
    DATA_MANIFEST="${PREPARED_DIR}/${task}/${task}_on_policy.yaml" \
    EXPERIMENT_DATA_INDEX="${DATA_INDEX}" \
    DISABLE_EVAL=1 \
    REQUIRE_EVAL=0 \
    OUTPUT_ROOT="${stage_output_root}" \
    RUN_NAME="${run_name}" \
    LOAD_CHECKPOINT="${stage_load_checkpoint}" \
    FRESH_START="${fresh_start}" \
    BATCH_PROFILE=responsive16 \
    ROLLOUT_BATCH_SIZE=16 \
    N_SAMPLES_PER_PROMPT=16 \
    GLOBAL_BATCH_SIZE=256 \
    SAVE_INTERVAL=100 \
    ADAMW_LR=1e-6 \
    USE_WANDB="${USE_WANDB}" \
    WANDB_GROUP=sequential_grpo \
      bash "${TRAIN_LAUNCHER}"
  fi

  current_checkpoint_resolved="$(resolve_checkpoint "${output_checkpoint_root}")"
  stage_resolved_checkpoints+=("${current_checkpoint_resolved}")
  python3 - "${SEQUENCE_DIR}/stage_boundaries.jsonl" "${stage_number}" "${label}" \
    "${previous_checkpoint_resolved}" "${current_checkpoint_resolved}" "${run_dir}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema_version": 1,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage_number": int(sys.argv[2]),
    "task": sys.argv[3],
    "input_checkpoint": str(Path(sys.argv[4]).resolve()),
    "output_checkpoint": str(Path(sys.argv[5]).resolve()),
    "run_dir": str(Path(sys.argv[6]).resolve()),
    "optimizer_state_at_boundary": "reset",
}
existing = []
if path.is_file():
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
existing = [item for item in existing if int(item["stage_number"]) != record["stage_number"]]
existing.append(record)
existing.sort(key=lambda item: int(item["stage_number"]))
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in existing))
os.replace(temporary, path)
PY

  boundary_measurements "${stage_number}_after_${label}" "${current_checkpoint_resolved}"
  previous_checkpoint_root="${current_checkpoint_resolved}"
  previous_checkpoint_resolved="${current_checkpoint_resolved}"
done

parameter_geometry_dir="${SEQUENCE_DIR}/parameter_geometry"
if [[ ! -s "${parameter_geometry_dir}/summary.json" ]]; then
  python3 "${SCRIPT_DIR}/analyze_sequential_updates.py" \
    --stage "math=${CODE_RESOLVED}::${stage_resolved_checkpoints[0]}" \
    --stage "knowledge=${stage_resolved_checkpoints[0]}::${stage_resolved_checkpoints[1]}" \
    --stage "if=${stage_resolved_checkpoints[1]}::${stage_resolved_checkpoints[2]}" \
    --output-dir "${parameter_geometry_dir}"
fi

python3 - "${SEQUENCE_DIR}/sequence_manifest.json" \
  "${stage_resolved_checkpoints[0]}" "${stage_resolved_checkpoints[1]}" "${stage_resolved_checkpoints[2]}" \
  "${RUN_BOUNDARY_EVAL}" "${RUN_GRADIENT_PROBES}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
manifest["status"] = "complete"
manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest["stage_endpoints"] = {
    "math": str(Path(sys.argv[2]).resolve()),
    "knowledge": str(Path(sys.argv[3]).resolve()),
    "if": str(Path(sys.argv[4]).resolve()),
}
manifest["task_boundary_evaluation"] = bool(int(sys.argv[5]))
manifest["same_checkpoint_raw_gradient_probes"] = bool(int(sys.argv[6]))
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY

echo "Sequential GRPO complete: ${SEQUENCE_DIR}"
