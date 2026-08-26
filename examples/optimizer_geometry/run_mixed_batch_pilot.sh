#!/usr/bin/env bash
# One-update capacity pilot for actor dynamic token packing on four 96 GiB GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MIXED_LAUNCHER="${MIXED_LAUNCHER:-${SCRIPT_DIR}/run_mixed_grpo.sh}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh}"
MIXED_DIR="${MIXED_DIR:-${SLIME_DIR}/outputs/raw_gradient_interference/qwen3_1.7b/seed42/mixed_code_math_qa_if}"
PREPARED_DIR="${PREPARED_DIR:-${MIXED_DIR}/prepared_data}"
PILOT_ROOT="${PILOT_ROOT:-${SLIME_DIR}/outputs/mixed_batch_token_pilot}"
PILOT_TAG="${PILOT_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
TOKEN_BUDGETS="${TOKEN_BUDGETS:-10240 14336 16384}"
MAX_SAFE_PEAK_RESERVED_MIB="${MAX_SAFE_PEAK_RESERVED_MIB:-87000}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${AVAILABLE_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}" ]]; then
  echo "Set AVAILABLE_CUDA_DEVICES (or CUDA_VISIBLE_DEVICES) to four idle 96 GiB GPUs." >&2
  exit 2
fi
IFS=',' read -r -a pilot_gpus <<< "${AVAILABLE_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
if (( ${#pilot_gpus[@]} != 4 )); then
  echo "The mixed batch pilot requires exactly four GPUs; got ${#pilot_gpus[@]}." >&2
  exit 2
fi
if ! [[ "${MAX_SAFE_PEAK_RESERVED_MIB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAFE_PEAK_RESERVED_MIB must be a positive integer." >&2
  exit 2
fi
case "${DRY_RUN}" in
  0|1) ;;
  *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
read -r -a token_budgets <<< "${TOKEN_BUDGETS}"
if (( ${#token_budgets[@]} == 0 )); then
  echo "TOKEN_BUDGETS must contain at least one integer." >&2
  exit 2
fi
for budget in "${token_budgets[@]}"; do
  if ! [[ "${budget}" =~ ^[1-9][0-9]*$ ]] || (( budget < 10240 )); then
    echo "Every TOKEN_BUDGETS entry must be an integer >= 10240; got ${budget}." >&2
    exit 2
  fi
done

# Prepare and validate the same frozen 4,800-prompt/task plan used by the main run.
PLAN_ONLY=1 \
RUN_FINAL_EVAL=0 \
MIXED_DIR="${MIXED_DIR}" \
PREPARED_DIR="${PREPARED_DIR}" \
  bash "${MIXED_LAUNCHER}"

MIXED_MANIFEST="${PREPARED_DIR}/mixed_on_policy.yaml"
MIXED_DATA_INDEX="${PREPARED_DIR}/mixed_data_index.json"
mapfile -t model_paths < <(
  python3 - "${MIXED_DATA_INDEX}" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))["model"]
print(model["hf_checkpoint"])
print(model["base_torch_dist_checkpoint"])
print(model["model_config"])
PY
)
if (( ${#model_paths[@]} != 3 )); then
  echo "Could not read base-model paths from ${MIXED_DATA_INDEX}." >&2
  exit 2
fi

pilot_output="${PILOT_ROOT}/${PILOT_TAG}"
mkdir -p -- "${pilot_output}"
failed=0
for budget in "${token_budgets[@]}"; do
  run_name="mixed_tokens_${budget}_seed42"
  echo "Running one mixed update with MAX_TOKENS_PER_GPU=${budget}: ${pilot_output}/${run_name}"
  if OUTPUT_ROOT="${pilot_output}" \
    RUN_NAME="${run_name}" \
    HF_CHECKPOINT="${model_paths[0]}" \
    LOAD_CHECKPOINT="${model_paths[1]}" \
    MODEL_CONFIG="${model_paths[2]}" \
    DATA_MANIFEST="${MIXED_MANIFEST}" \
    EXPERIMENT_DATA_INDEX="${MIXED_DATA_INDEX}" \
    REWARD_CONFIG="${REWARD_CONFIG:-${SCRIPT_DIR}/configs/rewards.example.yaml}" \
    TASK=mixed_code_math_qa_if_batch_pilot \
    ALGORITHM=grpo \
    OPTIMIZER=adamw \
    BATCH_PROFILE=responsive16 \
    ROLLOUT_BATCH_SIZE=16 \
    N_SAMPLES_PER_PROMPT=16 \
    GLOBAL_BATCH_SIZE=256 \
    TARGET_PROMPT_BUDGET=16 \
    ADAMW_LR=1e-6 \
    WEIGHT_DECAY=0.0 \
    MAX_PROMPT_LEN=2048 \
    MAX_RESPONSE_LEN=8192 \
    MAX_TOKENS_PER_GPU="${budget}" \
    TRAIN_GPU_COUNT=4 \
    ROLLOUT_NUM_GPUS=4 \
    ROLLOUT_GPUS_PER_ENGINE=1 \
    SGLANG_MEM_FRACTION=0.6 \
    SGLANG_MAX_RUNNING_REQUESTS=44 \
    DISABLE_EVAL=1 \
    SAVE_CHECKPOINTS=0 \
    USE_WANDB=0 \
    FRESH_START=1 \
    DRY_RUN="${DRY_RUN}" \
    START_TEACHER=0 \
    SEED=42 \
      bash "${TRAIN_LAUNCHER}"; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "Pilot budget ${budget} command validated."
    else
      echo "Pilot budget ${budget} completed."
    fi
  else
    failed=1
    echo "Pilot budget ${budget} failed; retaining its provenance/log artifacts." >&2
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; all candidate experiment commands validated without launching a pilot."
  exit 0
fi

python3 - "${pilot_output}" "${MAX_SAFE_PEAK_RESERVED_MIB}" "${token_budgets[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
safe_reserved = int(sys.argv[2])
budgets = [int(value) for value in sys.argv[3:]]
rows = []
for budget in budgets:
    run = root / f"mixed_tokens_{budget}_seed42"
    completion = run / "run_complete.json"
    status = "failed"
    if completion.is_file():
        status = str(json.loads(completion.read_text()).get("status") or "unknown")
    metrics = {}
    rollout_metrics = run / "metrics/rollout.jsonl"
    if rollout_metrics.is_file():
        for line in rollout_metrics.read_text().splitlines():
            record = json.loads(line).get("metrics") or {}
            metrics.update(record)
    train = {}
    train_metrics = run / "metrics/train.jsonl"
    if train_metrics.is_file():
        for line in train_metrics.read_text().splitlines():
            train.update(json.loads(line).get("metrics") or {})
    rows.append(
        {
            "max_tokens_per_gpu": budget,
            "status": status,
            "actor_train_tokens_per_second": metrics.get("perf/actor_train_tok_per_s"),
            "actor_train_tflops": metrics.get("perf/actor_train_tflops"),
            "actor_train_seconds": metrics.get("perf/actor_train_time"),
            "rollout_seconds": metrics.get("perf/rollout_time"),
            "step_seconds": metrics.get("perf/step_time"),
            "peak_allocated_mib": train.get("train/gpu_peak_allocated_mib"),
            "peak_reserved_mib": train.get("train/gpu_peak_reserved_mib"),
        }
    )

eligible = [
    row
    for row in rows
    if row["status"] == "complete"
    and row["actor_train_tokens_per_second"] is not None
    and row["peak_reserved_mib"] is not None
    and row["peak_reserved_mib"] <= safe_reserved
]
recommended = max(eligible, key=lambda row: row["actor_train_tokens_per_second"], default=None)
summary = {
    "hardware_contract": "4 x 96 GiB GPUs",
    "fixed_rl_batch": {
        "rollout_prompt_groups": 16,
        "responses_per_prompt": 16,
        "global_trajectories": 256,
    },
    "safe_peak_reserved_mib": safe_reserved,
    "runs": rows,
    "recommended_max_tokens_per_gpu": (
        recommended["max_tokens_per_gpu"] if recommended is not None else None
    ),
}
(root / "batch_pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print("max_tokens\tstatus\tactor_tok/s\tactor_TFLOPS\tactor_s\tstep_s\tpeak_reserved_MiB")
for row in rows:
    values = (
        row["max_tokens_per_gpu"],
        row["status"],
        row["actor_train_tokens_per_second"],
        row["actor_train_tflops"],
        row["actor_train_seconds"],
        row["step_seconds"],
        row["peak_reserved_mib"],
    )
    print("\t".join("-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value) for value in values))
print(f"Summary: {root / 'batch_pilot_summary.json'}")
if recommended is None:
    print("No completed candidate stayed inside the configured memory-safety ceiling.")
else:
    print(f"Recommended MAX_TOKENS_PER_GPU={recommended['max_tokens_per_gpu']}")
PY

exit "${failed}"
