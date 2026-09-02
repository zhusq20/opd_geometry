#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
case "${STAGE}" in warm|middle|late) ;; *) echo "Usage: $0 {warm|middle|late}" >&2; exit 2 ;; esac
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
OUTPUT_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_64k_v3}"
if [[ "${STAGE}" == warm ]]; then
  CHECKPOINT_ROOT="${MOPD_WARM_DIR:-${OUTPUT_ROOT}/warm_start-seed42}/checkpoints"
  TARGET=512
else
  CHECKPOINT_ROOT="${OUTPUT_ROOT}/uniform_k1_taskwise-seed42/checkpoints"
  [[ "${STAGE}" == middle ]] && TARGET=32768 || TARGET=64000
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  CHECKPOINT_ROOT="${MOPD_BASE_MEGATRON:-/workspace/dev/checkpoints/Qwen3-1.7B_torch_dist}"
  CHECKPOINT_ID=0
  RESIDENT_TASK=math
else
  INDEX="${CHECKPOINT_ROOT}/mopd_checkpoint_index.json"
  [[ -f "${INDEX}" ]] || { echo "Missing checkpoint index ${INDEX}." >&2; exit 2; }
  read -r CHECKPOINT_ID RESIDENT_TASK < <(python3 - "${INDEX}" "${CHECKPOINT_ROOT}" "${TARGET}" <<'PY'
import json, pathlib, sys, torch
index = json.loads(pathlib.Path(sys.argv[1]).read_text())
target = int(sys.argv[3])
matches = [entry for entry in index if int(entry["attempted_responses"]) == target]
if len(matches) != 1:
    raise SystemExit(f"expected one checkpoint at {target} responses, got {matches}")
rollout_id = int(matches[0]["rollout_id"])
state_path = pathlib.Path(sys.argv[2]) / "rollout" / f"mopd_dataset_state_dict_{rollout_id}.pt"
state = torch.load(state_path, map_location="cpu", weights_only=False)
resident = int(state["controller"]["resident_teacher"])
print(rollout_id, ("math", "code", "if", "science")[resident])
PY
  )
fi
export MOPD_BANK_LOAD_CHECKPOINT="${CHECKPOINT_ROOT}"
export MOPD_BANK_CHECKPOINT_ID="${CHECKPOINT_ID}"
export MOPD_BANK_RESIDENT_TASK="${RESIDENT_TASK}"
exec bash "${EXAMPLE_DIR}/_launch_mopd.sh" bank "frozen_bank_${STAGE}" 1 uniform taskwise
