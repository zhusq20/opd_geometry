#!/usr/bin/env bash
# Materialize math/code/science training manifests plus the independent online
# eval datasets used by this study (selectable AIME'24/MATH-500 and
# GPQA-Diamond). Math rows that overlap MATH-500 are removed from the
# single-task training view. Coding eval is prepared separately from
# LiveCodeBench.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
M2RL_DATA_ROOT="${M2RL_DATA_ROOT:-${SLIME_DIR}/data/m2rl}"
RL_MANIFEST="${RL_MANIFEST:-${M2RL_DATA_ROOT}/train/multitask_manifest.yaml}"
OUTPUT_DIR="${SINGLE_TASK_CONFIG_ROOT:-${M2RL_DATA_ROOT}/single_task}"
INDEX_PATH="${OUTPUT_DIR}/single_task_index.json"
EVAL_DATA_DIR="${M2RL_EVAL_DATA_DIR:-${M2RL_DATA_ROOT}/eval/m2rl_online}"
MATH500_PATH="${EVAL_DATA_DIR}/math500.parquet"
GPQA_PATH="${EVAL_DATA_DIR}/gpqa_diamond.parquet"
SEED="${SEED:-42}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
REQUIRE_GPQA="${REQUIRE_GPQA:-0}"
MATH_EVAL_DATASETS="${MATH_EVAL_DATASETS:-math500}"
read -r -a MATH_EVAL_DATASET_LIST <<< "${MATH_EVAL_DATASETS//,/ }"

case "${FORCE_REBUILD}" in
  0|1) ;;
  *) echo "FORCE_REBUILD must be 0 or 1." >&2; exit 2 ;;
esac
case "${REQUIRE_GPQA}" in
  0|1) ;;
  *) echo "REQUIRE_GPQA must be 0 or 1." >&2; exit 2 ;;
esac

if [[ ! -s "${RL_MANIFEST}" ]]; then
  echo "Prepared M2RL manifest not found: ${RL_MANIFEST}" >&2
  echo "Run: bash ${SCRIPT_DIR}/prepare_m2rl_dataset.sh" >&2
  exit 2
fi

EVAL_FORCE_ARGS=()
if [[ "${FORCE_REBUILD}" == "1" ]]; then
  EVAL_FORCE_ARGS+=(--force)
fi

python3 "${SCRIPT_DIR}/prepare_m2rl_eval_data.py" \
  --output-dir "${EVAL_DATA_DIR}" \
  --datasets aime24 math500 \
  --seed "${SEED}" \
  --math-eval-config-dir "${OUTPUT_DIR}/math" \
  --math-eval-datasets "${MATH_EVAL_DATASET_LIST[@]}" \
  "${EVAL_FORCE_ARGS[@]}"

GPQA_READY=0
if python3 "${SCRIPT_DIR}/prepare_m2rl_eval_data.py" \
  --output-dir "${EVAL_DATA_DIR}" \
  --datasets gpqa_diamond \
  --seed "${SEED}" \
  "${EVAL_FORCE_ARGS[@]}"; then
  GPQA_READY=1
elif python3 - "${GPQA_PATH}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as parquet

path = Path(sys.argv[1])
raise SystemExit(0 if path.is_file() and parquet.ParquetFile(path).metadata.num_rows == 198 else 1)
PY
then
  GPQA_READY=1
  echo "WARNING: GPQA refresh failed; retaining the existing validated 198-row file." >&2
elif [[ "${REQUIRE_GPQA}" == "1" ]]; then
  echo "GPQA-Diamond is required but unavailable. Export HF_TOKEN after accepting the dataset terms." >&2
  exit 2
else
  echo "WARNING: GPQA-Diamond is gated and unavailable; science online eval will be disabled." >&2
  echo "Export HF_TOKEN and rerun with FORCE_REBUILD=1 REQUIRE_GPQA=1 to match M2RL exactly." >&2
fi

if [[ "${FORCE_REBUILD}" == "0" && -s "${INDEX_PATH}" ]]; then
  if python3 - "${INDEX_PATH}" "${GPQA_READY}" "$(IFS=,; echo "${MATH_EVAL_DATASET_LIST[*]}")" <<'PY'
import json
import sys
from pathlib import Path

import yaml

index = json.loads(Path(sys.argv[1]).read_text())
tasks = index.get("tasks", {})
expected = {
    "math": ("external_benchmark", "math500", 1, 5),
    "science": ("external_benchmark", "gpqa", 4, 0) if sys.argv[2] == "1" else ("disabled", None, None, 0),
}
for task, values in expected.items():
    item = tasks.get(task, {})
    observed = (
        item.get("eval_kind"),
        item.get("eval_name"),
        item.get("eval_samples_per_prompt"),
        item.get("excluded_eval_overlap_rows", 0),
    )
    if observed != values or not Path(item.get("on_policy_manifest", "")).is_file():
        raise SystemExit(1)
    eval_config = item.get("eval_config")
    if eval_config is not None and not Path(eval_config).is_file():
        raise SystemExit(1)
code = tasks.get("code", {})
code_eval = (code.get("eval_kind"), code.get("eval_name"), code.get("eval_samples_per_prompt"))
if code_eval not in {("disabled", None, None), ("external_benchmark", "livecodebench_online", 1)}:
    raise SystemExit(1)
if not Path(code.get("on_policy_manifest", "")).is_file():
    raise SystemExit(1)
if code.get("eval_config") is not None and not Path(code["eval_config"]).is_file():
    raise SystemExit(1)
math_config = Path(tasks["math"]["eval_config"])
config = yaml.safe_load(math_config.read_text())
configured_names = [dataset["name"] for dataset in config["eval"]["datasets"]]
requested_names = [name for name in ("aime24", "math500") if name in set(sys.argv[3].split(","))]
if configured_names != requested_names or config["eval"]["defaults"].get("max_response_len") != 32768:
    raise SystemExit(1)
PY
  then
    echo "M2RL single-task training/evaluation data is already prepared: ${INDEX_PATH}"
    exit 0
  fi
fi

PREPARE_ARGS=(
  --rl-manifest "${RL_MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --tasks math code science
  --eval "math=${MATH500_PATH}"
  --eval-name "math=math500"
  --eval-rm-type "math=deepscaler"
  --eval-samples "math=1"
  --eval-max-response-len-override "math=32768"
  --eval-temperature "math=0"
  --eval-top-p-override "math=1"
  --exclude-eval-overlap math
  --skip-eval code
  --eval-max-response-len 16384
  --eval-top-p 0.7
  --seed "${SEED}"
)
if [[ "${GPQA_READY}" == "1" ]]; then
  PREPARE_ARGS+=(
    --eval "science=${GPQA_PATH}"
    --eval-name "science=gpqa"
    --eval-rm-type "science=gpqa"
    --eval-samples "science=4"
  )
else
  PREPARE_ARGS+=(--skip-eval science)
fi

python3 "${SCRIPT_DIR}/prepare_single_task_data.py" "${PREPARE_ARGS[@]}"

# ``prepare_single_task_data.py`` uses MATH-500 as the overlap reference and
# therefore writes a one-dataset config. Restore the selected active alias and
# all reusable single/combined variants after materializing the train view.
python3 "${SCRIPT_DIR}/prepare_m2rl_eval_data.py" \
  --output-dir "${EVAL_DATA_DIR}" \
  --datasets aime24 math500 \
  --seed "${SEED}" \
  --math-eval-config-dir "${OUTPUT_DIR}/math" \
  --math-eval-datasets "${MATH_EVAL_DATASET_LIST[@]}"

echo "Prepared benchmark-disjoint Math plus Code/Science training manifests and online eval metadata: ${INDEX_PATH}"
