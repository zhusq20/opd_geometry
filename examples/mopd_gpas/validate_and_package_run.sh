#!/usr/bin/env bash
set -euo pipefail

CONFIG_ID="${1:-}"
case "${CONFIG_ID}" in
  uniform|gpas|cost_gpas|raw_noise|loss_gap|std_mopd|d3_mopd|open_mopd) ;;
  *) echo "Usage: $0 CONFIG_ID" >&2; exit 2 ;;
esac

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
OUTPUT_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_v4}"
PACKAGE_DIR="${MOPD_PACKAGE_DIR:-${SLIME_ROOT}/outputs/mopd_packages_v4}"
RUN_NAME="${CONFIG_ID}-seed42"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
CAPABILITY_DIR="${RUN_DIR}/capability_eval/response_32000"

python3 - "${RUN_DIR}" "${CAPABILITY_DIR}" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
capability_dir = pathlib.Path(sys.argv[2])
completion = json.loads((run_dir / "run_complete.json").read_text(encoding="utf-8"))
if completion.get("status") != "complete":
    raise SystemExit(f"training is not complete: {completion}")
rows = [json.loads(line) for line in (run_dir / "allocation/allocation.jsonl").read_text().splitlines() if line]
if not rows or int(rows[-1]["attempted_responses_after"]) != 32_000:
    raise SystemExit("training did not finish at exactly 32,000 attempted responses")
capability = json.loads((capability_dir / "run_complete.json").read_text(encoding="utf-8"))
if capability.get("status") != "complete":
    raise SystemExit(f"capability evaluation is not complete: {capability}")
print(f"validated {run_dir.name}: training and capability evaluation are complete")
PY

mkdir -p "${PACKAGE_DIR}"
tar -czf "${PACKAGE_DIR}/${RUN_NAME}-analysis.tar.gz" -C "${OUTPUT_ROOT}" \
  "${RUN_NAME}/run_complete.json" \
  "${RUN_NAME}/provenance" \
  "${RUN_NAME}/allocation" \
  "${RUN_NAME}/metrics" \
  "${RUN_NAME}/teacher_loss_eval" \
  "${RUN_NAME}/capability_eval"
echo "Analysis package written to ${PACKAGE_DIR}/${RUN_NAME}-analysis.tar.gz"
