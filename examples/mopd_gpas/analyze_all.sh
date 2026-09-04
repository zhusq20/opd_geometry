#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
export PYTHONPATH="${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MOPD_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_v4}"
REPORT_ROOT="${REPORT_ROOT:-${SLIME_ROOT}/outputs/mopd_reports_v4}"
PROTOCOL="${MOPD_GENERATED_DIR:-${SLIME_ROOT}/local/mopd_generated}/protocol.json"

mkdir -p "${REPORT_ROOT}"
python3 "${EXAMPLE_DIR}/analyze_mopd.py" \
  --root "${MOPD_ROOT}" --protocol "${PROTOCOL}" --output "${REPORT_ROOT}/mopd.json"
python3 "${EXAMPLE_DIR}/analyze_capability.py" \
  --root "${MOPD_ROOT}" --output "${REPORT_ROOT}/mopd_capability.json" \
  --mopd-report "${REPORT_ROOT}/mopd.json"
python3 "${EXAMPLE_DIR}/analyze_heldout_variance.py" \
  --root "${MOPD_ROOT}/heldout_variance" \
  --output "${REPORT_ROOT}/heldout_gradient_variance.json"

PLOT_ARGS=(
  --mopd "${REPORT_ROOT}/mopd.json"
  --capability "${REPORT_ROOT}/mopd_capability.json"
  --output-dir "${REPORT_ROOT}/figures"
)
PLOT_ARGS+=(--heldout-variance "${REPORT_ROOT}/heldout_gradient_variance.json")
python3 "${EXAMPLE_DIR}/plot_results.py" "${PLOT_ARGS[@]}"
