#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
MOPD_ROOT="${MOPD_OUTPUT_ROOT:-${SLIME_ROOT}/outputs/mopd_gpas_64k_v3}"
REPORT_ROOT="${REPORT_ROOT:-${SLIME_ROOT}/outputs/mopd_reports_64k_v3}"
CONTROLLED_ROOT="${MOPD_CONTROLLED_RESULTS_ROOT:-${EXAMPLE_DIR}/generated/controlled/experiments}"
mkdir -p "${REPORT_ROOT}"
python3 "${EXAMPLE_DIR}/analyze_mopd.py" \
  --root "${MOPD_ROOT}" --output "${REPORT_ROOT}/mopd.json"
python3 "${EXAMPLE_DIR}/analyze_capability.py" \
  --root "${MOPD_ROOT}" --output "${REPORT_ROOT}/mopd_capability.json"
python3 "${EXAMPLE_DIR}/analyze_frozen_bank.py" \
  --root "${MOPD_ROOT}" --mopd-report "${REPORT_ROOT}/mopd.json" \
  --output "${REPORT_ROOT}/frozen_bank.json"
python3 "${EXAMPLE_DIR}/plot_results.py" \
  --mopd "${REPORT_ROOT}/mopd.json" \
  --capability "${REPORT_ROOT}/mopd_capability.json" \
  --frozen-bank "${REPORT_ROOT}/frozen_bank.json" \
  --controlled-sampling "${CONTROLLED_ROOT}/controlled_optimizer_sampling_results.csv" \
  --controlled-moments "${CONTROLLED_ROOT}/controlled_adamw_moment_results.csv" \
  --output-dir "${REPORT_ROOT}/figures"
