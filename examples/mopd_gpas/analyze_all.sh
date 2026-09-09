#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
export PYTHONPATH="${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${EXAMPLE_DIR}/_profile.sh"
python3 "${EXAMPLE_DIR}/analyze_paper.py" \
  --root "${MOPD_OUTPUT_ROOT}" \
  --output "${REPORT_ROOT:-${MOPD_OUTPUT_ROOT}_reports}"
