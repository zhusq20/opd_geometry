#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ID="${1:-}"
case "${CONFIG_ID}" in
  uniform|gpas|cost_gpas|raw_noise|loss_gap|std_mopd|d3_mopd|open_mopd) ALLOCATION="${CONFIG_ID}" ;;
  gpas-smoke) export MOPD_SMOKE_TEST=1; ALLOCATION=gpas ;;
  gpas-quick-smoke) export MOPD_QUICK_SMOKE_TEST=1; ALLOCATION=gpas ;;
  *) echo "Usage: $0 {uniform|gpas|cost_gpas|raw_noise|loss_gap|std_mopd|d3_mopd|open_mopd|gpas-smoke|gpas-quick-smoke}" >&2; exit 2 ;;
esac
exec bash "${EXAMPLE_DIR}/_launch_mopd.sh" "${CONFIG_ID}" "${ALLOCATION}"
