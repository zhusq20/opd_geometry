#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ID="${1:-m-tk-dr}"
case "${CONFIG_ID}" in
  s-pg|s-tk|m-pg|m-tk-dr|m-tk-dt|m-tk-gt|s-tk64|m-tk64-dr|m-tk64-dt|m-tk64-gt|m-intersection64-dr)
    exec bash "${EXAMPLE_DIR}/_launch_paper.sh" "${CONFIG_ID}" ;;
  uniform|uniform-s1) CONFIG_ID=uniform-s1; ALLOCATION=uniform ;;
  gpas|gpas-s1) CONFIG_ID=gpas-s1; ALLOCATION=gpas ;;
  raw_noise|gpas-raw-s1) CONFIG_ID=gpas-raw-s1; ALLOCATION=raw_noise ;;
  d3_fixed|d3-fixed-s1) CONFIG_ID=d3-fixed-s1; ALLOCATION=d3_fixed ;;
  gpas-smoke) export MOPD_SMOKE_TEST=1; ALLOCATION=gpas ;;
  gpas-quick-smoke) export MOPD_QUICK_SMOKE_TEST=1; ALLOCATION=gpas ;;
  *) echo "Usage: $0 {s-pg|s-tk|m-pg|m-tk-dr|m-tk-dt|m-tk-gt|s-tk64|m-tk64-dr|m-tk64-dt|m-tk64-gt|m-intersection64-dr|uniform-s1|gpas-s1|gpas-raw-s1|d3-fixed-s1|gpas-smoke|gpas-quick-smoke}" >&2; exit 2 ;;
esac
exec bash "${EXAMPLE_DIR}/_launch_mopd.sh" "${CONFIG_ID}" "${ALLOCATION}"
