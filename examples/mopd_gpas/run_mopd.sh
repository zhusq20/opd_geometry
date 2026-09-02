#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ID="${1:-}"
case "${CONFIG_ID}" in
  uniform_k1_conventional) K=1; ALLOCATION=uniform; ADAMW_STATE=conventional ;;
  uniform_k1_taskwise) K=1; ALLOCATION=uniform; ADAMW_STATE=taskwise ;;
  gpas_k1_taskwise) K=1; ALLOCATION=gpas; ADAMW_STATE=taskwise ;;
  cost_gpas_k1_taskwise) K=1; ALLOCATION=cost_gpas; ADAMW_STATE=taskwise ;;
  uniform_k2_taskwise) K=2; ALLOCATION=uniform; ADAMW_STATE=taskwise ;;
  cost_gpas_k2_taskwise) K=2; ALLOCATION=cost_gpas; ADAMW_STATE=taskwise ;;
  all_k4_taskwise) K=4; ALLOCATION=all; ADAMW_STATE=taskwise ;;
  all_k4_conventional) K=4; ALLOCATION=all; ADAMW_STATE=conventional ;;
  *)
    echo "Usage: $0 {uniform_k1_conventional|uniform_k1_taskwise|gpas_k1_taskwise|cost_gpas_k1_taskwise|uniform_k2_taskwise|cost_gpas_k2_taskwise|all_k4_taskwise|all_k4_conventional}" >&2
    exit 2
    ;;
esac
exec bash "${EXAMPLE_DIR}/_launch_mopd.sh" train "${CONFIG_ID}" "${K}" "${ALLOCATION}" "${ADAMW_STATE}"
