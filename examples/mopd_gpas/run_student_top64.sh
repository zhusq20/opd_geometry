#!/usr/bin/env bash
# Student Top64 reverse-KL distillation using the normalized detached advantage.
# Same optimizer and response/token reductions as Top16; no PPO or new domain weights.
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ID="${1:-m-tk64-dr}"
case "${CONFIG_ID}" in
  s-tk64|m-tk64-dr|m-tk64-dt|m-tk64-gt) ;;
  *) echo "Usage: $0 [s-tk64|m-tk64-dr|m-tk64-dt|m-tk64-gt]" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Expected at most one experiment condition." >&2; exit 2; }
exec bash "${EXAMPLE_DIR}/run_mopd.sh" "${CONFIG_ID}"
