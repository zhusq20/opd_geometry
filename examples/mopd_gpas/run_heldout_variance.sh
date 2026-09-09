#!/usr/bin/env bash
# Compatibility entry point for the current Uniform/250 comparison.
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ "${1:-250}" == 250 || "${1:-all}" == all ]] || { echo "The core comparison uses only Uniform/250." >&2; exit 2; }
exec bash "${EXAMPLE_DIR}/run_common_checkpoint.sh"
