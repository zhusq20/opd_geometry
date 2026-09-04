#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for step in 50 250 500; do
  bash "${EXAMPLE_DIR}/run_heldout_variance.sh" "${step}"
done
