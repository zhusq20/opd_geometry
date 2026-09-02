#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for stage in warm middle late; do
  bash "${EXAMPLE_DIR}/run_frozen_bank.sh" "${stage}"
done
