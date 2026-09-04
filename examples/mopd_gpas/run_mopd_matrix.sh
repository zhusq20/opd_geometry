#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for config in \
  uniform \
  gpas \
  cost_gpas \
  raw_noise \
  loss_gap \
  std_mopd \
  d3_mopd \
  open_mopd; do
  bash "${EXAMPLE_DIR}/run_mopd.sh" "${config}"
done
