#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for config in \
  uniform_k1_conventional uniform_k1_taskwise gpas_k1_taskwise cost_gpas_k1_taskwise \
  uniform_k2_taskwise cost_gpas_k2_taskwise all_k4_taskwise all_k4_conventional; do
  bash "${EXAMPLE_DIR}/run_capability_eval.sh" "${config}"
done
