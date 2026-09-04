#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for target in \
  initial_student teacher_math teacher_if teacher_qwen3_4b \
  uniform gpas cost_gpas raw_noise loss_gap std_mopd d3_mopd open_mopd; do
  bash "${EXAMPLE_DIR}/run_capability_eval.sh" "${target}"
done
