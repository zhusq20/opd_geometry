#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for target in initial_student teacher_math teacher_code teacher_if teacher_science; do
  bash "${EXAMPLE_DIR}/run_capability_eval.sh" "${target}"
done
for target in uniform-s1 gpas-s1; do
  for responses in 16000 32000; do
    MOPD_CAPABILITY_RESPONSE="${responses}" bash "${EXAMPLE_DIR}/run_capability_eval.sh" "${target}"
  done
done
for target in gpas-raw-s1 d3-fixed-s1; do
  MOPD_CAPABILITY_RESPONSE=32000 bash "${EXAMPLE_DIR}/run_capability_eval.sh" "${target}"
done
