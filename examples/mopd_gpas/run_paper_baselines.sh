#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for config in d3_mopd open_mopd; do
  bash "${EXAMPLE_DIR}/run_mopd.sh" "${config}"
done
