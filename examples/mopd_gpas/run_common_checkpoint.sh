#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MOPD_COMMON_CHECKPOINT=1
exec bash "${EXAMPLE_DIR}/_launch_mopd.sh" common-checkpoint-250 uniform
