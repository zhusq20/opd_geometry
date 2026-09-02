#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXAMPLE_DIR}/_launch_mopd.sh" warm warm_start 1 uniform taskwise
