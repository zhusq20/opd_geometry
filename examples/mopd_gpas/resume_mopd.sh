#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ $# == 1 ]] || { echo "Usage: $0 CONFIG_ID" >&2; exit 2; }
export MOPD_RESUME=1
exec bash "${EXAMPLE_DIR}/run_mopd.sh" "$1"
