#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${EXAMPLE_DIR}/_profile.sh"
exec python3 "${EXAMPLE_DIR}/fetch_profile.py"
