#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
export PYTHONPATH="${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${EXAMPLE_DIR}/convert_teachers.py" "$@"
