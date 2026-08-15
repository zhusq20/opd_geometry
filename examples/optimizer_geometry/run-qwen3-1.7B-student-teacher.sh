#!/usr/bin/env bash
# Preferred generic entry point. The compatibility launcher contains the shared
# implementation so old commands keep working unchanged.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run-qwen3-1.7B-student-8B-teacher.sh" "$@"
