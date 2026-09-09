#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for config in s-pg s-tk m-pg m-tk-dr m-tk-dt m-tk-gt; do
  bash "${EXAMPLE_DIR}/run_mopd.sh" "${config}"
done
