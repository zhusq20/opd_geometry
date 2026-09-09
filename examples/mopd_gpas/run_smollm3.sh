#!/usr/bin/env bash
# SmolLM3 MixSFT, three routed RL teachers, PG / Top64 intersection.
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  pg) CONDITION=m-pg ;;
  intersection64) CONDITION=m-intersection64-dr ;;
  *) echo "Usage: $0 {pg|intersection64} [train|smoke]" >&2; exit 2 ;;
esac
export MOPD_PROFILE=smollm3
source "${EXAMPLE_DIR}/_profile.sh"
python3 - "${MOPD_ASSET_ROOT}/assets.json" "${MOPD_GENERATED_DIR}/protocol.json" "${MOPD_HF_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

assets_path, protocol_path, student = map(Path, sys.argv[1:])
if not assets_path.is_file() or not protocol_path.is_file():
    raise SystemExit('Run fetch-assets and prepare with the SmolLM3 MixSFT profile first')
assets = json.loads(assets_path.read_text())
protocol = json.loads(protocol_path.read_text())
expected = {'repository': 'BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT',
            'revision': 'c9e7bad031667828656ead188d7e8ea162c048a4'}
if assets.get('student') != expected or (protocol.get('assets') or {}).get('student') != expected:
    raise SystemExit('This entry requires the pinned MixSFT student. Use fresh assets/generated paths; do not reuse Base runs')
if Path(protocol['student']['hf_config']['path']).resolve() != (student / 'config.json').resolve():
    raise SystemExit('Prepared student path differs from MOPD_HF_CHECKPOINT; rerun prepare with the correct paths')
PY
# Verifiers do not enter either objective. Enable explicitly for accuracy curves.
export MOPD_OBSERVE_TASK_REWARDS="${MOPD_OBSERVE_TASK_REWARDS:-0}"
export MOPD_EVAL_DURING_TRAINING="${MOPD_EVAL_DURING_TRAINING:-0}"
case "${2:-train}" in
  train) ;;
  smoke)
    export MOPD_TOTAL_STEPS=2 MOPD_CHECKPOINT_STEPS=2
    export MOPD_RESPONSES_PER_UPDATE=12 MOPD_MAX_RESPONSE_LEN=128
    export MOPD_EVAL_DURING_TRAINING=0
    export MOPD_RUN_ID="${MOPD_RUN_ID:-smoke-${CONDITION}-s${MOPD_SEED:-42}}"
    ;;
  *) echo "Expected train or smoke" >&2; exit 2 ;;
esac
exec bash "${EXAMPLE_DIR}/run_mopd.sh" "${CONDITION}"
