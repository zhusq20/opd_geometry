#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
source "${EXAMPLE_DIR}/_profile.sh"
TARGET="${1:-initial_student}"
case "${TARGET}" in
  initial_student|teacher_*) CHECKPOINT_STEP=0 ;;
  *) CHECKPOINT_STEP="${MOPD_EVAL_STEP:-500}" ;;
esac
MODEL="${MOPD_EVAL_MODEL_PATH:-}"
if [[ -z "${MODEL}" ]]; then
  case "${TARGET}" in
    initial_student) MODEL="${MOPD_HF_CHECKPOINT}" ;;
    teacher_*) MODEL="${MOPD_TEACHER_HF_ROOT}/${TARGET#teacher_}" ;;
    *)
      MODEL="$(python3 - "${MOPD_OUTPUT_ROOT}/${TARGET}/checkpoints/mopd_checkpoint_index.json" "${CHECKPOINT_STEP}" <<'PY'
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text())
matches = [row for row in rows if int(row['optimizer_step']) == int(sys.argv[2]) and row.get('hf_checkpoint')]
if len(matches) != 1:
    raise SystemExit('Expected one exported HF checkpoint at the requested MOPD_EVAL_STEP')
print(matches[0]['hf_checkpoint'])
PY
)" ;;
  esac
fi
EVAL_CONFIG="${MOPD_CAPABILITY_EVAL_CONFIG}"
if [[ "${TARGET}" == teacher_* ]]; then
  EVAL_CONFIG="${MOPD_GENERATED_DIR}/capability_${TARGET}.yaml"
  python3 - "${MOPD_CAPABILITY_EVAL_CONFIG}" "${EVAL_CONFIG}" "${TARGET#teacher_}" <<'PY'
import pathlib, sys, yaml
config=yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
tokens={'math': ('math','aime'), 'code': ('code',), 'if': ('ifbench','ifeval'), 'science': ('gpqa',)}[sys.argv[3]]
config['eval']['datasets']=[row for row in config['eval']['datasets'] if any(t in row['name'] for t in tokens)]
config['eval']['defaults']['chat_template_suffix_to_remove']=None
pathlib.Path(sys.argv[2]).write_text(yaml.safe_dump(config,sort_keys=False))
PY
fi
OUTPUT="${MOPD_OUTPUT_ROOT}/${TARGET}/capability_eval/step_${CHECKPOINT_STEP}"
export CUDA_VISIBLE_DEVICES="${CAPABILITY_CUDA_VISIBLE_DEVICES:-${MOPD_INFERENCE_GPU:-1}}"
export PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH:-/root/Megatron-LM}${PYTHONPATH:+:${PYTHONPATH}}"
source "${SLIME_ROOT}/scripts/models/${MOPD_MODEL_CONFIG}"
CMD=(python3 "${SLIME_ROOT}/train.py" "${MODEL_ARGS[@]}"
  --hf-checkpoint "${MODEL}" --load "${MODEL}" --no-load-optim --no-load-rng --start-rollout-id 0
  --debug-rollout-only --mopd-profile "${MOPD_PROFILE}" --prompt-data "${MOPD_GENERATED_DIR}/train.yaml"
  --data-source-path slime_plugins.m2rl.data_source.MultiTaskRolloutDataSource
  --input-key prompt --label-key label --metadata-key metadata --rollout-global-dataset
  --num-rollout 0 --rollout-batch-size 1 --global-batch-size 1 --n-samples-per-prompt 1
  --eval-config "${EVAL_CONFIG}" --eval-interval 1 --eval-max-response-len 32768
  --eval-checkpoint-step "${CHECKPOINT_STEP}"
  --eval-function-path slime_plugins.mopd.eval.generate_capability_eval
  --eval-max-concurrency "${EVAL_MAX_CONCURRENCY:-8}" --log-passrate
  --m2rl-reward-config "${MOPD_REWARD_CONFIG:-${MOPD_GENERATED_DIR}/rewards.yaml}"
  --eval-artifact-dir "${OUTPUT}/artifacts" --metrics-output-dir "${OUTPUT}/metrics"
  --completion-marker-path "${OUTPUT}/run_complete.json"
  --experiment-task mopd_capability --experiment-teacher "${TARGET}"
  --experiment-condition capability --experiment-name "${MOPD_PROFILE}-${TARGET}-eval"
  --experiment-optimizer none --experiment-data-index "${MOPD_GENERATED_DIR}/protocol.json"
  --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --num-gpus-per-node 1
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 --colocate
  --sglang-context-length "${MOPD_CONTEXT_LENGTH}" --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.7}"
  --seed "${MOPD_SEED:-42}")
if [[ "${USE_WANDB}" == 1 ]]; then
  CMD+=(--use-wandb --wandb-mode "${WANDB_MODE:-online}" --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${MOPD_PROFILE}-paper" --wandb-run-name "${TARGET}-eval-step${CHECKPOINT_STEP}")
  [[ -z "${WANDB_ENTITY:-}" ]] || CMD+=(--wandb-team "${WANDB_ENTITY}")
fi
printf '%q ' "${CMD[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" != 1 ]] || exit 0
exec "${CMD[@]}"
