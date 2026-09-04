#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

OFFICIAL_REVISION="4809a96cf85a869106ff0ff3f37d0a51e12010ae"
FULL_ROOT="${OPEN_MOPD_FULL_ROOT:-${REPO_ROOT}/local/open_mopd_full}"
SOURCE_ROOT="${OPEN_MOPD_SOURCE_ROOT:-${FULL_ROOT}/source}"
ASSET_ROOT="${OPEN_MOPD_ASSET_ROOT:-${FULL_ROOT}/assets}"
OUTPUT_ROOT="${OPEN_MOPD_OUTPUT_ROOT:-${FULL_ROOT}/runs/open_mopd_full}"
PYTHON_BIN="${OPEN_MOPD_PYTHON:-python3}"
EXECUTE=0

usage() {
  echo "Usage: $0 [--dry-run|--run]"
  echo "  --dry-run  print the frozen paper command (default)"
  echo "  --run      validate the pinned source/assets and launch on 1x8 GPUs"
  echo
  echo "Optional roots: OPEN_MOPD_FULL_ROOT, OPEN_MOPD_SOURCE_ROOT, OPEN_MOPD_ASSET_ROOT, OPEN_MOPD_OUTPUT_ROOT"
}

die() {
  echo "[open-mopd-full] error: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --run) EXECUTE=1 ;;
    --dry-run) EXECUTE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

STUDENT="${ASSET_ROOT}/models/mixsft"
MATH_TEACHER="${ASSET_ROOT}/models/teachers/math"
CODE_TEACHER="${ASSET_ROOT}/models/teachers/code"
IF_TEACHER="${ASSET_ROOT}/models/teachers/if"
TRAIN_FILE="${ASSET_ROOT}/data/rl_prompt_mix/train.parquet"
VAL_FILE="${ASSET_ROOT}/data/eval/math/aime24.parquet"
SAMPLER="${SOURCE_ROOT}/training/verl/verl/utils/dataset/domain_weighted_sampler.py"

cmd=(
  "${PYTHON_BIN}" -m verl.trainer.main_ppo
  "algorithm.adv_estimator=token_reward_direct"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.train_batch_size=1024"
  "data.max_prompt_length=2048"
  "data.max_response_length=16384"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.sampler.class_path=${SAMPLER}"
  "data.sampler.class_name=DomainWeightedSampler"
  "data.dataloader_num_workers=0"
  "+data.domain_weights={math:2,code:2,if:1}"
  "actor_rollout_ref.model.path=${STUDENT}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "+actor_rollout_ref.rollout.reward_mode=mt_opd"
  "actor_rollout_ref.rollout.n=1"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.top_p=0.99"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.max_model_len=32768"
  "+actor_rollout_ref.rollout.log_prob_top_k=16"
  "+actor_rollout_ref.rollout.top_k_strategy=only_stu"
  "+actor_rollout_ref.rollout.reward_weight_mode=student_p"
  "+actor_rollout_ref.rollout.kl_estimator=k1"
  "++actor_rollout_ref.rollout.train_kwargs_by_data_source.math_dapo_boxed.max_tokens=16384"
  "++actor_rollout_ref.rollout.train_kwargs_by_data_source.primeintellect.max_tokens=16384"
  "++actor_rollout_ref.rollout.train_kwargs_by_data_source.taco.max_tokens=16384"
  "++actor_rollout_ref.rollout.train_kwargs_by_data_source.nemotron_if_rl.max_tokens=2048"
  "actor_rollout_ref.actor.ppo_mini_batch_size=256"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.clip_ratio=0.2"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.optim.lr=1.5e-6"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=constant"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
  "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0"
  "+actor_rollout_ref.actor.opd_refresh_advantage=True"
  "+actor_rollout_ref.actor.opd_reward_weight_mode=student_p"
  "reward_model.enable=True"
  "reward_model.model.path=${MATH_TEACHER}"
  "reward_model.model.use_remove_padding=True"
  "reward_model.use_dynamic_bsz=True"
  "reward_model.forward_max_token_len_per_gpu=32768"
  "+mt_opd.teacher_domains=[math,code,if]"
  "+mt_opd.n_additional_teachers=2"
  "+mt_opd.domain_weighting=domain_routing"
  "+mt_opd.target_share_domains=[math,code,if]"
  "+mt_opd.target_share_values=[0.3333333333333333,0.3333333333333333,0.3333333333333333]"
  "+mt_opd.normalize_reward_scale=1.0"
  "+mt_opd.reward_scale_stat=mean"
  "+mt_opd.reward_scale_direction=multiply"
  "+mt_opd.reward_scale_anchored=False"
  "+mt_opd.conflict_policy=none"
  "+mt_reward_model_1.enable=True"
  "+mt_reward_model_1.model.path=${CODE_TEACHER}"
  "+mt_reward_model_1.model.input_tokenizer=null"
  "+mt_reward_model_1.model.use_remove_padding=True"
  "+mt_reward_model_1.model.fsdp_config.param_offload=True"
  "+mt_reward_model_2.enable=True"
  "+mt_reward_model_2.model.path=${IF_TEACHER}"
  "+mt_reward_model_2.model.input_tokenizer=null"
  "+mt_reward_model_2.model.use_remove_padding=True"
  "+mt_reward_model_2.model.fsdp_config.param_offload=True"
  "trainer.n_gpus_per_node=8"
  "trainer.nnodes=1"
  "trainer.total_training_steps=600"
  "trainer.val_before_train=False"
  "trainer.test_freq=-1"
  "trainer.save_freq=50"
  "trainer.default_local_dir=${OUTPUT_ROOT}/checkpoints"
  "trainer.project_name=Open-MOPD-paper"
  "trainer.experiment_name=open-mopd-full"
  "trainer.logger=['console']"
)

echo "[open-mopd-full] official revision: ${OFFICIAL_REVISION}"
echo "[open-mopd-full] recipe: ${SCRIPT_DIR}/recipe.json"
printf '[open-mopd-full]'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${EXECUTE}" == 0 ]]; then
  echo "[open-mopd-full] dry-run only; pass --run to execute"
  exit 0
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
[[ -d "${SOURCE_ROOT}/.git" ]] || die "official source missing; run fetch.sh source"
ACTUAL_REVISION="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_REVISION}" == "${OFFICIAL_REVISION}" ]] \
  || die "official source revision is ${ACTUAL_REVISION}, expected ${OFFICIAL_REVISION}"
for required in "${STUDENT}" "${MATH_TEACHER}" "${CODE_TEACHER}" "${IF_TEACHER}" "${TRAIN_FILE}" "${VAL_FILE}" "${SAMPLER}"; do
  [[ -e "${required}" ]] || die "required asset missing: ${required}"
done

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${SOURCE_ROOT}/training/verl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
cd "${SOURCE_ROOT}/training"
exec "${cmd[@]}"
