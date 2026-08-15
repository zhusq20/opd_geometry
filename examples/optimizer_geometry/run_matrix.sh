#!/usr/bin/env bash
# Sequentially submit the complete paper matrix. Override the three space-
# separated lists to run a subset, e.g. OPTIMIZERS="adamw muon".

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_LAUNCHER="${EXPERIMENT_LAUNCHER:-${SCRIPT_DIR}/run_experiment.sh}"
read -r -a OPTIMIZER_LIST <<< "${OPTIMIZERS:-adamw sgd muon}"
read -r -a ALGORITHM_LIST <<< "${ALGORITHMS:-grpo ppo opd}"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44 45 46}"

for seed in "${SEED_LIST[@]}"; do
  for algorithm in "${ALGORITHM_LIST[@]}"; do
    for optimizer in "${OPTIMIZER_LIST[@]}"; do
      echo "Submitting algorithm=${algorithm} optimizer=${optimizer} seed=${seed}"
      OPTIMIZER="${optimizer}" \
      ALGORITHM="${algorithm}" \
      SEED="${seed}" \
      RUN_NAME="${RUN_NAME_PREFIX:-}${algorithm}_${optimizer}_${BATCH_PROFILE:-responsive16}_seed${seed}" \
      bash "${EXPERIMENT_LAUNCHER}"
    done
  done
done
