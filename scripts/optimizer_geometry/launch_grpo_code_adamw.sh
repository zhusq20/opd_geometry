#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR=/workspace/dev/iclr2027/slime_opd_geometry
RAY_TEMP_DIR=/tmp/ray_gca8_retry1
RAY_JOB_ADDRESS=http://127.0.0.1:8267
RAY_HEAD_PID=""

cleanup() {
  if [[ -n "${RAY_HEAD_PID}" ]]; then
    kill -TERM "${RAY_HEAD_PID}" 2>/dev/null || true
    wait "${RAY_HEAD_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export CUDA_VISIBLE_DEVICES=2,3,5,7
export RAY_TMPDIR="${RAY_TEMP_DIR}"

ray start \
  --head \
  --node-ip-address 127.0.0.1 \
  --port 6381 \
  --ray-client-server-port 21001 \
  --min-worker-port 21002 \
  --max-worker-port 29999 \
  --num-gpus 4 \
  --disable-usage-stats \
  --dashboard-host 127.0.0.1 \
  --dashboard-port 8267 \
  --dashboard-agent-listen-port 52366 \
  --dashboard-agent-grpc-port 52367 \
  --runtime-env-agent-port 52368 \
  --metrics-export-port 52369 \
  --temp-dir "${RAY_TEMP_DIR}" \
  --block &
RAY_HEAD_PID=$!

ray_ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
    echo "Code Ray head exited before its Jobs API became ready." >&2
    wait "${RAY_HEAD_PID}" || true
    exit 1
  fi
  if ray job list --address "${RAY_JOB_ADDRESS}" >/dev/null 2>&1; then
    ray_ready=1
    break
  fi
  sleep 1
done
if [[ "${ray_ready}" != "1" ]]; then
  echo "Code Ray Jobs API did not become ready at ${RAY_JOB_ADDRESS}." >&2
  exit 1
fi
sleep 3

cd "${SLIME_DIR}"
export PYTHONUNBUFFERED=1
export AVAILABLE_CUDA_DEVICES=2,3,5,7
export RAY_ADDRESS="${RAY_JOB_ADDRESS}"
export START_RAY=0
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
export M2RL_SANDBOX_PREFLIGHT_MARKER=/tmp/slime-sandbox/sandboxfusion_preflight.json
export TASK=code
export RL_ALGORITHM=grpo
export OPTIMIZERS=adamw

bash examples/optimizer_geometry/run_single_task_rl.sh
