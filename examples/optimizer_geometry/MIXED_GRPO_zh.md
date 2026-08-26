# Code + Math + QA + IF 同 batch 混合 GRPO

主入口从 Qwen3-1.7B base model 开始训练，Code 不再只是序贯实验的 warm start：

```bash
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
export M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json
export FOUR_HEALTHY_GPUS=1,2,3,4  # 启动前按 nvidia-smi 的实际健康/空闲卡替换

AVAILABLE_CUDA_DEVICES="$FOUR_HEALTHY_GPUS" \
  bash examples/optimizer_geometry/run_mixed_grpo.sh
```

训练期间的 pre-eval 和 periodic eval 均被关闭。训练成功后，入口默认只对最终 checkpoint 运行一次
Code/Math/QA/IF 联合评测。若希望把评测安排到另一个时间，可训练时设置 `RUN_FINAL_EVAL=0`，之后执行：

```bash
AVAILABLE_CUDA_DEVICES="$FOUR_HEALTHY_GPUS" \
  bash examples/optimizer_geometry/evaluate_mixed_grpo.sh
```

## 模型与等量数据合同

现有序贯脚本的路径是 `Code checkpoint -> Math -> Knowledge/Science -> IF`。Code checkpoint 的
provenance 表明它由以下 base model 开始训练：

```text
HF/tokenizer:  /workspace/dev/checkpoints/Qwen3-1.7B
torch-dist:    /workspace/dev/checkpoints/Qwen3-1.7B_torch_dist
model config:  scripts/models/qwen3-1.7B.sh
thinking:      false
```

混合入口从同一个 `Qwen3-1.7B_torch_dist` base checkpoint 开始，不加载 Code 训练后的权重。Code
checkpoint 只用来校验原 Code run 的数据流、batch、seed、模型和超参数合同。

| 训练域 | 原始可用训练量 | 混合 run 实际消费 | reward |
|---|---:|---:|---|
| Code | 19,169 | 4,800 | `unit_test` / SandboxFusion |
| Math | 18,231（已去重并排除 eval overlap） | 4,800 | `deepscaler` |
| QA | 19,670（代码中的 canonical 名为 `science`） | 4,800 | `gpqa` |
| IF | 16,575 | 4,800 | `ifevalg` |

四个域严格各消费 4,800 个 prompt group，总量 19,200。Math、QA、IF 复用序贯实验已经冻结的前
4,800 行 view；Code 使用与原 Code run 相同的 seed-42 全源过滤、shuffle 流，并恰好消费其前
4,800 个可用 prompt。`required_samples: 4800` 会在 chat-template/长度过滤后再次检查容量，避免某个域
不足时静默循环采样。

只检查计划与数据合同、不启动训练：

```bash
PLAN_ONLY=1 RUN_FINAL_EVAL=0 \
  bash examples/optimizer_geometry/run_mixed_grpo.sh
```

## 混合发生在哪里

原多任务 manifest 的 `uniform + unit: batch` 会先抽一个任务，再把该任务索引重复到整个 rollout
batch，所以一个 batch 内任务相同，任务只在不同 batch 之间切换。

本实验使用 `stratified + unit: prompt`。每个 rollout/update 的精确构成为：

```text
16 prompt groups
  = 4 Code + 4 Math + 4 QA + 4 IF

每个 prompt group
  = 同一个 prompt 的 16 条采样 response

每个 optimizer update
  = 16 prompts x 16 responses
  = 256 trajectories
  = 每个域 64 trajectories
```

因此“同 batch 混合”与 GRPO 的“组内相同数据”同时满足：四个任务在同一个 rollout batch 和同一个
optimizer update 中出现，但每个 GRPO group 内的 16 条 response 始终对应同一个 prompt。四个域的
4,800 个 prompt 都消费完时正好得到 1,200 个 on-policy updates；每次 rollout 后只做一次 update。

## 启动 Code Sandbox

SandboxFusion 必须在宿主机启动；不要在训练容器内运行 Docker 启动器。首次部署、补丁变化或 pin
失效时，在宿主机运行：

```bash
cd /path/to/slime_opd_geometry  # 宿主机上的真实仓库路径
export SANDBOX_STATE="$HOME/.local/state/slime-opd-geometry/sandboxfusion"

SANDBOXFUSION_STATE_DIR="$SANDBOX_STATE" \
  bash scripts/optimizer_geometry/rebuild_and_start_sandboxfusion_on_host.sh
```

已有匹配当前补丁的固定 image pin 时，只重新启动并执行完整安全探针：

```bash
SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
SANDBOX_PREFLIGHT_MARKER="$SANDBOX_STATE/sandboxfusion_preflight.json" \
  bash examples/optimizer_geometry/start_sandboxfusion.sh
```

训练容器必须使用 `--network host`，并把 marker 只读挂载到容器。例如将宿主机
`$SANDBOX_STATE` 挂到 `/workspace/sandboxfusion-state` 后，在训练容器内检查：

```bash
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
export M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json

test -s "$M2RL_SANDBOX_PREFLIGHT_MARKER"
curl --noproxy '*' -fsS http://127.0.0.1:8080/v1/ping
```

预期返回 `"pong"`。训练预检会校验 marker 的 schema、时间、固定 image、cgroup v2、namespace、网络
隔离和内存限制；缺失或失效时 fail closed。完整容器挂载与安全说明见
`examples/optimizer_geometry/SANDBOXFUSION_CGROUP2_zh.md`。

本次检查时，当前容器中的 `127.0.0.1:8080` 返回 `"pong"`，且
`/workspace/sandboxfusion-state/sandboxfusion_preflight.json` 为 `safe=true`；marker 的探针时间是
2026-08-17。宿主机重启、服务重启、补丁变化或 marker 超龄后仍须重新执行宿主机启动/探针。

## 四张 96 GiB GPU 的 batch 建议

当前机器实际报告的是四卡使用场景下的 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，每张约
97,887 MiB；不是常见的 48 GiB RTX A6000。物理 GPU 0 当前有不可纠正 ECC 计数，不能使用；launcher
会检查所选卡的剩余显存与 ECC，并 fail closed。正式可比 run 固定：

```text
data parallel / rollout engines: 4 / 4
tensor parallel:                 1
rollout batch:                   16 prompt groups
GRPO group size:                 16 responses
global batch:                    256 trajectories
SGLang mem fraction:             0.6
SGLang max running requests:     44
actor max tokens/GPU:            10,240（保守默认）
```

不要为了让显存数字更高而直接增大 rollout/global batch：那会减少同一 19,200-prompt 预算下的策略更新
次数，改变 on-policy 刷新频率和优化问题，无法再与序贯实验直接比较。`16 x 16` 已经产生 256 个待生成
请求，足以让四个 TP=1 rollout engine 保持队列；训练阶段没有 32k eval，因此使用已在 8k response
上校准过的 `max_running_requests=44`。

可以在不改变 RL global batch 的前提下，提高 actor 动态 micro-batch 的 token packing。已有 Code n=16
run 在 `max_tokens_per_gpu=10240` 下每卡 peak allocated 约 54.0 GiB；peak reserved 中位数约 56.7 GiB、
p99 约 68.8 GiB、最坏一次约 80.1 GiB。更大的 token budget 可能提升 actor 吞吐，但必须实测，且 actor
阶段只占现有长 response 总 step 的约 3%，所以端到端收益通常有限。

仓库提供三个各一 update、无 eval、无 checkpoint 的容量 pilot：

```bash
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
export M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json

AVAILABLE_CUDA_DEVICES="$FOUR_HEALTHY_GPUS" \
TOKEN_BUDGETS="10240 14336 16384" \
  bash examples/optimizer_geometry/run_mixed_batch_pilot.sh
```

pilot 按 actor tokens/s 排序，只推荐 peak reserved 不超过 87,000 MiB 的候选，并生成
`outputs/mixed_batch_token_pilot/<timestamp>/batch_pilot_summary.json`。将其推荐值传给正式入口：

```bash
MAX_TOKENS_PER_GPU=14336 \
AVAILABLE_CUDA_DEVICES="$FOUR_HEALTHY_GPUS" \
  bash examples/optimizer_geometry/run_mixed_grpo.sh
```

上例的 `14336` 只是用法示例，不是未实测的默认结论；应使用 pilot 在四张空闲目标 GPU 上实际输出的
推荐值。如果只追求吞吐并愿意改变 RL 语义，应把更大的 rollout/global batch 作为单独 ablation，而
不是混入当前 sequential-vs-mixed 主比较。

## 主要产物

```text
outputs/raw_gradient_interference/qwen3_1.7b/seed42/mixed_code_math_qa_if/
  prepared_data/
    mixed_on_policy.yaml
    mixed_data_index.json
    mixed_final_eval.yaml
  qwen3_1.7b_code_math_qa_if_mixed_batch_grpo_adamw_seed42/
    checkpoints/
    geometry/
    metrics/
    provenance/
    run_complete.json
    final_eval/
```

异常退出后，确认 checkpoint 与 sampler state 完整，再显式设置 `RESUME_INCOMPLETE_RUN=1`。最终评测
目录已有完整 `run_complete.json` 时会自动跳过，避免重复支付评测成本。
