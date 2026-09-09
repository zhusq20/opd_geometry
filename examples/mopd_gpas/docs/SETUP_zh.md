# 环境与资产配置

## 硬件与软件

每条主轨迹固定使用两张不同的 GPU：一张至少 90,000 MiB 的训练卡和一张 45,000–55,000 MiB 的推理卡。`verify_hardware.py` 会检查卡型与编号。需要可用的 Megatron-LM、Ray、SGLang、Transformers、Hugging Face CLI 和 W&B（可通过 `USE_WANDB=0` 关闭）。

```bash
cp examples/mopd_gpas/configs/site.example.env local/mopd.env
vim local/mopd.env
source local/mopd.env
```

## 下载与准备

```bash
bash examples/mopd_gpas/run_stage.sh fetch-assets
```

该命令下载固定 model/data bundle，并把 `Qwen/Qwen3-1.7B-Base` revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` 下载到 `${MOPD_HF_CHECKPOINT}`，使用与四个 RL teacher 相同的 tokenizer 和生成配置。默认本仓库路径是 `local/mopd_assets/models/qwen3-1.7b-base`。

准备 Base student 和四域 RL teacher，然后生成两周核心协议：

```bash
bash examples/mopd_gpas/run_stage.sh convert-teachers
bash examples/mopd_gpas/run_stage.sh prepare
bash examples/mopd_gpas/run_stage.sh dry-run
```

`prepare` 从每任务原始流 `[16384:16575]` 经 student 模板渲染、删除末尾空 thinking 块和 2048-token 长度过滤后固定前 64 条为长期 loss prompts，其余作为独立 diagnostic pool。训练只使用 `[0:16384]` 的候选；相同输入处理和 seed-42 确定性洗牌后，每个任务截取恰好 16,000 条作为固定流。teacher 对同一 response 评分时由 router 插回空 thinking 块。目标始终为四域等权，无需初始 KL 测量。请将现有环境文件中的输出目录改为 `local/mopd_no_think_generated` 和 `outputs/mopd_no_think`，重新 prepare 并从初始 Base 训练；旧 bank 和 checkpoint 不可跨前缀协议复用。完整运行步骤见[实验手册](EXPERIMENTS_zh.md)。

## 常驻 teacher

```bash
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_stage.sh status-teacher
# 实验结束后：
bash examples/mopd_gpas/run_stage.sh stop-teacher
```

四个 SGLang server 默认绑定 `${MOPD_INFERENCE_GPU}`，端口默认为 31001–31004。训练过程中不做 checkpoint 热切换。student rollout engine 的静态显存比例默认 0.32；teacher 的默认比例总和为 0.36。Teacher 使用 SGLang 原生分块 logprob 计算，默认每块 512 个位置（`TEACHER_LOGPROBS_CHUNK_SIZE`），限制 dense 全词表 logits 的临时显存及各服务保留的 allocator cache；每个位置仍使用完整词表归一化。原有 TP2 learner 和分布式 teacher 参数仍保留，使用时记录实际硬件与 GPU 占用。

所有模型、数据和代码快照会写入每个 run 的 `provenance/run_manifest.json`。访问令牌只放在本机环境，不写入仓库。

## LiveCodeBench capability sandbox

训练、smoke test 和 held-out teacher-loss 评测不依赖 SandboxFusion；只需在执行 `capability` 前启动，因而不会占用训练 GPU。构建和启动必须在能直接访问 rootful Docker daemon 的宿主机执行：

```bash
export SANDBOX_STATE="${HOME}/.local/state/slime-opd-geometry/sandboxfusion"
mkdir -p "${SANDBOX_STATE}"
chmod 700 "${SANDBOX_STATE}"

# 每台宿主机首次部署或补丁变化后构建一次。
SANDBOXFUSION_PIN_FILE="${SANDBOX_STATE}/sandboxfusion-image.env" \
  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh

# 每次服务或宿主机重启后重新启动并生成安全 attestation。
SANDBOXFUSION_PIN_FILE="${SANDBOX_STATE}/sandboxfusion-image.env" \
SANDBOX_PREFLIGHT_MARKER="${SANDBOX_STATE}/sandboxfusion_preflight.json" \
  bash examples/optimizer_geometry/start_sandboxfusion.sh

export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
export M2RL_SANDBOX_PREFLIGHT_MARKER="${SANDBOX_STATE}/sandboxfusion_preflight.json"
bash examples/mopd_gpas/run_stage.sh capability all
```

如果 capability eval 在训练容器内运行，容器必须使用 host network，并以只读方式挂载该 marker；`M2RL_SANDBOX_PREFLIGHT_MARKER` 应指向容器内路径。launcher 会在创建 Ray 或输出目录前验证 endpoint、marker 新鲜度和 cgroup-v2 attestation，并把 marker 路径传给 Ray worker。详细的隔离模型和宿主机要求见 [`examples/optimizer_geometry/SANDBOXFUSION_CGROUP2_zh.md`](../../optimizer_geometry/SANDBOXFUSION_CGROUP2_zh.md)。
