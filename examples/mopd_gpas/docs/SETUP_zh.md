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

该命令下载固定 model/data bundle，并把 `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c` 单独下载到 `${MOPD_QWEN3_4B}`。默认本仓库路径是 `local/mopd_assets/models/qwen3-4b`。

依次生成 held-out 集、转换 math/IF teacher、实测初始 KL 并冻结协议：

```bash
bash examples/mopd_gpas/run_stage.sh prepare-heldout
bash examples/mopd_gpas/run_stage.sh convert-teachers
CUDA_VISIBLE_DEVICES="${MOPD_INFERENCE_GPU}" \
  bash examples/mopd_gpas/run_stage.sh measure-initial
bash examples/mopd_gpas/run_stage.sh prepare
bash examples/mopd_gpas/run_stage.sh preflight
```

`prepare-heldout` 从每任务原始流 `[16384:16575]` 经同一 non-thinking 模板和 2048-token 长度过滤后固定前 128 条。训练只使用 `[0:16384]` 的候选；长度过滤和 seed-42 确定性洗牌后，每个任务截取恰好 16,000 条作为固定流。`measure-initial` 在 128×4 条 held-out prompt 上实测 sampled reverse KL；若最大/最小值之比大于 10，协议自动固定等权，否则固定为 inverse-initial-loss 权重。

## 常驻 teacher

```bash
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_stage.sh status-teacher
# 实验结束后：
bash examples/mopd_gpas/run_stage.sh stop-teacher
```

四个 SGLang server 都绑定 `${MOPD_INFERENCE_GPU}`，端口默认为 31001–31004。训练过程中不做 checkpoint 热切换。student rollout engine 的静态显存比例默认 0.32；teacher 的默认比例总和为 0.56，可在 48GB 卡的正式 smoke test 后通过环境变量微调。

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
