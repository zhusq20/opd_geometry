# SmolLM3-3B MixSFT：给协作者的 MOPD 运行说明

本页在**本 slime 仓库**中运行两组实验：三教师 sampled-token PG，以及 teacher Top64 与 student Top64 交集上的蒸馏 loss。Student 使用 Open-MOPD 发布的 **MixSFT（第 4 个 epoch）**；math、code、IF 使用对应的 RL teacher。模型来源和初始化关系见 [MixSFT model card](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT) 与 [Open-MOPD 论文](https://arxiv.org/pdf/2608.19098v1)。

**训练可以不装 sandbox。** 两种 loss 只需要 student rollout 和 teacher log-prob，不使用任务正确率作为 reward。本文的专用入口默认关闭训练过程中的 verifier 观察和在线能力评测，仍训练全部 math/code/IF 数据。需要正确率曲线时，再完成第 6 节的评测 setup；关闭评分时不输出任务准确率，日志明确记录 `task_reward_observed=0`。

## 1. 机器与环境

推荐分卡部署，避免三个 teacher 与 rollout engine 抢显存：

| 用途 | 6 张 48GB 卡的配置 | 5 张卡的配置 |
|---|---|---|
| student learner | GPU 0,1，TP=2 | GPU 0，80/96GB，TP=1 |
| student rollout | GPU 2 | GPU 1 |
| math / code / IF teacher | GPU 3 / 4 / 5 | GPU 2 / 3 / 4 |

GPU 编号是 `nvidia-smi` 的物理编号。TP=2 保留 FP32 梯度、Adam 状态和相同的领域平均方式；设置 `MOPD_PAPER_MEASUREMENTS=0` 关闭只支持 TP=1 的额外参数几何快照，训练日志、HF 权重和可恢复 optimizer checkpoint 照常保存。两条 smoke 加上模型、数据和 checkpoint 需预留至少 150GB；两组默认 500-step 实验建议预留 350GB 以上，开启几何快照需更多。checkpoint 较大，建议把输出放在 NVMe/SSD。内存建议至少 128GB；开启完整几何测量建议 256GB 以上。

A6000 示例配置使用 FlashInfer，并设 `MOPD_DETERMINISTIC_INFERENCE=0`：SGLang 的确定性 FP32 BMM 在 SmolLM3 RoPE 上需要 128KiB shared memory，超过 A6000 的 99KiB 限制。仍使用 PyTorch sampler 和逐请求固定 seed，但不承诺跨 batch/硬件逐 bit 相同。在支持该 kernel 的机器上可显式设为 1；两组对照保持相同设置。

使用已有的 slime 训练镜像，或从本仓库 Dockerfile 构建环境。不要仅在普通 Python 环境 `pip install slime`，也不要切换到官方 Open-MOPD 的 verl trainer。下面命令在**宿主机、当前仓库根目录**执行：

```bash
docker build -f docker/Dockerfile -t slime-smollm3:local .
docker run --rm -it --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:$PWD" -w "$PWD" slime-smollm3:local bash
```

在训练容器内安装当前 checkout，并始终让它排在导入路径最前面：

```bash
python3 -m pip install -e . --no-deps
export MEGATRON_PATH=/root/Megatron-LM
export PYTHONPATH="$PWD:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import slime, torch, transformers, sglang, ray; print(slime.__file__); print(torch.__version__, transformers.__version__, sglang.__version__, ray.__version__)'
```

`slime.__file__` 必须指向当前仓库，不能是镜像中的另一个 `/root/slime`。本次验证的库版本、GPU 与实际运行结果另见 [验证记录](VALIDATION_SMOLLM3_2026-09-09.md)。镜像构建本身不是该验证的一部分。

## 2. 配置 GPU，下载公开模型和数据

以下操作从仓库根目录的**新 shell**开始，避免继承历史 Base/Qwen 实验的路径：

```bash
mkdir -p local
cp examples/mopd_gpas/configs/smollm3_mixsft.site.example.env local/smollm3.env
# 编辑 local/smollm3.env 中的 GPU 编号；默认 6 张 48GB 卡。
source local/smollm3.env
bash examples/mopd_gpas/run_stage.sh fetch-assets
bash examples/mopd_gpas/run_stage.sh prepare
```

下载使用固定 revision，约 27GB 模型与数据，不需要获取 SFT 训练全集或重新训练老师：

| 角色 | Hugging Face 仓库后缀（均位于 `BytedTsinghua-SIA/`） | 固定 revision |
|---|---|---|
| student | `Open-MOPD-SmolLM3-3B-MixSFT` | `c9e7bad031667828656ead188d7e8ea162c048a4` |
| math teacher | `Open-MOPD-SmolLM3-3B-RL-Math` | `5e901bb626b69d711074d2832c41ce1aa4232da8` |
| code teacher | `Open-MOPD-SmolLM3-3B-RL-Code` | `e2ce9beec52381350edcdaefebfd08ea67c21b42` |
| IF teacher | `Open-MOPD-SmolLM3-3B-RL-IF` | `8948051a805883e2db988cae4522de3c29d1d112` |
| data | [`Open-MOPD-Data`](https://huggingface.co/datasets/BytedTsinghua-SIA/Open-MOPD-Data) | `9e897efe3257599d4300e2d5ee865a1cc714af87` |

仅下载 `rl_prompt_mix/` 和六项评测的 parquet。`prepare` 自动转换成 slime JSONL，保留 code 测试、IF 指令约束和领域标签；每域以 seed 42 确定性预留 64 条 heldout 和 64 条 diagnostic，剩余数据作为训练流。训练时应用 MixSFT 原生 chat template，`enable_thinking=true`，prompt 上限 2048。脚本核对 student 与三个 teacher 的 tokenizer ID 映射。Student 直接加载 HF 权重，无需运行历史 Qwen 的 `convert-teachers` 阶段。

默认目录如下，可通过对应环境变量整体搬到高速磁盘；**路径变量要在首次 source profile 前设置**：

| 环境变量 | 默认目录 |
|---|---|
| `MOPD_ASSET_ROOT` | `local/mopd_smollm3_mixsft_assets` |
| `MOPD_GENERATED_DIR` | `local/mopd_smollm3_mixsft_generated` |
| `MOPD_OUTPUT_ROOT` | `outputs/mopd_smollm3_mixsft` |

不要复用旧 `local/mopd_smollm3_assets` 中的 Base student 或其 optimizer checkpoint。下载器会拒绝在已记录为 Base 的资产目录中覆盖成 MixSFT。换机器应重新 `prepare`，因为生成的 YAML 包含绝对路径。改 teacher 端口或评分服务地址后也需要重新 `prepare`，并给新实验使用独立 generated/output 目录。

## 3. 启动 teacher，跑两条 smoke

```bash
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_stage.sh status-teacher

DRY_RUN=1 bash examples/mopd_gpas/run_smollm3.sh pg smoke
DRY_RUN=1 bash examples/mopd_gpas/run_smollm3.sh intersection64 smoke

set -o pipefail
bash examples/mopd_gpas/run_smollm3.sh pg smoke 2>&1 | tee local/smollm3-pg-smoke.log
bash examples/mopd_gpas/run_smollm3.sh intersection64 smoke 2>&1 | tee local/smollm3-intersection64-smoke.log
```

每条 smoke 使用真实模型、真实三域 prompt 和三个 RL teacher，做 **2 次 Adam 更新，每次 12 条 response，每域 4 条，response 上限 128**。第二次更新会使用更新后的 student rollout；结束时保存 HF 和 optimizer checkpoint。短 response 用于检查工程链路，不能当作能力实验结果。

检查两条 pipeline 均完成，不能仅凭进程启动成功或 dry-run 判断：

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['MOPD_OUTPUT_ROOT'])
for name in ('smoke-m-pg-s42', 'smoke-m-intersection64-dr-s42'):
    run = root / name
    complete = json.loads((run / 'run_complete.json').read_text())
    assert complete['status'] == 'complete' and complete['final_num_updates'] == 2
    print(name, complete)
    index = json.loads((run / 'checkpoints/mopd_checkpoint_index.json').read_text())
    final = next(row for row in index if row['optimizer_step'] == 2)
    assert final['optimizer_state_retained']
    assert (Path(final['hf_checkpoint']) / 'model.safetensors.index.json').is_file()
    assert (run / 'checkpoints' / f"iter_{final['rollout_id']:07d}" / '.metadata').is_file()
    assert not (run / 'run_failed.json').exists()
    print('checkpoint OK')
PY
```

默认 seed 是 42，改过 seed 或 `MOPD_RUN_ID` 时相应替换检查路径。重复 smoke 请指定新 ID，例如 `MOPD_RUN_ID=smoke-pg-retry1 bash .../run_smollm3.sh pg smoke`。两个训练进程复用同一组训练卡，应依次运行。

## 4. 正式训练：PG 和 Top64 交集

```bash
bash examples/mopd_gpas/run_smollm3.sh pg train 2>&1 | tee local/smollm3-pg.log
bash examples/mopd_gpas/run_smollm3.sh intersection64 train 2>&1 | tee local/smollm3-intersection64.log
```

两组默认都从 MixSFT 独立初始化，500 次更新，每次 48 个 prompt，每域 16 个，每 prompt 一条 response；训练 response 上限 4096。每 4 个 response 构成一个 backward slice，每个 rollout 只有一次 Adam 更新。两组使用等领域权重、领域内等 response 权重、response 内有效 token 平均；学习率 `2.5e-7`，Adam `(0.9, 0.98)`，epsilon `1e-8`，weight decay 0，grad clip 1.0。设置 `MOPD_TOTAL_STEPS`、`MOPD_CHECKPOINT_STEPS`、`MOPD_SEED`、`MOPD_LR` 可显式调整；两组对照应使用相同设置。专用入口不修改现有六组默认实验矩阵。

两种目标的具体含义（`p` 为 student，`q` 为对应领域 teacher；log-prob 均以完整词表归一化）：

```python
# PG：只在 student 采样到的 token y 上计算
adv = (log_q_y - log_p_y).detach()
loss_t = -adv * log_p_y

# intersection64：S 是 rollout 时的 student Top64；T 是同一前缀的 teacher Top64
# actor forward 在保存的 S 上重算 student 概率；S、T 和交集 mask 不参与求导
weight = log_p_on_S.softmax(-1)
adv = (weight * (log_q_on_S - log_p_on_S)).detach()
adv = adv.masked_fill(~shared_mask, 0)
loss_t = -(adv * log_p_on_S).sum(-1)
```

交集外 teacher 数值使用有限占位符并严格 mask；**权重按原 student Top64 归一化，交集后不再次归一化**。空交集贡献零 loss、零梯度。PG 无 PPO ratio / clip，也不使用任务 verifier reward。这与论文默认的 student Top16 不同，本文运行的是指定的两个对照；不启用论文完整系统的动态 token-share、gap allocation 或 rollout 内 4 次 reward refresh。

`m-intersection64-dr` 才是交集条件；历史 `m-tk64-dr` 使用整个 student Top64，请勿混用。日志记录 `student_teacher_top64_intersection_size`、`empty_fraction`、`retained_weight`、`normalized_logratio` 等指标。该 log-ratio 可以为负，不能据其符号判定为非负 KL。

## 5. 日志、checkpoint、恢复

每条正式实验分别写到 `${MOPD_OUTPUT_ROOT}/m-pg-s42/` 和 `m-intersection64-dr-s42/`：

- `metrics/*.jsonl`：loss、梯度、领域 token 数、长度、截断率和耗时；无 sandbox 模式记录 `task_reward_observed=0`。
- `provenance/run_manifest.json`：执行命令、模型/数据身份、源码快照和退出状态。
- `checkpoints/mopd_checkpoint_index.json`：optimizer step 到 HF/optimizer checkpoint 的映射。第 2 次更新对应 rollout ID 1，不要直接把目录编号当更新次数。
- `weights/`：用于独立能力评测的 HF 模型；`run_complete.json` / `run_failed.json`：运行完成/失败记录。

注意：backward slice 日志中的 `train/optimizer_step_executed=0`、`train/grad_norm=0` 表示尚在累积；实际更新次数和梯度范数看 `mopd/optimizer_updates`、`mopd/aggregate_grad_norm`。通用训练器中的 `rollout/rewards=0` / `raw_reward=0` 是蒸馏协议占位值，不是代码或数学准确率；准确率只看启用 verifier 后的 `rollout/reward/<task>/...`。

默认不开 W&B，JSONL 始终保留。要使用 W&B，在运行前设置 `USE_WANDB=1`、`WANDB_PROJECT` 并完成认证，或设置 `WANDB_MODE=offline`。

恢复中断实验时保持相同 profile、loss、K、GPU 拓扑、generated 配置和 run ID：

```bash
MOPD_RESUME=1 bash examples/mopd_gpas/run_smollm3.sh pg train
# 交集实验：
MOPD_RESUME=1 bash examples/mopd_gpas/run_smollm3.sh intersection64 train
```

仅能从已保存的完整 checkpoint 恢复；切换 loss、从 Base 切到 MixSFT 或使用短 smoke 权重做正式实验，都应启动新 run。

## 6. 需要正确率时：额外安装 verifiers / SandboxFusion

这一节用于记录训练 reward 或跑 AIME24/25、LCB v5/v6、IFEval、IFBench 的能力评测。前三节训练不需要执行本节。

先在训练容器获取固定版本的 Open-MOPD **评分代码**，不安装其训练栈：

```bash
export OPEN_MOPD_ROOT="$PWD/local/open_mopd_scorers"
OPEN_MOPD_SOURCE_ROOT="$OPEN_MOPD_ROOT" \
  bash examples/mopd_gpas/open_mopd_full/fetch.sh source
python3 -m pip install -r examples/eval_multi_task/requirements_ifbench.txt
python3 -m nltk.downloader punkt_tab
```

SandboxFusion 在有 rootful Docker、cgroup v2 的**宿主机**构建与启动，提供 CPU 代码执行，不占训练 GPU：

```bash
export SANDBOX_STATE="$HOME/.local/state/slime-opd-geometry/sandboxfusion"
mkdir -p "$SANDBOX_STATE"
SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh
SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
SANDBOX_PREFLIGHT_MARKER="$SANDBOX_STATE/sandboxfusion_preflight.json" \
  bash examples/optimizer_geometry/start_sandboxfusion.sh
```

首次构建需为大型镜像另留约 60GB 以上空间。服务会生成通过隔离探针的 marker；**不能用随手启动的 `/run_code` 服务或伪造 marker 替代**。详细要求见 [sandbox 部署说明](../optimizer_geometry/SANDBOXFUSION_CGROUP2_zh.md)。

启动训练容器时增加 `-v "$SANDBOX_STATE:/workspace/sandboxfusion-state:ro"`，保留 `--network host`。然后在容器内重新 source 配置，并设定：

```bash
export OPEN_MOPD_ROOT="$PWD/local/open_mopd_scorers"
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
export M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json
bash examples/mopd_gpas/run_stage.sh prepare

# 对 MixSFT 本身做基线评测
bash examples/mopd_gpas/run_stage.sh capability initial_student
# 对已导出的第 500 次更新评测
MOPD_EVAL_STEP=500 bash examples/mopd_gpas/run_stage.sh capability m-pg-s42
MOPD_EVAL_STEP=500 bash examples/mopd_gpas/run_stage.sh capability m-intersection64-dr-s42
```

评测使用 student rollout GPU，训练结束后再运行。也可单独指定空闲 `CAPABILITY_CUDA_VISIBLE_DEVICES`。AIME 每题 64 次、LCB 每题 10 次、IF 每题 1 次；本仓库评测 response 上限 32768、SmolLM 原生总上下文 65536。完整评测耗时显著高于 smoke。要在训练中观察正确率，在新的正式 run 开始前设置 `MOPD_OBSERVE_TASK_REWARDS=1`；要在线能力评测，再设置 `MOPD_EVAL_DURING_TRAINING=1`（默认间隔 250 更新）。

## 7. 常见问题

- **启动后找不到模块/新参数**：检查 `slime.__file__` 和 `PYTHONPATH` 是否指向此 checkout，避免镜像中旧 slime 抢先导入。
- **teacher 显存不足/启动超时**：确认三个 teacher 分别占独立 GPU，查看 `${MOPD_TEACHER_SERVER_DIR}/*.log`。不要在 48GB rollout 卡上同时塞三个 3B teacher。
- **训练显存不足**：48GB 卡用双卡 TP=2；单卡 TP=1 的完整 FP32 状态与领域梯度缓冲需要更大显存。缩短 response 不能消除模型/optimizer 本身的开销。
- **代码 reward 报 marker/scorer 缺失**：纯训练保持 `MOPD_OBSERVE_TASK_REWARDS=0` 和 `MOPD_EVAL_DURING_TRAINING=0`；要准确率则完成第 6 节，不要关闭 sandbox 隔离验证。
- **数据目录是旧机器的绝对路径**：重新 `prepare`。不要手工删除 code 样本来绕过评分问题。
- **prompt 太长被过滤**：训练 prompt cap 为 2048，这是预期行为；原始数据文件保留在 assets，生成的数据身份与 split 记录在 `protocol.json`。

结束使用后释放本次 teacher 服务：

```bash
bash examples/mopd_gpas/run_stage.sh stop-teacher
```
