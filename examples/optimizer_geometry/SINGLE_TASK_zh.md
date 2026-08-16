# OPD 单任务参数几何：研究设计与运行说明

本目录把基础任务落实为 Qwen3-1.7B student 在 math/code/science 上的可复现实验。用户侧的
`coding` 在 M2RL manifest 中统一写成 `code`。支持的 teacher 是：

- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-4B-Thinking-2507`

teacher 由独立 SGLang server 提供逐 token log-prob；只有 student 需要转换成 Megatron `torch_dist` 格式。

## 0. 当前机器上的快速入口

当前环境已经定位并验证以下 checkpoint：

```text
Qwen3-1.7B HF:       /workspace/dev/checkpoints/Qwen3-1.7B
Qwen3-1.7B Megatron: /workspace/dev/checkpoints/Qwen3-1.7B_torch_dist
Qwen3-8B teacher:    /workspace/dev/iclr2027/checkpoints/Qwen3-8B
```

宿主机上的项目必须从 disk2 真实路径
`/mnt/disk2_from_server2/siqizhu4/iclr2027/slime_opd_geometry` 编辑和启动，并通过
bind mount 映射为容器内的 `/workspace/dev/iclr2027/slime_opd_geometry`。
`/mnt/data_from_server2/siqizhu4/iclr2027` 是 mergerfs 视图，可能优先展示 disk3
上的另一份同名文件，不能把它作为本项目的宿主机代码路径。启动器会先找项目
checkpoint，再找共享的 `/workspace/dev/checkpoints`，无需手工改脚本。1.7B 与
8B 的 151,669 项 tokenizer 映射、added tokens 和 special-token IDs 已逐项验证一致。

单任务数据已经准备在：

```text
data/m2rl/single_task/
  math/     22,051 train（已去除 benchmark 重合）+ MATH-500 online eval
  code/     19,169 train + LiveCodeBench v5 64 题 online eval + 880 题 final eval
  science/  19,670 train + GPQA-Diamond online eval
  single_task_index.json

data/m2rl/eval/m2rl_online/
  aime24.parquet        30 problems
  math500.parquet       500 problems
  gpqa_diamond.parquet  198 problems（HF gated，需先取得访问权限）
```

Frozen OPD 保留 `MAX_PROMPT_LEN=2048` 过滤。用正式 Qwen3 tokenizer 复核后，三份数据的
raw/usable 数分别为 Math 22,051/22,050、Code 19,169/19,125、Science 19,670/19,668；长度过滤数
分别为 1 / 44 / 2。Math 原始 22,056 行中另有 5 行包含 4 道 MATH-500 题，准备脚本会先将其从
单任务训练视图剔除；完整的 500 道 benchmark 题全部保留。下文“一轮数据”均指这批过滤后的
usable prompts。

当前环境已落盘固定 revision 的 AIME'24、MATH-500 与 GPQA-Diamond；GPQA 使用 seed 42 固定选项顺序，文件行数与
SHA-256 记录在 `data/m2rl/eval/m2rl_online/eval_data_index.json`。Science online eval 已标成
`external_benchmark`，不会回退到训练集 holdout。

如需重建：

```bash
# 首次使用需先在 GPQA 页面同意 gated 条款，然后二选一：hf auth login 或 export HF_TOKEN=...
FORCE_REBUILD=1 REQUIRE_GPQA=1 \
  bash examples/optimizer_geometry/prepare_single_task_dataset.sh
```

Math eval 会同时准备三个可复用配置：`math_eval_aime24.yaml`、`math_eval_math500.yaml` 和
`math_eval_aime24_math500.yaml`；`math_eval.yaml` 是当前默认别名。默认仍为 MATH-500。准备时或
启动 OPD Math 时可直接选择，逗号和空格分隔都支持：

```bash
# 仅 AIME'24
MATH_EVAL_DATASETS=aime24 bash examples/optimizer_geometry/run_opd_math_adamw.sh

# 仅 MATH-500（默认）
MATH_EVAL_DATASETS=math500 bash examples/optimizer_geometry/run_opd_math_adamw.sh

# 同一次 eval 同时报告 eval/aime24 与 eval/math500
MATH_EVAL_DATASETS="aime24 math500" bash examples/optimizer_geometry/run_opd_math_adamw.sh
```

显式 `EVAL_CONFIG=/path/to/config.yaml` 的优先级更高。无论运行时选哪种 Math eval，训练 manifest
都固定使用去掉 5 条 MATH-500 重合行的同一份训练视图，避免评测选择反过来改变 optimizer 对照的
训练数据。

Plain GRPO/PPO 的正式 Math 入口默认不使用单数据集别名，而是选择
`math_eval_aime24_math500.yaml`，同时报告 AIME'24（每题 8 次采样）和 MATH-500（每题 1 次 greedy
采样）。训练 rollout response cap 固定为 8,192，`max_tokens_per_gpu=10,240`；evaluation 独立使用
32,768-token cap。两阶段共用 SGLang engine，因此当前 96 GiB Qwen3-1.7B 部署按更长的 eval cap
限制为每个 engine 12 个并发请求。Plain GRPO/PPO 还固定使用 `enable_thinking=false`；response-cap
pilot 仍是显式的校准例外。

默认快捷准备不再切 256 条训练 holdout；Code/Science 使用全量训练数据，Math 只去掉上述 5 条
benchmark 重合行。OPD Math 默认使用 MATH-500（每题 1 次 greedy 采样），也可改为 AIME'24
（每题 8 次采样）或同时运行两者；Science
使用 GPQA-Diamond（每题 4 次），Code 使用固定且与
训练 prompt 零重合的 LiveCodeBench v5 64 题子集（greedy pass@1）。Code 最终评测另用完整
release_v5 的 880 题，每题 10 次采样，报告 pass@1/5/10。`HF_TOKEN` 仅从环境读取，不会写入
脚本或索引；若 GPQA 尚未获权，准备脚本
会明确把 science eval 标为 disabled，而 science 快捷启动器默认拒绝在缺少独立 eval 时启动。

M2RL 论文最终表格使用另一套 benchmark suite：Math 为 AIME'24/25，Coding
为 LiveCodeBench v5/v6，Science 为 HLE/GPQA-Diamond，并额外报告 MMLU-Redux。它们不是从
训练集切出的 holdout，也不等同于当前 OPD 脚本使用的 MATH-500 online eval。当前 online eval
每 50 个 optimizer updates（满批时 3,200 个训练 prompts）执行；在线与最终 eval 的 response cap
按任务固定为 Math 32,768、Code/Science 16,384。旧的 256 holdout 方式仍
可通过 `prepare_single_task_data.py` 显式使用，仅用于 pipeline/smoke test。

W&B 已默认开启，entity 为 `zsqzz`，project 为 `iclr2027-opd-geometry`。密钥只从环境
变量读取，不应写入脚本：

```bash
export WANDB_API_KEY='<your-wandb-api-key>'
export WANDB_MODE=online
```

RL 使用 4 张 student 卡；固定 seed 42，并在通过 2,048-token prompt 过滤后完整训练一个可用
数据集 epoch（包含真实尾批）。Plain GRPO/PPO 训练统一使用 8,192-token rollout 并关闭 thinking；
Math 统一评测 AIME'24 + MATH-500，evaluation rollout 独立使用 32,768-token cap。每个命令依次运行
AdamW、vanilla SGD、Muon，共 3 个 cell；
`SEEDS`/`SEED` 只接受 42，且拒绝旧的 `NUM_ROLLOUT`/`TARGET_PROMPT_BUDGET` 固定 prompt 预算。
2026-08-13 的本机检查发现物理 GPU 0 有不可纠正 ECC 错误，不能使用；下面的
设备号是当时通过 preflight 且空闲的物理卡，正式启动前脚本仍会检查 ECC 和至少 90% 空闲显存：

```bash
export AVAILABLE_CUDA_DEVICES=1,4,6,8
TASK=math bash examples/optimizer_geometry/run_single_task_rl.sh
TASK=code bash examples/optimizer_geometry/run_single_task_rl.sh
TASK=science bash examples/optimizer_geometry/run_single_task_rl.sh
```

OPD 再给 Qwen3-8B teacher 一张卡（默认前 4 张为 student、最后 1 张为 teacher）。单轮数据集
正式矩阵固定为 seed 42，并拆成以下 9 个互相独立、可单独调度的脚本：

```bash
export AVAILABLE_CUDA_DEVICES=1,4,6,8,7
bash examples/optimizer_geometry/run_opd_math_adamw.sh
bash examples/optimizer_geometry/run_opd_math_sgd.sh
bash examples/optimizer_geometry/run_opd_math_muon.sh
bash examples/optimizer_geometry/run_opd_code_adamw.sh
bash examples/optimizer_geometry/run_opd_code_sgd.sh
bash examples/optimizer_geometry/run_opd_code_muon.sh
bash examples/optimizer_geometry/run_opd_science_adamw.sh
bash examples/optimizer_geometry/run_opd_science_sgd.sh
bash examples/optimizer_geometry/run_opd_science_muon.sh
```

这些 cell 脚本会清除外层 `SEEDS/NUM_ROLLOUT/TARGET_PROMPT_BUDGET`，并显式固定
`opd64x1`（64 prompts × 1 response）、`NUM_EPOCH=1`、Qwen3 `enable_thinking=false`、训练 prompt/response cap 2048/4096、
`SGLANG_MAX_RUNNING_REQUESTS=72`、`EVAL_INTERVAL=50`、Math `EVAL_MAX_RESPONSE_LEN=32768`（Code/Science 为 16384）和
eval-only 全局并发 48。通用的
`TASK=... bash run_single_task_opd.sh` 入口仍保留，但默认也只跑 seed 42。
Code 的训练和评测必须先启动通过主动隔离检查的 SandboxFusion；endpoint 与安全 marker 由环境
变量传入，配置不会回退到本地 shell。即使 pure OPD 的训练信号不使用 verifier，其 Code 在线/
最终评测仍需要 sandbox。可先用 `DRY_RUN=1` 检查所有路径和参数。

W&B 会保存完整展开后的模型、数据 manifest、task、teacher、algorithm、optimizer、seed、
batch/token budget、学习率和各 optimizer 超参；曲线包括 `train/*` loss/grad/KL、
`rollout/reward/*`、每个 source 的样本比例、`eval/aime24`、`eval/math500`、GPQA/LiveCodeBench
的 `eval/*`、吞吐，以及
`geometry/{global,optimizer_branch_adam,optimizer_branch_sgd,optimizer_branch_muon_matrix,optimizer_branch_adam_fallback}/*`。
分支来自真实 optimizer membership，不按参数维度猜测。完整逐层/算子精确几何与低频投影仍写入
run 目录的 `geometry/actor/`。设置 `GEOMETRY_WANDB_GROUPS=all` 可把逐层量也上传 W&B。

## 1. 先把问题拆成三个可识别的因果轴

| 实验 | 改变量 | 固定量 | 主要观测 |
| --- | --- | --- | --- |
| optimizer × OPD | AdamW / SGD / Muon | student 初值、teacher、task、rollout seed、token budget | reward、梯度/更新范数、gradient-update cosine、有效步长、相对初值位移 |
| teacher distribution × OPD | 8B / 4B-Thinking-2507 | optimizer 与全部 student 侧设置 | sampled reverse KL、reward、同一层的更新/位移几何 |
| mixed loss × OPD | pure OPD / SFT+OPD / GRPO+OPD / PPO+OPD | optimizer、teacher、task、总 batch/token budget | 总梯度和实际参数更新的几何、reward、混合项 loss |

第二个实验应称为 **teacher identity/distribution 对照**。`Qwen3-8B` 与
`Qwen3-4B-Thinking-2507` 不只参数量不同，后训练方法和输出分布也不同，因而不能把差异单独归因为 teacher size。若论文需要严格的 size 因果结论，应再加入同系列、同训练配方、只改变规模的 teacher（例如对应的普通 4B/8B 对照）。

### 1.1 候选起点、调参后冻结的主对照超参数

启动器默认值是最接近文献的候选起点，不应未经验证就称为“最优”或“已冻结”。先在不进入论文
主表的 tuning split 和 seeds 1042/1043 上，用相同候选数、相同 12,800-prompt 预算（默认
`responsive16` 为 800 updates）选择 OPD 系数与
各 optimizer LR；再用一个 confirmation seed 验证。选择完成后写入预注册配置并冻结，禁止根据
math/code/science 报告集或 seeds 42--46 的结果继续改参。这样比较的是“在等额调参预算下各
optimizer 的性能”，同时可将这里列出的统一文献起点作为严格配方对照。

| 类别 | 固定值 |
| --- | --- |
| 训练预算 | `NUM_EPOCH=1`；去重并过滤后 Math 22,050、Code 19,125、Science 19,668 usable prompts，各题恰好一次 |
| 采样与 batch | `BATCH_PROFILE=opd64x1`；每个满 rollout 为 64 prompts × 1 response，`GLOBAL_BATCH_SIZE=64`；末尾不足 64 的题使用真实尾批，不丢弃、不回卷 |
| token/采样 | frozen OPD cell 显式使用 `apply_chat_template_kwargs={"enable_thinking":false}`；student 用 non-thinking 模板生成，teacher 对同一串 token 打分且不自行生成；训练 prompt/response 上限为 2048/4096，packing cap 10240；evaluation response cap 独立固定为 Math 32768、Code/Science 16384；temperature 1，top-p 1，top-k -1 |
| 公共训练项 | constant LR、warmup 0、global grad clip 1、dropout 0、weight decay 0 |
| pure OPD | `OPD_KL_COEF=1`、task reward weight 0、symmetric clip 0.2/0.2，不启用 TIS |
| GRPO | clip 0.2/0.28、entropy 0、TIS clip `[0,2]` |
| AdamW | LR `2.5e-7`，betas `(0.9,0.9987381276)`，epsilon `1e-8` |
| vanilla SGD | momentum 0；GRPO LR `2.5e-2`，OPD/PPO LR `2.5e-3` |
| Muon | LR `2.5e-7`，momentum 0.95，Newton--Schulz 5 步，spectral scale，再乘 0.2；不使用 Nesterov；非矩阵 AdamW fallback 使用 `(0.9,0.9987381276,1e-8)` |
| PPO critic | 固定 AdamW；LR `2.5e-6`，beta2 `0.9987381276`，weight decay `0.025` |
| 复现 | OPD 主矩阵只用 seed 42；tuning seeds 1042/1043；SGLang deterministic inference；eval 每 50 updates；checkpoint 每 80 updates；eval 全局并发 48；精确 geometry 每个 update，CountSketch/坐标分布/支持集/抽样矩阵每 4 updates（256 prompts），投影维度 256 |

选择逻辑如下：

- 旧的 51,200 来自 M2RL 配方的 `256 prompts × 200 steps`，目的是让不同 cell 使用固定采样预算，
  并不是这些单任务数据集的 epoch 大小。Frozen OPD 主矩阵现已取消该人工预算，直接训练一轮数据。
- Qwen3-1.7B 的直接 SGD-RL 研究给出 GRPO 的 vanilla SGD LR `0.1`；直接 OPD-SGD
  消融给出 SGD LR `0.01` 与 AdamW LR `1e-6`。`opd64x1` 每步仍有 64 个 response，与旧
  `responsive16` 的 `16×4=64` 相同，因此暂时沿用已经调定的 LR，而不是因 prompt 分组改变再缩放。
- 小 batch 语言模型研究表明 Adam 在减小 batch 时应显著增大 beta2，使二阶矩在 token/sample
  维度的半衰期保持不变；相对于 reference 的 1,024 responses/update，当前 64 responses/update 使用
  `0.98^(64/1024)=0.9987381276`，同时保留 beta1=0.9。
- Scalable Muon 推导出 `0.2*sqrt(max(m,n))` 的更新缩放，使 Muon 与 AdamW 的 update RMS
  处在同一单位；因此 Muon 与 AdamW 共用同一 profile LR（`opd64x1` 为 `2.5e-7`），
  而不是把 Muon LR 再放大 10 倍。
- frozen OPD 主矩阵把 response 上限固定为 4096，并按当前要求保留 2048 prompt 过滤；三种
  optimizer 使用完全相同的过滤和截断规则，packing cap 维持 10240。
- weight decay 设为 0 是因果隔离决定：SGD 的 LR 比 AdamW 大 4--5 个数量级，非零 decay
  会让每步收缩量也随 LR 改变，从而把 optimizer 与正则化混在一起。

依据均来自预先指定的、与当前模型/目标最接近的公开配方：
[M2RL](https://arxiv.org/abs/2602.12566)、
[Small Batch Size Training for Language Models](https://arxiv.org/abs/2507.07101)、
[Scalable Muon](https://arxiv.org/abs/2502.16982)、
[Qwen3-1.7B 上的 SGD-RL](https://arxiv.org/abs/2602.07729)、
[OPD 的 SGD 消融（v1）](https://arxiv.org/abs/2606.13657v1)。

具体候选网格、稳定性排除规则和 one-standard-error tie break 见
[`OPTIMIZER_COMPARISON_PROTOCOL_zh.md`](OPTIMIZER_COMPARISON_PROTOCOL_zh.md)，对应脚本为
`run_opd_coefficient_tuning.sh`、`run_optimizer_tuning.sh` 和 `select_tuning_hparams.py`。
response 上限用 `run_response_cap_pilot.sh` + `select_response_cap.py` 冻结；统一终点用
`run_budget_pilot.sh` + `select_training_budget.py` 在只观察 AdamW 的情况下决定。

这里的 `step` 指 rollout/optimizer update。64-prompt 满批下，Math 为 345 steps（尾批 34），Code
为 299 steps（尾批 53），Science 为 308 steps（尾批 20）。论文曲线必须同时报告 update 和累计
prompt/token，最后一步的真实 global batch 不能误记成 64。
脚本默认检查 `GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT`，避免一次
rollout 被拆成多个 update 后悄悄改变优化动力学。
历史 `reference256/responsive16/responsive8` 的 8192-response GRPO smoke 只能说明旧 profile 的
容量与延迟，不能直接当作当前 `64×1`、4096-response OPD 的实测耗时。当前 frozen cell 的终点由
数据集一轮唯一决定，不按 optimizer early-stop，也不从中间点挑最高点。某个 optimizer 若在该
预注册配方下发散，应把它作为结果报告，而不是事后换学习率重跑并只保留成功 cell。

## 2. 本实现中的 loss 定义

对 student 当前策略生成的 token $a_t\sim\pi_\theta(\cdot\mid h_t)$，OPD 使用 sampled reverse-KL advantage：

\[
\hat A_t^{\mathrm{OPD}}
=-\lambda_{\mathrm{OPD}}
\left[\log\pi_\theta(a_t\mid h_t)-\log\pi_T(a_t\mid h_t)\right].
\]

这里的 student/teacher 输入均为每条 response 的一维 sampled-token log-prob 向量
`[response_tokens]`。Teacher 请求设置 `max_new_tokens=0`，只取 student 实际采样的 $a_t$ 的
log-prob；不会传入或构造 `[response_tokens, vocab]` 的 teacher target，也没有 full-vocabulary
distillation loss。运行时会检查样本数、token 长度和张量维度，误接 full-vocab tensor 会立即报错。

`grpo_opd` 和 `ppo_opd` 在同一批 on-policy token 上把任务 advantage 与上述 OPD advantage 相加。

`sft_opd` 则在同一个 optimizer step 中混合两类独立样本：

\[
L=\alpha L_{\mathrm{OPD}}+\beta L_{\mathrm{SFT}}.
\]

- OPD 行由当前 student rollout 生成，仅计算 OPD policy loss；
- SFT 行必须包含标准 assistant response，仅计算 supervised token NLL；
- `prepare_single_task_data.py --sft-ratio` 控制两类样本数量比例；
- `HYBRID_OPD_LOSS_COEF` 与 `SFT_LOSS_COEF` 控制给定样本上的 loss 系数。

这里没有把 student 自己采样的 token 当作 SFT target，因为那是 entropy minimization，而不是真正的 SFT。

## 3. 模型准备

建议目录：

```text
../checkpoints/
  Qwen3-1.7B/
  Qwen3-1.7B_torch_dist/
  Qwen3-8B/
  Qwen3-4B-Thinking-2507/
```

下载模型并转换 student：

```bash
hf download Qwen/Qwen3-1.7B \
  --local-dir ../checkpoints/Qwen3-1.7B
hf download Qwen/Qwen3-8B \
  --local-dir ../checkpoints/Qwen3-8B
hf download Qwen/Qwen3-4B-Thinking-2507 \
  --local-dir ../checkpoints/Qwen3-4B-Thinking-2507

source scripts/models/qwen3-1.7B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint ../checkpoints/Qwen3-1.7B \
  --save ../checkpoints/Qwen3-1.7B_torch_dist
```

启动器在真正训练前运行 `validate_tokenizers.py`，逐项比较 student/teacher 的 token→ID 映射。当前官方三个 checkpoint 的 151,669 项词表、added tokens 与 special-token IDs 已核验一致；启动时仍会基于本地实际文件重检。这个检查不可省略：OPD teacher 评分的是 student 的原始 token IDs。

## 4. 准备现有 math/code/science 数据

先把 `nvidia/Nemotron-3-Nano-RL-Training-Blend` 转成 M2RL 格式：

```bash
python examples/optimizer_geometry/prepare_m2rl_data.py \
  --input /data/Nemotron-3-Nano-RL-Training-Blend/train_complete.jsonl \
  --output-dir /data/m2rl_rl \
  --sampling uniform \
  --sampling-unit batch \
  --seed 42
```

然后生成单任务 manifest。M2RL 的 messages-format SFT 文件通常是
`Nemotron-Math-v2.parquet` 与 `Nemotron-Competitive-Programming-v1.parquet`：

```bash
python examples/optimizer_geometry/prepare_single_task_data.py \
  --rl-manifest /data/m2rl_rl/multitask_manifest.yaml \
  --output-dir /data/opd_single \
  --tasks math code science \
  --sft math=/data/m2rl_sft/Nemotron-Math-v2.parquet \
  --sft code=/data/m2rl_sft/Nemotron-Competitive-Programming-v1.parquet \
  --sft-ratio 0.5 \
  --eval math=/data/eval/math500.parquet \
  --eval-name math=math500 \
  --eval-samples math=1 \
  --eval-temperature math=0 \
  --eval-top-p-override math=1 \
  --exclude-eval-overlap math \
  --eval science=/data/eval/gpqa_diamond.parquet \
  --eval-name science=gpqa \
  --eval-samples science=4 \
  --skip-eval code \
  --seed 42
```

没有 `--eval`/`--skip-eval` 时才会从 RL JSONL 做 holdout；这只适合 pipeline smoke test，
不应代替论文 benchmark。M2RL 训练期与论文最终评测的集合应分开记录。

code reward 会执行模型生成的程序，必须使用 SandboxFusion 等隔离服务，并在
`configs/rewards.example.yaml` 中设置 endpoint；不能连接到无隔离的通用 shell 服务。

本仓库提供基于官方固定 digest 的 cgroup-v2 派生镜像、仅监听 localhost、
`SANDBOX_CONFIG=ci` 的启动配置，以及执行、宿主/服务临时文件不可见、逐请求 namespace、
禁公网、只读 cgroup/实际内存限制、降权、seccomp/userns 拒绝、timeout 和 teardown 主动探针：

```bash
export HOST_ROOT=/mnt/data_from_server2/siqizhu4
export DISK2_ROOT=/mnt/disk2_from_server2/siqizhu4/iclr2027
export SLIME_REPO="$DISK2_ROOT/slime_opd_geometry"
export TRAIN_CACHE="$DISK2_ROOT/.training-cache/root-cache"
export SANDBOX_STATE="$HOME/.local/state/slime-opd-geometry/sandboxfusion"

test -f "$SLIME_REPO/examples/optimizer_geometry/run_single_task_rl.sh"
mkdir -p "$TRAIN_CACHE" "$SANDBOX_STATE"
chmod 700 "$SANDBOX_STATE"
cd "$SLIME_REPO"

SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh
SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
SANDBOX_PREFLIGHT_MARKER="$SANDBOX_STATE/sandboxfusion_preflight.json" \
  bash examples/optimizer_geometry/start_sandboxfusion.sh
```

该流程要求 Linux 宿主机提供 cgroup v2、Docker daemon，以及经过白名单限定的 mount/netns
capabilities；启动器会明确拒绝 privileged container。当前训练容器没有 Docker socket 且不允许
unshare，已确认会 fail closed；必须由服务器
管理员在宿主机构建并启动，不能改用 SandboxFusion 的 `local`/`isolation=none` 配置。每个
code cell 前应重跑主动探针，launcher 和 reward route 会校验 marker 的 URL、时效、patch
hash、固定 upstream digest、cgroup v2 attestation 和 `safe=true`。启动前旧 marker 会先变为
`safe=false`，任何失败都会关闭新服务。完整操作见 `SANDBOXFUSION_CGROUP2_zh.md`。

训练容器必须通过 `--network host` 访问服务，并把上述状态目录只读挂载到
`/workspace/sandboxfusion-state`。进入训练容器后使用：

```bash
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
export M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json
```

## 5. 运行一个实验

下面把 4 张卡给 student train/rollout，第 5 张卡给 teacher：

```bash
export AVAILABLE_CUDA_DEVICES=1,4,6,8,7
export OUTPUT_ROOT=/data/opd_geometry_runs
export REWARD_CONFIG=$PWD/examples/optimizer_geometry/configs/rewards.example.yaml

TASK=math \
TEACHER=qwen3-8b \
ALGORITHM=opd \
OPTIMIZER=adamw \
DATA_MANIFEST=/data/opd_single/math/math_on_policy.yaml \
EVAL_CONFIG=/data/opd_single/math/math_eval.yaml \
  bash examples/optimizer_geometry/run-qwen3-1.7B-student-teacher.sh
```

loss 条件对应关系：

| `ALGORITHM` | 数据 manifest | 训练信号 |
| --- | --- | --- |
| `opd` | `TASK_on_policy.yaml` | pure OPD，task reward weight = 0 |
| `sft_opd` | `TASK_sft_opd.yaml` | 两类样本的 SFT NLL + OPD |
| `grpo_opd` | `TASK_on_policy.yaml` | GRPO task advantage + OPD |
| `ppo_opd` | `TASK_on_policy.yaml` | PPO/GAE task advantage + OPD，critic 固定 AdamW |

完成独立 tuning 后冻结的主对照参数（仅在做明确标注的非主实验时覆盖）：

```bash
OPD_KL_COEF=1.0
OPD_TASK_REWARD_WEIGHT=1.0  # 只用于 grpo_opd / ppo_opd
SFT_LOSS_COEF=1.0           # 只用于 sft_opd
HYBRID_OPD_LOSS_COEF=1.0    # 只用于 sft_opd
GEOMETRY_INTERVAL=16          # 仍是每 256 prompts 一次
GEOMETRY_PROJECTION_DIM=256
WEIGHT_DECAY=0              # 主几何对照固定为 0
```

## 6. 运行矩阵

当前预算缩减矩阵跑 3 tasks × 1 teacher × pure OPD × 3 optimizers × 1 seed，共 9 个 cell，
对应第 0 节列出的 9 个独立脚本。单 seed 结果只能作为描述性比较，不能报告跨 seed 方差或
Student-t 置信区间。
显式扩展算法时可用：

```bash
export SINGLE_TASK_CONFIG_ROOT=/data/opd_single
export AVAILABLE_CUDA_DEVICES=1,4,6,8,7
export OUTPUT_ROOT=/data/opd_geometry_runs

TASKS="math code science" \
TEACHERS="qwen3-8b" \
ALGORITHMS="opd sft_opd grpo_opd ppo_opd" \
OPTIMIZERS="adamw sgd muon" \
SEEDS="42" \
  bash examples/optimizer_geometry/run_single_task_matrix.sh
```

这会产生 36 个 cell。学习率必须先按第 1.1 节的等预算 tuning 规则确定并冻结，不再使用论文
报告任务的 reward 选参。
比较 optimizer 时保持初始化、rollout seed、采样温度、batch/token budget、clip、几何间隔完全一致。

## 7. 混合 loss 系数扫描

固定 task、teacher 与 optimizer 后，可单独扫描 OPD/SFT/RL 的相对系数：

```bash
export SINGLE_TASK_CONFIG_ROOT=/data/opd_single
export AVAILABLE_CUDA_DEVICES=1,4,6,8,7

TASK=math \
TEACHER=qwen3-8b \
OPTIMIZER=adamw \
  bash examples/optimizer_geometry/run_mixed_loss_sweep.sh
```

默认覆盖 all-OPD reference、固定 hybrid batch 组成下的 SFT-only/OPD-only
端点与三档 SFT:OPD 权重，以及 pure GRPO/PPO 与三档
OPD:task-reward 权重。可通过空格分隔的
`MIXTURE_SETTINGS="algorithm:opd_kl:sft:hybrid_opd:task_reward ..."`
覆盖默认设置。SFT/OPD 的**样本比例**与 loss 系数是不同因素：前者由准备数据时的
`--sft-ratio` 控制，若要研究二者交互，应为每个 ratio 生成独立 config root。

## 8. 独立最终评估

训练脚本会在训练前和训练中每 50 个 updates 评估一次。Math 的 response cap 为 32,768，
Code/Science 为 16,384；三者
共享独立于训练 server cap 的 eval-only 全局并发 48。也可对任一 checkpoint 做 eval-only：

当前 4 个单卡 rollout engine 下，48 个全局 eval slot 平均为 12 requests/engine。已有同机
Qwen3-1.7B 长序列 smoke 在 64 个请求、平均 7,085 output tokens 时用 181.1 秒；在没有真实 32k
Math eval pilot 前，Math 按每个 48-request cap-hit wave 约 12--16 分钟、其他任务仍按 6--8 分钟做保守排程：

| eval | prompts | 每题采样 | 每次生成请求 | 48 并发 waves | cap-hit 排程估计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Math / AIME'24 online | 30 | 8 | 240 | 5 | 约 1--1.3 小时 |
| Math / MATH-500 online | 500 | 1 | 500 | 11 | 约 2.2--3.0 小时 |
| Math / AIME'24 + MATH-500 | 530 | 各自配置 | 740 | 16 | 约 3.2--4.3 小时 |
| Code / LiveCodeBench online | 64 | 1 | 64 | 2 | 约 12--16 分钟，另加 sandbox |
| Science / GPQA-Diamond online | 198 | 4 | 792 | 17 | 约 1.7--2.3 小时 |
| Code / LiveCodeBench final | 880 | 10 | 8,800 | 184 | 约 18--25 小时，另加 sandbox |

这是所有 response 都接近各自 cap 的容量排程，不是已实测的 eval wall-clock；若实际平均 response
长度为 `L`，Math 生成部分可先近似乘以 `L/32768`，其他任务乘以 `L/16384`。Code 的 eval semaphore 覆盖生成和 reward，且
SandboxFusion 另有并发 8，所以代码执行会增加长尾。按当前单轮步数，Math 的 eval 点为
`0,50,...,300,345`（8 次），Code 为 `0,50,...,250,299`（7 次），
Science 为 `0,50,...,300,308`（8 次）；首次 pre-train eval 应作为后续排程的实测校准值。

```bash
LOAD_CHECKPOINT=/data/opd_geometry_runs/RUN/checkpoints \
TASK=math EVAL_MAX_RESPONSE_LEN=32768 \
DATA_MANIFEST=/data/opd_single/math/math_on_policy.yaml \
EVAL_CONFIG=/data/opd_single/math/math_eval.yaml \
OUTPUT_DIR=/data/opd_geometry_runs/RUN/final_eval \
CUDA_VISIBLE_DEVICES=1,4,6,8 NUM_GPUS=4 \
  bash examples/optimizer_geometry/evaluate_single_task.sh
```

Code 最终表应把 `EVAL_CONFIG` 换成 `code_eval_final.yaml`。结果除聚合曲线外，还会写逐样本
response、label、reward、status、长度和版本信息；final eval 完成后运行产物校验器。

## 9. 几何与 reward 汇总

当前实现的主几何量不依赖事后加载 checkpoint：observer 在每个 optimizer update 前后在线扫描，
直接写出精确的 raw/clipped gradient、optimizer direction、FP32 intended data/WD update、模型 dtype
realized update、displacement、dot/cosine、ULP 实现率和真实 optimizer 分支能量占比。带 `_sketch`
的 CountSketch、坐标直方图/支持窗口和固定抽样矩阵指标只按低频 interval 运行；OPD/RL 有效 token/
sequence/相对位置分布写入 `geometry/rollout/metrics.jsonl`，训练 prompt/response/label/reward/版本逐样本
原子写入 `geometry/rollout/samples/*.jsonl`。保留 `geometry/actor/` 即可复算现有 analyzer 的标量、
相邻步 cosine 和 task-centroid cosine。checkpoint 仍用于精确 resume、独立 final eval，以及在同一个
参数状态上分别回放 math/code/science 固定 probe batch 的 same-checkpoint per-task gradient 因果对照；
最后一种量不能由当前在线、不同参数状态的梯度记录替代。

每个训练 run 的核心文件是：

```text
RUN/
  provenance/run_manifest.json
  provenance/source_snapshot.tar.gz
  provenance/inputs/*          # data/reward/eval/teacher/role config 的精确副本
  metrics/{rollout,train,eval,geometry,forgetting}.jsonl
  geometry/rollout/metrics.jsonl
  geometry/rollout/samples/*.jsonl
  geometry/actor/metrics.jsonl
  geometry/actor/exact_reference/rank_*.pt
  geometry/actor/support_state/rank_*.pt
  geometry/actor/vectors/*.pt
  forgetting/metrics.jsonl
  eval_artifacts/index.jsonl
  eval_artifacts/<dataset>/*.jsonl
  checkpoints/                 # PPO 另有 checkpoints/critic/
  wandb_run_id.txt
  wandb/
  run_complete.json
```

`run_complete.json` 只在最终 eval 和预期 update 都完成后写入；`run_failed.json` 表示失败，二者
不得并存。以下命令流式核对 update 数、最终逐样本行数/SHA/schema、几何、checkpoint、provenance
和 W&B run id，再生成 paired effect、95% CI、CSV/LaTeX 表及 PNG/PDF 图：

```bash
python examples/optimizer_geometry/validate_run_artifacts.py /data/opd_geometry_runs/RUN
python examples/optimizer_geometry/paper_statistics.py /data/opd_geometry_runs/RUN_* \
  --output-dir /data/analysis/paper
```

展开逐层/逐步几何：

```bash
python examples/optimizer_geometry/analyze_geometry.py \
  /data/opd_geometry_runs/RUN/geometry/actor \
  --output-prefix /data/analysis/RUN_geometry
```

把最终 reward 与 global geometry 合并为一行一个 run 的 CSV：

```bash
python examples/optimizer_geometry/summarize_single_task.py \
  /data/opd_geometry_runs/RUN_A \
  /data/opd_geometry_runs/RUN_B \
  --output /data/analysis/single_task_summary.csv
```

主结果至少报告：task reward、sampled reverse KL、gradient norm、update/weight ratio、effective step size、gradient-update cosine、相对初始化 displacement，以及这些量的 layerwise 分布。不要跨 seed 直接比较 CountSketch 坐标；本仓库的 analyzer 只在同一 run 内计算向量 cosine。

## 10. 向统一理论衔接

可以把 optimizer 写成局部预条件更新

\[
\Delta\theta_t=-\eta_t P_t g_t,
\]

其中 $g_t$ 是 OPD、SFT 或 RL+OPD 的总梯度，$P_t$ 由 AdamW/SGD/Muon 的状态与结构决定。实验中直接测量

- $\|g_t\|,\|\Delta\theta_t\|,\|\theta_t-\theta_0\|$；
- $\cos(g_t,\Delta\theta_t)$ 与有效步长；
- layer/module 的更新集中度；
- 上述几何量对 reward 增益和遗忘的预测能力。

单任务先辨认“哪种预条件几何带来更有效的 teacher imitation / reward transfer”，多任务 continual learning 再研究同一几何是否对应正迁移或 interference。这样基础任务与拓展任务使用同一组可观测量，而不是两套互不相连的叙述。

这里应把“统一理论”定位成可证伪的局部预条件框架，而不是直接声称坐标无关的普适定理：欧氏参数几何会随重参数化改变。因此所有比较都固定 student 架构与参数化，并同时锚定 function-space 的 sampled KL、task reward；混合系数会机械地缩放梯度范数，结论应优先依赖方向、归一化更新量及其对 reward 的预测，而不只比较原始 norm。
