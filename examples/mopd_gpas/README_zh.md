# Qwen3-1.7B 四任务 micro-batch MOPD / GPAS

本目录实现相邻论文仓库实验计划的 v4 协议，不修改论文正文。每条训练轨迹使用一张 96GB 训练卡和一张 48GB 推理卡；推理卡同时常驻 student rollout engine 和四个 teacher endpoint。math、IF teacher 是 Qwen3-1.7B 领域 RL checkpoint，code、science 都使用本地 `Qwen/Qwen3-4B` 原版权重（两个独立 endpoint、同一模型目录），全程 `enable_thinking=false`。

固定八个配置：`uniform`、`gpas`、`cost_gpas`、`raw_noise`、`loss_gap`、`std_mopd`、`d3_mopd`、`open_mopd`。每个配置 seed 42，500 个 optimizer step；每步 16 个 task micro-batch、每个 4 prompt、每 prompt 1 response，因此严格消耗 32,000 条 attempted response。每任务训练流独立且不重复；候选集做确定性洗牌后，各自固定前 16,000 条有效 prompt。

`d3_mopd` 按 [D³-MOPD](https://arxiv.org/abs/2608.24987) 论文 Table 3 的数值实现动态调度。`open_mopd` 按 [Open-MOPD](https://arxiv.org/abs/2608.19098) 实现 token-share balancing 与 forward gap-following；它明确是当前主协议的 K=1 sampled-token 适配版。K=1 时 reward refresh 数学上为恒等操作，因此本 launcher 不声称复现论文另一套 K=4、student-top-k=16 dense reward 系统。

论文完整系统已作为独立的 [`open_mopd_full`](open_mopd_full/README_zh.md) 复现通道加入：固定官方代码 commit、公开 SmolLM3-3B student/三教师/数据，运行 K=4、dense student-top-k=16 和 reward refresh。它作为论文协议的外部参考单独报告，不混入控制变量完全不同的 Qwen3 四任务主表。

执行顺序：

```bash
cp examples/mopd_gpas/configs/site.example.env local/mopd.env
# 编辑本机路径与两张 GPU 编号
source local/mopd.env

bash examples/mopd_gpas/run_stage.sh fetch-assets
bash examples/mopd_gpas/run_stage.sh prepare-heldout
bash examples/mopd_gpas/run_stage.sh convert-teachers
bash examples/mopd_gpas/run_stage.sh measure-initial
bash examples/mopd_gpas/run_stage.sh prepare
bash examples/mopd_gpas/run_stage.sh preflight
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_stage.sh smoke
bash examples/mopd_gpas/run_stage.sh train uniform
bash examples/mopd_gpas/run_stage.sh baselines
```

完整 Open-MOPD 使用单独的 1x8 GPU 论文协议：

```bash
bash examples/mopd_gpas/run_stage.sh open-full-fetch all
bash examples/mopd_gpas/run_stage.sh open-full-train --dry-run
bash examples/mopd_gpas/run_stage.sh open-full-train --run
```

中断后使用 `bash examples/mopd_gpas/run_stage.sh resume uniform`，不要在原目录重新启动 fresh run。Uniform 完成后运行 `variance all`，它在 step 50/250/500 各生成每任务 32 个新 micro-batch，只落盘标量范数，并用对应 checkpoint 中训练期的 `tau_i`/`C` 做 Cost-GPAS 反事实分配。能力评测用 `capability all`：先按[环境配置](docs/SETUP_zh.md#livecodebench-capability-sandbox)启动并验证 LiveCodeBench 专用 SandboxFusion，再评初始学生、math teacher、IF teacher 和共享的 Qwen3-4B teacher，最后评八个最终模型。训练、smoke test 和 held-out teacher-loss 评测不使用该 sandbox。全部结果到齐后运行 `analyze`；分析会直接生成 JSON、主表 CSV、held-out 方差 CSV 和 PDF 图组。

每 50 step 的 Hugging Face 权重全部保留；完整 optimizer state 只保留当前最新 resume 点，以及 Uniform 的 step 50/250/500 三个方差检查点。

详细说明见 [环境配置](docs/SETUP_zh.md)、[实验协议](docs/EXPERIMENTS_zh.md) 和 [协作手册](docs/COLLABORATION_zh.md)。
