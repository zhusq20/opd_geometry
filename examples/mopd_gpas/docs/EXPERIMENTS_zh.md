# 两周核心实验协议（v6，2026-09-06 前缀修正）

本实现以相邻论文的 `EXPERIMENT_PLAN_QWEN3_1.7B_4T_MOPD_GPAS_zh.md` 为依据。完整训练仅四个 run；s1 对应整数 42。所有结果须由实际运行产生，仓库中的旧输出不属于这次实验。

## 训练目标与分配

学生 Qwen3-1.7B-Base，non-thinking；math/code/IF/science 分别使用对应领域的 Qwen3-1.7B RL teacher。每域冻结 16,000 条有效 prompt，使用相同域内排列且无放回消费。每个 micro-batch 为同域 4 prompts × 1 response；每步 16 个 micro-batches，计数边界 `[2,8]`，500 步共 32,000 条计划 responses。

师生使用不同前缀。student 的 chat template 仍设置 `enable_thinking=false`，之后仅删除末尾精确字符串 `<think>\n\n</think>\n\n`，角色边界保持不变。student rollout 和 learner 前向均接收 `prompt + response`。teacher router 的每个 route 配置同一 `prompt_suffix`，在 student prompt 与 response 的 token 边界插入它，接收 `prompt + 空 thinking 块 + 同一 response`。仅按 response 相对位置取 teacher 分布；额外前缀不占 student loss 位置，response token IDs、EOS 和 top-64 目标不变。

`protocol.json` schema 为 6，`prompt_format` 同时记录 `student_chat_template_suffix_to_remove` 与 `teacher_prompt_suffix`。训练 manifest 保留 version 4，并新增相同声明。预渲染的 heldout/diagnostic prompts 已完成删除，对应数据配置将 `chat_template_suffix_to_remove` 显式设为 null，避免重复处理。能力评测的学生输入遵循相同规则；教师能力评测覆盖此配置为 null，以保留其空 thinking 块。

每个 response 先对有效位置平均，再对 responses 平均。每位置 loss 为 teacher top-64 上的 `sum[p*(log p-log q)-p+q]`；p/q 均保留完整词表归一化，保留 `-p` 的梯度。SGLang 返回 teacher 的 `input_top_logprobs`；learner 从完整 logits 中提取相同 token IDs。截断 response 保留有效 token。缺失或无效的 dense teacher targets 会使本次执行失败，不以 sampled-token loss 或虚构 penalty 替代；恢复和失败记录由 provenance 保留。

| Run ID | 分配统计 | 梯度汇总 |
|---|---|---|
| `uniform-s1` | `(4,4,4,4)` | `sum_i (1/4)*mean_s(g_i,s)` |
| `gpas-s1` | 更新前 bias-corrected Adam 二阶矩定义 D；Welford 估计噪声 | 同上 |
| `gpas-raw-s1` | 统计使用 D=I，优化器仍为 AdamW | 同上 |
| `d3-fixed-s1` | remaining gap × descent velocity | 同上 |

G/R 首步 Uniform、D=I，噪声 EMA decay=0.9，首个观测直接初始化。下一步读取已完成步骤的 EMA。穷举 149 个计数向量，最小化 `sum(w_i²*e_i/m_i)`；并列优先最近 Uniform，再按 math/code/if/science 字典序。U/H 不采集 micro-batch 噪声。G/R 的统计耗时和内存计入自己的运行成本。

D³ 保留固定实现的 first-5 initial loss、EMA window 10、W=10、最多 3 个窗口、每 10 步更新、KL floor 0.15、max-normalization、temperature 0.5、probability floor 0.10、jitter 0.30。warmup 也使用 jitter，最早在第 20 个观测后更新调度信号；概率通过最小化 `sum(m_i-16*p_i)²` 投影到相同的 149 个向量。版本记为 `d3-table3-synchronous-v1-integer-projection`，方法名为 **D³ signal, fixed weights**。

AdamW 配方沿用已有 lr=2.5e-7、betas=(0.9,0.98)、eps=1e-8、constant schedule、weight decay=0、全局 clipping=1。每批仅做一次更新，之后同步学生权重。

## 固定 loss 和能力评估

`prepare` 生成等权 `train.yaml`、每域 64 prompts 的 `teacher_loss_eval.yaml`、独立 `diagnostic.yaml` 和 `protocol.json`。无需 `initial_kl.json`。旧 v3 训练 manifest 不能用于 dense 新协议。

本次默认使用 `local/mopd_no_think_generated` 与 `outputs/mopd_no_think`。现有环境文件须更新这两个目录并重新 `prepare`，从初始 Base 开始训练。旧 v5 的 bank、diagnostic prompts 和 sampler/checkpoint 状态不符合 v6，不能跨协议恢复。旧评测结果保持原始记录，分析脚本仍支持显式指定 v5 输入。

训练自动创建公共 `reference_bank.pt`：初始学生生成一次 256 responses、缓存 teacher top-64；初始评分也共享一次。所有方法在 0/100/200/300/400/500 步由 learner 评分同一 bank。最终模型另生成一次 fresh bank，0 步 fresh loss 等于初始 bank loss。固定与 fresh loss 写到每个 run 的 `fixed_loss/`。

能力评估采用 MATH-500 greedy pass@1、固定 LiveCodeBench 切片 pass@1、IFBench strict accuracy、GPQA-Diamond average@4。GPQA 先平均同题四次回答，再平均题目。仅评初始学生一次；U/G 的 250、500；R/H 的 500，合计七组学生 checkpoint。四个 Qwen3-1.7B RL teacher 各评对应域。

```bash
bash examples/mopd_gpas/run_stage.sh capability all
# 可选：只产生 math/IF/science 三域结果，沿用现有无沙箱子集。
MOPD_CAPABILITY_SUITE=noncode bash examples/mopd_gpas/run_stage.sh capability uniform-s1
```

非代码子集写入 `capability_eval_noncode` / `capability_references_noncode`，不进入四域完整主表。完整 Code benchmark 沿用现有 SandboxFusion 配置；其他任务不要求 Code 沙箱。保存题目级结果，按配对题目 bootstrap；GPQA 的四次回答作为一簇。只有一个训练种子，不报告跨训练种子的标准差。

## Uniform/250 机制对照

```bash
bash examples/mopd_gpas/run_stage.sh mechanism
```

只恢复 `uniform-s1/checkpoints/iter_0000249`（第 250 次更新后）。evaluation bank 为每域 64 个独立 diagnostic prompts；calibration 为每域 16 个 micro-batches。先固定 GPAS 计数，再分别独立生成两分支各 10 次 64 responses。每次 trial 恢复相同模型、AdamW、LR 和随机状态，rollout engine 始终保留更新前模型，更新后的模型仅评分同一 evaluation bank。

用固定 pre-step D 对 clipping 前 A 的 trial 间方差做 Welford 累积，只保留每分支一个 CPU 均值 buffer。输出原始方差、G/U 方差比、逐域实际 loss 下降、非下降频率和全部 trial 点。evaluation-bank bootstrap 索引在全部 trials 中共享，trial 重采样在分支间独立。不测 evaluation 梯度、不构造方向协方差或 K 矩阵，也不选择其他 checkpoint 来替代无收益的结果。

产物位于 `common-checkpoint-250/`，包括 `calibration.json`、`before.json`、20 条 `trials/*.json` 和 `common_checkpoint.json`。生成预算 1,792 responses，更新后评分 5,120 次 response forwards。

## 运行、恢复与分析

```bash
bash examples/mopd_gpas/run_stage.sh prepare
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_stage.sh dry-run
bash examples/mopd_gpas/run_stage.sh train all
bash examples/mopd_gpas/run_stage.sh resume gpas-s1  # 仅中断时
bash examples/mopd_gpas/run_stage.sh capability all
bash examples/mopd_gpas/run_stage.sh mechanism
bash examples/mopd_gpas/run_stage.sh analyze
```

`uniform`、`gpas`、`raw_noise`、`d3_fixed` 仍可作为四个 run ID 的简写。主训练保存 100/200/300/400/500 步 HF 权重，U/G 额外保存 250；完整状态保留最新恢复点与 U/250。20-step 和 1-step smoke 命令保留为手动工程调试工具，不属于核心实验清单或开跑 gate。

默认 `analyze` 使用 `analyze_core.py`，输出原始百分比分数、算术均值、逐域及最差域相对 Uniform 的差值、fixed/fresh loss 和训练 GPU hours。达到共同目标 `F_ref,U(500)` 的成本按首次向下穿越的相邻点插值；未达标记 `unreached`，不外推。图包括固定 loss 对 steps/GPU hours、U/G 的三点能力曲线、局部机制图和 GPAS 分配/噪声轨迹。

训练计时包括 rollout、teacher scoring、梯度、统计、同步和等待；checkpoint 的占用另记 `checkpoint_costs.jsonl` 并纳入分析。`--mopd-occupied-gpus` 包含实际独占的 learner、rollout 和外部 teacher GPUs；设备型号及分配由 provenance 记录。不同硬件配置须在结果中分列。评估和机制诊断独立计费，失败重算由 provenance/resume archive 单列。默认两卡设置为一张 96GB learner 加一张 48GB inference；已有 TP2/分布式 teacher 环境参数仍可使用，必须记录实际设备和成本。

研究总 GPU hours 按 provenance 中各次执行的起止时间累计，包含失败重算及启动开销，排除中断后等待恢复的离线间隔。若强制退出未留下终止时间，则总成本标记为未知，保留已记录的训练、评估和诊断分项。

总计划生成量为 128,000 训练 + 1,280 长期 loss banks + 1,792 局部诊断 = 131,072 responses，能力 benchmark 和重算另计。旧八配置、单任务参照、cost-aware、完整外部配方及旧多 checkpoint probe 均不进入本轮自动执行和分析。
