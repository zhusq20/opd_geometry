# 多教师 OPD：token 平衡、参数更新稀疏性与监督密度

**SmolLM 协作者入口：** [MixSFT + PG / Top64 交集运行说明](README_SMOLLM3_zh.md)，包含无需 sandbox 的训练流程和独立能力评测 setup。

当前入口对应[最新版论文三项研究计划](../../../Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/MOPD_THREE_CONTRIBUTIONS_2026-09-05_zh.md)。目录名沿用历史名称，默认执行六组配置：

| 配置 | 教师 | 损失 | 平均方式 |
|---|---|---|---|
| `s-pg` | 代表性单域 RL teacher | sampled-token PG | domain-response mean |
| `s-tk` | 同一个 RL teacher | student Top16 | domain-response mean |
| `m-pg` | 所有路由 RL teachers | sampled-token PG | domain-response mean |
| `m-tk-dr` | 所有路由 RL teachers | student Top16 | domain-response mean |
| `m-tk-dt` | 所有路由 RL teachers | student Top16 | domain-token mean |
| `m-tk-gt` | 所有路由 RL teachers | student Top16 | global-token mean |

TopK 入口使用 `--mopd-loss student_topk`，**默认 K=16**；`--mopd-topk` 支持 16 和 64。rollout 时由 student 在每个前缀选择 K 个 token，teacher 对相同 IDs 打分。`log_p`、`log_q` 是这些 IDs 对应的全词表 log-prob，每个前缀的 loss 为：

```python
advantage = (log_p.softmax(-1) * (log_q - log_p)).detach()
loss = -(advantage * log_p).sum(-1)
```

只对 student 的选集权重做每前缀归一化；log-ratio 保留全词表概率。actor 每次 forward 在保存的 rollout 选集上重新计算 student 概率与 detached advantage。这采用 [Open-MOPD 的 TopK advantage](https://github.com/BytedTsinghua-SIA/Open-MOPD/blob/4809a96cf85a869106ff0ff3f37d0a51e12010ae/training/verl/verl/workers/actor/dp_actor.py#L672)，不加入 PPO ratio、PPO 裁剪或额外领域加权，表中的 reduction 保留。SGLang 的 `token_ids_logprob` 每个请求只接受一组 IDs，因此 teacher 分块请求选集并集，再逐位置取回各自的 TopK。Top16 每块 32 个位置，Top64 每块 8 个位置，均将并集限制在 512 个 IDs 内，并严格验证位置与 ID 对齐。

日志分别记录可正可负的 `student_top16_normalized_logratio`（或 `student_top64_normalized_logratio`）、surrogate 值，以及 student 和 teacher 在 student 选集上的概率质量；该 log-ratio 指标不是非负 KL。历史 `teacher_topk` 仍表示 corrected teacher Top64。切换 loss 或 K 时使用新 run ID；resume 会拒绝修改原 run 的 loss 或选集大小。

两套 profile 复用相同训练和记录格式：`qwen3` 使用 Qwen3-1.7B-Base、四个 Qwen3-1.7B RL teachers 与现有数学/代码/IF/科学数据；`smollm3` 使用 Open-MOPD 发布的 SmolLM3-3B **MixSFT** student、数学/代码/IF RL teachers 与 Open-MOPD-Data。SmolLM 使用已发布的 MixSFT 权重与 tokenizer；新资产和输出使用 `smollm3_mixsft` 目录，避免与历史 Base run 混用。prepare 会记录模型 revision、数据身份与前缀规则。

在仓库根目录的新 shell 中选择一个 profile，并配置本站 GPU 和资产路径：

```bash
source examples/mopd_gpas/configs/smollm3.env  # 或 configs/qwen3.env
bash examples/mopd_gpas/run_stage.sh fetch-assets
bash examples/mopd_gpas/run_stage.sh prepare
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_mopd.sh m-tk-dr
# 执行六组矩阵：
bash examples/mopd_gpas/run_mopd_matrix.sh
bash examples/mopd_gpas/analyze_all.sh
```

student Top64 reverse-KL 实验沿用同一套模型 profile 和本站配置：

```bash
bash examples/mopd_gpas/run_student_top64.sh            # 默认 m-tk64-dr：多教师，domain-response mean
bash examples/mopd_gpas/run_student_top64.sh s-tk64     # 单教师；MOPD_SINGLE_TASK 默认为 math
bash examples/mopd_gpas/run_student_top64.sh m-tk64-dt  # 多教师，domain-token mean
bash examples/mopd_gpas/run_student_top64.sh m-tk64-gt  # 多教师，global-token mean
```

这些配置使用上面的归一化 detached loss，将 K 改为 64，也可通过 `run_mopd.sh` 启动。默认 run ID 如 `m-tk64-dr-s42`，checkpoint 与 W&B 记录独立于 Top16。设置 `DRY_RUN=1` 可仅验证启动参数。原来的六组默认矩阵保持不变。

`MOPD_SINGLE_TASK=math` 选择代表性单教师。训练 response 上限为 **4096 tokens**，能力评估上限为 **32768 tokens**，同时受原生上下文减去实际 prompt 长度后的剩余空间约束。Qwen3 保持原生 32768-token 上下文，因此评估 response 的实际预算为 `min(32768, 32768 - prompt_tokens)`；SmolLM3 使用原生 65536-token 上下文。逐条评估产物和 W&B 能力指标记录实际生成预算。默认 500 次更新。多域保持相等 prompt 配额，三种 reduction 只改变损失平均方式。两套配置的资产、生成文件和结果分别位于 `local/mopd_<profile>_*` 与 `outputs/mopd_<profile>`。

默认启用 W&B，项目为 `iclr2027-mopd-dynamics`。通过 `WANDB_API_KEY` 认证；需要离线记录时设置 `WANDB_MODE=offline`。所有标量同时写入 `metrics/*.jsonl`。OPD 仍调用各域 reward scorer，记录 `rollout/reward/<task>/mean`、分位数与 pass rate；`rollout/reward_used_in_loss=0` 明确这些分数用于观察进度，训练优化教师蒸馏损失。代码奖励使用配置的 SandboxFusion 服务与已有 preflight marker；通过 `SANDBOXFUSION_BASE_URL`、`M2RL_SANDBOX_PREFLIGHT_MARKER` 指向本站服务。

记录包含 response 长度、截断/完成比例、各域有效 token 份额、更新步数、累计 token 暴露与实测 GPU 成本。能力图同时按更新步数、token 暴露和 GPU 小时绘制。诊断记录存于每个 run 的 `paper/`：原始梯度、拟执行 Adam 更新、FP32 累计参数变化、BF16 checkpoint 变化分别报告。teacher overlap 保留教师对、support fraction、层、checkpoint 与 prefix draw；JS 散点只匹配同一 checkpoint 和相同公共 prefixes 的记录。full-vocabulary 诊断表示局部监督对照，PG/Top16 在线训练表示累计轨迹。

对保存的 checkpoint，在可用 GPU 上执行公共前缀测量：

```bash
RUN_DIR="${MOPD_OUTPUT_ROOT}/m-tk-dr-s42"
python examples/mopd_gpas/probe_checkpoint.py \
  --snapshot "${RUN_DIR}/paper/checkpoint_step_0250.pt" \
  --manifest "${MOPD_GENERATED_DIR}/diagnostic.yaml" \
  --teachers "${MOPD_TEACHER_ROUTER_CONFIG}" \
  --output "${RUN_DIR}/paper/probe_step_0250" \
  --wandb-project "${WANDB_PROJECT}" --wandb-mode "${WANDB_MODE:-online}"
```

单教师和多教师实验各自在选定的早期/中期/最终 checkpoint 重复此命令。每个教师/损失分支从导出的同一 FP32 master weights 与 Adam 状态开始；记录固定 batch 的长度—梯度精确分解，并在相同 prefixes 上计算 teacher JS。TopK probe 自动读取 snapshot 的 K（旧 student snapshot 默认为 16）；标记为 `teacher_topk` 的 snapshot 继续使用 teacher Top64，也可用 `--topk-loss`、`--topk` 显式选择。主 overlap 热图/散点展示 top-5% 优化器更新并保留 loss 与 K 标记；全部阈值、top-1/5/10% supports、损失、层与重复抽样保留在 CSV 中。

`analyze_all.sh` 导出 CSV、PDF 与 PNG。没有实测数据的项目不生成虚构结果；`report.json` 保留运行状态、原始记录和已生成图表清单。`supervision_comparison.csv` 汇集带明确 loss 标记的 PG/TopK/full-vocabulary 对照数据。训练完成时，W&B plot-data artifact 包含标量日志及紧凑的论文诊断文件。

历史 GPAS 分配实验及分析器保留用于旧结果复查，不属于当前六组默认矩阵。当前入口与输出以本页为准；历史 `docs/EXPERIMENTS_zh.md`、`analyze_core.py` 描述的是旧协议。

本站配置、自动检查与真实模型验证证据见 [2026-09-06 实现验证记录](VALIDATION_2026-09-06.md)。
