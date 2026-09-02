# 实验协议与运行手册

## 固定协议

| 项目 | 固定值 |
|---|---|
| student | Qwen3-1.7B |
| tasks | `math`, `code`, `if`, `science` |
| seed | 42 |
| task unit | 16 prompts × 4 responses = 64 attempted responses |
| response cap | 8192 tokens |
| target task weights | 每项 0.25 |
| common warm start | 8 units，每任务 2 个，round-robin |
| 总训练时钟 | 64,000 attempted responses，包含 warm 与 probe |
| teacher-loss eval | 2,048 / 4,096 / 8,192 / 16,384 / 32,768 / 49,152 / 64,000 |
| checkpoints | 16,384 / 32,768 / 64,000 |

主 campaign 只运行一个训练 seed。执行者不能修改 response budget、seed、task slice、采样参数、评测点或优化器超参数；否则结果属于另一个 campaign。

GPAS 对 task-local score EMA 做 bias correction，inclusion floor 为 0.05，`A_max=50` processed task units。`K=2` 使用全部六个二元集合。梯度先按共享 norm 1.0 clip，再乘最终 inclusion marginal 对应的 importance correction。一次 K-task AdamW update 使用 `beta1^K`、`beta2^K` 和复合 decoupled weight decay；taskwise 版本使用未混合的加权平方观测。

## 主配置

| 配置 ID | K | allocation | AdamW second moment |
|---|---:|---|---|
| `uniform_k1_conventional` | 1 | Uniform | conventional |
| `uniform_k1_taskwise` | 1 | Uniform | taskwise |
| `gpas_k1_taskwise` | 1 | GPAS | taskwise |
| `cost_gpas_k1_taskwise` | 1 | resident-aware Cost-GPAS | taskwise |
| `uniform_k2_taskwise` | 2 | Uniform exact set | taskwise |
| `cost_gpas_k2_taskwise` | 2 | set-aware Cost-GPAS | taskwise |
| `all_k4_taskwise` | 4 | all tasks | taskwise |
| `all_k4_conventional` | 4 | all tasks | conventional |

## 单条主轨迹

每个 worker 只运行协调者分配的一个 `CONFIG_ID`：

```bash
source local/mopd.env
CONFIG_ID=uniform_k1_taskwise

bash examples/mopd_gpas/run_stage.sh train "${CONFIG_ID}"
```

训练入口会使用 `MOPD_WARM_DIR/checkpoints`，从公共 warm sampler state 的 rollout 8 分叉。每个 config 的输出目录固定为：

```text
${MOPD_OUTPUT_ROOT}/${CONFIG_ID}-seed42/
├── allocation/allocation.jsonl
├── checkpoints/
├── metrics/
├── provenance/run_manifest.json
├── teacher_loss_eval/
└── run_complete.json
```

新运行拒绝写入非空同名目录。发生中断时保留目录并执行：

```bash
bash examples/mopd_gpas/run_stage.sh resume "${CONFIG_ID}"
```

恢复入口选择最后一个完整的 16k/32k/64k checkpoint，同时恢复 optimizer、RNG、采样器和 teacher residency。checkpoint 后未持久化的日志尾部会归档到 `provenance/resume_rewind_*.tar.gz`。

## 最终能力评测

训练成功后运行：

```bash
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
bash examples/mopd_gpas/run_stage.sh capability "${CONFIG_ID}"
```

评测包括：

- MATH-500 greedy pass@1；
- LiveCodeBench v6 online128 pass@1；
- IFBench strict；
- GPQA-Diamond average@4。

结果写入 `${CONFIG_ID}-seed42/capability_eval/response_64000/`。

## Frozen-gradient bank

- `warm` 只依赖公共 warm checkpoint；
- `middle` 和 `late` 分别依赖 `uniform_k1_taskwise` 的 32k 和 64k checkpoint；
- 每个阶段采集每任务 8 个完整 unit，不更新参数。

```bash
bash examples/mopd_gpas/run_stage.sh bank warm
bash examples/mopd_gpas/run_stage.sh bank middle
bash examples/mopd_gpas/run_stage.sh bank late
```

这三项由协调者指定一台机器运行，不应由所有 worker 重复执行。

## 成功标准与交付

一个 config 只有同时满足以下条件才标记为 `complete`：

1. 主目录 `run_complete.json` 的 `status` 为 `complete`；
2. allocation 最后一行的 `attempted_responses_after` 为 64,000；
3. `capability_eval/response_64000/run_complete.json` 为 complete；
4. W&B 上的 config ID、seed 和本地目录一致。

验证并打包集中分析所需文件：

```bash
bash examples/mopd_gpas/run_stage.sh package "${CONFIG_ID}"
```

包默认写入 `outputs/mopd_packages_64k_v3/`，不包含大 checkpoint 和 W&B cache。它适合交给协调者做最终分析；如果要让别人续训，必须另外传输完整 config 输出目录。

## 集中分析

协调者把公共 warm、八个 config、三项 bank 解包到同一个 `MOPD_OUTPUT_ROOT` 后执行：

```bash
bash examples/mopd_gpas/run_stage.sh analyze
```

报告写入 `outputs/mopd_reports_64k_v3/`，包含主 outcome 表、paired bootstrap、系统 trace、frozen-bank 结果、能力评测和论文图。
