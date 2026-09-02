# Qwen3-1.7B 四任务 exact-set MOPD / GPAS

这里是 64k MOPD/GPAS 实验的统一入口。协议使用 `math/code/if/science` 四个任务、一个 student rollout GPU 和一个可热切换 teacher GPU；八条 seed-42 主轨迹共享同一个 warm checkpoint，之后可以分发到不同机器独立运行。

## 文档入口

- [环境与资产配置](docs/SETUP_zh.md)：新机器、Hugging Face 资产、GPU/W&B/SandboxFusion。
- [实验协议与运行手册](docs/EXPERIMENTS_zh.md)：固定超参数、配置 ID、命令、恢复和成功标准。
- [多人并行协作](docs/COLLABORATION_zh.md)：任务 DAG、认领方式、结果打包和集中分析。
- [机器配置模板](configs/site.example.env)：每台机器只修改这一层。
- [任务分配模板](configs/campaign.example.yaml)：协调者记录 owner、machine 和 status。

科学协议与机器配置必须分开：seed、预算、评测点和优化器条件不能由执行者修改；路径、GPU 编号、端口、输出目录和 W&B 项目可以按机器修改。

## 执行者快速开始

所有命令都从仓库根目录执行：

```bash
mkdir -p local
cp examples/mopd_gpas/configs/site.example.env local/mopd.env
# 按机器修改 GPU 编号和本地目录。
source local/mopd.env

bash examples/mopd_gpas/run_stage.sh fetch-assets
bash examples/mopd_gpas/run_stage.sh preflight

CONFIG_ID=cost_gpas_k2_taskwise  # 替换为分配给你的唯一配置
bash examples/mopd_gpas/run_stage.sh train "${CONFIG_ID}"
bash examples/mopd_gpas/run_stage.sh capability "${CONFIG_ID}"
bash examples/mopd_gpas/run_stage.sh package "${CONFIG_ID}"
```

训练中断时不要重新开始：

```bash
bash examples/mopd_gpas/run_stage.sh resume "${CONFIG_ID}"
```

## 八条主轨迹

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

每条主轨迹的成功条件是训练目录和 `capability_eval/response_64000/` 中都存在 `run_complete.json`，且训练 allocation 的最后一个 `attempted_responses_after` 恰好为 64,000。`package` 会检查这些条件，并生成不含 checkpoint 和 W&B cache 的集中分析包。

W&B 默认开启，项目默认为 `iclr2027-mopd-gpas-64k`。访问令牌只通过本机登录或环境变量提供，不写入仓库配置。
