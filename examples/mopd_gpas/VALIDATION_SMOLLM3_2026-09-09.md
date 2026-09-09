# 2026-09-09 SmolLM3 MixSFT pipeline 验证

运行说明见 [协作者 README](README_SMOLLM3_zh.md)。本记录验证真实训练链路，不报告 500-step 收敛或最终能力结果。

## 固定资产与数据

Student 是 `BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT` revision `c9e7bad031667828656ead188d7e8ea162c048a4`，不是 Base。权重 SHA256 为 `b5d4c1d8ae80f5b9086ef460ff1b63746e729beea7571b769f438ebd1d29f1c3`，与 Hub 发布的 LFS 哈希一致；文件中的 embedding 与 tied LM head 逐元素相等。

三个常驻 RL teacher 复用本机已有文件；已逐分片计算 SHA256，并与这次重新下载的固定 revision 比较，全部一致。见 [teacher 哈希记录](../../local/smollm3_handoff_20260909/teacher_weight_hashes.json)。模型/数据 revision 完整列于协作者 README。

从公开 `rl_prompt_mix/train.parquet` 准备的训练候选数为 math 17,789、code 23,539、IF 45,219；实际加载时按 2048-token prompt 上限过滤 2 条 IF，IF 可用数为 45,217；各域另有 64 条 heldout 和 64 条 diagnostic。保留代码测试和 IF 约束。实际生成配置见 [protocol.json](../../local/mopd_smollm3_mixsft_generated/protocol.json)。

## 环境

| 项目 | 本次实际版本 / 设置 |
|---|---|
| GPU | NVIDIA RTX A6000，48GB，TP=2 learner |
| PyTorch | `2.11.0+cu129` |
| Transformers | `5.12.1` |
| SGLang | `0.5.15.post1` |
| Ray | `2.56.1` |
| Transformer Engine | `2.16.1` |
| Megatron | `1dcf0dafa884ad52ffb243625717a3471643e087` |
| rollout | FlashInfer + PyTorch sampler，保留逐请求 seed |
| optimizer | FP32 梯度/Adam 状态，LR `2.5e-7` |
| 可选组件 | 未启用任务 reward、能力评测、W&B、参数几何快照 |

SmolLM3 使用 SGLang 的 Transformers backend。A6000 配置关闭 batch-invariant inference：该模式的 FP32 BMM 在 SmolLM3 RoPE 上需要 131,072 bytes shared memory，超过本卡 101,376 bytes 的上限。该修正已写入可移植 site 配置，不需要修改安装目录中的 SGLang 源码。

本机与其他任务共享 GPU，因此 PG 使用 learner 0/1、rollout 2；交集验证使用 learner 4/5、rollout 6；三个 teacher 共用 GPU 3。teacher 的 `max-total-tokens=8192`、`max-running-requests=4` 适用于本次短序列验证。正式长序列配置在 README 中使用三个独立 teacher GPU 和更大的 KV 容量。

## 自动验证

相关回归 **177 passed**，见 [regression_final.log](../../local/smollm3_handoff_20260909/regression_final.log)：

- Top64 交集按 token ID 和前缀位置对齐，包含 EOS 与缺失/错位检测。
- 交集 loss 保留原 student Top64 权重归一化，检查梯度和空交集。
- 禁用训练 verifiers 时不访问 sandbox，保留 teacher payload，日志不产生虚假准确率。
- 默认使用固定 MixSFT 和独立路径，拒绝覆盖 Base 资产或用 Base 启动专用入口。
- 关闭 batch-invariant kernel 后仍传递逐请求固定采样 seed。
- 领域平均方式、optimizer 累积、训练指标与 checkpoint provenance 回归。

两种入口通过真实 CLI 参数解析；shell 语法、修改的 Python 模块编译和 diff 空白检查通过。没有构建新 Docker 镜像，也未执行本次独立能力评测或完整 500-step 训练。

## 真实 GPU 运行

每条流程使用原始 MixSFT、三域真实 prompt 和三个 RL teacher，每次 rollout 12 个 prompt（每域 4 个），response 上限 128，执行 2 次 Adam 更新。短响应有较多截断，不能用于判断模型能力。

**两条流程均完整结束，进程退出码为 0，provenance 状态为 `complete`。**

| 验证项 | PG | teacher/student Top64 交集 |
|---|---|---|
| run ID | `smoke-m-pg-s42-ampere2` | `smoke-m-intersection64-dr-s42` |
| 完成时间（UTC） | 03:32:28 | 03:41:19 |
| optimizer 更新 / response 数 | 2 / 24 | 2 / 24 |
| 两次 aggregate grad norm（clip 前） | 19.6271、75.6453 | 14.4870、7.9603 |
| teacher 打分失败（math/code/IF） | 0 / 0 / 0 | 0 / 0 / 0 |
| overflow | 0 | 0 |
| 两轮 rollout model version | 0、1 | 0、1 |
| learner 峰值 HBM | 38.371 GiB | 38.371 GiB |
| optimizer checkpoint | 完整，保留 optimizer 状态 | 完整，保留 optimizer 状态 |
| HF 导出 | 13 个分片可读取、索引一致 | 13 个分片可读取、索引一致 |

两条作业的 6 个 backward slice loss 均有限。交集流程每个 slice 的平均交集大小为 51.127–59.127，观测到的空交集比例均为 0。两轮 `task_reward_observed=0`；本次没有运行 sandbox/verifier，也不把通用 reward 占位值当作准确率。

从导出的 HF 分片抽查 `model.layers.0.self_attn.q_proj.weight`（2048×2048，BF16）：PG 有 **9,272** 个元素与初始 MixSFT 不同，交集有 **9,137** 个元素不同；张量数值均有限，最大绝对变化均为 `9.5367431640625e-7`。这证明导出包含实际训练更新。检查点索引均记录 `optimizer_step=2`、`rollout_id=1`，分布式 optimizer 的 `.metadata` 与 4 个数据分片完整。没有实际执行断点恢复，不能将上述文件检查等同于恢复验证。

可复查的本地证据（这些大型运行产物不会随 Git 自动传给协作者）：

- PG：[完成标记](../../outputs/mopd_smollm3_mixsft/smoke-m-pg-s42-ampere2/run_complete.json)、[checkpoint 索引](../../outputs/mopd_smollm3_mixsft/smoke-m-pg-s42-ampere2/checkpoints/mopd_checkpoint_index.json)、[运行日志](../../local/smollm3_handoff_20260909/pg_smoke_ampere2.log)。
- 交集：[完成标记](../../outputs/mopd_smollm3_mixsft/smoke-m-intersection64-dr-s42/run_complete.json)、[checkpoint 索引](../../outputs/mopd_smollm3_mixsft/smoke-m-intersection64-dr-s42/checkpoints/mopd_checkpoint_index.json)、[运行日志](../../local/smollm3_handoff_20260909/intersection_smoke.log)。
- [结构化验证结果](../../local/smollm3_handoff_20260909/validation_summary.json) 与 [导出检查脚本](../../local/smollm3_handoff_20260909/validate_exports.py)。

每条 optimizer checkpoint 约 40.10 GiB，HF 导出约 5.74 GiB。本机共享磁盘保存 optimizer 分别耗时约 9 分钟和 10 分钟；日志暂停期间是磁盘 flush，应等待完成标记。上述成功作业已退出，本次启动的三个 teacher 服务也已停止。
