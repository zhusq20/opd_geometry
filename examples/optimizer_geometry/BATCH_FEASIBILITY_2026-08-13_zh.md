# OPD/PPO/GRPO batch 可行性实测（2026-08-13 UTC）

## 结论

这里把 `reference256` 的历史容量/耗时实测与新的 b8/b16 实测放在一起；reference 不再是
launcher 默认值。该配置
`rollout_batch_size=256`、`n_samples_per_prompt=4`、`global_batch_size=1024` 能在四张空闲的
96 GiB RTX PRO 6000 Blackwell Server Edition 上完成 GRPO/PPO/OPD 的训练路径，但一次长序列
更新需 1,539.7 秒，W&B 反馈延迟不可接受。实际 scripts 的完整 update 实测为：
`responsive16` 199.0 秒，`responsive8` 190.4 秒。b8 只再缩短 4.3% 的单步延迟，却把固定
prompt 预算的 update 数翻倍，因此旧 fixed-budget/tuning scripts 默认 `responsive16`
（16 prompts、64 responses/update），`responsive8` 只做低延迟诊断；
`reference256` 只用于复现和 response-cap pilot。容量结论只适用于同一 Qwen3-1.7B student、
并行配置、checkpoint 和健康且空闲的卡；launcher 的 GPU preflight 不应关闭。

当前 frozen `run_opd_*` 已改为过滤后数据集一轮的 `opd64x1`（64 prompts × 1），response cap
4096、max-running 72，并显式使用 Qwen3 `enable_thinking=false`；本文件没有把旧的
thinking-on b16/8192 smoke 耗时冒充为该新 profile 的实测值。

物理 GPU 0 在真实 CUDA allocation 时报告 uncorrectable ECC，已从示例设备列表排除。实测
student 使用物理 GPU 1/4/6/8，OPD 的 Qwen3-8B teacher 使用物理 GPU 7。

## 亲自执行的测试

| run | 覆盖路径 | 结果 |
|---|---|---|
| `smoke_grpo_adamw_b256n4_r8192_c44_seed9052` | 默认 8192 response cap；1024 responses；完整 rollout、backward、AdamW update | 成功；1,539.7 s/update；训练时 `nvidia-smi` 最坏约 72.9 GiB/卡 |
| `smoke_grpo_adamw_responsive16_r8192_c44_seed9060` | 默认 b16 profile；64 responses；LR/beta2 已缩放；完整 rollout、backward、AdamW update | 成功；181.1 s rollout、199.0 s/update；grad norm 0.197，未 clipping |
| `smoke_grpo_adamw_responsive8_r8192_c44_seed9062` | b8 诊断 profile；32 responses；LR/beta2 已缩放；完整 rollout、backward、AdamW update | 成功；175.6 s rollout、190.4 s/update；grad norm 0.273，未 clipping |
| `smoke_ppo_adamw_b256n4_r256_seed9043` | PPO actor + critic 的完整 forward/backward/update/checkpoint，response cap 256 | 成功；actor grad norm 0.693；actor/critic checkpoint 完整 |
| `smoke_opd_adamw_b256n4_r256_seed9044` | 外部 Qwen3-8B teacher token log-prob、reverse KL、actor update，response cap 256 | 成功；teacher 约 68.5 GiB；reverse KL 0.299；actor grad norm 4.91 |
| `smoke_grpo_adamw_b256n4_r256_seed9042_actual2` | 短序列公共 GRPO 路径 | 成功；1024 responses 和一个完整 update |

这是因子化测试：默认长序列的公共生成/actor 训练路径，以及 PPO critic 与 OPD teacher 的特有
路径都分别真实运行；没有声称已把 8192-token PPO/OPD 各再耗时数小时完整复制。AdamW 是本比较
中 optimizer-state 显存较保守的条件，因此结果足以支持 batch feasibility，但正式运行前仍需逐
cell 检查占用和 ECC。

长序列首次未限制并发的测试让约 498k KV tokens 达到 100%，产生 request retract/recompute；它
不是 OOM，但吞吐不可接受。因此同一硬件/模型/`mem_fraction=0.6` 下已设置
`SGLANG_MAX_RUNNING_REQUESTS=44`，用最多 450,560 worst-case tokens 留出余量。受控复跑没有
retraction。配置改变后必须重新校准，不能机械复用 44。曾把 responsive profile 的上限降到
每引擎平均请求数（b16 为 16、b8 为 8），但 router 分配并不严格均匀：同 seed b16 的 step
从 199.0 秒恶化到 308.7 秒，b8 也出现某引擎运行 8 条同时排队 6 条。因此实际 launcher 对三个
profile 都保留安全上限 44；小 batch 自然不会用满它，却不会被人为限流。

## 科学性警告

长序列成功 run 的 response length mean 为 7,161.96，median/p95/max 均为 8,192；
686/1024（66.99%）response 触顶。它证明“能运行”，却说明 8192 很可能截断模型的 reasoning。
在论文矩阵前应预注册 response-cap pilot（8192/12288/16384 或明确的 non-thinking 模板），选择触顶
率低于 10% 且算力可承受的最小上限；若仍用 8192，应在主表旁报告 truncation，并至少做一个更长
上限的敏感性分析。

一次 reference256 长序列 GRPO update 的 rollout 为 1,372.4 秒、actor update 为 161.8 秒，
总计 1,539.7 秒。默认并发下，`responsive16` 实测 rollout 181.1 秒、actor update 14.7 秒、
总计 199.0 秒（3.32 分钟）；`responsive8` 分别为 175.6、11.8 和 190.4 秒（3.17 分钟）。
b8 没有接近线性减半，是因为 median 仍为 8,192 tokens，整步要等待长尾 response。为保持原
51,200-prompt 预算，二者分别运行 3,200/6,400 updates；按单个 smoke step 机械外推约 7.37/14.10
天，b8 反而约为 b16 的 1.91 倍。这个外推不含启动、eval/save、OPD teacher，也不代表策略演化后
response length 不变；它只用于排程预警。小 batch 改善的是 W&B 反馈延迟，不会自动减少总生成
工作量。eval/save/geometry 已按累计 prompt 数重算，避免昂贵操作频率放大。

超参数不是原值照搬：`responsive16` 的 AdamW/Muon LR 为 `2.5e-7`，GRPO-SGD 为 `2.5e-2`，
OPD/PPO-SGD 为 `2.5e-3`；Adam beta2 为 `0.98^(16/256)=0.9987381276`，保持二阶矩的样本
半衰期。`responsive8` 进一步使用 `1.8e-7`、`1.8e-2/1.8e-3` 和 beta2 `0.9993688646`。
这些是基于文献缩放的 tuning 起点，不是免调参的最优值。

## 产物位置与复核

实测 run 位于 `outputs/batch_feasibility/`。每个成功 run 具有 source snapshot、完整命令和硬件/
ECC/package provenance、durable train/rollout JSONL、geometry、completion marker；保存开启的 PPO
run 还含 actor/critic checkpoint。它们是容量 smoke test，刻意关闭 W&B，且没有冒充论文 final
eval。可用以下命令复核：

```bash
python examples/optimizer_geometry/validate_run_artifacts.py \
  outputs/batch_feasibility/smoke_grpo_adamw_b256n4_r8192_c44_seed9052
python examples/optimizer_geometry/validate_run_artifacts.py \
  outputs/batch_feasibility/smoke_ppo_adamw_b256n4_r256_seed9043
python examples/optimizer_geometry/validate_run_artifacts.py \
  outputs/batch_feasibility/smoke_opd_adamw_b256n4_r256_seed9044
python examples/optimizer_geometry/validate_run_artifacts.py \
  outputs/batch_feasibility/smoke_grpo_adamw_responsive16_r8192_c44_seed9060 \
  --expected-updates 1
python examples/optimizer_geometry/validate_run_artifacts.py \
  outputs/batch_feasibility/smoke_grpo_adamw_responsive8_r8192_c44_seed9062 \
  --expected-updates 1
```

正式论文 run 默认开启 W&B、checkpoint、pre/periodic/final eval 和逐样本 eval artifact，并应在纳入
分析前通过同一 validator 的 `--require-eval` 检查。
