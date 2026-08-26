# Qwen3-1.7B 原版 → Code → Math → Knowledge → IF 序贯训练分析协议

本文档规定 `seq_01_math_from_code_seed42`、`seq_02_knowledge_from_math_seed42`、
`seq_03_if_from_knowledge_seed42` 的论文级事后分析口径。所有行为结论来自固定测试样本，所有参数结论来自
实际落盘的 BF16 model tensors；两者不混作同一种“干扰”。

## 1. 五个测量边界

| 边界 | 权重状态 | 新增序贯 update |
|---|---|---:|
| `pre_code` | Qwen3-1.7B 原版 release checkpoint | 0 |
| `code_origin` | Code 阶段 update-300 checkpoint | 300 |
| `after_math` | Math 阶段 update-300 | 600 |
| `after_knowledge` | Knowledge 阶段 update-300 | 900 |
| `after_if` | IF 阶段 update-300 | 1200 |

任务边界只继承模型权重；AdamW moments、scheduler、RNG 和 sampler 都重置。因此本文研究的是
weight-only continual adaptation，不把它描述成 optimizer state 连续的单次训练。

## 2. 固定测试矩阵

五个边界使用完全相同的六数据集配置、prompt identity、seed、chat template 和关闭 thinking 的推理设置：

| 域 | 数据集 | 每 prompt 样本数 | 角色 |
|---|---|---:|---|
| Code | LiveCodeBench v5 online-128 | 1 | 主指标 |
| Math | MATH500 | 1 | 主指标 |
| Math | AIME24 | 8 | 辅助高难数学指标，同时报告 pass@8 |
| Knowledge | GPQA Diamond | 4 | 主指标，同时报告 pass@4 |
| IF | IFBench strict | 1 | 主指标 |
| IF | IFEval strict-prompt | 1 | 辅助指令遵循指标 |

主 continual-learning 矩阵只使用每个域一个预注册指标：LiveCodeBench、MATH500、GPQA、IFBench。AIME24
和 IFEval 单列，避免一个域因拥有更多 benchmark 而被隐式加权。

所有 Code 输出都经当前安全 attestation 锁定的 SandboxFusion 判分。若 LiveCodeBench artifact 缺少
`sandbox_eval` 诊断，或出现 infrastructure error，主分析直接失败，不把基础设施故障计作模型错误。

## 3. 行为遗忘与迁移

令 `R[b,t]` 为边界 `b` 上任务 `t` 的主指标均值。任务获得边界为：

```text
Code      -> code_origin
Math      -> after_math
Knowledge -> after_knowledge
IF        -> after_if
```

对任务 `t`：

```text
signed_final_change(t) = R[final,t] - R[acquisition(t),t]
forgetting(t)          = max(0, R[acquisition(t),t] - R[final,t])
forgetting_rate(t)     = forgetting(t) / R[acquisition(t),t]
retention(t)           = R[final,t] / R[acquisition(t),t]
peak_forgetting(t)     = max_{b>=acquisition(t)} R[b,t] - R[final,t]
```

最终平均准确度 `ACC` 是四个主指标的算术平均。经典 backward transfer 为：

```text
BWT = mean_t∈{Code,Math,Knowledge}(R[after_if,t] - R[acquisition(t),t])
```

IF 是最后训练任务，没有后续边界，因而不能识别 post-IF forgetting。以 `pre_code` 的原版预训练模型为
参照，报告：

```text
FWT_pretrained_base = mean(
  R[code_origin,Math] - R[pre_code,Math],
  R[after_math,Knowledge] - R[pre_code,Knowledge],
  R[after_knowledge,IF] - R[pre_code,IF]
)
```

它明确命名为 pretrained-base-referenced FWT，不冒充以随机初始化模型为参照的 FWT。Code 本任务的
`R[code_origin,Code]-R[pre_code,Code]` 单列为 acquisition gain。

相邻边界对每个数据集计算 `after - before`。负值称为 behavioral interference loss，正值称为
facilitation gain。统计单位是 prompt；多采样数据集先在 prompt 内求均值，再做 20,000 次配对
prompt-cluster bootstrap。另保存逐 sample 的 pass→fail、fail→pass 和逐 prompt 的 improved/regressed
计数。

## 4. 全参数 checkpoint 几何

阶段局部更新固定为：

```text
Δ_code      = θ_code_origin     - θ_pre_code
Δ_math      = θ_after_math      - θ_code_origin
Δ_knowledge = θ_after_knowledge - θ_after_math
Δ_if        = θ_after_if        - θ_after_knowledge
```

脚本从 torch-distributed checkpoint 流式读取 model tensors，显式排除 `optimizer.*` 和 RNG state，并以
FP64 reduction 计算：

```text
||Δ_i||₂
||Δ_i||₂ / ||θ_pre_code||₂
<Δ_i, Δ_j>
cos(Δ_i, Δ_j)
<Δ_i, Δ_j> / ||Δ_i||₂²
```

累计位移 `θ_boundary - θ_pre_code` 单独输出。累计位移共享历史路径，通常会有较大正 cosine，不能替代
阶段局部更新的任务方向比较。

## 5. 更新稀疏度与支持复用

同时报告四种互补稀疏度：

1. exact-zero：落盘 BF16 delta 恰好为零；
2. visible sparsity：`|Δ| <= 1e-6, 1e-5, 1e-4`；
3. BF16-aware：`|Δ| <= 1e-3 max(|θ_before|, |θ_after|)`；
4. relative：每个逻辑 tensor block 内 `|Δ| < 1e-3 RMS(θ_before)`。

在支持集 `S_i={k:|Δ_i[k]|>1e-5}` 上计算：

```text
directional overlap = |S_i ∩ S_j| / |S_i|
independent baseline = |S_j| / d
overlap lift         = directional overlap / independent baseline
Jaccard              = |S_i ∩ S_j| / |S_i ∪ S_j|
union sparsity       = 1 - |S_i ∪ S_j| / d
sign conflict        = intersection 中 Δ_i[k]Δ_j[k] < 0 的比例
```

这一区分“全局方向近乎正交”和“稀疏支持是否复用”；二者可以同时成立。

## 6. 谱几何与子空间锁定

在第 0/7/14/21/27 层，对 attention output、QKV、MLP down、MLP gate/up 四类矩阵，分析 Code/Math/Knowledge/IF 每阶段
update-100/200/300 的 realized update matrix。使用固定 Gaussian sketch、rank 16、oversampling 16、
4 次 power iteration 的 deterministic randomized SVD，报告：

```text
stable_rank(ΔW) = ||ΔW||_F² / σ₁(ΔW)²
top-k energy    = sum_{i<=k} σ_i(ΔW)² / ||ΔW||_F², k∈{1,8,16}
spectral shift  = ||σ_1:16(W_t)-σ_1:16(W_start)||₂ / ||σ_1:16(W_start)||₂
locking(t)      = ||V_16(t)^T V_16(final)||_F² / 16
```

这是分层抽样的近似谱分析，不冒充全模型精确 SVD。流水线内含一个 exact-SVD spot check，量化近似误差。

## 7. 同 checkpoint 原始梯度干扰

原训练没有保存任务边界的 cross-task raw gradients，所以 checkpoint-delta cosine 不能被写成草稿理论式中的
局部梯度项。补充 probe 在五个边界对 Code、Math、Knowledge、IF 使用完全相同的冻结 probe 设计：每任务
16 prompt groups × 16 responses，共 256 sampled responses；LR=0、禁止 `optimizer.step`，在 clipping 与
AdamW 之前保存 optimizer-owned FP32 raw-gradient shards。

对同一 checkpoint 的任务梯度计算精确 FP64：

```text
||g_i||₂, <g_i,g_j>, cos(g_i,g_j), <g_i,g_j>/||g_i||₂²
```

负 dot 对应草稿一阶展开中的局部 conflict term。它只说明当前 checkpoint、当前 on-policy probe 分布上的
瞬时方向冲突，不能单独推出最终测试遗忘或因果关系。

若某个 probe 的所有组内 reward 方差均为零，GRPO raw gradient 可能严格为零；此时 cosine 的分母为零，
方向在数学上未定义。汇总文件与图中写为 `null/n/a`，而不是写成 0 并宣称正交。

## 8. GRPO 信号与优化动力学

每阶段锁定 300 条 rollout、train 和 optimizer-observer 记录，报告前/后 50 updates：

- on-policy reward；
- all-wrong、mixed/informative、all-correct group fraction；
- raw gradient norm、clipping fraction、entropy；
- response length、truncation；
- realized BF16 coordinate-change fraction；
- intended update 落在 half-ULP 以下的比例、被量化清零的能量比例；
- intended update 与 realized update 的 cosine。

其中 `informative_group_fraction = 1 - all_wrong - all_correct`，直接对应 GRPO 是否仍有组内相对优势信号。
Code 来源 run 的 manifest 保留为 `running` 且没有 `run_complete.json`；分析不篡改该状态，只在最终 checkpoint
指纹、训练 lineage 以及 rollout/train/geometry 各 300 条有效记录全部通过校验后纳入 Code 动力学。

## 9. 参考工作与差异

- [SFT Conflicts, RL Coexists](https://arxiv.org/abs/2608.03573)：采用其 checkpoint update norm/cosine、
  `1e-5` 支持、跨阶段性能矩阵和 same-checkpoint gradient dot/cosine；本文额外区分 acquisition forgetting、
  配对 prompt bootstrap 和 BF16 realization。
- [On the Geometry of On-Policy Distillation](https://arxiv.org/abs/2606.07082)：采用 BF16-aware unchanged、
  stable rank、spectral shift、principal-subspace locking；本文明确 randomized-SVD 的抽样层与误差锚点。
- [Dense Supervision, Sparse Updates](https://arxiv.org/abs/2606.13657)：采用多阈值稀疏度、relative threshold、
  directional support overlap、独立基线、overlap lift、union sparsity 与 top-coordinate/spectral energy 思路。

草稿中的一阶梯度干扰公式用于解释 raw-gradient probe，不被错误套用到累计 checkpoint 位移。最终行为结论
始终以固定测试矩阵为准。
