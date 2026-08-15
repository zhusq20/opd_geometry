# OPD × Optimizer × 多任务持续学习指标

## 记录规则

- 粒度：`global`、真实 optimizer 分支、`layer`、`operator_type(q/k/v/o/gate/up/down)`；单参数矩阵只做低频分析。
- Muon 必须区分 `Muon matrix` 与 `Adam fallback`，不得用参数维度猜测 optimizer 分支。
- 每条记录保存：`schema_version`、run/seed/task、`rollout_id`、`num_updates`、`model_version`、累计 prompt/token、参数 dtype、学习率、实际 batch size、`update_successful`。
- 所有近似量加 `_sketch` 后缀；无后缀的 norm、dot、cosine 必须精确计算。
- 失败或跳过的 update 保留事件和原因，但不进入有效更新统计。
- 参数张量统一记为：
  - `theta_before`：更新前模型参数；
  - `g_raw`：backward 累积完成、裁剪前梯度；
  - `clip_scale=min(1, clip_threshold/||g_raw||)`；
  - `g_opt`：实际送入 optimizer 的裁剪后梯度；
  - `d_data`：momentum/preconditioner/NS 处理后的数据方向；
  - `d_wd`：weight-decay 方向；
  - `delta_data_fp32`：应用前以 FP32 表示的数据更新；
  - `delta_wd_fp32`：应用前以 FP32 表示的 weight-decay 更新；
  - `delta_intended_fp32=delta_data_fp32+delta_wd_fp32`：应用前的总更新；
  - `delta_model`：模型参数更新前后的实际差，另存 `model_dtype`；BF16 模型时即 `delta_bf16`；
  - `displacement=theta_after-theta_reference`。

## 每个有效 update 必须记录

- `theta_before/g_raw/g_opt/d_data/d_wd/delta_data_fp32/delta_wd_fp32/delta_intended_fp32/delta_model/displacement` 的精确 `L2`、RMS、Linf、exact-zero 比例。
- 精确 cosine：
  - `cos(g_raw,g_opt)`；
  - `cos(g_opt,d_data)`；
  - `cos(g_opt,delta_intended_fp32)`；
  - `cos(delta_intended_fp32,delta_model)`；
  - `cos(theta_before,delta_model)`；
  - `cos(delta_model,displacement)`。
- 精确 dot：
  - `<g_opt,d_data>`；
  - `<g_opt,delta_intended_fp32>`；
  - `<g_opt,delta_model>`。
- 比率：
  - `||g_raw||/||theta||`；
  - `||d_data||/||g_opt||`；
  - `||delta_wd_fp32||/||delta_data_fp32||`；
  - `||delta_intended_fp32||/||theta||`；
  - `||delta_model||/||theta||`；
  - `||displacement||/||theta_reference||`；
  - `-<g_opt,delta>/||g_opt||^2`，字段名使用 `gradient_directional_step`，不得称为 coordinate learning rate。
- 裁剪：`grad_norm_raw`、`clip_threshold`、`clip_scale`、`grad_clipped`、run 内 clip fraction。
- 每个真实 optimizer 分支的参数量、梯度能量、intended-update 能量和 realized-update 能量占比。

## FP32 → 模型 dtype 的实现指标

- `model_change_fraction = nnz(delta_model)/numel`。
- `intended_below_half_ulp_fraction = mean(|delta_intended_fp32| < 0.5*ULP(theta_before))`。
- `energy_survival = ||delta_model||^2/||delta_intended_fp32||^2`。
- `quantization_residual = ||delta_model-delta_intended_fp32||/||delta_intended_fp32||`。
- `cos(delta_intended_fp32,delta_model)`。
- 在 `|delta_intended_fp32|/ULP(theta_before)` 固定分桶内记录：实际非零率、能量保留率、符号翻转率。
- 记录被归零、被放大、被衰减的 intended-update 能量比例。
- 低频记录相邻 `delta_model!=0` 支持集 Jaccard、窗口内坐标更新频率、从未变化坐标比例。

## Optimizer 专属指标

### Muon

- 张量：`g_opt`、momentum `m`、实际 NS 输入 `n`、post-NS `q`、缩放后 `d_data`、`delta_intended_fp32`、`delta_model`。
- 精确 norm/cosine：`cos(g_opt,m)`、`cos(m,n)`、`cos(n,q)`、`cos(g_opt,q)`、`cos(q,delta_intended_fp32)`。
- momentum：`||beta*m_prev||/||(1-beta)*g_opt||`、`cos(m_t,m_prev)`。
- NS：`rows<=cols` 时记录 `||q*q^T-I||_F/sqrt(rows)`，否则记录 `||q^T*q-I||_F/sqrt(cols)`；另记奇异值 log-spread、regularized `s95/s5`、row/column-norm CV。
- 按 `q/k/v/o/gate/up/down` 报告每类矩阵的 median、IQR 和固定抽样矩阵轨迹。
- Q/K/V 分块前后分别统计；单独报告 Muon matrix 与 Adam fallback。

### Adam / AdamW

- 张量：`g_opt`、`m_hat`、`v_hat`、`d_adam=m_hat/(sqrt(v_hat)+eps)`、`d_wd`、总方向、`delta_intended_fp32`、`delta_model`。
- 精确 cosine：`cos(g_opt,m_hat)`、`cos(g_opt,d_adam)`、`cos(d_adam,total_direction)`、`cos(d_wd,d_adam)`。
- `||d_adam||/||g_opt||`、`||d_wd||/||d_adam||`。
- `eta/(sqrt(v_hat)+eps)` 的 gradient-energy-weighted `p1/p50/p99/CV`，权重固定为 `g_opt^2`。
- `sqrt(v_hat)` 的 `p1/p50/p99`、`p99/p1`，以及 `sqrt(v_hat)<=eps` 和 `<=10*eps` 的坐标比例。
- first/second-moment carry ratio、`cos(m_t,m_prev)`。
- `weight_decay=0` 时 WD 指标记为 `not_applicable`，主表名称写清 `Adam (AdamW implementation, wd=0)`。

### SGD

- 张量：`g_opt`、momentum/velocity `v`、Nesterov `d_data`、`d_wd`、总方向、`delta_intended_fp32`、`delta_model`。
- `cos(g_opt,v)`、`cos(g_opt,d_data)`、`||d_data||/||g_opt||`。
- momentum carry ratio、`cos(v_t,v_prev)`、`||d_wd||/||d_data||`。
- vanilla SGD 只记录 `g_raw/g_opt`、WD、总方向和两种 delta。

## OPD 与 loss 指标

- 当前实现统一命名为 `sampled_reverse_kl_logratio = log pi_student(a|h)-log pi_teacher(a|h)`，不得写成 full-vocabulary KL。
- sampled log-ratio 按 token 和 sequence 记录 mean/std/p10/p50/p90/p99、负值比例、L2/RMS、最大绝对值。
- 按 response 相对位置分桶记录 student log-prob、teacher log-prob、sampled log-ratio、advantage 和 entropy。
- 记录有效 response length、截断率、importance ratio、policy clip fraction、TIS/OIS clip fraction、entropy、总 advantage 分布。
- 周期性 top-k probe：student/teacher top-k overlap、共享概率质量、entropy gap；仅在 teacher API 返回 top-k 分布时启用。
- 周期性样本/微批梯度 coherence：`||sum_i g_i||/sum_i ||g_i||`，以及 `||sum_i g_i||^2/(n*sum_i ||g_i||^2)`。
- mixed loss 只对实际启用的分量记录：未乘系数梯度、乘系数梯度、与总梯度 cosine、两两 cosine、`<g_component,g_total>/||g_total||^2`。
- loss 分量及系数：OPD、policy、value、entropy、SFT；未启用分量记为 `not_applicable`。

## Task reward 与固定 probe

- task reward 与优化信号分字段保存：`task_reward_observed`、`reward_used_in_loss`、`reward_loss_coefficient`。
- 纯 OPD 必须满足 `reward_used_in_loss=false`、`reward_loss_coefficient=0`；task reward 可异步离线计算。
- 实际 rollout 保存 task、prompt/sample ID、response、label、reward、pass/fail、`num_updates`、`model_version`；按 task 汇总 mean/std/p10/p50/p90/pass-rate。
- 每个 checkpoint 固定运行：seen-train probe、matched held-out probe、所有旧 task memory probe。
- 每个 probe 记录 reward/pass@k、NLL、teacher sampled reverse-KL、相对初始和 phase-start checkpoint 的 logit KL；top-k/full-vocabulary KL 仅在接口提供相应分布时记录。
- 记录 seen-held-out gap、逐样本 pass→fail/fail→pass、best-so-far minus current、response format/error rate。

## 同 checkpoint 的多任务与持续学习指标

- 在同一个 `theta_t` 上计算所有 task probe gradient `g_task`；不得用不同训练步的 task centroid 代替。
- 每个 task：`||g_task||`；每对 task：精确 dot、cosine、norm ratio。
- 对实际训练更新记录：
  - `<g_task,delta_intended_fp32>`；
  - `<g_task,delta_model>`；
  - `-<g_task,delta>/||g_task||||delta||`；
  - 各 layer/operator 的正 loss-change 内积比例和内积能量比例。
- 定义等范数欧氏梯度基线：`u_euclidean=-||delta_data_fp32||*g_opt/||g_opt||`。
- 对每个旧 task 精确分解：
  - raw-gradient contribution：`<g_task,u_euclidean>`；
  - optimizer-geometry contribution：`<g_task,delta_data_fp32-u_euclidean>`；
  - weight-decay contribution：`<g_task,delta_wd_fp32>`；
  - precision contribution：`<g_task,delta_model-delta_intended_fp32>`；
  - predicted loss change：`<g_task,delta_model>`。
- 实际 probe loss change：`L_task(theta_after)-L_task(theta_before)`。
- curvature residual：实际 loss change 减 predicted loss change。
- 每个 task phase 保存 performance matrix `R[i,j]`，并计算 ACC、BWT、FWT、best-current forgetting。
- 对 reward 不同尺度的 task：先报告原始 task 单位；跨 task 汇总使用预先固定的 task scale 标准化变化，并保存该 scale，不得直接平均未标准化 reward。

## 矩阵与稀疏性低频指标

- 对 `delta_intended_fp32`、`delta_model`、`displacement` 记录：exact-zero、`|x|<=tau*RMS(x)`（`tau=1e-3,1e-2`）、top-0.1%/1%/5% L2-energy。
- 固定抽样矩阵记录 stable rank、99%-energy effective rank、spectral entropy、`s95/max(s5,eps*s95)`、row/column-norm CV，并保存 `eps`。
- 不使用未正则化的全矩阵 condition number 作为主指标。
- Hoyer sparsity、L1 和完整坐标分位数只放附录。
- 同时报告：参数加权 global/micro 统计，以及矩阵等权或 operator 等权 macro median/IQR。
- 稀疏阈值至少报告 exact-zero、RMS-relative、ULP-relative 三种口径及敏感性分析。

## 干预实验指标

- 在 checkpoint clone 上分别施加 `delta_intended_fp32` 与 `delta_model`，比较各 task 的 NLL、KL、reward/pass@k 变化。
- 比较 BF16、FP32 参数应用和 stochastic rounding 的 update realization 与 task change。
- 对 update support mask、top-energy mask、低秩重建分别报告保留的 update energy、probe loss change 和最终 task performance。
- shadow optimizer 使用同一 `theta/g_opt` 产生候选方向，比较方向 cosine、精度实现率和各 task predicted loss change。

## 汇总与统计

- 横轴同时报告 optimizer update、累计 prompt、累计有效 token；尾批使用真实 batch size。
- 每个 run 报告 final、固定网格 AUC、best、失败/跳过 update 数、clip fraction、吞吐和额外观测开销。
- 主比较使用 paired seeds；至少报告均值、SD、95% CI 和 paired optimizer difference。
- 时间序列主检验预测“下一次固定 probe 的 loss/reward change”，不得把同一 run 的所有 step 当成独立样本。
- 报告三层嵌套模型的增量预测能力：raw-gradient 指标；加 optimizer-geometry 指标；再加 precision 指标。
- leave-one-seed-out 和 leave-one-task-out 验证；多指标探索结果与预注册主指标分开报告。

## 可选指标

- Gradient noise scale：仅在 batch-size/microbatch 是实验变量时启用，并分解 task 间、prompt 间、同 prompt 采样内方差。
- Fisher-weighted distance：仅作参数保持附录指标，并固定 Fisher probe、reference checkpoint 和归一化方式。
- Canary/exposure：不进入本论文主实验。

参考：[OPD update geometry](https://arxiv.org/abs/2606.13657)、[SGD-RL](https://arxiv.org/abs/2602.07729)、[Rethinking OPD](https://arxiv.org/abs/2604.13016)、[GEM](https://arxiv.org/abs/1706.08840)、[PCGrad](https://arxiv.org/abs/2001.06782)、[NorMuon](https://arxiv.org/abs/2510.05491)、[AMO](https://arxiv.org/abs/2605.17806)、[Muon-OGD](https://arxiv.org/abs/2605.08949)。
