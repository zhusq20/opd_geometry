# 多任务 LLM 后训练中的梯度干扰：机理实验与 Path-OPD 方法计划

> 更新日期：2026-08-18
>
> 论文定位：多任务 LLM post-training 的机理分析 + 由机理直接导出的方法
>
> 研究对象：PPO、GRPO、final-teacher OPD、Path-OPD
>
> 主模型：Qwen3-1.7B；规模复现：Qwen3-4B；optimizer 固定为 AdamW

## 0. 论文最终主线

论文只讲一条完整故事：PPO、GRPO 和 OPD 都在 student 自己生成的轨迹上训练，但三者的学习信号
随训练进展以不同方式变化，因此产生不同的 task-gradient norm、跨任务方向关系和多任务训练动态。

机制部分直接报告三类现象：

1. PPO 的有效信号随失败样本减少而减弱；
2. GRPO 在 mixed-reward group 内仍保留标准化 advantage，但这样的 group 会逐渐减少；
3. OPD 的信号由 teacher–student log-ratio 决定，student 接近 teacher 时梯度持续收缩。

方法部分据此设计 **Path-OPD**：不让初始 student 从第一步就追逐最终 RL teacher，而是依次使用
同一条 domain-RL 路径上的 25%、50%、75% 和 100% checkpoints。方法不改 OPD loss、不做
gradient surgery，也不引入动态阈值；唯一改动是固定的 teacher-checkpoint curriculum。

全文不设置 H0/H1、证伪门、提前停止规则或复杂统计检验。所有预定实验均仅使用 seed 42 运行，
最终表格报告单次运行的原始结果和完整训练曲线。

## 1. 模型谱系与 teacher 构造

### 1.1 同源、分任务 RL teachers

草稿中的 OPD teacher 写作 \(\pi_i^T\)，因此主实验采用已有多教师 OPD 工作中的标准设置：从同一个
初始化 \(\theta_0\) 出发，分别训练 Math、Code 和 Science 的 domain-specialized GRPO teacher：

~~~text
                          ┌─ Math GRPO ─── θMath,25 ─ θMath,50 ─ θMath,75 ─ θMath,100
θ0 ──────────────────────┼─ Code GRPO ─── θCode,25 ─ θCode,50 ─ θCode,75 ─ θCode,100
                          └─ Science GRPO ─ θSci,25  ─ θSci,50  ─ θSci,75  ─ θSci,100
~~~

这些 single-task GRPO runs 同时承担三项作用：

- 产生 final-teacher OPD 和 Path-OPD 所需的 frozen teachers；
- 提供每个 domain 的 specialist 参考结果；
- 提供标准的 single-task checkpoint update norm/cosine 分析。

仅使用 seed 42 产生一组 teachers，所有 OPD students 使用这组 seed 42 teacher 路径。
teacher 与 student 使用相同 architecture、tokenizer、chat template 和初始化来源；teacher scoring
只接收 student 已经生成的 exact token IDs。

### 1.2 四条主要多任务训练路径

在相同的 \(\theta_0\)、数据顺序和 task mixture 下运行：

~~~text
θ0 ── balanced multi-task PPO ─────────────────────────────────────────► θPPO
θ0 ── balanced multi-task GRPO ────────────────────────────────────────► θGRPO
θ0 ── final-teacher OPD：每个 task 始终路由到 θtask,100 ──────────────► θFinal-OPD
θ0 ── Path-OPD：每个 task 依次路由到 θtask,25/50/75/100 ─────────────► θPath-OPD
~~~

两个 OPD students 都从新的 \(\theta_0\) 开始；它们不从任一 RL checkpoint warm start。纯 OPD 中
verifier reward 只用于记录生成质量，不进入训练 loss。

## 2. 固定数据、模型和训练设置

### 2.1 Tasks 与正式 benchmarks

训练使用当前已经准备好的 M2RL Math、Code 和 Science 数据。完成 prompt 去重、benchmark overlap
过滤和长度过滤后，每个 task 固定取 18,000 个训练 prompts；三个 task 各留出 128 个不进入训练、
也不属于正式 benchmark 的 prompts 作为 gradient probe set。训练子集和 probe set 只冻结一次，所有
模型和方法共用。所有训练统一使用 seed 42，并用它控制训练 prompt 顺序、rollout sampling 和
dropout。

三类 RLVR 训练 reward 都固定为二值 \(r\in\{0,1\}\)：Math/Science 使用 exact-answer correctness，
Code 只有通过全部选定 unit tests 才记为 1，不使用 pass-rate partial reward。这与草稿中 PPO/GRPO 的
binary-reward 推导一致。

正式结果只用常见 benchmark 和原始任务指标：

| Domain | 正文主 benchmark | 补充 benchmark | 报告指标 |
| --- | --- | --- | --- |
| Math | MATH-500 | AIME'24 | MATH-500 pass@1；AIME avg@8 |
| Code | LiveCodeBench release_v5 | release_v6 | pass@1；完整 pass@5/10 放 appendix |
| Science | GPQA-Diamond | — | accuracy / avg@4 |

三项正文主指标可额外给一个等权 macro average，但所有结论必须同时展示逐 domain 原始分数；不再使用
worst-task improvement、自定义归一化分数或把 AIME 和 MATH-500 当成两个独立 domain 重复加权。

### 2.2 模型和随机种子

| 用途 | 模型 | seed |
| --- | --- | --- |
| 完整机制矩阵、方法主结果和消融 | Qwen3-1.7B | 42 |
| 方法规模复现 | Qwen3-4B | 42 |

所有实验都从完全相同的 \(\theta_0\) 开始。seed 42 控制训练 prompt 顺序、rollout sampling 和
dropout；正式 evaluation 也固定使用 decoding seed 42，使不同方法逐题可配对。

### 2.3 训练 recipe

主实验冻结以下设置：

| 项目 | 固定设置 |
| --- | --- |
| optimizer | AdamW；weight decay 0；gradient clipping 1.0 |
| task schedule | Math → Code → Science 的 task-homogeneous round robin |
| task budget | 每个 task 18,000 个 prompts，一轮、不丢尾批 |
| GRPO | group size 16；actor LR \(1\times10^{-6}\) |
| PPO | 每 prompt 4 responses；actor LR \(2.5\times10^{-7}\)；critic 使用固定 AdamW recipe |
| OPD / Path-OPD | 每 prompt 1 response；sampled-token reverse-KL；coefficient 1.0；LR \(1\times10^{-6}\) |
| rollout sampling | temperature 1.0，top-p 1.0，显式关闭 Qwen3 thinking mode |
| training prompt / response cap | 2,048 / 8,192，所有方法一致 |
| evaluation cap | Math 32,768；Code / Science 16,384 |
| checkpoints | 0%、25%、50%、75%、100% consumed-prompt progress |
| online evaluation | 每 10% progress，并强制评测最终 checkpoint |

不同 objective 使用文献和当前 launcher 中常见的 method-native rollout multiplicity，因此生成 response
数和 teacher scoring 成本不同。正文横轴统一使用 **每个 task 已消费的 prompts**，同时在 appendix
报告 student responses、有效 response tokens、optimizer updates、wall-clock 和 GPU hours。论文不把
prompt-matched 写成 compute-matched，也不根据某一种方法的最终结果临时延长预算。

Final-OPD、Path-OPD 及其消融之间则完全对齐 prompt、response、token、update 和 teacher-scoring
budget，构成方法效果的严格对照。

### 2.4 Evaluation decoding

- MATH-500 使用固定 greedy pass@1；
- AIME'24 每题 8 次采样并报告 avg@8；
- LiveCodeBench final evaluation 每题生成 10 次，正文报告 unbiased pass@1；
- GPQA-Diamond 每题 4 次独立生成并报告平均 accuracy；
- 同一 benchmark 的 prompt template、sampling 参数、verifier、sandbox 和 response cap 对所有模型一致；
- 保存逐题 response、判分结果和错误状态，不只保存聚合分数。

## 3. 固定运行矩阵

### 3.1 Qwen3-1.7B 主矩阵

| 类别 | ID | 训练内容 | seed | 用途 |
| --- | --- | --- | --- | --- |
| teacher | ST-GRPO-{Math,Code,Science} | 三条 domain-specialized GRPO 路径 | 42 | teachers + specialist reference |
| mechanism | MT-PPO | balanced multi-task PPO | 42 | PPO 机制与 direct-RL baseline |
| mechanism | MT-GRPO | balanced multi-task GRPO | 42 | GRPO 机制与 Mix-RL baseline |
| baseline | MT-FinalOPD | task-routed final-teacher OPD | 42 | vanilla multi-teacher OPD |
| method | MT-PathOPD | task-routed 25→50→75→100% teachers | 42 | 本文方法 |
| baseline | MT-OfflineKD | 在 frozen final teachers 生成的轨迹上做 SFT | 42 | 常见 off-policy distillation baseline |
| baseline | TIES-Merge | 合并三个 final domain teachers | 42 | 常见 parameter-merge baseline，无额外训练 |

这里的 “specialist reference” 不是一个可部署的统一模型：主表中每个 domain 的 specialist 数值来自该
domain 对应的 final teacher，必须用 `Specialist oracle (three models)` 明确标注。

### 3.2 Path-OPD 消融

除主矩阵中已有的 final-only 和 four-stage Path-OPD 外，再运行：

| ID | teacher schedule | seed | 作用 |
| --- | --- | --- | --- |
| MT-PathOPD-2stage | 训练前半使用 50% teacher，后半使用 100% teacher | 42 | checkpoint 数量消融 |
| MT-FinalOPD-warmup | 始终使用 100% teacher，OPD coefficient 在前 25% 线性升到 1.0 | 42 | 常规 loss-strength warmup 对照 |

不运行反向 teacher 顺序、随机 teacher 顺序、按阈值动态切换或按结果挑选 stage boundary。主方法固定
使用四个等长阶段。

### 3.3 Qwen3-4B 规模复现

在 Qwen3-4B 上用相同数据、seed 42 和 evaluation protocol 运行：

- 三个 ST-GRPO domain teacher paths；
- MT-GRPO；
- MT-FinalOPD；
- MT-PathOPD。

规模复现不重复全部消融，也不重新选择 Path-OPD schedule。

### 3.4 OPD teacher / loss 标准对照

草稿 appendix 同时讨论 sampled-token 与 Top-K reverse-KL，正文又涉及 teacher selection。为覆盖这两点，
在 Qwen3-1.7B 的 MT-FinalOPD pipeline 上固定 seed 42，运行一个小型 2×2 对照：

| teacher | OPD loss |
| --- | --- |
| 三个 domain 都使用 same-origin final teachers | sampled-token reverse-KL；renormalized Top-64 reverse-KL |
| 只把 Math teacher 换成 tokenizer-compatible Qwen3-8B，Code/Science teachers 不变 | sampled-token reverse-KL；renormalized Top-64 reverse-KL |

Top-64 loss 在对应 cell 的三个 domains 中统一启用，四个 cells 使用同一多任务数据和 budget。
same-origin + sampled-token cell 可直接复用 MT-FinalOPD-42，不重复跑。表格报告全部 domain 分数，
机制图重点展示 Math 的 MATH-500/AIME、teacher–student log-ratio、entropy、gradient norm 和训练稳定性；
不扩展为 teacher-size sweep 或 Top-K sweep。

### 3.5 Response-cap 标准对照

当前本地 8,192-token 长序列试跑出现过较高 truncation，因此在 Qwen3-1.7B、seed 42 上把 MT-PPO、
MT-GRPO、MT-FinalOPD 和 MT-PathOPD 各固定复现一次，唯一改动是把 training response cap 提高到
16,384。该组实验直接报告 benchmark、mean response length 和 truncation rate，放入 appendix；不根据
这组结果回头选择或删除主配置，也不扩展为多档 response-cap sweep。

### 3.6 训练数量

不计 evaluation/probe 和无需训练的 TIES-Merge：

- Qwen3-1.7B 主矩阵与消融：10 条 full training runs；
- Qwen3-1.7B teacher/loss 对照：新增 3 条 runs；
- Qwen3-1.7B response-cap 对照：新增 4 条 runs；
- Qwen3-4B 规模复现：6 条 runs；
- 总计：23 条训练 runs。

## 4. 机理实验

### 4.1 Same-checkpoint raw task-gradient probe

在 MT-PPO、MT-GRPO、MT-FinalOPD 和 MT-PathOPD 的 0/25/50/75/100% checkpoints 上运行固定
probe。每个 task 使用同一组 128 prompts，并用与对应训练方法一致的 sampling 和 loss。probe 只做
forward/backward，不执行 optimizer step。

为与论文第 1 节的 ascent-gradient 记号一致，定义

\[
g_i(\theta)=-\nabla_\theta L_i(\theta).
\]

这里的 \(g_i\) 是 loss reduction 完成后、gradient clipping 之前、进入 AdamW 之前的 raw objective
gradient。PPO、GRPO 和 OPD 各自沿用正式训练时的固定 loss reduction；同一种方法的三个 tasks 使用
完全相同的 reduction。跨方法不直接比较 raw norm 的绝对倍数，主要比较每种方法内部随 progress 的变化。
每个 checkpoint 在**同一个模型参数状态**上分别计算 Math、Code 和 Science gradients，不使用不同训练
step 的 task centroid 代替。

每个 probe 保存并报告：

\[
\|g_i\|_2,
\qquad
\cos(g_i,g_j)
=
\frac{g_i^\top g_j}{\|g_i\|_2\|g_j\|_2},
\qquad
g_i^\top g_j.
\]

正文使用 task gradient norm 和 pairwise cosine；与论文公式直接对应的 raw dot products 放在 appendix
完整表格。所有量使用 exact distributed reduction，不使用 CountSketch、随机投影或跨 checkpoint centroid。

### 4.2 PPO 学习信号

在 PPO 的训练日志和固定 probes 上按 task 报告：

- verifier pass rate 与 failure rate；
- correct responses 和 incorrect responses 各自的 mean absolute advantage；
- actor gradient norm；
- Math–Code、Math–Science、Code–Science gradient cosine；
- appendix 中给出 importance ratio、clip fraction、old-policy approximate KL 和 critic loss。

正文做一张随训练 progress 变化的组合图：上半部分是 pass/failure rate 与两类 advantage magnitude，
下半部分是 gradient norm 和三个 task-pair cosines。

### 4.3 GRPO 学习信号

对每个 task 的 rollout groups 直接统计：

- all-correct group fraction；
- all-wrong group fraction；
- mixed-reward / informative group fraction；
- mixed groups 内 group-normalized advantage RMS；
- actor gradient norm 和三个 task-pair gradient cosines。

这些都是 GRPO/DAPO 实验中常见的 reward-group 统计。正文把 informative-group fraction 与 conditional
advantage RMS 放在同一图中，直接展示“有效 group 数量变化”和“有效 group 内信号尺度”这两个不同量。

### 4.4 OPD 学习信号

当前 sampled-token OPD 对 student 采样 token 记录

\[
\ell_t=\log\pi_\theta(a_t\mid h_t)-\log\pi_T(a_t\mid h_t).
\]

按 task 报告：

- token-level sampled log-ratio mean（student 采样下 reverse-KL 的 Monte Carlo estimate）；
- sampled log-ratio RMS；
- policy entropy；
- actor gradient norm；
- 三个 task-pair gradient cosines。

日志和图注始终写 `sampled log-ratio` 或 `sampled reverse-KL estimate`，不把单 token log-ratio 误写成
full-vocabulary KL。Final-OPD 与 Path-OPD 用完全相同的统计口径；Path-OPD 图中用竖线标出三次固定
teacher switch。

### 4.5 常规训练动态

四种主要训练方法统一记录：

- per-task rollout reward / verifier pass rate；
- formal online evaluation score；
- policy 或 OPD loss；
- actor gradient norm；
- token entropy；
- mean response length 与 truncation rate；
- invalid response、reward exception 和 sandbox timeout 计数。

这些曲线同时用于解释最终能力和排查训练异常。除上述量外，不新增 Hessian、loss landscape、Cauchy
utilization、path integral、shadow optimizer 或 clone-one-step 指标。

### 4.6 标准 checkpoint-update 几何

使用三个 ST-GRPO runs 的 checkpoints 定义

\[
\Delta\theta_i(t)=\theta_i^{\mathrm{ST\text{-}GRPO}}(t)-\theta_0.
\]

appendix 报告 25/50/75/100% progress 的 \(\|\Delta\theta_i\|_2\) 和 task-pair cosine heatmap；另给
一张 layer-wise relative update-norm heatmap。这里只采用现有多任务 RL 论文常见的 update norm/cosine，
不再计算自定义 parameter-path distance、spectral rank、support-mask overlap 或 task arithmetic closure。

## 5. 本文方法：Path-OPD

### 5.1 固定 teacher schedule

对来自 task \(i\) 的 prompt，Path-OPD 使用

\[
\theta_i^T(u)=
\begin{cases}
\theta_{i,25}^{\mathrm{RL}}, & 0\le u<0.25,\\
\theta_{i,50}^{\mathrm{RL}}, & 0.25\le u<0.50,\\
\theta_{i,75}^{\mathrm{RL}}, & 0.50\le u<0.75,\\
\theta_{i,100}^{\mathrm{RL}}, & 0.75\le u\le 1,
\end{cases}
\]

其中 \(u\) 是 Path-OPD 自己已经消费的 per-task prompt budget 比例。每个 checkpoint 在 teacher 阶段
开始前已经训练完并被冻结。18,000-prompt budget 的三个边界固定为每 task 4,500、9,000 和 13,500
prompts；若边界落在 global batch 内就拆分该 batch，不把边界四舍五入。round-robin 保证三个 task
同步进入下一阶段。

### 5.2 与 Final-OPD 的唯一差别

Path-OPD 与 Final-OPD 保持下列内容完全一致：

- student 初始化和 seed 42；
- per-task prompts、顺序和 task routing；
- rollout sampling、response cap 和 total budget；
- sampled-token reverse-KL、coefficient 和 clipping；
- AdamW、learning rate 和 scheduler；
- checkpoint、probe 和 evaluation cadence。

两者唯一差别是当前 teacher checkpoint。Path-OPD 不增加 domain-teacher 训练成本，因为 25/50/75%
checkpoints 已经由 specialist training 正常保存；正文另外报告 teacher 服务切换和缓存带来的 wall-clock
开销。

### 5.3 方法结果直接报告什么

Final-OPD、2-stage Path-OPD、4-stage Path-OPD 和 Final-OPD coefficient warmup 直接比较：

- MATH-500、AIME'24、LiveCodeBench、GPQA-Diamond；
- 三个正文主指标的 macro average；
- benchmark learning curves；
- sampled reverse-KL estimate、entropy 和 gradient norm curves；
- raw task-gradient norm/cosine at five anchors；
- wall-clock、teacher GPU hours 和有效 tokens/s。

方法的主要证据由最终 benchmark、收敛曲线和标准 OPD 训练动态构成；gradient 分析用于解释，不作为
决定某条 run 是否进入论文的筛选条件。

## 6. 正文表格与图

### Table 1：Qwen3-1.7B 多任务主结果

行：

- Base student；
- Specialist oracle (three models)；
- MT-PPO；
- MT-GRPO；
- TIES-Merge；
- MT-OfflineKD；
- MT-FinalOPD；
- MT-PathOPD。

列：MATH-500、AIME'24、LiveCodeBench v5、GPQA-Diamond、three-domain macro average。训练方法
报告 seed 42 的单次运行原始结果；specialist oracle 每列使用对应 seed 42 task specialist 的结果。
Table 1 不报告跨 seed mean、SD 或误差条。
student/teacher GPU hours 单独放 appendix 的 compute table，避免主结果表过宽。

### Table 2：Path-OPD 消融与规模复现

上半表是 Qwen3-1.7B 的 Final-OPD、Final-OPD-warmup、2-stage Path-OPD、4-stage Path-OPD；下半表
是 Qwen3-4B 的 MT-GRPO、Final-OPD 和 Path-OPD。仍然报告逐 benchmark 原始分数，不只展示平均值。

### Figure 1：训练谱系和 Path-OPD

画出 shared initialization、三条 domain-RL teacher paths、final-teacher routing 和 checkpoint-path
routing。图中明确两个 OPD students 都 reset 到 \(\theta_0\)。

### Figure 2：PPO / GRPO / OPD 的机制曲线

三列分别对应 PPO、GRPO、OPD：

- PPO：failure rate 与 correct/incorrect advantage magnitude；
- GRPO：informative-group fraction 与 conditional advantage RMS；
- OPD：sampled log-ratio RMS 与 reverse-KL estimate；
- 每列下方统一给 task gradient norm。

### Figure 3：跨任务 raw-gradient geometry

上排画三个 task gradient norms 随 progress 的变化；下排画三个 task-pair cosine。分别展示 MT-PPO、
MT-GRPO、MT-FinalOPD 和 MT-PathOPD 的 seed 42 单次运行曲线，不画跨 seed 误差带。起点、中点和终点的完整
norm/cosine heatmaps 放 appendix。

### Figure 4：最终能力与 Path-OPD 训练动态

前三个 panels 分别画 Math、Code、Science formal evaluation score；后两个 panels 画 Final-OPD 与
Path-OPD 的 sampled reverse-KL estimate 和 entropy。横轴使用 per-task consumed prompts，竖线标出
Path-OPD teacher switches。

## 7. Appendix 固定内容

Appendix 直接包含：

1. 所有模型和算法的完整 config 表；
2. 每条 run 的逐 checkpoint benchmark 数值；
3. reward、entropy、length、truncation、grad norm、PPO KL/clip、OPD log-ratio dashboard；
4. exact gradient dot-product 表与完整 cosine heatmaps；
5. ST-GRPO checkpoint-update norm/cosine 和 layer-wise update norm；
6. LiveCodeBench v6 与 pass@5/10；
7. same-origin/external teacher × sampled/Top-64 对照；
8. 四种主要方法的 8,192 / 16,384 response-cap 对照；
9. 每条 run 的 prompts、responses、有效 tokens、wall-clock 和 GPU hours；
10. seed 42 的完整单次运行曲线，不隐藏失败或异常 checkpoint。

## 8. 执行顺序

1. 冻结 \(\theta_0\)、18,000 prompts/task、128 prompts/task probe set 和全部正式 eval configs。
2. 使用 seed 42 运行 Qwen3-1.7B 的 3 条 ST-GRPO task paths，保存 25/50/75/100% teachers。
3. 对这组 teachers 完成 tokenizer/token-ID 检查并启动三个 task-routed scoring endpoints。
4. 运行 Qwen3-1.7B 的 MT-PPO、MT-GRPO、MT-FinalOPD、MT-PathOPD 和 MT-OfflineKD。
5. 从同一组 final teachers 生成 TIES-Merge baselines。
6. 运行 2-stage Path-OPD、Final-OPD-warmup、teacher/loss 对照和固定的 response-cap 对照。
7. 对所有主要 checkpoints 运行 formal evaluation 与 same-checkpoint raw-gradient probes。
8. 用完全相同流程运行 Qwen3-4B 的 specialist teachers、MT-GRPO、Final-OPD 和 Path-OPD。
9. 汇总 seed 42 的单次运行结果，直接生成 Table 1–2、Figure 1–4 和 appendix dashboard。

这个顺序没有“结果达到某个阈值才继续”的分支；后续 runs 只依赖前序 teacher checkpoints 是否完整产出。

## 9. 需要补齐的实现

正式开跑前直接完成四项工程工作：

1. launcher 支持从当前 run 对应的同尺寸 ST-GRPO checkpoint 启动 task-routed teacher endpoints；
2. Path-OPD 在 25/50/75% consumed-prompt boundary 原子切换三个 teacher endpoints，并记录 teacher hash；
3. 新增 checkpoint-only task-gradient probe：同一 checkpoint 依次 backward 三个固定 task batches，
   保存 exact raw norm/dot/cosine，不调用 optimizer step；
4. appendix 对照支持 renormalized teacher Top-64 log-prob payload，主方法仍使用当前 sampled-token 路径。

这些都是执行计划所需的固定实现，不构成新的论文实验轴。

## 10. 每条 run 必须保存的产物

- 完整 config、git commit、seed、模型/tokenizer hash、数据和 benchmark revision；
- teacher checkpoint hash、endpoint config 和每次 switch 事件；
- train/rollout/eval metrics JSONL；
- 0/25/50/75/100% checkpoints；
- formal evaluation 的逐题 response、reward/pass、错误状态；
- gradient probe 的 task norm、pairwise dot/cosine 和有效 token 数；
- sampled prompts、responses、有效 tokens、wall-clock、student/teacher GPU hours；
- completion/failure marker，失败 run 不覆盖原日志。

建议目录：

~~~text
outputs/raw_gradient_interference/
  qwen3_1.7b/
    seed42/
      teachers/{math,code,science}/
      mt_{ppo,grpo,final_opd,path_opd,offline_kd}/
      ablations/
  qwen3_4b/
    seed42/
      teachers/{math,code,science}/
      mt_{grpo,final_opd,path_opd}/
  paper/
    tables/
    figures/
    appendix/
~~~

## 11. 与现有论文实验设计的对应关系

- [M2RL](https://arxiv.org/abs/2602.12566)：沿用 Math/Code/Science、多任务 joint RL、domain teachers、
  MATH/AIME/LiveCodeBench/GPQA 评测和 task-routed OPD 的常规设置；
- [MOPD](https://arxiv.org/abs/2606.30406)：沿用 shared initialization、same-origin domain-RL teachers、
  final-teacher OPD、Mix-RL/Offline-KD/parameter-merge baselines，以及 KL/entropy/accuracy 训练曲线；
- [SFT Conflicts, RL Coexists](https://arxiv.org/abs/2608.03573v2)：沿用 single-task checkpoint update
  L2 norm/cosine、multi-task vs specialist 结果表和第二模型规模复现；
- [DAPO](https://arxiv.org/abs/2503.14476)：沿用 reward/pass rate、informative-group fraction、entropy、
  response length 和正式 accuracy curve 等标准 RL 训练动态；
- [Dense Supervision, Sparse Updates](https://arxiv.org/abs/2606.13657v3)：只沿用 checkpoint relative
  update norm 和 layer-wise summary，不引入其 spectral/support-mask 指标。

最终论文结构就是：

\[
\underbrace{\text{PPO/GRPO/OPD 信号如何变化}
+\text{raw task-gradient norm/cosine}}_{\text{机理}}
\quad\Longrightarrow\quad
\underbrace{\text{沿 domain-RL checkpoints 逐段蒸馏}}_{\text{Path-OPD}}
\quad\Longrightarrow\quad
\underbrace{\text{标准 benchmark、消融与规模复现}}_{\text{方法证据}}.
\]
