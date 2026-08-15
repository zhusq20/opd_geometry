# OPD / PPO / GRPO optimizer 比较的预注册协议

这份协议把“算法差异”“optimizer 差异”和“算力预算差异”分开。同一 task 的 AdamW/SGD/Muon
都看同一轮、同一顺序的题；task 之间不再人为补到相同 prompt 数。wall-clock 只作为效率指标，
不能作为提前终止某个 optimizer 的依据。

## 固定不动的主实验量

- student、tokenizer、初始 checkpoint、prompt 顺序和数据混合权重；
- frozen pure-OPD `run_opd_*` 使用 `BATCH_PROFILE=opd64x1`：`rollout_batch_size=64`、每 prompt
  采样 `n=1`、`global_batch_size=64`，并用 `NUM_EPOCH=1` 覆盖通过 `max_prompt_len=2048` 过滤后的全部 usable prompts；最后不足 64 的真实
  尾批仍做一次 update。`responsive16/responsive8/reference256` 继续供旧配方、RL 和 tuning 使用；
- train sampling 为 `temperature=1.0, top_p=1.0, top_k=-1`，所有 cell 使用同一 seed；
- 所有 plain GRPO/PPO 与 OPD variant 均显式使用 Qwen3 `enable_thinking=false`；OPD student 生成
  non-thinking trajectory，teacher 以 `max_new_tokens=0` 对完全相同的 token IDs 逐 token 打分；
- frozen `run_opd_*` 训练使用 `max_prompt_len=2048`、`max_response_len=4096`；在线和最终 evaluation 的 response cap
  按任务固定：Math 为 32,768，Code/Science 为 16,384；
- plain GRPO/PPO 训练统一使用 `max_response_len=8192`；Math online evaluation 固定同时加载
  AIME'24 和 MATH-500，并独立使用 32,768-token eval cap；
- BF16、4-way DP、TP/PP/CP=1、full recomputation、gradient clipping=1；
- PPO critic 在所有 actor optimizer cell 中固定使用同一 AdamW 配置和同一初值；
- OPD teacher checkpoint、服务版本、sampling/logprob 参数与 `opd_kl_coef` 固定；
- reward/sandbox 版本、eval prompts、解码参数和每题采样数固定。

`max_response_len` 不应仅因“显存能放下”就冻结。先在不进入主结果的 prompt 样本上比较
4096/8192/12288/16384，并在当前明确冻结的 non-thinking 模板下报告触顶率、pass rate、吞吐和 OOM；主配置
优先选择触顶率低于 10% 且计算可承受的最小上限。当前 frozen OPD cell 固定为 4096，必须把截断率
作为主诊断量，并用至少一个更长上限做敏感性分析。同一 algorithm 内的所有 optimizer 使用同一已冻结上限。
在当前 96 GiB 卡和 `SGLANG_MEM_FRACTION=0.6` 上，每个 engine 实测约有 498k KV-cache token。
原 2048+8192 cap 下 44 请求占 450,560 token。保留 2048 prompt 过滤后，4096 response cap 可把
安全上限提高到 72（442,368 token，约 11.2% 余量）；32,768 response cap 使用 12
（417,792 token，约 16% 余量）。
换 batch、模型、显卡、cap、GPU 拓扑或 memory fraction 后必须重新校准。
eval 另设跨全部 eval 数据集共享的客户端全局并发 48；4 个 rollout engine 下平均为
12 requests/engine。Math 的 2,048 prompt + 32,768 response 上限每个 engine 最多预留 417,792
KV-cache tokens，仍低于当前实测约 498k 的容量；Code/Science 的 16k cap 余量更大。该阀门与训练用的
`SGLANG_MAX_RUNNING_REQUESTS=72` 相互独立。

```bash
TUNING_DATA_MANIFEST=/data/tuning/train.yaml \
  bash examples/optimizer_geometry/run_response_cap_pilot.sh
python examples/optimizer_geometry/select_response_cap.py \
  outputs/response_cap_pilot --max-truncation 0.10 \
  --output /data/tuning/response_cap_selection.json
```

默认对每个 cap 使用同一组初始模型、prompt stream 和 seeds 1042/1043，并显式使用
`PILOT_BATCH_PROFILE=reference256` 执行更新前的一次 256-prompt rollout（随后做一个可行性
update），以免用 8/16 个 prompt 估计截断率。选择 matched-seed 平均触顶率不超过 10% 的最小 cap。默认
`cap:max_running` 组合仅适用于本机 96 GiB 校准；其他硬件必须显式覆盖 `RESPONSE_PROFILES`。

每个日志事件必须同时记录 `rollout_id`、一基的 `num_updates` 和 `model_version`。
生成数据属于更新前的 model version；geometry/train 事件属于刚完成的更新；eval 明确标注
`pre_train`、`post_update` 或 `final`。

## 只用 tuning split 选择的量

不能用论文报告的 MATH-500、AIME、GPQA、LiveCodeBench 或其最终 seed 选超参数。先固定一个不进入主表的
tuning split 和两个 tuning seeds，所有 optimizer 得到相同候选数与相同 12,800-prompt 预算；
在 `responsive16` 下对应 800 updates：

| 参数 | 候选 |
|---|---|
| AdamW actor LR | `1e-7, 2.5e-7, 5e-7, 1e-6` |
| Muon actor LR | `1e-7, 2.5e-7, 5e-7, 1e-6` |
| vanilla SGD actor LR (GRPO) | `1e-2, 2.5e-2, 5e-2, 1e-1` |
| vanilla SGD actor LR (OPD/PPO) | `1e-3, 2.5e-3, 5e-3, 1e-2` |
| weight decay | 主分析固定 `0`; 仅做 `0.01` 敏感性分析 |
| grad clip | 主分析固定 `1.0`; pilot 检查 `0.5, 2.0` |
| OPD KL coefficient | `0.1, 0.3, 1.0, 3.0`，先用 AdamW 选定后对所有 optimizer 固定 |

上表是默认 `responsive16` 网格。若显式选择 `responsive8`，脚本会围绕其缩放起点重排为 adaptive
`9e-8, 1.8e-7, 3.6e-7, 7.2e-7`、GRPO-SGD
`9e-3, 1.8e-2, 3.6e-2, 7.2e-2`、OPD/PPO-SGD
`9e-4, 1.8e-3, 3.6e-3, 7.2e-3`，而不是拿 b16 网格机械套用。

选择准则预先固定为：12,800 prompts 内的 validation reward AUC 最大；若差异小于跨 seed 标准误，选择
较小 LR。任何 NaN/Inf、连续 3 次 grad norm 爆炸或 clip fraction 超过 50% 的候选判为不稳定。
选定后用一个额外、仍不进入主表的 confirmation seed 跑到完整预算。不要给每个主任务重新选 LR，
否则 optimizer 与调参自由度混杂。

实现命令为：

```bash
# 先固定 OPD 强度，再分别选择 algorithm × optimizer 的 LR。
TUNING_DATA_MANIFEST=/data/tuning/train.yaml \
TUNING_EVAL_CONFIG=/data/tuning/eval.yaml \
  bash examples/optimizer_geometry/run_opd_coefficient_tuning.sh

TUNING_DATA_MANIFEST=/data/tuning/train.yaml \
TUNING_EVAL_CONFIG=/data/tuning/eval.yaml \
  bash examples/optimizer_geometry/run_optimizer_tuning.sh

python examples/optimizer_geometry/select_tuning_hparams.py \
  /data/opd_coefficient_tuning --metric eval/tuning --parameter opd_kl_coef \
  --output /data/tuning/opd_selection.json
python examples/optimizer_geometry/select_tuning_hparams.py \
  /data/optimizer_tuning --metric eval/tuning --parameter lr \
  --output /data/tuning/lr_selection.json
```

选择器要求每个候选具有相同 seed 集合和完整 update 数，以固定 eval 网格上的 trapezoid AUC 为
精确目标；它会排除 non-finite、连续大梯度和频繁 clipping 的 cell，并在最佳候选差异不超过合并
标准误时选较小数值。选参 JSON、所有候选的失败信息和原始 run 都应随论文产物保存。

`responsive16` 的 Adam 使用 `beta1=0.9, beta2=0.98^(16/256)=0.9987381276`；`opd64x1`
按 response batch 比例得到同一数值 `0.98^(64/1024)=0.9987381276`，
`eps=1e-8`。这里不缩放 beta1，是因为小 batch 语言模型研究发现 0.9 跨 batch 仍有效，而 beta2
必须按 token/sample 半衰期调整。Muon momentum=0.95、Newton–Schulz steps=5、spectral
scale=0.2，以及 vanilla SGD momentum=0 是 optimizer 定义的一部分；变更它们属于独立消融，
不应在看过主结果后选择。参考依据为
[Small Batch Size Training for Language Models](https://arxiv.org/abs/2507.07101)、
[M2RL](https://arxiv.org/abs/2602.12566)、
[SGD-RL](https://arxiv.org/abs/2602.07729) 和
[OPD optimizer ablation](https://arxiv.org/abs/2606.13657v1)。

## Frozen OPD 的 steps

旧的 `NUM_ROLLOUT=3200` / 51,200 prompts 来自 M2RL 的 `256 prompts × 200 steps` 等预算协议，
不是数据集大小，现已从 frozen `run_opd_*` 删除。当前 tokenizer 在 2048 prompt cap 下的数据为：

| task | prompts = responses | updates (`ceil(N/64)`) | final batch |
| --- | ---: | ---: | ---: |
| Math | 22,050 | 345 | 34 |
| Code | 19,125 | 299 | 53 |
| Science | 19,668 | 308 | 20 |

普通 train/rollout scalar 每个 update 都记录；geometry 每 4 updates（满批 256 prompts）；checkpoint
每 80 updates，并在最终尾批强制保存；eval 每 50 updates，并额外保证最终 checkpoint 有评测。

## 旧 fixed-budget pilot（不控制 frozen `run_opd_*`）

当前机器实测 reference256 的默认长序列 GRPO update 为 1,539.7 秒（25.66 分钟）。默认并发 44
下，`responsive16` 为 199.0 秒（3.32 分钟；rollout 181.1 秒），`responsive8` 为 190.4 秒
（3.17 分钟；rollout 175.6 秒）。b8 因 8192-token 长尾只再快 4.3%。按一个 smoke step 外推
51,200 prompts 分别约 7.37 和 14.10 天；该数值不含启动、eval/save、OPD teacher，也会随策略
response length 改变，只能用于排程预警。小 batch 改善 W&B 反馈延迟，不保证总 cell 时间缩短。

若运行旧的等预算 GRPO/PPO/OPD 对照，可只使用 AdamW 和不进入主表的 seeds 1044--1046 跑到 3,200，
每 160 步 eval（2,560 prompts）、每 800 步 checkpoint（12,800 prompts）。在不知道 SGD/Muon
结果的情况下应用下面的规则一次性确认预算：

1. 计算最近三个 eval 区间的斜率和均值；
2. 对每个 algorithm，若跨 seed 的平均 1,600→3,200 改进小于 1 个百分点，且各 seed 最近三点
   OLS 斜率的 Student-t 95% CI 不完全位于 0 以上，判为在 51,200 prompts 附近饱和或已退化；
3. 只有三个 algorithm 都饱和才采用 3,200；任一仍明显上升，所有 optimizer/algorithm 一起改为
   6,400，并从同一 3,200-step checkpoint 精确 resume；
4. 若方法在 3,200 前退化，主表仍报告统一固定终点，同时报告 best-validation checkpoint 和 AUC，
   不允许按方法各自提前停止。

因此 3,200 不是数据驱动的 epoch 长度，只是旧 51,200-prompt 预算在小 batch 下的表示，不用于
当前 frozen OPD 终点。理想确认实验仍应使用至少 3、最好 5 个独立 seeds；当前算力约束下的 OPD
矩阵只运行 seed 42。其结果只能报告每个 task 的曲线、终点和 matched-task 差异，不能计算跨 seed
sample SD、Student-t CI 或把单次胜负表述为稳定的总体效应。

```bash
TUNING_DATA_MANIFEST=/data/tuning/train.yaml \
TUNING_EVAL_CONFIG=/data/tuning/eval.yaml \
  bash examples/optimizer_geometry/run_budget_pilot.sh
python examples/optimizer_geometry/select_training_budget.py \
  outputs/budget_pilot --metric eval/tuning \
  --output /data/tuning/budget_selection.json

# 仅当选择 JSON 建议 6,400 时，对原 run 精确续跑：
EXTEND_TO_DOUBLE_BUDGET=1 TUNING_DATA_MANIFEST=/data/tuning/train.yaml \
TUNING_EVAL_CONFIG=/data/tuning/eval.yaml \
  bash examples/optimizer_geometry/run_budget_pilot.sh
```

## 当前服务器的 batch 可行性实测（2026-08-13）

服务器共有 10 张 97,887 MiB 的 RTX PRO 6000 Blackwell Server Edition。物理 GPU 0 在真实 CUDA
分配时触发 uncorrectable ECC，不能进入实验池；launcher 现在会同时检查 volatile/aggregate ECC、
绝对空闲显存和空闲比例。物理 1/4/6/8 用作 student、物理 7 用作 OPD teacher 时通过检查。

| 路径 | 实际测试 | 结果与峰值证据 |
|---|---|---|
| 公共长序列 actor 路径 | GRPO，256 prompts × 4，response cap 8192，完整 rollout + backward + update | 完成 1024 responses；训练时 `nvidia-smi` 最坏约 72.9 GiB/卡，距 96 GiB 容量约 24 GiB |
| PPO 特有路径 | 同一 batch，response cap 256，完整 critic + actor update | 完成，actor grad norm 0.693；actor/critic checkpoint 均可校验 |
| OPD 特有路径 | 同一 batch，response cap 256，8B teacher 打分 + actor update | 完成；teacher 约 68.5 GiB，student train 约 25.7 GiB，reverse KL 0.299 |

这是保守的**因子化可行性测试**：最重的 8192-token 公共生成/actor 反向路径与 PPO critic、OPD
teacher 的算法特有路径都真实执行过；没有把耗时数小时的 8192-token PPO 和 OPD 各再完整复制
一遍。AdamW 的 optimizer state 至少不比本研究的 SGD 更省显存，因此该结果支持历史
`reference256` batch 在健康、
空闲的 96 GiB 卡上运行 OPD/PPO/GRPO，但每次正式 cell 仍必须通过 preflight，且不适用于被其他
用户占用的卡或不同 checkpoint/并行配置。

长序列实测的 response 长度均值 7,162、p50/p95/max 都为 8,192，686/1024（66.99%）触顶。
因此结论是“batch 能放下”，不是“8192 已科学合理”。在完成前述 response-cap pilot 前，不应一次性
提交完整论文矩阵。

## 评测和 coding

- Math/Science 在线 eval 使用固定独立集合；最终点必须保存逐样本 response/reward/status。
- Coding 在线曲线使用 LiveCodeBench recent slice 的固定 64 题、greedy pass@1；它只用于曲线，
  不冒充官方 full benchmark。
- Coding 最终表使用官方 `release_v5`，`n=10, temperature=0.2, top_p=0.95`，由保存的逐样本
  结果离线计算 pass@1/pass@5/pass@10。
- 所有模型生成代码只能发送到通过 `sandbox_preflight.py` 的 SandboxFusion endpoint；
  `isolation=none`、可见 host/service canary、可访问公网、缺少逐请求 namespace/cgroup/seccomp、
  内存或 timeout 未实际执行的服务一律 fail closed。

## 主结果之外必须报告的诊断量

reward/pass@k、response length/truncation、KL、entropy、clip fraction、importance ratio、grad norm、
update/weight ratio、effective step size、gradient-update cosine、wall-clock/token throughput、OOM/重试，
actor/critic 跨 rank CUDA peak allocated/reserved memory，以及 PPO critic loss/value error。任何样本过滤、
sandbox timeout 和 reward exception 都要有计数，不能静默丢弃。
