# v4 实验协议与运行手册

## 冻结协议

- student：Qwen3-1.7B，non-thinking。
- teacher：math/IF 为同源 1.7B RL teacher；code/science 为原版 Qwen3-4B，non-thinking。
- seed 42；500 step；每步 `G=16` 个 task micro-batch，每个 `b=4` prompt，每 prompt 一条回复；`2 <= m_i <= 8` 且 `sum_i m_i=16`。
- 每任务独立 matched prompt stream，不重复；每任务准备 16,000 条。每步恰好 64 条 response，总预算 32,000。
- response cap 4096；AdamW 与学习率 schedule 在八个配置间一致。论文 baseline 只引入它们的算法超参，不修改共享 student/data/response budget/optimizer，不做调参。
- held-out 每任务固定 128 prompt，在 step 0、50、…、500 评测；方法差值使用 prompt-level paired bootstrap 1,000 次。
- checkpoint 每 50 step 保存 Hugging Face 权重；完整 optimizer/controller/prompt cursor 状态只保留最新 resume 点，Uniform 额外永久保留 step 50/250/500。

## 八个配置

| 配置 | 分配或 loss |
|---|---|
| `uniform` | 每步 `[4,4,4,4]`，固定目标 |
| `gpas` | `m_i` 按 `w_i sqrt(e_i)` 有界整数分配 |
| `cost_gpas` | 穷举使 `(C + sum m_i tau_i) sum w_i^2 e_i/m_i` 最小 |
| `raw_noise` | 用 raw-gradient noise 替换 AdamW-scaled noise |
| `loss_gap` | 按固定权重乘 teacher-loss EMA 分配 |
| `std_mopd` | `[4,4,4,4]`，整步所有有效 token 直接求均值 |
| `d3_mopd` | D³-MOPD 的 remaining-gap × descent-velocity 动态采样，加 batch jitter；整步 token mean |
| `open_mopd` | `[4,4,4,4]`；按当前 batch 的 token share 与 forward reward gap 重加权 |

原有自适应方法首步为 Uniform；D³-MOPD 依论文在 warmup 期也启用 jitter。`e_i` 使用当前步直接估计；`tau_i`、`C` 和 loss 使用 0.9 EMA。固定目标方法先 token mean、再 response mean、再 micro-batch mean，任务聚合系数为 `w_i/m_i`，最后只执行一次普通 AdamW update 和一次全局 clipping。StdMOPD、D³-MOPD 与 Open-MOPD 会改变有效训练目标，因此只比较各任务 `L_i` 与 benchmark，不把到固定目标阈值的 GPU-hour 与其他方法并列。

### 论文 baseline 忠实性边界

- D³-MOPD 使用论文 Table 3：`n=10, W=10, R=3, S0=5, EMA window=10, epsilon_KL=0.15, T=0.5, epsilon=0.10, eta=0.30`。watcher 在 controller 的同一 rollout frontier 同步计算，与论文异步 status-file 版在数学上等价，且可随 checkpoint 原子恢复。论文概率经 jitter 后，再投影到本主协议共享的 `[2,8]` 计数边界；这是为了不突破每任务 16k 非重复 prompt stream。
- Open-MOPD 使用论文 Eq. 8/9 与 Table 8：四任务等目标 `g*=1/4`、`alpha=1.0`、forward gap factor clip `[0.05,20]`，`m_d` 是当前 batch 的每 token 绝对 sampled-OPD reward magnitude，并保证 `sum_d w_d s_d^tok=1`。主协议每 rollout 只有一次 optimizer update，所以论文明示的 K=1 情形下 reward refresh 为 identity。这是可公平接入当前 sampled-token OPD 的 MVP baseline，不是论文 SmolLM3-3B、K=4、dense student-top-k=16 整套系统的 bitwise reproduction。

### 完整 Open-MOPD 外部参考

`examples/mopd_gpas/open_mopd_full/` 另外提供论文 final-stage 的完整复现通道。该通道固定官方 commit 和公开资产，恢复三领域 hard routing、batch 1024 / minibatch 256 的 K=4、student top-k=16 dense reward、1/3 token-share target、forward gap-following 与每个 inner update 的 reward refresh。由于它同时改用 SmolLM3-3B、三任务公开数据、16K/2K response cap 和 1x8 GPU，因此只单列为 paper-protocol reference，不作为第九条主协议曲线，也不参与 32,000-response 同预算排名。完整配置和命令见 [`../open_mopd_full/README_zh.md`](../open_mopd_full/README_zh.md)。

## 命令

```bash
source local/mopd.env
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_stage.sh smoke
bash examples/mopd_gpas/run_stage.sh train gpas
```

可用配置就是上表八项。只运行新增论文 baseline：

```bash
bash examples/mopd_gpas/run_stage.sh baselines
```

中断恢复：

```bash
bash examples/mopd_gpas/run_stage.sh resume gpas
```

Uniform 的 held-out 梯度方差检查：

```bash
bash examples/mopd_gpas/run_stage.sh variance all
```

每个检查点、每个任务重新生成 32 个 micro-batch；16/16 两半交叉估计与评估，并报告 2/4 micro-batch 在线子集误差。artifact 只包含 raw/AdamW-scaled 范数、loss 和计时标量，不保存梯度向量；Cost-GPAS 使用 Uniform checkpoint 内的训练期 EMA `tau_i` 与 `C`，不使用 probe 自身的额外测量开销。

最终能力评测与打包：

```bash
bash examples/mopd_gpas/run_stage.sh capability all
bash examples/mopd_gpas/run_stage.sh package gpas
```

`capability all` 包含四个唯一 reference 模型（初始学生、math teacher、IF teacher、code/science 共用的 Qwen3-4B）和八个最终配置。分析按任务选择对应 teacher，计算 `(s-s_init)/(s_teacher-s_init)`；teacher/student 差距小于 0.03 时在报告中标为不可靠。

八条主轨迹、三个方差 probe 和十二个唯一模型的能力结果到齐后：

```bash
bash examples/mopd_gpas/run_stage.sh analyze
```

成功条件是 `run_complete.json` 标记 500 updates、allocation log 恰好 500 行且最终 response clock 为 32,000，并存在 11 个 held-out step-clock artifact。GPU-hour 统一按完整 step wall time 乘两张分配 GPU 计算。

正式运行前先用 `run_stage.sh smoke` 跑冻结的 GPAS 20-step smoke test；它在 step 10/20 存 checkpoint 并在 step 0/10/20 评测。检查单步不超过约两分钟、截断率不超过 5%、显存峰值和 `e_i` 变化；需要验证恢复时，在 step 10 后中断并执行 `run_stage.sh resume gpas-smoke`。正式八条轨迹始终使用 500-step 参数。
