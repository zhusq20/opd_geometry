# 2026-09-06 实现验证

本页记录代码与管线验证；这些小批结果不构成论文主实验结论。使用方式见 [README_zh.md](README_zh.md)。

## 配置与测量范围

- `qwen3`：Qwen3-1.7B Base、现有四个 RL teachers 与数学/代码/IF/科学数据。
- `smollm3`：原始 SmolLM3-3B Base 权重、Open-MOPD 发布的三个 RL teachers 与原始数据。两个学生权重分片的完整 SHA256 与指定 Base revision 一致；采用 Open-MOPD 的 tokenizer/template。
- 两套配置支持 `s-pg`、`s-tk`、`m-pg`、`m-tk-dr`、`m-tk-dt`、`m-tk-gt`。多域保持相等 prompt 配额。
- 训练输出上限 4096。评估请求上限 32768，同时受原生总上下文约束：Qwen 为 32768，实际输出预算为 `min(32768, 32768 - prompt_tokens)`；Smol 为 65536。没有扩展 Qwen 上下文。
- OPD 保留任务 reward scorer，奖励系数为零；W&B 同时记录 reward、长度/截断、token 份额、能力指标、实际成本与参数测量。
- PG/top-64 用于在线训练；full-vocabulary 用于同 checkpoint 的局部诊断。FP32 master 更新、累计 FP32 位移与 BF16 checkpoint 位移分别记录。

## 自动与数值验证

最终主回归：**293 passed，5 skipped**。日志：[final_regression.log](../../local/paper_implementation_20260906/final_regression.log)。实际多进程日志暴露的同一步字段合并问题另有回归检查，分析器保留各 worker 的 reward、token 份额和成本字段。

其他已完成检查：

- 12 个 profile/condition 的实际参数解析，训练、评估和 teacher 服务的原生上下文设置。
- SmolLM3 的真实 GPU HF–Megatron forward 数值对齐，覆盖绑定 embedding 与每四层一次的无 RoPE attention。
- 三种 reduction 的损失/梯度/Adam 更新一致性；采样 PG 与 full-normalized corrected top-64 公式。
- 原始 Open-MOPD IF、IFBench、代码 scorer；真实 SandboxFusion 请求。
- Qwen 的未命名模型词表位置：tokenizer 命名 151669 个 ID，但学生和教师均有 151936 个模型词表位置。保留 tokenizer 映射一致性检查，并按模型词表边界校验。实时 teacher 请求确认边界位置的数值评分。
- 初始快照复用检查：当前 FP32/BF16 参数、配置和 Adam 状态必须逐项匹配已有初始快照；不匹配会拒绝复用。

## 真实运行证据

| 验证 | 范围 | W&B |
|---|---|---|
| Qwen 原生上下文能力评估 | 四域各一条；两个长响应在原生上下文内截断；已完整结束 | [tg9hedta](https://wandb.ai/zsqzz/iclr2027-mopd-dynamics/runs/tg9hedta) |
| Qwen checkpoint 局部验证 | 一个真实 math teacher、一个公共前缀、三种损失；已完整结束 | [rzamlnhz](https://wandb.ai/zsqzz/iclr2027-mopd-dynamics/runs/rzamlnhz) |
| Qwen 四域 top-64 在线验证 | 16 条响应、一次更新；完成状态由 run marker 记录 | [ms18m9po](https://wandb.ai/zsqzz/iclr2027-mopd-dynamics/runs/ms18m9po) |
| Smol 三域 PG 在线验证 | 12 条响应、一次更新；完成状态由 run marker 记录 | [phzc4uga](https://wandb.ai/zsqzz/iclr2027-mopd-dynamics/runs/phzc4uga) |

局部 checkpoint 验证覆盖 1,720,574,976 个参数的有限梯度和提议更新，每种损失另检查 930 个 Adam 坐标的精确一致性及原状态不变性。该检查使用 FP32 master forward，不把它等同于 BF16 在线训练轨迹。完整 teacher-pair/support 科学实验网格尚未执行；两种架构的小模型测试覆盖完整诊断 runner。

W&B shared run 的 summary 可能不包含其他 worker 的字段；已通过在线 history API 核对 reward 与 paper 记录，不能仅用 summary 判断数据是否上传。完整 plot-data artifact 在运行成功结束时上传。CSV/PDF/PNG 由实际记录生成。

## 本机启动

在仓库根目录的新 shell 中执行。本站配置保留正式实验的默认预算，不继承上面一次更新的验证预算。

```bash
source local/paper_implementation_20260906/qwen3_site.env
# 或：source local/paper_implementation_20260906/smollm3_site.env
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_mopd.sh m-tk-dr
bash examples/mopd_gpas/analyze_all.sh
```

本站配置使用已准备的模型、数据、reward 服务及健康 GPU。通用配置与重新获取资产的步骤见 README。
