# Code checkpoint warm-start 的无 sandbox 顺序 GRPO

完整的边界测试、遗忘、稀疏支持、谱几何和同-checkpoint 梯度干扰口径见
[`SEQUENTIAL_ANALYSIS_PROTOCOL_zh.md`](SEQUENTIAL_ANALYSIS_PROTOCOL_zh.md)。

入口：

```bash
AVAILABLE_CUDA_DEVICES=0,1,2,3 \
  bash examples/optimizer_geometry/run_sequential_grpo.sh
```

默认路径为：

```text
已有 Code final checkpoint
  -> Math GRPO
  -> Knowledge/Science GRPO
  -> IF GRPO
```

脚本不会重新训练 Code。默认 Code 起点显式固定为当前 seed-42、update-300（目录
`iter_0000299`）的 Qwen3-1.7B Code GRPO checkpoint，不会随 checkpoint root 的 latest marker 漂移；也可以显式覆盖：

当前默认 checkpoint 的 `.metadata`/`common.pt` 完整，但原 run 的 provenance 仍标为 `running` 且没有
`run_complete.json`。脚本按用户指定复用该权重，并把这个事实写入 `sequence_manifest.json`；它不会把原
run 误报为完整实验。

```bash
CODE_CHECKPOINT=/absolute/path/to/checkpoints \
SEQUENCE_DIR=/absolute/path/to/new_sequence \
AVAILABLE_CUDA_DEVICES=0,1,2,3 \
  bash examples/optimizer_geometry/run_sequential_grpo.sh
```

先检查计划而不启动训练：

```bash
PLAN_ONLY=1 bash examples/optimizer_geometry/run_sequential_grpo.sh
```

## 固定实验语义

- 任务顺序固定为 Code → Math → Knowledge → IF；代码内部名称 `science` 在产物中显示为
  `knowledge`。
- 起点和任务边界只继承模型权重；AdamW moments、scheduler/RNG 和 task sampler 均重置。
- 三个新增阶段使用 AdamW、LR `1e-6`、weight decay 0、group size 16、seed 42、关闭 thinking。
- 复用的 Code checkpoint 的实际超参数会从原 `provenance/run_manifest.json` 自动提取到
  `sequence_manifest.json`，不会被误记成新增阶段的超参数。若覆盖为别的 checkpoint，则记录该
  checkpoint 可找到的实际 provenance。
- 每个新增任务默认只使用前 4,800 个 prompt，rollout 时再按 seed 42 打乱；在 16 prompts × 16
  responses 的配置下正好是 300 个 optimizer updates，与复用的 Code update-300 起点对齐。可用
  `TRAIN_PROMPTS_PER_TASK` 覆盖，但必须是 16 的正整数倍。
- 固定 probe 仍选择完整源文件中恰好出现一次的末尾 128 个 prompt，并与训练前缀保持不相交。这里只
  准备 Math、Knowledge 和 IF，因此不会扫描 Code JSONL，也不会调用代码判分服务。
- checkpoint 每 100 个 optimizer updates 保存一次；默认 300-update 短跑会得到 update 100、200、300
  三个 checkpoint（目录编号分别为 `iter_0000099`、`iter_0000199`、`iter_0000299`）。
- 段内使用现有 geometry observer 逐 update 保存 raw-gradient norm、optimizer direction、模型更新、
  displacement、逐层统计和低频 CountSketch 轨迹。
- 默认只训练，不运行边界 eval 或 same-checkpoint gradient probe；需要时可显式开启。
- 核心 Math→Knowledge→IF 训练 launcher 不运行 Code 判分；完整分析另用固定六数据集 boundary evaluator，
  在原版、Code、Math、Knowledge、IF 五个边界补测 Code，并要求已验证的 SandboxFusion。

这里的 optimizer reset 是实验定义，不应在论文中写成无缝延续 AdamW 状态。如果要研究是否保留
optimizer state，应作为单独的实验轴实现，不能与当前结果混报。

## Same-checkpoint raw-gradient probe

补充分析为 Code、Math、Knowledge 和 IF 各冻结同一组 probe prompts。每个边界 checkpoint 上，四个任务分别运行
固定 probe backward batches；当前论文产物为每任务 16 prompts × 16 responses。probe-only 模式不会调用
optimizer.step 或 scheduler.step（LR 仍固定为 0，作为额外保护）；observer 在 clipping 与 AdamW 之前把
八个 batch 的 FP32 raw gradients 求平均，并按 distributed optimizer 的唯一拥有区间落盘。manifest 同时
保存每批与总 effective-token count。

`analyze_raw_gradient_probes.py` 流式计算：

```text
||g_i||_2,  g_i^T g_j,  cosine(g_i, g_j)
```

这里没有 CountSketch 或随机投影。artifact 保存的是 loss gradient；实验手册中的 ascent gradient
`g=-grad(L)` 同时翻转所有任务符号，因此 norm、pairwise dot 和 cosine 均不变。
若任一任务的 probe 梯度范数为零，则该任务没有可比较的方向；对应 cosine 和冲突符号记为
`null/n/a`，不能把零向量解释为与其他任务正交。

probe launcher 会先把 checkpoint root 的 `latest_checkpointed_iteration.txt` 解析为不可变的 `iter_*` 或
`release` 目录，所以四个任务不会因 marker 后续变化而静默加载不同参数状态。

单独重跑某个 checkpoint 的 probe：

```bash
LOAD_CHECKPOINT=/path/to/checkpoints \
PROBE_CONFIG_ROOT=/path/to/sequence/prepared_data \
OUTPUT_DIR=/path/to/sequence/gradient_probes/custom_anchor \
ANCHOR_NAME=custom_anchor \
PROBE_TASKS="code math science if" \
AVAILABLE_CUDA_DEVICES=0,1,2,3 \
  bash examples/optimizer_geometry/run_raw_gradient_probe.sh
```

## 可选测量开关

默认命令就是 4,800 prompts/task 的 train-only 快速路径：

```bash
AVAILABLE_CUDA_DEVICES=0,1,2,3 \
  bash examples/optimizer_geometry/run_sequential_grpo.sh
```

如需恢复 sandbox-free 的三任务边界评测和 raw-gradient probe：

```bash
RUN_BOUNDARY_EVAL=1 RUN_GRADIENT_PROBES=1 RUN_ORIGIN_MEASUREMENTS=1 \
AVAILABLE_CUDA_DEVICES=0,1,2,3 \
  bash examples/optimizer_geometry/run_sequential_grpo.sh
```

若训练阶段异常退出，先检查日志和 checkpoint，再显式设置：

```bash
RESUME_INCOMPLETE_STAGE=1 \
AVAILABLE_CUDA_DEVICES=0,1,2,3 \
  bash examples/optimizer_geometry/run_sequential_grpo.sh
```

该模式只 resume 当前阶段自己的 checkpoint、sampler 和 geometry frontier，不会从前一任务重新开始。

## 主要产物

```text
SEQUENCE_DIR/
  sequence_manifest.json
  stage_boundaries.jsonl
  prepared_data/
    sequential_data_index.json
    all_tasks_eval.yaml
    {math,science,if}/*_gradient_probe.{jsonl,yaml}
    {math,science,if}/*_on_policy.yaml  # 默认指向原数据的前 4,800 行 slice
  stages/{01_math,02_knowledge,03_if}/RUN/
    geometry/actor/metrics.jsonl
    geometry/rollout/{metrics.jsonl,samples/}
    checkpoints/
    provenance/run_manifest.json
    run_complete.json
  evaluations_v2/{pre_code,00_code_origin,01_after_math,02_after_knowledge,03_after_if}/  # 可选
  gradient_probes/ANCHOR/  # 可选
    raw_gradients/{code,math,knowledge,if}/
    analysis/{summary.json,raw_gradient_geometry.csv,global_cosine_matrix.csv}
  parameter_geometry/
    summary.json
    stage_update_geometry.csv
    cumulative_update_geometry.csv
    stage_cosine_matrix.csv
    cumulative_cosine_matrix.csv
```

`parameter_geometry` 同时区分：

- stage-local update：Code、Math、Knowledge、IF 每一段实际造成的参数变化；
- cumulative update：相对 Qwen3-1.7B 原版起点，经过每个阶段后的累计参数变化。

两类结果均为 torch-dist checkpoint 中 BF16 realized model tensors 的 full-dimensional FP64
norm/dot/cosine；reader 显式排除 `optimizer.*` FP32 master weights 和 Adam moments，因而不会把任务边界的
optimizer reset 错算为参数空间位移。
