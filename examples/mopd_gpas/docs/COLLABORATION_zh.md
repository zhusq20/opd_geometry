# 多机协作手册

八个主配置互相独立，可各占一个同型 slot 并发运行；一个 slot 是一张 96GB 训练卡加一张 48GB 推理卡。用 `configs/campaign.example.yaml` 记录 owner、machine、GPU 编号和状态，配置 ID 不得重复认领。

共享前置产物只有：student/base Megatron checkpoint、math/IF teacher、Qwen3-4B、训练数据、128×4 held-out、`initial_kl.json` 和由它冻结的 `protocol.json`。协调者先生成并校验这些文件，再分发同一份哈希；不存在共享 warm-start。Uniform 执行者还需回传 step 50/250/500 的完整 optimizer/controller checkpoint，供三个 scalar-only variance probe 使用。

建议状态流：`pending -> running -> capability -> packaged -> complete`。执行者提交 `${CONFIG_ID}-seed42-analysis.tar.gz`，包内包含 provenance、allocation、metrics、teacher-loss artifacts 和 capability artifacts，不包含 checkpoint/W&B cache。

```bash
bash examples/mopd_gpas/run_stage.sh train "${CONFIG_ID}"
bash examples/mopd_gpas/run_stage.sh capability "${CONFIG_ID}"
bash examples/mopd_gpas/run_stage.sh package "${CONFIG_ID}"
```

协调者把八个 run 目录恢复到同一个 `${MOPD_OUTPUT_ROOT}`，运行 `run_stage.sh variance all` 和 `run_stage.sh capability all`，再运行 `run_stage.sh analyze`。分析会严格检查 500-step / 32k clock、step 0–500 的 11 个 held-out 点、两 GPU wall-time 计费、128 prompt 配对关系和三个 16/16 held-out 方差交叉拟合。单种子结论的区间只表示 held-out prompt bootstrap，不表示种子间不确定性。
