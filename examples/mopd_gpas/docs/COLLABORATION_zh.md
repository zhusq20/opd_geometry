# 两周核心实验协作

仅认领 `uniform-s1`、`gpas-s1`、`gpas-raw-s1`、`d3-fixed-s1` 四个 run，均 seed 42。U/G 优先；有空闲同型 slot 时并行加速已有清单。用 `configs/campaign.example.yaml` 记录 owner、machine、GPU 分配和状态。

共享相同的 student/base Megatron、指定 teachers、数据和 v5 `protocol.json`。初始 reference bank 及其评分只创建一次；并行进程使用文件锁，跨机器执行时分发同一份 bank 和哈希。无需 initial-KL 资格测量。四域 RL teacher 的实际权重哈希必须一致。

Uniform 执行者保留第 250 步完整模型/优化器/数据/随机状态，供唯一的共同 checkpoint 对照。各执行者交付 0/100/200/300/400/500 fixed loss、final fresh loss、500 条 allocation logs、指定能力评估与 provenance。U/G 还交付 250 步能力结果。

恢复使用 `run_stage.sh resume RUN_ID`，不用新 run 覆盖旧输出。归档使用 `run_stage.sh package RUN_ID`；公共 reference bank 和 U/250 完整 checkpoint 另外共享。最后统一执行 `run_stage.sh analyze`，保留无收益及未达目标的结果，不新增种子或可选方法。
