# Open-MOPD 论文完整系统复现通道

这里运行的是 Open-MOPD 论文的完整 final-stage 系统，不是 Qwen3 四任务主协议中的 `open_mopd` K=1 适配版。它固定使用论文的 SmolLM3-3B MixSFT student、math/code/IF 三个 RL teacher、公开 prompt mixture、K=4、student top-k=16 dense reward 和 inner-update reward refresh。

## 为什么独立运行

当前主实验固定为 Qwen3-1.7B、四任务、每个 rollout 64 条 response、每个 rollout 一次 AdamW update、4096 response cap，并通过 SGLang teacher endpoint 只取得 sampled-token log-prob。论文完整系统改动了 student、任务数、数据、response cap、batch size、teacher 打分张量和每次 rollout 的更新次数。把它直接替换进八条主轨迹会同时改变多个控制变量，所以完整系统只作为论文复现/外部参考报告，不进入 Qwen3 四任务主表；主表中的 `open_mopd` 继续承担同协议 K=1 baseline。

## 固定内容

- 官方代码固定在 commit `4809a96cf85a869106ff0ff3f37d0a51e12010ae`。
- 1x8 GPU；train batch 1024；mini batch 256，因此每个 rollout 有 K=4 个顺序更新。
- 学习率 `1.5e-6` constant；PPO clip low/high `0.2/0.28`；KL 关闭；600 steps。
- prompt limit 2048；math/code response limit 16384；IF response limit 2048。
- student top-k=16、student probability reward weighting、rollout nucleus `p=0.99`、domain-label hard routing。
- math:code:IF prompt sampler 为 2:2:1；token-share target 为 1/3:1/3:1/3。
- forward gap-following `alpha=1.0`，factor clip `[0.05,20]`；reward refresh 开启。

完整机器可读配置见 `recipe.json`。训练脚本只显式设置论文给出的算法超参；dynamic batching、teacher offload 和每 50 step 保存等项目是执行/可恢复性设置，不参与调参。

## 运行

先拉取固定版本的官方实现和公开资产：

```bash
bash examples/mopd_gpas/open_mopd_full/fetch.sh all
cd local/open_mopd_full/source/training
bash install_requirements.sh
cd -
```

先检查完整命令，再启动：

```bash
bash examples/mopd_gpas/open_mopd_full/train.sh --dry-run
bash examples/mopd_gpas/open_mopd_full/train.sh --run
```

也可以通过统一入口运行：

```bash
bash examples/mopd_gpas/run_stage.sh open-full-fetch all
bash examples/mopd_gpas/run_stage.sh open-full-train --dry-run
bash examples/mopd_gpas/run_stage.sh open-full-train --run
```

默认路径都在 `local/open_mopd_full/`。可用 `OPEN_MOPD_FULL_ROOT` 整体搬迁，也可分别设置 `OPEN_MOPD_SOURCE_ROOT`、`OPEN_MOPD_ASSET_ROOT` 和 `OPEN_MOPD_OUTPUT_ROOT`。

## 评测

默认评测官方发布的 final checkpoint；设置 `OPEN_MOPD_EVAL_MODEL` 可改为自己合并后的 Hugging Face checkpoint：

```bash
bash examples/mopd_gpas/open_mopd_full/evaluate.sh --dry-run
bash examples/mopd_gpas/open_mopd_full/evaluate.sh --run
```

评测固定使用论文六个数据集：AIME24/25 mean@64、LiveCodeBench v5/v6 mean@10、IFEval/IFBench_test mean@1，并按“数据集 -> domain -> 三领域”做 macro average。正式评测前需按官方 `evals/README.md` 安装 verifier 依赖及其固定版本的第三方仓库。

训练产出的 FSDP checkpoint 可按官方 verl merger 转成 HF 目录，例如：

```bash
cd local/open_mopd_full/source/training
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /path/to/global_step_600/actor \
  --target_dir /path/to/open_mopd_full_hf
```

随后设置 `OPEN_MOPD_EVAL_MODEL=/path/to/open_mopd_full_hf` 再运行评测。
