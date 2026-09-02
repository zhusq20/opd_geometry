# 环境与资产配置

本页回答一个问题：一台刚拿到代码的新机器，怎样准备成可运行 MOPD 的 worker。

## 1. 固定代码版本

协调者在分配任务前应公布唯一的 Git commit 或 tag。所有 worker 必须检出同一版本，并且不要在运行期间修改训练代码：

```bash
git clone https://github.com/zhusq20/opd_geometry.git
cd opd_geometry
git checkout <campaign-commit-or-tag>
```

每次正式运行会把 commit、工作区状态、完整命令和代码 snapshot 写入 `provenance/run_manifest.json`。

## 2. 运行环境

推荐使用与本仓库 `build_conda.sh` 对齐的 slime 容器。当前实验栈以 SGLang `v0.5.15.post1`、Megatron-LM `1dcf0dafa884ad52ffb243625717a3471643e087` 为基线；Blackwell 机器使用 CUDA 13 镜像：

```bash
docker pull slimerl/slime:v0.5.15.post1-cu130
docker run --rm -it --gpus all --ipc=host --shm-size=64g \
  -v "$PWD":/workspace/opd_geometry \
  -w /workspace/opd_geometry \
  slimerl/slime:v0.5.15.post1-cu130 bash
pip install -e . --no-deps
```

不用 Docker 时执行仓库根目录的 `build_conda.sh`。不要只根据未锁版本的 `requirements.txt` 重建训练环境。

## 3. 下载实验资产

资产不进入 Git 历史，分成两个公开 Hugging Face 仓库：

- 模型与公共 warm checkpoint：`zsqzz/mopd-gpas-64k-models`
- 训练、评测数据与受控实验输入：`zsqzz/mopd-gpas-64k-data`

`site.example.env` 已固定两个仓库的 revision，所有 worker 使用同一版资产；升级资产时由协调者统一修改 revision。

模型仓库包含 `student_hf/`、`student_megatron/`、`teachers_hf/{math,code,if,science}/` 和 `warm_start-seed42/`。数据仓库包含 `m2rl/train/`、能力评测数据、索引以及 `frozen/controlled/`。

```bash
mkdir -p local
cp examples/mopd_gpas/configs/site.example.env local/mopd.env
source local/mopd.env
bash examples/mopd_gpas/run_stage.sh fetch-assets
```

默认下载到 `local/mopd_assets/`。模型资产约 45 GB，数据约 6.5 GB；还需要为本次训练 checkpoint 和临时文件预留空间。

## 4. 每台机器只修改 site 配置

编辑 `local/mopd.env`：

```bash
export MOPD_TRAIN_CUDA_VISIBLE_DEVICES=0
export MOPD_TEACHER_GPU=1
export CAPABILITY_CUDA_VISIBLE_DEVICES=0
export MOPD_TEACHER_PORT=31001

export WANDB_ENTITY=zsqzz
export WANDB_PROJECT=iclr2027-mopd-gpas-64k
```

如果资产已经在共享盘，只需把以下变量改成共享盘上的实际目录，不必再次下载：

- `MOPD_HF_CHECKPOINT`
- `MOPD_BASE_MEGATRON`
- `MOPD_TEACHER_HF_ROOT`
- `MOPD_WARM_DIR`
- `MOPD_DATA_ROOT`
- `MOPD_CONTROLLED_RESULTS_ROOT`

不要把访问令牌写进 `local/mopd.env`。W&B 使用 `wandb login` 或进程环境；公开 Hugging Face 资产下载不需要登录。

## 5. 硬件约束

正式协议每条训练使用两张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，每张 97,887 MiB：

- student GPU：Megatron training 与 colocated SGLang rollout；
- teacher GPU：四个 teacher 在同一个 SGLang slot 中热切换。

启动器会按型号和显存检查。其他 GPU 上的试跑可用于调试代码，但不能与正式 campaign 的 GPU-hour、HBM 和 teacher-transfer 系统指标合并。

## 6. 外部服务

W&B 默认开启。正式运行前在容器中确认：

```bash
python -c 'import wandb; assert wandb.api.api_key'
```

只有最终 LiveCodeBench 能力评测需要 SandboxFusion：

```bash
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
curl -fsS "${SANDBOXFUSION_BASE_URL}/docs" >/dev/null
```

## 7. Preflight

```bash
source local/mopd.env
bash examples/mopd_gpas/run_stage.sh preflight
```

它会生成本机绝对路径的 train/eval manifest，验证四个已发布 teacher，并解析 warm、八条主训练、三个 frozen bank 和八项能力评测的真实参数。Preflight 不占 GPU；通过后再认领正式任务。

常见失败含义：

- `Missing ...`: `local/mopd.env` 的资产路径不正确或下载未完成；
- `Teacher architecture differs`: student 与 teacher 不是同一 Qwen3-1.7B 结构；
- `GPU ... requires`: 当前机器不是正式协议硬件；
- W&B authentication 错误：先登录，或仅在本地调试时设置 `USE_WANDB=0`。
