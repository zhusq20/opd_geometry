# Four-task micro-batch MOPD / GPAS

This directory implements the protocol-v4 Qwen3-1.7B experiment described in the adjacent paper repository. Each run uses one 96 GB training GPU and one 48 GB inference GPU. The inference GPU hosts the student rollout engine and four persistent teacher servers; code and science share the weights of the locally downloaded `Qwen/Qwen3-4B` but use distinct endpoints.

The frozen eight configurations are `uniform`, `gpas`, `cost_gpas`, `raw_noise`, `loss_gap`, `std_mopd`, `d3_mopd`, and `open_mopd`. Every run uses seed 42, 500 optimizer steps, 16 task micro-batches per step, four prompts per micro-batch, and exactly 32,000 attempted responses.

`d3_mopd` implements the scheduler from [D³-MOPD](https://arxiv.org/abs/2608.24987) with the paper's Table 3 values. `open_mopd` implements the token-share and forward gap-following rules from [Open-MOPD](https://arxiv.org/abs/2608.19098). It is explicitly the shared protocol's K=1 sampled-token adaptation: reward refresh is an identity at K=1, and this launcher does not claim to reproduce the paper's separate K=4, dense student-top-k=16 system.

The complete paper system is available as a separate, pinned reproduction lane under [`open_mopd_full`](open_mopd_full/README_zh.md). It uses the released SmolLM3-3B student and three teachers, the official data, K=4, dense student-top-k=16 rewards, and reward refresh. It is reported as an external paper-protocol reference rather than mixed into the controlled Qwen3 four-task table.

See the Chinese runbooks:

- [Setup](docs/SETUP_zh.md)
- [Protocol and commands](docs/EXPERIMENTS_zh.md)
- [Parallel campaign coordination](docs/COLLABORATION_zh.md)

LiveCodeBench capability evaluation additionally requires the audited cgroup-v2 SandboxFusion service. Build and attest it with `examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh` and `start_sandboxfusion.sh` immediately before `run_stage.sh capability`; training, smoke tests, and held-out teacher-loss evaluation do not use this service. See the [setup runbook](docs/SETUP_zh.md#livecodebench-capability-sandbox).

Quick start from the repository root:

```bash
cp examples/mopd_gpas/configs/site.example.env local/mopd.env
source local/mopd.env
bash examples/mopd_gpas/run_stage.sh fetch-assets
bash examples/mopd_gpas/run_stage.sh prepare-heldout
bash examples/mopd_gpas/run_stage.sh convert-teachers
bash examples/mopd_gpas/run_stage.sh measure-initial
bash examples/mopd_gpas/run_stage.sh prepare
bash examples/mopd_gpas/run_stage.sh preflight
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_stage.sh smoke
bash examples/mopd_gpas/run_stage.sh train uniform
bash examples/mopd_gpas/run_stage.sh baselines
bash examples/mopd_gpas/run_stage.sh variance all
bash examples/mopd_gpas/run_stage.sh capability all
bash examples/mopd_gpas/run_stage.sh analyze
```

Every 50-step boundary keeps an archived Hugging Face weight checkpoint. Full optimizer state is retained only for the latest resume frontier, plus Uniform steps 50, 250, and 500 used by the scalar-only held-out gradient-variance probe.
