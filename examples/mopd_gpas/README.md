# Four-task exact-set MOPD / GPAS

This directory is the portable entry point for the Qwen3-1.7B, four-task, 64k-response MOPD/GPAS campaign. Eight seed-42 trajectories share one warm checkpoint and can then run independently on separate two-GPU workers.

Start with the Chinese runbooks:

- [Machine setup and Hugging Face assets](docs/SETUP_zh.md)
- [Experiment protocol and commands](docs/EXPERIMENTS_zh.md)
- [Multi-machine collaboration](docs/COLLABORATION_zh.md)
- [Per-machine configuration template](configs/site.example.env)
- [Campaign assignment template](configs/campaign.example.yaml)

Worker quick start:

```bash
mkdir -p local
cp examples/mopd_gpas/configs/site.example.env local/mopd.env
source local/mopd.env
bash examples/mopd_gpas/run_stage.sh fetch-assets
bash examples/mopd_gpas/run_stage.sh preflight
bash examples/mopd_gpas/run_stage.sh train "$CONFIG_ID"
bash examples/mopd_gpas/run_stage.sh capability "$CONFIG_ID"
bash examples/mopd_gpas/run_stage.sh package "$CONFIG_ID"
```
