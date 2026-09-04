---
license: other
---

# MOPD/GPAS data assets

Expected layout:

- `m2rl/train/{math,code,if,science}.jsonl`
- `m2rl/eval/m2rl_online/`
- `m2rl/single_task/code/livecodebench_v6_online128.parquet`
- the corresponding immutable evaluation index files

The preparation script reserves disjoint train and held-out candidate slices, renders all Qwen3 prompts with `enable_thinking=false`, and writes the exact local manifests used by the run.
