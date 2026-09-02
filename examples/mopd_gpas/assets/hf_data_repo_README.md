---
license: other
---

# MOPD/GPAS 64k data assets

Frozen data inputs for the four-task Qwen3-1.7B MOPD/GPAS campaign in https://github.com/zhusq20/opd_geometry.

Repository layout:

- `m2rl/train/{math,code,if,science}.jsonl`
- `m2rl/eval/m2rl_online/`
- `m2rl/single_task/code/` with the frozen LiveCodeBench v6 online128 subset and index
- `frozen/controlled/` with the controlled optimizer/sampling inputs used by the final plotter

The bundle combines derived inputs from multiple upstream benchmarks. Consult the source datasets and the code repository for their individual terms. Use it for research reproduction of this campaign.
