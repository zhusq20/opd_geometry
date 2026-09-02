---
license: apache-2.0
---

# MOPD/GPAS 64k model assets

Model artifacts for the four-task Qwen3-1.7B MOPD/GPAS campaign in https://github.com/zhusq20/opd_geometry.

Repository layout:

- `student_hf/`: base Hugging Face checkpoint
- `student_megatron/`: base Megatron torch-dist checkpoint used to create the warm start
- `teachers_hf/{math,code,if,science}/`: converted single-task GRPO teachers
- `warm_start-seed42/`: shared eight-unit warm run used by every main trajectory

Use `examples/mopd_gpas/fetch_assets.sh` from the code repository instead of downloading individual files manually.
