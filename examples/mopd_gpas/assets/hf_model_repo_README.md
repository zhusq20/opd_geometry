---
license: apache-2.0
---

# MOPD/GPAS base model assets

Expected model asset layout:

- `qwen3-1.7b-base/`: Qwen3-1.7B-Base weights with the shared Qwen3 tokenizer
- `qwen3-1.7b-base_torch_dist/`: matching Megatron checkpoint
- `teachers_hf/{math,code,if,science}/`: four converted Qwen3-1.7B domain-RL teachers

`fetch_assets.sh` downloads the pinned upstream `Qwen/Qwen3-1.7B-Base` weights into `models/qwen3-1.7b-base/`; the bundle’s `student_hf/` and `student_megatron/` retain the original Qwen3-1.7B assets used to convert the RL teachers.
