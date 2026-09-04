---
license: apache-2.0
---

# MOPD/GPAS base model assets

Expected bundle layout:

- `student_hf/`: Qwen3-1.7B Hugging Face checkpoint
- `student_megatron/`: matching Megatron checkpoint
- `teachers_hf/math/` and `teachers_hf/if/`: converted domain-RL teachers

Code and science do not use converted bundle teachers. `fetch_assets.sh` downloads the pinned upstream `Qwen/Qwen3-4B` revision into `models/qwen3-4b/` and verifies it separately.
