"""Shared model/data identities for the current MOPD paper experiments."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_RUNS = {
    "s-pg": ("sampled_reverse_kl", "domain_response", True),
    "s-tk": ("student_topk", "domain_response", True),
    "m-pg": ("sampled_reverse_kl", "domain_response", False),
    "m-tk-dr": ("student_topk", "domain_response", False),
    "m-tk-dt": ("student_topk", "domain_token", False),
    "m-tk-gt": ("student_topk", "global_token", False),
    "s-tk64": ("student_topk", "domain_response", True),
    "m-tk64-dr": ("student_topk", "domain_response", False),
    "m-tk64-dt": ("student_topk", "domain_token", False),
    "m-tk64-gt": ("student_topk", "global_token", False),
    "m-intersection64-dr": ("topk_intersection", "domain_response", False),
}
PAPER_TOPK = {name: 64 if "64" in name else 16 for name, (loss, _, _) in PAPER_RUNS.items()
              if loss in {"student_topk", "topk_intersection"}}
PROFILES = {
    "qwen3": {
        "tasks": ("math", "code", "if", "science"),
        "student_repo": "Qwen/Qwen3-1.7B-Base",
        "student_revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "model_config": "qwen3-1.7B.sh",
        "data_repo": "zsqzz/mopd-gpas-64k-data",
        "data_revision": "e3ead6f98b7089e0def39516cd06cf711b10b1ec",
        "teacher_repo": "zsqzz/mopd-gpas-64k-models",
        "teacher_revision": "46c307ad13f274d8db7c1df949e4df8359ae6b32",
        "chat_kwargs": {"enable_thinking": False},
        "student_suffix_to_remove": "<think>\n\n</think>\n\n",
        "teacher_suffix": "<think>\n\n</think>\n\n",
    },
    "smollm3": {
        "tasks": ("math", "code", "if"),
        "student_repo": "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT",
        "student_revision": "c9e7bad031667828656ead188d7e8ea162c048a4",
        "base_repo": "HuggingFaceTB/SmolLM3-3B-Base",
        "base_revision": "d78a42f79198603e614095753484a04c10c2b940",
        "model_config": "smollm3-3B.sh",
        "data_repo": "BytedTsinghua-SIA/Open-MOPD-Data",
        "data_revision": "9e897efe3257599d4300e2d5ee865a1cc714af87",
        "template_repo": "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT",
        "template_revision": "c9e7bad031667828656ead188d7e8ea162c048a4",
        "teacher_repos": {
            task: f"BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-{suffix}"
            for task, suffix in (("math", "Math"), ("code", "Code"), ("if", "IF"))
        },
        "teacher_revisions": {
            "math": "5e901bb626b69d711074d2832c41ce1aa4232da8",
            "code": "e2ce9beec52381350edcdaefebfd08ea67c21b42",
            "if": "8948051a805883e2db988cae4522de3c29d1d112",
        },
        "chat_kwargs": {"enable_thinking": True},
        "student_suffix_to_remove": None,
        "teacher_suffix": "",
    },
}


def profile(name: str | None = None) -> dict:
    name = name or os.environ.get("MOPD_PROFILE", "qwen3")
    spec = dict(PROFILES[name])
    spec["name"] = name
    directory_name = "smollm3_mixsft" if name == "smollm3" else name
    assets = Path(os.environ.get("MOPD_ASSET_ROOT", ROOT / "local" / f"mopd_{directory_name}_assets"))
    spec.update({
        "asset_root": assets,
        "student_hf": Path(os.environ.get("MOPD_HF_CHECKPOINT", assets / "models/student")),
        "student_megatron": Path(os.environ.get("MOPD_BASE_MEGATRON", assets / "models/student_torch_dist")),
        "teacher_root": Path(os.environ.get("MOPD_TEACHER_HF_ROOT", assets / "models/teachers_hf")),
        "data_root": Path(os.environ.get("MOPD_DATA_ROOT", assets / "data" / ("m2rl" if name == "qwen3" else ""))),
        "generated": Path(os.environ.get("MOPD_GENERATED_DIR", ROOT / "local" / f"mopd_{directory_name}_generated")),
        "output": Path(os.environ.get("MOPD_OUTPUT_ROOT", ROOT / "outputs" / f"mopd_{directory_name}")),
    })
    return spec
