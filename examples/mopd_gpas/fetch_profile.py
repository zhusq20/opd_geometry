#!/usr/bin/env python3
"""Fetch pinned student weights, domain RL teachers, and the matching paper data."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
if __package__:
    from .profiles import profile
else:
    from profiles import profile

MODEL_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"]


def fetch(repo, revision, target, *, dataset=False, patterns=None):
    return Path(snapshot_download(repo, revision=revision, repo_type="dataset" if dataset else "model",
                                  local_dir=target, allow_patterns=patterns or MODEL_FILES))


def install_tokenizer(student: Path, template: Path):
    """Keep base parameters but adopt the RL family's exact chat serialization."""
    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained(student)
    target = AutoTokenizer.from_pretrained(template)
    if base.get_vocab() != target.get_vocab():
        raise ValueError("Base and RL tokenizer token IDs differ; cannot share teacher token scores")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
                 "merges.txt", "vocab.json", "generation_config.json"):
        if (template / name).is_file():
            shutil.copy2(template / name, student / name)
    # The raw base EOS is end-of-text; chat responses must stop at the RL
    # family's end-of-message token in both HF and SGLang generation.
    config_path = student / "config.json"
    config = json.loads(config_path.read_text())
    config.update(eos_token_id=target.eos_token_id, pad_token_id=target.pad_token_id,
                  bos_token_id=target.bos_token_id)
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main():
    p = profile()
    repo = os.environ.get("MOPD_STUDENT_REPO", p["student_repo"])
    revision = os.environ.get("MOPD_STUDENT_REVISION") or (
        p["student_revision"] if repo == p["student_repo"] else HfApi().model_info(repo).sha)
    identity_path = p["asset_root"] / "assets.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text()).get("student")
        if previous != {"repository": repo, "revision": revision}:
            raise ValueError("Student identity changed; use a new MOPD_ASSET_ROOT and unset old model/data path overrides. "
                             "Do not overwrite Base assets or resume their runs as MixSFT.")
    fetch(repo, revision, p["student_hf"])
    if p["name"] == "smollm3":
        if repo != p["template_repo"]:
            template = fetch(p["template_repo"], p["template_revision"], p["asset_root"] / "models/template",
                             patterns=["*.json", "*.jinja"])
            install_tokenizer(p["student_hf"], template)
        teachers = {}
        for task in p["tasks"]:
            fetch(p["teacher_repos"][task], p["teacher_revisions"][task], p["teacher_root"] / task)
            teachers[task] = {"repository": p["teacher_repos"][task], "revision": p["teacher_revisions"][task]}
        fetch(p["data_repo"], p["data_revision"], p["data_root"], dataset=True,
              patterns=["rl_prompt_mix/*", "eval/**/*.parquet", "README.md"])
    else:
        fetch(p["teacher_repo"], p["teacher_revision"], p["asset_root"] / "models",
              patterns=["teachers_hf/**", "student_hf/**"])
        install_tokenizer(p["student_hf"], p["asset_root"] / "models/student_hf")
        fetch(p["data_repo"], p["data_revision"], p["data_root"].parent, dataset=True, patterns=["m2rl/**"])
        teachers = {task: {"repository": p["teacher_repo"], "revision": p["teacher_revision"]}
                    for task in p["tasks"]}
    identity = {"profile": p["name"], "student": {"repository": repo, "revision": revision},
                "teachers": teachers, "data": {"repository": p["data_repo"], "revision": p["data_revision"]}}
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    print(f"Fetched {p['name']} assets to {p['asset_root']}")


if __name__ == "__main__":
    main()
