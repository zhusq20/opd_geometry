#!/usr/bin/env python3
"""Freeze manifests for the Qwen3-1.7B four-task exact-set MOPD protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

TASKS = ("math", "code", "if", "science")
RM_TYPES = {"math": "deepscaler", "code": "unit_test", "if": "ifevalg", "science": "gpqa"}
TRAIN_USABLE_PROMPTS = {"math": 15_999, "code": 15_964, "if": 15_993, "science": 15_998}
SPLITS = {
    "train": (0, 16_000),
    "probe": (16_000, 16_064),
    "bank": (16_064, 16_192),
    "teacher_loss": (16_384, 16_575),
}
CONFIGS = (
    ("uniform_k1_conventional", 1, "uniform", "conventional"),
    ("uniform_k1_taskwise", 1, "uniform", "taskwise"),
    ("gpas_k1_taskwise", 1, "gpas", "taskwise"),
    ("cost_gpas_k1_taskwise", 1, "cost_gpas", "taskwise"),
    ("uniform_k2_taskwise", 2, "uniform", "taskwise"),
    ("cost_gpas_k2_taskwise", 2, "cost_gpas", "taskwise"),
    ("all_k4_taskwise", 4, "all", "taskwise"),
    ("all_k4_conventional", 4, "all", "conventional"),
)
MAIN_RESPONSE_BUDGET = 64_000
MAIN_CHECKPOINTS = (16_384, 32_768, 64_000)
MAIN_EVALUATIONS = (2_048, 4_096, 8_192, 16_384, 32_768, 49_152, 64_000)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def dump_yaml(path: Path, value: Any) -> None:
    atomic_text(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def expand_yaml(path: Path, replacements: dict[str, str]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    for name, value in replacements.items():
        text = text.replace("${" + name + "}", value)
    return yaml.safe_load(text)


def sliced(path: Path, split: str) -> str:
    start, stop = SPLITS[split]
    return f"{path.resolve()}@[{start}:{stop}]"


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=here.parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--teacher-hf-root", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    output = (
        args.output if args.output is not None else Path(os.environ.get("MOPD_GENERATED_DIR", here / "generated/mopd"))
    ).resolve()
    data_root = (
        args.data_root if args.data_root is not None else Path(os.environ.get("MOPD_DATA_ROOT", root / "data/m2rl"))
    ).resolve()
    teacher_hf_root = (
        args.teacher_hf_root
        if args.teacher_hf_root is not None
        else Path(os.environ.get("MOPD_TEACHER_HF_ROOT", here / "generated/teachers_hf"))
    ).resolve()

    initial_path = here / "configs/initial_teacher_losses.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    if (
        initial["source"].get("event") != "update=0"
        or list(initial["source"].get("heldout_slice") or ()) != list(SPLITS["teacher_loss"])
        or int(initial["source"].get("generation_seed", -1)) != 42
    ):
        raise ValueError("the initial-loss source does not match the frozen update-0 held-out protocol")
    for task in TASKS:
        if task not in initial["values"]:
            raise ValueError(f"the frozen initial loss is missing task {task}")

    datasets: dict[str, Any] = {}
    for task in TASKS:
        path = data_root / "train" / f"{task}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            records = sum(1 for _ in stream)
        if records < SPLITS["teacher_loss"][1]:
            raise ValueError(f"{path} has only {records} records")
        datasets[task] = {
            **file_record(path),
            "records": records,
            "usable_train_prompts": TRAIN_USABLE_PROMPTS[task],
        }

    checkpoint_config = expand_yaml(
        here / "configs/teacher_checkpoints.yaml",
        {
            "SLIME_ROOT": str(root),
            "MOPD_HF_CHECKPOINT": os.environ.get("MOPD_HF_CHECKPOINT", "/workspace/dev/checkpoints/Qwen3-1.7B"),
            "MOPD_BASE_MEGATRON": os.environ.get(
                "MOPD_BASE_MEGATRON", "/workspace/dev/checkpoints/Qwen3-1.7B_torch_dist"
            ),
            "MOPD_TEACHER_HF_ROOT": str(teacher_hf_root),
        },
    )
    base_hf = Path(checkpoint_config["base_hf"])
    base_megatron = Path(checkpoint_config["base_megatron"])
    if not (base_hf / "config.json").is_file():
        raise FileNotFoundError(base_hf / "config.json")
    if not (base_megatron / "latest_checkpointed_iteration.txt").is_file():
        raise FileNotFoundError(base_megatron / "latest_checkpointed_iteration.txt")

    teachers: dict[str, Any] = {}
    for task in TASKS:
        config = checkpoint_config["teachers"][task]
        step = int(config["step"])
        converted = Path(config["output_hf"])
        if not (converted / "config.json").is_file() or not list(converted.glob("*.safetensors")):
            raise FileNotFoundError(f"converted {task} teacher is missing under {converted}")
        teachers[task] = {
            "source": "published converted single-task GRPO teacher",
            "step": step,
            "converted_hf": str(converted.resolve()),
            "converted_config": file_record(converted / "config.json"),
            "conversion_manifest": file_record(converted / "conversion_manifest.json"),
        }

    sources = []
    for index, task in enumerate(TASKS):
        data_path = data_root / "train" / f"{task}.jsonl"
        loss0 = float(initial["values"][task])
        sources.append(
            {
                "name": task,
                "path": sliced(data_path, "train"),
                "probe_path": sliced(data_path, "probe"),
                "bank_path": sliced(data_path, "bank"),
                "input_key": "prompt",
                "label_key": "label",
                "metadata_key": "metadata",
                "tool_key": "tools",
                "apply_chat_template": True,
                "apply_chat_template_kwargs": {"enable_thinking": False},
                "rm_type": RM_TYPES[task],
                "teacher": task,
                "target_weight": 0.25,
                "required_samples": TRAIN_USABLE_PROMPTS[task],
                "shuffle_seed": args.seed + index * 100_003,
                "initial_teacher_loss": loss0,
                "relative_loss_scale": 1.0 / loss0,
                "metadata": {"task_name": task, "teacher": task, "protocol_split": "train"},
            }
        )
    dump_yaml(
        output / "train.yaml",
        {
            "version": 2,
            "protocol": "qwen3_1.7b_4t_exact_set_mopd_gpas",
            "sampling": {"seed": args.seed, "repeat": True},
            "sources": sources,
        },
    )
    dump_yaml(
        output / "teacher_loss_eval.yaml",
        {
            "eval": {
                "defaults": {
                    "apply_chat_template": True,
                    "apply_chat_template_kwargs": {"enable_thinking": False},
                    "custom_rm_path": "slime_plugins.mopd.eval.generation_only_reward",
                    "n_samples_per_eval_prompt": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_response_len": 8192,
                },
                "datasets": [
                    {
                        "name": task,
                        "path": sliced(data_root / "train" / f"{task}.jsonl", "teacher_loss"),
                        "input_key": "prompt",
                        "label_key": "label",
                        "metadata_key": "metadata",
                        "tool_key": "tools",
                        "metadata_overrides": {
                            "task_name": task,
                            "teacher": task,
                            "protocol_split": "teacher_loss",
                        },
                    }
                    for task in TASKS
                ],
            }
        },
    )

    protocol = {
        "schema_version": 3,
        "name": "Qwen3-1.7B-4T-exact-set-MOPD-GPAS-64K",
        "campaign_role": "single_seed_primary",
        "seed": args.seed,
        "student": {
            "hf": file_record(base_hf / "config.json"),
            "megatron": file_record(base_megatron / "latest_checkpointed_iteration.txt"),
        },
        "teachers": teachers,
        "datasets": datasets,
        "splits": {name: list(bounds) for name, bounds in SPLITS.items()},
        "initial_teacher_losses": {**initial, "config": file_record(initial_path)},
        "objective": {
            "task": "relative_loss_i = teacher_loss_i / initial_teacher_loss_i",
            "aggregate": "mean over math, code, if, science",
            "teacher_loss": "valid-token mean per response, then four responses per prompt, then 16 prompts",
            "common_threshold": 0.75,
        },
        "task_unit": {"prompts": 16, "responses_per_prompt": 4, "attempted_responses": 64},
        "online_probe": {"prompts": 2, "attempted_responses": 8, "full_unit_noise_correction": True},
        "warm_start": {"order": list(TASKS), "units_per_task": 2, "total_units": 8},
        "training": {
            "configs": [
                {"id": name, "K": width, "allocation": allocation, "adamw_state": state}
                for name, width, allocation, state in CONFIGS
            ],
            "response_budget": MAIN_RESPONSE_BUDGET,
            "response_checkpoints": list(MAIN_CHECKPOINTS),
            "response_cap_tokens": 8192,
            "failure_penalty": 10.0,
            "ema_decay": 0.95,
            "inclusion_floor": 0.05,
            "score_max_age_task_units": 50,
            "importance_correction": "target_weight / final inclusion marginal",
            "exact_set_distribution": (
                "maximum entropy for GPAS; resident-aware direct set optimization over "
                "rollout-overlapped transfer-tail critical path for Cost-GPAS"
            ),
            "adamw": {
                "learning_rate": 2.5e-7,
                "beta1": 0.9,
                "beta2": 0.9987381276,
                "epsilon": 1e-8,
                "weight_decay": 0.0,
                "task_set_decay": "beta^K and (1-lr*wd)^K",
                "shared_pre_correction_clip": 1.0,
            },
            "generation": {
                "temperature": 1.0,
                "top_p": 1.0,
                "thinking": False,
                "seed": args.seed,
            },
        },
        "frozen_bank": {
            "trajectory": "uniform_k1_taskwise",
            "checkpoints": ["warm", "middle", "late"],
            "units_per_task_per_checkpoint": 8,
            "cross_fit": "4/4 swap",
            "K_values": [1, 2, 4],
            "K2_sets": ["math+code", "math+if", "math+science", "code+if", "code+science", "if+science"],
        },
        "system": {
            "student_gpus_per_run": 1,
            "teacher_slots_per_run": 1,
            "parallel_run_pairs": 2,
            "total_campaign_gpus": 4,
            "teacher_order": "resident first when selected, then fixed task ID; final teacher remains resident",
            "dtype": "bfloat16 model, FP32 Adam moments",
            "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "gpu_memory_mib": 97_887,
        },
        "evaluation": {
            "teacher_loss_response_milestones": list(MAIN_EVALUATIONS),
            "paired_generation_seed": 42,
            "benchmarks": [
                "MATH-500 greedy pass@1",
                "LiveCodeBench v6 online128 pass@1",
                "IFBench strict",
                "GPQA-Diamond average@4",
            ],
        },
    }
    atomic_text(output / "protocol.json", json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(f"MOPD protocol v3 (single seed {args.seed}) written to {output}")


if __name__ == "__main__":
    main()
