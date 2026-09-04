#!/usr/bin/env python3
"""Freeze data manifests and objective weights for the revised four-task protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer

TASKS = ("math", "code", "if", "science")
RM_TYPES = {"math": "deepscaler", "code": "unit_test", "if": "ifevalg", "science": "gpqa"}
CONFIGS = (
    "uniform",
    "gpas",
    "cost_gpas",
    "raw_noise",
    "loss_gap",
    "std_mopd",
    "d3_mopd",
    "open_mopd",
)
TRAIN_CANDIDATE_SLICE = (0, 16_384)
HELDOUT_CANDIDATE_SLICE = (16_384, 16_575)
TRAIN_PROMPTS_PER_TASK = 16_000
HELDOUT_PROMPTS_PER_TASK = 128
QWEN3_4B_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def dump_yaml(path: Path, value: Any) -> None:
    atomic_text(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def sliced(path: Path, bounds: tuple[int, int]) -> str:
    return f"{path.resolve()}@[{bounds[0]}:{bounds[1]}]"


def expanded_yaml(path: Path, replacements: dict[str, str]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    for name, value in replacements.items():
        text = text.replace("${" + name + "}", value)
    return yaml.safe_load(os.path.expandvars(text))


def materialize_heldout(data_root: Path, output: Path, student_hf: Path) -> dict[str, Any]:
    """Render and length-filter the frozen held-out pool, then retain exactly 128/task."""

    tokenizer = load_tokenizer(str(student_hf), trust_remote_code=True)
    processor = load_processor(str(student_hf), trust_remote_code=True)
    records: dict[str, Any] = {}
    for task in TASKS:
        source = data_root / "train" / f"{task}.jsonl"
        dataset = Dataset(
            sliced(source, HELDOUT_CANDIDATE_SLICE),
            tokenizer=tokenizer,
            processor=processor,
            max_length=2_048,
            prompt_key="prompt",
            label_key="label",
            metadata_key="metadata",
            tool_key="tools",
            apply_chat_template=True,
            apply_chat_template_kwargs={"enable_thinking": False},
            seed=42,
        )
        if len(dataset) < HELDOUT_PROMPTS_PER_TASK:
            raise ValueError(f"{task} has only {len(dataset)} usable held-out prompts")
        rows = []
        for sample in dataset.origin_samples[:HELDOUT_PROMPTS_PER_TASK]:
            metadata = dict(sample.metadata or {})
            metadata.update({"task_name": task, "teacher": task, "protocol_split": "heldout"})
            rows.append({"prompt": sample.prompt, "label": sample.label, "metadata": metadata})
        target = output / "heldout" / f"{task}.jsonl"
        atomic_text(target, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        records[task] = {
            "candidate_slice": list(HELDOUT_CANDIDATE_SLICE),
            "usable_candidates": len(dataset),
            "selected": len(rows),
            "file": file_record(target),
        }
    return records


def objective_from_measurement(path: Path) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("schema_version", -1)) != 2 or tuple(value.get("task_order") or ()) != TASKS:
        raise ValueError(f"{path} is not a version-2 initial-KL measurement for {TASKS}")
    losses = {task: float(value["tasks"][task]["ell0"]) for task in TASKS}
    ratio = max(losses.values()) / min(losses.values())
    if ratio > 10:
        weights = {task: 0.25 for task in TASKS}
        rule = "equal_due_to_max_min_ratio_gt_10"
    else:
        inverse = {task: 1.0 / losses[task] for task in TASKS}
        total = sum(inverse.values())
        weights = {task: inverse[task] / total for task in TASKS}
        rule = "inverse_initial_teacher_loss"
    measured_rule = str(value.get("weight_rule"))
    measured_weights = {task: float(value["target_weights"][task]) for task in TASKS}
    if measured_rule != rule or any(abs(measured_weights[task] - weights[task]) > 1e-12 for task in TASKS):
        raise ValueError("initial-KL file does not apply the protocol's >10 equal-weight rule")
    return losses, weights, value


def complete_hf(path: Path) -> bool:
    index_path = path / "model.safetensors.index.json"
    if not all((path / name).is_file() for name in ("config.json", "tokenizer.json", "tokenizer_config.json")):
        return False
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return all((path / shard).is_file() for shard in set(index["weight_map"].values()))
    return any(path.glob("*.safetensors"))


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=here.parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--teacher-hf-root", type=Path)
    parser.add_argument("--qwen3-4b", type=Path)
    parser.add_argument("--initial-kl", type=Path)
    parser.add_argument("--heldout-only", action="store_true")
    args = parser.parse_args()

    root = args.repo.resolve()
    output = (args.output or Path(os.environ.get("MOPD_GENERATED_DIR", root / "local/mopd_generated"))).resolve()
    data_root = (args.data_root or Path(os.environ.get("MOPD_DATA_ROOT", root / "data/m2rl"))).resolve()
    student_hf = Path(os.environ.get("MOPD_HF_CHECKPOINT", "/workspace/dev/checkpoints/Qwen3-1.7B")).resolve()
    base_megatron = Path(
        os.environ.get("MOPD_BASE_MEGATRON", "/workspace/dev/checkpoints/Qwen3-1.7B_torch_dist")
    ).resolve()
    teacher_hf_root = (
        args.teacher_hf_root
        or Path(os.environ.get("MOPD_TEACHER_HF_ROOT", root / "local/mopd_assets/models/teachers_hf"))
    ).resolve()
    qwen3_4b = (
        args.qwen3_4b or Path(os.environ.get("MOPD_QWEN3_4B", root / "local/mopd_assets/models/qwen3-4b"))
    ).resolve()
    initial_path = (args.initial_kl or Path(os.environ.get("MOPD_INITIAL_KL", output / "initial_kl.json"))).resolve()

    for path in (student_hf / "config.json", base_megatron / "latest_checkpointed_iteration.txt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    datasets: dict[str, Any] = {}
    for task in TASKS:
        path = data_root / "train" / f"{task}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            record_count = sum(1 for _ in stream)
        if record_count < HELDOUT_CANDIDATE_SLICE[1]:
            raise ValueError(f"{path} has only {record_count} records")
        datasets[task] = {**file_record(path), "records": record_count}

    heldout = materialize_heldout(data_root, output, student_hf)
    if args.heldout_only:
        print(f"Wrote exactly 128 rendered held-out prompts per task to {output / 'heldout'}")
        return

    if not initial_path.is_file():
        raise FileNotFoundError(
            f"Missing measured initial KL: {initial_path}. Run measure_initial_kl.py after --heldout-only."
        )
    initial_losses, weights, initial = objective_from_measurement(initial_path)

    teacher_config = expanded_yaml(
        here / "configs/teacher_checkpoints.yaml",
        {
            "SLIME_ROOT": str(root),
            "MOPD_HF_CHECKPOINT": str(student_hf),
            "MOPD_BASE_MEGATRON": str(base_megatron),
            "MOPD_TEACHER_HF_ROOT": str(teacher_hf_root),
            "MOPD_QWEN3_4B": str(qwen3_4b),
        },
    )
    teachers: dict[str, Any] = {}
    for task in TASKS:
        item = teacher_config["teachers"][task]
        model_path = Path(item["model_path"])
        if not complete_hf(model_path):
            raise FileNotFoundError(f"incomplete {task} teacher under {model_path}")
        teachers[task] = {
            "kind": item["kind"],
            "model_path": str(model_path.resolve()),
            "revision": item.get("revision"),
            "checkpoint_step": item.get("step"),
            "config": file_record(model_path / "config.json"),
        }
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha256(student_hf / name) != sha256(qwen3_4b / name):
            raise ValueError(f"Qwen3-4B {name} differs from the student tokenizer")

    sources = []
    for index, task in enumerate(TASKS):
        sources.append(
            {
                "name": task,
                "path": sliced(data_root / "train" / f"{task}.jsonl", TRAIN_CANDIDATE_SLICE),
                "input_key": "prompt",
                "label_key": "label",
                "metadata_key": "metadata",
                "tool_key": "tools",
                "apply_chat_template": True,
                "apply_chat_template_kwargs": {"enable_thinking": False},
                "rm_type": RM_TYPES[task],
                "teacher": task,
                "target_weight": weights[task],
                "required_samples": TRAIN_PROMPTS_PER_TASK,
                "shuffle_seed": 42 + index * 100_003,
                "initial_teacher_loss": initial_losses[task],
                "metadata": {"task_name": task, "teacher": task, "protocol_split": "train"},
            }
        )
    dump_yaml(
        output / "train.yaml",
        {
            "version": 3,
            "protocol": "qwen3_1.7b_4t_microbatch_mopd_gpas",
            "sampling": {"seed": 42, "repeat": False},
            "sources": sources,
        },
    )
    dump_yaml(
        output / "teacher_loss_eval.yaml",
        {
            "eval": {
                "defaults": {
                    "apply_chat_template": False,
                    "custom_rm_path": "slime_plugins.mopd.eval.generation_only_reward",
                    "n_samples_per_eval_prompt": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_response_len": 4_096,
                    "min_eval_samples": HELDOUT_PROMPTS_PER_TASK,
                },
                "datasets": [
                    {
                        "name": task,
                        "path": str((output / "heldout" / f"{task}.jsonl").resolve()),
                        "input_key": "prompt",
                        "label_key": "label",
                        "metadata_key": "metadata",
                    }
                    for task in TASKS
                ],
            }
        },
    )
    dump_yaml(
        output / "heldout_variance.yaml",
        {
            "version": 3,
            "protocol": "qwen3_1.7b_4t_heldout_gradient_variance",
            "sampling": {"seed": 42, "repeat": False},
            "sources": [
                {
                    "name": task,
                    "path": str((output / "heldout" / f"{task}.jsonl").resolve()),
                    "input_key": "prompt",
                    "label_key": "label",
                    "metadata_key": "metadata",
                    "apply_chat_template": False,
                    "teacher": task,
                    "target_weight": weights[task],
                    "initial_teacher_loss": initial_losses[task],
                    "required_samples": HELDOUT_PROMPTS_PER_TASK,
                }
                for task in TASKS
            ],
        },
    )

    protocol = {
        "schema_version": 4,
        "name": "Qwen3-1.7B-4T-microbatch-MOPD-GPAS-500-step",
        "seed": 42,
        "student": {
            "model": "Qwen3-1.7B",
            "thinking": False,
            "hf_config": file_record(student_hf / "config.json"),
            "megatron_frontier": file_record(base_megatron / "latest_checkpointed_iteration.txt"),
        },
        "teachers": teachers,
        "qwen3_4b": {
            "repository": "Qwen/Qwen3-4B",
            "revision": QWEN3_4B_REVISION,
            "path": str(qwen3_4b),
        },
        "datasets": datasets,
        "train_candidate_slice": list(TRAIN_CANDIDATE_SLICE),
        "training_prompts_per_task": TRAIN_PROMPTS_PER_TASK,
        "heldout": heldout,
        "initial_kl": {**initial, "file": file_record(initial_path)},
        "objective": {
            "weights": weights,
            "aggregation": "per-token, then per-response, then per-microbatch; sum_i w_i mean_s loss_i,s",
            "std_mopd_aggregation": "exact valid-token mean over the complete step",
            "paper_baselines": {
                "d3_mopd": {
                    "paper": "https://arxiv.org/abs/2608.24987",
                    "scheduler": {
                        "update_cadence": 10,
                        "window": 10,
                        "windows": 3,
                        "initial_steps": 5,
                        "ema_window": 10,
                        "kl_denominator_floor": 0.15,
                        "temperature": 0.5,
                        "mixture_floor": 0.10,
                        "jitter": 0.30,
                    },
                    "protocol_projection": "paper probabilities projected to the shared [2,8] microbatch bounds",
                },
                "open_mopd": {
                    "paper": "https://arxiv.org/abs/2608.19098",
                    "variant": "K=1 sampled-token in-protocol adaptation",
                    "share_target": dict.fromkeys(TASKS, 0.25),
                    "gap_alpha": 1.0,
                    "gap_factor_bounds": [0.05, 20.0],
                    "reward_refresh": "identity_at_K=1",
                },
            },
        },
        "training": {
            "configs": list(CONFIGS),
            "optimizer": "conventional AdamW",
            "steps": 500,
            "microbatches_per_step": 16,
            "prompts_per_microbatch": 4,
            "responses_per_prompt": 1,
            "responses_per_step": 64,
            "attempted_response_budget": 32_000,
            "microbatch_bounds": [2, 8],
            "checkpoint_steps": list(range(50, 501, 50)),
            "response_cap_tokens": 4_096,
            "ema_decay": 0.9,
            "generation": {"temperature": 1.0, "top_p": 1.0, "thinking": False},
        },
        "system": {
            "gpus_per_run": 2,
            "training_gpu": "one 96GB GPU",
            "inference_gpu": "one 48GB GPU shared by student rollout and four resident teachers",
            "teacher_order": list(TASKS),
        },
        "evaluation": {
            "steps": list(range(0, 501, 50)),
            "responses_per_task": 128,
            "paired_bootstrap_replicates": 1_000,
        },
        "heldout_gradient_variance": {
            "checkpoint_steps": [50, 250, 500],
            "microbatches_per_task": 32,
            "prompts_per_microbatch": 4,
            "stored_values": "scalar raw and AdamW-scaled norms only",
        },
    }
    atomic_text(output / "protocol.json", json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(f"MOPD protocol v4 written to {output}")


if __name__ == "__main__":
    main()
