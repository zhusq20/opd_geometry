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
from slime_plugins.mopd.prompting import EMPTY_THINKING_SUFFIX, PROMPT_FORMAT
from slime_plugins.mopd.sampler import CORE_RUNS

TASKS = ("math", "code", "if", "science")
RM_TYPES = {"math": "deepscaler", "code": "unit_test", "if": "ifevalg", "science": "gpqa"}
CONFIGS = tuple(CORE_RUNS)
TRAIN_CANDIDATE_SLICE = (0, 16_384)
HELDOUT_CANDIDATE_SLICE = (16_384, 16_575)
TRAIN_PROMPTS_PER_TASK = 16_000
HELDOUT_PROMPTS_PER_TASK = 64
STUDENT_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"


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
    """Freeze 64 reference prompts and a separate diagnostic pool per domain."""

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
            chat_template_suffix_to_remove=EMPTY_THINKING_SUFFIX,
            seed=42,
        )
        if len(dataset) < HELDOUT_PROMPTS_PER_TASK + 64:
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
        diagnostic_rows = []
        for sample in dataset.origin_samples[HELDOUT_PROMPTS_PER_TASK:]:
            metadata = dict(sample.metadata or {})
            metadata.update({"task_name": task, "teacher": task, "protocol_split": "diagnostic"})
            diagnostic_rows.append({"prompt": sample.prompt, "label": sample.label, "metadata": metadata})
        diagnostic_path = output / "diagnostic" / f"{task}.jsonl"
        atomic_text(diagnostic_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in diagnostic_rows))
        records[task]["diagnostic"] = {"selected": len(diagnostic_rows), "file": file_record(diagnostic_path)}
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


def hardware_spec(profile: str) -> dict[str, Any]:
    if profile == "frozen-96gb-tp1":
        return {
            "profile": profile,
            "gpus_per_run": 2,
            "training_gpu": "one 96GB GPU",
            "training_gpu_count": 1,
            "tensor_model_parallel_size": 1,
            "inference_gpu": "one 48GB GPU shared by student rollout and four resident teachers",
            "teacher_order": list(TASKS),
        }
    if profile == "dual-48gb-tp2":
        return {
            "profile": profile,
            "gpus_per_run": 3,
            "training_gpu": "two 48GB GPUs",
            "training_gpu_count": 2,
            "tensor_model_parallel_size": 2,
            "inference_gpu": "one dedicated 48GB student-rollout GPU",
            "teacher_service": "external frozen endpoints; service GPUs may be shared across runs",
            "teacher_order": list(TASKS),
        }
    raise ValueError(f"unknown MOPD hardware profile {profile!r}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=here.parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--teacher-hf-root", type=Path)
    parser.add_argument("--heldout-only", action="store_true")
    parser.add_argument(
        "--hardware-profile",
        choices=("frozen-96gb-tp1", "dual-48gb-tp2"),
        default=os.environ.get("MOPD_HARDWARE_PROFILE", "frozen-96gb-tp1"),
    )
    args = parser.parse_args()

    root = args.repo.resolve()
    output = (args.output or Path(os.environ.get("MOPD_GENERATED_DIR", root / "local/mopd_no_think_generated"))).resolve()
    existing_protocol = output / "protocol.json"
    if existing_protocol.is_file():
        previous = json.loads(existing_protocol.read_text(encoding="utf-8"))
        if previous.get("schema_version") != 6 or previous.get("prompt_format") != PROMPT_FORMAT:
            raise ValueError(f"{output} contains a different prompt protocol; select a new output directory")
    elif any((output / name).exists() for name in ("train.yaml", "diagnostic.yaml", "teacher_loss_eval.yaml")):
        raise ValueError(f"{output} contains manifests without a prompt protocol; select a new output directory")
    data_root = (args.data_root or Path(os.environ.get("MOPD_DATA_ROOT", root / "data/m2rl"))).resolve()
    student_hf = Path(os.environ.get("MOPD_HF_CHECKPOINT", "/workspace/dev/checkpoints/Qwen3-1.7B-Base")).resolve()
    base_megatron = Path(
        os.environ.get("MOPD_BASE_MEGATRON", "/workspace/dev/checkpoints/Qwen3-1.7B-Base_torch_dist")
    ).resolve()
    teacher_hf_root = (
        args.teacher_hf_root
        or Path(os.environ.get("MOPD_TEACHER_HF_ROOT", root / "local/mopd_assets/models/teachers_hf"))
    ).resolve()

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
        print(f"Wrote 64 reference prompts and a separate diagnostic pool per task to {output}")
        return

    weights = dict.fromkeys(TASKS, 0.25)

    teacher_config = expanded_yaml(
        here / "configs/teacher_checkpoints.yaml",
        {
            "SLIME_ROOT": str(root),
            "MOPD_TEACHER_BASE_HF": os.environ.get("MOPD_TEACHER_BASE_HF", "/workspace/dev/checkpoints/Qwen3-1.7B"),
            "MOPD_TEACHER_HF_ROOT": str(teacher_hf_root),
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
            "temporary_substitute": False,
            "intended_teacher": f"Qwen3-1.7B {task} RL teacher",
            "config": file_record(model_path / "config.json"),
            "weight_files": [file_record(path) for path in sorted(model_path.glob("*.safetensors"))],
        }
        for name in ("tokenizer.json", "tokenizer_config.json"):
            if sha256(student_hf / name) != sha256(model_path / name):
                raise ValueError(f"{task} RL teacher {name} differs from the student tokenizer")

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
                "chat_template_suffix_to_remove": EMPTY_THINKING_SUFFIX,
                "rm_type": RM_TYPES[task],
                "teacher": task,
                "target_weight": weights[task],
                "required_samples": TRAIN_PROMPTS_PER_TASK,
                "shuffle_seed": 42 + index * 100_003,
                "metadata": {"task_name": task, "teacher": task, "protocol_split": "train"},
            }
        )
    dump_yaml(
        output / "train.yaml",
        {
            "version": 4,
            "protocol": "qwen3_1.7b_4t_microbatch_mopd_gpas",
            "prompt_format": PROMPT_FORMAT,
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
                    "chat_template_suffix_to_remove": None,
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
        output / "diagnostic.yaml",
        {
            "version": 3,
            "protocol": "qwen3_1.7b_base_4t_common_checkpoint",
            "prompt_format": PROMPT_FORMAT,
            "sampling": {"seed": 42, "repeat": False},
            "sources": [
                {
                    "name": task,
                    "path": str((output / "diagnostic" / f"{task}.jsonl").resolve()),
                    "input_key": "prompt",
                    "label_key": "label",
                    "metadata_key": "metadata",
                    "apply_chat_template": False,
                    "chat_template_suffix_to_remove": None,
                    "teacher": task,
                    "target_weight": weights[task],
                }
                for task in TASKS
            ],
        },
    )

    protocol = {
        "schema_version": 6,
        "prompt_format": PROMPT_FORMAT,
        "name": "Qwen3-1.7B-Base-4T-microbatch-MOPD-GPAS-500-step",
        "seed": 42,
        "student": {
            "model": "Qwen3-1.7B-Base",
            "thinking": False,
            "hf_config": file_record(student_hf / "config.json"),
            "tokenizer": file_record(student_hf / "tokenizer.json"),
            "tokenizer_config": file_record(student_hf / "tokenizer_config.json"),
            "megatron_frontier": file_record(base_megatron / "latest_checkpointed_iteration.txt"),
        },
        "teachers": teachers,
        "student_base": {
            "repository": "Qwen/Qwen3-1.7B-Base",
            "revision": os.environ.get("MOPD_STUDENT_REVISION", STUDENT_REVISION),
            "path": str(student_hf),
        },
        "datasets": datasets,
        "train_candidate_slice": list(TRAIN_CANDIDATE_SLICE),
        "training_prompts_per_task": TRAIN_PROMPTS_PER_TASK,
        "heldout": heldout,
        "objective": {
            "weights": weights,
            "loss": "teacher_top64_corrected_reverse_kl",
            "support": "teacher top-64; p and q use full-vocabulary normalization; differentiate -p",
            "aggregation": "per-token, then per-response, then per-microbatch; sum_i w_i mean_s loss_i,s",
            "paper_baselines": {
                "d3_fixed": {
                    "label": "D³ signal, fixed weights",
                    "implementation_version": "d3-table3-synchronous-v1-integer-projection",
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
                    "protocol_projection": "minimize squared distance to 16*p over the 149 feasible counts",
                },
            },
        },
        "training": {
            "configs": list(CONFIGS),
            "allocations": CORE_RUNS,
            "optimizer": "conventional AdamW",
            "steps": 500,
            "microbatches_per_step": 16,
            "prompts_per_microbatch": 4,
            "responses_per_prompt": 1,
            "responses_per_step": 64,
            "attempted_response_budget": 32_000,
            "microbatch_bounds": [2, 8],
            "checkpoint_steps": [100, 200, 300, 400, 500],
            "additional_checkpoints": {"uniform-s1": [250], "gpas-s1": [250]},
            "retained_full_state": {"uniform-s1": [250], "all": "latest resume checkpoint"},
            "noise_estimator": "Welford, unweighted unclipped microbatch gradients; pre-step bias-corrected D",
            "allocation": "149-vector exact enumeration; ties nearest Uniform then task order",
            "response_cap_tokens": 4_096,
            "ema_decay": 0.9,
            "generation": {"temperature": 1.0, "top_p": 1.0, "thinking": False},
        },
        "system": hardware_spec(args.hardware_profile),
        "evaluation": {
            "steps": list(range(0, 501, 100)),
            "responses_per_task": 64,
            "reference_bank": str(output / "reference_bank.pt"),
            "fresh_policy_steps": [0, 500],
            "capability_steps": {
                "initial_student": [0],
                "uniform-s1": [250, 500],
                "gpas-s1": [250, 500],
                "gpas-raw-s1": [500],
                "d3-fixed-s1": [500],
            },
            "paired_bootstrap_replicates": 1_000,
        },
        "common_checkpoint": {
            "run": "uniform-s1",
            "step": 250,
            "branches": ["uniform", "gpas"],
            "calibration_microbatches_per_task": 16,
            "trials_per_branch": 10,
            "evaluation_prompts_per_task": 64,
            "prompts_per_microbatch": 4,
            "generated_responses": 1792,
            "student_scoring_forwards_after_updates": 5120,
        },
    }
    atomic_text(output / "protocol.json", json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(f"MOPD two-week core protocol v6 written to {output}; no initial-KL measurement is required")


if __name__ == "__main__":
    main()
