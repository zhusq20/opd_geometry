#!/usr/bin/env python3
"""Build a frozen, equal-volume, within-batch four-task GRPO manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

TASK_SPECS = (
    ("code", "code", "unit_test"),
    ("math", "math", "deepscaler"),
    ("science", "qa", "gpqa"),
    ("if", "if", "ifevalg"),
)
SEQUENTIAL_TASKS = ("math", "science", "if")
EVAL_CONFIG_FILES = {
    "code": "code/code_eval.yaml",
    "math": "math/math_eval_aime24_math500.yaml",
    "science": "science/science_eval.yaml",
    "if": "if/if_eval.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(os.path.expandvars(stream.read()))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return value


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def resolve_data_path(value: str, manifest_path: Path) -> Path:
    unsliced = value.split("@[", 1)[0]
    path = Path(os.path.expandvars(os.path.expanduser(unsliced)))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Data file referenced by {manifest_path} does not exist: {path}")
    return path


def command_value(command: list[Any], flag: str) -> str:
    if flag not in command:
        raise ValueError(f"Code-origin command is missing {flag}.")
    index = command.index(flag) + 1
    if index >= len(command):
        raise ValueError(f"Code-origin command has no value after {flag}.")
    return str(command[index])


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / ".metadata").is_file() and (path / "common.pt").is_file():
        return path
    marker = path / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError(f"Invalid torch-dist checkpoint: {path}")
    iteration = int(marker.read_text(encoding="utf-8").strip())
    resolved = path / f"iter_{iteration:07d}"
    if not (resolved / ".metadata").is_file() or not (resolved / "common.pt").is_file():
        raise FileNotFoundError(f"Incomplete latest torch-dist checkpoint: {resolved}")
    return resolved


def one_source(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_yaml(manifest_path)
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise ValueError(f"Expected exactly one source in {manifest_path}.")
    return manifest, dict(sources[0])


def validate_provenance_input(provenance: dict[str, Any], path: Path) -> None:
    resolved = path.resolve()
    record = next(
        (
            entry
            for entry in provenance.get("inputs", [])
            if isinstance(entry, dict) and Path(str(entry.get("path", ""))).resolve() == resolved
        ),
        None,
    )
    if record is None:
        raise ValueError(f"Code-origin provenance did not archive required input {resolved}.")
    current_sha256 = sha256(resolved)
    if record.get("sha256") != current_sha256:
        raise ValueError(
            f"Required code-origin input changed after training: {resolved}; "
            f"expected {record.get('sha256')}, got {current_sha256}."
        )


def validate_code_origin(
    *,
    checkpoint: Path,
    provenance_path: Path,
    code_manifest_path: Path,
    model_config_path: Path,
    prompts_per_task: int,
    rollout_batch_size: int,
    group_size: int,
    seed: int,
) -> dict[str, Any]:
    checkpoint = resolve_checkpoint(checkpoint)
    match = re.fullmatch(r"iter_(\d+)", checkpoint.name)
    if match is None:
        raise ValueError(f"Code-origin checkpoint must be named iter_NNNNNNN: {checkpoint}")
    completed_updates = int(match.group(1)) + 1
    expected_code_prompts = completed_updates * rollout_batch_size
    if expected_code_prompts != prompts_per_task:
        raise ValueError(
            f"Code checkpoint {checkpoint.name} represents {expected_code_prompts} prompt groups "
            f"at rollout_batch_size={rollout_batch_size}, not the requested {prompts_per_task}."
        )

    provenance_path = provenance_path.expanduser().resolve()
    provenance = read_json(provenance_path)
    command = provenance.get("command")
    if not isinstance(command, list):
        raise ValueError(f"Code-origin provenance has no command list: {provenance_path}")

    expected_values = {
        "--rollout-batch-size": rollout_batch_size,
        "--n-samples-per-prompt": group_size,
        "--global-batch-size": rollout_batch_size * group_size,
        "--rollout-seed": seed,
        "--seed": seed,
        "--num-epoch": 1,
    }
    for flag, expected in expected_values.items():
        actual = int(command_value(command, flag))
        if actual != expected:
            raise ValueError(f"Code-origin {flag}={actual} does not match the mixed-run contract {expected}.")
    if command_value(command, "--advantage-estimator") != "grpo":
        raise ValueError("Code-origin checkpoint was not trained with GRPO.")
    if command_value(command, "--optimizer") != "adam":
        raise ValueError("Code-origin checkpoint was not trained with AdamW/Megatron Adam.")
    if float(command_value(command, "--lr")) != 1e-6 or float(command_value(command, "--weight-decay")) != 0.0:
        raise ValueError("Code-origin AdamW hyperparameters do not match LR=1e-6 and weight_decay=0.")
    if "--rollout-shuffle" not in command:
        raise ValueError("Code-origin command did not enable deterministic rollout shuffling.")
    template_kwargs = json.loads(command_value(command, "--apply-chat-template-kwargs"))
    if not isinstance(template_kwargs, dict) or template_kwargs.get("enable_thinking") is not False:
        raise ValueError("Code-origin command did not freeze Qwen thinking off.")

    provenance_manifest = Path(command_value(command, "--prompt-data")).expanduser().resolve()
    if provenance_manifest != code_manifest_path.resolve():
        raise ValueError(
            f"Code-origin prompt manifest {provenance_manifest} does not match {code_manifest_path.resolve()}."
        )
    validate_provenance_input(provenance, code_manifest_path)
    validate_provenance_input(provenance, model_config_path)

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_metadata_sha256": sha256(checkpoint / ".metadata"),
        "checkpoint_common_sha256": sha256(checkpoint / "common.pt"),
        "completed_updates": completed_updates,
        "provenance": str(provenance_path),
        "provenance_sha256": sha256(provenance_path),
        "hf_checkpoint": str(Path(command_value(command, "--hf-checkpoint")).expanduser().resolve()),
        "base_torch_dist_checkpoint": str(Path(command_value(command, "--load")).expanduser().resolve()),
        "code_manifest_sha256": sha256(code_manifest_path),
    }


def source_record(
    *,
    canonical_task: str,
    task_alias: str,
    expected_rm_type: str,
    source: dict[str, Any],
    source_manifest: Path,
    prompts_per_task: int,
    seed: int,
    prompt_fit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rm_type = str(source.get("rm_type") or "")
    if rm_type != expected_rm_type:
        raise ValueError(
            f"{canonical_task} source in {source_manifest} uses rm_type={rm_type!r}, "
            f"expected {expected_rm_type!r}."
        )
    data_path = resolve_data_path(str(source.get("path") or ""), source_manifest)
    mixed_source = dict(source)
    mixed_source.pop("phase_samples", None)
    mixed_source.update(
        {
            "name": task_alias,
            "weight": 1.0,
            "shuffle_seed": seed,
            "required_samples": prompts_per_task,
        }
    )
    metadata = dict(mixed_source.get("metadata") or {})
    metadata.update(
        {
            "task_name": task_alias,
            "canonical_task_name": canonical_task,
            "mixed_batch_training": True,
        }
    )
    mixed_source["metadata"] = metadata
    record = {
        "task": task_alias,
        "canonical_task": canonical_task,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256(source_manifest),
        "source_data": str(data_path),
        "source_data_bytes": data_path.stat().st_size,
        "source_data_mtime_ns": data_path.stat().st_mtime_ns,
        "path_view": str(mixed_source["path"]),
        "rm_type": rm_type,
        "prompts": prompts_per_task,
        "shuffle_seed": seed,
    }
    if prompt_fit is not None:
        record["prompt_fit"] = prompt_fit
    return mixed_source, record


def fit_usable_prompt_prefix(
    *,
    source: dict[str, Any],
    source_manifest: Path,
    task_record: dict[str, Any],
    prompts_per_task: int,
    max_prompt_len: int,
    tokenizer: Any,
    processor: Any,
    apply_chat_template_kwargs: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extend a frozen raw prefix only enough to retain the requested usable prompts."""

    from slime.utils.data import Dataset

    original_view = str(source.get("path") or "")
    match = re.fullmatch(r"(?P<path>.*)@\[(?P<start>-?\d*):(?P<stop>-?\d*)\]", original_view)
    if match is None:
        raise ValueError(f"Prompt fitting requires a bounded source prefix, got {original_view!r}.")
    start = int(match.group("start")) if match.group("start") else None
    stop = int(match.group("stop")) if match.group("stop") else None
    if start not in {None, 0} or stop is None or stop <= 0:
        raise ValueError(f"Prompt fitting requires a non-negative prefix slice starting at zero: {original_view!r}.")
    if stop != prompts_per_task:
        raise ValueError(
            f"Frozen source prefix contains {stop} raw rows, expected prompts_per_task={prompts_per_task}."
        )

    available_train_rows = int(task_record.get("available_train_rows", -1))
    if available_train_rows < stop:
        raise ValueError(
            f"Sequential source has only {available_train_rows} rows before its reserved probe tail, "
            f"below the frozen prefix of {stop}."
        )

    data_path = resolve_data_path(original_view, source_manifest)
    current_stop = stop
    usable_prompts = -1
    while usable_prompts < prompts_per_task:
        fitted_view = f"{data_path}@[0:{current_stop}]"
        dataset = Dataset(
            fitted_view,
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_prompt_len,
            prompt_key=str(source.get("input_key") or "prompt"),
            multimodal_keys=source.get("multimodal_keys"),
            label_key=str(source.get("label_key") or "label"),
            metadata_key=str(source.get("metadata_key") or "metadata"),
            tool_key=str(source.get("tool_key") or "tools"),
            apply_chat_template=bool(source.get("apply_chat_template", True)),
            apply_chat_template_kwargs=dict(
                source.get("apply_chat_template_kwargs") or apply_chat_template_kwargs
            ),
            seed=seed,
        )
        usable_prompts = len(dataset)
        if usable_prompts > prompts_per_task:
            raise RuntimeError(
                f"Prompt fitting overshot for {original_view}: {usable_prompts} usable prompts."
            )
        if usable_prompts == prompts_per_task:
            break
        current_stop += prompts_per_task - usable_prompts
        if current_stop > available_train_rows:
            raise ValueError(
                f"Cannot find {prompts_per_task} usable prompts before the sequential probe tail; "
                f"needed raw prefix {current_stop}, limit is {available_train_rows}."
            )

    fitted_source = dict(source)
    fitted_source["path"] = fitted_view
    prompt_fit = {
        "original_path_view": original_view,
        "fitted_path_view": fitted_view,
        "original_raw_prefix_rows": stop,
        "fitted_raw_prefix_rows": current_stop,
        "usable_prompts": usable_prompts,
        "excluded_by_prompt_filter": current_stop - usable_prompts,
        "max_prompt_len": max_prompt_len,
        "reserved_probe_tail_untouched": True,
    }
    print(
        f"Prompt-fit {source.get('name', source_manifest.stem)}: "
        f"raw_prefix={current_stop}, usable={usable_prompts}, "
        f"filtered={current_stop - usable_prompts}"
    )
    return fitted_source, prompt_fit


def final_eval_config(config_root: Path) -> dict[str, Any]:
    aliases = {canonical_task: task_alias for canonical_task, task_alias, _ in TASK_SPECS}
    datasets = []
    for canonical_task, _, _ in TASK_SPECS:
        config_path = (config_root / EVAL_CONFIG_FILES[canonical_task]).resolve()
        config = read_yaml(config_path)
        eval_config = config.get("eval")
        if not isinstance(eval_config, dict):
            raise ValueError(f"Evaluation config lacks `eval`: {config_path}")
        defaults = eval_config.get("defaults") or {}
        raw_datasets = eval_config.get("datasets")
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ValueError(f"Evaluation config has no datasets: {config_path}")
        for raw_dataset in raw_datasets:
            if not isinstance(raw_dataset, dict):
                raise ValueError(f"Evaluation dataset entries must be mappings: {config_path}")
            dataset = {**defaults, **raw_dataset}
            dataset["path"] = str(resolve_data_path(str(dataset.get("path") or ""), config_path))
            dataset.setdefault("top_k", -1)
            dataset["metadata_overrides"] = {
                **dict(dataset.get("metadata_overrides") or {}),
                "mixed_domain": aliases[canonical_task],
                "canonical_domain": canonical_task,
            }
            datasets.append(dataset)
    names = [str(dataset.get("name") or "") for dataset in datasets]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"Final-evaluation dataset names must be non-empty and unique: {names}")
    return {"eval": {"defaults": {}, "datasets": datasets}}


def prepare(args: argparse.Namespace) -> None:
    if args.rollout_batch_size % len(TASK_SPECS) != 0:
        raise ValueError(f"rollout_batch_size must be divisible by {len(TASK_SPECS)} tasks.")
    prompts_per_task_per_batch = args.rollout_batch_size // len(TASK_SPECS)
    if args.prompts_per_task % prompts_per_task_per_batch != 0:
        raise ValueError("prompts_per_task must be divisible by the per-task prompt count in each rollout batch.")

    sequential_index_path = args.sequential_data_index.expanduser().resolve()
    sequential_index = read_json(sequential_index_path)
    if sequential_index.get("seed") != args.seed:
        raise ValueError("Sequential data seed does not match the mixed-run seed.")
    if sequential_index.get("train_prompts_per_task") != args.prompts_per_task:
        raise ValueError("Sequential and mixed runs must use the same prompts_per_task.")
    if not set(SEQUENTIAL_TASKS).issubset(sequential_index.get("tasks", {})):
        raise ValueError(f"Sequential data index must contain {SEQUENTIAL_TASKS}.")

    code_manifest_path = args.code_manifest.expanduser().resolve()
    model_config_path = args.model_config.expanduser().resolve()
    if not code_manifest_path.is_file() or not model_config_path.is_file():
        raise FileNotFoundError("The code manifest and Qwen3-1.7B model config must both exist.")
    code_origin = validate_code_origin(
        checkpoint=args.code_checkpoint,
        provenance_path=args.code_origin_provenance,
        code_manifest_path=code_manifest_path,
        model_config_path=model_config_path,
        prompts_per_task=args.prompts_per_task,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        seed=args.seed,
    )

    _, code_source = one_source(code_manifest_path)
    source_by_task: dict[str, tuple[Path, dict[str, Any], dict[str, Any] | None]] = {
        "code": (code_manifest_path, code_source, None)
    }
    tokenizer = processor = None
    if args.max_prompt_len is not None:
        from slime.utils.processing_utils import load_processor, load_tokenizer

        tokenizer = load_tokenizer(code_origin["hf_checkpoint"], trust_remote_code=True)
        processor = load_processor(code_origin["hf_checkpoint"], trust_remote_code=True)
    for task in SEQUENTIAL_TASKS:
        task_record = sequential_index["tasks"][task]
        if int(task_record.get("train_rows", -1)) != args.prompts_per_task:
            raise ValueError(f"Sequential {task} train_rows does not match prompts_per_task.")
        train_manifest_path = Path(task_record["train_manifest"]).expanduser().resolve()
        if sha256(train_manifest_path) != task_record.get("train_manifest_sha256"):
            raise ValueError(f"Prepared sequential manifest changed after freezing: {train_manifest_path}")
        _, source = one_source(train_manifest_path)
        prompt_fit = None
        if args.max_prompt_len is not None:
            source, prompt_fit = fit_usable_prompt_prefix(
                source=source,
                source_manifest=train_manifest_path,
                task_record=task_record,
                prompts_per_task=args.prompts_per_task,
                max_prompt_len=args.max_prompt_len,
                tokenizer=tokenizer,
                processor=processor,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.seed,
            )
        source_by_task[task] = (train_manifest_path, source, prompt_fit)

    mixed_sources = []
    task_records = {}
    for canonical_task, task_alias, rm_type in TASK_SPECS:
        source_manifest, source, prompt_fit = source_by_task[canonical_task]
        mixed_source, record = source_record(
            canonical_task=canonical_task,
            task_alias=task_alias,
            expected_rm_type=rm_type,
            source=source,
            source_manifest=source_manifest,
            prompts_per_task=args.prompts_per_task,
            seed=args.seed,
            prompt_fit=prompt_fit,
        )
        mixed_sources.append(mixed_source)
        task_records[task_alias] = record

    rollout_batches = args.prompts_per_task // prompts_per_task_per_batch
    manifest = {
        "version": 1,
        "sampling": {
            "strategy": "stratified",
            "unit": "prompt",
            "seed": args.seed,
            "repeat": False,
        },
        "sources": mixed_sources,
    }
    manifest_text = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    manifest_path = args.output_dir.expanduser().resolve() / "mixed_on_policy.yaml"
    eval_text = yaml.safe_dump(
        final_eval_config(Path(sequential_index["config_root"]).expanduser().resolve()),
        sort_keys=False,
        allow_unicode=True,
    )
    eval_path = manifest_path.with_name("mixed_final_eval.yaml")
    index = {
        "schema_version": 2,
        "purpose": "Equal-volume within-rollout-batch Code/Math/QA/IF GRPO comparison",
        "manifest": str(manifest_path),
        "manifest_sha256": text_sha256(manifest_text),
        "final_eval_config": str(eval_path),
        "final_eval_config_sha256": text_sha256(eval_text),
        "sequential_data_index": str(sequential_index_path),
        "sequential_data_index_sha256": sha256(sequential_index_path),
        "code_origin": code_origin,
        "model": {
            "family": "Qwen3-1.7B",
            "model_config": str(model_config_path),
            "model_config_sha256": sha256(model_config_path),
            "hf_checkpoint": code_origin["hf_checkpoint"],
            "base_torch_dist_checkpoint": code_origin["base_torch_dist_checkpoint"],
            "enable_thinking": False,
        },
        "sampling": {
            "strategy": "stratified",
            "unit": "prompt",
            "seed": args.seed,
            "per_task_shuffle_seed": args.seed,
        },
        "batch_contract": {
            "tasks": [task_alias for _, task_alias, _ in TASK_SPECS],
            "rollout_batch_size_prompt_groups": args.rollout_batch_size,
            "prompt_groups_per_task_per_batch": prompts_per_task_per_batch,
            "grpo_responses_per_prompt": args.group_size,
            "trajectories_per_task_per_batch": prompts_per_task_per_batch * args.group_size,
            "global_batch_size_trajectories": args.rollout_batch_size * args.group_size,
            "rollout_batches_and_optimizer_updates": rollout_batches,
        },
        "prompts_per_task": args.prompts_per_task,
        "total_prompt_groups": args.prompts_per_task * len(TASK_SPECS),
        "tasks": task_records,
        "prompt_filter_contract": {
            "max_prompt_len": args.max_prompt_len,
            "apply_chat_template_kwargs": args.apply_chat_template_kwargs,
            "sequential_prefixes_fitted_to_usable_count": args.max_prompt_len is not None,
        },
        "comparison_notes": [
            "Code reuses the seed-matched stream consumed by the existing code-origin checkpoint.",
            "Math, QA (canonical source: science), and IF reuse the frozen sequential prefixes; a prefix is extended only when prompt-length filtering would otherwise reduce its usable count.",
            "Any prefix extension remains before the sequential gradient-probe tail and stops at exactly prompts_per_task usable prompts.",
            "Each GRPO prompt group still contains repeated responses to one identical prompt.",
        ],
    }
    index_text = json.dumps(index, indent=2, sort_keys=True) + "\n"
    index_path = manifest_path.with_name("mixed_data_index.json")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.iterdir())
    if existing and not args.force:
        if (
            manifest_path.is_file()
            and eval_path.is_file()
            and index_path.is_file()
            and manifest_path.read_text(encoding="utf-8") == manifest_text
            and eval_path.read_text(encoding="utf-8") == eval_text
            and index_path.read_text(encoding="utf-8") == index_text
        ):
            print(f"Prepared mixed data already exists: {index_path}")
            return
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --force to replace the plan.")

    atomic_text(manifest_path, manifest_text)
    atomic_text(eval_path, eval_text)
    atomic_text(index_path, index_text)
    print(f"Wrote equal-volume mixed GRPO manifest to {manifest_path}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-data-index", type=Path, required=True)
    parser.add_argument("--code-checkpoint", type=Path, required=True)
    parser.add_argument("--code-origin-provenance", type=Path, required=True)
    parser.add_argument(
        "--code-manifest",
        type=Path,
        default=root / "data/m2rl/single_task/code/code_on_policy.yaml",
    )
    parser.add_argument("--model-config", type=Path, default=root / "scripts/models/qwen3-1.7B.sh")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts-per-task", type=int, default=4800)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=None,
        help="Fit sequential raw prefixes to this post-template tokenizer limit before training.",
    )
    parser.add_argument(
        "--apply-chat-template-kwargs",
        default='{"enable_thinking":false}',
        help="JSON object used by the training-time chat template during prompt-prefix fitting.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("prompts_per_task", "rollout_batch_size", "group_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.max_prompt_len is not None and args.max_prompt_len <= 0:
        parser.error("--max-prompt-len must be positive.")
    try:
        args.apply_chat_template_kwargs = json.loads(args.apply_chat_template_kwargs)
    except (TypeError, ValueError) as error:
        parser.error(f"--apply-chat-template-kwargs must be valid JSON: {error}")
    if not isinstance(args.apply_chat_template_kwargs, dict):
        parser.error("--apply-chat-template-kwargs must decode to a JSON object.")
    return args


if __name__ == "__main__":
    prepare(parse_args())
