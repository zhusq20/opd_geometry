#!/usr/bin/env python3
"""Build reproducible math/code/science single-task manifests from prepared M2RL data.

The input is the manifest emitted by ``prepare_m2rl_data.py``.  For each
selected task this script writes an on-policy manifest, an optional SFT+OPD
manifest, and (unless explicitly disabled) an evaluation config. A supplied
benchmark file is preferred. Without one, ``tail_view`` makes a disjoint
smoke-test holdout without copying the full training corpus; ``seeded_copy``
retains the older exact-random split mode. For benchmarks that preserve the
raw question in ``metadata.problem``, ``--exclude-eval-overlap`` can also
materialize a benchmark-disjoint training view.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml


GENERALIZED_PATH = re.compile(r"^(.*?)(@\[-?\d*:-?\d*\])?$")


def parse_assignments(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected TASK=PATH, got {value!r}.")
        task, path = value.split("=", 1)
        task = task.strip()
        if not task or not path:
            raise ValueError(f"Expected non-empty TASK=PATH, got {value!r}.")
        result[task] = path
    return result


def parse_positive_int_assignments(values: list[str]) -> dict[str, int]:
    parsed = parse_assignments(values)
    result: dict[str, int] = {}
    for task, value in parsed.items():
        try:
            number = int(value)
        except ValueError as exc:
            raise ValueError(f"Expected TASK=POSITIVE_INT, got {task}={value!r}.") from exc
        if number <= 0:
            raise ValueError(f"Expected TASK=POSITIVE_INT, got {task}={value!r}.")
        result[task] = number
    return result


def parse_float_assignments(values: list[str], option: str) -> dict[str, float]:
    parsed = parse_assignments(values)
    result: dict[str, float] = {}
    for task, value in parsed.items():
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"Expected TASK=FLOAT for {option}, got {task}={value!r}.") from exc
        if not math.isfinite(number):
            raise ValueError(f"Expected a finite TASK=FLOAT for {option}, got {task}={value!r}.")
        result[task] = number
    return result


def resolve_generalized_path(value: str, base: Path) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    match = GENERALIZED_PATH.fullmatch(expanded)
    assert match is not None
    real_path = Path(match.group(1))
    if not real_path.is_absolute():
        real_path = base / real_path
    return f"{real_path.resolve()}{match.group(2) or ''}"


def real_path(value: str) -> Path:
    match = GENERALIZED_PATH.fullmatch(value)
    assert match is not None
    return Path(match.group(1))


def count_rows(value: str) -> int | None:
    """Return a cheap row count for the JSONL/Parquet formats used here."""

    path = real_path(value)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    if suffix == ".parquet":
        import pyarrow.parquet as parquet

        return parquet.ParquetFile(path).metadata.num_rows
    return None


def overlap_fingerprint(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).lower()
    return "".join(character for character in normalized if character.isalnum())


def benchmark_problem_fingerprints(path: Path) -> dict[str, list[str]]:
    """Load raw benchmark problems used to make a training view disjoint."""

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    elif path.suffix.lower() == ".parquet":
        import pyarrow.parquet as parquet

        rows = parquet.read_table(path, columns=["metadata"]).to_pylist()
    else:
        raise ValueError("--exclude-eval-overlap requires a JSONL or Parquet evaluation dataset.")

    fingerprints: dict[str, list[str]] = {}
    for row_index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        problem = metadata.get("problem") if isinstance(metadata, dict) else None
        if not problem:
            raise ValueError(
                f"--exclude-eval-overlap requires metadata.problem on every row; missing at {path} row {row_index}."
            )
        fingerprint = overlap_fingerprint(problem)
        if not fingerprint:
            raise ValueError(f"Evaluation problem at {path} row {row_index} normalizes to an empty string.")
        eval_id = str(metadata.get("unique_id") or row_index)
        fingerprints.setdefault(fingerprint, []).append(eval_id)
    if not fingerprints:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return fingerprints


def filter_eval_overlap_jsonl(
    input_path: Path,
    output_path: Path,
    benchmark_path: Path,
    *,
    input_key: str,
) -> tuple[int, list[dict[str, Any]]]:
    """Materialize a training JSONL view with benchmark problems removed."""

    if input_path.suffix.lower() != ".jsonl":
        raise ValueError("--exclude-eval-overlap currently supports prepared JSONL training sources only.")
    fingerprints = benchmark_problem_fingerprints(benchmark_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    kept = 0
    removed: list[dict[str, Any]] = []
    completed = False
    try:
        with input_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
            for row_index, line in enumerate(source):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {input_path}:{row_index + 1}: {exc}") from exc
                prompt = row.get(input_key)
                if isinstance(prompt, list):
                    prompt_text = "\n".join(
                        str(message.get("content", "")) if isinstance(message, dict) else str(message)
                        for message in prompt
                    )
                else:
                    prompt_text = str(prompt or "")
                normalized_prompt = overlap_fingerprint(prompt_text)
                matching_ids = [
                    eval_id
                    for fingerprint, eval_ids in fingerprints.items()
                    if fingerprint in normalized_prompt
                    for eval_id in eval_ids
                ]
                if matching_ids:
                    metadata = row.get("metadata") or {}
                    removed.append(
                        {
                            "train_row": row_index,
                            "eval_ids": matching_ids,
                            "original_dataset": metadata.get("original_dataset"),
                            "original_index": metadata.get("original_index"),
                        }
                    )
                    continue
                output.write(line if line.endswith("\n") else line + "\n")
                kept += 1
        if kept == 0:
            raise ValueError(f"All training rows in {input_path} overlap {benchmark_path}.")
        temporary.replace(output_path)
        completed = True
    finally:
        if not completed:
            temporary.unlink(missing_ok=True)
    return kept, removed


def split_jsonl(input_path: Path, train_path: Path, eval_path: Path, count: int, seed: int) -> tuple[int, int]:
    if input_path.suffix != ".jsonl":
        raise ValueError("Automatic holdout currently supports prepared JSONL sources only. Pass --eval TASK=PATH for Parquet or benchmark evaluation data.")
    if count <= 0:
        raise ValueError("--holdout-count must be positive when no external eval file is supplied.")
    with input_path.open(encoding="utf-8") as stream:
        total = sum(1 for line in stream if line.strip())
    if total < 2:
        raise ValueError(f"Cannot split {input_path}: it has only {total} non-empty row(s).")
    eval_count = min(count, total - 1)
    selected = set(random.Random(seed).sample(range(total), eval_count))

    train_path.parent.mkdir(parents=True, exist_ok=True)
    row_index = 0
    with (
        input_path.open(encoding="utf-8") as source,
        train_path.open("w", encoding="utf-8") as train_stream,
        eval_path.open("w", encoding="utf-8") as eval_stream,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}: {exc}") from exc
            output = eval_stream if row_index in selected else train_stream
            output.write(line if line.endswith("\n") else line + "\n")
            row_index += 1
    return total - eval_count, eval_count


def tail_holdout_jsonl(input_path: Path, eval_path: Path, count: int) -> tuple[int, int]:
    """Write only the final holdout rows; training remains a slice of the source."""

    if input_path.suffix != ".jsonl":
        raise ValueError("Automatic holdout currently supports prepared JSONL sources only. Pass --eval TASK=PATH for Parquet or benchmark evaluation data.")
    if count <= 0:
        raise ValueError("--holdout-count must be positive when no external eval file is supplied.")
    total = 0
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}: {exc}") from exc
            total += 1
    if total < 2:
        raise ValueError(f"Cannot split {input_path}: it has only {total} non-empty row(s).")

    eval_count = min(count, total - 1)
    train_count = total - eval_count
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = eval_path.with_name(f".{eval_path.name}.tmp.{os.getpid()}")
    row_index = 0
    completed = False
    try:
        with input_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
            for line in source:
                if not line.strip():
                    continue
                if row_index >= train_count:
                    output.write(line if line.endswith("\n") else line + "\n")
                row_index += 1
        completed = True
    finally:
        if not completed:
            temporary.unlink(missing_ok=True)
    temporary.replace(eval_path)
    return train_count, eval_count


def find_source(manifest: dict[str, Any], task: str) -> dict[str, Any]:
    matches = []
    for source in manifest.get("sources") or []:
        metadata = source.get("metadata") or {}
        if source.get("name") == task or metadata.get("task_name") == task:
            matches.append(source)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source for task {task!r}; found {len(matches)}.")
    return dict(matches[0])


def on_policy_source(source: dict[str, Any], task: str, path: str) -> dict[str, Any]:
    result = dict(source)
    result["name"] = f"{task}_opd"
    result["path"] = path
    result["weight"] = 1.0
    metadata = dict(result.get("metadata") or {})
    metadata.update({"task_name": task, "training_mode": "opd"})
    result["metadata"] = metadata
    return result


def sft_source(task: str, path: str, rm_type: str, ratio: float) -> dict[str, Any]:
    return {
        "name": f"{task}_sft",
        "path": path,
        "input_key": "messages",
        "metadata_key": "metadata",
        "tool_key": "tools",
        "apply_chat_template": False,
        "rm_type": rm_type,
        "weight": ratio,
        "metadata": {"task_name": task, "training_mode": "sft", "rm_type": rm_type},
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.rl_manifest.resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream) or {}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), list):
        raise ValueError("--rl-manifest must contain a sources list.")

    sft_paths = parse_assignments(args.sft)
    eval_paths = parse_assignments(args.eval)
    eval_names = parse_assignments(getattr(args, "eval_name", []))
    eval_rm_types = parse_assignments(getattr(args, "eval_rm_type", []))
    eval_samples = parse_positive_int_assignments(getattr(args, "eval_samples", []))
    eval_max_response_lengths = parse_positive_int_assignments(
        getattr(args, "eval_max_response_len_override", [])
    )
    eval_temperatures = parse_float_assignments(getattr(args, "eval_temperature", []), "--eval-temperature")
    eval_top_p_overrides = parse_float_assignments(
        getattr(args, "eval_top_p_override", []), "--eval-top-p-override"
    )
    invalid_temperatures = {task: value for task, value in eval_temperatures.items() if value < 0}
    if invalid_temperatures:
        raise ValueError(f"--eval-temperature values must be non-negative: {invalid_temperatures}.")
    invalid_top_p = {task: value for task, value in eval_top_p_overrides.items() if not 0 < value <= 1}
    if invalid_top_p:
        raise ValueError(f"--eval-top-p-override values must be in (0, 1]: {invalid_top_p}.")
    exclude_eval_overlap = set(getattr(args, "exclude_eval_overlap", []))
    skip_eval = set(getattr(args, "skip_eval", []))
    configured_tasks = (
        set(sft_paths)
        | set(eval_paths)
        | set(eval_names)
        | set(eval_rm_types)
        | set(eval_samples)
        | set(eval_max_response_lengths)
        | set(eval_temperatures)
        | set(eval_top_p_overrides)
        | exclude_eval_overlap
    )
    unknown = (configured_tasks | skip_eval) - set(args.tasks)
    if unknown:
        raise ValueError(f"Assignments were supplied for unselected tasks: {sorted(unknown)}.")
    conflicts = skip_eval & set(eval_paths)
    if conflicts:
        raise ValueError(f"Tasks cannot use both --eval and --skip-eval: {sorted(conflicts)}.")
    eval_metadata_without_data = (
        set(eval_names)
        | set(eval_rm_types)
        | set(eval_samples)
        | set(eval_max_response_lengths)
        | set(eval_temperatures)
        | set(eval_top_p_overrides)
        | exclude_eval_overlap
    ) - set(eval_paths)
    if eval_metadata_without_data:
        raise ValueError(
            "Evaluation overrides require a matching --eval for tasks: "
            f"{sorted(eval_metadata_without_data)}."
        )
    if not 0 < args.sft_ratio < 1:
        raise ValueError("--sft-ratio must be strictly between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdout_mode = getattr(args, "holdout_mode", "seeded_copy")
    summary: dict[str, Any] = {"seed": args.seed, "holdout_mode": holdout_mode, "tasks": {}}
    for task_index, task in enumerate(args.tasks):
        original = find_source(manifest, task)
        original_path = resolve_generalized_path(str(original["path"]), manifest_path.parent)
        if not real_path(original_path).exists():
            raise FileNotFoundError(f"RL source for {task} does not exist: {original_path}")

        task_dir = args.output_dir / task
        external_eval = eval_paths.get(task)
        excluded_overlap_rows: list[dict[str, Any]] = []
        eval_value: str | None
        if task in skip_eval:
            train_value = original_path
            train_count = count_rows(original_path)
            eval_value = None
            eval_count = 0
            eval_kind = "disabled"
        elif external_eval is None:
            eval_path = task_dir / f"{task}_holdout.jsonl"
            if holdout_mode == "seeded_copy":
                train_path = task_dir / f"{task}_train.jsonl"
                train_count, eval_count = split_jsonl(
                    real_path(original_path),
                    train_path,
                    eval_path,
                    args.holdout_count,
                    args.seed + task_index * 100_003,
                )
                train_value = str(train_path.resolve())
                eval_kind = "in_distribution_seeded_holdout"
            elif holdout_mode == "tail_view":
                match = GENERALIZED_PATH.fullmatch(original_path)
                assert match is not None
                if match.group(2):
                    raise ValueError("tail_view does not accept an already-sliced RL source; use seeded_copy.")
                train_count, eval_count = tail_holdout_jsonl(real_path(original_path), eval_path, args.holdout_count)
                train_value = f"{original_path}@[0:{train_count}]"
                eval_kind = "in_distribution_tail_holdout"
            else:
                raise ValueError(f"Unknown holdout mode: {holdout_mode!r}.")
            eval_value = str(eval_path.resolve())
        else:
            eval_value = resolve_generalized_path(external_eval, Path.cwd())
            if not real_path(eval_value).exists():
                raise FileNotFoundError(f"Evaluation source for {task} does not exist: {eval_value}")
            if task in exclude_eval_overlap:
                if "@[" in original_path or "@[" in eval_value:
                    raise ValueError("--exclude-eval-overlap does not accept sliced training or evaluation paths.")
                filtered_train_path = task_dir / f"{task}_train_eval_disjoint.jsonl"
                train_count, excluded_overlap_rows = filter_eval_overlap_jsonl(
                    real_path(original_path),
                    filtered_train_path,
                    real_path(eval_value),
                    input_key=str(original.get("input_key") or "prompt"),
                )
                train_value = str(filtered_train_path.resolve())
            else:
                train_value = original_path
                train_count = count_rows(original_path)
            eval_count = count_rows(eval_value)
            eval_kind = "external_benchmark"

        opd_source = on_policy_source(original, task, train_value)
        on_policy_manifest = {
            "version": 1,
            "sampling": {"strategy": "weighted", "unit": "batch", "seed": args.seed, "repeat": True},
            "sources": [opd_source],
        }
        on_policy_path = task_dir / f"{task}_on_policy.yaml"
        write_yaml(on_policy_path, on_policy_manifest)

        rm_type = str(original.get("rm_type") or (original.get("metadata") or {}).get("rm_type") or "")
        eval_config_path = task_dir / f"{task}_eval.yaml"
        eval_name: str | None = None
        eval_rm_type: str | None = None
        eval_samples_per_prompt: int | None = None
        if eval_value is not None:
            eval_name = eval_names.get(task, task)
            eval_rm_type = eval_rm_types.get(task, rm_type)
            eval_samples_per_prompt = eval_samples.get(task, args.eval_samples_per_prompt)
            eval_top_p = eval_top_p_overrides.get(task, getattr(args, "eval_top_p", 0.7))
            eval_defaults = {
                "max_response_len": eval_max_response_lengths.get(task, args.eval_max_response_len),
                "top_p": eval_top_p,
                "n_samples_per_eval_prompt": eval_samples_per_prompt,
                "apply_chat_template": True,
                "custom_rm_path": "slime_plugins.m2rl.rewards.reward",
            }
            if task in eval_temperatures:
                eval_defaults["temperature"] = eval_temperatures[task]
            eval_config = {
                "eval": {
                    "defaults": eval_defaults,
                    "datasets": [
                        {
                            "name": eval_name,
                            "path": eval_value,
                            "rm_type": eval_rm_type,
                        }
                    ],
                }
            }
            write_yaml(eval_config_path, eval_config)
        else:
            # Do not let a stale holdout config be mistaken for a benchmark.
            eval_config_path.unlink(missing_ok=True)

        hybrid_path = None
        if task in sft_paths:
            resolved_sft = resolve_generalized_path(sft_paths[task], Path.cwd())
            if not real_path(resolved_sft).exists():
                raise FileNotFoundError(f"SFT source for {task} does not exist: {resolved_sft}")
            if args.sft_max_samples > 0 and "@[" not in resolved_sft:
                resolved_sft = f"{resolved_sft}@[0:{args.sft_max_samples}]"
            hybrid_opd_source = dict(opd_source)
            hybrid_opd_source["weight"] = 1.0 - args.sft_ratio
            hybrid_manifest = {
                "version": 1,
                "sampling": {
                    "strategy": "stratified",
                    "unit": "prompt",
                    "seed": args.seed,
                    "repeat": True,
                },
                "sources": [
                    hybrid_opd_source,
                    sft_source(task, resolved_sft, rm_type, args.sft_ratio),
                ],
            }
            hybrid_path = task_dir / f"{task}_sft_opd.yaml"
            write_yaml(hybrid_path, hybrid_manifest)

        summary["tasks"][task] = {
            "on_policy_manifest": str(on_policy_path),
            "sft_opd_manifest": str(hybrid_path) if hybrid_path else None,
            "eval_config": str(eval_config_path) if eval_value is not None else None,
            "eval_kind": eval_kind,
            "eval_name": eval_name,
            "eval_rm_type": eval_rm_type,
            "eval_samples_per_prompt": eval_samples_per_prompt,
            "train_rows": train_count,
            "eval_rows": eval_count,
        }
        if task in exclude_eval_overlap:
            summary["tasks"][task]["excluded_eval_overlap_rows"] = len(excluded_overlap_rows)
            summary["tasks"][task]["excluded_eval_overlaps"] = excluded_overlap_rows

    summary_path = args.output_dir / "single_task_index.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["math", "code", "science"])
    parser.add_argument(
        "--sft",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Processed messages-format SFT file; may be repeated.",
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Held-out benchmark in Slime format; may be repeated. Without it a seeded JSONL holdout is made.",
    )
    parser.add_argument(
        "--skip-eval",
        action="append",
        default=[],
        metavar="TASK",
        help="Keep the complete training source and omit online evaluation for this task; may be repeated.",
    )
    parser.add_argument(
        "--eval-name",
        action="append",
        default=[],
        metavar="TASK=NAME",
        help="Metric/dataset name for an external --eval source; may be repeated.",
    )
    parser.add_argument(
        "--eval-rm-type",
        action="append",
        default=[],
        metavar="TASK=RM_TYPE",
        help="Reward-model type for an external --eval source; may be repeated.",
    )
    parser.add_argument(
        "--eval-samples",
        action="append",
        default=[],
        metavar="TASK=COUNT",
        help="Samples per prompt for an external --eval source; may be repeated.",
    )
    parser.add_argument(
        "--eval-temperature",
        action="append",
        default=[],
        metavar="TASK=FLOAT",
        help="Sampling temperature override for an external --eval source; may be repeated.",
    )
    parser.add_argument(
        "--eval-max-response-len-override",
        action="append",
        default=[],
        metavar="TASK=COUNT",
        help="Maximum response length override for an external --eval source; may be repeated.",
    )
    parser.add_argument(
        "--eval-top-p-override",
        action="append",
        default=[],
        metavar="TASK=FLOAT",
        help="Top-p override for an external --eval source; may be repeated.",
    )
    parser.add_argument(
        "--exclude-eval-overlap",
        action="append",
        default=[],
        metavar="TASK",
        help=(
            "Materialize a training JSONL view with rows containing external benchmark metadata.problem "
            "removed; may be repeated."
        ),
    )
    parser.add_argument("--holdout-count", type=int, default=256)
    parser.add_argument(
        "--holdout-mode",
        choices=["tail_view", "seeded_copy"],
        default="tail_view",
        help=("tail_view stores only the final holdout rows and references the original training file by slice; seeded_copy writes a fully shuffled/disjoint train copy."),
    )
    parser.add_argument("--sft-ratio", type=float, default=0.5)
    parser.add_argument("--sft-max-samples", type=int, default=100_000)
    parser.add_argument("--eval-max-response-len", type=int, default=16384)
    parser.add_argument("--eval-samples-per-prompt", type=int, default=1)
    parser.add_argument("--eval-top-p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    result = prepare(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
