#!/usr/bin/env python3
"""Freeze train/probe splits and evaluation data for sequential GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

TASKS = ("math", "code", "science", "if")
DEFAULT_EVAL_FILES = {
    "math": "math/math_eval_aime24_math500.yaml",
    "code": "code/code_eval.yaml",
    "science": "science/science_eval.yaml",
    "if": "if/if_eval.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(os.path.expandvars(stream.read()))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def resolve_data_path(value: str, manifest: Path) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Data file referenced by {manifest} does not exist: {path}")
    return path


def value_identity(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_identity(row: dict[str, Any], input_key: str) -> str:
    if input_key not in row:
        raise KeyError(f"Probe source row is missing input key {input_key!r}.")
    return value_identity(row[input_key])


def select_globally_unique_jsonl_tail(
    path: Path,
    *,
    input_key: str,
    probe_prompts: int,
) -> tuple[list[dict[str, Any]], int, Counter[str], int, str]:
    """Select a globally unique tail set with one linear scan and no data copy."""

    counts: Counter[str] = Counter()
    positions: list[tuple[str, int]] = []
    source_digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            source_digest.update(raw)
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON objects in {path}.")
            identity = prompt_identity(value, input_key)
            counts[identity] += 1
            positions.append((identity, offset))

        selected_positions = []
        for row_index in range(len(positions) - 1, -1, -1):
            identity, offset = positions[row_index]
            if counts[identity] != 1:
                continue
            selected_positions.append((row_index, offset))
            if len(selected_positions) == probe_prompts:
                break
        if len(selected_positions) != probe_prompts:
            raise ValueError(
                f"Data source {path} has only {len(selected_positions)} globally unique prompts; "
                f"cannot reserve {probe_prompts} without train/probe leakage."
            )
        selected_positions.sort()
        probe = []
        for _, offset in selected_positions:
            stream.seek(offset)
            value = json.loads(stream.readline())
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON objects in {path}.")
            probe.append(value)

    source_rows = len(positions)
    excluded_tail_rows = source_rows - selected_positions[0][0]
    return probe, excluded_tail_rows, counts, source_rows, source_digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def materialize_eval_config(
    config_root: Path,
    output_dir: Path,
    tasks: tuple[str, ...],
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = []
    inputs = []
    dataset_inputs = []
    for task in tasks:
        config_path = (config_root / DEFAULT_EVAL_FILES[task]).resolve()
        config = read_mapping(config_path)
        eval_config = config.get("eval")
        if not isinstance(eval_config, dict):
            raise ValueError(f"Evaluation config lacks `eval`: {config_path}")
        defaults = eval_config.get("defaults") or {}
        raw_datasets = eval_config.get("datasets")
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ValueError(f"Evaluation config has no dataset list: {config_path}")
        for raw in raw_datasets:
            entry = {**defaults, **dict(raw)}
            data_path = resolve_data_path(str(entry["path"]), config_path)
            entry["path"] = str(data_path)
            entry.setdefault("top_k", -1)
            entry["metadata_overrides"] = {
                **dict(entry.get("metadata_overrides") or {}),
                "sequential_domain": "knowledge" if task == "science" else task,
            }
            datasets.append(entry)
            dataset_inputs.append(
                {
                    "name": str(entry.get("name") or ""),
                    "task": task,
                    "path": str(data_path),
                    "sha256": sha256(data_path),
                    "bytes": data_path.stat().st_size,
                    "mtime_ns": data_path.stat().st_mtime_ns,
                    "input_key": str(entry.get("input_key") or "prompt"),
                }
            )
        inputs.append(
            {
                "task": task,
                "path": str(config_path),
                "sha256": sha256(config_path),
            }
        )

    names = [str(entry.get("name") or "") for entry in datasets]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"Combined evaluation dataset names must be non-empty and unique: {names}")
    path = output_dir / "all_tasks_eval.yaml"
    atomic_text(path, yaml.safe_dump({"eval": {"defaults": {}, "datasets": datasets}}, sort_keys=False))
    return path, inputs, dataset_inputs


def input_values(path: Path, input_key: str):
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        table = parquet.read_table(path, columns=[input_key])
        yield from table.column(input_key).to_pylist()
        return
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if input_key not in row:
                    raise KeyError(f"Evaluation row in {path} lacks input key {input_key!r}.")
                yield row[input_key]
        return
    raise ValueError(f"Probe/eval overlap validation does not support {path.suffix}: {path}")


def validate_probe_eval_disjoint(
    probe_identities: dict[str, set[str]],
    eval_datasets: list[dict[str, Any]],
) -> None:
    owners: dict[str, str] = {}
    for task, identities in probe_identities.items():
        for identity in identities:
            previous = owners.setdefault(identity, task)
            if previous != task:
                raise ValueError(f"Probe prompt identity is shared by {previous} and {task}.")

    overlaps = []
    for dataset in eval_datasets:
        path = Path(dataset["path"])
        input_key = str(dataset["input_key"])
        for row_index, value in enumerate(input_values(path, input_key)):
            task = owners.get(value_identity(value))
            if task is not None:
                overlaps.append(
                    {
                        "probe_task": task,
                        "eval_dataset": dataset["name"],
                        "eval_row": row_index,
                    }
                )
    if overlaps:
        raise ValueError(f"Frozen gradient probes overlap formal evaluation prompts: {overlaps[:20]}")


def prepare(args: argparse.Namespace) -> None:
    config_root = args.config_root.resolve()
    output_dir = args.output_dir.resolve()
    tasks = tuple(args.tasks)
    index_path = output_dir / "sequential_data_index.json"
    source_index_path = config_root / "single_task_index.json"
    source_index_sha256 = sha256(source_index_path) if source_index_path.is_file() else None
    source_index = json.loads(source_index_path.read_text(encoding="utf-8")) if source_index_path.is_file() else {}
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        if index_path.is_file():
            current = json.loads(index_path.read_text(encoding="utf-8"))
            expected = {
                "schema_version": 3,
                "config_root": str(config_root),
                "seed": args.seed,
                "probe_prompts_per_task": args.probe_prompts,
                "train_prompts_per_task": args.train_prompts_per_task,
                "source_index_sha256": source_index_sha256,
                "tasks_prepared": list(tasks),
            }
            mismatched = {
                key: (current.get(key), value) for key, value in expected.items() if current.get(key) != value
            }
            if mismatched:
                raise ValueError(f"Existing sequential data index is incompatible: {mismatched}")
            required = [Path(current["all_tasks_eval_config"])]
            required.extend(
                Path(record[key])
                for record in current["tasks"].values()
                for key in ("train_manifest", "probe_manifest", "probe_data")
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Prepared sequential artifacts are missing: {missing}")
            changed_prepared = []
            if sha256(Path(current["all_tasks_eval_config"])) != current["all_tasks_eval_config_sha256"]:
                changed_prepared.append(current["all_tasks_eval_config"])
            for record in current["tasks"].values():
                for path_key, digest_key in (
                    ("train_manifest", "train_manifest_sha256"),
                    ("probe_manifest", "probe_manifest_sha256"),
                    ("probe_data", "probe_data_sha256"),
                ):
                    path = Path(record[path_key])
                    if sha256(path) != record[digest_key]:
                        changed_prepared.append(str(path))
            if changed_prepared:
                raise ValueError(f"Prepared sequential artifacts were modified: {changed_prepared}")
            changed_sources = []
            for task, record in current["tasks"].items():
                source_manifest = Path(record["source_manifest"])
                source_data = Path(record["source_data"])
                if not source_manifest.is_file() or sha256(source_manifest) != record["source_manifest_sha256"]:
                    changed_sources.append(f"{task}:source_manifest")
                if not source_data.is_file():
                    changed_sources.append(f"{task}:source_data_missing")
                elif (
                    source_data.stat().st_size != record["source_data_bytes"]
                    or source_data.stat().st_mtime_ns != record["source_data_mtime_ns"]
                ):
                    changed_sources.append(f"{task}:source_data")
            if changed_sources:
                raise ValueError(
                    "Prepared sequential artifacts no longer match their source inputs: "
                    f"{changed_sources}. Use a new output directory or pass --force."
                )
            changed_eval_inputs = []
            for record in [
                *current.get("eval_config_inputs", []),
                *current.get("eval_dataset_inputs", []),
            ]:
                path = Path(record["path"])
                if not path.is_file() or sha256(path) != record["sha256"]:
                    changed_eval_inputs.append(str(path))
            if changed_eval_inputs:
                raise ValueError(
                    "Prepared sequential evaluation inputs changed: "
                    f"{changed_eval_inputs}. Use a new output directory or pass --force."
                )
            print(f"Prepared sequential data already exists: {index_path}")
            return
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --force to replace known files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_records: dict[str, Any] = {}
    probe_identities: dict[str, set[str]] = {}
    for task in tasks:
        source_manifest = (config_root / task / f"{task}_on_policy.yaml").resolve()
        manifest = read_mapping(source_manifest)
        sources = manifest.get("sources")
        if not isinstance(sources, list) or len(sources) != 1:
            raise ValueError(
                f"Sequential preparation currently requires exactly one source for {task}: {source_manifest}"
            )
        source = dict(sources[0])
        input_key = str(source.get("input_key") or "prompt")
        source_data = resolve_data_path(str(source.get("path") or ""), source_manifest)
        if source_data.suffix != ".jsonl":
            raise ValueError(f"Sequential train/probe splitting requires JSONL input: {source_data}")
        (
            probe,
            excluded_tail_rows,
            prompt_counts,
            scanned_source_rows,
            source_data_sha256,
        ) = select_globally_unique_jsonl_tail(
            source_data,
            input_key=input_key,
            probe_prompts=args.probe_prompts,
        )
        probe_identities[task] = {prompt_identity(row, input_key) for row in probe}

        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        probe_path = task_dir / f"{task}_gradient_probe.jsonl"
        atomic_jsonl(probe_path, probe)

        available_train_rows = scanned_source_rows - excluded_tail_rows
        train_rows = args.train_prompts_per_task or available_train_rows
        if train_rows > available_train_rows:
            raise ValueError(
                f"Requested {train_rows} training prompts for {task}, but only "
                f"{available_train_rows} remain after reserving probes."
            )
        train_slice = (
            f"{source_data}@[0:{train_rows}]"
            if args.train_prompts_per_task is not None
            else f"{source_data}@[0:-{excluded_tail_rows}]"
        )
        train_source = {**source, "path": train_slice, "name": f"{task}_sequential_train"}
        probe_source = {**source, "path": str(probe_path), "name": f"{task}_gradient_probe"}
        train_manifest = {
            "version": int(manifest.get("version", 1)),
            "sampling": {**dict(manifest.get("sampling") or {}), "seed": args.seed, "repeat": True},
            "sources": [train_source],
        }
        probe_manifest = {
            "version": int(manifest.get("version", 1)),
            "sampling": {"strategy": "weighted", "unit": "batch", "seed": args.seed, "repeat": False},
            "sources": [probe_source],
        }
        train_manifest_path = task_dir / f"{task}_on_policy.yaml"
        probe_manifest_path = task_dir / f"{task}_gradient_probe.yaml"
        atomic_text(train_manifest_path, yaml.safe_dump(train_manifest, sort_keys=False))
        atomic_text(probe_manifest_path, yaml.safe_dump(probe_manifest, sort_keys=False))

        indexed_source_rows = (source_index.get("tasks") or {}).get(task, {}).get("train_rows")
        if indexed_source_rows is not None and int(indexed_source_rows) != scanned_source_rows:
            raise ValueError(
                f"Source row count for {task} differs from single_task_index.json: "
                f"{scanned_source_rows} != {indexed_source_rows}."
            )
        source_rows = scanned_source_rows
        task_records[task] = {
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256(source_manifest),
            "source_data": str(source_data),
            "source_data_sha256": source_data_sha256,
            "source_data_bytes": source_data.stat().st_size,
            "source_data_mtime_ns": source_data.stat().st_mtime_ns,
            "source_rows": source_rows,
            "source_unique_prompt_count": len(prompt_counts),
            "source_duplicate_prompt_rows": sum(count - 1 for count in prompt_counts.values()),
            "train_rows": train_rows,
            "available_train_rows": available_train_rows,
            "probe_rows": len(probe),
            "excluded_tail_rows": excluded_tail_rows,
            "train_path_slice": train_slice,
            "train_manifest": str(train_manifest_path),
            "train_manifest_sha256": sha256(train_manifest_path),
            "probe_manifest": str(probe_manifest_path),
            "probe_manifest_sha256": sha256(probe_manifest_path),
            "probe_data": str(probe_path),
            "probe_data_sha256": sha256(probe_path),
        }

    eval_path, eval_inputs, eval_dataset_inputs = materialize_eval_config(config_root, output_dir, tasks)
    validate_probe_eval_disjoint(probe_identities, eval_dataset_inputs)
    index = {
        "schema_version": 3,
        "purpose": "Frozen train/probe/eval artifacts for sequential GRPO",
        "config_root": str(config_root),
        "source_index": str(source_index_path) if source_index_path.is_file() else None,
        "source_index_sha256": source_index_sha256,
        "seed": args.seed,
        "train_prompts_per_task": args.train_prompts_per_task,
        "tasks_prepared": list(tasks),
        "probe_selection": (
            f"Last {args.probe_prompts} prompts whose identity occurs exactly once in each complete source. "
            "The consumed tail rows are excluded from training with Dataset path slicing, so multi-GB "
            "sources are scanned once but not copied."
        ),
        "probe_prompts_per_task": args.probe_prompts,
        "tasks": task_records,
        "all_tasks_eval_config": str(eval_path),
        "all_tasks_eval_config_sha256": sha256(eval_path),
        "eval_config_inputs": eval_inputs,
        "eval_dataset_inputs": eval_dataset_inputs,
        "probe_eval_overlap_count": 0,
    }
    atomic_text(index_path, json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Wrote sequential train/probe/eval artifacts to {output_dir}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=root / "data/m2rl/single_task",
        help="Existing prepared single-task configuration root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
        help="Tasks whose train/probe splits and evaluation datasets should be materialized.",
    )
    parser.add_argument("--probe-prompts", type=int, default=128)
    parser.add_argument(
        "--train-prompts-per-task",
        type=int,
        help="Use only this many prefix prompts per task for training; the pool is shuffled at rollout time.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.probe_prompts <= 0:
        parser.error("--probe-prompts must be positive.")
    if args.train_prompts_per_task is not None and args.train_prompts_per_task <= 0:
        parser.error("--train-prompts-per-task must be positive.")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("--tasks cannot contain duplicates.")
    return args


if __name__ == "__main__":
    prepare(parse_args())
