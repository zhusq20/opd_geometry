#!/usr/bin/env python3
"""Convert the Nemotron/M2RL training blend into Slime multi-task JSONL files.

The converter is streaming and does not require Hugging Face ``datasets``.
It accepts JSONL or Parquet inputs, preserves verifier metadata, and emits a
manifest consumable by ``MultiTaskRolloutDataSource``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    import orjson
except ImportError:  # pragma: no cover - exercised in minimal environments
    orjson = None


CATEGORY_MAP = {
    "nano_v3_sft_profiled_dapo17k": ("math", "deepscaler"),
    "nano_v3_sft_profiled_skywork_no_omni": ("math", "deepscaler"),
    "nano_v3_sft_profiled_stem_mcqa": ("science", "gpqa"),
    "nano_v3_sft_profiled_instruction_following": ("if", "ifevalg"),
    "nano_v3_sft_profiled_comp_coding_50tests": ("code", "unit_test"),
    "nano_v3_sft_profiled_workbench": ("agent", "workbench"),
}

TASK_ORDER = ("math", "science", "if", "code", "agent")
# The geometry study intentionally excludes WorkBench/Agent because it requires
# a stateful external environment and introduces a different source of systems
# variance. Agent conversion remains supported when explicitly requested.
DEFAULT_TASKS = TASK_ORDER[:-1]


def _json_loads(line: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def _json_line(value: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(value) + b"\n"
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def iter_rows(paths: Iterable[Path]):
    for path in paths:
        if path.suffix == ".jsonl":
            with path.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            yield _json_loads(line)
                        except ValueError as exc:
                            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        elif path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError("Parquet input requires pyarrow; JSONL input has no optional dependency.") from exc
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches():
                yield from batch.to_pylist()
        else:
            raise ValueError(f"Unsupported input {path}; expected .jsonl or .parquet.")


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    params = row.get("responses_create_params") or {}
    messages = copy.deepcopy(params.get("input") or row.get("prompt"))
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Row has no usable responses_create_params.input/prompt.")
    return messages


def convert_row(row: dict[str, Any], index: int) -> tuple[str, dict[str, Any]] | None:
    dataset_name = row.get("dataset")
    if dataset_name not in CATEGORY_MAP:
        return None
    task, rm_type = CATEGORY_MAP[dataset_name]
    messages = _messages(row)
    prompt_text = str(messages[0].get("content", ""))
    label = None
    metadata: dict[str, Any] = {"rm_type": rm_type, "original_dataset": dataset_name, "original_index": index}

    if task == "math":
        messages[0]["content"] = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes), where $Answer is the answer to the problem.\n\n" + prompt_text
        label = row.get("expected_answer")
    elif task == "science":
        prefix = "Answer the following multiple choice question."
        question = prompt_text[len(prefix) :] if prompt_text.startswith(prefix) else prompt_text
        messages[0]["content"] = "Answer the following multiple choice question step by step." + question
        label = row.get("expected_answer")
        metadata.update({key: row[key] for key in ("valid_letters", "correct_letter") if key in row})
    elif task == "if":
        metadata.update(
            {
                "prompt_text": row.get("prompt", prompt_text),
                "instruction_id_list": row.get("instruction_id_list", []),
                "kwargs": row.get("kwargs", []),
            }
        )
    elif task == "code":
        metadata.update(row.get("verifier_metadata") or {})
    elif task == "agent":
        metadata["ground_truth_tool_calls"] = row.get("ground_truth")

    output = {"prompt": messages, "label": label, "metadata": metadata, "data_source": task}
    tools = (row.get("responses_create_params") or {}).get("tools") or row.get("tools")
    if tools is not None:
        output["tools"] = tools
    return task, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-name", default="multitask_manifest.yaml")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_ORDER,
        default=list(DEFAULT_TASKS),
        help="Tasks to materialize. Defaults to the four non-Agent M2RL domains.",
    )
    parser.add_argument("--sampling", choices=["uniform", "proportional", "weighted", "round_robin", "sequential"], default="uniform")
    parser.add_argument("--sampling-unit", choices=["prompt", "batch"], default="batch")
    parser.add_argument("--phase-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_tasks = [task for task in TASK_ORDER if task in set(args.tasks)]
    output_paths = {task: args.output_dir / f"{task}.jsonl" for task in selected_tasks}
    temporary_paths = {task: path.with_name(f".{path.name}.tmp.{os.getpid()}") for task, path in output_paths.items()}
    streams = {task: temporary_paths[task].open("wb") for task in selected_tasks}
    counts: Counter[str] = Counter()
    digests = {task: hashlib.sha256() for task in selected_tasks}
    completed = False
    try:
        for index, row in enumerate(iter_rows(args.input)):
            converted = convert_row(row, index)
            if converted is None:
                continue
            task, output = converted
            if task not in streams:
                continue
            line = _json_line(output)
            streams[task].write(line)
            digests[task].update(line)
            counts[task] += 1
        missing = [task for task in selected_tasks if counts[task] == 0]
        if missing:
            raise ValueError(f"No records were found for requested tasks: {', '.join(missing)}")
        completed = True
    finally:
        for stream in streams.values():
            stream.close()
        if not completed:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)

    for task in selected_tasks:
        temporary_paths[task].replace(output_paths[task])

    reward_by_task = {task: rm_type for _, (task, rm_type) in CATEGORY_MAP.items()}
    sources = []
    for task in selected_tasks:
        source: dict[str, Any] = {
            "name": task,
            "path": output_paths[task].name,
            "input_key": "prompt",
            "label_key": "label",
            "metadata_key": "metadata",
            "tool_key": "tools",
            "apply_chat_template": task != "agent",
            "rm_type": reward_by_task[task],
            "weight": 1.0,
        }
        if args.phase_samples is not None:
            source["phase_samples"] = args.phase_samples
        sources.append(source)

    manifest = {
        "version": 1,
        "sampling": {
            "strategy": args.sampling,
            "unit": args.sampling_unit,
            "seed": args.seed,
            "repeat": True,
        },
        "sources": sources,
    }
    manifest_path = args.output_dir / args.manifest_name
    import yaml

    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    with manifest_tmp.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    manifest_tmp.replace(manifest_path)

    dataset_info = {
        "format_version": 1,
        "inputs": [str(path.resolve()) for path in args.input],
        "manifest": str(manifest_path.resolve()),
        "tasks": selected_tasks,
        "files": {
            task: {
                "path": output_paths[task].name,
                "records": counts[task],
                "bytes": output_paths[task].stat().st_size,
                "sha256": digests[task].hexdigest(),
            }
            for task in selected_tasks
        },
    }
    info_path = args.output_dir / "dataset_info.json"
    info_tmp = info_path.with_name(f".{info_path.name}.tmp.{os.getpid()}")
    with info_tmp.open("w", encoding="utf-8") as stream:
        json.dump(dataset_info, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    info_tmp.replace(info_path)
    print(json.dumps({"manifest": str(manifest_path), "counts": dict(counts)}, sort_keys=True))


if __name__ == "__main__":
    main()
