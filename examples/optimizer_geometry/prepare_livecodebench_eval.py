#!/usr/bin/env python3
"""Prepare official LiveCodeBench code-generation data for Slime + SandboxFusion."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import pickle
import random
import re
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset


BASE_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main"
VERSION_FILES = {
    "v5": ["test5.jsonl"],
    "v6": ["test6.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}
EXPECTED_ROWS = {"v5": 167, "v6": 175, "release_v5": 880, "release_v6": 1055}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalized_text(value: Any) -> str:
    if isinstance(value, list):
        value = "\n".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in value)
    value = re.sub(r"\s+", " ", str(value)).strip().lower()
    return value


def prompt_hash(value: Any) -> str:
    return hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()


def training_texts(path: Path | None) -> list[str]:
    if path is None:
        return []
    texts = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                texts.append(normalized_text(json.loads(line).get("prompt", "")))
    return texts


class _PrimitiveOnlyUnpickler(pickle.Unpickler):
    """Decode legacy LiveCodeBench payloads without importing pickle globals."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"pickle global {module}.{name} is forbidden")


def decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        decoded = zlib.decompress(base64.b64decode(value, validate=True))
        try:
            payload: Any = decoded.decode("utf-8")
        except UnicodeDecodeError:
            # Official LiveCodeBench releases encode private cases as a
            # zlib-compressed pickle containing only primitive Python data.
            # A restricted unpickler prevents a malformed download from
            # importing or invoking arbitrary globals during preparation.
            payload = _PrimitiveOnlyUnpickler(io.BytesIO(decoded)).load()
        return json.loads(payload) if isinstance(payload, str) else payload


def make_prompt(question: str, starter_code: str) -> str:
    instruction = (
        "### Instruction\nYou are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program that matches the "
        "specification and passes all tests. You will NOT return anything except for the program.\n\n"
    )
    if starter_code:
        format_text = (
            "### Format: You will use the following starter code to write the solution to the problem "
            "and enclose your code within delimiters.\n```python\n"
            + starter_code
            + "\n```"
        )
    else:
        format_text = (
            "### Format: Read the inputs from stdin, solve the problem, and write the answer to stdout "
            "(do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n"
            "```python\n```"
        )
    return f"{instruction}### Question:\n{question}\n\n{format_text}\n\n### Answer: (use the provided format with backticks)\n"


def sandboxfusion_test(row: dict[str, Any]) -> dict[str, Any]:
    public = decode_json(row.get("public_test_cases") or "[]") or []
    private = decode_json(row.get("private_test_cases") or "[]") or []
    cases = [*public, *private]
    metadata = decode_json(row.get("metadata") or "{}") or {}
    input_output: dict[str, Any] = {
        "inputs": [case.get("input") for case in cases],
        "outputs": [case.get("output") for case in cases],
    }
    function_name = metadata.get("func_name") or metadata.get("function_name")
    if function_name:
        input_output["fn_name"] = function_name
    return {"input_output": json.dumps(input_output, separators=(",", ":"), ensure_ascii=False)}


def convert(row: dict[str, Any]) -> dict[str, Any]:
    question_id = str(row["question_id"])
    prompt = make_prompt(str(row["question_content"]), str(row.get("starter_code") or ""))
    sf_row = {
        "id": question_id,
        "labels": json.dumps(
            {
                "difficulty": row.get("difficulty"),
                "platform": row.get("platform"),
                "contest_date": row.get("contest_date"),
            },
            sort_keys=True,
        ),
        "content": prompt,
        "test": json.dumps(sandboxfusion_test(row), separators=(",", ":"), ensure_ascii=False),
    }
    return {
        "prompt": prompt,
        "label": "",
        "data_source": "livecodebench-code-generation-lite",
        "metadata": {
            "rm_type": "livecodebench",
            "question_id": question_id,
            "question_title": str(row.get("question_title") or ""),
            "platform": str(row.get("platform") or ""),
            "contest_id": str(row.get("contest_id") or ""),
            "contest_date": str(row.get("contest_date") or ""),
            "difficulty": str(row.get("difficulty") or "unknown"),
            "question_sha256": prompt_hash(row["question_content"]),
            "sandboxfusion_row": json.dumps(sf_row, separators=(",", ":"), ensure_ascii=False),
        },
    }


def download(path: Path, url: str) -> None:
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def balanced_subset(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return rows
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["metadata"]["difficulty"]].append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    names = sorted(groups)
    while len(selected) < count and any(groups.values()):
        for name in names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
    return sorted(selected, key=lambda item: item["metadata"]["question_id"])


def atomic_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.parquet")
    try:
        Dataset.from_list(rows).to_parquet(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_configs(output_dir: Path, online_path: Path, final_path: Path) -> None:
    common = {
        "apply_chat_template": True,
        "custom_rm_path": "slime_plugins.m2rl.rewards.reward",
        "max_response_len": 16384,
        "top_k": -1,
    }
    online = {
        "eval": {
            "defaults": {**common, "n_samples_per_eval_prompt": 1, "temperature": 0.0, "top_p": 1.0},
            "datasets": [{"name": "livecodebench_online", "path": str(online_path.resolve()), "rm_type": "livecodebench"}],
        }
    }
    final = {
        "eval": {
            "defaults": {**common, "n_samples_per_eval_prompt": 10, "temperature": 0.2, "top_p": 0.95},
            "datasets": [{"name": "livecodebench_final", "path": str(final_path.resolve()), "rm_type": "livecodebench"}],
        }
    }
    (output_dir / "code_eval.yaml").write_text(yaml.safe_dump(online, sort_keys=False), encoding="utf-8")
    (output_dir / "code_eval_final.yaml").write_text(yaml.safe_dump(final, sort_keys=False), encoding="utf-8")


def load_version(
    version: str,
    raw_dir: Path,
    train_prompts: list[str],
    collision_cache: dict[str, bool],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    converted = []
    collisions = []
    sources = []
    seen = set()
    for filename in VERSION_FILES[version]:
        path = raw_dir / filename
        url = f"{BASE_URL}/{filename}"
        download(path, url)
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                source = json.loads(line)
                question_id = str(source["question_id"])
                if question_id in seen:
                    continue
                seen.add(question_id)
                count += 1
                question = normalized_text(source["question_content"])
                question_digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
                collision = collision_cache.get(question_digest)
                if collision is None:
                    # M2RL wraps coding problems in its own instruction prefix,
                    # so comparing only the complete prompt hashes would miss
                    # an exact duplicated problem. Require the entire normalized
                    # LCB problem statement to occur in a training prompt.
                    collision = bool(question) and any(question in prompt for prompt in train_prompts)
                    collision_cache[question_digest] = collision
                if collision:
                    collisions.append(question_id)
                else:
                    converted.append(convert(source))
        sources.append({"filename": filename, "url": url, "sha256": sha256_file(path), "unique_rows": count})
    expected = EXPECTED_ROWS[version]
    if len(converted) + len(collisions) != expected:
        raise ValueError(
            f"Expected {expected} unique rows for {version}, got {len(converted)} plus {len(collisions)} collisions."
        )
    return converted, collisions, sources


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = args.output_dir / "raw"
    online_version = args.online_version or args.version
    train_prompts = training_texts(args.training_data)
    collision_cache: dict[str, bool] = {}
    converted, collisions, final_sources = load_version(
        args.version, raw_dir, train_prompts, collision_cache
    )
    if online_version == args.version:
        online_candidates = converted
        online_collisions = collisions
        online_sources = final_sources
    else:
        online_candidates, online_collisions, online_sources = load_version(
            online_version, raw_dir, train_prompts, collision_cache
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / f"livecodebench_{args.version}.parquet"
    online_path = args.output_dir / f"livecodebench_{online_version}_online{args.online_samples}.parquet"
    online_rows = balanced_subset(online_candidates, args.online_samples, args.seed)
    atomic_parquet(converted, final_path)
    atomic_parquet(online_rows, online_path)
    write_configs(args.output_dir, online_path, final_path)
    index = {
        "schema_version": 1,
        "dataset": "livecodebench/code_generation_lite",
        "version": args.version,
        "online_version": online_version,
        "expected_rows": EXPECTED_ROWS[args.version],
        "rows_after_training_overlap_filter": len(converted),
        "exact_normalized_training_prompt_collisions": sorted(collisions),
        "online_exact_normalized_training_prompt_collisions": sorted(online_collisions),
        "online_rows": len(online_rows),
        "online_seed": args.seed,
        "sources": {"final": final_sources, "online": online_sources},
        "artifacts": {
            "final": {"path": str(final_path.resolve()), "sha256": sha256_file(final_path)},
            "online": {"path": str(online_path.resolve()), "sha256": sha256_file(online_path)},
        },
        "protocols": {
            "online": {"n": 1, "temperature": 0.0, "purpose": "checkpoint curves only"},
            "final": {"n": 10, "temperature": 0.2, "top_p": 0.95, "purpose": "paper pass@k"},
        },
    }
    benchmark_index_path = args.output_dir / "livecodebench_index.json"
    atomic_json(benchmark_index_path, index)
    single_task_index_path = args.output_dir.parent / "single_task_index.json"
    if single_task_index_path.is_file():
        single_task_index = json.loads(single_task_index_path.read_text(encoding="utf-8"))
        code = single_task_index.setdefault("tasks", {}).setdefault("code", {})
        code.update(
            {
                "eval_config": str((args.output_dir / "code_eval.yaml").resolve()),
                "eval_kind": "external_benchmark",
                "eval_name": "livecodebench_online",
                "eval_rm_type": "livecodebench",
                "eval_rows": len(online_rows),
                "eval_samples_per_prompt": 1,
                "final_eval_config": str((args.output_dir / "code_eval_final.yaml").resolve()),
                "final_eval_name": "livecodebench_final",
                "final_eval_rows": len(converted),
                "final_eval_samples_per_prompt": 10,
                "benchmark_index": str(benchmark_index_path.resolve()),
                "benchmark_index_sha256": sha256_file(benchmark_index_path),
            }
        )
        atomic_json(single_task_index_path, single_task_index)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", choices=sorted(VERSION_FILES), default="v5")
    parser.add_argument(
        "--online-version",
        choices=sorted(VERSION_FILES),
        help="Optional recent slice for checkpoint curves (for example v5 with final release_v5).",
    )
    parser.add_argument("--online-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-data", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))
