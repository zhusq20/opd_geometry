#!/usr/bin/env python3
"""Prepare the public Logic-RL Knights-and-Knaves train/validation task.

The public Parallel-RL repository does not include its private Logic subset.
This preparer therefore pins the public Logic-RL 3--7 person data, whose
committed split contains 4,500 training puzzles and 500 validation puzzles.
It rebuilds ordinary chat messages from ``quiz`` rather than reusing
Logic-RL's already-rendered Qwen prompt.

The underlying K-and-K dataset is licensed CC-BY-NC-SA-4.0. Generated indexes
record the source URL, revision, file hashes, and license identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
import yaml

LOGIC_RL_REPOSITORY = "https://github.com/Unakar/Logic-RL"
LOGIC_RL_REVISION = "9d2c457525ec14639e85afa12d49bb16efb053a4"
K_AND_K_DATASET = "https://huggingface.co/datasets/K-and-K/knights-and-knaves"
K_AND_K_LICENSE = "CC-BY-NC-SA-4.0"
DIFFICULTIES = (3, 4, 5, 6, 7)
EXPECTED_ROWS_PER_FILE = {"train": 900, "test": 100}
EXPECTED_OUTPUT_ROWS = {"train": 4_500, "validation": 500}
SOURCE_SHA256 = {
    (3, "train"): "3562e95e729ca9b87afc0496bcd0022d0b4714f41d917273f99fc7d473dd4701",
    (3, "test"): "0361ec695ae77e6ae11ad2683097a958cd4dfdb64ea965ae9a7929d0b2e9290c",
    (4, "train"): "e55481a7d79d6bc80b962492979515b2adf82a19268b1c45b291503a60d2ddff",
    (4, "test"): "47d8641333d8946cefafd58ac15dcb7a7924eb5065f483c95743103480414d54",
    (5, "train"): "d73d503efc2b87da3394f096dfd5e2501865c7e9319814499266c866998afac0",
    (5, "test"): "438550a48e95293f9b7802a4d0e48baa266295700c2bcdb64a3bdc48e3d29bda",
    (6, "train"): "24edd3bd45e09fca8fc64a7c2c04a9467faa002864104657ae393846316c978c",
    (6, "test"): "9345d9a29507b735f00007e4ad5a50c12c696c4b8100178cdd090113f728d639",
    (7, "train"): "18166d884c7d0c6aae33211cfd0d9a426f127f8c9b21dfb6fdc5bd29ceeddfea",
    (7, "test"): "6921d865103220e4ba4a672ec2a2c20919d4fae476e7cd3786ade554077188a0",
}

LOGIC_SYSTEM_PROMPT = """You solve Knights-and-Knaves logic puzzles. Knights always tell the truth and knaves always lie. You may reason before giving the final answer.

Finish your response with exactly one <answer>...</answer> block. Inside that block, write exactly one non-empty line for every inhabitant, using one of these forms:
Name is a knight
Name is a knave

List every inhabitant exactly once. The line order may differ from the question. Do not include any other text inside the answer block, and do not write anything after </answer>."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(difficulty: int, split: str) -> str:
    return (
        f"https://raw.githubusercontent.com/Unakar/Logic-RL/{LOGIC_RL_REVISION}/"
        f"data/kk/instruct/{difficulty}ppl/{split}.parquet"
    )


def _download_source(url: str, path: Path, expected_sha256: str, *, offline: bool, retries: int) -> None:
    if path.is_file() and sha256_file(path) == expected_sha256:
        return
    if offline:
        raise FileNotFoundError(f"Missing a valid cached Logic-RL source while --offline is set: {path}")
    if retries <= 0:
        raise ValueError("Download retries must be positive.")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    last_error: Exception | None = None
    try:
        for attempt in range(retries):
            temporary.unlink(missing_ok=True)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "slime-optimizer-geometry/logic-kk"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                observed = sha256_file(temporary)
                if observed != expected_sha256:
                    raise ValueError(
                        f"SHA-256 mismatch for {url}: expected {expected_sha256}, observed {observed}."
                    )
                temporary.replace(path)
                return
            except (OSError, urllib.error.URLError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Failed to download pinned Logic-RL source {url}: {last_error}") from last_error
    finally:
        temporary.unlink(missing_ok=True)


def _provided_source_path(source_dir: Path, difficulty: int, split: str) -> Path:
    canonical = source_dir / f"{difficulty}ppl" / f"{split}.parquet"
    if canonical.is_file():
        return canonical
    flat = source_dir / f"{difficulty}_{split}.parquet"
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"No provided Logic-RL source for {difficulty}ppl/{split}; checked {canonical} and {flat}."
    )


def resolve_sources(
    raw_dir: Path,
    *,
    source_dir: Path | None,
    offline: bool,
    retries: int,
) -> dict[tuple[int, str], Path]:
    paths: dict[tuple[int, str], Path] = {}
    for difficulty in DIFFICULTIES:
        for split in ("train", "test"):
            key = (difficulty, split)
            if source_dir is not None:
                path = _provided_source_path(source_dir, difficulty, split)
            else:
                path = raw_dir / f"{difficulty}ppl" / f"{split}.parquet"
                _download_source(
                    source_url(difficulty, split),
                    path,
                    SOURCE_SHA256[key],
                    offline=offline,
                    retries=retries,
                )
            observed = sha256_file(path)
            if observed != SOURCE_SHA256[key]:
                raise ValueError(
                    f"Pinned Logic-RL source hash mismatch for {path}: "
                    f"expected {SOURCE_SHA256[key]}, observed {observed}."
                )
            paths[key] = path.resolve()
    return paths


def convert_row(row: dict[str, Any], *, difficulty: int, source_split: str) -> dict[str, Any]:
    """Convert one pinned Logic-RL row to the local RLVR schema."""

    quiz = str(row.get("quiz") or "").strip()
    names_value = row.get("names")
    solution_value = row.get("solution")
    if not quiz:
        raise ValueError("Logic-RL row has an empty `quiz`.")
    if not isinstance(names_value, (list, tuple)) or not isinstance(solution_value, (list, tuple)):
        raise ValueError("Logic-RL rows require list-valued `names` and `solution` fields.")
    names = [str(name).strip() for name in names_value]
    if len(names) != difficulty or len(solution_value) != difficulty:
        raise ValueError(
            f"Logic-RL {difficulty}ppl row has {len(names)} names and {len(solution_value)} solution entries."
        )
    if any(not name for name in names):
        raise ValueError("Logic-RL names must be non-empty strings.")
    normalized_names = [" ".join(name.casefold().split()) for name in names]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(f"Logic-RL row contains duplicate names: {names}.")
    if any(not isinstance(value, bool) for value in solution_value):
        raise ValueError("Logic-RL `solution` entries must be booleans.")
    if source_split not in {"train", "validation"}:
        raise ValueError(f"Unexpected converted Logic-RL split: {source_split!r}.")
    source_index = row.get("index")
    if not isinstance(source_index, int) or isinstance(source_index, bool):
        raise ValueError("Logic-RL rows require an integer `index`.")
    statements = str(row.get("statements") or "").strip()
    if not statements:
        raise ValueError("Logic-RL row has empty symbolic `statements` metadata.")

    roles = ["knight" if value else "knave" for value in solution_value]
    source_id = f"{difficulty}ppl/{source_split}/{source_index}"
    return {
        "prompt": [
            {"role": "system", "content": LOGIC_SYSTEM_PROMPT},
            {"role": "user", "content": quiz},
        ],
        "label": {"names": names, "roles": roles},
        "data_source": "logic_rl_kk",
        "metadata": {
            "rm_type": "kk",
            "task_name": "logic",
            "dataset": "logic_kk",
            "difficulty": difficulty,
            "source_split": source_split,
            "source_index": source_index,
            "source_id": source_id,
            "statements": statements,
            "logic_rl_revision": LOGIC_RL_REVISION,
        },
        "tools": [],
    }


def load_converted_rows(
    source_paths: dict[tuple[int, str], Path],
    *,
    expected_rows_per_file: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = expected_rows_per_file or EXPECTED_ROWS_PER_FILE
    outputs = {"train": [], "validation": []}
    for difficulty in DIFFICULTIES:
        for upstream_split, output_split in (("train", "train"), ("test", "validation")):
            path = source_paths[(difficulty, upstream_split)]
            table = parquet.read_table(path)
            expected_count = expected[upstream_split]
            if table.num_rows != expected_count:
                raise ValueError(
                    f"Pinned Logic-RL file {path} has {table.num_rows} rows; expected {expected_count}."
                )
            for row in table.to_pylist():
                outputs[output_split].append(
                    convert_row(row, difficulty=difficulty, source_split=output_split)
                )
    return outputs["train"], outputs["validation"]


def validate_splits(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    expected_output_rows: dict[str, int] | None = None,
) -> None:
    expected = expected_output_rows or EXPECTED_OUTPUT_ROWS
    observed = {"train": len(train_rows), "validation": len(validation_rows)}
    if observed != expected:
        raise ValueError(f"Unexpected converted Logic-RL split sizes: expected {expected}, observed {observed}.")

    split_quizzes: dict[str, set[str]] = {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        source_ids = [str(row["metadata"]["source_id"]) for row in rows]
        quizzes = [str(row["prompt"][1]["content"]) for row in rows]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"Converted Logic-RL {split} split contains duplicate source IDs.")
        if len(quizzes) != len(set(quizzes)):
            raise ValueError(f"Converted Logic-RL {split} split contains duplicate quizzes.")
        difficulty_counts = Counter(int(row["metadata"]["difficulty"]) for row in rows)
        per_difficulty = expected[split] // len(DIFFICULTIES)
        expected_counts = Counter({difficulty: per_difficulty for difficulty in DIFFICULTIES})
        if difficulty_counts != expected_counts:
            raise ValueError(
                f"Unexpected {split} difficulty counts: expected {dict(expected_counts)}, "
                f"observed {dict(difficulty_counts)}."
            )
        split_quizzes[split] = set(quizzes)
    overlap = split_quizzes["train"] & split_quizzes["validation"]
    if overlap:
        raise ValueError(f"Logic-RL train/validation quiz overlap detected ({len(overlap)} rows).")


def atomic_write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def augment_manifest(base_manifest_path: Path, output_manifest_path: Path, train_path: Path) -> dict[str, Any]:
    with base_manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream) or {}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), list):
        raise ValueError("--base-manifest must contain a `sources` list.")
    if not manifest["sources"]:
        raise ValueError("--base-manifest must contain at least one source.")

    existing_logic = []
    for source in manifest["sources"]:
        if not isinstance(source, dict):
            raise ValueError("Every base-manifest source must be a mapping.")
        metadata = source.get("metadata") or {}
        if source.get("name") == "logic" or metadata.get("task_name") == "logic":
            existing_logic.append(source.get("name"))
    if existing_logic:
        raise ValueError(
            "--base-manifest already contains a Logic task; pass the original M2RL manifest "
            "instead of an already augmented manifest."
        )

    output = dict(manifest)
    output["sources"] = [dict(source) for source in manifest["sources"]]
    output["sources"].append(
        {
            "name": "logic",
            "path": str(train_path.resolve()),
            "input_key": "prompt",
            "label_key": "label",
            "metadata_key": "metadata",
            "tool_key": "tools",
            "apply_chat_template": True,
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "rm_type": "kk",
            "weight": 1.0,
            "metadata": {
                "task_name": "logic",
                "rm_type": "kk",
                "dataset": "logic_kk",
            },
        }
    )
    atomic_write_yaml(output, output_manifest_path)
    return output


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    base_manifest = args.base_manifest.resolve()
    output_manifest = args.output_manifest.resolve()
    if base_manifest == output_manifest:
        raise ValueError("--output-manifest must differ from --base-manifest; the pinned base is immutable.")
    if args.download_retries <= 0:
        raise ValueError("--download-retries must be positive.")
    output_dir = args.output_dir.resolve()
    raw_dir = (args.raw_dir or (output_dir / "raw")).resolve()
    source_dir = args.source_dir.resolve() if args.source_dir is not None else None
    source_paths = resolve_sources(
        raw_dir,
        source_dir=source_dir,
        offline=args.offline,
        retries=args.download_retries,
    )
    train_rows, validation_rows = load_converted_rows(source_paths)
    validate_splits(train_rows, validation_rows)

    train_path = output_dir / "logic_train.jsonl"
    validation_path = output_dir / "logic_validation.jsonl"
    atomic_write_jsonl(train_rows, train_path)
    atomic_write_jsonl(validation_rows, validation_path)
    augment_manifest(base_manifest, output_manifest, train_path)

    source_files = []
    for difficulty in DIFFICULTIES:
        for split in ("train", "test"):
            key = (difficulty, split)
            source_files.append(
                {
                    "difficulty": difficulty,
                    "split": split,
                    "rows": EXPECTED_ROWS_PER_FILE[split],
                    "url": source_url(difficulty, split),
                    "path": str(source_paths[key]),
                    "sha256": SOURCE_SHA256[key],
                }
            )
    index = {
        "schema_version": 1,
        "dataset": "logic_kk",
        "task": "logic",
        "split_policy": (
            "Concatenate Logic-RL 3--7ppl train.parquet files for training and "
            "their committed test.parquet files for online validation."
        ),
        "upstream": {
            "repository": LOGIC_RL_REPOSITORY,
            "revision": LOGIC_RL_REVISION,
            "dataset": K_AND_K_DATASET,
            "license": K_AND_K_LICENSE,
            "files": source_files,
        },
        "outputs": {
            "train": {
                "path": str(train_path.resolve()),
                "rows": len(train_rows),
                "sha256": sha256_file(train_path),
                "difficulties": list(DIFFICULTIES),
            },
            "validation": {
                "path": str(validation_path.resolve()),
                "rows": len(validation_rows),
                "sha256": sha256_file(validation_path),
                "difficulties": list(DIFFICULTIES),
            },
            "augmented_manifest": {
                "path": str(output_manifest),
                "sha256": sha256_file(output_manifest),
            },
        },
    }
    atomic_write_json(index, output_dir / "logic_data_index.json")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Use pre-downloaded verified files under DIFFICULTYppl/SPLIT.parquet (or D_SPLIT.parquet).",
    )
    parser.add_argument("--offline", action="store_true", help="Require all pinned sources to exist in --raw-dir.")
    parser.add_argument("--download-retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))
