#!/usr/bin/env python3
"""Validate that SandboxFusion can stage the largest LiveCodeBench row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


MAX_COMPLETION_BYTES = 8 * 1024 * 1024
# The pinned evaluator currently stages a roughly 29 KiB testing utility next
# to the test payload and extracted completion. Keep a generous fixed reserve
# so an upstream utility update cannot silently consume the whole budget.
RUNNER_SUPPORT_BYTES = 1024 * 1024


def required_upload_bytes(sandboxfusion_row: str | dict[str, Any]) -> int:
    if isinstance(sandboxfusion_row, str):
        sandboxfusion_row = json.loads(sandboxfusion_row)
    if not isinstance(sandboxfusion_row, dict):
        raise ValueError("sandboxfusion_row must decode to an object")
    test = sandboxfusion_row.get("test")
    if not isinstance(test, str):
        raise ValueError("sandboxfusion_row.test must be a JSON string")
    test_cases = json.loads(test)
    if not isinstance(test_cases, dict):
        raise ValueError("sandboxfusion_row.test must decode to an object")
    staged_test_bytes = len(json.dumps(test_cases).encode("utf-8"))
    return staged_test_bytes + MAX_COMPLETION_BYTES + RUNNER_SUPPORT_BYTES


def max_required_upload_bytes(rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    maximum = 0
    maximum_problem_id = ""
    for row in rows:
        metadata = row.get("metadata") or {}
        required = required_upload_bytes(metadata.get("sandboxfusion_row"))
        if required > maximum:
            maximum = required
            maximum_problem_id = str(metadata.get("question_id") or "")
    return maximum, maximum_problem_id


def parquet_requirements(path: Path) -> tuple[int, str]:
    import pyarrow.parquet as pq

    maximum = 0
    maximum_problem_id = ""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1, columns=["metadata"]):
        required, problem_id = max_required_upload_bytes(batch.to_pylist())
        if required > maximum:
            maximum = required
            maximum_problem_id = problem_id
    if not maximum:
        raise ValueError(f"LiveCodeBench parquet contains no rows: {path}")
    return maximum, maximum_problem_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--max-upload-bytes", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    required, problem_id = parquet_requirements(args.parquet)
    report = {
        "max_upload_bytes": args.max_upload_bytes,
        "max_required_upload_bytes": required,
        "max_problem_id": problem_id,
        "parquet": str(args.parquet.resolve()),
        "safe": required <= args.max_upload_bytes,
    }
    print(json.dumps(report, sort_keys=True))
    if not report["safe"]:
        raise SystemExit(
            f"SandboxFusion upload limit {args.max_upload_bytes} is smaller than "
            f"the required {required} bytes for LiveCodeBench problem {problem_id}."
        )
