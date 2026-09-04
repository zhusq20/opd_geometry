#!/usr/bin/env python3
"""Validate that SandboxFusion can stage the largest LiveCodeBench row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


MAX_COMPLETION_BYTES = 8 * 1024 * 1024
MAX_TEST_PAYLOAD_BYTES = 192 * 1024 * 1024
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


def encoded_test_payload_bytes(sandboxfusion_row: str | dict[str, Any]) -> int:
    if isinstance(sandboxfusion_row, str):
        sandboxfusion_row = json.loads(sandboxfusion_row)
    if not isinstance(sandboxfusion_row, dict):
        raise ValueError("sandboxfusion_row must decode to an object")
    test = sandboxfusion_row.get("test")
    if not isinstance(test, str):
        raise ValueError("sandboxfusion_row.test must be a JSON string")
    return len(test.encode("utf-8"))


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


def parquet_requirements(path: Path) -> tuple[int, str, int, str]:
    import pyarrow.parquet as pq

    maximum = 0
    maximum_problem_id = ""
    maximum_test_payload = 0
    maximum_test_payload_problem_id = ""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1, columns=["metadata"]):
        rows = batch.to_pylist()
        required, problem_id = max_required_upload_bytes(rows)
        if required > maximum:
            maximum = required
            maximum_problem_id = problem_id
        metadata = rows[0].get("metadata") or {}
        payload_bytes = encoded_test_payload_bytes(metadata.get("sandboxfusion_row"))
        if payload_bytes > maximum_test_payload:
            maximum_test_payload = payload_bytes
            maximum_test_payload_problem_id = str(metadata.get("question_id") or "")
    if not maximum:
        raise ValueError(f"LiveCodeBench parquet contains no rows: {path}")
    return maximum, maximum_problem_id, maximum_test_payload, maximum_test_payload_problem_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--max-upload-bytes", type=int, required=True)
    parser.add_argument("--max-test-payload-bytes", type=int, default=MAX_TEST_PAYLOAD_BYTES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    required, problem_id, test_payload, test_payload_problem_id = parquet_requirements(args.parquet)
    report = {
        "max_upload_bytes": args.max_upload_bytes,
        "max_required_upload_bytes": required,
        "max_problem_id": problem_id,
        "max_test_payload_bytes": args.max_test_payload_bytes,
        "max_observed_test_payload_bytes": test_payload,
        "max_test_payload_problem_id": test_payload_problem_id,
        "parquet": str(args.parquet.resolve()),
        "safe": required <= args.max_upload_bytes and test_payload <= args.max_test_payload_bytes,
    }
    print(json.dumps(report, sort_keys=True))
    if required > args.max_upload_bytes:
        raise SystemExit(
            f"SandboxFusion upload limit {args.max_upload_bytes} is smaller than "
            f"the required {required} bytes for LiveCodeBench problem {problem_id}."
        )
    if test_payload > args.max_test_payload_bytes:
        raise SystemExit(
            f"SandboxFusion test payload limit {args.max_test_payload_bytes} is smaller than "
            f"the required {test_payload} bytes for LiveCodeBench problem {test_payload_problem_id}."
        )
