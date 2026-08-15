# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Fail-closed online-judge API for the training deployment.

The upstream endpoint is intentionally generic.  In particular it accepts
Python in ``custom_extract_logic`` and legacy pickle-encoded LiveCodeBench
tests.  Both are unsafe when the request is supplied by an API client because
they are processed by the control plane before the generated program reaches
the sandbox.  This deployment only needs the JSON LiveCodeBench submission
path, so validate that narrow contract before dispatching it.
"""

from __future__ import annotations

import json
import math
from typing import Any

from fastapi import APIRouter, HTTPException

from sandbox.datasets.types import (
    CodingDataset,
    EvalResult,
    GetMetricsFunctionRequest,
    GetMetricsFunctionResult,
    GetMetricsRequest,
    GetPromptByIdRequest,
    GetPromptsRequest,
    Prompt,
    SubmitRequest,
    TestConfig,
)
from sandbox.registry import get_all_dataset_ids, get_coding_class_by_dataset, get_coding_class_by_name

oj_router = APIRouter()

MAX_COMPLETION_BYTES = 8 * 1024 * 1024
MAX_PROMPT_BYTES = 8 * 1024 * 1024
MAX_LABELS_BYTES = 1024 * 1024
MAX_TEST_PAYLOAD_BYTES = 48 * 1024 * 1024
MAX_TEST_CASES = 2048
MAX_CASE_BYTES = 8 * 1024 * 1024
MAX_CASE_TIMEOUT_SECONDS = 30.0
LIVE_CODE_BENCH_CLASS = "LiveCodeBenchDataset"
LIVE_CODE_BENCH_ROW_KEYS = {"id", "labels", "content", "test"}


def get_dataset_cls(dataset_id: str, config: TestConfig | None = None) -> CodingDataset:
    internal_cls = get_coding_class_by_dataset(dataset_id)
    if internal_cls is not None:
        return internal_cls
    if config is None or config.dataset_type is None:
        raise HTTPException(status_code=400, detail=f"no eval class found for dataset {dataset_id}")
    config_cls = get_coding_class_by_name(config.dataset_type)
    if config_cls is None:
        raise HTTPException(status_code=400, detail=f"eval class {config.dataset_type} not found")
    return config_cls


def _json_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{field} must be a JSON string")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field} must contain valid JSON") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(status_code=400, detail=f"{field} must decode to an object")
    return decoded


def _bounded_text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{field} must be text")
    if len(value.encode("utf-8")) > maximum:
        raise HTTPException(status_code=413, detail=f"{field} exceeds the deployment limit")
    return value


def _validate_livecodebench_submit(request: SubmitRequest, dataset: CodingDataset) -> None:
    config = request.config
    if getattr(dataset, "__name__", "") != LIVE_CODE_BENCH_CLASS:
        raise HTTPException(status_code=400, detail="only LiveCodeBenchDataset submissions are enabled")
    if config.dataset_type != LIVE_CODE_BENCH_CLASS:
        raise HTTPException(status_code=400, detail="dataset_type must be LiveCodeBenchDataset")
    if config.custom_extract_logic is not None:
        raise HTTPException(status_code=400, detail="custom extraction code is disabled")
    if (
        any(value is not None for value in (config.language, config.locale, config.is_fewshot, config.compile_timeout))
        or config.extra
    ):
        raise HTTPException(status_code=400, detail="unsupported LiveCodeBench submit configuration")

    timeout = 6.0 if config.run_timeout is None else float(config.run_timeout)
    if not math.isfinite(timeout) or not 0 < timeout <= MAX_CASE_TIMEOUT_SECONDS:
        raise HTTPException(status_code=400, detail="run_timeout is outside the allowed range")

    row = config.provided_data
    if not isinstance(row, dict) or set(row) != LIVE_CODE_BENCH_ROW_KEYS:
        raise HTTPException(status_code=400, detail="provided_data must be one exact LiveCodeBench row")
    if str(row["id"]) != str(request.id):
        raise HTTPException(status_code=400, detail="provided_data id does not match submission id")

    _bounded_text(request.completion, "completion", MAX_COMPLETION_BYTES)
    _bounded_text(row["content"], "provided_data.content", MAX_PROMPT_BYTES)
    encoded_labels = _bounded_text(row["labels"], "provided_data.labels", MAX_LABELS_BYTES)
    labels = _json_object(encoded_labels, "provided_data.labels")
    if not isinstance(labels, dict):  # pragma: no cover - documented invariant
        raise HTTPException(status_code=400, detail="provided_data.labels must decode to an object")

    encoded_test = _bounded_text(row["test"], "provided_data.test", MAX_TEST_PAYLOAD_BYTES)
    test_payload = _json_object(encoded_test, "provided_data.test")
    input_output = _json_object(test_payload.get("input_output"), "provided_data.test.input_output")
    if set(input_output) - {"inputs", "outputs", "fn_name"}:
        raise HTTPException(status_code=400, detail="unexpected LiveCodeBench test fields")
    inputs = input_output.get("inputs")
    outputs = input_output.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise HTTPException(status_code=400, detail="LiveCodeBench inputs and outputs must be lists")
    if not inputs or len(inputs) != len(outputs) or len(inputs) > MAX_TEST_CASES:
        raise HTTPException(status_code=400, detail="invalid LiveCodeBench test count")
    if not all(isinstance(item, str) for item in [*inputs, *outputs]):
        raise HTTPException(status_code=400, detail="LiveCodeBench cases must contain text")
    if any(len(item.encode("utf-8")) > MAX_CASE_BYTES for item in [*inputs, *outputs]):
        raise HTTPException(status_code=413, detail="a LiveCodeBench test case exceeds the deployment limit")
    if "fn_name" in input_output and not isinstance(input_output["fn_name"], str):
        raise HTTPException(status_code=400, detail="LiveCodeBench fn_name must be text")


@oj_router.get("/list_datasets", description="List all registered datasets", tags=["datasets"])
async def list_datasets() -> list[str]:
    return get_all_dataset_ids()


@oj_router.post("/list_ids", description="List all ids of a dataset", tags=["datasets"])
async def list_ids(request: GetPromptsRequest) -> list[int | str]:
    dataset = get_dataset_cls(request.dataset, request.config)
    return await dataset.get_ids(request)


@oj_router.post("/get_prompts", description="Get prompts of a dataset", tags=["datasets"])
async def get_prompt(request: GetPromptsRequest) -> list[Prompt]:
    dataset = get_dataset_cls(request.dataset, request.config)
    return await dataset.get_prompts(request)


@oj_router.post("/get_prompt_by_id", description="Get a single prompt given id", tags=["datasets"])
async def get_prompt_by_id(request: GetPromptByIdRequest) -> Prompt:
    dataset = get_dataset_cls(request.dataset, request.config)
    return await dataset.get_prompt_by_id(request)


@oj_router.post("/submit", description="Submit one JSON LiveCodeBench problem", tags=["datasets"])
async def submit(request: SubmitRequest) -> EvalResult:
    dataset = get_dataset_cls(request.dataset, request.config)
    _validate_livecodebench_submit(request, dataset)
    return await dataset.evaluate_single(request)


@oj_router.post("/get_metrics", description="Get metrics for problem results", tags=["datasets"])
async def get_metrics(request: GetMetricsRequest) -> dict[str, Any]:
    dataset = get_dataset_cls(request.dataset, request.config)
    if hasattr(dataset, "get_metrics"):
        return await dataset.get_metrics(request.results)
    return {}


@oj_router.post("/get_metrics_function", description="Get the dataset metrics function", tags=["datasets"])
async def get_metrics_function(request: GetMetricsFunctionRequest) -> GetMetricsFunctionResult:
    dataset = get_dataset_cls(request.dataset, request.config)
    if hasattr(dataset, "get_metrics_function"):
        return GetMetricsFunctionResult(function=dataset.get_metrics_function())
    return GetMetricsFunctionResult(function=None)
