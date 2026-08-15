# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

import os
import traceback
from enum import Enum

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from sandbox.runners import (
    CODE_RUNNERS,
    CellRunResult,
    CodeRunArgs,
    CodeRunResult,
    CommandRunResult,
    CommandRunStatus,
    Language,
    RunJupyterRequest,
    run_jupyter,
)

sandbox_router = APIRouter()
logger = structlog.stdlib.get_logger()


class RunCodeRequest(BaseModel):
    compile_timeout: float = Field(10, gt=0, le=60, description="compile timeout")
    # LiveCodeBench's official evaluator amortizes a per-case alarm across all
    # cases and caps the enclosing runner at 160 seconds.
    run_timeout: float = Field(10, gt=0, le=160, description="code run timeout")
    memory_limit_MB: int = Field(4096, ge=16, le=4096, description="maximum memory allowed in MiB")
    code: str = Field(
        ...,
        min_length=1,
        max_length=8 * 1024 * 1024,
        examples=['print("hello")'],
        description="the code to run",
    )
    stdin: str | None = Field(None, max_length=8 * 1024 * 1024, examples=[""], description="optional stdin")
    language: Language = Field(..., examples=["python"], description="execution language")
    files: dict[str, str | None] = Field({}, max_length=256, description="base64-encoded input files")
    fetch_files: list[str] = Field([], max_length=256, description="file paths to fetch after execution")


class RunStatus(str, Enum):
    Success = "Success"
    Failed = "Failed"
    SandboxError = "SandboxError"


class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: CommandRunResult | None = None
    run_result: CommandRunResult | None = None
    executor_pod_name: str | None = None
    files: dict[str, str] = {}


class RunJupyterResponse(BaseModel):
    status: RunStatus
    message: str
    driver: CommandRunResult | None = None
    cells: list[CellRunResult] = []
    executor_pod_name: str | None = None
    files: dict[str, str] = {}


def parse_run_status(result: CodeRunResult) -> tuple[RunStatus, str]:
    outcomes = []
    return_codes = []
    errors = []
    for command_result in (result.compile_result, result.run_result):
        if command_result is None:
            continue
        outcomes.append(command_result.status)
        errors.append(command_result.stderr or "")
        if command_result.return_code is not None:
            return_codes.append(command_result.return_code)
    for outcome, error in zip(outcomes, errors, strict=True):
        if outcome == CommandRunStatus.Error:
            return RunStatus.SandboxError, error
    if any(outcome == CommandRunStatus.TimeLimitExceeded for outcome in outcomes):
        return RunStatus.Failed, ""
    if any(return_code != 0 for return_code in return_codes):
        return RunStatus.Failed, ""
    if not outcomes:
        return RunStatus.SandboxError, "runner returned no command result"
    return RunStatus.Success, ""


@sandbox_router.post("/run_code", response_model=RunCodeResponse, tags=["sandbox"])
async def run_code(request: RunCodeRequest):
    response = RunCodeResponse(
        status=RunStatus.Success,
        message="",
        executor_pod_name=os.environ.get("MY_POD_NAME"),
    )
    try:
        logger.debug(
            "start processing code request",
            language=request.language,
            file_names=list(request.files.keys()),
            memory_limit_MB=request.memory_limit_MB,
        )
        result = await CODE_RUNNERS[request.language](CodeRunArgs(**request.model_dump()))
        response.compile_result = result.compile_result
        response.run_result = result.run_result
        response.files = result.files
        response.status, response.message = parse_run_status(result)
    except Exception as exc:
        response.message = (
            f"exception on running {request.language} code: {exc} " f"{traceback.print_tb(exc.__traceback__)}"
        )
        logger.warning(response.message)
        response.status = RunStatus.SandboxError
    return response


@sandbox_router.post("/run_jupyter", name="Run Code in Jupyter", response_model=RunJupyterResponse, tags=["sandbox"])
async def run_jupyter_handler(request: RunJupyterRequest):
    response = RunJupyterResponse(
        status=RunStatus.Success,
        message="",
        executor_pod_name=os.environ.get("MY_POD_NAME"),
    )
    try:
        result = await run_jupyter(request)
        response.driver = result.driver
        if result.status != CommandRunStatus.Finished:
            response.status = RunStatus.Failed
        else:
            response.cells = result.cells
            response.files = result.files
    except Exception as exc:
        response.message = f"exception on running jupyter: {exc} {traceback.print_tb(exc.__traceback__)}"
        logger.warning(response.message)
        response.status = RunStatus.SandboxError
    return response
