# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class CommandRunStatus(str, Enum):
    Finished = "Finished"
    Error = "Error"
    TimeLimitExceeded = "TimeLimitExceeded"


class CommandRunResult(BaseModel):
    status: CommandRunStatus
    execution_time: float | None = None
    return_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None


class CodeRunArgs(BaseModel):
    code: str
    files: dict[str, str | None] = {}
    compile_timeout: float = 10
    run_timeout: float = 10
    memory_limit_MB: int = 4096
    stdin: str | None = None
    fetch_files: list[str] = []


class CodeRunResult(BaseModel):
    compile_result: CommandRunResult | None = None
    run_result: CommandRunResult | None = None
    files: dict[str, str] = {}


class RunJupyterRequest(BaseModel):
    cells: list[str] = Field(
        ...,
        min_length=1,
        max_length=256,
        examples=[
            [
                "a = 123",
                "a",
                "print(a)",
                'import sys; sys.stderr.write("stderr message")',
                'raise RuntimeError("error message")',
            ]
        ],
        description="list of code blocks to run in jupyter notebook",
    )
    cell_timeout: float = Field(0, ge=0, le=60, description="max run time for each of the cells")
    total_timeout: float = Field(45, gt=0, le=300, description="max run time for all of the cells")
    memory_limit_MB: int = Field(4096, ge=16, le=4096, description="maximum memory allowed in MiB")
    kernel: Literal["python3"] = "python3"
    files: dict[str, str] = Field({}, max_length=256, description="base64-encoded input files")
    fetch_files: list[str] = Field([], max_length=256, description="file paths to fetch after execution")


class CellRunResult(BaseModel):
    stdout: str
    stderr: str
    display: list[dict[str, Any]]
    error: list[dict[str, Any]]


class RunJupyterResult(BaseModel):
    status: CommandRunStatus
    driver: CommandRunResult
    cells: list[CellRunResult] = []
    files: dict[str, str] = {}


Language = Literal[
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "bash",
    "typescript",
    "sql",
    "rust",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "swift",
    "racket",
]
compile_languages: list[Language] = ["cpp", "go", "java"]
cpu_languages: list[Language] = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "bash",
    "typescript",
    "sql",
    "rust",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "swift",
    "racket",
]
gpu_languages: list[Language] = []
