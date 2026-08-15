# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

# GPU runners in server-20250609 call run_command_bare directly, C# initializes
# a project outside isolation, and Lean explicitly disables PID isolation. This
# hardened profile intentionally does not register any of those runners.
from sandbox.runners.jupyter import run_jupyter
from sandbox.runners.major import MAJOR_RUNNERS
from sandbox.runners.minor import MINOR_RUNNERS
from sandbox.runners.types import (  # nopycln: import
    CellRunResult,
    CodeRunArgs,
    CodeRunResult,
    CommandRunResult,
    CommandRunStatus,
    Language,
    RunJupyterRequest,
)

DISABLED_RUNNERS = {"csharp", "lean"}
CODE_RUNNERS = {
    name: runner for name, runner in {**MAJOR_RUNNERS, **MINOR_RUNNERS}.items() if name not in DISABLED_RUNNERS
}

__all__ = [
    "CODE_RUNNERS",
    "CodeRunArgs",
    "CodeRunResult",
    "CommandRunResult",
    "CommandRunStatus",
    "RunJupyterRequest",
    "Language",
    "CellRunResult",
    "run_jupyter",
]
