# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

import base64
import json
import os
import shutil
import tempfile

from sandbox.runners.base import restore_files, run_commands
from sandbox.runners.major import get_python_rt_env
from sandbox.runners.types import CodeRunArgs, CommandRunStatus, RunJupyterRequest, RunJupyterResult
from sandbox.utils.execution import get_tmp_dir


async def run_jupyter(args: RunJupyterRequest) -> RunJupyterResult:
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        deps_dir = os.path.abspath(os.path.join(__file__, "../../../runtime/jupyter"))
        shutil.copy2(os.path.join(deps_dir, "main.py"), tmp_dir)
        output_name = "tmp/sandbox/configs/output.json"
        input_path = os.path.join(tmp_dir, "tmp/sandbox/configs/input.json")
        os.makedirs(os.path.dirname(input_path))
        with open(input_path, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "kernel": args.kernel,
                    "cells": args.cells,
                    "cell_timeout": args.cell_timeout,
                    "total_timeout": args.total_timeout,
                },
                stream,
                indent=2,
            )

        driver = await run_commands(
            None,
            "python main.py",
            tmp_dir,
            get_python_rt_env("sandbox-runtime"),
            CodeRunArgs(
                code="",
                run_timeout=args.total_timeout + 10,
                memory_limit_MB=args.memory_limit_MB,
                fetch_files=args.fetch_files + [output_name],
            ),
        )
        if driver.run_result is None or driver.run_result.status != CommandRunStatus.Finished:
            return RunJupyterResult(
                status=CommandRunStatus.Error,
                driver=driver.run_result
                or driver.compile_result
                or {"status": CommandRunStatus.Error, "stderr": "runner returned no result"},
            )
        if output_name not in driver.files:
            return RunJupyterResult(
                status=CommandRunStatus.Error,
                driver=driver.run_result,
                cells=[],
                files=driver.files,
            )
        output = json.loads(base64.b64decode(driver.files.pop(output_name).encode()).decode())
        return RunJupyterResult(
            status=output["status"],
            driver=driver.run_result,
            cells=output["cells"],
            files=driver.files,
        )
