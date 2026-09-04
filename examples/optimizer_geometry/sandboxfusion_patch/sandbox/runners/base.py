# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Derived from SandboxFusion server-20250609.  This version places every lite
# execution in a cgroup v2 group, denies network access, and drops privileges
# before evaluating untrusted code.

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import time
import traceback

import structlog

from sandbox.configs.run_config import RunConfig
from sandbox.runners.file_security import (
    copy_worktree_into_overlay,
    read_regular_file_beneath,
    restore_base64_files,
)
from sandbox.runners.isolation import tmp_cgroup, tmp_cgroup_view, tmp_netns, tmp_overlayfs
from sandbox.runners.types import CodeRunArgs, CodeRunResult, CommandRunResult, CommandRunStatus
from sandbox.utils.execution import get_output_non_blocking

logger = structlog.stdlib.get_logger()
config = RunConfig.get_instance_sync()

MAX_MEMORY_LIMIT_MB = int(os.environ.get("SANDBOX_MAX_MEMORY_MB", "4096"))
RUN_UID = int(os.environ.get("SANDBOX_RUN_UID", "1000"))
MAX_UPLOAD_BYTES = int(os.environ.get("SANDBOX_MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))
MAX_FETCH_BYTES = int(os.environ.get("SANDBOX_MAX_FETCH_BYTES", str(64 * 1024 * 1024)))
SAFE_ENV_KEYS = {
    "DOTNET_ROOT",
    "JAVA_HOME",
    "LANG",
    "LC_ALL",
    "NODE_PATH",
    "OMP_NUM_THREADS",
    "PATH",
    "QT_QPA_PLATFORM",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
}


async def run_command_bare(
    command: str | list[str],
    timeout: float = 10,
    stdin: str | None = None,
    cwd: str | None = None,
    extra_env: dict[str, str] | None = None,
    use_exec: bool = False,
    preexec_fn=None,
) -> CommandRunResult:
    try:
        logger.debug(f"running command {command}")
        child_env = {name: os.environ[name] for name in SAFE_ENV_KEYS if name in os.environ}
        child_env.update(extra_env or {})
        child_env.update(
            {
                "HOME": "/home/app",
                "LOGNAME": "app",
                "TMPDIR": "/tmp",
                "USER": "app",
                "XDG_CACHE_HOME": "/tmp/.cache",
            }
        )
        if use_exec:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env,
                preexec_fn=preexec_fn,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                executable="/bin/bash",
                env=child_env,
                preexec_fn=preexec_fn,
            )
        if stdin is not None:
            process.stdin.write(stdin.encode())
        process.stdin.close()
        start_time = time.time()
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
            execution_time = time.time() - start_time
            logger.debug(f"stop running command {command}")
        except asyncio.TimeoutError:
            return CommandRunResult(
                status=CommandRunStatus.TimeLimitExceeded,
                execution_time=time.time() - start_time,
                stdout=await get_output_non_blocking(process.stdout),
                stderr=await get_output_non_blocking(process.stderr),
            )
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    logger.warning(f"timed out reaping process: {process.pid}")
                logger.info(f"process killed: {process.pid}")
            # Per-request cgroup cleanup terminates residual descendants after
            # compile/run. The upstream container-wide process scan is unsafe
            # with multiple workers and is unnecessary with this isolation.

        return CommandRunResult(
            status=CommandRunStatus.Finished,
            execution_time=execution_time,
            return_code=process.returncode,
            stdout=await get_output_non_blocking(process.stdout),
            stderr=await get_output_non_blocking(process.stderr),
        )
    except Exception as exc:
        message = f"exception on running command {command}: {exc} | " f"{traceback.print_tb(exc.__traceback__)}"
        logger.warning(message)
        return CommandRunResult(status=CommandRunStatus.Error, stderr=message)


def _effective_memory_limit_mb(args: CodeRunArgs) -> int:
    requested = int(getattr(args, "memory_limit_MB", MAX_MEMORY_LIMIT_MB))
    if requested <= 0:
        raise ValueError("lite isolation refuses an unlimited memory request")
    if requested > MAX_MEMORY_LIMIT_MB:
        raise ValueError(
            f"requested memory limit {requested} MiB exceeds server maximum " f"{MAX_MEMORY_LIMIT_MB} MiB"
        )
    return requested


def _drop_privilege_prefix() -> list[str]:
    return [
        "/usr/local/libexec/sandboxfusion-restrict-exec",
        "--drop-to",
        str(RUN_UID),
        str(RUN_UID),
        "--",
    ]


async def run_commands(
    compile_command: str | None,
    run_command: str,
    cwd: str,
    extra_env: dict[str, str] | None,
    args: CodeRunArgs,
    **kwargs,
) -> CodeRunResult:
    files = {}
    compile_res = None
    run_res = None

    if config.sandbox.isolation == "none":
        raise RuntimeError("isolation=none is disabled in the patched SandboxFusion image")

    if config.sandbox.isolation != "lite":
        raise RuntimeError(f"unsupported sandbox isolation mode: {config.sandbox.isolation}")
    if kwargs.get("disable_pid_isolation", False):
        raise RuntimeError("disabling PID isolation is forbidden by the patched lite profile")

    memory_limit_mb = _effective_memory_limit_mb(args)
    async with tmp_overlayfs() as root:
        sandbox_cwd = await asyncio.to_thread(copy_worktree_into_overlay, root, cwd)
        async with (
            tmp_cgroup(mem_limit=f"{memory_limit_mb}M", cpu_limit=1.0) as cgroup,
            tmp_cgroup_view(root, cgroup.path),
            tmp_netns(no_bridge=True) as netns,
        ):
            prefix = list(cgroup.command_prefix)
            prefix += ["ip", "netns", "exec", netns]
            prefix += [
                "chroot",
                root,
                "unshare",
                "--pid",
                "--ipc",
                "--uts",
                "--fork",
                "--mount",
                "--mount-proc=/proc",
            ]
            prefix += _drop_privilege_prefix()

            if compile_command is not None:
                compile_res = await run_command_bare(
                    prefix + ["bash", "-c", f"cd {cwd} && {compile_command}"],
                    args.compile_timeout,
                    None,
                    cwd,
                    extra_env,
                    True,
                )
            if compile_res is None or (
                compile_res.status == CommandRunStatus.Finished and compile_res.return_code == 0
            ):
                run_res = await run_command_bare(
                    prefix + ["bash", "-c", f"cd {cwd} && {run_command}"],
                    args.run_timeout,
                    args.stdin,
                    cwd,
                    extra_env,
                    True,
                )

        # tmp_cgroup has now killed every residual task. Fetching from the
        # still-mounted overlay is therefore not subject to symlink races.
        fetched_bytes = 0
        for filename in args.fetch_files:
            content = read_regular_file_beneath(
                sandbox_cwd,
                filename,
                max_bytes=MAX_FETCH_BYTES - fetched_bytes,
            )
            if content is not None:
                fetched_bytes += len(content)
                files[filename] = base64.b64encode(content).decode("utf-8")
        return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)


def restore_files(directory: str, files: dict[str, str | None]):
    restore_base64_files(directory, files, max_bytes=MAX_UPLOAD_BYTES)
