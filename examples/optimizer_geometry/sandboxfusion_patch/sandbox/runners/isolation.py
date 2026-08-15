# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Derived from SandboxFusion server-20250609.  The cgroup v2 and fail-closed
# cleanup changes are maintained by the ICLR 2027 experiment integration.

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections.abc import AsyncIterator

import aiofiles.os
from sandbox.runners.cgroup_v2 import CgroupV2Manager


def resource_id() -> str:
    return secrets.token_hex(8)


@dataclass(frozen=True)
class CgroupCommand:
    """Command prefix that moves a runner into its resource cgroup."""

    command_prefix: list[str]
    path: str


async def execute_command(cmd: list[str], raise_nonzero: bool = True):
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0 and raise_nonzero:
        raise RuntimeError(f'Failed to execute {" ".join(cmd)}: {stdout.decode()}\n{stderr.decode()}')
    return process.returncode, stdout, stderr


async def mount_tmpfs(mount_point: str):
    await execute_command(["mount", "-t", "tmpfs", "tmpfs", mount_point])


async def unmount_fs(mount_point: str):
    # Never use lazy unmount here: it can report success while references and
    # the detached mount remain alive, defeating teardown attestation.
    return await execute_command(["umount", mount_point], raise_nonzero=False)


@asynccontextmanager
async def tmp_cgroup_view(root: str, cgroup_path: str) -> AsyncIterator[None]:
    """Expose only this execution's cgroup controls, read-only, in its chroot."""

    target = f"{root}/sys/fs/cgroup"
    mounted = False
    try:
        await execute_command(["mount", "--bind", cgroup_path, target])
        mounted = True
        await execute_command(
            [
                "mount",
                "-o",
                "remount,bind,ro,nosuid,nodev,noexec",
                target,
            ]
        )
        yield
    finally:
        if mounted:
            returncode, _, stderr = await unmount_fs(target)
            if returncode != 0:
                raise RuntimeError(f"failed to unmount cgroup view {target}: {stderr.decode()}")


@asynccontextmanager
async def tmp_overlayfs() -> AsyncIterator[str]:
    """Create a fresh overlay root and always tear it down after one request."""

    base_dir = f"/tmp/overlay_{resource_id()}"
    merged_dir = f"{base_dir}/merged"
    tmpfs_dir = f"{base_dir}/tmpfs"
    upper_dir = f"{tmpfs_dir}/upper"
    work_dir = f"{tmpfs_dir}/work"
    tmpfs_mounted = False
    merged_mounted = False
    proc_mounted = False
    sys_mounted = False
    dev_mounted = False
    sandbox_tmp_mounted = False
    sandbox_run_mounted = False
    sandbox_var_tmp_mounted = False
    try:
        for sub_dir in [tmpfs_dir, merged_dir]:
            await aiofiles.os.makedirs(sub_dir)
        await mount_tmpfs(tmpfs_dir)
        tmpfs_mounted = True
        for sub_dir in [upper_dir, work_dir]:
            await aiofiles.os.makedirs(sub_dir)

        await execute_command(
            [
                "mount",
                "-t",
                "overlay",
                "overlay",
                "-o",
                f"lowerdir=/,upperdir={upper_dir},workdir={work_dir}",
                merged_dir,
            ]
        )
        merged_mounted = True
        await execute_command(
            [
                "mount",
                "-t",
                "tmpfs",
                "-o",
                "mode=1777,nosuid,nodev,size=4g",
                "tmpfs",
                f"{merged_dir}/tmp",
            ]
        )
        sandbox_tmp_mounted = True
        await execute_command(
            [
                "mount",
                "-t",
                "tmpfs",
                "-o",
                "mode=755,nosuid,nodev,noexec,size=16m",
                "tmpfs",
                f"{merged_dir}/run",
            ]
        )
        sandbox_run_mounted = True
        await execute_command(
            [
                "mount",
                "-t",
                "tmpfs",
                "-o",
                "mode=1777,nosuid,nodev,size=4g",
                "tmpfs",
                f"{merged_dir}/var/tmp",
            ]
        )
        sandbox_var_tmp_mounted = True
        # util-linux 2.34's --mount-proc first detaches an existing procfs and
        # fails with EINVAL when its target is only an empty directory. This
        # trusted outer procfs is replaced inside the new mount/PID namespace
        # before untrusted code starts, then removed during strict teardown.
        await execute_command(
            ["mount", "-t", "proc", "-o", "nosuid,nodev,noexec", "proc", f"{merged_dir}/proc"]
        )
        proc_mounted = True
        await execute_command(["mount", "-t", "sysfs", "-o", "ro,nosuid,nodev,noexec", "sysfs", f"{merged_dir}/sys"])
        sys_mounted = True
        await execute_command(
            ["mount", "-t", "tmpfs", "-o", "mode=755,nosuid,noexec,size=16m", "tmpfs", f"{merged_dir}/dev"]
        )
        dev_mounted = True
        for name, major, minor in [
            ("null", "1", "3"),
            ("zero", "1", "5"),
            ("random", "1", "8"),
            ("urandom", "1", "9"),
        ]:
            await execute_command(["mknod", "-m", "0666", f"{merged_dir}/dev/{name}", "c", major, minor])
        await execute_command(["mkdir", "-m", "1777", "-p", f"{merged_dir}/dev/shm"])
        for name, target in [
            ("fd", "/proc/self/fd"),
            ("stdin", "/proc/self/fd/0"),
            ("stdout", "/proc/self/fd/1"),
            ("stderr", "/proc/self/fd/2"),
        ]:
            await execute_command(["ln", "-s", target, f"{merged_dir}/dev/{name}"])
        await execute_command(["cp", "/etc/hosts", f"{merged_dir}/etc/"])
        await execute_command(["cp", "/etc/resolv.conf", f"{merged_dir}/etc/"])
        yield merged_dir
    finally:
        cleanup_errors = []
        if proc_mounted:
            returncode, _, stderr = await unmount_fs(f"{merged_dir}/proc")
            if returncode != 0:
                cleanup_errors.append(stderr.decode())
        if dev_mounted:
            returncode, _, stderr = await unmount_fs(f"{merged_dir}/dev")
            if returncode != 0:
                cleanup_errors.append(stderr.decode())
        if sys_mounted:
            returncode, _, stderr = await unmount_fs(f"{merged_dir}/sys")
            if returncode != 0:
                cleanup_errors.append(stderr.decode())
        if sandbox_tmp_mounted:
            returncode, _, stderr = await unmount_fs(f"{merged_dir}/tmp")
            if returncode != 0:
                cleanup_errors.append(stderr.decode())
        if sandbox_var_tmp_mounted:
            returncode, _, stderr = await unmount_fs(f"{merged_dir}/var/tmp")
            if returncode != 0:
                cleanup_errors.append(stderr.decode())
        if sandbox_run_mounted:
            returncode, _, stderr = await unmount_fs(f"{merged_dir}/run")
            if returncode != 0:
                cleanup_errors.append(stderr.decode())
        if merged_mounted:
            returncode, _, stderr = await unmount_fs(merged_dir)
            if returncode != 0:
                cleanup_errors.append(stderr.decode())
        if tmpfs_mounted:
            returncode, _, stderr = await unmount_fs(tmpfs_dir)
            if returncode != 0:
                cleanup_errors.append(stderr.decode())
        if not cleanup_errors:
            try:
                await asyncio.to_thread(shutil.rmtree, base_dir)
            except OSError as exc:
                cleanup_errors.append(f"failed to remove overlay directory {base_dir}: {exc}")
        if cleanup_errors:
            raise RuntimeError("failed to tear down sandbox mounts: " + " | ".join(cleanup_errors))


@asynccontextmanager
async def tmp_cgroup(
    mem_limit: str | None = None,
    cpu_limit: float | None = None,
) -> AsyncIterator[CgroupCommand]:
    """Create a per-request cgroup in the delegated cgroup v2 subtree."""

    if mem_limit is None and cpu_limit is None:
        raise RuntimeError("every cgroup resource is unlimited; refusing unsafe execution")

    manager = CgroupV2Manager()
    handle = manager.create(
        memory_limit=mem_limit or "4G",
        cpu_limit=cpu_limit or 1.0,
        pids_limit=int(os.environ.get("SANDBOX_PIDS_LIMIT", "512")),
    )
    try:
        yield CgroupCommand(command_prefix=handle.command_prefix, path=str(handle.path))
    finally:
        await asyncio.to_thread(manager.destroy, handle)


@asynccontextmanager
async def tmp_netns(no_bridge: bool = True) -> AsyncIterator[str]:
    """Create a loopback-only network namespace for untrusted code."""

    if not no_bridge:
        raise RuntimeError("network-enabled lite sandboxes are disabled by policy")
    name = f"sandbox_{resource_id()}"
    created = False
    try:
        await execute_command(["ip", "netns", "add", name])
        created = True
        await execute_command(["ip", "netns", "exec", name, "ip", "link", "set", "lo", "up"])
        yield name
    finally:
        if created:
            returncode, _, stderr = await execute_command(["ip", "netns", "delete", name], raise_nonzero=False)
            if returncode != 0:
                raise RuntimeError(f"failed to delete network namespace {name}: {stderr.decode()}")
