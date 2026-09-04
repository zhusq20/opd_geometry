#!/usr/bin/env python3
"""Prepare an empty, controller-delegated host cgroup for SandboxFusion."""

from __future__ import annotations

import os
import re
import signal
import time
from pathlib import Path

DEFAULT_HOST_ROOT = "/host-cgroup"
DEFAULT_NAME = "sandboxfusion"
EXECUTION_PREFIX = "sandboxfusion-"
REQUIRED_CONTROLLERS = {"cpu", "memory", "pids"}
DEFAULT_AGGREGATE_MEMORY_BYTES = 32 * 1024**3
DEFAULT_AGGREGATE_PIDS = 4096


def read_tokens(path: Path) -> set[str]:
    return set(path.read_text(encoding="utf-8").split())


def write_control(path: Path, value: str | int) -> None:
    path.write_text(f"{value}\n", encoding="utf-8")


def enable_controllers(path: Path, required: set[str]) -> None:
    available = read_tokens(path / "cgroup.controllers")
    missing = required - available
    if missing:
        raise RuntimeError(f"cgroup {path} does not expose controllers: {', '.join(sorted(missing))}")
    enabled = read_tokens(path / "cgroup.subtree_control")
    for controller in sorted(required - enabled):
        write_control(path / "cgroup.subtree_control", f"+{controller}")


def populated(path: Path) -> bool:
    events = path / "cgroup.events"
    if events.exists():
        values = dict(
            line.split(maxsplit=1) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        return values.get("populated", "0") != "0"
    return bool((path / "cgroup.procs").read_text(encoding="utf-8").strip())


def kill_member_pids(path: Path) -> None:
    for value in (path / "cgroup.procs").read_text(encoding="utf-8").split():
        try:
            os.kill(int(value), signal.SIGKILL)
        except ProcessLookupError:
            continue


def clean_execution_group(path: Path, timeout: float = 5.0) -> None:
    kill_file = path / "cgroup.kill"
    if populated(path):
        if kill_file.exists():
            write_control(kill_file, 1)
        else:
            kill_member_pids(path)
    deadline = time.monotonic() + timeout
    while populated(path) and time.monotonic() < deadline:
        if not kill_file.exists():
            kill_member_pids(path)
        time.sleep(0.02)
    if populated(path):
        raise RuntimeError(f"stale cgroup remains populated: {path}")
    path.rmdir()


def positive_integer(value: str | int, setting: str) -> int:
    text = str(value)
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise ValueError(f"{setting} must be a positive integer, got {value!r}")
    return int(text)


def prepare(
    host_root: Path,
    name: str,
    *,
    aggregate_memory_bytes: int = DEFAULT_AGGREGATE_MEMORY_BYTES,
    aggregate_pids: int = DEFAULT_AGGREGATE_PIDS,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
        raise ValueError(f"invalid SandboxFusion cgroup name: {name!r}")
    host_root = host_root.resolve(strict=True)
    if not (host_root / "cgroup.controllers").is_file():
        raise RuntimeError(f"host does not expose cgroup v2 at {host_root}")

    enable_controllers(host_root, REQUIRED_CONTROLLERS)
    delegated = host_root / name
    delegated.mkdir(mode=0o750, exist_ok=True)
    delegated.chmod(0o755)
    if (delegated / "cgroup.procs").read_text(encoding="utf-8").strip():
        raise RuntimeError(f"delegated SandboxFusion cgroup contains direct processes: {delegated}")

    for child in sorted(path for path in delegated.iterdir() if path.is_dir()):
        if not child.name.startswith(EXECUTION_PREFIX):
            raise RuntimeError(f"unexpected child in SandboxFusion cgroup: {child}")
        clean_execution_group(child)

    write_control(delegated / "memory.max", positive_integer(aggregate_memory_bytes, "aggregate memory"))
    if (delegated / "memory.swap.max").exists():
        write_control(delegated / "memory.swap.max", 0)
    write_control(delegated / "pids.max", positive_integer(aggregate_pids, "aggregate PID limit"))
    enable_controllers(delegated, REQUIRED_CONTROLLERS)
    return delegated


if __name__ == "__main__":
    result = prepare(
        Path(os.environ.get("SANDBOXFUSION_HOST_CGROUP_ROOT", DEFAULT_HOST_ROOT)),
        os.environ.get("SANDBOXFUSION_CGROUP_NAME", DEFAULT_NAME),
        aggregate_memory_bytes=positive_integer(
            os.environ.get("SANDBOXFUSION_AGGREGATE_MEMORY_BYTES", DEFAULT_AGGREGATE_MEMORY_BYTES),
            "SANDBOXFUSION_AGGREGATE_MEMORY_BYTES",
        ),
        aggregate_pids=positive_integer(
            os.environ.get("SANDBOXFUSION_AGGREGATE_PIDS", DEFAULT_AGGREGATE_PIDS),
            "SANDBOXFUSION_AGGREGATE_PIDS",
        ),
    )
    print(result)
