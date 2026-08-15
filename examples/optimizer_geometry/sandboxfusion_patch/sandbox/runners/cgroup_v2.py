"""Minimal, fail-closed cgroup v2 management for SandboxFusion runners.

The upstream ``server-20250609`` image uses libcgroup's cgroup v1 CLI.  This
module deliberately uses the kernel cgroup v2 filesystem API instead.  The
SandboxFusion service container must have the audited capability allowlist,
use the host cgroup namespace, and have only its delegated cgroup2 subtree
available read-write.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import signal
import time
from dataclasses import dataclass
from pathlib import Path

CGROUP_NAME_PREFIX = "sandboxfusion-"
DEFAULT_CGROUP_ROOT = "/sys/fs/cgroup/sandboxfusion"
DEFAULT_PIDS_LIMIT = 512
CPU_PERIOD_US = 100_000


class CgroupV2Error(RuntimeError):
    """Raised when a cgroup v2 safety invariant cannot be established."""


@dataclass(frozen=True)
class CgroupV2Handle:
    """A configured per-execution cgroup."""

    path: Path

    @property
    def command_prefix(self) -> list[str]:
        wrapper = os.environ.get(
            "SANDBOX_CGROUP2_EXEC",
            "/usr/local/libexec/sandboxfusion-cgroup2-exec",
        )
        return [wrapper, str(self.path)]


def parse_byte_limit(value: str | int) -> int:
    """Parse a positive binary byte limit such as ``256M`` or ``4GiB``."""

    if isinstance(value, int):
        if value <= 0:
            raise ValueError("cgroup memory limit must be positive")
        return value
    match = re.fullmatch(r"\s*([1-9][0-9]*)\s*([KMGT]?)(?:i?B?)?\s*", value, re.IGNORECASE)
    if match is None:
        raise ValueError(f"invalid cgroup memory limit: {value!r}")
    amount = int(match.group(1))
    exponent = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}[match.group(2).upper()]
    return amount * (1024**exponent)


def _read_tokens(path: Path) -> set[str]:
    return set(path.read_text(encoding="utf-8").split())


def _write_control(path: Path, value: str | int) -> None:
    try:
        path.write_text(f"{value}\n", encoding="utf-8")
    except OSError as exc:
        raise CgroupV2Error(f"cannot write cgroup control {path}: {exc}") from exc


def _mkdir_cgroup(path: Path) -> None:
    # The leaf is exposed read-only as the sandbox's /sys/fs/cgroup. Its
    # controls are kernel-owned and non-writable to uid 1000, but the mount
    # root itself must be searchable so active probes and runtimes can read it.
    path.mkdir(mode=0o755)
    path.chmod(0o755)


class CgroupV2Manager:
    """Create, configure, terminate, and remove direct child cgroups."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured_root = root or os.environ.get("SANDBOX_CGROUP2_ROOT", DEFAULT_CGROUP_ROOT)
        self.root = Path(configured_root).resolve(strict=True)
        self._validate_root()

    def _validate_root(self) -> None:
        controllers_path = self.root / "cgroup.controllers"
        subtree_path = self.root / "cgroup.subtree_control"
        if not controllers_path.is_file() or not subtree_path.is_file():
            raise CgroupV2Error(f"{self.root} is not a cgroup v2 delegation root")
        required = {"cpu", "memory", "pids"}
        available = _read_tokens(controllers_path)
        missing_available = required - available
        if missing_available:
            raise CgroupV2Error(
                "required cgroup v2 controllers are unavailable: " + ", ".join(sorted(missing_available))
            )
        enabled = _read_tokens(subtree_path)
        missing_enabled = required - enabled
        if missing_enabled:
            raise CgroupV2Error(
                "required cgroup v2 controllers are not delegated at "
                f"{self.root}: {', '.join(sorted(missing_enabled))}"
            )
        if not os.access(self.root, os.W_OK):
            raise CgroupV2Error(f"cgroup v2 root is not writable: {self.root}")

    def create(
        self,
        *,
        memory_limit: str | int,
        cpu_limit: float,
        pids_limit: int = DEFAULT_PIDS_LIMIT,
    ) -> CgroupV2Handle:
        if not math.isfinite(cpu_limit) or cpu_limit <= 0:
            raise ValueError("cgroup CPU limit must be positive and finite")
        if pids_limit <= 0:
            raise ValueError("cgroup PID limit must be positive")

        memory_bytes = parse_byte_limit(memory_limit)
        name = f"{CGROUP_NAME_PREFIX}{secrets.token_hex(12)}"
        path = self.root / name
        _mkdir_cgroup(path)
        try:
            required_files = {
                "cgroup.procs",
                "cpu.max",
                "memory.max",
                "pids.max",
            }
            missing = sorted(name for name in required_files if not (path / name).exists())
            if missing:
                raise CgroupV2Error(f"new cgroup {path} lacks required controls: {', '.join(missing)}")
            _write_control(path / "memory.max", memory_bytes)
            if (path / "memory.swap.max").exists():
                _write_control(path / "memory.swap.max", 0)
            if (path / "memory.oom.group").exists():
                _write_control(path / "memory.oom.group", 1)
            quota = max(1, round(cpu_limit * CPU_PERIOD_US))
            _write_control(path / "cpu.max", f"{quota} {CPU_PERIOD_US}")
            _write_control(path / "pids.max", pids_limit)
            return CgroupV2Handle(path=path)
        except Exception:
            self.destroy(CgroupV2Handle(path=path), ignore_missing=True)
            raise

    def destroy(
        self,
        handle: CgroupV2Handle,
        *,
        ignore_missing: bool = False,
        timeout: float = 5.0,
    ) -> None:
        path = handle.path.resolve(strict=False)
        if path.parent != self.root or not path.name.startswith(CGROUP_NAME_PREFIX):
            raise CgroupV2Error(f"refusing to remove unexpected cgroup path: {path}")
        if not path.exists():
            if ignore_missing:
                return
            raise CgroupV2Error(f"cgroup disappeared before cleanup: {path}")

        kill_file = path / "cgroup.kill"
        if kill_file.exists():
            _write_control(kill_file, 1)
        else:
            self._kill_member_pids(path)

        deadline = time.monotonic() + timeout
        while self._is_populated(path) and time.monotonic() < deadline:
            if not kill_file.exists():
                self._kill_member_pids(path)
            time.sleep(0.02)
        if self._is_populated(path):
            raise CgroupV2Error(f"cgroup remains populated after SIGKILL: {path}")
        last_error: OSError | None = None
        remove_deadline = time.monotonic() + timeout
        while True:
            try:
                path.rmdir()
                return
            except OSError as exc:
                last_error = exc
                if time.monotonic() >= remove_deadline:
                    break
                time.sleep(0.02)
        raise CgroupV2Error(f"cannot remove cgroup {path}: {last_error}") from last_error

    @staticmethod
    def _kill_member_pids(path: Path) -> None:
        procs = path / "cgroup.procs"
        if not procs.exists():
            return
        for value in procs.read_text(encoding="utf-8").split():
            try:
                os.kill(int(value), signal.SIGKILL)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                raise CgroupV2Error(f"cannot kill PID {value} in {path}: {exc}") from exc

    @staticmethod
    def _is_populated(path: Path) -> bool:
        events = path / "cgroup.events"
        if events.exists():
            values = dict(
                line.split(maxsplit=1) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            return values.get("populated", "0") != "0"
        procs = path / "cgroup.procs"
        return procs.exists() and bool(procs.read_text(encoding="utf-8").strip())
