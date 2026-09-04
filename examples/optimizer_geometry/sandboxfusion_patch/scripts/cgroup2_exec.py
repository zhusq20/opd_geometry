#!/usr/bin/env python3
"""Move this trusted launcher into one SandboxFusion cgroup, then exec."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PREFIX = "sandboxfusion-"
DEFAULT_ROOT = "/sys/fs/cgroup/sandboxfusion"


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: sandboxfusion-cgroup2-exec CGROUP COMMAND [ARG ...]")
    root = Path(os.environ.get("SANDBOX_CGROUP2_ROOT", DEFAULT_ROOT)).resolve(strict=True)
    cgroup = Path(sys.argv[1]).resolve(strict=True)
    if cgroup.parent != root or not cgroup.name.startswith(PREFIX):
        raise SystemExit(f"refusing unexpected cgroup path: {cgroup}")
    try:
        (cgroup / "cgroup.procs").write_text(f"{os.getpid()}\n", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot join cgroup {cgroup}: {exc}") from exc
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
