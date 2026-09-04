"""Path and file-size guards for SandboxFusion's pre-isolation staging area."""

from __future__ import annotations

import base64
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path

TRUSTED_SYMLINK_ROOTS = (Path("/root/sandbox/runtime"),)


def _validate_relative_name(filename: str) -> tuple[str, ...]:
    if not filename or "\x00" in filename or Path(filename).is_absolute():
        raise ValueError(f"unsafe sandbox file path: {filename!r}")
    parts = Path(filename).parts
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"sandbox file path contains traversal: {filename!r}")
    return parts


def safe_relative_path(root: str | Path, filename: str) -> Path:
    root_path = Path(root).resolve(strict=True)
    _validate_relative_name(filename)
    candidate = (root_path / filename).resolve(strict=False)
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"sandbox file path escapes its work directory: {filename!r}") from exc
    if candidate == root_path:
        raise ValueError(f"sandbox file path names a directory: {filename!r}")
    return candidate


def read_regular_file_beneath(
    directory: str | Path,
    filename: str,
    *,
    max_bytes: int,
) -> bytes | None:
    """Read one regular file without following any user-created symlink.

    The untrusted cgroup must already be empty before this function is called,
    which makes the component-by-component ``openat`` checks race-free.
    """

    if max_bytes < 0:
        raise ValueError("remaining fetch byte limit cannot be negative")
    parts = _validate_relative_name(filename)
    root = Path(directory).resolve(strict=True)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    directory_fd = os.open(root, directory_flags)
    file_fd: int | None = None
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise ValueError(f"unsafe sandbox fetch path: {filename!r}: {exc}") from exc
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ValueError(f"unsafe sandbox fetch path: {filename!r}: {exc}") from exc
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            return None
        with os.fdopen(file_fd, "rb", closefd=True) as stream:
            file_fd = None
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"sandbox fetched files exceed their {max_bytes}-byte remainder")
        return data
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def _validate_trusted_symlink(path: Path) -> None:
    try:
        target = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"broken symbolic link in sandbox work tree: {path}") from exc
    for trusted_root in TRUSTED_SYMLINK_ROOTS:
        try:
            resolved_root = trusted_root.resolve(strict=True)
        except OSError:
            continue
        if _is_relative_to(target, resolved_root):
            return
    raise ValueError(f"untrusted symbolic link in sandbox work tree: {path} -> {target}")


def make_tree_world_writable(directory: str | Path) -> None:
    root = Path(directory).resolve(strict=True)
    for current_root, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            path = Path(current_root) / name
            if path.is_symlink():
                _validate_trusted_symlink(path)
                continue
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError(f"non-directory entry in sandbox directory list: {path}")
            path.chmod(0o777)
        for name in files:
            path = Path(current_root) / name
            if path.is_symlink():
                _validate_trusted_symlink(path)
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"non-regular file in sandbox work tree: {path}")
            path.chmod(0o777)
    root.chmod(0o777)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def copy_worktree_into_overlay(overlay_root: str | Path, directory: str | Path) -> Path:
    """Copy one runner worktree into the overlay's private ``/tmp`` mount."""

    source = Path(directory).resolve(strict=True)
    host_tmp = Path("/tmp").resolve(strict=True)
    try:
        relative = source.relative_to(host_tmp)
    except ValueError as exc:
        raise ValueError(f"sandbox worktree is not beneath /tmp: {source}") from exc
    if relative == Path("."):
        raise ValueError("refusing to use all of /tmp as a sandbox worktree")
    make_tree_world_writable(source)

    root = Path(overlay_root).resolve(strict=True)
    destination = root / "tmp" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=True)
    make_tree_world_writable(destination)
    return destination


def restore_base64_files(
    directory: str | Path,
    files: Mapping[str, str | None],
    *,
    max_bytes: int,
) -> None:
    total_bytes = 0
    for filename, content in files.items():
        if not isinstance(content, str) or "IGNORE_THIS_FILE" in filename:
            continue
        remaining = max_bytes - total_bytes
        maximum_encoded_length = 4 * ((remaining + 2) // 3)
        if len(content) > maximum_encoded_length:
            raise ValueError(f"sandbox uploads exceed {max_bytes} bytes")
        decoded = base64.b64decode(content, validate=True)
        total_bytes += len(decoded)
        if total_bytes > max_bytes:
            raise ValueError(f"sandbox uploads exceed {max_bytes} bytes")
        filepath = safe_relative_path(directory, filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("xb") as stream:
            stream.write(decoded)
