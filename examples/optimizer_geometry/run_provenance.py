#!/usr/bin/env python3
"""Create, resume, and finalize reproducibility artifacts for one experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ROOTS = ("slime", "slime_plugins", "examples/optimizer_geometry", "scripts/models")
SOURCE_FILES = ("train.py", "requirements.txt", "pyproject.toml", "setup.py")
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".txt", ".md"}
PACKAGES = ("torch", "transformers", "ray", "sglang", "wandb", "megatron-core", "emerging-optimizers")
MAX_ARCHIVED_INPUT_BYTES = 16 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def command(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        list(args), cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_paths(repo: Path) -> list[Path]:
    paths: list[Path] = []
    for name in SOURCE_FILES:
        path = repo / name
        if path.is_file():
            paths.append(path)
    for root_name in SOURCE_ROOTS:
        root = repo / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            relative = path.relative_to(repo)
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            if any(part in {"__pycache__", "data", "outputs", ".git", "wandb"} for part in relative.parts):
                continue
            paths.append(path)
    return sorted(set(paths))


def source_snapshot_content_sha256(path: Path) -> str:
    """Hash archive member names and bytes, independent of gzip/tar timestamps."""

    digest = hashlib.sha256()
    with tarfile.open(path, "r:gz") as archive:
        members = sorted((member for member in archive.getmembers() if member.isfile()), key=lambda member: member.name)
        for member in members:
            name = member.name.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(member.size.to_bytes(8, "big"))
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Could not read source snapshot member {member.name}")
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def make_source_snapshot(repo: Path, path: Path) -> dict[str, Any]:
    paths = source_paths(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as archive:
        for source in paths:
            archive.add(source, arcname=str(source.relative_to(repo)), recursive=False)
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "run_relative_path": str(Path("provenance") / path.name),
        "files": len(paths),
        "sha256": sha256_file(path),
        "content_sha256": source_snapshot_content_sha256(path),
        "bytes": path.stat().st_size,
    }


def file_record(value: str) -> dict[str, Any]:
    path = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return record


def archive_inputs(values: list[str], provenance_dir: Path, *, prefix: str = "") -> list[dict[str, Any]]:
    """Hash every declared input and retain a bounded exact copy of configs."""

    records = []
    archive_dir = provenance_dir / "inputs"
    for index, value in enumerate(values):
        record = file_record(value)
        source = Path(record["path"])
        if source.is_file() and source.stat().st_size <= MAX_ARCHIVED_INPUT_BYTES:
            archive_dir.mkdir(parents=True, exist_ok=True)
            destination = archive_dir / f"{prefix}{index:02d}_{source.name}"
            shutil.copy2(source, destination)
            record["archived_path"] = str(destination.resolve())
            record["archived_run_relative_path"] = str(destination.relative_to(provenance_dir.parent))
            record["archived_sha256"] = sha256_file(destination)
        elif source.is_file():
            record["archive_skipped_reason"] = f"input exceeds {MAX_ARCHIVED_INPUT_BYTES} bytes"
        records.append(record)
    return records


def directory_fingerprint(value: str) -> dict[str, Any]:
    root = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    record: dict[str, Any] = {"path": str(root), "exists": root.exists()}
    if not root.is_dir():
        return record
    entries = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        relative = str(path.relative_to(root))
        entries.append((relative, stat.st_size, stat.st_mtime_ns))
        total += stat.st_size
    payload = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    record.update(
        {
            "files": len(entries),
            "bytes": total,
            "name_size_mtime_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    for name in ("config.json", "latest_checkpointed_iteration.txt"):
        metadata = root / name
        if metadata.is_file():
            record[name] = {"bytes": metadata.stat().st_size, "sha256": sha256_file(metadata)}
    return record


def gpu_inventory() -> list[dict[str, str]]:
    query = (
        "index,uuid,name,memory.total,memory.used,memory.free,driver_version,"
        "ecc.errors.uncorrected.volatile.device_memory,ecc.errors.uncorrected.aggregate.device_memory"
    )
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        return []
    keys = (
        "index",
        "uuid",
        "name",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "driver_version",
        "volatile_uncorrected_ecc",
        "aggregate_uncorrected_ecc",
    )
    return [dict(zip(keys, (part.strip() for part in line.split(",")), strict=True)) for line in result.stdout.splitlines()]


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def start(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    provenance_dir = run_dir / "provenance"
    manifest_path = provenance_dir / "run_manifest.json"
    resume_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_path = provenance_dir / (
        f"source_snapshot_resume_{resume_stamp}.tar.gz"
        if manifest_path.exists()
        else "source_snapshot.tar.gz"
    )
    snapshot = make_source_snapshot(repo, snapshot_path)
    current = {
        "at_utc": now(),
        "source_snapshot": snapshot,
        "command": args.training_command,
        "cwd": str(repo),
        "inputs": [file_record(value) for value in args.input],
        "checkpoint_fingerprints": [directory_fingerprint(value) for value in args.checkpoint],
    }

    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite existing manifest {manifest_path} without --resume.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        original_snapshot = manifest["source_snapshot"]
        original_content_sha = original_snapshot.get("content_sha256")
        if original_content_sha is None:
            original_path = Path(str(original_snapshot.get("path") or ""))
            if original_path.is_file():
                original_content_sha = source_snapshot_content_sha256(original_path)
                original_snapshot["content_sha256"] = original_content_sha
        source_matches = original_content_sha == snapshot["content_sha256"]
        if not source_matches and not args.allow_source_change:
            raise RuntimeError(
                "Source changed since the original run; exact resume refused. "
                "Use a new RUN_NAME, or explicitly pass --allow-source-change for a non-primary recovery."
            )
        if source_matches:
            snapshot_path.unlink()
            current["source_snapshot"] = manifest["source_snapshot"]
        current["inputs"] = archive_inputs(args.input, provenance_dir, prefix=f"resume_{resume_stamp}_")
        archived_markers = {}
        for marker_name in ("run_complete.json", "run_failed.json"):
            marker_path = run_dir / marker_name
            if marker_path.is_file():
                archive_path = provenance_dir / f"{Path(marker_name).stem}_before_resume_{resume_stamp}.json"
                os.replace(marker_path, archive_path)
                archived_markers[marker_name] = file_record(archive_path)
        current["archived_terminal_markers"] = archived_markers
        manifest.setdefault("resume_events", []).append(current)
        manifest["status"] = "running"
        manifest["last_started_at_utc"] = current["at_utc"]
        manifest["last_command"] = args.training_command
        atomic_json(manifest_path, manifest)
        return manifest

    git_commit = command(repo, "git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD")
    git_status = command(repo, "git", "-c", f"safe.directory={repo}", "status", "--porcelain=v1", "--untracked-files=all")
    manifest = {
        "schema_version": 2,
        "status": "running",
        "created_at_utc": current["at_utc"],
        "command": args.training_command,
        "cwd": str(repo),
        "git": {"commit": git_commit, "dirty": bool(git_status), "status_porcelain": git_status or ""},
        "source_snapshot": snapshot,
        "inputs": archive_inputs(args.input, provenance_dir),
        "checkpoint_fingerprints": [directory_fingerprint(value) for value in args.checkpoint],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "available_cuda_devices": os.environ.get("AVAILABLE_CUDA_DEVICES"),
            "batch_profile": os.environ.get("BATCH_PROFILE"),
            "ppo_critic_hparams": {
                "lr": os.environ.get("OPTIMIZER_GEOMETRY_CRITIC_LR"),
                "weight_decay": os.environ.get("OPTIMIZER_GEOMETRY_CRITIC_WEIGHT_DECAY"),
                "adam_beta2": os.environ.get("OPTIMIZER_GEOMETRY_CRITIC_BETA2"),
            },
            "gpus": gpu_inventory(),
        },
    }
    atomic_json(manifest_path, manifest)
    return manifest


def finish(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.run_dir.resolve() / "provenance" / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion_marker = args.run_dir.resolve() / "run_complete.json"
    effective_exit_code = int(args.exit_code)
    if effective_exit_code == 0 and not completion_marker.is_file():
        effective_exit_code = 3
        manifest["completion_error"] = f"Training exited zero but did not write {completion_marker}."
    manifest["exit_code"] = effective_exit_code
    manifest["finished_at_utc"] = now()
    manifest["status"] = "complete" if effective_exit_code == 0 else "failed"
    atomic_json(manifest_path, manifest)
    failure_marker = args.run_dir.resolve() / "run_failed.json"
    if effective_exit_code == 0:
        # A retry in the same explicitly resumed run may have left a failure
        # marker.  A successful, completion-marker-backed finish supersedes it;
        # never leave contradictory terminal states for downstream analysis.
        failure_marker.unlink(missing_ok=True)
    else:
        atomic_json(
            failure_marker,
            {"schema_version": 1, "status": "failed", "exit_code": effective_exit_code, "at_utc": now()},
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--repo", type=Path, required=True)
    start_parser.add_argument("--run-dir", type=Path, required=True)
    start_parser.add_argument("--input", action="append", default=[])
    start_parser.add_argument("--checkpoint", action="append", default=[])
    start_parser.add_argument("--resume", action="store_true")
    start_parser.add_argument("--allow-source-change", action="store_true")
    start_parser.add_argument("training_command", nargs=argparse.REMAINDER)
    finish_parser = subparsers.add_parser("finish")
    finish_parser.add_argument("--run-dir", type=Path, required=True)
    finish_parser.add_argument("--exit-code", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = start(parsed) if parsed.action == "start" else finish(parsed)
    print(json.dumps(result, indent=2, sort_keys=True))
    if parsed.action == "finish" and int(result.get("exit_code", 0)) != 0:
        raise SystemExit(int(result["exit_code"]))
