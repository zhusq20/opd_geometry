#!/usr/bin/env python3
"""Record the exact command and pinned inputs for one MOPD run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
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


def file_record(value: str) -> dict[str, Any]:
    path = Path(value).expanduser().resolve()
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    return record


def checkpoint_record(value: str) -> dict[str, Any]:
    root = Path(value).expanduser().resolve()
    record: dict[str, Any] = {"path": str(root), "exists": root.is_dir()}
    if not root.is_dir():
        return record
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "latest_checkpointed_iteration.txt",
        ".metadata",
        "common.pt",
    ):
        path = root / name
        if path.is_file():
            record[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    selector = root / "latest_checkpointed_iteration.txt"
    if selector.is_file():
        selected_name = selector.read_text(encoding="utf-8").strip()
        if not selected_name:
            raise ValueError(f"Checkpoint selector is empty: {selector}")
        try:
            selected_dir = root / f"iter_{int(selected_name):07d}"
        except ValueError:
            selected_dir = root / selected_name
        if not selected_dir.is_dir():
            raise FileNotFoundError(f"Selected checkpoint directory is missing: {selected_dir}")
        selected = {"path": str(selected_dir.resolve()), "selector": selected_name}
        for name in (".metadata", "common.pt"):
            path = selected_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Selected checkpoint anchor is missing: {path}")
            selected[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        record["selected_checkpoint"] = selected
    return record


def git_record(repo: Path) -> dict[str, Any]:
    def run(*command: str) -> str:
        result = subprocess.run(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip()

    commit = run("git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD")
    status = run("git", "-c", f"safe.directory={repo}", "status", "--porcelain=v1", "--untracked-files=all")
    return {"commit": commit or None, "dirty": bool(status), "status_porcelain": status}


def source_snapshot(repo: Path, run_dir: Path, values: list[str], name: str) -> dict[str, Any] | None:
    """Archive the exact experiment source, including untracked working-tree files."""

    if not values:
        return None
    excluded_parts = {".git", "__pycache__", ".pytest_cache", "generated", "outputs"}
    files: dict[str, Path] = {}
    for value in values:
        root = Path(value).expanduser().resolve()
        try:
            root.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"Source snapshot path is outside the repository: {root}") from exc
        if not root.exists():
            raise FileNotFoundError(f"Source snapshot path does not exist: {root}")
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(repo)
            if excluded_parts.intersection(relative.parts):
                continue
            files[str(relative)] = path
    if not files:
        raise ValueError("Source snapshot contains no files.")

    destination = run_dir / "provenance" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    records = []
    with tarfile.open(temporary, "w:gz") as archive:
        for relative, path in sorted(files.items()):
            # Editors can replace or truncate source files while a job starts.
            # Archive and hash the same captured bytes, not later file reads.
            for _ in range(3):
                with path.open("rb") as stream:
                    before = os.fstat(stream.fileno())
                    content = stream.read()
                    after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and len(
                    content
                ) == after.st_size:
                    break
            else:
                raise RuntimeError(f"Source kept changing while creating its snapshot: {path}")
            info = tarfile.TarInfo(relative)
            info.size = len(content)
            info.mtime = before.st_mtime
            info.mode = before.st_mode & 0o7777
            archive.addfile(info, io.BytesIO(content))
            records.append({"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "files": records,
    }


def versions() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "transformers", "ray", "sglang", "megatron-core"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def hardware_record() -> dict[str, Any]:
    """Record the machine and physical GPUs used by the launch."""

    record: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["nvidia_smi"] = {
            "available": False,
            "query": command[1],
            "gpus": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
        return record
    if result.returncode:
        error = result.stderr.strip() or result.stdout.strip() or f"nvidia-smi exited with code {result.returncode}"
        record["nvidia_smi"] = {
            "available": False,
            "query": command[1],
            "gpus": [],
            "error": error,
        }
        return record
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        record["nvidia_smi"] = {
            "available": False,
            "query": command[1],
            "gpus": [],
            "error": "nvidia-smi returned an empty GPU inventory",
        }
        return record
    record["nvidia_smi"] = {
        "available": True,
        "query": command[1],
        "gpus": lines,
        "error": None,
    }
    return record


def selected_command_options(command: list[str]) -> dict[str, Any]:
    """Expose the protocol-critical CLI settings as structured manifest data."""

    names = (
        "optimizer",
        "lr",
        "weight-decay",
        "adam-beta1",
        "adam-beta2",
        "adam-eps",
        "clip-grad",
        "tensor-model-parallel-size",
        "pipeline-model-parallel-size",
        "context-parallel-size",
        "actor-num-nodes",
        "actor-num-gpus-per-node",
        "rollout-num-gpus",
        "rollout-num-gpus-per-engine",
        "start-rollout-id",
        "ckpt-step",
        "eval-interval",
        "eval-config",
        "eval-function-path",
        "mopd-seed",
        "mopd-loss",
        "mopd-topk",
        "mopd-profile",
        "mopd-tasks",
        "mopd-reduction",
        "mopd-responses-per-update",
        "mopd-skip-task-rewards",
        "mopd-skip-paper-measurements",
        "rollout-max-response-len",
        "hf-checkpoint",
        "sglang-enable-deterministic-inference",
        "sglang-sampling-backend",
        "sglang-attention-backend",
        "mopd-reference-bank",
        "mopd-common-checkpoint",
        "mopd-diagnostic-data",
        "mopd-occupied-gpus",
        "custom-loss-function-path",
        "disable-compute-advantages-and-returns",
        "mopd-allocation",
        "mopd-total-steps",
        "mopd-microbatches-per-step",
        "mopd-prompts-per-microbatch",
        "mopd-min-microbatches",
        "mopd-max-microbatches",
        "mopd-response-budget",
        "mopd-checkpoint-steps",
        "mopd-eval-responses",
        "mopd-eval-on-start",
        "mopd-heldout-variance",
        "mopd-variance-checkpoint-step",
        "mopd-variance-controller-state",
    )
    selected: dict[str, Any] = {}
    for name in names:
        flag = f"--{name}"
        positions = [index for index, value in enumerate(command) if value == flag]
        if not positions:
            continue
        index = positions[-1]
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            selected[name.replace("-", "_")] = True
        else:
            selected[name.replace("-", "_")] = command[index + 1]
    return selected


def start(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    path = run_dir / "provenance/run_manifest.json"
    if path.exists():
        raise FileExistsError(f"run manifest already exists: {path}")
    event = {
        "at_utc": now(),
        "command": args.training_command,
        "inputs": [file_record(value) for value in args.input],
        "checkpoints": [checkpoint_record(value) for value in args.checkpoint],
        "source_snapshot": source_snapshot(repo, run_dir, args.source, "source_snapshot.tar.gz"),
    }
    git = git_record(repo)
    snapshot = event["source_snapshot"]
    manifest = {
        "schema_version": 2,
        "status": "running",
        "created_at_utc": event["at_utc"],
        "cwd": str(repo),
        "git": git,
        "revisions": {
            "training_code_commit": git["commit"],
            "evaluation_code_commit": git["commit"],
            "source_snapshot_sha256": None if snapshot is None else snapshot["sha256"],
        },
        "command": event["command"],
        "protocol_cli": selected_command_options(event["command"]),
        "inputs": event["inputs"],
        "checkpoints": event["checkpoints"],
        "source_snapshot": event["source_snapshot"],
        "environment": {
            "python": sys.version,
            "packages": versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ray_temp_dir": os.environ.get("RAY_TEMP_DIR"),
            "ray_dashboard_port": os.environ.get("RAY_DASHBOARD_PORT"),
            "ray_gcs_port": os.environ.get("RAY_GCS_PORT"),
            "ray_aux_port_base": os.environ.get("RAY_AUX_PORT_BASE"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "mopd_hardware_profile": os.environ.get("MOPD_HARDWARE_PROFILE", "frozen-96gb-tp1"),
            "mopd_training_gpu": os.environ.get("MOPD_TRAIN_GPU"),
            "mopd_training_gpus": os.environ.get("MOPD_TRAIN_GPUS", os.environ.get("MOPD_TRAIN_GPU")),
            "mopd_inference_gpu": os.environ.get("MOPD_INFERENCE_GPU"),
            "mopd_teacher_ports": {
                task: os.environ.get(f"MOPD_TEACHER_{task.upper()}_PORT") for task in ("math", "code", "if", "science")
            },
            "mopd_teacher_gpus": {
                task: os.environ.get(f"MOPD_TEACHER_{task.upper()}_GPU") for task in ("math", "code", "if", "science")
            },
            "mopd_teacher_hf_root": os.environ.get("MOPD_TEACHER_HF_ROOT"),
            "mopd_student_revision": os.environ.get("MOPD_STUDENT_REVISION"),
        },
        "hardware": hardware_record(),
    }
    atomic_json(path, manifest)
    return manifest


def resume(args: argparse.Namespace) -> dict[str, Any]:
    """Append immutable resume provenance and reopen an interrupted run."""

    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    path = run_dir / "provenance/run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"run manifest does not exist: {path}")
    if (run_dir / "run_complete.json").is_file():
        raise ValueError(f"completed run cannot be resumed: {run_dir}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    previous_loss = manifest.get("protocol_cli", {}).get("mopd_loss")
    resumed_loss = selected_command_options(args.training_command).get("mopd_loss")
    if previous_loss != resumed_loss and (previous_loss is not None or resumed_loss is not None):
        raise ValueError(
            f"Cannot resume {previous_loss!r} as {resumed_loss!r}; use a new run ID for a different loss."
        )
    if previous_loss in {"student_topk", "topk_intersection"}:
        previous_k = int(manifest.get("protocol_cli", {}).get("mopd_topk", 16))
        resumed_k = int(selected_command_options(args.training_command).get("mopd_topk", 16))
        if previous_k != resumed_k:
            raise ValueError(f"Cannot resume student Top{previous_k} as Top{resumed_k}; use a new run ID.")
    ordinal = len(manifest.get("resume_events") or []) + 1
    event = {
        "ordinal": ordinal,
        "at_utc": now(),
        "previous_status": manifest.get("status"),
        "previous_exit_code": manifest.get("exit_code"),
        "command": args.training_command,
        "protocol_cli": selected_command_options(args.training_command),
        "inputs": [file_record(value) for value in args.input],
        "checkpoints": [checkpoint_record(value) for value in args.checkpoint],
        "source_snapshot": source_snapshot(repo, run_dir, args.source, f"source_snapshot_resume_{ordinal:03d}.tar.gz"),
        "git": git_record(repo),
    }
    failed = run_dir / "run_failed.json"
    if failed.is_file():
        destination = run_dir / "provenance" / f"run_failed_before_resume_{ordinal:03d}.json"
        os.replace(failed, destination)
        event["previous_failure_marker"] = file_record(str(destination))
    manifest.setdefault("resume_events", []).append(event)
    manifest.update(
        {
            "status": "running",
            "command": event["command"],
            "protocol_cli": event["protocol_cli"],
            "last_resumed_at_utc": event["at_utc"],
        }
    )
    manifest.pop("exit_code", None)
    manifest.pop("finished_at_utc", None)
    atomic_json(path, manifest)
    return manifest


def finish(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    path = run_dir / "provenance/run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    exit_code = int(args.exit_code)
    if exit_code == 0 and not (run_dir / "run_complete.json").is_file():
        exit_code = 3
    manifest.update(
        {
            "status": "complete" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "finished_at_utc": now(),
        }
    )
    atomic_json(path, manifest)
    if exit_code:
        atomic_json(
            run_dir / "run_failed.json",
            {"schema_version": 1, "status": "failed", "exit_code": exit_code, "at_utc": now()},
        )
    else:
        (run_dir / "run_failed.json").unlink(missing_ok=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    begin = subparsers.add_parser("start")
    begin.add_argument("--repo", type=Path, required=True)
    begin.add_argument("--run-dir", type=Path, required=True)
    begin.add_argument("--input", action="append", default=[])
    begin.add_argument("--checkpoint", action="append", default=[])
    begin.add_argument("--source", action="append", default=[])
    begin.add_argument("training_command", nargs=argparse.REMAINDER)
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--repo", type=Path, required=True)
    resume_parser.add_argument("--run-dir", type=Path, required=True)
    resume_parser.add_argument("--input", action="append", default=[])
    resume_parser.add_argument("--checkpoint", action="append", default=[])
    resume_parser.add_argument("--source", action="append", default=[])
    resume_parser.add_argument("training_command", nargs=argparse.REMAINDER)
    end = subparsers.add_parser("finish")
    end.add_argument("--run-dir", type=Path, required=True)
    end.add_argument("--exit-code", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    actions = {"start": start, "resume": resume, "finish": finish}
    output = actions[arguments.action](arguments)
    print(json.dumps(output, indent=2, sort_keys=True))
    if arguments.action == "finish" and output["exit_code"]:
        raise SystemExit(output["exit_code"])
