#!/usr/bin/env python3
"""Validate and, when necessary, rewind a MOPD run to its latest complete checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

TASKS = ("math", "code", "if", "science")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _allocation_path(run_dir: Path) -> Path:
    paths = [run_dir / "allocation.jsonl", run_dir / "allocation/allocation.jsonl"]
    existing = [path for path in paths if path.is_file()]
    if len(existing) > 1:
        raise ValueError(f"Ambiguous allocation logs under {run_dir}")
    return existing[0] if existing else paths[1]


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_root = run_dir / "checkpoints"
    index_path = checkpoint_root / "mopd_checkpoint_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"No MOPD checkpoint index exists under {run_dir}")
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    complete: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in entries:
        rollout_id = int(entry["rollout_id"])
        model = checkpoint_root / f"iter_{rollout_id:07d}"
        sampler = checkpoint_root / "rollout" / f"mopd_dataset_state_dict_{rollout_id}.pt"
        if not all((model / name).is_file() for name in (".metadata", "common.pt")) or not sampler.is_file():
            continue
        state = torch.load(sampler, map_location="cpu", weights_only=False)
        controller = state.get("controller") or {}
        if (
            int(controller.get("completed_steps", -1)) != int(entry["optimizer_updates"])
            or int(controller.get("attempted_responses", -1)) != int(entry["attempted_responses"])
            or controller.get("pending") is not None
        ):
            continue
        complete.append((entry, state))
    if not complete:
        raise ValueError("No indexed checkpoint has complete model, optimizer, and sampler state")
    return max(complete, key=lambda item: int(item[0]["attempted_responses"]))


def inspect(run_dir: Path, *, protocol_path: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if (run_dir / "run_complete.json").is_file():
        raise ValueError(f"Run is already complete and cannot be resumed: {run_dir}")
    entry, state = _checkpoint(run_dir)
    if protocol_path is not None:
        from slime_plugins.mopd.prompting import protocol_identity

        if state.get("protocol_sha256") != protocol_identity(protocol_path):
            raise ValueError("Saved MOPD prompt protocol differs; start a new run for asymmetric prefixes")
    rollout_id = int(entry["rollout_id"])
    operation_index = int(entry["operation_index"])
    step = int(entry["optimizer_updates"])
    allocations = _jsonl(_allocation_path(run_dir))
    if not allocations or int(allocations[-1]["operation_index"]) < operation_index:
        raise ValueError("The allocation log does not reach the selected checkpoint")
    selected = [row for row in allocations if int(row["operation_index"]) == operation_index]
    if len(selected) != 1 or int(selected[0]["rollout_id"]) != rollout_id:
        raise ValueError("The checkpoint index and allocation log disagree")

    eval_rows = _jsonl(run_dir / "teacher_loss_eval/index.jsonl")
    matching_eval = [row for row in eval_rows if int(row["rollout_id"]) == rollout_id]
    eval_complete = len(matching_eval) == 1 and set(matching_eval[0].get("datasets") or {}) == set(TASKS)
    if eval_complete:
        eval_complete = all(
            Path(record["path"]).is_file() and _sha256(Path(record["path"])) == record["sha256"]
            for record in matching_eval[0]["datasets"].values()
        )
    if (run_dir / "fixed_loss").is_dir():
        eval_complete = step % 100 != 0 or (run_dir / "fixed_loss" / f"step_{step:04d}.json").is_file()
        if step == 500:
            eval_complete = eval_complete and (run_dir / "fixed_loss/fresh_final.json").is_file()
    metric_limits = {
        "eval.jsonl": ("eval/rollout_id", rollout_id),
        "mopd.jsonl": ("mopd/update", int(entry["optimizer_updates"])),
        "rollout.jsonl": ("rollout/step", rollout_id),
        "train.jsonl": ("train/rollout_id", rollout_id),
    }
    metric_tail = any(
        any(int(row["metrics"][key]) > limit for row in _jsonl(run_dir / "metrics" / name))
        for name, (key, limit) in metric_limits.items()
    )
    eval_tail = any(int(row["rollout_id"]) > rollout_id for row in eval_rows)
    fixed_loss_tail = any(
        int(json.loads(path.read_text(encoding="utf-8"))["step"]) > step
        for path in (run_dir / "fixed_loss").glob("*.json")
    )
    checkpoint_cost_tail = any(int(row["step"]) > step for row in _jsonl(run_dir / "checkpoint_costs.jsonl"))
    return {
        "run_dir": str(run_dir),
        "checkpoint_root": str((run_dir / "checkpoints").resolve()),
        "rollout_id": rollout_id,
        "next_rollout_id": rollout_id + 1,
        "operation_index": operation_index,
        "attempted_responses": int(entry["attempted_responses"]),
        "optimizer_updates": int(entry["optimizer_updates"]),
        "allocation_frontier": int(allocations[-1]["operation_index"]),
        "rewind_required": (
            int(allocations[-1]["operation_index"]) > operation_index
            or metric_tail
            or eval_tail
            or fixed_loss_tail
            or checkpoint_cost_tail
        ),
        "eval_on_start": not eval_complete,
    }


def _archive_mutable_state(run_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = run_dir / "provenance" / f"resume_rewind_{stamp}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    candidates = [
        _allocation_path(run_dir),
        run_dir / "checkpoints/mopd_checkpoint_index.json",
        run_dir / "run_failed.json",
        run_dir / "wandb_run_id.txt",
        run_dir / "provenance/run_manifest.json",
        *sorted((run_dir / "metrics").glob("*.jsonl")),
        *sorted((run_dir / "teacher_loss_eval").rglob("*")),
        *sorted((run_dir / "fixed_loss").glob("*.json")),
        run_dir / "fixed_loss/fresh_bank.pt",
        run_dir / "checkpoint_costs.jsonl",
    ]
    with tarfile.open(destination, "w:gz") as archive:
        for path in candidates:
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(run_dir)), recursive=False)
    return destination


def apply(run_dir: Path, *, protocol_path: Path | None = None) -> dict[str, Any]:
    result = inspect(run_dir, protocol_path=protocol_path)
    run_dir = Path(result["run_dir"])
    rollout_id = int(result["rollout_id"])
    operation_index = int(result["operation_index"])
    archive = _archive_mutable_state(run_dir)

    allocation_path = _allocation_path(run_dir)
    _atomic_jsonl(
        allocation_path,
        [row for row in _jsonl(allocation_path) if int(row["operation_index"]) <= operation_index],
    )
    index_path = run_dir / "checkpoints/mopd_checkpoint_index.json"
    checkpoint_entries = json.loads(index_path.read_text(encoding="utf-8"))
    _atomic_json(
        index_path,
        [row for row in checkpoint_entries if int(row["operation_index"]) <= operation_index],
    )

    metric_limits = {
        "eval.jsonl": ("eval/rollout_id", rollout_id),
        "mopd.jsonl": ("mopd/update", int(result["optimizer_updates"])),
        "rollout.jsonl": ("rollout/step", rollout_id),
        "train.jsonl": ("train/rollout_id", rollout_id),
    }
    for name, (key, limit) in metric_limits.items():
        path = run_dir / "metrics" / name
        rows = _jsonl(path)
        if rows:
            _atomic_jsonl(path, [row for row in rows if int(row["metrics"][key]) <= limit])

    eval_index = run_dir / "teacher_loss_eval/index.jsonl"
    kept_eval = [row for row in _jsonl(eval_index) if int(row["rollout_id"]) <= rollout_id]
    if result["eval_on_start"]:
        kept_eval = [row for row in kept_eval if int(row["rollout_id"]) != rollout_id]
    if eval_index.exists():
        _atomic_jsonl(eval_index, kept_eval)
    referenced = {
        str(Path(record["path"]).resolve()) for row in kept_eval for record in (row.get("datasets") or {}).values()
    }
    artifact_root = run_dir / "teacher_loss_eval"
    for path in (artifact_root.rglob("*.jsonl") if artifact_root.is_dir() else ()):
        if path != eval_index and str(path.resolve()) not in referenced:
            path.unlink()

    step = int(result["optimizer_updates"])
    for path in (run_dir / "fixed_loss").glob("*.json"):
        if int(json.loads(path.read_text(encoding="utf-8"))["step"]) > step:
            path.unlink()
    if step < 500:
        (run_dir / "fixed_loss/fresh_bank.pt").unlink(missing_ok=True)

    # A rewind starts a linked W&B run so stale post-checkpoint points cannot
    # remain mixed with the canonical resumed trajectory.
    if result["rewind_required"]:
        (run_dir / "wandb_run_id.txt").unlink(missing_ok=True)
    costs_path = run_dir / "checkpoint_costs.jsonl"
    if costs_path.is_file():
        _atomic_jsonl(costs_path, [row for row in _jsonl(costs_path) if int(row["step"]) <= step])
    result["archive"] = str(archive)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, help="Require checkpoint identity to match this prompt protocol")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = (
        apply(args.run_dir, protocol_path=args.protocol)
        if args.apply
        else inspect(args.run_dir, protocol_path=args.protocol)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
