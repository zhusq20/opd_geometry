"""Durable, analysis-ready training-rollout samples for geometry runs."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slime.utils.metric_utils import num_updates_before_rollout
from slime.utils.types import Sample


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, enum.Enum):
        return _json_safe(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _scalar_task_reward(args: Any, sample: Sample) -> float | None:
    metadata = sample.metadata or {}
    value = metadata.get("task_reward_observed", metadata.get("raw_task_reward"))
    if value is None:
        value = sample.get_reward_value(args)
    if isinstance(value, dict):
        value = value.get("task_reward")
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
        return float(value)
    return None


def _sample_task(args: Any, sample: Sample) -> str:
    metadata = sample.metadata or {}
    task = (
        metadata.get("task_name")
        or metadata.get("source_name")
        or getattr(sample, "source", None)
        or getattr(args, "experiment_task", None)
        or "unknown"
    )
    return str(task)


def _sample_identifier(sample: Sample, position: int) -> Any:
    metadata = sample.metadata or {}
    return _json_safe(metadata.get("sample_id", sample.index if sample.index is not None else position))


def _prompt_identifier(sample: Sample, position: int) -> Any:
    metadata = sample.metadata or {}
    fallback = sample.group_index if sample.group_index is not None else sample.index
    return _json_safe(metadata.get("prompt_id", fallback if fallback is not None else position))


def _record(args: Any, rollout_id: int, sample: Sample, position: int) -> dict[str, Any]:
    num_updates = num_updates_before_rollout(args, rollout_id)
    reward = _scalar_task_reward(args, sample)
    reward_observed = reward is not None
    use_opd = bool(getattr(args, "use_opd", False))
    configured_coefficient = float(getattr(args, "opd_task_reward_weight", 0.0) or 0.0) if use_opd else 1.0
    reward_used = reward_observed and configured_coefficient != 0.0 and not sample.remove_sample
    return {
        "schema_version": 1,
        "record_type": "training_rollout_sample",
        "run_id": getattr(args, "experiment_name", None),
        "seed": int(getattr(args, "seed", 0)),
        "task": _sample_task(args, sample),
        "rollout_id": int(rollout_id),
        "num_updates": num_updates,
        "model_version": num_updates,
        "sample_position": position,
        "prompt_id": _prompt_identifier(sample, position),
        "sample_id": _sample_identifier(sample, position),
        "source_index": _json_safe(sample.index),
        "group_index": _json_safe(sample.group_index),
        "sample_rollout_id": _json_safe(sample.rollout_id),
        "prompt": _json_safe(sample.prompt),
        "response": sample.response,
        "label": _json_safe(sample.label),
        "reward": reward,
        "passed": reward == 1.0 if reward_observed else None,
        "task_reward_observed": reward_observed,
        "reward_used_in_loss": reward_used,
        "reward_loss_coefficient": configured_coefficient if reward_used else 0.0,
        "status": sample.status.value,
        "remove_sample": bool(sample.remove_sample),
        "response_length": int(sample.response_length),
        "effective_response_length": int(sample.effective_response_length),
        "weight_versions": _json_safe(sample.weight_versions),
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, payload: bytes) -> None:
    """Atomically create *path*; an identical replay is idempotent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"Refusing to replace divergent rollout artifact: {path}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link publishes without ever overwriting a resume artifact.
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() != payload:
                raise FileExistsError(f"Refusing to replace divergent rollout artifact: {path}") from error
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def persist_rollout_samples(rollout_id: int, args: Any, samples: list[Sample]) -> dict[str, Any]:
    """Persist one deterministic JSONL file for an actual training rollout."""

    output_root = Path(os.path.expandvars(os.path.expanduser(str(args.geometry_output_dir))))
    path = output_root / "rollout" / "samples" / f"rollout_{int(rollout_id):08d}.jsonl"
    payload = b"".join(
        (json.dumps(_record(args, rollout_id, sample, position), sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        for position, sample in enumerate(samples)
    )
    _publish_once(path, payload)
    return {
        "path": str(path.resolve()),
        "samples": len(samples),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


__all__ = ["persist_rollout_samples"]
