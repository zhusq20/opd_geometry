"""Persist an exact, averaged raw-gradient probe without full-vector all-gather.

Each distributed optimizer rank owns disjoint parameter ranges.  This module
accumulates those local FP32 ranges on CPU across a fixed number of equal-sized
probe backward batches, then writes one tensor per range. Cross-task dot products can
therefore be reconstructed exactly by streaming matching files from probes
that used the same checkpoint and parallel topology.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .optimizer_views import OptimizerParameterView


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _distributed() else 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if _distributed() else 1


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RawGradientProbeAccumulator:
    """Average pre-clipping raw gradients and persist optimizer-owned shards."""

    def __init__(self, args: Any, views: Sequence[OptimizerParameterView]):
        raw_output_dir = str(getattr(args, "geometry_raw_gradient_probe_dir", "") or "").strip()
        expected_updates = int(getattr(args, "geometry_raw_gradient_probe_updates", 0) or 0)
        if bool(raw_output_dir) != bool(expected_updates):
            raise ValueError(
                "--geometry-raw-gradient-probe-dir and " "--geometry-raw-gradient-probe-updates must be set together."
            )
        if expected_updates < 0:
            raise ValueError("--geometry-raw-gradient-probe-updates must be non-negative.")

        self.enabled = bool(raw_output_dir)
        self.output_dir = Path(raw_output_dir).expanduser().resolve() if raw_output_dir else None
        self.expected_updates = expected_updates
        self.args = args
        self.views = tuple(views)
        self._sums: list[torch.Tensor] | None = None
        self._updates = 0
        self._batch_sizes: list[int] = []
        self._effective_token_counts: list[int] = []
        self._source_counts: Counter[str] = Counter()
        self._finished = False

        if self.enabled:
            if not self.views:
                raise ValueError("Raw-gradient probing requires at least one optimizer-owned parameter range.")
            assert self.output_dir is not None
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def add(
        self,
        *,
        observation_id: int,
        source_names: Sequence[str],
        actual_batch_size: int,
        effective_token_count: int,
    ) -> None:
        if not self.enabled:
            return
        if self._finished:
            raise RuntimeError("Raw-gradient probe received an update after its artifact was finalized.")
        if observation_id != self._updates:
            raise ValueError(
                "Raw-gradient probes must start from observation 0 and be contiguous; "
                f"expected {self._updates}, received {observation_id}."
            )
        if actual_batch_size <= 0:
            raise ValueError("Raw-gradient probes require a positive, fixed actual batch size.")
        if effective_token_count < 0:
            raise ValueError("Raw-gradient probe effective-token counts must be non-negative.")
        if self._batch_sizes and actual_batch_size != self._batch_sizes[0]:
            raise ValueError(
                "Raw-gradient probe updates must have equal batch sizes so their arithmetic mean "
                f"matches the fixed probe objective; got {self._batch_sizes[0]} then {actual_batch_size}."
            )

        if self._sums is None:
            self._sums = []
            for view in self.views:
                gradient = view.raw_gradient()
                if gradient is None:
                    raise RuntimeError(f"Missing raw gradient for optimizer-owned range {view.name!r}.")
                self._sums.append(gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float32).clone())
        else:
            for total, view in zip(self._sums, self.views, strict=True):
                gradient = view.raw_gradient()
                if gradient is None:
                    raise RuntimeError(f"Missing raw gradient for optimizer-owned range {view.name!r}.")
                total.add_(gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float32))

        self._batch_sizes.append(int(actual_batch_size))
        self._effective_token_counts.append(int(effective_token_count))
        self._source_counts.update(str(name) for name in source_names)
        self._updates += 1
        if self._updates == self.expected_updates:
            self._write()

    def _write(self) -> None:
        assert self.output_dir is not None and self._sums is not None
        rank = _rank()
        rank_dir = self.output_dir / f"rank_{rank:05d}"
        if rank_dir.exists() and any(rank_dir.iterdir()):
            raise FileExistsError(f"Raw-gradient rank directory is not empty: {rank_dir}")
        rank_dir.mkdir(parents=True, exist_ok=True)

        view_records = []
        scale = 1.0 / self._updates
        for view_id, (view, total) in enumerate(zip(self.views, self._sums, strict=True)):
            total.mul_(scale)
            if not bool(torch.isfinite(total).all()):
                raise FloatingPointError(f"Non-finite raw gradient in {view.name!r}.")
            filename = f"view_{view_id:05d}.pt"
            path = rank_dir / filename
            _atomic_torch_save(total, path)
            view_records.append(
                {
                    "file": filename,
                    "group_names": list(view.group_names),
                    "name": view.name,
                    "numel": int(total.numel()),
                    "optimizer_branch": view.optimizer_branch,
                    "start": int(view.start),
                    "stop": None if view.stop is None else int(view.stop),
                    "bytes": path.stat().st_size,
                }
            )

        rank_manifest = {
            "schema_version": 1,
            "rank": rank,
            "world_size": _world_size(),
            "task": str(getattr(self.args, "experiment_task", "unknown")),
            "checkpoint": str(getattr(self.args, "load", "")),
            "checkpoint_step": getattr(self.args, "ckpt_step", None),
            "seed": int(getattr(self.args, "seed", 0)),
            "rollout_seed": int(getattr(self.args, "rollout_seed", 0)),
            "expected_updates": self.expected_updates,
            "observed_updates": self._updates,
            "actual_batch_size_per_update": self._batch_sizes[0],
            "rollout_batch_size": int(getattr(self.args, "rollout_batch_size", 0)),
            "n_samples_per_prompt": int(getattr(self.args, "n_samples_per_prompt", 0)),
            "probe_prompt_count": self.expected_updates * int(getattr(self.args, "rollout_batch_size", 0)),
            "effective_token_counts": self._effective_token_counts,
            "effective_token_count_total": sum(self._effective_token_counts),
            "optimizer_step_executed": not bool(getattr(self.args, "geometry_raw_gradient_probe_only", False)),
            "source_counts": dict(sorted(self._source_counts.items())),
            "gradient_definition": (
                "Arithmetic mean of FP32 loss gradients after the configured loss reduction, "
                "before clipping and before the optimizer step."
            ),
            "views": view_records,
        }
        _atomic_json(rank_manifest, rank_dir / "manifest.json")

        if _distributed():
            torch.distributed.barrier()
        if rank == 0:
            rank_manifests = []
            for rank_id in range(_world_size()):
                path = self.output_dir / f"rank_{rank_id:05d}" / "manifest.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing raw-gradient rank manifest: {path}")
                rank_manifests.append(str(path.relative_to(self.output_dir)))
            _atomic_json(
                {
                    "schema_version": 1,
                    "task": rank_manifest["task"],
                    "checkpoint": rank_manifest["checkpoint"],
                    "checkpoint_step": rank_manifest["checkpoint_step"],
                    "seed": rank_manifest["seed"],
                    "world_size": _world_size(),
                    "expected_updates": self.expected_updates,
                    "actual_batch_size_per_update": self._batch_sizes[0],
                    "rollout_batch_size": rank_manifest["rollout_batch_size"],
                    "n_samples_per_prompt": rank_manifest["n_samples_per_prompt"],
                    "probe_prompt_count": rank_manifest["probe_prompt_count"],
                    "effective_token_counts": self._effective_token_counts,
                    "effective_token_count_total": sum(self._effective_token_counts),
                    "optimizer_step_executed": rank_manifest["optimizer_step_executed"],
                    "rank_manifests": rank_manifests,
                    "gradient_definition": rank_manifest["gradient_definition"],
                },
                self.output_dir / "manifest.json",
            )
        if _distributed():
            torch.distributed.barrier()

        self._sums = None
        self._finished = True


__all__ = ["RawGradientProbeAccumulator"]
