"""Deterministic bounded-memory sketches of realized-update support.

Storing one boolean per model coordinate is prohibitive for multi-billion
parameter models.  This module keeps a fixed deterministic coordinate sample
per optimizer-owned view, labels all estimates with ``_sketch``, and persists
its short history at low-frequency observation points.  A non-contiguous
resume resets the history instead of reporting a false adjacent-update
Jaccard.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _sample_indices(descriptor: dict[str, Any], sample_size: int) -> torch.Tensor:
    numel = int(descriptor["numel"])
    count = min(numel, sample_size)
    if count == 0:
        return torch.empty(0, dtype=torch.int64)
    if count == numel:
        return torch.arange(numel, dtype=torch.int64)
    payload = repr(sorted(descriptor.items())).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    start = int.from_bytes(digest[:8], "little") % numel
    stride = int.from_bytes(digest[8:], "little") % numel or 1
    while math.gcd(stride, numel) != 1:
        stride = (stride + 1) % numel or 1
    return torch.remainder(start + stride * torch.arange(count, dtype=torch.int64), numel)


class SupportWindowSketch:
    """Track sampled ``delta_model != 0`` support across successful updates."""

    # Parameter-weighted previous population, intersection, union, current
    # changed population, sampled population, window-frequency sum,
    # never-changed population, history population, and actual sample count.
    _COLUMN_COUNT = 9

    def __init__(
        self,
        *,
        group_names: list[str],
        descriptors: list[dict[str, Any]],
        sample_size: int,
        window: int,
        device: torch.device,
        path: Path,
    ) -> None:
        if sample_size <= 0 or window <= 0:
            raise ValueError("Support sketch sample_size and window must be positive.")
        self.group_names = tuple(group_names)
        self.descriptors = descriptors
        self.sample_size = int(sample_size)
        self.window = int(window)
        self.device = device
        self.path = path
        self.indices = [_sample_indices(descriptor, self.sample_size).to(device=device) for descriptor in descriptors]
        self.previous: list[torch.Tensor | None] = [None] * len(descriptors)
        self.history: list[list[torch.Tensor]] = [[] for _ in descriptors]
        self.last_successful_update = 0
        self.history_contiguous = True
        self._report = False
        self._metrics = torch.zeros(
            (len(group_names), self._COLUMN_COUNT),
            dtype=torch.float64,
            device=device,
        )
        self._load()

    def _signature(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "group_names": list(self.group_names),
            "descriptors": self.descriptors,
            "sample_size": self.sample_size,
            "window": self.window,
        }

    def _load(self) -> None:
        if not self.path.is_file():
            return
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        signature = self._signature()
        if any(payload.get(name) != value for name, value in signature.items()):
            raise ValueError(
                f"Support-sketch state {self.path} is incompatible with current groups, views, or settings."
            )
        previous = payload.get("previous", [])
        history = payload.get("history", [])
        if len(previous) != len(self.descriptors) or len(history) != len(self.descriptors):
            raise ValueError(f"Support-sketch state {self.path} has the wrong view count.")
        self.previous = [
            None if value is None else value.to(device=self.device, dtype=torch.bool).reshape(-1) for value in previous
        ]
        self.history = [
            [entry.to(device=self.device, dtype=torch.bool).reshape(-1) for entry in entries] for entries in history
        ]
        for view_id, indices in enumerate(self.indices):
            expected = int(indices.numel())
            values = ([] if self.previous[view_id] is None else [self.previous[view_id]]) + self.history[view_id]
            if any(value.numel() != expected for value in values):
                raise ValueError(f"Support-sketch state {self.path} has an invalid sampled-vector length.")
            if len(self.history[view_id]) > self.window:
                raise ValueError(f"Support-sketch state {self.path} exceeds its configured history window.")
        self.last_successful_update = int(payload.get("last_successful_update", 0))

    def _reset_history(self) -> None:
        self.previous = [None] * len(self.descriptors)
        self.history = [[] for _ in self.descriptors]
        self.history_contiguous = False

    def begin(self, successful_update: int, *, report: bool) -> None:
        expected = self.last_successful_update + 1
        if self.last_successful_update and successful_update != expected:
            self._reset_history()
        self._report = bool(report)
        self._metrics.zero_()
        self._current_successful_update = int(successful_update)

    def _accumulate(
        self,
        group_id: int,
        current: torch.Tensor,
        previous: torch.Tensor | None,
        history: list[torch.Tensor],
        coordinate_weight: float,
    ) -> None:
        count = int(current.numel())
        if count == 0:
            return
        row = self._metrics[group_id]
        if previous is not None:
            row[0] += count * coordinate_weight
            row[1].add_(torch.count_nonzero(current & previous), alpha=coordinate_weight)
            row[2].add_(torch.count_nonzero(current | previous), alpha=coordinate_weight)
        row[3].add_(torch.count_nonzero(current), alpha=coordinate_weight)
        row[4] += count * coordinate_weight
        stacked = torch.stack(history)
        frequencies = stacked.to(torch.float64).mean(dim=0)
        row[5].add_(frequencies.sum(), alpha=coordinate_weight)
        row[6].add_(torch.count_nonzero(~stacked.any(dim=0)), alpha=coordinate_weight)
        row[7] += count * coordinate_weight
        row[8] += count

    @torch.no_grad()
    def add(
        self,
        view_id: int,
        delta_model: torch.Tensor,
        *,
        group_ids: tuple[int, ...],
        semantic_groups: dict[int, torch.Tensor] | None = None,
    ) -> None:
        indices = self.indices[view_id]
        flat = delta_model.detach().reshape(-1)
        if flat.numel() != int(self.descriptors[view_id]["numel"]):
            raise ValueError("Support sketch delta_model length changed after initialization.")
        current = flat[indices] != 0
        previous = self.previous[view_id]
        history = [*self.history[view_id], current]
        history = history[-self.window :]
        coordinate_weight = int(self.descriptors[view_id]["numel"]) / int(indices.numel()) if indices.numel() else 0.0

        if self._report:
            for group_id in group_ids:
                self._accumulate(
                    group_id,
                    current,
                    previous,
                    history,
                    coordinate_weight,
                )
            for group_id, selection in (semantic_groups or {}).items():
                selected = selection.detach().to(device=self.device, dtype=torch.bool).reshape(-1)
                if selected.numel() != current.numel():
                    raise ValueError("Support sketch semantic selection has the wrong length.")
                self._accumulate(
                    group_id,
                    current[selected],
                    None if previous is None else previous[selected],
                    [entry[selected] for entry in history],
                    coordinate_weight,
                )

        self.previous[view_id] = current
        self.history[view_id] = history

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        payload = {
            **self._signature(),
            "last_successful_update": self.last_successful_update,
            "previous": [None if value is None else value.to(device="cpu") for value in self.previous],
            "history": [[entry.to(device="cpu") for entry in entries] for entries in self.history],
        }
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def finish(self) -> dict[str, dict[str, Any]]:
        self.last_successful_update = self._current_successful_update
        if not self._report:
            return {}
        metrics = self._metrics.clone()
        if _distributed():
            torch.distributed.all_reduce(metrics)
        metrics = metrics.cpu()
        output: dict[str, dict[str, Any]] = {}
        for group_id, group_name in enumerate(self.group_names):
            (
                previous_count,
                intersection,
                union,
                current_changed,
                sample_count,
                frequency_sum,
                never_count,
                history_count,
                actual_sample_count,
            ) = (float(value) for value in metrics[group_id])
            if sample_count == 0:
                continue
            if previous_count == 0:
                jaccard = None
            elif union == 0:
                jaccard = 1.0
            else:
                jaccard = intersection / union
            output[group_name] = {
                "delta_model_support_sample_count": int(actual_sample_count),
                "delta_model_support_estimated_population_count_sketch": sample_count,
                "delta_model_support_fraction_sketch": current_changed / sample_count,
                "delta_model_support_jaccard_previous_update_sketch": jaccard,
                "delta_model_window_update_frequency_mean_sketch": (
                    frequency_sum / history_count if history_count else None
                ),
                "delta_model_window_never_changed_fraction_sketch": (
                    never_count / history_count if history_count else None
                ),
                "delta_model_support_history_contiguous": self.history_contiguous,
            }
        self.history_contiguous = True
        self._save()
        return output


__all__ = ["SupportWindowSketch"]
