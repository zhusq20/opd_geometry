"""Map model parameters to the tensors updated by the real optimizer branch.

Megatron may update a full FP32 master parameter, a ZeRO-owned flat shard, or
one leaf of a chained Muon/Adam optimizer.  Geometry code must use those actual
memberships; parameter dimensionality is never used to guess a branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "to_local"):
        value = value.to_local()
    if hasattr(value, "_local_tensor"):
        value = value._local_tensor
    return value


def _raw_gradient(parameter: torch.nn.Parameter) -> torch.Tensor | None:
    for attribute in ("main_grad", "decoupled_grad", "grad"):
        value = getattr(parameter, attribute, None)
        if value is not None:
            return _local_tensor(value)
    return None


def _optimizer_gradient(parameter: torch.Tensor) -> torch.Tensor | None:
    value = getattr(parameter, "decoupled_grad", None)
    if value is None:
        value = getattr(parameter, "grad", None)
    return _local_tensor(value) if value is not None else None


def _is_unique_model_parallel_parameter(parameter: torch.nn.Parameter) -> bool:
    try:
        from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
        from megatron.core.transformer.module import param_is_not_shared

        return bool(param_is_not_tensor_parallel_duplicate(parameter) and param_is_not_shared(parameter))
    except (ImportError, RuntimeError, AssertionError):
        return True


def _is_data_parallel_contributor() -> bool:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return True
    try:
        from megatron.core import mpu

        return mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    except (ImportError, RuntimeError, AssertionError):
        return torch.distributed.get_rank() == 0


def _leaf_optimizers(optimizer: Any):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        yield optimizer
        return
    for child in children:
        yield from _leaf_optimizers(child)


def _inner_optimizer(leaf: Any) -> Any:
    inner = getattr(leaf, "optimizer", None)
    return inner if inner is not None else leaf


def _optimizer_kind(inner: Any) -> str:
    qualified = f"{type(inner).__module__}.{type(inner).__name__}".lower()
    if "muon" in qualified or "orthogonalizedoptimizer" in qualified:
        return "muon"
    if "adam" in qualified:
        return "adam"
    if "sgd" in qualified:
        return "sgd"
    return "unknown"


def _branch_name(kind: str, requested_optimizer: str) -> str:
    requested = requested_optimizer.lower()
    if kind == "muon":
        return "muon_matrix"
    if requested in {"muon", "dist_muon"} and kind == "adam":
        return "adam_fallback"
    return kind


def _group_for_parameter(inner: Any, parameter: torch.Tensor) -> dict[str, Any]:
    for group in getattr(inner, "param_groups", []):
        if any(candidate is parameter for candidate in group["params"]):
            return group
    raise KeyError("Optimizer parameter is not present in its inner optimizer groups.")


@dataclass
class OptimizerParameterView:
    """One uniquely owned coordinate range and its actual optimizer branch."""

    name: str
    model_parameter: torch.nn.Parameter
    optimizer_parameter: torch.Tensor
    inner_optimizer: Any
    optimizer_group: dict[str, Any]
    optimizer_kind: str
    optimizer_branch: str
    group_names: tuple[str, ...]
    start: int = 0
    stop: int | None = None

    def _slice(self, value: torch.Tensor) -> torch.Tensor:
        flat = _local_tensor(value).reshape(-1)
        return flat[self.start : self.stop]

    def model_value(self) -> torch.Tensor:
        return self._slice(self.model_parameter.detach())

    def raw_gradient(self) -> torch.Tensor | None:
        value = _raw_gradient(self.model_parameter)
        return self._slice(value) if value is not None else None

    def optimizer_value(self) -> torch.Tensor:
        return _local_tensor(self.optimizer_parameter.detach()).reshape(-1)

    def optimizer_gradient(self) -> torch.Tensor | None:
        value = _optimizer_gradient(self.optimizer_parameter)
        return value.reshape(-1) if value is not None else None

    @property
    def optimizer_state(self) -> dict[str, Any]:
        return getattr(self.inner_optimizer, "state", {}).get(self.optimizer_parameter, {})

    @property
    def numel(self) -> int:
        return self.model_value().numel()


def _add_view(
    output: list[OptimizerParameterView],
    *,
    entry: tuple[str, torch.nn.Parameter, list[str]],
    optimizer_parameter: torch.Tensor,
    inner: Any,
    kind: str,
    requested_optimizer: str,
    start: int = 0,
    stop: int | None = None,
) -> None:
    name, model_parameter, groups = entry
    if not _is_unique_model_parallel_parameter(model_parameter):
        return
    branch = _branch_name(kind, requested_optimizer)
    optimizer_group = _group_for_parameter(inner, optimizer_parameter)
    output.append(
        OptimizerParameterView(
            name=name,
            model_parameter=model_parameter,
            optimizer_parameter=optimizer_parameter,
            inner_optimizer=inner,
            optimizer_group=optimizer_group,
            optimizer_kind=kind,
            optimizer_branch=branch,
            group_names=tuple((*groups, f"optimizer_branch/{branch}")),
            start=start,
            stop=stop,
        )
    )


def build_optimizer_parameter_views(
    entries: list[tuple[str, torch.nn.Parameter, list[str]]],
    optimizer: Any,
    *,
    requested_optimizer: str,
) -> list[OptimizerParameterView]:
    """Resolve every selected model parameter to its real optimizer tensor."""

    entry_by_id = {id(parameter): entry for entry in entries for parameter in (entry[1],)}
    output: list[OptimizerParameterView] = []
    root_is_layerwise = "layerwise" in type(optimizer).__name__.lower()

    for leaf in _leaf_optimizers(optimizer):
        inner = _inner_optimizer(leaf)
        kind = _optimizer_kind(inner)

        # ZeRO owns arbitrary flat ranges of model parameters on every data
        # parallel rank.  Its public maps are the authoritative membership and
        # range source.
        distributed_map = getattr(leaf, "model_param_group_index_map", None)
        if distributed_map is not None and hasattr(leaf, "_get_model_param_range_map"):
            for model_parameter, (group_id, parameter_id) in distributed_map.items():
                entry = entry_by_id.get(id(model_parameter))
                if entry is None:
                    continue
                parameter_range = leaf._get_model_param_range_map(model_parameter)["param"]
                optimizer_parameter = inner.param_groups[group_id]["params"][parameter_id]
                _add_view(
                    output,
                    entry=entry,
                    optimizer_parameter=optimizer_parameter,
                    inner=inner,
                    kind=kind,
                    requested_optimizer=requested_optimizer,
                    start=int(parameter_range.start),
                    stop=int(parameter_range.end),
                )
            continue

        # Mixed-precision Megatron wrappers expose aligned model/master lists.
        aligned = False
        for model_groups_name, main_groups_name in (
            ("float16_groups", "fp32_from_float16_groups"),
            ("fp32_from_fp32_groups", "fp32_from_fp32_groups"),
        ):
            model_groups = getattr(leaf, model_groups_name, None)
            main_groups = getattr(leaf, main_groups_name, None)
            if model_groups is None or main_groups is None:
                continue
            aligned = True
            for model_group, main_group in zip(model_groups, main_groups, strict=True):
                for model_parameter, optimizer_parameter in zip(model_group, main_group, strict=True):
                    entry = entry_by_id.get(id(model_parameter))
                    if entry is None:
                        continue
                    if not root_is_layerwise and not _is_data_parallel_contributor():
                        continue
                    _add_view(
                        output,
                        entry=entry,
                        optimizer_parameter=optimizer_parameter,
                        inner=inner,
                        kind=kind,
                        requested_optimizer=requested_optimizer,
                    )
        if aligned:
            continue

        # Plain torch optimizers are used by CPU tests and lightweight callers.
        if not _is_data_parallel_contributor():
            continue
        for group in getattr(inner, "param_groups", []):
            for parameter in group["params"]:
                entry = entry_by_id.get(id(parameter))
                if entry is None:
                    continue
                _add_view(
                    output,
                    entry=entry,
                    optimizer_parameter=parameter,
                    inner=inner,
                    kind=kind,
                    requested_optimizer=requested_optimizer,
                )

    # Duplicate identities indicate an optimizer adapter bug and would silently
    # double-count energy, so fail before recording scientifically invalid data.
    identities = [(id(view.model_parameter), view.start, view.stop) for view in output]
    if len(identities) != len(set(identities)):
        raise RuntimeError("A model parameter range was assigned to multiple optimizer branches.")
    return output


__all__ = ["OptimizerParameterView", "build_optimizer_parameter_views"]
