"""Megatron train-step hooks for distributed parameter geometry.

Exact observations keep only the pre-step/reference tensors required to recover
realized model-dtype changes and stream scalar sufficient statistics without
ever concatenating a full-model vector. Bounded CountSketch, histogram, support,
and matrix diagnostics run on the configured low-frequency cadence.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .directions import UpdateVectors, compute_update_vectors
from .exact import ExactGeometryAccumulator
from .histograms import SPARSITY_VECTORS, LowFrequencyHistogramAccumulator
from .matrix_metrics import matrix_diagnostics, matrix_macro_summary, selected_view_ids
from .metrics import geometry_metrics
from .optimizer_views import OptimizerParameterView, build_optimizer_parameter_views
from .projection import count_sketch_many
from .support import SupportWindowSketch


@dataclass
class _Snapshot:
    weight: torch.Tensor
    gradient: torch.Tensor
    weight_sq: torch.Tensor
    gradient_sq: torch.Tensor
    parameter_count: torch.Tensor


@dataclass
class _ExactBefore:
    view: OptimizerParameterView
    theta: torch.Tensor
    reference: torch.Tensor
    main: torch.Tensor | None
    group_ids: tuple[int, ...]
    raw_gradient_sq: torch.Tensor
    had_gradient: bool
    semantic_raw_gradient_sq: dict[str, torch.Tensor]


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _distributed() else 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if _distributed() else 1


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    while hasattr(module, "module"):
        module = module.module
    return module


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "to_local"):
        value = value.to_local()
    if hasattr(value, "_local_tensor"):
        value = value._local_tensor
    return value


def _gradient(parameter: torch.nn.Parameter) -> torch.Tensor | None:
    for attribute in ("main_grad", "decoupled_grad", "grad"):
        value = getattr(parameter, attribute, None)
        if value is not None:
            return _local_tensor(value)
    return None


@torch.no_grad()
def _squared_l2(value: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Accumulate a non-projected squared norm without a full FP64 clone."""

    flat = _local_tensor(value.detach()).reshape(-1)
    result = torch.zeros((), dtype=torch.float64, device=flat.device)
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size].to(torch.float32)
        result.add_(torch.sum(chunk.square(), dtype=torch.float64))
    return result


def _is_unique_model_parallel_parameter(parameter: torch.nn.Parameter) -> bool:
    try:
        from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
        from megatron.core.transformer.module import param_is_not_shared

        return bool(param_is_not_tensor_parallel_duplicate(parameter) and param_is_not_shared(parameter))
    except (ImportError, RuntimeError, AssertionError):
        # CPU unit tests and non-Megatron callers have neither parallel state nor
        # Megatron parameter annotations, in which case every parameter is unique.
        return True


def _is_data_parallel_contributor() -> bool:
    if not _distributed():
        return True
    try:
        from megatron.core import mpu

        return mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    except (ImportError, RuntimeError, AssertionError):
        return _rank() == 0


def _is_sample_contributor() -> bool:
    if not _distributed():
        return True
    try:
        from megatron.core import mpu

        return bool(
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
            and (not hasattr(mpu, "get_expert_model_parallel_rank") or mpu.get_expert_model_parallel_rank() == 0)
        )
    except (ImportError, RuntimeError, AssertionError):
        return _rank() == 0


def _module_group(parameter_name: str) -> str:
    name = re.sub(r"\.(?:weight|bias)$", "", parameter_name)
    return f"module/{name}"


def _operator_groups(parameter_name: str) -> list[str]:
    """Classify transformer operators from semantic module names."""

    name = parameter_name.lower()
    if any(token in name for token in ("linear_qkv", "query_key_value", "qkv_proj")):
        return ["operator_type/qkv_fused"]
    if any(token in name for token in ("linear_fc1", "dense_h_to_4h", "gate_up_proj")):
        return ["operator_type/gate_up_fused"]
    patterns = (
        (("q_proj", "linear_q_proj", ".query."), "q"),
        (("k_proj", "linear_k_proj", ".key."), "k"),
        (("v_proj", "linear_v_proj", ".value."), "v"),
        (("o_proj", "linear_proj", "out_proj", "dense_attention"), "o"),
        (("gate_proj",), "gate"),
        (("up_proj",), "up"),
        (("down_proj", "linear_fc2", "dense_4h_to_h"), "down"),
    )
    for tokens, operator in patterns:
        if any(token in name for token in tokens):
            return [f"operator_type/{operator}"]
    return []


def _aggregate_counter(local: Counter) -> dict[str, int]:
    if _distributed():
        gathered: list[dict[str, int] | None] = [None] * _world_size()
        torch.distributed.all_gather_object(gathered, dict(local))
        local = Counter()
        for values in gathered:
            local.update(values or {})
    return dict(sorted(local.items()))


def _aggregate_optimizer_metadata(
    local: dict[str, dict[str, set[Any]]],
) -> dict[str, dict[str, Any]]:
    serializable = {
        branch: {name: list(values) for name, values in fields.items()} for branch, fields in local.items()
    }
    gathered: list[dict[str, dict[str, list[Any]]] | None] = [serializable]
    if _distributed():
        gathered = [None] * _world_size()
        torch.distributed.all_gather_object(gathered, serializable)
    merged: dict[str, dict[str, set[Any]]] = {}
    for rank_metadata in gathered:
        for branch, fields in (rank_metadata or {}).items():
            destination = merged.setdefault(branch, {})
            for name, values in fields.items():
                destination.setdefault(name, set()).update(values)
    output: dict[str, dict[str, Any]] = {}
    for branch, fields in sorted(merged.items()):
        output[branch] = {}
        for name, values in sorted(fields.items()):
            ordered = sorted(values, key=lambda value: (type(value).__name__, repr(value)))
            output[branch][name] = ordered[0] if len(ordered) == 1 else ordered
    return output


def _last_jsonl_record(path: Path) -> dict[str, Any] | None:
    """Read only the final non-empty JSONL record, including after a large run."""

    if not path.is_file():
        return None
    with path.open("rb") as stream:
        position = stream.seek(0, os.SEEK_END)
        buffer = b""
        while position > 0:
            read_size = min(position, 8192)
            position -= read_size
            stream.seek(position)
            buffer = stream.read(read_size) + buffer
            stripped = buffer.rstrip()
            boundary = stripped.rfind(b"\n")
            if boundary >= 0:
                return json.loads(stripped[boundary + 1 :])
        stripped = buffer.strip()
        return json.loads(stripped) if stripped else None


class GeometryObserver:
    def __init__(self, args: Any, model: Sequence[torch.nn.Module]):
        self.args = args
        self.role = getattr(args, "_slime_model_role", "actor")
        self.enabled = self.role in set(args.geometry_roles)
        self.dim = int(args.geometry_projection_dim)
        self.interval = int(args.geometry_interval)
        self.seed = int(args.geometry_seed)
        self.chunk_size = int(args.geometry_sketch_chunk_size)
        self.include = re.compile(args.geometry_parameter_include)
        self.exclude = re.compile(args.geometry_parameter_exclude) if args.geometry_parameter_exclude else None
        self.output_dir = Path(os.path.expandvars(os.path.expanduser(args.geometry_output_dir))) / self.role
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir = self.output_dir / "vectors"
        if args.geometry_save_vectors:
            self.vector_dir.mkdir(parents=True, exist_ok=True)

        self._step_counter = 0
        self._persisted_frontier: tuple[int, int, int] | None = None
        self._last_attempt_position: tuple[int, int] | None = None
        self._successful_updates = 0
        self._failed_updates = 0
        self._clipped_updates = 0
        self._cumulative_prompts = 0
        self._cumulative_effective_tokens = 0
        metrics_path = self.output_dir / "metrics.jsonl"
        previous = _last_jsonl_record(metrics_path)
        if previous is not None:
            missing_frontier = {"observation_id", "rollout_id", "step_id"}.difference(previous)
            if missing_frontier:
                raise ValueError(
                    f"Last geometry record in {metrics_path} is missing resume fields: {sorted(missing_frontier)}."
                )
            self._persisted_frontier = (
                int(previous["observation_id"]),
                int(previous["rollout_id"]),
                int(previous["step_id"]),
            )
            self._step_counter = self._persisted_frontier[0] + 1
            counters = previous.get("run_update_counters", {})
            self._successful_updates = int(counters.get("successful", previous.get("num_updates", 0)))
            self._failed_updates = int(counters.get("failed_or_skipped", 0))
            self._clipped_updates = int(counters.get("clipped", 0))
            self._cumulative_prompts = int(previous.get("cumulative_prompt_count", 0))
            self._cumulative_effective_tokens = int(previous.get("cumulative_effective_token_count", 0))
        self._active = False
        self._sketch_active = False
        self._resume_frontier_checked = False
        self._before: _Snapshot | None = None
        self._initial_weight: torch.Tensor | None = None
        self._optimizer_identity: int | None = None
        self._optimizer_views: list[OptimizerParameterView] = []
        self._exact_references: list[torch.Tensor] | None = None
        self._exact_before: list[_ExactBefore] = []
        self._exact_accumulator: ExactGeometryAccumulator | None = None
        self._support_sketch: SupportWindowSketch | None = None
        self._matrix_view_ids: set[int] = set()
        self._semantic_layouts: dict[int, tuple[str, tuple[int, ...]]] = {}
        self._entries = self._build_entries(model)
        self.group_names = self._build_group_names(self._entries)
        self.group_index = {name: index for index, name in enumerate(self.group_names)}

    def _build_entries(self, model: Sequence[torch.nn.Module]):
        entries = []
        seen: set[int] = set()
        for chunk_index, wrapped_chunk in enumerate(model):
            chunk = _unwrap(wrapped_chunk)
            layer_by_parameter: dict[int, int] = {}
            for _, submodule in chunk.named_modules():
                layer_number = getattr(submodule, "layer_number", None)
                if not isinstance(layer_number, int):
                    continue
                for parameter in submodule.parameters(recurse=True):
                    layer_by_parameter.setdefault(id(parameter), layer_number - 1)

            for local_name, parameter in chunk.named_parameters():
                if id(parameter) in seen or not parameter.requires_grad:
                    continue
                seen.add(id(parameter))
                name = f"chunk{chunk_index}.{local_name}"
                if not self.include.search(name) or (self.exclude and self.exclude.search(name)):
                    continue

                specific_groups: list[str] = []
                if self.args.geometry_group_by == "layer":
                    layer_id = layer_by_parameter.get(id(parameter))
                    if layer_id is not None:
                        specific_groups.append(f"layer/{layer_id:04d}")
                    elif "output_layer" in name or ".lm_head" in name:
                        specific_groups.append("output")
                    elif "embedding" in name:
                        specific_groups.append("embedding")
                    elif getattr(parameter, "is_embedding_or_output_parameter", False):
                        specific_groups.append("embedding_or_output")
                    else:
                        specific_groups.append("other")
                elif self.args.geometry_group_by == "module":
                    specific_groups.append(_module_group(name))

                specific_groups.extend(_operator_groups(name))
                if len(parameter.shape) == 2 and any(
                    token in name.lower() for token in ("linear_qkv", "query_key_value", "qkv_proj")
                ):
                    config = getattr(chunk, "config", None)
                    if config is not None:
                        query_groups = int(getattr(config, "num_query_groups", 0) or 0)
                        attention_heads = int(getattr(config, "num_attention_heads", 0) or 0)
                        kv_channels = int(getattr(config, "kv_channels", 0) or 0)
                        if query_groups > 0 and attention_heads % query_groups == 0 and kv_channels > 0:
                            self._semantic_layouts[id(parameter)] = (
                                "qkv",
                                (
                                    attention_heads // query_groups * kv_channels,
                                    kv_channels,
                                    kv_channels,
                                ),
                            )
                elif len(parameter.shape) == 2 and any(
                    token in name.lower() for token in ("linear_fc1", "dense_h_to_4h", "gate_up_proj")
                ):
                    if parameter.shape[0] % 2 == 0:
                        self._semantic_layouts[id(parameter)] = (
                            "gate_up",
                            (int(parameter.shape[0]) // 2,),
                        )
                entries.append((name, parameter, specific_groups))
        return entries

    @staticmethod
    def _build_group_names(entries) -> list[str]:
        local_names = sorted({group for _, _, groups in entries for group in groups})
        if _distributed():
            gathered: list[list[str] | None] = [None] * _world_size()
            torch.distributed.all_gather_object(gathered, local_names)
            local_names = sorted({name for names in gathered for name in (names or [])})
        return ["global", *local_names]

    def _device(self) -> torch.device:
        if self._entries:
            return _local_tensor(self._entries[0][1]).device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _ensure_optimizer_views(self, optimizer: Any) -> None:
        if self._optimizer_identity == id(optimizer):
            return
        views = build_optimizer_parameter_views(
            self._entries,
            optimizer,
            requested_optimizer=str(self.args.optimizer),
        )
        branches_by_name: dict[str, set[str]] = {}
        for view in views:
            branches_by_name.setdefault(view.name, set()).add(f"optimizer_branch/{view.optimizer_branch}")
        if _distributed():
            gathered: list[dict[str, set[str]] | None] = [None] * _world_size()
            torch.distributed.all_gather_object(gathered, branches_by_name)
            branches_by_name = {}
            for rank_mapping in gathered:
                for name, branches in (rank_mapping or {}).items():
                    branches_by_name.setdefault(name, set()).update(branches)
        for name, _, groups in self._entries:
            for branch in sorted(branches_by_name.get(name, ())):
                if branch not in groups:
                    groups.append(branch)

        self.group_names = self._build_group_names(self._entries)
        semantic_groups = {group for view in views for group in self._semantic_group_names(view)}
        if _distributed():
            gathered_semantic: list[set[str] | None] = [None] * _world_size()
            torch.distributed.all_gather_object(gathered_semantic, semantic_groups)
            semantic_groups = set().union(*(groups or set() for groups in gathered_semantic))
        self.group_names = [
            "global",
            *sorted(set(self.group_names[1:]).union(semantic_groups)),
        ]
        self.group_index = {name: index for index, name in enumerate(self.group_names)}
        self._optimizer_views = views
        self._exact_references = None
        support_descriptors = [
            {
                "name": view.name,
                "start": int(view.start),
                "stop": None if view.stop is None else int(view.stop),
                "numel": int(view.numel),
                "optimizer_branch": view.optimizer_branch,
                "seed": self.seed,
            }
            for view in views
        ]
        self._support_sketch = SupportWindowSketch(
            group_names=self.group_names,
            descriptors=support_descriptors,
            sample_size=int(getattr(self.args, "geometry_support_sample_size", 1024)),
            window=int(getattr(self.args, "geometry_support_window", 8)),
            device=self._device(),
            path=self.output_dir / "support_state" / f"rank_{_rank():05d}.pt",
        )
        self._matrix_view_ids = selected_view_ids(
            views,
            seed=self.seed,
            count=int(getattr(self.args, "geometry_matrix_sample_count", 1)),
        )
        self._optimizer_identity = id(optimizer)

    def _semantic_group_names(self, view: OptimizerParameterView) -> tuple[str, ...]:
        layout = self._semantic_layouts.get(id(view.model_parameter))
        if layout is None:
            return ()
        if layout[0] == "qkv":
            return ("operator_type/q", "operator_type/k", "operator_type/v")
        if layout[0] == "gate_up":
            return ("operator_type/gate", "operator_type/up")
        raise RuntimeError(f"Unknown semantic operator layout {layout[0]!r}.")

    def _semantic_masks(self, view: OptimizerParameterView) -> dict[str, torch.Tensor]:
        layout = self._semantic_layouts.get(id(view.model_parameter))
        if layout is None:
            return {}
        parameter = _local_tensor(view.model_parameter.detach())
        if parameter.ndim != 2:
            raise ValueError(f"Semantic operator split requires a matrix: {view.name}.")
        start = int(view.start)
        stop = int(view.stop) if view.stop is not None else parameter.numel()
        flat_indices = torch.arange(start, stop, dtype=torch.int64, device=parameter.device)
        rows = torch.div(flat_indices, int(parameter.shape[1]), rounding_mode="floor")
        if layout[0] == "qkv":
            q_size, k_size, v_size = layout[1]
            block = q_size + k_size + v_size
            if int(parameter.shape[0]) % block != 0:
                raise ValueError(
                    f"Fused QKV rows for {view.name} are not divisible by configured Q/K/V block {layout[1]}."
                )
            within = torch.remainder(rows, block)
            return {
                "operator_type/q": within < q_size,
                "operator_type/k": (within >= q_size) & (within < q_size + k_size),
                "operator_type/v": within >= q_size + k_size,
            }
        half = layout[1][0]
        return {
            "operator_type/gate": rows < half,
            "operator_type/up": rows >= half,
        }

    def _matrix_records_for_view(
        self,
        view: OptimizerParameterView,
        vectors: dict[str, torch.Tensor],
        semantic_masks: dict[str, torch.Tensor],
    ) -> list[dict[str, Any]]:
        """Diagnose one fixed full matrix before and after semantic splitting."""

        parameter = _local_tensor(view.model_parameter.detach())
        if parameter.ndim != 2 or view.numel != parameter.numel():
            return []
        columns = int(parameter.shape[1])
        whole_operators = [
            group.removeprefix("operator_type/") for group in view.group_names if group.startswith("operator_type/")
        ]
        whole_operator = whole_operators[0] if whole_operators else "unclassified"
        slices: list[tuple[str, torch.Tensor | None]] = [(whole_operator, None)]
        slices.extend((name.removeprefix("operator_type/"), mask) for name, mask in semantic_masks.items())
        selected_vectors = {
            name: vectors[name]
            for name in (
                "post_ns",
                "delta_intended_fp32",
                "delta_model",
                "displacement",
            )
            if name in vectors
        }
        output: list[dict[str, Any]] = []
        randomized_rank = int(getattr(self.args, "geometry_matrix_randomized_rank", 16))
        for operator, mask in slices:
            diagnostics: dict[str, dict[str, Any]] = {}
            shape: list[int] | None = None
            for vector_name, vector in selected_vectors.items():
                flat = vector.reshape(-1)
                selected = flat if mask is None else flat[mask]
                if selected.numel() % columns:
                    raise ValueError(f"Sampled matrix slice {view.name}/{operator} does not contain complete rows.")
                matrix = selected.reshape(-1, columns)
                shape = list(matrix.shape)
                diagnostics[vector_name] = matrix_diagnostics(
                    matrix,
                    name=f"{view.name}:{operator}:{vector_name}",
                    seed=self.seed,
                    randomized_rank=randomized_rank,
                    include_orthogonality=vector_name == "post_ns",
                )
            if diagnostics:
                output.append(
                    {
                        "rank": _rank(),
                        "name": view.name,
                        "optimizer_branch": view.optimizer_branch,
                        "operator": operator,
                        "shape": shape,
                        "vectors": diagnostics,
                    }
                )
        return output

    def _load_or_create_exact_references(self) -> list[torch.Tensor]:
        directory = self.output_dir / "exact_reference"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank_{_rank():05d}.pt"
        descriptors = [
            {
                "name": view.name,
                "start": view.start,
                "stop": view.stop,
                "shape": list(view.model_value().shape),
                "dtype": str(view.model_value().dtype),
                "optimizer_branch": view.optimizer_branch,
            }
            for view in self._optimizer_views
        ]

        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload.get("schema_version") != 2
                or payload.get("world_size") != _world_size()
                or payload.get("projection_seed") != self.seed
                or payload.get("views") != descriptors
            ):
                raise ValueError(
                    f"Exact geometry reference at {path} is incompatible with the current model, "
                    "optimizer branches, parameter ownership, seed, or distributed topology."
                )
            tensors = payload.get("tensors", [])
            if len(tensors) != len(self._optimizer_views):
                raise ValueError(f"Exact geometry reference at {path} has the wrong tensor count.")
            return [
                tensor.to(device=view.model_value().device, dtype=view.model_value().dtype)
                for tensor, view in zip(tensors, self._optimizer_views, strict=True)
            ]

        if (self.output_dir / "metrics.jsonl").exists():
            raise ValueError(
                f"Exact geometry reference {path} is missing for an existing metrics log. "
                "Restore it with the checkpoint or use a new geometry output directory."
            )
        references = [view.model_value().detach().clone() for view in self._optimizer_views]
        self._atomic_torch_save(
            {
                "schema_version": 2,
                "world_size": _world_size(),
                "projection_seed": self.seed,
                "views": descriptors,
                "tensors": [tensor.cpu() for tensor in references],
            },
            path,
        )
        return references

    def _snapshot(self, include_gradient: bool) -> _Snapshot:
        group_count = len(self.group_names)
        device = self._device()
        weight = torch.zeros((group_count, self.dim), dtype=torch.float32, device=device)
        gradient = torch.zeros_like(weight)
        weight_sq = torch.zeros(group_count, dtype=torch.float64, device=device)
        gradient_sq = torch.zeros_like(weight_sq)
        parameter_count = torch.zeros(group_count, dtype=torch.float64, device=device)

        contribute = _is_data_parallel_contributor()
        rank = _rank()
        if contribute:
            for name, parameter, specific_groups in self._entries:
                if not _is_unique_model_parallel_parameter(parameter):
                    continue
                parameter_value = _local_tensor(parameter.detach())
                group_ids = [0, *(self.group_index[group] for group in specific_groups)]
                projection_name = f"rank{rank}:{name}"
                grad = _gradient(parameter) if include_gradient else None
                projection_values = [parameter_value, *([grad] if grad is not None else [])]
                projections = count_sketch_many(
                    projection_values,
                    self.dim,
                    seed=self.seed,
                    name=projection_name,
                    chunk_size=self.chunk_size,
                )
                parameter_projection = projections[0]
                parameter_sq = _squared_l2(parameter_value, self.chunk_size)
                for group_id in group_ids:
                    weight[group_id].add_(parameter_projection)
                    weight_sq[group_id].add_(parameter_sq)
                    parameter_count[group_id].add_(parameter_value.numel())

                if grad is not None:
                    grad_projection = projections[1]
                    grad_sq = _squared_l2(grad, self.chunk_size)
                    for group_id in group_ids:
                        gradient[group_id].add_(grad_projection)
                        gradient_sq[group_id].add_(grad_sq)

        if _distributed():
            projected = torch.cat((weight.reshape(-1), gradient.reshape(-1)))
            torch.distributed.all_reduce(projected)
            split = weight.numel()
            weight.copy_(projected[:split].reshape_as(weight))
            gradient.copy_(projected[split:].reshape_as(gradient))

            scalar_sums = torch.cat((weight_sq, gradient_sq, parameter_count))
            torch.distributed.all_reduce(scalar_sums)
            group_count = weight_sq.numel()
            weight_sq.copy_(scalar_sums[:group_count])
            gradient_sq.copy_(scalar_sums[group_count : 2 * group_count])
            parameter_count.copy_(scalar_sums[2 * group_count :])
        return _Snapshot(weight, gradient, weight_sq, gradient_sq, parameter_count)

    def _baseline_signature(self, parameter_count: torch.Tensor) -> dict[str, Any]:
        return {
            "role": self.role,
            "optimizer": str(self.args.optimizer),
            "advantage_estimator": str(self.args.advantage_estimator),
            "use_opd": bool(self.args.use_opd),
            "optimizer_config": {
                "lr": float(getattr(self.args, "lr", 0.0) or 0.0),
                "weight_decay": float(getattr(self.args, "weight_decay", 0.0) or 0.0),
                "adam_beta1": float(getattr(self.args, "adam_beta1", 0.0) or 0.0),
                "adam_beta2": float(getattr(self.args, "adam_beta2", 0.0) or 0.0),
                "adam_eps": float(getattr(self.args, "adam_eps", 0.0) or 0.0),
                "sgd_momentum": float(getattr(self.args, "sgd_momentum", 0.0) or 0.0),
                "clip_grad": float(getattr(self.args, "clip_grad", 0.0) or 0.0),
                "decoupled_weight_decay": bool(getattr(self.args, "decoupled_weight_decay", False)),
                "lr_decay_style": getattr(self.args, "lr_decay_style", None),
                "lr_warmup_iters": int(getattr(self.args, "lr_warmup_iters", 0) or 0),
                "muon_momentum": float(getattr(self.args, "muon_momentum", 0.0) or 0.0),
                "muon_use_nesterov": bool(getattr(self.args, "muon_use_nesterov", False)),
                "muon_split_qkv": bool(getattr(self.args, "muon_split_qkv", True)),
                "muon_num_ns_steps": int(getattr(self.args, "muon_num_ns_steps", 0) or 0),
                "muon_scale_mode": getattr(self.args, "muon_scale_mode", None),
                "muon_tp_mode": getattr(self.args, "muon_tp_mode", None),
                "muon_fp32_matmul_prec": getattr(self.args, "muon_fp32_matmul_prec", None),
                "muon_extra_scale_factor": float(getattr(self.args, "muon_extra_scale_factor", 1.0)),
            },
            "algorithm_config": {
                "loss_type": str(getattr(self.args, "loss_type", "policy_loss")),
                "custom_loss_function_path": getattr(self.args, "custom_loss_function_path", None),
                "entropy_coef": float(getattr(self.args, "entropy_coef", 0.0) or 0.0),
                "eps_clip": float(getattr(self.args, "eps_clip", 0.0) or 0.0),
                "eps_clip_high": float(getattr(self.args, "eps_clip_high", 0.0) or 0.0),
                "value_clip": float(getattr(self.args, "value_clip", 0.0) or 0.0),
                "normalize_advantages": bool(getattr(self.args, "normalize_advantages", False)),
                "use_tis": bool(getattr(self.args, "use_tis", False)),
                "tis_clip": float(getattr(self.args, "tis_clip", 0.0) or 0.0),
                "tis_clip_low": float(getattr(self.args, "tis_clip_low", 0.0) or 0.0),
                "opd_type": getattr(self.args, "opd_type", None),
                "opd_kl_coef": float(getattr(self.args, "opd_kl_coef", 0.0) or 0.0),
                "opd_task_reward_weight": float(getattr(self.args, "opd_task_reward_weight", 0.0) or 0.0),
                "hybrid_sft_loss_coef": float(getattr(self.args, "hybrid_sft_loss_coef", 0.0) or 0.0),
                "hybrid_opd_loss_coef": float(getattr(self.args, "hybrid_opd_loss_coef", 0.0) or 0.0),
            },
            "data_config": {
                "prompt_data": getattr(self.args, "prompt_data", None),
                "task_sampling_seed": getattr(self.args, "m2rl_task_sampling_seed", None),
                "rollout_seed": int(getattr(self.args, "rollout_seed", 0) or 0),
                "num_rollout": int(getattr(self.args, "num_rollout", 0) or 0),
                "rollout_batch_size": int(getattr(self.args, "rollout_batch_size", 0) or 0),
                "n_samples_per_prompt": int(getattr(self.args, "n_samples_per_prompt", 1) or 1),
                "global_batch_size": int(getattr(self.args, "global_batch_size", 0) or 0),
                "rollout_max_prompt_len": getattr(self.args, "rollout_max_prompt_len", None),
                "rollout_max_response_len": getattr(self.args, "rollout_max_response_len", None),
                "rollout_temperature": float(getattr(self.args, "rollout_temperature", 0.0) or 0.0),
                "rollout_top_p": float(getattr(self.args, "rollout_top_p", 0.0) or 0.0),
                "rollout_top_k": int(getattr(self.args, "rollout_top_k", -1)),
                "rollout_shuffle": bool(getattr(self.args, "rollout_shuffle", False)),
                "sglang_deterministic_inference": bool(
                    getattr(self.args, "sglang_enable_deterministic_inference", False)
                ),
                "experiment_task": getattr(self.args, "experiment_task", None),
                "experiment_teacher": getattr(self.args, "experiment_teacher", None),
                "experiment_condition": getattr(self.args, "experiment_condition", None),
                "experiment_name": getattr(self.args, "experiment_name", None),
                "experiment_optimizer": getattr(self.args, "experiment_optimizer", None),
                "experiment_data_index": getattr(self.args, "experiment_data_index", None),
            },
            "group_by": self.args.geometry_group_by,
            "interval": self.interval,
            "support_sample_size": int(getattr(self.args, "geometry_support_sample_size", 1024)),
            "support_window": int(getattr(self.args, "geometry_support_window", 8)),
            "matrix_sample_count": int(getattr(self.args, "geometry_matrix_sample_count", 1)),
            "matrix_randomized_rank": int(getattr(self.args, "geometry_matrix_randomized_rank", 16)),
            "parameter_include": self.args.geometry_parameter_include,
            "parameter_exclude": self.args.geometry_parameter_exclude,
            "parameter_count": parameter_count.detach().cpu().tolist(),
            "parallel_sizes": {
                "tensor": int(getattr(self.args, "tensor_model_parallel_size", 1)),
                "pipeline": int(getattr(self.args, "pipeline_model_parallel_size", 1)),
                "context": int(getattr(self.args, "context_parallel_size", 1)),
                "expert": int(getattr(self.args, "expert_model_parallel_size", 1)),
            },
        }

    def _validate_resume_frontier(self, rollout_id: int, step_id: int, observation_id: int) -> None:
        """Reject replay into geometry records that are ahead of a checkpoint."""

        if self._resume_frontier_checked:
            return
        frontier = -1
        frontier_error: Exception | None = None
        if _rank() == 0:
            try:
                persisted = self._persisted_frontier
                frontier = persisted[0] if persisted is not None else -1
                stored_position = persisted[1:] if persisted is not None else None
                current_position = (int(rollout_id), int(step_id))
                if observation_id <= frontier or (stored_position is not None and current_position <= stored_position):
                    raise ValueError(
                        f"Geometry output {self.output_dir} already ends at observation {frontier}, "
                        f"rollout/step {stored_position}, but training is attempting observation {observation_id}, "
                        f"rollout/step {current_position}. The geometry log is "
                        "ahead of (or equal to) the resumed checkpoint. Restore a matching checkpoint, "
                        "archive/truncate the stale records, or use a new output directory."
                    )
            except Exception as exc:
                frontier_error = exc

        if _distributed():
            state = torch.tensor(
                [int(frontier_error is not None), frontier],
                dtype=torch.int64,
                device=self._device(),
            )
            torch.distributed.broadcast(state, src=0)
            if state[0].item():
                if frontier_error is not None:
                    raise frontier_error
                raise RuntimeError(
                    f"Rank 0 rejected geometry resume at observation {observation_id}; "
                    f"the stored frontier is {int(state[1].item())}."
                )
        elif frontier_error is not None:
            raise frontier_error
        self._resume_frontier_checked = True

    def _load_or_create_initial(self, current: torch.Tensor, parameter_count: torch.Tensor) -> torch.Tensor:
        path = self.output_dir / "initial_projection.pt"
        signature = self._baseline_signature(parameter_count)
        exists = path.exists() if _rank() == 0 else False
        if _distributed():
            exists_tensor = torch.tensor([int(exists)], dtype=torch.int64, device=current.device)
            torch.distributed.broadcast(exists_tensor, src=0)
            exists = bool(exists_tensor.item())

        initial = torch.zeros_like(current)
        baseline_error: Exception | None = None
        if not exists:
            initial.copy_(current)
        if _rank() == 0:
            try:
                if exists:
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        payload["group_names"] != self.group_names
                        or payload["projection_dim"] != self.dim
                        or payload.get("seed") != self.seed
                        or payload.get("world_size") != _world_size()
                        or payload.get("signature") != signature
                    ):
                        raise ValueError(
                            f"Geometry baseline at {path} is incompatible with the current groups, projection, "
                            "seed, parallel topology, parameter selection, optimizer/algorithm condition, or "
                            "model size. Use a new geometry output directory."
                        )
                    initial.copy_(payload["weight"].to(device=current.device, dtype=current.dtype))
                else:
                    self._atomic_torch_save(
                        {
                            "group_names": self.group_names,
                            "projection_dim": self.dim,
                            "seed": self.seed,
                            "world_size": _world_size(),
                            "signature": signature,
                            "weight": initial.cpu(),
                        },
                        path,
                    )
            except Exception as exc:  # make every distributed rank fail instead of hanging
                baseline_error = exc

        if _distributed():
            failed = torch.tensor([int(baseline_error is not None)], dtype=torch.int64, device=current.device)
            torch.distributed.broadcast(failed, src=0)
            if failed.item():
                if baseline_error is not None:
                    raise baseline_error
                raise RuntimeError(f"Rank 0 failed to load or save geometry baseline {path}.")
            torch.distributed.broadcast(initial, src=0)
        elif baseline_error is not None:
            raise baseline_error
        return initial

    @staticmethod
    def _source_counts(source_names: Iterable[str]) -> dict[str, int]:
        local = Counter(str(name) for name in source_names) if _is_sample_contributor() else Counter()
        if _distributed():
            gathered: list[dict[str, int] | None] = [None] * _world_size()
            torch.distributed.all_gather_object(gathered, dict(local))
            local = Counter()
            for counts in gathered:
                local.update(counts or {})
        return dict(sorted(local.items()))

    def after_backward(
        self,
        rollout_id: int,
        step_id: int,
        source_names: Iterable[str],
        optimizer: Any | None = None,
        actual_batch_size: int = 0,
        effective_token_count: int = 0,
    ) -> None:
        position = (int(rollout_id), int(step_id))
        if self._last_attempt_position is not None and position <= self._last_attempt_position:
            raise ValueError(
                f"Geometry update positions must increase monotonically; got {position} after "
                f"{self._last_attempt_position}."
            )
        observation_id = self._step_counter
        self._observation_started_at = time.perf_counter()
        self._cumulative_prompts += int(actual_batch_size)
        self._cumulative_effective_tokens += int(effective_token_count)
        self._pending_ids = (rollout_id, step_id, observation_id)
        self._pending_source_names = list(source_names)
        self._pending_batch_size = int(actual_batch_size)
        self._pending_effective_tokens = int(effective_token_count)
        selected = self.enabled
        self._sketch_active = selected and observation_id % self.interval == 0
        self._step_counter = observation_id + 1
        self._active = selected
        if not selected:
            return
        self._validate_resume_frontier(rollout_id, step_id, observation_id)
        self._last_attempt_position = position
        if optimizer is not None:
            self._ensure_optimizer_views(optimizer)
        self._before = None
        if self._sketch_active:
            self._before = self._snapshot(include_gradient=True)
            if self._initial_weight is None:
                self._initial_weight = self._load_or_create_initial(self._before.weight, self._before.parameter_count)
        self._pending_source_counts = self._source_counts(source_names)

        self._exact_before = []
        self._exact_accumulator = None
        if optimizer is not None:
            if self._exact_references is None:
                self._exact_references = self._load_or_create_exact_references()
            self._exact_accumulator = ExactGeometryAccumulator(self.group_names, self._device())
            for view, reference in zip(self._optimizer_views, self._exact_references, strict=True):
                theta = view.model_value().detach().clone()
                raw_gradient = view.raw_gradient()
                had_gradient = raw_gradient is not None
                if raw_gradient is None:
                    raw_gradient = torch.zeros_like(view.model_value(), dtype=torch.float32)
                group_ids = tuple(dict.fromkeys([0, *(self.group_index[name] for name in view.group_names)]))
                self._exact_accumulator.add(group_ids, {"g_raw": raw_gradient})
                raw_gradient_sq = torch.sum(raw_gradient.reshape(-1).to(torch.float32).square(), dtype=torch.float64)
                semantic_raw_gradient_sq: dict[str, torch.Tensor] = {}
                for semantic_group, mask in self._semantic_masks(view).items():
                    semantic_gradient = raw_gradient.reshape(-1)[mask]
                    self._exact_accumulator.add([self.group_index[semantic_group]], {"g_raw": semantic_gradient})
                    semantic_raw_gradient_sq[semantic_group] = torch.sum(
                        semantic_gradient.to(torch.float32).square(), dtype=torch.float64
                    )
                main = (
                    view.optimizer_value().detach().clone()
                    if view.optimizer_kind in {"muon", "unknown"} and had_gradient
                    else None
                )
                self._exact_before.append(
                    _ExactBefore(
                        view=view,
                        theta=theta,
                        reference=reference,
                        main=main,
                        group_ids=group_ids,
                        raw_gradient_sq=raw_gradient_sq,
                        had_gradient=had_gradient,
                        semantic_raw_gradient_sq=semantic_raw_gradient_sq,
                    )
                )

    def _record_unobserved_failure(
        self,
        *,
        grad_norm: float | int | None,
        num_zeros_in_grad: Any,
        failure_reason: str | None,
    ) -> None:
        rollout_id, step_id, observation_id = self._pending_ids
        self._validate_resume_frontier(rollout_id, step_id, observation_id)
        source_counts = self._source_counts(self._pending_source_names)
        if _rank() != 0:
            return
        record = {
            "schema_version": 2,
            "record_type": "failed_or_skipped_update",
            "run_id": getattr(self.args, "experiment_name", None),
            "seed": int(getattr(self.args, "seed", 0)),
            "task": getattr(self.args, "experiment_task", None),
            "role": self.role,
            "optimizer": str(self.args.optimizer),
            "learning_rate": float(getattr(self.args, "lr", 0.0)),
            "model_dtype_parameter_counts": {},
            "experiment_name": getattr(self.args, "experiment_name", None),
            "experiment_task": getattr(self.args, "experiment_task", None),
            "experiment_condition": getattr(self.args, "experiment_condition", None),
            "experiment_seed": int(getattr(self.args, "seed", 0)),
            "rollout_seed": int(getattr(self.args, "rollout_seed", 0)),
            "rollout_id": int(rollout_id),
            "step_id": int(step_id),
            "observation_id": int(observation_id),
            "num_updates": self._successful_updates,
            "model_version": self._successful_updates,
            "actual_batch_size": self._pending_batch_size,
            "effective_token_count": self._pending_effective_tokens,
            "cumulative_prompt_count": self._cumulative_prompts,
            "cumulative_effective_token_count": self._cumulative_effective_tokens,
            "update_successful": False,
            "failure_reason": failure_reason or "update_skipped",
            "valid_update_metrics": False,
            "low_frequency_observation": False,
            "reported_grad_norm": grad_norm,
            "reported_num_zeros_in_grad": _to_number(num_zeros_in_grad),
            "run_clip_fraction": _safe_ratio_number(self._clipped_updates, self._successful_updates),
            "run_update_counters": {
                "successful": self._successful_updates,
                "failed_or_skipped": self._failed_updates,
                "clipped": self._clipped_updates,
            },
            "world_size": _world_size(),
            "source_counts": source_counts,
            "groups": {},
            "vector_file": None,
        }
        self._append_jsonl(self.output_dir / "metrics.jsonl", record)
        self._log_wandb(record)

    def after_step(
        self,
        *,
        update_successful: bool,
        grad_norm: Any,
        num_zeros_in_grad: Any,
        optimizer: Any | None = None,
        failure_reason: str | None = None,
    ) -> None:
        reported_grad_norm = _to_number(grad_norm)
        clip_threshold = float(getattr(self.args, "clip_grad", 0.0) or 0.0)
        grad_clipped = bool(
            update_successful
            and clip_threshold > 0.0
            and reported_grad_norm is not None
            and reported_grad_norm > clip_threshold
        )
        if update_successful:
            self._successful_updates += 1
            self._clipped_updates += int(grad_clipped)
        else:
            self._failed_updates += 1
        if not self._active:
            if not update_successful:
                self._record_unobserved_failure(
                    grad_norm=reported_grad_norm,
                    num_zeros_in_grad=num_zeros_in_grad,
                    failure_reason=failure_reason,
                )
            return
        rollout_id, step_id, observation_id = self._pending_ids

        group_metrics: dict[str, dict[str, Any]] = {}
        vectors: dict[str, dict[str, torch.Tensor]] = {}
        if update_successful and self._sketch_active:
            assert self._before is not None and self._initial_weight is not None
            after = self._snapshot(include_gradient=False)
            for group_id, group_name in enumerate(self.group_names):
                parameter_count = int(after.parameter_count[group_id].item())
                if parameter_count == 0:
                    continue
                scalars, group_vectors = geometry_metrics(
                    weight_before=self._before.weight[group_id],
                    weight_after=after.weight[group_id],
                    gradient=self._before.gradient[group_id],
                    initial_weight=self._initial_weight[group_id],
                    exact_weight_before_sq=float(self._before.weight_sq[group_id].item()),
                    exact_weight_after_sq=float(after.weight_sq[group_id].item()),
                    exact_gradient_sq=float(self._before.gradient_sq[group_id].item()),
                    parameter_count=parameter_count,
                )
                group_metrics[group_name] = scalars
                vectors[group_name] = {name: value.detach().cpu() for name, value in group_vectors.items()}

        clip_scale = 1.0
        optimizer_clip_scale = 1.0
        if clip_threshold > 0.0 and reported_grad_norm is not None and reported_grad_norm > 0.0:
            clip_scale = min(1.0, clip_threshold / reported_grad_norm)
            # Megatron uses the epsilon below in clip_grad_by_total_norm_fp32.
            optimizer_clip_scale = min(1.0, clip_threshold / (reported_grad_norm + 1.0e-6))

        optimizer_metadata: dict[str, dict[str, set[Any]]] = {}
        actual_optimizer_metadata: dict[str, dict[str, Any]] | None = None
        local_matrix_records: list[dict[str, Any]] = []
        histogram_accumulator = (
            LowFrequencyHistogramAccumulator(
                self.group_names,
                self._device(),
                chunk_size=self.chunk_size,
            )
            if update_successful and self._sketch_active
            else None
        )
        if update_successful and self._exact_accumulator is not None:
            if optimizer is None:
                raise RuntimeError("Exact geometry was activated without the optimizer at after_step.")
            if self._support_sketch is not None:
                self._support_sketch.begin(
                    self._successful_updates,
                    report=self._sketch_active,
                )
            for view_id, before in enumerate(self._exact_before):
                view = before.view
                if before.had_gradient:
                    result = compute_update_vectors(
                        view,
                        theta_before=before.theta,
                        theta_reference=before.reference,
                        theta_before_main=before.main,
                    )
                else:
                    theta_after = view.model_value().reshape(-1).to(torch.float32)
                    theta_before = before.theta.reshape(-1)
                    reference = before.reference.reshape(-1)
                    zeros = torch.zeros_like(theta_after)
                    result_vectors = {
                        "theta_before": theta_before,
                        "theta_reference": reference,
                        "g_opt": zeros,
                        "d_data": zeros,
                        "d_wd": zeros,
                        "delta_data_fp32": zeros,
                        "delta_wd_fp32": zeros,
                        "delta_intended_fp32": zeros,
                        "delta_model": theta_after - theta_before.to(torch.float32),
                        "displacement": theta_after - reference.to(torch.float32),
                    }
                    result = UpdateVectors(vectors=result_vectors)
                all_vectors = {**result.vectors, **result.optimizer_vectors}
                self._exact_accumulator.add(
                    before.group_ids,
                    all_vectors,
                )
                self._exact_accumulator.add_dot(
                    before.group_ids,
                    ("g_raw", "g_opt"),
                    before.raw_gradient_sq * optimizer_clip_scale,
                )
                if histogram_accumulator is not None:
                    histogram_accumulator.add_sparsity(
                        before.group_ids,
                        {name: all_vectors[name] for name in SPARSITY_VECTORS if name in all_vectors},
                    )
                    if view.optimizer_kind == "adam":
                        histogram_accumulator.add_adam(
                            before.group_ids,
                            sqrt_v_hat=all_vectors["sqrt_v_hat"],
                            effective_eta=all_vectors["effective_eta"],
                            gradient_energy=all_vectors["g_opt_squared"],
                            eps=float(result.metadata["adam_eps"]),
                        )
                semantic_masks = self._semantic_masks(view)
                if self._sketch_active and view_id in self._matrix_view_ids:
                    local_matrix_records.extend(self._matrix_records_for_view(view, all_vectors, semantic_masks))
                for semantic_group, mask in semantic_masks.items():
                    semantic_group_id = self.group_index[semantic_group]
                    self._exact_accumulator.add(
                        [semantic_group_id],
                        {name: value.reshape(-1)[mask] for name, value in all_vectors.items()},
                    )
                    self._exact_accumulator.add_dot(
                        [semantic_group_id],
                        ("g_raw", "g_opt"),
                        before.semantic_raw_gradient_sq[semantic_group] * optimizer_clip_scale,
                    )
                    if histogram_accumulator is not None:
                        histogram_accumulator.add_sparsity(
                            [semantic_group_id],
                            {
                                name: all_vectors[name].reshape(-1)[mask]
                                for name in SPARSITY_VECTORS
                                if name in all_vectors
                            },
                        )
                        if view.optimizer_kind == "adam":
                            histogram_accumulator.add_adam(
                                [semantic_group_id],
                                sqrt_v_hat=all_vectors["sqrt_v_hat"].reshape(-1)[mask],
                                effective_eta=all_vectors["effective_eta"].reshape(-1)[mask],
                                gradient_energy=all_vectors["g_opt_squared"].reshape(-1)[mask],
                                eps=float(result.metadata["adam_eps"]),
                            )
                if self._support_sketch is not None:
                    sampled_indices = self._support_sketch.indices[view_id]
                    semantic_sample_groups = {
                        self.group_index[name]: mask[sampled_indices.to(mask.device)]
                        for name, mask in semantic_masks.items()
                    }
                    self._support_sketch.add(
                        view_id,
                        all_vectors["delta_model"],
                        group_ids=before.group_ids,
                        semantic_groups=semantic_sample_groups,
                    )
                branch_metadata = optimizer_metadata.setdefault(view.optimizer_branch, {})
                for name, value in result.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        branch_metadata.setdefault(name, set()).add(value)

            exact_groups = self._exact_accumulator.finalize()
            for group_name, exact_metrics in exact_groups.items():
                group_metrics.setdefault(group_name, {}).update(exact_metrics)
            if histogram_accumulator is not None:
                for group_name, histogram_metrics in histogram_accumulator.finalize().items():
                    group_metrics.setdefault(group_name, {}).update(histogram_metrics)
            if self._support_sketch is not None:
                for group_name, support_metrics in self._support_sketch.finish().items():
                    group_metrics.setdefault(group_name, {}).update(support_metrics)
            actual_optimizer_metadata = _aggregate_optimizer_metadata(optimizer_metadata)

            global_metrics = group_metrics.get("global", {})
            global_parameter_count = float(global_metrics.get("parameter_count", 0))
            energy_fields = {
                "gradient_energy_fraction": "g_opt_l2",
                "intended_update_energy_fraction": "delta_intended_fp32_l2",
                "realized_update_energy_fraction": "delta_model_l2",
            }
            for group_name, metrics in group_metrics.items():
                if not group_name.startswith("optimizer_branch/"):
                    continue
                metrics["parameter_fraction"] = _safe_ratio_number(
                    float(metrics.get("parameter_count", 0)), global_parameter_count
                )
                for output_name, norm_name in energy_fields.items():
                    numerator = float(metrics.get(norm_name, 0.0)) ** 2
                    denominator = float(global_metrics.get(norm_name, 0.0)) ** 2
                    metrics[output_name] = _safe_ratio_number(numerator, denominator)

                branch = group_name.removeprefix("optimizer_branch/")
                branch_weight_decays = actual_optimizer_metadata.get(branch, {}).get("weight_decay", [])
                if not isinstance(branch_weight_decays, list):
                    branch_weight_decays = [branch_weight_decays]
                weight_decay_enabled = any(float(value or 0.0) != 0.0 for value in branch_weight_decays)
                if weight_decay_enabled:
                    metrics["weight_decay_metrics_applicability"] = "applicable"
                    metrics["d_wd_to_d_data_ratio"] = _safe_ratio_number(
                        float(metrics.get("d_wd_l2", 0.0)),
                        float(metrics.get("d_data_l2", 0.0)),
                    )
                else:
                    metrics["weight_decay_metrics_applicability"] = "not_applicable"
                    metrics["d_wd_to_d_data_ratio"] = None
                    metrics["delta_wd_to_delta_data_ratio"] = None
                if branch == "muon_matrix":
                    metrics["momentum_carry_ratio"] = _safe_ratio_number(
                        float(metrics.get("momentum_carry_l2", 0.0)),
                        float(metrics.get("momentum_innovation_l2", 0.0)),
                    )
                    aliases = {
                        "cos_g_opt_m": "cos_g_opt_momentum",
                        "cos_m_n": "cos_momentum_ns_input",
                        "cos_n_q": "cos_ns_input_post_ns",
                        "cos_g_opt_q": "cos_g_opt_post_ns",
                        "cos_q_delta_intended_fp32": "cos_post_ns_delta_intended_fp32",
                        "cos_m_t_m_prev": "cos_momentum_momentum_previous",
                    }
                elif branch in {"adam", "adam_fallback"}:
                    metrics["first_moment_carry_ratio"] = _safe_ratio_number(
                        float(metrics.get("first_moment_carry_l2", 0.0)),
                        float(metrics.get("first_moment_innovation_l2", 0.0)),
                    )
                    metrics["second_moment_carry_ratio"] = _safe_ratio_number(
                        float(metrics.get("second_moment_carry_l2", 0.0)),
                        float(metrics.get("second_moment_innovation_l2", 0.0)),
                    )
                    metrics["d_adam_to_g_opt_ratio"] = _safe_ratio_number(
                        float(metrics.get("d_adam_l2", 0.0)),
                        float(metrics.get("g_opt_l2", 0.0)),
                    )
                    metrics["d_wd_to_d_adam_ratio"] = (
                        _safe_ratio_number(
                            float(metrics.get("d_wd_l2", 0.0)),
                            float(metrics.get("d_adam_l2", 0.0)),
                        )
                        if weight_decay_enabled
                        else None
                    )
                    aliases = {
                        "cos_m_t_m_prev": "cos_first_moment_first_moment_previous",
                    }
                elif branch == "sgd":
                    metrics["momentum_carry_ratio"] = _safe_ratio_number(
                        float(metrics.get("velocity_carry_l2", 0.0)),
                        float(metrics.get("velocity_innovation_l2", 0.0)),
                    )
                    aliases = {
                        "cos_g_opt_v": "cos_g_opt_velocity",
                        "cos_v_t_v_prev": "cos_velocity_velocity_previous",
                    }
                else:
                    aliases = {}
                for alias, source in aliases.items():
                    metrics[alias] = metrics.get(source)

        matrix_records: list[dict[str, Any]] = []
        matrix_macro: dict[str, Any] = {}
        if update_successful and self._sketch_active:
            gathered_matrix_records: list[list[dict[str, Any]] | None] = [local_matrix_records]
            if _distributed():
                gathered_matrix_records = [None] * _world_size()
                torch.distributed.all_gather_object(
                    gathered_matrix_records,
                    local_matrix_records,
                )
            matrix_records = [record for rank_records in gathered_matrix_records for record in (rank_records or [])]
            matrix_macro = matrix_macro_summary(matrix_records)
            for operator, vector_summaries in matrix_macro.items():
                group_name = f"operator_type/{operator}"
                if group_name not in group_metrics:
                    continue
                destination = group_metrics[group_name]
                for vector_name, metric_summaries in vector_summaries.items():
                    for metric_name, summary in metric_summaries.items():
                        destination[f"matrix_macro_{vector_name}_{metric_name}_median_sketch"] = summary[
                            "median_sketch"
                        ]
                        destination[f"matrix_macro_{vector_name}_{metric_name}_iqr_sketch"] = summary["iqr_sketch"]
                        destination[f"matrix_macro_{vector_name}_{metric_name}_sample_count"] = summary["matrix_count"]

        vector_file = None
        if update_successful and self._sketch_active and self.args.geometry_save_vectors and _rank() == 0:
            vector_file = f"rollout_{rollout_id:08d}_step_{step_id:04d}_obs_{observation_id:08d}.pt"
            self._atomic_torch_save(
                {
                    "schema_version": 2,
                    "projection": "countsketch",
                    "projection_dim": self.dim,
                    "seed": self.seed,
                    "groups": vectors,
                },
                self.vector_dir / vector_file,
            )

        dtype_counts = _aggregate_counter(
            Counter(
                {
                    str(dtype): sum(
                        before.view.numel for before in self._exact_before if before.view.model_value().dtype == dtype
                    )
                    for dtype in {before.view.model_value().dtype for before in self._exact_before}
                }
            )
        )
        if actual_optimizer_metadata is None:
            actual_optimizer_metadata = _aggregate_optimizer_metadata(optimizer_metadata)
        actual_learning_rates = sorted(
            {
                float(value)
                for metadata in actual_optimizer_metadata.values()
                for value in (
                    metadata.get("learning_rate", [])
                    if isinstance(metadata.get("learning_rate"), list)
                    else [metadata.get("learning_rate")]
                )
                if value is not None
            }
        )
        observation_wall_time_ms = (time.perf_counter() - self._observation_started_at) * 1000.0

        if _rank() == 0:
            requested_optimizer = str(self.args.optimizer)
            configured_weight_decay = float(self.args.weight_decay)
            if requested_optimizer == "adam" and configured_weight_decay == 0.0:
                optimizer_display_name = "Adam (AdamW implementation, wd=0)"
            elif requested_optimizer == "adam":
                optimizer_display_name = "AdamW"
            else:
                optimizer_display_name = requested_optimizer
            record = {
                "schema_version": 2,
                "record_type": "optimizer_update",
                "run_id": getattr(self.args, "experiment_name", None),
                "seed": int(getattr(self.args, "seed", 0)),
                "task": getattr(self.args, "experiment_task", None),
                "role": self.role,
                "optimizer": (
                    "adamw"
                    if requested_optimizer == "adam" and getattr(self.args, "decoupled_weight_decay", True)
                    else requested_optimizer
                ),
                "optimizer_display_name": optimizer_display_name,
                "megatron_optimizer": requested_optimizer,
                "learning_rate": (
                    actual_learning_rates[0]
                    if len(actual_learning_rates) == 1
                    else (actual_learning_rates if actual_learning_rates else float(self.args.lr))
                ),
                "configured_learning_rate": float(self.args.lr),
                "actual_optimizer_branches": actual_optimizer_metadata,
                "model_dtype_parameter_counts": dtype_counts,
                "weight_decay": configured_weight_decay,
                "sgd_momentum": float(getattr(self.args, "sgd_momentum", 0.0)),
                "muon_momentum": float(getattr(self.args, "muon_momentum", 0.0)),
                "muon_num_ns_steps": int(getattr(self.args, "muon_num_ns_steps", 0)),
                "muon_scale_mode": getattr(self.args, "muon_scale_mode", None),
                "optimizer_hyperparameters": {
                    "clip_grad": float(getattr(self.args, "clip_grad", 0.0)),
                    "adam_beta1": float(getattr(self.args, "adam_beta1", 0.0)),
                    "adam_beta2": float(getattr(self.args, "adam_beta2", 0.0)),
                    "adam_eps": float(getattr(self.args, "adam_eps", 0.0)),
                    "sgd_momentum": float(getattr(self.args, "sgd_momentum", 0.0)),
                    "muon_momentum": float(getattr(self.args, "muon_momentum", 0.0)),
                    "muon_use_nesterov": bool(getattr(self.args, "muon_use_nesterov", False)),
                    "muon_split_qkv": bool(getattr(self.args, "muon_split_qkv", True)),
                    "muon_scale_mode": getattr(self.args, "muon_scale_mode", None),
                    "muon_fp32_matmul_prec": getattr(self.args, "muon_fp32_matmul_prec", None),
                    "muon_num_ns_steps": int(getattr(self.args, "muon_num_ns_steps", 0)),
                    "muon_tp_mode": getattr(self.args, "muon_tp_mode", None),
                    "muon_extra_scale_factor": float(getattr(self.args, "muon_extra_scale_factor", 1.0)),
                },
                "advantage_estimator": str(self.args.advantage_estimator),
                "experiment_task": getattr(self.args, "experiment_task", None),
                "experiment_teacher": getattr(self.args, "experiment_teacher", None),
                "experiment_condition": getattr(self.args, "experiment_condition", None),
                "experiment_name": getattr(self.args, "experiment_name", None),
                "experiment_optimizer": getattr(self.args, "experiment_optimizer", None),
                "experiment_data_index": getattr(self.args, "experiment_data_index", None),
                "loss_type": str(getattr(self.args, "loss_type", "policy_loss")),
                "custom_loss_function_path": getattr(self.args, "custom_loss_function_path", None),
                "use_opd": bool(self.args.use_opd),
                "opd_type": self.args.opd_type,
                "opd_kl_coef": float(getattr(self.args, "opd_kl_coef", 0.0)),
                "opd_task_reward_weight": float(getattr(self.args, "opd_task_reward_weight", 0.0)),
                "hybrid_sft_loss_coef": float(getattr(self.args, "hybrid_sft_loss_coef", 0.0)),
                "hybrid_opd_loss_coef": float(getattr(self.args, "hybrid_opd_loss_coef", 0.0)),
                "algorithm_hyperparameters": {
                    "eps_clip": float(getattr(self.args, "eps_clip", 0.0)),
                    "eps_clip_high": float(getattr(self.args, "eps_clip_high", 0.0)),
                    "value_clip": float(getattr(self.args, "value_clip", 0.0)),
                    "gamma": float(getattr(self.args, "gamma", 1.0)),
                    "lambda": float(getattr(self.args, "lambd", 1.0)),
                    "normalize_advantages": bool(getattr(self.args, "normalize_advantages", False)),
                    "rewards_normalization": bool(getattr(self.args, "rewards_normalization", False)),
                    "grpo_std_normalization": bool(getattr(self.args, "grpo_std_normalization", False)),
                    "n_samples_per_prompt": int(getattr(self.args, "n_samples_per_prompt", 1)),
                    "global_batch_size": int(getattr(self.args, "global_batch_size", 0) or 0),
                },
                "experiment_seed": int(getattr(self.args, "seed", 0)),
                "rollout_seed": int(getattr(self.args, "rollout_seed", 0)),
                "rollout_id": int(rollout_id),
                "step_id": int(step_id),
                "observation_id": int(observation_id),
                "num_updates": self._successful_updates,
                "model_version": self._successful_updates,
                "actual_batch_size": self._pending_batch_size,
                "effective_token_count": self._pending_effective_tokens,
                "cumulative_prompt_count": self._cumulative_prompts,
                "cumulative_effective_token_count": self._cumulative_effective_tokens,
                "update_successful": bool(update_successful),
                "failure_reason": failure_reason if not update_successful else None,
                "valid_update_metrics": bool(update_successful),
                "low_frequency_observation": self._sketch_active and update_successful,
                "low_frequency_approximation": (
                    {
                        "coordinate_distribution": "fixed_log2_histogram",
                        "bins_per_octave": 4,
                        "log2_range": [-149, 128],
                        "approximate_fields_suffix": "_sketch",
                        "support_sample_size_per_parameter_range": int(
                            getattr(self.args, "geometry_support_sample_size", 1024)
                        ),
                        "support_window_successful_updates": int(getattr(self.args, "geometry_support_window", 8)),
                        "matrix_sample_count_per_optimizer_branch_per_rank": int(
                            getattr(self.args, "geometry_matrix_sample_count", 1)
                        ),
                    }
                    if self._sketch_active and update_successful
                    else None
                ),
                "geometry_observation_wall_time_ms": observation_wall_time_ms,
                "matrix_diagnostic_status": (
                    "available"
                    if matrix_records
                    else (
                        "no_full_matrix_optimizer_view_on_any_rank"
                        if self._sketch_active and update_successful
                        else "not_scheduled"
                    )
                ),
                "matrix_diagnostics": matrix_records,
                "matrix_macro": matrix_macro,
                "reported_grad_norm": reported_grad_norm,
                "grad_norm_raw": reported_grad_norm,
                "clip_threshold": clip_threshold,
                "clip_scale": clip_scale,
                "optimizer_clip_scale": optimizer_clip_scale,
                "grad_clipped": grad_clipped,
                "run_clip_fraction": _safe_ratio_number(self._clipped_updates, self._successful_updates),
                "run_update_counters": {
                    "successful": self._successful_updates,
                    "failed_or_skipped": self._failed_updates,
                    "clipped": self._clipped_updates,
                },
                "reported_num_zeros_in_grad": _to_number(num_zeros_in_grad),
                "world_size": _world_size(),
                "projection": "countsketch",
                "projection_dim": self.dim,
                "projection_seed": self.seed,
                "geometry_group_by": self.args.geometry_group_by,
                "geometry_parameter_include": self.args.geometry_parameter_include,
                "geometry_parameter_exclude": self.args.geometry_parameter_exclude,
                "geometry_support_sample_size": int(getattr(self.args, "geometry_support_sample_size", 1024)),
                "geometry_support_window": int(getattr(self.args, "geometry_support_window", 8)),
                "geometry_matrix_sample_count": int(getattr(self.args, "geometry_matrix_sample_count", 1)),
                "geometry_matrix_randomized_rank": int(getattr(self.args, "geometry_matrix_randomized_rank", 16)),
                "source_counts": self._pending_source_counts,
                "groups": group_metrics,
                "vector_file": f"vectors/{vector_file}" if vector_file else None,
            }
            self._append_jsonl(self.output_dir / "metrics.jsonl", record)
            self._log_wandb(record)

        self._active = False
        self._sketch_active = False
        self._before = None
        self._exact_before = []
        self._exact_accumulator = None

    def _log_wandb(self, record: dict[str, Any]) -> None:
        """Mirror selected scalar geometry to the shared W&B run.

        Full layerwise records and projected vectors remain on disk. The
        default W&B selection keeps the global signal plus the matrix/non-matrix
        optimizer partitions, which is compact enough to log every observed
        step. Set ``--geometry-wandb-groups all`` for layerwise dashboards.
        """

        if not bool(getattr(self.args, "use_wandb", False)):
            return
        configured = str(getattr(self.args, "geometry_wandb_groups", "") or "").strip()
        if not configured:
            return
        selected = {item.strip() for item in configured.split(",") if item.strip()}
        all_groups = "all" in selected
        metrics: dict[str, float | int] = {
            "geometry/step": int(record["observation_id"]),
            "geometry/num_updates": int(record["num_updates"]),
            "geometry/model_version": int(record["model_version"]),
            "geometry/update_successful": int(record["update_successful"]),
        }
        for name in ("reported_grad_norm", "reported_num_zeros_in_grad"):
            value = record[name]
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[f"geometry/{name}"] = value
        for source, count in record["source_counts"].items():
            metrics[f"geometry/source_count/{source}"] = int(count)
        for group_name, scalars in record["groups"].items():
            if not all_groups and group_name not in selected:
                continue
            metric_group = group_name.replace("/", "_")
            for name, value in scalars.items():
                if isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
                    metrics[f"geometry/{metric_group}/{name}"] = value

        # Import lazily so geometry-only/offline analysis does not initialize
        # the tracking SDK. This process has already joined the shared run.
        from slime.utils import logging_utils

        logging_utils.log(self.args, metrics, step_key="geometry/step")

    @staticmethod
    def _atomic_torch_save(payload: Any, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())


def _to_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().item()
    value = float(value)
    if not torch.isfinite(torch.tensor(value)):
        return None
    return value


def _safe_ratio_number(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


_OBSERVERS: dict[tuple[int, str, int, tuple[int, ...]], GeometryObserver] = {}


def _observer(args: Any, model: Sequence[torch.nn.Module]) -> GeometryObserver:
    role = getattr(args, "_slime_model_role", "actor")
    # Include model identity so release/recreate workflows never retain stale
    # Parameter objects.  The persisted baseline is still reused and validated.
    key = (id(args), role, _rank(), tuple(id(_unwrap(chunk)) for chunk in model))
    if key not in _OBSERVERS:
        stale_keys = [old_key for old_key in _OBSERVERS if old_key[:3] == key[:3] and old_key != key]
        for stale_key in stale_keys:
            del _OBSERVERS[stale_key]
        _OBSERVERS[key] = GeometryObserver(args, model)
    return _OBSERVERS[key]


def _step_context(data_iterator: Sequence[Any], num_microbatches: int) -> tuple[list[str], int]:
    if not data_iterator:
        return [], 0
    iterator = data_iterator[0]
    rollout_data = iterator.rollout_data
    source_names = iterator.rollout_data.get("source_names") or []
    stop = int(iterator.offset)
    start = max(0, stop - int(num_microbatches))
    indices = [index for microbatch in iterator.micro_batch_indices[start:stop] for index in microbatch]
    selected_sources = [source_names[index] for index in indices if index < len(source_names)]
    local_effective_tokens = 0
    if _is_sample_contributor():
        loss_masks = rollout_data.get("loss_masks") or []
        for index in indices:
            if index >= len(loss_masks):
                continue
            mask = loss_masks[index]
            local_effective_tokens += int(mask.detach().sum().item() if torch.is_tensor(mask) else sum(mask))
    if _distributed():
        gathered: list[int | None] = [None] * _world_size()
        torch.distributed.all_gather_object(gathered, local_effective_tokens)
        local_effective_tokens = sum(value or 0 for value in gathered)
    return selected_sources, local_effective_tokens


def after_backward(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Sequence[torch.nn.Module],
    optimizer: Any,
    opt_param_scheduler: Any,
    *,
    data_iterator: Sequence[Any] | None = None,
    num_microbatches: int = 0,
    actual_batch_size: int = 0,
) -> None:
    del opt_param_scheduler
    source_names, effective_token_count = _step_context(data_iterator or [], num_microbatches)
    _observer(args, model).after_backward(
        rollout_id,
        step_id,
        source_names,
        optimizer=optimizer,
        actual_batch_size=actual_batch_size,
        effective_token_count=effective_token_count,
    )


def after_optimizer_step(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Sequence[torch.nn.Module],
    optimizer: Any,
    opt_param_scheduler: Any,
    *,
    update_successful: bool,
    grad_norm: Any,
    num_zeros_in_grad: Any,
    failure_reason: str | None = None,
) -> None:
    del rollout_id, step_id, opt_param_scheduler
    _observer(args, model).after_step(
        update_successful=update_successful,
        grad_norm=grad_norm,
        num_zeros_in_grad=num_zeros_in_grad,
        optimizer=optimizer,
        failure_reason=failure_reason,
    )
