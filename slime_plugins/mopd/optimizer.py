"""Per-task gradient capture and exact taskwise/conventional AdamW commits."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32

from .optimizer_views import OptimizerParameterView, build_optimizer_parameter_views

MOMENT_RULE_KEY = "mopd_adamw_state"
MOMENT_CLOCK_KEY = "mopd_task_unit_clock"
BASE_BETAS_KEY = "mopd_base_betas"


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _distributed() else 0


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    while hasattr(module, "module"):
        module = module.module
    return module


def _model_entries(model: Sequence[torch.nn.Module]):
    entries = []
    seen: set[int] = set()
    for chunk_id, wrapped in enumerate(model):
        for name, parameter in _unwrap(wrapped).named_parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                entries.append((f"chunk{chunk_id}.{name}", parameter, []))
    return entries


def _leaf_optimizers(optimizer: Any):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        yield optimizer
        return
    for child in children:
        yield from _leaf_optimizers(child)


def _inner_optimizer(leaf: Any) -> Any:
    return getattr(leaf, "optimizer", None) or leaf


def _gradient_for_parameter(parameter: torch.Tensor, owner: Any) -> torch.Tensor | None:
    if bool(getattr(owner, "use_decoupled_grad", False)):
        return getattr(parameter, "decoupled_grad", None)
    gradient = getattr(parameter, "grad", None)
    if gradient is None:
        gradient = getattr(parameter, "decoupled_grad", None)
    return gradient


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "to_local"):
        value = value.to_local()
    if hasattr(value, "_local_tensor"):
        value = value._local_tensor
    return value


def _view_key(view: OptimizerParameterView) -> tuple[int, int, int | None]:
    return id(view.optimizer_parameter), int(view.start), view.stop


@torch.no_grad()
def clip_prepared_gradients(optimizer: Any, grad_norm: float | torch.Tensor) -> None:
    """Apply the shared task-unit clip before inclusion correction."""

    for leaf in _leaf_optimizers(optimizer):
        if getattr(leaf, "is_stub_optimizer", False):
            continue
        parameters = leaf.get_parameters()
        clip = float(getattr(leaf.config, "clip_grad", 0.0) or 0.0)
        if not parameters or clip <= 0:
            continue
        gradients = [
            gradient
            for parameter in parameters
            if (gradient := _gradient_for_parameter(parameter, leaf.config)) is not None
        ]
        if gradients and not all(gradient.is_cuda for gradient in gradients):
            norm = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            coefficient = min(1.0, clip / (norm + 1e-6))
            for gradient in gradients:
                gradient.mul_(coefficient)
        else:
            clip_grad_by_total_norm_fp32(
                parameters,
                max_norm=clip,
                total_norm=grad_norm,
                use_decoupled_grad=bool(
                    getattr(leaf.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
                ),
            )


def _state_step(view: OptimizerParameterView) -> int:
    value = view.optimizer_group.get(MOMENT_CLOCK_KEY)
    if value is None:
        value = view.optimizer_state.get("step", 0)
    if torch.is_tensor(value):
        value = value.detach().item()
    return int(value or 0)


def _adam_hyperparameters(view: OptimizerParameterView) -> tuple[float, float, float]:
    beta1, beta2 = map(float, view.optimizer_group.get(BASE_BETAS_KEY, view.optimizer_group["betas"]))
    eps = float(view.optimizer_group.get("eps", 1e-8))
    return beta1, beta2, eps


def _preconditioner_denominator(
    view: OptimizerParameterView, start: int, stop: int, *, device: torch.device
) -> torch.Tensor:
    _, beta2, eps = _adam_hyperparameters(view)
    exp_avg_sq = view.optimizer_state.get("exp_avg_sq")
    if exp_avg_sq is None:
        return torch.full((stop - start,), eps, dtype=torch.float32, device=device)
    flat = _local_tensor(exp_avg_sq).detach().reshape(-1)
    value = flat[start:stop].to(device=device, dtype=torch.float32)
    clock = _state_step(view)
    if clock > 0:
        value = value / (1.0 - beta2**clock)
    return value.clamp_min_(0).sqrt_().add_(eps)


def _preconditioner_denominator_at(
    view: OptimizerParameterView, indices: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    _, beta2, eps = _adam_hyperparameters(view)
    exp_avg_sq = view.optimizer_state.get("exp_avg_sq")
    if exp_avg_sq is None:
        return torch.full((indices.numel(),), eps, dtype=torch.float32, device=device)
    flat = _local_tensor(exp_avg_sq).detach().reshape(-1)
    value = flat.index_select(0, indices.to(flat.device)).to(device=device, dtype=torch.float32)
    clock = _state_step(view)
    if clock > 0:
        value = value / (1.0 - beta2**clock)
    return value.clamp_min_(0).sqrt_().add_(eps)


@torch.no_grad()
def score_prepared_gradients(
    views: Sequence[OptimizerParameterView], *, chunk_size: int = 1_048_576
) -> tuple[float, float]:
    first = next((g for view in views if (g := view.optimizer_gradient()) is not None), None)
    device = first.device if first is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sums = torch.zeros(2, dtype=torch.float64, device=device)
    for view in views:
        gradient = view.optimizer_gradient()
        if gradient is None:
            continue
        flat = gradient.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_size):
            stop = min(start + chunk_size, flat.numel())
            chunk = flat[start:stop].float()
            denominator = _preconditioner_denominator(view, start, stop, device=chunk.device)
            sums[0] += torch.sum(chunk.square(), dtype=torch.float64)
            sums[1] += torch.sum((chunk / denominator).square(), dtype=torch.float64)
    if _distributed():
        torch.distributed.all_reduce(sums)
    return math.sqrt(float(sums[0])), math.sqrt(float(sums[1]))


def _initialize_adam_state(inner: Any, parameter: torch.Tensor) -> dict[str, Any]:
    state = inner.state[parameter]
    if "exp_avg" in state and "exp_avg_sq" in state:
        return state
    initialize = getattr(inner, "initialize_state", None)
    if initialize is not None:
        initialize(parameter, bool(getattr(inner, "store_param_remainders", False)))
        return inner.state[parameter]
    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
    if type(inner).__module__.startswith("torch.optim"):
        group = next(group for group in inner.param_groups if any(p is parameter for p in group["params"]))
        device = parameter.device if group.get("capturable") or group.get("fused") else torch.device("cpu")
        state["step"] = torch.zeros((), dtype=torch.float32, device=device)
        if group.get("amsgrad"):
            state["max_exp_avg_sq"] = torch.zeros_like(parameter)
    return state


def _set_step(container: dict[str, Any], value: int) -> None:
    if "step" not in container:
        return
    if torch.is_tensor(container["step"]):
        container["step"].fill_(value)
    else:
        container["step"] = value


@torch.no_grad()
def _prepare_compound_adamw(
    optimizer: Any,
    second_observation: dict[int, torch.Tensor],
    *,
    task_width: int,
    task_unit_clock_before: int,
    adamw_state: str,
) -> list[tuple[dict[str, Any], tuple[float, float], float]]:
    """Temporarily install beta^K/compound-WD and the requested square observation."""

    if task_unit_clock_before % task_width:
        raise ValueError("task-unit clock must be divisible by K for exact Adam bias correction")
    restored: list[tuple[dict[str, Any], tuple[float, float], float]] = []
    for leaf in _leaf_optimizers(optimizer):
        if getattr(leaf, "is_stub_optimizer", False):
            continue
        inner = _inner_optimizer(leaf)
        for group in inner.param_groups:
            base_betas = tuple(map(float, group.get(BASE_BETAS_KEY, group["betas"])))
            beta1, beta2 = base_betas
            effective = (beta1**task_width, beta2**task_width)
            lr, weight_decay = float(group["lr"]), float(group.get("weight_decay", 0.0))
            if lr * weight_decay >= 1:
                raise ValueError("AdamW lr*weight_decay must be below one for compound decay")
            compound_wd = 0.0 if lr == 0 else (1.0 - (1.0 - lr * weight_decay) ** task_width) / lr
            restored.append((group, tuple(group["betas"]), weight_decay))
            group[BASE_BETAS_KEY] = base_betas
            group["betas"] = effective
            group["weight_decay"] = compound_wd
            group[MOMENT_RULE_KEY] = adamw_state
            group[MOMENT_CLOCK_KEY] = int(task_unit_clock_before)
            _set_step(group, task_unit_clock_before // task_width)
            for parameter in group["params"]:
                gradient = getattr(parameter, "decoupled_grad", None)
                if gradient is None:
                    gradient = getattr(parameter, "grad", None)
                if gradient is None:
                    continue
                state = _initialize_adam_state(inner, parameter)
                _set_step(state, task_unit_clock_before // task_width)
                if adamw_state == "taskwise":
                    desired = second_observation[id(parameter)].to(
                        device=gradient.device, dtype=torch.float32
                    ).reshape_as(_local_tensor(gradient))
                    second = _local_tensor(state["exp_avg_sq"])
                    combined = _local_tensor(gradient).float()
                    correction = (1.0 - effective[1]) / effective[1]
                    second.add_(desired - combined.square(), alpha=correction)
    return restored


def _restore_compound_groups(
    restored: Sequence[tuple[dict[str, Any], tuple[float, float], float]], clock_after: int
) -> None:
    for group, betas, weight_decay in restored:
        group["betas"] = betas
        group["weight_decay"] = weight_decay
        group[MOMENT_CLOCK_KEY] = int(clock_after)


def _current_step_indices(data_iterator: Sequence[Any], num_microbatches: int) -> list[int]:
    iterator = data_iterator[0]
    used = iterator.micro_batch_indices[iterator.offset - num_microbatches : iterator.offset]
    return [index for microbatch in used for index in microbatch]


def _step_values(data_iterator: Sequence[Any], key: str, num_microbatches: int) -> list[Any]:
    indices = _current_step_indices(data_iterator, num_microbatches)
    values = data_iterator[0].rollout_data.get(key)
    if values is None:
        raise KeyError(f"MOPD rollout data is missing {key}")
    return [values[index] for index in indices]


def _uniform_step_value(data_iterator: Sequence[Any], key: str, num_microbatches: int) -> Any:
    values = _step_values(data_iterator, key, num_microbatches)
    normalized = [v.item() if torch.is_tensor(v) and v.numel() == 1 else v for v in values]
    first = normalized[0]
    if any(value != first for value in normalized[1:]):
        raise ValueError(f"one MOPD backward slice contains multiple {key} values")
    return first


def _teacher_loss(data_iterator: Sequence[Any], num_microbatches: int) -> float:
    values = _step_values(data_iterator, "sampled_reverse_kl_logratio", num_microbatches)
    masks = _step_values(data_iterator, "loss_masks", num_microbatches)
    penalties = _step_values(data_iterator, "mopd_failure_penalties", num_microbatches)
    device = torch.as_tensor(values[0]).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for value, mask, penalty in zip(values, masks, penalties, strict=True):
        penalty = float(penalty)
        tensor = torch.as_tensor(value, device=device).reshape(-1).float()
        mask_tensor = torch.as_tensor(mask, device=device).reshape(-1).float()
        if penalty > 0:
            loss = penalty
        elif tensor.numel() and float(mask_tensor.sum()) > 0:
            loss = float((tensor * mask_tensor).sum() / mask_tensor.sum())
        else:
            raise ValueError("an empty MOPD response is missing its numerical failure penalty")
        total[0] += loss
        total[1] += 1
    if _distributed():
        torch.distributed.all_reduce(total)
    return float(total[0] / total[1])


def _scale_optimizer_gradients(views: Sequence[OptimizerParameterView], scale: float) -> None:
    seen: set[int] = set()
    for view in views:
        parameter = view.optimizer_parameter
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        gradient = view.optimizer_gradient()
        if gradient is not None:
            gradient.mul_(scale)


def _sample_bank_vectors(
    views: Sequence[OptimizerParameterView], coordinate_count: int
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    active = [view for view in views if view.optimizer_gradient() is not None]
    if not active:
        raise RuntimeError("frozen-bank capture found no optimizer gradients")
    if coordinate_count <= 0:
        raise ValueError("frozen-bank coordinate count must be positive")
    sizes = [int(view.optimizer_gradient().numel()) for view in active]
    total = sum(sizes)
    target = min(int(coordinate_count), total)
    if target == 1:
        global_indices = torch.zeros(1, dtype=torch.int64)
    else:
        global_indices = torch.div(
            torch.arange(target, dtype=torch.int64) * (total - 1),
            target - 1,
            rounding_mode="floor",
        )
    raw_parts, scaled_parts, layout = [], [], []
    offset = 0
    for view, size in zip(active, sizes, strict=True):
        gradient = view.optimizer_gradient().detach().reshape(-1).float()
        mask = (global_indices >= offset) & (global_indices < offset + size)
        indices = global_indices[mask] - offset
        if not indices.numel():
            offset += size
            continue
        device_indices = indices.to(gradient.device)
        raw = gradient.index_select(0, device_indices)
        denominator = _preconditioner_denominator_at(view, device_indices, device=gradient.device)
        raw_parts.append(raw.cpu())
        scaled_parts.append((raw / denominator).cpu())
        layout.append(
            {
                "name": view.name,
                "offset": offset,
                "numel": size,
                "take": int(indices.numel()),
                "indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
            }
        )
        offset += size
    if sum(part.numel() for part in raw_parts) != target:
        raise RuntimeError("frozen-bank global coordinate sampler produced the wrong vector size")
    return torch.cat(raw_parts), torch.cat(scaled_parts), layout


def _write_bank_observation(args: Any, metadata: dict[str, Any], views: Sequence[OptimizerParameterView]) -> None:
    output = Path(args.mopd_bank_dir).resolve()
    raw, scaled, layout = _sample_bank_vectors(views, int(args.mopd_bank_coordinates))
    path = output / f"unit_{int(metadata['operation_index']):03d}_{metadata['task']}_rank_{_rank():04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save({"schema_version": 1, "metadata": metadata, "layout": layout, "raw": raw, "scaled": scaled}, temporary)
    os.replace(temporary, path)


class _OperationAccumulator:
    def __init__(self, operation: str, adamw_state: str):
        self.operation = operation
        self.adamw_state = adamw_state
        self.entries: list[dict[str, Any]] = []

    def add(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)


def begin_mopd_operation(optimizer: Any, operation: str, adamw_state: str) -> None:
    if hasattr(optimizer, "_mopd_operation_accumulator"):
        raise RuntimeError("the previous MOPD operation was not finalized")
    optimizer._mopd_operation_accumulator = _OperationAccumulator(operation, adamw_state)


@torch.no_grad()
def mopd_capture_step(
    args: Any,
    data_iterator: Sequence[Any],
    model: Sequence[torch.nn.Module],
    optimizer: Any,
    *,
    num_microbatches: int,
) -> tuple[bool, float, Any, dict[str, Any]]:
    operation = str(_uniform_step_value(data_iterator, "mopd_operations", num_microbatches))
    task = str(_uniform_step_value(data_iterator, "mopd_tasks", num_microbatches))
    relative_scale = float(_uniform_step_value(data_iterator, "mopd_relative_loss_scales", num_microbatches))
    importance = float(_uniform_step_value(data_iterator, "mopd_importance_corrections", num_microbatches))
    inclusion = float(_uniform_step_value(data_iterator, "mopd_inclusion_probabilities", num_microbatches))
    target = float(_uniform_step_value(data_iterator, "mopd_target_weights", num_microbatches))
    clock_before = int(_uniform_step_value(data_iterator, "mopd_processed_task_units_before", num_microbatches))
    operation_index = int(_uniform_step_value(data_iterator, "mopd_operation_indices", num_microbatches))
    raw_teacher_loss = _teacher_loss(data_iterator, num_microbatches)

    accumulator: _OperationAccumulator = optimizer._mopd_operation_accumulator
    if operation != accumulator.operation:
        raise ValueError(
            f"captured operation {operation!r} differs from initialized operation {accumulator.operation!r}"
        )
    found_inf = optimizer.prepare_grads()
    if bool(found_inf):
        raise FloatingPointError(f"non-finite gradient in MOPD {operation} slice for task {task}")
    views = build_optimizer_parameter_views(_model_entries(model), optimizer, requested_optimizer="adam")
    _scale_optimizer_gradients(views, relative_scale)
    grad_norm = optimizer.get_grad_norm()
    preclip = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
    raw_score, adam_score = score_prepared_gradients(views, chunk_size=int(args.mopd_score_chunk_size))
    bank_payload = None
    if operation == "bank":
        bank_payload = {"operation_index": operation_index, "task": task, "raw_score": raw_score, "adam_score": adam_score}
        _write_bank_observation(args, bank_payload, views)
    probe_vectors = None
    if operation == "probe":
        probe_vectors = {
            _view_key(view): (
                view.optimizer_gradient().detach().float().clone(),
                view.optimizer_gradient().detach().float().clone()
                / _preconditioner_denominator(view, 0, view.optimizer_gradient().numel(), device=view.optimizer_gradient().device),
            )
            for view in views
            if view.optimizer_gradient() is not None
        }
    clip_prepared_gradients(optimizer, grad_norm)
    gradients = {
        _view_key(view): view.optimizer_gradient().detach().float().clone()
        for view in views
        if view.optimizer_gradient() is not None
    }
    accumulator.add(
        {
            "task": task,
            "operation": operation,
            "importance": importance,
            "inclusion": inclusion,
            "target_weight": target,
            "clock_before": clock_before,
            "views": list(views),
            "gradients": gradients,
            "probe_vectors": probe_vectors,
            "raw_score": raw_score,
            "adam_score": adam_score,
            "raw_teacher_loss": raw_teacher_loss,
            "relative_teacher_loss": raw_teacher_loss * relative_scale,
            "preclip_grad_norm": preclip,
            "clip_flag": bool(float(args.clip_grad) > 0 and preclip > float(args.clip_grad)),
        }
    )
    zeros = optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else None
    return True, preclip, zeros, {"mopd": True, "task": task, "captured": True}


def _probe_score(entries: Sequence[dict[str, Any]], vector_index: int) -> float:
    if len(entries) != 2:
        raise ValueError("an online score probe must contain exactly two prompt-group gradients")
    device = next(iter(entries[0]["probe_vectors"].values()))[vector_index].device
    values = torch.zeros(2, dtype=torch.float64, device=device)
    for key in entries[0]["probe_vectors"]:
        first = entries[0]["probe_vectors"][key][vector_index].reshape(-1)
        second = entries[1]["probe_vectors"][key][vector_index].reshape(-1)
        values[0] += torch.dot(first.double(), second.double())
        values[1] += 0.5 * torch.sum((first.double() - second.double()).square())
    if _distributed():
        torch.distributed.all_reduce(values)
    full_unit_square = max(float(values[0] + values[1] / 16.0), 0.0)
    return math.sqrt(full_unit_square)


@torch.no_grad()
def finish_mopd_operation(args: Any, optimizer: Any) -> tuple[bool, float, dict[str, Any]]:
    accumulator: _OperationAccumulator = optimizer._mopd_operation_accumulator
    entries = accumulator.entries
    if not entries:
        raise RuntimeError("MOPD operation captured no gradients")
    operation = accumulator.operation
    update_successful = True
    # Bank captures and probes do not form or commit an aggregate optimizer
    # gradient.  Their operation-level update norm is therefore exactly zero;
    # the measured per-task norms remain available in ``task_units``.
    aggregate_norm = 0.0

    if operation == "train":
        keys = tuple(entries[0]["gradients"])
        if any(tuple(entry["gradients"]) != keys for entry in entries[1:]):
            raise RuntimeError("task gradients use different optimizer coordinate layouts")
        combined: dict[tuple[int, int, int | None], torch.Tensor] = {}
        second_by_parameter: dict[int, torch.Tensor] = {}
        for key in keys:
            combined[key] = sum(entry["importance"] * entry["gradients"][key] for entry in entries)
            if key[0] in second_by_parameter:
                raise RuntimeError("one optimizer tensor is represented by multiple MOPD gradient views")
            second_by_parameter[key[0]] = sum(
                entry["importance"] * entry["gradients"][key].square() for entry in entries
            )
        views = entries[0]["views"]
        for view in views:
            key = _view_key(view)
            if key in combined:
                view.set_optimizer_gradient(combined[key])
        norm_square = next(iter(combined.values())).new_zeros((), dtype=torch.float64)
        for gradient in combined.values():
            norm_square += torch.sum(gradient.double().square())
        if _distributed():
            torch.distributed.all_reduce(norm_square)
        aggregate_norm = math.sqrt(float(norm_square))
        width = len(entries)
        clock_before = int(entries[0]["clock_before"])
        restored = _prepare_compound_adamw(
            optimizer,
            second_by_parameter,
            task_width=width,
            task_unit_clock_before=clock_before,
            adamw_state=accumulator.adamw_state,
        )
        update_successful = bool(optimizer.step_with_ready_grads())
        _restore_compound_groups(restored, clock_before + width)

    if operation == "probe":
        task_units = [
            {
                "task": entries[0]["task"],
                "raw_score": _probe_score(entries, 0),
                "adam_score": _probe_score(entries, 1),
                "raw_teacher_loss": sum(entry["raw_teacher_loss"] for entry in entries) / 2,
                "relative_teacher_loss": sum(entry["relative_teacher_loss"] for entry in entries) / 2,
                "importance_correction": entries[0]["importance"],
                "inclusion_probability": entries[0]["inclusion"],
                "target_weight": entries[0]["target_weight"],
                "clip_flag": any(entry["clip_flag"] for entry in entries),
            }
        ]
    else:
        task_units = [
            {
                "task": entry["task"],
                "raw_score": entry["raw_score"],
                "adam_score": entry["adam_score"],
                "raw_teacher_loss": entry["raw_teacher_loss"],
                "relative_teacher_loss": entry["relative_teacher_loss"],
                "importance_correction": entry["importance"],
                "inclusion_probability": entry["inclusion"],
                "target_weight": entry["target_weight"],
                "clip_flag": entry["clip_flag"],
            }
            for entry in entries
        ]
    feedback = {
        "mopd": True,
        "operation": operation,
        "adamw_state": accumulator.adamw_state,
        "task_units": task_units,
        "optimizer_step_executed": operation == "train",
        "update_successful": update_successful,
        "aggregate_grad_norm": aggregate_norm,
        "attempted_responses": 8 if operation == "probe" else 64 * len(task_units),
        "overflow_flag": False,
    }
    if operation == "train" and not update_successful:
        feedback["failure_reason"] = "optimizer_step_rejected"
    del optimizer._mopd_operation_accumulator
    return update_successful, aggregate_norm, feedback


__all__ = [
    "BASE_BETAS_KEY", "MOMENT_CLOCK_KEY", "MOMENT_RULE_KEY", "begin_mopd_operation",
    "clip_prepared_gradients", "finish_mopd_operation", "mopd_capture_step",
    "score_prepared_gradients",
]
