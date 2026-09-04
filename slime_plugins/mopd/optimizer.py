"""Streaming micro-batch gradient statistics and one conventional AdamW commit."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32

from .optimizer_views import OptimizerParameterView, build_optimizer_parameter_views
from .sampler import (
    OPEN_MOPD_GAP_ALPHA,
    OPEN_MOPD_GAP_FACTOR_MAX,
    OPEN_MOPD_GAP_FACTOR_MIN,
    PROMPTS_PER_MICROBATCH,
    TASKS,
)


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


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
    """Apply the configured clip once, to the final all-task gradient."""

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
                use_decoupled_grad=bool(getattr(leaf.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)),
            )


def _state_step(view: OptimizerParameterView) -> int:
    value = view.optimizer_state.get("step", view.optimizer_group.get("step", 0))
    if torch.is_tensor(value):
        value = value.detach().item()
    return int(value or 0)


def _preconditioner_denominator(
    view: OptimizerParameterView, start: int, stop: int, *, device: torch.device
) -> torch.Tensor:
    beta2 = float(view.optimizer_group["betas"][1])
    eps = float(view.optimizer_group.get("eps", 1e-8))
    exp_avg_sq = view.optimizer_state.get("exp_avg_sq")
    if exp_avg_sq is None:
        return torch.full((stop - start,), eps, dtype=torch.float32, device=device)
    value = _local_tensor(exp_avg_sq).detach().reshape(-1)[start:stop].to(device=device, dtype=torch.float32)
    clock = _state_step(view)
    if clock and bool(view.optimizer_group.get("bias_correction", True)):
        value = value / (1.0 - beta2**clock)
    return value.clamp_min_(0).sqrt_().add_(eps)


@torch.no_grad()
def _gradient_square_sums(
    views: Sequence[OptimizerParameterView],
    vectors: dict[tuple[int, int, int | None], torch.Tensor] | None = None,
    *,
    divisor: float = 1.0,
    chunk_size: int = 1_048_576,
) -> tuple[float, float]:
    first = next(
        (
            vectors[_view_key(view)] if vectors is not None else view.optimizer_gradient()
            for view in views
            if (vectors is not None and _view_key(view) in vectors) or view.optimizer_gradient() is not None
        ),
        None,
    )
    device = first.device if first is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sums = torch.zeros(2, dtype=torch.float64, device=device)
    for view in views:
        gradient = vectors.get(_view_key(view)) if vectors is not None else view.optimizer_gradient()
        if gradient is None:
            continue
        flat = gradient.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_size):
            stop = min(start + chunk_size, flat.numel())
            chunk = flat[start:stop].float() / divisor
            denominator = _preconditioner_denominator(view, start, stop, device=chunk.device)
            sums[0] += torch.sum(chunk.square(), dtype=torch.float64)
            sums[1] += torch.sum((chunk / denominator).square(), dtype=torch.float64)
    if _distributed():
        torch.distributed.all_reduce(sums)
    return float(sums[0]), float(sums[1])


def score_prepared_gradients(
    views: Sequence[OptimizerParameterView], *, chunk_size: int = 1_048_576
) -> tuple[float, float]:
    raw, scaled = _gradient_square_sums(views, chunk_size=chunk_size)
    return math.sqrt(raw), math.sqrt(scaled)


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
    normalized = [value.item() if torch.is_tensor(value) and value.numel() == 1 else value for value in values]
    if any(value != normalized[0] for value in normalized[1:]):
        raise ValueError(f"one MOPD gradient micro-batch contains multiple {key} values")
    return normalized[0]


def _teacher_loss_and_tokens(data_iterator: Sequence[Any], num_microbatches: int) -> tuple[float, int, float]:
    values = _step_values(data_iterator, "sampled_reverse_kl_logratio", num_microbatches)
    masks = _step_values(data_iterator, "loss_masks", num_microbatches)
    penalties = _step_values(data_iterator, "mopd_failure_penalties", num_microbatches)
    device = torch.as_tensor(values[0]).device
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for value, mask, penalty in zip(values, masks, penalties, strict=True):
        tensor = torch.as_tensor(value, device=device).reshape(-1).float()
        valid = torch.as_tensor(mask, device=device).reshape(-1).float()
        token_count = int(valid.sum().item())
        loss = float(penalty) if float(penalty) > 0 else float((tensor * valid).sum() / max(token_count, 1))
        totals[0] += loss
        totals[1] += 1
        totals[2] += token_count
        totals[3] += (tensor.abs() * valid).sum()
    if _distributed():
        torch.distributed.all_reduce(totals)
    return float(totals[0] / totals[1]), int(totals[2]), float(totals[3] / totals[2].clamp_min(1))


def open_mopd_domain_weights(
    token_counts: Sequence[int],
    reward_magnitudes: Sequence[float],
    target_shares: Sequence[float],
) -> dict[str, list[float] | float]:
    """Compute Open-MOPD token-share and forward gap-following weights."""

    if not token_counts or not (len(token_counts) == len(reward_magnitudes) == len(target_shares)):
        raise ValueError("Open-MOPD requires aligned, non-empty domain statistics")
    tokens = [int(value) for value in token_counts]
    if any(value <= 0 for value in tokens):
        raise ValueError("Open-MOPD requires at least one valid response token from every domain")
    magnitudes = [float(value) for value in reward_magnitudes]
    if any(not math.isfinite(value) or value < 0 for value in magnitudes):
        raise ValueError("Open-MOPD reward magnitudes must be finite and non-negative")
    targets = [float(value) for value in target_shares]
    if any(not math.isfinite(value) or value <= 0 for value in targets):
        raise ValueError("Open-MOPD target shares must be finite and positive")
    target_total = sum(targets)
    targets = [value / target_total for value in targets]

    token_total = sum(tokens)
    token_shares = [value / token_total for value in tokens]
    share_weights = [target / observed for target, observed in zip(targets, token_shares, strict=True)]
    reward_reference = sum(magnitudes) / len(magnitudes)
    if reward_reference <= 1e-12:
        gap_factors = [1.0] * len(magnitudes)
    else:
        gap_factors = [
            min(
                max((value / reward_reference) ** OPEN_MOPD_GAP_ALPHA, OPEN_MOPD_GAP_FACTOR_MIN),
                OPEN_MOPD_GAP_FACTOR_MAX,
            )
            for value in magnitudes
        ]
    unnormalized = [share * factor for share, factor in zip(share_weights, gap_factors, strict=True)]
    normalizer = sum(weight * observed for weight, observed in zip(unnormalized, token_shares, strict=True))
    weights = [value / normalizer for value in unnormalized]
    effective_shares = [weight * observed for weight, observed in zip(weights, token_shares, strict=True)]
    return {
        "token_shares": token_shares,
        "share_weights": share_weights,
        "reward_reference": reward_reference,
        "gap_factors": gap_factors,
        "weights": weights,
        "effective_shares": effective_shares,
    }


@dataclass
class _TaskAccumulator:
    task: str
    expected_microbatches: int
    target_weight: float
    gradients: dict[tuple[int, int, int | None], torch.Tensor]
    token_gradients: dict[tuple[int, int, int | None], torch.Tensor] | None = None
    microbatches: int = 0
    raw_square_sum: float = 0.0
    scaled_square_sum: float = 0.0
    teacher_loss_sum: float = 0.0
    reward_abs_token_sum: float = 0.0
    valid_tokens: int = 0
    raw_microbatch_sq: list[float] = field(default_factory=list)
    scaled_microbatch_sq: list[float] = field(default_factory=list)
    teacher_loss_microbatch: list[float] = field(default_factory=list)
    half_gradients: list[dict[tuple[int, int, int | None], torch.Tensor]] | None = None
    subset_mean_sq: dict[int, list[tuple[float, float]]] = field(default_factory=lambda: {2: [], 4: []})


@dataclass
class _OperationAccumulator:
    operation: str
    aggregation: str
    aggregate: dict[tuple[int, int, int | None], torch.Tensor] = field(default_factory=dict)
    current: _TaskAccumulator | None = None
    task_units: list[dict[str, Any]] = field(default_factory=list)
    views: list[OptimizerParameterView] = field(default_factory=list)
    total_valid_tokens: int = 0
    open_task_gradients: list[dict[tuple[int, int, int | None], torch.Tensor]] = field(default_factory=list)


def begin_mopd_operation(optimizer: Any, operation: str, aggregation: str) -> None:
    if hasattr(optimizer, "_mopd_operation_accumulator"):
        raise RuntimeError("the previous MOPD operation was not finalized")
    if aggregation not in {"fixed_objective", "token_mean", "open_mopd", "variance_probe"}:
        raise ValueError(f"unsupported MOPD aggregation {aggregation!r}")
    if (operation == "heldout_variance") != (aggregation == "variance_probe"):
        raise ValueError("held-out variance operations must use variance_probe aggregation")
    optimizer._mopd_operation_accumulator = _OperationAccumulator(operation, aggregation)


@torch.no_grad()
def _finish_task(accumulator: _OperationAccumulator, chunk_size: int) -> None:
    task = accumulator.current
    if task is None:
        return
    if task.microbatches != task.expected_microbatches:
        raise ValueError(
            f"task {task.task} produced {task.microbatches} micro-batches, expected {task.expected_microbatches}"
        )
    mean_raw, mean_scaled = _gradient_square_sums(
        accumulator.views,
        task.gradients,
        divisor=task.microbatches,
        chunk_size=chunk_size,
    )
    raw_noise = max((task.raw_square_sum - task.microbatches * mean_raw) / (task.microbatches - 1), 0.0)
    scaled_noise = max(
        (task.scaled_square_sum - task.microbatches * mean_scaled) / (task.microbatches - 1),
        0.0,
    )
    if accumulator.aggregation == "fixed_objective":
        scale = task.target_weight / task.microbatches
        for key, gradient in task.gradients.items():
            accumulator.aggregate[key].add_(gradient, alpha=scale)
    elif accumulator.aggregation == "open_mopd":
        if task.token_gradients is None:
            raise RuntimeError("Open-MOPD task block did not collect token-sum gradients")
        accumulator.open_task_gradients.append(task.token_gradients)
    unit = {
        "task": task.task,
        "microbatches": task.microbatches,
        "raw_noise": raw_noise,
        "scaled_noise": scaled_noise,
        "sum_raw_gradient_sq": task.raw_square_sum,
        "sum_scaled_gradient_sq": task.scaled_square_sum,
        "raw_task_mean_sq": mean_raw,
        "scaled_task_mean_sq": mean_scaled,
        "teacher_loss": task.teacher_loss_sum / task.microbatches,
        "reward_abs_mean": task.reward_abs_token_sum / max(task.valid_tokens, 1),
        "valid_response_tokens": task.valid_tokens,
        "target_weight": task.target_weight,
    }
    if accumulator.aggregation == "variance_probe":
        if task.microbatches != 32 or task.half_gradients is None:
            raise ValueError("held-out variance requires two halves of 16 micro-batches")
        half_squares = [
            _gradient_square_sums(
                accumulator.views,
                gradients,
                divisor=task.microbatches // 2,
                chunk_size=chunk_size,
            )
            for gradients in task.half_gradients
        ]
        unit.update(
            {
                "raw_microbatch_sq": task.raw_microbatch_sq,
                "scaled_microbatch_sq": task.scaled_microbatch_sq,
                "teacher_loss_microbatch": task.teacher_loss_microbatch,
                "raw_half_mean_sq": [value[0] for value in half_squares],
                "scaled_half_mean_sq": [value[1] for value in half_squares],
                "raw_subset_mean_sq": {
                    str(size): [value[0] for value in task.subset_mean_sq[size]] for size in (2, 4)
                },
                "scaled_subset_mean_sq": {
                    str(size): [value[1] for value in task.subset_mean_sq[size]] for size in (2, 4)
                },
            }
        )
    accumulator.task_units.append(unit)
    accumulator.current = None


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
    aggregation = str(_uniform_step_value(data_iterator, "mopd_aggregations", num_microbatches))
    expected = int(_uniform_step_value(data_iterator, "mopd_task_microbatch_counts", num_microbatches))
    target_weight = float(_uniform_step_value(data_iterator, "mopd_target_weights", num_microbatches))
    teacher_loss, valid_tokens, reward_abs_mean = _teacher_loss_and_tokens(data_iterator, num_microbatches)

    accumulator: _OperationAccumulator = optimizer._mopd_operation_accumulator
    if operation != accumulator.operation or aggregation != accumulator.aggregation:
        raise ValueError("MOPD gradient metadata differs from the initialized operation")
    found_inf = optimizer.prepare_grads()
    if bool(found_inf):
        raise FloatingPointError(f"non-finite gradient in MOPD micro-batch for task {task}")
    views = build_optimizer_parameter_views(_model_entries(model), optimizer, requested_optimizer="adam")
    raw_square, scaled_square = _gradient_square_sums(
        views,
        chunk_size=int(args.mopd_score_chunk_size),
    )
    gradient_map = {
        _view_key(view): view.optimizer_gradient().detach().float().clone()
        for view in views
        if view.optimizer_gradient() is not None
    }
    if not accumulator.aggregate and aggregation != "variance_probe":
        accumulator.aggregate = {key: torch.zeros_like(gradient) for key, gradient in gradient_map.items()}
    if accumulator.current is not None and accumulator.current.task != task:
        _finish_task(accumulator, int(args.mopd_score_chunk_size))
    if accumulator.current is None:
        gradients = {key: torch.zeros_like(gradient) for key, gradient in gradient_map.items()}
        token_gradients = None
        if aggregation == "open_mopd":
            token_gradients = {key: torch.zeros_like(gradient) for key, gradient in gradient_map.items()}
        halves = None
        if aggregation == "variance_probe":
            halves = [{key: torch.zeros_like(gradient) for key, gradient in gradient_map.items()} for _ in range(2)]
        accumulator.current = _TaskAccumulator(
            task,
            expected,
            target_weight,
            gradients,
            token_gradients=token_gradients,
            half_gradients=halves,
        )
    accumulator.views = list(views)
    if tuple(accumulator.current.gradients) != tuple(gradient_map):
        raise RuntimeError("MOPD gradient coordinate layout changed within a task block")
    for key, gradient in gradient_map.items():
        accumulator.current.gradients[key].add_(gradient)
    if aggregation == "variance_probe":
        half_size = expected // 2
        half_index = accumulator.current.microbatches // half_size
        local_count = accumulator.current.microbatches % half_size + 1
        half_gradients = accumulator.current.half_gradients[half_index]
        for key, gradient in gradient_map.items():
            half_gradients[key].add_(gradient)
        if local_count in (2, 4):
            accumulator.current.subset_mean_sq[local_count].append(
                _gradient_square_sums(
                    views,
                    half_gradients,
                    divisor=local_count,
                    chunk_size=int(args.mopd_score_chunk_size),
                )
            )
    accumulator.current.microbatches += 1
    accumulator.current.raw_square_sum += raw_square
    accumulator.current.scaled_square_sum += scaled_square
    accumulator.current.teacher_loss_sum += teacher_loss
    accumulator.current.reward_abs_token_sum += reward_abs_mean * valid_tokens
    accumulator.current.valid_tokens += valid_tokens
    accumulator.current.raw_microbatch_sq.append(raw_square)
    accumulator.current.scaled_microbatch_sq.append(scaled_square)
    accumulator.current.teacher_loss_microbatch.append(teacher_loss)
    accumulator.total_valid_tokens += valid_tokens
    if aggregation == "token_mean":
        for key, gradient in gradient_map.items():
            accumulator.aggregate[key].add_(gradient, alpha=valid_tokens)
    elif aggregation == "open_mopd":
        if accumulator.current.token_gradients is None:
            raise RuntimeError("Open-MOPD token-gradient accumulator is missing")
        for key, gradient in gradient_map.items():
            accumulator.current.token_gradients[key].add_(gradient, alpha=valid_tokens)

    grad_norm = math.sqrt(raw_square)
    zeros = optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else None
    return True, grad_norm, zeros, {"mopd": True, "task": task, "captured": True}


@torch.no_grad()
def finish_mopd_operation(args: Any, optimizer: Any) -> tuple[bool, float, dict[str, Any]]:
    accumulator: _OperationAccumulator = optimizer._mopd_operation_accumulator
    _finish_task(accumulator, int(args.mopd_score_chunk_size))
    if [unit["task"] for unit in accumulator.task_units] != list(TASKS):
        raise ValueError("a MOPD optimizer step must contain all four task blocks in protocol order")
    if accumulator.aggregation == "token_mean":
        for gradient in accumulator.aggregate.values():
            gradient.div_(accumulator.total_valid_tokens)
    elif accumulator.aggregation == "open_mopd":
        if len(accumulator.open_task_gradients) != len(TASKS):
            raise RuntimeError("Open-MOPD did not retain one token-gradient map per domain")
        weighting = open_mopd_domain_weights(
            [int(unit["valid_response_tokens"]) for unit in accumulator.task_units],
            [float(unit["reward_abs_mean"]) for unit in accumulator.task_units],
            [float(unit["target_weight"]) for unit in accumulator.task_units],
        )
        for gradient in accumulator.aggregate.values():
            gradient.zero_()
        for task_gradients, weight in zip(accumulator.open_task_gradients, weighting["weights"], strict=True):
            for key, gradient in task_gradients.items():
                accumulator.aggregate[key].add_(gradient, alpha=float(weight))
        for gradient in accumulator.aggregate.values():
            gradient.div_(accumulator.total_valid_tokens)
        for index, unit in enumerate(accumulator.task_units):
            unit.update(
                {
                    "open_mopd_token_share": float(weighting["token_shares"][index]),
                    "open_mopd_share_weight": float(weighting["share_weights"][index]),
                    "open_mopd_reward_reference": float(weighting["reward_reference"]),
                    "open_mopd_gap_factor": float(weighting["gap_factors"][index]),
                    "open_mopd_loss_weight": float(weighting["weights"][index]),
                    "open_mopd_effective_share": float(weighting["effective_shares"][index]),
                }
            )
    is_probe = accumulator.aggregation == "variance_probe"
    if is_probe:
        aggregate_norm = 0.0
        update_successful = True
    else:
        for view in accumulator.views:
            key = _view_key(view)
            if key in accumulator.aggregate:
                view.set_optimizer_gradient(accumulator.aggregate[key])
        norm_square = next(iter(accumulator.aggregate.values())).new_zeros((), dtype=torch.float64)
        for gradient in accumulator.aggregate.values():
            norm_square += torch.sum(gradient.double().square())
        if _distributed():
            torch.distributed.all_reduce(norm_square)
        aggregate_norm = math.sqrt(float(norm_square))
        clip_prepared_gradients(optimizer, aggregate_norm)
        update_successful = bool(optimizer.step_with_ready_grads())
    attempted_responses = sum(int(unit["microbatches"]) * PROMPTS_PER_MICROBATCH for unit in accumulator.task_units)
    feedback = {
        "mopd": True,
        "operation": accumulator.operation,
        "aggregation": accumulator.aggregation,
        "task_units": accumulator.task_units,
        "optimizer_step_executed": not is_probe,
        "update_successful": update_successful,
        "aggregate_grad_norm": aggregate_norm,
        "aggregate_grad_clipped": bool(float(args.clip_grad) > 0 and aggregate_norm > float(args.clip_grad)),
        "attempted_responses": attempted_responses,
        "overflow_flag": False,
    }
    del optimizer._mopd_operation_accumulator
    return update_successful, aggregate_norm, feedback


__all__ = [
    "begin_mopd_operation",
    "clip_prepared_gradients",
    "finish_mopd_operation",
    "mopd_capture_step",
    "open_mopd_domain_weights",
    "score_prepared_gradients",
]
