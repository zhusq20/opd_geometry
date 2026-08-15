"""Recover optimizer directions and intended FP32 updates after a real step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .optimizer_views import OptimizerParameterView


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _step_number(state: dict[str, Any], group: dict[str, Any]) -> int:
    # Apex/Transformer-Engine FusedAdam advances one shared step on the
    # parameter group.  Native torch Adam stores it in each parameter state.
    value = group.get("step")
    if value is None:
        value = state.get("step", 0)
    if torch.is_tensor(value):
        value = value.detach().item()
    return int(value)


def _is_decoupled_adam(view: OptimizerParameterView) -> bool:
    name = type(view.inner_optimizer).__name__.lower()
    if "adamw" in name:
        return True
    if hasattr(view.inner_optimizer, "adam_w_mode"):
        return bool(view.inner_optimizer.adam_w_mode)
    if "decoupled_weight_decay" in view.optimizer_group:
        return bool(view.optimizer_group["decoupled_weight_decay"])
    # Megatron's supported Adam path is AdamW unless explicitly configured as
    # coupled Adam.  Unknown third-party Adam implementations fail closed when
    # nonzero decay makes the distinction observable.
    config = getattr(view.inner_optimizer, "defaults", {})
    return bool(config.get("decoupled_weight_decay", False))


def _muon_scale_factor_for_shape(
    view: OptimizerParameterView,
    rows: int,
    columns: int,
) -> float:
    try:
        from emerging_optimizers.orthogonalized_optimizers import get_muon_scale_factor
    except ImportError as exc:  # pragma: no cover - Muon environment contract
        raise RuntimeError("Muon geometry requires NVIDIA Emerging-Optimizers.") from exc

    size = [int(rows), int(columns)]
    inner = view.inner_optimizer
    mode = getattr(inner, "mode", "blockwise")
    partition_dim = None if mode == "blockwise" else getattr(view.optimizer_parameter, "partition_dim", None)
    if partition_dim == -1:
        partition_dim = None
    if partition_dim is not None:
        collection = getattr(inner, "pg_collection", None)
        group = None
        if collection is not None:
            group = collection.expt_tp if getattr(view.model_parameter, "expert_tp", False) else collection.tp
        if group is not None:
            size[int(partition_dim)] *= torch.distributed.get_world_size(group)
    scale_mode = getattr(inner, "slime_muon_scale_mode", None)
    if scale_mode is None:
        raise RuntimeError("Muon optimizer is missing its geometry scale-mode annotation.")
    return float(get_muon_scale_factor(size[0], size[1], mode=scale_mode)) * float(
        getattr(inner, "slime_muon_extra_scale_factor", 1.0)
    )


def _muon_scale_factor(
    view: OptimizerParameterView,
) -> tuple[float | torch.Tensor, dict[str, Any]]:
    """Return the exact scale applied by Muon, including fused-QKV splits."""

    shape = tuple(view.model_parameter.shape)
    if len(shape) != 2:
        raise ValueError(f"Muon optimizer member {view.name} is not a matrix: shape={shape}.")
    rows, columns = (int(value) for value in shape)
    inner = view.inner_optimizer
    is_qkv_fn = getattr(inner, "is_qkv_fn", None)
    split_qkv = bool(getattr(inner, "split_qkv", False))
    is_qkv = bool(callable(is_qkv_fn) and is_qkv_fn(view.optimizer_parameter))
    split_shapes = tuple(int(value) for value in (getattr(inner, "qkv_split_shapes", ()) or ()))
    if not (split_qkv and is_qkv and len(split_shapes) == 3):
        factor = _muon_scale_factor_for_shape(view, rows, columns)
        return factor, {
            "muon_scale_application": "whole_matrix",
            "muon_scale_factor": factor,
        }

    block_rows = sum(split_shapes)
    if block_rows <= 0 or rows % block_rows:
        raise ValueError(f"Muon QKV member {view.name} rows={rows} are incompatible with split shapes {split_shapes}.")
    query_groups = rows // block_rows
    component_factors = tuple(
        _muon_scale_factor_for_shape(view, query_groups * component_rows, columns) for component_rows in split_shapes
    )
    one_group = torch.cat(
        [
            torch.full(
                (component_rows,),
                factor,
                dtype=torch.float32,
                device=view.model_parameter.device,
            )
            for component_rows, factor in zip(split_shapes, component_factors, strict=True)
        ]
    )
    # Keep one factor per row and rely on broadcasting when recovering
    # post-NS. Expanding this to every matrix coordinate would allocate an
    # avoidable full-size FP32 tensor on every observed update.
    scale = one_group.repeat(query_groups).reshape(rows, 1)
    return scale, {
        "muon_scale_application": "qkv_componentwise",
        "muon_q_scale_factor": component_factors[0],
        "muon_k_scale_factor": component_factors[1],
        "muon_v_scale_factor": component_factors[2],
    }


@dataclass
class UpdateVectors:
    vectors: dict[str, torch.Tensor]
    optimizer_vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def _adam_vectors(
    view: OptimizerParameterView,
    theta_after_main: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    state = view.optimizer_state
    if "exp_avg" not in state or "exp_avg_sq" not in state:
        raise RuntimeError(f"Adam state was not initialized for {view.name} after a successful update.")
    group = view.optimizer_group
    beta1, beta2 = group.get("betas", (0.9, 0.999))
    beta1, beta2 = _number(beta1), _number(beta2)
    step = _step_number(state, group)
    if step <= 0:
        raise RuntimeError(f"Adam step must be positive after an update for {view.name}, got {step}.")
    lr = _number(group.get("lr"))
    eps = _number(group.get("eps", 1e-8))
    weight_decay = _number(group.get("weight_decay"))
    exp_avg = state["exp_avg"].detach().reshape(-1).to(torch.float32)
    exp_avg_sq = state["exp_avg_sq"].detach().reshape(-1).to(torch.float32)
    denominator_second_moment = state.get("max_exp_avg_sq", state["exp_avg_sq"])
    denominator_exp_avg_sq = denominator_second_moment.detach().reshape(-1).to(torch.float32)
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    m_hat = exp_avg / bias_correction1
    v_hat = denominator_exp_avg_sq / bias_correction2
    d_adam = m_hat / (torch.sqrt(v_hat) + eps)
    delta_data = -lr * d_adam

    decoupled = _is_decoupled_adam(view)
    if weight_decay != 0.0 and not decoupled:
        raise RuntimeError(
            f"Coupled Adam with nonzero weight decay cannot be split into exact data/WD updates for {view.name}; "
            "use Megatron's decoupled AdamW implementation."
        )
    decay_factor = 1.0 - lr * weight_decay if decoupled else 1.0
    if decay_factor == 0.0:
        raise RuntimeError(f"AdamW decay factor is zero for {view.name}; cannot recover theta_before.")
    theta_before_main = (theta_after_main - delta_data) / decay_factor
    d_wd = weight_decay * theta_before_main if decoupled else torch.zeros_like(theta_before_main)
    delta_wd = -lr * d_wd
    g_opt = view.optimizer_gradient()
    if g_opt is None:
        g_opt = torch.zeros_like(exp_avg)
    else:
        g_opt = g_opt.reshape(-1).to(torch.float32)
    if beta1 == 0.0:
        first_moment_previous = torch.zeros_like(exp_avg)
    else:
        first_moment_previous = (exp_avg - (1.0 - beta1) * g_opt) / beta1
    if beta2 == 0.0:
        second_moment_previous = torch.zeros_like(exp_avg_sq)
    else:
        second_moment_previous = (exp_avg_sq - (1.0 - beta2) * g_opt.square()) / beta2
    metadata = {
        "optimizer_step": step,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "weight_decay_applicable": bool(weight_decay != 0.0),
        "adam_beta1": beta1,
        "adam_beta2": beta2,
        "adam_eps": eps,
    }
    return (
        {
            "d_data": d_adam,
            "d_wd": d_wd,
            "delta_data_fp32": delta_data,
            "delta_wd_fp32": delta_wd,
            "delta_intended_fp32": delta_data + delta_wd,
        },
        {
            "m_hat": m_hat,
            "first_moment": exp_avg,
            "first_moment_previous": first_moment_previous,
            "first_moment_carry": beta1 * first_moment_previous,
            "first_moment_innovation": (1.0 - beta1) * g_opt,
            "second_moment": exp_avg_sq,
            "second_moment_previous": second_moment_previous,
            "second_moment_carry": beta2 * second_moment_previous,
            "second_moment_innovation": (1.0 - beta2) * g_opt.square(),
            "sqrt_v_hat": torch.sqrt(v_hat),
            "d_adam": d_adam,
            "total_direction": d_adam + d_wd,
            "effective_eta": lr / (torch.sqrt(v_hat) + eps),
            "g_opt_squared": g_opt.square(),
        },
        metadata,
    )


@torch.no_grad()
def _sgd_vectors(
    view: OptimizerParameterView,
    theta_after_main: torch.Tensor,
    g_opt: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    group = view.optimizer_group
    lr = _number(group.get("lr"))
    weight_decay = _number(group.get("weight_decay"))
    momentum = _number(group.get("momentum"))
    dampening = _number(group.get("dampening"))
    nesterov = bool(group.get("nesterov", False))
    state = view.optimizer_state
    buffer = state.get("momentum_buffer")
    velocity = buffer.detach().reshape(-1).to(torch.float32) if buffer is not None else None

    if lr == 0.0:
        theta_before_main = theta_after_main
        total_direction = torch.zeros_like(theta_after_main)
    elif momentum != 0.0:
        if velocity is None:
            raise RuntimeError(f"SGD momentum state was not initialized for {view.name}.")
        if nesterov:
            # new = old - lr * (g + wd*old + momentum*velocity_t)
            denominator = 1.0 - lr * weight_decay
            if denominator == 0.0:
                raise RuntimeError(f"Nesterov SGD decay factor is zero for {view.name}.")
            theta_before_main = (theta_after_main + lr * (g_opt + momentum * velocity)) / denominator
            total_direction = g_opt + weight_decay * theta_before_main + momentum * velocity
        else:
            total_direction = velocity
            theta_before_main = theta_after_main + lr * total_direction
    else:
        denominator = 1.0 - lr * weight_decay
        if denominator == 0.0:
            raise RuntimeError(f"SGD decay factor is zero for {view.name}.")
        theta_before_main = (theta_after_main + lr * g_opt) / denominator
        total_direction = g_opt + weight_decay * theta_before_main

    d_wd = weight_decay * theta_before_main
    # This algebraic split is exact even with momentum: all carried history is
    # assigned to the data branch, while d_wd is the current explicit L2 term.
    d_data = total_direction - d_wd
    delta_data = -lr * d_data
    delta_wd = -lr * d_wd
    optimizer_vectors: dict[str, torch.Tensor] = {}
    if velocity is not None:
        grad_with_decay = g_opt + d_wd
        velocity_previous = (
            (velocity - (1.0 - dampening) * grad_with_decay) / momentum
            if momentum != 0.0
            else torch.zeros_like(velocity)
        )
        optimizer_vectors = {
            "velocity": velocity,
            "velocity_previous": velocity_previous,
            "velocity_carry": momentum * velocity_previous,
            "velocity_innovation": (1.0 - dampening) * g_opt,
        }
    metadata = {
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "weight_decay_applicable": bool(weight_decay != 0.0),
        "sgd_momentum": momentum,
        "sgd_dampening": dampening,
        "sgd_nesterov": nesterov,
    }
    return (
        {
            "d_data": d_data,
            "d_wd": d_wd,
            "delta_data_fp32": delta_data,
            "delta_wd_fp32": delta_wd,
            "delta_intended_fp32": delta_data + delta_wd,
        },
        optimizer_vectors,
        metadata,
    )


@torch.no_grad()
def _muon_vectors(
    view: OptimizerParameterView,
    theta_before_main: torch.Tensor | None,
    theta_after_main: torch.Tensor,
    g_opt: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    if theta_before_main is None:
        raise RuntimeError(f"Muon exact intended updates require a pre-step FP32 master snapshot for {view.name}.")
    group = view.optimizer_group
    lr = _number(group.get("lr"))
    weight_decay = _number(group.get("weight_decay"))
    beta = _number(group.get("momentum_beta", 0.0))
    use_nesterov = bool(getattr(view.inner_optimizer, "use_nesterov", False))
    state = view.optimizer_state
    momentum = state.get("momentum_buffer")
    if momentum is None:
        raise RuntimeError(f"Muon momentum state was not initialized for {view.name}.")
    momentum = momentum.detach().reshape(-1).to(torch.float32)
    if beta == 0.0:
        momentum_previous = torch.zeros_like(momentum)
    else:
        momentum_previous = (momentum - (1.0 - beta) * g_opt) / beta
    ns_input = torch.lerp(g_opt, momentum, beta) if use_nesterov else momentum

    d_wd = weight_decay * theta_before_main
    delta_wd = -lr * d_wd
    delta_total = theta_after_main - theta_before_main
    delta_data = delta_total - delta_wd
    d_data = -delta_data / lr if lr != 0.0 else torch.zeros_like(delta_data)
    scale_factor, scale_metadata = _muon_scale_factor(view)
    if torch.is_tensor(scale_factor):
        scale_factor = scale_factor.to(device=d_data.device, dtype=torch.float32)
        matrix = d_data.reshape(int(view.model_parameter.shape[0]), int(view.model_parameter.shape[1]))
        post_ns = torch.where(scale_factor != 0.0, matrix / scale_factor, torch.zeros_like(matrix)).reshape(-1)
    else:
        post_ns = d_data / scale_factor if scale_factor != 0.0 else torch.zeros_like(d_data)
    metadata = {
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "weight_decay_applicable": bool(weight_decay != 0.0),
        "muon_momentum": beta,
        "muon_use_nesterov": use_nesterov,
        **scale_metadata,
    }
    return (
        {
            "d_data": d_data,
            "d_wd": d_wd,
            "delta_data_fp32": delta_data,
            "delta_wd_fp32": delta_wd,
            "delta_intended_fp32": delta_data + delta_wd,
        },
        {
            "momentum": momentum,
            "momentum_previous": momentum_previous,
            "momentum_carry": beta * momentum_previous,
            "momentum_innovation": (1.0 - beta) * g_opt,
            "ns_input": ns_input,
            "post_ns": post_ns,
        },
        metadata,
    )


@torch.no_grad()
def compute_update_vectors(
    view: OptimizerParameterView,
    *,
    theta_before: torch.Tensor,
    theta_reference: torch.Tensor,
    theta_before_main: torch.Tensor | None = None,
) -> UpdateVectors:
    """Return the specification's common vectors for one successful update."""

    model_after = view.model_value()
    theta_before_model = theta_before.reshape(-1)
    theta_reference_model = theta_reference.reshape(-1)
    theta_before_fp32 = theta_before_model.to(torch.float32)
    theta_reference_fp32 = theta_reference_model.to(torch.float32)
    model_after_fp32 = model_after.reshape(-1).to(torch.float32)
    main_after = view.optimizer_value().to(torch.float32)
    g_opt = view.optimizer_gradient()
    if g_opt is None:
        g_opt = torch.zeros_like(main_after)
    else:
        g_opt = g_opt.to(torch.float32)
    if main_after.numel() != theta_before_fp32.numel() or g_opt.numel() != main_after.numel():
        raise ValueError(f"Optimizer/model view size mismatch for {view.name}.")

    if view.optimizer_kind == "adam":
        common, optimizer_vectors, metadata = _adam_vectors(view, main_after)
    elif view.optimizer_kind == "sgd":
        common, optimizer_vectors, metadata = _sgd_vectors(view, main_after, g_opt)
    elif view.optimizer_kind == "muon":
        common, optimizer_vectors, metadata = _muon_vectors(
            view,
            theta_before_main.reshape(-1).to(torch.float32) if theta_before_main is not None else None,
            main_after,
            g_opt,
        )
    else:
        if theta_before_main is None:
            if model_after.dtype != torch.float32:
                raise RuntimeError(
                    f"Unsupported optimizer {type(view.inner_optimizer).__name__} needs an FP32 master snapshot."
                )
            theta_before_main = theta_before_fp32
        lr = _number(view.optimizer_group.get("lr"))
        intended = main_after - theta_before_main.reshape(-1).to(torch.float32)
        d_data = -intended / lr if lr != 0.0 else torch.zeros_like(intended)
        zeros = torch.zeros_like(intended)
        common = {
            "d_data": d_data,
            "d_wd": zeros,
            "delta_data_fp32": intended,
            "delta_wd_fp32": zeros,
            "delta_intended_fp32": intended,
        }
        optimizer_vectors = {}
        metadata = {"learning_rate": lr, "weight_decay_applicable": False}

    vectors = {
        "theta_before": theta_before_model,
        "theta_reference": theta_reference_model,
        "g_opt": g_opt,
        **common,
        "delta_model": model_after_fp32 - theta_before_fp32,
        "displacement": model_after_fp32 - theta_reference_fp32,
    }
    return UpdateVectors(vectors=vectors, optimizer_vectors=optimizer_vectors, metadata=metadata)


__all__ = ["UpdateVectors", "compute_update_vectors"]
