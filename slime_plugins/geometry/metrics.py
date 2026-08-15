"""Pure metric calculations for projected optimizer geometry."""

from __future__ import annotations

import math
from typing import Any

import torch


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.to(torch.float64)).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left64 = left.to(torch.float64)
    right64 = right.to(torch.float64)
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    if float(denominator.item()) == 0.0:
        return None
    return float((torch.dot(left64, right64) / denominator).item())


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0 else None


def geometry_metrics(
    *,
    weight_before: torch.Tensor,
    weight_after: torch.Tensor,
    gradient: torch.Tensor,
    initial_weight: torch.Tensor,
    exact_weight_before_sq: float,
    exact_weight_after_sq: float,
    exact_gradient_sq: float,
    parameter_count: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Calculate scalar geometry and return the named projected vectors."""

    update = weight_after - weight_before
    displacement = weight_after - initial_weight
    weight_norm_before = math.sqrt(max(exact_weight_before_sq, 0.0))
    weight_norm_after = math.sqrt(max(exact_weight_after_sq, 0.0))
    gradient_norm = math.sqrt(max(exact_gradient_sq, 0.0))
    update_norm = _norm(update)
    displacement_norm = _norm(displacement)
    gradient_update_inner_product = float(torch.dot(gradient.to(torch.float64), update.to(torch.float64)).item())
    cos_gradient_update = _cosine(gradient, update)

    scalars: dict[str, Any] = {
        "parameter_count": int(parameter_count),
        # ``weight_norm`` remains the compact compatibility name and is the
        # pre-step denominator used by all relative-update metrics.
        "weight_norm": weight_norm_before,
        "weight_norm_before": weight_norm_before,
        "weight_norm_after": weight_norm_after,
        "relative_weight_norm_change": _ratio(weight_norm_after - weight_norm_before, weight_norm_before),
        "gradient_norm": gradient_norm,
        "update_norm_sketch": update_norm,
        "displacement_norm_sketch": displacement_norm,
        "update_to_weight_ratio_sketch": _ratio(update_norm, weight_norm_before),
        "gradient_to_weight_ratio": _ratio(gradient_norm, weight_norm_before),
        "update_to_gradient_ratio_sketch": _ratio(update_norm, gradient_norm),
        "gradient_update_inner_product_sketch": gradient_update_inner_product,
        "gradient_directional_step_sketch": _ratio(-gradient_update_inner_product, exact_gradient_sq),
        "cos_gradient_update_sketch": cos_gradient_update,
        "descent_alignment_sketch": -cos_gradient_update if cos_gradient_update is not None else None,
        "cos_weight_update_sketch": _cosine(weight_before, update),
        "cos_gradient_displacement_sketch": _cosine(gradient, displacement),
        "cos_update_displacement_sketch": _cosine(update, displacement),
    }
    vectors = {
        "weight_before": weight_before,
        "weight_after": weight_after,
        "gradient": gradient,
        "update": update,
        "displacement": displacement,
    }
    return scalars, vectors
