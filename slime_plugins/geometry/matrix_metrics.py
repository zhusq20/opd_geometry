"""Low-frequency diagnostics for a fixed deterministic matrix sample."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import torch

MATRIX_EPS = 1.0e-6
EXACT_MAX_MIN_DIM = 256
EXACT_MAX_NUMEL = 1_000_000
HUTCHINSON_PROBES = 8


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def _coefficient_of_variation(values: torch.Tensor) -> float | None:
    values = values.to(torch.float64)
    mean = float(values.mean()) if values.numel() else 0.0
    return float(values.std(unbiased=False) / mean) if mean != 0.0 else None


def _seed(name: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF


@torch.no_grad()
def _randomized_singular_values(matrix: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    rows, columns = matrix.shape
    rank = max(1, min(int(rank), rows, columns))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    omega = torch.randint(0, 2, (columns, rank), generator=generator, dtype=torch.int8)
    omega = omega.to(device=matrix.device, dtype=torch.float32).mul_(2).sub_(1)
    value = matrix.to(torch.float32)
    basis, _ = torch.linalg.qr(value @ omega, mode="reduced")
    # One power iteration materially improves the leading singular estimates
    # while retaining O(rank * numel) work.
    basis, _ = torch.linalg.qr(value @ (value.transpose(0, 1) @ basis), mode="reduced")
    compressed = basis.transpose(0, 1) @ value
    return torch.linalg.svdvals(compressed).to(torch.float64)


@torch.no_grad()
def _orthogonality_hutchinson(matrix: torch.Tensor, seed: int) -> float:
    rows, columns = matrix.shape
    dimension = min(rows, columns)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    probes = torch.randint(
        0,
        2,
        (dimension, HUTCHINSON_PROBES),
        generator=generator,
        dtype=torch.int8,
    )
    probes = probes.to(device=matrix.device, dtype=torch.float32).mul_(2).sub_(1)
    value = matrix.to(torch.float32)
    if rows <= columns:
        residual = value @ (value.transpose(0, 1) @ probes) - probes
    else:
        residual = value.transpose(0, 1) @ (value @ probes) - probes
    frobenius_sq = float(residual.square().sum(dtype=torch.float64) / HUTCHINSON_PROBES)
    return math.sqrt(max(frobenius_sq, 0.0)) / math.sqrt(dimension)


@torch.no_grad()
def matrix_diagnostics(
    matrix: torch.Tensor,
    *,
    name: str,
    seed: int,
    randomized_rank: int,
    eps: float = MATRIX_EPS,
    include_orthogonality: bool = False,
) -> dict[str, Any]:
    """Compute exact small-matrix or explicitly sketched large-matrix metrics."""

    if matrix.ndim != 2:
        raise ValueError(f"Matrix diagnostics require rank two, got shape {tuple(matrix.shape)}.")
    if eps <= 0.0:
        raise ValueError("Matrix diagnostic epsilon must be positive.")
    value = matrix.detach().to(torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Matrix diagnostics require finite coordinates.")
    rows, columns = (int(size) for size in value.shape)
    if rows == 0 or columns == 0:
        raise ValueError("Matrix diagnostics do not support an empty matrix.")
    minimum_dimension = min(rows, columns)
    row_norms = torch.linalg.vector_norm(value, dim=1)
    column_norms = torch.linalg.vector_norm(value, dim=0)
    frobenius_sq = float(torch.sum(value.square(), dtype=torch.float64))
    exact = minimum_dimension <= EXACT_MAX_MIN_DIM and value.numel() <= EXACT_MAX_NUMEL
    singular_values = (
        torch.linalg.svdvals(value).to(torch.float64)
        if exact
        else _randomized_singular_values(
            value,
            randomized_rank,
            _seed(name, seed),
        )
    )
    singular_values = torch.sort(singular_values, descending=True).values
    spectral = float(singular_values[0]) if singular_values.numel() else 0.0
    captured_energy = singular_values.square()
    captured_sum = float(captured_energy.sum())
    total_energy = max(frobenius_sq, 0.0)
    suffix = "" if exact else "_sketch"
    metrics: dict[str, Any] = {
        "rows": rows,
        "columns": columns,
        "eps": eps,
        "row_norm_cv": _coefficient_of_variation(row_norms),
        "column_norm_cv": _coefficient_of_variation(column_norms),
        f"spectral_norm{suffix}": spectral,
        f"stable_rank{suffix}": _safe_ratio(total_energy, spectral * spectral),
        "spectrum_method": "exact_svd" if exact else "deterministic_randomized_svd",
        "randomized_rank": None if exact else int(singular_values.numel()),
    }

    if total_energy == 0.0:
        effective_rank = 0
        entropy = 0.0
    else:
        cumulative = torch.cumsum(captured_energy, dim=0)
        reached = torch.nonzero(cumulative >= 0.99 * total_energy)
        effective_rank = int(reached[0].item() + 1) if reached.numel() else None
        probabilities = captured_energy / total_energy
        tail = max(total_energy - captured_sum, 0.0) / total_energy
        entropy_terms = probabilities[probabilities > 0]
        entropy = float(-(entropy_terms * torch.log(entropy_terms)).sum())
        # Treating all unresolved tail energy as one component is a lower
        # bound on spectral entropy; this is exact when no tail remains.
        if tail > 0.0:
            entropy -= tail * math.log(tail)
    if exact:
        metrics["effective_rank_99_energy"] = effective_rank
        metrics["spectral_entropy"] = entropy
    else:
        metrics["effective_rank_99_energy_sketch"] = (
            effective_rank if effective_rank is not None else int(singular_values.numel()) + 1
        )
        metrics["effective_rank_99_energy_censored_sketch"] = effective_rank is None
        metrics["spectral_entropy_sketch"] = entropy
        metrics["captured_spectral_energy_fraction_sketch"] = _safe_ratio(captured_sum, total_energy)

    if singular_values.numel():
        s5, s95 = torch.quantile(
            singular_values,
            torch.tensor([0.05, 0.95], dtype=torch.float64, device=singular_values.device),
        )
        regularized = float(s95 / torch.maximum(s5, eps * s95)) if float(s95) != 0.0 else None
        log_spread = (
            math.log(float(singular_values[0]) / max(float(singular_values[-1]), eps * float(singular_values[0])))
            if float(singular_values[0]) != 0.0
            else None
        )
    else:
        regularized = None
        log_spread = None
    metrics[f"regularized_s95_to_s5{suffix}"] = regularized
    metrics[f"singular_value_log_spread{suffix}"] = log_spread

    if include_orthogonality:
        if exact:
            identity = torch.eye(minimum_dimension, device=value.device, dtype=torch.float32)
            gram = value @ value.transpose(0, 1) if rows <= columns else value.transpose(0, 1) @ value
            metrics["orthogonality_error"] = float(
                torch.linalg.vector_norm((gram - identity).to(torch.float64)) / math.sqrt(minimum_dimension)
            )
        else:
            metrics["orthogonality_error_sketch"] = _orthogonality_hutchinson(
                value,
                _seed(f"{name}:orthogonality", seed),
            )
    return metrics


def selected_view_ids(
    views: list[Any],
    *,
    seed: int,
    count: int,
) -> set[int]:
    """Choose fixed full matrices per real optimizer branch."""

    candidates: dict[str, list[tuple[bytes, int]]] = defaultdict(list)
    for view_id, view in enumerate(views):
        parameter = view.model_parameter
        full_view = int(view.start) == 0 and (view.stop is None or int(view.stop) == parameter.numel())
        if parameter.ndim != 2 or not full_view:
            continue
        key = f"{seed}:{view.name}:{view.optimizer_branch}".encode()
        candidates[str(view.optimizer_branch)].append((hashlib.blake2b(key, digest_size=16).digest(), view_id))
    return {view_id for branch_candidates in candidates.values() for _, view_id in sorted(branch_candidates)[:count]}


def matrix_macro_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    """Matrix-equal median/IQR summaries by semantic operator and vector."""

    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        operator = str(record["operator"])
        for vector_name, metrics in record.get("vectors", {}).items():
            for metric_name, value in metrics.items():
                if metric_name in {"rows", "columns", "eps", "randomized_rank"}:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value is None:
                    continue
                if not math.isfinite(float(value)):
                    continue
                values[(operator, vector_name, metric_name)].append(float(value))
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for (operator, vector_name, metric_name), samples in sorted(values.items()):
        tensor = torch.tensor(samples, dtype=torch.float64)
        q1, median, q3 = torch.quantile(
            tensor,
            torch.tensor([0.25, 0.50, 0.75], dtype=torch.float64),
        )
        output.setdefault(operator, {}).setdefault(vector_name, {})[metric_name] = {
            "median_sketch": float(median),
            "iqr_sketch": float(q3 - q1),
            "matrix_count": len(samples),
        }
    return output


__all__ = [
    "EXACT_MAX_MIN_DIM",
    "EXACT_MAX_NUMEL",
    "MATRIX_EPS",
    "matrix_diagnostics",
    "matrix_macro_summary",
    "selected_view_ids",
]
