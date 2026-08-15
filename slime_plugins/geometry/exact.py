"""Exact streaming statistics for optimizer-update geometry.

The accumulator keeps only scalar sufficient statistics per reporting group.
It never concatenates model tensors and performs two compact distributed
collectives at finalization (one SUM and one MAX), so exact norms, dots, and
cosines do not require full-vector copies or many latency-bound reductions.
Inputs are interpreted in FP32, matching the metric specification; sums and
inner products are accumulated in FP64.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

VECTOR_NAMES = (
    "theta_before",
    "theta_reference",
    "g_raw",
    "g_opt",
    "d_data",
    "d_wd",
    "delta_data_fp32",
    "delta_wd_fp32",
    "delta_intended_fp32",
    "delta_model",
    "displacement",
    # Optimizer-specific vectors. Missing vectors simply leave no fields for a
    # branch, so disabled/not-applicable components are never fabricated.
    "momentum",
    "momentum_previous",
    "momentum_carry",
    "momentum_innovation",
    "ns_input",
    "post_ns",
    "m_hat",
    "first_moment",
    "first_moment_previous",
    "first_moment_carry",
    "first_moment_innovation",
    "second_moment",
    "second_moment_previous",
    "second_moment_carry",
    "second_moment_innovation",
    "sqrt_v_hat",
    "d_adam",
    "total_direction",
    "effective_eta",
    "g_opt_squared",
    "velocity",
    "velocity_previous",
    "velocity_carry",
    "velocity_innovation",
)

PAIR_NAMES = (
    ("g_raw", "g_opt"),
    ("g_opt", "d_data"),
    ("g_opt", "delta_intended_fp32"),
    ("delta_intended_fp32", "delta_model"),
    ("theta_before", "delta_model"),
    ("delta_model", "displacement"),
    ("g_opt", "delta_model"),
    ("g_opt", "momentum"),
    ("momentum", "ns_input"),
    ("ns_input", "post_ns"),
    ("g_opt", "post_ns"),
    ("post_ns", "delta_intended_fp32"),
    ("momentum", "momentum_previous"),
    ("g_opt", "m_hat"),
    ("g_opt", "d_adam"),
    ("d_adam", "total_direction"),
    ("d_wd", "d_adam"),
    ("first_moment", "first_moment_previous"),
    ("g_opt", "velocity"),
    ("velocity", "velocity_previous"),
)

# Ratio of |intended update| to one ULP of theta_before.  These fixed bins
# straddle the half-ULP rounding boundary required by the specification.
ULP_BIN_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _as_fp32(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    if hasattr(value, "_local_tensor"):
        value = value._local_tensor
    return value.reshape(-1).to(torch.float32)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def _model_ulp(theta_before: torch.Tensor) -> torch.Tensor:
    """Return one positive-direction ULP in the model parameter dtype."""

    theta = theta_before.detach()
    if hasattr(theta, "to_local"):
        theta = theta.to_local()
    if hasattr(theta, "_local_tensor"):
        theta = theta._local_tensor
    if not theta.is_floating_point():
        raise TypeError(f"ULP metrics require a floating tensor, got {theta.dtype}.")
    positive_inf = torch.full_like(theta, math.inf)
    negative_inf = torch.full_like(theta, -math.inf)
    spacing_up = torch.nextafter(theta, positive_inf) - theta
    # The positive neighbour of the largest finite value is infinity.  Use the
    # finite negative-direction spacing only for that boundary case.
    spacing_down = theta - torch.nextafter(theta, negative_inf)
    spacing = torch.where(torch.isfinite(spacing_up), spacing_up, spacing_down)
    # Some accelerator kernels flush ``nextafter(0, +inf)``. Recover the IEEE
    # smallest subnormal analytically instead of substituting the much larger
    # smallest normal value (``tiny``).
    dtype_info = torch.finfo(theta.dtype)
    smallest_subnormal = float(dtype_info.tiny) * float(dtype_info.eps)
    spacing = spacing.to(torch.float32)
    return torch.where(spacing > 0, spacing, spacing.new_full((), smallest_subnormal))


class ExactGeometryAccumulator:
    """Accumulate exact vector and FP32-to-model-dtype realization metrics."""

    def __init__(
        self,
        group_names: Sequence[str],
        device: torch.device,
        *,
        vector_names: Sequence[str] = VECTOR_NAMES,
        pair_names: Sequence[tuple[str, str]] = PAIR_NAMES,
    ) -> None:
        self.group_names = tuple(group_names)
        self.vector_names = tuple(vector_names)
        self.pair_names = tuple(pair_names)
        self._vector_index = {name: index for index, name in enumerate(self.vector_names)}
        self._pair_index = {pair: index for index, pair in enumerate(self.pair_names)}
        group_count = len(self.group_names)
        vector_count = len(self.vector_names)
        pair_count = len(self.pair_names)

        self.sum_sq = torch.zeros((group_count, vector_count), dtype=torch.float64, device=device)
        self.max_abs = torch.zeros((group_count, vector_count), dtype=torch.float32, device=device)
        self.zero_count = torch.zeros((group_count, vector_count), dtype=torch.int64, device=device)
        self.element_count = torch.zeros((group_count, vector_count), dtype=torch.int64, device=device)
        self.dots = torch.zeros((group_count, pair_count), dtype=torch.float64, device=device)

        bin_count = len(ULP_BIN_EDGES) - 1
        self.realization_counts = torch.zeros((group_count, 3), dtype=torch.int64, device=device)
        self.realization_energy = torch.zeros((group_count, 6), dtype=torch.float64, device=device)
        self.ulp_bin_counts = torch.zeros((group_count, bin_count, 3), dtype=torch.int64, device=device)
        self.ulp_bin_energy = torch.zeros((group_count, bin_count, 2), dtype=torch.float64, device=device)

    @staticmethod
    def _group_ids(group_ids: Sequence[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(group_ids, dtype=torch.int64, device=device).reshape(-1)

    @torch.no_grad()
    def add(
        self,
        group_ids: Sequence[int] | torch.Tensor,
        vectors: Mapping[str, torch.Tensor],
    ) -> None:
        """Add one parameter view to every listed reporting group."""

        if not vectors:
            return
        unknown = set(vectors).difference(self._vector_index)
        if unknown:
            raise KeyError(f"Unknown exact-geometry vectors: {sorted(unknown)}")

        first = next(iter(vectors.values()))
        ids = self._group_ids(group_ids, first.device)
        if ids.numel() == 0:
            return

        fp32 = {name: _as_fp32(value) for name, value in vectors.items()}
        numel = next(iter(fp32.values())).numel()
        if any(value.numel() != numel for value in fp32.values()):
            raise ValueError("All geometry vectors for one parameter view must have equal length.")

        for name, value in fp32.items():
            vector_id = self._vector_index[name]
            square_sum = torch.sum(value.square(), dtype=torch.float64)
            absolute_max = value.abs().amax() if value.numel() else value.new_zeros(())
            zeros = torch.count_nonzero(value == 0)
            repeats = ids.numel()
            self.sum_sq[:, vector_id].index_add_(0, ids, square_sum.expand(repeats))
            current_max = self.max_abs[:, vector_id].index_select(0, ids)
            self.max_abs[:, vector_id].index_copy_(0, ids, torch.maximum(current_max, absolute_max))
            self.zero_count[:, vector_id].index_add_(0, ids, zeros.expand(repeats))
            self.element_count[:, vector_id].index_add_(
                0,
                ids,
                torch.full((repeats,), numel, dtype=torch.int64, device=value.device),
            )

        for pair, pair_id in self._pair_index.items():
            if pair[0] not in fp32 or pair[1] not in fp32:
                continue
            dot = torch.sum(fp32[pair[0]].to(torch.float64) * fp32[pair[1]], dtype=torch.float64)
            self.dots[:, pair_id].index_add_(0, ids, dot.expand(ids.numel()))

        required = {"theta_before", "delta_intended_fp32", "delta_model"}
        if required.issubset(fp32):
            self._add_realization(ids, vectors["theta_before"], fp32["delta_intended_fp32"], fp32["delta_model"])

    @torch.no_grad()
    def add_dot(
        self,
        group_ids: Sequence[int] | torch.Tensor,
        pair: tuple[str, str],
        value: torch.Tensor | float,
    ) -> None:
        """Add a precomputed exact dot without retaining either source vector."""

        if pair not in self._pair_index:
            raise KeyError(f"Unknown exact-geometry pair: {pair}")
        scalar = torch.as_tensor(value, dtype=torch.float64, device=self.dots.device).reshape(())
        ids = self._group_ids(group_ids, self.dots.device)
        self.dots[:, self._pair_index[pair]].index_add_(0, ids, scalar.expand(ids.numel()))

    @torch.no_grad()
    def _add_realization(
        self,
        ids: torch.Tensor,
        theta_before: torch.Tensor,
        intended: torch.Tensor,
        realized: torch.Tensor,
    ) -> None:
        ulp = _model_ulp(theta_before).reshape(-1)
        if ulp.numel() != intended.numel():
            raise ValueError("theta_before and update vectors must have equal length for ULP metrics.")

        intended_abs = intended.abs()
        realized_abs = realized.abs()
        intended_energy = intended.square()
        realized_energy = realized.square()
        residual_energy = (realized - intended).square()
        changed = realized != 0
        below_half_ulp = intended_abs < (0.5 * ulp)
        sign_flip = changed & (intended != 0) & (torch.signbit(realized) != torch.signbit(intended))

        counts = torch.stack(
            (
                torch.tensor(intended.numel(), dtype=torch.int64, device=intended.device),
                torch.count_nonzero(changed),
                torch.count_nonzero(below_half_ulp),
            )
        )
        energy = torch.stack(
            (
                torch.sum(intended_energy, dtype=torch.float64),
                torch.sum(realized_energy, dtype=torch.float64),
                torch.sum(residual_energy, dtype=torch.float64),
                torch.sum(intended_energy[~changed], dtype=torch.float64),
                torch.sum(intended_energy[changed & (realized_abs > intended_abs)], dtype=torch.float64),
                torch.sum(intended_energy[changed & (realized_abs < intended_abs)], dtype=torch.float64),
            )
        )
        self.realization_counts.index_add_(0, ids, counts.expand(ids.numel(), -1))
        self.realization_energy.index_add_(0, ids, energy.expand(ids.numel(), -1))

        ratio = intended_abs / ulp
        for bin_id, (lower, upper) in enumerate(zip(ULP_BIN_EDGES[:-1], ULP_BIN_EDGES[1:], strict=True)):
            selected = ratio >= lower
            if math.isfinite(upper):
                selected &= ratio < upper
            bin_counts = torch.stack(
                (
                    torch.count_nonzero(selected),
                    torch.count_nonzero(selected & changed),
                    torch.count_nonzero(selected & sign_flip),
                )
            )
            bin_energy = torch.stack(
                (
                    torch.sum(intended_energy[selected], dtype=torch.float64),
                    torch.sum(realized_energy[selected], dtype=torch.float64),
                )
            )
            self.ulp_bin_counts[:, bin_id].index_add_(0, ids, bin_counts.expand(ids.numel(), -1))
            self.ulp_bin_energy[:, bin_id].index_add_(0, ids, bin_energy.expand(ids.numel(), -1))

    def _reduced(self) -> tuple[torch.Tensor, ...]:
        sum_values = (
            self.sum_sq.clone(),
            self.zero_count.clone(),
            self.element_count.clone(),
            self.dots.clone(),
            self.realization_counts.clone(),
            self.realization_energy.clone(),
            self.ulp_bin_counts.clone(),
            self.ulp_bin_energy.clone(),
        )
        maximum = self.max_abs.clone()
        if _distributed():
            sizes = [value.numel() for value in sum_values]
            packed = torch.cat([value.reshape(-1).to(torch.float64) for value in sum_values])
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            reduced_sum_values: list[torch.Tensor] = []
            offset = 0
            for original, size in zip(sum_values, sizes, strict=True):
                reduced = packed[offset : offset + size].reshape(original.shape)
                if not original.is_floating_point():
                    reduced = reduced.round()
                reduced_sum_values.append(reduced.to(original.dtype))
                offset += size
            sum_values = tuple(reduced_sum_values)
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)

        (
            sum_sq,
            zero_count,
            element_count,
            dots,
            realization_counts,
            realization_energy,
            ulp_bin_counts,
            ulp_bin_energy,
        ) = sum_values
        return (
            sum_sq,
            maximum,
            zero_count,
            element_count,
            dots,
            realization_counts,
            realization_energy,
            ulp_bin_counts,
            ulp_bin_energy,
        )

    @torch.no_grad()
    def finalize(self) -> dict[str, dict[str, Any]]:
        """Use two packed reductions and return JSON-safe scalar group metrics."""

        (
            sum_sq,
            max_abs,
            zero_count,
            element_count,
            dots,
            realization_counts,
            realization_energy,
            ulp_bin_counts,
            ulp_bin_energy,
        ) = (value.cpu() for value in self._reduced())

        output: dict[str, dict[str, Any]] = {}
        for group_id, group_name in enumerate(self.group_names):
            metrics: dict[str, Any] = {}
            norms: dict[str, float] = {}
            for vector_id, name in enumerate(self.vector_names):
                count = int(element_count[group_id, vector_id])
                if count == 0:
                    continue
                square_sum = float(sum_sq[group_id, vector_id])
                absolute_max = float(max_abs[group_id, vector_id])
                if not math.isfinite(square_sum) or not math.isfinite(absolute_max):
                    raise ValueError(f"Exact geometry received non-finite values for {group_name}/{name}.")
                norm = math.sqrt(max(square_sum, 0.0))
                norms[name] = norm
                if name == "theta_before":
                    metrics["parameter_count"] = count
                metrics[f"{name}_l2"] = norm
                metrics[f"{name}_rms"] = math.sqrt(max(square_sum / count, 0.0))
                metrics[f"{name}_linf"] = absolute_max
                metrics[f"{name}_exact_zero_fraction"] = float(zero_count[group_id, vector_id]) / count

            for pair_id, (left, right) in enumerate(self.pair_names):
                if left not in norms or right not in norms:
                    continue
                dot = float(dots[group_id, pair_id])
                if not math.isfinite(dot):
                    raise ValueError(f"Exact geometry received a non-finite dot for {group_name}/{left}/{right}.")
                metrics[f"dot_{left}_{right}"] = dot
                denominator = norms[left] * norms[right]
                metrics[f"cos_{left}_{right}"] = _safe_ratio(dot, denominator)

            self._add_ratios(metrics, norms)
            self._finalize_realization(
                metrics,
                realization_counts[group_id],
                realization_energy[group_id],
                ulp_bin_counts[group_id],
                ulp_bin_energy[group_id],
            )
            if metrics:
                output[group_name] = metrics
        return output

    @staticmethod
    def _add_ratios(metrics: dict[str, Any], norms: Mapping[str, float]) -> None:
        def ratio(name: str, numerator: str, denominator: str) -> None:
            if numerator in norms and denominator in norms:
                metrics[name] = _safe_ratio(norms[numerator], norms[denominator])

        ratio("g_raw_to_theta_ratio", "g_raw", "theta_before")
        ratio("d_data_to_g_opt_ratio", "d_data", "g_opt")
        ratio("delta_wd_to_delta_data_ratio", "delta_wd_fp32", "delta_data_fp32")
        ratio("delta_intended_to_theta_ratio", "delta_intended_fp32", "theta_before")
        ratio("delta_model_to_theta_ratio", "delta_model", "theta_before")
        ratio("displacement_to_reference_ratio", "displacement", "theta_reference")
        grad_sq = norms.get("g_opt", 0.0) ** 2
        dot = metrics.get("dot_g_opt_delta_intended_fp32")
        metrics["gradient_directional_step"] = _safe_ratio(-float(dot), grad_sq) if dot is not None else None

    @staticmethod
    def _finalize_realization(
        metrics: dict[str, Any],
        counts: torch.Tensor,
        energy: torch.Tensor,
        bin_counts: torch.Tensor,
        bin_energy: torch.Tensor,
    ) -> None:
        count, changed, below_half = (int(value) for value in counts)
        if count == 0:
            return
        intended, realized, residual, zeroed, amplified, attenuated = (float(value) for value in energy)
        if not all(math.isfinite(value) for value in (intended, realized, residual, zeroed, amplified, attenuated)):
            raise ValueError("Exact update-realization energy contains non-finite values.")
        metrics["model_change_fraction"] = changed / count
        metrics["intended_below_half_ulp_fraction"] = below_half / count
        metrics["energy_survival"] = _safe_ratio(realized, intended)
        metrics["quantization_residual"] = math.sqrt(max(residual, 0.0) / intended) if intended != 0.0 else None
        metrics["intended_energy_zeroed_fraction"] = _safe_ratio(zeroed, intended)
        metrics["intended_energy_amplified_fraction"] = _safe_ratio(amplified, intended)
        metrics["intended_energy_attenuated_fraction"] = _safe_ratio(attenuated, intended)

        bins: dict[str, dict[str, Any]] = {}
        for bin_id, (lower, upper) in enumerate(zip(ULP_BIN_EDGES[:-1], ULP_BIN_EDGES[1:], strict=True)):
            total, nonzero, flips = (int(value) for value in bin_counts[bin_id])
            intended_bin, realized_bin = (float(value) for value in bin_energy[bin_id])
            label = f"[{lower:g},{upper:g})" if math.isfinite(upper) else f"[{lower:g},inf)"
            bins[label] = {
                "coordinate_count": total,
                "realized_nonzero_fraction": nonzero / total if total else None,
                "energy_survival": _safe_ratio(realized_bin, intended_bin),
                "sign_flip_fraction": flips / total if total else None,
            }
        metrics["ulp_ratio_bins"] = bins


__all__ = [
    "ExactGeometryAccumulator",
    "PAIR_NAMES",
    "ULP_BIN_EDGES",
    "VECTOR_NAMES",
]
