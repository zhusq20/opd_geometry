"""Bounded-memory low-frequency coordinate distributions.

Full-model quantiles and top-energy statistics cannot be computed by
concatenating billions of coordinates.  This accumulator uses fixed log2
histograms, labels every derived approximation with ``_sketch``, and keeps
exact scalar sums/counts where they are sufficient (for example weighted CV
and Adam epsilon-threshold fractions).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

SPARSITY_VECTORS = (
    "delta_intended_fp32",
    "delta_model",
    "displacement",
)
RMS_RELATIVE_THRESHOLDS = (1.0e-3, 1.0e-2)
TOP_COORDINATE_FRACTIONS = (1.0e-3, 1.0e-2, 5.0e-2)

# Every finite nonzero FP32 magnitude lies in [2^-149, 2^128).  Four bins per
# octave keep the relative resolution fixed over the entire representable
# range without data-dependent edges.
LOG2_MIN = -149.0
LOG2_MAX = 128.0
BINS_PER_OCTAVE = 4
BIN_COUNT = int((LOG2_MAX - LOG2_MIN) * BINS_PER_OCTAVE)


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _local_flat(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if hasattr(value, "to_local"):
        value = value.to_local()
    if hasattr(value, "_local_tensor"):
        value = value._local_tensor
    return value.reshape(-1)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def _representative(bin_id: int) -> float:
    return 2.0 ** (LOG2_MIN + (bin_id + 0.5) / BINS_PER_OCTAVE)


class LowFrequencyHistogramAccumulator:
    """Accumulate fixed histogram sketches and exact sufficient statistics."""

    def __init__(
        self,
        group_names: Sequence[str],
        device: torch.device,
        *,
        chunk_size: int,
    ) -> None:
        self.group_names = tuple(group_names)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("Histogram chunk_size must be positive.")
        group_count = len(self.group_names)
        vector_count = len(SPARSITY_VECTORS)
        self._sparsity_index = {name: index for index, name in enumerate(SPARSITY_VECTORS)}

        self.sparsity_hist_count = torch.zeros(
            (group_count, vector_count, BIN_COUNT), dtype=torch.int64, device=device
        )
        self.sparsity_hist_energy = torch.zeros(
            (group_count, vector_count, BIN_COUNT), dtype=torch.float64, device=device
        )
        self.sparsity_zero_count = torch.zeros((group_count, vector_count), dtype=torch.int64, device=device)
        self.sparsity_total_count = torch.zeros_like(self.sparsity_zero_count)
        self.sparsity_sum_sq = torch.zeros((group_count, vector_count), dtype=torch.float64, device=device)
        self.sparsity_invalid_count = torch.zeros_like(self.sparsity_total_count)

        self.sqrt_v_hist_count = torch.zeros((group_count, BIN_COUNT), dtype=torch.int64, device=device)
        self.sqrt_v_zero_count = torch.zeros(group_count, dtype=torch.int64, device=device)
        self.sqrt_v_total_count = torch.zeros_like(self.sqrt_v_zero_count)
        self.sqrt_v_le_eps = torch.zeros_like(self.sqrt_v_zero_count)
        self.sqrt_v_le_10eps = torch.zeros_like(self.sqrt_v_zero_count)
        self.sqrt_v_invalid_count = torch.zeros_like(self.sqrt_v_zero_count)

        self.eta_hist_weight = torch.zeros((group_count, BIN_COUNT), dtype=torch.float64, device=device)
        self.eta_zero_weight = torch.zeros(group_count, dtype=torch.float64, device=device)
        # total weight, sum(w*x), sum(w*x^2)
        self.eta_moments = torch.zeros((group_count, 3), dtype=torch.float64, device=device)
        # non-finite eta, non-finite weight, negative weight
        self.eta_invalid_counts = torch.zeros((group_count, 3), dtype=torch.int64, device=device)

    @staticmethod
    def _ids(group_ids: Sequence[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(group_ids, dtype=torch.int64, device=device).reshape(-1)

    @staticmethod
    def _bins(nonzero_abs: torch.Tensor) -> torch.Tensor:
        indices = torch.floor((torch.log2(nonzero_abs) - LOG2_MIN) * BINS_PER_OCTAVE)
        return indices.to(torch.int64).clamp_(0, BIN_COUNT - 1)

    @torch.no_grad()
    def _unweighted_histogram(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        flat = _local_flat(value)
        counts = torch.zeros(BIN_COUNT, dtype=torch.int64, device=flat.device)
        energy = torch.zeros(BIN_COUNT, dtype=torch.float64, device=flat.device)
        zero_count = torch.zeros((), dtype=torch.int64, device=flat.device)
        invalid_count = torch.zeros((), dtype=torch.int64, device=flat.device)
        total_count = int(flat.numel())
        sum_sq = torch.zeros((), dtype=torch.float64, device=flat.device)
        for start in range(0, total_count, self.chunk_size):
            chunk = flat[start : start + self.chunk_size].to(torch.float32)
            finite = torch.isfinite(chunk)
            invalid_count.add_(torch.count_nonzero(~finite))
            chunk = torch.where(finite, chunk, torch.zeros_like(chunk))
            absolute = chunk.abs()
            zero_count.add_(torch.count_nonzero(absolute == 0))
            selected = absolute > 0
            nonzero = absolute[selected]
            squares = nonzero.square()
            sum_sq.add_(torch.sum(squares, dtype=torch.float64))
            if nonzero.numel():
                bins = self._bins(nonzero)
                counts.add_(torch.bincount(bins, minlength=BIN_COUNT))
                energy.add_(
                    torch.bincount(
                        bins,
                        weights=squares.to(torch.float64),
                        minlength=BIN_COUNT,
                    )
                )
        return counts, energy, zero_count, total_count, sum_sq, invalid_count

    @torch.no_grad()
    def add_sparsity(
        self,
        group_ids: Sequence[int] | torch.Tensor,
        vectors: Mapping[str, torch.Tensor],
    ) -> None:
        unknown = set(vectors).difference(self._sparsity_index)
        if unknown:
            raise KeyError(f"Unknown sparsity vectors: {sorted(unknown)}")
        for name, value in vectors.items():
            ids = self._ids(group_ids, value.device)
            if ids.numel() == 0:
                continue
            counts, energy, zeros, total, sum_sq, invalid = self._unweighted_histogram(value)
            vector_id = self._sparsity_index[name]
            repeats = ids.numel()
            self.sparsity_hist_count[:, vector_id].index_add_(0, ids, counts.expand(repeats, -1))
            self.sparsity_hist_energy[:, vector_id].index_add_(0, ids, energy.expand(repeats, -1))
            self.sparsity_zero_count[:, vector_id].index_add_(0, ids, zeros.expand(repeats))
            self.sparsity_total_count[:, vector_id].index_add_(
                0,
                ids,
                torch.full((repeats,), total, dtype=torch.int64, device=value.device),
            )
            self.sparsity_sum_sq[:, vector_id].index_add_(0, ids, sum_sq.expand(repeats))
            self.sparsity_invalid_count[:, vector_id].index_add_(0, ids, invalid.expand(repeats))

    @torch.no_grad()
    def add_adam(
        self,
        group_ids: Sequence[int] | torch.Tensor,
        *,
        sqrt_v_hat: torch.Tensor,
        effective_eta: torch.Tensor,
        gradient_energy: torch.Tensor,
        eps: float,
    ) -> None:
        sqrt_flat = _local_flat(sqrt_v_hat)
        eta_flat = _local_flat(effective_eta)
        weight_flat = _local_flat(gradient_energy)
        if not (sqrt_flat.numel() == eta_flat.numel() == weight_flat.numel()):
            raise ValueError("Adam distribution tensors must have equal length.")
        ids = self._ids(group_ids, sqrt_flat.device)
        if ids.numel() == 0:
            return

        sqrt_counts, _, sqrt_zeros, total, _, sqrt_invalid = self._unweighted_histogram(sqrt_flat)
        le_eps = torch.zeros((), dtype=torch.int64, device=sqrt_flat.device)
        le_10eps = torch.zeros((), dtype=torch.int64, device=sqrt_flat.device)
        eta_hist = torch.zeros(BIN_COUNT, dtype=torch.float64, device=sqrt_flat.device)
        eta_zero_weight = torch.zeros((), dtype=torch.float64, device=sqrt_flat.device)
        moments = torch.zeros(3, dtype=torch.float64, device=sqrt_flat.device)
        invalid_counts = torch.zeros(3, dtype=torch.int64, device=sqrt_flat.device)
        for start in range(0, total, self.chunk_size):
            sqrt_chunk = sqrt_flat[start : start + self.chunk_size].to(torch.float32)
            eta_chunk = eta_flat[start : start + self.chunk_size].to(torch.float32)
            weights = weight_flat[start : start + self.chunk_size].to(torch.float32)
            finite_sqrt = torch.isfinite(sqrt_chunk)
            sqrt_chunk = torch.where(finite_sqrt, sqrt_chunk, torch.zeros_like(sqrt_chunk))
            finite_eta = torch.isfinite(eta_chunk)
            finite_weights = torch.isfinite(weights)
            invalid_counts[0].add_(torch.count_nonzero(~finite_eta))
            invalid_counts[1].add_(torch.count_nonzero(~finite_weights))
            invalid_counts[2].add_(torch.count_nonzero(finite_weights & (weights < 0)))
            eta_chunk = torch.where(finite_eta, eta_chunk, torch.zeros_like(eta_chunk))
            weights = torch.where(finite_weights & (weights >= 0), weights, torch.zeros_like(weights))
            le_eps.add_(torch.count_nonzero(sqrt_chunk <= eps))
            le_10eps.add_(torch.count_nonzero(sqrt_chunk <= 10.0 * eps))
            weights64 = weights.to(torch.float64)
            eta64 = eta_chunk.to(torch.float64)
            moments[0].add_(weights64.sum())
            moments[1].add_(torch.sum(weights64 * eta64, dtype=torch.float64))
            moments[2].add_(torch.sum(weights64 * eta64.square(), dtype=torch.float64))
            positive = eta_chunk > 0
            eta_zero_weight.add_(weights64[~positive].sum())
            bins = self._bins(eta_chunk[positive])
            eta_hist.add_(
                torch.bincount(
                    bins,
                    weights=weights64[positive],
                    minlength=BIN_COUNT,
                )
            )

        repeats = ids.numel()
        self.sqrt_v_hist_count.index_add_(0, ids, sqrt_counts.expand(repeats, -1))
        for destination, value in (
            (self.sqrt_v_zero_count, sqrt_zeros),
            (self.sqrt_v_le_eps, le_eps),
            (self.sqrt_v_le_10eps, le_10eps),
        ):
            destination.index_add_(0, ids, value.expand(repeats))
        self.sqrt_v_total_count.index_add_(
            0,
            ids,
            torch.full((repeats,), total, dtype=torch.int64, device=sqrt_flat.device),
        )
        self.sqrt_v_invalid_count.index_add_(0, ids, sqrt_invalid.expand(repeats))
        self.eta_hist_weight.index_add_(0, ids, eta_hist.expand(repeats, -1))
        self.eta_zero_weight.index_add_(0, ids, eta_zero_weight.expand(repeats))
        self.eta_moments.index_add_(0, ids, moments.expand(repeats, -1))
        self.eta_invalid_counts.index_add_(0, ids, invalid_counts.expand(repeats, -1))

    def _reduced(self) -> tuple[torch.Tensor, ...]:
        values = (
            self.sparsity_hist_count.clone(),
            self.sparsity_hist_energy.clone(),
            self.sparsity_zero_count.clone(),
            self.sparsity_total_count.clone(),
            self.sparsity_sum_sq.clone(),
            self.sparsity_invalid_count.clone(),
            self.sqrt_v_hist_count.clone(),
            self.sqrt_v_zero_count.clone(),
            self.sqrt_v_total_count.clone(),
            self.sqrt_v_le_eps.clone(),
            self.sqrt_v_le_10eps.clone(),
            self.sqrt_v_invalid_count.clone(),
            self.eta_hist_weight.clone(),
            self.eta_zero_weight.clone(),
            self.eta_moments.clone(),
            self.eta_invalid_counts.clone(),
        )
        if _distributed():
            sizes = [value.numel() for value in values]
            packed = torch.cat([value.reshape(-1).to(torch.float64) for value in values])
            torch.distributed.all_reduce(packed)
            reduced: list[torch.Tensor] = []
            offset = 0
            for original, size in zip(values, sizes, strict=True):
                value = packed[offset : offset + size].reshape(original.shape)
                if not original.is_floating_point():
                    value = value.round()
                reduced.append(value.to(original.dtype))
                offset += size
            values = tuple(reduced)
        return values

    @staticmethod
    def _hist_quantile(counts: torch.Tensor, zero_count: int, total: int, quantile: float) -> float | None:
        if total == 0:
            return None
        target = quantile * max(total - 1, 0) + 1.0
        if target <= zero_count:
            return 0.0
        cumulative = float(zero_count)
        for bin_id, count in enumerate(counts.tolist()):
            cumulative += int(count)
            if cumulative >= target:
                return _representative(bin_id)
        return _representative(BIN_COUNT - 1)

    @staticmethod
    def _weighted_hist_quantile(
        weights: torch.Tensor,
        zero_weight: float,
        total_weight: float,
        quantile: float,
    ) -> float | None:
        if total_weight == 0.0:
            return None
        target = quantile * total_weight
        if target <= zero_weight:
            return 0.0
        cumulative = zero_weight
        for bin_id, weight in enumerate(weights.tolist()):
            cumulative += float(weight)
            if cumulative >= target:
                return _representative(bin_id)
        return _representative(BIN_COUNT - 1)

    @staticmethod
    def _top_energy_fraction(
        counts: torch.Tensor,
        energy: torch.Tensor,
        total_count: int,
        total_energy: float,
        fraction: float,
    ) -> float | None:
        if total_count == 0 or total_energy == 0.0:
            return None
        remaining = max(1, math.ceil(fraction * total_count))
        selected_energy = 0.0
        for count, bin_energy in zip(reversed(counts.tolist()), reversed(energy.tolist()), strict=True):
            count = int(count)
            if count == 0:
                continue
            take = min(remaining, count)
            selected_energy += float(bin_energy) * (take / count)
            remaining -= take
            if remaining == 0:
                break
        return selected_energy / total_energy

    @torch.no_grad()
    def finalize(self) -> dict[str, dict[str, Any]]:
        (
            sparsity_count,
            sparsity_energy,
            sparsity_zero,
            sparsity_total,
            sparsity_sum_sq,
            sparsity_invalid,
            sqrt_count,
            sqrt_zero,
            sqrt_total,
            sqrt_le_eps,
            sqrt_le_10eps,
            sqrt_invalid,
            eta_weight,
            eta_zero_weight,
            eta_moments,
            eta_invalid,
        ) = (value.cpu() for value in self._reduced())

        if int(sparsity_invalid.sum()) or int(sqrt_invalid.sum()):
            raise ValueError("Geometry histograms received non-finite coordinates.")
        if int(eta_invalid[:, :2].sum()):
            raise ValueError("Adam geometry distributions require finite eta and gradient energy.")
        if int(eta_invalid[:, 2].sum()):
            raise ValueError("Adam gradient-energy weights must be non-negative.")

        output: dict[str, dict[str, Any]] = {}
        for group_id, group_name in enumerate(self.group_names):
            metrics: dict[str, Any] = {}
            for vector_id, vector_name in enumerate(SPARSITY_VECTORS):
                total = int(sparsity_total[group_id, vector_id])
                if total == 0:
                    continue
                square_sum = float(sparsity_sum_sq[group_id, vector_id])
                rms = math.sqrt(max(square_sum / total, 0.0))
                counts = sparsity_count[group_id, vector_id]
                energies = sparsity_energy[group_id, vector_id]
                zeros = int(sparsity_zero[group_id, vector_id])
                for threshold in RMS_RELATIVE_THRESHOLDS:
                    absolute_threshold = threshold * rms
                    approximate_count = zeros
                    for bin_id, count in enumerate(counts.tolist()):
                        if _representative(bin_id) <= absolute_threshold:
                            approximate_count += int(count)
                    metrics[f"{vector_name}_le_{threshold:g}_rms_fraction_sketch"] = approximate_count / total
                for fraction in TOP_COORDINATE_FRACTIONS:
                    metrics[f"{vector_name}_top_{100.0 * fraction:g}pct_coordinate_energy_fraction_sketch"] = (
                        self._top_energy_fraction(
                            counts,
                            energies,
                            total,
                            square_sum,
                            fraction,
                        )
                    )

            adam_total = int(sqrt_total[group_id])
            if adam_total:
                sqrt_quantiles = {
                    name: self._hist_quantile(sqrt_count[group_id], int(sqrt_zero[group_id]), adam_total, quantile)
                    for quantile, name in ((0.01, "p1"), (0.50, "p50"), (0.99, "p99"))
                }
                for name, value in sqrt_quantiles.items():
                    metrics[f"sqrt_v_hat_{name}_sketch"] = value
                p1 = sqrt_quantiles["p1"]
                p99 = sqrt_quantiles["p99"]
                metrics["sqrt_v_hat_p99_to_p1_ratio_sketch"] = (
                    _safe_ratio(float(p99), float(p1)) if p1 is not None and p99 is not None else None
                )
                metrics["sqrt_v_hat_le_eps_fraction"] = int(sqrt_le_eps[group_id]) / adam_total
                metrics["sqrt_v_hat_le_10eps_fraction"] = int(sqrt_le_10eps[group_id]) / adam_total

                total_weight, weighted_sum, weighted_square_sum = (float(value) for value in eta_moments[group_id])
                for quantile, name in ((0.01, "p1"), (0.50, "p50"), (0.99, "p99")):
                    metrics[f"effective_eta_gradient_energy_weighted_{name}_sketch"] = self._weighted_hist_quantile(
                        eta_weight[group_id],
                        float(eta_zero_weight[group_id]),
                        total_weight,
                        quantile,
                    )
                if total_weight != 0.0:
                    mean = weighted_sum / total_weight
                    variance = max(weighted_square_sum / total_weight - mean * mean, 0.0)
                    metrics["effective_eta_gradient_energy_weighted_mean"] = mean
                    metrics["effective_eta_gradient_energy_weighted_cv"] = (
                        math.sqrt(variance) / mean if mean != 0.0 else None
                    )
            if metrics:
                output[group_name] = metrics
        return output


__all__ = [
    "BIN_COUNT",
    "BINS_PER_OCTAVE",
    "LOG2_MAX",
    "LOG2_MIN",
    "LowFrequencyHistogramAccumulator",
    "RMS_RELATIVE_THRESHOLDS",
    "SPARSITY_VECTORS",
    "TOP_COORDINATE_FRACTIONS",
]
