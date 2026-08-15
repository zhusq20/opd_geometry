#!/usr/bin/env python3
"""Compare exact matrix spectra between a reference and torch-dist checkpoints.

The reader loads one distributed-checkpoint chunk at a time.  This matters for
training checkpoints that also contain optimizer state and can be tens of GB,
even though the model itself is much smaller.

The current semantic split follows the Megatron Qwen layout used by the
optimizer-geometry experiments: interleaved grouped QKV rows and concatenated
gate/up rows.  The analysis covers q, k, v, o, gate, up, and down projection
matrices in every selected transformer layer.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pickle
import re
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


LINEAR_PROJ = "decoder.layers.self_attention.linear_proj.weight"
LINEAR_QKV = "decoder.layers.self_attention.linear_qkv.weight"
LINEAR_FC1 = "decoder.layers.mlp.linear_fc1.weight"
LINEAR_FC2 = "decoder.layers.mlp.linear_fc2.weight"
OPERATORS = ("q", "k", "v", "o", "gate", "up", "down")


@dataclass(frozen=True)
class ModelShape:
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    head_dim: int


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    iteration: int | None


class DistributedCheckpointReader:
    """Read individual tensor chunks without materializing the full checkpoint."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        metadata_path = self.root / ".metadata"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing torch-dist metadata: {metadata_path}")
        with metadata_path.open("rb") as stream:
            # Checkpoint metadata is trusted experiment output and uses the
            # native torch distributed-checkpoint pickle format.
            self.metadata = pickle.load(stream)  # noqa: S301
        self._entries: dict[tuple[str, int], list[tuple[Any, Any]]] = defaultdict(list)
        for index, storage in self.metadata.storage_data.items():
            offsets = getattr(index, "offset", None)
            if offsets is None or len(offsets) != 3:
                continue
            self._entries[(index.fqn, int(offsets[0]))].append((index, storage))
        self._stack = ExitStack()
        self._streams: dict[str, Any] = {}

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> DistributedCheckpointReader:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def load_layer_parameter(self, name: str, layer: int) -> torch.Tensor:
        entries = self._entries.get((name, layer), [])
        if not entries:
            raise KeyError(f"Checkpoint {self.root} has no layer {layer} chunk for {name}.")

        chunks: list[tuple[int, torch.Tensor]] = []
        for index, storage in entries:
            descriptors = getattr(storage, "transform_descriptors", None)
            if descriptors:
                raise ValueError(f"Unsupported transformed checkpoint chunk for {name}: {descriptors}")
            relative_path = str(storage.relative_path)
            stream = self._streams.get(relative_path)
            if stream is None:
                stream = self._stack.enter_context((self.root / relative_path).open("rb"))
                self._streams[relative_path] = stream
            stream.seek(int(storage.offset))
            payload = stream.read(int(storage.length))
            if len(payload) != int(storage.length):
                raise IOError(
                    f"Short read for {name} in {relative_path}: "
                    f"expected {storage.length}, received {len(payload)} bytes."
                )
            tensor = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
                raise TypeError(f"Unexpected checkpoint payload for {name}: {type(tensor)!r}, shape={tensor.shape}")
            offsets = tuple(int(value) for value in index.offset)
            if offsets[2] != 0:
                raise ValueError(f"Column-sharded {name} is not supported: offsets={offsets}.")
            chunks.append((offsets[1], tensor.squeeze(0)))

        chunks.sort(key=lambda item: item[0])
        expected_row = 0
        for row_offset, tensor in chunks:
            if row_offset != expected_row:
                raise ValueError(
                    f"Non-contiguous chunks for {name}, layer {layer}: "
                    f"expected row {expected_row}, found {row_offset}."
                )
            expected_row += int(tensor.shape[0])
        parameter = chunks[0][1] if len(chunks) == 1 else torch.cat([tensor for _, tensor in chunks], dim=0)
        expected = self.metadata.state_dict_metadata[name].size
        if tuple(parameter.shape) != (int(expected[1]), int(expected[2])):
            raise ValueError(
                f"Reconstructed {name}, layer {layer} has shape {tuple(parameter.shape)}; "
                f"expected {(int(expected[1]), int(expected[2]))}."
            )
        return parameter


def load_model_shape(checkpoint: Path) -> ModelShape:
    common_path = checkpoint / "common.pt"
    if not common_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint configuration: {common_path}")
    payload = torch.load(common_path, map_location="cpu", weights_only=False)  # noqa: S614
    args = payload["args"]
    hidden_size = int(args.hidden_size)
    num_attention_heads = int(args.num_attention_heads)
    num_query_groups = int(args.num_query_groups)
    head_dim = int(args.kv_channels or hidden_size // num_attention_heads)
    return ModelShape(
        num_layers=int(args.num_layers),
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        head_dim=head_dim,
    )


def projection_matrices(
    reader: DistributedCheckpointReader,
    shape: ModelShape,
    layer: int,
    selected_operators: set[str],
) -> dict[str, torch.Tensor]:
    matrices: dict[str, torch.Tensor] = {}
    if "o" in selected_operators:
        matrices["o"] = reader.load_layer_parameter(LINEAR_PROJ, layer)

    if selected_operators.intersection({"q", "k", "v"}):
        qkv = reader.load_layer_parameter(LINEAR_QKV, layer)
        queries_per_group = shape.num_attention_heads // shape.num_query_groups
        expected_rows = shape.num_query_groups * (queries_per_group + 2) * shape.head_dim
        if qkv.shape != (expected_rows, shape.hidden_size):
            raise ValueError(
                f"Unexpected QKV shape {tuple(qkv.shape)} for layer {layer}; "
                f"expected {(expected_rows, shape.hidden_size)}."
            )
        grouped = qkv.reshape(
            shape.num_query_groups,
            queries_per_group + 2,
            shape.head_dim,
            shape.hidden_size,
        )
        if "q" in selected_operators:
            matrices["q"] = grouped[:, :queries_per_group].reshape(-1, shape.hidden_size)
        if "k" in selected_operators:
            matrices["k"] = grouped[:, queries_per_group].reshape(-1, shape.hidden_size)
        if "v" in selected_operators:
            matrices["v"] = grouped[:, queries_per_group + 1].reshape(-1, shape.hidden_size)

    if selected_operators.intersection({"gate", "up"}):
        fc1 = reader.load_layer_parameter(LINEAR_FC1, layer)
        if fc1.shape[0] % 2:
            raise ValueError(f"Cannot split odd FC1 row count {fc1.shape[0]} in layer {layer}.")
        gate, up = fc1.chunk(2, dim=0)
        if "gate" in selected_operators:
            matrices["gate"] = gate
        if "up" in selected_operators:
            matrices["up"] = up

    if "down" in selected_operators:
        matrices["down"] = reader.load_layer_parameter(LINEAR_FC2, layer)
    return matrices


def tensor_square_sum(value: torch.Tensor) -> float:
    return float(torch.sum(value.to(torch.float32).square(), dtype=torch.float64))


def quantiles(value: torch.Tensor, probabilities: tuple[float, ...]) -> list[float]:
    points = torch.tensor(probabilities, device=value.device, dtype=value.dtype)
    return [float(item) for item in torch.quantile(value, points).cpu()]


def stable_svd(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the accurate CUDA QR driver; Jacobi has a visible floor for tiny deltas."""

    if value.device.type == "cuda":
        return torch.linalg.svd(value, full_matrices=False, driver="gesvd")
    return torch.linalg.svd(value, full_matrices=False)


def stable_svdvals(value: torch.Tensor) -> torch.Tensor:
    if value.device.type == "cuda":
        return torch.linalg.svdvals(value, driver="gesvd")
    return torch.linalg.svdvals(value)


def principal_angle_summary(
    reference_vectors: torch.Tensor,
    trained_vectors: torch.Tensor,
    ranks: list[int],
    prefix: str,
) -> tuple[dict[str, float], dict[int, np.ndarray]]:
    metrics: dict[str, float] = {}
    spectra: dict[int, np.ndarray] = {}
    available = min(reference_vectors.shape[1], trained_vectors.shape[1])
    for requested_rank in ranks:
        rank = min(requested_rank, available)
        if rank <= 0 or rank in spectra:
            continue
        # Small principal angles are sensitive to cancellation in V0^T V1.
        # Use FP64 for the reported top-k blocks; keep full-width diagnostics
        # in FP32 so a 2k x 2k auxiliary SVD does not dominate the analysis.
        overlap_dtype = torch.float64 if rank <= 256 else torch.float32
        reference_block = reference_vectors[:, :rank].to(overlap_dtype)
        trained_block = trained_vectors[:, :rank].to(overlap_dtype)
        overlap = reference_block.transpose(0, 1) @ trained_block
        cosines = stable_svdvals(overlap).clamp_(0.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosines))
        median, p95 = quantiles(angles, (0.50, 0.95))
        key = f"{prefix}_top{rank}_principal_angle"
        metrics[f"{key}_mean_deg"] = float(angles.mean())
        metrics[f"{key}_median_deg"] = median
        metrics[f"{key}_p95_deg"] = p95
        metrics[f"{key}_max_deg"] = float(angles.max())
        metrics[f"{key}_rms_sin"] = float(torch.sqrt(torch.mean(torch.sin(torch.deg2rad(angles)).square())))
        spectra[rank] = angles.cpu().numpy()
    return metrics, spectra


def compare_matrix(
    reference: torch.Tensor,
    trained: torch.Tensor,
    reference_svd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    svd_dtype: torch.dtype,
    subspace_ranks: list[int],
) -> tuple[dict[str, Any], torch.Tensor, dict[str, np.ndarray]]:
    if reference.shape != trained.shape:
        raise ValueError(f"Matrix shape changed from {tuple(reference.shape)} to {tuple(trained.shape)}.")

    reference32 = reference.to(torch.float32)
    trained32 = trained.to(torch.float32)
    delta = trained32 - reference32
    reference_sq = tensor_square_sum(reference32)
    trained_sq = tensor_square_sum(trained32)
    delta_sq = tensor_square_sum(delta)
    reference_norm = math.sqrt(reference_sq)
    trained_norm = math.sqrt(trained_sq)
    delta_norm = math.sqrt(delta_sq)

    reference_u, reference_s, reference_vh = reference_svd
    trained_u, trained_s, trained_vh = stable_svd(trained32.to(device=device, dtype=svd_dtype))
    singular_delta = trained_s - reference_s
    singular_delta_sq = float(torch.sum(singular_delta.square(), dtype=torch.float64))
    singular_delta_norm = math.sqrt(singular_delta_sq)
    ratio = singular_delta_norm / delta_norm if delta_norm else 0.0
    if ratio > 1.001:
        raise RuntimeError(
            f"Computed singular-value displacement exceeds the weight displacement: {ratio:.6f}. "
            "This violates Mirsky's inequality and indicates insufficient SVD accuracy."
        )
    ratio = min(max(ratio, 0.0), 1.0)
    absolute_singular_delta = singular_delta.abs()
    median_delta, p95_delta = quantiles(absolute_singular_delta, (0.50, 0.95))
    singular_cosine = float(
        torch.dot(reference_s.to(torch.float64), trained_s.to(torch.float64))
        / (
            torch.linalg.vector_norm(reference_s.to(torch.float64))
            * torch.linalg.vector_norm(trained_s.to(torch.float64))
        )
    )

    metrics: dict[str, Any] = {
        "rows": int(reference.shape[0]),
        "columns": int(reference.shape[1]),
        "parameter_count": int(reference.numel()),
        "changed_coordinate_count": int(torch.count_nonzero(delta)),
        "changed_coordinate_fraction": float(torch.count_nonzero(delta) / delta.numel()),
        "weight_fro_initial": reference_norm,
        "weight_fro_trained": trained_norm,
        "weight_norm_relative_change": (trained_norm - reference_norm) / reference_norm,
        "weight_delta_fro": delta_norm,
        "relative_weight_change": delta_norm / reference_norm,
        "singular_delta_l2": singular_delta_norm,
        "relative_singular_change": singular_delta_norm / reference_norm,
        "singular_to_weight_change_ratio": ratio,
        "rotation_compatible_displacement_energy_fraction": 1.0 - ratio * ratio,
        "singular_cosine": singular_cosine,
        "top_singular_initial": float(reference_s[0]),
        "top_singular_trained": float(trained_s[0]),
        "top_singular_relative_change": float((trained_s[0] - reference_s[0]) / reference_s[0]),
        "median_abs_singular_change": median_delta,
        "p95_abs_singular_change": p95_delta,
        "max_abs_singular_change": float(absolute_singular_delta.max()),
        "stable_rank_initial": reference_sq / float(reference_s[0].square()),
        "stable_rank_trained": trained_sq / float(trained_s[0].square()),
    }

    left_metrics, left_spectra = principal_angle_summary(reference_u, trained_u, subspace_ranks, "left")
    right_metrics, right_spectra = principal_angle_summary(
        reference_vh.transpose(0, 1),
        trained_vh.transpose(0, 1),
        subspace_ranks,
        "right",
    )
    metrics.update(left_metrics)
    metrics.update(right_metrics)
    angle_spectra = {
        **{f"left_top{rank}_angles_deg": values for rank, values in left_spectra.items()},
        **{f"right_top{rank}_angles_deg": values for rank, values in right_spectra.items()},
    }

    rows, columns = reference.shape
    if rows > columns:
        full_metrics, full_spectra = principal_angle_summary(
            reference_u,
            trained_u,
            [columns],
            "left_nontrivial_full",
        )
        metrics.update(full_metrics)
        angle_spectra["left_nontrivial_full_angles_deg"] = full_spectra[columns]
    elif columns > rows:
        full_metrics, full_spectra = principal_angle_summary(
            reference_vh.transpose(0, 1),
            trained_vh.transpose(0, 1),
            [rows],
            "right_nontrivial_full",
        )
        metrics.update(full_metrics)
        angle_spectra["right_nontrivial_full_angles_deg"] = full_spectra[rows]

    return metrics, trained_s.cpu(), angle_spectra


def parse_checkpoint(value: str) -> CheckpointSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoints must use LABEL=PATH syntax.")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Checkpoints must use a non-empty LABEL=PATH pair.")
    path = Path(raw_path)
    match = re.fullmatch(r"iter_(\d+)", path.name)
    return CheckpointSpec(label=label, path=path, iteration=int(match.group(1)) if match else None)


def parse_integer_selection(value: str, upper_bound: int, name: str) -> list[int]:
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            raw_start, raw_stop = item.split("-", 1)
            start, stop = int(raw_start), int(raw_stop)
            if stop < start:
                raise ValueError(f"Invalid {name} range {item!r}.")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(item))
    invalid = sorted(value for value in selected if value < 0 or value >= upper_bound)
    if invalid:
        raise ValueError(f"{name.capitalize()} outside [0, {upper_bound - 1}]: {invalid}")
    if not selected:
        raise ValueError(f"At least one {name} must be selected.")
    return sorted(selected)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["checkpoint"]), "global")].append(row)
        grouped[(str(row["checkpoint"]), f"operator/{row['operator']}")].append(row)

    output: list[dict[str, Any]] = []
    for (checkpoint, scope), values in grouped.items():
        reference_sq = sum(float(row["weight_fro_initial"]) ** 2 for row in values)
        trained_sq = sum(float(row["weight_fro_trained"]) ** 2 for row in values)
        delta_sq = sum(float(row["weight_delta_fro"]) ** 2 for row in values)
        singular_delta_sq = sum(float(row["singular_delta_l2"]) ** 2 for row in values)
        reference_norm = math.sqrt(reference_sq)
        trained_norm = math.sqrt(trained_sq)
        delta_norm = math.sqrt(delta_sq)
        singular_delta_norm = math.sqrt(singular_delta_sq)
        ratio = singular_delta_norm / delta_norm if delta_norm else 0.0
        ratios = torch.tensor([float(row["singular_to_weight_change_ratio"]) for row in values])
        relative_weight_changes = torch.tensor([float(row["relative_weight_change"]) for row in values])
        relative_singular_changes = torch.tensor([float(row["relative_singular_change"]) for row in values])
        ratio_q25, ratio_median, ratio_q75 = quantiles(ratios, (0.25, 0.50, 0.75))
        output.append(
            {
                "checkpoint": checkpoint,
                "iteration": values[0]["iteration"],
                "scope": scope,
                "matrix_count": len(values),
                "parameter_count": sum(int(row["parameter_count"]) for row in values),
                "changed_coordinate_fraction": sum(int(row["changed_coordinate_count"]) for row in values)
                / sum(int(row["parameter_count"]) for row in values),
                "weight_fro_initial": reference_norm,
                "weight_fro_trained": trained_norm,
                "weight_norm_relative_change": (trained_norm - reference_norm) / reference_norm,
                "weight_delta_fro": delta_norm,
                "relative_weight_change": delta_norm / reference_norm,
                "singular_delta_l2": singular_delta_norm,
                "relative_singular_change": singular_delta_norm / reference_norm,
                "singular_to_weight_change_ratio": ratio,
                "rotation_compatible_displacement_energy_fraction": 1.0 - min(ratio, 1.0) ** 2,
                "matrix_median_singular_to_weight_change_ratio": ratio_median,
                "matrix_iqr_singular_to_weight_change_ratio": ratio_q75 - ratio_q25,
                "matrix_median_relative_weight_change": float(relative_weight_changes.median()),
                "matrix_median_relative_singular_change": float(relative_singular_changes.median()),
            }
        )
    return output


def save_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    spectra: dict[str, np.ndarray],
    checkpoints: list[CheckpointSpec],
    paper_layer: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global_rows = {row["checkpoint"]: row for row in aggregates if row["scope"] == "global"}
    labels = [checkpoint.label for checkpoint in checkpoints]
    x = np.arange(len(labels))
    figure, left = plt.subplots(figsize=(8.0, 4.8))
    left.plot(x, [global_rows[label]["relative_weight_change"] for label in labels], "o-", label="weight change")
    left.plot(
        x,
        [global_rows[label]["relative_singular_change"] for label in labels],
        "s-",
        label="singular-value change",
    )
    left.set_yscale("log")
    left.set_ylabel("relative Frobenius change")
    left.set_xticks(x, labels, rotation=25, ha="right")
    left.grid(alpha=0.25)
    right = left.twinx()
    right.plot(
        x,
        [100.0 * global_rows[label]["singular_to_weight_change_ratio"] for label in labels],
        "^-",
        color="tab:green",
        label="spectral / raw displacement",
    )
    right.set_ylabel("spectral / raw displacement (%)")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "global_trajectory.png", dpi=180)
    plt.close(figure)

    final_label = labels[-1]
    operator_rows = {
        row["scope"].removeprefix("operator/"): row
        for row in aggregates
        if row["checkpoint"] == final_label and row["scope"].startswith("operator/")
    }
    operators = [operator for operator in OPERATORS if operator in operator_rows]
    x = np.arange(len(operators))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(
        x - width / 2,
        [operator_rows[operator]["relative_weight_change"] for operator in operators],
        width,
        label="weight change",
    )
    axis.bar(
        x + width / 2,
        [operator_rows[operator]["relative_singular_change"] for operator in operators],
        width,
        label="singular-value change",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, operators)
    axis.set_ylabel("relative Frobenius change")
    axis.set_title(final_label)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "final_operator_changes.png", dpi=180)
    plt.close(figure)

    final_rows = [row for row in rows if row["checkpoint"] == final_label]
    layers = sorted({int(row["layer"]) for row in final_rows})
    heatmap = np.full((len(layers), len(operators)), np.nan)
    layer_index = {layer: index for index, layer in enumerate(layers)}
    operator_index = {operator: index for index, operator in enumerate(operators)}
    for row in final_rows:
        heatmap[layer_index[int(row["layer"])], operator_index[str(row["operator"])]] = 100.0 * float(
            row["singular_to_weight_change_ratio"]
        )
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    image = axis.imshow(heatmap, aspect="auto", vmin=0.0, vmax=float(np.nanmax(heatmap)) * 1.05, cmap="viridis")
    axis.set_xticks(np.arange(len(operators)), operators)
    axis.set_yticks(np.arange(len(layers)), layers)
    axis.set_xlabel("projection matrix")
    axis.set_ylabel("transformer layer")
    axis.set_title(f"spectral / raw displacement: {final_label}")
    figure.colorbar(image, ax=axis, label="spectral / raw displacement (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "final_layer_operator_ratio.png", dpi=180)
    plt.close(figure)

    if paper_layer in layers and all(operator in operators for operator in ("q", "k", "v")):
        figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
        for axis, operator in zip(axes, ("q", "k", "v"), strict=True):
            reference_key = f"reference__layer_{paper_layer:02d}__{operator}__singular_values"
            reference_values = spectra[reference_key]
            for checkpoint in checkpoints:
                trained_key = f"{checkpoint.label}__layer_{paper_layer:02d}__{operator}__singular_values"
                delta = spectra[trained_key] - reference_values
                axis.plot(np.arange(delta.size), delta, label=checkpoint.label, linewidth=0.9)
            axis.axhline(0.0, color="black", linewidth=0.6)
            axis.set_title(f"layer {paper_layer} {operator}_proj")
            axis.set_xlabel("singular-value index")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("trained - initial singular value")
        axes[-1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"paper_style_layer{paper_layer}_qkv_delta.png", dpi=180)
        plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    reference_path = args.reference.resolve()
    checkpoints = [
        CheckpointSpec(spec.label, spec.path.resolve(), spec.iteration) for spec in args.checkpoint
    ]
    shape = load_model_shape(reference_path)
    for checkpoint in checkpoints:
        checkpoint_shape = load_model_shape(checkpoint.path)
        if checkpoint_shape != shape:
            raise ValueError(
                f"Model shape mismatch for {checkpoint.label}: {checkpoint_shape} != reference {shape}."
            )

    layers = (
        list(range(shape.num_layers))
        if args.layers is None
        else parse_integer_selection(args.layers, shape.num_layers, "layer")
    )
    operators = set(OPERATORS if args.operators is None else args.operators.split(","))
    invalid_operators = sorted(operators.difference(OPERATORS))
    if invalid_operators:
        raise ValueError(f"Unknown operators: {invalid_operators}; choose from {OPERATORS}.")
    subspace_ranks = sorted({int(value) for value in args.subspace_ranks.split(",") if value})
    if any(rank <= 0 for rank in subspace_ranks):
        raise ValueError("Subspace ranks must be positive integers.")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --force to overwrite outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    svd_dtype = {"float32": torch.float32, "float64": torch.float64}[args.svd_dtype]

    rows: list[dict[str, Any]] = []
    spectra: dict[str, np.ndarray] = {}
    with DistributedCheckpointReader(reference_path) as reference_reader, ExitStack() as stack:
        trained_readers = {
            checkpoint.label: stack.enter_context(DistributedCheckpointReader(checkpoint.path))
            for checkpoint in checkpoints
        }
        for layer_index, layer in enumerate(layers, start=1):
            print(f"[{layer_index}/{len(layers)}] loading reference layer {layer}", flush=True)
            reference_matrices = projection_matrices(reference_reader, shape, layer, operators)
            reference_decompositions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for operator, matrix in reference_matrices.items():
                decomposition = stable_svd(matrix.to(device=device, dtype=svd_dtype))
                reference_decompositions[operator] = decomposition
                spectra[f"reference__layer_{layer:02d}__{operator}__singular_values"] = decomposition[1].cpu().numpy()

            for checkpoint in checkpoints:
                print(f"[{layer_index}/{len(layers)}] comparing {checkpoint.label} layer {layer}", flush=True)
                trained_matrices = projection_matrices(trained_readers[checkpoint.label], shape, layer, operators)
                for operator in OPERATORS:
                    if operator not in operators:
                        continue
                    metrics, trained_s, angle_spectra = compare_matrix(
                        reference_matrices[operator],
                        trained_matrices[operator],
                        reference_decompositions[operator],
                        device,
                        svd_dtype,
                        subspace_ranks,
                    )
                    rows.append(
                        {
                            "checkpoint": checkpoint.label,
                            "checkpoint_path": str(checkpoint.path),
                            "iteration": checkpoint.iteration,
                            "layer": layer,
                            "operator": operator,
                            **metrics,
                        }
                    )
                    prefix = f"{checkpoint.label}__layer_{layer:02d}__{operator}"
                    spectra[f"{prefix}__singular_values"] = trained_s.numpy()
                    for name, values in angle_spectra.items():
                        spectra[f"{prefix}__{name}"] = values
                del trained_matrices
            del reference_matrices, reference_decompositions
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregates = aggregate_rows(rows)
    write_csv(output_dir / "matrix_metrics.csv", rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregates)
    np.savez_compressed(output_dir / "spectra_and_angles.npz", **spectra)
    summary = {
        "schema_version": 1,
        "method": {
            "svd": (
                f"exact torch.linalg.svd in {args.svd_dtype} on realized checkpoint matrices; "
                "CUDA QR driver gesvd"
            ),
            "spectral_orbit_distance": "L2 distance between ordered singular-value vectors",
            "rotation_compatible_displacement_energy_fraction": "1 - ||delta_sigma||_2^2 / ||delta_W||_F^2",
        },
        "reference": str(reference_path),
        "checkpoints": [
            {"label": checkpoint.label, "path": str(checkpoint.path), "iteration": checkpoint.iteration}
            for checkpoint in checkpoints
        ],
        "model_shape": vars(shape),
        "layers": layers,
        "operators": [operator for operator in OPERATORS if operator in operators],
        "subspace_ranks": subspace_ranks,
        "aggregate_metrics": aggregates,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plots:
        save_plots(output_dir, rows, aggregates, spectra, checkpoints, args.paper_layer)
    print(f"Wrote exact spectral analysis to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Initial torch-dist checkpoint directory.")
    parser.add_argument(
        "--checkpoint",
        type=parse_checkpoint,
        action="append",
        required=True,
        help="Trained checkpoint as LABEL=PATH; repeat for a trajectory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="SVD device (default: cuda:0).")
    parser.add_argument(
        "--svd-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="SVD precision (default: float64, recommended for small checkpoint deltas).",
    )
    parser.add_argument("--layers", help="Comma-separated layers/ranges, e.g. 0,5,10-12; default: all.")
    parser.add_argument("--operators", help=f"Comma-separated subset of {','.join(OPERATORS)}; default: all.")
    parser.add_argument("--subspace-ranks", default="16,64,256")
    parser.add_argument("--paper-layer", type=int, default=5, help="Layer for the paper-style Q/K/V delta plot.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite known outputs in a non-empty directory.")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
