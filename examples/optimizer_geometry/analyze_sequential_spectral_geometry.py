#!/usr/bin/env python3
"""Measure deterministic randomized-SVD geometry along sequential stage trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_checkpoint_updates import TorchDistTensorReader, sha256

MATRIX_NAMES = (
    "decoder.layers.self_attention.linear_proj.weight",
    "decoder.layers.self_attention.linear_qkv.weight",
    "decoder.layers.mlp.linear_fc2.weight",
    "decoder.layers.mlp.linear_fc1.weight",
)
SHORT_NAME = {
    "decoder.layers.self_attention.linear_proj.weight": "attention_output",
    "decoder.layers.self_attention.linear_qkv.weight": "attention_qkv",
    "decoder.layers.mlp.linear_fc2.weight": "mlp_down",
    "decoder.layers.mlp.linear_fc1.weight": "mlp_gate_up",
}


@dataclass(frozen=True)
class Stage:
    task: str
    start: Path
    checkpoints: tuple[tuple[int, Path], ...]


@dataclass(frozen=True)
class ApproximateSVD:
    singular_values: torch.Tensor
    right_vectors: torch.Tensor


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON objects in {path}.")
                rows.append(value)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_iteration(path: Path) -> int:
    value = path.name.removeprefix("iter_")
    if not value.isdigit():
        raise ValueError(f"Checkpoint directory must be named iter_N: {path}")
    return int(value)


def resolve_torch_dist_checkpoint(path: Path) -> Path:
    path = path.resolve()
    if (path / ".metadata").is_file() and (path / "common.pt").is_file():
        return path
    marker = path / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError(f"Not a torch-dist checkpoint or save root: {path}")
    value = marker.read_text(encoding="utf-8").strip()
    if value == "release":
        checkpoint = path / "release"
    elif value.isdigit():
        checkpoint = path / f"iter_{int(value):07d}"
    else:
        raise ValueError(f"Unsupported checkpoint marker {value!r} in {marker}.")
    if not (checkpoint / ".metadata").is_file() or not (checkpoint / "common.pt").is_file():
        raise FileNotFoundError(f"Resolved torch-dist checkpoint is incomplete: {checkpoint}")
    return checkpoint.resolve()


def load_stage_checkpoints(run_dir: Path, task: str) -> tuple[tuple[int, Path], ...]:
    checkpoint_paths = tuple(
        (checkpoint_iteration(path) + 1, path.resolve())
        for path in sorted((run_dir / "checkpoints").glob("iter_*"))
        if (path / ".metadata").is_file() and (path / "common.pt").is_file()
    )
    if [updates for updates, _ in checkpoint_paths] != [100, 200, 300]:
        raise ValueError(f"Expected update-100/200/300 checkpoints for {task}, found {checkpoint_paths}.")
    return checkpoint_paths


def load_stages(sequence_root: Path, pre_code_checkpoint: Path) -> tuple[dict[str, Any], list[Stage]]:
    manifest = read_json(sequence_root / "sequence_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Sequential training is not complete.")
    boundaries = sorted(
        read_jsonl(sequence_root / "stage_boundaries.jsonl"),
        key=lambda row: int(row["stage_number"]),
    )
    origin = Path(str(manifest["code_origin"])).resolve()
    code_manifest = Path(str(manifest["code_origin_training"]["provenance_manifest"])).resolve()
    code_run_dir = code_manifest.parent.parent
    code_command = [str(item) for item in read_json(code_manifest)["command"]]
    if "--load" not in code_command or code_command.index("--load") + 1 >= len(code_command):
        raise ValueError(f"Code provenance command is missing --load: {code_manifest}")
    recorded_code_input = resolve_torch_dist_checkpoint(Path(code_command[code_command.index("--load") + 1]))
    pre_code_checkpoint = resolve_torch_dist_checkpoint(pre_code_checkpoint)
    if recorded_code_input != pre_code_checkpoint:
        raise ValueError(
            f"Code training loaded {recorded_code_input}, but the requested pre-Code checkpoint is {pre_code_checkpoint}."
        )
    code_checkpoints = load_stage_checkpoints(code_run_dir, "code")
    if code_checkpoints[-1][1] != origin:
        raise ValueError("Final Code spectral checkpoint does not match the sequence Code origin.")
    stages = [Stage("code", pre_code_checkpoint, code_checkpoints)]
    for record in boundaries:
        run_dir = Path(str(record["run_dir"])).resolve()
        checkpoint_paths = load_stage_checkpoints(run_dir, str(record["task"]))
        start = Path(str(record["input_checkpoint"])).resolve()
        if checkpoint_paths[-1][1] != Path(str(record["output_checkpoint"])).resolve():
            raise ValueError(f"Final spectral checkpoint does not match stage boundary for {record['task']}.")
        stages.append(Stage(str(record["task"]), start, checkpoint_paths))
    if [stage.task for stage in stages] != ["code", "math", "knowledge", "if"]:
        raise ValueError(f"Unexpected stage order: {[stage.task for stage in stages]}")
    return manifest, stages


def assemble_layer_matrix(
    reader: TorchDistTensorReader,
    name: str,
    layer: int,
) -> torch.Tensor:
    shape = reader.shape(name)
    if len(shape) != 3 or not 0 <= layer < shape[0]:
        raise ValueError(f"Invalid layer {layer} for {name} with shape {shape}.")
    chunks = reader.load_chunks(name, leading_index=layer)
    matrix = torch.empty((shape[1], shape[2]), dtype=torch.float32)
    covered = torch.zeros(shape[1], dtype=torch.bool)
    for offset, value in chunks.items():
        if len(offset) != 3 or offset[0] != layer or offset[2] != 0 or value.shape[0] != 1:
            raise ValueError(f"Unsupported matrix chunk for {name}, layer {layer}: {offset}/{value.shape}")
        row_start = offset[1]
        row_stop = row_start + value.shape[1]
        if bool(torch.any(covered[row_start:row_stop])):
            raise ValueError(f"Overlapping matrix chunks for {name}, layer {layer}.")
        matrix[row_start:row_stop] = value[0].to(torch.float32)
        covered[row_start:row_stop] = True
    if not bool(torch.all(covered)):
        raise ValueError(f"Incomplete matrix chunks for {name}, layer {layer}.")
    return matrix


def matrix_seed(base_seed: int, name: str, layer: int) -> int:
    digest = hashlib.sha256(f"{name}::{layer}".encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**63 - 1)


def randomized_svd(
    matrix: torch.Tensor,
    *,
    rank: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
) -> ApproximateSVD:
    rows, columns = matrix.shape
    sketch_rank = min(rank + oversampling, rows, columns)
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(seed)
    omega = torch.randn(columns, sketch_rank, generator=generator, device=matrix.device)
    basis = torch.linalg.qr(matrix @ omega, mode="reduced").Q
    for _ in range(power_iterations):
        basis = torch.linalg.qr(matrix @ (matrix.T @ basis), mode="reduced").Q
    compressed = basis.T @ matrix
    _, singular_values, vh = torch.linalg.svd(compressed, full_matrices=False)
    return ApproximateSVD(
        singular_values=singular_values[:rank].to(torch.float64).cpu(),
        right_vectors=vh[:rank].T.to(torch.float32).cpu(),
    )


def spectral_metrics(
    update: torch.Tensor,
    approximation: ApproximateSVD,
    rank: int,
) -> dict[str, Any]:
    frobenius_square = float(torch.sum(update.to(torch.float64).square()).item())
    singular_square = approximation.singular_values.square()
    top1_square = float(singular_square[0]) if singular_square.numel() else 0.0
    minimum_dimension = min(update.shape)
    return {
        "rows": update.shape[0],
        "columns": update.shape[1],
        "parameter_count": update.numel(),
        "frobenius_norm": math.sqrt(frobenius_square),
        "approx_spectral_norm": float(approximation.singular_values[0]),
        "approx_stable_rank": frobenius_square / top1_square if top1_square else None,
        "approx_normalized_stable_rank": (frobenius_square / top1_square / minimum_dimension if top1_square else None),
        "approx_top1_energy_fraction": top1_square / frobenius_square if frobenius_square else None,
        "approx_top8_energy_fraction": (
            float(torch.sum(singular_square[: min(8, rank)])) / frobenius_square if frobenius_square else None
        ),
        f"approx_top{rank}_energy_fraction": (
            float(torch.sum(singular_square)) / frobenius_square if frobenius_square else None
        ),
    }


def subspace_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    rank = min(left.shape[1], right.shape[1])
    if rank == 0:
        return float("nan")
    gram = left[:, :rank].T.to(torch.float64) @ right[:, :rank].to(torch.float64)
    value = float(torch.sum(gram.square()).item() / rank)
    return min(1.0, max(0.0, value))


def aggregate_rows(rows: list[dict[str, Any]], rank: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["stage"]), int(row["stage_update"]), str(row["matrix_type"]))
        groups.setdefault(key, []).append(row)
    outputs = []
    metric_names = (
        "frobenius_norm",
        "approx_spectral_norm",
        "approx_stable_rank",
        "approx_normalized_stable_rank",
        "approx_top1_energy_fraction",
        "approx_top8_energy_fraction",
        f"approx_top{rank}_energy_fraction",
        "topk_weight_spectral_shift",
        f"subspace_locking_top{rank}",
    )
    for (stage, update, matrix_type), values in sorted(groups.items()):
        output: dict[str, Any] = {
            "stage": stage,
            "stage_update": update,
            "matrix_type": matrix_type,
            "sampled_layer_count": len(values),
        }
        for metric in metric_names:
            present = [float(row[metric]) for row in values if row.get(metric) is not None]
            output[f"mean/{metric}"] = float(np.mean(present)) if present else None
            output[f"median/{metric}"] = float(np.median(present)) if present else None
        outputs.append(output)
    return outputs


def save_plots(output_dir: Path, rows: list[dict[str, Any]], rank: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    for stage, color in (
        ("code", "tab:purple"),
        ("math", "tab:blue"),
        ("knowledge", "tab:orange"),
        ("if", "tab:green"),
    ):
        for metric, axis in (
            ("approx_stable_rank", axes[0]),
            (f"approx_top{rank}_energy_fraction", axes[1]),
            (f"subspace_locking_top{rank}", axes[2]),
        ):
            values = [row for row in rows if row["stage"] == stage and row.get(metric) is not None]
            by_update = {
                update: float(np.mean([float(row[metric]) for row in values if row["stage_update"] == update]))
                for update in sorted({int(row["stage_update"]) for row in values})
            }
            axis.plot(list(by_update), list(by_update.values()), marker="o", label=stage, color=color)
    axes[0].set_title("Approximate stable rank of stage update")
    axes[1].set_title(f"Approximate top-{rank} update energy")
    axes[2].set_title(f"Top-{rank} subspace locking to stage final")
    for axis in axes:
        axis.set_xlabel("stage optimizer updates")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "spectral_trajectory.png", dpi=200)
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    sequence_root = args.sequence_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --force.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _, stages = load_stages(sequence_root, args.pre_code_checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    layers = sorted(set(args.layer))
    if not layers:
        raise ValueError("At least one layer is required.")

    unique_paths = list(
        dict.fromkeys(
            [
                *(stage.start for stage in stages),
                *(path for stage in stages for _, path in stage.checkpoints),
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    approximation_spot_check: dict[str, Any] | None = None
    with ExitStack() as stack:
        readers = {path: stack.enter_context(TorchDistTensorReader(path)) for path in unique_paths}
        for name in MATRIX_NAMES:
            for layer in layers:
                print(f"{SHORT_NAME[name]} layer {layer}", flush=True)
                weights = {path: assemble_layer_matrix(readers[path], name, layer) for path in unique_paths}
                seed = matrix_seed(args.seed, name, layer)
                weight_svd = {
                    path: randomized_svd(
                        value.to(device),
                        rank=args.rank,
                        oversampling=args.oversampling,
                        power_iterations=args.power_iterations,
                        seed=seed,
                    )
                    for path, value in weights.items()
                }
                for stage in stages:
                    start_weight = weights[stage.start]
                    start_singular = weight_svd[stage.start].singular_values
                    update_svd: dict[int, ApproximateSVD] = {}
                    updates: dict[int, torch.Tensor] = {}
                    for stage_update, checkpoint in stage.checkpoints:
                        update = weights[checkpoint] - start_weight
                        approximation = randomized_svd(
                            update.to(device),
                            rank=args.rank,
                            oversampling=args.oversampling,
                            power_iterations=args.power_iterations,
                            seed=seed,
                        )
                        updates[stage_update] = update
                        update_svd[stage_update] = approximation
                    if (
                        approximation_spot_check is None
                        and name == MATRIX_NAMES[0]
                        and layer == layers[0]
                        and stage.task == "math"
                    ):
                        final_update = stage.checkpoints[-1][0]
                        approximate_values = update_svd[final_update].singular_values
                        exact_values = (
                            torch.linalg.svdvals(updates[final_update].to(device))[: args.rank].to(torch.float64).cpu()
                        )
                        approximation_spot_check = {
                            "matrix": name,
                            "layer": layer,
                            "stage": stage.task,
                            "stage_update": final_update,
                            "sigma1_relative_error": float(
                                torch.abs(approximate_values[0] - exact_values[0]) / exact_values[0]
                            ),
                            f"top{args.rank}_energy_relative_error": float(
                                torch.abs(torch.sum(approximate_values.square()) - torch.sum(exact_values.square()))
                                / torch.sum(exact_values.square())
                            ),
                            f"max_top{args.rank}_singular_value_relative_error": float(
                                torch.max(torch.abs(approximate_values - exact_values) / exact_values)
                            ),
                        }
                    final_vectors = update_svd[stage.checkpoints[-1][0]].right_vectors
                    for stage_update, checkpoint in stage.checkpoints:
                        approximation = update_svd[stage_update]
                        current_singular = weight_svd[checkpoint].singular_values
                        spectral_shift = float(
                            torch.linalg.vector_norm(current_singular - start_singular)
                            / torch.linalg.vector_norm(start_singular)
                        )
                        rows.append(
                            {
                                "stage": stage.task,
                                "stage_update": stage_update,
                                "stage_fraction": stage_update / stage.checkpoints[-1][0],
                                "checkpoint": str(checkpoint),
                                "matrix": name,
                                "matrix_type": SHORT_NAME[name],
                                "layer": layer,
                                **spectral_metrics(updates[stage_update], approximation, args.rank),
                                "topk_weight_spectral_shift": spectral_shift,
                                f"subspace_locking_top{args.rank}": subspace_similarity(
                                    approximation.right_vectors, final_vectors
                                ),
                            }
                        )
                del weights, weight_svd
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    aggregates = aggregate_rows(rows, args.rank)
    write_csv(output_dir / "spectral_update_geometry.csv", rows)
    write_csv(output_dir / "spectral_update_aggregates.csv", aggregates)
    checkpoint_fingerprints = {
        str(path): {
            "metadata_sha256": sha256(path / ".metadata"),
            "common_sha256": sha256(path / "common.pt"),
        }
        for path in unique_paths
    }
    summary = {
        "schema_version": 1,
        "method": (
            "Deterministic randomized SVD of realized BF16 checkpoint-update matrices, converted to FP32. "
            "Metrics are a stratified layer sample and are explicitly approximate, not full-model exact reductions."
        ),
        "layers": layers,
        "matrix_types": SHORT_NAME,
        "stages": [stage.task for stage in stages],
        "randomized_svd": {
            "rank": args.rank,
            "oversampling": args.oversampling,
            "power_iterations": args.power_iterations,
            "seed": args.seed,
            "shared_test_matrix_policy": "same deterministic Gaussian sketch per layer/matrix across checkpoints",
            "exact_spot_check": approximation_spot_check,
        },
        "definitions": {
            "approx_stable_rank": "Frobenius(update)^2 / approximate_sigma_1(update)^2",
            "approx_topk_energy_fraction": "sum of approximate top-k squared singular values / exact Frobenius(update)^2",
            "topk_weight_spectral_shift": (
                "L2 change of approximate top-k weight singular values divided by their stage-start L2 norm"
            ),
            "subspace_locking": "||V_k(t)^T V_k(stage_final)||_F^2 / k",
        },
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "aggregates": aggregates,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plots:
        save_plots(output_dir, rows, args.rank)
    print(f"Wrote sequential spectral geometry to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--pre-code-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, action="append", default=[])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--oversampling", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.rank <= 0 or args.oversampling < 0 or args.power_iterations < 0:
        parser.error("Rank must be positive; oversampling and power iterations must be non-negative.")
    if any(layer < 0 for layer in args.layer):
        parser.error("Layer indices must be non-negative.")
    return args


if __name__ == "__main__":
    analyze(parse_args())
