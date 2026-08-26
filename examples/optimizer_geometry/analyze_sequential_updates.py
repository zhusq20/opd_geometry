#!/usr/bin/env python3
"""Analyze exact stage-local and cumulative parameter changes in sequential GRPO."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from analyze_checkpoint_updates import (
    Contribution,
    TorchDistTensorReader,
    contribution,
    sha256,
)

ABSOLUTE_THRESHOLDS = ((1e-6, "1e-6"), (1e-5, "1e-5"), (1e-4, "1e-4"))
SUPPORT_THRESHOLD = 1e-5
SUPPORT_THRESHOLD_NAME = "1e-5"
BF16_RELATIVE_TOLERANCE = 1e-3
RELATIVE_THRESHOLD = 1e-3


@dataclass(frozen=True)
class StageSpec:
    label: str
    before: Path
    after: Path


@dataclass
class Scope:
    parameter_count: int = 0
    reference_square_sum: float = 0.0
    squares: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    dots: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    changed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    absolute_unchanged: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    bf16_aware_unchanged: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    relative_unchanged: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    support_intersections: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    support_sign_conflicts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / ".metadata").is_file():
        return path
    marker = path / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError(f"Checkpoint has neither .metadata nor latest marker: {path}")
    iteration = int(marker.read_text(encoding="utf-8").strip())
    resolved = path / f"iter_{iteration:07d}"
    if not (resolved / ".metadata").is_file():
        raise FileNotFoundError(f"Latest checkpoint metadata is missing: {resolved}")
    return resolved


def parse_stage(value: str) -> StageSpec:
    if "=" not in value or "::" not in value:
        raise argparse.ArgumentTypeError("Stages must use LABEL=BEFORE::AFTER syntax.")
    label, paths = value.split("=", 1)
    before, after = paths.split("::", 1)
    if not label or not before or not after:
        raise argparse.ArgumentTypeError("Stage labels and checkpoint paths must be non-empty.")
    return StageSpec(label, Path(before), Path(after))


def add(
    scope: Scope,
    value: Contribution,
    updates: dict[str, torch.Tensor],
    sources: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> None:
    scope.parameter_count += value.parameter_count
    scope.reference_square_sum += value.reference_square_sum
    for label, square_sum in value.update_square_sums.items():
        scope.squares[label] += square_sum
    for pair, dot in value.update_dots.items():
        scope.dots[pair] += dot
    for label, changed_count in value.changed_counts.items():
        scope.changed[label] += changed_count

    supports: dict[str, torch.Tensor] = {}
    for label, update in updates.items():
        absolute = update.abs()
        for threshold, threshold_name in ABSOLUTE_THRESHOLDS:
            scope.absolute_unchanged[(label, threshold_name)] += int(torch.count_nonzero(absolute <= threshold))

        source = sources[label]
        target = targets[label]
        bf16_tolerance = BF16_RELATIVE_TOLERANCE * torch.maximum(source.abs(), target.abs())
        scope.bf16_aware_unchanged[label] += int(torch.count_nonzero(absolute <= bf16_tolerance))

        source_rms = float(torch.sqrt(torch.mean(source.square())).item())
        relative_cutoff = RELATIVE_THRESHOLD * max(source_rms, torch.finfo(torch.float32).tiny)
        scope.relative_unchanged[label] += int(torch.count_nonzero(absolute < relative_cutoff))
        supports[label] = absolute > SUPPORT_THRESHOLD

    labels = list(updates)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            intersection = torch.logical_and(supports[left], supports[right])
            scope.support_intersections[(left, right)] += int(torch.count_nonzero(intersection))
            sign_conflict = torch.logical_and(
                intersection,
                torch.logical_xor(torch.signbit(updates[left]), torch.signbit(updates[right])),
            )
            scope.support_sign_conflicts[(left, right)] += int(torch.count_nonzero(sign_conflict))


def row(scope_name: str, scope: Scope, labels: list[str]) -> dict[str, Any]:
    reference_norm = math.sqrt(scope.reference_square_sum)
    output: dict[str, Any] = {
        "scope": scope_name,
        "parameter_count": scope.parameter_count,
        "origin_parameter_norm": reference_norm,
    }
    for label in labels:
        norm = math.sqrt(scope.squares[label])
        output[f"update_norm/{label}"] = norm
        output[f"relative_update_norm/{label}"] = norm / reference_norm if reference_norm else None
        output[f"changed_count/{label}"] = scope.changed[label]
        output[f"changed_fraction/{label}"] = scope.changed[label] / scope.parameter_count
        output[f"exact_zero_sparsity/{label}"] = 1.0 - scope.changed[label] / scope.parameter_count
        for _, threshold_name in ABSOLUTE_THRESHOLDS:
            output[f"visible_sparsity@{threshold_name}/{label}"] = (
                scope.absolute_unchanged[(label, threshold_name)] / scope.parameter_count
            )
        output[f"bf16_aware_sparsity_eta1e-3/{label}"] = scope.bf16_aware_unchanged[label] / scope.parameter_count
        output[f"relative_sparsity_tau1e-3/{label}"] = scope.relative_unchanged[label] / scope.parameter_count
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            denominator = math.sqrt(scope.squares[left] * scope.squares[right])
            output[f"dot/{left}__{right}"] = scope.dots[(left, right)]
            output[f"cosine/{left}__{right}"] = scope.dots[(left, right)] / denominator if denominator else None
            output[f"interference_ratio/{left}_self_from_{right}"] = (
                scope.dots[(left, right)] / scope.squares[left] if scope.squares[left] else None
            )
            output[f"interference_ratio/{right}_self_from_{left}"] = (
                scope.dots[(left, right)] / scope.squares[right] if scope.squares[right] else None
            )

            left_support = scope.parameter_count - scope.absolute_unchanged[(left, SUPPORT_THRESHOLD_NAME)]
            right_support = scope.parameter_count - scope.absolute_unchanged[(right, SUPPORT_THRESHOLD_NAME)]
            intersection = scope.support_intersections[(left, right)]
            union = left_support + right_support - intersection
            left_overlap = intersection / left_support if left_support else None
            right_overlap = intersection / right_support if right_support else None
            left_baseline = right_support / scope.parameter_count
            right_baseline = left_support / scope.parameter_count
            output[f"support_intersection@{SUPPORT_THRESHOLD_NAME}/{left}__{right}"] = intersection
            output[f"support_jaccard@{SUPPORT_THRESHOLD_NAME}/{left}__{right}"] = (
                intersection / union if union else None
            )
            output[f"support_union_sparsity@{SUPPORT_THRESHOLD_NAME}/{left}__{right}"] = (
                1.0 - union / scope.parameter_count
            )
            output[f"support_overlap@{SUPPORT_THRESHOLD_NAME}/{left}_to_{right}"] = left_overlap
            output[f"support_overlap@{SUPPORT_THRESHOLD_NAME}/{right}_to_{left}"] = right_overlap
            output[f"support_independent_baseline@{SUPPORT_THRESHOLD_NAME}/{left}_to_{right}"] = left_baseline
            output[f"support_independent_baseline@{SUPPORT_THRESHOLD_NAME}/{right}_to_{left}"] = right_baseline
            output[f"support_overlap_lift@{SUPPORT_THRESHOLD_NAME}/{left}_to_{right}"] = (
                left_overlap / left_baseline if left_overlap is not None and left_baseline else None
            )
            output[f"support_overlap_lift@{SUPPORT_THRESHOLD_NAME}/{right}_to_{left}"] = (
                right_overlap / right_baseline if right_overlap is not None and right_baseline else None
            )
            output[f"support_sign_conflict_fraction@{SUPPORT_THRESHOLD_NAME}/{left}__{right}"] = (
                scope.support_sign_conflicts[(left, right)] / intersection if intersection else None
            )
    return output


def support_overlap_rows(scope: Scope, labels: list[str], update_family: str) -> list[dict[str, Any]]:
    rows = []
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            left_support = scope.parameter_count - scope.absolute_unchanged[(left, SUPPORT_THRESHOLD_NAME)]
            right_support = scope.parameter_count - scope.absolute_unchanged[(right, SUPPORT_THRESHOLD_NAME)]
            intersection = scope.support_intersections[(left, right)]
            union = left_support + right_support - intersection
            left_overlap = intersection / left_support if left_support else None
            right_overlap = intersection / right_support if right_support else None
            left_baseline = right_support / scope.parameter_count
            right_baseline = left_support / scope.parameter_count
            rows.append(
                {
                    "update_family": update_family,
                    "left": left,
                    "right": right,
                    "threshold": SUPPORT_THRESHOLD,
                    "parameter_count": scope.parameter_count,
                    "left_support_count": left_support,
                    "right_support_count": right_support,
                    "intersection_count": intersection,
                    "jaccard": intersection / union if union else None,
                    "union_sparsity": 1.0 - union / scope.parameter_count,
                    "left_to_right_overlap": left_overlap,
                    "left_to_right_independent_baseline": left_baseline,
                    "left_to_right_lift": (
                        left_overlap / left_baseline if left_overlap is not None and left_baseline else None
                    ),
                    "right_to_left_overlap": right_overlap,
                    "right_to_left_independent_baseline": right_baseline,
                    "right_to_left_lift": (
                        right_overlap / right_baseline if right_overlap is not None and right_baseline else None
                    ),
                    "intersection_sign_conflict_fraction": (
                        scope.support_sign_conflicts[(left, right)] / intersection if intersection else None
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for item in rows for key in item))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cosine_matrix(path: Path, global_row: dict[str, Any], labels: list[str]) -> None:
    rows = []
    for left_index, left in enumerate(labels):
        item = {"update": left}
        for right_index, right in enumerate(labels):
            if left == right:
                item[right] = 1.0
            else:
                pair = (left, right) if left_index < right_index else (right, left)
                item[right] = global_row[f"cosine/{pair[0]}__{pair[1]}"]
        rows.append(item)
    write_csv(path, rows)


def analyze(args: argparse.Namespace) -> None:
    stages = [
        StageSpec(spec.label, resolve_checkpoint(spec.before), resolve_checkpoint(spec.after)) for spec in args.stage
    ]
    labels = [stage.label for stage in stages]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Stage labels must be unique: {labels}")
    for previous, current in zip(stages, stages[1:], strict=False):
        if previous.after != current.before:
            raise ValueError(
                f"Sequential chain is broken: {previous.label} ends at {previous.after}, "
                f"but {current.label} starts at {current.before}."
            )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --force to overwrite outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_paths = list(dict.fromkeys([stages[0].before, *(stage.after for stage in stages)]))
    stage_scopes: dict[str, Scope] = defaultdict(Scope)
    cumulative_scopes: dict[str, Scope] = defaultdict(Scope)
    with ExitStack() as stack:
        readers = {path: stack.enter_context(TorchDistTensorReader(path)) for path in unique_paths}
        origin_reader = readers[stages[0].before]
        origin_names = set(origin_reader.tensor_names)
        for path, reader in readers.items():
            if set(reader.tensor_names) != origin_names:
                raise ValueError(f"Model tensor names differ at checkpoint {path}.")
            changed_shapes = {
                name: (origin_reader.shape(name), reader.shape(name))
                for name in origin_names
                if origin_reader.shape(name) != reader.shape(name)
            }
            if changed_shapes:
                raise ValueError(f"Model tensor shapes differ at checkpoint {path}: {changed_shapes}")

        for tensor_index, name in enumerate(origin_reader.tensor_names, start=1):
            print(f"[{tensor_index}/{len(origin_reader.tensor_names)}] {name}", flush=True)
            chunks_by_path = {path: readers[path].load_chunks(name) for path in unique_paths}
            expected_offsets = set(chunks_by_path[stages[0].before])
            for path, chunks in chunks_by_path.items():
                if set(chunks) != expected_offsets:
                    raise ValueError(f"Tensor chunk layout for {name} differs at {path}.")

            for offset in expected_offsets:
                values_by_path = {path: chunks_by_path[path][offset].to(torch.float32) for path in unique_paths}
                origin = values_by_path[stages[0].before]
                stage_vectors = {
                    stage.label: torch.subtract(values_by_path[stage.after], values_by_path[stage.before])
                    for stage in stages
                }
                cumulative_vectors = {
                    stage.label: torch.subtract(values_by_path[stage.after], origin) for stage in stages
                }
                stage_sources = {stage.label: values_by_path[stage.before] for stage in stages}
                stage_targets = {stage.label: values_by_path[stage.after] for stage in stages}
                cumulative_sources = {stage.label: origin for stage in stages}
                cumulative_targets = {stage.label: values_by_path[stage.after] for stage in stages}
                if name.startswith("decoder.layers."):
                    if not offset or origin.ndim == 0:
                        raise ValueError(f"Layer tensor {name} has no leading layer coordinate.")
                    for local_layer in range(origin.shape[0]):
                        layer = offset[0] + local_layer
                        stage_contribution = contribution(
                            origin[local_layer],
                            {label: value[local_layer] for label, value in stage_vectors.items()},
                        )
                        cumulative_contribution = contribution(
                            origin[local_layer],
                            {label: value[local_layer] for label, value in cumulative_vectors.items()},
                        )
                        stage_updates = {label: value[local_layer] for label, value in stage_vectors.items()}
                        stage_source_values = {label: value[local_layer] for label, value in stage_sources.items()}
                        stage_target_values = {label: value[local_layer] for label, value in stage_targets.items()}
                        cumulative_updates = {label: value[local_layer] for label, value in cumulative_vectors.items()}
                        cumulative_source_values = {
                            label: value[local_layer] for label, value in cumulative_sources.items()
                        }
                        cumulative_target_values = {
                            label: value[local_layer] for label, value in cumulative_targets.items()
                        }
                        add(
                            stage_scopes["global"],
                            stage_contribution,
                            stage_updates,
                            stage_source_values,
                            stage_target_values,
                        )
                        add(
                            stage_scopes[f"layer/{layer}"],
                            stage_contribution,
                            stage_updates,
                            stage_source_values,
                            stage_target_values,
                        )
                        add(
                            cumulative_scopes["global"],
                            cumulative_contribution,
                            cumulative_updates,
                            cumulative_source_values,
                            cumulative_target_values,
                        )
                        add(
                            cumulative_scopes[f"layer/{layer}"],
                            cumulative_contribution,
                            cumulative_updates,
                            cumulative_source_values,
                            cumulative_target_values,
                        )
                else:
                    add(
                        stage_scopes["global"],
                        contribution(origin, stage_vectors),
                        stage_vectors,
                        stage_sources,
                        stage_targets,
                    )
                    add(
                        cumulative_scopes["global"],
                        contribution(origin, cumulative_vectors),
                        cumulative_vectors,
                        cumulative_sources,
                        cumulative_targets,
                    )

    ordered = ["global"]
    ordered.extend(
        sorted(
            (name for name in stage_scopes if name.startswith("layer/")),
            key=lambda name: int(name.split("/", 1)[1]),
        )
    )
    stage_rows = [row(name, stage_scopes[name], labels) for name in ordered]
    cumulative_rows = [row(name, cumulative_scopes[name], labels) for name in ordered]
    write_csv(output_dir / "stage_update_geometry.csv", stage_rows)
    write_csv(output_dir / "cumulative_update_geometry.csv", cumulative_rows)
    cosine_matrix(output_dir / "stage_cosine_matrix.csv", stage_rows[0], labels)
    cosine_matrix(output_dir / "cumulative_cosine_matrix.csv", cumulative_rows[0], labels)
    stage_support_rows = support_overlap_rows(stage_scopes["global"], labels, "stage_local")
    cumulative_support_rows = support_overlap_rows(cumulative_scopes["global"], labels, "cumulative")
    write_csv(output_dir / "stage_support_overlap.csv", stage_support_rows)
    write_csv(output_dir / "cumulative_support_overlap.csv", cumulative_support_rows)

    checkpoints = [stages[0].before, *(stage.after for stage in stages)]
    summary = {
        "schema_version": 2,
        "method": (
            "Exact FP64 norm/dot reductions over BF16 realized model tensors in torch-dist checkpoints; "
            "optimizer and RNG entries are excluded."
        ),
        "origin": str(stages[0].before),
        "optimizer_state_policy": "Not inferred here; read the sequential run lineage manifest.",
        "metric_definitions": {
            "exact_zero_sparsity": "Fraction of stored BF16 coordinates with an exactly zero checkpoint delta.",
            "visible_sparsity": "Fraction with |delta| <= epsilon for epsilon in {1e-6, 1e-5, 1e-4}.",
            "bf16_aware_sparsity_eta1e-3": (
                "Fraction with |after-before| <= 1e-3 * max(|before|, |after|), following arXiv:2606.07082."
            ),
            "relative_sparsity_tau1e-3": (
                "Coordinate-weighted fraction with |delta| < 1e-3 * RMS(source), computed per logical tensor block."
            ),
            "support_overlap": (
                "Directional overlap, independent-support baseline, lift, Jaccard, union sparsity, and sign-conflict "
                "fraction for supports |delta| > 1e-5, following arXiv:2606.13657."
            ),
            "interference_ratio": "Cross-update dot product divided by the receiving update's squared norm.",
        },
        "stages": [{"label": stage.label, "before": str(stage.before), "after": str(stage.after)} for stage in stages],
        "checkpoint_fingerprints": {
            str(path): {
                "metadata_sha256": sha256(path / ".metadata"),
                "common_sha256": sha256(path / "common.pt"),
            }
            for path in checkpoints
        },
        "stage_local_global": stage_rows[0],
        "cumulative_global": cumulative_rows[0],
        "stage_local_support_overlap": stage_support_rows,
        "cumulative_support_overlap": cumulative_support_rows,
        "stage_local_layerwise": stage_rows[1:],
        "cumulative_layerwise": cumulative_rows[1:],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote sequential checkpoint geometry to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=parse_stage, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
