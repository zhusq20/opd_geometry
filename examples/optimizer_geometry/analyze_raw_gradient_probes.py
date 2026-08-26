#!/usr/bin/env python3
"""Compute exact norm, dot, and cosine from persisted raw-gradient probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from analyze_checkpoint_updates import fp64_gram


@dataclass(frozen=True)
class ProbeSpec:
    label: str
    path: Path


@dataclass
class Scope:
    parameter_count: int = 0
    squares: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    dots: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))


def parse_probe(value: str) -> ProbeSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Probes must use LABEL=PATH syntax.")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Probe labels and paths must be non-empty.")
    return ProbeSpec(label, Path(raw_path))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def layout(rank_manifest: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            view["name"],
            int(view["start"]),
            None if view["stop"] is None else int(view["stop"]),
            int(view["numel"]),
            tuple(view["group_names"]),
            view["optimizer_branch"],
            view["file"],
        )
        for view in rank_manifest["views"]
    ]


def scope_row(name: str, scope: Scope, labels: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {"scope": name, "parameter_count": scope.parameter_count}
    for label in labels:
        row[f"gradient_norm/{label}"] = math.sqrt(scope.squares[label])
        row[f"gradient_direction_defined/{label}"] = scope.squares[label] > 0.0
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            pair = (left, right)
            denominator = math.sqrt(scope.squares[left] * scope.squares[right])
            row[f"dot/{left}__{right}"] = scope.dots[pair]
            row[f"cosine/{left}__{right}"] = scope.dots[pair] / denominator if denominator else None
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> None:
    probes = [ProbeSpec(spec.label, spec.path.resolve()) for spec in args.probe]
    labels = [probe.label for probe in probes]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Probe labels must be unique: {labels}")
    if len(labels) < 2:
        raise ValueError("At least two task probes are required for interference geometry.")

    roots = {probe.label: read_json(probe.path / "manifest.json") for probe in probes}
    first = roots[labels[0]]
    common_fields = (
        "world_size",
        "expected_updates",
        "actual_batch_size_per_update",
        "rollout_batch_size",
        "n_samples_per_prompt",
        "probe_prompt_count",
        "optimizer_step_executed",
    )
    for label in labels[1:]:
        mismatched = {
            field: (first[field], roots[label][field])
            for field in common_fields
            if roots[label][field] != first[field]
        }
        if mismatched:
            raise ValueError(f"Probe {label} is not comparable to {labels[0]}: {mismatched}")
        if (
            roots[label]["checkpoint"] != first["checkpoint"]
            or roots[label]["checkpoint_step"] != first["checkpoint_step"]
        ):
            raise ValueError(
                f"Probe {label} used a different checkpoint: "
                f"{roots[label]['checkpoint']}@{roots[label]['checkpoint_step']} != "
                f"{first['checkpoint']}@{first['checkpoint_step']}"
            )
    if first["optimizer_step_executed"]:
        raise ValueError("Same-checkpoint raw-gradient analysis requires backward-only probes.")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --force to overwrite outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    scopes: dict[str, Scope] = defaultdict(Scope)
    rank_count = int(first["world_size"])
    for rank_id in range(rank_count):
        manifests = {}
        for probe in probes:
            path = probe.path / f"rank_{rank_id:05d}" / "manifest.json"
            manifests[probe.label] = read_json(path)
        expected_layout = layout(manifests[labels[0]])
        for label in labels[1:]:
            if layout(manifests[label]) != expected_layout:
                raise ValueError(f"Raw-gradient optimizer view layout differs for task {label}, rank {rank_id}.")

        for view_id, descriptor in enumerate(expected_layout):
            _, _, _, numel, group_names, _, filename = descriptor
            gradients = {}
            for probe in probes:
                path = probe.path / f"rank_{rank_id:05d}" / filename
                value = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(value, torch.Tensor) or value.numel() != numel:
                    raise ValueError(f"Invalid raw-gradient tensor: {path}")
                value = value.reshape(-1).to(torch.float32)
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Non-finite raw-gradient tensor: {path}")
                gradients[probe.label] = value

            contributions = ["global", *dict.fromkeys(group_names)]
            squares, dots = fp64_gram(gradients)
            for scope_name in contributions:
                scope = scopes[scope_name]
                scope.parameter_count += numel
                for label, value in squares.items():
                    scope.squares[label] += value
                for pair, value in dots.items():
                    scope.dots[pair] += value

    ordered_scopes = ["global"]
    ordered_scopes.extend(
        sorted(
            (name for name in scopes if name.startswith("layer/")),
            key=lambda name: int(name.split("/", 1)[1]),
        )
    )
    ordered_scopes.extend(sorted(name for name in scopes if name not in set(ordered_scopes)))
    rows = [scope_row(name, scopes[name], labels) for name in ordered_scopes]
    write_csv(output_dir / "raw_gradient_geometry.csv", rows)

    global_row = rows[0]
    matrix = []
    for left_index, left in enumerate(labels):
        row = {"task": left}
        for right_index, right in enumerate(labels):
            if left == right:
                row[right] = 1.0 if global_row[f"gradient_direction_defined/{left}"] else None
            else:
                pair = (left, right) if left_index < right_index else (right, left)
                row[right] = global_row[f"cosine/{pair[0]}__{pair[1]}"]
        matrix.append(row)
    write_csv(output_dir / "global_cosine_matrix.csv", matrix)

    summary = {
        "schema_version": 2,
        "method": (
            "Exact FP64 norm/dot reductions over matching optimizer-owned FP32 raw-gradient shards; "
            "no random projection. Each task vector is the arithmetic mean of equal-sized probe updates."
        ),
        "ascent_gradient_note": (
            "Artifacts store loss gradients. The experiment defines g=-grad(L); simultaneous sign reversal "
            "preserves every reported norm, pairwise dot, and cosine."
        ),
        "cosine_definition_note": (
            "Cosine is null when either raw-gradient norm is zero; a zero vector has no direction and must not "
            "be interpreted as orthogonal to another task."
        ),
        "checkpoint": first["checkpoint"],
        "checkpoint_step": first["checkpoint_step"],
        "probe_updates": first["expected_updates"],
        "actual_batch_size_per_update": first["actual_batch_size_per_update"],
        "probe_prompt_count": first["probe_prompt_count"],
        "optimizer_step_executed": first["optimizer_step_executed"],
        "tasks": {label: roots[label] for label in labels},
        "global": global_row,
        "layerwise": [row for row in rows if str(row["scope"]).startswith("layer/")],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote exact raw-gradient interference analysis to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=parse_probe, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
