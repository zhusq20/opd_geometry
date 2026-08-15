#!/usr/bin/env python3
"""Flatten geometry records and estimate cross-task gradient interference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.to(torch.float64)
    right = right.to(torch.float64)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if not float(denominator):
        return None
    return float(torch.dot(left, right) / denominator)


def records(paths: Iterable[Path]):
    for path in paths:
        metrics_path = path / "metrics.jsonl" if path.is_dir() else path
        base = metrics_path.parent
        run_id = str(base.resolve())
        for line in metrics_path.read_text().splitlines():
            record = json.loads(line)
            vectors = None
            if record.get("vector_file"):
                vectors = torch.load(base / record["vector_file"], map_location="cpu", weights_only=True)["groups"]
            yield run_id, record, vectors


def homogeneous_task(source_counts: dict[str, int]) -> str | None:
    nonempty = [name for name, count in source_counts.items() if count]
    return nonempty[0] if len(nonempty) == 1 else None


def analyze(
    paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float | None]]]]:
    rows: list[dict[str, Any]] = []
    # Projection coordinates are only comparable inside one run: launch seeds
    # deliberately change the CountSketch map.  Keep run identity in both the
    # temporal and task-centroid aggregations so separate seeds never get
    # averaged in incompatible coordinate systems.
    task_gradients: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    previous: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}

    for run_id, record, vectors in records(paths):
        task = homogeneous_task(record.get("source_counts", {}))
        for group, metrics in record["groups"].items():
            row = {
                "run_id": run_id,
                "optimizer": record["optimizer"],
                "experiment_task": record.get("experiment_task"),
                "experiment_teacher": record.get("experiment_teacher"),
                "experiment_condition": record.get("experiment_condition"),
                "advantage_estimator": record["advantage_estimator"],
                "loss_type": record.get("loss_type", "policy_loss"),
                "use_opd": record["use_opd"],
                "opd_kl_coef": record.get("opd_kl_coef"),
                "opd_task_reward_weight": record.get("opd_task_reward_weight"),
                "hybrid_sft_loss_coef": record.get("hybrid_sft_loss_coef"),
                "hybrid_opd_loss_coef": record.get("hybrid_opd_loss_coef"),
                "learning_rate": record.get("learning_rate"),
                "weight_decay": record.get("weight_decay"),
                "role": record["role"],
                "rollout_id": record["rollout_id"],
                "step_id": record["step_id"],
                "observation_id": record["observation_id"],
                "projection_dim": record.get("projection_dim"),
                "projection_seed": record.get("projection_seed"),
                "task": task or "mixed",
                "group": group,
                **metrics,
            }
            if vectors is not None and group in vectors:
                current = vectors[group]
                previous_key = (run_id, record["role"], group)
                last = previous.get(previous_key)
                row["cos_gradient_previous_gradient_sketch"] = (
                    cosine(current["gradient"], last["gradient"]) if last is not None else None
                )
                row["cos_update_previous_update_sketch"] = (
                    cosine(current["update"], last["update"]) if last is not None else None
                )
                previous[previous_key] = current
                if task:
                    task_gradients[run_id][group][task].append(current["gradient"])
            rows.append(row)

    matrices: dict[str, dict[str, dict[str, float | None]]] = {}
    for run_id, by_group in task_gradients.items():
        matrices[run_id] = {}
        for group, by_task in by_group.items():
            means = {task: torch.stack(values).mean(dim=0) for task, values in by_task.items()}
            matrices[run_id][group] = {
                f"cos_{left}__{right}_sketch": cosine(means[left], means[right])
                for left in sorted(means)
                for right in sorted(means)
            }
    return rows, matrices


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Actor geometry directories or metrics.jsonl files.")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scalar_rows, task_matrices = analyze(args.inputs)
    write_csv(args.output_prefix.with_suffix(".csv"), scalar_rows)
    args.output_prefix.with_suffix(".task_cosines.json").write_text(
        json.dumps(task_matrices, indent=2, sort_keys=True) + "\n"
    )
