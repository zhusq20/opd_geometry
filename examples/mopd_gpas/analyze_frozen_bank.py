#!/usr/bin/env python3
"""Cross-fit the real-checkpoint frozen gradient banks for K=1,2,4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slime_plugins.mopd.sampler import (
    TASKS,
    bounded_inclusion_probabilities,
    cost_optimized_set_distribution,
    inclusion_probabilities,
    maximum_entropy_set_distribution,
    predicted_set_seconds,
    task_sets,
)

METHODS = ("uniform", "raw_norm", "gpas", "cost_gpas")
# The frozen-bank launcher deliberately uses one colocated student/rollout GPU.
# Each operation therefore produces one optimizer-rank shard.
EXPECTED_BANK_RANKS = 1


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_stage(path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    allocations = jsonl(path / "allocation/allocation.jsonl")
    if len(allocations) != 32:
        raise ValueError(f"frozen bank must contain 32 full units: {path}")
    files: dict[tuple[int, str], list[Path]] = {}
    for file in (path / "bank").glob("unit_*_rank_*.pt"):
        payload = torch.load(file, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        files.setdefault((int(metadata["operation_index"]), str(metadata["task"])), []).append(file)
    observations = {task: [] for task in TASKS}
    expected_layout: dict[int, list[Any]] = {}
    for row in allocations:
        operation = int(row["operation_index"])
        task = str(row["execution_order"][0])
        rank_files = sorted(files.get((operation, task), []))
        rank_ids = [int(file.stem.rsplit("_", 1)[-1]) for file in rank_files]
        if rank_ids != list(range(EXPECTED_BANK_RANKS)):
            raise ValueError(
                f"bank operation {operation} ({task}) requires ranks "
                f"0..{EXPECTED_BANK_RANKS - 1}, got {rank_ids}"
            )
        raw_parts, scaled_parts, layouts = [], [], []
        metadata = None
        for file in rank_files:
            payload = torch.load(file, map_location="cpu", weights_only=False)
            if int(payload.get("schema_version", -1)) != 1:
                raise ValueError(f"unsupported bank tensor schema: {file}")
            payload_metadata = payload["metadata"]
            if (
                int(payload_metadata["operation_index"]) != operation
                or str(payload_metadata["task"]) != task
            ):
                raise ValueError(f"bank tensor metadata disagrees with allocation log: {file}")
            raw_parts.append(payload["raw"].float())
            scaled_parts.append(payload["scaled"].float())
            layouts.append(payload["layout"])
            metadata = payload_metadata
        if operation % 4 in expected_layout and layouts != expected_layout[operation % 4]:
            raise ValueError(f"bank coordinate layout changed for task {task}")
        expected_layout[operation % 4] = layouts
        unit = row["feedback"]["task_units"][0]
        observations[task].append(
            {
                "raw": torch.cat(raw_parts).double().numpy(),
                "adam": torch.cat(scaled_parts).double().numpy(),
                "raw_score": float(metadata["raw_score"]),
                "adam_score": float(metadata["adam_score"]),
                "task_seconds": float(unit["predicted_full_task_seconds"]),
                "switch_seconds": float(unit["teacher_transfer_tail_seconds"]),
            }
        )
    if any(len(values) != 8 for values in observations.values()):
        raise ValueError("each frozen checkpoint must contain eight units per task")
    return observations, allocations


def _distribution(
    method: str,
    width: int,
    raw_scores: np.ndarray,
    adam_scores: np.ndarray,
    task_seconds: np.ndarray,
    switch_seconds: np.ndarray,
    resident: int,
) -> tuple[np.ndarray, dict[tuple[int, ...], float]]:
    sets = task_sets(4, width)
    if width == 4:
        return np.ones(4), {sets[0]: 1.0}
    if method == "uniform":
        return np.full(4, width / 4), {subset: 1 / len(sets) for subset in sets}
    score = raw_scores if method == "raw_norm" else adam_scores
    if method == "cost_gpas":
        costs = {
            subset: predicted_set_seconds(subset, resident, task_seconds, switch_seconds)
            for subset in sets
        }
        distribution = cost_optimized_set_distribution(0.25 * score, width, 0.05, costs)
        marginals = inclusion_probabilities(distribution.keys(), distribution.values(), 4)
    else:
        marginals = bounded_inclusion_probabilities(0.25 * score, width, 0.05)
        distribution = maximum_entropy_set_distribution(marginals, width)
    return np.asarray(marginals), distribution


def _estimator_error(
    vectors: list[np.ndarray], marginals: np.ndarray, distribution: dict[tuple[int, ...], float]
) -> tuple[float, float, float]:
    target = sum(0.25 * vector for vector in vectors)
    target_norm = float(np.dot(target, target))
    error = 0.0
    conventional = np.zeros_like(target)
    taskwise = np.zeros_like(target)
    target_second = sum(0.25 * np.square(vector) for vector in vectors)
    for subset, probability in distribution.items():
        estimate = sum(0.25 / marginals[index] * vectors[index] for index in subset)
        error += probability * float(np.dot(estimate - target, estimate - target))
        conventional += probability * np.square(estimate)
        taskwise += probability * sum(
            0.25 / marginals[index] * np.square(vectors[index]) for index in subset
        )
    denominator = max(float(np.dot(target_second, target_second)), 1e-30)
    conventional_error = math.sqrt(float(np.dot(conventional - target_second, conventional - target_second)) / denominator)
    taskwise_error = math.sqrt(float(np.dot(taskwise - target_second, taskwise - target_second)) / denominator)
    return error / max(target_norm, 1e-30), conventional_error, taskwise_error


def _fold(observations: dict[str, list[dict[str, Any]]], train: slice, test: slice, resident: int) -> dict[str, Any]:
    raw_scores = np.asarray(
        [
            math.sqrt(np.mean([unit["raw_score"] ** 2 for unit in observations[task][train]]))
            for task in TASKS
        ]
    )
    adam_scores = np.asarray(
        [
            math.sqrt(np.mean([unit["adam_score"] ** 2 for unit in observations[task][train]]))
            for task in TASKS
        ]
    )
    task_seconds = np.asarray(
        [np.mean([unit["task_seconds"] for unit in observations[task][train]]) for task in TASKS]
    )
    switch_seconds = np.asarray(
        [np.mean([unit["switch_seconds"] for unit in observations[task][train]]) for task in TASKS]
    )
    result: dict[str, Any] = {"scores": {}, "K": {}}
    for index, task in enumerate(TASKS):
        result["scores"][task] = {
            "raw_norm": float(raw_scores[index]),
            "adamw_scaled_norm": float(adam_scores[index]),
            "task_seconds": float(task_seconds[index]),
            "switch_seconds": float(switch_seconds[index]),
        }
    test_indices = list(range(8))[test]
    for width in (1, 2, 4):
        by_method = {}
        for method in METHODS:
            marginals, distribution = _distribution(
                method, width, raw_scores, adam_scores, task_seconds, switch_seconds, resident
            )
            raw_errors, adam_errors, conventional_errors, taskwise_errors = [], [], [], []
            for offset in test_indices:
                raw_vectors = [observations[task][offset]["raw"] for task in TASKS]
                adam_vectors = [observations[task][offset]["adam"] for task in TASKS]
                raw_error, conventional, taskwise = _estimator_error(raw_vectors, marginals, distribution)
                adam_error, _, _ = _estimator_error(adam_vectors, marginals, distribution)
                raw_errors.append(raw_error)
                adam_errors.append(adam_error)
                conventional_errors.append(conventional)
                taskwise_errors.append(taskwise)
            by_method[method] = {
                "marginals": dict(zip(TASKS, map(float, marginals), strict=True)),
                "set_distribution": {
                    "+".join(TASKS[index] for index in subset): float(probability)
                    for subset, probability in distribution.items()
                },
                "relative_raw_estimator_mse": float(np.mean(raw_errors)),
                "relative_adamw_estimator_mse": float(np.mean(adam_errors)),
                "conventional_second_moment_relative_error": float(np.mean(conventional_errors)),
                "taskwise_second_moment_relative_error": float(np.mean(taskwise_errors)),
            }
        result["K"][str(width)] = by_method
    return result


def _average_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    output = {"scores": {}, "K": {}}
    for task in TASKS:
        output["scores"][task] = {
            key: float(np.mean([fold["scores"][task][key] for fold in folds]))
            for key in folds[0]["scores"][task]
        }
    for width in (1, 2, 4):
        output["K"][str(width)] = {}
        for method in METHODS:
            template = folds[0]["K"][str(width)][method]
            output["K"][str(width)][method] = {
                "marginals": {
                    task: float(np.mean([fold["K"][str(width)][method]["marginals"][task] for fold in folds]))
                    for task in TASKS
                },
                "set_distribution": {
                    subset: float(np.mean([fold["K"][str(width)][method]["set_distribution"].get(subset, 0.0) for fold in folds]))
                    for subset in template["set_distribution"]
                },
                **{
                    key: float(np.mean([fold["K"][str(width)][method][key] for fold in folds]))
                    for key in (
                        "relative_raw_estimator_mse",
                        "relative_adamw_estimator_mse",
                        "conventional_second_moment_relative_error",
                        "taskwise_second_moment_relative_error",
                    )
                },
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mopd-report", type=Path)
    args = parser.parse_args()
    stages = {}
    for stage in ("warm", "middle", "late"):
        path = args.root / f"frozen_bank_{stage}-seed42"
        observations, allocations = _load_stage(path)
        resident_name = allocations[0]["resident_teacher_before"]
        resident = 0 if resident_name is None else TASKS.index(str(resident_name))
        folds = [
            _fold(observations, slice(0, 4), slice(4, 8), resident),
            _fold(observations, slice(4, 8), slice(0, 4), resident),
        ]
        stages[stage] = {"cross_fit": _average_folds(folds), "folds": folds}
    result: dict[str, Any] = {
        "schema_version": 2,
        "units_per_task_per_checkpoint": 8,
        "cross_fit": "4/4 swap",
        "stages": stages,
    }
    if args.mopd_report:
        mopd = json.loads(args.mopd_report.read_text(encoding="utf-8"))
        result["online_diagnostics"] = {
            config: {
                key: mopd["outcomes"][config][key]
                for key in ("probe_count", "probe_gpu_hours", "max_score_age")
            }
            for config in mopd["configs"]
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Frozen-bank report written to {args.output}")


if __name__ == "__main__":
    main()
