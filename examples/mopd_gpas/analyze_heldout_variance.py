#!/usr/bin/env python3
"""Cross-fit the scalar-only held-out gradient measurements at steps 50/250/500."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from slime_plugins.mopd.sampler import TASKS, cost_gpas_allocation, largest_remainder_allocation

CHECKPOINTS = (50, 250, 500)
METHODS = ("uniform", "raw_noise", "loss_gap", "gpas", "cost_gpas")
HALF_SIZE = 16


def _noise(square_values: list[float], mean_square: float) -> float:
    count = len(square_values)
    return max((sum(square_values) - count * float(mean_square)) / (count - 1), 0.0)


def _task_half(unit: dict[str, Any], half: int) -> dict[str, float]:
    start = half * HALF_SIZE
    stop = start + HALF_SIZE
    return {
        "raw_noise": _noise(unit["raw_microbatch_sq"][start:stop], unit["raw_half_mean_sq"][half]),
        "scaled_noise": _noise(unit["scaled_microbatch_sq"][start:stop], unit["scaled_half_mean_sq"][half]),
        "teacher_loss": float(np.mean(unit["teacher_loss_microbatch"][start:stop])),
    }


def _allocation(
    method: str,
    weights: list[float],
    estimates: list[dict[str, float]],
    task_seconds: list[float],
    fixed_seconds: float,
) -> list[int]:
    if method == "uniform":
        return [4, 4, 4, 4]
    if method == "raw_noise":
        scores = [weight * math.sqrt(value["raw_noise"]) for weight, value in zip(weights, estimates, strict=True)]
    elif method == "loss_gap":
        scores = [weight * value["teacher_loss"] for weight, value in zip(weights, estimates, strict=True)]
    elif method == "gpas":
        scores = [weight * math.sqrt(value["scaled_noise"]) for weight, value in zip(weights, estimates, strict=True)]
    elif method == "cost_gpas":
        return cost_gpas_allocation(
            weights,
            [value["scaled_noise"] for value in estimates],
            task_seconds,
            fixed_seconds,
        )
    else:
        raise ValueError(method)
    return largest_remainder_allocation(scores)


def _variance(weights: list[float], noise: list[float], counts: list[int]) -> float:
    return sum(weight * weight * value / count for weight, value, count in zip(weights, noise, counts, strict=True))


def _subset_noise(unit: dict[str, Any], half: int, size: int, kind: str) -> float:
    start = half * HALF_SIZE
    squares = unit[f"{kind}_microbatch_sq"][start : start + size]
    mean_square = unit[f"{kind}_subset_mean_sq"][str(size)][half]
    return _noise(squares, mean_square)


def analyze_checkpoint(path: Path, expected_step: int) -> dict[str, Any]:
    completion = json.loads((path / "run_complete.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError(f"held-out variance run is incomplete: {path}")
    record = json.loads((path / "heldout_gradient_scalars.json").read_text(encoding="utf-8"))
    if int(record.get("schema_version", -1)) != 1 or int(record["checkpoint_step"]) != expected_step:
        raise ValueError(f"wrong held-out variance artifact at {path}")
    units = record["feedback"]["task_units"]
    if [unit["task"] for unit in units] != list(TASKS):
        raise ValueError(f"held-out variance task order differs from {TASKS}")
    weights = [float(record["target_weights"][task]) for task in TASKS]
    training_state = record["training_controller_state"]
    if int(training_state["checkpoint_step"]) != expected_step:
        raise ValueError(f"wrong training controller state at {path}")
    task_seconds = [float(training_state["task_seconds"][task]) for task in TASKS]
    fixed_seconds = float(training_state["fixed_seconds"])
    halves = [[_task_half(unit, half) for unit in units] for half in (0, 1)]

    folds = []
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for train_half, test_half in ((0, 1), (1, 0)):
        estimates = halves[train_half]
        test_noise = [value["scaled_noise"] for value in halves[test_half]]
        uniform_variance = _variance(weights, test_noise, [4, 4, 4, 4])
        if uniform_variance <= 0:
            raise ValueError(f"Uniform held-out variance is zero at step {expected_step}")
        fold = {"estimate_half": train_half, "evaluation_half": test_half, "methods": {}}
        for method in METHODS:
            counts = _allocation(method, weights, estimates, task_seconds, fixed_seconds)
            value = _variance(weights, test_noise, counts)
            result = {
                "allocation": dict(zip(TASKS, counts, strict=True)),
                "variance": value,
                "relative_variance": value / uniform_variance,
            }
            fold["methods"][method] = result
            by_method[method].append(result)
        folds.append(fold)

    methods = {
        method: {
            "relative_variance": float(np.mean([fold["relative_variance"] for fold in values])),
            "variance": float(np.mean([fold["variance"] for fold in values])),
            "fold_allocations": [fold["allocation"] for fold in values],
        }
        for method, values in by_method.items()
    }
    diagnostics = {}
    for index, task in enumerate(TASKS):
        unit = units[index]
        task_result: dict[str, Any] = {
            "halves": halves[0][index] | {f"other_{key}": value for key, value in halves[1][index].items()},
            "half_relative_error": {},
            "online_subset_relative_error": {},
        }
        for kind in ("raw", "scaled"):
            first = halves[0][index][f"{kind}_noise"]
            second = halves[1][index][f"{kind}_noise"]
            task_result["half_relative_error"][kind] = abs(first - second) / max((first + second) / 2, 1e-30)
            full = _noise(unit[f"{kind}_microbatch_sq"], unit[f"{kind}_task_mean_sq"])
            task_result[f"{kind}_noise_32"] = full
            task_result["online_subset_relative_error"][kind] = {
                str(size): float(
                    np.mean([abs(_subset_noise(unit, half, size, kind) - full) / max(full, 1e-30) for half in (0, 1)])
                )
                for size in (2, 4)
            }
        diagnostics[task] = task_result
    return {
        **methods,
        "folds": folds,
        "diagnostics": diagnostics,
        "task_seconds": dict(zip(TASKS, task_seconds, strict=True)),
        "fixed_seconds": fixed_seconds,
    }


def _write_table(path: Path, checkpoints: dict[str, dict[str, Any]]) -> None:
    fields = ["checkpoint", *(f"relative_variance_{method}" for method in METHODS)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in CHECKPOINTS:
            writer.writerow(
                {
                    "checkpoint": step,
                    **{
                        f"relative_variance_{method}": checkpoints[str(step)][method]["relative_variance"]
                        for method in METHODS
                    },
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = {str(step): analyze_checkpoint(args.root / f"step_{step:03d}", step) for step in CHECKPOINTS}
    result = {
        "schema_version": 1,
        "seed": 42,
        "checkpoint_steps": list(CHECKPOINTS),
        "microbatches_per_task": 32,
        "cross_fit": "16/16 swap",
        "methods": list(METHODS),
        "checkpoints": checkpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = args.output.with_name("heldout_gradient_variance.csv")
    _write_table(table, checkpoints)
    result["table_csv"] = str(table.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Held-out gradient variance report written to {args.output}")


if __name__ == "__main__":
    main()
