#!/usr/bin/env python3
"""Apply the preregistered half-to-full prompt-budget gain and tail-slope rule."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def command_value(command: list[str], flag: str, default: Any = None) -> Any:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def estimate(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return {"n": len(array), "mean": mean, "sd": None, "ci95_low": None, "ci95_high": None}
    sd = float(array.std(ddof=1))
    critical = T_CRITICAL_95.get(len(array) - 1, 1.96)
    half = critical * sd / math.sqrt(len(array))
    return {"n": len(array), "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def inspect_run(run: Path, metric: str, comparison_step: int, target_step: int) -> dict[str, Any]:
    manifest_path = run / "provenance" / "run_manifest.json"
    marker_path = run / "run_complete.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else {}
    command = manifest.get("command") or []
    errors = []
    if manifest.get("status") != "complete" or marker.get("status") != "complete":
        errors.append("run is incomplete")
    if int(marker.get("final_num_updates", -1)) != target_step:
        errors.append(f"run does not end at target update {target_step}")
    if (run / "run_failed.json").exists():
        errors.append("run has a failure marker")
    points: dict[int, float] = {}
    for event in jsonl(run / "metrics" / "eval.jsonl"):
        values = event.get("metrics") or {}
        step = values.get("eval/num_updates", values.get("eval/step"))
        value = values.get(metric)
        if step is not None and value is not None:
            if math.isfinite(float(value)):
                points[int(step)] = float(value)
            else:
                errors.append(f"non-finite {metric} at update {step}")
    if comparison_step not in points or target_step not in points:
        errors.append(f"metric must contain updates {comparison_step} and {target_step}")
    tail_steps = sorted(step for step in points if step <= target_step)[-3:]
    if len(tail_steps) < 3 or tail_steps[-1] != target_step:
        errors.append("fewer than three tail evaluation points ending at target")
        slope = None
    else:
        x = np.asarray(tail_steps, dtype=np.float64)
        y = np.asarray([points[step] for step in tail_steps], dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
    algorithm = command_value(command, "--experiment-condition")
    optimizer = command_value(command, "--experiment-optimizer")
    if optimizer not in ("adamw", "adam"):
        errors.append(f"budget pilot must use AdamW, got {optimizer}")
    return {
        "run_dir": str(run.resolve()),
        "algorithm": algorithm,
        "optimizer": optimizer,
        "seed": int(command_value(command, "--seed", -1)),
        "eval_steps": sorted(points),
        "gain": (
            points[target_step] - points[comparison_step]
            if target_step in points and comparison_step in points
            else None
        ),
        "tail_steps": tail_steps,
        "tail_slope_per_update": slope,
        "stable": not errors,
        "errors": errors,
    }


def decide(
    runs: list[dict[str, Any]],
    required_seeds: int,
    gain_threshold: float,
    required_algorithms: list[str] | None = None,
    target_step: int = 3200,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[str(run["algorithm"])].append(run)
    decisions = []
    algorithms = sorted(set(required_algorithms or groups))
    for algorithm in algorithms:
        values = groups.get(algorithm, [])
        stable = [run for run in values if run["stable"]]
        seeds = [run["seed"] for run in stable]
        grids = {tuple(run["eval_steps"]) for run in stable}
        complete = len(set(seeds)) >= required_seeds and len(seeds) == len(set(seeds)) and len(grids) == 1
        gains = estimate([float(run["gain"]) for run in stable]) if stable else None
        slopes = estimate([float(run["tail_slope_per_update"]) for run in stable]) if stable else None
        slope_not_clearly_positive = bool(
            slopes
            and slopes["ci95_low"] is not None
            and slopes["ci95_low"] <= 0
        )
        choose_target = bool(complete and gains and gains["mean"] < gain_threshold and slope_not_clearly_positive)
        decisions.append(
            {
                "algorithm": algorithm,
                "seeds": sorted(seeds),
                "complete": complete,
                "gain_comparison_to_target": gains,
                "tail_slope_per_update": slopes,
                "tail_slope_not_clearly_positive": slope_not_clearly_positive,
                "recommended_steps": target_step if choose_target else 2 * target_step,
            }
        )
    return {
        "algorithm_decisions": decisions,
        "recommended_common_steps": max((row["recommended_steps"] for row in decisions), default=None),
        "rule": (
            f"use {target_step} updates only when every algorithm has mean half-to-full-budget gain below "
            f"threshold and its tail-slope 95% CI is not wholly positive; otherwise use {2 * target_step}"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--metric", required=True, help="Exact durable eval key, e.g. eval/tuning")
    parser.add_argument("--comparison-step", type=int, default=1600)
    parser.add_argument("--target-step", type=int, default=3200)
    parser.add_argument("--gain-threshold", type=float, default=0.01)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--required-algorithms", nargs="+", default=["grpo", "ppo", "opd"])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.required_seeds < 2:
        raise SystemExit("--required-seeds must be at least 2")
    if not 0 < args.comparison_step < args.target_step:
        raise SystemExit("Require 0 < comparison-step < target-step")
    run_dirs = sorted(path.parent.parent for path in args.root.rglob("provenance/run_manifest.json"))
    runs = [inspect_run(run, args.metric, args.comparison_step, args.target_step) for run in run_dirs]
    result = {
        "schema_version": 2,
        "metric": args.metric,
        "comparison_step": args.comparison_step,
        "target_step": args.target_step,
        "gain_threshold": args.gain_threshold,
        "runs": runs,
        **decide(runs, args.required_seeds, args.gain_threshold, args.required_algorithms, args.target_step),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["recommended_common_steps"] is None or any(
        not row["complete"] for row in result["algorithm_decisions"]
    ):
        raise SystemExit("Incomplete or unmatched budget pilot; no valid recommendation.")
