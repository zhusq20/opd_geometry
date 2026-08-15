#!/usr/bin/env python3
"""Select a frozen LR or OPD coefficient from complete disjoint-split runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FLAGS = {
    "algorithm": "--experiment-condition",
    "optimizer": "--experiment-optimizer",
    "seed": "--seed",
    "lr": "--lr",
    "opd_kl_coef": "--opd-kl-coef",
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flag(command: list[str], name: str, default: Any = None) -> Any:
    try:
        return command[command.index(FLAGS[name]) + 1]
    except (ValueError, IndexError):
        return default


def normalized_auc(points: dict[int, float]) -> float | None:
    if len(points) < 2:
        return None
    x = np.asarray(sorted(points), dtype=np.float64)
    y = np.asarray([points[int(step)] for step in x], dtype=np.float64)
    span = float(x[-1] - x[0])
    # Spell out the trapezoid rule: np.trapezoid is absent from NumPy < 2.0,
    # while np.trapz is deprecated in newer releases.
    area = np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5)
    return float(area / span) if span > 0 else None


def inspect_run(run: Path, metric: str, parameter: str, grad_norm_limit: float) -> dict[str, Any]:
    manifest_path = run / "provenance" / "run_manifest.json"
    marker_path = run / "run_complete.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else {}
    command = manifest.get("command") or []
    errors: list[str] = []
    if manifest.get("status") != "complete" or marker.get("status") != "complete":
        errors.append("run is incomplete")
    if (run / "run_failed.json").exists():
        errors.append("run has a failure marker")

    points: dict[int, float] = {}
    for event in jsonl(run / "metrics" / "eval.jsonl"):
        values = event.get("metrics") or {}
        value = values.get(metric)
        step = values.get("eval/num_updates", values.get("eval/step"))
        if value is not None and step is not None:
            if math.isfinite(float(value)):
                points[int(step)] = float(value)
            else:
                errors.append(f"metric {metric!r} is non-finite at update {step}")
    expected = int(marker.get("final_num_updates", -1))
    if expected < 0 or expected not in points:
        errors.append(f"metric {metric!r} does not reach the final update")

    grad_norms: list[float] = []
    clipped: list[int] = []
    nonfinite = False
    for event in jsonl(run / "metrics" / "train.jsonl"):
        values = event.get("metrics") or {}
        if "train/grad_norm" in values:
            value = values["train/grad_norm"]
            if value is None or not math.isfinite(float(value)):
                nonfinite = True
            else:
                grad_norms.append(float(value))
        if values.get("train/grad_clipped") is not None:
            clipped.append(int(values["train/grad_clipped"]))
    if nonfinite:
        errors.append("non-finite gradient norm")
    if any(all(value > grad_norm_limit for value in grad_norms[start : start + 3]) for start in range(max(0, len(grad_norms) - 2))):
        errors.append(f"three consecutive grad norms exceed {grad_norm_limit:g}")
    if clipped and sum(clipped) / len(clipped) > 0.5:
        errors.append("gradient clipping occurs on more than 50% of updates")

    value = flag(command, parameter)
    if value is None:
        errors.append(f"command is missing {FLAGS[parameter]}")
    return {
        "run_dir": str(run.resolve()),
        "algorithm": flag(command, "algorithm"),
        "optimizer": flag(command, "optimizer"),
        "seed": int(flag(command, "seed", -1)),
        "parameter": parameter,
        "value": float(value) if value is not None else None,
        "auc": normalized_auc(points),
        "final_score": points.get(expected),
        "eval_points": len(points),
        "eval_steps": sorted(points),
        "expected_updates": expected,
        "stable": not errors,
        "errors": errors,
    }


def candidate_rows(runs: list[dict[str, Any]], required_seeds: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["algorithm"], run["optimizer"], run["value"])].append(run)
    comparison_groups: dict[tuple[Any, Any], list[list[dict[str, Any]]]] = defaultdict(list)
    for (algorithm, optimizer, _value), values in groups.items():
        comparison_groups[(algorithm, optimizer)].append(values)
    rows = []
    for (algorithm, optimizer, value), values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        candidate_sets = comparison_groups[(algorithm, optimizer)]
        stable_seed_sets = [
            {row["seed"] for row in candidate if row["stable"] and row["auc"] is not None}
            for candidate in candidate_sets
        ]
        shared_seeds = set.intersection(*stable_seed_sets) if stable_seed_sets else set()
        valid = [
            row
            for row in values
            if row["stable"] and row["auc"] is not None and row["seed"] in shared_seeds
        ]
        seed_counts = {seed: sum(row["seed"] == seed for row in valid) for seed in shared_seeds}
        duplicate_seeds = sorted(seed for seed, count in seed_counts.items() if count != 1)
        matched_runs = [
            row
            for candidate in candidate_sets
            for row in candidate
            if row["stable"] and row["auc"] is not None and row["seed"] in shared_seeds
        ]
        eval_grids = {tuple(row["eval_steps"]) for row in matched_runs}
        update_budgets = {row["expected_updates"] for row in matched_runs}
        aucs = np.asarray([row["auc"] for row in valid], dtype=np.float64)
        complete = (
            len(shared_seeds) >= required_seeds
            and not duplicate_seeds
            and len(eval_grids) == 1
            and len(update_budgets) == 1
        )
        mean = float(aucs.mean()) if len(aucs) else None
        sd = float(aucs.std(ddof=1)) if len(aucs) > 1 else None
        se = sd / math.sqrt(len(aucs)) if sd is not None else None
        rows.append(
            {
                "algorithm": algorithm,
                "optimizer": optimizer,
                "value": value,
                "n": len(aucs),
                "seeds": sorted(row["seed"] for row in valid),
                "shared_seed_set": sorted(shared_seeds),
                "eval_grid": list(next(iter(eval_grids))) if len(eval_grids) == 1 else None,
                "expected_updates": next(iter(update_budgets)) if len(update_budgets) == 1 else None,
                "eligibility_errors": [
                    *([f"only {len(shared_seeds)} shared stable seeds"] if len(shared_seeds) < required_seeds else []),
                    *([f"duplicate runs for seeds {duplicate_seeds}"] if duplicate_seeds else []),
                    *(["eval step grids differ"] if len(eval_grids) != 1 else []),
                    *(["final update budgets differ"] if len(update_budgets) != 1 else []),
                ],
                "auc_mean": mean,
                "auc_sd": sd,
                "auc_se": se,
                "complete": complete,
            }
        )
    return rows


def selections(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["complete"] and row["auc_mean"] is not None:
            groups[(row["algorithm"], row["optimizer"])].append(row)
    selected = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        best = max(rows, key=lambda row: row["auc_mean"])
        tied = []
        for row in rows:
            combined_se = math.sqrt((best["auc_se"] or 0.0) ** 2 + (row["auc_se"] or 0.0) ** 2)
            if best["auc_mean"] - row["auc_mean"] <= combined_se:
                tied.append(row)
        choice = min(tied, key=lambda row: row["value"])
        selected.append(
            {
                "algorithm": key[0],
                "optimizer": key[1],
                "selected_value": choice["value"],
                "selected_auc_mean": choice["auc_mean"],
                "best_observed_value": best["value"],
                "rule": "maximum mean validation AUC; choose smaller value when within one combined SE",
            }
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--metric", required=True, help="Exact durable eval key, e.g. eval/tuning")
    parser.add_argument("--parameter", choices=("lr", "opd_kl_coef"), default="lr")
    parser.add_argument("--required-seeds", type=int, default=2)
    parser.add_argument("--grad-norm-limit", type=float, default=1e4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.required_seeds < 2:
        raise SystemExit("--required-seeds must be at least 2")
    run_dirs = sorted(path.parent.parent for path in args.root.rglob("provenance/run_manifest.json"))
    runs = [inspect_run(run, args.metric, args.parameter, args.grad_norm_limit) for run in run_dirs]
    candidates = candidate_rows(runs, args.required_seeds)
    result = {
        "schema_version": 1,
        "metric": args.metric,
        "parameter": args.parameter,
        "required_seeds": args.required_seeds,
        "runs": runs,
        "candidates": candidates,
        "selections": selections(candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not result["selections"]:
        raise SystemExit("No complete stable candidate group could be selected.")
