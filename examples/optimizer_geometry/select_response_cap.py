#!/usr/bin/env python3
"""Select the smallest response cap meeting a preregistered truncation bound."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def command_value(command: list[str], flag: str, default: Any = None) -> Any:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def inspect_run(run: Path) -> dict[str, Any]:
    manifest_path = run / "provenance" / "run_manifest.json"
    marker_path = run / "run_complete.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else {}
    command = manifest.get("command") or []
    errors = []
    if manifest.get("status") != "complete" or marker.get("status") != "complete":
        errors.append("run is incomplete")
    if (run / "run_failed.json").exists():
        errors.append("run has a failure marker")
    cap = command_value(command, "--rollout-max-response-len")
    seed = command_value(command, "--seed")
    metrics: dict[str, Any] = {}
    for event in jsonl(run / "metrics" / "rollout.jsonl"):
        metrics.update(event.get("metrics") or {})
    required = ("rollout/truncated_ratio", "rollout/response_len/p95", "perf/rollout_time")
    for key in required:
        value = metrics.get(key)
        if value is None or not math.isfinite(float(value)):
            errors.append(f"missing or non-finite {key}")
    return {
        "run_dir": str(run.resolve()),
        "cap": int(cap) if cap is not None else None,
        "seed": int(seed) if seed is not None else None,
        "truncated_ratio": metrics.get("rollout/truncated_ratio"),
        "response_p95": metrics.get("rollout/response_len/p95"),
        "response_mean": metrics.get("rollout/response_len/mean"),
        "reward_mean": metrics.get("rollout/reward/mean"),
        "pass_at_1": metrics.get("passrate/pass@1"),
        "rollout_time_seconds": metrics.get("perf/rollout_time"),
        "step_time_seconds": metrics.get("perf/step_time"),
        "stable": not errors,
        "errors": errors,
    }


def select(runs: list[dict[str, Any]], required_seeds: int, max_truncation: float) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["cap"] is not None:
            groups[run["cap"]].append(run)
    stable_sets = [
        {run["seed"] for run in values if run["stable"]}
        for values in groups.values()
    ]
    shared_seeds = set.intersection(*stable_sets) if stable_sets else set()
    candidates = []
    for cap, values in sorted(groups.items()):
        matched = [run for run in values if run["stable"] and run["seed"] in shared_seeds]
        truncation = [float(run["truncated_ratio"]) for run in matched]
        duplicate = len(matched) != len({run["seed"] for run in matched})
        complete = len(shared_seeds) >= required_seeds and not duplicate
        mean = sum(truncation) / len(truncation) if truncation else None
        eligible = complete and mean is not None and mean <= max_truncation
        candidates.append(
            {
                "cap": cap,
                "seeds": sorted(run["seed"] for run in matched),
                "complete": complete,
                "mean_truncated_ratio": mean,
                "max_seed_truncated_ratio": max(truncation) if truncation else None,
                "mean_response_p95": (
                    sum(float(run["response_p95"]) for run in matched) / len(matched) if matched else None
                ),
                "mean_rollout_time_seconds": (
                    sum(float(run["rollout_time_seconds"]) for run in matched) / len(matched) if matched else None
                ),
                "eligible": eligible,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    return {
        "shared_seed_set": sorted(shared_seeds),
        "max_mean_truncated_ratio": max_truncation,
        "candidates": candidates,
        "selected_cap": min(candidate["cap"] for candidate in eligible) if eligible else None,
        "rule": "smallest complete cap whose matched-seed mean truncation ratio is at most the threshold",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--required-seeds", type=int, default=2)
    parser.add_argument("--max-truncation", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.required_seeds < 2:
        raise SystemExit("--required-seeds must be at least 2")
    if not 0 <= args.max_truncation < 1:
        raise SystemExit("--max-truncation must be in [0, 1)")
    run_dirs = sorted(path.parent.parent for path in args.root.rglob("provenance/run_manifest.json"))
    runs = [inspect_run(run) for run in run_dirs]
    result = {"schema_version": 1, "runs": runs, **select(runs, args.required_seeds, args.max_truncation)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["selected_cap"] is None:
        raise SystemExit("No response cap met the preregistered truncation threshold.")
