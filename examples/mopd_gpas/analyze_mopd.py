#!/usr/bin/env python3
"""Analyze the frozen eight-run, 500-step MOPD experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("math", "code", "if", "science")
CONFIGS = (
    "uniform",
    "gpas",
    "cost_gpas",
    "raw_noise",
    "loss_gap",
    "std_mopd",
    "d3_mopd",
    "open_mopd",
)
NON_FIXED_OBJECTIVE = {"std_mopd", "d3_mopd", "open_mopd"}
SEED = 42
STEPS = tuple(range(0, 501, 50))
RESPONSES_PER_STEP = 64
RESPONSE_BUDGET = 32_000
BOOTSTRAP_REPLICATES = 1_000
BASELINE = "uniform"


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value["schema_version"]) != 4 or tuple(value["training"]["configs"]) != CONFIGS:
        raise ValueError(f"unexpected MOPD protocol: {path}")
    weights = {task: float(value["objective"]["weights"][task]) for task in TASKS}
    losses = {task: float(value["initial_kl"]["tasks"][task]["ell0"]) for task in TASKS}
    return weights, losses


def _completion(path: Path) -> None:
    value = json.loads((path / "run_complete.json").read_text(encoding="utf-8"))
    if value.get("status") != "complete" or int(value["final_num_updates"]) != 500:
        raise ValueError(f"incomplete 500-step run: {path}")


def _evaluations(path: Path, weights: dict[str, float]) -> dict[int, dict[str, Any]]:
    rows = [row["metrics"] for row in jsonl(path / "metrics/eval.jsonl") if "metrics" in row]
    output: dict[int, dict[str, Any]] = {}
    for metrics in rows:
        if "eval/weighted_teacher_loss" not in metrics:
            continue
        step = int(metrics["eval/num_updates"])
        raw = {task: float(metrics[f"eval/teacher_loss/{task}"]) for task in TASKS}
        output[step] = {
            "step": step,
            "raw_losses": raw,
            "normalized_losses": {task: float(metrics[f"eval/normalized_teacher_loss/{task}"]) for task in TASKS},
            "weighted_loss": sum(weights[task] * raw[task] for task in TASKS),
        }
    if tuple(sorted(output)) != STEPS:
        raise ValueError(f"expected held-out evaluations at steps {STEPS}, got {tuple(sorted(output))}: {path}")
    return output


def _system_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace = []
    for row in rows:
        feedback = row["feedback"]
        counts = {task: int(row["counts"][task]) for task in TASKS}
        task_seconds_ema = {task: float(row["task_seconds_after"][task]) for task in TASKS}
        variable_ema = sum(counts[task] * task_seconds_ema[task] for task in TASKS)
        fixed_ema = float(row["fixed_seconds_after"])
        before_task_seconds = row["task_seconds_before"]
        before_fixed_seconds = row["fixed_seconds_before"]
        predicted = None
        if before_fixed_seconds is not None and all(before_task_seconds[task] is not None for task in TASKS):
            predicted = float(before_fixed_seconds) + sum(
                counts[task] * float(before_task_seconds[task]) for task in TASKS
            )
        total_wall_seconds = float(feedback["total_wall_seconds"])
        trace.append(
            {
                "step": int(row["optimizer_updates_after"]),
                "attempted_responses": int(row["attempted_responses_after"]),
                "counts": counts,
                "H": float(row["H"]),
                "raw_noise": {task: float(row["raw_noise_after"][task]) for task in TASKS},
                "scaled_noise": {task: float(row["scaled_noise_after"][task]) for task in TASKS},
                "loss_ema": {task: float(row["loss_ema_after"][task]) for task in TASKS},
                "task_seconds_ema": task_seconds_ema,
                "fixed_seconds_ema": fixed_ema,
                "fixed_to_variable_ratio": fixed_ema / variable_ema,
                "time_model_prediction_before": predicted,
                "time_model_residual_seconds": (None if predicted is None else total_wall_seconds - predicted),
                "aggregate_grad_norm": float(feedback["aggregate_grad_norm"]),
                "aggregate_grad_clipped": bool(feedback["aggregate_grad_clipped"]),
                "total_wall_seconds": total_wall_seconds,
                "valid_response_tokens": int(feedback["valid_response_tokens"]),
                "generated_tokens": int(feedback["generated_tokens"]),
                "truncated_responses": int(feedback["truncated_responses"]),
                "invalid_responses": int(feedback["invalid_responses"]),
                "student_peak_hbm_gib": float(feedback["peak_hbm_bytes"]) / 2**30,
                "inference_peak_hbm_gib": float(feedback["teacher_peak_memory_mib"]) / 1024,
                "allocation_details": row.get("allocation_details"),
                "open_mopd_weights": {
                    unit["task"]: {
                        key: unit[key]
                        for key in (
                            "open_mopd_token_share",
                            "open_mopd_share_weight",
                            "open_mopd_gap_factor",
                            "open_mopd_loss_weight",
                            "open_mopd_effective_share",
                        )
                        if key in unit
                    }
                    for unit in feedback["task_units"]
                },
            }
        )
    return trace


def _mechanism_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    noise = {}
    for kind in ("raw_noise", "scaled_noise"):
        cross_task_ratios = []
        for row in trace:
            values = [float(row[kind][task]) for task in TASKS]
            cross_task_ratios.append(max(values) / max(min(values), 1e-30))
        all_values = [float(row[kind][task]) for row in trace for task in TASKS]
        noise[kind] = {
            "minimum": min(all_values),
            "maximum": max(all_values),
            "cross_task_ratio_median": float(np.median(cross_task_ratios)),
            "cross_task_ratio_p95": float(np.percentile(cross_task_ratios, 95)),
            "final_over_initial": {
                task: float(trace[-1][kind][task]) / max(float(trace[0][kind][task]), 1e-30) for task in TASKS
            },
        }

    ranking_disagreements = 0
    for row in trace:
        raw_order = sorted(TASKS, key=lambda task: (row["raw_noise"][task], task))
        scaled_order = sorted(TASKS, key=lambda task: (row["scaled_noise"][task], task))
        ranking_disagreements += raw_order != scaled_order

    h_values = [float(row["H"]) for row in trace]
    residuals = [
        float(row["time_model_residual_seconds"]) for row in trace if row["time_model_residual_seconds"] is not None
    ]
    return {
        "noise": noise,
        "raw_scaled_ranking_disagreement_fraction": ranking_disagreements / len(trace),
        "count_at_lower_bound_fraction": {
            task: sum(row["counts"][task] == 2 for row in trace) / len(trace) for task in TASKS
        },
        "count_at_upper_bound_fraction": {
            task: sum(row["counts"][task] == 8 for row in trace) / len(trace) for task in TASKS
        },
        "H": {
            "q25": float(np.percentile(h_values, 25)),
            "median": float(np.median(h_values)),
            "q75": float(np.percentile(h_values, 75)),
        },
        "time_model_forecast_residual_seconds": {
            "q25": float(np.percentile(residuals, 25)),
            "median": float(np.median(residuals)),
            "q75": float(np.percentile(residuals, 75)),
            "absolute_p95": float(np.percentile(np.abs(residuals), 95)),
        },
    }


def _interpolate(curve: list[dict[str, Any]], coordinate: str, threshold: float) -> float | None:
    for index, row in enumerate(curve):
        current = float(row["weighted_loss"])
        if current > threshold:
            continue
        if index == 0:
            return float(row[coordinate])
        previous = curve[index - 1]
        previous_loss = float(previous["weighted_loss"])
        fraction = (
            1.0
            if previous_loss == current
            else np.clip((previous_loss - threshold) / (previous_loss - current), 0.0, 1.0)
        )
        return float(previous[coordinate] + fraction * (row[coordinate] - previous[coordinate]))
    return None


def load_run(root: Path, config: str, weights: dict[str, float]) -> dict[str, Any]:
    path = root / f"{config}-seed{SEED}"
    _completion(path)
    allocations = jsonl(path / "allocation/allocation.jsonl")
    if len(allocations) != 500:
        raise ValueError(f"{config} has {len(allocations)} allocation records, expected 500")
    for index, row in enumerate(allocations):
        if (
            int(row["operation_index"]) != index
            or int(row["optimizer_updates_after"]) != index + 1
            or int(row["attempted_responses_after"]) != (index + 1) * RESPONSES_PER_STEP
        ):
            raise ValueError(f"non-contiguous step/response clock in {path} at record {index}")
    if int(allocations[-1]["attempted_responses_after"]) != RESPONSE_BUDGET:
        raise ValueError(f"{config} did not reach exactly {RESPONSE_BUDGET} responses")

    evaluations = _evaluations(path, weights)
    cumulative_gpu_hours = [0.0]
    cumulative_tokens = [0]
    for row in allocations:
        feedback = row["feedback"]
        wall = float(feedback["total_wall_seconds"])
        if not np.isclose(float(feedback["total_gpu_seconds"]), 2.0 * wall, rtol=1e-6, atol=1e-6):
            raise ValueError(f"GPU-hour accounting is not wall time x 2 in {path}")
        cumulative_gpu_hours.append(cumulative_gpu_hours[-1] + 2.0 * wall / 3600.0)
        cumulative_tokens.append(cumulative_tokens[-1] + int(feedback["valid_response_tokens"]))

    curve = []
    for step in STEPS:
        curve.append(
            {
                **evaluations[step],
                "attempted_responses": step * RESPONSES_PER_STEP,
                "gpu_hours": cumulative_gpu_hours[step],
                "valid_response_tokens": cumulative_tokens[step],
            }
        )
    trace = _system_trace(allocations)
    final = curve[-1]
    outcome = {
        **final,
        "objective_comparable": config not in NON_FIXED_OBJECTIVE,
        "gpu_hours": cumulative_gpu_hours[-1],
        "responses_per_gpu_hour": RESPONSE_BUDGET / cumulative_gpu_hours[-1],
        "tokens_per_gpu_hour": cumulative_tokens[-1] / cumulative_gpu_hours[-1],
        "responses_to_uniform_final": None,
        "gpu_hours_to_uniform_final": None,
        "step_time_p50_seconds": float(np.percentile([row["total_wall_seconds"] for row in trace], 50)),
        "step_time_p95_seconds": float(np.percentile([row["total_wall_seconds"] for row in trace], 95)),
        "H_median": float(np.median([row["H"] for row in trace])),
        "fixed_to_variable_ratio_median": float(np.median([row["fixed_to_variable_ratio"] for row in trace])),
        "truncation_rate": sum(row["truncated_responses"] for row in trace) / RESPONSE_BUDGET,
        "invalid_rate": sum(row["invalid_responses"] for row in trace) / RESPONSE_BUDGET,
        "peak_training_hbm_gib": max(row["student_peak_hbm_gib"] for row in trace),
        "peak_inference_hbm_gib": max(row["inference_peak_hbm_gib"] for row in trace),
        "final_counts": trace[-1]["counts"],
    }
    return {"path": str(path), "curve": curve, "outcome": outcome, "system_trace": trace}


def _artifact_for_step(path: Path, step: int) -> dict[str, Any]:
    matches = [
        row
        for row in jsonl(path / "teacher_loss_eval/index.jsonl")
        if int(row["num_updates"]) == step and row["eval_phase"] == "step_clock"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one step-{step} held-out artifact index under {path}")
    return matches[0]


def _prompt_losses(path: Path, step: int) -> dict[str, tuple[list[str], np.ndarray]]:
    index = _artifact_for_step(path, step)
    output = {}
    for task in TASKS:
        descriptor = index["datasets"][task]
        artifact = Path(descriptor["path"])
        if not artifact.is_file():
            artifact = path / descriptor["run_relative_path"]
        if sha256(artifact) != descriptor["sha256"]:
            raise ValueError(f"held-out artifact hash mismatch: {artifact}")
        records = sorted(jsonl(artifact), key=lambda row: int(row["prompt_index"]))
        if len(records) != 128 or [int(row["prompt_index"]) for row in records] != list(range(128)):
            raise ValueError(f"paired bootstrap requires 128 ordered prompts for {task}: {artifact}")
        prompt_keys = [json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True) for row in records]
        losses = np.asarray([float(row["metadata"]["sampled_reverse_kl"]) for row in records])
        output[task] = prompt_keys, losses
    return output


def paired_bootstrap(
    runs: dict[str, dict[str, Any]], weights: dict[str, float], replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    result: dict[str, Any] = {
        "replicates": replicates,
        "seed": SEED,
        "baseline": BASELINE,
        "checkpoints": {},
    }
    for step in STEPS:
        losses = {config: _prompt_losses(Path(run["path"]), step) for config, run in runs.items()}
        resamples = {task: rng.integers(0, 128, size=(replicates, 128)) for task in TASKS}
        for task in TASKS:
            reference = losses[BASELINE][task][0]
            if any(losses[config][task][0] != reference for config in CONFIGS[1:]):
                raise ValueError(f"held-out prompts are not paired at step {step} for {task}")
        task_samples = {
            config: {task: losses[config][task][1][resamples[task]].mean(axis=1) for task in TASKS}
            for config in CONFIGS
        }
        weighted = {config: sum(weights[task] * task_samples[config][task] for task in TASKS) for config in CONFIGS}
        checkpoint: dict[str, Any] = {}
        for config in CONFIGS:
            tasks = {}
            for task in TASKS:
                samples = task_samples[config][task]
                delta = samples - task_samples[BASELINE][task]
                tasks[task] = {
                    "loss_95ci": list(map(float, np.percentile(samples, [2.5, 97.5]))),
                    "paired_delta_vs_uniform": float(np.mean(delta)),
                    "paired_delta_95ci": list(map(float, np.percentile(delta, [2.5, 97.5]))),
                }
            item: dict[str, Any] = {"objective_comparable": config not in NON_FIXED_OBJECTIVE, "tasks": tasks}
            if config not in NON_FIXED_OBJECTIVE:
                delta = weighted[config] - weighted[BASELINE]
                item.update(
                    {
                        "weighted_loss_95ci": list(map(float, np.percentile(weighted[config], [2.5, 97.5]))),
                        "paired_delta_vs_uniform": float(np.mean(delta)),
                        "paired_delta_95ci": list(map(float, np.percentile(delta, [2.5, 97.5]))),
                    }
                )
            checkpoint[config] = item
        result["checkpoints"][str(step)] = checkpoint
    return result


def _write_outcomes(path: Path, outcomes: dict[str, dict[str, Any]]) -> None:
    fields = [
        "config",
        "objective_comparable",
        "final_weighted_loss",
        *(f"final_loss_{task}" for task in TASKS),
        "responses_to_uniform_final",
        "gpu_hours_to_uniform_final",
        "gpu_hours",
        "responses_per_gpu_hour",
        "tokens_per_gpu_hour",
        "step_time_p50_seconds",
        "step_time_p95_seconds",
        "H_median",
        "fixed_to_variable_ratio_median",
        "truncation_rate",
        "invalid_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for config in CONFIGS:
            outcome = outcomes[config]
            writer.writerow(
                {
                    "config": config,
                    "objective_comparable": outcome["objective_comparable"],
                    "final_weighted_loss": "" if config in NON_FIXED_OBJECTIVE else outcome["weighted_loss"],
                    **{f"final_loss_{task}": outcome["raw_losses"][task] for task in TASKS},
                    **{field: outcome[field] for field in fields[7:]},
                }
            )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(os.environ.get("MOPD_GENERATED_DIR", here.parents[1] / "local/mopd_generated")) / "protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weights, initial_losses = _protocol(args.protocol)
    runs = {config: load_run(args.root, config, weights) for config in CONFIGS}
    threshold = float(runs[BASELINE]["curve"][-1]["weighted_loss"])
    for config, run in runs.items():
        if config in NON_FIXED_OBJECTIVE:
            continue
        run["outcome"]["responses_to_uniform_final"] = _interpolate(run["curve"], "attempted_responses", threshold)
        run["outcome"]["gpu_hours_to_uniform_final"] = _interpolate(run["curve"], "gpu_hours", threshold)

    result = {
        "schema_version": 4,
        "seed": SEED,
        "configs": list(CONFIGS),
        "steps": list(STEPS),
        "response_budget": RESPONSE_BUDGET,
        "responses_per_step": RESPONSES_PER_STEP,
        "target_weights": weights,
        "initial_teacher_losses": initial_losses,
        "common_threshold": threshold,
        "non_fixed_objective_note": (
            "StdMOPD, D3-MOPD, and Open-MOPD change the training objective; compare per-task loss and "
            "capability, not fixed-objective threshold efficiency."
        ),
        "curves": {config: run["curve"] for config, run in runs.items()},
        "outcomes": {config: run["outcome"] for config, run in runs.items()},
        "system_traces": {config: run["system_trace"] for config, run in runs.items()},
        "mechanism_summary": {config: _mechanism_summary(run["system_trace"]) for config, run in runs.items()},
        "paired_bootstrap": paired_bootstrap(runs, weights),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = args.output.with_name("mopd_outcomes.csv")
    _write_outcomes(table, result["outcomes"])
    result["outcome_table_csv"] = str(table.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MOPD protocol-v4 report written to {args.output}")


if __name__ == "__main__":
    main()
