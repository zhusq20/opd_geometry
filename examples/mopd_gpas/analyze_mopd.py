#!/usr/bin/env python3
"""Build response/GPU-hour curves and the complete eight-configuration outcome table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("math", "code", "if", "science")
SEED = 42
CONFIGS = (
    "uniform_k1_conventional",
    "uniform_k1_taskwise",
    "gpas_k1_taskwise",
    "cost_gpas_k1_taskwise",
    "uniform_k2_taskwise",
    "cost_gpas_k2_taskwise",
    "all_k4_taskwise",
    "all_k4_conventional",
)
THRESHOLD = 0.75
INITIAL_LOSSES = {
    task: float(value)
    for task, value in json.loads(
        (Path(__file__).resolve().parent / "configs/initial_teacher_losses.json").read_text(encoding="utf-8")
    )["values"].items()
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _metric_rows(path: Path) -> list[dict[str, Any]]:
    return [row["metrics"] for row in jsonl(path) if "metrics" in row]


def _completion(path: Path) -> dict[str, Any]:
    marker = path / "run_complete.json"
    if not marker.is_file():
        raise ValueError(f"incomplete run: {path}")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"failed run: {path}")
    return value


def _evaluation(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout_id": int(metrics["eval/rollout_id"]),
        "optimizer_updates": int(metrics["eval/num_updates"]),
        "mean_relative_loss": float(metrics["eval/mean_relative_teacher_loss"]),
        "relative_losses": {task: float(metrics[f"eval/relative_teacher_loss/{task}"]) for task in TASKS},
        "raw_losses": {task: float(metrics[f"eval/teacher_loss/{task}"]) for task in TASKS},
    }


def _interpolate(curve: list[dict[str, Any]], coordinate: str, threshold: float) -> float | None:
    for index, row in enumerate(curve):
        if row["mean_relative_loss"] > threshold:
            continue
        if index == 0:
            return float(row[coordinate])
        previous = curve[index - 1]
        high, low = float(previous["mean_relative_loss"]), float(row["mean_relative_loss"])
        fraction = 1.0 if high == low else np.clip((high - threshold) / (high - low), 0.0, 1.0)
        return float(previous[coordinate] + fraction * (row[coordinate] - previous[coordinate]))
    return None


def _system_trace(allocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "operation_index": int(row["operation_index"]),
            "rollout_id": int(row["rollout_id"]),
            "attempted_responses_before": int(row["attempted_responses_before"]),
            "attempted_responses_after": int(row["attempted_responses_after"]),
            "operation": str(row["operation"]),
            "selected_set": list(row["selected_set"]),
            "execution_order": list(row["execution_order"]),
            "resident_teacher_before": row["resident_teacher_before"],
            "resident_teacher_after": row["resident_teacher_after"],
            "optimizer_updates_after": int(row["optimizer_updates_after"]),
            "processed_task_units_after": int(row["processed_task_units_after"]),
            "probe_count_after": int(row["probe_count_after"]),
            "inclusion_probabilities": {task: float(row["inclusion_probabilities"][task]) for task in TASKS},
            "set_distribution": {
                str(task_set): float(probability) for task_set, probability in row["set_distribution"].items()
            },
            "score_ages": {task: int(row["score_ages_after"][task]) for task in TASKS},
            "raw_gradient_rms": {task: float(row["raw_gradient_rms_after"][task]) for task in TASKS},
            "adam_gradient_rms": {task: float(row["adam_gradient_rms_after"][task]) for task in TASKS},
            "predicted_set_seconds": {
                str(task_set): float(seconds) for task_set, seconds in row["predicted_set_seconds"].items()
            },
            "allocated_gpu_count": int(row["feedback"]["allocated_gpu_count"]),
            "total_wall_seconds": float(row["feedback"]["total_wall_seconds"]),
            "total_gpu_seconds": float(row["feedback"]["total_gpu_seconds"]),
            "active_gpu_seconds": float(row["feedback"]["active_gpu_seconds"]),
            "valid_response_tokens": int(row["feedback"]["valid_response_tokens"]),
            "completed_responses": int(row["feedback"]["completed_responses"]),
            "invalid_responses": int(row["feedback"]["invalid_responses"]),
            "truncated_responses": int(row["feedback"]["truncated_responses"]),
            "component_gpu_seconds": {
                key: float(row["feedback"][key])
                for key in (
                    "rollout_gpu_seconds",
                    "teacher_gpu_seconds",
                    "actor_forward_backward_gpu_seconds",
                    "optimizer_gpu_seconds",
                )
            },
            "component_wall_seconds": {
                key: float(row["feedback"][key])
                for key in (
                    "rollout_wall_seconds",
                    "teacher_wall_seconds",
                    "rollout_and_teacher_wall_seconds",
                    "actor_forward_backward_wall_seconds",
                    "optimizer_wall_seconds",
                )
            },
            "student_peak_hbm_gib": float(row["feedback"]["peak_hbm_bytes"]) / 2**30,
            "teacher_peak_hbm_gib": float(row["feedback"]["teacher_peak_memory_mib"]) / 1024,
            "task_units": [
                {
                    **{
                        key: unit[key]
                        for key in (
                            "task",
                            "switched",
                            "teacher_switch_seconds",
                            "teacher_transfer_tail_seconds",
                            "teacher_memory_mib",
                            "student_rollout_seconds",
                            "teacher_scoring_seconds",
                            "actor_forward_backward_wall_seconds",
                            "optimizer_wall_seconds",
                        )
                    },
                    "student_peak_hbm_gib": float(unit["student_peak_hbm_bytes"]) / 2**30,
                }
                for unit in row["feedback"]["task_units"]
            ],
        }
        for row in allocations
    ]


def _load_warm(root: Path) -> tuple[dict[str, Any], float, int, list[dict[str, Any]]]:
    path = root / f"warm_start-seed{SEED}"
    _completion(path)
    allocations = jsonl(path / "allocation/allocation.jsonl")
    if len(allocations) != 8 or int(allocations[-1]["attempted_responses_after"]) != 512:
        raise ValueError("warm start must contain eight task units and 512 attempted responses")
    gpu_hours = sum(float(row["feedback"]["total_gpu_seconds"]) for row in allocations) / 3600.0
    tokens = sum(int(row["feedback"]["valid_response_tokens"]) for row in allocations)
    evals = [
        _evaluation(row)
        for row in _metric_rows(path / "metrics/eval.jsonl")
        if "eval/mean_relative_teacher_loss" in row
    ]
    if len(evals) != 1:
        raise ValueError("warm start must have exactly one final held-out evaluation")
    return evals[0], gpu_hours, tokens, _system_trace(allocations)


def load_run(
    root: Path,
    config: str,
    warm_eval: dict[str, Any],
    warm_gpu: float,
    warm_tokens: int,
    response_budget: int,
) -> dict[str, Any]:
    path = root / f"{config}-seed{SEED}"
    _completion(path)
    allocations = jsonl(path / "allocation/allocation.jsonl")
    if not allocations or [int(row["operation_index"]) for row in allocations] != list(range(len(allocations))):
        raise ValueError(f"non-contiguous allocation log: {path}")
    if int(allocations[0]["attempted_responses_before"]) != 512:
        raise ValueError(f"{config} did not branch from the common post-warm response clock")
    if int(allocations[-1]["attempted_responses_after"]) != response_budget:
        raise ValueError(f"{config} did not reach exactly {response_budget} attempted responses")

    cumulative_gpu = warm_gpu
    cumulative_tokens = warm_tokens
    coordinates: dict[int, tuple[int, float, int]] = {}
    for row in allocations:
        feedback = row["feedback"]
        cumulative_gpu += float(feedback["total_gpu_seconds"]) / 3600.0
        cumulative_tokens += int(feedback["valid_response_tokens"])
        coordinates[int(row["rollout_id"])] = (
            int(row["attempted_responses_after"]),
            cumulative_gpu,
            cumulative_tokens,
        )

    evaluations = [
        _evaluation(row)
        for row in _metric_rows(path / "metrics/eval.jsonl")
        if "eval/mean_relative_teacher_loss" in row
    ]
    curve = [
        {
            "attempted_responses": 0,
            "gpu_hours": 0.0,
            "valid_response_tokens": 0,
            "mean_relative_loss": 1.0,
            "relative_losses": {task: 1.0 for task in TASKS},
            "raw_losses": dict(INITIAL_LOSSES),
        },
        {
            "attempted_responses": 512,
            "gpu_hours": warm_gpu,
            "valid_response_tokens": warm_tokens,
            **{key: warm_eval[key] for key in ("mean_relative_loss", "relative_losses", "raw_losses")},
        },
    ]
    for evaluation in evaluations:
        responses, gpu_hours, tokens = coordinates[evaluation["rollout_id"]]
        curve.append(
            {
                "attempted_responses": responses,
                "gpu_hours": gpu_hours,
                "valid_response_tokens": tokens,
                **{key: evaluation[key] for key in ("mean_relative_loss", "relative_losses", "raw_losses")},
            }
        )
    if curve[-1]["attempted_responses"] != response_budget:
        raise ValueError(f"{config} is missing its final response-clock evaluation")

    feedbacks = [row["feedback"] for row in allocations]
    units = [unit for feedback in feedbacks for unit in feedback["task_units"]]
    final = curve[-1]
    total_gpu = final["gpu_hours"]
    probe_feedback = [feedback for feedback in feedbacks if feedback["operation"] == "probe"]
    last_train = next((row for row in reversed(allocations) if row["operation"] == "train"), None)
    if last_train is None:
        raise ValueError(f"{config} contains no optimizer update")
    final_inclusion = last_train["inclusion_probabilities"]
    system_trace = _system_trace(allocations)
    endpoint = {
        "mean_relative_loss": final["mean_relative_loss"],
        "relative_losses": final["relative_losses"],
        "raw_losses": final["raw_losses"],
        "attempted_responses": response_budget,
        "gpu_hours": total_gpu,
        "responses_per_gpu_hour": response_budget / total_gpu,
        "tokens_per_gpu_hour": final["valid_response_tokens"] / total_gpu,
        "peak_hbm_gib": max(
            max(float(feedback["peak_hbm_bytes"]) / 2**30 for feedback in feedbacks),
            max(float(feedback["teacher_peak_memory_mib"]) / 1024 for feedback in feedbacks),
        ),
        "switch_rate": float(np.mean([bool(unit["switched"]) for unit in units])),
        "transfer_tail_p95_seconds": float(
            np.percentile([float(unit["teacher_transfer_tail_seconds"]) for unit in units], 95)
        ),
        "operation_time_p50_seconds": float(np.percentile([row["total_step_seconds"] for row in feedbacks], 50)),
        "operation_time_p95_seconds": float(np.percentile([row["total_step_seconds"] for row in feedbacks], 95)),
        "probe_count": len(probe_feedback),
        "probe_gpu_hours": sum(float(row["total_gpu_seconds"]) for row in probe_feedback) / 3600.0,
        "max_score_age": max(max(map(int, row["score_ages_after"].values())) for row in allocations),
        "final_train_operation_index": int(last_train["operation_index"]),
        "final_inclusion_probabilities": final_inclusion,
        "final_importance_multipliers": {task: 0.25 / float(final_inclusion[task]) for task in TASKS},
        "responses_to_threshold": _interpolate(curve, "attempted_responses", THRESHOLD),
        "gpu_hours_to_threshold": _interpolate(curve, "gpu_hours", THRESHOLD),
    }
    return {
        "path": str(path),
        "curve": curve,
        "endpoint": endpoint,
        "system_trace": system_trace,
        "final_rollout_id": int(allocations[-1]["rollout_id"]),
    }


def _final_prompt_losses(run: dict[str, Any]) -> dict[str, tuple[list[str], np.ndarray]]:
    path = Path(run["path"])
    matches = [
        row
        for row in jsonl(path / "teacher_loss_eval/index.jsonl")
        if int(row["rollout_id"]) == int(run["final_rollout_id"])
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one final eval artifact index under {path}")
    output = {}
    for task in TASKS:
        records = jsonl(Path(matches[0]["datasets"][task]["path"]))
        records.sort(key=lambda row: int(row["prompt_index"]))
        if len(records) != 191 or [int(row["prompt_index"]) for row in records] != list(range(191)):
            raise ValueError(f"paired bootstrap requires 191 ordered prompts for {task}: {path}")
        prompts = [str(row["prompt"]) for row in records]
        values = np.asarray([float(row["metadata"]["relative_teacher_loss"]) for row in records], dtype=np.float64)
        output[task] = prompts, values
    return output


def _paired_bootstrap(runs: dict[str, dict[str, Any]], replicates: int = 10_000) -> dict[str, Any]:
    values = {config: _final_prompt_losses(run) for config, run in runs.items()}
    baseline = CONFIGS[0]
    for task in TASKS:
        reference = values[baseline][task][0]
        if any(values[config][task][0] != reference for config in CONFIGS[1:]):
            raise ValueError(f"final held-out prompts are not paired for task {task}")
    rng = np.random.default_rng(SEED)
    resamples = {task: rng.integers(0, 191, size=(replicates, 191)) for task in TASKS}
    task_bootstraps = {}
    for config in CONFIGS:
        task_bootstraps[config] = {task: values[config][task][1][resamples[task]].mean(axis=1) for task in TASKS}
    bootstraps = {config: sum(task_bootstraps[config].values()) / len(TASKS) for config in CONFIGS}
    report = {"replicates": replicates, "seed": SEED, "baseline": baseline, "configs": {}}
    for config in CONFIGS:
        samples = bootstraps[config]
        delta = samples - bootstraps[baseline]
        report["configs"][config] = {
            "mean_relative_loss_95ci": list(map(float, np.percentile(samples, [2.5, 97.5]))),
            "paired_delta_vs_baseline": float(np.mean(delta)),
            "paired_delta_95ci": list(map(float, np.percentile(delta, [2.5, 97.5]))),
            "worst_task_loss_95ci": list(
                map(
                    float,
                    np.percentile(
                        np.max(np.column_stack(list(task_bootstraps[config].values())), axis=1),
                        [2.5, 97.5],
                    ),
                )
            ),
            "tasks": {
                task: {
                    "relative_loss_95ci": list(map(float, np.percentile(task_bootstraps[config][task], [2.5, 97.5]))),
                    "paired_delta_vs_baseline": float(
                        np.mean(task_bootstraps[config][task] - task_bootstraps[baseline][task])
                    ),
                    "paired_delta_95ci": list(
                        map(
                            float,
                            np.percentile(
                                task_bootstraps[config][task] - task_bootstraps[baseline][task],
                                [2.5, 97.5],
                            ),
                        )
                    ),
                }
                for task in TASKS
            },
        }
    return report


def _write_outcome_table(path: Path, outcomes: dict[str, dict[str, Any]]) -> None:
    fields = [
        "config",
        "final_mean_relative_loss",
        *(f"relative_loss_{task}" for task in TASKS),
        "responses_to_threshold",
        "gpu_hours_to_threshold",
        "gpu_hours",
        "responses_per_gpu_hour",
        "tokens_per_gpu_hour",
        "peak_hbm_gib",
        "switch_rate",
        "transfer_tail_p95_seconds",
        "operation_time_p50_seconds",
        "operation_time_p95_seconds",
        "probe_count",
        "probe_gpu_hours",
        "max_score_age",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for config in CONFIGS:
            outcome = outcomes[config]
            writer.writerow(
                {
                    "config": config,
                    "final_mean_relative_loss": outcome["mean_relative_loss"],
                    **{f"relative_loss_{task}": outcome["relative_losses"][task] for task in TASKS},
                    **{field: outcome[field] for field in fields[6:]},
                }
            )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-budget", type=int, default=64_000)
    args = parser.parse_args()
    warm_eval, warm_gpu, warm_tokens, warm_trace = _load_warm(args.root)
    runs = {
        config: load_run(args.root, config, warm_eval, warm_gpu, warm_tokens, args.response_budget)
        for config in CONFIGS
    }
    paired_bootstrap = _paired_bootstrap(runs)
    result = {
        "schema_version": 2,
        "seed": SEED,
        "response_budget": args.response_budget,
        "configs": list(CONFIGS),
        "common_threshold": THRESHOLD,
        "warm_start": {
            "initial_teacher_losses": dict(INITIAL_LOSSES),
            "attempted_responses": 512,
            "gpu_hours": warm_gpu,
            "mean_relative_loss": warm_eval["mean_relative_loss"],
            "system_trace": warm_trace,
        },
        "curves": {config: run["curve"] for config, run in runs.items()},
        "outcomes": {config: run["endpoint"] for config, run in runs.items()},
        "system_traces": {
            "warm_start": warm_trace,
            **{config: run["system_trace"] for config, run in runs.items()},
        },
        "paired_bootstrap": paired_bootstrap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outcome_table = args.output.with_name("mopd_outcomes.csv")
    _write_outcome_table(outcome_table, result["outcomes"])
    result["outcome_table_csv"] = str(outcome_table.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MOPD v2 report written to {args.output}")


if __name__ == "__main__":
    main()
