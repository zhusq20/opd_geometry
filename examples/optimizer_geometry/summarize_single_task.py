#!/usr/bin/env python3
"""Join final evaluation reward with global parameter-geometry summaries."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

GEOMETRY_FIELDS = (
    "g_raw_l2",
    "g_opt_l2",
    "delta_intended_fp32_l2",
    "delta_model_l2",
    "displacement_l2",
    "g_raw_to_theta_ratio",
    "delta_intended_to_theta_ratio",
    "delta_model_to_theta_ratio",
    "displacement_to_reference_ratio",
    "gradient_directional_step",
    "cos_g_opt_delta_intended_fp32",
    "cos_delta_intended_fp32_delta_model",
    "cos_delta_model_displacement",
    "model_change_fraction",
    "energy_survival",
    "quantization_residual",
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric_values(events: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for event in events:
        value = (event.get("metrics") or {}).get(key)
        if value is not None:
            values.append(float(value))
    return values


def fixed_grid_auc(points: dict[int, float]) -> tuple[float | None, float | None]:
    """Return raw and update-span-normalized trapezoidal AUC on the saved grid."""

    ordered = sorted(points.items())
    if len(ordered) < 2:
        return None, None
    raw = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value) in pairwise(ordered)
    )
    span = ordered[-1][0] - ordered[0][0]
    return raw, raw / span if span > 0 else None


def final_eval_metrics(run_dir: Path) -> tuple[dict[str, dict[str, Any]], int | None]:
    index = jsonl(run_dir / "eval_artifacts" / "index.jsonl")
    if not index:
        return {}, None
    final_phase = [record for record in index if record.get("eval_phase") == "final"]
    selected = final_phase or index
    final_update = max(int(record.get("num_updates", -1)) for record in selected)
    records = [record for record in selected if int(record.get("num_updates", -1)) == final_update]
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        for dataset, details in (record.get("datasets") or {}).items():
            sample_path = Path(details["path"])
            if not sample_path.is_absolute() and not sample_path.exists():
                sample_path = run_dir / sample_path
            if not sample_path.exists() and details.get("run_relative_path"):
                sample_path = run_dir / details["run_relative_path"]
            samples = jsonl(sample_path)
            rewards = [float(sample["reward"]) for sample in samples if sample.get("reward") is not None]
            n = int(details.get("n_samples_per_prompt", 1) or 1)
            metrics: dict[str, Any] = {"score": sum(rewards) / len(rewards) if rewards else None}
            if rewards and len(rewards) % n == 0:
                from slime.utils.metric_utils import compute_pass_rate

                metrics.update(compute_pass_rate(rewards, group_size=n))
            output[dataset] = metrics
    return output, final_update


def summarize_run(run_dir: Path) -> list[dict[str, Any]]:
    geometry_records = jsonl(run_dir / "geometry" / "actor" / "metrics.jsonl")
    if not geometry_records:
        raise FileNotFoundError(f"No actor geometry metrics found under {run_dir}.")
    first = geometry_records[0]
    last = geometry_records[-1]
    global_records = [record["groups"]["global"] for record in geometry_records if "global" in record["groups"]]
    final_geometry = global_records[-1]
    eval_records = jsonl(run_dir / "forgetting" / "metrics.jsonl")
    final_eval = eval_records[-1].get("tasks", {}) if eval_records else {}
    eval_points: dict[str, dict[int, float]] = {}
    for record in eval_records:
        update = int(record.get("num_updates", record.get("rollout_id", -1)))
        for task, values in record.get("tasks", {}).items():
            score = values.get("score")
            if score is not None:
                eval_points.setdefault(task, {})[update] = float(score)
    artifact_eval, final_eval_num_updates = final_eval_metrics(run_dir)
    for task, values in artifact_eval.items():
        final_eval.setdefault(task, {}).update(values)
    tasks = final_eval or {str(last.get("experiment_task") or "unknown"): {}}
    train_events = jsonl(run_dir / "metrics" / "train.jsonl")
    rollout_events = jsonl(run_dir / "metrics" / "rollout.jsonl")

    def mean_metric(events: list[dict[str, Any]], key: str) -> float | None:
        values = metric_values(events, key)
        return sum(values) / len(values) if values else None

    def max_metric(events: list[dict[str, Any]], key: str) -> float | None:
        values = metric_values(events, key)
        return max(values) if values else None

    def sum_metric(events: list[dict[str, Any]], key: str) -> float | None:
        values = metric_values(events, key)
        return sum(values) if values else None

    step_times = metric_values(rollout_events, "perf/step_time")
    geometry_observation_times = [
        float(record["geometry_observation_wall_time_ms"])
        for record in geometry_records
        if record.get("geometry_observation_wall_time_ms") is not None
    ]
    rollout_geometry_records = jsonl(run_dir / "geometry" / "rollout" / "metrics.jsonl")
    rollout_observation_times = [
        float(record["observation_wall_time_ms"])
        for record in rollout_geometry_records
        if record.get("observation_wall_time_ms") is not None
    ]
    recorded_step_time_ms = 1000.0 * sum(step_times)
    recorded_observation_ms = sum(geometry_observation_times) + sum(rollout_observation_times)
    update_counters = last.get("run_update_counters") or {}
    durable_summary = {
        "train_peak_gpu_allocated_mib": max_metric(train_events, "train/gpu_peak_allocated_mib"),
        "train_peak_gpu_reserved_mib": max_metric(train_events, "train/gpu_peak_reserved_mib"),
        "critic_peak_gpu_allocated_mib": max_metric(train_events, "train/critic-gpu_peak_allocated_mib"),
        "critic_peak_gpu_reserved_mib": max_metric(train_events, "train/critic-gpu_peak_reserved_mib"),
        "train_grad_clip_fraction": mean_metric(train_events, "train/grad_clipped"),
        "critic_grad_clip_fraction": mean_metric(train_events, "train/critic-grad_clipped"),
        "rollout_truncated_ratio_mean": mean_metric(rollout_events, "rollout/truncated_ratio"),
        "rollout_response_p95_mean": mean_metric(rollout_events, "rollout/response_len/p95"),
        "sampled_logratio_token_mean": mean_metric(
            rollout_events,
            "rollout/geometry/sampled_reverse_kl_logratio/token/mean",
        ),
        "sampled_logratio_token_std": mean_metric(
            rollout_events,
            "rollout/geometry/sampled_reverse_kl_logratio/token/std",
        ),
        "sampled_logratio_negative_fraction": mean_metric(
            rollout_events,
            "rollout/geometry/sampled_reverse_kl_logratio/token/negative_fraction",
        ),
        "valid_response_length_mean": mean_metric(
            rollout_events,
            "rollout/geometry/response/valid_length/mean",
        ),
        "rollout_effective_tokens_per_gpu_per_sec_mean": mean_metric(
            rollout_events, "perf/effective_tokens_per_gpu_per_sec"
        ),
        "step_time_seconds_mean": sum(step_times) / len(step_times) if step_times else None,
        "step_time_seconds_total": sum(step_times) if step_times else None,
        "parameter_geometry_observation_ms_mean": (
            sum(geometry_observation_times) / len(geometry_observation_times) if geometry_observation_times else None
        ),
        "parameter_geometry_observation_ms_total": (
            sum(geometry_observation_times) if geometry_observation_times else None
        ),
        "rollout_geometry_observation_ms_mean": (
            sum(rollout_observation_times) / len(rollout_observation_times) if rollout_observation_times else None
        ),
        "rollout_geometry_observation_ms_total": (
            sum(rollout_observation_times) if rollout_observation_times else None
        ),
        "recorded_observation_to_step_time_ratio": (
            recorded_observation_ms / recorded_step_time_ms if recorded_step_time_ms else None
        ),
        "sandbox_errors_total": sum_metric(rollout_events, "rollout/sandbox/errors"),
        "sandbox_infrastructure_errors_total": sum_metric(rollout_events, "rollout/sandbox/infrastructure_errors"),
        "sandbox_execution_errors_total": sum_metric(rollout_events, "rollout/sandbox/execution_errors"),
        "sandbox_timeouts_total": sum_metric(rollout_events, "rollout/sandbox/timeouts"),
    }

    rows = []
    for task, eval_values in sorted(tasks.items()):
        auc_raw, auc_normalized = fixed_grid_auc(eval_points.get(task, {}))
        row: dict[str, Any] = {
            "run_dir": str(run_dir.resolve()),
            "task": task,
            "teacher": last.get("experiment_teacher"),
            "condition": last.get("experiment_condition"),
            "optimizer": last.get("optimizer"),
            "learning_rate": last.get("learning_rate"),
            "weight_decay": last.get("weight_decay"),
            "loss_type": last.get("loss_type", "policy_loss"),
            "advantage_estimator": last.get("advantage_estimator"),
            "opd_kl_coef": last.get("opd_kl_coef"),
            "opd_task_reward_weight": last.get("opd_task_reward_weight"),
            "hybrid_sft_loss_coef": last.get("hybrid_sft_loss_coef"),
            "hybrid_opd_loss_coef": last.get("hybrid_opd_loss_coef"),
            "seed": last.get("experiment_seed"),
            "geometry_successful_updates": len(global_records),
            "geometry_failed_or_skipped_updates": int(update_counters.get("failed_or_skipped", 0)),
            "geometry_run_clip_fraction": last.get("run_clip_fraction"),
            "first_observation_id": first.get("observation_id"),
            "last_observation_id": last.get("observation_id"),
            "final_cumulative_prompt_count": last.get("cumulative_prompt_count"),
            "final_cumulative_effective_token_count": last.get("cumulative_effective_token_count"),
            "final_actual_batch_size": last.get("actual_batch_size"),
            "minimum_actual_batch_size": min(int(record.get("actual_batch_size", 0)) for record in geometry_records),
            "eval_score": eval_values.get("score"),
            "eval_best": eval_values.get("best"),
            "eval_fixed_grid_auc_raw": auc_raw,
            "eval_fixed_grid_auc_update_normalized": auc_normalized,
            "eval_fixed_grid_point_count": len(eval_points.get(task, {})),
            "eval_forgetting": eval_values.get("forgetting"),
            "eval_backward_transfer": eval_values.get("backward_transfer"),
            "eval_num_updates": final_eval_num_updates,
            "eval_pass_at_1": eval_values.get("pass@1"),
            "eval_pass_at_5": eval_values.get("pass@5"),
            "eval_pass_at_10": eval_values.get("pass@10"),
            **durable_summary,
        }
        for field in GEOMETRY_FIELDS:
            row[f"final_{field}"] = final_geometry.get(field)
            values = [record.get(field) for record in global_records if record.get(field) is not None]
            row[f"mean_{field}"] = sum(values) / len(values) if values else None
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_rows = [row for run in args.runs for row in summarize_run(run)]
    write_csv(args.output, output_rows)
