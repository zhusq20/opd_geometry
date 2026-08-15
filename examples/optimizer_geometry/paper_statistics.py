#!/usr/bin/env python3
"""Aggregate seeds, paired optimizer effects, curves, plots, and a LaTeX table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from .summarize_single_task import GEOMETRY_FIELDS, summarize_run
    from .validate_run_artifacts import validate as validate_run
except ImportError:  # Direct execution: python examples/.../paper_statistics.py
    from summarize_single_task import GEOMETRY_FIELDS, summarize_run
    from validate_run_artifacts import validate as validate_run


GROUP = ("task", "teacher", "condition", "optimizer")
PAIR_GROUP = ("task", "teacher", "condition")


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
    }
    return table.get(df, 1.96)


def estimate(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    n = len(array)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = float(array.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "ci95_low": None, "ci95_high": None}
    sd = float(array.std(ddof=1))
    half = t_critical_95(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fields:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = (
        "eval_score",
        "eval_best",
        "eval_fixed_grid_auc_raw",
        "eval_fixed_grid_auc_update_normalized",
        "eval_pass_at_1",
        "eval_pass_at_5",
        "eval_pass_at_10",
        "eval_forgetting",
        "eval_backward_transfer",
        "train_peak_gpu_allocated_mib",
        "train_peak_gpu_reserved_mib",
        "critic_peak_gpu_allocated_mib",
        "critic_peak_gpu_reserved_mib",
        "train_grad_clip_fraction",
        "critic_grad_clip_fraction",
        "rollout_truncated_ratio_mean",
        "rollout_response_p95_mean",
        "rollout_effective_tokens_per_gpu_per_sec_mean",
        "step_time_seconds_mean",
        "step_time_seconds_total",
        "parameter_geometry_observation_ms_mean",
        "parameter_geometry_observation_ms_total",
        "rollout_geometry_observation_ms_mean",
        "rollout_geometry_observation_ms_total",
        "recorded_observation_to_step_time_ratio",
        "geometry_failed_or_skipped_updates",
        "geometry_run_clip_fraction",
        "final_cumulative_prompt_count",
        "final_cumulative_effective_token_count",
        "final_actual_batch_size",
        "minimum_actual_batch_size",
        "sandbox_errors_total",
        "sandbox_infrastructure_errors_total",
        "sandbox_execution_errors_total",
        "sandbox_timeouts_total",
    ) + tuple(f"final_{field}" for field in GEOMETRY_FIELDS)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in GROUP)].append(row)
    output = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(GROUP, key, strict=True))
        record["seeds"] = ",".join(str(row.get("seed")) for row in sorted(group_rows, key=lambda row: row.get("seed")))
        for metric in numeric:
            values = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
            for name, value in estimate(values).items():
                record[f"{metric}_{name}"] = value
        output.append(record)
    return output


def paired(rows: list[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    indexed = {
        tuple(row.get(key) for key in PAIR_GROUP) + (row.get("seed"), row.get("optimizer")): row for row in rows
    }
    optimizers = sorted({str(row.get("optimizer")) for row in rows if row.get("optimizer") != baseline})
    groups = sorted({tuple(row.get(key) for key in PAIR_GROUP) for row in rows}, key=lambda x: tuple(map(str, x)))
    output = []
    effect_metrics = (
        "eval_score",
        "eval_best",
        "eval_fixed_grid_auc_update_normalized",
        "eval_pass_at_1",
        "eval_pass_at_5",
        "eval_pass_at_10",
    )
    for group in groups:
        seeds = sorted({key[-2] for key in indexed if key[: len(PAIR_GROUP)] == group})
        for optimizer in optimizers:
            for metric in effect_metrics:
                differences = []
                paired_seeds = []
                for seed in seeds:
                    left = indexed.get(group + (seed, optimizer))
                    right = indexed.get(group + (seed, baseline))
                    if left and right and left.get(metric) is not None and right.get(metric) is not None:
                        differences.append(float(left[metric]) - float(right[metric]))
                        paired_seeds.append(seed)
                if not differences:
                    continue
                record = dict(zip(PAIR_GROUP, group, strict=True))
                record.update(
                    {
                        "optimizer": optimizer,
                        "baseline": baseline,
                        "metric": metric,
                        "paired_seeds": ",".join(map(str, paired_seeds)),
                    }
                )
                record.update({f"difference_{name}": value for name, value in estimate(differences).items()})
                output.append(record)
    return output


def curves(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run in run_dirs:
        geometry = jsonl(run / "geometry" / "actor" / "metrics.jsonl")
        if not geometry:
            continue
        metadata = geometry[-1]
        base = {
            "run_dir": str(run.resolve()),
            "task": metadata.get("experiment_task"),
            "teacher": metadata.get("experiment_teacher"),
            "condition": metadata.get("experiment_condition"),
            "optimizer": metadata.get("optimizer"),
            "seed": metadata.get("experiment_seed"),
        }
        axes_by_update = {
            int(record.get("num_updates", int(record["observation_id"]) + 1)): {
                "cumulative_prompt_count": record.get("cumulative_prompt_count"),
                "cumulative_effective_token_count": record.get("cumulative_effective_token_count"),
                "actual_batch_size": record.get("actual_batch_size"),
            }
            for record in geometry
            if record.get("update_successful") is True
        }
        for record in geometry:
            if record.get("update_successful") is not True:
                continue
            rows.append(
                {
                    **base,
                    "curve": "geometry",
                    "num_updates": record.get("num_updates", int(record["observation_id"]) + 1),
                    "cumulative_prompt_count": record.get("cumulative_prompt_count"),
                    "cumulative_effective_token_count": record.get("cumulative_effective_token_count"),
                    "actual_batch_size": record.get("actual_batch_size"),
                    **{
                        f"geometry_{key}": value
                        for key, value in (record.get("groups", {}).get("global") or {}).items()
                    },
                }
            )
        for record in jsonl(run / "forgetting" / "metrics.jsonl"):
            update = int(record.get("num_updates", record.get("rollout_id", -1)))
            axes = axes_by_update.get(update, {})
            for task, values in record.get("tasks", {}).items():
                rows.append(
                    {
                        **base,
                        "curve": "evaluation",
                        "eval_dataset": task,
                        "num_updates": update,
                        **axes,
                        **{f"eval_{key}": value for key, value in values.items()},
                    }
                )
    return rows


def latex_table(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    def interval(row: dict[str, Any], metric: str) -> str:
        mean = row.get(f"{metric}_mean")
        low = row.get(f"{metric}_ci95_low")
        high = row.get(f"{metric}_ci95_high")
        if mean is None:
            return "--"
        if low is None or high is None:
            return f"{mean:.4f}"
        return f"{mean:.4f} $\\pm$ {(high - low) / 2:.4f}"

    lines = [
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Task & Condition & Teacher & Optimizer & Score & pass@1 & pass@5 & pass@10 \\",
        r"\midrule",
    ]
    for row in aggregate_rows:
        fields = [
            row.get("task"),
            row.get("condition"),
            row.get("teacher"),
            row.get("optimizer"),
            interval(row, "eval_score"),
            interval(row, "eval_pass_at_1"),
            interval(row, "eval_pass_at_5"),
            interval(row, "eval_pass_at_10"),
        ]
        lines.append(" & ".join(str(field).replace("_", r"\_") for field in fields) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_latex_table(path: Path, paired_rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lllllrrr}",
        r"\toprule",
        r"Task & Condition & Teacher & Contrast & Metric & $n$ & Mean difference & 95\% CI \\",
        r"\midrule",
    ]
    for row in paired_rows:
        mean = row.get("difference_mean")
        low = row.get("difference_ci95_low")
        high = row.get("difference_ci95_high")
        interval = "--" if low is None or high is None else f"[{low:.4f}, {high:.4f}]"
        fields = [
            row.get("task"),
            row.get("condition"),
            row.get("teacher"),
            f"{row.get('optimizer')} - {row.get('baseline')}",
            row.get("metric"),
            row.get("difference_n"),
            "--" if mean is None else f"{mean:.4f}",
            interval,
        ]
        lines.append(" & ".join(str(field).replace("_", r"\_") for field in fields) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_figure(figure: Any, output_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.pdf")
    figure.savefig(output_dir / f"{stem}.png", dpi=200)


def _curve_plot(
    output_dir: Path,
    curve_rows: list[dict[str, Any]],
    *,
    curve: str,
    metric: str,
    ylabel: str,
    stem: str,
) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in curve_rows if row.get("curve") == curve and row.get(metric) is not None]
    if not selected:
        return
    panel_keys = sorted(
        {(row.get("task"), row.get("eval_dataset") if curve == "evaluation" else None) for row in selected},
        key=lambda value: tuple(map(str, value)),
    )
    figure, axes = plt.subplots(len(panel_keys), 1, figsize=(7.5, max(4.0, 3.4 * len(panel_keys))), squeeze=False)
    for axis, panel in zip(axes.ravel(), panel_keys, strict=True):
        panel_rows = [
            row
            for row in selected
            if (row.get("task"), row.get("eval_dataset") if curve == "evaluation" else None) == panel
        ]
        series: dict[tuple[Any, ...], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in panel_rows:
            key = (row.get("teacher"), row.get("condition"), row.get("optimizer"))
            series[key][int(row["num_updates"])].append(float(row[metric]))
        for key, points in sorted(series.items(), key=lambda item: tuple(map(str, item[0]))):
            updates = sorted(points)
            estimates = [estimate(points[update]) for update in updates]
            means = np.asarray([value["mean"] for value in estimates], dtype=np.float64)
            lows = np.asarray(
                [value["mean"] if value["ci95_low"] is None else value["ci95_low"] for value in estimates],
                dtype=np.float64,
            )
            highs = np.asarray(
                [value["mean"] if value["ci95_high"] is None else value["ci95_high"] for value in estimates],
                dtype=np.float64,
            )
            label = "/".join(str(value) for value in key if value not in (None, "none", "unspecified"))
            (line,) = axis.plot(updates, means, marker="o", markersize=3, label=label)
            if any(value["n"] > 1 for value in estimates):
                axis.fill_between(updates, lows, highs, color=line.get_color(), alpha=0.16)
        title = str(panel[0])
        if panel[1] is not None:
            title += f" — {panel[1]}"
        axis.set_title(title)
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    _save_figure(figure, output_dir, stem)
    plt.close(figure)


def plots(output_dir: Path, aggregate_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    score_rows = [row for row in aggregate_rows if row.get("eval_score_mean") is not None]
    labels = [f"{row['task']}\n{row['condition']}\n{row['optimizer']}" for row in score_rows]
    means = [row["eval_score_mean"] for row in score_rows]
    errors = [
        0 if row.get("eval_score_ci95_low") is None else row["eval_score_mean"] - row["eval_score_ci95_low"]
        for row in score_rows
    ]
    if score_rows:
        width = max(7, len(labels) * 0.55)
        figure, axis = plt.subplots(figsize=(width, 4.5))
        axis.bar(np.arange(len(labels)), means, yerr=errors, capsize=3)
        axis.set_ylabel("Final evaluation score")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
        _save_figure(figure, output_dir, "final_scores")
        plt.close(figure)
    _curve_plot(
        output_dir,
        curve_rows,
        curve="evaluation",
        metric="eval_score",
        ylabel="Evaluation score",
        stem="learning_curves",
    )
    _curve_plot(
        output_dir,
        curve_rows,
        curve="geometry",
        metric="geometry_delta_intended_to_theta_ratio",
        ylabel="Intended update / parameter norm",
        stem="geometry_update_ratio_curves",
    )
    _curve_plot(
        output_dir,
        curve_rows,
        curve="geometry",
        metric="geometry_cos_g_opt_delta_intended_fp32",
        ylabel="cos(optimizer gradient, intended update)",
        stem="geometry_alignment_curves",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(run_dirs: list[Path], *, skip: bool) -> list[dict[str, Any]]:
    reports = []
    for run in run_dirs:
        report = validate_run(SimpleNamespace(run_dir=run, expected_updates=None, require_eval=not skip))
        reports.append(report)
        if not skip and not report["valid"]:
            joined = "; ".join(report["errors"])
            raise ValueError(f"Refusing incomplete paper input {run}: {joined}")
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="adamw")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Allow exploratory analysis of incomplete runs; never use for a paper table.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_reports = validate_inputs(args.runs, skip=args.skip_validation)
    per_run = [row for run in args.runs for row in summarize_run(run)]
    aggregate_rows = aggregate(per_run)
    paired_rows = paired(per_run, args.baseline)
    curve_rows = curves(args.runs)
    write_csv(args.output_dir / "per_run.csv", per_run)
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows)
    write_csv(args.output_dir / "paired_effects.csv", paired_rows)
    write_csv(args.output_dir / "curves.csv", curve_rows)
    latex_table(args.output_dir / "final_scores.tex", aggregate_rows)
    paired_latex_table(args.output_dir / "paired_effects.tex", paired_rows)
    if not args.no_plots:
        plots(args.output_dir, aggregate_rows, curve_rows)
    generated = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "analysis_manifest.json"
    )
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "path": str(run.resolve()),
                        "run_manifest_sha256": sha256_file(run / "provenance" / "run_manifest.json"),
                    }
                    for run in args.runs
                ],
                "baseline": args.baseline,
                "interval": "two-sided 95% Student-t confidence interval across independent seeds",
                "paired_effect": "within-seed optimizer minus baseline",
                "analysis_availability": {
                    "final_best_fixed_grid_auc": "available",
                    "update_prompt_effective_token_axes": "available",
                    "paired_seed_effects": "available_when_matching_seeds_exist",
                    "same_checkpoint_task_gradient_prediction": (
                        "not_available_requires_separate_fixed-probe_backward_artifacts"
                    ),
                    "nested_raw_optimizer_precision_models": (
                        "not_available_without_same-checkpoint_next-probe_targets"
                    ),
                    "leave_one_seed_and_task_out": ("not_available_until_nested_prediction_inputs_are_collected"),
                    "checkpoint_clone_interventions": ("not_available_requires_separate_intervention_runner"),
                },
                "validated": not args.skip_validation,
                "validation_reports": validation_reports,
                "outputs": {
                    path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in generated
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
