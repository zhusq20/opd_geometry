#!/usr/bin/env python3
"""Render the preregistered single-seed MOPD result gallery: nine rows, three panels each."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
TASKS = ("math", "code", "if", "science")
CONFIG_LABELS = {
    "uniform_k1_conventional": "Uniform K1 / conventional",
    "uniform_k1_taskwise": "Uniform K1 / taskwise",
    "gpas_k1_taskwise": "GPAS K1",
    "cost_gpas_k1_taskwise": "Cost-GPAS K1",
    "uniform_k2_taskwise": "Uniform K2",
    "cost_gpas_k2_taskwise": "Cost-GPAS K2",
    "all_k4_taskwise": "All K4 / taskwise",
    "all_k4_conventional": "All K4 / conventional",
}
CONFIGS = tuple(CONFIG_LABELS)
COLORS = dict(zip(CONFIGS, plt.get_cmap("tab10").colors[: len(CONFIGS)], strict=True))
TASK_LABELS = {"math": "Math", "code": "Code", "if": "IF", "science": "Science"}
TASK_COLORS = dict(zip(TASKS, ("#3B77B4", "#E69F00", "#7A5195", "#2A9D8F"), strict=True))
ADAPTIVE_CONFIGS = ("gpas_k1_taskwise", "cost_gpas_k1_taskwise", "cost_gpas_k2_taskwise")
ROW_FILES = (
    "row_01_controlled_checks.pdf",
    "row_02_learning_efficiency.pdf",
    "row_03_endpoint_efficiency.pdf",
    "row_04_task_balance.pdf",
    "row_05_online_diagnostics.pdf",
    "row_06_frozen_bank.pdf",
    "row_07_allocation_dynamics.pdf",
    "row_08_system_costs.pdf",
    "row_09_capability.pdf",
)
PANEL_FILES = (
    "panel_01_controlled_adamw_error.pdf",
    "panel_02_controlled_time_error.pdf",
    "panel_03_controlled_moment_bias.pdf",
    "panel_04_loss_vs_responses.pdf",
    "panel_05_loss_vs_tokens.pdf",
    "panel_06_loss_vs_gpu_hours.pdf",
    "panel_07_final_loss.pdf",
    "panel_08_responses_to_threshold.pdf",
    "panel_09_gpu_hours_to_threshold.pdf",
    "panel_10_task_loss_heatmap.pdf",
    "panel_11_task_delta_heatmap.pdf",
    "panel_12_worst_task_loss.pdf",
    "panel_13_raw_vs_adamw_priority.pdf",
    "panel_14_score_age.pdf",
    "panel_15_probe_overhead.pdf",
    "panel_16_bank_k1.pdf",
    "panel_17_bank_k2.pdf",
    "panel_18_bank_k4.pdf",
    "panel_19_gpas_k1_allocation.pdf",
    "panel_20_cost_gpas_k1_allocation.pdf",
    "panel_21_cost_gpas_k2_sets.pdf",
    "panel_22_critical_path.pdf",
    "panel_23_transfer_tail.pdf",
    "panel_24_peak_hbm.pdf",
    "panel_25_capability_heatmap.pdf",
    "panel_26_capability_delta.pdf",
    "panel_27_loss_capability_alignment.pdf",
)


plt.rcParams.update(
    {
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _finish_axis(axis, *, grid: str | None = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid is not None:
        axis.grid(axis=grid, color="#D9DEE3", linewidth=0.6, alpha=0.75)
        axis.set_axisbelow(True)


def _config_ticks(axis) -> None:
    axis.set_xticks(
        np.arange(len(CONFIGS)),
        [CONFIG_LABELS[config] for config in CONFIGS],
        rotation=35,
        ha="right",
    )


def _error_magnitudes(points: list[float], intervals: list[list[float]]) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    bounds = np.asarray(intervals, dtype=float)
    return np.vstack(
        (
            np.maximum(0.0, values - bounds[:, 0]),
            np.maximum(0.0, bounds[:, 1] - values),
        )
    )


def _save_row(fig, axes, row_path: Path, panel_dir: Path, panel_names: tuple[str, str, str]) -> None:
    row_path.parent.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(w_pad=1.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_bounds = [
        axis.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(0.06) for axis in axes
    ]
    fig.savefig(row_path, bbox_inches="tight")
    for name, bounds in zip(panel_names, panel_bounds, strict=True):
        fig.savefig(panel_dir / name, bbox_inches=bounds)
    plt.close(fig)


def _heatmap(
    axis,
    values: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    *,
    title: str,
    centered: bool = False,
) -> None:
    if centered:
        limit = max(float(np.max(np.abs(values))), 1e-12)
        axis.imshow(values, aspect="auto", cmap="coolwarm_r", vmin=-limit, vmax=limit)
    else:
        axis.imshow(values, aspect="auto", cmap="viridis_r")
    axis.set_xticks(np.arange(len(xlabels)), xlabels)
    axis.set_yticks(np.arange(len(ylabels)), ylabels)
    axis.set_title(title)
    threshold = float(np.nanmedian(values))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            if centered:
                color = "white" if abs(value) > 0.55 * limit else "black"
            else:
                color = "white" if value > threshold else "black"
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=5.5, color=color)


def plot_controlled_row(
    sampling: list[dict[str, str]],
    moments: list[dict[str, str]],
    row_path: Path,
    panel_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    method_colors = ["#7B8794", "#E69F00", "#3B77B4", "#2A9D8F"]
    method_labels = ["Uniform", "GradNorm", "GPAS", "Cost"]
    x = np.arange(len(sampling))
    for axis, empirical_key, theory_key, title in (
        (axes[0], "adamw_empirical_mse_ratio", "adamw_theory_mse_ratio", "AdamW-metric estimator error"),
        (axes[1], "cost_empirical_ratio", "cost_theory_ratio", "Time-weighted objective"),
    ):
        empirical = [float(row[empirical_key]) for row in sampling]
        theory = [float(row[theory_key]) for row in sampling]
        axis.bar(x, empirical, color=method_colors, width=0.68)
        axis.scatter(x, theory, color="black", marker="x", s=24, label="calculated", zorder=3)
        axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_xticks(x, method_labels, rotation=20)
        axis.set_ylabel("relative to Uniform")
        axis.set_title(title)
        axis.legend(frameon=False)
        _finish_axis(axis)

    optimizer_names = ("Standard AdamW", "Moment-consistent AdamW")
    moment_names = ("First moment", "Second moment")
    moment_x = np.arange(len(moment_names))
    for index, optimizer in enumerate(optimizer_names):
        rows = [
            next(row for row in moments if row["optimizer"] == optimizer and row["moment"] == moment)
            for moment in moment_names
        ]
        offset = (index - 0.5) * 0.34
        axes[2].bar(
            moment_x + offset,
            [float(row["mc_relative_bias"]) for row in rows],
            width=0.34,
            color=("#D55E00", "#2A9D8F")[index],
            label=optimizer.replace(" AdamW", ""),
        )
        axes[2].scatter(
            moment_x + offset,
            [float(row["calculated_relative_bias"]) for row in rows],
            color="black",
            marker="x",
            s=24,
            zorder=3,
        )
    axes[2].set_xticks(moment_x, ("first", "second"))
    axes[2].set_ylabel("relative bias")
    axes[2].set_title("Expected AdamW moment change")
    axes[2].legend(frameon=False)
    _finish_axis(axes[2])
    _save_row(fig, axes, row_path, panel_dir, PANEL_FILES[0:3])


def plot_learning_row(report: dict[str, Any], row_path: Path, panel_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    coordinates = (
        ("attempted_responses", 1_000.0, "Attempted responses (thousands)"),
        ("valid_response_tokens", 1_000_000.0, "Valid response tokens (millions)"),
        ("gpu_hours", 1.0, "Resident GPU hours"),
    )
    for config in CONFIGS:
        curve = report["curves"][config]
        loss = [row["mean_relative_loss"] for row in curve]
        for axis, (coordinate, scale, xlabel) in zip(axes, coordinates, strict=True):
            axis.plot(
                [row[coordinate] / scale for row in curve],
                loss,
                marker="o",
                markersize=2.8,
                linewidth=1.1,
                label=CONFIG_LABELS[config],
                color=COLORS[config],
            )
            axis.set_xlabel(xlabel)
    for axis in axes:
        axis.axhline(report["common_threshold"], color="black", linestyle="--", linewidth=0.8)
        _finish_axis(axis, grid="both")
    axes[0].set_ylabel("Mean relative teacher loss")
    axes[0].set_title("Response efficiency")
    axes[1].set_title("Token efficiency")
    axes[2].set_title("Wall-clock efficiency")
    axes[2].legend(frameon=False, ncol=2)
    _save_row(fig, axes, row_path, panel_dir, PANEL_FILES[3:6])


def _plot_threshold(
    axis,
    report: dict[str, Any],
    key: str,
    cap_key: str | None,
    ylabel: str,
) -> None:
    values, censored = [], []
    for config in CONFIGS:
        outcome = report["outcomes"][config]
        observed = outcome[key]
        cap = report["response_budget"] if cap_key is None else outcome[cap_key]
        values.append(float(cap if observed is None else observed))
        censored.append(observed is None)
    bars = axis.bar(np.arange(len(CONFIGS)), values, color=[COLORS[config] for config in CONFIGS])
    for bar, is_censored in zip(bars, censored, strict=True):
        if is_censored:
            bar.set_facecolor("none")
            bar.set_edgecolor("#555555")
            bar.set_hatch("///")
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                "NR",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    _config_ticks(axis)
    axis.set_ylabel(ylabel)
    _finish_axis(axis)


def plot_endpoint_row(report: dict[str, Any], row_path: Path, panel_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    x = np.arange(len(CONFIGS))
    points = [float(report["outcomes"][config]["mean_relative_loss"]) for config in CONFIGS]
    intervals = [report["paired_bootstrap"]["configs"][config]["mean_relative_loss_95ci"] for config in CONFIGS]
    axes[0].bar(x, points, color=[COLORS[config] for config in CONFIGS])
    axes[0].errorbar(
        x,
        points,
        yerr=_error_magnitudes(points, intervals),
        fmt="none",
        color="black",
        capsize=2,
        linewidth=0.8,
    )
    axes[0].axhline(report["common_threshold"], color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Final mean relative loss")
    axes[0].set_title("Final loss with paired-prompt 95% CI")
    _config_ticks(axes[0])
    _finish_axis(axes[0])
    _plot_threshold(axes[1], report, "responses_to_threshold", None, "Attempted responses")
    axes[1].set_title("Responses to common threshold")
    _plot_threshold(
        axes[2],
        report,
        "gpu_hours_to_threshold",
        "gpu_hours",
        "Resident GPU hours",
    )
    axes[2].set_title("GPU hours to common threshold")
    _save_row(fig, axes, row_path, panel_dir, PANEL_FILES[6:9])


def plot_task_row(report: dict[str, Any], row_path: Path, panel_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    losses = np.asarray(
        [[report["outcomes"][config]["relative_losses"][task] for task in TASKS] for config in CONFIGS]
    )
    baseline = losses[0]
    deltas = losses - baseline[None, :]
    labels = [CONFIG_LABELS[config] for config in CONFIGS]
    task_labels = [TASK_LABELS[task] for task in TASKS]
    _heatmap(axes[0], losses, task_labels, labels, title="Final task-relative loss")
    _heatmap(
        axes[1],
        deltas,
        task_labels,
        labels,
        title="Task loss delta vs baseline",
        centered=True,
    )

    worst = list(map(float, np.max(losses, axis=1)))
    intervals = [report["paired_bootstrap"]["configs"][config]["worst_task_loss_95ci"] for config in CONFIGS]
    x = np.arange(len(CONFIGS))
    axes[2].bar(x, worst, color=[COLORS[config] for config in CONFIGS])
    axes[2].errorbar(
        x,
        worst,
        yerr=_error_magnitudes(worst, intervals),
        fmt="none",
        color="black",
        capsize=2,
        linewidth=0.8,
    )
    axes[2].set_ylabel("Maximum task-relative loss")
    axes[2].set_title("Worst-task outcome")
    _config_ticks(axes[2])
    _finish_axis(axes[2])
    _save_row(fig, axes, row_path, panel_dir, PANEL_FILES[9:12])


def _train_trace(report: dict[str, Any], config: str) -> list[dict[str, Any]]:
    return [row for row in report["system_traces"][config] if row["operation"] == "train"]


def plot_online_row(report: dict[str, Any], row_path: Path, panel_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    priority_trace = _train_trace(report, "gpas_k1_taskwise")
    for task_index, task in enumerate(TASKS):
        raw_share, adam_share = [], []
        for row in priority_trace:
            raw = np.asarray([row["raw_gradient_rms"][name] for name in TASKS], dtype=float)
            adam = np.asarray([row["adam_gradient_rms"][name] for name in TASKS], dtype=float)
            if raw.sum() > 0 and adam.sum() > 0:
                raw_share.append(raw[task_index] / raw.sum())
                adam_share.append(adam[task_index] / adam.sum())
        axes[0].scatter(
            raw_share,
            adam_share,
            s=8,
            alpha=0.35,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
        )
    axes[0].plot((0, 1), (0, 1), color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Raw-gradient priority share")
    axes[0].set_ylabel("AdamW-scaled priority share")
    axes[0].set_title("Optimizer-induced task reprioritization")
    axes[0].legend(frameon=False)
    _finish_axis(axes[0], grid="both")

    for config in ADAPTIVE_CONFIGS:
        trace = report["system_traces"][config]
        axes[1].plot(
            [row["attempted_responses_after"] / 1_000 for row in trace],
            [max(row["score_ages"].values()) for row in trace],
            color=COLORS[config],
            label=CONFIG_LABELS[config],
            linewidth=1.0,
        )
    axes[1].set_xlabel("Attempted responses (thousands)")
    axes[1].set_ylabel("Maximum score age (task units)")
    axes[1].set_title("Online score freshness")
    axes[1].legend(frameon=False)
    _finish_axis(axes[1], grid="both")

    for config in ADAPTIVE_CONFIGS:
        total_gpu, probe_gpu, x, y = 0.0, 0.0, [], []
        for row in report["system_traces"][config]:
            total_gpu += row["total_gpu_seconds"]
            if row["operation"] == "probe":
                probe_gpu += row["total_gpu_seconds"]
            x.append(row["attempted_responses_after"] / 1_000)
            y.append(100.0 * probe_gpu / total_gpu)
        axes[2].plot(x, y, color=COLORS[config], label=CONFIG_LABELS[config], linewidth=1.0)
    axes[2].set_xlabel("Attempted responses (thousands)")
    axes[2].set_ylabel("Cumulative probe GPU cost (%)")
    axes[2].set_title("Probe overhead")
    axes[2].legend(frameon=False)
    _finish_axis(axes[2], grid="both")
    _save_row(fig, axes, row_path, panel_dir, PANEL_FILES[12:15])


def plot_bank_row(report: dict[str, Any], row_path: Path, panel_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    stages = ("warm", "middle", "late")
    methods = ("uniform", "raw_norm", "gpas", "cost_gpas")
    labels = {
        "uniform": "Uniform",
        "raw_norm": "Raw norm",
        "gpas": "GPAS",
        "cost_gpas": "Cost-GPAS",
    }
    method_colors = {
        "uniform": "#7B8794",
        "raw_norm": "#E69F00",
        "gpas": "#3B77B4",
        "cost_gpas": "#2A9D8F",
    }
    x = np.arange(len(stages))
    for axis, width_k in zip(axes, (1, 2, 4), strict=True):
        for method in methods:
            values = [
                report["stages"][stage]["cross_fit"]["K"][str(width_k)][method]["relative_adamw_estimator_mse"]
                for stage in stages
            ]
            axis.plot(x, values, marker="o", label=labels[method], color=method_colors[method])
        axis.set_xticks(x, [stage.title() for stage in stages])
        axis.set_title(f"Frozen-bank estimator error, K={width_k}")
        _finish_axis(axis)
    axes[0].set_ylabel("Held-out relative AdamW-metric MSE")
    axes[-1].legend(frameon=False)
    _save_row(fig, axes, row_path, panel_dir, PANEL_FILES[15:18])


def _plot_inclusion(axis, report: dict[str, Any], config: str) -> None:
    trace = _train_trace(report, config)
    x = [row["attempted_responses_after"] / 1_000 for row in trace]
    for task in TASKS:
        axis.plot(
            x,
            [row["inclusion_probabilities"][task] for row in trace],
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
        )
    axis.set_xlabel("Attempted responses (thousands)")
    axis.set_ylabel("Inclusion probability")
    axis.set_title(CONFIG_LABELS[config])
    axis.legend(frameon=False, ncol=2)
    _finish_axis(axis, grid="both")


def plot_allocation_row(report: dict[str, Any], row_path: Path, panel_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    _plot_inclusion(axes[0], report, "gpas_k1_taskwise")
    _plot_inclusion(axes[1], report, "cost_gpas_k1_taskwise")

    trace = _train_trace(report, "cost_gpas_k2_taskwise")
    set_names = sorted({name for row in trace for name in row["set_distribution"]})
    probabilities = np.asarray([[row["set_distribution"].get(name, 0.0) for row in trace] for name in set_names])
    left = trace[0]["attempted_responses_after"] / 1_000
    right = trace[-1]["attempted_responses_after"] / 1_000
    axes[2].imshow(
        probabilities,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=max(0.5, float(probabilities.max())),
        extent=(left, right, -0.5, len(set_names) - 0.5),
    )
    axes[2].set_yticks(np.arange(len(set_names)), set_names)
    axes[2].set_xlabel("Attempted responses (thousands)")
    axes[2].set_ylabel("Exact K=2 task set")
    axes[2].set_title("Cost-GPAS K2 set distribution")
    _save_row(fig, axes, row_path, panel_dir, PANEL_FILES[18:21])


def plot_system_row(report: dict[str, Any], row_path: Path, panel_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    x = np.arange(len(CONFIGS))
    components = (
        ("rollout_and_teacher_wall_seconds", "Rollout + teacher", "#4C78A8"),
        ("actor_forward_backward_wall_seconds", "Actor F/B", "#F58518"),
        ("optimizer_wall_seconds", "Optimizer", "#54A24B"),
    )
    bottom = np.zeros(len(CONFIGS))
    for key, label, color in components:
        values = np.asarray(
            [
                np.median([row["component_wall_seconds"][key] for row in _train_trace(report, config)])
                for config in CONFIGS
            ]
        )
        axes[0].bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    _config_ticks(axes[0])
    axes[0].set_ylabel("Median seconds per training operation")
    axes[0].set_title("Critical-path decomposition")
    axes[0].legend(frameon=False)
    _finish_axis(axes[0])

    for config in CONFIGS:
        tails = np.asarray(
            [
                unit["teacher_transfer_tail_seconds"]
                for row in report["system_traces"][config]
                for unit in row["task_units"]
                if unit["switched"]
            ],
            dtype=float,
        )
        if tails.size == 0:
            continue
        tails.sort()
        axes[1].step(
            tails,
            np.arange(1, tails.size + 1) / tails.size,
            where="post",
            color=COLORS[config],
            label=CONFIG_LABELS[config],
        )
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_xlabel("Unhidden teacher transfer tail (seconds)")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("Teacher-switch critical-path tail")
    axes[1].legend(frameon=False, ncol=2)
    _finish_axis(axes[1], grid="both")

    width = 0.38
    student = [max(row["student_peak_hbm_gib"] for row in report["system_traces"][config]) for config in CONFIGS]
    teacher = [max(row["teacher_peak_hbm_gib"] for row in report["system_traces"][config]) for config in CONFIGS]
    axes[2].bar(x - width / 2, student, width, label="Student/rollout GPU", color="#4C78A8")
    axes[2].bar(x + width / 2, teacher, width, label="Teacher GPU", color="#72B7B2")
    _config_ticks(axes[2])
    axes[2].set_ylabel("Peak HBM (GiB)")
    axes[2].set_title("Peak memory by reserved GPU")
    axes[2].legend(frameon=False)
    _finish_axis(axes[2])
    _save_row(fig, axes, row_path, panel_dir, PANEL_FILES[21:24])


def plot_capability_row(
    mopd: dict[str, Any],
    capability: dict[str, Any],
    row_path: Path,
    panel_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    scores = np.asarray(
        [[capability["configs"][config]["domains"][task]["score"] for task in TASKS] for config in CONFIGS]
    )
    labels = [CONFIG_LABELS[config] for config in CONFIGS]
    task_labels = [TASK_LABELS[task] for task in TASKS]
    _heatmap(axes[0], scores, task_labels, labels, title="Final benchmark scores")

    baseline = capability["paired_bootstrap"]["baseline"]
    points = [
        capability["paired_bootstrap"]["configs"][config]["paired_macro_delta_vs_baseline"] for config in CONFIGS
    ]
    intervals = [capability["paired_bootstrap"]["configs"][config]["paired_macro_delta_95ci"] for config in CONFIGS]
    x = np.arange(len(CONFIGS))
    axes[1].bar(x, points, color=[COLORS[config] for config in CONFIGS])
    axes[1].errorbar(
        x,
        points,
        yerr=_error_magnitudes(points, intervals),
        fmt="none",
        color="black",
        capsize=2,
        linewidth=0.8,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    _config_ticks(axes[1])
    axes[1].set_ylabel(f"Macro benchmark delta vs {CONFIG_LABELS[baseline]}")
    axes[1].set_title("Capability change with paired-prompt 95% CI")
    _finish_axis(axes[1])

    baseline_loss = mopd["outcomes"][baseline]["relative_losses"]
    baseline_scores = capability["configs"][baseline]["domains"]
    for task in TASKS:
        loss_improvement = [
            baseline_loss[task] - mopd["outcomes"][config]["relative_losses"][task] for config in CONFIGS[1:]
        ]
        capability_change = [
            capability["configs"][config]["domains"][task]["score"] - baseline_scores[task]["score"]
            for config in CONFIGS[1:]
        ]
        axes[2].scatter(
            loss_improvement,
            capability_change,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
            s=26,
            alpha=0.8,
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].axvline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Task-relative teacher-loss improvement")
    axes[2].set_ylabel("Matched benchmark-score change")
    axes[2].set_title("Surrogate/capability alignment")
    axes[2].legend(frameon=False)
    _finish_axis(axes[2], grid="both")
    _save_row(fig, axes, row_path, panel_dir, PANEL_FILES[24:27])


def render_all(
    mopd: dict[str, Any],
    capability: dict[str, Any],
    bank: dict[str, Any],
    sampling: list[dict[str, str]],
    moments: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, Any]:
    if int(mopd.get("seed", -1)) != SEED or int(capability.get("seed", -1)) != SEED:
        raise ValueError(f"the figure suite is fixed to the single training seed {SEED}")
    if tuple(mopd["configs"]) != CONFIGS or set(capability["configs"]) != set(CONFIGS):
        raise ValueError("the figure suite requires the fixed eight-configuration ordering")

    row_dir = output_dir / "rows"
    panel_dir = output_dir / "panels"
    plot_controlled_row(sampling, moments, row_dir / ROW_FILES[0], panel_dir)
    plot_learning_row(mopd, row_dir / ROW_FILES[1], panel_dir)
    plot_endpoint_row(mopd, row_dir / ROW_FILES[2], panel_dir)
    plot_task_row(mopd, row_dir / ROW_FILES[3], panel_dir)
    plot_online_row(mopd, row_dir / ROW_FILES[4], panel_dir)
    plot_bank_row(bank, row_dir / ROW_FILES[5], panel_dir)
    plot_allocation_row(mopd, row_dir / ROW_FILES[6], panel_dir)
    plot_system_row(mopd, row_dir / ROW_FILES[7], panel_dir)
    plot_capability_row(mopd, capability, row_dir / ROW_FILES[8], panel_dir)

    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "layout": "nine rows with three panels per row",
        "row_count": len(ROW_FILES),
        "panel_count": len(PANEL_FILES),
        "rows": [str(Path("rows") / name) for name in ROW_FILES],
        "panels": [str(Path("panels") / name) for name in PANEL_FILES],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mopd", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--frozen-bank", type=Path, required=True)
    parser.add_argument("--controlled-sampling", type=Path, required=True)
    parser.add_argument("--controlled-moments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = render_all(
        _load_json(args.mopd),
        _load_json(args.capability),
        _load_json(args.frozen_bank),
        _load_csv(args.controlled_sampling),
        _load_csv(args.controlled_moments),
        args.output_dir,
    )
    print(
        f"Single-seed MOPD gallery written to {args.output_dir}: "
        f"{manifest['row_count']} rows, {manifest['panel_count']} panels"
    )


if __name__ == "__main__":
    main()
