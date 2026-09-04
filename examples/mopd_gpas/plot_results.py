#!/usr/bin/env python3
"""Render the protocol-v4 MOPD result figures."""

from __future__ import annotations

import argparse
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
    "uniform": "Uniform",
    "gpas": "GPAS",
    "cost_gpas": "Cost-GPAS",
    "raw_noise": "RawNoise",
    "loss_gap": "LossGap",
    "std_mopd": "StdMOPD",
    "d3_mopd": "D³-MOPD",
    "open_mopd": "Open-MOPD (K=1)",
}
CONFIGS = tuple(CONFIG_LABELS)
COLORS = dict(zip(CONFIGS, plt.get_cmap("tab10").colors[: len(CONFIGS)], strict=True))
TASK_LABELS = {"math": "Math", "code": "Code", "if": "IF", "science": "Science"}
TASK_COLORS = dict(zip(TASKS, plt.get_cmap("Dark2").colors[: len(TASKS)], strict=True))

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
    }
)


def _finish(axis, grid: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid, color="#D9DEE3", linewidth=0.6)
    axis.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_interval(report: dict[str, Any], step: int, config: str) -> list[float]:
    return report["paired_bootstrap"]["checkpoints"][str(step)][config]["weighted_loss_95ci"]


def plot_learning(report: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    panels = (
        (axes[0], ("uniform", "gpas", "raw_noise", "loss_gap"), "step", "Optimizer step"),
        (axes[1], ("uniform", "gpas", "cost_gpas"), "gpu_hours", "GPU-hours (wall time × 2)"),
    )
    for axis, configs, coordinate, label in panels:
        for config in configs:
            curve = report["curves"][config]
            x = np.asarray([row[coordinate] for row in curve], dtype=float)
            y = np.asarray([row["weighted_loss"] for row in curve], dtype=float)
            intervals = np.asarray([_bootstrap_interval(report, int(row["step"]), config) for row in curve])
            axis.plot(x, y, marker="o", markersize=2.5, color=COLORS[config], label=CONFIG_LABELS[config])
            axis.fill_between(x, intervals[:, 0], intervals[:, 1], color=COLORS[config], alpha=0.13)
        axis.axhline(report["common_threshold"], color="#555555", linestyle="--", linewidth=0.8)
        axis.set_xlabel(label)
        axis.set_ylabel("Held-out fixed-objective F")
        axis.legend(frameon=False)
        _finish(axis)
    axes[0].set_title("Synchronization efficiency")
    axes[1].set_title("End-to-end efficiency")
    _save(fig, path)


def plot_task_losses(report: dict[str, Any], path: Path) -> None:
    values = np.asarray([[report["outcomes"][config]["raw_losses"][task] for task in TASKS] for config in CONFIGS])
    figure, axis = plt.subplots(figsize=(6.7, 3.8))
    image = axis.imshow(values, aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(TASKS)), [TASK_LABELS[task] for task in TASKS])
    axis.set_yticks(range(len(CONFIGS)), [CONFIG_LABELS[config] for config in CONFIGS])
    axis.set_title("Final held-out teacher loss (non-fixed objectives: per-task only)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    _save(figure, path)


def plot_dynamics(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), sharex=True)

    uniform = report["system_traces"]["uniform"]
    steps = [row["step"] for row in uniform]
    for task in TASKS:
        axes[0, 0].plot(
            steps,
            [row["scaled_noise"][task] for row in uniform],
            color=TASK_COLORS[task],
            label=f"{TASK_LABELS[task]} scaled",
        )
        axes[0, 0].plot(
            steps,
            [row["raw_noise"][task] for row in uniform],
            color=TASK_COLORS[task],
            linestyle="--",
            alpha=0.75,
            label=f"{TASK_LABELS[task]} raw",
        )
    axes[0, 0].set_yscale("symlog", linthresh=1e-8)
    axes[0, 0].set_ylabel("Noise estimate eᵢ")
    axes[0, 0].set_title("(a) Uniform raw vs. AdamW-scaled noise")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=6)

    for config, linestyle in (("gpas", "-"), ("cost_gpas", "--"), ("d3_mopd", ":")):
        trace = report["system_traces"][config]
        for task in TASKS:
            axes[0, 1].plot(
                [row["step"] for row in trace],
                [row["counts"][task] for row in trace],
                color=TASK_COLORS[task],
                linestyle=linestyle,
                label=f"{CONFIG_LABELS[config]} {TASK_LABELS[task]}",
            )
    axes[0, 1].set_ylim(1.7, 8.3)
    axes[0, 1].set_ylabel("Micro-batches mᵢ")
    axes[0, 1].set_title("(b) Adaptive allocation")
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=6)

    for config in ("gpas", "cost_gpas", "raw_noise", "loss_gap"):
        trace = report["system_traces"][config]
        axes[1, 0].plot(
            [row["step"] for row in trace],
            [row["H"] for row in trace],
            color=COLORS[config],
            label=CONFIG_LABELS[config],
        )
    axes[1, 0].axhline(2.0, color="#555555", linestyle="--", linewidth=0.8, label="bound = 2")
    axes[1, 0].set_ylim(0.5, 2.1)
    axes[1, 0].set_ylabel("H")
    axes[1, 0].set_title("(c) Variance ratio vs. Uniform")
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=6)

    for task in TASKS:
        axes[1, 1].plot(
            steps,
            [row["task_seconds_ema"][task] for row in uniform],
            color=TASK_COLORS[task],
            label=f"τ {TASK_LABELS[task]}",
        )
    ratio_axis = axes[1, 1].twinx()
    ratio_axis.plot(
        steps,
        [row["fixed_to_variable_ratio"] for row in uniform],
        color="#333333",
        linestyle="--",
        label="C / Σmᵢτᵢ",
    )
    axes[1, 1].set_ylabel("Seconds / micro-batch")
    ratio_axis.set_ylabel("Fixed / variable cost")
    axes[1, 1].set_title("(d) Measured time model")
    lines = axes[1, 1].lines + ratio_axis.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], frameon=False, ncol=2, fontsize=6)
    ratio_axis.grid(False)

    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        _finish(axis)
    _save(figure, path)


def plot_system(report: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    x = np.arange(len(CONFIGS))
    outcomes = report["outcomes"]
    axes[0].bar(x, [outcomes[c]["gpu_hours"] for c in CONFIGS], color=[COLORS[c] for c in CONFIGS])
    axes[0].set_ylabel("GPU-hours")
    axes[0].set_title("Full-run cost")
    axes[1].bar(
        x,
        [outcomes[c]["fixed_to_variable_ratio_median"] for c in CONFIGS],
        color=[COLORS[c] for c in CONFIGS],
    )
    axes[1].set_ylabel("C / Σ mᵢτᵢ")
    axes[1].set_title("Measured fixed-cost ratio")
    axes[2].bar(
        x,
        [outcomes[c]["step_time_p50_seconds"] for c in CONFIGS],
        yerr=[outcomes[c]["step_time_p95_seconds"] - outcomes[c]["step_time_p50_seconds"] for c in CONFIGS],
        color=[COLORS[c] for c in CONFIGS],
        capsize=2,
    )
    axes[2].set_ylabel("Seconds")
    axes[2].set_title("Step time: p50 to p95")
    for axis in axes:
        axis.set_xticks(x, [CONFIG_LABELS[c] for c in CONFIGS], rotation=35, ha="right")
        _finish(axis)
    _save(figure, path)


def plot_capability(report: dict[str, Any], path: Path) -> None:
    values = np.asarray(
        [[report["configs"][config]["domains"][task]["normalized_gain"] for task in TASKS] for config in CONFIGS]
    )
    figure, axis = plt.subplots(figsize=(6.7, 3.8))
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(TASKS)), [TASK_LABELS[task] for task in TASKS])
    axis.set_yticks(range(len(CONFIGS)), [CONFIG_LABELS[config] for config in CONFIGS])
    axis.set_title("Normalized capability gain")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    _save(figure, path)


def plot_variance(report: dict[str, Any], path: Path) -> None:
    checkpoints = (50, 250, 500)
    methods = ("uniform", "raw_noise", "loss_gap", "gpas", "cost_gpas")
    values = np.asarray(
        [[report["checkpoints"][str(step)][method]["relative_variance"] for method in methods] for step in checkpoints]
    )
    figure, axis = plt.subplots(figsize=(7.0, 3.4))
    width = 0.15
    x = np.arange(len(checkpoints))
    for index, method in enumerate(methods):
        axis.bar(x + (index - 2) * width, values[:, index], width, label=CONFIG_LABELS[method])
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    axis.set_xticks(x, [f"step {step}" for step in checkpoints])
    axis.set_ylabel("Held-out gradient variance / Uniform")
    axis.legend(frameon=False, ncol=3)
    _finish(axis)
    _save(figure, path)


def render_all(
    mopd: dict[str, Any], capability: dict[str, Any], output_dir: Path, variance: dict[str, Any] | None = None
) -> dict[str, Any]:
    if mopd["seed"] != SEED or capability["seed"] != SEED:
        raise ValueError("result gallery requires the frozen single training seed 42")
    if tuple(mopd["configs"]) != CONFIGS or set(capability["configs"]) != set(CONFIGS):
        raise ValueError("result reports do not contain the frozen eight configurations")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "learning_efficiency": output_dir / "learning_efficiency.pdf",
        "task_losses": output_dir / "task_losses.pdf",
        "allocation_dynamics": output_dir / "allocation_dynamics.pdf",
        "system_costs": output_dir / "system_costs.pdf",
        "capability": output_dir / "capability.pdf",
    }
    plot_learning(mopd, figures["learning_efficiency"])
    plot_task_losses(mopd, figures["task_losses"])
    plot_dynamics(mopd, figures["allocation_dynamics"])
    plot_system(mopd, figures["system_costs"])
    plot_capability(capability, figures["capability"])
    if variance is not None:
        figures["heldout_gradient_variance"] = output_dir / "heldout_gradient_variance.pdf"
        plot_variance(variance, figures["heldout_gradient_variance"])
    manifest = {
        "schema_version": 4,
        "seed": SEED,
        "figures": {name: path.name for name, path in figures.items()},
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mopd", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--heldout-variance", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    variance = None if args.heldout_variance is None else json.loads(args.heldout_variance.read_text())
    render_all(
        json.loads(args.mopd.read_text()),
        json.loads(args.capability.read_text()),
        args.output_dir,
        variance,
    )
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
