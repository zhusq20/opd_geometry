#!/usr/bin/env python3
"""Plot empirical parameter-space geometry for the current four GRPO runs.

The trajectory panels use the online 256-dimensional CountSketch displacement
vectors saved during training.  Every quantitative norm, support fraction, and
cosine panel uses exact full-parameter statistics computed from torch-dist
checkpoints by ``analyze_checkpoint_updates.py`` or by the online exact
geometry observer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

TASKS = ("math", "code", "knowledge", "if")
DISPLAY = {
    "math": "Math",
    "code": "Code",
    "knowledge": "Knowledge",
    "if": "Instruction following",
}
SHORT_DISPLAY = {
    "math": "Math",
    "code": "Code",
    "knowledge": "Knowledge",
    "if": "IF",
}
COLORS = {
    "math": "#0072B2",
    "code": "#D55E00",
    "knowledge": "#009E73",
    "if": "#CC79A7",
}
RUN_PATHS = {
    "math": (
        "outputs/qwen3_1.7b_math_code_grpo_20260817T065129Z/"
        "qwen3_1.7b_math_grpo_adamw_responsive16_trainr8192_evalr32768_seed42"
    ),
    "code": (
        "outputs/qwen3_1.7b_code_grpo_after_sandbox_20260817T194524Z/"
        "qwen3_1.7b_code_grpo_adamw_responsive16_trainr8192_seed42"
    ),
    "knowledge": (
        "outputs/qwen3_1.7b_science_grpo_adamw_lr1e-6_n4_20260817T0655Z/"
        "qwen3_1.7b_science_grpo_adamw_responsive16_trainr8192_seed42"
    ),
    "if": (
        "outputs/qwen3_1.7b_if_grpo_eosfix_dualeval_20260818T202612Z/"
        "qwen3_1.7b_if_grpo_adamw_responsive16_trainr8192_evalr32768_seed42"
    ),
}


@dataclass
class TaskDynamics:
    task: str
    rows: list[dict[str, float]]
    alpha: float
    alpha_r2: float


@dataclass
class ProjectionResult:
    rows: list[dict[str, float | str]]
    energy_fraction: np.ndarray
    checkpoint_updates: tuple[int, ...]


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair_name(left: str, right: str) -> str:
    if TASKS.index(left) > TASKS.index(right):
        left, right = right, left
    return f"{left}__{right}"


def read_exact_summaries(exact_root: Path) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    for update in (100, 200, 300):
        path = exact_root / f"exact_u{update}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run analyze_checkpoint_updates.py at the four matched checkpoints first."
            )
        summary = json.loads(path.read_text(encoding="utf-8"))
        labels = [row["label"] for row in summary["checkpoints"]]
        if labels != list(TASKS):
            raise ValueError(f"{path}: expected checkpoint order {TASKS}, received {labels}")
        expected_iteration = update - 1
        if any(int(row["iteration"]) != expected_iteration for row in summary["checkpoints"]):
            raise ValueError(f"{path}: checkpoint iteration does not match update {update}")
        summaries[update] = summary
    return summaries


def read_dynamics(task: str, run_path: Path, max_update: int) -> TaskDynamics:
    metrics_path = run_path / "geometry" / "actor" / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    by_update: dict[int, dict[str, float]] = {}
    with metrics_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            update = int(record.get("num_updates", -1))
            if (
                update < 1
                or update > max_update
                or not record.get("valid_update_metrics", False)
                or not record.get("update_successful", False)
            ):
                continue
            global_row = record["groups"]["global"]
            by_update[update] = {
                "update": float(update),
                "step_norm": float(global_row["delta_model_l2"]),
                "displacement_norm": float(global_row["displacement_l2"]),
                "step_displacement_cosine": float(global_row["cos_delta_model_displacement"]),
                "step_changed_fraction": float(global_row["model_change_fraction"]),
            }
    if sorted(by_update) != list(range(1, max_update + 1)):
        missing = sorted(set(range(1, max_update + 1)).difference(by_update))
        raise ValueError(f"{task}: incomplete exact dynamics through update {max_update}: {missing[:10]}")

    rows = [by_update[update] for update in sorted(by_update)]
    path_length = 0.0
    for row in rows:
        path_length += row["step_norm"]
        row["path_length"] = path_length
        row["path_efficiency"] = row["displacement_norm"] / path_length

    fit_rows = [row for row in rows if row["update"] >= 25]
    log_update = np.log([row["update"] for row in fit_rows])
    log_distance = np.log([row["displacement_norm"] for row in fit_rows])
    alpha, intercept = np.polyfit(log_update, log_distance, 1)
    prediction = intercept + alpha * log_update
    residual_sum = float(np.sum((log_distance - prediction) ** 2))
    total_sum = float(np.sum((log_distance - np.mean(log_distance)) ** 2))
    return TaskDynamics(
        task=task,
        rows=rows,
        alpha=float(alpha),
        alpha_r2=1.0 - residual_sum / total_sum,
    )


def load_sketch_trajectories(run_paths: dict[str, Path], max_update: int) -> ProjectionResult:
    import torch

    raw_rows: list[tuple[str, int, np.ndarray]] = []
    signatures: list[tuple[int, int, tuple[str, ...]]] = []
    file_pattern = re.compile(r"rollout_(\d+)_step_\d+_obs_\d+\.pt")
    for task in TASKS:
        actor_dir = run_paths[task] / "geometry" / "actor"
        initial = torch.load(actor_dir / "initial_projection.pt", map_location="cpu", weights_only=True)
        signatures.append(
            (
                int(initial["seed"]),
                int(initial["projection_dim"]),
                tuple(initial["group_names"]),
            )
        )
        for path in sorted((actor_dir / "vectors").glob("*.pt")):
            match = file_pattern.fullmatch(path.name)
            if match is None:
                continue
            update = int(match.group(1)) + 1
            if update > max_update:
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["projection"] != "countsketch" or int(payload["projection_dim"]) != 256:
                raise ValueError(f"Unexpected projection metadata in {path}")
            displacement = payload["groups"]["global"]["displacement"]
            raw_rows.append((task, update, displacement.numpy().astype(np.float64, copy=False)))
    if len(set(signatures)) != 1:
        raise ValueError("The four runs do not share the same CountSketch mapping signature")
    for task in TASKS:
        updates = [update for row_task, update, _ in raw_rows if row_task == task]
        if not updates or updates[0] != 1 or updates[-1] < max_update - 16:
            raise ValueError(f"{task}: insufficient trajectory sketch coverage: {updates[:1]}...{updates[-1:]}")

    matrix = np.stack([vector for _, _, vector in raw_rows])
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    coordinates = matrix @ right_vectors[:4].T
    final_indices = [
        max(index for index, row in enumerate(raw_rows) if row[0] == task) for task in TASKS
    ]
    for component in range(4):
        anchor = final_indices[int(np.argmax(np.abs(coordinates[final_indices, component])))]
        if coordinates[anchor, component] < 0:
            coordinates[:, component] *= -1.0
    energy = singular_values**2
    energy /= energy.sum()
    rows: list[dict[str, float | str]] = []
    for (task, update, vector), coordinate in zip(raw_rows, coordinates, strict=True):
        rows.append(
            {
                "task": task,
                "update": update,
                "sketch_norm": float(np.linalg.norm(vector)),
                "pc1": float(coordinate[0]),
                "pc2": float(coordinate[1]),
                "pc3": float(coordinate[2]),
                "pc4": float(coordinate[3]),
            }
        )
    common_updates = tuple(
        sorted(
            set.intersection(
                *(
                    {int(row["update"]) for row in rows if row["task"] == task}
                    for task in TASKS
                )
            )
        )
    )
    return ProjectionResult(rows=rows, energy_fraction=energy, checkpoint_updates=common_updates)


def panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
    )


def plot_trajectory_axis(
    axis: Any,
    projection: ProjectionResult,
    x_component: int,
    y_component: int,
) -> None:
    x_key = f"pc{x_component}"
    y_key = f"pc{y_component}"
    axis.scatter([0.0], [0.0], marker="*", s=42, color="#222222", zorder=5)
    axis.annotate("base", (0.0, 0.0), xytext=(4, 3), textcoords="offset points", fontsize=6.2)
    for task in TASKS:
        rows = sorted(
            (row for row in projection.rows if row["task"] == task),
            key=lambda row: int(row["update"]),
        )
        x = np.asarray([0.0] + [float(row[x_key]) for row in rows])
        y = np.asarray([0.0] + [float(row[y_key]) for row in rows])
        color = COLORS[task]
        axis.plot(x, y, color=color, alpha=0.92, label=SHORT_DISPLAY[task])
        marker_indices = [
            index + 1
            for index, row in enumerate(rows)
            if int(row["update"]) in {97, 193, 289}
        ]
        axis.scatter(
            x[marker_indices],
            y[marker_indices],
            s=14,
            facecolors="white",
            edgecolors=color,
            linewidths=0.8,
            zorder=4,
        )
        axis.annotate(
            SHORT_DISPLAY[task],
            (x[-1], y[-1]),
            xytext=(4, 1),
            textcoords="offset points",
            color=color,
            fontsize=6.3,
            va="center",
        )
    axis.axhline(0.0, color="#777777", linewidth=0.45, alpha=0.45)
    axis.axvline(0.0, color="#777777", linewidth=0.45, alpha=0.45)
    axis.grid(alpha=0.16, linewidth=0.45)
    axis.set_xlabel(
        f"PC{x_component} ({100 * projection.energy_fraction[x_component - 1]:.1f}% sketch energy)"
    )
    axis.set_ylabel(
        f"PC{y_component} ({100 * projection.energy_fraction[y_component - 1]:.1f}% sketch energy)"
    )
    axis.set_aspect("equal", adjustable="datalim")


def endpoint_matrix(axis: Any, summary: dict[str, Any]) -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    global_row = summary["global"]
    size = len(TASKS)
    matrix = np.full((size, size), np.nan, dtype=np.float64)
    off_diagonal: list[float] = []
    for row_index, left in enumerate(TASKS):
        for column_index, right in enumerate(TASKS):
            if row_index == column_index:
                continue
            value = float(global_row[f"cosine/{pair_name(left, right)}"])
            matrix[row_index, column_index] = value
            off_diagonal.append(value)
    limit = max(0.008, 1.08 * max(abs(value) for value in off_diagonal))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8E8E8")
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks(np.arange(size), [SHORT_DISPLAY[task] for task in TASKS], rotation=30, ha="right")
    axis.set_yticks(np.arange(size), [SHORT_DISPLAY[task] for task in TASKS])
    for row_index, left in enumerate(TASKS):
        for column_index, right in enumerate(TASKS):
            if row_index == column_index:
                norm = float(global_row[f"update_norm/{left}"])
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1.0,
                        1.0,
                        facecolor="#E8E8E8",
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
                axis.text(
                    column_index,
                    row_index,
                    f"$\\|\\Delta\\theta\\|_2$\n{norm:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                )
            else:
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:+.4f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                )
    axis.spines[:].set_visible(False)
    axis.tick_params(length=0)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.047, pad=0.04)
    colorbar.set_label(r"cosine$(\Delta\theta_i,\Delta\theta_j)$ (zoomed)", fontsize=6.7)
    colorbar.ax.tick_params(labelsize=6.1)
    return image


def trajectory_figure(
    projection: ProjectionResult,
    dynamics: dict[str, TaskDynamics],
    exact: dict[int, dict[str, Any]],
) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(7.15, 6.2), gridspec_kw={"hspace": 0.40, "wspace": 0.34})
    plot_trajectory_axis(axes[0, 0], projection, 1, 2)
    axes[0, 0].set_title("Parameter trajectory: PC1–PC2", fontweight="bold")
    panel_label(axes[0, 0], "a")
    plot_trajectory_axis(axes[0, 1], projection, 3, 4)
    axes[0, 1].set_title("Parameter trajectory: PC3–PC4", fontweight="bold")
    panel_label(axes[0, 1], "b")

    axis = axes[1, 0]
    final_efficiencies: list[float] = []
    for task in TASKS:
        rows = dynamics[task].rows
        axis.plot(
            [row["update"] for row in rows],
            [row["path_efficiency"] for row in rows],
            color=COLORS[task],
            label=SHORT_DISPLAY[task],
        )
        final_efficiencies.append(100.0 * rows[-1]["path_efficiency"])
    axis.set_yscale("log")
    axis.set_ylim(0.025, 1.15)
    axis.set_xlim(1, 307)
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel(r"Net displacement / traveled path")
    axis.set_title("Curved paths retain only ≈4% of traveled distance", fontweight="bold")
    axis.grid(alpha=0.20, linewidth=0.45, which="both")
    axis.text(
        0.98,
        0.08,
        f"update 300: {min(final_efficiencies):.2f}–{max(final_efficiencies):.2f}%",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#555555",
    )
    panel_label(axis, "c")

    endpoint_matrix(axes[1, 1], exact[300])
    axes[1, 1].set_title("Exact 300-step update geometry", fontweight="bold")
    panel_label(axes[1, 1], "d")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=4,
        frameon=False,
        handlelength=2.1,
    )
    figure.text(
        0.5,
        0.006,
        "Trajectory: shared 256-D CountSketch every 16 updates (updates ≤300; top four PCs retain "
        f"{100 * projection.energy_fraction[:4].sum():.1f}% energy). Curvature and matrix: exact BF16 realized updates.",
        ha="center",
        fontsize=5.9,
        color="#555555",
    )
    figure.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.09)
    return figure


def evidence_figure(
    dynamics: dict[str, TaskDynamics],
    exact: dict[int, dict[str, Any]],
) -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.75), gridspec_kw={"hspace": 0.42, "wspace": 0.34})

    axis = axes[0, 0]
    for task in TASKS:
        item = dynamics[task]
        axis.plot(
            [row["update"] for row in item.rows],
            [row["displacement_norm"] for row in item.rows],
            color=COLORS[task],
            label=rf"{SHORT_DISPLAY[task]}  $\alpha$={item.alpha:.2f}",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel(r"Exact $\|\theta_t-\theta_0\|_2$")
    axis.set_title(r"Sublinear displacement: $\|\Delta\theta_t\|\propto t^\alpha$", fontweight="bold")
    axis.grid(alpha=0.20, linewidth=0.45, which="both")
    axis.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8)
    panel_label(axis, "a")

    axis = axes[0, 1]
    updates = np.asarray([100, 200, 300])
    endpoint_offsets = {"math": -9, "code": 9, "knowledge": 0, "if": 0}
    for task in TASKS:
        support = [100.0 * exact[update]["global"][f"changed_fraction/{task}"] for update in updates]
        axis.plot(updates, support, marker="o", markersize=3.5, color=COLORS[task], label=SHORT_DISPLAY[task])
        axis.annotate(
            f"{support[-1]:.1f}%",
            (updates[-1], support[-1]),
            xytext=(3, endpoint_offsets[task]),
            textcoords="offset points",
            color=COLORS[task],
            fontsize=6.1,
            va="center",
        )
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel("Coordinates changed from base (%)")
    axis.set_title("Realized BF16 support expands", fontweight="bold")
    axis.set_xticks(updates)
    axis.grid(alpha=0.20, linewidth=0.45)
    panel_label(axis, "b")

    axis = axes[1, 0]
    pair_styles = {
        "math__code": ("#0072B2", "-"),
        "math__knowledge": ("#009E73", "-"),
        "code__knowledge": ("#E69F00", "-"),
        "math__if": ("#CC79A7", "--"),
        "code__if": ("#D55E00", "--"),
        "knowledge__if": ("#6A3D9A", "--"),
    }
    for pair, (color, linestyle) in pair_styles.items():
        left, right = pair.split("__")
        values = [exact[update]["global"][f"cosine/{pair}"] for update in updates]
        axis.plot(
            updates,
            values,
            marker="o",
            markersize=3.0,
            color=color,
            linestyle=linestyle,
            label=f"{SHORT_DISPLAY[left]}–{SHORT_DISPLAY[right]}",
        )
        endpoint_offset = {"math__if": 5, "code__if": -5}.get(pair, 0)
        axis.annotate(
            f"{SHORT_DISPLAY[left]}–{SHORT_DISPLAY[right]}",
            (updates[-1], values[-1]),
            xytext=(4, endpoint_offset),
            textcoords="offset points",
            color=color,
            fontsize=5.8,
            va="center",
        )
    axis.axhline(0.0, color="#333333", linewidth=0.7)
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel(r"Exact cosine$(\Delta\theta_i,\Delta\theta_j)$")
    axis.set_title("Task directions stay within 0.43° of orthogonal", fontweight="bold")
    axis.set_xticks(updates)
    axis.set_xlim(90, 360)
    axis.set_ylim(-0.0085, 0.0085)
    axis.grid(alpha=0.20, linewidth=0.45)
    panel_label(axis, "c")

    axis = axes[1, 1]
    layer_rows = exact[300]["layerwise"]
    layer_indices = [int(str(row["scope"]).split("/", 1)[1]) for row in layer_rows]
    pairs = list(pair_styles)
    heatmap = np.asarray(
        [[float(row[f"cosine/{pair}"]) for row in layer_rows] for pair in pairs],
        dtype=np.float64,
    )
    limit = max(0.022, 1.04 * float(np.max(np.abs(heatmap))))
    image = axis.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks(
        [index for index, layer in enumerate(layer_indices) if layer % 4 == 0 or layer == layer_indices[-1]],
        [str(layer) for layer in layer_indices if layer % 4 == 0 or layer == layer_indices[-1]],
    )
    axis.set_yticks(
        np.arange(len(pairs)),
        [
            f"{SHORT_DISPLAY[pair.split('__')[0]]}–{SHORT_DISPLAY[pair.split('__')[1]]}"
            for pair in pairs
        ],
    )
    axis.set_xlabel("Transformer layer")
    axis.set_title("Weak coupling is structured by layer", fontweight="bold")
    axis.axvline(14.5, color="#333333", linewidth=0.55, linestyle=(0, (2, 2)), alpha=0.65)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.04)
    colorbar.set_label("layer cosine", fontsize=6.7)
    colorbar.ax.tick_params(labelsize=6.1)
    panel_label(axis, "d")

    figure.text(
        0.5,
        0.006,
        "All panels use exact full-parameter BF16 realized updates. Support means coordinates unequal to the shared base checkpoint.",
        ha="center",
        fontsize=5.9,
        color="#555555",
    )
    figure.subplots_adjust(left=0.095, right=0.97, top=0.94, bottom=0.09)
    return figure


def single_panel_figures(
    projection: ProjectionResult,
    dynamics: dict[str, TaskDynamics],
    exact: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Build title-free standalone figures suitable for LaTeX subfigures."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    figures: dict[str, Any] = {}

    for x_component, y_component in ((1, 2), (3, 4)):
        figure, axis = plt.subplots(figsize=(3.45, 3.0))
        plot_trajectory_axis(axis, projection, x_component, y_component)
        figure.tight_layout(pad=0.35)
        figures[f"grpo_trajectory_pc{x_component}{y_component}"] = figure

    figure, axis = plt.subplots(figsize=(3.45, 2.75))
    for task in TASKS:
        rows = dynamics[task].rows
        axis.plot(
            [row["update"] for row in rows],
            [row["path_efficiency"] for row in rows],
            color=COLORS[task],
            label=SHORT_DISPLAY[task],
        )
    axis.set_yscale("log")
    axis.set_ylim(0.025, 1.15)
    axis.set_xlim(1, 307)
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel("Net displacement / traveled path")
    axis.grid(alpha=0.20, linewidth=0.45, which="both")
    axis.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.8)
    figure.tight_layout(pad=0.35)
    figures["grpo_path_efficiency"] = figure

    figure, axis = plt.subplots(figsize=(3.55, 3.15))
    endpoint_matrix(axis, exact[300])
    figure.subplots_adjust(left=0.25, right=0.88, bottom=0.21, top=0.98)
    figures["grpo_endpoint_geometry"] = figure

    figure, axis = plt.subplots(figsize=(3.45, 2.75))
    for task in TASKS:
        item = dynamics[task]
        axis.plot(
            [row["update"] for row in item.rows],
            [row["displacement_norm"] for row in item.rows],
            color=COLORS[task],
            label=rf"{SHORT_DISPLAY[task]}  $\alpha$={item.alpha:.2f}",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel(r"Exact $\|\theta_t-\theta_0\|_2$")
    axis.grid(alpha=0.20, linewidth=0.45, which="both")
    axis.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8)
    figure.tight_layout(pad=0.35)
    figures["grpo_displacement_scaling"] = figure

    figure, axis = plt.subplots(figsize=(3.45, 2.75))
    updates = np.asarray([100, 200, 300])
    endpoint_offsets = {"math": -10, "code": 10, "knowledge": 0, "if": 0}
    for task in TASKS:
        support = [100.0 * exact[update]["global"][f"changed_fraction/{task}"] for update in updates]
        axis.plot(updates, support, marker="o", markersize=3.5, color=COLORS[task])
        axis.annotate(
            f"{SHORT_DISPLAY[task]} {support[-1]:.1f}%",
            (updates[-1], support[-1]),
            xytext=(4, endpoint_offsets[task]),
            textcoords="offset points",
            color=COLORS[task],
            fontsize=6.2,
            va="center",
        )
    axis.set_xlim(90, 355)
    axis.set_xticks(updates)
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel("Coordinates changed from base (%)")
    axis.grid(alpha=0.20, linewidth=0.45)
    figure.tight_layout(pad=0.35)
    figures["grpo_bf16_support"] = figure

    pair_styles = {
        "math__code": ("#0072B2", "-"),
        "math__knowledge": ("#009E73", "-"),
        "code__knowledge": ("#E69F00", "-"),
        "math__if": ("#CC79A7", "--"),
        "code__if": ("#D55E00", "--"),
        "knowledge__if": ("#6A3D9A", "--"),
    }
    figure, axis = plt.subplots(figsize=(4.15, 2.85))
    for pair, (color, linestyle) in pair_styles.items():
        left, right = pair.split("__")
        values = [exact[update]["global"][f"cosine/{pair}"] for update in updates]
        axis.plot(
            updates,
            values,
            marker="o",
            markersize=3.0,
            color=color,
            linestyle=linestyle,
        )
        endpoint_offset = {"math__if": 5, "code__if": -5}.get(pair, 0)
        axis.annotate(
            f"{SHORT_DISPLAY[left]}–{SHORT_DISPLAY[right]}",
            (updates[-1], values[-1]),
            xytext=(4, endpoint_offset),
            textcoords="offset points",
            color=color,
            fontsize=5.8,
            va="center",
        )
    axis.axhline(0.0, color="#333333", linewidth=0.7)
    axis.set_xlim(90, 360)
    axis.set_ylim(-0.0085, 0.0085)
    axis.set_xticks(updates)
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel(r"Exact cosine$(\Delta\theta_i,\Delta\theta_j)$")
    axis.grid(alpha=0.20, linewidth=0.45)
    figure.tight_layout(pad=0.35)
    figures["grpo_pairwise_cosine"] = figure

    figure, axis = plt.subplots(figsize=(5.15, 2.7))
    layer_rows = exact[300]["layerwise"]
    layer_indices = [int(str(row["scope"]).split("/", 1)[1]) for row in layer_rows]
    pairs = list(pair_styles)
    heatmap = np.asarray(
        [[float(row[f"cosine/{pair}"]) for row in layer_rows] for pair in pairs],
        dtype=np.float64,
    )
    limit = max(0.022, 1.04 * float(np.max(np.abs(heatmap))))
    image = axis.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    tick_indices = [
        index for index, layer in enumerate(layer_indices) if layer % 4 == 0 or layer == layer_indices[-1]
    ]
    axis.set_xticks(tick_indices, [str(layer_indices[index]) for index in tick_indices])
    axis.set_yticks(
        np.arange(len(pairs)),
        [
            f"{SHORT_DISPLAY[pair.split('__')[0]]}–{SHORT_DISPLAY[pair.split('__')[1]]}"
            for pair in pairs
        ],
    )
    axis.set_xlabel("Transformer layer")
    axis.axvline(14.5, color="#333333", linewidth=0.55, linestyle=(0, (2, 2)), alpha=0.65)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.038, pad=0.025)
    colorbar.set_label("Layer cosine", fontsize=6.7)
    colorbar.ax.tick_params(labelsize=6.1)
    figure.subplots_adjust(left=0.23, right=0.91, bottom=0.19, top=0.98)
    figures["grpo_layerwise_cosine"] = figure

    return figures


def analysis_rows(
    dynamics: dict[str, TaskDynamics],
    exact: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    for task in TASKS:
        item = dynamics[task]
        late_cosines = [row["step_displacement_cosine"] for row in item.rows[-50:]]
        row: dict[str, Any] = {
            "task": task,
            "power_law_alpha_updates25_to300": item.alpha,
            "power_law_log_r2": item.alpha_r2,
            "update300_displacement_norm": item.rows[-1]["displacement_norm"],
            "update300_cumulative_path_length": item.rows[-1]["path_length"],
            "update300_path_efficiency": item.rows[-1]["path_efficiency"],
            "late50_step_displacement_cosine": float(np.mean(late_cosines)),
        }
        for update in (100, 200, 300):
            global_row = exact[update]["global"]
            row[f"update{update}_norm"] = global_row[f"update_norm/{task}"]
            row[f"update{update}_relative_norm"] = global_row[f"relative_update_norm/{task}"]
            row[f"update{update}_changed_fraction"] = global_row[f"changed_fraction/{task}"]
        task_rows.append(row)

    pair_rows: list[dict[str, Any]] = []
    for update in (100, 200, 300):
        global_row = exact[update]["global"]
        for left_index, left in enumerate(TASKS):
            for right in TASKS[left_index + 1 :]:
                pair = pair_name(left, right)
                row = {
                    "update": update,
                    "task_left": left,
                    "task_right": right,
                    "global_dot": global_row[f"dot/{pair}"],
                    "global_cosine": global_row[f"cosine/{pair}"],
                }
                if update == 300:
                    layer_values = np.asarray(
                        [item[f"cosine/{pair}"] for item in exact[300]["layerwise"]],
                        dtype=np.float64,
                    )
                    extreme_index = int(np.argmax(np.abs(layer_values)))
                    row.update(
                        {
                            "layer_cosine_mean": float(layer_values.mean()),
                            "layer_cosine_min": float(layer_values.min()),
                            "layer_cosine_max": float(layer_values.max()),
                            "max_abs_layer": extreme_index,
                            "max_abs_layer_cosine": float(layer_values[extreme_index]),
                        }
                    )
                pair_rows.append(row)
    return task_rows, pair_rows


def save_figure(figure: Any, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.pdf")
    figure.savefig(output_dir / f"{stem}.png", dpi=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=ROOT / "outputs/raw_gradient_interference/paper/parameter_geometry",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/raw_gradient_interference/paper/figures/grpo_parameter_geometry",
    )
    parser.add_argument("--max-update", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_update != 300:
        parser.error("The matched-checkpoint paper layout currently requires --max-update 300.")
    return args


def main() -> None:
    args = parse_args()
    exact_root = args.exact_root if args.exact_root.is_absolute() else ROOT / args.exact_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    exact_root = exact_root.resolve()
    output_dir = output_dir.resolve()
    expected = ("grpo_parameter_space.pdf", "grpo_geometry_evidence.pdf", "grpo_parameter_geometry_analysis.pdf")
    if output_dir.exists() and not args.force and any((output_dir / name).exists() for name in expected):
        raise FileExistsError(f"Outputs already exist in {output_dir}; pass --force to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_paths = {task: (ROOT / RUN_PATHS[task]).resolve() for task in TASKS}
    exact = read_exact_summaries(exact_root)
    dynamics = {task: read_dynamics(task, run_paths[task], args.max_update) for task in TASKS}
    projection = load_sketch_trajectories(run_paths, args.max_update)
    task_rows, pair_rows = analysis_rows(dynamics, exact)

    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    trajectory = trajectory_figure(projection, dynamics, exact)
    evidence = evidence_figure(dynamics, exact)
    standalone = single_panel_figures(projection, dynamics, exact)
    save_figure(trajectory, output_dir, "grpo_parameter_space")
    save_figure(evidence, output_dir, "grpo_geometry_evidence")
    for stem, figure in standalone.items():
        save_figure(figure, output_dir, stem)
    with PdfPages(
        output_dir / "grpo_parameter_geometry_analysis.pdf",
        metadata={
            "Title": "Four-task GRPO parameter-space geometry",
            "Author": "Generated from local experiment artifacts",
            "Subject": "Exact checkpoint-update geometry and online CountSketch trajectories",
        },
    ) as pdf:
        pdf.savefig(trajectory)
        pdf.savefig(evidence)
    plt.close(trajectory)
    plt.close(evidence)
    for figure in standalone.values():
        plt.close(figure)

    write_csv(output_dir / "task_geometry_summary.csv", task_rows)
    write_csv(output_dir / "pairwise_geometry_summary.csv", pair_rows)
    write_csv(output_dir / "trajectory_projection.csv", projection.rows)

    input_manifest: dict[str, Any] = {"exact": {}, "online": {}}
    for update in (100, 200, 300):
        path = exact_root / f"exact_u{update}" / "summary.json"
        input_manifest["exact"][str(update)] = {"path": str(path), "sha256": sha256(path)}
    for task in TASKS:
        metrics_path = run_paths[task] / "geometry" / "actor" / "metrics.jsonl"
        vector_paths = sorted((run_paths[task] / "geometry" / "actor" / "vectors").glob("*.pt"))
        input_manifest["online"][task] = {
            "run_path": str(run_paths[task]),
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256(metrics_path),
            "trajectory_vector_count": len(vector_paths),
            "trajectory_vectors_sha256": hashlib.sha256(
                "".join(f"{path.name}:{sha256(path)}\n" for path in vector_paths).encode()
            ).hexdigest(),
        }
    figure_paths = sorted(output_dir.glob("grpo_*.pdf")) + sorted(output_dir.glob("grpo_*.png"))
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_paper": "https://arxiv.org/abs/2608.03573v2",
        "inputs": input_manifest,
        "trajectory_projection": {
            "method": "Uncentered SVD of shared-seed 256-D CountSketch cumulative parameter displacements",
            "max_update": args.max_update,
            "saved_interval_updates": 16,
            "energy_fraction_pc1_to_pc4": projection.energy_fraction[:4].tolist(),
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in figure_paths
        },
        "limitations": [
            "All four training runs use seed 42; no training-seed uncertainty is estimated.",
            "Trajectory panels are CountSketch projections for visualization; all reported norms and cosines are exact.",
            "Checkpoint deltas are cumulative updates from separate runs, not same-checkpoint per-task raw gradients.",
            "Changed-coordinate support is measured on BF16 realized model tensors and is precision dependent.",
            "There is no matched SFT baseline in these artifacts, so the figures do not establish an RL-versus-SFT contrast.",
        ],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "CAPTIONS.md").write_text(
        "# Suggested captions\n\n"
        "**Four-task GRPO trajectories in parameter space.** (a,b) Online cumulative parameter "
        "displacements projected with a shared 256-dimensional CountSketch and visualized in the "
        "first four uncentered principal directions. Open circles mark updates 97, 193, and 289. "
        "(c) The ratio between net displacement and cumulative realized step length falls to about "
        "4%, showing that each optimizer path is strongly curved rather than a straight ray. "
        "(d) Exact full-parameter geometry after 300 matched updates: diagonal cells report update "
        "norms and off-diagonal cells report task-pair cosines. The task directions are nearly "
        "orthogonal despite comparable displacement magnitudes.\n\n"
        "**Multi-scale evidence for task-specific GRPO updates.** (a) Exact displacement grows "
        "sublinearly with optimizer updates; fitted exponents use updates 25--300. (b) The fraction "
        "of BF16 coordinates that differ from the shared base checkpoint grows steadily, with the "
        "instruction-following run changing the largest support. (c) Exact cross-task cosines remain "
        "within 0.0075 in magnitude at 100, 200, and 300 matched updates. (d) Layer-wise cosines "
        "reveal weak but structured coupling hidden by the global average, including negative "
        "Knowledge--IF coupling around the middle layers. These are cumulative checkpoint updates "
        "from separate seed-42 runs, not same-state raw-gradient interference measurements.\n",
        encoding="utf-8",
    )
    (output_dir / "SINGLE_PANEL_CAPTIONS.md").write_text(
        "# Standalone figure captions\n\n"
        "## `grpo_trajectory_pc12.pdf`\n\n"
        "Online GRPO parameter trajectories for Math, Code, Knowledge, and instruction following "
        "in the first two uncentered principal directions of a shared 256-dimensional CountSketch. "
        "The star denotes the shared base model; open circles mark updates 97, 193, and 289.\n\n"
        "## `grpo_trajectory_pc34.pdf`\n\n"
        "The same online GRPO parameter trajectories in principal directions three and four. "
        "Together, the first four projected directions retain 69.7% of the trajectory-sketch energy.\n\n"
        "## `grpo_path_efficiency.pdf`\n\n"
        "Ratio between exact net parameter displacement and cumulative realized optimizer-step "
        "length. After 300 updates, only 3.74--3.91% of the traveled distance remains as net "
        "displacement, indicating strongly curved optimization paths.\n\n"
        "## `grpo_endpoint_geometry.pdf`\n\n"
        "Exact full-parameter geometry after 300 matched GRPO updates. Diagonal cells report "
        "the realized update norm; off-diagonal cells report cosine similarity between task-specific "
        "cumulative updates. All cross-task cosines have magnitude below 0.007.\n\n"
        "## `grpo_displacement_scaling.pdf`\n\n"
        "Exact parameter displacement as a function of optimizer updates. Power-law exponents are "
        "fitted over updates 25--300; instruction following grows faster ($\\alpha=0.73$) than "
        "Math, Code, and Knowledge ($\\alpha=0.62$--$0.64$).\n\n"
        "## `grpo_bf16_support.pdf`\n\n"
        "Fraction of realized BF16 model coordinates that differ from the shared base checkpoint "
        "at matched training horizons. Instruction following reaches 12.2% at update 300, compared "
        "with approximately 10.5--10.6% for the other tasks.\n\n"
        "## `grpo_pairwise_cosine.pdf`\n\n"
        "Exact pairwise cosine similarity of cumulative task updates at 100, 200, and 300 matched "
        "optimizer updates. The maximum absolute cosine is 0.00744, corresponding to at most "
        "0.43 degrees of deviation from orthogonality.\n\n"
        "## `grpo_layerwise_cosine.pdf`\n\n"
        "Layer-wise cosine similarity of task-specific cumulative updates at update 300. Global "
        "near-orthogonality hides weak structured coupling, including negative Knowledge--IF "
        "coupling around the middle transformer layers.\n\n"
        "All quantitative panels use exact full-parameter BF16 realized updates from seed-42 runs. "
        "The trajectory panels are CountSketch visualizations. Cumulative checkpoint updates should "
        "not be interpreted as same-checkpoint raw-gradient interference.\n",
        encoding="utf-8",
    )
    print(f"Wrote parameter-geometry figures to {output_dir}")


if __name__ == "__main__":
    main()
