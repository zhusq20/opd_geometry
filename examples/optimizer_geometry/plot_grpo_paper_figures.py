#!/usr/bin/env python3
"""Create paper-ready figures from the current single-task GRPO artifacts.

The script is intentionally read-only with respect to training runs.  It joins
rollout, geometry, train, and evaluation records at their recorded update IDs,
then writes vector PDF figures plus PNG previews and machine-readable summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUNS = (
    "math=outputs/qwen3_1.7b_math_code_grpo_20260817T065129Z/"
    "qwen3_1.7b_math_grpo_adamw_responsive16_trainr8192_evalr32768_seed42@300",
    "code=outputs/qwen3_1.7b_code_grpo_after_sandbox_20260817T194524Z/"
    "qwen3_1.7b_code_grpo_adamw_responsive16_trainr8192_seed42@300",
    "knowledge=outputs/qwen3_1.7b_science_grpo_adamw_lr1e-6_n4_20260817T0655Z/"
    "qwen3_1.7b_science_grpo_adamw_responsive16_trainr8192_seed42@600",
    "if=outputs/qwen3_1.7b_if_grpo_eosfix_dualeval_20260818T202612Z/"
    "qwen3_1.7b_if_grpo_adamw_responsive16_trainr8192_evalr32768_seed42@300",
)

DISPLAY_NAMES = {
    "math": "Math",
    "code": "Code",
    "knowledge": "Knowledge",
    "science": "Knowledge",
    "if": "Instruction following",
}

DATASET_NAMES = {
    "aime24": "AIME'24 avg@8",
    "math500": "MATH-500",
    "livecodebench_online": "LiveCodeBench online",
    "gpqa": "GPQA-Diamond avg@4",
    "ifbench_strict": "IFBench strict",
    "ifeval_strict_prompt": "IFEval strict",
}

TASK_COLORS = {
    "math": "#0072B2",
    "code": "#D55E00",
    "knowledge": "#009E73",
    "science": "#009E73",
    "if": "#CC79A7",
}

GROUP_COLORS = {
    "all_wrong": "#D55E00",
    "mixed": "#E69F00",
    "all_correct": "#0072B2",
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    max_updates: int


@dataclass
class RunData:
    spec: RunSpec
    group_size: int
    rollout_batch_size: int
    signal_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    status: str

    @property
    def prompts_consumed(self) -> int:
        return int(self.signal_rows[-1]["prompts_consumed"])


def parse_run(value: str) -> RunSpec:
    if "=" not in value or "@" not in value:
        raise argparse.ArgumentTypeError("Run must use LABEL=RUN_DIR@MAX_UPDATES syntax.")
    label, remainder = value.split("=", 1)
    raw_path, raw_updates = remainder.rsplit("@", 1)
    try:
        max_updates = int(raw_updates)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid update count in {value!r}") from error
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    if not label or max_updates <= 0:
        raise argparse.ArgumentTypeError(f"Invalid run specification: {value!r}")
    return RunSpec(label=label, path=path.resolve(), max_updates=max_updates)


def read_metric_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A live JSONL writer may expose a final partial line.
                continue
            metrics = record.get("metrics")
            if isinstance(metrics, dict):
                rows.append(metrics)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def command_value(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError as error:
        raise ValueError(f"Run command is missing {option}: {command[:3]}") from error
    return command[index + 1]


def run_status(path: Path) -> str:
    if (path / "run_complete.json").is_file():
        return "complete"
    if (path / "run_failed.json").is_file():
        return "failed"
    return "incomplete"


def collect_run(spec: RunSpec) -> RunData:
    manifest_path = spec.path / "provenance" / "run_manifest.json"
    metrics_dir = spec.path / "metrics"
    required = (
        manifest_path,
        metrics_dir / "rollout.jsonl",
        metrics_dir / "train.jsonl",
        metrics_dir / "eval.jsonl",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    command = manifest["command"]
    group_size = int(command_value(command, "--n-samples-per-prompt"))
    rollout_batch_size = int(command_value(command, "--rollout-batch-size"))

    rollout_rows = read_metric_rows(metrics_dir / "rollout.jsonl")
    train_rows = read_metric_rows(metrics_dir / "train.jsonl")
    eval_metric_rows = read_metric_rows(metrics_dir / "eval.jsonl")

    rewards = {
        int(row["rollout/num_updates"]): row
        for row in rollout_rows
        if "rollout/reward/mean" in row
        and int(row["rollout/num_updates"]) < spec.max_updates
    }
    geometry = {
        int(row["rollout/num_updates"]): row
        for row in rollout_rows
        if "rollout/geometry/advantage/sequence_mean/rms" in row
        and int(row["rollout/num_updates"]) < spec.max_updates
    }
    train = {
        int(row["train/num_updates"]) - 1: row
        for row in train_rows
        if "train/grad_norm" in row
        and int(row["train/num_updates"]) <= spec.max_updates
    }
    common_steps = sorted(set(rewards) & set(geometry) & set(train))
    if len(common_steps) != spec.max_updates:
        raise ValueError(
            f"{spec.label}: expected {spec.max_updates} joined updates, found {len(common_steps)}; "
            f"reward/geometry/train={len(rewards)}/{len(geometry)}/{len(train)}"
        )

    signal_rows: list[dict[str, Any]] = []
    prompts_consumed = 0
    expected_conditional_rms = math.sqrt((group_size - 1) / group_size)
    for rollout_step in common_steps:
        reward = rewards[rollout_step]
        geometry_row = geometry[rollout_step]
        train_row = train[rollout_step]
        reward_count = int(reward["rollout/reward/count"])
        if reward_count % group_size:
            raise ValueError(
                f"{spec.label} step {rollout_step}: {reward_count} responses not divisible by G={group_size}"
            )
        group_count = reward_count // group_size
        prompts_consumed += group_count
        all_wrong_count = int(reward.get("rollout/zero_std/count_0.0", 0))
        all_correct_count = int(reward.get("rollout/zero_std/count_1.0", 0))
        mixed_count = group_count - all_wrong_count - all_correct_count
        if mixed_count < 0:
            raise ValueError(f"{spec.label} step {rollout_step}: negative mixed group count")
        mixed_fraction = mixed_count / group_count
        advantage_rms = float(geometry_row["rollout/geometry/advantage/sequence_mean/rms"])
        conditional_rms = advantage_rms / math.sqrt(mixed_fraction) if mixed_fraction else math.nan
        signal_rows.append(
            {
                "task": spec.label,
                "update": rollout_step + 1,
                "prompts_consumed": prompts_consumed,
                "reward_mean": float(reward["rollout/reward/mean"]),
                "all_wrong_fraction": all_wrong_count / group_count,
                "mixed_fraction": mixed_fraction,
                "all_correct_fraction": all_correct_count / group_count,
                "advantage_rms_all": advantage_rms,
                "advantage_rms_mixed": conditional_rms,
                "advantage_rms_mixed_expected": expected_conditional_rms,
                "grad_norm": float(train_row["train/grad_norm"]),
                "grad_clipped": int(train_row["train/grad_clipped"]),
                "entropy": float(train_row["train/entropy_loss"]),
                "response_len_mean": float(reward["rollout/response_len/mean"]),
                "truncated_fraction": float(reward.get("rollout/truncated_ratio", 0.0)),
            }
        )

    prompts_by_update = {0: 0}
    prompts_by_update.update(
        {int(row["update"]): int(row["prompts_consumed"]) for row in signal_rows}
    )
    eval_rows: list[dict[str, Any]] = []
    for metrics in eval_metric_rows:
        update = int(metrics.get("eval/num_updates", -1))
        if update < 0 or update > spec.max_updates or update not in prompts_by_update:
            continue
        dataset_keys = sorted(
            key
            for key in metrics
            if key.startswith("eval/")
            and key.count("/") == 1
            and f"{key}/reward/count" in metrics
        )
        for key in dataset_keys:
            eval_rows.append(
                {
                    "task": spec.label,
                    "dataset": key.removeprefix("eval/"),
                    "update": update,
                    "prompts_consumed": prompts_by_update[update],
                    "score": float(metrics[key]),
                    "response_count": int(metrics[f"{key}/reward/count"]),
                    "response_len_mean": float(metrics[f"{key}/response_len/mean"]),
                    "truncated_fraction": float(metrics.get(f"{key}-truncated_ratio", 0.0)),
                }
            )

    return RunData(
        spec=spec,
        group_size=group_size,
        rollout_batch_size=rollout_batch_size,
        signal_rows=signal_rows,
        eval_rows=eval_rows,
        status=run_status(spec.path),
    )


def smooth_xy(rows: list[dict[str, Any]], key: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([float(row["prompts_consumed"]) / 1000.0 for row in rows])
    y = np.asarray([float(row[key]) for row in rows])
    effective_window = min(window, len(rows))
    if effective_window <= 1:
        return x, y
    valid = np.isfinite(y).astype(np.float64)
    numerator = np.convolve(np.nan_to_num(y), np.ones(effective_window), mode="valid")
    denominator = np.convolve(valid, np.ones(effective_window), mode="valid")
    averaged = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    return x[effective_window - 1 :], averaged


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure(figure: Any, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.pdf")
    figure.savefig(output_dir / f"{stem}.png", dpi=300)


def mechanism_figure(runs: list[RunData], smoothing_window: int) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        3,
        len(runs),
        figsize=(7.15, 6.35),
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0.16, "wspace": 0.18},
    )
    if len(runs) == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for column, run in enumerate(runs):
        rows = run.signal_rows
        x_raw = np.asarray([float(row["prompts_consumed"]) / 1000.0 for row in rows])
        title = DISPLAY_NAMES.get(run.spec.label, run.spec.label)
        axes[0, column].set_title(
            f"{title}\n$G$={run.group_size} · {run.prompts_consumed / 1000:.1f}k prompts",
            pad=4,
            fontweight="bold",
        )

        group_series = (
            ("all_wrong_fraction", "all wrong", GROUP_COLORS["all_wrong"]),
            ("mixed_fraction", "mixed / informative", GROUP_COLORS["mixed"]),
            ("all_correct_fraction", "all correct", GROUP_COLORS["all_correct"]),
        )
        for key, label, color in group_series:
            raw = np.asarray([float(row[key]) for row in rows])
            axes[0, column].plot(x_raw, raw, color=color, alpha=0.10, linewidth=0.45)
            x, y = smooth_xy(rows, key, smoothing_window)
            axes[0, column].plot(x, y, color=color, label=label)
        axes[0, column].set_ylim(-0.02, 1.02)
        axes[0, column].grid(axis="y", alpha=0.20, linewidth=0.5)

        for key, label, color in (
            ("advantage_rms_all", "all responses", "#CC79A7"),
            ("advantage_rms_mixed", "mixed groups", "#009E73"),
        ):
            x, y = smooth_xy(rows, key, smoothing_window)
            axes[1, column].plot(x, y, color=color, label=label)
        expected = float(rows[0]["advantage_rms_mixed_expected"])
        axes[1, column].axhline(
            expected,
            color="#333333",
            linestyle=(0, (2.2, 2.2)),
            linewidth=0.9,
            label=r"$\sqrt{(G-1)/G}$",
        )
        axes[1, column].set_ylim(-0.02, 1.05)
        axes[1, column].grid(axis="y", alpha=0.20, linewidth=0.5)

        grad_raw = np.asarray([float(row["grad_norm"]) for row in rows])
        axes[2, column].plot(x_raw, grad_raw, color="#0072B2", alpha=0.12, linewidth=0.45)
        x, grad = smooth_xy(rows, "grad_norm", smoothing_window)
        axes[2, column].plot(x, grad, color="#0072B2", label="raw grad norm")
        axes[2, column].set_yscale("log")
        axes[2, column].set_ylim(0.12, 25.0)
        axes[2, column].grid(axis="y", alpha=0.20, linewidth=0.5, which="both")
        clip_axis = axes[2, column].twinx()
        x, clipped = smooth_xy(rows, "grad_clipped", smoothing_window)
        clip_axis.plot(
            x,
            clipped,
            color="#777777",
            linestyle=(0, (3.0, 2.0)),
            linewidth=1.0,
            label="clipped fraction",
        )
        clip_axis.set_ylim(-0.02, 1.02)
        clip_axis.spines["top"].set_visible(False)
        if column != len(runs) - 1:
            clip_axis.tick_params(axis="y", right=False, labelright=False)
            clip_axis.spines["right"].set_visible(False)
        else:
            clip_axis.set_ylabel("Clipped fraction", color="#666666", labelpad=3)
            clip_axis.tick_params(axis="y", colors="#666666")

        axes[2, column].set_xlim(0, max(x_raw) * 1.01)

    axes[0, 0].set_ylabel("Group fraction")
    axes[1, 0].set_ylabel("Advantage RMS")
    axes[2, 0].set_ylabel("Pre-clip grad norm")
    axes[0, 0].legend(loc="upper right", frameon=False, handlelength=1.8)
    axes[1, 0].legend(loc="lower right", frameon=False, handlelength=1.8)
    axes[2, 0].legend(loc="upper right", frameon=False, handlelength=1.8)
    figure.supxlabel("Consumed prompts (thousands)", x=0.5, y=0.032, fontsize=8.0)
    figure.text(
        0.5,
        0.003,
        f"Seed 42; trailing {smoothing_window}-update mean. Partial single-task runs; group sizes and horizons differ.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#555555",
    )
    figure.subplots_adjust(left=0.075, right=0.935, top=0.91, bottom=0.095)
    return figure


def evaluation_figure(runs: list[RunData]) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), gridspec_kw={"hspace": 0.38, "wspace": 0.24})
    axes_flat = axes.ravel()
    dataset_palette = ("#0072B2", "#E69F00", "#009E73")
    for axis, run in zip(axes_flat, runs, strict=True):
        datasets = sorted({str(row["dataset"]) for row in run.eval_rows})
        all_scores: list[float] = []
        latest_eval_prompts = 0
        for index, dataset in enumerate(datasets):
            rows = sorted(
                (row for row in run.eval_rows if row["dataset"] == dataset),
                key=lambda row: int(row["update"]),
            )
            x = np.asarray([float(row["prompts_consumed"]) / 1000.0 for row in rows])
            y = np.asarray([100.0 * float(row["score"]) for row in rows])
            all_scores.extend(y.tolist())
            latest_eval_prompts = max(latest_eval_prompts, int(rows[-1]["prompts_consumed"]))
            color = dataset_palette[index % len(dataset_palette)]
            axis.plot(
                x,
                y,
                color=color,
                marker="o",
                markersize=3.2,
                markeredgewidth=0.0,
                label=DATASET_NAMES.get(dataset, dataset),
            )
            axis.axhline(y[0], color=color, linestyle=(0, (2.0, 2.2)), linewidth=0.75, alpha=0.55)
            axis.annotate(
                f"{y[-1]:.1f}",
                xy=(x[-1], y[-1]),
                xytext=(3, 0),
                textcoords="offset points",
                color=color,
                va="center",
                fontsize=6.6,
            )

        display = DISPLAY_NAMES.get(run.spec.label, run.spec.label)
        suffix = ""
        if latest_eval_prompts < run.prompts_consumed:
            suffix = f" · eval through {latest_eval_prompts / 1000:.1f}k"
        axis.set_title(f"{display}{suffix}", fontweight="bold")
        axis.set_xlabel("Consumed prompts (thousands)")
        axis.set_ylabel("Evaluation score (%)")
        axis.grid(axis="y", alpha=0.22, linewidth=0.5)
        axis.legend(loc="best", frameon=False)
        if all_scores:
            low = max(0.0, math.floor((min(all_scores) - 6.0) / 5.0) * 5.0)
            high = min(100.0, math.ceil((max(all_scores) + 6.0) / 5.0) * 5.0)
            if high - low < 15.0:
                middle = (high + low) / 2.0
                low = max(0.0, middle - 7.5)
                high = min(100.0, middle + 7.5)
            axis.set_ylim(low, high)
        axis.set_xlim(left=0)

    figure.text(
        0.5,
        0.008,
        "Seed 42, fixed per-benchmark decoding. Dashed lines mark the base-model score; IF update-300 evaluation is in progress.",
        ha="center",
        fontsize=6.0,
        color="#555555",
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.10)
    return figure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def endpoint_analysis(
    run: RunData,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    index_path = run.spec.path / "eval_artifacts" / "index.jsonl"
    index_rows = [
        row
        for row in read_jsonl(index_path)
        if int(row.get("num_updates", -1)) <= run.spec.max_updates
    ]
    if not index_rows:
        return []
    baseline_candidates = [row for row in index_rows if int(row.get("num_updates", -1)) == 0]
    if len(baseline_candidates) != 1:
        raise ValueError(f"{run.spec.label}: expected exactly one baseline eval artifact")
    baseline_record = baseline_candidates[0]
    latest_update = max(int(row["num_updates"]) for row in index_rows)
    latest_candidates = [row for row in index_rows if int(row["num_updates"]) == latest_update]
    if len(latest_candidates) != 1:
        raise ValueError(f"{run.spec.label}: expected one latest eval artifact at {latest_update}")
    latest_record = latest_candidates[0]

    output: list[dict[str, Any]] = []
    datasets = sorted(set(baseline_record["datasets"]) & set(latest_record["datasets"]))
    for dataset_index, dataset in enumerate(datasets):
        baseline_info = baseline_record["datasets"][dataset]
        latest_info = latest_record["datasets"][dataset]
        baseline_path = run.spec.path / baseline_info["run_relative_path"]
        latest_path = run.spec.path / latest_info["run_relative_path"]
        for path, info in ((baseline_path, baseline_info), (latest_path, latest_info)):
            actual = sha256(path)
            if actual != info["sha256"]:
                raise ValueError(f"Artifact hash mismatch for {path}: {actual} != {info['sha256']}")

        baseline_rows = read_jsonl(baseline_path)
        latest_rows = read_jsonl(latest_path)
        baseline = {
            (int(row["prompt_index"]), int(row["sample_within_prompt"])): row
            for row in baseline_rows
        }
        latest = {
            (int(row["prompt_index"]), int(row["sample_within_prompt"])): row
            for row in latest_rows
        }
        if baseline.keys() != latest.keys():
            raise ValueError(f"{run.spec.label}/{dataset}: paired sample identities differ")

        prompt_deltas: dict[int, list[float]] = {}
        for key in sorted(baseline):
            prompt_deltas.setdefault(key[0], []).append(
                float(latest[key]["reward"]) - float(baseline[key]["reward"])
            )
        prompt_ids = sorted(prompt_deltas)
        deltas = np.asarray([np.mean(prompt_deltas[prompt_id]) for prompt_id in prompt_ids])
        rng = np.random.default_rng(bootstrap_seed + 100 * len(output) + dataset_index)
        resampled = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
        bootstrap_means = deltas[resampled].mean(axis=1)
        low, high = np.quantile(bootstrap_means, [0.025, 0.975])

        baseline_score = float(np.mean([float(row["reward"]) for row in baseline_rows]))
        latest_score = float(np.mean([float(row["reward"]) for row in latest_rows]))
        baseline_truncated = float(np.mean([row["status"] == "truncated" for row in baseline_rows]))
        latest_truncated = float(np.mean([row["status"] == "truncated" for row in latest_rows]))
        output.append(
            {
                "task": run.spec.label,
                "dataset": dataset,
                "display_name": f"{DISPLAY_NAMES.get(run.spec.label, run.spec.label)} — "
                f"{DATASET_NAMES.get(dataset, dataset)}",
                "latest_update": latest_update,
                "latest_prompts": next(
                    int(row["prompts_consumed"])
                    for row in run.signal_rows
                    if int(row["update"]) == latest_update
                ),
                "prompts": len(prompt_ids),
                "samples": len(baseline_rows),
                "baseline_score": baseline_score,
                "latest_score": latest_score,
                "delta": latest_score - baseline_score,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "baseline_response_len": float(
                    np.mean([float(row["effective_response_length"]) for row in baseline_rows])
                ),
                "latest_response_len": float(
                    np.mean([float(row["effective_response_length"]) for row in latest_rows])
                ),
                "baseline_truncated_fraction": baseline_truncated,
                "latest_truncated_fraction": latest_truncated,
            }
        )
    return output


def endpoint_figure(endpoint_rows: list[dict[str, Any]]) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.55), gridspec_kw={"wspace": 0.52})
    y = np.arange(len(endpoint_rows))
    labels = [str(row["display_name"]).replace(" — ", "\n") for row in endpoint_rows]
    colors = [TASK_COLORS.get(str(row["task"]), "#0072B2") for row in endpoint_rows]

    deltas = np.asarray([100.0 * float(row["delta"]) for row in endpoint_rows])
    lows = np.asarray([100.0 * float(row["ci95_low"]) for row in endpoint_rows])
    highs = np.asarray([100.0 * float(row["ci95_high"]) for row in endpoint_rows])
    for index, color in enumerate(colors):
        axes[0].errorbar(
            deltas[index],
            y[index],
            xerr=np.asarray([[deltas[index] - lows[index]], [highs[index] - deltas[index]]]),
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=2.2,
        )
    axes[0].scatter(deltas, y, c=colors, s=25, zorder=3, edgecolors="white", linewidths=0.45)
    axes[0].axvline(0.0, color="#333333", linewidth=0.8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Score change (percentage points)")
    axes[0].set_title("Paired endpoint gain", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.22, linewidth=0.5)

    baseline_length = np.asarray([float(row["baseline_response_len"]) for row in endpoint_rows])
    latest_length = np.asarray([float(row["latest_response_len"]) for row in endpoint_rows])
    axes[1].hlines(
        y,
        np.minimum(baseline_length, latest_length),
        np.maximum(baseline_length, latest_length),
        color="#B5B5B5",
        linewidth=1.5,
        zorder=1,
    )
    axes[1].scatter(
        baseline_length,
        y,
        facecolors="white",
        edgecolors="#777777",
        linewidths=1.0,
        s=24,
        zorder=2,
    )
    axes[1].scatter(
        latest_length,
        y,
        c=colors,
        edgecolors="white",
        linewidths=0.45,
        s=29,
        zorder=3,
    )
    axes[1].set_xscale("log")
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean response tokens (log scale)")
    axes[1].set_title("Response-length shift", fontweight="bold")
    axes[1].text(
        0.02,
        0.985,
        "open = base · filled = trained",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#555555",
    )
    axes[1].grid(axis="x", alpha=0.22, linewidth=0.5, which="both")
    axes[1].set_xlim(
        min(baseline_length.min(), latest_length.min()) * 0.78,
        max(baseline_length.max(), latest_length.max()) * 1.75,
    )
    axes[1].set_xticks([1000, 2000, 4000, 8000], ["1k", "2k", "4k", "8k"])
    for index, row in enumerate(endpoint_rows):
        axes[1].text(
            max(baseline_length[index], latest_length[index]) * 1.05,
            index,
            f"trunc {100 * float(row['baseline_truncated_fraction']):.1f}→"
            f"{100 * float(row['latest_truncated_fraction']):.1f}%",
            va="center",
            fontsize=5.8,
            color="#555555",
        )

    figure.text(
        0.5,
        0.008,
        "95% paired prompt-cluster bootstrap CI; uncertainty is over benchmark prompts, not training seeds.",
        ha="center",
        fontsize=6.0,
        color="#555555",
    )
    figure.subplots_adjust(left=0.24, right=0.985, top=0.90, bottom=0.16)
    return figure


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else math.nan


def signal_summary(runs: list[RunData], window: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run in runs:
        effective_window = min(window, len(run.signal_rows))
        early = run.signal_rows[:effective_window]
        late = run.signal_rows[-effective_window:]
        finite_conditional = [
            float(row["advantage_rms_mixed"])
            for row in run.signal_rows
            if math.isfinite(float(row["advantage_rms_mixed"]))
        ]
        expected = float(run.signal_rows[0]["advantage_rms_mixed_expected"])
        summaries.append(
            {
                "task": run.spec.label,
                "status": run.status,
                "group_size": run.group_size,
                "updates": len(run.signal_rows),
                "prompts_consumed": run.prompts_consumed,
                "early_reward": mean(early, "reward_mean"),
                "late_reward": mean(late, "reward_mean"),
                "early_all_wrong_fraction": mean(early, "all_wrong_fraction"),
                "late_all_wrong_fraction": mean(late, "all_wrong_fraction"),
                "early_mixed_fraction": mean(early, "mixed_fraction"),
                "late_mixed_fraction": mean(late, "mixed_fraction"),
                "early_all_correct_fraction": mean(early, "all_correct_fraction"),
                "late_all_correct_fraction": mean(late, "all_correct_fraction"),
                "conditional_advantage_rms": float(np.mean(finite_conditional)),
                "expected_conditional_advantage_rms": expected,
                "max_conditional_rms_abs_error": max(
                    abs(value - expected) for value in finite_conditional
                ),
                "early_grad_norm": mean(early, "grad_norm"),
                "late_grad_norm": mean(late, "grad_norm"),
                "early_clipped_fraction": mean(early, "grad_clipped"),
                "late_clipped_fraction": mean(late, "grad_clipped"),
                "late_response_len_mean": mean(late, "response_len_mean"),
                "late_training_truncated_fraction": mean(late, "truncated_fraction"),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        help="LABEL=RUN_DIR@MAX_UPDATES; repeat in the desired panel order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/raw_gradient_interference/paper/figures/grpo_current",
    )
    parser.add_argument("--smoothing-window", type=int, default=21)
    parser.add_argument("--summary-window", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.run:
        args.run = [parse_run(value) for value in DEFAULT_RUNS]
    if args.smoothing_window <= 0 or args.summary_window <= 0 or args.bootstrap_samples <= 0:
        parser.error("Windows and bootstrap sample count must be positive.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    expected_outputs = (
        "grpo_signal_dynamics.pdf",
        "grpo_evaluation_curves.pdf",
        "grpo_endpoint_deltas.pdf",
        "grpo_current_analysis.pdf",
    )
    if output_dir.exists() and not args.force and any((output_dir / name).exists() for name in expected_outputs):
        raise FileExistsError(f"Figure outputs already exist in {output_dir}; pass --force to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [collect_run(spec) for spec in args.run]
    if len(runs) != 4:
        raise ValueError(f"The paper layout expects four runs, received {len(runs)}")
    endpoint_rows = [
        row
        for run in runs
        for row in endpoint_analysis(run, args.bootstrap_samples, args.bootstrap_seed)
    ]
    summaries = signal_summary(runs, args.summary_window)

    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    mechanism = mechanism_figure(runs, args.smoothing_window)
    evaluation = evaluation_figure(runs)
    endpoints = endpoint_figure(endpoint_rows)
    save_figure(mechanism, output_dir, "grpo_signal_dynamics")
    save_figure(evaluation, output_dir, "grpo_evaluation_curves")
    save_figure(endpoints, output_dir, "grpo_endpoint_deltas")
    with PdfPages(
        output_dir / "grpo_current_analysis.pdf",
        metadata={
            "Title": "Current four-task GRPO analysis",
            "Author": "Generated from local experiment artifacts",
            "Subject": "Partial single-task GRPO mechanism and evaluation curves",
        },
    ) as pdf:
        pdf.savefig(mechanism)
        pdf.savefig(evaluation)
        pdf.savefig(endpoints)
    plt.close(mechanism)
    plt.close(evaluation)
    plt.close(endpoints)

    write_csv(output_dir / "grpo_signal_summary.csv", summaries)
    write_csv(output_dir / "grpo_endpoint_summary.csv", endpoint_rows)
    input_files = {
        run.spec.label: {
            "run_dir": str(run.spec.path),
            "max_updates": run.spec.max_updates,
            "status": run.status,
            "group_size": run.group_size,
            "prompts_consumed": run.prompts_consumed,
            "sha256": {
                name: sha256(run.spec.path / "metrics" / f"{name}.jsonl")
                for name in ("rollout", "train", "eval")
            },
        }
        for run in runs
    }
    figure_files = sorted(output_dir.glob("grpo_*.pdf")) + sorted(output_dir.glob("grpo_*.png"))
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoothing_window_updates": args.smoothing_window,
        "summary_window_updates": args.summary_window,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "inputs": input_files,
        "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in figure_files},
        "limitations": [
            "All runs are seed 42 and currently lack completion markers.",
            "Group sizes and consumed-prompt horizons differ across tasks.",
            "IF update-300 evaluation was incomplete when the figures were generated.",
            "Paired confidence intervals describe benchmark-prompt uncertainty, not training-seed variance.",
        ],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "CAPTIONS.md").write_text(
        "# Suggested captions\n\n"
        "**GRPO signal dynamics.** Fractions of all-wrong, mixed-reward, and all-correct "
        "rollout groups (top), advantage RMS across all responses and conditional on mixed "
        "groups (middle), and pre-clipping actor gradient norm with the clipped-update fraction "
        "(bottom). Curves are trailing 21-update means. Conditional advantage RMS remains at "
        r"$\sqrt{(G-1)/G}$ because the implementation uses sample-standard-deviation "
        "normalization, while the fraction of informative groups changes with training. These "
        "are partial seed-42 single-task runs with unequal group sizes and horizons.\n\n"
        "**Held-out evaluation curves.** Per-task held-out scores along the same partial GRPO "
        "runs. Dashed horizontal lines denote the base-model score. IF evaluation is complete "
        "through 4,000 consumed prompts in this snapshot.\n\n"
        "**Endpoint gains and response drift.** Paired score changes from the base model to the "
        "latest completed evaluation artifact, with 95% prompt-cluster bootstrap confidence "
        "intervals (left), and the associated mean response-length and truncation shifts "
        "(right). The intervals do not estimate training-seed variability.\n",
        encoding="utf-8",
    )
    print(f"Wrote paper figures to {output_dir}")


if __name__ == "__main__":
    main()
