#!/usr/bin/env python3
"""Summarize single-task GRPO runs at explicitly locked checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    checkpoint_iteration: int

    @property
    def num_updates(self) -> int:
        # Megatron checkpoint iter_0000099 is the state after update 100.
        return self.checkpoint_iteration + 1


def parse_run(value: str) -> RunSpec:
    if "=" not in value or "@" not in value:
        raise argparse.ArgumentTypeError("Runs must use LABEL=RUN_DIR@CHECKPOINT_ITERATION syntax.")
    label, remainder = value.split("=", 1)
    raw_path, raw_iteration = remainder.rsplit("@", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Runs must have a non-empty label and path.")
    try:
        iteration = int(raw_iteration)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid checkpoint iteration: {raw_iteration!r}") from error
    if iteration < 0:
        raise argparse.ArgumentTypeError("Checkpoint iteration must be non-negative.")
    return RunSpec(label, Path(raw_path), iteration)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def read_metric_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"Missing metrics object in {path}:{line_number}")
            rows.append(metrics)
    return rows


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def command_value(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError as error:
        raise ValueError(f"Run manifest command is missing {option}.") from error
    if index + 1 == len(command):
        raise ValueError(f"Run manifest command has no value after {option}.")
    return command[index + 1]


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def difference(left: float | None, right: float | None) -> float | None:
    return right - left if left is not None and right is not None else None


def pass_at_k(rewards: list[float], k: int) -> float:
    correct = sum(reward > 0.0 for reward in rewards)
    sample_count = len(rewards)
    if k > sample_count:
        raise ValueError(f"Cannot compute pass@{k} from only {sample_count} samples.")
    if sample_count - correct < k:
        return 1.0
    return 1.0 - math.comb(sample_count - correct, k) / math.comb(sample_count, k)


def paired_eval_analysis(
    spec: RunSpec,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    index_path = spec.path / "eval_artifacts" / "index.jsonl"
    index_rows = read_json_rows(index_path)
    baseline_records = [row for row in index_rows if int(row["num_updates"]) == 0]
    final_records = [row for row in index_rows if int(row["num_updates"]) == spec.num_updates]
    if len(baseline_records) != 1 or len(final_records) != 1:
        raise ValueError(
            f"{spec.label}: expected one artifact index entry at updates 0 and {spec.num_updates}, "
            f"found {len(baseline_records)} and {len(final_records)}."
        )
    baseline_record = baseline_records[0]
    final_record = final_records[0]
    datasets = sorted(set(baseline_record["datasets"]) & set(final_record["datasets"]))
    if set(baseline_record["datasets"]) != set(final_record["datasets"]):
        raise ValueError(f"{spec.label}: baseline and final artifact datasets differ.")

    paired_summaries = []
    pass_summaries = []
    artifact_provenance: dict[str, Any] = {
        "locked_index_entries_sha256": metric_rows_sha256([baseline_record, final_record]),
        "datasets": {},
    }
    for dataset_index, dataset in enumerate(datasets):
        baseline_info = baseline_record["datasets"][dataset]
        final_info = final_record["datasets"][dataset]
        baseline_path = spec.path / baseline_info["run_relative_path"]
        final_path = spec.path / final_info["run_relative_path"]
        for path, info in ((baseline_path, baseline_info), (final_path, final_info)):
            actual_hash = sha256(path)
            if actual_hash != info["sha256"]:
                raise ValueError(f"Artifact hash mismatch for {path}: {actual_hash} != {info['sha256']}")

        baseline_rows = read_json_rows(baseline_path)
        final_rows = read_json_rows(final_path)
        baseline = {
            (int(row["prompt_index"]), int(row["sample_within_prompt"])): row
            for row in baseline_rows
        }
        final = {
            (int(row["prompt_index"]), int(row["sample_within_prompt"])): row
            for row in final_rows
        }
        if len(baseline) != len(baseline_rows) or len(final) != len(final_rows):
            raise ValueError(f"{spec.label}/{dataset}: duplicate sample identities.")
        if baseline.keys() != final.keys():
            raise ValueError(f"{spec.label}/{dataset}: baseline and final sample identities differ.")
        for key in baseline:
            if baseline[key]["prompt"] != final[key]["prompt"] or baseline[key]["label"] != final[key]["label"]:
                raise ValueError(f"{spec.label}/{dataset}: prompt or label changed for sample {key}.")

        prompt_baseline: dict[int, list[float]] = defaultdict(list)
        prompt_final: dict[int, list[float]] = defaultdict(list)
        improved_samples = 0
        regressed_samples = 0
        for key in sorted(baseline):
            baseline_reward = float(baseline[key]["reward"])
            final_reward = float(final[key]["reward"])
            prompt_baseline[key[0]].append(baseline_reward)
            prompt_final[key[0]].append(final_reward)
            improved_samples += baseline_reward <= 0.0 < final_reward
            regressed_samples += final_reward <= 0.0 < baseline_reward

        n_samples = int(baseline_info["n_samples_per_prompt"])
        if any(len(rewards) != n_samples for rewards in prompt_baseline.values()) or any(
            len(rewards) != n_samples for rewards in prompt_final.values()
        ):
            raise ValueError(f"{spec.label}/{dataset}: incomplete per-prompt sample groups.")
        prompt_ids = sorted(prompt_baseline)
        prompt_deltas = np.asarray(
            [np.mean(prompt_final[index]) - np.mean(prompt_baseline[index]) for index in prompt_ids]
        )
        rng = np.random.default_rng(seed + dataset_index)
        resampled = rng.integers(0, len(prompt_deltas), size=(bootstrap_samples, len(prompt_deltas)))
        bootstrap_means = prompt_deltas[resampled].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
        baseline_reward = float(np.mean([float(row["reward"]) for row in baseline_rows]))
        final_reward = float(np.mean([float(row["reward"]) for row in final_rows]))
        paired_summaries.append(
            {
                "task": spec.label,
                "dataset": dataset,
                "prompts": len(prompt_ids),
                "samples_per_prompt": n_samples,
                "samples": len(baseline_rows),
                "baseline_reward": baseline_reward,
                "final_reward": final_reward,
                "paired_reward_change": final_reward - baseline_reward,
                "paired_bootstrap_ci95_low": float(ci_low),
                "paired_bootstrap_ci95_high": float(ci_high),
                "improved_samples": improved_samples,
                "regressed_samples": regressed_samples,
                "net_improved_samples": improved_samples - regressed_samples,
                "baseline_response_len_mean": float(
                    np.mean([float(row["effective_response_length"]) for row in baseline_rows])
                ),
                "final_response_len_mean": float(
                    np.mean([float(row["effective_response_length"]) for row in final_rows])
                ),
                "baseline_truncated_fraction": float(
                    np.mean([row["status"] == "truncated" for row in baseline_rows])
                ),
                "final_truncated_fraction": float(
                    np.mean([row["status"] == "truncated" for row in final_rows])
                ),
            }
        )
        for k in (1, 2, 4, 5, 8, 10):
            if k > n_samples:
                continue
            baseline_score = float(np.mean([pass_at_k(prompt_baseline[index], k) for index in prompt_ids]))
            final_score = float(np.mean([pass_at_k(prompt_final[index], k) for index in prompt_ids]))
            pass_summaries.append(
                {
                    "task": spec.label,
                    "dataset": dataset,
                    "k": k,
                    "baseline_pass_at_k": baseline_score,
                    "final_pass_at_k": final_score,
                    "absolute_change": final_score - baseline_score,
                }
            )
        artifact_provenance["datasets"][dataset] = {
            "baseline_path": str(baseline_path),
            "baseline_sha256": baseline_info["sha256"],
            "final_path": str(final_path),
            "final_sha256": final_info["sha256"],
        }
    return paired_summaries, pass_summaries, artifact_provenance


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def smooth(values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    effective_window = min(window, len(array))
    if effective_window <= 1:
        return np.arange(len(array)), array
    averaged = np.convolve(array, np.ones(effective_window) / effective_window, mode="valid")
    return np.arange(effective_window - 1, len(array)), averaged


def plot_series(axis: Any, rows: list[dict[str, Any]], x_key: str, y_key: str, window: int, **kwargs: Any) -> None:
    points = [(float(row[x_key]), float(row[y_key])) for row in rows if row.get(y_key) is not None]
    if not points:
        return
    x = np.asarray([point[0] for point in points])
    y = [point[1] for point in points]
    axis.plot(x, y, alpha=0.13, linewidth=0.8, color=kwargs.get("color"))
    indices, averaged = smooth(y, window)
    axis.plot(x[indices], averaged, **kwargs)


def save_plots(
    output_dir: Path,
    specs: list[RunSpec],
    rollout_by_label: dict[str, list[dict[str, Any]]],
    train_by_label: dict[str, list[dict[str, Any]]],
    eval_rows: list[dict[str, Any]],
    paired_eval_rows: list[dict[str, Any]],
    window: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(specs), 4, figsize=(18, 3.7 * len(specs)), squeeze=False)
    for row_index, spec in enumerate(specs):
        rollout = rollout_by_label[spec.label]
        train = train_by_label[spec.label]

        axis = axes[row_index, 0]
        plot_series(axis, rollout, "update", "reward_mean", window, label="reward", color="tab:blue")
        plot_series(
            axis,
            rollout,
            "update",
            "informative_group_fraction",
            window,
            label="mixed groups",
            color="tab:orange",
        )
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel(spec.label)
        axis.set_title("Reward and usable GRPO groups")
        axis.legend(fontsize=8)

        axis = axes[row_index, 1]
        for key, label, color in (
            ("all_wrong_group_fraction", "all wrong", "tab:red"),
            ("informative_group_fraction", "mixed", "tab:green"),
            ("all_correct_group_fraction", "all correct", "tab:purple"),
        ):
            plot_series(axis, rollout, "update", key, window, label=label, color=color)
        axis.set_ylim(-0.02, 1.02)
        axis.set_title("Group outcome composition")
        axis.legend(fontsize=8)

        axis = axes[row_index, 2]
        plot_series(axis, train, "update", "grad_norm", window, label="grad norm", color="tab:blue")
        entropy_axis = axis.twinx()
        plot_series(
            entropy_axis,
            train,
            "update",
            "entropy",
            window,
            label="entropy",
            color="tab:orange",
        )
        axis.set_title("Optimization dynamics")
        axis.set_ylabel("grad norm")
        entropy_axis.set_ylabel("entropy")

        axis = axes[row_index, 3]
        plot_series(axis, rollout, "update", "response_len_mean", window, label="length", color="tab:blue")
        truncation_axis = axis.twinx()
        plot_series(
            truncation_axis,
            rollout,
            "update",
            "truncated_fraction",
            window,
            label="truncated",
            color="tab:red",
        )
        axis.set_title("Response length and truncation")
        axis.set_ylabel("mean tokens")
        truncation_axis.set_ylabel("fraction")

        for axis in axes[row_index]:
            axis.set_xlabel("optimizer updates")
            axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "training_dynamics.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(specs), figsize=(5.4 * len(specs), 4.2), squeeze=False)
    for column, spec in enumerate(specs):
        axis = axes[0, column]
        task_rows = [row for row in eval_rows if row["task"] == spec.label]
        for dataset in sorted({str(row["dataset"]) for row in task_rows}):
            dataset_rows = [row for row in task_rows if row["dataset"] == dataset]
            axis.plot(
                [row["update"] for row in dataset_rows],
                [row["score"] for row in dataset_rows],
                marker="o",
                label=dataset,
            )
        axis.set_title(spec.label)
        axis.set_xlabel("optimizer updates")
        axis.set_ylabel("formal evaluation score")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "formal_eval_curves.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    positions = np.arange(len(paired_eval_rows))
    changes = np.asarray([row["paired_reward_change"] for row in paired_eval_rows])
    lower = changes - np.asarray([row["paired_bootstrap_ci95_low"] for row in paired_eval_rows])
    upper = np.asarray([row["paired_bootstrap_ci95_high"] for row in paired_eval_rows]) - changes
    axis.bar(
        positions,
        changes,
        yerr=np.vstack([lower, upper]),
        capsize=4,
        color="tab:blue",
        alpha=0.85,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(
        positions,
        [f"{row['task']}\n{row['dataset']}" for row in paired_eval_rows],
    )
    axis.set_ylabel("final − baseline reward")
    axis.set_title("Paired evaluation change (95% prompt-cluster bootstrap CI)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "paired_eval_changes.png", dpi=200)
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    specs = [RunSpec(spec.label, spec.path.resolve(), spec.checkpoint_iteration) for spec in args.run]
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Run labels must be unique: {labels}")

    output_dir = args.output_dir.resolve()
    known_outputs = (
        "rollout_curves.csv",
        "train_curves.csv",
        "eval_curves.csv",
        "run_summary.csv",
        "eval_summary.csv",
        "paired_eval_summary.csv",
        "pass_at_k_summary.csv",
        "summary.json",
        "training_dynamics.png",
        "formal_eval_curves.png",
        "paired_eval_changes.png",
    )
    if output_dir.exists() and not args.force and any((output_dir / name).exists() for name in known_outputs):
        raise FileExistsError(f"Analysis outputs already exist in {output_dir}; pass --force to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    paired_eval_rows: list[dict[str, Any]] = []
    pass_at_k_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    rollout_by_label: dict[str, list[dict[str, Any]]] = {}
    train_by_label: dict[str, list[dict[str, Any]]] = {}

    for spec in specs:
        metrics_dir = spec.path / "metrics"
        manifest_path = spec.path / "provenance" / "run_manifest.json"
        input_paths = {
            "manifest": manifest_path,
            "rollout": metrics_dir / "rollout.jsonl",
            "train": metrics_dir / "train.jsonl",
            "eval": metrics_dir / "eval.jsonl",
        }
        for name, path in input_paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} input for {spec.label}: {path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        command = manifest["command"]
        n_samples = int(command_value(command, "--n-samples-per-prompt"))
        rollout_batch_size = int(command_value(command, "--rollout-batch-size"))
        learning_rate = float(command_value(command, "--lr"))

        locked_rollout = [
            metrics
            for metrics in read_metric_rows(input_paths["rollout"])
            if "rollout/reward/mean" in metrics
            and int(metrics["rollout/num_updates"]) < spec.num_updates
        ]
        locked_train = [
            metrics
            for metrics in read_metric_rows(input_paths["train"])
            if "train/grad_norm" in metrics and int(metrics["train/num_updates"]) <= spec.num_updates
        ]
        locked_eval = [
            metrics
            for metrics in read_metric_rows(input_paths["eval"])
            if int(metrics["eval/num_updates"]) <= spec.num_updates
        ]
        if len(locked_rollout) != spec.num_updates or len(locked_train) != spec.num_updates:
            raise ValueError(
                f"{spec.label}: checkpoint expects {spec.num_updates} rollout/train records, found "
                f"{len(locked_rollout)}/{len(locked_train)}."
            )

        task_rollout_rows = []
        for metrics in locked_rollout:
            reward_count = int(metrics["rollout/reward/count"])
            if reward_count % n_samples:
                raise ValueError(
                    f"{spec.label}: reward count {reward_count} is not divisible by n={n_samples}."
                )
            group_count = reward_count // n_samples
            all_wrong_count = int(metrics.get("rollout/zero_std/count_0.0", 0))
            all_correct_count = int(metrics.get("rollout/zero_std/count_1.0", 0))
            informative_count = group_count - all_wrong_count - all_correct_count
            if informative_count < 0:
                raise ValueError(f"{spec.label}: inconsistent zero-variance group counts.")
            row = {
                "task": spec.label,
                "update": int(metrics["rollout/num_updates"]) + 1,
                "reward_mean": float(metrics["rollout/reward/mean"]),
                "reward_count": reward_count,
                "group_count": group_count,
                "all_wrong_group_fraction": all_wrong_count / group_count,
                "all_correct_group_fraction": all_correct_count / group_count,
                "informative_group_fraction": informative_count / group_count,
                "response_len_mean": float(metrics["rollout/response_len/mean"]),
                "truncated_fraction": float(metrics.get("rollout/truncated_ratio", 0.0)),
                "repetition_fraction": float(metrics.get("rollout/repetition_frac", 0.0)),
                "completed_fraction": float(metrics.get("rollout/status/fraction_completed", 0.0)),
                "sandbox_timeouts": int(metrics.get("rollout/sandbox/timeouts", 0)),
                "sandbox_infrastructure_errors": int(
                    metrics.get("rollout/sandbox/infrastructure_errors", 0)
                ),
            }
            rollout_rows.append(row)
            task_rollout_rows.append(row)
        rollout_by_label[spec.label] = task_rollout_rows

        task_train_rows = []
        for metrics in locked_train:
            row = {
                "task": spec.label,
                "update": int(metrics["train/num_updates"]),
                "grad_norm": float(metrics["train/grad_norm"]),
                "entropy": float(metrics["train/entropy_loss"]),
                "policy_loss": float(metrics["train/pg_loss"]),
                "ppo_kl": float(metrics["train/ppo_kl"]),
                "clip_fraction": float(metrics["train/pg_clipfrac"]),
                "gradient_clipped": int(metrics["train/grad_clipped"]),
                "tis_absolute_deviation": float(metrics["train/tis_abs"]),
            }
            train_rows.append(row)
            task_train_rows.append(row)
        train_by_label[spec.label] = task_train_rows

        for metrics in locked_eval:
            update = int(metrics["eval/num_updates"])
            phase = str(metrics["eval/phase"])
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
                        "update": update,
                        "phase": phase,
                        "dataset": key.removeprefix("eval/"),
                        "score": float(metrics[key]),
                        "response_count": int(metrics[f"{key}/reward/count"]),
                        "truncated_fraction": float(metrics.get(f"{key}-truncated_ratio", 0.0)),
                    }
                )

        window = min(args.summary_window, spec.num_updates)
        early_rollout = task_rollout_rows[:window]
        late_rollout = task_rollout_rows[-window:]
        early_train = task_train_rows[:window]
        late_train = task_train_rows[-window:]
        early_reward = mean(early_rollout, "reward_mean")
        late_reward = mean(late_rollout, "reward_mean")
        early_informative = mean(early_rollout, "informative_group_fraction")
        late_informative = mean(late_rollout, "informative_group_fraction")
        early_grad = mean(early_train, "grad_norm")
        late_grad = mean(late_train, "grad_norm")
        run_summaries.append(
            {
                "task": spec.label,
                "checkpoint_iteration": spec.checkpoint_iteration,
                "num_updates": spec.num_updates,
                "learning_rate": learning_rate,
                "n_samples_per_prompt": n_samples,
                "rollout_batch_size": rollout_batch_size,
                "prompts_consumed": sum(int(row["group_count"]) for row in task_rollout_rows),
                "responses_consumed": sum(int(row["reward_count"]) for row in task_rollout_rows),
                "summary_window_updates": window,
                "early_reward_mean": early_reward,
                "late_reward_mean": late_reward,
                "reward_change": difference(early_reward, late_reward),
                "early_informative_group_fraction": early_informative,
                "late_informative_group_fraction": late_informative,
                "informative_group_fraction_change": difference(early_informative, late_informative),
                "late_all_wrong_group_fraction": mean(late_rollout, "all_wrong_group_fraction"),
                "late_all_correct_group_fraction": mean(late_rollout, "all_correct_group_fraction"),
                "early_grad_norm": early_grad,
                "late_grad_norm": late_grad,
                "grad_norm_change": difference(early_grad, late_grad),
                "early_entropy": mean(early_train, "entropy"),
                "late_entropy": mean(late_train, "entropy"),
                "late_response_len_mean": mean(late_rollout, "response_len_mean"),
                "late_truncated_fraction": mean(late_rollout, "truncated_fraction"),
                "total_sandbox_timeouts": sum(int(row["sandbox_timeouts"]) for row in task_rollout_rows),
                "total_sandbox_infrastructure_errors": sum(
                    int(row["sandbox_infrastructure_errors"]) for row in task_rollout_rows
                ),
            }
        )
        paired_rows, pass_rows, artifact_provenance = paired_eval_analysis(
            spec,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        paired_eval_rows.extend(paired_rows)
        pass_at_k_rows.extend(pass_rows)
        provenance.append(
            {
                "task": spec.label,
                "run_path": str(spec.path),
                "checkpoint_iteration": spec.checkpoint_iteration,
                "num_updates": spec.num_updates,
                "manifest_sha256": sha256(manifest_path),
                "locked_metric_rows_sha256": {
                    "rollout": metric_rows_sha256(locked_rollout),
                    "train": metric_rows_sha256(locked_train),
                    "eval": metric_rows_sha256(locked_eval),
                },
                "eval_artifacts": artifact_provenance,
            }
        )

    eval_summaries = []
    for spec in specs:
        task_rows = [row for row in eval_rows if row["task"] == spec.label]
        for dataset in sorted({str(row["dataset"]) for row in task_rows}):
            dataset_rows = sorted(
                (row for row in task_rows if row["dataset"] == dataset), key=lambda row: int(row["update"])
            )
            baseline = dataset_rows[0]
            final = dataset_rows[-1]
            eval_summaries.append(
                {
                    "task": spec.label,
                    "dataset": dataset,
                    "baseline_update": baseline["update"],
                    "baseline_score": baseline["score"],
                    "final_update": final["update"],
                    "final_score": final["score"],
                    "absolute_change": float(final["score"]) - float(baseline["score"]),
                    "final_response_count": final["response_count"],
                    "final_truncated_fraction": final["truncated_fraction"],
                }
            )

    write_csv(output_dir / "rollout_curves.csv", rollout_rows)
    write_csv(output_dir / "train_curves.csv", train_rows)
    write_csv(output_dir / "eval_curves.csv", eval_rows)
    write_csv(output_dir / "run_summary.csv", run_summaries)
    write_csv(output_dir / "eval_summary.csv", eval_summaries)
    write_csv(output_dir / "paired_eval_summary.csv", paired_eval_rows)
    write_csv(output_dir / "pass_at_k_summary.csv", pass_at_k_rows)
    summary = {
        "schema_version": 1,
        "checkpoint_mapping": "iter_N is analyzed after N+1 optimizer updates",
        "runs": provenance,
        "run_summary": run_summaries,
        "eval_summary": eval_summaries,
        "paired_eval_summary": paired_eval_rows,
        "pass_at_k_summary": pass_at_k_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.no_plots:
        save_plots(
            output_dir,
            specs,
            rollout_by_label,
            train_by_label,
            eval_rows,
            paired_eval_rows,
            args.smoothing_window,
        )
    print(f"Wrote locked ST training analysis to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=parse_run,
        action="append",
        required=True,
        help="Run lock as LABEL=RUN_DIR@CHECKPOINT_ITERATION; repeat for each task.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-window", type=int, default=50)
    parser.add_argument("--smoothing-window", type=int, default=21)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.summary_window <= 0 or args.smoothing_window <= 0 or args.bootstrap_samples <= 0:
        parser.error("Window sizes must be positive.")
    return args


if __name__ == "__main__":
    analyze(parse_args())
