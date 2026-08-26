#!/usr/bin/env python3
"""Collate exact four-task raw-gradient interference across sequential boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ANCHORS = ("pre_code", "00_code_origin", "01_after_math", "02_after_knowledge", "03_after_if")
TASKS = ("code", "math", "knowledge", "if")
MANIFEST_TASK = {"code": "code", "math": "math", "knowledge": "science", "if": "if"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def read_rollout_metrics(path: Path) -> dict[str, Any]:
    """Return the unique rollout summary record used to form a probe batch."""
    matches = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record.get("metrics")
            if isinstance(metrics, dict) and "rollout/reward/mean" in metrics:
                matches.append((line_number, metrics))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one rollout reward summary in {path}, found {len(matches)}.")
    return matches[0][1]


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


def save_plot(output_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairwise_cosines = [
        abs(float(value))
        for anchor in ANCHORS
        for left_index, left in enumerate(TASKS)
        for right in TASKS[left_index + 1 :]
        if (value := summaries[anchor]["global"][f"cosine/{left}__{right}"]) is not None
    ]
    color_limit = max(0.01, 1.05 * max(pairwise_cosines, default=0.0))
    color_map = plt.get_cmap("coolwarm").copy()
    color_map.set_bad("#eeeeee")

    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), layout="constrained")
    for axis_index, (axis, anchor) in enumerate(zip(axes.flat, ANCHORS, strict=False)):
        global_row = summaries[anchor]["global"]
        matrix = np.full((len(TASKS), len(TASKS)), np.nan, dtype=np.float64)
        for left_index, left in enumerate(TASKS):
            for right_index in range(left_index + 1, len(TASKS)):
                right = TASKS[right_index]
                value = global_row[f"cosine/{left}__{right}"]
                if value is not None:
                    matrix[left_index, right_index] = matrix[right_index, left_index] = float(value)
        image = axis.imshow(matrix, vmin=-color_limit, vmax=color_limit, cmap=color_map)
        axis.set_xticks(np.arange(len(TASKS)), TASKS, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(TASKS)), TASKS)
        if axis_index % 3:
            axis.tick_params(axis="y", labelleft=False)
        for row_index in range(len(TASKS)):
            for column_index in range(len(TASKS)):
                if row_index == column_index:
                    label = "—"
                elif np.isfinite(matrix[row_index, column_index]):
                    label = f"{matrix[row_index, column_index]:+.3f}"
                else:
                    label = "n/a"
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        axis.set_title(anchor)
    for axis in axes.flat[len(ANCHORS) :]:
        axis.set_visible(False)
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="raw-gradient cosine",
        shrink=0.82,
        pad=0.035,
    )
    figure.suptitle("Same-checkpoint task-gradient interference")
    figure.savefig(output_dir / "raw_gradient_cosines_by_boundary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    probe_root = args.probe_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --force.")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    provenance = []
    expected_probe_design = None
    norm_rows = []
    pair_rows = []
    layer_rows = []
    quality_rows = []
    fixed_probe_inputs = {}
    for anchor_index, anchor in enumerate(ANCHORS):
        path = probe_root / anchor / "analysis" / "summary.json"
        summary = read_json(path)
        tasks = set(summary["tasks"])
        if tasks != set(TASKS):
            raise ValueError(f"{anchor}: expected tasks {TASKS}, found {sorted(tasks)}.")
        if bool(summary["optimizer_step_executed"]):
            raise ValueError(f"{anchor}: raw-gradient probe executed an optimizer step.")
        design = {
            "probe_updates": int(summary["probe_updates"]),
            "actual_batch_size_per_update": int(summary["actual_batch_size_per_update"]),
            "probe_prompt_count": int(summary["probe_prompt_count"]),
        }
        if expected_probe_design is None:
            expected_probe_design = design
        elif design != expected_probe_design:
            raise ValueError(f"Probe design changed at {anchor}: {design} != {expected_probe_design}")
        summaries[anchor] = summary
        global_row = summary["global"]
        task_artifacts = {}
        for task in TASKS:
            manifest_path = probe_root / anchor / "raw_gradients" / task / "manifest.json"
            manifest = read_json(manifest_path)
            if manifest["task"] != MANIFEST_TASK[task]:
                raise ValueError(
                    f"{manifest_path}: task is {manifest['task']!r}, " f"expected {MANIFEST_TASK[task]!r}."
                )
            if bool(manifest["optimizer_step_executed"]):
                raise ValueError(f"{manifest_path}: raw-gradient probe executed an optimizer step.")
            if int(manifest["expected_updates"]) != design["probe_updates"]:
                raise ValueError(f"{manifest_path}: expected_updates does not match probe design.")
            if int(manifest["actual_batch_size_per_update"]) != design["actual_batch_size_per_update"]:
                raise ValueError(f"{manifest_path}: actual batch size does not match probe design.")
            if int(manifest["probe_prompt_count"]) != design["probe_prompt_count"]:
                raise ValueError(f"{manifest_path}: prompt count does not match probe design.")

            run_dirs = sorted((probe_root / anchor / "runs").glob(f"probe_{anchor}_{task}_seed*"))
            if len(run_dirs) != 1:
                raise ValueError(f"{anchor}/{task}: expected one completed run directory, found {len(run_dirs)}.")
            run_dir = run_dirs[0]
            completion_path = run_dir / "run_complete.json"
            completion = read_json(completion_path)
            if completion.get("status") != "complete":
                raise ValueError(f"{completion_path}: run status is not complete.")
            if int(completion["final_num_updates"]) != design["probe_updates"]:
                raise ValueError(f"{completion_path}: final updates do not match probe design.")
            if int(completion["num_rollout"]) != design["probe_updates"]:
                raise ValueError(f"{completion_path}: rollout count does not match probe design.")

            metrics_path = run_dir / "metrics" / "rollout.jsonl"
            metrics = read_rollout_metrics(metrics_path)
            input_configs = sorted((run_dir / "provenance" / "inputs").glob("00_*_gradient_probe.yaml"))
            if len(input_configs) != 1:
                raise ValueError(f"{run_dir}: expected one archived probe input config, found {len(input_configs)}.")
            input_config_path = input_configs[0]
            input_config = yaml.safe_load(input_config_path.read_text(encoding="utf-8"))
            sources = input_config.get("sources") if isinstance(input_config, dict) else None
            if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
                raise ValueError(f"{input_config_path}: expected exactly one probe source.")
            prompt_data_path = Path(str(sources[0]["path"])).resolve()
            if not prompt_data_path.is_file():
                raise FileNotFoundError(f"Probe prompt data is missing: {prompt_data_path}")
            input_fingerprint = {
                "archived_config_sha256": sha256(input_config_path),
                "prompt_data_path": str(prompt_data_path),
                "prompt_data_sha256": sha256(prompt_data_path),
            }
            if task not in fixed_probe_inputs:
                fixed_probe_inputs[task] = input_fingerprint
            elif input_fingerprint != fixed_probe_inputs[task]:
                raise ValueError(f"Fixed probe input changed for {task} at {anchor}.")
            task_artifacts[task] = {
                "manifest": manifest_path,
                "metrics": metrics_path,
                "completion": completion_path,
                "input_config": input_config_path,
                "prompt_data": prompt_data_path,
            }
            response_count = int(metrics["rollout/reward/count"])
            if response_count != design["actual_batch_size_per_update"]:
                raise ValueError(
                    f"{metrics_path}: reward count {response_count} does not match probe batch "
                    f"{design['actual_batch_size_per_update']}."
                )
            all_wrong_groups = int(metrics.get("rollout/zero_std/count_0.0", 0))
            all_correct_groups = int(metrics.get("rollout/zero_std/count_1.0", 0))
            zero_std_groups = all_wrong_groups + all_correct_groups
            prompt_count = design["probe_prompt_count"]
            if zero_std_groups > prompt_count:
                raise ValueError(
                    f"{metrics_path}: all-wrong plus all-correct groups exceed the {prompt_count} probe prompts."
                )
            effective_tokens = int(manifest["effective_token_count_total"])
            sandbox_infrastructure_errors = metrics.get("rollout/sandbox/infrastructure_errors")
            if task == "code":
                if sandbox_infrastructure_errors is None:
                    raise ValueError(f"{metrics_path}: Code probe is missing SandboxFusion diagnostics.")
                if int(sandbox_infrastructure_errors) != 0:
                    raise ValueError(
                        f"{metrics_path}: Code probe has {sandbox_infrastructure_errors} SandboxFusion "
                        "infrastructure errors; refusing to treat them as model failures."
                    )
            quality_rows.append(
                {
                    "anchor": anchor,
                    "anchor_index": anchor_index,
                    "task": task,
                    "reward_mean": float(metrics["rollout/reward/mean"]),
                    "all_wrong_group_count": all_wrong_groups,
                    "all_correct_group_count": all_correct_groups,
                    "zero_std_group_count": zero_std_groups,
                    "informative_group_count": prompt_count - zero_std_groups,
                    "informative_group_fraction": (prompt_count - zero_std_groups) / prompt_count,
                    "response_length_mean": float(metrics["rollout/response_len/mean"]),
                    "response_length_median": float(metrics["rollout/response_len/median"]),
                    "response_length_p95": float(metrics["rollout/response_len/p95"]),
                    "response_length_max": int(metrics["rollout/response_len/max"]),
                    "truncated_fraction": float(metrics["rollout/truncated_ratio"]),
                    "effective_token_count_total": effective_tokens,
                    "effective_tokens_per_response": effective_tokens / response_count,
                    "sandbox_infrastructure_errors": sandbox_infrastructure_errors,
                }
            )
            norm_rows.append(
                {
                    "anchor": anchor,
                    "anchor_index": anchor_index,
                    "checkpoint": summary["checkpoint"],
                    "checkpoint_step": summary["checkpoint_step"],
                    "task": task,
                    "gradient_norm": float(global_row[f"gradient_norm/{task}"]),
                }
            )
        for left_index, left in enumerate(TASKS):
            for right in TASKS[left_index + 1 :]:
                dot = float(global_row[f"dot/{left}__{right}"])
                left_square = float(global_row[f"gradient_norm/{left}"]) ** 2
                right_square = float(global_row[f"gradient_norm/{right}"]) ** 2
                cosine = global_row[f"cosine/{left}__{right}"]
                direction_defined = cosine is not None
                pair_rows.append(
                    {
                        "anchor": anchor,
                        "anchor_index": anchor_index,
                        "checkpoint": summary["checkpoint"],
                        "checkpoint_step": summary["checkpoint_step"],
                        "left_task": left,
                        "right_task": right,
                        "dot": dot,
                        "cosine": float(cosine) if direction_defined else None,
                        "cosine_defined": direction_defined,
                        "left_self_interference_ratio_from_right": dot / left_square if left_square else None,
                        "right_self_interference_ratio_from_left": dot / right_square if right_square else None,
                        "conflicting_direction": dot < 0.0 if direction_defined else None,
                    }
                )
        for row in summary["layerwise"]:
            layer_rows.append({"anchor": anchor, **row})
        provenance.append(
            {
                "anchor": anchor,
                "summary_path": str(path),
                "summary_sha256": sha256(path),
                "checkpoint": summary["checkpoint"],
                "checkpoint_step": summary["checkpoint_step"],
                "task_manifest_sha256": {task: sha256(task_artifacts[task]["manifest"]) for task in TASKS},
                "task_rollout_metrics_sha256": {task: sha256(task_artifacts[task]["metrics"]) for task in TASKS},
                "task_run_complete_sha256": {task: sha256(task_artifacts[task]["completion"]) for task in TASKS},
                "task_archived_input_config_sha256": {
                    task: sha256(task_artifacts[task]["input_config"]) for task in TASKS
                },
                "task_prompt_data_sha256": {task: sha256(task_artifacts[task]["prompt_data"]) for task in TASKS},
            }
        )

    pair_changes = []
    for left_index, left in enumerate(TASKS):
        for right in TASKS[left_index + 1 :]:
            relevant = [row for row in pair_rows if row["left_task"] == left and row["right_task"] == right]
            for before, after in zip(relevant, relevant[1:], strict=False):
                direction_defined_at_both = bool(before["cosine_defined"] and after["cosine_defined"])
                pair_changes.append(
                    {
                        "left_task": left,
                        "right_task": right,
                        "before_anchor": before["anchor"],
                        "after_anchor": after["anchor"],
                        "before_cosine": before["cosine"],
                        "after_cosine": after["cosine"],
                        "cosine_change": (
                            float(after["cosine"]) - float(before["cosine"]) if direction_defined_at_both else None
                        ),
                        "direction_defined_at_both": direction_defined_at_both,
                        "conflict_status_changed": (
                            before["conflicting_direction"] != after["conflicting_direction"]
                            if direction_defined_at_both
                            else None
                        ),
                    }
                )

    if expected_probe_design is None:
        raise ValueError(f"No probes found under {probe_root}.")
    write_csv(output_dir / "raw_gradient_norms.csv", norm_rows)
    write_csv(output_dir / "raw_gradient_interference.csv", pair_rows)
    write_csv(output_dir / "raw_gradient_interference_changes.csv", pair_changes)
    write_csv(output_dir / "raw_gradient_layerwise.csv", layer_rows)
    write_csv(output_dir / "raw_gradient_probe_quality.csv", quality_rows)
    summary = {
        "schema_version": 3,
        "method": (
            "Exact FP64 reductions over matching optimizer-owned FP32 raw-gradient shards at each fixed checkpoint."
        ),
        "probe_design": expected_probe_design,
        "anchors": list(ANCHORS),
        "fixed_probe_inputs": fixed_probe_inputs,
        "ascent_gradient_note": (
            "Artifacts store loss gradients; jointly negating all task vectors preserves pairwise dots and cosines."
        ),
        "cosine_definition_note": (
            "Cosine and directional-conflict fields are null when either gradient norm is zero. A zero vector has "
            "no direction and is not labeled orthogonal."
        ),
        "interpretation": (
            "Negative dot products are the local first-order conflict term. They must not be treated as a causal "
            "estimate of final test forgetting."
        ),
        "provenance": provenance,
        "gradient_norms": norm_rows,
        "probe_quality": quality_rows,
        "pairwise_interference": pair_rows,
        "pairwise_changes": pair_changes,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plots:
        save_plot(output_dir, summaries)
    print(f"Wrote sequential raw-gradient interference summary to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
