#!/usr/bin/env python3
"""Analyze behavior, forgetting, and training dynamics for one sequential GRPO run."""

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

BOUNDARIES = (
    ("pre_code", "pre_code", None),
    ("code_origin", "00_code_origin", "code"),
    ("after_math", "01_after_math", "math"),
    ("after_knowledge", "02_after_knowledge", "knowledge"),
    ("after_if", "03_after_if", "if"),
)
DOMAINS = ("code", "math", "knowledge", "if")
PRIMARY_DATASET = {
    "code": "livecodebench_online",
    "math": "math500",
    "knowledge": "gpqa",
    "if": "ifbench_strict",
}
DATASET_DOMAIN_FALLBACK = {
    "livecodebench_online": "code",
    "aime24": "math",
    "math500": "math",
    "gpqa": "knowledge",
    "ifeval_strict_prompt": "if",
    "ifbench_strict": "if",
}


@dataclass(frozen=True)
class Stage:
    number: int
    task: str
    run_dir: Path
    input_checkpoint: Path
    output_checkpoint: Path
    updates: int


@dataclass(frozen=True)
class Boundary:
    index: int
    label: str
    trained_task: str | None
    eval_dir: Path
    checkpoint: Path


@dataclass
class EvalDataset:
    name: str
    domain: str
    samples_per_prompt: int
    rows: list[dict[str, Any]]
    prompt_rewards: dict[int, list[float]]
    artifact_path: Path
    artifact_sha256: str


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            rows.append(value)
    return rows


def metric_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, record in enumerate(read_jsonl(path), start=1):
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Missing metrics object at {path}:{line_number}.")
        rows.append(metrics)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command_value(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError as error:
        raise ValueError(f"Run command is missing {option}.") from error
    if index + 1 >= len(command):
        raise ValueError(f"Run command has no value after {option}.")
    return str(command[index + 1])


def optional_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def optional_change(before: float | None, after: float | None) -> float | None:
    return after - before if before is not None and after is not None else None


def resolve_torch_dist_checkpoint(path: Path) -> Path:
    """Resolve either a torch-dist checkpoint directory or its save root."""
    path = path.resolve()
    if (path / ".metadata").is_file() and (path / "common.pt").is_file():
        return path

    marker = path / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError(f"Not a torch-dist checkpoint or save root: {path}")
    value = marker.read_text(encoding="utf-8").strip()
    if value == "release":
        checkpoint = path / "release"
    elif value.isdigit():
        checkpoint = path / f"iter_{int(value):07d}"
    elif value.startswith("iter_") and value.removeprefix("iter_").isdigit():
        checkpoint = path / value
    else:
        raise ValueError(f"Unsupported checkpoint marker {value!r} in {marker}.")
    if not (checkpoint / ".metadata").is_file() or not (checkpoint / "common.pt").is_file():
        raise FileNotFoundError(f"Resolved torch-dist checkpoint is incomplete: {checkpoint}")
    return checkpoint.resolve()


def load_sequence(sequence_root: Path, pre_code_checkpoint: Path) -> tuple[dict[str, Any], list[Stage]]:
    manifest_path = sequence_root / "sequence_manifest.json"
    boundaries_path = sequence_root / "stage_boundaries.jsonl"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"Sequential manifest is not complete: {manifest_path}")
    if tuple(manifest.get("sequence", [])) != ("code_origin", "math", "knowledge", "if"):
        raise ValueError(f"Unexpected task order in {manifest_path}: {manifest.get('sequence')}")

    boundary_rows = sorted(read_jsonl(boundaries_path), key=lambda row: int(row["stage_number"]))
    if [str(row["task"]) for row in boundary_rows] != ["math", "knowledge", "if"]:
        raise ValueError(f"Unexpected stage records in {boundaries_path}.")
    pre_code_checkpoint = resolve_torch_dist_checkpoint(pre_code_checkpoint)
    origin = Path(str(manifest["code_origin"])).resolve()
    code_training = manifest["code_origin_training"]
    code_manifest_path = Path(str(code_training["provenance_manifest"])).resolve()
    code_run_dir = code_manifest_path.parent.parent
    code_manifest = read_json(code_manifest_path)
    if sha256(code_manifest_path) != str(code_training["provenance_manifest_sha256"]):
        raise ValueError(f"Code provenance manifest fingerprint changed: {code_manifest_path}")
    if code_manifest.get("status") != code_training.get("source_run_status"):
        raise ValueError("Code run status differs from the status locked in the sequence manifest.")
    completion_present = (code_run_dir / "run_complete.json").is_file()
    if completion_present != bool(code_training.get("source_run_complete")):
        raise ValueError("Code completion-marker state differs from the state locked in the sequence manifest.")
    code_command = [str(item) for item in code_manifest["command"]]
    recorded_code_input = resolve_torch_dist_checkpoint(Path(command_value(code_command, "--load")))
    recorded_code_save = Path(command_value(code_command, "--save")).resolve()
    if recorded_code_input != pre_code_checkpoint:
        raise ValueError(
            f"Code training loaded {recorded_code_input}, but --pre-code-checkpoint resolves to {pre_code_checkpoint}."
        )
    if origin.parent != recorded_code_save:
        raise ValueError(f"Code-origin checkpoint {origin} is not under the recorded save root {recorded_code_save}.")
    code_iteration = origin.name.removeprefix("iter_")
    if not code_iteration.isdigit():
        raise ValueError(f"Invalid Code-origin checkpoint directory name: {origin}")
    code_updates = int(code_iteration) + 1
    stages = [Stage(0, "code", code_run_dir, pre_code_checkpoint, origin, code_updates)]

    previous_checkpoint = origin
    for expected_number, record in enumerate(boundary_rows, start=1):
        number = int(record["stage_number"])
        task = str(record["task"])
        run_dir = Path(str(record["run_dir"])).resolve()
        input_checkpoint = Path(str(record["input_checkpoint"])).resolve()
        output_checkpoint = Path(str(record["output_checkpoint"])).resolve()
        if number != expected_number or input_checkpoint != previous_checkpoint:
            raise ValueError(f"Broken sequential lineage at stage {number} ({task}).")
        if output_checkpoint != Path(str(manifest["stage_endpoints"][task])).resolve():
            raise ValueError(f"Stage endpoint mismatch for {task}.")
        completion = read_json(run_dir / "run_complete.json")
        if completion.get("status") != "complete":
            raise ValueError(f"Stage run is not complete: {run_dir}")
        iteration_text = output_checkpoint.name.removeprefix("iter_")
        if not iteration_text.isdigit():
            raise ValueError(f"Invalid checkpoint directory name: {output_checkpoint}")
        updates = int(iteration_text) + 1
        if int(completion["final_num_updates"]) != updates:
            raise ValueError(f"Completion marker/checkpoint update mismatch for {task}.")
        stages.append(Stage(number, task, run_dir, input_checkpoint, output_checkpoint, updates))
        previous_checkpoint = output_checkpoint

    for path, expected in (
        (origin / ".metadata", manifest["code_origin_metadata_sha256"]),
        (origin / "common.pt", manifest["code_origin_common_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Code-origin checkpoint fingerprint changed: {path}")
    return manifest, stages


def parse_training(stages: list[Stage], summary_window: int) -> tuple[dict[str, Any], ...]:
    rollout_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    sequence_offset = 0

    for stage in stages:
        manifest_path = stage.run_dir / "provenance" / "run_manifest.json"
        run_manifest = read_json(manifest_path)
        command = [str(item) for item in run_manifest["command"]]
        n_samples = int(command_value(command, "--n-samples-per-prompt"))
        rollout_batch_size = int(command_value(command, "--rollout-batch-size"))
        learning_rate = float(command_value(command, "--lr"))

        raw_rollout = [
            row
            for row in metric_rows(stage.run_dir / "metrics" / "rollout.jsonl")
            if row.get("rollout/reward/mean") is not None and 0 <= int(row["rollout/num_updates"]) < stage.updates
        ]
        raw_train = [
            row
            for row in metric_rows(stage.run_dir / "metrics" / "train.jsonl")
            if row.get("train/grad_norm") is not None and 1 <= int(row["train/num_updates"]) <= stage.updates
        ]
        raw_geometry = [
            row
            for row in metric_rows(stage.run_dir / "metrics" / "geometry.jsonl")
            if row.get("geometry/global/parameter_count") is not None
            and 1 <= int(row["geometry/num_updates"]) <= stage.updates
        ]
        for name, rows in (("rollout", raw_rollout), ("train", raw_train), ("geometry", raw_geometry)):
            if len(rows) != stage.updates:
                raise ValueError(f"{stage.task}: expected {stage.updates} {name} records, found {len(rows)}.")

        stage_rollout = []
        for metrics in raw_rollout:
            reward_count = int(metrics["rollout/reward/count"])
            if reward_count % n_samples:
                raise ValueError(f"{stage.task}: reward count {reward_count} is not divisible by n={n_samples}.")
            group_count = reward_count // n_samples
            all_wrong = int(metrics.get("rollout/zero_std/count_0.0", 0))
            all_correct = int(metrics.get("rollout/zero_std/count_1.0", 0))
            informative = group_count - all_wrong - all_correct
            if informative < 0:
                raise ValueError(f"{stage.task}: invalid GRPO group composition.")
            local_update = int(metrics["rollout/num_updates"]) + 1
            row = {
                "stage": stage.task,
                "stage_number": stage.number,
                "local_update": local_update,
                "sequence_update": sequence_offset + local_update,
                "reward_mean": float(metrics["rollout/reward/mean"]),
                "reward_count": reward_count,
                "group_count": group_count,
                "all_wrong_group_fraction": all_wrong / group_count,
                "all_correct_group_fraction": all_correct / group_count,
                "informative_group_fraction": informative / group_count,
                "response_len_mean": float(metrics["rollout/response_len/mean"]),
                "truncated_fraction": float(metrics.get("rollout/truncated_ratio", 0.0)),
                "completed_fraction": float(metrics.get("rollout/status/fraction_completed", 0.0)),
                "repetition_fraction": float(metrics.get("rollout/repetition_frac", 0.0)),
            }
            rollout_rows.append(row)
            stage_rollout.append(row)

        stage_train = []
        for metrics in raw_train:
            local_update = int(metrics["train/num_updates"])
            row = {
                "stage": stage.task,
                "stage_number": stage.number,
                "local_update": local_update,
                "sequence_update": sequence_offset + local_update,
                "grad_norm": float(metrics["train/grad_norm"]),
                "grad_clip_threshold": float(metrics["train/grad_clip_threshold"]),
                "gradient_clipped": int(metrics["train/grad_clipped"]),
                "entropy": float(metrics["train/entropy_loss"]),
                "policy_loss": float(metrics["train/pg_loss"]),
                "ppo_kl": float(metrics["train/ppo_kl"]),
                "policy_clip_fraction": float(metrics["train/pg_clipfrac"]),
                "tis_absolute_deviation": float(metrics["train/tis_abs"]),
            }
            train_rows.append(row)
            stage_train.append(row)

        stage_geometry = []
        for metrics in raw_geometry:
            local_update = int(metrics["geometry/num_updates"])
            row = {
                "stage": stage.task,
                "stage_number": stage.number,
                "local_update": local_update,
                "sequence_update": sequence_offset + local_update,
                "model_change_fraction": float(metrics["geometry/global/model_change_fraction"]),
                "realized_update_l2": float(metrics["geometry/global/delta_model_l2"]),
                "intended_update_l2": float(metrics["geometry/global/delta_intended_fp32_l2"]),
                "intended_below_half_ulp_fraction": float(metrics["geometry/global/intended_below_half_ulp_fraction"]),
                "intended_energy_zeroed_fraction": float(metrics["geometry/global/intended_energy_zeroed_fraction"]),
                "energy_survival": float(metrics["geometry/global/energy_survival"]),
                "quantization_residual": float(metrics["geometry/global/quantization_residual"]),
                "cos_raw_optimizer_gradient": (
                    float(metrics["geometry/global/cos_g_raw_g_opt"])
                    if metrics.get("geometry/global/cos_g_raw_g_opt") is not None
                    else None
                ),
                "cos_optimizer_gradient_realized_update": (
                    float(metrics["geometry/global/cos_g_opt_delta_model"])
                    if metrics.get("geometry/global/cos_g_opt_delta_model") is not None
                    else None
                ),
                "cos_intended_realized_update": float(metrics["geometry/global/cos_delta_intended_fp32_delta_model"]),
            }
            geometry_rows.append(row)
            stage_geometry.append(row)

        window = min(summary_window, stage.updates)
        early_rollout, late_rollout = stage_rollout[:window], stage_rollout[-window:]
        early_train, late_train = stage_train[:window], stage_train[-window:]
        early_geometry, late_geometry = stage_geometry[:window], stage_geometry[-window:]
        summaries.append(
            {
                "stage": stage.task,
                "stage_number": stage.number,
                "updates": stage.updates,
                "source_run_manifest_status": run_manifest.get("status"),
                "completion_marker_present": (stage.run_dir / "run_complete.json").is_file(),
                "learning_rate": learning_rate,
                "n_samples_per_prompt": n_samples,
                "rollout_batch_size": rollout_batch_size,
                "prompts_consumed": sum(int(row["group_count"]) for row in stage_rollout),
                "responses_consumed": sum(int(row["reward_count"]) for row in stage_rollout),
                "summary_window_updates": window,
                "early_reward_mean": optional_mean(early_rollout, "reward_mean"),
                "late_reward_mean": optional_mean(late_rollout, "reward_mean"),
                "reward_change": optional_change(
                    optional_mean(early_rollout, "reward_mean"),
                    optional_mean(late_rollout, "reward_mean"),
                ),
                "early_informative_group_fraction": optional_mean(early_rollout, "informative_group_fraction"),
                "late_informative_group_fraction": optional_mean(late_rollout, "informative_group_fraction"),
                "late_all_wrong_group_fraction": optional_mean(late_rollout, "all_wrong_group_fraction"),
                "late_all_correct_group_fraction": optional_mean(late_rollout, "all_correct_group_fraction"),
                "early_grad_norm": optional_mean(early_train, "grad_norm"),
                "late_grad_norm": optional_mean(late_train, "grad_norm"),
                "grad_norm_change": optional_change(
                    optional_mean(early_train, "grad_norm"),
                    optional_mean(late_train, "grad_norm"),
                ),
                "gradient_clipped_fraction": optional_mean(stage_train, "gradient_clipped"),
                "early_entropy": optional_mean(early_train, "entropy"),
                "late_entropy": optional_mean(late_train, "entropy"),
                "late_response_len_mean": optional_mean(late_rollout, "response_len_mean"),
                "late_truncated_fraction": optional_mean(late_rollout, "truncated_fraction"),
                "early_model_change_fraction": optional_mean(early_geometry, "model_change_fraction"),
                "late_model_change_fraction": optional_mean(late_geometry, "model_change_fraction"),
                "late_intended_below_half_ulp_fraction": optional_mean(
                    late_geometry, "intended_below_half_ulp_fraction"
                ),
                "late_intended_energy_zeroed_fraction": optional_mean(
                    late_geometry, "intended_energy_zeroed_fraction"
                ),
                "late_energy_survival": optional_mean(late_geometry, "energy_survival"),
                "late_cos_intended_realized_update": optional_mean(late_geometry, "cos_intended_realized_update"),
            }
        )
        provenance.append(
            {
                "stage": stage.task,
                "run_dir": str(stage.run_dir),
                "input_checkpoint": str(stage.input_checkpoint),
                "output_checkpoint": str(stage.output_checkpoint),
                "run_manifest_sha256": sha256(manifest_path),
                "run_manifest_status": run_manifest.get("status"),
                "completion_marker_present": (stage.run_dir / "run_complete.json").is_file(),
                "locked_metric_rows_sha256": {
                    "rollout": rows_sha256(raw_rollout),
                    "train": rows_sha256(raw_train),
                    "geometry": rows_sha256(raw_geometry),
                },
            }
        )
        sequence_offset += stage.updates

    return rollout_rows, train_rows, geometry_rows, summaries, provenance


def resolve_eval_artifact(eval_dir: Path, info: dict[str, Any]) -> Path:
    relative = info.get("run_relative_path")
    if relative:
        path = (eval_dir / str(relative)).resolve()
    else:
        path = Path(str(info["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation artifact: {path}")
    return path


def infer_domain(dataset: str, rows: list[dict[str, Any]]) -> str:
    domains = {
        str(row.get("metadata", {}).get("sequential_domain"))
        for row in rows
        if row.get("metadata", {}).get("sequential_domain") is not None
    }
    if len(domains) > 1:
        raise ValueError(f"Dataset {dataset} has inconsistent sequential_domain metadata: {domains}")
    if domains:
        return next(iter(domains))
    try:
        return DATASET_DOMAIN_FALLBACK[dataset]
    except KeyError as error:
        raise ValueError(f"Cannot infer sequential domain for dataset {dataset}.") from error


def checkpoint_from_eval_manifest(eval_dir: Path) -> Path:
    manifest = read_json(eval_dir / "provenance" / "run_manifest.json")
    command = [str(item) for item in manifest["command"]]
    load_root = Path(command_value(command, "--load")).resolve()
    if "--ckpt-step" not in command:
        return resolve_torch_dist_checkpoint(load_root)
    step = int(command_value(command, "--ckpt-step"))
    return resolve_torch_dist_checkpoint(load_root / f"iter_{step:07d}")


def load_evaluations(
    sequence_root: Path,
    eval_root: Path,
    manifest: dict[str, Any],
    pre_code_checkpoint: Path,
) -> tuple[list[Boundary], dict[str, dict[str, EvalDataset]], list[dict[str, Any]]]:
    expected_checkpoints = {
        "pre_code": resolve_torch_dist_checkpoint(pre_code_checkpoint),
        "code_origin": Path(str(manifest["code_origin"])).resolve(),
        "after_math": Path(str(manifest["stage_endpoints"]["math"])).resolve(),
        "after_knowledge": Path(str(manifest["stage_endpoints"]["knowledge"])).resolve(),
        "after_if": Path(str(manifest["stage_endpoints"]["if"])).resolve(),
    }
    boundaries = []
    evaluations: dict[str, dict[str, EvalDataset]] = {}
    provenance = []
    expected_dataset_names: set[str] | None = None
    reference_identities: dict[str, dict[tuple[int, int], tuple[Any, ...]]] = {}

    for index, (label, dirname, trained_task) in enumerate(BOUNDARIES):
        eval_dir = (eval_root / dirname).resolve()
        completion_path = eval_dir / "run_complete.json"
        completion = read_json(completion_path)
        if completion.get("status") != "complete":
            raise ValueError(f"Evaluation is not complete: {eval_dir}")
        checkpoint = checkpoint_from_eval_manifest(eval_dir)
        if checkpoint != expected_checkpoints[label]:
            raise ValueError(f"Boundary {label} evaluated {checkpoint}, expected {expected_checkpoints[label]}.")
        boundary = Boundary(index, label, trained_task, eval_dir, checkpoint)
        boundaries.append(boundary)

        index_path = eval_dir / "eval_artifacts" / "index.jsonl"
        index_rows = read_jsonl(index_path)
        if len(index_rows) != 1:
            raise ValueError(f"Expected exactly one evaluation artifact index row in {index_path}.")
        index_record = index_rows[0]
        if index_record.get("eval_phase") != "eval_only":
            raise ValueError(f"Boundary {label} is not an eval_only measurement.")
        dataset_names = set(index_record["datasets"])
        if expected_dataset_names is None:
            expected_dataset_names = dataset_names
        elif dataset_names != expected_dataset_names:
            raise ValueError(f"Dataset set changed at boundary {label}.")

        boundary_datasets = {}
        dataset_provenance = {}
        for dataset, info in sorted(index_record["datasets"].items()):
            path = resolve_eval_artifact(eval_dir, info)
            actual_hash = sha256(path)
            if actual_hash != info["sha256"]:
                raise ValueError(f"Evaluation artifact hash mismatch: {path}")
            rows = read_jsonl(path)
            if len(rows) != int(info["samples"]):
                raise ValueError(f"Evaluation artifact sample count mismatch: {path}")
            group_size = int(info["n_samples_per_prompt"])
            if group_size <= 0:
                raise ValueError(f"Invalid n_samples_per_prompt for {dataset}.")

            identities = {}
            prompt_rewards: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                key = (int(row["prompt_index"]), int(row["sample_within_prompt"]))
                if key in identities:
                    raise ValueError(f"Duplicate sample identity {key} in {path}.")
                identity = (
                    row.get("source_index"),
                    row.get("prompt"),
                    row.get("label"),
                )
                identities[key] = identity
                reward = float(row["reward"])
                if not math.isfinite(reward):
                    raise ValueError(f"Non-finite reward for {dataset} at {label}: {key}")
                prompt_rewards[key[0]].append(reward)
            if any(len(rewards) != group_size for rewards in prompt_rewards.values()):
                raise ValueError(f"Incomplete prompt sample group in {path}.")
            if len(prompt_rewards) != int(info["prompts"]):
                raise ValueError(f"Evaluation artifact prompt count mismatch: {path}")
            if dataset not in reference_identities:
                reference_identities[dataset] = identities
            elif identities != reference_identities[dataset]:
                raise ValueError(f"Fixed evaluation identities changed for {dataset} at {label}.")
            identity_sha256 = rows_sha256(
                [
                    {
                        "prompt_index": prompt_index,
                        "sample_within_prompt": sample_index,
                        "source_index": identity[0],
                        "prompt": identity[1],
                        "label": identity[2],
                    }
                    for (prompt_index, sample_index), identity in sorted(identities.items())
                ]
            )

            domain = infer_domain(dataset, rows)
            boundary_datasets[dataset] = EvalDataset(
                dataset,
                domain,
                group_size,
                rows,
                dict(prompt_rewards),
                path,
                actual_hash,
            )
            dataset_provenance[dataset] = {
                "path": str(path),
                "sha256": actual_hash,
                "samples": len(rows),
                "prompts": len(prompt_rewards),
                "n_samples_per_prompt": group_size,
                "fixed_sample_identity_sha256": identity_sha256,
            }
        evaluations[label] = boundary_datasets
        provenance.append(
            {
                "boundary": label,
                "eval_dir": str(eval_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_metadata_sha256": sha256(checkpoint / ".metadata"),
                "checkpoint_common_sha256": sha256(checkpoint / "common.pt"),
                "run_manifest_sha256": sha256(eval_dir / "provenance" / "run_manifest.json"),
                "completion_marker_sha256": sha256(completion_path),
                "artifact_index_sha256": sha256(index_path),
                "artifact_index_record_sha256": rows_sha256([index_record]),
                "datasets": dataset_provenance,
            }
        )

    if expected_dataset_names is None:
        raise ValueError(f"No evaluation datasets found under {eval_root}.")
    required = set(DATASET_DOMAIN_FALLBACK)
    if expected_dataset_names != required:
        raise ValueError(
            f"Expected the locked six-dataset suite {sorted(required)}, found {sorted(expected_dataset_names)}."
        )
    for domain, dataset in PRIMARY_DATASET.items():
        actual_domain = evaluations["code_origin"][dataset].domain
        if actual_domain != domain:
            raise ValueError(f"Primary dataset {dataset} is tagged {actual_domain}, expected {domain}.")
    return boundaries, evaluations, provenance


def binary_rewards(dataset: EvalDataset) -> bool:
    return all(float(row["reward"]) in {0.0, 1.0} for row in dataset.rows)


def prompt_values(dataset: EvalDataset) -> tuple[np.ndarray, np.ndarray]:
    prompt_ids = np.asarray(sorted(dataset.prompt_rewards), dtype=np.int64)
    values = np.asarray(
        [float(np.mean(dataset.prompt_rewards[int(prompt_id)])) for prompt_id in prompt_ids],
        dtype=np.float64,
    )
    return prompt_ids, values


def bootstrap_mean_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty vector.")
    rng = np.random.default_rng(seed)
    # Chunk resampling to keep the analysis bounded for large evaluation sets.
    means = np.empty(samples, dtype=np.float64)
    chunk_size = max(1, min(samples, 2_000_000 // max(values.size, 1)))
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def dataset_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode()).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def pass_at_k(rewards: list[float], k: int) -> float:
    successes = sum(reward == 1.0 for reward in rewards)
    sample_count = len(rewards)
    if k > sample_count:
        raise ValueError(f"Cannot compute pass@{k} from {sample_count} samples.")
    if sample_count - successes < k:
        return 1.0
    return 1.0 - math.comb(sample_count - successes, k) / math.comb(sample_count, k)


def summarize_evaluations(
    boundaries: list[Boundary],
    evaluations: dict[str, dict[str, EvalDataset]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_rows = []
    pass_rows = []
    for boundary in boundaries:
        for dataset_name, dataset in sorted(evaluations[boundary.label].items()):
            _, values = prompt_values(dataset)
            ci_low, ci_high = bootstrap_mean_interval(
                values,
                bootstrap_samples,
                dataset_seed(bootstrap_seed, boundary.label, dataset_name, "score"),
            )
            rewards = [float(row["reward"]) for row in dataset.rows]
            status_errors = sum(str(row.get("status")) not in {"completed", "truncated"} for row in dataset.rows)
            truncated = sum(str(row.get("status")) == "truncated" for row in dataset.rows)
            sandbox_records = [
                row.get("metadata", {}).get("sandbox_eval")
                for row in dataset.rows
                if isinstance(row.get("metadata", {}).get("sandbox_eval"), dict)
            ]
            sandbox_infrastructure_errors = sum(
                int(record.get("infrastructure_errors", 0) or 0) for record in sandbox_records
            )
            if dataset_name == "livecodebench_online":
                if len(sandbox_records) != len(dataset.rows):
                    raise ValueError(
                        f"LiveCodeBench diagnostics are missing at boundary {boundary.label}: "
                        f"{len(sandbox_records)}/{len(dataset.rows)} samples."
                    )
                if sandbox_infrastructure_errors:
                    raise ValueError(
                        f"LiveCodeBench has {sandbox_infrastructure_errors} infrastructure errors "
                        f"at boundary {boundary.label}; refusing to count them as model failures."
                    )
            score_rows.append(
                {
                    "boundary": boundary.label,
                    "boundary_index": boundary.index,
                    "trained_task": boundary.trained_task,
                    "checkpoint": str(boundary.checkpoint),
                    "domain": dataset.domain,
                    "dataset": dataset_name,
                    "is_primary_domain_metric": dataset_name == PRIMARY_DATASET[dataset.domain],
                    "prompts": len(dataset.prompt_rewards),
                    "samples_per_prompt": dataset.samples_per_prompt,
                    "samples": len(dataset.rows),
                    "mean_reward": float(np.mean(rewards)),
                    "prompt_cluster_ci95_low": ci_low,
                    "prompt_cluster_ci95_high": ci_high,
                    "reward_std": float(np.std(rewards)),
                    "mean_effective_response_length": float(
                        np.mean([float(row["effective_response_length"]) for row in dataset.rows])
                    ),
                    "truncated_fraction": truncated / len(dataset.rows),
                    "response_error_fraction": status_errors / len(dataset.rows),
                    "sandbox_evaluated_samples": len(sandbox_records),
                    "sandbox_infrastructure_errors": sandbox_infrastructure_errors,
                    "sandbox_execution_errors": sum(
                        int(record.get("execution_errors", 0) or 0) for record in sandbox_records
                    ),
                    "sandbox_timeouts": sum(int(record.get("timeouts", 0) or 0) for record in sandbox_records),
                    "binary_reward": binary_rewards(dataset),
                }
            )
            if binary_rewards(dataset):
                for k in (1, 2, 4, 5, 8, 10):
                    if k > dataset.samples_per_prompt:
                        continue
                    pass_rows.append(
                        {
                            "boundary": boundary.label,
                            "boundary_index": boundary.index,
                            "domain": dataset.domain,
                            "dataset": dataset_name,
                            "k": k,
                            "pass_at_k": float(
                                np.mean(
                                    [
                                        pass_at_k(dataset.prompt_rewards[prompt_id], k)
                                        for prompt_id in sorted(dataset.prompt_rewards)
                                    ]
                                )
                            ),
                            "prompts": len(dataset.prompt_rewards),
                            "samples_per_prompt": dataset.samples_per_prompt,
                        }
                    )
    return score_rows, pass_rows


def paired_change(
    before: EvalDataset,
    after: EvalDataset,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    before_ids, before_values = prompt_values(before)
    after_ids, after_values = prompt_values(after)
    if not np.array_equal(before_ids, after_ids):
        raise ValueError(f"Prompt identities differ for paired dataset {before.name}.")
    deltas = after_values - before_values
    ci_low, ci_high = bootstrap_mean_interval(deltas, bootstrap_samples, seed)

    before_by_key = {
        (int(row["prompt_index"]), int(row["sample_within_prompt"])): float(row["reward"]) for row in before.rows
    }
    after_by_key = {
        (int(row["prompt_index"]), int(row["sample_within_prompt"])): float(row["reward"]) for row in after.rows
    }
    if before_by_key.keys() != after_by_key.keys():
        raise ValueError(f"Sample identities differ for paired dataset {before.name}.")
    binary = binary_rewards(before) and binary_rewards(after)
    pass_to_fail = None
    fail_to_pass = None
    if binary:
        pass_to_fail = sum(before_by_key[key] == 1.0 and after_by_key[key] == 0.0 for key in before_by_key)
        fail_to_pass = sum(before_by_key[key] == 0.0 and after_by_key[key] == 1.0 for key in before_by_key)

    before_score = float(before_values.mean())
    after_score = float(after_values.mean())
    delta = after_score - before_score
    return {
        "before_score": before_score,
        "after_score": after_score,
        "signed_change": delta,
        "paired_bootstrap_ci95_low": ci_low,
        "paired_bootstrap_ci95_high": ci_high,
        "interference_loss": max(0.0, -delta),
        "facilitation_gain": max(0.0, delta),
        "relative_signed_change": delta / before_score if before_score > 0 else None,
        "prompt_improved_count": int(np.count_nonzero(deltas > 0)),
        "prompt_unchanged_count": int(np.count_nonzero(deltas == 0)),
        "prompt_regressed_count": int(np.count_nonzero(deltas < 0)),
        "sample_pass_to_fail_count": pass_to_fail,
        "sample_fail_to_pass_count": fail_to_pass,
        "prompts": int(deltas.size),
        "samples": len(before.rows),
    }


def bootstrap_equal_task_mean_change_interval(
    pairs: list[tuple[EvalDataset, EvalDataset]],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not pairs:
        raise ValueError("At least one paired task is required.")
    task_deltas = []
    for before, after in pairs:
        before_ids, before_values = prompt_values(before)
        after_ids, after_values = prompt_values(after)
        if not np.array_equal(before_ids, after_ids):
            raise ValueError(f"Prompt identities differ for paired dataset {before.name}.")
        task_deltas.append(after_values - before_values)

    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    maximum_prompts = max(values.size for values in task_deltas)
    chunk_size = max(1, min(samples, 2_000_000 // maximum_prompts))
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        equal_task_mean = np.zeros(stop - start, dtype=np.float64)
        for values in task_deltas:
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            equal_task_mean += values[indices].mean(axis=1)
        means[start:stop] = equal_task_mean / len(task_deltas)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def transfer_and_forgetting(
    boundaries: list[Boundary],
    evaluations: dict[str, dict[str, EvalDataset]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    adjacent_rows = []
    primary_transfer = []
    boundary_by_label = {boundary.label: boundary for boundary in boundaries}

    for before_boundary, after_boundary in zip(boundaries, boundaries[1:], strict=False):
        if after_boundary.trained_task is None:
            raise ValueError(f"Boundary {after_boundary.label} has no trained task.")
        for dataset_name, before_dataset in sorted(evaluations[before_boundary.label].items()):
            after_dataset = evaluations[after_boundary.label][dataset_name]
            paired = paired_change(
                before_dataset,
                after_dataset,
                bootstrap_samples,
                dataset_seed(
                    bootstrap_seed,
                    before_boundary.label,
                    after_boundary.label,
                    dataset_name,
                    "adjacent",
                ),
            )
            row = {
                "training_stage": after_boundary.trained_task,
                "before_boundary": before_boundary.label,
                "after_boundary": after_boundary.label,
                "domain": before_dataset.domain,
                "dataset": dataset_name,
                "is_own_task": before_dataset.domain == after_boundary.trained_task,
                "is_previously_trained_task": DOMAINS.index(before_dataset.domain)
                < DOMAINS.index(after_boundary.trained_task),
                "is_future_task": DOMAINS.index(before_dataset.domain) > DOMAINS.index(after_boundary.trained_task),
                **paired,
            }
            adjacent_rows.append(row)
            if dataset_name == PRIMARY_DATASET[before_dataset.domain]:
                primary_transfer.append(row.copy())

    acquisition_boundary = {
        "code": "code_origin",
        "math": "after_math",
        "knowledge": "after_knowledge",
        "if": "after_if",
    }
    final_label = boundaries[-1].label
    forgetting_rows = []
    for domain in DOMAINS:
        dataset_name = PRIMARY_DATASET[domain]
        acquired_label = acquisition_boundary[domain]
        acquired = evaluations[acquired_label][dataset_name]
        final = evaluations[final_label][dataset_name]
        paired = paired_change(
            acquired,
            final,
            bootstrap_samples,
            dataset_seed(bootstrap_seed, domain, "acquisition_to_final"),
        )
        acquired_score = paired["before_score"]
        final_score = paired["after_score"]
        observed_labels = [
            boundary.label for boundary in boundaries if boundary.index >= boundary_by_label[acquired_label].index
        ]
        observed_scores = [
            float(np.mean([float(row["reward"]) for row in evaluations[label][dataset_name].rows]))
            for label in observed_labels
        ]
        peak_index = int(np.argmax(observed_scores))
        peak_score = observed_scores[peak_index]
        classical_forgetting = max(0.0, acquired_score - final_score)
        peak_forgetting = max(0.0, peak_score - final_score)
        forgetting_rows.append(
            {
                "domain": domain,
                "dataset": dataset_name,
                "acquisition_boundary": acquired_label,
                "final_boundary": final_label,
                "acquisition_score": acquired_score,
                "final_score": final_score,
                "signed_final_change": final_score - acquired_score,
                "signed_final_change_ci95_low": paired["paired_bootstrap_ci95_low"],
                "signed_final_change_ci95_high": paired["paired_bootstrap_ci95_high"],
                "forgetting": classical_forgetting,
                "forgetting_rate": classical_forgetting / acquired_score if acquired_score > 0 else None,
                "retention_ratio": final_score / acquired_score if acquired_score > 0 else None,
                "peak_boundary": observed_labels[peak_index],
                "peak_score": peak_score,
                "peak_forgetting": peak_forgetting,
                "peak_forgetting_rate": peak_forgetting / peak_score if peak_score > 0 else None,
                "sample_pass_to_fail_count": paired["sample_pass_to_fail_count"],
                "sample_fail_to_pass_count": paired["sample_fail_to_pass_count"],
            }
        )

    final_scores = {row["domain"]: float(row["final_score"]) for row in forgetting_rows}
    prior_rows = [row for row in forgetting_rows if row["domain"] != "if"]
    pre_acquisition_boundary = {
        "math": "code_origin",
        "knowledge": "after_math",
        "if": "after_knowledge",
    }
    forward_transfer_rows = []
    forward_transfer_pairs = []
    for domain, before_label in pre_acquisition_boundary.items():
        dataset_name = PRIMARY_DATASET[domain]
        reference_dataset = evaluations["pre_code"][dataset_name]
        pre_acquisition_dataset = evaluations[before_label][dataset_name]
        forward_transfer_pairs.append((reference_dataset, pre_acquisition_dataset))
        paired = paired_change(
            reference_dataset,
            pre_acquisition_dataset,
            bootstrap_samples,
            dataset_seed(bootstrap_seed, domain, "pretrained_base_referenced_forward_transfer"),
        )
        forward_transfer_rows.append(
            {
                "domain": domain,
                "dataset": dataset_name,
                "reference_boundary": "pre_code",
                "pre_acquisition_boundary": before_label,
                "reference_score": paired["before_score"],
                "pre_acquisition_score": paired["after_score"],
                "forward_transfer": paired["signed_change"],
                "paired_bootstrap_ci95_low": paired["paired_bootstrap_ci95_low"],
                "paired_bootstrap_ci95_high": paired["paired_bootstrap_ci95_high"],
            }
        )

    base_to_final_rows = []
    base_to_final_pairs = []
    for domain in DOMAINS:
        dataset_name = PRIMARY_DATASET[domain]
        reference_dataset = evaluations["pre_code"][dataset_name]
        final_dataset = evaluations[final_label][dataset_name]
        base_to_final_pairs.append((reference_dataset, final_dataset))
        paired = paired_change(
            reference_dataset,
            final_dataset,
            bootstrap_samples,
            dataset_seed(bootstrap_seed, domain, "pretrained_base_to_final"),
        )
        base_to_final_rows.append(
            {
                "domain": domain,
                "dataset": dataset_name,
                "reference_boundary": "pre_code",
                "final_boundary": final_label,
                "reference_score": paired["before_score"],
                "final_score": paired["after_score"],
                "signed_change": paired["signed_change"],
                "paired_bootstrap_ci95_low": paired["paired_bootstrap_ci95_low"],
                "paired_bootstrap_ci95_high": paired["paired_bootstrap_ci95_high"],
            }
        )

    code_acquisition = next(
        row for row in primary_transfer if row["training_stage"] == "code" and row["domain"] == "code"
    )
    forward_transfer_ci = bootstrap_equal_task_mean_change_interval(
        forward_transfer_pairs,
        bootstrap_samples,
        dataset_seed(bootstrap_seed, "pretrained_base_referenced_FWT", "equal_task_mean"),
    )
    base_to_final_ci = bootstrap_equal_task_mean_change_interval(
        base_to_final_pairs,
        bootstrap_samples,
        dataset_seed(bootstrap_seed, "pretrained_base_to_final", "equal_task_mean"),
    )
    base_scores = {row["domain"]: float(row["reference_score"]) for row in base_to_final_rows}
    continual = {
        "primary_metric_policy": PRIMARY_DATASET,
        "pretrained_base_ACC": float(np.mean(list(base_scores.values()))),
        "final_ACC": float(np.mean(list(final_scores.values()))),
        "pretrained_base_to_final_ACC_change": float(
            np.mean([float(row["signed_change"]) for row in base_to_final_rows])
        ),
        "pretrained_base_to_final_ACC_change_ci95_low": base_to_final_ci[0],
        "pretrained_base_to_final_ACC_change_ci95_high": base_to_final_ci[1],
        "code_acquisition_gain": float(code_acquisition["signed_change"]),
        "code_acquisition_gain_ci95_low": float(code_acquisition["paired_bootstrap_ci95_low"]),
        "code_acquisition_gain_ci95_high": float(code_acquisition["paired_bootstrap_ci95_high"]),
        "classical_BWT": float(np.mean([float(row["signed_final_change"]) for row in prior_rows])),
        "classical_BWT_tasks": [row["domain"] for row in prior_rows],
        "mean_nonnegative_forgetting": float(np.mean([float(row["forgetting"]) for row in prior_rows])),
        "mean_forgetting_rate": float(
            np.mean([float(row["forgetting_rate"]) for row in prior_rows if row["forgetting_rate"] is not None])
        ),
        "pretrained_base_referenced_FWT": float(
            np.mean([float(row["forward_transfer"]) for row in forward_transfer_rows])
        ),
        "pretrained_base_referenced_FWT_ci95_low": forward_transfer_ci[0],
        "pretrained_base_referenced_FWT_ci95_high": forward_transfer_ci[1],
        "pretrained_base_referenced_FWT_tasks": [row["domain"] for row in forward_transfer_rows],
        "FWT_reference": "original_pretrained_Qwen3-1.7B_pre_code_boundary",
        "classical_random_initialization_FWT": None,
        "FWT_status": (
            "pretrained-base-referenced_FWT_available; random-initialization-referenced_FWT_not_applicable"
        ),
        "if_forgetting_status": "not_identifiable_because_IF_is_the_final_stage",
    }
    return (
        adjacent_rows,
        primary_transfer,
        forgetting_rows,
        forward_transfer_rows,
        base_to_final_rows,
        continual,
    )


def load_parameter_geometry(
    path: Path | None,
    primary_transfer: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return [], [], [], None
    summary = read_json(path)
    global_row = summary["stage_local_global"]
    labels = [str(stage["label"]) for stage in summary["stages"]]
    if tuple(labels) != DOMAINS:
        raise ValueError(f"Parameter geometry stages are {labels}, expected {list(DOMAINS)}.")
    stage_rows = []
    for label in labels:
        stage_rows.append(
            {
                "stage": label,
                "parameter_count": int(global_row["parameter_count"]),
                "update_norm": float(global_row[f"update_norm/{label}"]),
                "relative_update_norm": float(global_row[f"relative_update_norm/{label}"]),
                "exact_zero_sparsity": float(global_row[f"exact_zero_sparsity/{label}"]),
                "visible_sparsity@1e-6": float(global_row[f"visible_sparsity@1e-6/{label}"]),
                "visible_sparsity@1e-5": float(global_row[f"visible_sparsity@1e-5/{label}"]),
                "visible_sparsity@1e-4": float(global_row[f"visible_sparsity@1e-4/{label}"]),
                "bf16_aware_sparsity_eta1e-3": float(global_row[f"bf16_aware_sparsity_eta1e-3/{label}"]),
                "relative_sparsity_tau1e-3": float(global_row[f"relative_sparsity_tau1e-3/{label}"]),
            }
        )
    pair_rows = []
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            pair_rows.append(
                {
                    "left_stage": left,
                    "right_stage": right,
                    "dot": float(global_row[f"dot/{left}__{right}"]),
                    "cosine": float(global_row[f"cosine/{left}__{right}"]),
                    "left_interference_ratio_from_right": float(
                        global_row[f"interference_ratio/{left}_self_from_{right}"]
                    ),
                    "right_interference_ratio_from_left": float(
                        global_row[f"interference_ratio/{right}_self_from_{left}"]
                    ),
                    "support_jaccard@1e-5": float(global_row[f"support_jaccard@1e-5/{left}__{right}"]),
                    "support_union_sparsity@1e-5": float(global_row[f"support_union_sparsity@1e-5/{left}__{right}"]),
                    "left_to_right_support_overlap_lift@1e-5": float(
                        global_row[f"support_overlap_lift@1e-5/{left}_to_{right}"]
                    ),
                    "right_to_left_support_overlap_lift@1e-5": float(
                        global_row[f"support_overlap_lift@1e-5/{right}_to_{left}"]
                    ),
                    "support_sign_conflict_fraction@1e-5": float(
                        global_row[f"support_sign_conflict_fraction@1e-5/{left}__{right}"]
                    ),
                }
            )

    transfer_lookup = {(str(row["training_stage"]), str(row["domain"])): row for row in primary_transfer}
    pair_lookup = {(str(row["left_stage"]), str(row["right_stage"])): row for row in pair_rows}
    alignment_rows = []
    for current_index, current in enumerate(labels):
        for previous in labels[:current_index]:
            pair = pair_lookup[(previous, current)]
            behavior = transfer_lookup[(current, previous)]
            alignment_rows.append(
                {
                    "training_stage": current,
                    "previous_task": previous,
                    "previous_task_performance_change": behavior["signed_change"],
                    "previous_task_change_ci95_low": behavior["paired_bootstrap_ci95_low"],
                    "previous_task_change_ci95_high": behavior["paired_bootstrap_ci95_high"],
                    "stage_update_cosine": pair["cosine"],
                    "stage_update_dot": pair["dot"],
                    "support_jaccard@1e-5": pair["support_jaccard@1e-5"],
                    "support_sign_conflict_fraction@1e-5": pair["support_sign_conflict_fraction@1e-5"],
                    "interpretation_limit": (
                        "Checkpoint-delta alignment is descriptive and is not a same-checkpoint raw-gradient dot product."
                    ),
                }
            )
    return stage_rows, pair_rows, alignment_rows, summary


def smooth_xy(rows: list[dict[str, Any]], key: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([float(row["sequence_update"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    effective = min(window, len(y))
    if effective <= 1:
        return x, y
    smoothed = np.convolve(y, np.ones(effective) / effective, mode="valid")
    return x[effective - 1 :], smoothed


def save_plots(
    output_dir: Path,
    rollout_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    primary_transfer: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
    parameter_stage_rows: list[dict[str, Any]],
    parameter_pair_rows: list[dict[str, Any]],
    smoothing_window: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"code": "tab:purple", "math": "tab:blue", "knowledge": "tab:orange", "if": "tab:green"}
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    for stage in DOMAINS:
        rollout = [row for row in rollout_rows if row["stage"] == stage]
        train = [row for row in train_rows if row["stage"] == stage]
        geometry = [row for row in geometry_rows if row["stage"] == stage]
        for axis, rows, key, label in (
            (axes[0, 0], rollout, "reward_mean", stage),
            (axes[0, 1], rollout, "informative_group_fraction", stage),
            (axes[1, 0], train, "grad_norm", stage),
            (axes[1, 1], geometry, "model_change_fraction", stage),
        ):
            x, y = smooth_xy(rows, key, smoothing_window)
            axis.plot(x, y, label=label, color=colors[stage])
    axes[0, 0].set_title("On-policy training reward")
    axes[0, 1].set_title("Informative GRPO group fraction")
    axes[1, 0].set_title("Raw gradient norm")
    axes[1, 1].set_title("Realized BF16 coordinate-change fraction")
    for axis in axes.flat:
        axis.set_xlabel("sequential optimizer updates from pretrained base")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "training_dynamics.png", dpi=200)
    plt.close(figure)

    boundary_labels = [item[0] for item in BOUNDARIES]
    primary_matrix = np.asarray(
        [
            [
                next(
                    float(row["mean_reward"])
                    for row in score_rows
                    if row["boundary"] == boundary and row["dataset"] == PRIMARY_DATASET[domain]
                )
                for domain in DOMAINS
            ]
            for boundary in boundary_labels
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    image = axis.imshow(primary_matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    axis.set_xticks(np.arange(len(DOMAINS)), DOMAINS)
    axis.set_yticks(np.arange(len(boundary_labels)), boundary_labels)
    for row_index in range(primary_matrix.shape[0]):
        for column_index in range(primary_matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{primary_matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis, label="mean task reward")
    axis.set_title("Sequential primary-task performance matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "primary_performance_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    stages = DOMAINS
    transfer_matrix = np.asarray(
        [
            [
                next(
                    float(row["signed_change"])
                    for row in primary_transfer
                    if row["training_stage"] == stage and row["domain"] == domain
                )
                for domain in DOMAINS
            ]
            for stage in stages
        ],
        dtype=np.float64,
    )
    limit = max(float(np.max(np.abs(transfer_matrix))), np.finfo(np.float64).eps)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    image = axis.imshow(transfer_matrix, vmin=-limit, vmax=limit, cmap="coolwarm")
    axis.set_xticks(np.arange(len(DOMAINS)), DOMAINS)
    axis.set_yticks(np.arange(len(stages)), stages)
    axis.set_ylabel("training stage")
    for row_index in range(transfer_matrix.shape[0]):
        for column_index in range(transfer_matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{transfer_matrix[row_index, column_index]:+.3f}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis, label="score after − before")
    axis.set_title("Behavioral transfer/interference on primary metrics")
    figure.tight_layout()
    figure.savefig(output_dir / "primary_transfer_matrix.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    positions = np.arange(len(forgetting_rows))
    pretrained_base = np.asarray(
        [
            next(
                float(row["mean_reward"])
                for row in score_rows
                if row["boundary"] == "pre_code" and row["dataset"] == PRIMARY_DATASET[str(item["domain"])]
            )
            for item in forgetting_rows
        ]
    )
    acquired = np.asarray([float(row["acquisition_score"]) for row in forgetting_rows])
    final = np.asarray([float(row["final_score"]) for row in forgetting_rows])
    width = 0.25
    axis.bar(positions - width, pretrained_base, width, label="pretrained base")
    axis.bar(positions, acquired, width, label="at acquisition")
    axis.bar(positions + width, final, width, label="final")
    axis.set_xticks(positions, [str(row["domain"]) for row in forgetting_rows])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("primary metric score")
    axis.set_title("Pretrained baseline, task acquisition, and final retention")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "task_retention.png", dpi=200)
    plt.close(figure)

    if parameter_stage_rows and parameter_pair_rows:
        figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
        axes[0].bar(
            [str(row["stage"]) for row in parameter_stage_rows],
            [float(row["visible_sparsity@1e-5"]) for row in parameter_stage_rows],
        )
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_ylabel("fraction |Δθ| ≤ 1e-5")
        axes[0].set_title("Visible update sparsity")
        matrix = np.zeros((len(parameter_stage_rows), len(parameter_stage_rows)), dtype=np.float64)
        labels = [str(row["stage"]) for row in parameter_stage_rows]
        for row in parameter_pair_rows:
            left = labels.index(str(row["left_stage"]))
            right = labels.index(str(row["right_stage"]))
            matrix[left, right] = matrix[right, left] = float(row["cosine"])
        image = axes[1].imshow(matrix, vmin=-0.02, vmax=0.02, cmap="coolwarm")
        axes[1].set_xticks(np.arange(len(labels)), labels)
        axes[1].set_yticks(np.arange(len(labels)), labels)
        for row_index in range(len(labels)):
            for column_index in range(len(labels)):
                axes[1].text(
                    column_index,
                    row_index,
                    "—" if row_index == column_index else f"{matrix[row_index, column_index]:+.4f}",
                    ha="center",
                    va="center",
                )
        figure.colorbar(image, ax=axes[1], label="stage-local update cosine")
        axes[1].set_title("Cross-stage checkpoint-update alignment")
        for axis in axes:
            axis.grid(alpha=0.18)
        figure.tight_layout()
        figure.savefig(output_dir / "parameter_update_geometry.png", dpi=200)
        plt.close(figure)

        pair_labels = [f"{row['left_stage']}\n{row['right_stage']}" for row in parameter_pair_rows]
        overlap_lifts = [
            0.5
            * (
                float(row["left_to_right_support_overlap_lift@1e-5"])
                + float(row["right_to_left_support_overlap_lift@1e-5"])
            )
            for row in parameter_pair_rows
        ]
        figure, axes = plt.subplots(1, 3, figsize=(14.6, 4.4))
        for axis, values, title, ylabel, baseline in (
            (
                axes[0],
                [float(row["support_jaccard@1e-5"]) for row in parameter_pair_rows],
                "Visible-support Jaccard",
                "fraction",
                None,
            ),
            (axes[1], overlap_lifts, "Support-overlap lift", "relative to independence", 1.0),
            (
                axes[2],
                [float(row["support_sign_conflict_fraction@1e-5"]) for row in parameter_pair_rows],
                "Opposite signs on intersection",
                "fraction",
                0.5,
            ),
        ):
            axis.bar(pair_labels, values)
            if baseline is not None:
                axis.axhline(baseline, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.tick_params(axis="x", labelsize=8)
            axis.grid(axis="y", alpha=0.22)
        figure.suptitle("Sparse coordinate reuse despite near-zero global cosine")
        figure.tight_layout()
        figure.savefig(output_dir / "parameter_support_reuse.png", dpi=200)
        plt.close(figure)


def build_report(
    output_dir: Path,
    score_rows: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
    forward_transfer_rows: list[dict[str, Any]],
    continual: dict[str, Any],
    parameter_pair_rows: list[dict[str, Any]],
    raw_gradient_summary: dict[str, Any] | None,
) -> None:
    boundary_labels = [item[0] for item in BOUNDARIES]
    lines = [
        "# Sequential pretrained base → Code → Math → Knowledge → IF analysis",
        "",
        "This report is generated from locked test artifacts. Scores are mean task rewards; the primary metric policy "
        "is Code/LiveCodeBench, Math/MATH500, Knowledge/GPQA, and IF/IFBench.",
        "",
        "## Primary performance matrix",
        "",
        "| boundary | code | math | knowledge | if |",
        "|---|---:|---:|---:|---:|",
    ]
    for boundary in boundary_labels:
        values = {
            domain: next(
                float(row["mean_reward"])
                for row in score_rows
                if row["boundary"] == boundary and row["dataset"] == PRIMARY_DATASET[domain]
            )
            for domain in DOMAINS
        }
        lines.append(
            f"| {boundary} | {values['code']:.4f} | {values['math']:.4f} | "
            f"{values['knowledge']:.4f} | {values['if']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Acquisition-to-final retention",
            "",
            "| task | acquired | final | signed Δ | forgetting | rate | retention |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in forgetting_rows:
        rate = "NA" if row["forgetting_rate"] is None else f"{float(row['forgetting_rate']):.4f}"
        retention = "NA" if row["retention_ratio"] is None else f"{float(row['retention_ratio']):.4f}"
        lines.append(
            f"| {row['domain']} | {float(row['acquisition_score']):.4f} | "
            f"{float(row['final_score']):.4f} | {float(row['signed_final_change']):+.4f} | "
            f"{float(row['forgetting']):.4f} | {rate} | {retention} |"
        )
    lines.extend(
        [
            "",
            f"Final ACC = {float(continual['final_ACC']):.4f}; classical BWT over Code/Math/Knowledge = "
            f"{float(continual['classical_BWT']):+.4f}.",
            "",
            f"The original pretrained Qwen3-1.7B ACC is {float(continual['pretrained_base_ACC']):.4f}; "
            f"base-to-final ACC change is {float(continual['pretrained_base_to_final_ACC_change']):+.4f} "
            f"[{float(continual['pretrained_base_to_final_ACC_change_ci95_low']):+.4f}, "
            f"{float(continual['pretrained_base_to_final_ACC_change_ci95_high']):+.4f}]. "
            f"Code acquisition gain is {float(continual['code_acquisition_gain']):+.4f}, and pretrained-base-"
            f"referenced FWT over Math/Knowledge/IF is "
            f"{float(continual['pretrained_base_referenced_FWT']):+.4f} "
            f"[{float(continual['pretrained_base_referenced_FWT_ci95_low']):+.4f}, "
            f"{float(continual['pretrained_base_referenced_FWT_ci95_high']):+.4f}].",
            "Per-task base-referenced forward transfer is "
            + ", ".join(f"{row['domain']} {float(row['forward_transfer']):+.4f}" for row in forward_transfer_rows)
            + ".",
            "",
            "The FWT reference is the original pretrained model, not random initialization. IF is the final stage, "
            "so post-IF forgetting is not identifiable.",
            "",
            "## Interpretation guardrails",
            "",
            "Negative adjacent test-score changes are behavioral interference. Checkpoint-delta cosine is a parameter-"
            "trajectory descriptor, not a same-checkpoint raw-gradient interference term and not a causal estimate.",
        ]
    )
    if parameter_pair_rows:
        lines.extend(
            [
                "",
                "Stage-local checkpoint-update cosines are: "
                + ", ".join(
                    f"{row['left_stage']}–{row['right_stage']} {float(row['cosine']):+.6f}"
                    for row in parameter_pair_rows
                )
                + ".",
            ]
        )
    if raw_gradient_summary is not None:
        pairwise = raw_gradient_summary["pairwise_interference"]
        changes = raw_gradient_summary["pairwise_changes"]
        defined_pairs = [row for row in pairwise if row["cosine"] is not None]
        defined_changes = [row for row in changes if row["cosine_change"] is not None]
        max_abs_cosine = max((abs(float(row["cosine"])) for row in defined_pairs), default=None)
        max_abs_cosine_text = "n/a" if max_abs_cosine is None else f"{max_abs_cosine:.4f}"
        sign_changes = sum(bool(row["conflict_status_changed"]) for row in defined_changes)
        anchor_count = len({str(row["anchor"]) for row in pairwise})
        lines.extend(
            [
                "",
                "## Same-checkpoint raw-gradient probes",
                "",
                f"Across {len(defined_pairs)}/{len(pairwise)} direction-defined checkpoint-pair measurements at "
                f"{anchor_count} anchors, max |cosine| = {max_abs_cosine_text}. The conflict sign changed at "
                f"{sign_changes}/{len(defined_changes)} direction-defined adjacent-anchor comparisons. Undefined "
                "cosines arise from zero-norm gradients and are not labeled orthogonal. These are exact reductions "
                "over each sampled batch, "
                "not causal estimates of long-horizon forgetting.",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_chinese_report(
    output_dir: Path,
    training_summaries: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    pass_rows: list[dict[str, Any]],
    primary_transfer: list[dict[str, Any]],
    forgetting_rows: list[dict[str, Any]],
    forward_transfer_rows: list[dict[str, Any]],
    base_to_final_rows: list[dict[str, Any]],
    continual: dict[str, Any],
    parameter_stage_rows: list[dict[str, Any]],
    parameter_pair_rows: list[dict[str, Any]],
    spectral_summary: dict[str, Any] | None,
    raw_gradient_summary: dict[str, Any] | None,
) -> None:
    lines = [
        "# Qwen3-1.7B 原版 → Code → Math → Knowledge → IF 序贯训练指标分析",
        "",
        "> 本报告由锁定 checkpoint、固定测试 artifact 和参数分析流水线自动生成。行为干扰、参数位移对齐与"
        "同-checkpoint 原始梯度冲突是三类不同证据，以下分别报告。",
        "",
        "## 1. 主任务性能矩阵",
        "",
        "主指标预注册为 Code/LiveCodeBench、Math/MATH500、Knowledge/GPQA、IF/IFBench；四域等权。",
        "",
        "| 边界 | Code | Math | Knowledge | IF |",
        "|---|---:|---:|---:|---:|",
    ]
    for boundary, _, _ in BOUNDARIES:
        values = {
            domain: next(
                float(row["mean_reward"])
                for row in score_rows
                if row["boundary"] == boundary and row["dataset"] == PRIMARY_DATASET[domain]
            )
            for domain in DOMAINS
        }
        lines.append(
            f"| `{boundary}` | {values['code']:.4f} | {values['math']:.4f} | "
            f"{values['knowledge']:.4f} | {values['if']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. 全部测试集与不确定性",
            "",
            "区间为 prompt-cluster bootstrap 95% CI；多采样数据集先在 prompt 内平均。",
            "",
            "| 边界 | 数据集 | 域 | score | 95% CI | n×samples | 截断率 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in score_rows:
        lines.append(
            f"| `{row['boundary']}` | {row['dataset']} | {row['domain']} | "
            f"{float(row['mean_reward']):.4f} | [{float(row['prompt_cluster_ci95_low']):.4f}, "
            f"{float(row['prompt_cluster_ci95_high']):.4f}] | {row['prompts']}×{row['samples_per_prompt']} | "
            f"{float(row['truncated_fraction']):.2%} |"
        )
    auxiliary_pass = [row for row in pass_rows if (row["dataset"], int(row["k"])) in {("aime24", 8), ("gpqa", 4)}]
    if auxiliary_pass:
        lines.extend(
            [
                "",
                "多采样辅助指标："
                + "；".join(
                    f"{row['boundary']} {row['dataset']} pass@{row['k']}={float(row['pass_at_k']):.4f}"
                    for row in auxiliary_pass
                )
                + "。",
            ]
        )
    livecode_rows = [row for row in score_rows if row["dataset"] == "livecodebench_online"]
    lines.extend(
        [
            "",
            "LiveCodeBench / SandboxFusion 审计（infrastructure error 必须为 0）：",
            "",
            "| 边界 | sandbox样本 | infrastructure errors | execution errors | timeouts |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in livecode_rows:
        lines.append(
            f"| `{row['boundary']}` | {row['sandbox_evaluated_samples']} | "
            f"{row['sandbox_infrastructure_errors']} | {row['sandbox_execution_errors']} | "
            f"{row['sandbox_timeouts']} |"
        )

    lines.extend(
        [
            "",
            "## 3. 获得后遗忘、保留率与 BWT",
            "",
            "| 任务 | 获得边界 | 获得分数 | 最终分数 | signed Δ [95% CI] | forgetting | rate | retention |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in forgetting_rows:
        rate = "NA" if row["forgetting_rate"] is None else f"{float(row['forgetting_rate']):.2%}"
        retention = "NA" if row["retention_ratio"] is None else f"{float(row['retention_ratio']):.3f}"
        lines.append(
            f"| {row['domain']} | `{row['acquisition_boundary']}` | "
            f"{float(row['acquisition_score']):.4f} | {float(row['final_score']):.4f} | "
            f"{float(row['signed_final_change']):+.4f} "
            f"[{float(row['signed_final_change_ci95_low']):+.4f}, "
            f"{float(row['signed_final_change_ci95_high']):+.4f}] | "
            f"{float(row['forgetting']):.4f} | {rate} | {retention} |"
        )
    lines.extend(
        [
            "",
            f"原版模型四域 ACC = **{float(continual['pretrained_base_ACC']):.4f}**，最终四域 ACC = "
            f"**{float(continual['final_ACC']):.4f}**（端到端 Δ="
            f"**{float(continual['pretrained_base_to_final_ACC_change']):+.4f}** "
            f"[{float(continual['pretrained_base_to_final_ACC_change_ci95_low']):+.4f}, "
            f"{float(continual['pretrained_base_to_final_ACC_change_ci95_high']):+.4f}]）；"
            "Code/Math/Knowledge 的 classical "
            f"BWT = **{float(continual['classical_BWT']):+.4f}**；平均非负遗忘 = "
            f"**{float(continual['mean_nonnegative_forgetting']):.4f}**。IF 是最后阶段，无法识别 post-IF "
            "forgetting。",
            "",
            f"Code 训练带来的主指标增益为 **{float(continual['code_acquisition_gain']):+.4f}** "
            f"[{float(continual['code_acquisition_gain_ci95_low']):+.4f}, "
            f"{float(continual['code_acquisition_gain_ci95_high']):+.4f}]。以原版 Qwen3-1.7B 为参照，"
            f"Math/Knowledge/IF 的 pretrained-base-referenced FWT = "
            f"**{float(continual['pretrained_base_referenced_FWT']):+.4f}** "
            f"[{float(continual['pretrained_base_referenced_FWT_ci95_low']):+.4f}, "
            f"{float(continual['pretrained_base_referenced_FWT_ci95_high']):+.4f}]；该数值不是以随机初始化模型为"
            "参照的 FWT。",
            "",
            "| 任务 | 原版分数 | 本任务训练前分数 | base-referenced FWT [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in forward_transfer_rows:
        lines.append(
            f"| {row['domain']} | {float(row['reference_score']):.4f} | "
            f"{float(row['pre_acquisition_score']):.4f} | {float(row['forward_transfer']):+.4f} "
            f"[{float(row['paired_bootstrap_ci95_low']):+.4f}, "
            f"{float(row['paired_bootstrap_ci95_high']):+.4f}] |"
        )
    lines.extend(
        [
            "",
            "原版到最终边界的逐域变化："
            + "；".join(
                f"{row['domain']} {float(row['signed_change']):+.4f} "
                f"[{float(row['paired_bootstrap_ci95_low']):+.4f}, "
                f"{float(row['paired_bootstrap_ci95_high']):+.4f}]"
                for row in base_to_final_rows
            )
            + "。",
            "",
            "## 4. 相邻阶段行为干扰矩阵",
            "",
            "正值表示促进，负值表示固定测试集性能干扰。星号表示配对 95% CI 不跨 0。",
            "",
            "| 训练阶段 | Code | Math | Knowledge | IF |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage in DOMAINS:
        values = {}
        for domain in DOMAINS:
            row = next(
                item for item in primary_transfer if item["training_stage"] == stage and item["domain"] == domain
            )
            significant = float(row["paired_bootstrap_ci95_low"]) > 0 or float(row["paired_bootstrap_ci95_high"]) < 0
            values[domain] = f"{float(row['signed_change']):+.4f}{'*' if significant else ''}"
        lines.append(f"| {stage} | {values['code']} | {values['math']} | {values['knowledge']} | {values['if']} |")

    lines.extend(
        [
            "",
            "## 5. 训练信号与优化动力学",
            "",
            "前/后窗口均为 50 updates。mixed group 是 GRPO 仍有组内相对优势的 prompt group。",
            "",
            "| 阶段 | 实际 prompts | reward early→late | mixed early→late | grad norm early→late | clip率 | late截断 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in training_summaries:
        lines.append(
            f"| {row['stage']} | {row['prompts_consumed']} | "
            f"{float(row['early_reward_mean']):.4f}→{float(row['late_reward_mean']):.4f} | "
            f"{float(row['early_informative_group_fraction']):.2%}→"
            f"{float(row['late_informative_group_fraction']):.2%} | "
            f"{float(row['early_grad_norm']):.3f}→{float(row['late_grad_norm']):.3f} | "
            f"{float(row['gradient_clipped_fraction']):.2%} | "
            f"{float(row['late_truncated_fraction']):.2%} |"
        )
    shortfalls = [row for row in training_summaries if int(row["usable_prompt_shortfall"]) > 0]
    if shortfalls:
        lines.extend(
            [
                "",
                "数据审计："
                + "；".join(
                    f"{row['stage']} 配置 {row['configured_train_prompts']} prompts，实际 "
                    f"{row['prompts_consumed']}，短缺 {row['usable_prompt_shortfall']}（runtime prompt-length filter）"
                    for row in shortfalls
                )
                + "。",
            ]
        )
    missing_completion_markers = [row for row in training_summaries if not row["completion_marker_present"]]
    if missing_completion_markers:
        lines.extend(
            [
                "",
                "来源状态审计："
                + "；".join(
                    f"{row['stage']} 的原始 run manifest 状态为 {row['source_run_manifest_status']} 且缺少 "
                    "run_complete.json，但最终 checkpoint 及 rollout/train/geometry 各 "
                    f"{row['updates']} 条锁定记录均已通过完整性校验"
                    for row in missing_completion_markers
                )
                + "。报告保留该 provenance 限制，不把缺失的完成标记改写为已完成。",
            ]
        )
    lines.extend(
        [
            "",
            "后 50 updates 的 BF16 实现审计：",
            "",
            "| 阶段 | 每步实际变化坐标 | intended≤half-ULP | intended能量被清零 | intended–realized cosine |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in training_summaries:
        lines.append(
            f"| {row['stage']} | {float(row['late_model_change_fraction']):.3%} | "
            f"{float(row['late_intended_below_half_ulp_fraction']):.3%} | "
            f"{float(row['late_intended_energy_zeroed_fraction']):.3%} | "
            f"{float(row['late_cos_intended_realized_update']):.4f} |"
        )

    if parameter_stage_rows:
        lines.extend(
            [
                "",
                "## 6. 精确全参数更新几何与稀疏度",
                "",
                "| 阶段 | ‖Δθ‖₂ | ‖Δθ‖/‖θ₀‖ | exact-zero | ≤1e-5 | BF16-aware | relative τ=1e-3 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in parameter_stage_rows:
            lines.append(
                f"| {row['stage']} | {float(row['update_norm']):.6f} | "
                f"{float(row['relative_update_norm']):.3e} | {float(row['exact_zero_sparsity']):.2%} | "
                f"{float(row['visible_sparsity@1e-5']):.2%} | "
                f"{float(row['bf16_aware_sparsity_eta1e-3']):.2%} | "
                f"{float(row['relative_sparsity_tau1e-3']):.2%} |"
            )
        lines.extend(
            [
                "",
                "| 阶段对 | cosine | dot | Jaccard@1e-5 | overlap lift | 交集反号率 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in parameter_pair_rows:
            lift = 0.5 * (
                float(row["left_to_right_support_overlap_lift@1e-5"])
                + float(row["right_to_left_support_overlap_lift@1e-5"])
            )
            lines.append(
                f"| {row['left_stage']}–{row['right_stage']} | {float(row['cosine']):+.6f} | "
                f"{float(row['dot']):+.3e} | {float(row['support_jaccard@1e-5']):.3f} | "
                f"{lift:.2f}× | {float(row['support_sign_conflict_fraction@1e-5']):.2%} |"
            )
        lines.extend(
            [
                "",
                "近零全局 cosine 并不表示任务使用互不相交的参数：支持 overlap lift 显著大于 1，说明可见更新"
                "坐标被高度复用；约一半交集坐标反号，使全局方向发生抵消。",
            ]
        )

    if spectral_summary is not None:
        aggregates = [row for row in spectral_summary["aggregates"] if int(row["stage_update"]) == 300]
        lines.extend(
            [
                "",
                "## 7. 抽样谱几何",
                "",
                "| 阶段 | mean stable rank | mean top-16 energy | mean top-16 weight spectral shift |",
                "|---|---:|---:|---:|",
            ]
        )
        for stage in DOMAINS:
            rows = [row for row in aggregates if row["stage"] == stage]
            stable = float(np.mean([float(row["mean/approx_stable_rank"]) for row in rows]))
            energy = float(np.mean([float(row["mean/approx_top16_energy_fraction"]) for row in rows]))
            shift = float(np.mean([float(row["mean/topk_weight_spectral_shift"]) for row in rows]))
            lines.append(f"| {stage} | {stable:.2f} | {energy:.2%} | {shift:.3e} |")
        spot = spectral_summary["randomized_svd"]["exact_spot_check"]
        lines.extend(
            [
                "",
                f"randomized-SVD 精确锚点：σ₁ 相对误差 {float(spot['sigma1_relative_error']):.3e}，"
                f"top-16 能量相对误差 {float(spot['top16_energy_relative_error']):.3%}。"
                "top-16 只解释小部分更新能量，未观察到低秩塌缩。",
            ]
        )

    lines.extend(["", "## 8. 同-checkpoint 原始梯度干扰", ""])
    if raw_gradient_summary is None:
        lines.append(
            "原始训练未采集跨任务 raw gradients；补充固定 probe 尚未完成。当前 checkpoint-delta cosine 不能"
            "解释成草稿公式中的局部梯度 inner product。"
        )
    else:
        design = raw_gradient_summary["probe_design"]
        samples_per_prompt = int(design["actual_batch_size_per_update"]) // int(design["probe_prompt_count"])
        lines.append(
            f"固定 probe：每任务 {int(design['probe_prompt_count'])} prompts，"
            f"每 prompt {samples_per_prompt} 个样本、实际 batch "
            f"{int(design['actual_batch_size_per_update'])} responses，"
            f"执行 {int(design['probe_updates'])} 次 backward；未执行 optimizer step。"
            "五个 checkpoint 锚点的 probe 配置与 prompt-data 哈希逐任务一致。"
        )
        lines.extend(
            [
                "",
                "原始梯度范数：",
                "",
                "| 边界 | Code | Math | Knowledge | IF |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        raw_anchors = list(dict.fromkeys(str(row["anchor"]) for row in raw_gradient_summary["gradient_norms"]))
        for anchor in raw_anchors:
            norms = {
                row["task"]: float(row["gradient_norm"])
                for row in raw_gradient_summary["gradient_norms"]
                if row["anchor"] == anchor
            }
            lines.append(
                f"| `{anchor}` | {norms['code']:.6f} | {norms['math']:.6f} | "
                f"{norms['knowledge']:.6f} | {norms['if']:.6f} |"
            )

        lines.extend(
            [
                "",
                "Probe rollout 质量（仅用于解释梯度估计，不是固定测试分数）：",
                "",
                "| 边界 | 任务 | reward | informative groups | mean长度 | p95长度 | 截断率 | effective tokens |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in raw_gradient_summary["probe_quality"]:
            lines.append(
                f"| `{row['anchor']}` | {row['task']} | {float(row['reward_mean']):.4f} | "
                f"{int(row['informative_group_count'])}/{int(design['probe_prompt_count'])} | "
                f"{float(row['response_length_mean']):.1f} | "
                f"{float(row['response_length_p95']):.1f} | "
                f"{float(row['truncated_fraction']):.2%} | "
                f"{int(row['effective_token_count_total']):,} |"
            )

        lines.extend(
            [
                "",
                "任务对干涉：",
                "",
                "| 边界 | 任务对 | dot | cosine | 冲突 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in raw_gradient_summary["pairwise_interference"]:
            cosine = "n/a" if row["cosine"] is None else f"{float(row['cosine']):+.4f}"
            conflict = (
                "未定义" if row["conflicting_direction"] is None else ("是" if row["conflicting_direction"] else "否")
            )
            lines.append(
                f"| {row['anchor']} | {row['left_task']}–{row['right_task']} | "
                f"{float(row['dot']):+.3e} | {cosine} | {conflict} |"
            )
        pairwise = raw_gradient_summary["pairwise_interference"]
        pair_changes = raw_gradient_summary["pairwise_changes"]
        defined_pairs = [row for row in pairwise if row["cosine"] is not None]
        defined_changes = [row for row in pair_changes if row["cosine_change"] is not None]
        max_abs_cosine = max((abs(float(row["cosine"])) for row in defined_pairs), default=None)
        max_abs_cosine_text = "n/a" if max_abs_cosine is None else f"{max_abs_cosine:.4f}"
        sign_changes = sum(bool(row["conflict_status_changed"]) for row in defined_changes)
        pair_lookup = {(row["anchor"], row["left_task"], row["right_task"]): row for row in pairwise}
        behavior_lookup = {(row["training_stage"], row["domain"]): row for row in primary_transfer}
        before_knowledge = pair_lookup[("01_after_math", "math", "knowledge")]
        math_after_knowledge = behavior_lookup[("knowledge", "math")]
        before_if = pair_lookup[("02_after_knowledge", "knowledge", "if")]
        knowledge_after_if = behavior_lookup[("if", "knowledge")]
        before_code = pair_lookup[("pre_code", "code", "math")]
        code_after_training = behavior_lookup[("code", "code")]
        math_after_code = behavior_lookup[("code", "math")]
        code_quality = [row for row in raw_gradient_summary["probe_quality"] if row["task"] == "code"]
        pre_code_quality = {
            row["task"]: row for row in raw_gradient_summary["probe_quality"] if row["anchor"] == "pre_code"
        }
        origin_quality = {
            row["task"]: row for row in raw_gradient_summary["probe_quality"] if row["anchor"] == "00_code_origin"
        }
        after_math_quality = {
            row["task"]: row for row in raw_gradient_summary["probe_quality"] if row["anchor"] == "01_after_math"
        }
        after_if_quality = {
            row["task"]: row for row in raw_gradient_summary["probe_quality"] if row["anchor"] == "03_after_if"
        }
        pre_code_norms = {
            row["task"]: float(row["gradient_norm"])
            for row in raw_gradient_summary["gradient_norms"]
            if row["anchor"] == "pre_code"
        }
        if before_code["cosine"] is None:
            code_math_geometry = f"Code 梯度范数为 {pre_code_norms['code']:.1f}，因此 Code–Math cosine 未定义"
        else:
            code_math_geometry = f"Code–Math cosine={float(before_code['cosine']):+.4f}"
        before_knowledge_cosine = (
            "未定义" if before_knowledge["cosine"] is None else f"{float(before_knowledge['cosine']):+.4f}"
        )
        before_if_cosine = "未定义" if before_if["cosine"] is None else f"{float(before_if['cosine']):+.4f}"
        lines.extend(
            [
                "",
                f"总体上 {len(defined_pairs)}/{len(pairwise)} 个方向已定义的“锚点×任务对”cosine 最大绝对值"
                f"为 {max_abs_cosine_text}；{len(defined_changes)} 个方向均已定义的相邻锚点比较中有 "
                f"{sign_changes} 个发生冲突符号翻转。零范数梯度对应的 cosine 记为未定义，不把它伪装成正交。"
                "因此这组局部梯度呈动态近正交，而不是稳定的固定任务冲突图。",
                "",
                f"Code 训练前，{int(pre_code_quality['code']['zero_std_group_count'])}/"
                f"{int(design['probe_prompt_count'])} 个 Code probe group 的组内奖励方差为零，"
                f"{code_math_geometry}；随后 "
                f"MATH500 变化 {float(math_after_code['signed_change']):+.4f} "
                f"[{float(math_after_code['paired_bootstrap_ci95_low']):+.4f}, "
                f"{float(math_after_code['paired_bootstrap_ci95_high']):+.4f}]。Code 自身的 LiveCodeBench "
                f"acquisition gain 为 {float(code_after_training['signed_change']):+.4f}。这把原版模型处的局部梯度"
                "基线与 Code 阶段的实际跨任务行为变化纳入了同一审计链。",
                "",
                f"Knowledge 训练前，Math–Knowledge cosine={before_knowledge_cosine}；"
                f"随后 MATH500 变化 {float(math_after_knowledge['signed_change']):+.4f} "
                f"[{float(math_after_knowledge['paired_bootstrap_ci95_low']):+.4f}, "
                f"{float(math_after_knowledge['paired_bootstrap_ci95_high']):+.4f}]。方向一致，但区间跨 0，"
                "不能据此宣称显著遗忘。",
                "",
                f"IF 训练前，Knowledge–IF cosine={before_if_cosine}、"
                f"dot={float(before_if['dot']):+.3e}，局部一阶上是兼容的；但 300 updates 后 GPQA "
                f"变化 {float(knowledge_after_if['signed_change']):+.4f} "
                f"[{float(knowledge_after_if['paired_bootstrap_ci95_low']):+.4f}, "
                f"{float(knowledge_after_if['paired_bootstrap_ci95_high']):+.4f}]。"
                "这直接说明单 batch 局部 inner product 不能外推完整 Adam 轨迹；梯度非平稳、预条件、"
                "IF 阶段高频 clipping 与 BF16 实现都会改变长期路径。",
                "",
                f"Code probe 在五个锚点中最少只有 "
                f"{min(int(row['informative_group_count']) for row in code_quality)}/"
                f"{int(design['probe_prompt_count'])} 个非零组内方差 group，其方向尤其低信噪。"
                "五个 Code probe 的 SandboxFusion infrastructure error 均为 0；精确 reduction 只消除了"
                "数值归约误差，并未消除 on-policy response sampling 方差。",
                "",
                f"响应分布本身也明显漂移：Code 训练后 Code/Math 的平均响应长度分别变为原版模型的 "
                f"{float(origin_quality['code']['response_length_mean']) / float(pre_code_quality['code']['response_length_mean']):.2f}×/"
                f"{float(origin_quality['math']['response_length_mean']) / float(pre_code_quality['math']['response_length_mean']):.2f}×；"
                f"Math 阶段后 Code/Math 又分别变为 Code 起点的 "
                f"{float(after_math_quality['code']['response_length_mean']) / float(origin_quality['code']['response_length_mean']):.2f}×/"
                f"{float(after_math_quality['math']['response_length_mean']) / float(origin_quality['math']['response_length_mean']):.2f}×；"
                f"IF 训练后 IF 响应长度降至起点的 "
                f"{float(after_if_quality['if']['response_length_mean']) / float(origin_quality['if']['response_length_mean']):.2f}×。"
                "这也是同 prompt 梯度估计随 checkpoint 改变的一部分。",
            ]
        )

    training_by_stage = {row["stage"]: row for row in training_summaries}
    support_lifts = [
        0.5
        * (
            float(row["left_to_right_support_overlap_lift@1e-5"])
            + float(row["right_to_left_support_overlap_lift@1e-5"])
        )
        for row in parameter_pair_rows
    ]
    support_lift_range = f"{min(support_lifts):.1f}–{max(support_lifts):.1f}×" if support_lifts else "未测量"
    final_spectral_rows = (
        [row for row in spectral_summary["aggregates"] if int(row["stage_update"]) == 300]
        if spectral_summary is not None
        else []
    )
    top16_energies = [float(row["mean/approx_top16_energy_fraction"]) for row in final_spectral_rows]
    top16_energy_range = f"{min(top16_energies):.0%}–{max(top16_energies):.0%}" if top16_energies else "未测量"
    lines.extend(
        [
            "",
            "## 9. 对当前草稿命题的证据强度",
            "",
            "- 草稿式 (1) 把同-checkpoint 梯度内积定义为局部一阶干涉；本报告的固定 probe 正面测量了"
            "这个量。它支持“干涉是局部量”的表述，也以 Knowledge–IF 的正局部内积和后续 GPQA 负变化"
            "说明：不能把该式直接解释成长程遗忘预测器。",
            f"- 草稿式 (8) 预言任务接近解决时 informative GRPO group 会变稀。本实验中 Code 为 "
            f"{float(training_by_stage['code']['early_informative_group_fraction']):.2%}→"
            f"{float(training_by_stage['code']['late_informative_group_fraction']):.2%}，Math 为 "
            f"{float(training_by_stage['math']['early_informative_group_fraction']):.2%}→"
            f"{float(training_by_stage['math']['late_informative_group_fraction']):.2%}，Knowledge 为 "
            f"{float(training_by_stage['knowledge']['early_informative_group_fraction']):.2%}→"
            f"{float(training_by_stage['knowledge']['late_informative_group_fraction']):.2%}，IF 为 "
            f"{float(training_by_stage['if']['early_informative_group_fraction']):.2%}→"
            f"{float(training_by_stage['if']['late_informative_group_fraction']):.2%}。"
            "这些阶段趋势并不一致；这不反驳渐近命题，但说明 300-update、未完全解决的 regime 不支持把它"
            "写成普遍单调的经验结论。",
            "- 这批数据只有 GRPO/AdamW，没有同 checkpoint、同数据预算的 PPO 或 OPD 对照，因而不能"
            "填充草稿摘要中的 PPO–GRPO–OPD 排序或“低方差 consolidation”主结果；它适合作为 GRPO "
            "case study 和测量协议验证。",
            "",
            "## 10. 结论边界与参考",
            "",
            "- 行为矩阵回答是否遗忘；同-checkpoint 梯度 dot 回答局部一阶冲突；checkpoint delta 回答实际参数"
            "路径。三者相关但不可互相替代。",
            "- 当前只有 seed 42 的一条序列；bootstrap 只量化固定测试 prompt 的抽样不确定性，不覆盖训练 seed"
            "方差，也不支持三点相关性的显著性推断。",
            "- optimizer 在每个阶段重置；结论不外推到保留 Adam moments 的 continual training。",
            "- 与 [SFT Conflicts, RL Coexists](https://arxiv.org/abs/2608.03573) 的近正交兼容图景一致，"
            f"本实验的全局梯度和阶段位移 cosine 都接近 0；但 {support_lift_range} support-overlap lift 表明"
            "近正交不等于使用互不相交的坐标。",
            "- 相比 [On the Geometry of On-Policy Distillation](https://arxiv.org/abs/2606.07082) 所讨论的"
            "早期子空间锁定，本实验在 100→200→300 updates 的抽样子空间重合是渐进增强，不能声称"
            "立即锁定。",
            "- [Dense Supervision, Sparse Updates](https://arxiv.org/abs/2606.13657) 强调更新稀疏性；"
            f"这里 GRPO 的坐标更新同样高度稀疏，但 top-16 能量仅约 {top16_energy_range}，未见低秩塌缩。由于没有"
            "同模型同数据的 SFT/OPD 对照，这些只能作机制对照，不能作方法优劣结论。",
        ]
    )
    (output_dir / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    sequence_root = args.sequence_root.resolve()
    eval_root = (args.eval_root or sequence_root / "evaluations_v2").resolve()
    output_dir = args.output_dir.resolve()
    known_outputs = (
        "summary.json",
        "performance_matrix.csv",
        "forgetting_summary.csv",
        "training_dynamics.png",
        "report.md",
        "report_zh.md",
    )
    if output_dir.exists() and not args.force and any((output_dir / name).exists() for name in known_outputs):
        raise FileExistsError(f"Analysis outputs already exist in {output_dir}; pass --force to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_code_checkpoint = resolve_torch_dist_checkpoint(args.pre_code_checkpoint)
    manifest, stages = load_sequence(sequence_root, pre_code_checkpoint)
    rollout_rows, train_rows, geometry_rows, training_summaries, training_provenance = parse_training(
        stages, args.summary_window
    )
    configured_prompts = int(manifest["new_stage_training"]["train_prompts_per_task"])
    for row in training_summaries:
        consumed = int(row["prompts_consumed"])
        row["configured_train_prompts"] = configured_prompts
        row["usable_prompt_shortfall"] = configured_prompts - consumed
        row["usable_prompt_shortfall_reason"] = (
            "runtime_prompt_length_filtering" if consumed < configured_prompts else None
        )
    boundaries, evaluations, evaluation_provenance = load_evaluations(
        sequence_root, eval_root, manifest, pre_code_checkpoint
    )
    score_rows, pass_rows = summarize_evaluations(boundaries, evaluations, args.bootstrap_samples, args.bootstrap_seed)
    (
        adjacent_rows,
        primary_transfer,
        forgetting_rows,
        forward_transfer_rows,
        base_to_final_rows,
        continual,
    ) = transfer_and_forgetting(boundaries, evaluations, args.bootstrap_samples, args.bootstrap_seed)

    parameter_path = args.parameter_geometry
    if parameter_path is None:
        candidates = (sequence_root / "parameter_geometry_v3" / "summary.json",)
        parameter_path = next((path for path in candidates if path.is_file()), None)
    elif not parameter_path.is_absolute():
        parameter_path = (Path.cwd() / parameter_path).resolve()
    parameter_stage_rows, parameter_pair_rows, alignment_rows, parameter_summary = load_parameter_geometry(
        parameter_path, primary_transfer
    )
    spectral_path = args.spectral_geometry
    if spectral_path is None:
        spectral_candidates = (sequence_root / "spectral_geometry_v3" / "summary.json",)
        spectral_path = next((path for path in spectral_candidates if path.is_file()), None)
    elif not spectral_path.is_absolute():
        spectral_path = (Path.cwd() / spectral_path).resolve()
    spectral_summary = read_json(spectral_path) if spectral_path is not None else None
    if spectral_summary is not None:
        spectral_stages = tuple(str(stage) for stage in spectral_summary.get("stages", []))
        if spectral_stages != DOMAINS:
            raise ValueError(f"Spectral geometry stages are {list(spectral_stages)}, expected {list(DOMAINS)}.")
    raw_gradient_path = args.raw_gradient_interference
    if raw_gradient_path is None:
        candidate = sequence_root / "gradient_probes_v2" / "summary" / "summary.json"
        raw_gradient_path = candidate if candidate.is_file() else None
    elif not raw_gradient_path.is_absolute():
        raw_gradient_path = (Path.cwd() / raw_gradient_path).resolve()
    raw_gradient_summary = read_json(raw_gradient_path) if raw_gradient_path is not None else None
    if raw_gradient_summary is not None:
        measured_anchors = {str(row["anchor"]) for row in raw_gradient_summary.get("pairwise_interference", [])}
        required_anchors = {dirname for _, dirname, _ in BOUNDARIES}
        if measured_anchors != required_anchors:
            if args.raw_gradient_interference is not None:
                raise ValueError(
                    f"Raw-gradient anchors are {sorted(measured_anchors)}, expected {sorted(required_anchors)}."
                )
            raw_gradient_path = None
            raw_gradient_summary = None

    domain_matrix_rows = []
    for boundary in boundaries:
        row: dict[str, Any] = {
            "boundary": boundary.label,
            "boundary_index": boundary.index,
            "trained_task": boundary.trained_task,
        }
        for domain in DOMAINS:
            dataset = PRIMARY_DATASET[domain]
            row[domain] = next(
                float(item["mean_reward"])
                for item in score_rows
                if item["boundary"] == boundary.label and item["dataset"] == dataset
            )
        domain_matrix_rows.append(row)

    stage_impact_rows = []
    for stage in DOMAINS:
        rows = [row for row in primary_transfer if row["training_stage"] == stage]
        prior = [float(row["signed_change"]) for row in rows if row["is_previously_trained_task"]]
        future = [float(row["signed_change"]) for row in rows if row["is_future_task"]]
        own = next(float(row["signed_change"]) for row in rows if row["is_own_task"])
        stage_impact_rows.append(
            {
                "training_stage": stage,
                "own_task_change": own,
                "prior_task_mean_change": float(np.mean(prior)) if prior else None,
                "prior_task_mean_interference_loss": float(np.mean([max(0.0, -v) for v in prior])) if prior else None,
                "future_task_mean_change": float(np.mean(future)) if future else None,
                "interpretation": "descriptive adjacent-boundary behavior; future-task changes are not classical FWT",
            }
        )

    outputs = {
        "rollout_curves.csv": rollout_rows,
        "train_curves.csv": train_rows,
        "optimizer_geometry_curves.csv": geometry_rows,
        "training_summary.csv": training_summaries,
        "performance_matrix.csv": score_rows,
        "primary_domain_matrix.csv": domain_matrix_rows,
        "pass_at_k.csv": pass_rows,
        "adjacent_dataset_transfer.csv": adjacent_rows,
        "primary_task_transfer_matrix.csv": primary_transfer,
        "stage_impact_summary.csv": stage_impact_rows,
        "forgetting_summary.csv": forgetting_rows,
        "pretrained_base_referenced_forward_transfer.csv": forward_transfer_rows,
        "pretrained_base_to_final.csv": base_to_final_rows,
        "parameter_stage_summary.csv": parameter_stage_rows,
        "parameter_update_pairs.csv": parameter_pair_rows,
        "behavior_parameter_alignment.csv": alignment_rows,
    }
    if raw_gradient_summary is not None:
        outputs.update(
            {
                "same_checkpoint_raw_gradient_norms.csv": raw_gradient_summary["gradient_norms"],
                "same_checkpoint_raw_gradient_probe_quality.csv": raw_gradient_summary["probe_quality"],
                "same_checkpoint_raw_gradient_interference.csv": raw_gradient_summary["pairwise_interference"],
                "same_checkpoint_raw_gradient_changes.csv": raw_gradient_summary["pairwise_changes"],
            }
        )
    for filename, rows in outputs.items():
        write_csv(output_dir / filename, rows)

    summary = {
        "schema_version": 2,
        "analysis_policy": {
            "primary_domain_metrics": PRIMARY_DATASET,
            "score": "mean task reward over fixed test samples",
            "uncertainty": (
                f"paired or one-sample prompt-cluster bootstrap, {args.bootstrap_samples} replicates, "
                f"base seed {args.bootstrap_seed}"
            ),
            "forgetting": "max(0, acquisition_score - final_score)",
            "forgetting_rate": "forgetting / acquisition_score when acquisition_score > 0",
            "retention": "final_score / acquisition_score when acquisition_score > 0",
            "classical_BWT": "mean(final_score - acquisition_score) over Code, Math, and Knowledge",
            "pretrained_base_referenced_FWT": (
                "mean(pre-acquisition score - original pretrained Qwen3-1.7B score) over Math, Knowledge, and IF"
            ),
            "behavioral_interference": "negative adjacent-boundary fixed-test score change",
            "parameter_interference_limit": (
                "Stage-local checkpoint-delta dot/cosine is descriptive; it is not a same-checkpoint raw-gradient dot."
            ),
        },
        "sequence_root": str(sequence_root),
        "eval_root": str(eval_root),
        "pre_code_checkpoint": str(pre_code_checkpoint),
        "pre_code_checkpoint_metadata_sha256": sha256(pre_code_checkpoint / ".metadata"),
        "pre_code_checkpoint_common_sha256": sha256(pre_code_checkpoint / "common.pt"),
        "sequence_manifest_sha256": sha256(sequence_root / "sequence_manifest.json"),
        "stage_boundaries_sha256": sha256(sequence_root / "stage_boundaries.jsonl"),
        "training_provenance": training_provenance,
        "evaluation_provenance": evaluation_provenance,
        "training_summary": training_summaries,
        "primary_domain_matrix": domain_matrix_rows,
        "stage_impact_summary": stage_impact_rows,
        "forgetting_summary": forgetting_rows,
        "pretrained_base_referenced_forward_transfer": forward_transfer_rows,
        "pretrained_base_to_final": base_to_final_rows,
        "continual_learning_summary": continual,
        "parameter_geometry": {
            "path": str(parameter_path) if parameter_path is not None else None,
            "sha256": sha256(parameter_path) if parameter_path is not None else None,
            "method": parameter_summary.get("method") if parameter_summary is not None else None,
            "stage_summary": parameter_stage_rows,
            "stage_pairs": parameter_pair_rows,
        },
        "spectral_geometry": {
            "path": str(spectral_path) if spectral_path is not None else None,
            "sha256": sha256(spectral_path) if spectral_path is not None else None,
            "method": spectral_summary.get("method") if spectral_summary is not None else None,
            "randomized_svd": (spectral_summary.get("randomized_svd") if spectral_summary is not None else None),
            "aggregates": spectral_summary.get("aggregates") if spectral_summary is not None else None,
        },
        "same_checkpoint_raw_gradient_interference": {
            "path": str(raw_gradient_path) if raw_gradient_path is not None else None,
            "sha256": sha256(raw_gradient_path) if raw_gradient_path is not None else None,
            "method": raw_gradient_summary.get("method") if raw_gradient_summary is not None else None,
            "probe_design": (raw_gradient_summary.get("probe_design") if raw_gradient_summary is not None else None),
            "fixed_probe_inputs": (
                raw_gradient_summary.get("fixed_probe_inputs") if raw_gradient_summary is not None else None
            ),
            "pairwise_interference": (
                raw_gradient_summary.get("pairwise_interference") if raw_gradient_summary is not None else None
            ),
            "pairwise_changes": (
                raw_gradient_summary.get("pairwise_changes") if raw_gradient_summary is not None else None
            ),
            "gradient_norms": (
                raw_gradient_summary.get("gradient_norms") if raw_gradient_summary is not None else None
            ),
            "probe_quality": (raw_gradient_summary.get("probe_quality") if raw_gradient_summary is not None else None),
        },
        "metric_availability": {
            "test_performance": "available",
            "paired_behavioral_interference": "available",
            "acquisition_to_final_forgetting": "available_for_Code_Math_Knowledge",
            "IF_post_training_forgetting": "unavailable_final_stage_has_no_later_boundary",
            "pretrained_base_referenced_FWT": "available",
            "random_initialization_referenced_FWT": "not_applicable_pretrained_model_sequence",
            "stage_checkpoint_delta_geometry": "available" if parameter_summary is not None else "missing",
            "sampled_spectral_trajectory_geometry": (
                "available_approximate" if spectral_summary is not None else "missing"
            ),
            "same_checkpoint_raw_gradient_interference": (
                "available_supplemental_fixed_probe"
                if raw_gradient_summary is not None
                else "not_collected_by_the_original_sequence; requires_separate_fixed_probe_backward_runs"
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_report(
        output_dir,
        score_rows,
        forgetting_rows,
        forward_transfer_rows,
        continual,
        parameter_pair_rows,
        raw_gradient_summary,
    )
    build_chinese_report(
        output_dir,
        training_summaries,
        score_rows,
        pass_rows,
        primary_transfer,
        forgetting_rows,
        forward_transfer_rows,
        base_to_final_rows,
        continual,
        parameter_stage_rows,
        parameter_pair_rows,
        spectral_summary,
        raw_gradient_summary,
    )
    if not args.no_plots:
        save_plots(
            output_dir,
            rollout_rows,
            train_rows,
            geometry_rows,
            score_rows,
            primary_transfer,
            forgetting_rows,
            parameter_stage_rows,
            parameter_pair_rows,
            args.smoothing_window,
        )
    print(f"Wrote sequential experiment analysis to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--pre-code-checkpoint", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--parameter-geometry", type=Path)
    parser.add_argument("--spectral-geometry", type=Path)
    parser.add_argument("--raw-gradient-interference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-window", type=int, default=50)
    parser.add_argument("--smoothing-window", type=int, default=21)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.summary_window <= 0 or args.smoothing_window <= 0 or args.bootstrap_samples <= 0:
        parser.error("Window sizes and bootstrap samples must be positive.")
    return args


if __name__ == "__main__":
    analyze(parse_args())
