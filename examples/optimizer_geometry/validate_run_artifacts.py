#!/usr/bin/env python3
"""Fail if a completed optimizer experiment is missing paper-critical artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def command_value(command: list[str], flag: str, default: Any = None) -> Any:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def inspect_jsonl(path: Path) -> tuple[int, str, dict[str, Any] | None]:
    digest = hashlib.sha256()
    count = 0
    first = None
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            if first is None:
                first = record
            count += 1
    return count, digest.hexdigest(), first


def recorded_path(run: Path, absolute: Any, relative: Any = None) -> Path:
    path = Path(str(absolute or ""))
    if path.is_file():
        return path
    return run / str(relative or absolute or "")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run_dir.resolve()
    errors = []
    marker_path = run / "run_complete.json"
    manifest_path = run / "provenance" / "run_manifest.json"
    marker = json_file(marker_path) if marker_path.is_file() else {}
    manifest = json_file(manifest_path) if manifest_path.is_file() else {}
    if (run / "run_failed.json").exists():
        errors.append("run_failed.json exists for a purportedly completed run")
    if marker.get("status") != "complete":
        errors.append("missing valid run_complete.json")
    if manifest.get("status") != "complete":
        errors.append("provenance/run_manifest.json is absent or not complete")
    if int(manifest.get("schema_version", 1)) >= 2:
        snapshot = manifest.get("source_snapshot") or {}
        snapshot_path = recorded_path(run, snapshot.get("path"), snapshot.get("run_relative_path"))
        if not snapshot_path.is_file():
            errors.append("source snapshot archive is missing")
        elif hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != snapshot.get("sha256"):
            errors.append("source snapshot SHA-256 mismatch")
        for record in manifest.get("inputs") or []:
            if not record.get("exists") or "sha256" not in record:
                continue
            archived = record.get("archived_path")
            relative = record.get("archived_run_relative_path")
            if archived is None:
                if not record.get("archive_skipped_reason"):
                    errors.append(f"input has neither archive nor skip reason: {record.get('path')}")
                continue
            archived_path = recorded_path(run, archived, relative)
            if not archived_path.is_file():
                errors.append(f"archived input is missing: {record.get('path')}")
                continue
            digest = hashlib.sha256(archived_path.read_bytes()).hexdigest()
            if digest != record.get("sha256") or digest != record.get("archived_sha256"):
                errors.append(f"archived input SHA-256 mismatch: {record.get('path')}")
    expected = args.expected_updates
    if expected is None:
        expected = marker.get("final_num_updates")
    if expected is None:
        errors.append("expected update count is unknown")
        expected = 0
    expected = int(expected)
    if marker and int(marker.get("final_num_updates", -1)) != expected:
        errors.append(f"completion marker update count != {expected}")

    command = manifest.get("command") or []
    checkpoint_expected = "--save" in command
    checkpoint_pointer = run / "checkpoints" / "latest_checkpointed_iteration.txt"
    if checkpoint_expected and not checkpoint_pointer.is_file():
        errors.append("final actor checkpoint pointer is missing")
    if checkpoint_expected and command_value(command, "--advantage-estimator") == "ppo":
        critic_save = run / "checkpoints" / "critic" / "latest_checkpointed_iteration.txt"
        if not critic_save.is_file():
            errors.append("final PPO critic checkpoint pointer is missing")
    geometry = jsonl(run / "geometry" / "actor" / "metrics.jsonl")
    geometry_interval = int(command_value(command, "--geometry-interval", 1))
    # Exact scalar geometry is emitted for every attempted update; the interval
    # gates only CountSketch vectors, histograms, support, and sampled matrices.
    successful_geometry = [record for record in geometry if record.get("update_successful") is True]
    expected_geometry = expected
    if len(successful_geometry) != expected_geometry:
        errors.append(
            f"actor geometry has {len(successful_geometry)} successful-update records, expected {expected_geometry}"
        )
    if expected and geometry and int(geometry[-1].get("num_updates", -1)) != expected:
        errors.append("actor geometry does not end at the expected update")
    required_geometry_fields = {
        "schema_version",
        "run_id",
        "seed",
        "task",
        "rollout_id",
        "observation_id",
        "num_updates",
        "model_version",
        "actual_batch_size",
        "effective_token_count",
        "cumulative_prompt_count",
        "cumulative_effective_token_count",
        "model_dtype_parameter_counts",
        "learning_rate",
        "update_successful",
        "valid_update_metrics",
        "low_frequency_observation",
        "groups",
    }
    exact_vector_names = {
        "theta_before",
        "theta_reference",
        "g_raw",
        "g_opt",
        "d_data",
        "d_wd",
        "delta_data_fp32",
        "delta_wd_fp32",
        "delta_intended_fp32",
        "delta_model",
        "displacement",
    }
    required_exact_metrics = {
        "parameter_count",
        "cos_g_raw_g_opt",
        "cos_g_opt_d_data",
        "cos_g_opt_delta_intended_fp32",
        "cos_delta_intended_fp32_delta_model",
        "cos_theta_before_delta_model",
        "cos_delta_model_displacement",
        "dot_g_opt_d_data",
        "dot_g_opt_delta_intended_fp32",
        "dot_g_opt_delta_model",
        "g_raw_to_theta_ratio",
        "d_data_to_g_opt_ratio",
        "delta_wd_to_delta_data_ratio",
        "delta_intended_to_theta_ratio",
        "delta_model_to_theta_ratio",
        "displacement_to_reference_ratio",
        "gradient_directional_step",
        "model_change_fraction",
        "intended_below_half_ulp_fraction",
        "energy_survival",
        "quantization_residual",
        "intended_energy_zeroed_fraction",
        "intended_energy_amplified_fraction",
        "intended_energy_attenuated_fraction",
        "ulp_ratio_bins",
    }
    for vector_name in exact_vector_names:
        required_exact_metrics.update(
            {
                f"{vector_name}_l2",
                f"{vector_name}_rms",
                f"{vector_name}_linf",
                f"{vector_name}_exact_zero_fraction",
            }
        )
    required_success_fields = {
        "actual_optimizer_branches",
        "grad_norm_raw",
        "clip_threshold",
        "clip_scale",
        "optimizer_clip_scale",
        "grad_clipped",
        "run_clip_fraction",
    }
    for index, record in enumerate(geometry):
        missing = required_geometry_fields - set(record)
        if missing:
            errors.append(f"actor geometry record {index} is missing: {sorted(missing)}")
        if record.get("update_successful") is True and not record.get("groups"):
            errors.append(f"successful actor geometry record {index} has no exact groups")
        if record.get("update_successful") is True:
            missing_success = required_success_fields - set(record)
            if missing_success:
                errors.append(f"successful actor geometry record {index} is missing: {sorted(missing_success)}")
            global_metrics = (record.get("groups") or {}).get("global") or {}
            missing_exact = required_exact_metrics - set(global_metrics)
            if missing_exact:
                errors.append(
                    f"successful actor geometry record {index} global group is missing exact metrics: "
                    f"{sorted(missing_exact)}"
                )
            branch_groups = {
                name: values
                for name, values in (record.get("groups") or {}).items()
                if name.startswith("optimizer_branch/")
            }
            if not branch_groups:
                errors.append(f"successful actor geometry record {index} has no optimizer branch group")
            for branch_name, values in branch_groups.items():
                missing_branch = {
                    "parameter_count",
                    "parameter_fraction",
                    "gradient_energy_fraction",
                    "intended_update_energy_fraction",
                    "realized_update_energy_fraction",
                    "weight_decay_metrics_applicability",
                } - set(values)
                if missing_branch:
                    errors.append(
                        f"successful actor geometry record {index} branch {branch_name} is missing: "
                        f"{sorted(missing_branch)}"
                    )
        if record.get("update_successful") is not True and record.get("groups"):
            errors.append(f"failed/skipped actor geometry record {index} contains valid-update groups")
    low_frequency = [record for record in successful_geometry if record.get("low_frequency_observation")]
    if expected and not low_frequency:
        errors.append(f"no low-frequency geometry observation was saved for interval {geometry_interval}")
    exact_references = list((run / "geometry" / "actor" / "exact_reference").glob("rank_*.pt"))
    if expected and not exact_references:
        errors.append("per-rank exact geometry reference tensors are missing")
    support_states = list((run / "geometry" / "actor" / "support_state").glob("rank_*.pt"))
    if low_frequency and not support_states:
        errors.append("low-frequency support-window state is missing")

    rollout_geometry = jsonl(run / "geometry" / "rollout" / "metrics.jsonl")
    if expected and not rollout_geometry:
        errors.append("durable OPD/RL rollout geometry distributions are missing")
    required_rollout_geometry_fields = {
        "schema_version",
        "record_type",
        "run_id",
        "seed",
        "task",
        "rollout_id",
        "num_updates",
        "model_version",
        "actual_batch_size",
        "effective_token_count",
        "cumulative_prompt_count",
        "cumulative_effective_token_count",
        "sampled_reverse_kl_definition",
        "task_reward_observed",
        "reward_used_in_loss",
        "reward_loss_coefficient",
        "loss_components",
        "availability",
        "observation_wall_time_ms",
        "metrics",
    }
    for index, record in enumerate(rollout_geometry):
        missing = required_rollout_geometry_fields - set(record)
        if missing:
            errors.append(f"rollout geometry record {index} is missing: {sorted(missing)}")
    rollout_geometry_ids = [int(record.get("rollout_id", -1)) for record in rollout_geometry]
    if len(rollout_geometry_ids) != len(set(rollout_geometry_ids)):
        errors.append("rollout geometry contains duplicate rollout ids")
    rollout_sample_files = sorted((run / "geometry" / "rollout" / "samples").glob("rollout_*.jsonl"))
    if expected and not rollout_sample_files:
        errors.append("durable per-sample training rollout artifacts are missing")
    required_rollout_sample_fields = {
        "schema_version",
        "run_id",
        "seed",
        "task",
        "rollout_id",
        "num_updates",
        "model_version",
        "prompt_id",
        "sample_id",
        "prompt",
        "response",
        "label",
        "reward",
        "passed",
        "task_reward_observed",
        "reward_used_in_loss",
        "reward_loss_coefficient",
        "status",
        "response_length",
        "effective_response_length",
    }
    rollout_sample_count = 0
    rollout_sample_ids: set[int] = set()
    for sample_path in rollout_sample_files:
        count, _, _ = inspect_jsonl(sample_path)
        rollout_sample_count += count
        if not count:
            errors.append(f"training rollout sample file is empty: {sample_path.name}")
            continue
        file_rollout_ids: set[int] = set()
        with sample_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                missing = required_rollout_sample_fields - set(sample)
                if missing:
                    errors.append(
                        f"training rollout sample schema for {sample_path.name}:{line_number} "
                        f"is missing: {sorted(missing)}"
                    )
                if bool(sample.get("task_reward_observed")) != (sample.get("reward") is not None):
                    errors.append(
                        f"training rollout reward availability is inconsistent in " f"{sample_path.name}:{line_number}"
                    )
                file_rollout_ids.add(int(sample.get("rollout_id", -1)))
        if len(file_rollout_ids) != 1:
            errors.append(f"training rollout sample file mixes rollout ids: {sample_path.name}")
        rollout_sample_ids.update(file_rollout_ids)
    if set(rollout_geometry_ids) != rollout_sample_ids:
        errors.append("rollout geometry ids do not match per-sample rollout artifact ids")

    train_events = jsonl(run / "metrics" / "train.jsonl")
    rollout_events = jsonl(run / "metrics" / "rollout.jsonl")
    if not train_events:
        errors.append("durable train scalar JSONL is missing")
    if not rollout_events:
        errors.append("durable rollout scalar JSONL is missing")
    train_updates = [
        event.get("metrics", {}).get("train/num_updates")
        for event in train_events
        if event.get("metrics", {}).get("train/num_updates") is not None
    ]
    if expected and (not train_updates or max(map(int, train_updates)) != expected):
        errors.append("durable train metrics do not reach the expected update")

    eval_index = jsonl(run / "eval_artifacts" / "index.jsonl")
    if args.require_eval:
        if not eval_index:
            errors.append("per-sample evaluation index is missing")
        final_eval_records = [
            record
            for record in eval_index
            if int(record.get("num_updates", -1)) == expected
            and record.get("eval_phase") in {"final", "post_update", "eval_only"}
        ]
        if not final_eval_records:
            errors.append("no final-checkpoint per-sample evaluation artifact at expected update")
        required_sample_fields = {
            "dataset",
            "num_updates",
            "model_version",
            "eval_phase",
            "prompt_index",
            "sample_within_prompt",
            "prompt",
            "response",
            "reward",
            "status",
            "response_length",
            "metadata_sha256",
        }
        for record in final_eval_records:
            for dataset, details in (record.get("datasets") or {}).items():
                sample_path = recorded_path(run, details.get("path"), details.get("run_relative_path"))
                if not sample_path.is_file():
                    errors.append(f"eval sample file is missing for {dataset}: {sample_path}")
                    continue
                count, digest, first = inspect_jsonl(sample_path)
                if count != int(details.get("samples", -1)):
                    errors.append(f"eval sample count mismatch for {dataset}")
                if digest != details.get("sha256"):
                    errors.append(f"eval sample SHA-256 mismatch for {dataset}")
                missing = required_sample_fields - set(first or {})
                if missing:
                    errors.append(f"eval sample schema for {dataset} is missing: {sorted(missing)}")
        if not jsonl(run / "forgetting" / "metrics.jsonl"):
            errors.append("forgetting/backward-transfer metrics are missing")

    wandb_expected = "--use-wandb" in command
    if wandb_expected:
        if not (run / "wandb_run_id.txt").is_file():
            errors.append("W&B run id file is missing")
        if not any(path.is_file() for path in (run / "wandb").rglob("*")):
            errors.append("W&B local cache is empty")
        if marker.get("wandb_run_id") and (run / "wandb_run_id.txt").is_file():
            persisted_id = (run / "wandb_run_id.txt").read_text(encoding="utf-8").strip()
            if persisted_id != marker.get("wandb_run_id"):
                errors.append("W&B run id does not match completion marker")

    report = {
        "schema_version": 1,
        "run_dir": str(run),
        "valid": not errors,
        "errors": errors,
        "expected_updates": expected,
        "counts": {
            "geometry": len(geometry),
            "successful_geometry": len(successful_geometry),
            "low_frequency_geometry": len(low_frequency),
            "rollout_geometry": len(rollout_geometry),
            "rollout_sample_files": len(rollout_sample_files),
            "rollout_samples": rollout_sample_count,
            "train_scalar_events": len(train_events),
            "rollout_scalar_events": len(rollout_events),
            "eval_events": len(eval_index),
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-updates", type=int)
    parser.add_argument("--require-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = validate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)
