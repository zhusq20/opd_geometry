"""Evaluation logger that persists per-task forgetting and backward transfer."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "numel") and value.numel() == 1:
        value = value.detach().cpu().item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _evaluation_key(record: dict[str, Any]) -> str:
    return (
        f"{int(record.get('num_updates', -1))}:"
        f"{int(record.get('model_version', -1))}:"
        f"{record.get('eval_phase', 'unspecified')}"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid forgetting metric at {path}:{line_number}: {error}") from error
    return records


def _reconcile_metrics_with_state(
    output_dir: Path,
    state: dict[str, Any],
    metrics_path: Path,
) -> list[dict[str, Any]]:
    """Drop only JSONL records that were never committed to state.json."""

    records = _read_jsonl(metrics_path)
    performance_matrix = state.get("performance_matrix", [])
    committed_count = len(performance_matrix)
    if len(records) < committed_count:
        raise ValueError(
            f"Forgetting state contains {committed_count} committed evaluations but {metrics_path} "
            f"contains only {len(records)} records; refusing an unsafe automatic recovery."
        )
    for index, performance_row in enumerate(performance_matrix):
        if records[index].get("performance_matrix_row") != performance_row:
            raise ValueError(
                f"Forgetting state and metrics diverge at committed evaluation {index}; "
                "refusing an unsafe automatic recovery."
            )

    stored_key = state.get("last_evaluation_key")
    if committed_count:
        committed_key = _evaluation_key(records[committed_count - 1])
        if stored_key is not None and stored_key != committed_key:
            raise ValueError(
                f"Forgetting state ends at {stored_key}, but the committed metrics prefix ends at {committed_key}."
            )
    elif stored_key is not None:
        raise ValueError(f"Forgetting state names {stored_key} as committed but has an empty performance matrix.")

    if len(records) == committed_count:
        return records

    committed_records = records[:committed_count]
    uncommitted_records = records[committed_count:]
    recovery_dir = output_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    first_key = _evaluation_key(uncommitted_records[0]).replace(":", "_")
    archive_path = recovery_dir / f"uncommitted_metrics.{first_key}.{os.getpid()}.jsonl"
    suffix = 1
    while archive_path.exists():
        archive_path = recovery_dir / f"uncommitted_metrics.{first_key}.{os.getpid()}.{suffix}.jsonl"
        suffix += 1
    _atomic_jsonl(archive_path, uncommitted_records)
    _atomic_jsonl(metrics_path, committed_records)
    logger.warning(
        "Archived %d uncommitted forgetting metric record(s) to %s and restored the state.json commit boundary.",
        len(uncommitted_records),
        archive_path,
    )
    return committed_records


def log_eval_and_forgetting(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None = None,
) -> bool:
    """Append task curves while allowing Slime's normal eval logger to run."""

    output_value = getattr(args, "forgetting_output_dir", None) or getattr(args, "geometry_output_dir", None)
    if output_value is None:
        raise ValueError("Forgetting logger requires --forgetting-output-dir or --geometry-output-dir.")
    output_dir = Path(os.path.expandvars(os.path.expanduser(output_value))) / "forgetting"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "schema_version": 2,
            "tasks": {},
            "phase_starts": {},
            "performance_matrix": [],
            "sample_outcomes": {},
            "last_evaluation_key": None,
        }
    )
    state["schema_version"] = 2
    state.setdefault("phase_starts", {})
    state.setdefault("performance_matrix", [])
    state.setdefault("sample_outcomes", {})

    extra_metrics = extra_metrics or {}
    num_updates = int(extra_metrics.get("eval/num_updates", rollout_id))
    model_version = int(extra_metrics.get("eval/model_version", num_updates))
    eval_phase = str(extra_metrics.get("eval/phase", "unspecified"))
    evaluation_key = f"{num_updates}:{model_version}:{eval_phase}"
    metrics_path = output_dir / "metrics.jsonl"
    metric_records = _reconcile_metrics_with_state(output_dir, state, metrics_path)
    if any(_evaluation_key(previous) == evaluation_key for previous in metric_records):
        raise ValueError(
            f"Forgetting metrics already contain evaluation {evaluation_key}; "
            "refusing to double-count a resumed fixed probe."
        )
    training_phase = str(
        extra_metrics.get(
            "eval/training_task",
            extra_metrics.get(
                "eval/phase_task",
                getattr(args, "experiment_task", None) or eval_phase,
            ),
        )
    )

    scores: dict[str, float] = {}
    task_records: dict[str, dict[str, Any]] = {}
    for task, task_data in sorted(data.items()):
        rewards = task_data.get("rewards") or []
        if not rewards:
            continue
        score = float(sum(float(reward) for reward in rewards) / len(rewards))
        if not math.isfinite(score):
            continue
        scores[task] = score
        previous = state["tasks"].get(task, {})
        baseline = float(previous.get("baseline", score))
        best = max(float(previous.get("best", score)), score)
        phase_key = f"{training_phase}::{task}"
        phase_start = float(state["phase_starts"].setdefault(phase_key, score))
        reward_array = np.asarray([float(reward) for reward in rewards], dtype=np.float64)

        previous_outcomes = state["sample_outcomes"].get(task, {})
        current_outcomes: dict[str, bool] = {}
        pass_to_fail = 0
        fail_to_pass = 0
        samples = task_data.get("samples") or []
        for sample_index, reward in enumerate(reward_array.tolist()):
            sample = samples[sample_index] if sample_index < len(samples) else None
            source_index = getattr(sample, "index", None) if sample is not None else None
            key = str(source_index if source_index is not None else sample_index)
            passed = reward == 1.0
            current_outcomes[key] = passed
            if key in previous_outcomes:
                pass_to_fail += int(bool(previous_outcomes[key]) and not passed)
                fail_to_pass += int(not bool(previous_outcomes[key]) and passed)
        state["sample_outcomes"][task] = current_outcomes

        status_errors = 0
        format_errors = 0
        for sample in samples:
            status = getattr(sample, "status", None)
            status_value = getattr(status, "value", status)
            status_errors += int(status_value not in {None, "completed"})
            metadata = getattr(sample, "metadata", None) or {}
            format_errors += int(bool(metadata.get("format_error", False)))
        task_records[task] = {
            "score": score,
            "score_unit": "raw_task_reward",
            "reward_std": float(reward_array.std()),
            "reward_p10": float(np.percentile(reward_array, 10)),
            "reward_p50": float(np.percentile(reward_array, 50)),
            "reward_p90": float(np.percentile(reward_array, 90)),
            "pass_rate": float(np.mean(reward_array == 1.0)),
            "baseline": baseline,
            "phase_start": phase_start,
            "best": best,
            "forgetting": best - score,
            "backward_transfer": score - baseline,
            "change_from_phase_start": score - phase_start,
            "pass_to_fail_count": pass_to_fail,
            "fail_to_pass_count": fail_to_pass,
            "matched_sample_count": sum(key in previous_outcomes for key in current_outcomes),
            "response_error_rate": status_errors / len(samples) if samples else None,
            "response_format_error_rate": format_errors / len(samples) if samples else None,
        }
        state["tasks"][task] = {"baseline": baseline, "best": best, "last": score}

    forgetting_values = [record["forgetting"] for record in task_records.values()]
    bwt_values = [record["backward_transfer"] for record in task_records.values()]
    performance_row = {
        "num_updates": num_updates,
        "model_version": model_version,
        "training_phase": training_phase,
        "scores": scores,
    }
    state["performance_matrix"].append(performance_row)
    record = {
        "schema_version": 2,
        "rollout_id": int(rollout_id),
        "num_updates": num_updates,
        "model_version": model_version,
        "eval_phase": eval_phase,
        "training_phase": training_phase,
        "tasks": task_records,
        "performance_matrix_row": performance_row,
        "ACC": sum(scores.values()) / len(scores) if scores else None,
        # This baseline is the first fixed-probe observation for each task.  A
        # classical phase-diagonal BWT/FWT needs explicit phase/task metadata;
        # retain an honest status instead of fabricating the diagonal.
        "BWT_first_observation": sum(bwt_values) / len(bwt_values) if bwt_values else None,
        "classical_BWT": None,
        "classical_FWT": None,
        "classical_transfer_status": "requires_explicit_phase_diagonal_and_untrained_baseline",
        "probe_metric_availability": {
            "reward_and_pass_rate": "available",
            "response_error_and_format_rate": "available_when_samples_are_returned",
            "nll": "not_collected_eval_interface_returns_reward_only",
            "teacher_sampled_reverse_kl": "not_collected_eval_interface_returns_reward_only",
            "logit_kl_to_initial": "not_collected_requires_checkpoint_probe_forward",
            "logit_kl_to_phase_start": "not_collected_requires_checkpoint_probe_forward",
            "seen_heldout_gap": "not_collected_requires_explicit_seen_heldout_pairing",
            "same_checkpoint_task_gradients": "not_collected_requires_separate_probe_backward",
        },
        "mean_forgetting": sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0.0,
        "mean_backward_transfer": sum(bwt_values) / len(bwt_values) if bwt_values else 0.0,
        "extra_metrics": extra_metrics,
    }
    _atomic_jsonl(metrics_path, [*metric_records, record])
    state["last_rollout_id"] = int(rollout_id)
    state["last_evaluation_key"] = evaluation_key
    _atomic_json(state_path, state)

    metrics: dict[str, float | int] = {
        "forgetting/step": record["num_updates"],
        "forgetting/num_updates": record["num_updates"],
        "forgetting/model_version": record["model_version"],
        "forgetting/mean": record["mean_forgetting"],
        "forgetting/mean_backward_transfer": record["mean_backward_transfer"],
    }
    for name in ("ACC", "BWT_first_observation"):
        value = record[name]
        if isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
            metrics[f"forgetting/{name}"] = value
    for task, values in task_records.items():
        safe_task = task.replace("/", "_")
        for name, value in values.items():
            if isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
                metrics[f"forgetting/{safe_task}/{name}"] = value
    from slime.utils import logging_utils

    logging_utils.log(args, metrics, step_key="forgetting/step")
    return False
