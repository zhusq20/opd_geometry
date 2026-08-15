"""Exact rollout-side OPD/RL distributions for geometry experiments.

The trainer stores token tensors in context-parallel (CP) shards.  This module
collects only valid response tokens, gathers the small metric payload over the
DP-with-CP Gloo group, and computes distributions once on that group's source
rank.  Normal training does not call this path unless parameter geometry is
enabled.

Only sampled-action quantities are named as such: ``sampled_reverse_kl_logratio``
is ``log pi_student(a|h) - log pi_teacher(a|h)`` for the rollout action.  No
full-vocabulary KL is inferred from those scalars.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.utils.metric_utils import compute_rollout_step, num_updates_before_rollout

from .cp_utils import get_logits_and_tokens_offset_with_cp

POSITION_BIN_COUNT = 10
_DISTRIBUTION_QUANTILES = ((0.10, "p10"), (0.50, "p50"), (0.90, "p90"), (0.99, "p99"))
_POSITION_FIELDS = (
    "student_log_prob",
    "teacher_log_prob",
    "sampled_reverse_kl_logratio",
    "advantage",
    "entropy",
)
_TOKEN_DISTRIBUTIONS = (
    "sampled_reverse_kl_logratio",
    "advantage",
    "entropy",
    "importance_ratio",
)


def distribution_statistics(values: torch.Tensor, *, signed: bool = True) -> dict[str, float | int]:
    """Return exact population statistics for a one-dimensional tensor."""

    values = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    count = int(values.numel())
    if count == 0:
        return {"count": 0}
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Rollout geometry distributions require finite values.")
    square_sum = float(torch.sum(values.square(), dtype=torch.float64))
    output: dict[str, float | int] = {
        "count": count,
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "l2": math.sqrt(max(square_sum, 0.0)),
        "rms": math.sqrt(max(square_sum / count, 0.0)),
        "max_abs": float(values.abs().max()),
    }
    quantiles = torch.tensor(
        [quantile for quantile, _ in _DISTRIBUTION_QUANTILES],
        dtype=torch.float64,
    )
    quantile_values = torch.quantile(values, quantiles)
    for (_, name), value in zip(_DISTRIBUTION_QUANTILES, quantile_values, strict=True):
        output[name] = float(value)
    if signed:
        output["negative_fraction"] = float(torch.count_nonzero(values < 0)) / count
    return output


def _local_response_layout(
    total_length: int,
    response_length: int,
    full_loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global response positions and the matching local CP mask."""

    full_loss_mask = full_loss_mask.detach().reshape(-1)
    if full_loss_mask.numel() != response_length:
        raise ValueError(
            "Response loss-mask length mismatch: " f"mask={full_loss_mask.numel()}, response={response_length}."
        )
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        positions = torch.arange(response_length, device=full_loss_mask.device, dtype=torch.int64)
        return positions, full_loss_mask

    prompt_length = total_length - response_length
    _, _, logits_offsets, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length)
    position_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    for start, stop in logits_offsets:
        if start == stop:
            continue
        response_start = start - (prompt_length - 1)
        response_stop = stop - (prompt_length - 1)
        position_parts.append(
            torch.arange(response_start, response_stop, device=full_loss_mask.device, dtype=torch.int64)
        )
        mask_parts.append(full_loss_mask[response_start:response_stop])
    if not position_parts:
        return (
            torch.empty(0, device=full_loss_mask.device, dtype=torch.int64),
            full_loss_mask[:0],
        )
    return torch.cat(position_parts), torch.cat(mask_parts)


def _as_tensor_list(value: Any, expected: int, name: str) -> list[torch.Tensor] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        count = len(value) if isinstance(value, (list, tuple)) else type(value).__name__
        raise ValueError(f"Rollout geometry expected {expected} {name} tensors, got {count}.")
    if not all(isinstance(item, torch.Tensor) for item in value):
        raise TypeError(f"Rollout geometry field {name} must contain tensors.")
    return list(value)


def _task_reward(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("task_reward")
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _dp_sample_rank() -> int:
    try:
        return int(mpu.get_data_parallel_rank(with_context_parallel=False))
    except (RuntimeError, AssertionError):
        return 0


@torch.no_grad()
def build_local_payload(args: Any, rollout_data: dict[str, Any]) -> dict[str, Any]:
    """Build a compact CPU payload from this rank's CP-owned response tokens."""

    response_lengths = [int(value) for value in rollout_data["response_lengths"]]
    total_lengths = [int(value) for value in rollout_data["total_lengths"]]
    loss_masks = rollout_data["loss_masks"]
    sample_count = len(response_lengths)
    if len(total_lengths) != sample_count or len(loss_masks) != sample_count:
        raise ValueError("Rollout geometry response lengths, total lengths, and masks must align.")

    sampled = _as_tensor_list(
        rollout_data.get("sampled_reverse_kl_logratio"),
        sample_count,
        "sampled_reverse_kl_logratio",
    )
    student = _as_tensor_list(
        rollout_data.get("log_probs") or rollout_data.get("rollout_log_probs"),
        sample_count,
        "student_log_prob",
    )
    teacher = _as_tensor_list(rollout_data.get("teacher_log_probs"), sample_count, "teacher_log_prob")
    advantages = _as_tensor_list(rollout_data.get("advantages"), sample_count, "advantage")
    entropy = _as_tensor_list(rollout_data.get("entropy"), sample_count, "entropy")
    rollout_log_probs = _as_tensor_list(rollout_data.get("rollout_log_probs"), sample_count, "rollout_log_probs")

    fields = {
        "student_log_prob": student,
        "teacher_log_prob": teacher,
        "sampled_reverse_kl_logratio": sampled,
        "advantage": advantages,
        "entropy": entropy,
    }
    source_names = rollout_data.get("source_names") or ["unknown"] * sample_count
    sample_indices = rollout_data.get("sample_indices") or list(range(sample_count))
    rollout_ids = rollout_data.get("rollout_ids") or [0] * sample_count
    truncated = rollout_data.get("truncated") or [0] * sample_count
    observed_task_rewards = rollout_data.get("task_rewards_observed")
    raw_rewards = rollout_data.get("local_raw_reward")
    if raw_rewards is None:
        raw_rewards = rollout_data.get("raw_reward")
    if raw_rewards is None:
        raw_rewards = [None] * sample_count
    if observed_task_rewards is None:
        observed_task_rewards = raw_rewards
    if not all(
        len(values) == sample_count
        for values in (source_names, sample_indices, rollout_ids, truncated, raw_rewards, observed_task_rewards)
    ):
        raise ValueError("Rollout geometry per-sample metadata does not align with response lengths.")

    dp_rank = _dp_sample_rank()
    samples: list[dict[str, Any]] = []
    for index, (total_length, response_length, loss_mask) in enumerate(
        zip(total_lengths, response_lengths, loss_masks, strict=True)
    ):
        positions, local_mask = _local_response_layout(total_length, response_length, loss_mask)
        valid = local_mask.to(torch.bool)
        local_length = int(local_mask.numel())
        valid_positions = positions[valid]
        if response_length > 0:
            position_bins = torch.clamp(
                torch.div(valid_positions * POSITION_BIN_COUNT, response_length, rounding_mode="floor"),
                max=POSITION_BIN_COUNT - 1,
            )
        else:
            position_bins = valid_positions

        item_values: dict[str, torch.Tensor] = {}
        position_stats: dict[str, tuple[list[float], list[int]]] = {}
        for field_name, values in fields.items():
            if values is None:
                continue
            tensor = values[index].detach().reshape(-1).to(torch.float32)
            if tensor.numel() != local_length:
                raise ValueError(
                    f"Rollout geometry sample {index} {field_name} length {tensor.numel()} "
                    f"does not match its local CP response length {local_length}."
                )
            selected = tensor[valid]
            if not bool(torch.isfinite(selected).all()):
                raise ValueError(f"Rollout geometry sample {index} {field_name} contains non-finite values.")
            item_values[field_name] = selected.cpu()
            sums = torch.zeros(POSITION_BIN_COUNT, dtype=torch.float64, device=tensor.device)
            counts = torch.zeros(POSITION_BIN_COUNT, dtype=torch.int64, device=tensor.device)
            if selected.numel():
                sums.index_add_(0, position_bins, selected.to(torch.float64))
                counts.index_add_(0, position_bins, torch.ones_like(position_bins, dtype=torch.int64))
            position_stats[field_name] = (sums.cpu().tolist(), counts.cpu().tolist())

        if student is not None and rollout_log_probs is not None:
            current = student[index].detach().reshape(-1).to(torch.float32)
            old = rollout_log_probs[index].detach().reshape(-1).to(device=current.device, dtype=torch.float32)
            if current.numel() != local_length or old.numel() != local_length:
                raise ValueError("Importance-ratio tensors do not align with the local CP response.")
            ratio = torch.exp(current[valid] - old[valid])
            if not bool(torch.isfinite(ratio).all()):
                raise ValueError("Importance ratio contains non-finite values.")
            item_values["importance_ratio"] = ratio.cpu()

            low = 1.0 - float(getattr(args, "eps_clip", 0.0) or 0.0)
            high = 1.0 + float(getattr(args, "eps_clip_high", 0.0) or 0.0)
            outside = (ratio < low) | (ratio > high)
            item_values["ois_outside_policy_bounds"] = outside.to(torch.float32).cpu()
            if advantages is not None:
                advantage = (
                    advantages[index].detach().reshape(-1).to(device=current.device, dtype=torch.float32)[valid]
                )
                if str(getattr(args, "advantage_estimator", "")) == "cispo":
                    policy_clipped = outside
                else:
                    unclipped_loss = -ratio * advantage
                    clipped_loss = -ratio.clamp(low, high) * advantage
                    policy_clipped = clipped_loss > unclipped_loss
                item_values["policy_clip"] = policy_clipped.to(torch.float32).cpu()

            if bool(getattr(args, "use_tis", False)) or bool(getattr(args, "get_mismatch_metrics", False)):
                tis_low = float(getattr(args, "tis_clip_low", 0.0))
                tis_high = float(getattr(args, "tis_clip", math.inf))
                item_values["tis_clip"] = ((ratio < tis_low) | (ratio > tis_high)).to(torch.float32).cpu()

        item_values["valid_token"] = torch.ones(int(valid.sum()), dtype=torch.float32)
        key = (dp_rank, int(rollout_ids[index]), str(sample_indices[index]), index)
        samples.append(
            {
                "key": key,
                "source": str(source_names[index]),
                "response_length": response_length,
                "truncated": bool(truncated[index]),
                "task_reward": _task_reward(observed_task_rewards[index]),
                "values": item_values,
                "position": position_stats,
            }
        )
    return {"samples": samples}


def summarize_payloads(payloads: list[dict[str, Any]]) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Merge DP/CP payloads and return flat metrics plus availability metadata."""

    by_sample: dict[tuple[Any, ...], dict[str, Any]] = {}
    token_values: dict[str, list[torch.Tensor]] = {name: [] for name in _TOKEN_DISTRIBUTIONS}
    binary_values: dict[str, list[torch.Tensor]] = {
        "policy_clip": [],
        "tis_clip": [],
        "ois_outside_policy_bounds": [],
    }
    position_sums = {name: torch.zeros(POSITION_BIN_COUNT, dtype=torch.float64) for name in _POSITION_FIELDS}
    position_counts = {name: torch.zeros(POSITION_BIN_COUNT, dtype=torch.int64) for name in _POSITION_FIELDS}

    for payload in payloads:
        for item in payload.get("samples", []):
            key = tuple(item["key"])
            state = by_sample.setdefault(
                key,
                {
                    "source": item["source"],
                    "response_length": int(item["response_length"]),
                    "truncated": bool(item["truncated"]),
                    "task_reward": item.get("task_reward"),
                    "values": {},
                },
            )
            if state["source"] != item["source"] or state["response_length"] != int(item["response_length"]):
                raise ValueError(f"CP ranks disagree on metadata for rollout sample {key}.")
            if item.get("task_reward") is not None:
                state["task_reward"] = float(item["task_reward"])
            for name, values in item.get("values", {}).items():
                tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
                state["values"].setdefault(name, []).append(tensor)
                if name in token_values:
                    token_values[name].append(tensor)
                if name in binary_values:
                    binary_values[name].append(tensor)
            for name, (sums, counts) in item.get("position", {}).items():
                if name not in position_sums:
                    continue
                position_sums[name].add_(torch.tensor(sums, dtype=torch.float64))
                position_counts[name].add_(torch.tensor(counts, dtype=torch.int64))

    metrics: dict[str, float | int] = {}
    availability: dict[str, Any] = {}
    for name, pieces in token_values.items():
        if not pieces:
            availability[name] = "not_available"
            continue
        values = torch.cat(pieces)
        availability[name] = "available"
        signed = name in {"sampled_reverse_kl_logratio", "advantage"}
        for statistic, value in distribution_statistics(values, signed=signed).items():
            metrics[f"{name}/token/{statistic}"] = value

    sampled_sequence_means: list[float] = []
    advantage_sequence_means: list[float] = []
    valid_lengths: list[float] = []
    sources: Counter[str] = Counter()
    truncated_count = 0
    task_rewards: list[float] = []
    task_reward_on_valid_sample = False
    for state in by_sample.values():
        sources[state["source"]] += 1
        truncated_count += int(state["truncated"])
        reward = state.get("task_reward")
        if reward is not None:
            task_rewards.append(float(reward))
        valid_pieces = state["values"].get("valid_token", [])
        valid_length = sum(int(piece.numel()) for piece in valid_pieces)
        valid_lengths.append(float(valid_length))
        task_reward_on_valid_sample |= reward is not None and valid_length > 0
        for name, destination in (
            ("sampled_reverse_kl_logratio", sampled_sequence_means),
            ("advantage", advantage_sequence_means),
        ):
            pieces = state["values"].get(name, [])
            if pieces:
                values = torch.cat(pieces)
                if values.numel():
                    destination.append(float(values.mean()))

    for name, values in (
        ("sampled_reverse_kl_logratio", sampled_sequence_means),
        ("advantage", advantage_sequence_means),
    ):
        if not values:
            continue
        statistics = distribution_statistics(torch.tensor(values, dtype=torch.float64), signed=True)
        for statistic, value in statistics.items():
            metrics[f"{name}/sequence_mean/{statistic}"] = value

    if valid_lengths:
        for statistic, value in distribution_statistics(
            torch.tensor(valid_lengths, dtype=torch.float64), signed=False
        ).items():
            metrics[f"response/valid_length/{statistic}"] = value
        metrics["response/truncation_fraction"] = truncated_count / len(valid_lengths)
        metrics["sample_count"] = len(valid_lengths)
        metrics["valid_token_count"] = int(sum(valid_lengths))

    for source, count in sorted(sources.items()):
        metrics[f"source_count/{source}"] = count
        metrics[f"source_fraction/{source}"] = count / len(by_sample) if by_sample else 0.0

    for name, pieces in binary_values.items():
        if pieces:
            values = torch.cat(pieces)
            metrics[f"{name}_fraction"] = float(values.mean()) if values.numel() else 0.0

    for bin_id in range(POSITION_BIN_COUNT):
        lower = bin_id / POSITION_BIN_COUNT
        upper = (bin_id + 1) / POSITION_BIN_COUNT
        label = f"bin_{bin_id:02d}_{lower:.1f}_{upper:.1f}"
        for name in _POSITION_FIELDS:
            count = int(position_counts[name][bin_id])
            if count:
                metrics[f"position/{label}/{name}_mean"] = float(position_sums[name][bin_id]) / count
                metrics[f"position/{label}/{name}_token_count"] = count

    if task_rewards:
        reward_stats = distribution_statistics(torch.tensor(task_rewards), signed=True)
        for statistic, value in reward_stats.items():
            metrics[f"task_reward/{statistic}"] = value
        metrics["task_reward/pass_rate"] = sum(reward == 1.0 for reward in task_rewards) / len(task_rewards)
    availability["task_reward_observed"] = bool(task_rewards)
    availability["task_reward_on_valid_sample"] = task_reward_on_valid_sample
    availability["teacher_topk_probe"] = "not_available_sampled_log_probs_only"
    availability["sample_gradient_coherence"] = "not_collected_requires_per_sample_backward"
    availability["mixed_loss_gradient_geometry"] = "not_collected_requires_component_backward"
    return metrics, availability


def _loss_components(args: Any) -> dict[str, dict[str, Any]]:
    hybrid = str(getattr(args, "custom_loss_function_path", "")) == ("slime_plugins.m2rl.hybrid.hybrid_loss_function")
    use_opd = bool(getattr(args, "use_opd", False))
    entropy_coefficient = float(getattr(args, "entropy_coef", 0.0) or 0.0)
    components = {
        "opd": {
            "status": "enabled" if use_opd else "not_applicable",
            "coefficient": float(getattr(args, "opd_kl_coef", 0.0) or 0.0) if use_opd else None,
        },
        "policy": {
            "status": (
                "enabled"
                if str(getattr(args, "loss_type", "policy_loss")) == "policy_loss" or hybrid
                else "not_applicable"
            ),
            "coefficient": float(getattr(args, "hybrid_opd_loss_coef", 1.0)) if hybrid else 1.0,
        },
        "value": {
            "status": "enabled" if str(getattr(args, "loss_type", "")) == "value_loss" else "not_applicable",
            "coefficient": 1.0 if str(getattr(args, "loss_type", "")) == "value_loss" else None,
        },
        "entropy": {
            "status": "enabled" if entropy_coefficient != 0.0 else "not_applicable",
            "coefficient": entropy_coefficient if entropy_coefficient != 0.0 else None,
        },
        "sft": {
            "status": (
                "enabled" if hybrid and float(getattr(args, "hybrid_sft_loss_coef", 0.0)) != 0.0 else "not_applicable"
            ),
            "coefficient": float(getattr(args, "hybrid_sft_loss_coef", 0.0)) if hybrid else None,
        },
    }
    return components


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(f"Failed to append rollout geometry to {path}.")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _last_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        position = stream.seek(0, os.SEEK_END)
        buffer = b""
        while position > 0:
            read_size = min(position, 8192)
            position -= read_size
            stream.seek(position)
            buffer = stream.read(read_size) + buffer
            stripped = buffer.rstrip()
            boundary = stripped.rfind(b"\n")
            if boundary >= 0:
                return json.loads(stripped[boundary + 1 :])
        stripped = buffer.strip()
        return json.loads(stripped) if stripped else None


def _previous_cumulative(path: Path, rollout_id: int) -> tuple[int, int]:
    record = _last_record(path)
    if record is None:
        return 0, 0
    frontier = int(record.get("rollout_id", -1))
    if int(rollout_id) <= frontier:
        raise ValueError(
            f"Rollout geometry {path} already ends at rollout {frontier}; "
            f"refusing to append replayed rollout {rollout_id}."
        )
    return int(record.get("cumulative_prompt_count", 0)), int(record.get("cumulative_effective_token_count", 0))


def collect_rollout_geometry(
    rollout_id: int,
    args: Any,
    rollout_data: dict[str, Any],
) -> dict[str, float | int] | None:
    """Gather, persist, and log rollout geometry on the DP-with-CP source."""

    started_at = time.perf_counter()
    local = build_local_payload(args, rollout_data)
    if dist.is_available() and dist.is_initialized():
        group = mpu.get_data_parallel_group_gloo(with_context_parallel=True)
        source = mpu.get_data_parallel_src_rank(with_context_parallel=True)
        size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        if dist.get_rank() == source:
            payloads: list[dict[str, Any] | None] = [None] * size
            dist.gather_object(local, payloads, dst=source, group=group)
            gathered = [payload for payload in payloads if payload is not None]
        else:
            dist.gather_object(local, None, dst=source, group=group)
            return None
    else:
        gathered = [local]

    metrics, availability = summarize_payloads(gathered)
    sample_count = int(metrics.get("sample_count", 0))
    valid_tokens = int(metrics.get("valid_token_count", 0))
    output_root = Path(os.path.expandvars(os.path.expanduser(str(args.geometry_output_dir))))
    path = output_root / "rollout" / "metrics.jsonl"
    cumulative_prompts, cumulative_tokens = _previous_cumulative(path, rollout_id)
    cumulative_prompts += sample_count
    cumulative_tokens += valid_tokens
    updates = num_updates_before_rollout(args, rollout_id)
    use_opd = bool(getattr(args, "use_opd", False))
    reward_coefficient = float(getattr(args, "opd_task_reward_weight", 0.0) or 0.0) if use_opd else 1.0
    reward_observed = bool(availability["task_reward_observed"])
    reward_used = bool(availability["task_reward_on_valid_sample"]) and reward_coefficient != 0.0
    observation_ms = (time.perf_counter() - started_at) * 1000.0
    record = {
        "schema_version": 1,
        "record_type": "rollout_geometry",
        "run_id": getattr(args, "experiment_name", None),
        "seed": int(getattr(args, "seed", 0)),
        "task": getattr(args, "experiment_task", None),
        "rollout_id": int(rollout_id),
        "num_updates": updates,
        "model_version": updates,
        "actual_batch_size": sample_count,
        "effective_token_count": valid_tokens,
        "cumulative_prompt_count": cumulative_prompts,
        "cumulative_effective_token_count": cumulative_tokens,
        "sampled_reverse_kl_definition": "log_pi_student_sampled_action_minus_log_pi_teacher_sampled_action",
        "task_reward_observed": reward_observed,
        "reward_used_in_loss": reward_used,
        "reward_loss_coefficient": reward_coefficient if reward_used else 0.0,
        "loss_components": _loss_components(args),
        "availability": availability,
        "observation_wall_time_ms": observation_ms,
        "metrics": metrics,
    }
    _append_record(path, record)

    logged = {
        "rollout/step": compute_rollout_step(args, rollout_id),
        "rollout/num_updates": updates,
        "rollout/model_version": updates,
        "rollout/geometry_observation_wall_time_ms": observation_ms,
        "rollout/task_reward_observed": int(reward_observed),
        "rollout/reward_used_in_loss": int(reward_used),
        "rollout/reward_loss_coefficient": record["reward_loss_coefficient"],
    }
    logged.update({f"rollout/geometry/{name}": value for name, value in metrics.items()})
    from slime.utils import logging_utils

    logging_utils.log(args, logged, step_key="rollout/step")
    return logged


__all__ = [
    "POSITION_BIN_COUNT",
    "build_local_payload",
    "collect_rollout_geometry",
    "distribution_statistics",
    "summarize_payloads",
]
