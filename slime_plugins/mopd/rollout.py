"""Matched student rollouts and sequential one-slot teacher scoring."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from argparse import Namespace
from collections import defaultdict
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sample_hooks import apply_rollout_sample_hooks
from slime.rollout.sglang_rollout import GenerateState, generate
from slime.utils.async_utils import run
from slime.utils.types import Sample

from .sampler import RESPONSES_PER_PROMPT, TASKS
from .teacher_slot import activate_teacher, score_teacher_samples


async def _generate_one(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    state = GenerateState(args)
    async with state.semaphore:
        if state.aborted:
            raise RuntimeError("MOPD does not allow rollout abortion")
        with state.dp_rank_context():
            sample = await generate(args, sample, sampling_params)
    return await apply_rollout_sample_hooks(args, sample, evaluation=False)


def _paired_sampling_seed(args: Namespace, sample: Sample) -> int:
    metadata = sample.metadata or {}
    fields = (
        int(args.mopd_seed),
        str(metadata["protocol_split"]),
        str(metadata["mopd_task"]),
        int(metadata["mopd_task_prompt_epoch"]),
        int(metadata["mopd_task_prompt_ordinal"]),
        int(metadata["mopd_response_ordinal"]),
    )
    digest = hashlib.sha256("\x1f".join(map(str, fields)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


async def _generate_task(args: Namespace, samples: list[Sample]) -> tuple[list[Sample], float]:
    state = GenerateState(args)
    started = time.perf_counter()
    pending = []
    for sample in samples:
        seed = _paired_sampling_seed(args, sample)
        sample.session_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"mopd:{seed}"))
        params = state.sampling_params.copy()
        if args.sglang_enable_deterministic_inference:
            params["sampling_seed"] = seed
        pending.append(asyncio.create_task(_generate_one(args, sample, params)))
    generated = await asyncio.gather(*pending)
    elapsed = time.perf_counter() - started
    for sample in generated:
        if sample.tokens is None or sample.rollout_log_probs is None:
            raise RuntimeError("a MOPD response lacks the token/log-prob arrays required for OPD")
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
    return generated, elapsed


async def _generate_train(args: Namespace, rollout_id: int, data_source: Any) -> RolloutFnTrainOutput:
    prompt_count = int(data_source.prompt_count_for_rollout(rollout_id))
    groups = data_source.get_samples(prompt_count)
    if len(groups) != prompt_count or any(len(group) != RESPONSES_PER_PROMPT for group in groups):
        raise RuntimeError("every MOPD prompt must produce exactly four attempted responses")

    pending_plan = data_source.controller.pending
    if pending_plan is None:
        raise RuntimeError("MOPD plan disappeared before generation")
    by_task: dict[str, list[Sample]] = defaultdict(list)
    for group in groups:
        for sample in group:
            by_task[str(sample.metadata["mopd_task"])].append(sample)

    all_samples: list[Sample] = []
    task_metrics: dict[str, dict[str, float | int | bool | str | None]] = {}
    wall_started = time.perf_counter()
    for task in pending_plan["execution_order"]:
        # Student rollout uses GPUs 0-3 while the independent one-slot teacher
        # on GPU 4 loads task weights. Starting both together makes the
        # uncovered transfer tail the Cost-GPAS critical-path cost.
        generation = asyncio.create_task(_generate_task(args, by_task[task]))
        activation = asyncio.create_task(activate_teacher(args, task))
        (generated, rollout_seconds), activated = await asyncio.gather(generation, activation)
        scored = await score_teacher_samples(
            args,
            task,
            generated,
            failure_penalty=float(args.mopd_failure_penalty),
            activation=activated,
        )
        truncated = sum(sample.status == Sample.Status.TRUNCATED for sample in generated)
        invalid = sum(sample.status in {Sample.Status.ABORTED, Sample.Status.FAILED} for sample in generated)
        empty = sum(int(sample.effective_response_length) == 0 for sample in generated)
        task_metrics[task] = {
            "task": task,
            "prompt_count": len(generated) // RESPONSES_PER_PROMPT,
            "attempted_responses": len(generated),
            "valid_response_tokens": sum(int(sample.effective_response_length) for sample in generated),
            "completed_responses": len(generated) - truncated - invalid,
            "truncated_responses": truncated,
            "invalid_responses": invalid,
            "empty_responses": empty,
            "student_rollout_seconds": rollout_seconds,
            "teacher_load_seconds": float(scored["teacher_load_seconds"]),
            "teacher_offload_seconds": float(scored["teacher_offload_seconds"]),
            "teacher_ready_seconds": float(scored["teacher_ready_seconds"]),
            "teacher_scoring_seconds": float(scored["teacher_scoring_seconds"]),
            "teacher_switch_seconds": float(scored["teacher_switch_seconds"]),
            "teacher_transfer_tail_seconds": max(
                float(scored["teacher_ready_seconds"]) - rollout_seconds,
                0.0,
            ),
            "teacher_scoring_failures": int(scored["teacher_scoring_failures"]),
            "teacher_scored_tokens": int(scored["teacher_scored_tokens"]),
            "teacher_memory_mib": float(scored["teacher_memory_mib"]),
            "switched": bool(scored["switched"]),
            "resident_before": scored["resident_before"],
            "resident_after": scored["resident_after"],
        }
        all_samples.extend(generated)

    all_samples.sort(key=lambda sample: int(sample.index))
    GenerateState(args).reset()
    total_wall = time.perf_counter() - wall_started
    attempted = len(all_samples)
    if attempted != int(pending_plan["attempted_responses"]):
        raise RuntimeError("rollout did not preserve the planned attempted-response count")
    rollout_seconds = sum(float(unit["student_rollout_seconds"]) for unit in task_metrics.values())
    teacher_seconds = sum(
        float(unit["teacher_ready_seconds"]) + float(unit["teacher_scoring_seconds"])
        for unit in task_metrics.values()
    )
    metrics: dict[str, float | int] = {
        "mopd/prompt_count": prompt_count,
        "mopd/attempted_responses": attempted,
        "mopd/response_count": attempted,
        "mopd/valid_response_tokens": sum(int(sample.effective_response_length) for sample in all_samples),
        "mopd/teacher_scored_tokens": sum(int(unit["teacher_scored_tokens"]) for unit in task_metrics.values()),
        "mopd/completed_responses": sum(int(unit["completed_responses"]) for unit in task_metrics.values()),
        "mopd/truncated_responses": sum(int(unit["truncated_responses"]) for unit in task_metrics.values()),
        "mopd/invalid_responses": sum(int(unit["invalid_responses"]) for unit in task_metrics.values()),
        "mopd/empty_responses": sum(int(unit["empty_responses"]) for unit in task_metrics.values()),
        "mopd/student_rollout_wall_seconds": rollout_seconds,
        "mopd/student_rollout_gpu_seconds": rollout_seconds * int(args.rollout_num_gpus),
        "mopd/teacher_wall_seconds": teacher_seconds,
        "mopd/teacher_pool_gpu_count": 1,
        "mopd/teacher_gpu_seconds": teacher_seconds,
        "mopd/reward_wall_seconds": 0.0,
        "mopd/rollout_and_teacher_wall_seconds": total_wall,
        "mopd/task_width": len(task_metrics),
        "mopd/operation_probe": int(pending_plan["operation"] == "probe"),
        "mopd/operation_bank": int(pending_plan["operation"] == "bank"),
        "mopd/resident_teacher_before_index": (
            -1
            if pending_plan["resident_teacher_before"] is None
            else TASKS.index(str(pending_plan["resident_teacher_before"]))
        ),
        "mopd/resident_teacher_after_index": TASKS.index(str(pending_plan["execution_order"][-1])),
        "mopd/teacher_peak_memory_mib": max(
            float(unit["teacher_memory_mib"]) for unit in task_metrics.values()
        ),
    }
    for task, unit in task_metrics.items():
        for key, value in unit.items():
            if isinstance(value, bool):
                metrics[f"mopd/task/{task}/{key}"] = int(value)
            elif isinstance(value, (int, float)):
                metrics[f"mopd/task/{task}/{key}"] = value
    return RolloutFnTrainOutput(samples=[[sample] for sample in all_samples], metrics=metrics)


def generate_rollout(args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False):
    if evaluation:
        from .eval import generate_teacher_loss_eval

        return generate_teacher_loss_eval(args, rollout_id, data_source, evaluation=True)
    return run(_generate_train(args, rollout_id, data_source))


__all__ = ["_paired_sampling_seed", "generate_rollout"]
