"""Matched student generation followed by four resident-teacher scoring blocks."""

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
from .teacher_slot import score_teacher_samples


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


async def _generate_samples(args: Namespace, samples: list[Sample]) -> tuple[list[Sample], float, dict[str, float]]:
    state = GenerateState(args)
    started = time.perf_counter()

    async def timed_generate(sample: Sample) -> tuple[Sample, float]:
        sample_started = time.perf_counter()
        result = await _generate_one(args, sample, sampling_params_by_index[int(sample.index)])
        return result, time.perf_counter() - sample_started

    sampling_params_by_index = {}
    for sample in samples:
        seed = _paired_sampling_seed(args, sample)
        sample.session_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"mopd:{seed}"))
        params = state.sampling_params.copy()
        if args.sglang_enable_deterministic_inference:
            params["sampling_seed"] = seed
        sampling_params_by_index[int(sample.index)] = params
    generated_with_seconds = await asyncio.gather(*(asyncio.create_task(timed_generate(sample)) for sample in samples))
    elapsed = time.perf_counter() - started
    generated = [sample for sample, _seconds in generated_with_seconds]
    task_seconds = {
        task: max(seconds for sample, seconds in generated_with_seconds if str(sample.metadata["mopd_task"]) == task)
        for task in TASKS
    }
    for sample in generated:
        if sample.tokens is None or sample.rollout_log_probs is None:
            raise RuntimeError("a MOPD response lacks token IDs or rollout log-probabilities")
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
    return generated, elapsed, task_seconds


async def _generate_train(args: Namespace, rollout_id: int, data_source: Any) -> RolloutFnTrainOutput:
    prompt_count = int(data_source.prompt_count_for_rollout(rollout_id))
    groups = data_source.get_samples(prompt_count)
    if len(groups) != prompt_count or any(len(group) != RESPONSES_PER_PROMPT for group in groups):
        raise RuntimeError("every MOPD prompt must produce exactly one attempted response")
    pending_plan = data_source.controller.pending
    if pending_plan is None:
        raise RuntimeError("MOPD allocation disappeared before generation")

    samples = [group[0] for group in groups]
    wall_started = time.perf_counter()
    generated, rollout_seconds, task_rollout_seconds = await _generate_samples(args, samples)
    by_task: dict[str, list[Sample]] = defaultdict(list)
    for sample in generated:
        by_task[str(sample.metadata["mopd_task"])].append(sample)

    task_metrics: dict[str, dict[str, float | int]] = {}
    for task in TASKS:
        task_samples = by_task[task]
        scored = await score_teacher_samples(
            args,
            task,
            task_samples,
            failure_penalty=float(args.mopd_failure_penalty),
        )
        truncated = sum(sample.status == Sample.Status.TRUNCATED for sample in task_samples)
        invalid = sum(sample.status in {Sample.Status.ABORTED, Sample.Status.FAILED} for sample in task_samples)
        empty = sum(int(sample.effective_response_length) == 0 for sample in task_samples)
        task_metrics[task] = {
            "task": task,
            "microbatches": int(pending_plan["counts"][task]),
            "prompt_count": len(task_samples),
            "attempted_responses": len(task_samples),
            "valid_response_tokens": sum(int(sample.effective_response_length) for sample in task_samples),
            "generated_tokens": sum(int(sample.response_length) for sample in task_samples),
            "completed_responses": len(task_samples) - truncated - invalid,
            "truncated_responses": truncated,
            "invalid_responses": invalid,
            "empty_responses": empty,
            "student_rollout_seconds": task_rollout_seconds[task],
            "teacher_scoring_seconds": float(scored["teacher_scoring_seconds"]),
            "teacher_scoring_failures": int(scored["teacher_scoring_failures"]),
            "teacher_scored_tokens": int(scored["teacher_scored_tokens"]),
            "teacher_memory_mib": float(scored["teacher_memory_mib"]),
            "teacher_memory_probe_failures": int(scored["teacher_memory_probe_failures"]),
        }

    generated.sort(key=lambda sample: int(sample.index))
    GenerateState(args).reset()
    total_wall = time.perf_counter() - wall_started
    attempted = len(generated)
    if attempted != int(pending_plan["attempted_responses"]):
        raise RuntimeError("rollout did not preserve the planned response count")
    teacher_seconds = sum(float(unit["teacher_scoring_seconds"]) for unit in task_metrics.values())
    metrics: dict[str, float | int] = {
        "mopd/prompt_count": prompt_count,
        "mopd/attempted_responses": attempted,
        "mopd/response_count": attempted,
        "mopd/valid_response_tokens": sum(int(sample.effective_response_length) for sample in generated),
        "mopd/generated_tokens": sum(int(sample.response_length) for sample in generated),
        "mopd/teacher_scored_tokens": sum(int(unit["teacher_scored_tokens"]) for unit in task_metrics.values()),
        "mopd/completed_responses": sum(int(unit["completed_responses"]) for unit in task_metrics.values()),
        "mopd/truncated_responses": sum(int(unit["truncated_responses"]) for unit in task_metrics.values()),
        "mopd/invalid_responses": sum(int(unit["invalid_responses"]) for unit in task_metrics.values()),
        "mopd/empty_responses": sum(int(unit["empty_responses"]) for unit in task_metrics.values()),
        "mopd/student_rollout_wall_seconds": rollout_seconds,
        "mopd/student_rollout_gpu_seconds": rollout_seconds,
        "mopd/teacher_wall_seconds": teacher_seconds,
        "mopd/teacher_gpu_seconds": teacher_seconds,
        "mopd/reward_wall_seconds": 0.0,
        "mopd/rollout_and_teacher_wall_seconds": total_wall,
        "mopd/task_width": len(TASKS),
        "mopd/teacher_peak_memory_mib": max(float(unit["teacher_memory_mib"]) for unit in task_metrics.values()),
        "mopd/teacher_memory_probe_failures": sum(
            int(unit["teacher_memory_probe_failures"]) for unit in task_metrics.values()
        ),
    }
    for task, unit in task_metrics.items():
        for key, value in unit.items():
            if isinstance(value, (int, float)):
                metrics[f"mopd/task/{task}/{key}"] = value
    return RolloutFnTrainOutput(samples=[[sample] for sample in generated], metrics=metrics)


def generate_rollout(args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False):
    if evaluation:
        from .eval import generate_teacher_loss_eval

        return generate_teacher_loss_eval(args, rollout_id, data_source, evaluation=True)
    return run(_generate_train(args, rollout_id, data_source))


__all__ = ["_paired_sampling_seed", "generate_rollout"]
