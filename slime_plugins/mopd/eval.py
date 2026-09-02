"""Paired held-out relative teacher loss through the shared teacher slot."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import numpy as np
import torch

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout
from slime.utils.async_utils import run
from slime.utils.types import Sample
from slime_plugins.m2rl.opd import _teacher_log_probs

from .sampler import TASKS
from .teacher_slot import activate_teacher, score_teacher_samples


async def generation_only_reward(
    args: Namespace, sample: Sample | list[Sample], **kwargs: Any
) -> float | list[float]:
    """A zero-cost eval RM; matched teachers are scored serially afterwards."""

    del args, kwargs
    return [0.0] * len(sample) if isinstance(sample, list) else 0.0


async def _score_all_tasks(args: Namespace, output: RolloutFnEvalOutput, restore_task: str) -> None:
    if tuple(output.data) != TASKS:
        raise ValueError(f"teacher-loss eval datasets must be ordered as {TASKS}, got {tuple(output.data)}")
    for task in TASKS:
        await score_teacher_samples(
            args,
            task,
            output.data[task]["samples"],
            failure_penalty=float(args.mopd_failure_penalty),
        )
    await activate_teacher(args, restore_task)


def _relative_scales(data_source: Any) -> dict[str, float]:
    configs = {str(source.config["name"]): source.config for source in data_source.sources}
    if tuple(configs) != TASKS:
        raise ValueError(f"MOPD source order must be {TASKS}")
    return {task: float(configs[task]["relative_loss_scale"]) for task in TASKS}


def generate_teacher_loss_eval(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnEvalOutput:
    if not evaluation:
        raise ValueError("generate_teacher_loss_eval is evaluation-only")
    output = default_generate_rollout(args, rollout_id, data_source, evaluation=True)
    resident = data_source.controller.resident_teacher
    restore_task = TASKS[0 if resident is None else int(resident)]
    run(_score_all_tasks(args, output, restore_task))
    scales = _relative_scales(data_source)
    relative_means: list[float] = []
    metrics: dict[str, float | int] = {}

    for task in TASKS:
        info = output.data[task]
        losses: list[float] = []
        token_logratios: list[float] = []
        for sample in info["samples"]:
            penalty = float((sample.metadata or {}).get("mopd_failure_penalty", 0.0))
            if penalty > 0:
                loss = penalty
            else:
                if not isinstance(sample.reward, dict):
                    raise TypeError(f"teacher slot did not return an SGLang payload for {task}")
                teacher = _teacher_log_probs(sample.reward, sample.response_length)
                student = torch.as_tensor(sample.rollout_log_probs, dtype=torch.float32)
                mask = torch.as_tensor(
                    sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length,
                    dtype=torch.float32,
                )
                if student.numel() != teacher.numel() or mask.numel() != student.numel():
                    raise ValueError(
                        f"student/teacher/mask span mismatch for {task}: "
                        f"{student.numel()}/{teacher.numel()}/{mask.numel()}"
                    )
                if not bool(torch.all((mask == 0) | (mask == 1)).item()) or float(mask.sum()) <= 0:
                    raise ValueError(f"nonempty teacher-loss response for {task} has an invalid mask")
                logratio = student - teacher
                loss = float((logratio * mask).sum().div(mask.sum()).item())
                token_logratios.extend(
                    float(value)
                    for value, valid in zip(logratio.tolist(), mask.tolist(), strict=True)
                    if valid
                )
            losses.append(loss)
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["sampled_reverse_kl"] = loss
            sample.metadata["relative_teacher_loss"] = loss * scales[task]
            sample.reward = loss

        if not losses:
            raise ValueError(f"teacher-loss eval dataset {task} is empty")
        raw_mean = float(np.mean(losses))
        relative_mean = raw_mean * scales[task]
        relative_means.append(relative_mean)
        info["rewards"] = losses
        metrics[f"eval/teacher_loss/{task}"] = raw_mean
        metrics[f"eval/relative_teacher_loss/{task}"] = relative_mean
        metrics[f"eval/teacher_loss/{task}/responses"] = len(losses)
        metrics[f"eval/teacher_loss/{task}/tokens"] = len(token_logratios)
        metrics[f"eval/teacher_loss/{task}/logratio_p95"] = (
            float(np.percentile(token_logratios, 95)) if token_logratios else 0.0
        )
        metrics[f"eval/teacher_loss/{task}/logratio_p99"] = (
            float(np.percentile(token_logratios, 99)) if token_logratios else 0.0
        )

    metrics["eval/mean_relative_teacher_loss"] = float(np.mean(relative_means))
    return RolloutFnEvalOutput(data=output.data, metrics=metrics)


__all__ = ["generation_only_reward", "generate_teacher_loss_eval"]
