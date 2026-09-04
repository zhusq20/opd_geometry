"""Score samples with four resident SGLang teacher endpoints on one GPU."""

from __future__ import annotations

import asyncio
import logging
import math
import subprocess
import time
from collections.abc import Sequence
from typing import Any

import torch

from slime.utils.types import Sample
from slime_plugins.m2rl.opd import _teacher_log_probs, load_teacher_router, teacher_reward

from .sampler import TASKS

logger = logging.getLogger(__name__)

_TEACHER_MEMORY_PROBE_ATTEMPTS = 3
_TEACHER_MEMORY_PROBE_TIMEOUT_SECONDS = 1.0
_TEACHER_MEMORY_PROBE_RETRY_DELAY_SECONDS = 0.1
_TEACHER_MEMORY_PROBE_COOLDOWN_SECONDS = 60.0
_teacher_memory_probe_retry_after: dict[str, float] = {}


def _slot_config(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_teacher_router(args.opd_teacher_router_config)
    if int(config.get("schema_version", -1)) != 3:
        raise ValueError("resident teacher router must use schema_version 3")
    pool = config.get("resident_pool")
    if not isinstance(pool, dict) or int(pool.get("gpu_count", 0)) != 1:
        raise ValueError("resident teacher router must define one shared physical GPU")
    if not str(pool.get("physical_gpu", "")).strip():
        raise ValueError("resident teacher pool is missing physical_gpu")
    teachers = config.get("teachers")
    if tuple(teachers or ()) != TASKS:
        raise ValueError(f"teacher router task order must be {TASKS}")
    urls = []
    for task, route_value in teachers.items():
        route = dict(route_value)
        if not str(route.get("url", "")).strip() or not str(route.get("model_path", "")).strip():
            raise ValueError(f"resident teacher route {task} requires url and model_path")
        urls.append(str(route["url"]))
    if len(set(urls)) != len(TASKS):
        raise ValueError("the four resident teachers must use four distinct endpoints")
    return config, pool


def teacher_memory_mib(args: Any) -> float | None:
    """Best-effort HBM telemetry for the shared inference GPU."""

    _config, pool = _slot_config(args)
    gpu = str(pool["physical_gpu"])
    if time.monotonic() < _teacher_memory_probe_retry_after.get(gpu, 0.0):
        return None
    command = ["nvidia-smi", "--id", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    last_error = "unknown error"
    for attempt in range(_TEACHER_MEMORY_PROBE_ATTEMPTS):
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=_TEACHER_MEMORY_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            stdout = result.stdout.strip()
            if result.returncode:
                last_error = result.stderr.strip() or stdout or f"exit code {result.returncode}"
            else:
                try:
                    value = float(stdout.splitlines()[0])
                except (IndexError, ValueError):
                    last_error = f"invalid stdout={stdout!r}"
                else:
                    if math.isfinite(value) and value > 0:
                        _teacher_memory_probe_retry_after.pop(gpu, None)
                        return value
                    last_error = f"invalid memory value={value!r}"
        if attempt + 1 < _TEACHER_MEMORY_PROBE_ATTEMPTS:
            time.sleep(_TEACHER_MEMORY_PROBE_RETRY_DELAY_SECONDS * (attempt + 1))
    _teacher_memory_probe_retry_after[gpu] = time.monotonic() + _TEACHER_MEMORY_PROBE_COOLDOWN_SECONDS
    logger.warning("Teacher HBM telemetry unavailable on GPU %s (%s).", gpu, last_error)
    return None


def _penalty_payload(sample: Sample, penalty: float) -> dict[str, Any]:
    teacher = [float(value) - penalty for value in list(sample.rollout_log_probs or [])]
    return {"meta_info": {"input_token_logprobs": [[0.0, 0]] + [[value, 0] for value in teacher]}}


async def score_teacher_samples(
    args: Any,
    task: str,
    samples: Sequence[Sample],
    *,
    failure_penalty: float,
) -> dict[str, Any]:
    """Score one task block; all model weights are already resident."""

    _slot_config(args)
    memory_before = await asyncio.to_thread(teacher_memory_mib, args)
    started = time.perf_counter()
    results = await teacher_reward(args, list(samples), return_exceptions=True)
    failures = 0
    for sample, result in zip(samples, results, strict=True):
        metadata = dict(sample.metadata or {})
        if isinstance(result, BaseException):
            failures += 1
            metadata["mopd_failure_kind"] = "teacher_scoring"
            metadata["mopd_failure_penalty"] = float(failure_penalty)
            sample.reward = _penalty_payload(sample, float(failure_penalty))
        elif int(sample.effective_response_length) == 0:
            failures += 1
            metadata["mopd_failure_kind"] = "empty_generation"
            metadata["mopd_failure_penalty"] = float(failure_penalty)
            sample.reward = _penalty_payload(sample, float(failure_penalty))
        else:
            try:
                teacher = _teacher_log_probs(result, sample.response_length)
                if teacher.numel() != sample.response_length or not bool(torch.isfinite(teacher).all()):
                    raise ValueError("misaligned teacher token log-probabilities")
                sample.reward = result
            except (IndexError, KeyError, TypeError, ValueError):
                failures += 1
                metadata["mopd_failure_kind"] = "teacher_payload"
                metadata["mopd_failure_penalty"] = float(failure_penalty)
                sample.reward = _penalty_payload(sample, float(failure_penalty))
        sample.metadata = metadata
    scoring_seconds = time.perf_counter() - started
    memory_after = await asyncio.to_thread(teacher_memory_mib, args)
    memory = [value for value in (memory_before, memory_after) if value is not None]
    return {
        "teacher_scoring_seconds": scoring_seconds,
        "teacher_scoring_failures": failures,
        "teacher_scored_tokens": sum(len(sample.tokens) for sample in samples if sample.reward is not None),
        "teacher_memory_mib": max(memory, default=0.0),
        "teacher_memory_probe_failures": 2 - len(memory),
    }


__all__ = ["score_teacher_samples", "teacher_memory_mib"]
