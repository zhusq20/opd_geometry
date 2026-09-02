"""Control and score against one hot-swappable external SGLang teacher slot."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import aiohttp
import torch

from slime.utils.types import Sample
from slime_plugins.m2rl.opd import _teacher_log_probs, load_teacher_router, teacher_reward

from .sampler import TASKS


def _slot_config(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_teacher_router(args.opd_teacher_router_config)
    if int(config.get("schema_version", -1)) != 2:
        raise ValueError("teacher router must use schema_version 2")
    slot = config.get("slot")
    if not isinstance(slot, dict):
        raise ValueError("teacher router must define one `slot` mapping")
    teachers = config.get("teachers")
    if tuple(teachers or ()) != TASKS:
        raise ValueError(f"teacher router task order must be {TASKS}")
    urls = {str(dict(route)["url"]) for route in teachers.values()}
    controls = {str(dict(route)["control_url"]) for route in teachers.values()}
    if len(urls) != 1 or len(controls) != 1 or int(slot.get("gpu_count", 0)) != 1:
        raise ValueError("all four teachers must share one single-GPU generate/control endpoint")
    for key in ("physical_gpu", "state_path"):
        if not str(slot.get(key, "")).strip():
            raise ValueError(f"teacher slot is missing `{key}`")
    for task, route_value in teachers.items():
        route = dict(route_value)
        for key in ("url", "control_url", "model_path", "weight_version"):
            if not str(route.get(key, "")).strip():
                raise ValueError(f"teacher route {task} is missing `{key}`")
        if str(route["weight_version"]) != task:
            raise ValueError(f"teacher route {task} must use weight_version={task}")
    return config, slot


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else {}


def _write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def teacher_memory_mib(args: Any) -> float:
    """Sample physical teacher-slot HBM without importing CUDA in the rollout actor."""

    _config, slot = _slot_config(args)
    gpu = str(slot["physical_gpu"])
    result = subprocess.run(
        ["nvidia-smi", "--id", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(f"failed to read teacher GPU {gpu} memory: {result.stderr.strip()}")
    value = float(result.stdout.strip().splitlines()[0])
    if value <= 0:
        raise RuntimeError(f"teacher GPU {gpu} reported non-positive memory usage")
    return value


async def activate_teacher(args: Any, task: str) -> dict[str, Any]:
    """Load one teacher's HF weights into the shared slot and persist residency."""

    config, slot = _slot_config(args)
    route = dict(config["teachers"][task])
    state_path = Path(os.path.expandvars(str(slot["state_path"]))).resolve()
    state = _read_state(state_path)
    resident_before = state.get("resident")
    started = time.perf_counter()
    memory_before = teacher_memory_mib(args)
    timeout = aiohttp.ClientTimeout(total=float(route.get("request_timeout", 900)))
    version_url = str(route["control_url"]).rsplit("/", 1)[0] + "/get_weight_version"
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(version_url) as response:
            response.raise_for_status()
            observed_version = str((await response.json())["weight_version"])
        if observed_version not in TASKS:
            raise RuntimeError(f"teacher slot returned unknown weight version {observed_version!r}")
        if resident_before != observed_version:
            raise RuntimeError(
                f"teacher residency state says {resident_before!r}, but the server reports {observed_version!r}"
            )
        resident_before = observed_version
        switched = resident_before != task
        if switched:
            payload = {
                "model_path": os.path.expandvars(str(route["model_path"])),
                "load_format": str(route.get("load_format", "auto")),
                "weight_version": str(route["weight_version"]),
            }
            async with session.post(str(route["control_url"]), json=payload) as response:
                response.raise_for_status()
                result = await response.json()
            if result.get("success") is not True:
                raise RuntimeError(f"teacher switch to {task} failed: {result}")
            async with session.get(version_url) as response:
                response.raise_for_status()
                version_after = str((await response.json())["weight_version"])
            if version_after != task:
                raise RuntimeError(f"teacher switch requested {task}, but server reports {version_after!r}")
            _write_state(
                state_path,
                {
                    "resident": task,
                    "model_path": payload["model_path"],
                    "server_pid": state.get("server_pid"),
                    "updated_unix": time.time(),
                },
            )
    elapsed = time.perf_counter() - started
    return {
        "resident_before": resident_before,
        "resident_after": task,
        "switched": switched,
        "teacher_load_seconds": elapsed if switched else 0.0,
        "teacher_offload_seconds": 0.0,
        "teacher_ready_seconds": elapsed,
        "teacher_switch_seconds": elapsed if switched else 0.0,
        "teacher_memory_mib": max(memory_before, teacher_memory_mib(args)),
    }


def _penalty_payload(sample: Sample, penalty: float) -> dict[str, Any]:
    student = list(sample.rollout_log_probs or [])
    teacher = [float(value) - penalty for value in student]
    # _teacher_log_probs discards the first entry and then takes the response tail.
    return {"meta_info": {"input_token_logprobs": [[0.0, 0]] + [[value, 0] for value in teacher]}}


async def score_teacher_samples(
    args: Any,
    task: str,
    samples: Sequence[Sample],
    *,
    failure_penalty: float,
    activation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one generated task batch and numerically penalize failures.

    ``activation`` is supplied by the MOPD rollout path after it overlaps the
    teacher weight switch with student generation. Direct callers may omit it
    and activate immediately before scoring.
    """

    if activation is None:
        activation = await activate_teacher(args, task)
    if str(activation.get("resident_after")) != task:
        raise RuntimeError(
            f"teacher activation completed with {activation.get('resident_after')!r}, expected {task!r}"
        )
    started = time.perf_counter()
    coroutines = [teacher_reward(args, sample) for sample in samples]
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    failures = 0
    for sample, result in zip(samples, results, strict=True):
        metadata = dict(sample.metadata or {})
        empty = int(sample.effective_response_length) == 0
        if isinstance(result, BaseException):
            failures += 1
            metadata["mopd_failure_kind"] = "teacher_scoring"
            metadata["mopd_failure_penalty"] = float(failure_penalty)
            sample.reward = _penalty_payload(sample, float(failure_penalty))
        elif empty:
            metadata["mopd_failure_kind"] = "empty_generation"
            metadata["mopd_failure_penalty"] = float(failure_penalty)
            sample.reward = _penalty_payload(sample, float(failure_penalty))
        else:
            try:
                teacher = _teacher_log_probs(result, sample.response_length)
                if teacher.numel() != sample.response_length or not bool(torch.isfinite(teacher).all()):
                    raise ValueError("teacher payload contains non-finite or misaligned token log-probabilities")
                sample.reward = result
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
                failures += 1
                metadata["mopd_failure_kind"] = "teacher_payload"
                metadata["mopd_failure_penalty"] = float(failure_penalty)
                sample.reward = _penalty_payload(sample, float(failure_penalty))
        sample.metadata = metadata
    scoring = time.perf_counter() - started
    return {
        **activation,
        "teacher_scoring_seconds": scoring,
        "teacher_scoring_failures": failures,
        "teacher_scored_tokens": sum(len(sample.tokens) for sample in samples if sample.reward is not None),
        "teacher_gpu_seconds": (activation["teacher_ready_seconds"] + scoring),
    }


__all__ = ["activate_teacher", "score_teacher_samples", "teacher_memory_mib"]
