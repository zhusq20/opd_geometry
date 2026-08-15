"""Per-task SGLang teacher routing for on-policy distillation."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import aiohttp
import torch
import yaml

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def load_teacher_router(path: str) -> dict[str, Any]:
    expanded = str(Path(os.path.expandvars(os.path.expanduser(path))).resolve())
    modified = os.path.getmtime(expanded)
    if expanded in _CACHE and _CACHE[expanded][0] == modified:
        return _CACHE[expanded][1]
    with open(expanded, encoding="utf-8") as stream:
        text = os.path.expandvars(stream.read())
    unresolved = sorted(set(re.findall(r"\$\{([^}]+)\}", text)))
    if unresolved:
        raise ValueError(f"Unresolved environment variables in {expanded}: {', '.join(unresolved)}")
    config = yaml.safe_load(text) if expanded.endswith((".yaml", ".yml")) else json.loads(text)
    if not isinstance(config, dict):
        raise ValueError("OPD teacher router config must be a mapping.")
    teachers = config.get("teachers", config)
    if not isinstance(teachers, dict) or not teachers:
        raise ValueError("OPD teacher router config requires a non-empty `teachers` mapping.")
    config["teachers"] = teachers
    _CACHE[expanded] = (modified, config)
    return config


def teacher_route(args: Any, sample: Sample) -> dict[str, Any]:
    path = getattr(args, "opd_teacher_router_config", None)
    if not path:
        if not getattr(args, "rm_url", None):
            raise ValueError("Multi-teacher OPD requires --opd-teacher-router-config or --rm-url.")
        return {"url": str(args.rm_url), "concurrency": 128, "request_timeout": 900}
    config = load_teacher_router(path)
    metadata = sample.metadata or {}
    teachers = config["teachers"]
    keys = [metadata.get("teacher"), metadata.get("task_name"), metadata.get("source_name"), metadata.get("rm_type")]
    for key in keys:
        if key is not None and str(key) in teachers:
            value = teachers[str(key)]
            route = dict(value) if isinstance(value, dict) else {"url": str(value)}
            route.setdefault("concurrency", config.get("concurrency", 128))
            route.setdefault("request_timeout", config.get("request_timeout", 900))
            return route
    default = config.get("default")
    if default:
        route = dict(default) if isinstance(default, dict) else {"url": str(default)}
        route.setdefault("concurrency", config.get("concurrency", 128))
        route.setdefault("request_timeout", config.get("request_timeout", 900))
        return route
    raise KeyError(f"No OPD teacher route for sample metadata keys {keys} and no default route.")


def teacher_url(args: Any, sample: Sample) -> str:
    """Backward-compatible URL accessor used by tests and external callers."""

    return str(teacher_route(args, sample)["url"])


async def _teacher_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    route: dict[str, Any],
    sample: Sample,
) -> dict[str, Any]:
    # The prompt was rendered once by the student tokenizer. For frozen Qwen3
    # OPD launches that render includes enable_thinking=false, and the teacher
    # receives the exact same token IDs. max_new_tokens=0 makes this a scoring
    # request only: input_token_logprobs contains one scalar for each observed
    # token, never a teacher-generated trajectory or a full-vocabulary target.
    payload: dict[str, Any] = {
        "input_ids": sample.tokens,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
        ]
    timeout = aiohttp.ClientTimeout(total=float(route.get("request_timeout", 900)))
    async with semaphore:
        # The timeout starts only after a request obtains its deterministic
        # client-side concurrency slot, so a large rollout cannot time out
        # merely while waiting behind earlier teacher-prefill batches.
        async with session.post(str(route["url"]), json=payload, timeout=timeout) as response:
            response.raise_for_status()
            return await response.json()


async def teacher_reward(
    args: Any, sample: Sample | list[Sample], **kwargs: Any
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(sample, list):
        del kwargs
        routes = [teacher_route(args, item) for item in sample]
        semaphores: dict[tuple[str, int], asyncio.Semaphore] = {}
        for route in routes:
            key = (str(route["url"]), int(route.get("concurrency", 128)))
            semaphores.setdefault(key, asyncio.Semaphore(key[1]))
        connector = aiohttp.TCPConnector(limit=sum(key[1] for key in semaphores))
        async with aiohttp.ClientSession(connector=connector) as session:
            return await asyncio.gather(
                *[
                    _teacher_request(
                        session,
                        semaphores[(str(route["url"]), int(route.get("concurrency", 128)))],
                        route,
                        item,
                    )
                    for item, route in zip(sample, routes, strict=True)
                ]
            )
    del kwargs
    route = teacher_route(args, sample)
    async with aiohttp.ClientSession() as session:
        return await _teacher_request(
            session,
            asyncio.Semaphore(int(route.get("concurrency", 128))),
            route,
            sample,
        )


def _teacher_log_probs(response: dict[str, Any], response_length: int) -> torch.Tensor:
    if response_length < 0:
        raise ValueError(f"Response length must be non-negative, got {response_length}.")
    entries = response.get("meta_info", {}).get("input_token_logprobs")
    if not entries:
        raise ValueError("SGLang teacher response is missing meta_info.input_token_logprobs.")
    values = torch.tensor([entry[0] for entry in entries[1:]], dtype=torch.float32)
    if values.ndim != 1:
        raise ValueError(
            "Sampled-token OPD expects one scalar teacher log-probability per input token; "
            f"received shape {tuple(values.shape)}. Full-vocabulary teacher distributions are not accepted."
        )
    if response_length == 0:
        return values[:0]
    if values.numel() < response_length:
        raise ValueError(
            f"Teacher returned {values.numel()} token log-probs for a response of length {response_length}."
        )
    return values[-response_length:]


def post_process_rewards(args: Any, samples: list[Sample], **kwargs: Any):
    del kwargs
    raw_teacher_responses = [sample.get_reward_value(args) for sample in samples]
    for sample, response in zip(samples, raw_teacher_responses, strict=True):
        sample.teacher_log_probs = _teacher_log_probs(response, sample.response_length)

    weight = float(getattr(args, "opd_task_reward_weight", 0.0))
    if weight == 0:
        task_rewards = [0.0] * len(samples)
    else:
        # Task rewards are evaluated only after teacher log-probs have been
        # extracted; this keeps one rollout response and combines both signals.
        raise ValueError(
            "--opd-task-reward-weight is non-zero, but synchronous reward post-processing cannot call async task "
            "rewards. Use slime_plugins.m2rl.opd.combined_reward as --custom-rm-path instead."
        )
    return task_rewards, task_rewards


async def combined_reward(
    args: Any, sample: Sample | list[Sample], **kwargs: Any
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(sample, list):
        import asyncio

        return await asyncio.gather(*(combined_reward(args, item, **kwargs) for item in sample))
    teacher, task = await _gather_teacher_and_task(args, sample, kwargs)
    return {"teacher": teacher, "task_reward": task}


async def _gather_teacher_and_task(args: Any, sample: Sample, kwargs: dict[str, Any]):
    import asyncio

    from slime_plugins.m2rl.rewards import reward as task_reward

    return await asyncio.gather(teacher_reward(args, sample, **kwargs), task_reward(args, sample, **kwargs))


def post_process_combined_rewards(args: Any, samples: list[Sample], **kwargs: Any):
    del kwargs
    weighted: list[float] = []
    weight = float(args.opd_task_reward_weight)
    for sample in samples:
        payload = sample.get_reward_value(args)
        sample.teacher_log_probs = _teacher_log_probs(payload["teacher"], sample.response_length)
        task_reward = float(payload["task_reward"])
        sample.metadata = dict(sample.metadata or {})
        # Preserve the verifier's native task unit separately from the
        # coefficient-weighted value sent into the advantage estimator.
        sample.metadata["task_reward_observed"] = task_reward
        weighted.append(weight * task_reward)
    normalized = list(weighted)
    if (
        args.advantage_estimator in {"grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"}
        and args.rewards_normalization
    ):
        values = torch.tensor(weighted, dtype=torch.float32).reshape(-1, args.n_samples_per_prompt)
        values = values - values.mean(dim=-1, keepdim=True)
        if args.advantage_estimator in {"grpo", "gspo", "cispo"} and args.grpo_std_normalization:
            values = values / (values.std(dim=-1, keepdim=True) + 1e-6)
        normalized = values.flatten().tolist()
    return weighted, normalized
