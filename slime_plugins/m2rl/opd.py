"""Per-task SGLang teacher routing for on-policy distillation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import aiohttp
import torch
import yaml

from slime.utils.processing_utils import encode_image_for_rollout_engine, load_tokenizer
from slime.utils.types import Sample

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def _opd_tokenizer(model_path: str):
    return load_tokenizer(model_path, trust_remote_code=True)


@lru_cache(maxsize=16)
def _opd_model_vocab_size(model_path: str) -> int:
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(model_path, trust_remote_code=True).vocab_size)


@lru_cache(maxsize=32)
def _teacher_suffix_tokens(
    model_path: str, suffix: str, student_model_path: str | None
) -> tuple[tuple[int, ...], range]:
    """Encode teacher-only context once and validate the shared token-ID space."""
    tokenizer = _opd_tokenizer(model_path)
    vocabulary = tokenizer.get_vocab()
    suffix_ids = tuple(tokenizer.encode(suffix, add_special_tokens=False))
    if (
        not suffix_ids
        or tokenizer.decode(suffix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != suffix
    ):
        raise ValueError("OPD teacher prompt_suffix must encode and decode exactly without added special tokens.")
    if student_model_path:
        student_tokenizer = _opd_tokenizer(student_model_path)
        if student_tokenizer.get_vocab() != vocabulary:
            raise ValueError("OPD teacher and student tokenizers must have identical token-to-ID vocabulary mappings.")
        if tuple(student_tokenizer.encode(suffix, add_special_tokens=False)) != suffix_ids:
            raise ValueError(
                "OPD teacher prompt_suffix must use the same token IDs in teacher and student tokenizers."
            )
    # Qwen includes unnamed vocabulary rows beyond tokenizer.get_vocab(). They
    # are valid sampled model IDs and must retain the same teacher coordinates.
    vocab_size = _opd_model_vocab_size(model_path)
    if student_model_path and _opd_model_vocab_size(student_model_path) != vocab_size:
        raise ValueError("OPD teacher and student must have the same model vocabulary size.")
    return suffix_ids, range(vocab_size)


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


async def _teacher_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    route: dict[str, Any],
    sample: Sample,
) -> dict[str, Any]:
    # Reuse the exact student response IDs. Teacher-only context is inserted at
    # the response boundary; neither the response nor the student input is
    # decoded, re-tokenized, or modified. max_new_tokens=0 only scores this path.
    input_ids = sample.tokens
    suffix = route.get("prompt_suffix", "")
    if not isinstance(suffix, str):
        raise ValueError("OPD teacher prompt_suffix must be a string.")
    if suffix:
        if not 0 <= sample.response_length < len(input_ids):
            raise ValueError(
                "OPD teacher prompt_suffix requires a valid response_length and a non-empty student prompt."
            )
        model_path = route.get("model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("OPD teacher prompt_suffix requires the route's model_path tokenizer.")
        suffix_ids, vocabulary_ids = _teacher_suffix_tokens(model_path, suffix, route.get("_student_model_path"))
        if any(token_id not in vocabulary_ids for token_id in input_ids):
            raise ValueError("OPD student input contains token IDs outside the teacher model vocabulary.")
        prompt_length = len(input_ids) - sample.response_length
        if tuple(input_ids[max(0, prompt_length - len(suffix_ids)) : prompt_length]) == suffix_ids:
            raise ValueError(
                "OPD student prompt already ends with the teacher prompt_suffix; remove it from the student."
            )
        input_ids = input_ids[:prompt_length] + list(suffix_ids) + input_ids[prompt_length:]
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if route.get("top_logprobs_num"):
        payload["top_logprobs_num"] = int(route["top_logprobs_num"])
        payload["logprob_start_len"] = max(0, len(input_ids) - sample.response_length - 1)
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
        ]
    timeout = aiohttp.ClientTimeout(total=float(route.get("request_timeout", 900)))
    if route.get("student_topk"):
        from slime_plugins.mopd.topk import STUDENT_TOPK, select_teacher_rows, student_support, teacher_score_chunk_size

        k = route.get("student_topk_k", STUDENT_TOPK)
        ids = student_support(sample, k=k).tolist()
        chunk_size = teacher_score_chunk_size(k)
        prompt_length = len(input_ids) - sample.response_length
        if prompt_length <= 0:
            raise ValueError("Student TopK teacher scoring requires a non-empty prompt")
        selected_rows, sampled_rows = [], []
        for start in range(0, sample.response_length, chunk_size):
            end = min(start + chunk_size, sample.response_length)
            chunk = dict(payload)
            chunk.pop("top_logprobs_num", None)
            chunk.update(
                input_ids=input_ids[:prompt_length + end],
                logprob_start_len=prompt_length + start - 1,
                token_ids_logprob=sorted({token for row in ids[start:end] for token in row}),
            )
            result = await _post_teacher_request(session, semaphore, route, sample, chunk, timeout)
            selected, sampled = select_teacher_rows(
                result, ids[start:end], input_ids[prompt_length + start:prompt_length + end], k=k
            )
            selected_rows.extend(selected)
            sampled_rows.extend(sampled)
        return {"meta_info": {"input_token_ids_logprobs": selected_rows,
                              "input_token_logprobs": [[None, input_ids[prompt_length - 1]], *sampled_rows]}}
    return await _post_teacher_request(session, semaphore, route, sample, payload, timeout)


async def _post_teacher_request(session, semaphore, route, sample, payload, timeout):
    async with semaphore:
        # The timeout starts only after a request obtains its deterministic
        # client-side concurrency slot, so a large rollout cannot time out
        # merely while waiting behind earlier teacher-prefill batches.
        for attempt in range(3):
            try:
                async with session.post(str(route["url"]), json=payload, timeout=timeout) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return _restore_teacher_vocab_boundary_id(result, payload, route.get("model_path"))
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as error:
                if attempt == 2:
                    raise
                # Scoring is read-only: retry the same tokens and top-k request
                # after a dropped connection, without generating another sample.
                logger.warning(
                    "Teacher scoring request for sample %s failed (%s: %s); retrying (%s/3)",
                    sample.index,
                    type(error).__name__,
                    error,
                    attempt + 2,
                )
                await asyncio.sleep(0.5 * 2**attempt)


def _restore_teacher_vocab_boundary_id(response, payload, model_path):
    """Undo SGLang's last-vocabulary-ID alias in input logprob metadata only.

    SchedulerLogprobResultProcessor clips IDs using ``x < vocab_size - 1``
    instead of ``x < vocab_size``. The model scores the correct input IDs, but
    the last valid ID is serialized as zero. Require the complete requested
    position range and allow only this exact alias; never repair shifted rows
    or change scores, native TopK IDs, or the student's tokens.
    """
    if not model_path:
        return response
    meta = response.get("meta_info", {})
    rows = meta.get("input_token_logprobs")
    expected = payload["input_ids"][payload["logprob_start_len"] :]
    if rows is None or len(rows) != len(expected) or any(row is None or len(row) < 2 for row in rows):
        return response
    mismatches = [i for i, (row, token) in enumerate(zip(rows, expected, strict=True)) if row[1] != token]
    if not mismatches:
        return response
    last_id = _opd_model_vocab_size(model_path) - 1
    if any(expected[i] != last_id or rows[i][1] != 0 for i in mismatches):
        return response
    restored = [list(row) for row in rows]
    for i in mismatches:
        restored[i][1] = expected[i]
    logger.warning("Restored %d SGLang input logprob metadata IDs at the vocabulary boundary (%d)",
                   len(mismatches), last_id)
    return {**response, "meta_info": {**meta, "input_token_logprobs": restored,
                                     "opd_restored_input_token_ids": len(mismatches)}}


async def teacher_reward(
    args: Any, sample: Sample | list[Sample], **kwargs: Any
) -> dict[str, Any] | list[dict[str, Any] | BaseException]:
    enabled = bool(getattr(args, "mopd_enabled", False))
    loss = getattr(args, "mopd_loss", None)
    from slime_plugins.mopd.topk import student_topk_size

    teacher_topk = enabled and loss in {"teacher_topk", "topk_intersection"}
    teacher_k = student_topk_size(args) if enabled and loss == "topk_intersection" else 64
    student_topk = enabled and loss == "student_topk"
    if isinstance(sample, list):
        return_exceptions = bool(kwargs.pop("return_exceptions", False))
        del kwargs
        routes = [teacher_route(args, item) for item in sample]
        for route in routes:
            if route.get("prompt_suffix") and getattr(args, "hf_checkpoint", None):
                route["_student_model_path"] = str(args.hf_checkpoint)
        if teacher_topk:
            for route in routes:
                route["top_logprobs_num"] = teacher_k
        if student_topk:
            for route in routes:
                route["student_topk"] = True
                route["student_topk_k"] = student_topk_size(args)
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
                ],
                return_exceptions=return_exceptions,
            )
    del kwargs
    route = teacher_route(args, sample)
    if route.get("prompt_suffix") and getattr(args, "hf_checkpoint", None):
        route["_student_model_path"] = str(args.hf_checkpoint)
    if teacher_topk:
        route["top_logprobs_num"] = teacher_k
    if student_topk:
        route["student_topk"] = True
        route["student_topk_k"] = student_topk_size(args)
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
