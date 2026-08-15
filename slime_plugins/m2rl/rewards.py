"""Reward router for the five M2RL domains.

Math and science reuse Slime's tested rule rewards. Instruction following uses
IFBench (its metadata schema is compatible with the IFEvalG blend). Code is
executed only by an external sandbox service. WorkBench trajectories already
contain their environment reward and are passed through unchanged.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import aiohttp

from slime.rollout.rm_hub import (
    compute_gpqa_reward,
    compute_score_dapo,
    extract_boxed_answer,
    f1_score,
    get_deepscaler_rule_based_reward,
    grade_answer_verl,
)
from slime.utils.types import Sample

from .sandbox_security import validate_preflight_marker

_CONFIG_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}


def _semaphore(concurrency: int) -> asyncio.Semaphore:
    if concurrency <= 0:
        raise ValueError("Sandbox concurrency must be positive.")
    # A module can be exercised by multiple asyncio.run() calls in tests and
    # utilities. Semaphores are event-loop objects, so never reuse one across
    # loops merely because it has the same numerical limit.
    key = (id(asyncio.get_running_loop()), concurrency)
    return _SEMAPHORES.setdefault(key, asyncio.Semaphore(concurrency))


def load_reward_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    expanded = str(Path(os.path.expandvars(os.path.expanduser(path))).resolve())
    modified = os.path.getmtime(expanded)
    cached = _CONFIG_CACHE.get(expanded)
    if cached and cached[0] == modified:
        return cached[1]
    with open(expanded, encoding="utf-8") as stream:
        text = os.path.expandvars(stream.read())
    unresolved = sorted(set(re.findall(r"\$\{([^}]+)\}", text)))
    if unresolved:
        raise ValueError(f"Unresolved environment variables in {expanded}: {', '.join(unresolved)}")
    if expanded.endswith((".yaml", ".yml")):
        import yaml

        config = yaml.safe_load(text) or {}
    else:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise ValueError("M2RL reward config must be a mapping.")
    _CONFIG_CACHE[expanded] = (modified, config)
    return config


def extract_python(response: str) -> str | None:
    matches = list(re.finditer(r"```(?:python|py)?\s*\n?(.*?)```", response or "", re.DOTALL | re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    return None


def _sandbox_payload(code: str, stdin: str, config: dict[str, Any]) -> dict[str, Any]:
    memory_limit_mb = config.get("memory_limit_mb")
    if memory_limit_mb is None:
        # Backward compatibility with M2RL's old byte-valued
        # ``memory_limit`` option.  SandboxFusion's actual API field is
        # ``memory_limit_MB``; sending the old name silently disabled the
        # limit under Pydantic's default extra-field handling.
        memory_limit_bytes = int(config.get("memory_limit", 4 * 1024**3))
        memory_limit_mb = math.ceil(memory_limit_bytes / 1024**2) if memory_limit_bytes > 0 else memory_limit_bytes
    return {
        "code": code,
        "stdin": stdin,
        "language": config.get("language", "python"),
        "compile_timeout": float(config.get("compile_timeout", 5)),
        "run_timeout": float(config.get("run_timeout", 10)),
        "memory_limit_MB": int(memory_limit_mb),
    }


async def _execute_code(
    session: aiohttp.ClientSession,
    url: str,
    code: str,
    stdin: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    concurrency = int(config.get("concurrency", 128))
    async with _semaphore(concurrency):
        async with session.post(url, json=_sandbox_payload(code, stdin, config)) as response:
            if response.status != 200:
                return {"stdout": None, "status": f"http_{response.status}"}
            payload = await response.json()
    top_status = str(payload.get("status", "success")).lower()
    if top_status not in {"success", "finished", "ok"}:
        return {"stdout": None, "status": top_status or "service_failed"}
    run_result = payload.get("run_result") or payload.get("result") or payload
    if not isinstance(run_result, dict):
        return {"stdout": None, "status": "malformed_result"}
    run_status = str(run_result.get("status", "finished")).lower()
    if run_status not in {"success", "finished", "ok"}:
        return {"stdout": None, "status": run_status or "execution_failed"}
    if run_result.get("return_code") not in {None, 0}:
        return {"stdout": run_result.get("stdout"), "status": f"return_code_{run_result.get('return_code')}"}
    return {"stdout": run_result.get("stdout"), "status": "success"}


async def code_reward(args: Any, sample: Sample, config: dict[str, Any]) -> float:
    code = extract_python(sample.response)
    if not code:
        return 0.0
    unit_tests = (sample.metadata or {}).get("unit_tests") or {}
    if isinstance(unit_tests, str):
        unit_tests = json.loads(unit_tests)
    inputs = list(unit_tests.get("inputs") or [])
    outputs = list(unit_tests.get("outputs") or [])
    if len(inputs) != len(outputs) or not inputs:
        return 0.0

    max_cases = int(config.get("max_cases", 20))
    if len(inputs) > max_cases:
        # Per-sample deterministic selection avoids reward noise across reruns.
        rng = random.Random(int(getattr(sample, "index", 0) or 0) + int(config.get("seed", 0)))
        chosen = sorted(rng.sample(range(len(inputs)), max_cases))
        inputs = [inputs[index] for index in chosen]
        outputs = [outputs[index] for index in chosen]

    url = config.get("url") or getattr(args, "code_sandbox_url", None)
    if not url:
        raise ValueError("unit_test reward requires code.url in --m2rl-reward-config or --code-sandbox-url.")
    validate_preflight_marker(config, str(config.get("preflight_url") or url))
    timeout = aiohttp.ClientTimeout(total=float(config.get("request_timeout", 30)))
    connector = aiohttp.TCPConnector(limit=int(config.get("concurrency", 128)))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        results = await asyncio.gather(
            *[_execute_code(session, str(url), code, str(stdin), config) for stdin in inputs],
            return_exceptions=True,
        )
    passed = 0
    errors = 0
    timeouts = 0
    for result, expected in zip(results, outputs, strict=True):
        if isinstance(result, Exception):
            status = type(result).__name__.lower()
            errors += 1
        else:
            status = str(result.get("status", "unknown")).lower()
            stdout = result.get("stdout")
            if status == "success" and isinstance(stdout, str) and stdout.strip() == str(expected).strip():
                passed += 1
            elif status != "success":
                errors += 1
        timeouts += int("time" in status)
    outcome = "accepted" if passed == len(inputs) else ("sandbox_error" if errors else "wrong_answer")
    sample.metadata["sandbox_eval"] = {
        "evaluator": "unit_test",
        "outcome": outcome,
        "cases_total": len(inputs),
        "cases_passed": passed,
        "errors": errors,
        "timeouts": timeouts,
    }
    metric = str(config.get("metric", "pass_all"))
    if metric == "pass_avg":
        return passed / len(inputs)
    if metric == "pass_all":
        return float(passed == len(inputs))
    raise ValueError(f"Unknown code reward metric {metric!r}; expected pass_all or pass_avg.")


def _livecodebench_diagnostics(result: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    accepted = result.get("accepted") is True
    execution_statuses: list[str] = []
    top_statuses: list[str] = []
    output_fragments: list[str] = []
    for test in result.get("tests") or []:
        if not isinstance(test, dict):
            continue
        execution = test.get("exec_info") or {}
        if isinstance(execution, dict):
            top_statuses.append(str(execution.get("status") or "unknown").lower())
            run_result = execution.get("run_result") or {}
            if isinstance(run_result, dict):
                execution_statuses.append(str(run_result.get("status") or "unknown").lower())
                output_fragments.extend(str(run_result.get(key) or "") for key in ("stdout", "stderr"))
    all_statuses = [*top_statuses, *execution_statuses]
    timeouts = int(any("time" in status for status in all_statuses))
    infrastructure_errors = int(any("sandboxerror" in status or "sandbox_error" in status for status in top_statuses))
    wrong_answer = "wrong answer" in "\n".join(output_fragments).lower()
    execution_errors = int(not accepted and not timeouts and not infrastructure_errors and not wrong_answer)
    if accepted:
        outcome = "accepted"
    elif timeouts:
        outcome = "timeout"
    elif infrastructure_errors:
        outcome = "sandbox_error"
    elif wrong_answer:
        outcome = "wrong_answer"
    else:
        outcome = "execution_error"

    try:
        test_payload = json.loads(row.get("test") or "{}")
        input_output = json.loads(test_payload.get("input_output") or "{}")
        cases_total = len(input_output.get("inputs") or [])
    except (AttributeError, TypeError, ValueError):
        # SandboxFusion already consumed the payload; this count is purely a
        # diagnostic and must never change pass/fail.
        cases_total = 0
    return {
        "evaluator": "livecodebench",
        "outcome": outcome,
        "cases_total": cases_total,
        "cases_passed": cases_total if accepted else 0,
        "errors": infrastructure_errors + execution_errors,
        "infrastructure_errors": infrastructure_errors,
        "execution_errors": execution_errors,
        "timeouts": timeouts,
        "runner_statuses": all_statuses,
    }


async def livecodebench_reward(args: Any, sample: Sample, config: dict[str, Any]) -> float:
    """Evaluate one official LiveCodeBench row through SandboxFusion's evaluator."""

    del args
    url = config.get("url")
    if not url:
        raise ValueError("LiveCodeBench reward requires routes.livecodebench.url.")
    validate_preflight_marker(config, str(config.get("preflight_url") or url))
    metadata = sample.metadata or {}
    row = metadata.get("sandboxfusion_row")
    if isinstance(row, str):
        row = json.loads(row)
    if not isinstance(row, dict):
        raise ValueError("LiveCodeBench sample metadata requires a sandboxfusion_row mapping.")
    problem_id = metadata.get("question_id") or row.get("id") or sample.index
    payload = {
        "dataset": str(config.get("dataset", "m2rl_livecodebench")),
        "id": problem_id,
        "completion": sample.response,
        "config": {
            "dataset_type": "LiveCodeBenchDataset",
            "provided_data": row,
            "run_timeout": float(config.get("run_timeout", 6)),
        },
    }
    timeout = aiohttp.ClientTimeout(total=float(config.get("request_timeout", 180)))
    concurrency = int(config.get("concurrency", 8))
    async with _semaphore(concurrency):
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(str(url), json=payload) as response:
                response.raise_for_status()
                result = await response.json()
    sample.metadata["sandbox_eval"] = _livecodebench_diagnostics(result, row)
    return float(result.get("accepted") is True)


async def remote_reward(args: Any, sample: Sample, config: dict[str, Any]) -> float | dict[str, Any]:
    url = config.get("url")
    if not url:
        raise ValueError("Remote M2RL reward route requires a `url`.")
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
        "metadata": sample.metadata,
    }
    timeout = aiohttp.ClientTimeout(total=float(config.get("timeout", 120)))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(str(url), json=payload) as response:
            response.raise_for_status()
            result = await response.json()
    if isinstance(result, dict) and "reward" in result and len(result) == 1:
        return result["reward"]
    return result


async def reward(
    args: Any, sample: Sample | list[Sample], **kwargs: Any
) -> float | dict[str, Any] | list[float | dict[str, Any]]:
    if isinstance(sample, list):
        return await asyncio.gather(*(reward(args, item, **kwargs) for item in sample))
    del kwargs
    metadata = sample.metadata or {}
    rm_type = str(metadata.get("rm_type") or getattr(args, "rm_type", "")).strip()
    config = load_reward_config(getattr(args, "m2rl_reward_config", None))
    route_config = dict((config.get("routes") or {}).get(rm_type) or {})

    if rm_type == "livecodebench":
        return await livecodebench_reward(args, sample, route_config)
    if route_config.get("url") and rm_type not in {"unit_test"}:
        return await remote_reward(args, sample, route_config)
    if rm_type == "unit_test":
        return await code_reward(args, sample, route_config or config.get("code", {}))
    if rm_type in {"ifevalg", "ifbench"}:
        from slime_plugins.m2rl.ifevalg import compute_ifevalg_reward

        return compute_ifevalg_reward(sample.response, sample.label, metadata=metadata)
    if rm_type == "workbench":
        return float(metadata.get("workbench_reward", sample.reward or 0.0))
    # Do not call rm_hub.async_rm here: args.custom_rm_path and the per-sample
    # custom_rm_path both point back to this router, which would recurse until
    # failure. Dispatch the built-in deterministic rules directly instead.
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]
    if rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    if rm_type == "dapo":
        return compute_score_dapo(response, label)
    if rm_type == "math":
        return float(grade_answer_verl(response, label))
    if rm_type == "f1":
        return f1_score(response, label)[0]
    if rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    raise NotImplementedError(f"M2RL reward route for {rm_type!r} is not implemented.")


async def batched_reward(args: Any, samples: list[Sample], **kwargs: Any) -> list[float | dict[str, Any]]:
    return await asyncio.gather(*(reward(args, sample, **kwargs) for sample in samples))
