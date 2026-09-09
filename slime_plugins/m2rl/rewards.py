"""Reward router for the four-task MOPD suite.

Math and science reuse Slime's tested rule rewards. Instruction-following
training uses the vendored IFEvalG verifier, while official IFBench evaluation
uses IFBench's distinct constraint registry. Code is executed only by an
external sandbox service. WorkBench trajectories already contain their
environment reward and are passed through unchanged. Logic-RL
Knights-and-Knaves uses a local strict binary verifier.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import math
import os
import random
import re
import pickle
import zlib
from pathlib import Path
from functools import lru_cache
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
_TERMINAL_INSTRUCTION_SPECIAL_TOKENS = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)+(\s*)\Z")


def _instruction_verifier_response(response: str | None) -> str | None:
    """Remove generated end tokens only when they terminate the response."""

    if response is None:
        return None
    return _TERMINAL_INSTRUCTION_SPECIAL_TOKENS.sub(r"\1", response)


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
    return {
        "code": code,
        "stdin": stdin,
        "language": config.get("language", "python"),
        "compile_timeout": float(config.get("compile_timeout", 5)),
        "run_timeout": float(config.get("run_timeout", 10)),
        "memory_limit_MB": int(config.get("memory_limit_mb", 4096)),
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
    function_name = unit_tests.get("fn_name")
    if function_name:
        code = _functional_test_program(code, str(function_name))
    inputs = list(unit_tests.get("inputs") or [])
    outputs = list(unit_tests.get("outputs") or [])
    if function_name:
        # Released TACO rows store arguments as arrays and each return value
        # inside a singleton output array. LCB-style string cases already use
        # one JSON argument per line and an unwrapped JSON result.
        inputs = [
            "\n".join(json.dumps(arg) for arg in value) if isinstance(value, list) else value for value in inputs
        ]
        outputs = [value[0] if isinstance(value, list) and len(value) == 1 else value for value in outputs]
    if len(inputs) != len(outputs) or not inputs:
        return 0.0

    max_cases = int(config.get("max_cases", 20))
    if len(inputs) > max_cases:
        # Every response for one prompt must be graded on the same cases so
        # GRPO's within-group comparison does not include evaluator noise.
        selection_index = sample.group_index if sample.group_index is not None else sample.index
        rng = random.Random(int(selection_index or 0) + int(config.get("seed", 0)))
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
    passed = errors = timeouts = infrastructure_errors = execution_errors = 0
    statuses: dict[str, int] = {}
    for result, expected in zip(results, outputs, strict=True):
        if isinstance(result, Exception):
            status = type(result).__name__.lower()
            errors += 1
            infrastructure_errors += 1
        else:
            status = str(result.get("status", "unknown")).lower()
            stdout = result.get("stdout")
            if (
                status == "success"
                and isinstance(stdout, str)
                and _test_output_matches(stdout, expected, bool(function_name))
            ):
                passed += 1
            elif status != "success":
                errors += 1
                infrastructure = status.startswith("http_") or status in {
                    "sandboxerror",
                    "sandbox_error",
                    "service_failed",
                    "malformed_result",
                }
                infrastructure_errors += int(infrastructure)
                execution_errors += int(not infrastructure)
        statuses[status] = statuses.get(status, 0) + 1
        timeouts += int("time" in status)
    if passed == len(inputs):
        outcome = "accepted"
    elif infrastructure_errors:
        outcome = "sandbox_error"
    elif timeouts:
        outcome = "timeout"
    elif execution_errors:
        outcome = "execution_error"
    else:
        outcome = "wrong_answer"
    sample.metadata["sandbox_eval"] = {
        "evaluator": "unit_test",
        "outcome": outcome,
        "cases_total": len(inputs),
        "cases_passed": passed,
        "errors": errors,
        "infrastructure_errors": infrastructure_errors,
        "execution_errors": execution_errors,
        "status_counts": statuses,
        "timeouts": timeouts,
    }
    metric = str(config.get("metric", "pass_all"))
    if metric == "pass_avg":
        return passed / len(inputs)
    if metric == "pass_all":
        return float(passed == len(inputs))
    raise ValueError(f"Unknown code reward metric {metric!r}; expected pass_all or pass_avg.")


def _functional_test_program(code: str, function_name: str) -> str:
    """Run TACO call-based tests inside the existing external sandbox."""
    return (
        "from typing import *\nimport json, sys\n"
        + code
        + "\n_opd_args = [json.loads(line) for line in sys.stdin.read().splitlines() if line.strip()]\n"
        + f"_opd_fn_name = {function_name!r}\n"
        + "_opd_fn = globals().get(_opd_fn_name)\n"
        + "if _opd_fn is None:\n    _opd_fn = getattr(Solution(), _opd_fn_name)\n"
        + "print(json.dumps(_opd_fn(*_opd_args)))\n"
    )


def _test_output_matches(stdout: str, expected: Any, functional: bool) -> bool:
    if functional:
        try:
            wanted = json.loads(expected) if isinstance(expected, str) else expected
            return json.loads(stdout) == wanted
        except (TypeError, ValueError):
            return False
    return stdout.strip() == str(expected).strip()


@lru_cache(maxsize=4)
def _open_mopd_if_scorer(path: str):
    specification = importlib.util.spec_from_file_location("open_mopd_instruction_reward", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.compute_score


def open_mopd_if_reward(sample: Sample, config: dict[str, Any]) -> float:
    """Reuse Open-MOPD's exact Nemotron registry and native reward convention."""
    if sample.metadata.get("open_mopd_evaluation") and sample.metadata.get("evaluator") == "ifbench":
        if not os.environ.get("OPENOPD_IFBENCH_REPO"):
            from slime.rollout.rm_hub.ifbench import _ensure_ifbench_repo

            os.environ["OPENOPD_IFBENCH_REPO"] = str(_ensure_ifbench_repo())
    scorer = _open_mopd_if_scorer(str(Path(config["scorer_path"]).resolve()))
    result = scorer(
        solution_str=_instruction_verifier_response(sample.response),
        ground_truth=sample.label,
        extra_info=sample.metadata,
        data_source=str(sample.metadata.get("source_dataset") or "nemotron_if_rl"),
        **({"scoring_mode": "official_eval"} if sample.metadata.get("open_mopd_evaluation") else {}),
    )
    sample.metadata["instruction_eval"] = result
    return float(result["score"])


def open_mopd_lcb_row(metadata: dict[str, Any]) -> dict[str, Any]:
    """Translate released LCB problem metadata to SandboxFusion's native row."""
    problem = metadata.get("metadata", metadata)
    if isinstance(problem, str):
        problem = json.loads(problem)
    public = problem["public_test_cases"]
    public = json.loads(public) if isinstance(public, str) else public
    private = problem["private_test_cases"]
    if isinstance(private, str):
        try:
            private = json.loads(private)
        except json.JSONDecodeError:
            # The pinned Open-MOPD release preserves LCB's serialized JSON
            # string representation for private test cases.
            private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(private))))
    tests = public + private
    details = problem.get("metadata") or {}
    details = json.loads(details) if isinstance(details, str) else details
    input_output = {
        "inputs": [test["input"] for test in tests],
        "outputs": [test["output"] for test in tests],
    }
    if details.get("func_name"):
        input_output["fn_name"] = details["func_name"]
    return {
        "id": problem["question_id"],
        "content": problem["question_content"],
        "labels": "{}",
        "test": json.dumps({"input_output": json.dumps(input_output)}),
    }


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
    retry_attempts = int(config.get("retry_attempts", 3))
    retry_backoff_seconds = float(config.get("retry_backoff_seconds", 1))
    if retry_attempts <= 0:
        raise ValueError("LiveCodeBench retry_attempts must be positive.")
    if retry_backoff_seconds < 0:
        raise ValueError("LiveCodeBench retry_backoff_seconds must be non-negative.")
    async with _semaphore(concurrency):
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(retry_attempts):
                async with session.post(str(url), json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        break
                    body = (await response.text())[:4096]
                    if response.status < 500 or attempt + 1 == retry_attempts:
                        raise RuntimeError(
                            f"SandboxFusion LiveCodeBench request failed for problem_id={problem_id!r}: "
                            f"HTTP {response.status}: {body}"
                        )
                await asyncio.sleep(retry_backoff_seconds * 2**attempt)
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
    if rm_type == "open_mopd_lcb":
        sample.metadata["sandboxfusion_row"] = open_mopd_lcb_row(metadata)
        return await livecodebench_reward(args, sample, route_config)
    if route_config.get("url") and rm_type not in {"unit_test"}:
        return await remote_reward(args, sample, route_config)
    if rm_type == "unit_test":
        return await code_reward(args, sample, route_config or config.get("code", {}))
    if rm_type == "open_mopd_if":
        return open_mopd_if_reward(sample, route_config)
    if rm_type == "ifevalg":
        from slime_plugins.m2rl.ifevalg import compute_ifevalg_reward

        return compute_ifevalg_reward(
            _instruction_verifier_response(sample.response),
            sample.label,
            metadata=metadata,
        )
    if rm_type == "ifbench":
        from slime.rollout.rm_hub.ifbench import compute_ifbench_reward

        return compute_ifbench_reward(
            _instruction_verifier_response(sample.response),
            sample.label,
            metadata=metadata,
        )
    if rm_type == "kk":
        from slime_plugins.m2rl.kk import compute_kk_reward

        if sample.metadata is not metadata:
            sample.metadata = metadata
        return compute_kk_reward(sample.response, sample.label, metadata=metadata)
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
