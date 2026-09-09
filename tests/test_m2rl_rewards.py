"""CPU tests for M2RL reward routing helpers."""

import asyncio
import json
import pickle
import sys
import types
from types import SimpleNamespace

import pytest

from slime.utils.types import Sample
from slime_plugins.m2rl import rewards
from slime_plugins.m2rl.kk import compute_kk_reward, evaluate_kk_response
from slime_plugins.m2rl.rewards import _livecodebench_diagnostics, _sandbox_payload, extract_python

NUM_GPUS = 0


def test_extract_python_uses_last_code_block():
    response = "first\n```python\nprint(1)\n```\nlast\n```py\nprint(2)\n```"
    assert extract_python(response) == "print(2)"


def test_extract_python_rejects_unfenced_text():
    assert extract_python("print(1)") is None


def test_sandbox_payload_uses_sandboxfusion_memory_field():
    payload = _sandbox_payload("print(1)", "", {"memory_limit_mb": 512})
    assert payload["memory_limit_MB"] == 512
    assert "memory_limit" not in payload


def test_code_reward_uses_same_test_subset_for_grpo_group(monkeypatch):
    calls = []

    class FakeClientSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    async def fake_execute(_session, _url, _code, stdin, _config):
        calls.append(stdin)
        return {"stdout": stdin, "status": "success"}

    monkeypatch.setattr(rewards.aiohttp, "ClientSession", FakeClientSession)
    monkeypatch.setattr(rewards, "_execute_code", fake_execute)
    monkeypatch.setattr(rewards, "validate_preflight_marker", lambda *_args: None)
    unit_tests = {"inputs": [str(index) for index in range(50)], "outputs": [str(index) for index in range(50)]}
    config = {"url": "http://sandbox/run_code", "max_cases": 20, "seed": 42}

    first = Sample(
        group_index=17,
        index=100,
        response="```python\nprint(input())\n```",
        metadata={"unit_tests": unit_tests},
    )
    second = Sample(
        group_index=17,
        index=101,
        response="```python\nprint(input())\n```",
        metadata={"unit_tests": unit_tests},
    )

    assert asyncio.run(rewards.code_reward(SimpleNamespace(), first, config)) == 1.0
    first_subset, calls[:] = calls[:], []
    assert asyncio.run(rewards.code_reward(SimpleNamespace(), second, config)) == 1.0

    assert len(first_subset) == 20
    assert calls == first_subset


def test_livecodebench_diagnostics_distinguish_wrong_answer_from_sandbox_failure():
    row = {"test": json.dumps({"input_output": json.dumps({"inputs": ["1", "2"]})})}
    result = {
        "accepted": False,
        "tests": [
            {
                "exec_info": {
                    "status": "Failed",
                    "run_result": {"status": "Finished", "stdout": "Wrong Answer at test 1"},
                }
            }
        ],
    }

    diagnostics = _livecodebench_diagnostics(result, row)

    assert diagnostics["outcome"] == "wrong_answer"
    assert diagnostics["cases_total"] == 2
    assert diagnostics["errors"] == 0


def test_livecodebench_diagnostics_expose_timeout():
    result = {
        "accepted": False,
        "tests": [
            {
                "exec_info": {
                    "status": "Failed",
                    "run_result": {"status": "TimeLimitExceeded", "stdout": ""},
                }
            }
        ],
    }

    diagnostics = _livecodebench_diagnostics(result, {})

    assert diagnostics["outcome"] == "timeout"
    assert diagnostics["timeouts"] == 1


class _FakeResponse:
    def __init__(self, status, *, payload=None, body=""):
        self.status = status
        self.payload = payload
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return self.payload

    async def text(self):
        return self.body


def _fake_client_session(responses, calls):
    class FakeClientSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, *, json):
            calls.append((url, json))
            return responses.pop(0)

    return FakeClientSession


def _livecodebench_sample():
    row = {
        "id": "abc375_c",
        "labels": "{}",
        "content": "Rotate a grid.",
        "test": json.dumps({"input_output": json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]})}),
    }
    return Sample(
        index=28,
        response="```python\nprint(1)\n```",
        metadata={"rm_type": "livecodebench", "question_id": "abc375_c", "sandboxfusion_row": row},
    )


@pytest.mark.parametrize(
    "status,outcome,infrastructure,execution",
    [("Failed", "execution_error", 0, 1), ("SandboxError", "sandbox_error", 1, 0)],
)
def test_unit_test_diagnostics_separate_candidate_failure_from_service_failure(
    monkeypatch, status, outcome, infrastructure, execution
):
    calls = []
    responses = [_FakeResponse(200, payload={"status": status})]
    monkeypatch.setattr(rewards.aiohttp, "ClientSession", _fake_client_session(responses, calls))
    monkeypatch.setattr(rewards, "validate_preflight_marker", lambda *_args: None)
    sample = Sample(
        response="```python\nraise ValueError('candidate failed')\n```",
        metadata={"unit_tests": {"inputs": [""], "outputs": [""]}},
    )
    score = asyncio.run(rewards.code_reward(SimpleNamespace(), sample, {"url": "http://sandbox/run_code"}))
    diagnostics = sample.metadata["sandbox_eval"]
    assert score == 0.0
    assert diagnostics["outcome"] == outcome
    assert diagnostics["infrastructure_errors"] == infrastructure
    assert diagnostics["execution_errors"] == execution
    assert diagnostics["status_counts"] == {status.lower(): 1}


def test_livecodebench_reward_fails_closed_without_preflight_marker():
    with pytest.raises(ValueError, match="preflight_marker"):
        asyncio.run(
            rewards.livecodebench_reward(
                SimpleNamespace(),
                _livecodebench_sample(),
                {"url": "http://sandbox/submit"},
            )
        )


def test_livecodebench_retries_transient_server_error(monkeypatch):
    calls = []
    responses = [
        _FakeResponse(500, body="temporary failure"),
        _FakeResponse(200, payload={"accepted": True, "tests": []}),
    ]
    monkeypatch.setattr(rewards.aiohttp, "ClientSession", _fake_client_session(responses, calls))
    monkeypatch.setattr(rewards, "validate_preflight_marker", lambda *_args: None)

    result = asyncio.run(
        rewards.livecodebench_reward(
            SimpleNamespace(),
            _livecodebench_sample(),
            {"url": "http://sandbox/submit", "retry_attempts": 2, "retry_backoff_seconds": 0},
        )
    )

    assert result == 1.0
    assert len(calls) == 2


def test_livecodebench_http_error_is_serializable_and_identifies_problem(monkeypatch):
    calls = []
    responses = [_FakeResponse(500, body="sandbox uploads exceed 67108864 bytes")]
    monkeypatch.setattr(rewards.aiohttp, "ClientSession", _fake_client_session(responses, calls))
    monkeypatch.setattr(rewards, "validate_preflight_marker", lambda *_args: None)

    with pytest.raises(RuntimeError, match="abc375_c.*HTTP 500.*uploads exceed") as error:
        asyncio.run(
            rewards.livecodebench_reward(
                SimpleNamespace(),
                _livecodebench_sample(),
                {"url": "http://sandbox/submit", "retry_attempts": 1},
            )
        )

    pickle.dumps(error.value)
    assert len(calls) == 1


def test_reward_accepts_batched_custom_rm_contract(monkeypatch):
    def fake_deepscaler(_response, _label):
        return float(_response)

    monkeypatch.setattr(rewards, "get_deepscaler_rule_based_reward", fake_deepscaler)
    args = SimpleNamespace(m2rl_reward_config=None, rm_type="deepscaler")
    samples = [
        Sample(
            index=1,
            response="1",
            metadata={"rm_type": "deepscaler"},
            custom_rm_path="slime_plugins.m2rl.rewards.reward",
        ),
        Sample(
            index=2,
            response="2",
            metadata={"rm_type": "deepscaler"},
            custom_rm_path="slime_plugins.m2rl.rewards.reward",
        ),
    ]

    assert asyncio.run(rewards.reward(args, samples)) == [1.0, 2.0]


def test_builtin_rule_does_not_reenter_custom_router(monkeypatch):
    calls = []

    def fake_deepscaler(response, label):
        calls.append((response, label))
        return 1

    monkeypatch.setattr(rewards, "get_deepscaler_rule_based_reward", fake_deepscaler)
    args = SimpleNamespace(m2rl_reward_config=None, rm_type=None, custom_rm_path="slime_plugins.m2rl.rewards.reward")
    sample = Sample(
        response="answer",
        label="42",
        metadata={"rm_type": "deepscaler"},
        custom_rm_path="slime_plugins.m2rl.rewards.reward",
    )

    assert asyncio.run(rewards.reward(args, sample)) == 1
    assert calls == [("answer", "42")]


def test_deepscaler_route_grades_non_thinking_completion():
    args = SimpleNamespace(m2rl_reward_config=None, rm_type=None)
    sample = Sample(
        response=r"Final: \boxed{42}",
        label="42",
        metadata={"rm_type": "deepscaler"},
        custom_rm_path="slime_plugins.m2rl.rewards.reward",
    )

    assert asyncio.run(rewards.reward(args, sample)) == 1


def test_ifevalg_keyword_reward_uses_vendored_evaluator():
    from slime_plugins.m2rl.ifevalg import compute_ifevalg_reward

    metadata = {
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["geometry", "optimizer"]}],
        "prompt_text": "Mention both required terms.",
    }

    assert compute_ifevalg_reward("Optimizer geometry matters.", None, metadata) == 1.0
    assert compute_ifevalg_reward("Only geometry is mentioned.", None, metadata) == 0.0


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (None, None),
        ("answer<|im_end|>", "answer"),
        ("answer<|im_end|>\n", "answer\n"),
        ("answer<|im_end|><|endoftext|>", "answer"),
        ("literal <|im_end|> inside the answer", "literal <|im_end|> inside the answer"),
        ("answer<|im_end|>after", "answer<|im_end|>after"),
    ],
)
def test_instruction_verifier_response_removes_only_terminal_special_tokens(response, expected):
    assert rewards._instruction_verifier_response(response) == expected


def test_ifevalg_route_removes_terminal_special_token_before_scoring(monkeypatch):
    from slime_plugins.m2rl import ifevalg

    calls = []

    def fake_ifevalg(response, label, metadata):
        calls.append((response, label, metadata))
        return 1.0

    monkeypatch.setattr(ifevalg, "compute_ifevalg_reward", fake_ifevalg)
    args = SimpleNamespace(m2rl_reward_config=None, rm_type=None)
    metadata = {
        "rm_type": "ifevalg",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["geometry"]}],
        "prompt_text": "Mention geometry.",
    }
    sample = Sample(response="geometry<|im_end|>", label=None, metadata=metadata)

    assert asyncio.run(rewards.reward(args, sample)) == 1.0
    assert calls == [("geometry", None, metadata)]


def test_ifbench_route_uses_official_scorer_instead_of_ifevalg(monkeypatch):
    calls = []
    fake_module = types.ModuleType("slime.rollout.rm_hub.ifbench")

    def fake_ifbench(response, label, metadata):
        calls.append((response, label, metadata))
        return 0.75

    fake_module.compute_ifbench_reward = fake_ifbench
    monkeypatch.setitem(sys.modules, "slime.rollout.rm_hub.ifbench", fake_module)
    args = SimpleNamespace(m2rl_reward_config=None, rm_type=None)
    metadata = {
        "rm_type": "ifbench",
        "instruction_id_list": ["count:keywords_multiple"],
        "kwargs": [{"keyword1": "geometry"}],
        "prompt_text": "Follow the instruction.",
    }
    sample = Sample(response="geometry<|im_end|>", label=None, metadata=metadata)

    assert asyncio.run(rewards.reward(args, sample)) == 0.75
    assert calls == [("geometry", None, metadata)]


KK_LABEL = {
    "names": ["Michael", "Zoey", "Ethan"],
    "roles": ["knight", "knave", "knight"],
}


@pytest.mark.parametrize(
    "response",
    [
        (
            "Reasoning can appear before the final block.\n<answer>\n"
            "Michael is a knight\nZoey is a knave\nEthan is a knight\n</answer>"
        ),
        ("<ANSWER>\n(1) ethan is a knight.\n" "2. MICHAEL is a knight\n- Zoey is a knave\n</ANSWER><|im_end|>"),
    ],
)
def test_kk_reward_accepts_exact_assignment_in_any_order(response):
    metadata = {}

    assert compute_kk_reward(response, KK_LABEL, metadata) == 1.0
    assert metadata["kk_eval"]["correct"] is True
    assert metadata["kk_eval"]["format_valid"] is True
    assert metadata["format_error"] is False


def test_kk_reward_distinguishes_valid_wrong_answer_from_format_error():
    wrong = evaluate_kk_response(
        "<answer>\nMichael is a knight\nZoey is a knight\nEthan is a knight\n</answer>",
        KK_LABEL,
    )
    malformed = evaluate_kk_response(
        "<answer>Michael is a knight, Zoey is a knave, Ethan is a knight</answer>",
        KK_LABEL,
    )

    assert wrong["correct"] is False
    assert wrong["format_valid"] is True
    assert wrong["incorrect_names"] == ["Zoey"]
    assert malformed["correct"] is False
    assert malformed["format_valid"] is False
    assert "malformed_assignment_line" in malformed["format_errors"]


@pytest.mark.parametrize(
    ("response", "error", "detail_key", "detail_value"),
    [
        (
            "<answer>\nMichael is a knight\nZoey is a knave\n</answer>",
            "missing_name",
            "missing_names",
            ["Ethan"],
        ),
        (
            "<answer>\nMichael is a knight\nMichael is a knight\nZoey is a knave\nEthan is a knight\n</answer>",
            "duplicate_name",
            "duplicate_names",
            ["Michael"],
        ),
        (
            "<answer>\nMichael is a knight\nMichael is a knave\nZoey is a knave\nEthan is a knight\n</answer>",
            "conflicting_roles_for_name",
            "conflicting_names",
            ["Michael"],
        ),
        (
            "<answer>\nMichael is a knight\nZoey is a knave\nEthan is a knight\nAlice is a knight\n</answer>",
            "unexpected_name",
            "extra_names",
            ["Alice"],
        ),
        (
            "<answer>\nMichael is a knight\nZoey is a knave\nEthan is a knight\n</answer> trailing",
            "text_after_answer_block",
            "missing_names",
            [],
        ),
        (
            "<answer>\nMichael is a knight\nZoey is a knave\nEthan is a knight\n</answer><answer>x</answer>",
            "answer_tags_must_appear_exactly_once",
            "missing_names",
            ["Michael", "Zoey", "Ethan"],
        ),
    ],
)
def test_kk_reward_rejects_incomplete_duplicate_conflicting_extra_or_bad_tags(
    response,
    error,
    detail_key,
    detail_value,
):
    diagnostics = evaluate_kk_response(response, KK_LABEL)

    assert diagnostics["correct"] is False
    assert diagnostics["format_valid"] is False
    assert error in diagnostics["format_errors"]
    assert diagnostics[detail_key] == detail_value


@pytest.mark.parametrize(
    "label",
    [
        None,
        "free-form answer",
        {"names": ["Alice"], "roles": []},
        {"names": ["Alice", " alice "], "roles": ["knight", "knave"]},
        {"names": ["Alice"], "roles": ["unknown"]},
    ],
)
def test_kk_reward_raises_for_invalid_dataset_labels(label):
    with pytest.raises(ValueError, match="logic_kk"):
        compute_kk_reward("<answer>\nAlice is a knight\n</answer>", label)


def test_reward_router_dispatches_kk_and_records_diagnostics():
    args = SimpleNamespace(m2rl_reward_config=None, rm_type=None)
    sample = Sample(
        response="<answer>\nMichael is a knight\nZoey is a knave\nEthan is a knight\n</answer>",
        label=json.dumps(KK_LABEL),
        metadata={"rm_type": "kk"},
    )

    assert asyncio.run(rewards.reward(args, sample)) == 1.0
    assert sample.metadata["kk_eval"]["assigned_roles"] == {
        "Michael": "knight",
        "Zoey": "knave",
        "Ethan": "knight",
    }
