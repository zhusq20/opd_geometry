"""CPU tests for M2RL reward routing helpers."""

import asyncio
import json
from types import SimpleNamespace

from slime.utils.types import Sample
from slime_plugins.m2rl import rewards
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

    legacy = _sandbox_payload("print(1)", "", {"memory_limit": 4 * 1024**3})
    assert legacy["memory_limit_MB"] == 4096


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
