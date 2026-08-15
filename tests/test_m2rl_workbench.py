"""CPU tests for the ported M2RL WorkBench rollout adapter."""

import asyncio
import importlib
from types import SimpleNamespace

from slime.utils.types import Sample

NUM_GPUS = 0


def _interaction(module):
    return module.InteractionResult(
        prompt="rendered prompt",
        reward=1.0,
        messages=[],
        info={"trace_id": "trace-1"},
        response="done",
        response_log_probs=[-0.2],
        loss_mask=[1],
        tokens=[10, 11],
        status=module.Status.COMPLETED,
    )


def test_workbench_result_converts_to_current_sample_contract():
    module = importlib.import_module("slime_plugins.m2rl.workbench.generate_with_workplace_assistant")

    sample = module.res_to_sample(_interaction(module), index=4)

    assert sample.index == 4
    assert sample.status == Sample.Status.COMPLETED
    assert sample.response_length == 1
    assert sample.reward == 1.0
    assert sample.rollout_log_probs == [-0.2]


def test_workbench_generate_preserves_multitask_routing(monkeypatch):
    module = importlib.import_module("slime_plugins.m2rl.workbench.generate_with_workplace_assistant")

    class DummyAgent:
        rollout_args = object()
        sampling_params = {"max_new_tokens": 32}

        async def asolve(self, *_args):
            return _interaction(module)

    monkeypatch.setattr(module, "agent_factory", lambda **_kwargs: DummyAgent())
    args = SimpleNamespace(
        partial_rollout=False,
        workplace_assistant_resources_server_url="http://resources:12000",
        rollout_temperature=0.7,
    )
    original = Sample(
        index=7,
        group_index=3,
        rollout_id=9,
        prompt=[{"role": "user", "content": "task"}],
        label=None,
        metadata={"rm_type": "workbench", "source_name": "agent"},
        generate_function_path="generate.path",
        custom_rm_path="reward.path",
    )

    result = asyncio.run(module.generate(args, original, {"max_new_tokens": 32}))

    assert result.group_index == 3
    assert result.rollout_id == 9
    assert result.generate_function_path == "generate.path"
    assert result.custom_rm_path == "reward.path"
    assert result.metadata == {
        "rm_type": "workbench",
        "source_name": "agent",
        "trace_id": "trace-1",
    }
