import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from slime.rollout import sglang_rollout
from slime.utils.types import Sample


@pytest.fixture
def requests(monkeypatch):
    payloads = []
    monkeypatch.setattr(
        sglang_rollout, "GenerateState", lambda args: SimpleNamespace(tokenizer=None, processor=None)
    )
    monkeypatch.setattr(
        sglang_rollout, "trace_span", lambda *a, **k: nullcontext(SimpleNamespace(update=lambda value: None))
    )

    async def post(url, payload, headers=None):
        payloads.append(payload)
        return {"text": "", "meta_info": {"finish_reason": {"type": "stop"}}}

    monkeypatch.setattr(sglang_rollout, "post", post)
    return payloads


def args(profile="qwen3", context=32768):
    return SimpleNamespace(
        ci_test=False, sglang_router_ip="localhost", sglang_router_port=1,
        mopd_profile=profile, sglang_context_length=context, use_rollout_routing_replay=False,
    )


def test_qwen_eval_respects_native_context_without_sharing_prompt_budgets(requests):
    params = {"max_new_tokens": 32768}
    for prompt_length in (2048, 10):
        sample = Sample(tokens=[1] * prompt_length)
        asyncio.run(sglang_rollout.generate(args(), sample, params))
        payload = requests[-1]
        assert len(payload["input_ids"]) + payload["sampling_params"]["max_new_tokens"] == 32768
        assert sample.metadata["generation_max_new_tokens"] == 32768 - prompt_length
    assert params == {"max_new_tokens": 32768}
    assert requests[0]["sampling_params"]["max_new_tokens"] == 30720
    assert requests[1]["sampling_params"]["max_new_tokens"] == 32758


@pytest.mark.parametrize("profile,context,requested", [("qwen3", 32768, 4096), ("smollm3", 65536, 32768)])
def test_training_and_smollm_eval_keep_requested_response_budget(requests, profile, context, requested):
    asyncio.run(sglang_rollout.generate(args(profile, context), Sample(tokens=[1] * 2048), {"max_new_tokens": requested}))
    assert requests[-1]["sampling_params"]["max_new_tokens"] == requested


def test_continuation_counts_existing_response_once(requests):
    sample = Sample(tokens=[1] * 32760, response_length=10, rollout_log_probs=[0.0] * 10, status=Sample.Status.ABORTED)
    params = {"max_new_tokens": 32768, "min_new_tokens": 16}
    asyncio.run(sglang_rollout.generate(args(), sample, params))
    assert requests[-1]["sampling_params"] == {"max_new_tokens": 8, "min_new_tokens": 8}
    assert params["max_new_tokens"] == 32768


def test_full_context_is_truncated_without_sending_invalid_request(requests):
    sample = Sample(tokens=[1] * 32768)
    asyncio.run(sglang_rollout.generate(args(), sample, {"max_new_tokens": 32768}))
    assert sample.status == Sample.Status.TRUNCATED
    assert not requests
