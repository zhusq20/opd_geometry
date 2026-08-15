"""Unit tests for task-aware OPD routing and token alignment."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample
from slime_plugins.m2rl import opd
from slime_plugins.m2rl.opd import _teacher_log_probs, teacher_url

NUM_GPUS = 0


@pytest.mark.unit
def test_teacher_routing_uses_task_then_default(tmp_path):
    path = tmp_path / "teachers.json"
    path.write_text(json.dumps({"teachers": {"math": "http://math/generate"}, "default": "http://base/generate"}))
    args = SimpleNamespace(opd_teacher_router_config=str(path), rm_url=None)

    assert teacher_url(args, Sample(metadata={"task_name": "math"})) == "http://math/generate"
    assert teacher_url(args, Sample(metadata={"task_name": "code"})) == "http://base/generate"


@pytest.mark.unit
def test_teacher_log_probs_are_aligned_to_response_tail():
    response = {"meta_info": {"input_token_logprobs": [[0.0, 0], [-1.0, 1], [-2.0, 2], [-3.0, 3]]}}
    assert _teacher_log_probs(response, 2).tolist() == [-2.0, -3.0]


@pytest.mark.unit
def test_teacher_request_scores_exact_tokens_without_generating_or_requesting_vocab_targets():
    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def json(self):
            return {"meta_info": {"input_token_logprobs": [[0.0, 10]]}}

    class FakeSession:
        payload = None

        def post(self, _url, *, json, timeout):
            del timeout
            self.payload = json
            return FakeResponse()

    session = FakeSession()
    sample = Sample(tokens=[10, 11, 12])
    asyncio.run(
        opd._teacher_request(
            session,
            asyncio.Semaphore(1),
            {"url": "http://teacher/generate", "request_timeout": 1},
            sample,
        )
    )

    assert session.payload == {
        "input_ids": [10, 11, 12],
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }


@pytest.mark.unit
def test_teacher_log_probs_reject_full_vocab_payloads():
    response = {
        "meta_info": {
            "input_token_logprobs": [
                [[0.0, -1.0], 0],
                [[-2.0, -3.0], 1],
                [[-4.0, -5.0], 2],
            ]
        }
    }

    with pytest.raises(ValueError, match="one scalar teacher log-probability"):
        _teacher_log_probs(response, 2)


@pytest.mark.unit
def test_opd_advantage_is_sampled_token_log_ratio_and_rejects_vocab_axis():
    from slime.backends.megatron_utils.loss import apply_opd_kl_to_advantages

    args = SimpleNamespace(opd_type="sglang", opd_kl_coef=1.0)
    advantages = [torch.zeros(2)]
    student = [torch.tensor([-1.0, -1.5])]
    teacher = [torch.tensor([-0.2, -2.0])]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # A_t = log p_T(a_t|h_t) - log p_S(a_t|h_t), for the sampled a_t only.
    torch.testing.assert_close(advantages[0], torch.tensor([0.8, -0.5]))
    torch.testing.assert_close(rollout_data["sampled_reverse_kl_logratio"][0], torch.tensor([-0.8, 0.5]))

    with pytest.raises(ValueError, match=r"Full-vocabulary \[tokens, vocab\]"):
        apply_opd_kl_to_advantages(
            args,
            {"teacher_log_probs": [torch.zeros(2, 8)]},
            [torch.zeros(2)],
            [torch.zeros(2)],
        )


@pytest.mark.unit
def test_combined_reward_accepts_batched_custom_rm_contract(monkeypatch):
    async def fake_gather(_args, sample, _kwargs):
        return {"id": sample.index}, float(sample.index)

    monkeypatch.setattr(opd, "_gather_teacher_and_task", fake_gather)
    result = asyncio.run(opd.combined_reward(SimpleNamespace(), [Sample(index=1), Sample(index=2)]))

    assert result == [
        {"teacher": {"id": 1}, "task_reward": 1.0},
        {"teacher": {"id": 2}, "task_reward": 2.0},
    ]


@pytest.mark.unit
def test_combined_reward_postprocess_preserves_native_task_reward(monkeypatch):
    monkeypatch.setattr(
        opd,
        "_teacher_log_probs",
        lambda _response, response_length: torch.zeros(response_length),
    )
    sample = Sample(
        response_length=2,
        reward={"teacher": {"scores": []}, "task_reward": 0.75},
        metadata={"task_name": "science"},
    )
    args = SimpleNamespace(
        reward_key=None,
        opd_task_reward_weight=0.2,
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=1,
        grpo_std_normalization=False,
    )

    weighted, normalized = opd.post_process_combined_rewards(args, [sample])

    assert weighted == pytest.approx([0.15])
    assert normalized == pytest.approx([0.15])
    assert sample.metadata["task_reward_observed"] == pytest.approx(0.75)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
