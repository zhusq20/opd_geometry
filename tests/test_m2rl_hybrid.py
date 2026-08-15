"""CPU tests for the SFT + OPD mixed condition."""

from types import SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample
from slime_plugins.m2rl.hybrid import (
    _mode_token_masks,
    hybrid_advantages,
    hybrid_loss_function,
    post_process_sft_opd_rewards,
)

NUM_GPUS = 0


@pytest.mark.unit
def test_hybrid_reward_postprocess_scores_only_opd_rows():
    teacher_response = {
        "meta_info": {
            "input_token_logprobs": [[0.0, 10], [-1.0, 11], [-2.0, 12]],
        }
    }
    opd = Sample(
        response_length=2,
        reward=teacher_response,
        metadata={"training_mode": "opd"},
    )
    sft = Sample(
        response_length=3,
        reward=0.0,
        metadata={"training_mode": "sft"},
    )

    raw, normalized = post_process_sft_opd_rewards(SimpleNamespace(reward_key=None), [opd, sft])

    assert raw == normalized == [0.0, 0.0]
    assert opd.teacher_log_probs.tolist() == [-1.0, -2.0]
    assert sft.teacher_log_probs.tolist() == [0.0, 0.0, 0.0]
    assert [opd.train_metadata, sft.train_metadata] == [
        {"training_mode": "opd"},
        {"training_mode": "sft"},
    ]


@pytest.mark.unit
def test_hybrid_advantage_masks_default_opd_adjustment_on_sft_rows():
    student = [torch.tensor([-0.5, -0.7]), torch.tensor([-1.0])]
    teacher = [torch.tensor([-1.5, -1.7]), torch.tensor([-9.0])]
    data = {
        "log_probs": student,
        "teacher_log_probs": teacher,
        "metadata": [{"training_mode": "opd"}, {"training_mode": "sft"}],
    }

    hybrid_advantages(SimpleNamespace(), data)

    assert torch.equal(data["teacher_log_probs"][0], teacher[0])
    assert torch.equal(data["teacher_log_probs"][1], student[1])
    assert all(torch.count_nonzero(value) == 0 for value in data["advantages"])


@pytest.mark.unit
def test_mode_token_masks_follow_ragged_sample_lengths():
    opd, sft = _mode_token_masks(["opd", "sft"], [torch.zeros(2), torch.zeros(3)])
    assert opd.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert sft.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


@pytest.mark.unit
def test_hybrid_loss_has_separate_opd_and_sft_gradients(monkeypatch):
    import slime.backends.megatron_utils.loss as megatron_loss
    import slime_plugins.m2rl.hybrid as hybrid

    logits = torch.tensor([-1.0, -1.0, -2.0, -3.0], requires_grad=True)

    def fake_log_probs(_logits, **_kwargs):
        return None, {"log_probs": [_logits[:2], _logits[2:]]}

    def fake_policy_loss(ppo_kl, advantages, *_args, **_kwargs):
        return -torch.exp(-ppo_kl) * advantages, torch.zeros_like(ppo_kl)

    monkeypatch.setattr(megatron_loss, "get_log_probs_and_entropy", fake_log_probs)
    monkeypatch.setattr(hybrid, "compute_policy_loss", fake_policy_loss)
    args = SimpleNamespace(
        eps_clip=0.2,
        eps_clip_high=0.28,
        eps_clip_c=None,
        hybrid_opd_loss_coef=0.5,
        hybrid_sft_loss_coef=2.0,
    )
    batch = {
        "response_lengths": [2, 2],
        "total_lengths": [2, 2],
        "unconcat_tokens": [torch.zeros(2), torch.zeros(2)],
        "metadata": [{"training_mode": "opd"}, {"training_mode": "sft"}],
        "advantages": [torch.ones(2), torch.zeros(2)],
        "log_probs": [torch.tensor([-1.0, -1.0]), torch.tensor([-9.0, -9.0])],
    }

    loss, metrics = hybrid_loss_function(args, batch, logits, torch.sum)
    loss.backward()

    # OPD: 0.5 * (-1 - 1); SFT: 2 * (2 + 3).
    assert loss.item() == pytest.approx(9.0)
    assert metrics["opd_policy_loss"].item() == pytest.approx(-2.0)
    assert metrics["sft_loss"].item() == pytest.approx(5.0)
    assert logits.grad.tolist() == pytest.approx([-0.5, -0.5, -2.0, -2.0])
