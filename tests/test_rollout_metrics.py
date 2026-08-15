import base64
from argparse import Namespace

import numpy as np
import pytest
import torch

from slime.ray.rollout import (
    _compute_sample_outcome_metrics,
    _compute_top_p_kept_vocab_metrics,
    _compute_training_reward_metrics,
)
from slime.utils.metric_utils import compute_pass_rate, rollout_prompt_count
from slime.utils.misc import decode_int32_meta_array
from slime.utils.types import Sample

NUM_GPUS = 0


def _make_args():
    return Namespace(sglang_speculative_algorithm=False, num_layers=2, moe_router_topk=2)


@pytest.mark.unit
def test_rollout_prompt_count_preserves_exact_epoch_tail_and_restarts_next_epoch():
    args = Namespace(
        rollout_batch_size=64,
        rollout_prompts_per_epoch=19125,
        rollout_steps_per_epoch=299,
    )

    assert rollout_prompt_count(args, 0) == 64
    assert rollout_prompt_count(args, 297) == 64
    assert rollout_prompt_count(args, 298) == 53
    assert rollout_prompt_count(args, 299) == 64


@pytest.mark.unit
def test_rollout_prompt_count_keeps_fixed_size_num_rollout_behavior():
    assert rollout_prompt_count(Namespace(rollout_batch_size=16), 999) == 16


@pytest.mark.unit
def test_training_reward_metrics_report_values_and_source_composition():
    args = Namespace(reward_key=None)
    samples = [
        Sample(reward=1.0, metadata={"task_name": "math"}),
        Sample(reward=0.0, metadata={"task_name": "math"}),
        Sample(reward={"teacher": {}, "task_reward": 0.5}, metadata={"task_name": "science"}),
        # A pure OPD teacher response is deliberately not treated as a task reward.
        Sample(reward={"meta_info": {"input_token_logprobs": []}}, metadata={"task_name": "science"}),
    ]

    metrics = _compute_training_reward_metrics(args, samples)

    assert metrics["source_count/math"] == 2
    assert metrics["source_fraction/science"] == pytest.approx(0.5)
    assert metrics["reward/math/mean"] == pytest.approx(0.5)
    assert metrics["reward/science/mean"] == pytest.approx(0.5)
    assert metrics["reward/count"] == 3
    assert metrics["reward/mean"] == pytest.approx(0.5)


@pytest.mark.unit
def test_livecodebench_pass_rate_includes_standard_k_values():
    rewards = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    metrics = compute_pass_rate(rewards, group_size=10)

    assert {"pass@1", "pass@5", "pass@10"} <= metrics.keys()
    assert metrics["pass@1"] == pytest.approx(0.1)
    assert metrics["pass@5"] == pytest.approx(0.5)
    assert metrics["pass@10"] == pytest.approx(1.0)
    assert compute_pass_rate([1.0], group_size=1) == {"pass@1": 1.0}


@pytest.mark.unit
def test_sample_outcome_metrics_count_status_filter_and_sandbox_failures():
    samples = [
        Sample(
            status=Sample.Status.COMPLETED,
            metadata={"sandbox_eval": {"outcome": "accepted", "cases_total": 2, "cases_passed": 2}},
        ),
        Sample(
            status=Sample.Status.TRUNCATED,
            remove_sample=True,
            metadata={
                "sandbox_eval": {
                    "outcome": "sandbox_error",
                    "cases_total": 2,
                    "cases_passed": 0,
                    "errors": 1,
                    "infrastructure_errors": 1,
                    "execution_errors": 0,
                    "timeouts": 1,
                }
            },
        ),
    ]

    metrics = _compute_sample_outcome_metrics(samples)

    assert metrics["status/count_completed"] == 1
    assert metrics["status/fraction_truncated"] == pytest.approx(0.5)
    assert metrics["filtered/count"] == 1
    assert metrics["sandbox/cases"] == 4
    assert metrics["sandbox/errors"] == 1
    assert metrics["sandbox/infrastructure_errors"] == 1
    assert metrics["sandbox/execution_errors"] == 0
    assert metrics["sandbox/timeouts"] == 1


@pytest.mark.unit
def test_top_p_kept_vocab_metric_uses_loss_mask():
    samples = [
        Sample(
            response_length=4,
            loss_mask=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
            rollout_top_p_token_offsets=torch.tensor([0, 3, 8, 10, 20], dtype=torch.int32),
        ),
        Sample(
            response_length=2,
            loss_mask=None,
            rollout_top_p_token_offsets=torch.tensor([0, 4, 9], dtype=torch.int32),
        ),
    ]

    metrics = _compute_top_p_kept_vocab_metrics(None, samples)

    assert metrics["top_p_kept_vocab_per_token"] == pytest.approx(3.5)


@pytest.mark.unit
def test_top_p_kept_vocab_metric_skips_removed_samples():
    samples = [
        Sample(
            response_length=3,
            loss_mask=[1, 1, 1],
            remove_sample=True,
            rollout_top_p_token_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        )
    ]

    assert _compute_top_p_kept_vocab_metrics(None, samples) == {}


def _b64_int32(values: list[int]) -> str:
    return base64.b64encode(np.array(values, dtype=np.int32).tobytes()).decode("ascii")


@pytest.mark.unit
def test_decode_int32_meta_array_decodes_base64_to_tensor():
    decoded = decode_int32_meta_array({"routed_experts": _b64_int32([1, 2, 3])}, "routed_experts")

    assert torch.is_tensor(decoded)
    assert decoded.dtype == torch.int32
    torch.testing.assert_close(decoded, torch.tensor([1, 2, 3], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_merges_top_p_tensors():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.tokens == [0, 1, 10, 20]
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_can_skip_terminal_status_for_streaming_chunks():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
        update_terminal_info=False,
    )

    assert sample.status is Sample.Status.PENDING
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_decodes_routed_experts():
    sample = Sample(tokens=[101, 102, 103])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts.shape == (2, 2, 2)
    torch.testing.assert_close(
        sample.rollout_routed_experts,
        torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.int32),
    )


@pytest.mark.unit
def test_append_response_tokens_ignores_split_pd_routed_experts():
    sample = Sample(tokens=[101, 102, 103, 104])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "pd_prefill_routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "pd_decode_routed_experts": _b64_int32([8, 9, 10, 11]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts is None


@pytest.mark.unit
def test_append_response_tokens_rejects_mismatched_routed_experts_shape():
    sample = Sample(tokens=[101, 102, 103])

    with pytest.raises(ValueError, match="routed_experts element count"):
        sample.append_response_tokens(
            _make_args(),
            tokens=[],
            trainable=True,
            meta_info={
                "routed_experts": _b64_int32([0, 1, 2, 3]),
                "finish_reason": {"type": "stop"},
            },
        )


@pytest.mark.unit
def test_append_response_tokens_pads_top_p_for_non_trainable_tokens():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        rollout_top_p_token_ids=torch.tensor([10, 11], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 2], dtype=torch.int32),
    )

    sample.append_response_tokens(tokens=[200, 201, 202], trainable=False)

    assert sample.tokens == [0, 1, 200, 201, 202]
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 0, 0, 0]
    assert sample.rollout_log_probs == [-0.1, 0.0, 0.0, 0.0]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([10, 11], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_requires_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="trainable response tokens require rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], trainable=True)


@pytest.mark.unit
def test_append_response_tokens_rejects_non_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="non-trainable response tokens should not pass rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], log_probs=[-0.1], trainable=False)
