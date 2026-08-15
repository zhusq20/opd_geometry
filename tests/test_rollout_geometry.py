"""CPU tests for exact sampled-action OPD/RL rollout distributions."""

import json
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401,E402 - install CPU Megatron stubs before importing slime
import pytest
import torch

from slime.backends.megatron_utils import rollout_geometry

NUM_GPUS = 0


@pytest.mark.unit
def test_distribution_statistics_are_population_exact():
    metrics = rollout_geometry.distribution_statistics(torch.tensor([-2.0, 0.0, 2.0, 4.0]))

    assert metrics["mean"] == pytest.approx(1.0)
    assert metrics["std"] == pytest.approx((5.0) ** 0.5)
    assert metrics["p50"] == pytest.approx(1.0)
    assert metrics["negative_fraction"] == pytest.approx(0.25)
    assert metrics["l2"] == pytest.approx((24.0) ** 0.5)
    assert metrics["rms"] == pytest.approx(6.0**0.5)
    assert metrics["max_abs"] == 4.0


@pytest.mark.unit
def test_payload_summary_merges_cp_shards_before_sequence_statistics():
    def item(key, values, positions, *, source="math", reward=None):
        return {
            "key": key,
            "source": source,
            "response_length": 4,
            "truncated": False,
            "task_reward": reward,
            "values": {
                "sampled_reverse_kl_logratio": torch.tensor(values),
                "advantage": torch.tensor(values),
                "valid_token": torch.ones(len(values)),
            },
            "position": {
                "sampled_reverse_kl_logratio": (positions, [1 if value else 0 for value in positions]),
            },
        }

    first_key = (0, 5, "7", 0)
    second_key = (1, 5, "8", 0)
    payloads = [
        {"samples": [item(first_key, [-1.0, 1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])]},
        {"samples": [item(first_key, [-2.0, 2.0], [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0])]},
        {
            "samples": [
                item(
                    second_key,
                    [0.0, 4.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
                    source="code",
                    reward=0.5,
                )
            ]
        },
    ]

    metrics, availability = rollout_geometry.summarize_payloads(payloads)

    assert metrics["sampled_reverse_kl_logratio/token/count"] == 6
    assert metrics["sampled_reverse_kl_logratio/token/mean"] == pytest.approx(4 / 6)
    assert metrics["sampled_reverse_kl_logratio/token/negative_fraction"] == pytest.approx(2 / 6)
    # CP pieces of the first sample must form one sequence with mean 0; the
    # second sample mean is 2, so the sequence-level distribution is [0, 2].
    assert metrics["sampled_reverse_kl_logratio/sequence_mean/count"] == 2
    assert metrics["sampled_reverse_kl_logratio/sequence_mean/mean"] == pytest.approx(1.0)
    assert metrics["sampled_reverse_kl_logratio/sequence_mean/std"] == pytest.approx(1.0)
    assert metrics["response/valid_length/mean"] == pytest.approx(3.0)
    assert metrics["source_count/math"] == 1
    assert metrics["source_count/code"] == 1
    assert metrics["task_reward/pass_rate"] == 0.0
    assert availability["task_reward_observed"] is True
    assert availability["task_reward_on_valid_sample"] is True
    assert availability["sample_gradient_coherence"] == "not_collected_requires_per_sample_backward"
    assert availability["mixed_loss_gradient_geometry"] == "not_collected_requires_component_backward"


@pytest.mark.unit
def test_local_payload_uses_sampled_action_name_and_exact_clip_masks(monkeypatch):
    monkeypatch.setattr(rollout_geometry.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rollout_geometry, "_dp_sample_rank", lambda: 0)
    args = SimpleNamespace(
        eps_clip=0.2,
        eps_clip_high=0.2,
        advantage_estimator="cispo",
        use_tis=True,
        get_mismatch_metrics=False,
        tis_clip_low=0.5,
        tis_clip=1.5,
    )
    rollout_data = {
        "response_lengths": [3],
        "total_lengths": [5],
        "loss_masks": [torch.tensor([1, 0, 1])],
        "log_probs": [torch.tensor([0.0, 0.0, 1.0])],
        "rollout_log_probs": [torch.zeros(3)],
        "teacher_log_probs": [torch.tensor([0.5, 0.0, 0.5])],
        "sampled_reverse_kl_logratio": [torch.tensor([-0.5, 0.0, 0.5])],
        "advantages": [torch.tensor([1.0, 1.0, -1.0])],
        "entropy": [torch.tensor([0.1, 0.2, 0.3])],
        "source_names": ["science"],
        "sample_indices": [11],
        "rollout_ids": [4],
        "truncated": [1],
        "raw_reward": [{"task_reward": 0.75}],
    }

    payload = rollout_geometry.build_local_payload(args, rollout_data)
    values = payload["samples"][0]["values"]

    assert values["sampled_reverse_kl_logratio"].tolist() == pytest.approx([-0.5, 0.5])
    assert values["importance_ratio"].tolist() == pytest.approx([1.0, torch.exp(torch.tensor(1.0)).item()])
    assert values["policy_clip"].tolist() == [0.0, 1.0]
    assert values["tis_clip"].tolist() == [0.0, 1.0]
    assert payload["samples"][0]["task_reward"] == pytest.approx(0.75)


@pytest.mark.unit
def test_local_payload_prefers_dp_local_raw_rewards(monkeypatch):
    monkeypatch.setattr(rollout_geometry.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rollout_geometry, "_dp_sample_rank", lambda: 2)
    rollout_data = {
        "response_lengths": [1, 2],
        "total_lengths": [3, 5],
        "loss_masks": [torch.ones(1), torch.ones(2)],
        # log_passrate deliberately retains the whole rollout batch on every
        # DP rank, while geometry must pair rewards with this rank's samples.
        "raw_reward": [0.0, 0.1, 0.2, 0.3],
        "local_raw_reward": [0.2, 0.3],
    }

    payload = rollout_geometry.build_local_payload(SimpleNamespace(), rollout_data)

    assert [sample["task_reward"] for sample in payload["samples"]] == pytest.approx([0.2, 0.3])


@pytest.mark.unit
def test_rollout_geometry_resume_frontier_reads_only_the_last_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "rollout_id": rollout_id,
                    "cumulative_prompt_count": rollout_id + 10,
                    "cumulative_effective_token_count": rollout_id + 100,
                }
            )
            + "\n"
            for rollout_id in range(200)
        )
    )

    assert rollout_geometry._previous_cumulative(path, 200) == (209, 299)
    with pytest.raises(ValueError, match="replayed rollout 199"):
        rollout_geometry._previous_cumulative(path, 199)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
