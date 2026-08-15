"""Regression tests for scalar rollout reward metrics."""

import json
from types import SimpleNamespace

import pytest

from slime.ray.rollout import _compute_training_reward_metrics, _compute_zero_std_metrics, _save_eval_artifacts
from slime.utils.types import Sample
from slime_plugins.geometry.rollout_samples import persist_rollout_samples

NUM_GPUS = 0


@pytest.mark.unit
def test_zero_std_metrics_skip_opd_teacher_payloads():
    args = SimpleNamespace(advantage_estimator="grpo", reward_key=None)
    samples = [
        Sample(group_index=0, reward={"meta_info": {"input_token_logprobs": []}}),
        Sample(group_index=1, reward={"meta_info": {"input_token_logprobs": []}}),
    ]

    assert _compute_zero_std_metrics(args, samples) == {}


@pytest.mark.unit
def test_zero_std_metrics_keep_scalar_task_rewards():
    args = SimpleNamespace(advantage_estimator="grpo", reward_key=None)
    samples = [
        Sample(group_index=0, reward=1.0),
        Sample(group_index=0, reward=1.0),
        Sample(group_index=1, reward=0.0),
        Sample(group_index=1, reward=1.0),
    ]

    assert _compute_zero_std_metrics(args, samples) == {"zero_std/count_1.0": 1}


@pytest.mark.unit
def test_task_reward_metrics_separate_observation_from_loss_use():
    samples = [
        Sample(reward={"teacher": {}, "task_reward": 1.0}, metadata={"task_name": "math"}),
        Sample(reward={"teacher": {}, "task_reward": 0.0}, metadata={"task_name": "math"}),
    ]
    pure_opd = SimpleNamespace(
        reward_key=None,
        use_opd=True,
        opd_task_reward_weight=0.0,
    )

    metrics = _compute_training_reward_metrics(pure_opd, samples)

    assert metrics["reward/math/mean"] == pytest.approx(0.5)
    assert metrics["reward/math/std"] == pytest.approx(0.5)
    assert metrics["reward/math/p10"] == pytest.approx(0.1)
    assert metrics["reward/math/p90"] == pytest.approx(0.9)
    assert metrics["reward/math/pass_rate"] == pytest.approx(0.5)
    assert metrics["task_reward_observed"] == 1
    assert metrics["reward_used_in_loss"] == 0
    assert metrics["reward_loss_coefficient"] == 0.0


@pytest.mark.unit
def test_task_reward_metrics_use_native_metadata_value_before_weighted_reward():
    sample = Sample(
        reward=0.25,
        metadata={"task_name": "code", "task_reward_observed": 1.0},
    )
    args = SimpleNamespace(reward_key=None, use_opd=True, opd_task_reward_weight=0.25)

    metrics = _compute_training_reward_metrics(args, [sample])

    assert metrics["reward/code/mean"] == 1.0
    assert metrics["reward_used_in_loss"] == 1
    assert metrics["reward_loss_coefficient"] == pytest.approx(0.25)


@pytest.mark.unit
def test_removed_task_reward_is_observed_but_not_used():
    sample = Sample(reward=1.0, remove_sample=True)
    args = SimpleNamespace(reward_key=None, use_opd=False)

    metrics = _compute_training_reward_metrics(args, [sample])

    assert metrics["task_reward_observed"] == 1
    assert metrics["reward_used_in_loss"] == 0
    assert metrics["reward_loss_coefficient"] == 0.0


@pytest.mark.unit
def test_training_rollout_samples_are_durable_complete_and_idempotent(tmp_path):
    args = SimpleNamespace(
        geometry_output_dir=str(tmp_path),
        experiment_name="paper-run",
        experiment_task="multi",
        seed=17,
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        global_batch_size=2,
        reward_key=None,
        use_opd=True,
        opd_task_reward_weight=0.0,
    )
    sample = Sample(
        index=9,
        group_index=4,
        rollout_id=9,
        prompt="2 + 2 = ?",
        response="4",
        response_length=1,
        loss_mask=[1],
        label="4",
        reward={"teacher": {}, "task_reward": 1.0},
        status=Sample.Status.COMPLETED,
        metadata={"task_name": "math", "prompt_id": "math-4", "sample_id": "math-4-0"},
    )

    first = persist_rollout_samples(3, args, [sample])
    second = persist_rollout_samples(3, args, [sample])

    assert first == second
    path = tmp_path / "rollout" / "samples" / "rollout_00000003.jsonl"
    record = json.loads(path.read_text())
    assert record["task"] == "math"
    assert record["prompt_id"] == "math-4"
    assert record["sample_id"] == "math-4-0"
    assert record["prompt"] == "2 + 2 = ?"
    assert record["response"] == "4"
    assert record["label"] == "4"
    assert record["reward"] == 1.0
    assert record["passed"] is True
    assert record["num_updates"] == 3
    assert record["model_version"] == 3
    assert record["task_reward_observed"] is True
    assert record["reward_used_in_loss"] is False
    assert record["reward_loss_coefficient"] == 0.0

    sample.response = "divergent replay"
    with pytest.raises(FileExistsError, match="divergent rollout artifact"):
        persist_rollout_samples(3, args, [sample])


@pytest.mark.unit
def test_eval_artifacts_never_silently_overwrite_a_checkpoint_probe(tmp_path):
    args = SimpleNamespace(eval_artifact_dir=str(tmp_path), eval_datasets=[])
    sample = Sample(prompt="p", response="r", response_length=1, reward=1.0)
    data = {"heldout": {"samples": [sample], "rewards": [1.0]}}
    metrics = {"eval/num_updates": 7, "eval/model_version": 7, "eval/phase": "final"}

    _save_eval_artifacts(args, 6, data, metrics)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _save_eval_artifacts(args, 6, data, metrics)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
