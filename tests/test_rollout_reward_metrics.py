"""Regression tests for scalar rollout reward metrics."""

from types import SimpleNamespace

import pytest

from slime.ray.rollout import (
    _assemble_mopd_feedback,
    _compute_training_reward_metrics,
    _compute_zero_std_metrics,
    _save_eval_artifacts,
)
from slime.utils.types import Sample
from slime_plugins.mopd.sampler import TASKS

NUM_GPUS = 0


def _mopd_stage_metrics():
    metrics = {
        "mopd/student_rollout_gpu_seconds": 1.0,
        "mopd/teacher_gpu_seconds": 2.0,
        "mopd/student_rollout_wall_seconds": 1.0,
        "mopd/teacher_wall_seconds": 2.0,
        "mopd/rollout_and_teacher_wall_seconds": 3.0,
        "mopd/reward_wall_seconds": 0.5,
        "mopd/valid_response_tokens": 1_000,
        "mopd/generated_tokens": 1_100,
        "mopd/teacher_scored_tokens": 2_000,
        "mopd/prompt_count": 64,
        "mopd/completed_responses": 63,
        "mopd/truncated_responses": 1,
        "mopd/invalid_responses": 0,
        "mopd/teacher_peak_memory_mib": 4096,
    }
    for task in TASKS:
        prefix = f"mopd/task/{task}/"
        metrics.update(
            {
                prefix + "student_rollout_seconds": 0.25,
                prefix + "teacher_scoring_seconds": 0.5,
                prefix + "prompt_count": 16,
                prefix + "attempted_responses": 16,
                prefix + "valid_response_tokens": 250,
                prefix + "generated_tokens": 275,
                prefix + "teacher_scored_tokens": 500,
                prefix + "completed_responses": 16,
                prefix + "truncated_responses": 0,
                prefix + "invalid_responses": 0,
                prefix + "empty_responses": 0,
                prefix + "teacher_scoring_failures": 0,
                prefix + "teacher_memory_mib": 4096,
                prefix + "teacher_memory_probe_failures": 0,
            }
        )
    return metrics


def _mopd_trainer_feedback():
    return {
        "mopd": True,
        "driver_step_wall_seconds": 10.0,
        "actor_forward_backward_gpu_seconds": 3.0,
        "optimizer_gpu_seconds": 1.0,
        "actor_forward_backward_wall_seconds": 3.0,
        "optimizer_wall_seconds": 1.0,
        "operation": "train",
        "task_units": [
            {
                "task": task,
                "microbatches": 4,
                "actor_forward_backward_wall_seconds": 0.75,
                "optimizer_wall_seconds": 0.125,
            }
            for task in TASKS
        ],
        "peak_hbm_bytes": 8 * 2**30,
    }


@pytest.mark.unit
def test_mopd_cost_accounting_uses_end_to_end_wall_time_and_all_resident_gpus():
    args = SimpleNamespace(rollout_num_gpus=1, actor_num_nodes=1, actor_num_gpus_per_node=1)
    feedback = _assemble_mopd_feedback(args, _mopd_stage_metrics(), _mopd_trainer_feedback())

    assert feedback["component_wall_seconds"] == pytest.approx(7.5)
    assert feedback["total_step_seconds"] == pytest.approx(10.0)
    assert feedback["active_gpu_seconds"] == pytest.approx(7.0)
    assert feedback["allocated_gpu_count"] == 2
    assert feedback["total_gpu_seconds"] == pytest.approx(20.0)
    assert feedback["valid_response_tokens"] == 1_000


@pytest.mark.unit
def test_mopd_cost_accounting_rejects_a_driver_clock_shorter_than_critical_path():
    args = SimpleNamespace(rollout_num_gpus=1, actor_num_nodes=1, actor_num_gpus_per_node=1)
    feedback = _mopd_trainer_feedback()
    feedback["driver_step_wall_seconds"] = 6.0
    with pytest.raises(ValueError, match="shorter than"):
        _assemble_mopd_feedback(args, _mopd_stage_metrics(), feedback)


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
