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

NUM_GPUS = 0


def _mopd_stage_metrics():
    return {
        "mopd/student_rollout_gpu_seconds": 4.0,
        "mopd/teacher_gpu_seconds": 2.0,
        "mopd/student_rollout_wall_seconds": 1.0,
        "mopd/teacher_wall_seconds": 2.0,
        "mopd/rollout_and_teacher_wall_seconds": 2.0,
        "mopd/reward_wall_seconds": 0.5,
        "mopd/teacher_pool_gpu_count": 1,
        "mopd/valid_response_tokens": 65_600,
        "mopd/teacher_scored_tokens": 70_000,
        "mopd/prompt_count": 64,
        "mopd/completed_responses": 63,
        "mopd/truncated_responses": 1,
        "mopd/invalid_responses": 0,
        "mopd/teacher_peak_memory_mib": 4096,
        "mopd/resident_teacher_after_index": 0,
        "mopd/task/math/student_rollout_seconds": 1.0,
        "mopd/task/math/teacher_ready_seconds": 1.5,
        "mopd/task/math/teacher_scoring_seconds": 0.5,
        "mopd/task/math/teacher_switch_seconds": 1.5,
        "mopd/task/math/teacher_load_seconds": 1.5,
        "mopd/task/math/teacher_offload_seconds": 0.0,
        "mopd/task/math/prompt_count": 16,
        "mopd/task/math/attempted_responses": 64,
        "mopd/task/math/valid_response_tokens": 65_600,
        "mopd/task/math/teacher_scored_tokens": 70_000,
        "mopd/task/math/completed_responses": 63,
        "mopd/task/math/truncated_responses": 1,
        "mopd/task/math/invalid_responses": 0,
        "mopd/task/math/empty_responses": 0,
        "mopd/task/math/teacher_scoring_failures": 0,
        "mopd/task/math/teacher_memory_mib": 4096,
        "mopd/task/math/teacher_transfer_tail_seconds": 1.5,
        "mopd/task/math/switched": 1,
    }


def _mopd_trainer_feedback():
    return {
        "mopd": True,
        "driver_step_wall_seconds": 10.0,
        "actor_forward_backward_gpu_seconds": 12.0,
        "optimizer_gpu_seconds": 4.0,
        "actor_forward_backward_wall_seconds": 3.0,
        "optimizer_wall_seconds": 1.0,
        "operation": "train",
        "task_units": [
            {
                "task": "math",
                "actor_forward_backward_wall_seconds": 3.0,
                "optimizer_wall_seconds": 1.0,
            }
        ],
        "peak_hbm_bytes": 8 * 2**30,
    }


@pytest.mark.unit
def test_mopd_cost_accounting_uses_end_to_end_wall_time_and_all_resident_gpus():
    args = SimpleNamespace(rollout_num_gpus=4, actor_num_nodes=1, actor_num_gpus_per_node=4)
    feedback = _assemble_mopd_feedback(args, _mopd_stage_metrics(), _mopd_trainer_feedback())

    assert feedback["component_wall_seconds"] == pytest.approx(6.5)
    assert feedback["total_step_seconds"] == pytest.approx(10.0)
    assert feedback["active_gpu_seconds"] == pytest.approx(22.0)
    assert feedback["allocated_gpu_count"] == 5
    assert feedback["total_gpu_seconds"] == pytest.approx(50.0)
    assert feedback["valid_response_tokens"] == 65_600


@pytest.mark.unit
def test_mopd_cost_accounting_rejects_a_driver_clock_shorter_than_critical_path():
    args = SimpleNamespace(rollout_num_gpus=4, actor_num_nodes=1, actor_num_gpus_per_node=4)
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
