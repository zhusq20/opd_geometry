import copy
import asyncio
import json
from types import SimpleNamespace

import pytest

import train
from slime.ray.rollout import _assemble_mopd_feedback
from slime.utils.types import Sample
from slime_plugins.mopd.rollout import _paired_sampling_seed
from slime_plugins.mopd import teacher_slot
from slime_plugins.mopd.teacher_slot import _slot_config


class _RemoteCall:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def remote(self):
        self.events.append(self.name)
        return self.name


class _CheckpointRolloutManager:
    def __init__(self, events):
        self.offload = _RemoteCall(events, "offload")
        self.onload_weights = _RemoteCall(events, "onload_weights")
        self.onload_kv = _RemoteCall(events, "onload_kv")


class _CheckpointActor:
    def __init__(self, events, failure=None):
        self.events = events
        self.failure = failure

    def save_model(self, rollout_id, force_sync=False):
        self.events.append(("save", rollout_id, force_sync))
        if self.failure is not None:
            raise self.failure

    def update_weights(self):
        self.events.append("update_weights")


@pytest.mark.parametrize("failure", [None, RuntimeError("save failed")])
def test_mopd_checkpoint_releases_and_restores_colocated_rollout(monkeypatch, failure):
    events = []
    monkeypatch.setattr(train.ray, "get", lambda value: value)
    args = SimpleNamespace(offload_rollout=True)
    rollout_manager = _CheckpointRolloutManager(events)
    actor_model = _CheckpointActor(events, failure=failure)

    if failure is None:
        train._save_mopd_actor_checkpoint(args, rollout_manager, actor_model, rollout_id=7)
    else:
        with pytest.raises(RuntimeError, match="save failed"):
            train._save_mopd_actor_checkpoint(args, rollout_manager, actor_model, rollout_id=7)

    assert events == [
        "offload",
        ("save", 7, True),
        "onload_weights",
        "update_weights",
        "onload_kv",
    ]


def _sample(index: int) -> Sample:
    return Sample(
        index=index,
        metadata={
            "mopd_task": "math",
            "protocol_split": "train",
            "mopd_task_prompt_epoch": 0,
            "mopd_task_prompt_ordinal": 7,
            "mopd_response_ordinal": 2,
        },
    )


def test_paired_seed_depends_on_task_local_request_not_global_sample_index():
    args = SimpleNamespace(mopd_seed=42)
    first, second = _sample(3), _sample(999)
    second.metadata = copy.deepcopy(first.metadata)
    assert _paired_sampling_seed(args, first) == _paired_sampling_seed(args, second)
    second.metadata["mopd_task_prompt_ordinal"] += 1
    assert _paired_sampling_seed(args, first) != _paired_sampling_seed(args, second)


def test_teacher_router_requires_one_shared_single_gpu_slot(tmp_path):
    teachers = {
        task: {
            "url": "http://localhost:31001/generate",
            "control_url": "http://localhost:31001/update_weights_from_disk",
            "model_path": f"/{task}",
            "weight_version": task,
        }
        for task in ("math", "code", "if", "science")
    }
    path = tmp_path / "router.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slot": {"gpu_count": 1, "physical_gpu": "4", "state_path": str(tmp_path / "state")},
                "teachers": teachers,
            }
        )
    )
    config, slot = _slot_config(SimpleNamespace(opd_teacher_router_config=str(path)))
    assert slot["gpu_count"] == 1
    assert len({route["url"] for route in config["teachers"].values()}) == 1
    teachers["science"]["url"] = "http://localhost:31002/generate"
    bad_path = tmp_path / "bad_router.json"
    bad_path.write_text(json.dumps({"schema_version": 2, "slot": slot, "teachers": teachers}))
    with pytest.raises(ValueError, match="share one"):
        _slot_config(SimpleNamespace(opd_teacher_router_config=str(bad_path)))


def test_malformed_teacher_payload_is_converted_to_the_numeric_penalty(monkeypatch):
    async def fake_activate(_args, _task):
        return {
            "resident_before": "math",
            "resident_after": "math",
            "switched": False,
            "teacher_load_seconds": 0.0,
            "teacher_offload_seconds": 0.0,
            "teacher_ready_seconds": 0.0,
            "teacher_switch_seconds": 0.0,
            "teacher_memory_mib": 1000.0,
        }

    async def malformed(_args, _sample):
        return {"meta_info": {"input_token_logprobs": [[0.0, 0]]}}

    monkeypatch.setattr(teacher_slot, "activate_teacher", fake_activate)
    monkeypatch.setattr(teacher_slot, "teacher_reward", malformed)
    sample = Sample(tokens=[1, 2], response_length=2, rollout_log_probs=[-1.0, -2.0], loss_mask=[1, 1])
    result = asyncio.run(
        teacher_slot.score_teacher_samples(SimpleNamespace(), "math", [sample], failure_penalty=10.0)
    )
    assert result["teacher_scoring_failures"] == 1
    assert sample.metadata["mopd_failure_kind"] == "teacher_payload"
    assert sample.metadata["mopd_failure_penalty"] == 10.0
    assert sample.reward["meta_info"]["input_token_logprobs"][1][0] == -11.0


def _rollout_metrics():
    metrics = {
        "mopd/student_rollout_gpu_seconds": 8.0,
        "mopd/teacher_gpu_seconds": 3.0,
        "mopd/student_rollout_wall_seconds": 2.0,
        "mopd/teacher_wall_seconds": 3.0,
        "mopd/rollout_and_teacher_wall_seconds": 4.0,
        "mopd/reward_wall_seconds": 0.0,
        "mopd/valid_response_tokens": 1000,
        "mopd/teacher_scored_tokens": 1200,
        "mopd/prompt_count": 32,
        "mopd/completed_responses": 128,
        "mopd/truncated_responses": 0,
        "mopd/invalid_responses": 0,
        "mopd/teacher_pool_gpu_count": 1,
        "mopd/teacher_peak_memory_mib": 4096.0,
        "mopd/resident_teacher_after_index": 1,
    }
    for task, rollout, switch in (("math", 0.8, 0.2), ("code", 1.2, 0.3)):
        prefix = f"mopd/task/{task}/"
        metrics.update(
            {
                prefix + "student_rollout_seconds": rollout,
                prefix + "teacher_ready_seconds": switch,
                prefix + "teacher_scoring_seconds": 0.5,
                prefix + "teacher_switch_seconds": switch,
                prefix + "teacher_load_seconds": switch,
                prefix + "teacher_offload_seconds": 0.0,
                prefix + "prompt_count": 16,
                prefix + "attempted_responses": 64,
                prefix + "valid_response_tokens": 500,
                prefix + "teacher_scored_tokens": 600,
                prefix + "completed_responses": 64,
                prefix + "truncated_responses": 0,
                prefix + "invalid_responses": 0,
                prefix + "empty_responses": 0,
                prefix + "teacher_scoring_failures": 0,
                prefix + "teacher_memory_mib": 4096.0,
                prefix + "teacher_transfer_tail_seconds": switch,
                prefix + "switched": 1,
            }
        )
    return metrics


def test_feedback_accounts_one_teacher_gpu_and_per_task_service_time():
    trainer = {
        "mopd": True,
        "operation": "train",
        "task_units": [
            {"task": task, "actor_forward_backward_wall_seconds": 0.4, "optimizer_wall_seconds": 0.1}
            for task in ("math", "code")
        ],
        "actor_forward_backward_wall_seconds": 0.8,
        "actor_forward_backward_gpu_seconds": 3.2,
        "optimizer_wall_seconds": 0.4,
        "optimizer_gpu_seconds": 1.6,
        "driver_step_wall_seconds": 7.0,
        "peak_hbm_bytes": 8 * 2**30,
    }
    feedback = _assemble_mopd_feedback(
        SimpleNamespace(rollout_num_gpus=4, actor_num_nodes=1, actor_num_gpus_per_node=4),
        _rollout_metrics(),
        trainer,
    )
    assert feedback["teacher_pool_gpu_count"] == 1
    assert feedback["allocated_gpu_count"] == 5
    assert feedback["total_gpu_seconds"] == pytest.approx(35.0)
    assert feedback["resident_teacher_after"] == "code"
    assert feedback["task_units"][0]["predicted_full_task_seconds"] == pytest.approx(1.8)
    assert feedback["task_units"][1]["predicted_full_task_seconds"] == pytest.approx(2.2)
