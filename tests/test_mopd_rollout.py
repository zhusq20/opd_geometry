import asyncio
import copy
import json
import logging
import subprocess
from types import SimpleNamespace

import pytest

import train
from slime.ray.rollout import _assemble_mopd_feedback
from slime.utils.types import Sample
from slime_plugins.mopd import teacher_slot
from slime_plugins.mopd.rollout import _paired_sampling_seed
from slime_plugins.mopd.sampler import TASKS
from slime_plugins.mopd.teacher_slot import _slot_config

NUM_GPUS = 0


class _CheckpointActor:
    def __init__(self):
        self.events = []

    def save_model(self, rollout_id, force_sync=False):
        self.events.append((rollout_id, force_sync))


def test_mopd_checkpoint_is_synchronous_on_the_separate_training_gpu():
    actor = _CheckpointActor()
    train._save_mopd_actor_checkpoint(SimpleNamespace(offload_rollout=False), SimpleNamespace(), actor, rollout_id=49)
    assert actor.events == [(49, True)]


def test_checkpoint_policy_keeps_uniform_variance_states_and_latest_resume(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    entries = []
    for step in (50, 100, 250, 300, 500):
        rollout_id = step - 1
        checkpoint = checkpoint_root / f"iter_{rollout_id:07d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "common.pt").write_text("state", encoding="utf-8")
        entries.append({"rollout_id": rollout_id, "optimizer_step": step})

    train._mark_and_prune_mopd_optimizer_checkpoints(
        SimpleNamespace(save=str(checkpoint_root), mopd_allocation="uniform"), entries
    )

    assert [entry["optimizer_step"] for entry in entries if entry["optimizer_state_retained"]] == [
        50,
        250,
        500,
    ]
    assert sorted(path.name for path in checkpoint_root.glob("iter_*")) == [
        "iter_0000049",
        "iter_0000249",
        "iter_0000499",
    ]


def _sample(index: int) -> Sample:
    return Sample(
        index=index,
        metadata={
            "mopd_task": "math",
            "protocol_split": "train",
            "mopd_task_prompt_epoch": 0,
            "mopd_task_prompt_ordinal": 7,
            "mopd_response_ordinal": 0,
        },
    )


def test_paired_seed_depends_on_task_local_request_not_global_sample_index():
    args = SimpleNamespace(mopd_seed=42)
    first, second = _sample(3), _sample(999)
    second.metadata = copy.deepcopy(first.metadata)
    assert _paired_sampling_seed(args, first) == _paired_sampling_seed(args, second)
    second.metadata["mopd_task_prompt_ordinal"] += 1
    assert _paired_sampling_seed(args, first) != _paired_sampling_seed(args, second)


def test_teacher_router_requires_four_distinct_resident_endpoints(tmp_path):
    teachers = {
        task: {"url": f"http://localhost:{31001 + index}/generate", "model_path": f"/{task}"}
        for index, task in enumerate(TASKS)
    }
    path = tmp_path / "router.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "resident_pool": {"gpu_count": 1, "physical_gpu": "7"},
                "teachers": teachers,
            }
        )
    )
    config, pool = _slot_config(SimpleNamespace(opd_teacher_router_config=str(path)))
    assert pool == {"gpu_count": 1, "physical_gpu": "7"}
    assert len({route["url"] for route in config["teachers"].values()}) == 4

    teachers["science"]["url"] = teachers["code"]["url"]
    bad_path = tmp_path / "bad_router.json"
    bad_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "resident_pool": {"gpu_count": 1, "physical_gpu": "7"},
                "teachers": teachers,
            }
        )
    )
    with pytest.raises(ValueError, match="distinct endpoints"):
        _slot_config(SimpleNamespace(opd_teacher_router_config=str(bad_path)))


def test_teacher_memory_probe_retries_and_recovers(monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess([], 255, "Failed to initialize NVML: Unknown Error\n", ""),
            subprocess.CompletedProcess([], 0, "15159\n", ""),
        ]
    )
    calls = []
    monkeypatch.setattr(teacher_slot, "_slot_config", lambda _args: ({}, {"physical_gpu": "7"}))
    monkeypatch.setattr(teacher_slot, "_teacher_memory_probe_retry_after", {})
    monkeypatch.setattr(teacher_slot, "_TEACHER_MEMORY_PROBE_RETRY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(
        teacher_slot.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or next(results),
    )
    assert teacher_slot.teacher_memory_mib(SimpleNamespace()) == 15159.0
    assert len(calls) == 2


def test_teacher_memory_probe_failure_is_nonfatal_and_rate_limited(monkeypatch, caplog):
    calls = []
    failure = subprocess.CompletedProcess([], 255, "Failed to initialize NVML: Unknown Error\n", "")
    monkeypatch.setattr(teacher_slot, "_slot_config", lambda _args: ({}, {"physical_gpu": "7"}))
    monkeypatch.setattr(teacher_slot, "_teacher_memory_probe_retry_after", {})
    monkeypatch.setattr(teacher_slot, "_TEACHER_MEMORY_PROBE_RETRY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(
        teacher_slot.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or failure,
    )
    with caplog.at_level(logging.WARNING, logger=teacher_slot.__name__):
        assert teacher_slot.teacher_memory_mib(SimpleNamespace()) is None
        assert teacher_slot.teacher_memory_mib(SimpleNamespace()) is None
    assert len(calls) == teacher_slot._TEACHER_MEMORY_PROBE_ATTEMPTS
    assert "Teacher HBM telemetry unavailable" in caplog.text


def test_malformed_teacher_payload_is_converted_to_numeric_penalty(monkeypatch):
    async def malformed(_args, samples, **_kwargs):
        return [{"meta_info": {"input_token_logprobs": [[0.0, 0]]}} for _ in samples]

    measurements = iter([1000.0, 1100.0])
    monkeypatch.setattr(teacher_slot, "_slot_config", lambda _args: ({}, {}))
    monkeypatch.setattr(teacher_slot, "teacher_memory_mib", lambda _args: next(measurements))
    monkeypatch.setattr(teacher_slot, "teacher_reward", malformed)
    sample = Sample(tokens=[1, 2], response_length=2, rollout_log_probs=[-1.0, -2.0], loss_mask=[1, 1])
    result = asyncio.run(teacher_slot.score_teacher_samples(SimpleNamespace(), "math", [sample], failure_penalty=10.0))
    assert result["teacher_scoring_failures"] == 1
    assert result["teacher_memory_mib"] == 1100.0
    assert sample.metadata["mopd_failure_kind"] == "teacher_payload"
    assert sample.metadata["mopd_failure_penalty"] == 10.0
    assert sample.reward["meta_info"]["input_token_logprobs"][1][0] == -11.0


def _rollout_metrics():
    metrics = {
        "mopd/student_rollout_gpu_seconds": 3.0,
        "mopd/teacher_gpu_seconds": 2.0,
        "mopd/student_rollout_wall_seconds": 3.0,
        "mopd/teacher_wall_seconds": 2.0,
        "mopd/rollout_and_teacher_wall_seconds": 5.0,
        "mopd/reward_wall_seconds": 0.0,
        "mopd/valid_response_tokens": 1000,
        "mopd/generated_tokens": 1100,
        "mopd/teacher_scored_tokens": 2000,
        "mopd/prompt_count": 64,
        "mopd/completed_responses": 64,
        "mopd/truncated_responses": 0,
        "mopd/invalid_responses": 0,
        "mopd/teacher_peak_memory_mib": 43000.0,
        "mopd/teacher_memory_probe_failures": 0,
    }
    for task in TASKS:
        prefix = f"mopd/task/{task}/"
        metrics.update(
            {
                prefix + "student_rollout_seconds": 0.75,
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
                prefix + "teacher_memory_mib": 43000.0,
                prefix + "teacher_memory_probe_failures": 0,
            }
        )
    return metrics


def test_feedback_uses_two_allocated_gpus_and_microbatch_costs():
    trainer = {
        "mopd": True,
        "operation": "train",
        "attempted_responses": 64,
        "task_units": [
            {
                "task": task,
                "microbatches": 4,
                "actor_forward_backward_wall_seconds": 0.4,
                "actor_forward_backward_gpu_seconds": 0.4,
                "optimizer_wall_seconds": 0.1,
                "optimizer_gpu_seconds": 0.1,
            }
            for task in TASKS
        ],
        "actor_forward_backward_wall_seconds": 1.6,
        "actor_forward_backward_gpu_seconds": 1.6,
        "optimizer_wall_seconds": 1.0,
        "optimizer_gpu_seconds": 1.0,
        "driver_step_wall_seconds": 10.0,
        "peak_hbm_bytes": 8 * 2**30,
    }
    feedback = _assemble_mopd_feedback(
        SimpleNamespace(rollout_num_gpus=1, actor_num_nodes=1, actor_num_gpus_per_node=1),
        _rollout_metrics(),
        trainer,
    )
    assert feedback["allocated_gpu_count"] == 2
    assert feedback["total_gpu_seconds"] == pytest.approx(20.0)
    assert feedback["fixed_seconds"] == pytest.approx(6.0)
    assert [unit["microbatch_seconds"] for unit in feedback["task_units"]] == pytest.approx([0.25] * 4)
    assert [unit["student_rollout_seconds"] for unit in feedback["task_units"]] == pytest.approx([0.75] * 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
