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
from slime_plugins.m2rl import opd
from slime_plugins.mopd import rollout as mopd_rollout
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
        250,
        500,
    ]
    assert sorted(path.name for path in checkpoint_root.glob("iter_*")) == [
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


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("k", [16, 64])
@pytest.mark.parametrize(
    "enabled,loss,expected_topk",
    [(False, "teacher_topk", None), (True, "sampled_reverse_kl", None), (True, "teacher_topk", 64),
     (False, "student_topk", None), (True, "student_topk", None),
     (False, "topk_intersection", None), (True, "topk_intersection", "k")],
)
def test_dense_teacher_targets_are_requested_only_for_enabled_mopd(monkeypatch, batch, enabled, loss, expected_topk, k):
    routes = []

    async def request(session, semaphore, route, sample):
        routes.append(route)
        return {"meta_info": {}}

    monkeypatch.setattr(opd, "_teacher_request", request)
    args = SimpleNamespace(mopd_enabled=enabled, mopd_loss=loss, mopd_topk=k, rm_url="http://teacher/generate")
    samples = [_sample(0), _sample(1)] if batch else _sample(0)
    asyncio.run(opd.teacher_reward(args, samples))
    assert len(routes) == (2 if batch else 1)
    assert all(route.get("top_logprobs_num") == (k if expected_topk == "k" else expected_topk) for route in routes)
    assert all(bool(route.get("student_topk")) == (enabled and loss == "student_topk") for route in routes)
    assert all(route.get("student_topk_k") == (k if enabled and loss == "student_topk" else None) for route in routes)


def test_paired_seed_depends_on_task_local_request_not_global_sample_index():
    args = SimpleNamespace(mopd_seed=42)
    first, second = _sample(3), _sample(999)
    second.metadata = copy.deepcopy(first.metadata)
    assert _paired_sampling_seed(args, first) == _paired_sampling_seed(args, second)
    second.metadata["mopd_task_prompt_ordinal"] += 1
    assert _paired_sampling_seed(args, first) != _paired_sampling_seed(args, second)


def test_pytorch_sampling_keeps_request_seed_without_batch_invariant_kernels(monkeypatch):
    args = SimpleNamespace(mopd_seed=42, sglang_enable_deterministic_inference=False,
                           sglang_sampling_backend="pytorch")
    sample = _sample(0)
    requested = []

    async def generate(_args, value, sampling_params):
        requested.append(sampling_params)
        value.tokens, value.response_length, value.rollout_log_probs = [1, 2], 1, [-0.5]
        return value

    monkeypatch.setattr(mopd_rollout, "GenerateState", lambda _args: SimpleNamespace(sampling_params={"temperature": 1.0}))
    monkeypatch.setattr(mopd_rollout, "active_tasks", lambda _args: ("math",))
    monkeypatch.setattr(mopd_rollout, "_generate_one", generate)
    generated, _, _ = asyncio.run(mopd_rollout._generate_samples(args, [sample]))
    assert generated == [sample]
    assert requested == [{"temperature": 1.0, "sampling_seed": _paired_sampling_seed(args, sample)}]


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


def test_teacher_router_supports_task_specific_resident_gpus(tmp_path):
    teachers = {
        task: {"url": f"http://localhost:{31301 + index}/generate", "model_path": f"/{task}"}
        for index, task in enumerate(TASKS)
    }
    path = tmp_path / "distributed_router.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "resident_pool": {
                    "gpu_count": 2,
                    "physical_gpus": {"math": "8", "code": "9", "if": "8", "science": "9"},
                },
                "teachers": teachers,
            }
        )
    )
    config, pool = _slot_config(SimpleNamespace(opd_teacher_router_config=str(path)))
    assert pool["gpu_count"] == 2
    assert pool["physical_gpus"]["math"] == "8"
    assert len({route["url"] for route in config["teachers"].values()}) == 4


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
    monkeypatch.setattr(teacher_slot, "teacher_memory_mib", lambda _args, _task=None: next(measurements))
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


def _trainer_feedback():
    return {
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


def test_feedback_uses_two_allocated_gpus_and_microbatch_costs():
    feedback = _assemble_mopd_feedback(
        SimpleNamespace(rollout_num_gpus=1, actor_num_nodes=1, actor_num_gpus_per_node=1),
        _rollout_metrics(),
        _trainer_feedback(),
    )
    assert feedback["allocated_gpu_count"] == 2
    assert feedback["total_gpu_seconds"] == pytest.approx(20.0)
    assert feedback["fixed_seconds"] == pytest.approx(6.0)
    assert [unit["microbatch_seconds"] for unit in feedback["task_units"]] == pytest.approx([0.25] * 4)
    assert [unit["student_rollout_seconds"] for unit in feedback["task_units"]] == pytest.approx([0.75] * 4)


@pytest.mark.parametrize("reward_seconds", [0.0, 50.0])
def test_generated_feedback_counts_reward_verification_once(monkeypatch, reward_seconds):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(mopd_rollout, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    monkeypatch.setattr(mopd_rollout, "active_tasks", lambda _args: TASKS)
    monkeypatch.setattr(mopd_rollout, "GenerateState", lambda _args: SimpleNamespace(reset=lambda: None))
    samples = []
    for task in TASKS:
        for _ in range(16):
            sample = _sample(len(samples))
            sample.metadata["mopd_task"] = task
            sample.tokens, sample.response_length, sample.loss_mask = [1, 2], 2, [1, 1]
            sample.status = Sample.Status.COMPLETED
            samples.append(sample)

    async def generate(_args, pending):
        clock.now += 3.0
        return pending, 3.0, {task: 3.0 for task in TASKS}

    async def score(_args, _task, task_samples, **_kwargs):
        clock.now += 0.5
        return {
            "teacher_scoring_seconds": 0.5,
            "teacher_scoring_failures": 0,
            "teacher_scored_tokens": 2 * len(task_samples),
            "teacher_memory_mib": 1000.0,
            "teacher_memory_probe_failures": 0,
        }

    async def reward(_args, _samples):
        clock.now += reward_seconds
        return reward_seconds

    monkeypatch.setattr(mopd_rollout, "_generate_samples", generate)
    monkeypatch.setattr(mopd_rollout, "score_teacher_samples", score)
    monkeypatch.setattr(mopd_rollout, "_observe_task_rewards", reward)
    source = SimpleNamespace(
        prompt_count_for_rollout=lambda _rollout_id: 64,
        get_samples=lambda _count: [[sample] for sample in samples],
        controller=SimpleNamespace(pending={"counts": dict.fromkeys(TASKS, 4), "attempted_responses": 64}),
    )
    result = asyncio.run(mopd_rollout._generate_train(SimpleNamespace(mopd_failure_penalty=10.0), 1, source))
    trainer = _trainer_feedback()
    trainer["driver_step_wall_seconds"] = clock.now + 1.6 + 1.0
    feedback = _assemble_mopd_feedback(
        SimpleNamespace(rollout_num_gpus=1, actor_num_nodes=1, actor_num_gpus_per_node=1),
        result.metrics,
        trainer,
    )
    assert result.metrics["mopd/rollout_and_teacher_wall_seconds"] == pytest.approx(5.0)
    assert feedback["reward_wall_seconds"] == pytest.approx(reward_seconds)
    assert feedback["component_wall_seconds"] == pytest.approx(7.6 + reward_seconds)
    assert feedback["total_step_seconds"] == pytest.approx(7.6 + reward_seconds)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
