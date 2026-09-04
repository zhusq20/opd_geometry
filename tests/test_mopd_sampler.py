import copy
import json
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.mopd.data_source import (
    MOPDRolloutDataSource,
    MOPDVarianceDataSource,
    PROTOCOL,
    VARIANCE_PROTOCOL,
    _validate_protocol_manifest,
    _validate_variance_manifest,
)
from slime_plugins.m2rl.data_source import MultiTaskRolloutDataSource
from slime.utils.types import Sample
from slime_plugins.mopd.sampler import (
    ALLOCATIONS,
    MAIN_CHECKPOINT_RESPONSES,
    MAIN_CHECKPOINT_STEPS,
    MAIN_EVAL_RESPONSES,
    MAIN_RESPONSE_BUDGET,
    MOPDController,
    SMOKE_CHECKPOINT_STEPS,
    SMOKE_EVAL_RESPONSES,
    SMOKE_RESPONSE_BUDGET,
    SMOKE_STEPS,
    TASKS,
    cost_gpas_allocation,
    d3_scheduler_mixture,
    feasible_allocations,
    largest_remainder_allocation,
)

NUM_GPUS = 0


def _sources():
    return [
        {
            "name": task,
            "teacher": task,
            "target_weight": 0.25,
            "initial_teacher_loss": 0.1 + index,
            "required_samples": 16_000,
        }
        for index, task in enumerate(TASKS)
    ]


def _feedback(plan, *, scaled=(1.0, 4.0, 9.0, 16.0), raw=(16.0, 9.0, 4.0, 1.0), loss=None):
    losses = loss or (1.0, 2.0, 3.0, 4.0)
    return {
        "attempted_responses": 64,
        "fixed_seconds": 8.0,
        "task_units": [
            {
                "task": task,
                "microbatches": plan["counts"][task],
                "scaled_noise": scaled[index],
                "raw_noise": raw[index],
                "microbatch_seconds": 1.0 + index,
                "teacher_loss": losses[index],
            }
            for index, task in enumerate(TASKS)
        ],
    }


def test_manifest_freezes_four_nonrepeating_16000_prompt_streams():
    manifest = {"version": 3, "protocol": PROTOCOL, "sampling": {"repeat": False}}
    _validate_protocol_manifest(manifest, _sources())
    wrong = _sources()
    wrong[0]["required_samples"] = 15_999
    with pytest.raises(ValueError, match="16,000"):
        _validate_protocol_manifest(manifest, wrong)
    with pytest.raises(ValueError, match="must not repeat"):
        _validate_protocol_manifest({**manifest, "sampling": {"repeat": True}}, _sources())


def test_training_source_truncates_each_seeded_stream_to_16000(tmp_path, monkeypatch):
    class Dataset:
        def __init__(self):
            self.samples = list(range(16_001))
            self.origin_samples = self.samples

        def __len__(self):
            return len(self.samples)

    sources = [SimpleNamespace(config=config, dataset=Dataset()) for config in _sources()]

    def initialize(data_source, args):
        data_source.args = args
        data_source.manifest = {
            "version": 3,
            "protocol": PROTOCOL,
            "sampling": {"repeat": False},
        }
        data_source.sources = sources

    monkeypatch.setattr(MultiTaskRolloutDataSource, "__init__", initialize)
    args = SimpleNamespace(
        mopd_checkpoint_steps="50,100,150,200,250,300,350,400,450,500",
        mopd_allocation="uniform",
        mopd_seed=42,
        mopd_ema_decay=0.9,
        mopd_total_steps=500,
        mopd_microbatches_per_step=16,
        mopd_prompts_per_microbatch=4,
        mopd_min_microbatches=2,
        mopd_max_microbatches=8,
        mopd_output_dir=str(tmp_path),
        save=str(tmp_path / "checkpoints"),
    )
    MOPDRolloutDataSource(args)
    for source in sources:
        assert len(source.dataset) == 16_000
        assert source.dataset.samples[-1] == 15_999
        assert source.dataset.origin_samples is source.dataset.samples


def test_variance_manifest_freezes_128_heldout_prompts_per_task():
    manifest = {"version": 3, "protocol": VARIANCE_PROTOCOL, "sampling": {"repeat": False}}
    sources = _sources()
    for source in sources:
        source["required_samples"] = 128
    _validate_variance_manifest(manifest, sources)
    sources[2]["required_samples"] = 127
    with pytest.raises(ValueError, match="128 prompts"):
        _validate_variance_manifest(manifest, sources)


def test_variance_source_uses_uniform_checkpoint_cost_state_and_writes_scalars(tmp_path, monkeypatch):
    sources = [
        SimpleNamespace(
            config={
                "name": task,
                "teacher": task,
                "target_weight": 0.25,
                "required_samples": 128,
            },
            dataset=[Sample(prompt=f"{task}-{index}", metadata={}) for index in range(128)],
            offset=0,
            epoch=0,
        )
        for task in TASKS
    ]

    def initialize(data_source, args):
        data_source.args = args
        data_source.manifest = {
            "version": 3,
            "protocol": VARIANCE_PROTOCOL,
            "sampling": {"repeat": False},
        }
        data_source.sources = sources
        data_source.repeat_sources = False
        data_source.sample_group_index = 0
        data_source.sample_index = 0

    monkeypatch.setattr(MultiTaskRolloutDataSource, "__init__", initialize)
    checkpoint = tmp_path / "sampler.pt"
    torch.save(
        {
            "controller": {
                "config": {"task_names": list(TASKS)},
                "task_seconds": [1.0, 2.0, 3.0, 4.0],
                "fixed_seconds": 7.0,
                "completed_steps": 50,
                "pending": None,
            }
        },
        checkpoint,
    )
    args = SimpleNamespace(
        mopd_variance_controller_state=str(checkpoint),
        mopd_variance_checkpoint_step=50,
        mopd_output_dir=str(tmp_path / "output"),
    )
    source = MOPDVarianceDataSource(args)
    assert source.prompt_count_for_rollout(0) == 512
    samples = source.get_samples(512)
    assert [samples[index][0].metadata["mopd_task"] for index in (0, 128, 256, 384)] == list(TASKS)
    feedback = {
        "operation": "heldout_variance",
        "optimizer_step_executed": False,
        "task_units": [{"task": task, "microbatches": 32} for task in TASKS],
    }
    source.complete_update(0, feedback)
    artifact = json.loads((tmp_path / "output/heldout_gradient_scalars.json").read_text())
    assert artifact["training_controller_state"]["task_seconds"] == {
        "math": 1.0,
        "code": 2.0,
        "if": 3.0,
        "science": 4.0,
    }
    assert artifact["optimizer_updates_after"] == 50


def test_frozen_step_and_response_clocks_match_the_plan():
    assert MAIN_CHECKPOINT_STEPS == tuple(range(50, 501, 50))
    assert MAIN_CHECKPOINT_RESPONSES == tuple(range(3_200, 32_001, 3_200))
    assert MAIN_EVAL_RESPONSES == MAIN_CHECKPOINT_RESPONSES
    assert MAIN_RESPONSE_BUDGET == 32_000
    assert SMOKE_STEPS == 20
    assert SMOKE_CHECKPOINT_STEPS == (10, 20)
    assert SMOKE_EVAL_RESPONSES == (640, 1_280)
    assert SMOKE_RESPONSE_BUDGET == 1_280


def test_largest_remainder_preserves_sum_bounds_and_source_order_ties():
    assert largest_remainder_allocation([1, 1, 1, 1]) == [4, 4, 4, 4]
    counts = largest_remainder_allocation([1, 2, 3, 4])
    assert counts == [2, 3, 5, 6]
    assert sum(counts) == 16
    assert all(2 <= value <= 8 for value in counts)


def test_cost_gpas_enumerates_the_exact_integer_minimum():
    weights = [0.1, 0.2, 0.3, 0.4]
    noise = [2.0, 8.0, 1.0, 5.0]
    seconds = [1.0, 3.0, 2.0, 4.0]
    fixed = 7.0
    observed = cost_gpas_allocation(weights, noise, seconds, fixed)

    def objective(counts):
        return (fixed + sum(m * tau for m, tau in zip(counts, seconds, strict=True))) * sum(
            w * w * e / m for w, e, m in zip(weights, noise, counts, strict=True)
        )

    expected = min(feasible_allocations(), key=objective)
    assert tuple(observed) == expected


@pytest.mark.parametrize("allocation", [value for value in ALLOCATIONS if value != "d3_mopd"])
def test_every_configuration_starts_with_uniform_allocation(allocation):
    controller = MOPDController(allocation=allocation)
    plan = controller.plan(0)
    assert plan["counts"] == dict.fromkeys(TASKS, 4)
    assert plan["attempted_responses"] == 64
    expected = (
        "token_mean" if allocation == "std_mopd" else "open_mopd" if allocation == "open_mopd" else "fixed_objective"
    )
    assert plan["aggregation"] == expected


def test_d3_warmup_uses_uniform_base_mixture_with_paper_jitter():
    plan = MOPDController(allocation="d3_mopd").plan(0)
    assert plan["counts"] == {"math": 5, "code": 3, "if": 4, "science": 4}
    assert plan["aggregation"] == "token_mean"
    details = plan["allocation_details"]
    assert details["watcher_updated"] is False
    assert details["base_probabilities"] == dict.fromkeys(TASKS, 0.25)
    assert len(set(details["jitter"].values())) > 1


def test_open_mopd_uses_the_paper_equal_share_target():
    controller = MOPDController(allocation="open_mopd", target_weights=(0.1, 0.2, 0.3, 0.4))
    plan = controller.plan(0)
    assert [unit["target_weight"] for unit in plan["task_units"]] == [0.25] * 4
    assert plan["allocation_details"]["share_target"] == dict.fromkeys(TASKS, 0.25)


def test_d3_scheduler_applies_gap_times_descent_velocity_and_floor():
    descending = [1.0 - 0.01 * index for index in range(20)]
    flat = [1.0] * 20
    rising = [1.0 + 0.01 * index for index in range(20)]
    result = d3_scheduler_mixture([descending, flat, rising, flat], [1.0] * 4, 20)
    assert result["available_windows"] == 1
    assert result["descent_velocity"][0] > 0
    assert result["descent_velocity"][1:] == [0.0, 0.0, 0.0]
    assert result["mixture"][0] > max(result["mixture"][1:])
    assert min(result["mixture"]) >= 0.1
    assert sum(result["mixture"]) == pytest.approx(1.0)


def test_d3_controller_seeds_five_step_normalizer_and_updates_at_step_twenty():
    controller = MOPDController(allocation="d3_mopd")
    for step in range(20):
        plan = controller.plan(step)
        losses = (1.0 - 0.01 * step, 1.0, 1.0 + 0.01 * step, 1.0)
        controller.complete(step, _feedback(plan, loss=losses))
    assert controller.d3_initial_kl == pytest.approx([0.98, 1.0, 1.02, 1.0])
    plan = controller.plan(20)
    assert plan["allocation_details"]["watcher_updated"] is True
    assert plan["allocation_details"]["scheduler"]["available_windows"] == 1

    state = copy.deepcopy(controller.state_dict())
    resumed = MOPDController(allocation="d3_mopd")
    resumed.load_state_dict(state)
    assert resumed.plan(20) == plan


@pytest.mark.parametrize(
    ("allocation", "expected"),
    [
        ("gpas", [2, 3, 5, 6]),
        ("raw_noise", [6, 5, 3, 2]),
        ("loss_gap", [2, 3, 5, 6]),
    ],
)
def test_adaptive_rules_use_their_documented_signal(allocation, expected):
    controller = MOPDController(allocation=allocation)
    first = controller.plan(0)
    controller.complete(0, _feedback(first))
    second = controller.plan(1)
    assert list(second["counts"].values()) == expected
    scaled_noise = [1.0, 4.0, 9.0, 16.0]
    uniform_variance = sum(0.25**2 * value / 4 for value in scaled_noise)
    allocated_variance = sum(0.25**2 * value / count for value, count in zip(scaled_noise, expected, strict=True))
    assert second["H"] == pytest.approx(uniform_variance / allocated_variance)
    assert second["H"] <= 2


def test_noise_is_current_step_while_time_and_loss_are_ema_smoothed():
    controller = MOPDController(allocation="gpas")
    first = controller.plan(0)
    controller.complete(0, _feedback(first))
    second = controller.plan(1)
    record = controller.complete(
        1,
        _feedback(
            second,
            scaled=(5.0, 6.0, 7.0, 8.0),
            raw=(8.0, 7.0, 6.0, 5.0),
            loss=(5.0, 6.0, 7.0, 8.0),
        ),
    )
    assert list(record["scaled_noise_after"].values()) == [5.0, 6.0, 7.0, 8.0]
    assert list(record["raw_noise_after"].values()) == [8.0, 7.0, 6.0, 5.0]
    assert list(record["loss_ema_after"].values()) == pytest.approx([1.4, 2.4, 3.4, 4.4])
    assert list(record["task_seconds_after"].values()) == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert record["fixed_seconds_after"] == pytest.approx(8.0)


def test_pending_plan_and_completed_state_resume_exactly():
    controller = MOPDController(allocation="cost_gpas")
    first = controller.plan(0)
    controller.complete(0, _feedback(first))
    pending = controller.plan(1)
    pending_state = copy.deepcopy(controller.state_dict())
    resumed = MOPDController(allocation="cost_gpas")
    resumed.load_state_dict(pending_state)
    assert resumed.plan(1) == pending
    assert resumed.complete(1, _feedback(pending)) == controller.complete(1, _feedback(pending))

    completed_state = resumed.state_dict()
    next_controller = MOPDController(allocation="cost_gpas")
    next_controller.load_state_dict(completed_state)
    assert next_controller.plan(2) == resumed.plan(2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
