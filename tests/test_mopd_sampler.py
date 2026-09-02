import copy

import pytest

from slime_plugins.mopd.data_source import PROTOCOL, _validate_protocol_manifest
from slime_plugins.mopd.sampler import (
    CONFIRMATION_CHECKPOINT_RESPONSES,
    CONFIRMATION_RESPONSE_BUDGET,
    MAIN_CHECKPOINT_RESPONSES,
    MAIN_EVAL_RESPONSES,
    MAIN_RESPONSE_BUDGET,
    MOPDController,
    TASKS,
    bounded_inclusion_probabilities,
    inclusion_probabilities,
    maximum_entropy_set_distribution,
    crossed_response_milestone,
)

NUM_GPUS = 0


def _sources():
    losses = (0.1, 0.2, 0.3, 0.4)
    return [
        {
            "name": task,
            "teacher": task,
            "target_weight": 0.25,
            "initial_teacher_loss": loss,
            "relative_loss_scale": 1 / loss,
            "required_samples": 8,
            "probe_path": f"/{task}/probe",
            "bank_path": f"/{task}/bank",
        }
        for task, loss in zip(TASKS, losses, strict=True)
    ]


def _feedback(plan):
    return {
        "operation": plan["operation"],
        "attempted_responses": plan["attempted_responses"],
        "resident_teacher_after": plan["execution_order"][-1],
        "task_units": [
            {
                "task": task,
                "raw_score": 1.0 + TASKS.index(task),
                "adam_score": 2.0 + TASKS.index(task),
                "predicted_full_task_seconds": 3.0 + TASKS.index(task),
                "teacher_switch_seconds": 0.25,
                "teacher_transfer_tail_seconds": 0.125,
                "switched": task != plan["resident_teacher_before"],
            }
            for task in plan["execution_order"]
        ],
    }


def _warm_state():
    controller = MOPDController(run_mode="warm", allocation="uniform", task_width=1)
    for rollout_id in range(8):
        plan = controller.plan(rollout_id)
        assert plan["execution_order"] == [TASKS[rollout_id % 4]]
        controller.complete(rollout_id, _feedback(plan))
    return controller.state_dict()


def test_v2_manifest_freezes_relative_loss_and_disjoint_auxiliary_streams():
    _validate_protocol_manifest({"version": 2, "protocol": PROTOCOL}, _sources())
    wrong = _sources()
    wrong[0]["relative_loss_scale"] = 1.0
    with pytest.raises(ValueError, match="denominator"):
        _validate_protocol_manifest({"version": 2, "protocol": PROTOCOL}, wrong)

    missing_count = _sources()
    missing_count[0].pop("required_samples")
    with pytest.raises(ValueError, match="usable-prompt count"):
        _validate_protocol_manifest({"version": 2, "protocol": PROTOCOL}, missing_count)


def test_maximum_entropy_k2_has_all_six_sets_and_exact_marginals():
    marginals = bounded_inclusion_probabilities([1.0, 2.0, 3.0, 4.0], 2, 0.05)
    distribution = maximum_entropy_set_distribution(marginals, 2)
    assert len(distribution) == 6
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert inclusion_probabilities(distribution.keys(), distribution.values(), 4) == pytest.approx(marginals)
    assert sum(marginals) == pytest.approx(2.0)


def test_warm_start_is_exactly_two_round_robin_units_per_task():
    state = _warm_state()
    assert state["completed_operations"] == 8
    assert state["processed_task_units"] == 8
    assert state["attempted_responses"] == 512
    assert state["observation_counts"] == [2, 2, 2, 2]
    controller = MOPDController(run_mode="warm", allocation="uniform", task_width=1)
    controller.load_state_dict(state)
    assert controller._score_rms(controller.raw_gradient_sq_ema) == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert controller._score_rms(controller.adam_gradient_sq_ema) == pytest.approx([2.0, 3.0, 4.0, 5.0])
    assert controller._effective(controller.task_seconds_ema, 1.0) == pytest.approx([3.0, 4.0, 5.0, 6.0])
    assert controller.switch_observation_counts == [1, 2, 2, 2]
    assert controller._effective(
        controller.switch_seconds_ema, 0.0, controller.switch_observation_counts
    ) == pytest.approx([0.125] * 4)
    assert controller.budget_complete
    with pytest.raises(StopIteration):
        controller.plan(8)


def test_k2_response_clock_hits_64000_exactly_without_padding_responses():
    controller = MOPDController(
        run_mode="train", allocation="uniform", task_width=2, rollout_offset=8
    )
    controller.bootstrap(_warm_state())
    rollout_id = 8
    while not controller.budget_complete:
        plan = controller.plan(rollout_id)
        controller.complete(rollout_id, _feedback(plan))
        rollout_id += 1
    assert controller.attempted_responses == MAIN_RESPONSE_BUDGET
    assert controller.probe_count == 0
    assert controller.processed_task_units % 2 == 0


def test_shortened_campaign_milestones_and_confirmation_budget_are_exact():
    assert MAIN_CHECKPOINT_RESPONSES == (16_384, 32_768, 64_000)
    assert MAIN_EVAL_RESPONSES == (2_048, 4_096, 8_192, 16_384, 32_768, 49_152, 64_000)
    assert crossed_response_milestone(2_040, 2_048, MAIN_EVAL_RESPONSES)
    assert crossed_response_milestone(2_040, 2_056, MAIN_EVAL_RESPONSES)
    assert not crossed_response_milestone(2_048, 2_056, MAIN_EVAL_RESPONSES)

    controller = MOPDController(
        run_mode="train",
        allocation="uniform",
        task_width=1,
        rollout_offset=8,
        response_budget=CONFIRMATION_RESPONSE_BUDGET,
        checkpoint_responses=CONFIRMATION_CHECKPOINT_RESPONSES,
    )
    controller.bootstrap(_warm_state())
    rollout_id = 8
    while not controller.budget_complete:
        plan = controller.plan(rollout_id)
        controller.complete(rollout_id, _feedback(plan))
        rollout_id += 1
    assert controller.attempted_responses == CONFIRMATION_RESPONSE_BUDGET


def test_score_age_50_forces_an_eight_response_probe():
    controller = MOPDController(
        run_mode="train", allocation="gpas", task_width=1, rollout_offset=8
    )
    controller.bootstrap(_warm_state())
    controller.score_ages[2] = 50
    plan = controller.plan(8)
    assert plan["operation"] == "probe"
    assert plan["reason"] == "stale_score"
    assert plan["execution_order"] == ["if"]
    assert plan["attempted_responses"] == 8


def test_k2_probe_is_anticipated_so_score_age_never_exceeds_50():
    controller = MOPDController(
        run_mode="train", allocation="gpas", task_width=2, rollout_offset=8
    )
    controller.bootstrap(_warm_state())
    controller.score_ages[1] = 49
    plan = controller.plan(8)
    assert plan["operation"] == "probe"
    assert plan["execution_order"] == ["code"]
    controller.complete(8, _feedback(plan))
    assert max(controller.score_ages) <= 50


def test_uniform_control_does_not_spend_budget_on_adaptive_score_probes():
    controller = MOPDController(
        run_mode="train", allocation="uniform", task_width=1, rollout_offset=8
    )
    controller.bootstrap(_warm_state())
    controller.score_ages = [50, 50, 50, 50]
    assert controller.plan(8)["operation"] == "train"


def test_pending_plan_and_rng_resume_are_exact():
    controller = MOPDController(
        run_mode="train", allocation="cost_gpas", task_width=2, rollout_offset=8, seed=42
    )
    controller.bootstrap(_warm_state())
    plan = controller.plan(8)
    state = copy.deepcopy(controller.state_dict())
    resumed = MOPDController(
        run_mode="train", allocation="cost_gpas", task_width=2, rollout_offset=8, seed=42
    )
    resumed.load_state_dict(state)
    assert resumed.plan(8) == plan
    feedback = _feedback(plan)
    assert resumed.complete(8, feedback) == controller.complete(8, feedback)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
