import copy
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.mopd.optimizer import (
    _OperationAccumulator,
    _TaskAccumulator,
    _finish_task,
    _view_key,
    finish_mopd_operation,
    open_mopd_domain_weights,
    score_prepared_gradients,
)
from slime_plugins.mopd.optimizer_views import build_optimizer_parameter_views
from slime_plugins.mopd.sampler import TASKS

NUM_GPUS = 0


def _initialized_adam(parameter):
    optimizer = torch.optim.AdamW([parameter], lr=0.01, betas=(0.9, 0.8), eps=0.1, weight_decay=0.0)
    optimizer.state[parameter] = {
        "step": torch.tensor(2.0),
        "exp_avg": torch.tensor([0.3, -0.2]),
        "exp_avg_sq": torch.tensor([4.0, 9.0]),
    }
    return optimizer


class _MegatronAdamAdapter:
    def __init__(self, optimizer, parameter, clip_grad=0.0):
        self.optimizer = optimizer
        self.parameter = parameter
        self.config = SimpleNamespace(clip_grad=clip_grad, use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)

    def get_parameters(self):
        return [self.parameter]

    def step_with_ready_grads(self):
        self.optimizer.step()
        return True


def _ready_accumulator(view, gradient, *, aggregation="fixed_objective", total_tokens=1):
    return _OperationAccumulator(
        operation="train",
        aggregation=aggregation,
        aggregate={_view_key(view): gradient.clone()},
        task_units=[{"task": task, "microbatches": 4} for task in TASKS],
        views=[view],
        total_valid_tokens=total_tokens,
    )


def test_adamw_score_uses_bias_corrected_lagged_second_moment():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([2.0, 0.0])
    optimizer = _initialized_adam(parameter)
    views = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")
    raw, scaled = score_prepared_gradients(views, chunk_size=1)
    expected = 2.0 / ((4.0 / (1.0 - 0.8**2)) ** 0.5 + 0.1)
    assert raw == pytest.approx(2.0)
    assert scaled == pytest.approx(expected)


def test_adamw_score_reads_apex_fused_adam_group_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([2.0, 0.0])
    optimizer = _initialized_adam(parameter)
    optimizer.state[parameter].pop("step")
    optimizer.param_groups[0]["step"] = 2
    views = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")
    _raw, scaled = score_prepared_gradients(views, chunk_size=1)
    expected = 2.0 / ((4.0 / (1.0 - 0.8**2)) ** 0.5 + 0.1)
    assert scaled == pytest.approx(expected)


def test_task_block_noise_and_fixed_objective_weight_are_exact():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.zeros_like(parameter)
    optimizer = _initialized_adam(parameter)
    view = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")[0]
    key = _view_key(view)
    accumulator = _OperationAccumulator(
        operation="train",
        aggregation="fixed_objective",
        aggregate={key: torch.zeros(2)},
        views=[view],
        current=_TaskAccumulator(
            task="math",
            expected_microbatches=2,
            target_weight=0.25,
            gradients={key: torch.tensor([4.0, 6.0])},
            microbatches=2,
            raw_square_sum=30.0,
            scaled_square_sum=5.0,
            teacher_loss_sum=1.0,
            valid_tokens=7,
        ),
    )
    _finish_task(accumulator, chunk_size=1)
    unit = accumulator.task_units[0]
    assert unit["raw_task_mean_sq"] == pytest.approx(13.0)
    assert unit["raw_noise"] == pytest.approx(4.0)
    assert unit["teacher_loss"] == pytest.approx(0.5)
    assert accumulator.aggregate[key] == pytest.approx(torch.tensor([0.5, 0.75]))


def test_finish_commits_one_conventional_adamw_step_on_combined_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    optimizer = _initialized_adam(parameter)
    reference = _initialized_adam(reference_parameter)
    view = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")[0]
    gradient = torch.tensor([0.25, -0.75])
    adapter = _MegatronAdamAdapter(optimizer, parameter)
    adapter._mopd_operation_accumulator = _ready_accumulator(view, gradient)
    reference_parameter.grad = gradient.clone()
    reference.step()

    successful, aggregate_norm, feedback = finish_mopd_operation(
        SimpleNamespace(mopd_score_chunk_size=128, clip_grad=0.0), adapter
    )
    assert successful
    assert aggregate_norm == pytest.approx(float(torch.linalg.vector_norm(gradient)))
    assert parameter.detach() == pytest.approx(reference_parameter.detach())
    assert optimizer.state[parameter]["exp_avg"] == pytest.approx(reference.state[reference_parameter]["exp_avg"])
    assert optimizer.state[parameter]["exp_avg_sq"] == pytest.approx(
        reference.state[reference_parameter]["exp_avg_sq"]
    )
    assert feedback["optimizer_step_executed"] is True
    assert feedback["aggregation"] == "fixed_objective"


def test_std_mopd_divides_accumulated_token_sum_once_before_adamw():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    optimizer = _initialized_adam(parameter)
    reference = _initialized_adam(reference_parameter)
    view = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")[0]
    token_sum_gradient = torch.tensor([5.0, -10.0])
    adapter = _MegatronAdamAdapter(optimizer, parameter)
    adapter._mopd_operation_accumulator = _ready_accumulator(
        view, token_sum_gradient, aggregation="token_mean", total_tokens=5
    )
    reference_parameter.grad = token_sum_gradient / 5
    reference.step()

    finish_mopd_operation(SimpleNamespace(mopd_score_chunk_size=128, clip_grad=0.0), adapter)
    assert parameter.detach() == pytest.approx(reference_parameter.detach())


def test_open_mopd_weights_equalize_tokens_then_follow_the_forward_gap():
    equal_gap = open_mopd_domain_weights([90, 10], [1.0, 1.0], [0.5, 0.5])
    assert equal_gap["effective_shares"] == pytest.approx([0.5, 0.5])

    unequal_gap = open_mopd_domain_weights([90, 10], [2.0, 0.5], [0.5, 0.5])
    assert unequal_gap["gap_factors"] == pytest.approx([1.6, 0.4])
    assert unequal_gap["effective_shares"] == pytest.approx([0.8, 0.2])
    assert sum(unequal_gap["effective_shares"]) == pytest.approx(1.0)


def test_open_mopd_commits_equal_domain_token_means_when_gaps_match():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    optimizer = _initialized_adam(parameter)
    reference = _initialized_adam(reference_parameter)
    view = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")[0]
    key = _view_key(view)
    token_counts = [8, 4, 2, 2]
    per_token_gradients = [1.0, 2.0, 3.0, 4.0]
    adapter = _MegatronAdamAdapter(optimizer, parameter)
    adapter._mopd_operation_accumulator = _OperationAccumulator(
        operation="train",
        aggregation="open_mopd",
        aggregate={key: torch.zeros(2)},
        task_units=[
            {
                "task": task,
                "microbatches": 4,
                "valid_response_tokens": tokens,
                "reward_abs_mean": 1.0,
                "target_weight": 0.25,
            }
            for task, tokens in zip(TASKS, token_counts, strict=True)
        ],
        views=[view],
        total_valid_tokens=sum(token_counts),
        open_task_gradients=[
            {key: torch.tensor([gradient * tokens, -gradient * tokens])}
            for gradient, tokens in zip(per_token_gradients, token_counts, strict=True)
        ],
    )
    reference_parameter.grad = torch.tensor([2.5, -2.5])
    reference.step()

    _, _, feedback = finish_mopd_operation(SimpleNamespace(mopd_score_chunk_size=128, clip_grad=0.0), adapter)
    assert parameter.detach() == pytest.approx(reference_parameter.detach())
    assert [unit["open_mopd_effective_share"] for unit in feedback["task_units"]] == pytest.approx([0.25] * 4)


def test_aggregate_clip_is_applied_once_before_the_single_commit():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    optimizer = _initialized_adam(parameter)
    reference = _initialized_adam(reference_parameter)
    view = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")[0]
    gradient = torch.tensor([3.0, 4.0])
    adapter = _MegatronAdamAdapter(optimizer, parameter, clip_grad=1.0)
    adapter._mopd_operation_accumulator = _ready_accumulator(view, gradient)
    reference_parameter.grad = gradient / 5.0
    reference.step()

    _, _, feedback = finish_mopd_operation(SimpleNamespace(mopd_score_chunk_size=128, clip_grad=1.0), adapter)
    assert parameter.detach() == pytest.approx(reference_parameter.detach(), abs=1e-7)
    assert feedback["aggregate_grad_clipped"] is True


def test_optimizer_state_roundtrip_reproduces_the_next_conventional_update():
    first_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    first = _initialized_adam(first_parameter)
    state = copy.deepcopy(first.state_dict())
    resumed_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    resumed = torch.optim.AdamW([resumed_parameter], lr=0.01, betas=(0.9, 0.8), eps=0.1)
    resumed.load_state_dict(state)
    for parameter, optimizer in ((first_parameter, first), (resumed_parameter, resumed)):
        parameter.grad = torch.tensor([-0.4, 0.2])
        optimizer.step()
    assert resumed_parameter.detach() == pytest.approx(first_parameter.detach())
    assert resumed.state[resumed_parameter]["exp_avg_sq"] == pytest.approx(first.state[first_parameter]["exp_avg_sq"])


def test_heldout_variance_returns_scalars_without_an_optimizer_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = _initialized_adam(parameter)
    view = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")[0]
    adapter = _MegatronAdamAdapter(optimizer, parameter)
    adapter._mopd_operation_accumulator = _OperationAccumulator(
        operation="heldout_variance",
        aggregation="variance_probe",
        task_units=[{"task": task, "microbatches": 32} for task in TASKS],
        views=[view],
    )
    before_parameter = parameter.detach().clone()
    before_step = optimizer.state[parameter]["step"].clone()

    successful, aggregate_norm, feedback = finish_mopd_operation(
        SimpleNamespace(mopd_score_chunk_size=128, clip_grad=1.0), adapter
    )

    assert successful
    assert aggregate_norm == 0.0
    assert feedback["optimizer_step_executed"] is False
    assert feedback["attempted_responses"] == 512
    assert parameter.detach() == pytest.approx(before_parameter)
    assert optimizer.state[parameter]["step"] == before_step


def test_heldout_task_summary_contains_only_scalar_gradient_statistics():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = _initialized_adam(parameter)
    view = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")[0]
    key = _view_key(view)
    accumulator = _OperationAccumulator(
        operation="heldout_variance",
        aggregation="variance_probe",
        views=[view],
        current=_TaskAccumulator(
            task="math",
            expected_microbatches=32,
            target_weight=0.25,
            gradients={key: torch.tensor([32.0, 0.0])},
            microbatches=32,
            raw_square_sum=64.0,
            scaled_square_sum=64.0,
            teacher_loss_sum=16.0,
            valid_tokens=128,
            raw_microbatch_sq=[2.0] * 32,
            scaled_microbatch_sq=[2.0] * 32,
            teacher_loss_microbatch=[0.5] * 32,
            half_gradients=[
                {key: torch.tensor([16.0, 0.0])},
                {key: torch.tensor([16.0, 0.0])},
            ],
            subset_mean_sq={2: [(1.0, 1.0), (1.0, 1.0)], 4: [(1.0, 1.0), (1.0, 1.0)]},
        ),
    )

    _finish_task(accumulator, chunk_size=128)
    unit = accumulator.task_units[0]
    assert len(unit["raw_microbatch_sq"]) == 32
    assert len(unit["raw_half_mean_sq"]) == 2
    assert set(unit["scaled_subset_mean_sq"]) == {"2", "4"}
    assert not any(torch.is_tensor(value) for value in unit.values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
