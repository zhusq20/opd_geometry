import copy
import json
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.mopd.optimizer import (
    BASE_BETAS_KEY,
    MOMENT_CLOCK_KEY,
    MOMENT_RULE_KEY,
    _prepare_compound_adamw,
    _restore_compound_groups,
    _sample_bank_vectors,
    finish_mopd_operation,
    score_prepared_gradients,
)
from slime_plugins.mopd.optimizer_views import build_optimizer_parameter_views


def _initialized_adam(parameter, *, lr=0.01, betas=(0.9, 0.8), eps=1e-8, wd=0.0):
    optimizer = torch.optim.AdamW([parameter], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    optimizer.state[parameter] = {
        "step": torch.tensor(8.0),
        "exp_avg": torch.tensor([0.3, -0.2]),
        "exp_avg_sq": torch.tensor([0.4, 0.6]),
    }
    return optimizer


def test_adamw_score_uses_bias_corrected_lagged_second_moment():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([2.0, 0.0])
    optimizer = torch.optim.AdamW([parameter], betas=(0.9, 0.5), eps=0.1)
    optimizer.state[parameter] = {
        "step": torch.tensor(2.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.tensor([4.0, 9.0]),
    }
    views = build_optimizer_parameter_views([("parameter", parameter, [])], optimizer, requested_optimizer="adam")
    raw, adam = score_prepared_gradients(views, chunk_size=1)
    expected = 2.0 / ((4.0 / (1.0 - 0.5**2)) ** 0.5 + 0.1)
    assert raw == pytest.approx(2.0)
    assert adam == pytest.approx(expected)


def test_frozen_bank_samples_uniformly_over_the_concatenated_optimizer_vector():
    small = torch.nn.Parameter(torch.zeros(2))
    large = torch.nn.Parameter(torch.zeros(8))
    small.grad = torch.tensor([10.0, 11.0])
    large.grad = torch.arange(20.0, 28.0)
    optimizer = torch.optim.AdamW([small, large], betas=(0.9, 0.5), eps=0.1)
    for parameter in (small, large):
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }
    views = build_optimizer_parameter_views(
        [("small", small, []), ("large", large, [])], optimizer, requested_optimizer="adam"
    )
    raw, scaled, layout = _sample_bank_vectors(views, 5)
    assert raw == pytest.approx(torch.tensor([10.0, 20.0, 22.0, 24.0, 27.0]))
    assert scaled == pytest.approx(raw / (2.0**0.5 + 0.1))
    assert [entry["take"] for entry in layout] == [1, 4]


@pytest.mark.parametrize("operation", ["bank", "probe"])
def test_non_train_operation_feedback_is_strict_json_serializable(operation):
    def entry(value):
        vector = torch.tensor([value, value + 1.0])
        return {
            "task": "math",
            "raw_score": 2.0,
            "adam_score": 3.0,
            "raw_teacher_loss": 0.5,
            "relative_teacher_loss": 0.25,
            "importance": 1.0,
            "inclusion": 1.0,
            "target_weight": 0.25,
            "clip_flag": False,
            "probe_vectors": {("parameter", 0, None): (vector, vector / 2)},
        }

    entries = [entry(1.0), entry(2.0)] if operation == "probe" else [entry(1.0)]
    optimizer = SimpleNamespace(
        _mopd_operation_accumulator=SimpleNamespace(
            operation=operation,
            adamw_state="taskwise",
            entries=entries,
        )
    )

    successful, aggregate_norm, feedback = finish_mopd_operation(None, optimizer)

    assert successful
    assert aggregate_norm == 0.0
    assert feedback["aggregate_grad_norm"] == 0.0
    assert feedback["optimizer_step_executed"] is False
    json.dumps(feedback, allow_nan=False)


def test_taskwise_k2_commit_uses_beta_squared_and_unmixed_square_observation():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = _initialized_adam(parameter)
    old_first = optimizer.state[parameter]["exp_avg"].clone()
    old_second = optimizer.state[parameter]["exp_avg_sq"].clone()
    gradient = torch.tensor([0.25, -0.75])
    desired_second = torch.tensor([1.5, 0.5])
    parameter.grad = gradient.clone()

    restored = _prepare_compound_adamw(
        optimizer,
        {id(parameter): desired_second},
        task_width=2,
        task_unit_clock_before=8,
        adamw_state="taskwise",
    )
    assert optimizer.param_groups[0]["betas"] == pytest.approx((0.9**2, 0.8**2))
    optimizer.step()
    _restore_compound_groups(restored, 10)

    assert optimizer.state[parameter]["exp_avg"] == pytest.approx(
        0.9**2 * old_first + (1 - 0.9**2) * gradient
    )
    assert optimizer.state[parameter]["exp_avg_sq"] == pytest.approx(
        0.8**2 * old_second + (1 - 0.8**2) * desired_second
    )
    group = optimizer.param_groups[0]
    assert group["betas"] == pytest.approx((0.9, 0.8))
    assert group[BASE_BETAS_KEY] == pytest.approx((0.9, 0.8))
    assert group[MOMENT_RULE_KEY] == "taskwise"
    assert group[MOMENT_CLOCK_KEY] == 10


def test_conventional_k4_uses_square_of_the_combined_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = _initialized_adam(parameter)
    old_second = optimizer.state[parameter]["exp_avg_sq"].clone()
    gradient = torch.tensor([0.25, -0.75])
    parameter.grad = gradient.clone()
    restored = _prepare_compound_adamw(
        optimizer,
        {id(parameter): torch.tensor([99.0, 99.0])},
        task_width=4,
        task_unit_clock_before=8,
        adamw_state="conventional",
    )
    optimizer.step()
    _restore_compound_groups(restored, 12)
    assert optimizer.state[parameter]["exp_avg_sq"] == pytest.approx(
        0.8**4 * old_second + (1 - 0.8**4) * gradient.square()
    )


def test_compound_decoupled_weight_decay_matches_k_serial_decay_steps():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    optimizer = _initialized_adam(parameter, lr=0.01, wd=0.1)
    optimizer.state[parameter]["exp_avg"].zero_()
    optimizer.state[parameter]["exp_avg_sq"].zero_()
    before = parameter.detach().clone()
    parameter.grad = torch.zeros_like(parameter)
    restored = _prepare_compound_adamw(
        optimizer,
        {id(parameter): torch.zeros_like(parameter)},
        task_width=4,
        task_unit_clock_before=8,
        adamw_state="taskwise",
    )
    optimizer.step()
    _restore_compound_groups(restored, 12)
    assert parameter.detach() == pytest.approx(before * (1 - 0.01 * 0.1) ** 4)


def test_optimizer_state_roundtrip_reproduces_next_k2_update():
    first_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    first = _initialized_adam(first_parameter)
    first.param_groups[0][BASE_BETAS_KEY] = (0.9, 0.8)
    first.param_groups[0][MOMENT_CLOCK_KEY] = 10
    state = copy.deepcopy(first.state_dict())

    resumed_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    resumed = torch.optim.AdamW([resumed_parameter], lr=0.01, betas=(0.9, 0.8))
    resumed.load_state_dict(state)
    for parameter, optimizer in ((first_parameter, first), (resumed_parameter, resumed)):
        parameter.grad = torch.tensor([-0.4, 0.2])
        restored = _prepare_compound_adamw(
            optimizer,
            {id(parameter): torch.tensor([0.7, 0.9])},
            task_width=2,
            task_unit_clock_before=10,
            adamw_state="taskwise",
        )
        optimizer.step()
        _restore_compound_groups(restored, 12)
    assert resumed_parameter.detach() == pytest.approx(first_parameter.detach())
    assert resumed.state[resumed_parameter]["exp_avg"] == pytest.approx(first.state[first_parameter]["exp_avg"])
    assert resumed.state[resumed_parameter]["exp_avg_sq"] == pytest.approx(
        first.state[first_parameter]["exp_avg_sq"]
    )
