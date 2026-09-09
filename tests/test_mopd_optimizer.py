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


def test_first_preconditioner_is_identity_and_reading_does_not_mutate_adam():
    from slime_plugins.mopd.optimizer import _preconditioner_denominator

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter])
    view = build_optimizer_parameter_views([("p", parameter, [])], optimizer, requested_optimizer="adam")[0]
    assert _preconditioner_denominator(view, 0, 2, device=parameter.device).tolist() == [1.0, 1.0]
    optimizer = _initialized_adam(parameter)
    view = build_optimizer_parameter_views([("p", parameter, [])], optimizer, requested_optimizer="adam")[0]
    optimizer.param_groups[0]["bias_correction"] = False
    before = optimizer.state[parameter]["exp_avg_sq"].clone()
    _preconditioner_denominator(view, 0, 2, device=parameter.device)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], before)


def test_welford_is_stable_with_large_task_means_and_raw_ablation_skips_d(monkeypatch):
    from slime_plugins.mopd import optimizer as module

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = _initialized_adam(parameter)
    view = build_optimizer_parameter_views([("p", parameter, [])], optimizer, requested_optimizer="adam")[0]
    mean = {_view_key(view): torch.zeros(2)}
    gradients = torch.tensor([[1e6, 1e6], [1e6 + 2, 1e6 - 2], [1e6 + 4, 1e6 - 4]])
    monkeypatch.setattr(module, "_preconditioner_denominator", lambda *a, **kw: pytest.fail("raw ablation used D"))
    m2 = 0.0
    for count, gradient in enumerate(gradients, 1):
        parameter.grad = gradient
        raw, scaled = module.welford_gradients([view], mean, count, raw_only=True, chunk_size=1)
        m2 += raw
        assert raw == scaled
    assert m2 / 2 == pytest.approx(float(gradients.var(dim=0).sum()))
    torch.testing.assert_close(mean[_view_key(view)], gradients.mean(dim=0))


@pytest.mark.parametrize("support_key", ["teacher_topk_ids", "student_topk_ids"])
def test_uniform_dense_accumulates_task_means_without_collecting_noise(monkeypatch, support_key):
    from slime_plugins.mopd import optimizer as module

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    optimizer = _initialized_adam(parameter)
    adapter = _MegatronAdamAdapter(optimizer, parameter)
    adapter.prepare_grads = lambda: False
    module.begin_mopd_operation(adapter, "train", "fixed_objective", noise_mode="none")
    monkeypatch.setattr(module, "_gradient_square_sums", lambda *a, **kw: pytest.fail("Uniform measured noise"))
    monkeypatch.setattr(module, "welford_gradients", lambda *a, **kw: pytest.fail("Uniform measured noise"))
    for task_index, (task, count) in enumerate(zip(TASKS, [2, 2, 4, 8], strict=True)):
        data = {
            "mopd_operations": ["train"] * 4,
            "mopd_tasks": [task] * 4,
            "mopd_aggregations": ["fixed_objective"] * 4,
            "mopd_task_microbatch_counts": [count] * 4,
            "mopd_target_weights": [0.25] * 4,
            "metadata": [{support_key: [], "mopd_teacher_loss": 0.3} for _ in range(4)],
            "loss_masks": [torch.ones(1)] * 4,
        }
        iterator = SimpleNamespace(offset=1, micro_batch_indices=[[0, 1, 2, 3]], rollout_data=data)
        for _ in range(count):
            parameter.grad = torch.tensor([float(task_index + 1), 1.0])
            module.mopd_capture_step(
                SimpleNamespace(mopd_score_chunk_size=2), [iterator], [model], adapter, num_microbatches=1
            )
    module._finish_task(adapter._mopd_operation_accumulator, 2)
    captured = adapter._mopd_operation_accumulator
    torch.testing.assert_close(next(iter(captured.aggregate.values())), torch.tensor([2.5, 1.0]))
    assert all(unit["scaled_noise"] is None and not unit["noise_collected"] for unit in captured.task_units)


def test_trial_variance_uses_one_frozen_preconditioner_before_clipping():
    from slime_plugins.mopd.diagnostic import record_trial_gradient
    from slime_plugins.mopd.optimizer import _preconditioner_denominator

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = _initialized_adam(parameter)
    optimizer._mopd_trial_variance = {}
    views = build_optimizer_parameter_views([("p", parameter, [])], optimizer, requested_optimizer="adam")
    denominator = _preconditioner_denominator(views[0], 0, 2, device=parameter.device)
    for branch, scale in [("uniform", 1.0), ("gpas", 0.5)]:
        optimizer._mopd_trial_branch = branch
        draws = torch.tensor([[float(i), float(i * i)] for i in range(10)]) * scale
        for gradient in draws:
            parameter.grad = gradient
            record_trial_gradient(optimizer, views)
        result = optimizer._mopd_trial_variance[branch]
        assert result["m2"] / 9 == pytest.approx(float((draws / denominator).var(dim=0).sum()), rel=1e-5)
    assert optimizer._mopd_trial_variance["gpas"]["m2"] / optimizer._mopd_trial_variance["uniform"][
        "m2"
    ] == pytest.approx(0.25)


@pytest.mark.parametrize("broken_restore", [False, True])
def test_common_checkpoint_restores_distributed_master_parameters_and_moments(monkeypatch, broken_restore):
    from megatron.core.optimizer import distrib_optimizer
    from megatron.core.tensor_parallel import random as megatron_random

    from slime_plugins.mopd.diagnostic import assert_restored_state, restore_actor, snapshot_actor

    model = torch.nn.Linear(2, 1, bias=False)
    master = torch.nn.Parameter(model.weight.detach().float().clone())
    adam = torch.optim.AdamW([master], lr=0.01, weight_decay=0)
    master.grad = torch.ones_like(master)
    adam.step()
    with torch.no_grad():
        model.weight.copy_(master)

    class MetadataOnlyDistributedOptimizer:
        # Match Megatron's split between scalar metadata and parameter state.
        def state_dict(self):
            return {"lr": adam.param_groups[0]["lr"]}

        def load_state_dict(self, state):
            adam.param_groups[0]["lr"] = state["lr"]

        def get_parameter_state_dp_zero(self):
            return {"param": master, "optimizer": adam.state_dict()}

        def load_parameter_state_from_dp_zero(self, state):
            if broken_restore:
                return
            with torch.no_grad():
                master.copy_(state["param"])
            adam.load_state_dict(state["optimizer"])

    monkeypatch.setattr(distrib_optimizer, "DistributedOptimizer", MetadataOnlyDistributedOptimizer)
    tracker = SimpleNamespace(get_states=lambda: {}, set_states=lambda state: None)
    monkeypatch.setattr(megatron_random, "get_cuda_rng_tracker", lambda: tracker)
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: None)
    actor = SimpleNamespace(
        model=[model], optimizer=MetadataOnlyDistributedOptimizer(),
        opt_param_scheduler=torch.optim.lr_scheduler.StepLR(adam, step_size=10),
    )
    saved = snapshot_actor(actor)
    expected_parameter = master.detach().clone()
    expected_optimizer = copy.deepcopy(adam.state_dict())
    outcomes = []
    for _ in range(2):
        master.grad = torch.full_like(master, 3.0)
        adam.step()
        with torch.no_grad():
            model.weight.copy_(master)
        if broken_restore:
            with pytest.raises(RuntimeError, match="not restored exactly"):
                restore_actor(actor, saved)
            return
        restore_actor(actor, saved)
        torch.testing.assert_close(master, expected_parameter, rtol=0, atol=0)
        torch.testing.assert_close(model.weight, expected_parameter, rtol=0, atol=0)
        assert_restored_state(adam.state_dict(), expected_optimizer)
        master.grad = torch.full_like(master, 2.0)
        adam.step()
        outcomes.append(master.detach().clone())
    torch.testing.assert_close(outcomes[0], outcomes[1], rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


@pytest.mark.parametrize("reduction", ["global_token", "domain_token", "domain_response"])
def test_paper_unequal_lengths_match_direct_reduction_and_one_adam_step(reduction):
    from slime_plugins.mopd import optimizer as module
    from slime_plugins.mopd.paper_diagnostics import prefix_weights

    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    optimizer = _initialized_adam(parameter)
    reference = _initialized_adam(reference_parameter)
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    adapter = _MegatronAdamAdapter(optimizer, parameter)
    adapter.prepare_grads = lambda: False
    module.begin_mopd_operation(adapter, "train", reduction)
    args = SimpleNamespace(mopd_score_chunk_size=2, mopd_profile="smollm3", mopd_tasks="math,code,if",
                           mopd_prompts_per_microbatch=2, clip_grad=0.0)
    all_masks, domains, all_coefficients = [], [], []
    for task_index, task in enumerate(("math", "code", "if")):
        # Two unequal-length micro-batches in each task; two responses apiece.
        for batch_index in range(2):
            lengths = [1 + task_index + batch_index, 4 + 2 * task_index + batch_index]
            coefficients = [torch.arange(length * 2, dtype=torch.float32).reshape(length, 2) / 9 + task_index for length in lengths]
            losses = [coefficient @ parameter for coefficient in coefficients]
            token_reduction = reduction != "domain_response"
            batch_loss = sum(value.sum() for value in losses) / sum(lengths) if token_reduction else sum(value.mean() for value in losses) / 2
            optimizer.zero_grad()
            batch_loss.backward()
            masks = [torch.ones(length) for length in lengths]
            data = {
                "mopd_operations": ["train"] * 2, "mopd_tasks": [task] * 2,
                "mopd_aggregations": [reduction] * 2, "mopd_task_microbatch_counts": [2] * 2,
                "mopd_target_weights": [1 / 3] * 2,
                "metadata": [{"teacher_topk_ids": [], "mopd_teacher_loss": float(value.detach().mean())} for value in losses],
                "loss_masks": masks,
            }
            iterator = SimpleNamespace(offset=1, micro_batch_indices=[[0, 1]], rollout_data=data)
            module.mopd_capture_step(args, [iterator], [model], adapter, num_microbatches=1)
            all_masks.extend(masks)
            all_coefficients.extend(coefficients)
            domains.extend([task] * 2)
    positions = torch.cat(all_coefficients) @ reference_parameter
    (positions * prefix_weights(all_masks, domains, reduction)).sum().backward()
    expected_gradient = reference_parameter.grad.clone()
    reference.step()
    _, norm, feedback = module.finish_mopd_operation(args, adapter)
    torch.testing.assert_close(parameter, reference_parameter)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], reference.state[reference_parameter]["exp_avg"])
    assert norm == pytest.approx(float(expected_gradient.norm()))
    assert feedback["attempted_responses"] == 12
