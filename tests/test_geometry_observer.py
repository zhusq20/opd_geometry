"""CPU tests for deterministic projections and parameter geometry storage."""

import json
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.geometry.directions import _muon_scale_factor
from slime_plugins.geometry.exact import ExactGeometryAccumulator, _model_ulp
from slime_plugins.geometry.histograms import LowFrequencyHistogramAccumulator
from slime_plugins.geometry.matrix_metrics import matrix_diagnostics, matrix_macro_summary, selected_view_ids
from slime_plugins.geometry.observer import GeometryObserver
from slime_plugins.geometry.optimizer_views import build_optimizer_parameter_views
from slime_plugins.geometry.projection import _hash_indices, count_sketch
from slime_plugins.geometry.support import SupportWindowSketch

NUM_GPUS = 0


@pytest.mark.unit
def test_low_frequency_histograms_label_approximations_and_keep_exact_adam_sums():
    accumulator = LowFrequencyHistogramAccumulator(
        ["global"],
        torch.device("cpu"),
        chunk_size=2,
    )
    vector = torch.tensor([0.0, 1.0e-3, 1.0e-1, 1.0])
    accumulator.add_sparsity(
        [0],
        {
            "delta_intended_fp32": vector,
            "delta_model": vector,
            "displacement": vector,
        },
    )
    accumulator.add_adam(
        [0],
        sqrt_v_hat=torch.tensor([0.0, 1.0e-8, 1.0e-6, 1.0e-4]),
        effective_eta=torch.tensor([1.0, 2.0, 4.0, 8.0]),
        gradient_energy=torch.tensor([0.0, 1.0, 1.0, 2.0]),
        eps=1.0e-8,
    )

    metrics = accumulator.finalize()["global"]

    assert metrics["delta_model_le_0.001_rms_fraction_sketch"] == pytest.approx(0.25)
    assert metrics["delta_model_top_0.1pct_coordinate_energy_fraction_sketch"] > 0.98
    assert metrics["sqrt_v_hat_le_eps_fraction"] == pytest.approx(0.5)
    assert metrics["sqrt_v_hat_le_10eps_fraction"] == pytest.approx(0.5)
    assert metrics["sqrt_v_hat_p50_sketch"] is not None
    assert metrics["effective_eta_gradient_energy_weighted_mean"] == pytest.approx(5.5)
    assert metrics["effective_eta_gradient_energy_weighted_cv"] == pytest.approx((6.75**0.5) / 5.5)


@pytest.mark.unit
def test_support_window_sketch_is_adjacent_persistent_and_resets_on_gap(tmp_path):
    descriptor = {
        "name": "weight",
        "start": 0,
        "stop": None,
        "numel": 4,
        "optimizer_branch": "sgd",
        "seed": 7,
    }
    path = tmp_path / "support.pt"
    sketch = SupportWindowSketch(
        group_names=["global"],
        descriptors=[descriptor],
        sample_size=4,
        window=2,
        device=torch.device("cpu"),
        path=path,
    )
    sketch.begin(1, report=True)
    sketch.add(0, torch.tensor([1.0, 0.0, 1.0, 0.0]), group_ids=(0,))
    first = sketch.finish()["global"]
    assert first["delta_model_support_jaccard_previous_update_sketch"] is None

    sketch.begin(2, report=True)
    sketch.add(0, torch.tensor([1.0, 1.0, 0.0, 0.0]), group_ids=(0,))
    second = sketch.finish()["global"]
    assert second["delta_model_support_jaccard_previous_update_sketch"] == pytest.approx(1 / 3)
    assert second["delta_model_window_update_frequency_mean_sketch"] == pytest.approx(0.5)
    assert second["delta_model_window_never_changed_fraction_sketch"] == pytest.approx(0.25)

    resumed = SupportWindowSketch(
        group_names=["global"],
        descriptors=[descriptor],
        sample_size=4,
        window=2,
        device=torch.device("cpu"),
        path=path,
    )
    resumed.begin(3, report=True)
    resumed.add(0, torch.tensor([0.0, 1.0, 0.0, 1.0]), group_ids=(0,))
    third = resumed.finish()["global"]
    assert third["delta_model_support_jaccard_previous_update_sketch"] == pytest.approx(1 / 3)
    assert third["delta_model_support_history_contiguous"] is True

    resumed.begin(5, report=True)
    resumed.add(0, torch.tensor([1.0, 0.0, 0.0, 0.0]), group_ids=(0,))
    after_gap = resumed.finish()["global"]
    assert after_gap["delta_model_support_jaccard_previous_update_sketch"] is None
    assert after_gap["delta_model_support_history_contiguous"] is False


@pytest.mark.unit
def test_support_window_global_fraction_is_parameter_weighted(tmp_path):
    descriptors = [
        {"name": "large", "start": 0, "stop": None, "numel": 100, "optimizer_branch": "sgd", "seed": 3},
        {"name": "small", "start": 0, "stop": None, "numel": 10, "optimizer_branch": "sgd", "seed": 3},
    ]
    sketch = SupportWindowSketch(
        group_names=["global"],
        descriptors=descriptors,
        sample_size=2,
        window=2,
        device=torch.device("cpu"),
        path=tmp_path / "weighted.pt",
    )
    sketch.begin(1, report=True)
    sketch.add(0, torch.ones(100), group_ids=(0,))
    sketch.add(1, torch.zeros(10), group_ids=(0,))

    metrics = sketch.finish()["global"]

    assert metrics["delta_model_support_sample_count"] == 4
    assert metrics["delta_model_support_estimated_population_count_sketch"] == pytest.approx(110)
    assert metrics["delta_model_support_fraction_sketch"] == pytest.approx(100 / 110)


@pytest.mark.unit
def test_small_matrix_diagnostics_are_exact_and_macro_is_explicitly_sampled():
    metrics = matrix_diagnostics(
        torch.eye(2),
        name="identity",
        seed=3,
        randomized_rank=1,
        include_orthogonality=True,
    )

    assert metrics["spectrum_method"] == "exact_svd"
    assert metrics["stable_rank"] == pytest.approx(2.0)
    assert metrics["effective_rank_99_energy"] == 2
    assert metrics["spectral_entropy"] == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert metrics["regularized_s95_to_s5"] == pytest.approx(1.0)
    assert metrics["row_norm_cv"] == pytest.approx(0.0)
    assert metrics["column_norm_cv"] == pytest.approx(0.0)
    assert metrics["orthogonality_error"] == pytest.approx(0.0)

    records = [
        {
            "operator": "q",
            "vectors": {"delta_model": {"stable_rank": value}},
        }
        for value in (2.0, 4.0)
    ]
    summary = matrix_macro_summary(records)["q"]["delta_model"]["stable_rank"]
    assert summary == {
        "median_sketch": pytest.approx(3.0),
        "iqr_sketch": pytest.approx(1.0),
        "matrix_count": 2,
    }


@pytest.mark.unit
def test_fixed_matrix_sampling_covers_each_real_optimizer_branch():
    views = [
        SimpleNamespace(
            name=f"matrix_{index}",
            optimizer_branch=branch,
            model_parameter=torch.nn.Parameter(torch.zeros(2, 2)),
            start=0,
            stop=None,
        )
        for index, branch in enumerate(("muon_matrix", "muon_matrix", "adam_fallback", "adam_fallback"))
    ]

    selected = selected_view_ids(views, seed=7, count=1)

    assert len(selected) == 2
    assert {views[index].optimizer_branch for index in selected} == {"muon_matrix", "adam_fallback"}


@pytest.mark.unit
def test_exact_geometry_accumulator_records_required_norms_dots_and_ratios():
    accumulator = ExactGeometryAccumulator(["global", "optimizer_branch/sgd"], torch.device("cpu"))
    theta = torch.tensor([3.0, 4.0], dtype=torch.float32)
    reference = torch.tensor([2.0, 4.0], dtype=torch.float32)
    g_raw = torch.tensor([2.0, 0.0])
    g_opt = torch.tensor([1.0, 0.0])
    d_data = g_opt.clone()
    zeros = torch.zeros_like(theta)
    intended = torch.tensor([-0.25, 0.0])
    realized = torch.tensor([-0.25, 0.0])
    accumulator.add(
        [0, 1],
        {
            "theta_before": theta,
            "theta_reference": reference,
            "g_raw": g_raw,
            "g_opt": g_opt,
            "d_data": d_data,
            "d_wd": zeros,
            "delta_data_fp32": intended,
            "delta_wd_fp32": zeros,
            "delta_intended_fp32": intended,
            "delta_model": realized,
            "displacement": torch.tensor([0.75, 0.0]),
        },
    )

    metrics = accumulator.finalize()["global"]
    assert metrics["theta_before_l2"] == pytest.approx(5.0)
    assert metrics["theta_before_rms"] == pytest.approx((25 / 2) ** 0.5)
    assert metrics["g_raw_exact_zero_fraction"] == pytest.approx(0.5)
    assert metrics["dot_g_opt_delta_intended_fp32"] == pytest.approx(-0.25)
    assert metrics["cos_g_raw_g_opt"] == pytest.approx(1.0)
    assert metrics["gradient_directional_step"] == pytest.approx(0.25)
    assert metrics["displacement_to_reference_ratio"] == pytest.approx(0.75 / (20**0.5))
    assert accumulator.finalize()["optimizer_branch/sgd"] == metrics


@pytest.mark.unit
def test_exact_geometry_accumulator_records_bfloat16_realization_bins():
    theta = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.bfloat16)
    # BF16 ULP at one is 1/128.  Exercise below-half-ULP, zeroed,
    # attenuated, amplified, and sign-flip accounting with exact values.
    intended = torch.tensor([1 / 512, 1 / 128, 1 / 64, -1 / 128])
    realized = torch.tensor([0.0, 1 / 256, 1 / 32, 1 / 128])
    accumulator = ExactGeometryAccumulator(["global"], torch.device("cpu"))
    accumulator.add(
        [0],
        {
            "theta_before": theta,
            "theta_reference": theta,
            "delta_intended_fp32": intended,
            "delta_model": realized,
        },
    )

    metrics = accumulator.finalize()["global"]
    assert metrics["model_change_fraction"] == pytest.approx(0.75)
    assert metrics["intended_below_half_ulp_fraction"] == pytest.approx(0.25)
    assert metrics["energy_survival"] == pytest.approx(float(realized.square().sum() / intended.square().sum()))
    assert metrics["ulp_ratio_bins"]["[0.25,0.5)"]["realized_nonzero_fraction"] == 0.0
    assert metrics["ulp_ratio_bins"]["[1,2)"]["sign_flip_fraction"] == pytest.approx(0.5)


@pytest.mark.unit
def test_exact_geometry_rejects_nonfinite_success_metrics():
    accumulator = ExactGeometryAccumulator(["global"], torch.device("cpu"))
    accumulator.add([0], {"theta_before": torch.tensor([float("nan")])})

    with pytest.raises(ValueError, match="non-finite"):
        accumulator.finalize()


@pytest.mark.unit
def test_bfloat16_ulp_uses_the_parameter_positive_direction_for_negative_values():
    theta = torch.tensor([-1.0], dtype=torch.bfloat16)
    intended = torch.tensor([0.003], dtype=torch.float32)
    accumulator = ExactGeometryAccumulator(["global"], torch.device("cpu"))
    accumulator.add(
        [0],
        {
            "theta_before": theta,
            "theta_reference": theta,
            "delta_intended_fp32": intended,
            "delta_model": torch.zeros_like(intended),
        },
    )

    # nextafter(-1, +inf) - (-1) == 1/256 in BF16, so 0.003 is above
    # half an ULP. Using abs(theta) would incorrectly use the 1/128 spacing.
    assert accumulator.finalize()["global"]["intended_below_half_ulp_fraction"] == 0.0


@pytest.mark.unit
def test_bfloat16_zero_ulp_fallback_uses_the_smallest_subnormal(monkeypatch):
    monkeypatch.setattr(torch, "nextafter", lambda value, _direction: value.clone())

    ulp = _model_ulp(torch.tensor([0.0], dtype=torch.bfloat16))

    expected = torch.finfo(torch.bfloat16).tiny * torch.finfo(torch.bfloat16).eps
    assert float(ulp[0]) == pytest.approx(expected)


@pytest.mark.unit
def test_optimizer_branch_comes_from_membership_not_parameter_shape():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    entries = [(f"chunk0.{name}", parameter, []) for name, parameter in model.named_parameters()]

    views = build_optimizer_parameter_views(entries, optimizer, requested_optimizer="muon")

    assert {view.optimizer_branch for view in views} == {"adam_fallback"}
    assert {view.model_parameter.ndim for view in views} == {1, 2}


@pytest.mark.unit
def test_muon_fused_qkv_scale_is_recovered_componentwise():
    parameter = torch.nn.Parameter(torch.zeros(8, 2))
    optimizer_parameter = torch.nn.Parameter(torch.zeros_like(parameter))
    optimizer_parameter.is_qkv = True
    inner = SimpleNamespace(
        mode="blockwise",
        pg_collection=None,
        slime_muon_scale_mode="spectral",
        slime_muon_extra_scale_factor=0.2,
        split_qkv=True,
        qkv_split_shapes=(4, 2, 2),
        is_qkv_fn=lambda value: bool(getattr(value, "is_qkv", False)),
    )
    view = SimpleNamespace(
        name="linear_qkv.weight",
        model_parameter=parameter,
        optimizer_parameter=optimizer_parameter,
        inner_optimizer=inner,
    )

    scale, metadata = _muon_scale_factor(view)

    assert metadata["muon_scale_application"] == "qkv_componentwise"
    assert scale.shape == (8, 1)
    assert scale[:4].unique().item() == pytest.approx(0.4)
    assert scale[4:].unique().item() == pytest.approx((2.0**0.5) * 0.2)


@pytest.mark.unit
def test_observer_recovers_exact_adamw_data_and_decay_updates(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    model = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.1)
    args = _geometry_args(tmp_path)
    args.optimizer = "adam"
    args.lr = 0.01
    args.weight_decay = 0.1
    args.decoupled_weight_decay = True
    observer = GeometryObserver(args, [model])

    parameter.grad = torch.tensor([0.25, -0.5, 1.0])
    observer.after_backward(0, 0, ["math"], optimizer=optimizer)
    before = parameter.detach().clone()
    optimizer.step()
    observer.after_step(
        update_successful=True,
        grad_norm=float(torch.linalg.vector_norm(parameter.grad)),
        num_zeros_in_grad=0,
        optimizer=optimizer,
    )

    record = json.loads((tmp_path / "actor" / "metrics.jsonl").read_text())
    metrics = record["groups"]["global"]
    expected_delta = parameter.detach() - before
    assert metrics["delta_model_l2"] == pytest.approx(float(torch.linalg.vector_norm(expected_delta)))
    assert metrics["delta_wd_fp32_l2"] > 0
    assert metrics["cos_delta_intended_fp32_delta_model"] == pytest.approx(1.0, abs=2e-5)
    assert metrics["energy_survival"] == pytest.approx(1.0, abs=2e-5)
    assert record["actual_optimizer_branches"]["adam"]["adam_beta1"] == pytest.approx(0.8)
    branch_metrics = record["groups"]["optimizer_branch/adam"]
    assert branch_metrics["weight_decay_metrics_applicability"] == "applicable"
    assert branch_metrics["d_wd_to_d_data_ratio"] > 0
    assert branch_metrics["d_wd_to_d_adam_ratio"] > 0


@pytest.mark.unit
def test_observer_uses_fused_adam_parameter_group_step(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    model = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    args = _geometry_args(tmp_path)
    args.optimizer = "adam"
    args.lr = 0.01
    observer = GeometryObserver(args, [model])

    parameter.grad = torch.tensor([0.25, -0.5, 1.0])
    observer.after_backward(0, 0, ["math"], optimizer=optimizer)
    optimizer.step()

    # Apex/TE FusedAdam keeps the authoritative shared step on the group. A
    # restored per-parameter step may remain at zero because the fused kernel
    # does not advance it.
    optimizer.param_groups[0]["step"] = 1
    optimizer.state[parameter]["step"].zero_()
    observer.after_step(
        update_successful=True,
        grad_norm=float(torch.linalg.vector_norm(parameter.grad)),
        num_zeros_in_grad=0,
        optimizer=optimizer,
    )

    record = json.loads((tmp_path / "actor" / "metrics.jsonl").read_text())
    assert record["actual_optimizer_branches"]["adam"]["optimizer_step"] == 1


@pytest.mark.unit
def test_failed_update_is_saved_without_valid_geometry(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = _geometry_args(tmp_path)
    observer = GeometryObserver(args, [model])
    model(torch.ones(1, 2)).sum().backward()
    observer.after_backward(0, 0, ["science"], optimizer=optimizer)

    observer.after_step(
        update_successful=False,
        grad_norm=float("nan"),
        num_zeros_in_grad=None,
        optimizer=optimizer,
        failure_reason="nonfinite_gradient",
    )

    record = json.loads((tmp_path / "actor" / "metrics.jsonl").read_text())
    assert record["update_successful"] is False
    assert record["valid_update_metrics"] is False
    assert record["failure_reason"] == "nonfinite_gradient"
    assert record["groups"] == {}
    assert record["low_frequency_observation"] is False
    assert record["low_frequency_approximation"] is None
    assert record["run_update_counters"] == {
        "successful": 0,
        "failed_or_skipped": 1,
        "clipped": 0,
    }


@pytest.mark.unit
def test_failure_is_saved_even_outside_geometry_interval(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = _geometry_args(tmp_path)
    args.geometry_interval = 2
    observer = GeometryObserver(args, [model])

    model(torch.ones(1, 2)).sum().backward()
    observer.after_backward(0, 0, ["math"], optimizer=optimizer)
    optimizer.step()
    observer.after_step(update_successful=True, grad_norm=1.0, num_zeros_in_grad=0, optimizer=optimizer)

    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    observer.after_backward(
        0,
        1,
        ["science"],
        optimizer=optimizer,
        actual_batch_size=1,
        effective_token_count=3,
    )
    observer.after_step(
        update_successful=False,
        grad_norm=float("nan"),
        num_zeros_in_grad=None,
        optimizer=optimizer,
        failure_reason="nonfinite_gradient",
    )

    records = [json.loads(line) for line in (tmp_path / "actor" / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[-1]["observation_id"] == 1
    assert records[-1]["low_frequency_observation"] is False
    assert records[-1]["source_counts"] == {"science": 1}
    assert records[-1]["groups"] == {}


@pytest.mark.unit
def test_exact_geometry_is_saved_on_every_successful_update(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = _geometry_args(tmp_path)
    args.geometry_interval = 4
    observer = GeometryObserver(args, [model])

    for step in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        observer.after_backward(0, step, ["math"], optimizer=optimizer)
        optimizer.step()
        observer.after_step(
            update_successful=True,
            grad_norm=1.0,
            num_zeros_in_grad=0,
            optimizer=optimizer,
        )

    records = [json.loads(line) for line in (tmp_path / "actor" / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["low_frequency_observation"] is True
    assert records[0]["vector_file"] is not None
    assert records[1]["low_frequency_observation"] is False
    assert records[1]["vector_file"] is None
    assert records[1]["groups"]["global"]["delta_model_l2"] > 0
    assert "update_norm_sketch" not in records[1]["groups"]["global"]


@pytest.mark.unit
def test_exact_displacement_reference_survives_observer_recreation(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    model = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    args = _geometry_args(tmp_path)

    parameter.grad = torch.tensor([1.0, 2.0])
    first = GeometryObserver(args, [model])
    first.after_backward(0, 0, ["math"], optimizer=optimizer)
    optimizer.step()
    first.after_step(update_successful=True, grad_norm=5**0.5, num_zeros_in_grad=0, optimizer=optimizer)

    optimizer.zero_grad()
    parameter.grad = torch.tensor([-1.0, 1.0])
    second = GeometryObserver(args, [model])
    second.after_backward(1, 0, ["math"], optimizer=optimizer)
    optimizer.step()
    second.after_step(update_successful=True, grad_norm=2**0.5, num_zeros_in_grad=0, optimizer=optimizer)

    records = [json.loads(line) for line in (tmp_path / "actor" / "metrics.jsonl").read_text().splitlines()]
    assert [record["observation_id"] for record in records] == [0, 1]
    assert records[-1]["groups"]["global"]["displacement_l2"] == pytest.approx(
        float(torch.linalg.vector_norm(parameter.detach() - torch.tensor([1.0, -1.0])))
    )


@pytest.mark.unit
def test_fused_qkv_and_gate_up_are_reported_before_and_after_semantic_split(tmp_path):
    class FusedBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_attention_heads=2,
                num_query_groups=1,
                kv_channels=2,
            )
            self.linear_qkv = torch.nn.Linear(2, 8, bias=False)
            self.linear_fc1 = torch.nn.Linear(2, 6, bias=False)

        def forward(self, value):
            return self.linear_qkv(value).sum() + self.linear_fc1(value).sum()

    model = FusedBlock()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = _geometry_args(tmp_path)
    model(torch.ones(1, 2)).backward()
    observer = GeometryObserver(args, [model])
    observer.after_backward(0, 0, ["math"], optimizer=optimizer)
    optimizer.step()
    observer.after_step(update_successful=True, grad_norm=1.0, num_zeros_in_grad=0, optimizer=optimizer)

    groups = json.loads((tmp_path / "actor" / "metrics.jsonl").read_text())["groups"]
    assert groups["operator_type/qkv_fused"]["parameter_count"] == 16
    assert groups["operator_type/q"]["parameter_count"] == 8
    assert groups["operator_type/k"]["parameter_count"] == 4
    assert groups["operator_type/v"]["parameter_count"] == 4
    assert groups["operator_type/gate_up_fused"]["parameter_count"] == 12
    assert groups["operator_type/gate"]["parameter_count"] == 6
    assert groups["operator_type/up"]["parameter_count"] == 6


@pytest.mark.unit
def test_count_sketch_is_deterministic_and_linear():
    left = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    right = torch.linspace(-1, 1, 12).reshape(3, 4)
    kwargs = {"dim": 127, "seed": 19, "name": "rank0:param", "chunk_size": 5}

    first = count_sketch(left, **kwargs)
    second = count_sketch(left, **kwargs)
    combined = count_sketch(left + right, **kwargs)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(combined, first + count_sketch(right, **kwargs))


@pytest.mark.unit
def test_count_sketch_hash_does_not_tie_sign_to_power_of_two_bucket():
    hashes = _hash_indices(torch.arange(4096, dtype=torch.int64), base_seed=17)
    buckets = torch.remainder(hashes, 256)
    signs = torch.bitwise_and(torch.bitwise_right_shift(hashes, 31), 1)

    signs_by_bucket = {
        int(bucket): set(signs[buckets == bucket].tolist()) for bucket in torch.unique(buckets).tolist()
    }
    assert len(signs_by_bucket) == 256
    assert sum(values == {0, 1} for values in signs_by_bucket.values()) > 200


def _geometry_args(tmp_path):
    return SimpleNamespace(
        _slime_model_role="actor",
        geometry_roles=["actor"],
        geometry_projection_dim=4096,
        geometry_interval=1,
        geometry_seed=7,
        geometry_sketch_chunk_size=8,
        geometry_parameter_include=".*",
        geometry_parameter_exclude=None,
        geometry_output_dir=str(tmp_path),
        geometry_save_vectors=True,
        geometry_group_by="layer",
        optimizer="sgd",
        lr=0.1,
        clip_grad=0.0,
        weight_decay=0.0,
        sgd_momentum=0.0,
        muon_momentum=0.0,
        muon_num_ns_steps=0,
        muon_scale_mode=None,
        advantage_estimator="grpo",
        use_opd=False,
        opd_type=None,
        opd_kl_coef=0.0,
        opd_task_reward_weight=0.0,
    )


@pytest.mark.unit
def test_observer_records_gradient_update_and_displacement(tmp_path):
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = _geometry_args(tmp_path)
    observer = GeometryObserver(args, [model])

    loss = model(torch.randn(5, 4)).square().mean()
    loss.backward()
    observer.after_backward(
        rollout_id=2,
        step_id=0,
        source_names=["math", "science"],
        optimizer=optimizer,
        actual_batch_size=2,
        effective_token_count=7,
    )
    optimizer.step()
    observer.after_step(
        update_successful=True,
        grad_norm=1.25,
        num_zeros_in_grad=0,
        optimizer=optimizer,
    )

    metrics_path = tmp_path / "actor" / "metrics.jsonl"
    record = json.loads(metrics_path.read_text().strip())
    assert record["source_counts"] == {"math": 1, "science": 1}
    assert record["optimizer"] == "sgd"
    assert record["projection_dim"] == args.geometry_projection_dim
    assert record["projection_seed"] == args.geometry_seed
    assert record["groups"]["global"]["update_norm_sketch"] > 0
    assert record["groups"]["global"]["displacement_norm_sketch"] > 0
    assert record["groups"]["global"]["cos_gradient_update_sketch"] < -0.99
    assert record["groups"]["global"]["cos_g_opt_delta_model"] < -0.99
    assert record["groups"]["global"]["delta_intended_fp32_l2"] > 0
    assert record["groups"]["optimizer_branch/sgd"]["parameter_fraction"] == 1.0
    assert record["groups"]["optimizer_branch/sgd"]["weight_decay_metrics_applicability"] == "not_applicable"
    assert record["groups"]["optimizer_branch/sgd"]["d_wd_to_d_data_ratio"] is None
    assert record["groups"]["optimizer_branch/sgd"]["delta_wd_to_delta_data_ratio"] is None
    assert record["schema_version"] == 2
    assert record["actual_batch_size"] == 2
    assert record["effective_token_count"] == 7
    assert record["cumulative_prompt_count"] == 2
    assert record["cumulative_effective_token_count"] == 7
    assert record["valid_update_metrics"] is True
    assert record["actual_optimizer_branches"]["sgd"]["learning_rate"] == pytest.approx(0.1)
    assert (tmp_path / "actor" / record["vector_file"]).is_file()
    assert (tmp_path / "actor" / "initial_projection.pt").is_file()
    assert (tmp_path / "actor" / "exact_reference" / "rank_00000.pt").is_file()

    recreated = GeometryObserver(args, [model])
    with pytest.raises(ValueError, match=r"ahead of \(or equal to\) the resumed checkpoint"):
        recreated.after_backward(
            rollout_id=2,
            step_id=0,
            source_names=["math"],
        )


@pytest.mark.unit
def test_observer_mirrors_selected_geometry_groups_to_wandb(tmp_path, monkeypatch):
    from slime.utils import logging_utils

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = _geometry_args(tmp_path)
    args.use_wandb = True
    args.geometry_wandb_groups = "global,optimizer_branch/sgd"
    captured = {}
    monkeypatch.setattr(
        logging_utils,
        "log",
        lambda _args, metrics, step_key: captured.update({"metrics": metrics, "step_key": step_key}),
    )
    observer = GeometryObserver(args, [model])
    model(torch.ones(1, 3)).sum().backward()
    observer.after_backward(
        rollout_id=0,
        step_id=0,
        source_names=["science"],
        optimizer=optimizer,
    )
    optimizer.step()
    observer.after_step(
        update_successful=True,
        grad_norm=1.0,
        num_zeros_in_grad=0,
        optimizer=optimizer,
    )

    assert captured["step_key"] == "geometry/step"
    assert captured["metrics"]["geometry/source_count/science"] == 1
    assert "geometry/global/gradient_norm" in captured["metrics"]
    assert "geometry/optimizer_branch_sgd/delta_model_l2" in captured["metrics"]
    assert not any(key.startswith("geometry/other/") for key in captured["metrics"])


@pytest.mark.unit
def test_baseline_signature_rejects_changed_muon_scaling(tmp_path):
    model = torch.nn.Linear(3, 2)
    args = _geometry_args(tmp_path)
    args.optimizer = "muon"
    args.muon_momentum = 0.95
    args.muon_num_ns_steps = 5
    args.muon_scale_mode = "spectral"
    args.muon_extra_scale_factor = 0.2

    model(torch.ones(1, 3)).sum().backward()
    GeometryObserver(args, [model]).after_backward(
        rollout_id=0,
        step_id=0,
        source_names=["math"],
    )

    changed_args = SimpleNamespace(**vars(args))
    changed_args.muon_extra_scale_factor = 1.0
    model.zero_grad(set_to_none=True)
    model(torch.ones(1, 3)).sum().backward()
    with pytest.raises(ValueError, match="baseline.*incompatible"):
        GeometryObserver(changed_args, [model]).after_backward(
            rollout_id=1,
            step_id=0,
            source_names=["math"],
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
