"""CPU-only tests for AdamW/SGD/Muon optimizer dispatch."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.optimizer_factory import (
    build_megatron_optimizer,
    configure_optimizer_runtime,
)

NUM_GPUS = 0


@pytest.mark.unit
def test_adamw_keeps_distributed_optimizer():
    args = SimpleNamespace(
        optimizer="adam",
        fp16=False,
        use_distributed_optimizer=False,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
    )

    configure_optimizer_runtime(args)

    assert args.use_distributed_optimizer is True
    assert args.overlap_grad_reduce is True


@pytest.mark.unit
def test_sgd_disables_incompatible_megatron_flags():
    args = SimpleNamespace(
        optimizer="sgd",
        fp16=False,
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
    )

    configure_optimizer_runtime(args)

    assert args.use_distributed_optimizer is False
    assert args.overlap_grad_reduce is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False


@pytest.mark.unit
@pytest.mark.parametrize("name", ["muon", "dist_muon"])
def test_muon_disables_incompatible_megatron_flags(name):
    args = SimpleNamespace(
        optimizer=name,
        fp16=False,
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
    )

    configure_optimizer_runtime(args)

    assert args.use_distributed_optimizer is False
    assert args.overlap_grad_reduce is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False


@pytest.mark.unit
def test_muon_rejects_fp16():
    with pytest.raises(ValueError, match="FP16"):
        configure_optimizer_runtime(SimpleNamespace(optimizer="muon", fp16=True))


@pytest.mark.unit
@pytest.mark.parametrize("actor_optimizer", ["sgd", "muon"])
def test_ppo_adam_critic_restores_pre_actor_overlap_flags(actor_optimizer):
    args = SimpleNamespace(
        optimizer=actor_optimizer,
        fp16=False,
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=False,
    )
    configure_optimizer_runtime(args)
    args.optimizer = "adam"
    configure_optimizer_runtime(args)

    assert args.use_distributed_optimizer is True
    assert args.overlap_grad_reduce is True
    assert args.overlap_param_gather is True
    assert args.overlap_param_gather_with_optimizer_step is False


@pytest.mark.unit
def test_generic_optimizer_dispatch():
    calls = []
    config = SimpleNamespace(optimizer="sgd")
    sentinel = object()

    def generic_factory(**kwargs):
        calls.append(kwargs)
        return sentinel

    result = build_megatron_optimizer(
        config=config,
        model_chunks=["model"],
        use_gloo_process_groups=False,
        generic_factory=generic_factory,
    )

    assert result is sentinel
    assert calls == [{"config": config, "model_chunks": ["model"], "use_gloo_process_groups": False}]


@pytest.mark.unit
def test_sgd_rejects_distributed_optimizer_if_runtime_normalization_is_bypassed():
    config = SimpleNamespace(optimizer="sgd", use_distributed_optimizer=True)

    with pytest.raises(ValueError, match="SGD.*use_distributed_optimizer=False"):
        build_megatron_optimizer(
            config=config,
            model_chunks=["model"],
            use_gloo_process_groups=False,
            generic_factory=lambda **_: object(),
        )


@pytest.mark.unit
def test_sgd_materializes_fused_momentum_state_for_checkpoint_loading():
    param = torch.nn.Parameter(torch.ones(2))

    class FakeFusedSGD:
        def __init__(self):
            self.param_groups = [{"params": [param], "momentum": 0.0}]
            self.state = defaultdict(dict)

        def get_momentums(self, params):
            for current_param in params:
                self.state[current_param].setdefault("momentum_buffer", torch.zeros_like(current_param))

    base_optimizer = FakeFusedSGD()
    wrapped_optimizer = SimpleNamespace(optimizer=base_optimizer, init_state_fn=None)
    chained_optimizer = SimpleNamespace(chained_optimizers=[wrapped_optimizer])

    result = build_megatron_optimizer(
        config=SimpleNamespace(optimizer="sgd", use_distributed_optimizer=False),
        model_chunks=["model"],
        use_gloo_process_groups=False,
        generic_factory=lambda **_: chained_optimizer,
    )
    wrapped_optimizer.init_state_fn(base_optimizer)

    assert result is chained_optimizer
    assert torch.equal(base_optimizer.state[param]["momentum_buffer"], torch.zeros_like(param))


@pytest.mark.unit
@pytest.mark.parametrize("name, layer_wise", [("muon", False), ("dist_muon", True)])
def test_muon_optimizer_dispatch(name, layer_wise):
    calls = []
    config = SimpleNamespace(optimizer=name, use_distributed_optimizer=False, fp16=False)
    result = SimpleNamespace()

    def muon_factory(**kwargs):
        calls.append(kwargs)
        return result

    actual = build_megatron_optimizer(
        config=config,
        model_chunks=["model"],
        use_gloo_process_groups=True,
        muon_factory=muon_factory,
    )

    assert actual is result
    assert result.slime_optimizer_name == name
    assert calls[0]["layer_wise_distributed_optimizer"] is layer_wise


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
