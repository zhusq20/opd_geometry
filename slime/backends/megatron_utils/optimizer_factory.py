"""Optimizer dispatch helpers for Slime's Megatron backend.

Megatron exposes Muon through a separate factory because a Muon run is a
chained optimizer: Muon updates matrix-shaped transformer weights while AdamW
updates embeddings, norms, biases, and output parameters. Keeping the dispatch
here makes the behavior explicit and independently testable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

MUON_OPTIMIZERS = frozenset({"muon", "dist_muon"})


def _init_sgd_checkpoint_state(optimizer: Any, config: Any = None) -> None:
    """Materialize the lazy SGD state required by sharded checkpoint loading."""

    if optimizer is None:
        return
    for group in optimizer.param_groups:
        params = group["params"]
        if hasattr(optimizer, "get_momentums"):
            # Transformer Engine/Apex FusedSGD creates momentum buffers even
            # when momentum is zero, so match the state written by its first
            # optimizer step.
            optimizer.get_momentums(params)
        elif group.get("momentum", 0.0):
            for param in params:
                optimizer.state[param].setdefault("momentum_buffer", torch.zeros_like(param.data))


def _enable_sgd_checkpoint_loading(optimizer: Any) -> None:
    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        for child in children:
            _enable_sgd_checkpoint_loading(child)
        return
    if hasattr(optimizer, "init_state_fn"):
        optimizer.init_state_fn = _init_sgd_checkpoint_state


def _annotate_muon_geometry(optimizer: Any, config: Any) -> None:
    """Expose immutable Muon scaling choices to the observation adapter."""

    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        for child in children:
            _annotate_muon_geometry(child, config)
        return
    inner = getattr(optimizer, "optimizer", optimizer)
    if "muon" not in type(inner).__name__.lower():
        return
    inner.slime_muon_scale_mode = config.muon_scale_mode
    inner.slime_muon_extra_scale_factor = config.muon_extra_scale_factor


def is_muon_optimizer(name: str | None) -> bool:
    return str(name or "").lower() in MUON_OPTIMIZERS


def configure_optimizer_runtime(args: Any) -> Any:
    """Normalize runtime flags that are incompatible with the selected optimizer.

    Slime historically forced Megatron's distributed optimizer for every run.
    The pinned Megatron version supports that path only for Adam-compatible
    optimizers. SGD must use Megatron's regular optimizer wrapper, while Muon
    uses either replicated Muon or its layer-wise wrapper. Both reject the
    ordinary distributed optimizer and communication overlap flags. This
    function is idempotent so it can be applied both to the base arguments and
    to role-specific PPO overrides.
    """

    optimizer_name = str(getattr(args, "optimizer", "adam")).lower()
    if optimizer_name not in {"adam", "sgd", *MUON_OPTIMIZERS}:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; expected adam (AdamW), sgd, muon, or dist_muon.")

    overlap_flags = (
        "overlap_grad_reduce",
        "overlap_param_gather",
        "overlap_param_gather_with_optimizer_step",
    )
    if not hasattr(args, "_slime_adam_overlap_flags"):
        args._slime_adam_overlap_flags = {
            flag: bool(getattr(args, flag)) for flag in overlap_flags if hasattr(args, flag)
        }

    previous_optimizer = getattr(args, "_slime_last_configured_optimizer", None)
    if is_muon_optimizer(optimizer_name):
        if getattr(args, "fp16", False):
            raise ValueError("Muon in the pinned Megatron version requires BF16 or FP32; FP16 is unsupported.")
    if optimizer_name == "sgd" or is_muon_optimizer(optimizer_name):
        for flag in (
            "use_distributed_optimizer",
            *overlap_flags,
        ):
            if hasattr(args, flag):
                setattr(args, flag, False)
    else:
        # Preserve Slime's established ZeRO default for AdamW.
        args.use_distributed_optimizer = True
        # A PPO critic can override a Muon or SGD actor back to AdamW. Restore
        # the overlap choices parsed before non-Adam normalization so the fixed
        # critic has the same runtime configuration in every cell.
        if previous_optimizer == "sgd" or is_muon_optimizer(previous_optimizer):
            for flag, value in args._slime_adam_overlap_flags.items():
                setattr(args, flag, value)
        else:
            for flag in overlap_flags:
                if hasattr(args, flag):
                    args._slime_adam_overlap_flags[flag] = bool(getattr(args, flag))
    args._slime_last_configured_optimizer = optimizer_name
    return args


def build_megatron_optimizer(
    *,
    config: Any,
    model_chunks: list[Any],
    use_gloo_process_groups: bool,
    generic_factory: Callable[..., Any] | None = None,
    muon_factory: Callable[..., Any] | None = None,
) -> Any:
    """Build AdamW/SGD/Muon without importing optional Muon code eagerly."""

    optimizer_name = str(config.optimizer).lower()
    if optimizer_name not in MUON_OPTIMIZERS:
        if optimizer_name == "sgd" and getattr(config, "use_distributed_optimizer", False):
            raise ValueError(
                "SGD must be constructed with use_distributed_optimizer=False in the pinned Megatron version."
            )
        if generic_factory is None:
            from megatron.core.optimizer import get_megatron_optimizer

            generic_factory = get_megatron_optimizer
        optimizer = generic_factory(
            config=config,
            model_chunks=model_chunks,
            use_gloo_process_groups=use_gloo_process_groups,
        )
        if optimizer_name == "sgd":
            _enable_sgd_checkpoint_loading(optimizer)
        return optimizer

    if getattr(config, "use_distributed_optimizer", False):
        raise ValueError("Muon must be constructed with use_distributed_optimizer=False.")
    if getattr(config, "fp16", False):
        raise ValueError("Muon does not support FP16 in the pinned Megatron version; use BF16 or FP32.")

    if muon_factory is None:
        try:
            from megatron.core.optimizer import muon as megatron_muon
        except ImportError as exc:  # pragma: no cover - depends on external Megatron installation
            raise RuntimeError("The pinned Megatron installation does not provide Muon support.") from exc
        if not getattr(megatron_muon, "HAVE_EMERGING_OPTIMIZERS", False):
            raise RuntimeError(
                "Muon requires NVIDIA Emerging-Optimizers. Rebuild the Slime environment or run "
                "`pip install git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0`."
            )
        muon_factory = megatron_muon.get_megatron_muon_optimizer

    # Megatron's factory mutates config.optimizer to "adam" while constructing
    # the chained AdamW branch. Retain the experiment label on the returned
    # object so logging/checkpoint tooling can still identify the requested run.
    optimizer = muon_factory(
        config=config,
        model_chunks=model_chunks,
        use_gloo_process_groups=use_gloo_process_groups,
        layer_wise_distributed_optimizer=optimizer_name == "dist_muon",
    )
    optimizer.slime_optimizer_name = optimizer_name
    _annotate_muon_geometry(optimizer, config)
    return optimizer
