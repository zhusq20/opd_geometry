import logging
import math
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    # Keep Slime's historical power-of-two series, but always include the
    # benchmark-standard endpoints. In particular LiveCodeBench is normally
    # sampled with n=10 and papers need pass@1/pass@5/pass@10; the former code
    # silently emitted only pass@1/2/4/8. n=1 online evaluation also needs an
    # explicit pass@1 instead of an empty result.
    pass_rate_name_list = sorted(
        {
            1,
            group_size,
            *[2**i for i in range(int(math.log2(group_size)) + 1)],
            *[k for k in (5, 10) if k <= group_size],
        }
    )

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id


def updates_per_rollout(args) -> int:
    """Return the configured number of optimizer updates in one rollout."""

    sampled = int(args.rollout_batch_size) * int(args.n_samples_per_prompt)
    global_batch = int(args.global_batch_size)
    if sampled % global_batch:
        raise ValueError(
            "rollout_batch_size*n_samples_per_prompt must be divisible by global_batch_size "
            "to assign an unambiguous optimizer-update axis."
        )
    return sampled // global_batch


def num_updates_before_rollout(args, rollout_id: int) -> int:
    return int(rollout_id) * updates_per_rollout(args)


def rollout_prompt_count(args, rollout_id: int) -> int:
    """Return the prompt-group target for a rollout, including an epoch tail.

    ``RolloutManager`` annotates its private argument namespace with the usable
    dataset size when ``--num-epoch`` is used.  Keeping this calculation in a
    dependency-light helper makes the non-full final rollout explicit and
    unit-testable while preserving the historical fixed-size behavior for
    ``--num-rollout`` launches.
    """

    prompts_per_epoch = getattr(args, "rollout_prompts_per_epoch", None)
    steps_per_epoch = getattr(args, "rollout_steps_per_epoch", None)
    if prompts_per_epoch is None and steps_per_epoch is None:
        return int(args.rollout_batch_size)
    if prompts_per_epoch is None or steps_per_epoch is None:
        raise ValueError("Epoch rollout metadata must define both prompt and step counts.")

    prompts_per_epoch = int(prompts_per_epoch)
    steps_per_epoch = int(steps_per_epoch)
    batch_size = int(args.rollout_batch_size)
    if prompts_per_epoch <= 0 or steps_per_epoch <= 0 or batch_size <= 0:
        raise ValueError("Epoch rollout prompt, step, and batch counts must be positive.")

    step_in_epoch = int(rollout_id) % steps_per_epoch
    remaining = prompts_per_epoch - step_in_epoch * batch_size
    if remaining <= 0:
        raise ValueError(
            f"Rollout {rollout_id} is outside an epoch with {prompts_per_epoch} prompts, "
            f"batch size {batch_size}, and {steps_per_epoch} steps."
        )
    return min(batch_size, remaining)
