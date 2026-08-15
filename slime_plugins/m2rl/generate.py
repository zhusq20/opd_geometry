"""Route ordinary tasks to SGLang and WorkBench tasks to the multi-turn agent."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from slime.rollout.sglang_rollout import generate as sglang_generate
from slime.utils.types import Sample


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    metadata = sample.metadata or {}
    if metadata.get("training_mode") == "sft":
        from slime_plugins.m2rl.hybrid import make_sft_sample

        return make_sft_sample(args, sample)
    if metadata.get("rm_type") == "workbench":
        from slime_plugins.m2rl.workbench import generate as workbench_generate

        result = await workbench_generate(args, sample, sampling_params)
        if getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang":
            result.metadata["workbench_reward"] = result.reward
            # Let generate_and_rm invoke the OPD teacher reward function.
            result.reward = None
        return result
    result = await sglang_generate(args, sample, sampling_params)
    if metadata.get("training_mode") == "opd":
        result.train_metadata = {"training_mode": "opd"}
    return result
