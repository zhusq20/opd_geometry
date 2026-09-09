"""Independent calibration and twenty one-step draws from Uniform/250."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .optimizer_views import _leaf_optimizers
from .prompting import validate_prompt_runtime
from .reference_bank import bank_identity, loss_record, write_json
from .sampler import TASKS, integer_allocation


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return copy.deepcopy(value)


def snapshot_actor(actor):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return {
        "model": [cpu_state(module.state_dict()) for module in actor.model],
        "optimizer": cpu_state(actor.optimizer.state_dict()),
        # DistributedOptimizer.state_dict() excludes master parameters and
        # Adam moments. Its separate parameter-state API preserves both.
        "optimizer_parameter_state": [
            cpu_state(optimizer.get_parameter_state_dp_zero()) if isinstance(optimizer, DistributedOptimizer) else None
            for optimizer in _leaf_optimizers(actor.optimizer)
        ],
        "scheduler": copy.deepcopy(actor.opt_param_scheduler.state_dict()),
        "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()),
        "megatron_rng": copy.deepcopy(get_cuda_rng_tracker().get_states()),
    }


def assert_restored_state(actual, expected):
    if torch.is_tensor(expected):
        if not torch.equal(actual.detach().cpu(), expected):
            raise RuntimeError("Common-checkpoint optimizer parameter state was not restored exactly")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise RuntimeError("Common-checkpoint optimizer state keys differ after restoration")
        for key in expected:
            assert_restored_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_restored_state(actual_item, expected_item)
    elif actual != expected:
        raise RuntimeError("Common-checkpoint optimizer metadata differs after restoration")


def restore_actor(actor, state):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    for module, value in zip(actor.model, state["model"], strict=True):
        module.load_state_dict(value)
    # Some optimizers retain references to loaded state. Give each draw its own copy.
    actor.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    for optimizer, parameter_state in zip(
        _leaf_optimizers(actor.optimizer), state["optimizer_parameter_state"], strict=True
    ):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer.load_parameter_state_from_dp_zero(copy.deepcopy(parameter_state))
            # Verify every master parameter and moment before each trial.
            assert_restored_state(optimizer.get_parameter_state_dp_zero(), parameter_state)
    actor.opt_param_scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    python_rng, numpy_rng, torch_rng, cuda_rng = state["rng"]
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    torch.cuda.set_rng_state(cuda_rng)
    get_cuda_rng_tracker().set_states(copy.deepcopy(state["megatron_rng"]))


@torch.no_grad()
def record_trial_gradient(optimizer, views):
    """Welford over D*A before clipping; retain only one CPU mean per branch."""
    from .optimizer import _preconditioner_denominator

    branch = optimizer._mopd_trial_branch
    state = optimizer._mopd_trial_variance.setdefault(branch, {"count": 0, "mean": {}, "m2": 0.0})
    state["count"] += 1
    count = state["count"]
    device = views[0].optimizer_parameter.device
    increment = torch.zeros((), dtype=torch.float32, device=device)
    for view in views:
        if not view.contributes_to_norm:
            continue
        gradient = view.optimizer_gradient().detach().reshape(-1)
        if view.name not in state["mean"]:
            state["mean"][view.name] = torch.zeros(gradient.numel(), dtype=torch.float32)
        mean = state["mean"][view.name]
        for start in range(0, gradient.numel(), 1_048_576):
            stop = min(start + 1_048_576, gradient.numel())
            scaled = gradient[start:stop].float() / _preconditioner_denominator(view, start, stop, device=device)
            delta = scaled - mean[start:stop].to(device)
            mean[start:stop].add_(delta.cpu(), alpha=1.0 / count)
            increment += delta.square().sum() * ((count - 1) / count)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(increment)
    state["m2"] += float(increment)


def diagnostic_seed(seed, purpose, trial, task):
    payload = f"{seed}:common-checkpoint-250:{purpose}:{trial}:{task}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


async def generate_samples(args, purpose, counts, trial):
    from slime.utils.types import Sample

    from .rollout import _generate_samples
    from .teacher_slot import score_teacher_samples

    config = yaml.safe_load(Path(args.mopd_diagnostic_data).read_text())
    if config.get("protocol") != "qwen3_1.7b_base_4t_common_checkpoint":
        raise ValueError("the common-checkpoint experiment requires the separate diagnostic manifest")
    validate_prompt_runtime(args, config)
    sources = config["sources"]
    if tuple(source["name"] for source in sources) != TASKS:
        raise ValueError("diagnostic sources must follow protocol task order")
    if purpose not in {"evaluation", "calibration", "uniform", "gpas"}:
        raise ValueError("unknown diagnostic purpose")
    if purpose in {"evaluation", "calibration"} and counts != [16] * 4:
        raise ValueError("evaluation and calibration each need 64 responses per domain")
    if purpose in {"uniform", "gpas"} and (
        len(counts) != 4 or sum(counts) != 16 or any(m < 2 or m > 8 for m in counts)
    ):
        raise ValueError("trial counts must satisfy the four-domain update budget")
    samples = []
    microbatch = 0
    for task_index, (source, count) in enumerate(zip(sources, counts, strict=True)):
        task = source["name"]
        rows = [json.loads(line) for line in Path(source["path"]).read_text().splitlines() if line.strip()]
        rng = random.Random(diagnostic_seed(args.mopd_seed, purpose, trial, task))
        selected = rng.sample(rows, 64) if purpose == "evaluation" else rng.choices(rows, k=count * 4)
        for ordinal, row in enumerate(selected):
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    "task_name": task,
                    "teacher": task,
                    "source_name": task,
                    "protocol_split": f"diagnostic/{purpose}/{trial}",
                    "mopd_enabled": True,
                    "mopd_task": task,
                    "mopd_task_index": task_index,
                    "mopd_operation": "calibration" if purpose == "calibration" else "diagnostic_trial",
                    "mopd_operation_index": trial,
                    "mopd_target_weight": 0.25,
                    "mopd_aggregation": "fixed_objective",
                    "mopd_microbatch_index": microbatch + ordinal // 4,
                    "mopd_task_microbatch_count": count,
                    "mopd_failure_penalty": 0.0,
                    "mopd_step_global_batch_size": 4,
                    "mopd_task_prompt_epoch": 0,
                    "mopd_task_prompt_ordinal": ordinal,
                    "mopd_response_ordinal": 0,
                }
            )
            samples.append(
                Sample(
                    prompt=row["prompt"],
                    label=row.get("label"),
                    metadata=metadata,
                    index=len(samples),
                    group_index=len(samples),
                    rollout_id=len(samples),
                )
            )
        microbatch += count
    samples, _, _ = await _generate_samples(args, samples)
    for task in TASKS:
        await score_teacher_samples(
            args,
            task,
            [s for s in samples if s.metadata["mopd_task"] == task],
            failure_penalty=args.mopd_failure_penalty,
        )
    from slime.rollout.sglang_rollout import GenerateState

    GenerateState(args).reset()
    return samples


def run_common_checkpoint(args, actor_model, rollout_manager, *, startup_seconds=0.0):
    import ray

    root = Path(args.save).parent
    started = time.perf_counter()
    bank_path = ray.get(rollout_manager.generate_mopd_diagnostic.remote("evaluation", [16] * 4, 0))
    before = loss_record(bank_path, actor_model.score_mopd_bank(bank_path))
    write_json(root / "before.json", before)
    calibration_data = ray.get(rollout_manager.generate_mopd_diagnostic.remote("calibration", [16] * 4, 0))
    calibration = actor_model.mopd_diagnostic("calibration", calibration_data)
    noises = [unit["scaled_noise"] for unit in calibration["task_units"]]
    counts = {"uniform": [4] * 4, "gpas": integer_allocation(noises, weights=[0.25] * 4)}
    write_json(root / "calibration.json", {"counts": counts, "feedback": calibration, "checkpoint_step": 250})
    trials = []
    for branch in ("uniform", "gpas"):
        for trial in range(10):
            data = ray.get(rollout_manager.generate_mopd_diagnostic.remote(branch, counts[branch], trial))
            feedback = actor_model.mopd_diagnostic(branch, data)
            if not feedback.get("checkpoint_restore_verified"):
                raise RuntimeError("Common-checkpoint trial is missing optimizer restoration verification")
            after = loss_record(bank_path, actor_model.score_mopd_bank(bank_path))
            changes = {task: before["task_losses"][task] - after["task_losses"][task] for task in TASKS}
            record = {
                "branch": branch,
                "trial": trial,
                "counts": counts[branch],
                "after": after,
                "loss_decrease": changes,
                "mean_loss_decrease": sum(changes.values()) / 4,
                "feedback": feedback,
            }
            write_json(root / "trials" / f"{branch}_{trial:02d}.json", record)
            trials.append(record)
    variance = actor_model.mopd_diagnostic("summary", None)
    uniform_variance = variance["uniform"]["variance"]
    write_json(
        root / "common_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_step": 250,
            "independent_checkpoint_restores_verified": len(trials),
            "counts": counts,
            "variance": variance,
            "variance_ratio": variance["gpas"]["variance"] / uniform_variance if uniform_variance else None,
            "before": before,
            "trials": trials,
            "generated_responses": 1792,
            "student_scoring_forwards_after_updates": 5120,
            "wall_seconds": time.perf_counter() - started + startup_seconds,
            "occupied_gpus": args.mopd_occupied_gpus,
        },
    )


def save_evaluation_bank(args, samples):
    from .loss import post_process_rewards

    post_process_rewards(args, samples)
    path = Path(args.save).parent / "evaluation_bank.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "protocol_sha256": bank_identity(args),
            "samples": [
                {
                    "task": s.metadata["mopd_task"],
                    "tokens": s.tokens,
                    "response_length": s.response_length,
                    "loss_mask": s.loss_mask,
                    "metadata": s.train_metadata,
                }
                for s in samples
            ],
        },
        path,
    )
    return str(path)
