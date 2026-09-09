"""Shared, immutable response/teacher-score banks for the two-week experiment."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import torch

from .prompting import validate_prompt_runtime
from .sampler import TASKS


def bank_identity(args):
    identity = validate_prompt_runtime(args)
    if identity is not None:
        return identity
    protocol = Path(args.experiment_data_index)
    return hashlib.sha256(protocol.read_bytes()).hexdigest()


def prepare_bank(args, rollout_id, data_source, path, *, initial=False):
    from slime.rollout.sglang_rollout import generate_rollout
    from slime.utils.async_utils import run

    from .eval import _score_all_tasks
    from .loss import post_process_rewards

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = bank_identity(args)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            bank = torch.load(path, map_location="cpu", weights_only=False)
            if bank["protocol_sha256"] != identity:
                raise ValueError(f"reference bank belongs to a different protocol: {path}")
            return str(path)
        if not initial and Path(args.mopd_reference_bank).resolve() == path.resolve():
            raise FileNotFoundError("The shared reference bank must be created by the initial student")
        output = generate_rollout(args, rollout_id, data_source, evaluation=True)
        run(_score_all_tasks(args, output))
        samples = []
        for task in TASKS:
            task_samples = output.data[task]["samples"]
            if len(task_samples) != 64:
                raise ValueError("the fixed loss bank requires 64 responses per domain")
            post_process_rewards(args, task_samples)
            for index, sample in enumerate(task_samples):
                samples.append(
                    {
                        "task": task,
                        "prompt_index": index,
                        "tokens": sample.tokens,
                        "response_length": sample.response_length,
                        "loss_mask": sample.loss_mask or [1] * sample.response_length,
                        "metadata": sample.train_metadata,
                    }
                )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save({"schema_version": 1, "protocol_sha256": identity, "samples": samples}, temporary)
        os.replace(temporary, path)
    return str(path)


def loss_record(bank_path, values):
    bank_path = Path(bank_path)
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    if len(values) != len(bank["samples"]):
        raise ValueError("bank scoring did not return one loss for every response")
    tasks = {task: [] for task in TASKS}
    for sample, value in zip(bank["samples"], values, strict=True):
        tasks[sample["task"]].append(float(value))
    means = {task: sum(losses) / len(losses) for task, losses in tasks.items()}
    return {
        "bank": str(bank_path.resolve()),
        "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "loss": "teacher_top64_corrected_reverse_kl",
        "task_losses": means,
        "prompt_losses": tasks,
        "weighted_loss": sum(means.values()) / len(TASKS),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
