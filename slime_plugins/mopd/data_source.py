"""Matched, non-repeating prompt streams for four-task micro-batch MOPD."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from slime_plugins.m2rl.data_source import MultiTaskRolloutDataSource, load_manifest

from .prompting import validate_prompt_runtime
from .sampler import (
    MICROBATCHES_PER_STEP,
    PROMPTS_PER_MICROBATCH,
    RESPONSES_PER_PROMPT,
    TASKS,
    VARIANCE_CHECKPOINT_STEPS,
    VARIANCE_MICROBATCHES,
    VARIANCE_MICROBATCHES_PER_TASK,
    VARIANCE_RESPONSE_BUDGET,
    MOPDController,
    active_tasks,
    active_domain_weights,
)

PROTOCOL = "qwen3_1.7b_4t_microbatch_mopd_gpas"
VARIANCE_PROTOCOL = "qwen3_1.7b_4t_heldout_gradient_variance"


def _validate_protocol_manifest(manifest: dict[str, Any], source_configs: list[dict[str, Any]]) -> None:
    if manifest.get("version") not in {3, 4} or manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"MOPD requires the {PROTOCOL!r} manifest")
    if tuple(config.get("name") for config in source_configs) != TASKS:
        raise ValueError(f"task order must be {TASKS}")
    weights = []
    for task, config in zip(TASKS, source_configs, strict=True):
        if config.get("teacher") != task:
            raise ValueError(f"source {task} must use its same-named teacher route")
        if int(config.get("required_samples", 0)) != 16_000:
            raise ValueError(f"source {task} must freeze exactly 16,000 training prompts")
        weights.append(float(config.get("target_weight", 0.0)))
    if any(weight <= 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("the four fixed target weights must be positive and sum to one")
    if manifest.get("version") == 4 and any(weight != 0.25 for weight in weights):
        raise ValueError("the two-week protocol requires equal task weights of 1/4")
    if bool((manifest.get("sampling") or {}).get("repeat", True)):
        raise ValueError("MOPD training prompt streams must not repeat")


def _validate_variance_manifest(manifest: dict[str, Any], source_configs: list[dict[str, Any]]) -> None:
    if manifest.get("version") != 3 or manifest.get("protocol") != VARIANCE_PROTOCOL:
        raise ValueError(f"held-out variance requires the {VARIANCE_PROTOCOL!r} version-3 manifest")
    if tuple(config.get("name") for config in source_configs) != TASKS:
        raise ValueError(f"task order must be {TASKS}")
    for task, config in zip(TASKS, source_configs, strict=True):
        if config.get("teacher") != task or int(config.get("required_samples", 0)) != 128:
            raise ValueError(f"held-out variance source {task} must contain 128 prompts and its matching teacher")
        if float(config.get("target_weight", 0.0)) <= 0:
            raise ValueError(f"held-out variance source {task} is missing its fixed target weight")
    if abs(sum(float(config["target_weight"]) for config in source_configs) - 1.0) > 1e-12:
        raise ValueError("held-out variance target weights must sum to one")
    if bool((manifest.get("sampling") or {}).get("repeat", True)):
        raise ValueError("held-out variance prompts must not repeat")


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _last_jsonl_operation(path: Path) -> int | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return None if not rows else int(rows[-1]["operation_index"])


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class MOPDRolloutDataSource(MultiTaskRolloutDataSource):
    """Issue one 64-prompt step in task-contiguous four-prompt micro-batches."""

    STATE_SCHEMA_VERSION = 4

    def __init__(self, args: Any):
        manifest = load_manifest(args.prompt_data)[0] if getattr(args, "prompt_data", None) else None
        self.protocol_sha256 = validate_prompt_runtime(args, manifest)
        super().__init__(args)
        paper_profile = bool(getattr(args, "mopd_profile", None))
        self.tasks = active_tasks(args) if paper_profile else TASKS
        if paper_profile:
            by_name = {source.config["name"]: source for source in self.sources}
            self.sources = [by_name[task] for task in self.tasks]
        source_configs = [source.config for source in self.sources]
        if paper_profile:
            if self.manifest.get("version") != 7 or self.manifest.get("protocol") != "mopd_paper":
                raise ValueError("Paper profiles require the version-7 mopd_paper manifest")
            for task, config in zip(self.tasks, source_configs, strict=True):
                if config.get("teacher") != task:
                    raise ValueError(f"Source {task} must use its matching teacher")
        else:
            _validate_protocol_manifest(self.manifest, source_configs)
        if not paper_profile and getattr(args, "mopd_loss", None) == "teacher_topk" and self.manifest.get("version") != 4:
            raise ValueError("Regenerate the two-week manifest; dense MOPD cannot consume the old weighted protocol")
        for source in self.sources:
            required = int(source.config.get("required_samples", len(source.dataset)))
            if len(source.dataset) < required:
                raise ValueError(
                    f"source {source.config['name']} has {len(source.dataset)} usable prompts; {required} are required"
                )
            selected = list(source.dataset.samples[:required])
            source.dataset.origin_samples = selected
            source.dataset.samples = selected
        checkpoints = tuple(int(value) for value in str(args.mopd_checkpoint_steps).split(",") if value)
        microbatches = int(args.mopd_microbatches_per_step)
        if paper_profile and microbatches % len(self.tasks):
            raise ValueError("Equal paper prompt quotas require microbatches divisible by active task count")
        count = microbatches // len(self.tasks)
        self.controller = MOPDController(
            self.tasks,
            target_weights=active_domain_weights(args) if paper_profile else [float(config["target_weight"]) for config in source_configs],
            allocation="uniform" if paper_profile else args.mopd_allocation,
            seed=args.mopd_seed,
            rollout_offset=0,
            ema_decay=args.mopd_ema_decay,
            total_steps=args.mopd_total_steps,
            checkpoint_steps=checkpoints,
            microbatches_per_step=args.mopd_microbatches_per_step,
            prompts_per_microbatch=args.mopd_prompts_per_microbatch,
            min_microbatches=count if paper_profile else args.mopd_min_microbatches,
            max_microbatches=count if paper_profile else args.mopd_max_microbatches,
            reduction=args.mopd_reduction if paper_profile else None,
        )
        output = args.mopd_output_dir or str(Path(args.save).resolve().parent / "allocation")
        self.output_dir = Path(output).resolve()
        self.allocation_log = self.output_dir / "allocation.jsonl"
        self.train_parallel_config: dict[str, Any] | None = None

    def set_train_parallel_config(self, config: dict[str, Any]) -> None:
        self.train_parallel_config = dict(config)

    def budget_status(self) -> dict[str, Any]:
        return {
            "complete": self.controller.budget_complete,
            "attempted_responses": self.controller.attempted_responses,
            "optimizer_updates": self.controller.optimizer_updates,
            "completed_operations": self.controller.completed_operations,
        }

    def prompt_count_for_rollout(self, rollout_id: int) -> int:
        plan = self.controller.plan(rollout_id)
        dp_size = int((self.train_parallel_config or {}).get("dp_size", 1))
        if int(plan["step_global_batch_size"]) < dp_size:
            raise ValueError("a four-response gradient micro-batch must contain at least one sample per DP rank")
        return int(plan["prompt_count"])

    def get_samples(self, num_samples: int):
        pending = self.controller.pending
        if pending is None:
            raise RuntimeError("samples requested before a MOPD allocation was locked")
        if int(num_samples) != int(pending["prompt_count"]):
            raise ValueError(f"MOPD requested {num_samples} prompts, expected {pending['prompt_count']}")

        groups = []
        microbatch_index = 0
        planned_weights = {unit["task"]: float(unit["target_weight"]) for unit in pending["task_units"]}
        for task_index, task in enumerate(self.controller.task_names):
            count = int(pending["counts"][task])
            weight = planned_weights[task]
            for task_microbatch_index in range(count):
                for prompt_in_microbatch in range(self.controller.config.prompts_per_microbatch):
                    source = self.sources[task_index]
                    prompt = self._next_prompt(task_index)
                    metadata = dict(prompt.metadata or {})
                    metadata.update(
                        {
                            "source_name": task,
                            "task_name": task,
                            "teacher": task,
                            "protocol_split": "train",
                            "mopd_enabled": True,
                            "mopd_rollout_id": pending["rollout_id"],
                            "mopd_operation": "train",
                            "mopd_operation_index": pending["operation_index"],
                            "mopd_task": task,
                            "mopd_task_index": task_index,
                            "mopd_allocation": pending["allocation"],
                            "mopd_aggregation": pending["aggregation"],
                            "mopd_microbatch_index": microbatch_index,
                            "mopd_task_microbatch_index": task_microbatch_index,
                            "mopd_task_microbatch_count": count,
                            "mopd_prompt_in_microbatch": prompt_in_microbatch,
                            "mopd_target_weight": weight,
                            "mopd_step_global_batch_size": self.controller.config.prompts_per_microbatch,
                            "mopd_task_prompt_epoch": int(source.epoch),
                            "mopd_task_prompt_ordinal": int(source.offset - 1),
                            "mopd_response_ordinal": 0,
                            "mopd_failure_penalty": 0.0,
                        }
                    )
                    sample = copy.deepcopy(prompt)
                    sample.metadata = metadata
                    sample.group_index = self.sample_group_index
                    sample.index = self.sample_index
                    sample.rollout_id = self.sample_index
                    self.sample_group_index += 1
                    self.sample_index += 1
                    groups.append([sample])
                microbatch_index += 1
        pending["issued"] = len(groups)
        if microbatch_index != self.controller.config.microbatches_per_step or len(groups) != int(num_samples):
            raise RuntimeError("MOPD did not construct the configured task micro-batches")
        return groups

    def add_samples(self, samples) -> None:
        del samples
        raise RuntimeError("MOPD prompt streams are fixed and do not recycle samples")

    def complete_update(self, rollout_id: int, feedback: dict[str, Any]) -> dict[str, Any]:
        pending = self.controller.pending
        if pending is None or int(pending["issued"]) != int(pending["prompt_count"]):
            raise RuntimeError("MOPD step did not issue its complete prompt set")
        record = self.controller.complete(rollout_id, feedback)
        expected = int(record["operation_index"]) - 1
        if _last_jsonl_operation(self.allocation_log) != (expected if expected >= 0 else None):
            raise ValueError("allocation log and sampler state have different frontiers")
        _append_jsonl(self.allocation_log, record)
        return record

    def _state_path(self, root: str, rollout_id: Any) -> Path:
        return Path(root) / "rollout" / f"mopd_dataset_state_dict_{rollout_id}.pt"

    def save(self, rollout_id: Any) -> None:
        if not self.args.save:
            return
        if self.controller.pending is not None:
            raise RuntimeError("cannot checkpoint while a MOPD step awaits feedback")
        _atomic_torch_save(
            {
                "schema_version": self.STATE_SCHEMA_VERSION,
                "protocol_sha256": self.protocol_sha256,
                "source_names": [source.config["name"] for source in self.sources],
                "source_lengths": [len(source.dataset) for source in self.sources],
                "source_offsets": [source.offset for source in self.sources],
                "source_epochs": [source.epoch for source in self.sources],
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "controller": self.controller.state_dict(),
            },
            self._state_path(self.args.save, rollout_id),
        )

    def load(self, rollout_id: Any = None) -> None:
        common_checkpoint = getattr(self.args, "mopd_common_checkpoint", False)
        if common_checkpoint and self.protocol_sha256 is None:
            return
        if not self.args.load:
            return
        path = self._state_path(self.args.load, rollout_id)
        if not path.exists():
            if int(self.args.start_rollout_id) == 0:
                return
            raise FileNotFoundError(f"MOPD requires sampler checkpoint {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if int(state.get("schema_version", -1)) != self.STATE_SCHEMA_VERSION:
            raise ValueError("unsupported MOPD data-source checkpoint schema")
        if self.protocol_sha256 is not None and state.get("protocol_sha256") != self.protocol_sha256:
            raise ValueError("Saved MOPD prompt protocol differs; start a new run for asymmetric prefixes")
        if common_checkpoint:
            return
        if state["source_names"] != [source.config["name"] for source in self.sources]:
            raise ValueError("saved MOPD source names differ from the manifest")
        if state["source_lengths"] != [len(source.dataset) for source in self.sources]:
            raise ValueError("saved MOPD source lengths differ from the manifest")
        for source, offset, epoch in zip(self.sources, state["source_offsets"], state["source_epochs"], strict=True):
            source.offset, source.epoch = int(offset), int(epoch)
            if self.args.rollout_shuffle:
                source.dataset.shuffle(source.epoch)
        self.sample_group_index = int(state["sample_group_index"])
        self.sample_index = int(state["sample_index"])
        self.controller.load_state_dict(state["controller"])
        expected = self.controller.completed_steps - 1
        if _last_jsonl_operation(self.allocation_log) != (expected if expected >= 0 else None):
            raise ValueError("allocation log does not match the resumed sampler state")


class MOPDVarianceDataSource(MultiTaskRolloutDataSource):
    """Issue the frozen 32-micro-batch-per-task held-out gradient probe."""

    def __init__(self, args: Any):
        super().__init__(args)
        source_configs = [source.config for source in self.sources]
        _validate_variance_manifest(self.manifest, source_configs)
        if any(len(source.dataset) != 128 for source in self.sources):
            raise ValueError("held-out variance requires exactly 128 rendered prompts per task")
        checkpoint = torch.load(args.mopd_variance_controller_state, map_location="cpu", weights_only=False)
        controller = checkpoint["controller"]
        step = int(args.mopd_variance_checkpoint_step)
        if int(controller["completed_steps"]) != step or controller.get("pending") is not None:
            raise ValueError("held-out variance controller state does not match its Uniform checkpoint")
        if tuple(controller["config"]["task_names"]) != TASKS:
            raise ValueError(f"held-out variance controller task order must be {TASKS}")
        self.training_controller_state = {
            "checkpoint_step": step,
            "task_seconds": dict(zip(TASKS, map(float, controller["task_seconds"]), strict=True)),
            "fixed_seconds": float(controller["fixed_seconds"]),
        }
        self.controller = SimpleNamespace(pending=None)
        self.complete = False
        self.output_dir = Path(args.mopd_output_dir).resolve()
        self.train_parallel_config: dict[str, Any] | None = None

    def set_train_parallel_config(self, config: dict[str, Any]) -> None:
        self.train_parallel_config = dict(config)

    def budget_status(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "attempted_responses": VARIANCE_RESPONSE_BUDGET if self.complete else 0,
            "optimizer_updates": int(self.args.mopd_variance_checkpoint_step),
            "completed_operations": int(self.complete),
        }

    def prompt_count_for_rollout(self, rollout_id: int) -> int:
        if int(rollout_id) != 0 or self.complete:
            raise StopIteration("the held-out gradient probe contains exactly one operation")
        if self.controller.pending is None:
            weights = [float(source.config["target_weight"]) for source in self.sources]
            counts = {task: VARIANCE_MICROBATCHES_PER_TASK for task in TASKS}
            self.controller.pending = {
                "schema_version": 1,
                "rollout_id": 0,
                "operation_index": 0,
                "optimizer_step": int(self.args.mopd_variance_checkpoint_step),
                "operation": "heldout_variance",
                "run_mode": "heldout_variance",
                "allocation": "uniform",
                "aggregation": "variance_probe",
                "counts": counts,
                "execution_order": list(TASKS),
                "prompt_count": VARIANCE_RESPONSE_BUDGET,
                "responses_per_prompt": RESPONSES_PER_PROMPT,
                "attempted_responses": VARIANCE_RESPONSE_BUDGET,
                "attempted_responses_before": 0,
                "optimizer_updates_before": int(self.args.mopd_variance_checkpoint_step),
                "step_global_batch_size": PROMPTS_PER_MICROBATCH,
                "target_weights": dict(zip(TASKS, weights, strict=True)),
                "issued": 0,
            }
        return VARIANCE_RESPONSE_BUDGET

    def get_samples(self, num_samples: int):
        pending = self.controller.pending
        if pending is None or int(num_samples) != VARIANCE_RESPONSE_BUDGET:
            raise ValueError(f"held-out variance requires exactly {VARIANCE_RESPONSE_BUDGET} prompts")
        groups = []
        microbatch_index = 0
        for task_index, task in enumerate(TASKS):
            source = self.sources[task_index]
            weight = float(source.config["target_weight"])
            for task_microbatch_index in range(VARIANCE_MICROBATCHES_PER_TASK):
                for prompt_in_microbatch in range(PROMPTS_PER_MICROBATCH):
                    prompt = self._next_prompt(task_index)
                    metadata = dict(prompt.metadata or {})
                    metadata.update(
                        {
                            "source_name": task,
                            "task_name": task,
                            "teacher": task,
                            "protocol_split": "heldout_variance",
                            "mopd_enabled": True,
                            "mopd_rollout_id": 0,
                            "mopd_operation": "heldout_variance",
                            "mopd_operation_index": 0,
                            "mopd_task": task,
                            "mopd_task_index": task_index,
                            "mopd_allocation": "uniform",
                            "mopd_aggregation": "variance_probe",
                            "mopd_microbatch_index": microbatch_index,
                            "mopd_task_microbatch_index": task_microbatch_index,
                            "mopd_task_microbatch_count": VARIANCE_MICROBATCHES_PER_TASK,
                            "mopd_prompt_in_microbatch": prompt_in_microbatch,
                            "mopd_target_weight": weight,
                            "mopd_step_global_batch_size": PROMPTS_PER_MICROBATCH,
                            "mopd_task_prompt_epoch": 0,
                            "mopd_task_prompt_ordinal": int(source.offset - 1),
                            "mopd_response_ordinal": 0,
                            "mopd_failure_penalty": 0.0,
                        }
                    )
                    sample = copy.deepcopy(prompt)
                    sample.metadata = metadata
                    sample.group_index = self.sample_group_index
                    sample.index = self.sample_index
                    sample.rollout_id = self.sample_index
                    self.sample_group_index += 1
                    self.sample_index += 1
                    groups.append([sample])
                microbatch_index += 1
        pending["issued"] = len(groups)
        if microbatch_index != VARIANCE_MICROBATCHES:
            raise RuntimeError("held-out variance did not construct 128 four-prompt micro-batches")
        return groups

    def add_samples(self, samples) -> None:
        del samples
        raise RuntimeError("held-out variance prompts do not recycle")

    def complete_update(self, rollout_id: int, feedback: dict[str, Any]) -> dict[str, Any]:
        pending = self.controller.pending
        if pending is None or int(rollout_id) != 0 or int(pending["issued"]) != VARIANCE_RESPONSE_BUDGET:
            raise RuntimeError("held-out variance operation is not ready to complete")
        if feedback.get("operation") != "heldout_variance" or feedback.get("optimizer_step_executed"):
            raise ValueError("held-out variance must return scalar statistics without an optimizer update")
        units = list(feedback.get("task_units") or [])
        if [unit["task"] for unit in units] != list(TASKS) or any(
            int(unit["microbatches"]) != VARIANCE_MICROBATCHES_PER_TASK for unit in units
        ):
            raise ValueError("held-out variance feedback must contain 32 micro-batches for every task")
        record = {
            "schema_version": 1,
            "checkpoint_step": int(self.args.mopd_variance_checkpoint_step),
            "operation": "heldout_variance",
            "task_order": list(TASKS),
            "microbatches_per_task": VARIANCE_MICROBATCHES_PER_TASK,
            "prompts_per_microbatch": PROMPTS_PER_MICROBATCH,
            "attempted_responses": VARIANCE_RESPONSE_BUDGET,
            "target_weights": pending["target_weights"],
            "training_controller_state": self.training_controller_state,
            "feedback": copy.deepcopy(feedback),
            "operation_index": 0,
            "attempted_responses_before": 0,
            "attempted_responses_after": VARIANCE_RESPONSE_BUDGET,
            "optimizer_updates_after": int(self.args.mopd_variance_checkpoint_step),
            "checkpoint_due": False,
            "budget_complete": True,
        }
        _atomic_json(record, self.output_dir / "heldout_gradient_scalars.json")
        self.complete = True
        self.controller.pending = None
        return record

    def save(self, rollout_id: Any) -> None:
        del rollout_id

    def load(self, rollout_id: Any = None) -> None:
        del rollout_id


__all__ = [
    "MOPDRolloutDataSource",
    "MOPDVarianceDataSource",
    "PROTOCOL",
    "VARIANCE_PROTOCOL",
    "_validate_protocol_manifest",
    "_validate_variance_manifest",
]
