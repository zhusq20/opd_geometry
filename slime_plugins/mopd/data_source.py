"""Manifest-backed prompt streams for exact-set MOPD operations."""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample
from slime_plugins.m2rl.data_source import MultiTaskRolloutDataSource

from .sampler import MOPDController, RESPONSES_PER_PROMPT, TASKS

PROTOCOL = "qwen3_1.7b_4t_exact_set_mopd_gpas"


def _validate_protocol_manifest(manifest: dict[str, Any], source_configs: list[dict[str, Any]]) -> None:
    if manifest.get("version") != 2 or manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"MOPD requires the {PROTOCOL!r} version-2 manifest")
    if tuple(config.get("name") for config in source_configs) != TASKS:
        raise ValueError(f"task order must be {TASKS}")
    for task, config in zip(TASKS, source_configs, strict=True):
        if config.get("teacher") != task or float(config.get("target_weight", 0)) != 0.25:
            raise ValueError(f"source {task} must use its same-named teacher and target weight 0.25")
        initial = float(config.get("initial_teacher_loss", 0))
        scale = float(config.get("relative_loss_scale", 0))
        if initial <= 0 or not abs(scale * initial - 1.0) < 1e-10:
            raise ValueError(f"source {task} has an invalid frozen relative-loss denominator")
        required_samples = config.get("required_samples")
        if (
            isinstance(required_samples, bool)
            or not isinstance(required_samples, int)
            or required_samples <= 0
        ):
            raise ValueError(f"source {task} must freeze a positive usable-prompt count")
        for key in ("probe_path", "bank_path"):
            if not config.get(key):
                raise ValueError(f"source {task} is missing {key}")


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _last_jsonl_operation(path: Path) -> int | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    last = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid allocation JSON at {path}:{line_number}") from exc
    return None if last is None else int(last["operation_index"])


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _expanded_path(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    match = re.match(r"^(.*)(@\[-?\d*:-?\d*\])$", expanded)
    real, suffix = (match.group(1), match.group(2)) if match else (expanded, "")
    return str(Path(real).resolve()) + suffix


@dataclass
class _AuxStream:
    dataset: Dataset
    offset: int = 0
    epoch: int = 0


class MOPDRolloutDataSource(MultiTaskRolloutDataSource):
    """Issue full units from matched task streams and probes from a disjoint stream."""

    STATE_SCHEMA_VERSION = 3

    def __init__(self, args: Any):
        super().__init__(args)
        source_configs = [source.config for source in self.sources]
        _validate_protocol_manifest(self.manifest, source_configs)
        for source in self.sources:
            expected = int(source.config["required_samples"])
            if len(source.dataset) != expected:
                raise ValueError(
                    f"source {source.config['name']} froze {expected} usable prompts, "
                    f"but the current tokenizer and prompt cap produced {len(source.dataset)}"
                )
        checkpoints = tuple(int(v) for v in str(args.mopd_checkpoint_responses).split(",") if v)
        self.controller = MOPDController(
            TASKS,
            target_weights=[0.25] * 4,
            allocation=args.mopd_allocation,
            task_width=args.mopd_task_width,
            adamw_state=args.mopd_adamw_state,
            run_mode=args.mopd_run_mode,
            seed=args.mopd_seed,
            rollout_offset=args.start_rollout_id,
            ema_decay=args.mopd_ema_decay,
            inclusion_floor=args.mopd_inclusion_floor,
            score_max_age=args.mopd_score_max_age,
            response_budget=args.mopd_response_budget,
            checkpoint_responses=checkpoints,
            bank_units_per_task=args.mopd_bank_units_per_task,
        )
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        self.aux_streams: dict[str, list[_AuxStream]] = {"probe": [], "bank": []}
        for config in source_configs:
            for stream_name, path_key in (("probe", "probe_path"), ("bank", "bank_path")):
                dataset = Dataset(
                    _expanded_path(config[path_key]),
                    tokenizer=tokenizer,
                    processor=processor,
                    max_length=args.rollout_max_prompt_len,
                    prompt_key=config.get("input_key", args.input_key),
                    label_key=config.get("label_key", args.label_key),
                    metadata_key=config.get("metadata_key", args.metadata_key),
                    tool_key=config.get("tool_key", args.tool_key),
                    apply_chat_template=config.get("apply_chat_template", args.apply_chat_template),
                    apply_chat_template_kwargs=config.get(
                        "apply_chat_template_kwargs", args.apply_chat_template_kwargs
                    ),
                    seed=int(config["shuffle_seed"]) + (1_000_003 if stream_name == "probe" else 2_000_003),
                )
                if args.rollout_shuffle:
                    dataset.shuffle(0)
                self.aux_streams[stream_name].append(_AuxStream(dataset))
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
        response_count = int(plan["prompt_count"]) * RESPONSES_PER_PROMPT
        dp_size = int((self.train_parallel_config or {}).get("dp_size", 1))
        if int(plan["step_global_batch_size"]) < dp_size:
            raise ValueError("one MOPD backward slice must contain at least one response per DP rank")
        if response_count != int(plan["attempted_responses"]):
            raise RuntimeError("planned prompts and attempted responses disagree")
        return int(plan["prompt_count"])

    def _next_aux_prompt(self, stream_name: str, task_index: int) -> tuple[Sample, int, int]:
        stream = self.aux_streams[stream_name][task_index]
        if stream.offset >= len(stream.dataset):
            stream.epoch += 1
            stream.offset = 0
            if self.args.rollout_shuffle:
                stream.dataset.shuffle(stream.epoch)
        ordinal = stream.offset
        sample = copy.deepcopy(stream.dataset[ordinal])
        stream.offset += 1
        config = self.sources[task_index].config
        metadata = dict(config.get("metadata") or {})
        metadata.update(sample.metadata or {})
        metadata.update({"source_name": config["name"], "task_name": config["name"], "teacher": config["name"]})
        if config.get("rm_type") is not None:
            metadata["rm_type"] = config["rm_type"]
        sample.metadata = metadata
        sample.source = config["name"]
        return sample, stream.epoch, ordinal

    def _next_planned_prompt(self, task_index: int, operation: str) -> tuple[Sample, int, int, str]:
        if operation in {"probe", "bank"}:
            sample, epoch, ordinal = self._next_aux_prompt(operation, task_index)
            return sample, epoch, ordinal, operation
        source = self.sources[task_index]
        sample = self._next_prompt(task_index)
        return sample, int(source.epoch), int(source.offset - 1), "train"

    def get_samples(self, num_samples: int):
        pending = self.controller.pending
        if pending is None:
            raise RuntimeError("samples requested before an exact-set plan was locked")
        if int(pending["issued"]) + int(num_samples) > int(pending["prompt_count"]):
            raise RuntimeError("rollout requested more prompts than the locked MOPD operation")
        task_plan = {unit["task"]: unit for unit in pending["task_units"]}
        prompts_per_task = 2 if pending["operation"] == "probe" else 16
        groups: list[list[Sample]] = []
        for task_name in pending["execution_order"]:
            task_index = TASKS.index(task_name)
            unit = task_plan[task_name]
            for group_ordinal in range(prompts_per_task):
                prompt, epoch, ordinal, split = self._next_planned_prompt(task_index, pending["operation"])
                metadata = dict(prompt.metadata or {})
                metadata.update(
                    {
                        "source_name": task_name,
                        "task_name": task_name,
                        "teacher": task_name,
                        "protocol_split": split,
                        "mopd_enabled": True,
                        "mopd_rollout_id": pending["rollout_id"],
                        "mopd_operation": pending["operation"],
                        "mopd_operation_index": pending["operation_index"],
                        "mopd_task": task_name,
                        "mopd_task_index": task_index,
                        "mopd_allocation": pending["allocation"],
                        "mopd_adamw_state": pending["adamw_state"],
                        "mopd_task_width": pending["task_width"],
                        "mopd_selected_set": pending["selected_set"],
                        "mopd_execution_order": pending["execution_order"],
                        "mopd_inclusion_probability": unit["inclusion_probability"],
                        "mopd_target_weight": unit["target_weight"],
                        "mopd_importance_correction": unit["importance_correction"],
                        "mopd_relative_loss_scale": float(self.sources[task_index].config["relative_loss_scale"]),
                        "mopd_initial_teacher_loss": float(self.sources[task_index].config["initial_teacher_loss"]),
                        "mopd_attempted_responses_before": pending["attempted_responses_before"],
                        "mopd_processed_task_units_before": pending["processed_task_units_before"],
                        "mopd_optimizer_updates_before": pending["optimizer_updates_before"],
                        "mopd_step_global_batch_size": pending["step_global_batch_size"],
                        "mopd_task_prompt_epoch": epoch,
                        "mopd_task_prompt_ordinal": ordinal,
                        "mopd_prompt_group_ordinal": group_ordinal,
                    }
                )
                prompt.metadata = metadata
                group = []
                for response_ordinal in range(RESPONSES_PER_PROMPT):
                    sample = copy.deepcopy(prompt)
                    sample.metadata["mopd_response_ordinal"] = response_ordinal
                    sample.metadata["mopd_failure_penalty"] = 0.0
                    sample.group_index = self.sample_group_index
                    sample.index = self.sample_index
                    sample.rollout_id = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                self.sample_group_index += 1
                groups.append(group)
        if len(groups) != int(num_samples):
            raise RuntimeError(f"planned {num_samples} prompt groups but constructed {len(groups)}")
        pending["issued"] = int(pending["issued"]) + len(groups)
        return groups

    def add_samples(self, samples) -> None:
        del samples
        raise RuntimeError("MOPD operations have fixed attempted-response counts and cannot recycle samples")

    def complete_update(self, rollout_id: int, feedback: dict[str, Any]) -> dict[str, Any]:
        pending = self.controller.pending
        if pending is None or int(pending["issued"]) != int(pending["prompt_count"]):
            raise RuntimeError("MOPD operation did not issue its complete prompt set")
        record = self.controller.complete(rollout_id, feedback)
        frontier = _last_jsonl_operation(self.allocation_log)
        expected = int(record["operation_index"]) - 1
        if frontier != (expected if expected >= 0 else None):
            raise ValueError("allocation log and sampler checkpoint have different frontiers")
        _append_jsonl(self.allocation_log, record)
        return record

    def _state_path(self, root: str, rollout_id: Any) -> Path:
        return Path(root) / "rollout" / f"mopd_dataset_state_dict_{rollout_id}.pt"

    def _source_state(self) -> dict[str, Any]:
        return {
            "source_names": [source.config["name"] for source in self.sources],
            "source_lengths": [len(source.dataset) for source in self.sources],
            "source_shuffle_seeds": [source.config["shuffle_seed"] for source in self.sources],
            "source_offsets": [source.offset for source in self.sources],
            "source_epochs": [source.epoch for source in self.sources],
            "aux_offsets": {name: [stream.offset for stream in streams] for name, streams in self.aux_streams.items()},
            "aux_epochs": {name: [stream.epoch for stream in streams] for name, streams in self.aux_streams.items()},
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
        }

    def save(self, rollout_id: Any) -> None:
        if not self.args.save:
            return
        if self.controller.pending is not None:
            raise RuntimeError("cannot checkpoint while a MOPD operation awaits feedback")
        _atomic_torch_save(
            {"schema_version": self.STATE_SCHEMA_VERSION, **self._source_state(), "controller": self.controller.state_dict()},
            self._state_path(self.args.save, rollout_id),
        )

    def _restore_sources(self, state: dict[str, Any], *, restore_aux: bool) -> None:
        if state["source_names"] != [source.config["name"] for source in self.sources]:
            raise ValueError("saved MOPD source names differ from the manifest")
        if state["source_lengths"] != [len(source.dataset) for source in self.sources]:
            raise ValueError("saved MOPD source lengths differ from the manifest")
        for source, offset, epoch in zip(self.sources, state["source_offsets"], state["source_epochs"], strict=True):
            source.offset, source.epoch = int(offset), int(epoch)
            if self.args.rollout_shuffle:
                source.dataset.shuffle(source.epoch)
        if restore_aux:
            for name, streams in self.aux_streams.items():
                for stream, offset, epoch in zip(streams, state["aux_offsets"][name], state["aux_epochs"][name], strict=True):
                    stream.offset, stream.epoch = int(offset), int(epoch)
                    if self.args.rollout_shuffle:
                        stream.dataset.shuffle(stream.epoch)
        self.sample_group_index = int(state["sample_group_index"])
        self.sample_index = int(state["sample_index"])

    def load(self, rollout_id: Any = None) -> None:
        if not self.args.load:
            return
        path = self._state_path(self.args.load, rollout_id)
        if not path.exists():
            if int(self.args.start_rollout_id) != 0:
                raise FileNotFoundError(f"MOPD requires sampler checkpoint {path}")
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        if int(state.get("schema_version", -1)) != self.STATE_SCHEMA_VERSION:
            raise ValueError("unsupported MOPD data-source checkpoint schema")
        reset = bool(self.args.mopd_reset_sampler)
        self._restore_sources(state, restore_aux=not reset)
        if reset:
            self.controller.bootstrap(state["controller"])
            self.sample_group_index = 0
            self.sample_index = 0
            if _last_jsonl_operation(self.allocation_log) is not None:
                raise ValueError("a new branch requires an empty allocation output directory")
        else:
            self.controller.load_state_dict(state["controller"])
            expected = self.controller.completed_operations - 1
            if _last_jsonl_operation(self.allocation_log) != (expected if expected >= 0 else None):
                raise ValueError("allocation log does not match the resumed sampler state")


__all__ = ["MOPDRolloutDataSource", "PROTOCOL", "_validate_protocol_manifest"]
