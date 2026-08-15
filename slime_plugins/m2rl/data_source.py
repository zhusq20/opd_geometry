"""Deterministic multi-task rollout data source.

``--prompt-data`` points to a JSON/YAML manifest instead of a single dataset.
Each task keeps an independent shuffle/epoch cursor, and the sampler state is
checkpointed with Slime's rollout state so resumed experiments see exactly the
same task order.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from slime.rollout.data_source import DataSource
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _expand_path(path: str, base: Path) -> str:
    value = os.path.expandvars(os.path.expanduser(path))
    match = re.match(r"^(.*)(@\[-?\d*:-?\d*\])$", value)
    real_path, suffix = (match.group(1), match.group(2)) if match else (value, "")
    resolved = Path(real_path)
    if not resolved.is_absolute():
        resolved = base / resolved
    return f"{resolved.resolve()}{suffix}"


def load_manifest(path: str) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Multi-task manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        if manifest_path.suffix.lower() in {".yaml", ".yml"}:
            manifest = yaml.safe_load(stream) or {}
        else:
            manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), list):
        raise ValueError("Multi-task manifest must be a mapping with a non-empty `sources` list.")
    if not manifest["sources"]:
        raise ValueError("Multi-task manifest `sources` cannot be empty.")
    return manifest, manifest_path


class TaskSampler:
    """Serializable task-index sampler used independently of dataset loading."""

    STRATEGIES = {
        "uniform",
        "proportional",
        "weighted",
        "stratified",
        "round_robin",
        "sequential",
    }

    def __init__(self, sources: list[dict[str, Any]], lengths: list[int], sampling: dict[str, Any]):
        self.sources = sources
        self.lengths = lengths
        self.strategy = str(sampling.get("strategy", "proportional"))
        self.unit = str(sampling.get("unit", "prompt"))
        self.repeat = bool(sampling.get("repeat", True))
        self.rng = random.Random(int(sampling.get("seed", 0)))
        self.cursor = 0
        self.sequential_source = 0
        self.sequential_offset = 0

        if self.strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown task sampling strategy {self.strategy!r}; choose from {sorted(self.STRATEGIES)}."
            )
        if self.unit not in {"prompt", "batch"}:
            raise ValueError("Task sampling `unit` must be `prompt` or `batch`.")
        if len(sources) != len(lengths) or not sources:
            raise ValueError("TaskSampler requires one non-empty length per source.")
        if any(length <= 0 for length in lengths):
            raise ValueError("Every multi-task source must contain at least one usable sample.")

        if self.strategy == "uniform":
            self.weights = [1.0] * len(sources)
        elif self.strategy == "proportional":
            self.weights = [float(length) for length in lengths]
        elif self.strategy in {"weighted", "stratified"}:
            self.weights = [float(source.get("weight", 0.0)) for source in sources]
            if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
                raise ValueError(
                    f"{self.strategy.capitalize()} task sampling requires non-negative source weights "
                    "with a positive sum."
                )
        else:
            self.weights = [1.0] * len(sources)

        self.phase_sizes = [
            int(source.get("phase_samples", length)) for source, length in zip(sources, lengths, strict=True)
        ]
        if any(size <= 0 for size in self.phase_sizes):
            raise ValueError("Every sequential source phase_samples value must be positive.")

    def _random_index(self) -> int:
        threshold = self.rng.random() * sum(self.weights)
        cumulative = 0.0
        for index, weight in enumerate(self.weights):
            cumulative += weight
            if threshold < cumulative:
                return index
        return len(self.weights) - 1

    def _sequential_index(self) -> int:
        if self.sequential_source >= len(self.sources):
            if not self.repeat:
                raise StopIteration("The non-repeating sequential multi-task curriculum is exhausted.")
            self.sequential_source = 0
        index = self.sequential_source
        self.sequential_offset += 1
        if self.sequential_offset >= self.phase_sizes[index]:
            self.sequential_source += 1
            self.sequential_offset = 0
        return index

    def _next_index(self) -> int:
        if self.strategy == "round_robin":
            result = self.cursor % len(self.sources)
            self.cursor += 1
            return result
        if self.strategy == "sequential":
            return self._sequential_index()
        return self._random_index()

    def _stratified_indices(self, count: int) -> list[int]:
        """Return a deterministic, near-exact weighted composition per call.

        Ordinary weighted sampling is appropriate for task streams, but it can
        produce an all-SFT or all-OPD optimizer step when a hybrid loss uses a
        small batch.  ``stratified`` uses largest-remainder allocation and then
        shuffles the resulting source IDs with the checkpointed RNG.  Every
        call is within one sample of the requested proportions.
        """

        active = [index for index, weight in enumerate(self.weights) if weight > 0]
        if count < len(active):
            raise ValueError(
                f"Stratified batch size {count} is smaller than the {len(active)} positive-weight sources."
            )
        # Reserve one slot per active component. This makes the mixed-loss
        # contract literal even for an imbalanced ratio and a small batch.
        allocations = [int(weight > 0) for weight in self.weights]
        unallocated = count - sum(allocations)
        total_weight = sum(self.weights)
        quotas = [unallocated * weight / total_weight for weight in self.weights]
        quota_floors = [int(quota) for quota in quotas]
        allocations = [base + floor for base, floor in zip(allocations, quota_floors, strict=True)]
        remaining = count - sum(allocations)
        # Random tie breakers prevent source-order bias while remaining exactly
        # reproducible across checkpoint resume.
        tie_breakers = [self.rng.random() for _ in self.weights]
        order = sorted(
            range(len(self.weights)),
            key=lambda index: (quotas[index] - quota_floors[index], tie_breakers[index]),
            reverse=True,
        )
        for index in order[:remaining]:
            allocations[index] += 1

        result = [index for index, allocation in enumerate(allocations) for _ in range(allocation)]
        self.rng.shuffle(result)
        return result

    def select(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("Task sample count cannot be negative.")
        if count == 0:
            return []
        if self.strategy == "stratified":
            if self.unit != "prompt":
                raise ValueError("Stratified sampling requires `unit: prompt` so one batch can mix sources.")
            return self._stratified_indices(count)
        if self.unit == "batch" and self.strategy == "sequential":
            if self.sequential_source >= len(self.sources):
                if not self.repeat:
                    raise StopIteration("The non-repeating sequential multi-task curriculum is exhausted.")
                self.sequential_source = 0
            index = self.sequential_source
            self.sequential_offset += count
            if self.sequential_offset >= self.phase_sizes[index]:
                self.sequential_source += 1
                self.sequential_offset = 0
            return [index] * count
        if self.unit == "batch":
            return [self._next_index()] * count
        return [self._next_index() for _ in range(count)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "unit": self.unit,
            "repeat": self.repeat,
            "weights": self.weights,
            "phase_sizes": self.phase_sizes,
            "rng_state": self.rng.getstate(),
            "cursor": self.cursor,
            "sequential_source": self.sequential_source,
            "sequential_offset": self.sequential_offset,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state.get("strategy") != self.strategy
            or state.get("unit") != self.unit
            or state.get("repeat", self.repeat) != self.repeat
            or state.get("weights", self.weights) != self.weights
            or state.get("phase_sizes", self.phase_sizes) != self.phase_sizes
        ):
            raise ValueError("Saved task-sampler configuration does not match the current manifest.")
        self.rng.setstate(state["rng_state"])
        self.cursor = int(state["cursor"])
        self.sequential_source = int(state["sequential_source"])
        self.sequential_offset = int(state["sequential_offset"])


@dataclass
class _Source:
    config: dict[str, Any]
    dataset: Dataset
    offset: int = 0
    epoch: int = 0


class MultiTaskRolloutDataSource(DataSource):
    """Manifest-backed source supporting mixtures and sequential curricula."""

    def __init__(self, args: Any):
        if not args.rollout_global_dataset:
            raise ValueError("MultiTaskRolloutDataSource requires the global rollout dataset to be enabled.")
        if not args.prompt_data:
            raise ValueError("--prompt-data must point to a multi-task JSON/YAML manifest.")
        self.args = args
        self.manifest, manifest_path = load_manifest(args.prompt_data)
        base = manifest_path.parent
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        names: set[str] = set()
        self.sources: list[_Source] = []

        for source_index, original_config in enumerate(self.manifest["sources"]):
            config = dict(original_config)
            name = str(config.get("name", "")).strip()
            if not name or name in names:
                raise ValueError(f"Every manifest source needs a unique non-empty name; got {name!r}.")
            names.add(name)
            if "path" not in config:
                raise ValueError(f"Manifest source {name!r} is missing `path`.")
            config["path"] = _expand_path(str(config["path"]), base)
            dataset = Dataset(
                config["path"],
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=config.get("input_key", args.input_key),
                multimodal_keys=config.get("multimodal_keys", args.multimodal_keys),
                label_key=config.get("label_key", args.label_key),
                metadata_key=config.get("metadata_key", args.metadata_key),
                tool_key=config.get("tool_key", args.tool_key),
                apply_chat_template=config.get("apply_chat_template", args.apply_chat_template),
                apply_chat_template_kwargs=config.get("apply_chat_template_kwargs", args.apply_chat_template_kwargs),
                seed=args.rollout_seed + source_index * 100_003,
            )
            if args.rollout_shuffle:
                dataset.shuffle(0)
            self.sources.append(_Source(config=config, dataset=dataset))

        sampling = dict(self.manifest.get("sampling") or {})
        sampling.setdefault("seed", args.rollout_seed)
        if getattr(args, "m2rl_task_sampling_seed", None) is not None:
            sampling["seed"] = int(args.m2rl_task_sampling_seed)
        self.sampler = TaskSampler(
            [source.config for source in self.sources],
            [len(source.dataset) for source in self.sources],
            sampling,
        )
        self.strict_single_epoch = bool(
            len(self.sources) == 1
            and getattr(args, "include_epoch_tail", False)
            and getattr(args, "num_epoch", None) == 1
            and getattr(args, "num_rollout", None) is None
        )
        self.sample_group_index = 0
        self.sample_index = 0

    def _next_prompt(self, source_index: int) -> Sample:
        source = self.sources[source_index]
        if source.offset >= len(source.dataset):
            if self.strict_single_epoch:
                raise RuntimeError(
                    "Exact single-dataset epoch exhausted; refusing to wrap and repeat prompts. "
                    "A rollout requested more prompts than the filtered dataset contains."
                )
            source.epoch += 1
            source.offset = 0
            if self.args.rollout_shuffle:
                source.dataset.shuffle(source.epoch)
        sample = copy.deepcopy(source.dataset[source.offset])
        source.offset += 1
        config = source.config
        metadata = dict(config.get("metadata") or {})
        metadata.update(sample.metadata or {})
        metadata["source_name"] = config["name"]
        metadata.setdefault("task_name", config["name"])
        if config.get("rm_type") is not None:
            metadata.setdefault("rm_type", config["rm_type"])
        if config.get("teacher") is not None:
            metadata.setdefault("teacher", config["teacher"])
        sample.metadata = metadata
        sample.source = config["name"]
        if config.get("custom_rm_path"):
            sample.custom_rm_path = config["custom_rm_path"]
        if config.get("generate_function_path"):
            sample.generate_function_path = config["generate_function_path"]
        return sample

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        source_indices = self.sampler.select(num_samples)
        groups: list[list[Sample]] = []
        for source_index in source_indices:
            prompt_sample = self._next_prompt(source_index)
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def add_samples(self, samples: list[list[Sample]]) -> None:
        del samples
        raise RuntimeError("MultiTaskRolloutDataSource is read-only.")

    def _state_path(self, root: str, rollout_id: Any) -> Path:
        return Path(root) / "rollout" / f"multitask_dataset_state_dict_{rollout_id}.pt"

    def save(self, rollout_id: Any) -> None:
        if not self.args.save:
            return
        path = self._state_path(self.args.save, rollout_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source_names": [source.config["name"] for source in self.sources],
                "source_lengths": [len(source.dataset) for source in self.sources],
                "source_offsets": [source.offset for source in self.sources],
                "source_epochs": [source.epoch for source in self.sources],
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "sampler": self.sampler.state_dict(),
            },
            path,
        )

    def load(self, rollout_id: Any = None) -> None:
        if not self.args.load:
            return
        path = self._state_path(self.args.load, rollout_id)
        if not path.exists():
            logger.info("Multi-task dataset checkpoint %s does not exist.", path)
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        current_names = [source.config["name"] for source in self.sources]
        if state.get("source_names", current_names) != current_names:
            raise ValueError("Saved multi-task source names/order do not match the current manifest.")
        current_lengths = [len(source.dataset) for source in self.sources]
        if state.get("source_lengths", current_lengths) != current_lengths:
            raise ValueError("Saved multi-task source lengths do not match the current datasets.")
        if len(state["source_offsets"]) != len(self.sources):
            raise ValueError("Saved multi-task source count does not match the current manifest.")
        for source, offset, epoch in zip(self.sources, state["source_offsets"], state["source_epochs"], strict=True):
            source.offset = int(offset)
            source.epoch = int(epoch)
            if not 0 <= source.offset <= len(source.dataset) or source.epoch < 0:
                raise ValueError("Saved multi-task source cursor is outside the current dataset.")
            if self.args.rollout_shuffle:
                source.dataset.shuffle(source.epoch)
        self.sample_group_index = int(state["sample_group_index"])
        self.sample_index = int(state["sample_index"])
        self.sampler.load_state_dict(state["sampler"])

    def __len__(self) -> int:
        return sum(len(source.dataset) for source in self.sources)
