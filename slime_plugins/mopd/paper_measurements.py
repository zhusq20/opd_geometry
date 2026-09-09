"""Record actual optimizer updates and export their state for fixed-prefix probes."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

from .optimizer_views import build_optimizer_parameter_views
from .paper_diagnostics import tensor_metrics


def _cpu(value):
    return value.detach().to(device="cpu", copy=True)


def _snapshot(views, getter, vocab_size=None):
    values = {}
    for view in views:
        if not view.contributes_to_norm:
            continue
        value = getter(view)
        # Match the HF probe population: unused vocabulary padding and duplicate
        # tied parameters do not define model coordinates in either measurement.
        if vocab_size is not None and view.name.split(".", 1)[1] in {
            "embedding.word_embeddings.weight", "output_layer.weight"
        }:
            value = value.reshape(view.model_parameter.shape)[:vocab_size]
        values[view.name] = _cpu(value.reshape(-1))
    return values


def _atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def append_measurement(args, record):
    from slime.utils.logging_utils import log

    directory = Path(args.mopd_output_dir) / "paper"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "measurements.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    dimensions = ("kind", "loss", "topk", "reduction", "prefix_set", "probe_batch", "task", "teacher_a", "teacher_b",
                  "support_fraction", "threshold", "quantity", "layer")
    prefix = "/".join(str(record[key]) for key in dimensions if record.get(key) is not None)
    values = {f"paper/{prefix}/{key}": value for key, value in record["metrics"].items() if isinstance(value, (int, float))}
    log(args, {"paper/step": record["step"], **values}, step_key="paper/step")


def _export_metadata(args, views, step):
    from .sampler import active_tasks, active_domain_weights
    from .topk import student_topk_size

    return {
        "step": step, "profile": args.mopd_profile, "hf_checkpoint": args.hf_checkpoint,
        "clip_grad": float(args.clip_grad),
        "model_precision": str(views[0].model_parameter.dtype),
        "optimizer_precision": str(views[0].optimizer_parameter.dtype),
        "loss": args.mopd_loss, "reduction": args.mopd_reduction,
        "topk": student_topk_size(args) if args.mopd_loss in {"student_topk", "topk_intersection"} else None,
        "tasks": list(active_tasks(args)), "domain_weights": active_domain_weights(args),
        "pg_advantage_clip": float(getattr(args, "mopd_pg_advantage_clip", 0.0)),
    }


def _export_parameter_fields(args, views):
    """Yield one converted field at a time for export and exact retry validation."""
    from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf
    from .optimizer import _state_step

    model_name = "smollm3" if args.mopd_profile == "smollm3" else "qwen3"
    for view in views:
        if not view.contributes_to_norm:
            continue
        name = "module.module." + view.name.split(".", 1)[1]
        shape = view.model_parameter.shape
        if view.optimizer_value().numel() != view.model_parameter.numel():
            raise ValueError("Portable paper optimizer export needs complete local parameters")
        state = view.optimizer_state
        group = view.optimizer_group
        metadata = {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
        metadata.update(step=_state_step(view), bias_correction=bool(group.get("bias_correction", True)))
        for field, tensor in (
            ("value", view.optimizer_value()), ("model_value", view.model_value()),
            ("exp_avg", state.get("exp_avg")), ("exp_avg_sq", state.get("exp_avg_sq")),
        ):
            value = torch.zeros(shape, dtype=torch.float32) if tensor is None else _cpu(tensor).reshape(shape)
            for hf_name, hf_value in convert_to_hf(args, model_name, name, value):
                yield hf_name, metadata, field, hf_value


def _validate_initial_reuse(args, views, initial, checkpoint_path):
    """A retry before its first update may reuse identical durable snapshots."""
    for precision, getter in (("fp32", lambda view: view.optimizer_value()),
                              ("bf16", lambda view: view.model_value())):
        names = set()
        for view in views:
            current = _snapshot([view], getter, args.vocab_size)
            for name, value in current.items():
                names.add(name)
                saved = initial.get(precision, {}).get(name)
                if saved is None or saved.dtype != value.dtype or not torch.equal(saved, value):
                    raise ValueError(f"Cannot reuse step-zero initial parameters: {precision}/{name} differs")
        if names != set(initial.get(precision, {})):
            raise ValueError(f"Cannot reuse step-zero initial parameters: {precision} coordinates differ")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    for key, value in _export_metadata(args, views, 0).items():
        # Student snapshots written before K was configurable used Top16.
        default = 16 if key == "topk" and saved.get("loss") == "student_topk" else None
        if saved.get(key, default) != value:
            raise ValueError(f"Cannot reuse step-zero optimizer snapshot: {key} differs")
    names = set()
    for name, metadata, field, value in _export_parameter_fields(args, views):
        names.add(name)
        state = saved["parameters"].get(name, {})
        if any(state.get(key) != expected for key, expected in metadata.items()):
            raise ValueError(f"Cannot reuse step-zero optimizer snapshot: {name} Adam metadata differs")
        previous = state.get(field)
        if previous is None or previous.dtype != value.dtype or not torch.equal(previous, value):
            raise ValueError(f"Cannot reuse step-zero optimizer snapshot: {name}/{field} differs")
    if names != set(saved["parameters"]):
        raise ValueError("Cannot reuse step-zero optimizer snapshot: HF coordinates differ")


def export_hf_optimizer_state(args, views, step):
    """Export FP32 master values and Adam moments in the HF parameter layout.

    Both model families are dense and run with TP=DP=PP=1. The same permutation
    used for inference weights is applied to the moments, preserving coordinates.
    """
    parameters = {}
    for name, metadata, field, value in _export_parameter_fields(args, views):
        parameters.setdefault(name, dict(metadata))[field] = value.contiguous().clone()
    path = Path(args.mopd_output_dir) / "paper" / f"checkpoint_step_{step:04d}.pt"
    _atomic_save({**_export_metadata(args, views, step), "parameters": parameters}, path)
    return path


class PaperMeasurements:
    def __init__(self, args, views, start_step):
        self.args, self.views, self.step = args, views, start_step
        self.steps = {1, args.mopd_total_steps, *(int(x) for x in args.mopd_checkpoint_steps.split(",") if x)}
        self.directory = Path(args.mopd_output_dir) / "paper"
        self.directory.mkdir(parents=True, exist_ok=True)
        initial_path = self.directory / "initial_parameters.pt"
        if start_step:
            self.initial = torch.load(initial_path, map_location="cpu", weights_only=True)
            # Earlier exports retained flat padded vocabulary rows. Preserve
            # their original initialization while matching the current HF-sized
            # coordinate population when resuming those runs.
            for view in views:
                if view.contributes_to_norm and view.name.split(".", 1)[1] in {
                    "embedding.word_embeddings.weight", "output_layer.weight"
                }:
                    width = view.model_parameter.numel() // view.model_parameter.shape[0]
                    size = min(args.vocab_size, view.model_parameter.shape[0]) * width
                    for precision in ("fp32", "bf16"):
                        self.initial[precision][view.name] = self.initial[precision][view.name].reshape(-1)[:size]
        else:
            checkpoint_path = self.directory / "checkpoint_step_0000.pt"
            if initial_path.is_file() and checkpoint_path.is_file():
                self.initial = torch.load(initial_path, map_location="cpu", weights_only=True, mmap=True)
                _validate_initial_reuse(args, views, self.initial, checkpoint_path)
            else:
                self.initial = {
                    "fp32": _snapshot(views, lambda view: view.optimizer_value(), args.vocab_size),
                    "bf16": _snapshot(views, lambda view: view.model_value(), args.vocab_size),
                }
                _atomic_save(self.initial, initial_path)
                export_hf_optimizer_state(args, views, 0)
            # The initial displacement is identically zero; avoid sorting a full
            # model of zeros just to establish the origin of the plotted curve.
            count = sum(value.numel() for value in self.initial["fp32"].values())
            zero = {"parameters": count, "l2": 0.0, "energy90_fraction": None,
                    **{f"sparsity_at_{x:g}": 1.0 for x in (0, 1e-8, 1e-7, 1e-6)},
                    **{f"energy_at_{x:g}": None for x in (0.01, 0.05, 0.1)}}
            for quantity in ("delta_fp32", "delta_bf16"):
                append_measurement(args, self.record(quantity, zero))
        self.before = None

    def record(self, quantity, metrics, layer="all"):
        return {"kind": "sparsity", "step": self.step, "quantity": quantity, "layer": layer,
                "profile": self.args.mopd_profile, "loss": self.args.mopd_loss,
                "topk": getattr(self.args, "mopd_topk", 16) if self.args.mopd_loss in {"student_topk", "topk_intersection"} else None,
                "reduction": self.args.mopd_reduction, "metrics": metrics,
                "parameter_exclusions": "vocabulary_padding_and_shared_duplicates"}

    def before_step(self):
        if self.step + 1 not in self.steps:
            return
        self.before = {
            "fp32": _snapshot(self.views, lambda view: view.optimizer_value(), self.args.vocab_size),
            "bf16": _snapshot(self.views, lambda view: view.model_value(), self.args.vocab_size),
            "gradient": _snapshot(self.views, lambda view: view.optimizer_gradient(), self.args.vocab_size),
        }

    def _measure(self, quantity, values):
        append_measurement(self.args, self.record(quantity, tensor_metrics(values)))
        layers = {}
        for name, value in values.items():
            match = re.search(r"decoder\.layers\.(\d+)", name)
            layer = f"layer_{match[1]}" if match else "embedding_and_head"
            layers.setdefault(layer, []).append(value)
        for layer, tensors in layers.items():
            append_measurement(self.args, self.record(quantity, tensor_metrics(tensors), layer))

    def after_step(self):
        self.step += 1
        if self.before is None:
            return
        current = _snapshot(self.views, lambda view: view.optimizer_value(), self.args.vocab_size)
        current_bf16 = _snapshot(self.views, lambda view: view.model_value(), self.args.vocab_size)
        self._measure("raw_gradient", self.before["gradient"])
        self._measure("clipped_gradient", _snapshot(self.views, lambda view: view.optimizer_gradient(), self.args.vocab_size))
        self._measure("update", {key: value.float() - self.before["fp32"][key].float() for key, value in current.items()})
        self._measure("delta_fp32", {key: value.float() - self.initial["fp32"][key].float() for key, value in current.items()})
        self._measure("delta_bf16", {key: value.float() - self.initial["bf16"][key].float() for key, value in current_bf16.items()})
        export_hf_optimizer_state(self.args, self.views, self.step)
        self.before = None


def initialize_paper_measurements(args, model, optimizer, start_step):
    if getattr(args, "mopd_skip_paper_measurements", False):
        return
    from .optimizer import _model_entries

    views = build_optimizer_parameter_views(_model_entries(model), optimizer, requested_optimizer="adam")
    optimizer._paper_measurements = PaperMeasurements(args, views, start_step)
