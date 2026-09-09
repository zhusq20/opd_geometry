#!/usr/bin/env python3
"""Measure the three paper studies at one exported student/AdamW checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import copy
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slime_plugins.mopd.paper_diagnostics import (
    js_divergence, local_distillation_loss, normalization_decomposition,
    support_overlap, tensor_metrics,
)
from slime_plugins.mopd.loss import student_topk_advantage
from slime_plugins.mopd.topk import STUDENT_TOPK


def next_logits(model, tokens, device):
    inputs = torch.tensor([tokens], dtype=torch.long, device=device)
    # Avoid materializing sequence_length x vocabulary logits for long prefixes.
    hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[:, -1, :]
    return model.lm_head(hidden).float()


def proposed_adamw_step(parameters, gradients, clip_grad):
    """Use the checkpoint's real moments without modifying any checkpoint tensor."""
    norm = math.sqrt(sum(float(value.double().square().sum()) for value in gradients.values()))
    coefficient = min(1.0, clip_grad / (norm + 1e-6)) if clip_grad > 0 else 1.0
    clipped, updates = {}, {}
    for name, state in parameters.items():
        value = state["value"].float()
        gradient = gradients[name].float() * coefficient
        beta1, beta2 = state["betas"]
        step = int(state["step"]) + 1
        first = state["exp_avg"].float() * beta1 + gradient * (1 - beta1)
        second = state["exp_avg_sq"].float() * beta2 + gradient.square() * (1 - beta2)
        if state.get("bias_correction", True):
            first = first / (1 - beta1 ** step)
            second = second / (1 - beta2 ** step)
        after = value * (1 - float(state["lr"]) * float(state["weight_decay"]))
        after = after - float(state["lr"]) * first / (second.sqrt() + float(state["eps"]))
        clipped[name], updates[name] = gradient, after - value
    return clipped, updates


def mean_gradients(gradients, weights):
    return {name: sum((gradient[name] * weight for gradient, weight in zip(gradients, weights, strict=True)),
                      torch.zeros_like(gradients[0][name])) for name in gradients[0]}


def response_gradient(model, row, teacher_scores, loss, device, clip, topk=None):
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    for ordinal, position in enumerate(row["positions"]):
        logits = next_logits(model, row["prompt_ids"] + row["response_ids"][:position], device)
        objective = local_distillation_loss(
            logits, teacher_scores[ordinal].to(device).unsqueeze(0), loss=loss,
            weights=torch.tensor([1 / len(row["positions"])], device=device),
            action_ids=[row["fresh_actions"][ordinal]], advantage_clip=clip,
            topk=topk,
        )
        total_loss += float(objective.detach())
        objective.backward()
    return {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else torch.zeros_like(parameter, device="cpu")
            for name, parameter in model.named_parameters()}, total_loss


def generate_bank(model, tokenizer, manifest, tasks, args, batch):
    rng = random.Random(args.seed + batch * 100003)
    sources = {source["name"]: source for source in manifest["sources"]}
    rows = []
    requested_counts = getattr(args, "domain_response_counts", None)
    counts = list(map(int, requested_counts.split(","))) if requested_counts else [args.responses_per_domain] * len(tasks)
    if len(counts) != len(tasks) or any(count <= 0 for count in counts):
        raise ValueError("Domain response counts must give one positive count per active task")
    for task, count in zip(tasks, counts, strict=True):
        path = Path(os.path.expandvars(sources[task]["path"]))
        if not path.is_absolute():
            path = Path(args.manifest).resolve().parent / path
        prompts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for index in rng.sample(range(len(prompts)), count):
            prompt = prompts[index]["prompt"]
            if not isinstance(prompt, str):
                raise ValueError("Probe manifests must contain the prepared diagnostic string prompts")
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            with torch.no_grad():
                generated = model.generate(
                    torch.tensor([prompt_ids], device=args.device), do_sample=True,
                    temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id, use_cache=True,
                )[0].tolist()[len(prompt_ids):]
            if not generated:
                raise ValueError("A diagnostic prompt produced an empty response")
            positions = sorted(rng.sample(range(len(generated)), min(args.prefixes_per_response, len(generated))))
            row = {"task": task, "prompt_index": index, "prompt": prompt, "prompt_ids": prompt_ids,
                   "response_ids": generated, "loss_mask": [1] * len(generated), "positions": positions, "fresh_actions": []}
            with torch.no_grad():
                for position in positions:
                    probabilities = next_logits(model, prompt_ids + generated[:position], args.device).softmax(-1)
                    row["fresh_actions"].append(int(torch.multinomial(probabilities, 1)))
            rows.append(row)
    return rows


def parameter_layers(values):
    layers = {}
    for name, value in values.items():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        layer = "layer_" + match.group(1) if match else "embedding" if "embed_tokens" in name else "output"
        layers.setdefault(layer, {})[name] = value
    return layers


@torch.no_grad()
def heldout_loss(model, rows, teachers, tasks, domain_weights, device, loss="student_topk", topk=STUDENT_TOPK):
    totals = dict.fromkeys(tasks, 0.0)
    counts = dict.fromkeys(tasks, 0)
    for index, row in enumerate(rows):
        counts[row["task"]] += 1
        for ordinal, position in enumerate(row["positions"]):
            logits = next_logits(model, row["prompt_ids"] + row["response_ids"][:position], device)
            teacher = teachers[row["task"]][index][ordinal].to(device).unsqueeze(0)
            if loss == "student_topk":
                # Track the normalized log-ratio, not the numerical PG surrogate.
                log_p = logits.float().log_softmax(-1)
                selected, ids = log_p.topk(min(topk, log_p.shape[-1]), dim=-1)
                value = -student_topk_advantage(selected, teacher.gather(-1, ids)).sum()
                totals[row["task"]] += float(value) / len(row["positions"])
            else:
                totals[row["task"]] += float(local_distillation_loss(
                    logits, teacher, loss=loss, weights=torch.tensor([1 / len(row["positions"])], device=device)))
    means = {task: totals[task] / counts[task] for task in tasks}
    return {**{f"heldout_loss_{task}": value for task, value in means.items()},
            "heldout_loss": sum(means[task] * weight for task, weight in zip(tasks, domain_weights, strict=True))}


def run_probe(args):
    started = time.perf_counter()
    cpu_started = time.process_time()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    topk_loss = getattr(args, "topk_loss", None) or (
        "teacher_topk" if snapshot.get("loss") == "teacher_topk" else "student_topk"
    )
    topk = (getattr(args, "topk", None) or snapshot.get("topk") or STUDENT_TOPK) if topk_loss == "student_topk" else 64
    if args.pg_advantage_clip is None:
        args.pg_advantage_clip = float(snapshot.get("pg_advantage_clip", 0.0))
    manifest = yaml.safe_load(Path(args.manifest).read_text())
    routes = yaml.safe_load(os.path.expandvars(Path(args.teachers).read_text()))["teachers"]
    tasks = tuple(args.tasks.split(",")) if args.tasks else tuple(snapshot.get("tasks") or [source["name"] for source in manifest["sources"]])
    domain_weights = snapshot.get("domain_weights") or [1 / len(tasks)] * len(tasks)
    if isinstance(domain_weights, dict):
        domain_weights = [domain_weights[task] for task in tasks]
    if getattr(args, "domain_weights", None):
        domain_weights = list(map(float, args.domain_weights.split(",")))
    if len(domain_weights) != len(tasks) or any(weight < 0 for weight in domain_weights) or not math.isclose(sum(domain_weights), 1.0):
        raise ValueError("Domain weights must give one nonnegative weight per task and sum to one")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    measurements_path = output / "measurements.jsonl"
    if measurements_path.exists() and measurements_path.stat().st_size:
        raise ValueError(f"Probe output already contains measurements; choose a new --output directory: {output}")
    config = AutoConfig.from_pretrained(snapshot["hf_checkpoint"])
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32, attn_implementation="sdpa")
    parameters = dict(model.named_parameters())
    if set(parameters) != set(snapshot["parameters"]):
        raise ValueError(f"Exported optimizer coordinates differ from HF model: missing={set(parameters) - set(snapshot['parameters'])}; extra={set(snapshot['parameters']) - set(parameters)}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(snapshot["parameters"][name]["value"])
    model.to(args.device).eval()
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(snapshot["hf_checkpoint"])
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, name=output.name, mode=args.wandb_mode,
                         config={**vars(args), "snapshot_step": snapshot["step"], "profile": snapshot["profile"],
                                 "probe_forward_precision": "fp32_master", "teacher_forward_precision": "fp32"})
        run.define_metric("paper/step")
        run.define_metric("paper/*", step_metric="paper/step")
    identity = hashlib.file_digest(Path(args.snapshot).open("rb"), "sha256").hexdigest()
    base = {"step": snapshot["step"], "profile": snapshot["profile"], "snapshot_sha256": identity,
            "probe_forward_precision": "fp32_master", "teacher_forward_precision": "fp32",
            "online_model_precision": snapshot.get("model_precision"), "pg_advantage_clip": args.pg_advantage_clip,
            "loss_normalization": "full_vocabulary", "topk_loss": topk_loss, "topk": topk,
            "topk_advantage_normalization": "per_prefix_student_mass" if topk_loss == "student_topk" else "none",
            "prefix_sampling": "uniform_without_replacement",
            "parameter_exclusions": "vocabulary_padding_and_shared_duplicates"}
    probe_manifest = {**base, "arguments": vars(args), "tasks": list(tasks), "domain_weights": list(domain_weights),
                      "teachers": {task: routes[task] for task in tasks}, "status": "running"}
    (output / "probe_manifest.json").write_text(json.dumps(probe_manifest, indent=2, allow_nan=False) + "\n")
    records = []
    def emit(kind, metrics, **fields):
        if "batch" in fields:
            fields["probe_batch"] = fields.pop("batch")
        if "prefix_set" not in fields:
            fields["prefix_set"] = "shared" if kind == "teacher_distance" else "common_prefix" if fields.get("branch") == "common_prefix" else "routed"
        record = {**base, "kind": kind, **fields, "metrics": metrics}
        records.append(record)
        with (output / "measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if run is not None:
            prefix = "/".join(str(record.get(key, "all")) for key in ("kind", "branch", "loss", "reduction", "prefix_set", "probe_batch", "task", "teacher_a", "teacher_b", "support_fraction", "threshold", "quantity", "layer"))
            run.log({"paper/step": record["step"], **{f"paper/{prefix}/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}})

    def score_teachers(bank_rows):
        result = {}
        for task in tasks:
            route = routes[task]
            teacher_tokenizer = AutoTokenizer.from_pretrained(route["model_path"])
            if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
                raise ValueError(f"Teacher {task} vocabulary does not match student")
            suffix = teacher_tokenizer.encode(route.get("prompt_suffix", ""), add_special_tokens=False)
            teacher = AutoModelForCausalLM.from_pretrained(route["model_path"], torch_dtype=torch.float32,
                                                         attn_implementation="sdpa").to(args.device).eval()
            with torch.no_grad():
                result[task] = [torch.cat([next_logits(teacher, row["prompt_ids"] + suffix + row["response_ids"][:position], args.device).log_softmax(-1).cpu()
                                          for position in row["positions"]]) for row in bank_rows]
            del teacher
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        return result

    def normalization_step_record(gradient, reduction, rows, teacher_scores, before, batch):
        _, update = proposed_adamw_step(snapshot["parameters"], gradient, snapshot["clip_grad"])
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(snapshot["parameters"][name]["value"] + update[name])
            after = heldout_loss(model, rows, teacher_scores, tasks, domain_weights, args.device, topk_loss, topk)
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(snapshot["parameters"][name]["value"])
        metrics = {**after, "heldout_loss_before": before["heldout_loss"],
                   "heldout_loss_decrease": before["heldout_loss"] - after["heldout_loss"]}
        for task in tasks:
            metrics[f"heldout_loss_decrease_{task}"] = before[f"heldout_loss_{task}"] - after[f"heldout_loss_{task}"]
        emit("normalization", metrics, batch=batch, reduction=reduction, loss=topk_loss, branch="heldout_step",
             heldout_metric=f"student_top{topk}_normalized_logratio" if topk_loss == "student_topk" else "teacher_top64_corrected_reverse_kl")
        for quantity, values in (("raw_gradient", gradient), ("update", update)):
            emit("normalization", tensor_metrics(values), batch=batch, reduction=reduction, loss=topk_loss, quantity=quantity, layer="all")

    for batch in range(args.batches):
        rows = generate_bank(model, tokenizer, manifest, tasks, args, batch)
        torch.save({"snapshot_sha256": identity, "batch": batch, "rows": rows}, output / f"prefix_bank_{batch:02d}.pt")
        student_scores = []
        with torch.no_grad():
            for row in rows:
                student_scores.append(torch.cat([next_logits(model, row["prompt_ids"] + row["response_ids"][:position], args.device).log_softmax(-1).cpu()
                                                for position in row["positions"]]))
        teachers = score_teachers(rows)
        torch.save({"snapshot_sha256": identity, "probe_batch": batch, "rows": rows,
                    "teacher_log_probs": teachers, "student_log_probs": student_scores}, output / f"prefix_bank_{batch:02d}.pt")
        heldout_manifest_path = Path(getattr(args, "heldout_manifest", None) or Path(args.manifest).with_name("heldout.yaml"))
        heldout_args = copy.copy(args)
        heldout_args.manifest = str(heldout_manifest_path)
        heldout_args.seed = args.seed + 70000001
        heldout_manifest = yaml.safe_load(heldout_manifest_path.read_text())
        heldout_rows = generate_bank(model, tokenizer, heldout_manifest, tasks, heldout_args, batch)
        heldout_teachers = score_teachers(heldout_rows)
        torch.save({"snapshot_sha256": identity, "batch": batch, "rows": heldout_rows, "teacher_log_probs": heldout_teachers},
                   output / f"heldout_bank_{batch:02d}.pt")
        heldout_before = heldout_loss(model, heldout_rows, heldout_teachers, tasks, domain_weights, args.device, topk_loss, topk)
        # Every teacher pair uses every cached prefix with the same response/domain weights.
        response_counts = {task: sum(row["task"] == task for row in rows) for task in tasks}
        prefix_weight = torch.cat([torch.full((len(row["positions"]),), domain_weights[tasks.index(row["task"])] / response_counts[row["task"]] / len(row["positions"])) for row in rows])
        for left, right in itertools.combinations(tasks, 2):
            emit("teacher_distance", {"js": js_divergence(torch.cat(teachers[left]), torch.cat(teachers[right]), prefix_weight)},
                 batch=batch, teacher_a=left, teacher_b=right, distribution="full_vocab")
        for task in tasks:
            emit("teacher_distance", {"js": js_divergence(torch.cat(teachers[task]), torch.cat(student_scores), prefix_weight)},
                 batch=batch, teacher_a=task, teacher_b="student", distribution="full_vocab")
        branch_gradients, branch_steps = {}, {}
        routed_response_gradients = {}
        for loss in (topk_loss, "sampled_pg", "full_vocab"):
            for task in tasks:
                chosen = [index for index, row in enumerate(rows) if row["task"] == task]
                gradients, losses = zip(*(response_gradient(model, rows[index], teachers[task][index], loss, args.device, args.pg_advantage_clip, topk) for index in chosen), strict=True)
                gradient = mean_gradients(gradients, [1 / len(chosen)] * len(chosen))
                branch = f"routed/{task}/{loss}"
                branch_gradients[branch] = gradient
                if loss == topk_loss:
                    routed_response_gradients[task] = gradients
                clipped, step = proposed_adamw_step(snapshot["parameters"], gradient, snapshot["clip_grad"])
                branch_steps[branch] = step
                for quantity, values in (("raw_gradient", gradient), ("clipped_gradient", clipped), ("update", step)):
                    emit("sparsity", tensor_metrics(values), batch=batch, task=task, branch=branch, loss=loss,
                         quantity=quantity, reduction="domain_response", responses=len(chosen),
                         valid_tokens=sum(len(rows[index]["response_ids"]) for index in chosen),
                         measured_prefixes=sum(len(rows[index]["positions"]) for index in chosen), layer="all")
            joint = mean_gradients([branch_gradients[f"routed/{task}/{loss}"] for task in tasks], domain_weights)
            branch = f"joint/{loss}"
            branch_gradients[branch] = joint
            _, branch_steps[branch] = proposed_adamw_step(snapshot["parameters"], joint, snapshot["clip_grad"])
            for quantity, values in (("raw_gradient", joint), ("update", branch_steps[branch])):
                emit("sparsity", tensor_metrics(values), batch=batch, branch=branch, loss=loss, quantity=quantity,
                     reduction="domain_response", responses=len(rows), layer="all")
            for left, right in itertools.combinations(tasks, 2):
                for quantity, collection in (("raw_gradient", branch_gradients), ("update", branch_steps)):
                    a, b = collection[f"routed/{left}/{loss}"], collection[f"routed/{right}/{loss}"]
                    for layer, layer_a in {"all": a, **parameter_layers(a)}.items():
                        layer_b = b if layer == "all" else parameter_layers(b)[layer]
                        for fraction in (0.01, 0.05, 0.1):
                            emit("overlap", support_overlap(layer_a, layer_b, fraction), batch=batch, teacher_a=left, teacher_b=right,
                                 branch="routed", loss=loss, quantity=quantity, support_fraction=fraction, layer=layer)
                    for threshold in (0.0, 1e-8, 1e-7, 1e-6):
                        emit("overlap", support_overlap(a, b, threshold=threshold), batch=batch, teacher_a=left, teacher_b=right,
                             branch="routed", loss=loss, quantity=quantity, threshold=threshold, layer="all")
        # Exact fixed-batch normalization and length-covariance comparison.
        domain_lengths = {task: [len(row["response_ids"]) for row in rows if row["task"] == task] for task in tasks}
        token_gradients = []
        for task in tasks:
            lengths = domain_lengths[task]
            gradients = routed_response_gradients[task]
            token_gradients.append(mean_gradients(gradients, [length / sum(lengths) for length in lengths]))
            identity_record = normalization_decomposition([torch.cat([value.reshape(-1) for value in gradient.values()]) for gradient in gradients], lengths)
            emit("normalization", {"covariance_l2": float(identity_record["covariance"].norm()), "decomposition_residual_l2": identity_record["residual_l2"],
                                   "mean_response_length": identity_record["mean_length"], "domain_token_share": sum(lengths) / sum(sum(x) for x in domain_lengths.values())},
                 batch=batch, task=task, loss=topk_loss)
        reduction_gradients = {"domain_response": branch_gradients[f"joint/{topk_loss}"]}
        normalization_step_record(branch_gradients[f"joint/{topk_loss}"], "domain_response", heldout_rows, heldout_teachers, heldout_before, batch)
        for reduction, weights in (("domain_token", domain_weights), ("global_token", [sum(domain_lengths[task]) / sum(sum(x) for x in domain_lengths.values()) for task in tasks])):
            gradient = mean_gradients(token_gradients, weights)
            reduction_gradients[reduction] = gradient
            normalization_step_record(gradient, reduction, heldout_rows, heldout_teachers, heldout_before, batch)
        for left, right in itertools.combinations(reduction_gradients, 2):
            comparison = support_overlap(reduction_gradients[left], reduction_gradients[right], 0.05)
            emit("normalization", comparison, batch=batch, reduction_a=left, reduction_b=right,
                 loss=topk_loss, quantity="raw_gradient", branch="gradient_comparison")
        zero = {name: torch.zeros_like(state["value"]) for name, state in snapshot["parameters"].items()}
        _, zero_step = proposed_adamw_step(snapshot["parameters"], zero, snapshot["clip_grad"])
        emit("sparsity", tensor_metrics(zero_step), batch=batch, branch="zero_gradient", quantity="update", layer="all")
        for task in tasks:
            emit("overlap", support_overlap(branch_steps[f"routed/{task}/{topk_loss}"], zero_step, 0.05),
                 batch=batch, branch="zero_gradient_control", teacher_a=task, teacher_b="zero_gradient", loss=topk_loss,
                 quantity="update", support_fraction=0.05, layer="all")
        for loss in (topk_loss, "sampled_pg"):
            for branch in [f"routed/{task}" for task in tasks] + ["joint"]:
                for quantity, collection in (("raw_gradient", branch_gradients), ("update", branch_steps)):
                    emit("supervision", support_overlap(collection[f"{branch}/{loss}"], collection[f"{branch}/full_vocab"], 0.05),
                         batch=batch, branch=branch, loss=loss, reference_loss="full_vocab", quantity=quantity, support_fraction=0.05)
        # Smaller same-prefix control: all routed teachers use the same bank and paired PG actions.
        for loss in (topk_loss, "sampled_pg"):
            common, common_gradients = {}, {}
            for task in tasks:
                gradients = [response_gradient(model, row, teachers[task][index], loss, args.device, args.pg_advantage_clip, topk)[0] for index, row in enumerate(rows)]
                response_weights = [domain_weights[tasks.index(row["task"])] / response_counts[row["task"]] for row in rows]
                gradient = mean_gradients(gradients, response_weights)
                common_gradients[task] = gradient
                _, common[task] = proposed_adamw_step(snapshot["parameters"], gradient, snapshot["clip_grad"])
            for left, right in itertools.combinations(tasks, 2):
                for quantity, collection in (("update", common), ("raw_gradient", common_gradients)):
                    emit("overlap", support_overlap(collection[left], collection[right], 0.05), batch=batch, teacher_a=left, teacher_b=right,
                         branch="common_prefix", loss=loss, quantity=quantity, support_fraction=0.05, layer="all")
    wall_seconds = time.perf_counter() - started
    uses_cuda = str(args.device).startswith("cuda")
    emit("cost", {"wall_seconds": wall_seconds, "cpu_seconds": time.process_time() - cpu_started,
                  "occupied_gpu_seconds": wall_seconds if uses_cuda else 0.0,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if uses_cuda else 0},
         branch="local_probe", batches=args.batches)
    probe_manifest["status"] = "complete"
    probe_manifest["measurement_records"] = len(records)
    (output / "probe_manifest.json").write_text(json.dumps(probe_manifest, indent=2, allow_nan=False) + "\n")
    if run is not None:
        artifact = wandb.Artifact(output.name + "-measurements", type="paper-probe-data")
        artifact.add_file(str(measurements_path))
        artifact.add_file(str(output / "probe_manifest.json"))
        run.log_artifact(artifact)
        run.finish()
    return records


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--snapshot", required=True)
    result.add_argument("--manifest", required=True)
    result.add_argument("--teachers", required=True)
    result.add_argument("--heldout-manifest", help="Defaults to heldout.yaml beside the diagnostic manifest")
    result.add_argument("--output", required=True)
    result.add_argument("--tasks")
    result.add_argument("--batches", type=int, default=2)
    result.add_argument("--responses-per-domain", type=int, default=2)
    result.add_argument("--domain-response-counts", help="Optional comma-separated per-task diagnostic quotas")
    result.add_argument("--domain-weights", help="Optional comma-separated weights for unequal-quota normalization comparisons")
    result.add_argument("--prefixes-per-response", type=int, default=4)
    result.add_argument("--max-new-tokens", type=int, default=4096)
    result.add_argument("--pg-advantage-clip", type=float, help="Defaults to the exported online PG clipping value")
    result.add_argument("--topk-loss", choices=("student_topk", "teacher_topk"),
                        help="Defaults to student TopK, or legacy teacher Top64 for a teacher_topk snapshot")
    result.add_argument("--topk", type=int, choices=(16, 64),
                        help="Student support size; defaults to the snapshot's K, or 16 for old student snapshots")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda")
    result.add_argument("--wandb-project", default="mopd-optimization-dynamics")
    result.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return result


if __name__ == "__main__":
    run_probe(parser().parse_args())
