#!/usr/bin/env python3
"""Tables and figures for four single-seed runs and the Uniform/250 comparison."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from examples.mopd_gpas.analyze_capability import DOMAINS, TEACHER_BY_DOMAIN, evaluate
from slime_plugins.mopd.prompting import PROMPT_FORMAT
from slime_plugins.mopd.reference_bank import write_json
from slime_plugins.mopd.sampler import CORE_RUNS, TASKS

STEPS = (0, 100, 200, 300, 400, 500)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def execution_gpu_hours(path, occupied_gpus):
    """Include completed/failed sessions, excluding downtime before a resume."""
    manifest_path = path / "provenance/run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    started = manifest["created_at_utc"]
    seconds = 0.0
    for event in manifest.get("resume_events", []):
        marker = event.get("previous_failure_marker", {}).get("path")
        if not marker:
            return None
        marker = Path(marker)
        if not marker.is_file():
            marker = path / "provenance" / marker.name
        if not marker.is_file():
            # An abrupt kill may have no recorded end. Never count the idle
            # gap until the next launch as GPU occupancy or invent a cost.
            return None
        stopped = read_json(marker)["at_utc"]
        seconds += (datetime.fromisoformat(stopped) - datetime.fromisoformat(started)).total_seconds()
        started = event["at_utc"]
    stopped = manifest.get("finished_at_utc")
    if not stopped:
        return None
    seconds += (datetime.fromisoformat(stopped) - datetime.fromisoformat(started)).total_seconds()
    return seconds * occupied_gpus / 3600


def crossing_cost(curve, target):
    if curve[0]["weighted_loss"] <= target:
        return curve[0]["gpu_hours"]
    for previous, current in zip(curve, curve[1:]):
        if previous["weighted_loss"] > target >= current["weighted_loss"]:
            fraction = (previous["weighted_loss"] - target) / (previous["weighted_loss"] - current["weighted_loss"])
            return previous["gpu_hours"] + fraction * (current["gpu_hours"] - previous["gpu_hours"])
    return "unreached"


def load_run(root, run):
    path = root / run
    completion = read_json(path / "run_complete.json")
    if completion.get("status") != "complete" or completion.get("final_num_updates") != 500:
        raise ValueError(f"{run} is not a complete 500-step training run")
    rows = read_jsonl(path / "allocation/allocation.jsonl")
    if len(rows) != 500:
        raise ValueError(f"{run} must contain exactly 500 allocation records")
    exposure = dict.fromkeys(TASKS, 0)
    cumulative_seconds = [0.0]
    for step, row in enumerate(rows, 1):
        counts = [row["counts"][task] for task in TASKS]
        if row["optimizer_updates_after"] != step or row["attempted_responses_after"] != step * 64:
            raise ValueError(f"{run} has an inconsistent step/response clock")
        if sum(counts) != 16 or any(count < 2 or count > 8 for count in counts):
            raise ValueError(f"{run} violates the shared allocation bounds")
        for task, count in zip(TASKS, counts, strict=True):
            exposure[task] += count * 4
        cumulative_seconds.append(cumulative_seconds[-1] + float(row["feedback"]["total_gpu_seconds"]))
    if max(exposure.values()) > 16000:
        raise ValueError(f"{run} consumed more than the frozen task stream")
    checkpoint_costs = read_jsonl(path / "checkpoint_costs.jsonl")
    curve = []
    for step in STEPS:
        record = read_json(path / "fixed_loss" / f"step_{step:04d}.json")
        if any(len(record["prompt_losses"][task]) != 64 for task in TASKS):
            raise ValueError("fixed loss requires 64 prompt-level observations per domain")
        if not np.isclose(record["weighted_loss"], np.mean(list(record["task_losses"].values()))):
            raise ValueError("fixed loss must use equal task weights")
        save_cost = sum(row["wall_seconds"] * row["occupied_gpus"] for row in checkpoint_costs if row["step"] <= step)
        curve.append({**record, "gpu_hours": (cumulative_seconds[step] + save_cost) / 3600})
    fresh = read_json(path / "fixed_loss/fresh_final.json")
    occupied = rows[-1]["feedback"].get("allocated_gpu_count", 2)
    evaluation_hours = (
        (sum(row.get("evaluation_wall_seconds", 0.0) for row in curve) + fresh.get("evaluation_wall_seconds", 0.0))
        * occupied
        / 3600
    )
    hardware_hours = {}
    provenance_path = path / "provenance/run_manifest.json"
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        environment = provenance["environment"]
        devices = set(str(environment.get("cuda_visible_devices") or "").split(","))
        devices.update(str(value) for value in environment.get("mopd_teacher_gpus", {}).values() if value is not None)
        inventory = provenance.get("hardware", {}).get("nvidia_smi", {}).get("gpus", [])
        names = Counter(row.split(",")[1].strip() for row in inventory if row.split(",")[0].strip() in devices)
        hardware_hours = {name: curve[-1]["gpu_hours"] * count / occupied for name, count in names.items()}
    return {
        "curve": curve,
        "fresh": fresh,
        "exposure": exposure,
        "allocation": rows,
        "training_gpu_hours": curve[-1]["gpu_hours"],
        "loss_evaluation_gpu_hours": evaluation_hours,
        "training_gpu_hours_by_model": hardware_hours,
        "execution_gpu_hours": execution_gpu_hours(path, occupied),
    }


def capability_bootstrap(configs, replicates=1000):
    """Pair benchmark questions; GPQA's four responses have already been clustered."""
    baseline = configs["uniform-s1"]
    rng = np.random.default_rng(42)
    boot = {run: {} for run in configs}
    for task in TASKS:
        indices = baseline["domains"][task]["prompt_indices"]
        draws = rng.integers(len(indices), size=(replicates, len(indices)))
        for run, config in configs.items():
            values = config["domains"][task]
            if values["prompt_indices"] != indices:
                raise ValueError(f"benchmark question IDs differ for {run}/{task}")
            boot[run][task] = np.asarray(values["prompt_scores"])[draws].mean(axis=1) * 100
    result = {}
    for run in configs:
        domain_deltas = np.array([boot[run][task] - boot["uniform-s1"][task] for task in TASKS])
        macro = np.array(list(boot[run].values())).mean(axis=0)
        result[run] = {
            "mean_score_95ci": np.percentile(macro, [2.5, 97.5]).tolist(),
            "mean_delta_95ci": np.percentile(domain_deltas.mean(axis=0), [2.5, 97.5]).tolist(),
            "worst_domain_delta_95ci": np.percentile(domain_deltas.min(axis=0), [2.5, 97.5]).tolist(),
            "domain_delta_95ci": {
                task: np.percentile(domain_deltas[i], [2.5, 97.5]).tolist() for i, task in enumerate(TASKS)
            },
        }
    return {"replicates": replicates, "uncertainty": "paired benchmark questions; one training seed", "runs": result}


def mechanism_summary(record, replicates=1000):
    if record["checkpoint_step"] != 250 or len(record["trials"]) != 20:
        raise ValueError("the local comparison requires exactly 20 updates at Uniform/250")
    rng = np.random.default_rng(42)
    # Share bank resampling indices across every trial and branch.
    bank_draws = {task: rng.integers(64, size=(replicates, 64)) for task in TASKS}
    result = {}
    for branch in ("uniform", "gpas"):
        trials = [trial for trial in record["trials"] if trial["branch"] == branch]
        if sorted(trial["trial"] for trial in trials) != list(range(10)):
            raise ValueError(f"{branch} must contain all ten independent draws")
        if any(trial["after"]["bank_sha256"] != record["before"]["bank_sha256"] for trial in trials):
            raise ValueError("all local updates must be evaluated on the shared before/after bank")
        trial_draws = rng.integers(10, size=(replicates, 10))
        domains = {}
        means = []
        for task in TASKS:
            before = np.asarray(record["before"]["prompt_losses"][task])
            differences = np.array([before - np.asarray(t["after"]["prompt_losses"][task]) for t in trials])
            bootstrap = differences[:, bank_draws[task]].mean(axis=-1).T
            bootstrap = bootstrap[np.arange(replicates)[:, None], trial_draws].mean(axis=-1)
            means.append(bootstrap)
            trial_means = differences.mean(axis=1)
            domains[task] = {
                "trial_decreases": trial_means.tolist(),
                "mean_decrease": float(trial_means.mean()),
                "non_decrease_frequency": float(np.mean(trial_means <= 0)),
                "mean_decrease_95ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            }
        result[branch] = {
            "domains": domains,
            "mean_decrease_95ci": np.percentile(np.mean(means, axis=0), [2.5, 97.5]).tolist(),
        }
    return result


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for run, data in report["runs"].items():
        for axis, x in zip(axes, ("step", "gpu_hours"), strict=True):
            axis.plot([r[x] for r in data["curve"]], [r["weighted_loss"] for r in data["curve"]], "o-", label=run)
            axis.set(xlabel="Optimizer steps" if x == "step" else "Training GPU hours", ylabel="Fixed reference loss")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "fixed_loss.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for axis, task in zip(axes, TASKS, strict=True):
        for run in ("uniform-s1", "gpas-s1"):
            axis.plot(
                [0, 250, 500],
                [
                    report["capability_curves"][run][str(step)]["domains"][task]["score"] * 100
                    for step in (0, 250, 500)
                ],
                "o-",
                label=run,
            )
        axis.set(title=task, xlabel="Optimizer steps", ylabel="Score (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "capability.pdf")
    plt.close(fig)

    mechanism = report["common_checkpoint"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for index, branch in enumerate(("uniform", "gpas")):
        axes[0].bar(np.arange(4) + (index - 0.5) * 0.35, mechanism["counts"][branch], width=0.35, label=branch)
        for task_index, task in enumerate(TASKS):
            decreases = report["mechanism_summary"][branch]["domains"][task]["trial_decreases"]
            axes[2].scatter(
                np.full(10, task_index + (index - 0.5) * 0.16),
                decreases,
                label=branch if task_index == 0 else None,
                alpha=0.65,
            )
    axes[0].set(xticks=np.arange(4), xticklabels=TASKS, ylabel="Micro-batches")
    axes[0].legend()
    ratio = mechanism["variance_ratio"]
    if ratio is not None:
        axes[1].bar(["Uniform", "GPAS"], [1, ratio])
    axes[1].set(ylabel="Empirical update variance / Uniform")
    axes[2].set(xticks=np.arange(4), xticklabels=TASKS, ylabel="Fixed-bank one-step loss decrease")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "common_checkpoint.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    trajectory = report["runs"]["gpas-s1"]["allocation"]
    for task in TASKS:
        axes[0].plot(range(1, 501), [r["counts"][task] for r in trajectory], label=task)
        axes[1].plot(range(1, 501), [r["scaled_noise_after"][task] for r in trajectory], label=task)
    axes[0].set(xlabel="Optimizer steps", ylabel="GPAS micro-batches")
    axes[1].set(xlabel="Optimizer steps", ylabel="Preconditioned noise EMA")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "allocation.pdf")
    plt.close(fig)


def analyze(root, protocol, output):
    protocol = read_json(protocol)
    if protocol["schema_version"] not in (5, 6) or protocol["training"]["configs"] != list(CORE_RUNS):
        raise ValueError("analysis requires the two-week core protocol")
    if protocol["schema_version"] == 6 and protocol.get("prompt_format") != PROMPT_FORMAT:
        raise ValueError("protocol v6 requires the asymmetric student/teacher prompt format")
    runs = {run: load_run(root, run) for run in CORE_RUNS}
    bank_hashes = {row["bank_sha256"] for data in runs.values() for row in data["curve"]}
    if len(bank_hashes) != 1:
        raise ValueError("all methods and checkpoints must use the same fixed bank")
    initial_losses = [data["curve"][0]["prompt_losses"] for data in runs.values()]
    if any(value != initial_losses[0] for value in initial_losses):
        raise ValueError("initial student loss must be shared across runs")
    target = runs["uniform-s1"]["curve"][-1]["weighted_loss"]
    for data in runs.values():
        data["gpu_hours_to_uniform_final"] = crossing_cost(data["curve"], target)
    initial = evaluate(root / "capability_references/initial_student", DOMAINS)
    teachers = {}
    for teacher in set(TEACHER_BY_DOMAIN.values()):
        domains = {task: DOMAINS[task] for task in TASKS if TEACHER_BY_DOMAIN[task] == teacher}
        teachers[teacher] = evaluate(root / "capability_references" / teacher, domains)
    capability = {run: evaluate(root / run / "capability_eval/response_32000", DOMAINS) for run in CORE_RUNS}
    curves = {
        run: {
            "0": initial,
            "250": evaluate(root / run / "capability_eval/response_16000", DOMAINS),
            "500": capability[run],
        }
        for run in ("uniform-s1", "gpas-s1")
    }
    teacher_row = {
        "target": "assigned_teachers",
        "kind": "teacher",
        **{f"score_{task}": teachers[TEACHER_BY_DOMAIN[task]]["domains"][task]["score"] * 100 for task in TASKS},
        "teacher_note": "Math/code/IF/science: four domain-specific Qwen3-1.7B RL teachers",
    }
    rows = []
    for run, values in {"initial_student": initial, **capability}.items():
        scores = {task: values["domains"][task]["score"] * 100 for task in TASKS}
        row = {
            "target": run,
            "kind": "initial" if run == "initial_student" else "training",
            **{f"score_{task}": scores[task] for task in TASKS},
            "mean_score": float(np.mean(list(scores.values()))),
        }
        if run in runs:
            deltas = {task: scores[task] - capability["uniform-s1"]["domains"][task]["score"] * 100 for task in TASKS}
            row.update(
                {
                    **{f"delta_vs_uniform_{task}": value for task, value in deltas.items()},
                    "mean_delta": float(np.mean(list(deltas.values()))),
                    "worst_domain_delta": min(deltas.values()),
                    "training_gpu_hours": runs[run]["training_gpu_hours"],
                    "gpu_hours_to_uniform_final": runs[run]["gpu_hours_to_uniform_final"],
                    "fixed_loss": runs[run]["curve"][-1]["weighted_loss"],
                    "fresh_loss": runs[run]["fresh"]["weighted_loss"],
                }
            )
        rows.append(row)
    teacher_row["mean_score"] = float(np.mean([teacher_row[f"score_{task}"] for task in TASKS]))
    rows.insert(1, teacher_row)
    mechanism = read_json(root / "common-checkpoint-250/common_checkpoint.json")
    result = {
        "schema_version": protocol["schema_version"],
        "prompt_format": protocol.get("prompt_format"),
        "training_seed": 42,
        "teachers": protocol["teachers"],
        "runs": runs,
        "main_table": rows,
        "capability_curves": curves,
        "paired_capability_bootstrap": capability_bootstrap(capability),
        "common_checkpoint": mechanism,
        "mechanism_summary": mechanism_summary(mechanism),
        "target_fixed_loss": target,
    }
    capability_paths = [
        root / "capability_references/initial_student",
        *(root / "capability_references" / teacher for teacher in teachers),
        *(root / run / "capability_eval/response_32000" for run in CORE_RUNS),
        *(root / run / "capability_eval/response_16000" for run in curves),
    ]
    capability_hours = [execution_gpu_hours(path, 1) for path in capability_paths]
    training_execution_hours = [run["execution_gpu_hours"] for run in runs.values()]
    diagnostic_execution_hours = execution_gpu_hours(
        root / "common-checkpoint-250", mechanism.get("occupied_gpus") or 2
    )
    cost = {
        "training_gpu_hours": sum(run["training_gpu_hours"] for run in runs.values()),
        "loss_evaluation_gpu_hours": sum(run["loss_evaluation_gpu_hours"] for run in runs.values()),
        "diagnostic_gpu_hours": mechanism.get("wall_seconds", 0.0) * (mechanism.get("occupied_gpus") or 2) / 3600,
        "capability_gpu_hours": (
            sum(capability_hours) if all(value is not None for value in capability_hours) else None
        ),
        "training_execution_gpu_hours": (
            sum(training_execution_hours) if all(value is not None for value in training_execution_hours) else None
        ),
        "diagnostic_execution_gpu_hours": diagnostic_execution_hours,
        "retries": "execution costs include failed sessions; missing termination records leave total cost unknown",
    }
    cost["training_retry_and_launch_overhead_gpu_hours"] = (
        max(cost["training_execution_gpu_hours"] - cost["training_gpu_hours"] - cost["loss_evaluation_gpu_hours"], 0.0)
        if cost["training_execution_gpu_hours"] is not None
        else None
    )
    execution_costs = (
        cost["training_execution_gpu_hours"],
        cost["diagnostic_execution_gpu_hours"],
        cost["capability_gpu_hours"],
    )
    cost["research_gpu_hours"] = sum(execution_costs) if all(value is not None for value in execution_costs) else None
    result["cost"] = cost
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "main_table.csv", rows)
    write_table(
        output / "fixed_loss.csv",
        [
            {
                "run": run,
                "step": row["step"],
                "gpu_hours": row["gpu_hours"],
                "weighted_loss": row["weighted_loss"],
                **row["task_losses"],
            }
            for run, data in runs.items()
            for row in data["curve"]
        ],
    )
    write_json(output / "report.json", result)
    make_figures(result, output / "figures")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    analyze(cli.root, cli.protocol, cli.output)
