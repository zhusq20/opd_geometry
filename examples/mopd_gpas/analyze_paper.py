#!/usr/bin/env python3
"""Export measured data and figures for the current three empirical studies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scalar_events(path, step_key):
    """Merge worker fields at one clock; repeated fields use the latest value."""
    events = {}
    if path.is_file():
        for record in read_jsonl(path):
            values = record["metrics"]
            if step_key in values:
                events.setdefault(int(values[step_key]), {}).update(values)
    return sorted(events.items())


def measurement_rows(run):
    rows = []
    for directory in (run / "paper", run / "metrics/paper"):
        for path in sorted(directory.rglob("*")):
            if path.suffix == ".jsonl":
                records = read_jsonl(path)
            elif path.suffix == ".json":
                value = json.loads(path.read_text())
                records = value if isinstance(value, list) else [value]
            else:
                continue
            for record in records:
                if "kind" not in record:
                    continue
                rows.append({"run": run.name, **record, "source": str(path.relative_to(run))})
    return rows


def load_run(run):
    updates = scalar_events(run / "metrics/mopd.jsonl", "mopd/update")
    rollouts = dict(scalar_events(run / "metrics/rollout.jsonl", "rollout/step"))
    tokens = 0
    gpu_hours = 0.0
    curves = []
    for step, metrics in updates:
        tokens += int(metrics.get("mopd/valid_response_tokens", 0))
        gpu_seconds = metrics.get("mopd/total_gpu_seconds")
        gpu_hours = (
            gpu_hours + float(gpu_seconds) / 3600 if gpu_seconds is not None and gpu_hours is not None else None
        )
        row = {"run": run.name, "step": step, "valid_tokens": tokens, "gpu_hours": gpu_hours}
        rollout = rollouts.get(step - 1, {})
        row.update({**rollout, **metrics})
        curves.append(row)
    capabilities = []
    evaluation = {
        int(values.get("eval/num_updates", step)): values
        for step, values in scalar_events(run / "metrics/eval.jsonl", "eval/step")
    }
    for path in sorted((run / "capability_eval").glob("step_*/metrics/eval.jsonl")):
        step = 0 if run.name == "initial_student" or run.name.startswith("teacher_") else int(path.parents[1].name[5:])
        for _local_step, values in scalar_events(path, "eval/step"):
            evaluation[step] = values
    for step, metrics in sorted(evaluation.items()):
        previous = [row for row in curves if row["step"] <= step]
        cost = (
            previous[-1]
            if previous
            else {"valid_tokens": 0 if step == 0 else None, "gpu_hours": 0 if step == 0 else None}
        )
        for key, value in metrics.items():
            if key.startswith("eval/capability/") and key.count("/") == 2:
                capabilities.append(
                    {
                        "run": run.name,
                        "step": step,
                        "dataset": key.split("/")[-1],
                        "score": value,
                        "valid_tokens": cost["valid_tokens"],
                        "gpu_hours": cost["gpu_hours"],
                    }
                )
    marker = run / "run_complete.json"
    status = json.loads(marker.read_text())["status"] if marker.is_file() else "in_progress"
    return {
        "run": run.name,
        "status": status,
        "curves": curves,
        "capabilities": capabilities,
        "measurements": measurement_rows(run),
    }


def flatten_record(row):
    values = {key: value for key, value in row.items() if key != "metrics"}
    values.update(row.get("metrics", {}))
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in values.items()
    }


def write_csv(path, rows):
    rows = [flatten_record(row) for row in rows]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def teacher_distance_pairs(measurements):
    """Match teacher pairs at the same checkpoint and diagnostic prefix draw."""

    def identity(row):
        return (
            row["run"],
            row["step"],
            row.get("probe_batch", 0),
            *sorted((row["teacher_a"], row["teacher_b"])),
        )

    distances = {identity(row): row for row in measurements if row["kind"] == "teacher_distance"}
    pairs = []
    for row in measurements:
        if row["kind"] != "overlap" or identity(row) not in distances:
            continue
        distance = distances[identity(row)]
        pairs.append(
            {
                **row,
                "metrics": {
                    **row["metrics"],
                    "js": distance["metrics"]["js"],
                    "subnetwork_distance": (
                        1 - row["metrics"]["jaccard"] if row["metrics"]["jaccard"] is not None else None
                    ),
                },
            }
        )
    return pairs


def make_figures(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    output.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    capabilities = report["capabilities"]
    datasets = sorted({row["dataset"] for row in capabilities})
    if datasets:
        fig, axes = plt.subplots(len(datasets), 3, figsize=(12, 2.8 * len(datasets)), squeeze=False)
        for index, dataset in enumerate(datasets):
            for run in report["runs"]:
                rows = [r for r in capabilities if r["dataset"] == dataset and r["run"] == run["run"]]
                for axis, key, label in zip(
                    axes[index],
                    ("step", "valid_tokens", "gpu_hours"),
                    ("Optimizer steps", "Valid response tokens", "Training GPU hours"),
                    strict=True,
                ):
                    present = [row for row in rows if row[key] is not None]
                    axis.plot([r[key] for r in present], [100 * r["score"] for r in present], "o-", label=run["run"])
                    axis.set(xlabel=label, ylabel=f"{dataset} (%)")
            axes[index, 0].legend(fontsize=6)
        save(fig, "normalization_capability")

    traces = report["curves"]
    token_keys = sorted(
        {key for row in traces for key in row if key.startswith("mopd/task/") and key.endswith("/token_share")}
    )
    if token_keys:
        fig, axes = plt.subplots(len(token_keys), 1, figsize=(8, 2.3 * len(token_keys)), squeeze=False)
        for (axis,), key in zip(axes, token_keys, strict=True):
            for run in report["runs"]:
                rows = [row for row in traces if row["run"] == run["run"] and key in row]
                axis.plot([r["step"] for r in rows], [r[key] for r in rows], label=run["run"])
            axis.set(xlabel="Optimizer steps", ylabel=f"{key.split('/')[2]} token share")
            axis.legend(fontsize=6)
        save(fig, "normalization_token_shares")
    reward_keys = sorted(
        {key for row in traces for key in row if key.startswith("rollout/reward/") and key.endswith("/mean")}
    )
    if reward_keys:
        fig, axes = plt.subplots(len(reward_keys), 1, figsize=(8, 2.5 * len(reward_keys)), squeeze=False)
        for (axis,), key in zip(axes, reward_keys, strict=True):
            for run in report["runs"]:
                rows = [row for row in traces if row["run"] == run["run"] and key in row]
                axis.plot([r["step"] for r in rows], [r[key] for r in rows], label=run["run"])
            axis.set(xlabel="Optimizer steps", ylabel=key.removeprefix("rollout/reward/").removesuffix("/mean"))
            axis.legend(fontsize=6)
        save(fig, "rollout_reward")

    sparsity = [r for r in report["measurements"] if r["kind"] == "sparsity"]
    quantities = sorted({r["quantity"] for r in sparsity})
    if quantities:
        fig, axes = plt.subplots(len(quantities), 2, figsize=(10, 2.8 * len(quantities)), squeeze=False)
        for index, quantity in enumerate(quantities):
            groups = defaultdict(list)
            for row in sparsity:
                if row["quantity"] == quantity and row.get("layer", "all") == "all":
                    groups[
                        (row["run"], row.get("loss", ""), row.get("task", "joint"), row.get("branch", "online"))
                    ].append(row)
            for label, rows in groups.items():
                rows = sorted(rows, key=lambda row: row["step"])
                for axis, metric in zip(axes[index], ("energy90_fraction", "l2"), strict=True):
                    by_step = defaultdict(list)
                    for row in rows:
                        if row["metrics"][metric] is not None:
                            by_step[row["step"]].append(row["metrics"][metric])
                    axis.errorbar(
                        list(by_step),
                        [np.mean(values) for values in by_step.values()],
                        yerr=[np.std(values) for values in by_step.values()],
                        fmt="o-",
                        label=" / ".join(filter(None, label)),
                    )
                    axis.set(xlabel="Optimizer steps", ylabel=f"{quantity}: {metric}")
            axes[index, 0].legend(fontsize=5)
        save(fig, "update_sparsity")

    overlap_groups = defaultdict(list)
    for row in report["measurements"]:
        if (
            row["kind"] == "overlap"
            and row.get("quantity", "update") == "update"
            and row.get("support_fraction") == 0.05
            and row.get("layer", "all") == "all"
            and row.get("loss", "teacher_topk") in {"student_topk", "teacher_topk"}
        ):
            key = (
                row["run"],
                row["step"],
                row.get("quantity", "update"),
                row.get("support_fraction"),
                row.get("loss", ""),
                row.get("topk", ""),
                row.get("layer", "all"),
                row.get("prefix_set", "shared"),
            )
            overlap_groups[key].append(row)
    for index, (key, rows) in enumerate(overlap_groups.items()):
        teachers = sorted({r[k] for r in rows for k in ("teacher_a", "teacher_b")})
        matrix = np.full((len(teachers), len(teachers)), np.nan)
        np.fill_diagonal(matrix, 1)
        pair_values = defaultdict(list)
        for row in rows:
            a, b = teachers.index(row["teacher_a"]), teachers.index(row["teacher_b"])
            if row["metrics"]["jaccard"] is not None:
                pair_values[(a, b)].append(row["metrics"]["jaccard"])
        for (a, b), values in pair_values.items():
            matrix[a, b] = matrix[b, a] = np.mean(values)
        fig, axis = plt.subplots(figsize=(5, 4))
        shown = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set(
            xticks=range(len(teachers)),
            xticklabels=teachers,
            yticks=range(len(teachers)),
            yticklabels=teachers,
            title=" / ".join(map(str, key)),
        )
        fig.colorbar(shown, ax=axis, label="Support Jaccard (mean over prefix draws)")
        save(fig, f"teacher_overlap_{index:03d}")

    pairs = [
        row
        for row in report["teacher_distance_pairs"]
        if row.get("quantity", "update") == "update"
        and row.get("layer", "all") == "all"
        and row.get("support_fraction") == 0.05
        and row.get("loss", "teacher_topk") in {"student_topk", "teacher_topk"}
        and row["metrics"]["subnetwork_distance"] is not None
    ]
    if pairs:
        fig, axis = plt.subplots(figsize=(7, 4))
        for row in pairs:
            metrics = row["metrics"]
            axis.scatter(metrics["js"], metrics["subnetwork_distance"], s=20)
            axis.annotate(
                f"{row['run']}:{row['step']} {row['teacher_a']}/{row['teacher_b']} "
                f"{row.get('loss', 'teacher_topk')} K={row.get('topk', '')} {row.get('quantity', 'update')} {row.get('support_fraction', '')}",
                (metrics["js"], metrics["subnetwork_distance"]),
                fontsize=5,
            )
        axis.set(xlabel="Teacher JS on identical student prefixes (nats)", ylabel="Subnetwork distance (1 − Jaccard)")
        save(fig, "teacher_distance_subnetwork")
    return made


def analyze(root, output):
    def has_results(path):
        return any((path / name).is_dir() for name in ("metrics", "paper", "capability_eval"))

    paths = (
        [root] if has_results(root) else sorted(path for path in root.iterdir() if path.is_dir() and has_results(path))
    )
    if not paths:
        raise ValueError(f"No training metrics or paper measurements found in {root}")
    runs = [load_run(path) for path in paths]
    measurements = [row for run in runs for row in run["measurements"]]
    report = {
        "schema_version": 1,
        "study": "token_balancing_update_sparsity_supervision_density",
        "runs": runs,
        "curves": [row for run in runs for row in run["curves"]],
        "capabilities": [row for run in runs for row in run["capabilities"]],
        "measurements": measurements,
        "teacher_distance_pairs": teacher_distance_pairs(measurements),
        "interpretation": "Observed measurements only. Shared teachers and repeated probe batches are not independent training seeds. Full-vocabulary local probes are not online training trajectories.",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "training_curves.csv", report["curves"])
    write_csv(output / "capability.csv", report["capabilities"])
    write_csv(output / "measurements.csv", measurements)
    for kind in ("normalization", "sparsity", "overlap", "teacher_distance", "supervision"):
        write_csv(output / f"{kind}.csv", [row for row in measurements if row["kind"] == kind])
    write_csv(
        output / "supervision_comparison.csv",
        [row for row in measurements if row["kind"] in {"sparsity", "supervision"} and row.get("loss")],
    )
    write_csv(output / "teacher_distance_pairs.csv", report["teacher_distance_pairs"])
    report["figures"] = make_figures(report, output / "figures")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.output)
    print(
        json.dumps(
            {
                "runs": len(report["runs"]),
                "measurements": len(report["measurements"]),
                "figures": report["figures"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
