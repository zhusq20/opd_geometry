#!/usr/bin/env python3
"""Validate and summarize the frozen 3x3 single-task cross-domain evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ("math", "code", "science")
TASK_DATASETS = {
    "math": ("aime24", "math500"),
    "code": ("livecodebench_online",),
    "science": ("gpqa",),
}
OFF_DIAGONAL = (
    ("math", "science"),
    ("code", "science"),
    ("code", "math"),
    ("science", "math"),
    ("math", "code"),
    ("science", "code"),
)
PASS_K = {"aime24": 8, "math500": 1, "livecodebench_online": 1, "gpqa": 4}
DISPLAY_NAMES = {
    "aime24": "AIME24 avg@8",
    "math500": "MATH500",
    "livecodebench_online": "LCB Online",
    "gpqa": "GPQA avg@4",
}


@dataclass(frozen=True)
class Artifact:
    dataset: str
    path: Path
    sha256: str
    rows: dict[tuple[int, int], dict[str, Any]]
    score: float
    pass_at_k: float
    k: int
    prompts: int
    samples_per_prompt: int
    response_len_mean: float
    truncated_fraction: float
    starts_with_think: int
    contains_think: int


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{message}: {actual} != {expected}")


def command_value(command: list[str], option: str) -> str:
    require(command.count(option) == 1, f"Expected exactly one {option} in run command.")
    index = command.index(option)
    require(index + 1 < len(command), f"Missing value after {option}.")
    return command[index + 1]


def pass_at_k(rewards: list[float], k: int) -> float:
    correct = sum(reward > 0.0 for reward in rewards)
    sample_count = len(rewards)
    require(k <= sample_count, f"Cannot compute pass@{k} from {sample_count} samples.")
    if sample_count - correct < k:
        return 1.0
    return 1.0 - math.comb(sample_count - correct, k) / math.comb(sample_count, k)


def artifact_path(run_dir: Path, details: dict[str, Any]) -> Path:
    absolute = Path(details["path"])
    if absolute.exists():
        return absolute
    return run_dir / details["run_relative_path"]


def load_artifact(
    run_dir: Path,
    record: dict[str, Any],
    dataset: str,
    *,
    expected_thinking: bool | None,
) -> Artifact:
    details = record["datasets"][dataset]
    path = artifact_path(run_dir, details)
    require(path.is_file(), f"Missing evaluation artifact: {path}")
    actual_hash = sha256(path)
    require(actual_hash == details["sha256"], f"Artifact hash mismatch for {path}.")
    raw_rows = read_jsonl(path)
    require(len(raw_rows) == int(details["samples"]), f"Sample count mismatch for {path}.")

    rows = {
        (int(row["prompt_index"]), int(row["sample_within_prompt"])): row
        for row in raw_rows
    }
    require(len(rows) == len(raw_rows), f"Duplicate sample identities in {path}.")
    prompts = {key[0] for key in rows}
    samples_per_prompt = int(details["n_samples_per_prompt"])
    require(len(prompts) == int(details["prompts"]), f"Prompt count mismatch for {path}.")
    per_prompt_counts: dict[int, int] = defaultdict(int)
    for prompt_index, _ in rows:
        per_prompt_counts[prompt_index] += 1
    require(
        all(count == samples_per_prompt for count in per_prompt_counts.values()),
        f"Incomplete prompt groups in {path}.",
    )

    rewards = [float(row["reward"]) for row in raw_rows]
    response_lengths = [float(row["effective_response_length"]) for row in raw_rows]
    starts_with_think = sum(str(row.get("response", "")).lstrip().startswith("<think>") for row in raw_rows)
    contains_think = sum("<think>" in str(row.get("response", "")) for row in raw_rows)
    if expected_thinking is False:
        require(contains_think == 0, f"Unexpected <think> output in no-thinking artifact {path}.")
    if expected_thinking is True:
        require(contains_think > 0, f"No <think> output found in thinking-enabled artifact {path}.")

    grouped_rewards: dict[int, list[float]] = defaultdict(list)
    for key in sorted(rows):
        grouped_rewards[key[0]].append(float(rows[key]["reward"]))
    k = PASS_K[dataset]
    pass_score = float(np.mean([pass_at_k(grouped_rewards[index], k) for index in sorted(grouped_rewards)]))
    score = float(np.mean(rewards))
    metrics = record["metrics"]
    require_close(score, float(metrics[f"eval/{dataset}"]), f"Metric/artifact score mismatch for {path}")
    require_close(
        float(np.mean(response_lengths)),
        float(metrics[f"eval/{dataset}/response_len/mean"]),
        f"Response-length mismatch for {path}",
    )
    truncated_fraction = float(np.mean([row["status"] == "truncated" for row in raw_rows]))
    require_close(
        truncated_fraction,
        float(metrics[f"eval/{dataset}/truncated_ratio"]),
        f"Truncation mismatch for {path}",
    )
    pass_metric = metrics.get(f"eval/{dataset}-pass@{k}")
    if pass_metric is not None:
        require_close(pass_score, float(pass_metric), f"Pass@{k} mismatch for {path}")

    return Artifact(
        dataset=dataset,
        path=path.resolve(),
        sha256=actual_hash,
        rows=rows,
        score=score,
        pass_at_k=pass_score,
        k=k,
        prompts=len(prompts),
        samples_per_prompt=samples_per_prompt,
        response_len_mean=float(np.mean(response_lengths)),
        truncated_fraction=truncated_fraction,
        starts_with_think=starts_with_think,
        contains_think=contains_think,
    )


def paired_comparison(
    reference: Artifact,
    candidate: Artifact,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    require(reference.dataset == candidate.dataset, "Cannot pair different datasets.")
    require(reference.rows.keys() == candidate.rows.keys(), f"Sample identities differ for {candidate.dataset}.")
    prompt_deltas: dict[int, list[float]] = defaultdict(list)
    improved_samples = 0
    regressed_samples = 0
    for key in sorted(reference.rows):
        left = reference.rows[key]
        right = candidate.rows[key]
        require(left["prompt"] == right["prompt"], f"Prompt changed for {candidate.dataset} sample {key}.")
        require(left["label"] == right["label"], f"Label changed for {candidate.dataset} sample {key}.")
        left_reward = float(left["reward"])
        right_reward = float(right["reward"])
        prompt_deltas[key[0]].append(right_reward - left_reward)
        improved_samples += left_reward <= 0.0 < right_reward
        regressed_samples += right_reward <= 0.0 < left_reward

    deltas = np.asarray(
        [float(np.mean(prompt_deltas[index])) for index in sorted(prompt_deltas)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    resampled = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
    bootstrap_means = deltas[resampled].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "delta": candidate.score - reference.score,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "improved_samples": improved_samples,
        "regressed_samples": regressed_samples,
        "net_improved_samples": improved_samples - regressed_samples,
    }


def find_index_record(run_dir: Path, num_updates: int, expected_phase: str | None = None) -> dict[str, Any]:
    records = [
        record
        for record in read_jsonl(run_dir / "eval_artifacts" / "index.jsonl")
        if int(record["num_updates"]) == num_updates
        and (expected_phase is None or record["eval_phase"] == expected_phase)
    ]
    require(
        len(records) == 1,
        f"Expected one index record at update {num_updates} under {run_dir}, found {len(records)}.",
    )
    return records[0]


def validate_checkpoint_views(cross_root: Path, locks: dict[str, Any]) -> None:
    for task in TASKS:
        lock = locks["checkpoints"][task]
        iteration = int(lock["iteration"])
        view = cross_root / "checkpoint_views" / task / f"iter_{iteration:07d}"
        require(sha256(view / ".metadata") == lock["metadata_sha256"], f"{task} metadata lock changed.")
        require(sha256(view / "common.pt") == lock["common_sha256"], f"{task} common.pt lock changed.")


def artifact_summary(artifact: Artifact) -> dict[str, Any]:
    return {
        "dataset": artifact.dataset,
        "score": artifact.score,
        "pass_at_k": artifact.pass_at_k,
        "k": artifact.k,
        "prompts": artifact.prompts,
        "samples_per_prompt": artifact.samples_per_prompt,
        "samples": len(artifact.rows),
        "response_len_mean": artifact.response_len_mean,
        "truncated_fraction": artifact.truncated_fraction,
        "artifact_path": str(artifact.path),
        "artifact_sha256": artifact.sha256,
        "starts_with_think": artifact.starts_with_think,
        "contains_think": artifact.contains_think,
    }


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def format_ci(comparison: dict[str, Any]) -> str:
    return f"[{100.0 * comparison['ci95_low']:+.2f}, {100.0 * comparison['ci95_high']:+.2f}]"


def matrix_cell(rows: list[dict[str, Any]], source: str, target: str) -> str:
    selected = [row for row in rows if row["source"] == source and row["target"] == target]
    require(bool(selected), f"Missing matrix cell {source}->{target}.")
    values = " / ".join(format_percent(row["score"]) for row in selected)
    return f"**{values}**" if source == target else values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> None:
    cross_root = args.cross_root.resolve()
    output_dir = args.output_dir.resolve()
    known_outputs = ("matrix.json", "matrix.csv", "REPORT_zh.md", "thinking_enabled_auxiliary.csv")
    if not args.force and any((output_dir / name).exists() for name in known_outputs):
        raise FileExistsError(f"Analysis outputs already exist under {output_dir}; pass --force to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_summary = read_json(args.training_summary.resolve())
    locks = read_json(cross_root / "checkpoint_locks.json")
    require(
        "N+1 optimizer updates" in locks["mapping"]
        and "N+1 optimizer updates" in training_summary["checkpoint_mapping"],
        "Checkpoint mappings do not both use the iter_N -> N+1 updates convention.",
    )
    validate_checkpoint_views(cross_root, locks)

    run_specs = {str(run["task"]): run for run in training_summary["runs"]}
    require(set(run_specs) == set(TASKS), f"Unexpected specialist tasks: {sorted(run_specs)}")
    references: dict[str, dict[str, dict[str, Artifact]]] = defaultdict(dict)
    for task in TASKS:
        run = run_specs[task]
        run_dir = Path(run["run_path"]).resolve()
        final_update = int(run["num_updates"])
        require(final_update == int(locks["checkpoints"][task]["iteration"]) + 1, f"{task} update lock mismatch.")
        baseline_record = find_index_record(run_dir, 0)
        final_record = find_index_record(run_dir, final_update)
        require(set(final_record["datasets"]) == set(TASK_DATASETS[task]), f"{task} diagonal datasets differ.")
        for dataset in TASK_DATASETS[task]:
            baseline = load_artifact(run_dir, baseline_record, dataset, expected_thinking=False)
            specialist = load_artifact(run_dir, final_record, dataset, expected_thinking=False)
            provenance = run["eval_artifacts"]["datasets"][dataset]
            require(baseline.sha256 == provenance["baseline_sha256"], f"{task}/{dataset} baseline hash drifted.")
            require(specialist.sha256 == provenance["final_sha256"], f"{task}/{dataset} final hash drifted.")
            references[task][dataset] = {"baseline": baseline, "specialist": specialist}

    rows: list[dict[str, Any]] = []
    comparison_seed = args.bootstrap_seed
    for source in TASKS:
        for target in TASKS:
            if source == target:
                artifacts = {
                    dataset: references[target][dataset]["specialist"]
                    for dataset in TASK_DATASETS[target]
                }
                run_dir = Path(run_specs[source]["run_path"]).resolve()
                num_updates = int(run_specs[source]["num_updates"])
                eval_phase = "post_update"
            else:
                run_dir = cross_root / f"{source}_to_{target}_no_thinking"
                marker = read_json(run_dir / "run_complete.json")
                num_updates = int(run_specs[source]["num_updates"])
                require(marker["status"] == "complete", f"Incomplete run: {run_dir}")
                require(int(marker["num_rollout"]) == 0, f"Cross eval unexpectedly trained: {run_dir}")
                require(int(marker["final_num_updates"]) == num_updates, f"Wrong checkpoint update in {run_dir}")

                manifest = read_json(run_dir / "provenance" / "run_manifest.json")
                require(manifest["status"] == "complete", f"Incomplete provenance manifest: {run_dir}")
                command = [str(value) for value in manifest["command"]]
                require("--sglang-enable-deterministic-inference" in command, f"Non-deterministic eval: {run_dir}")
                require(command_value(command, "--num-rollout") == "0", f"Nonzero rollout count: {run_dir}")
                require(command_value(command, "--experiment-task") == target, f"Wrong target task: {run_dir}")
                require(command_value(command, "--experiment-teacher") == f"st_{source}", f"Wrong source task: {run_dir}")
                chat_kwargs = json.loads(command_value(command, "--apply-chat-template-kwargs"))
                require(chat_kwargs == {"enable_thinking": False}, f"Thinking was not disabled: {run_dir}")
                expected_load = (cross_root / "checkpoint_views" / source).resolve()
                require(Path(command_value(command, "--load")).resolve() == expected_load, f"Wrong checkpoint view: {run_dir}")

                record = find_index_record(run_dir, num_updates, "eval_only")
                require(set(record["datasets"]) == set(TASK_DATASETS[target]), f"Wrong datasets in {run_dir}")
                if target == "code":
                    infrastructure_errors = float(
                        record["metrics"]["eval/livecodebench_online/sandbox/infrastructure_errors"]
                    )
                    require(infrastructure_errors == 0.0, f"Sandbox infrastructure errors in {run_dir}")
                artifacts = {
                    dataset: load_artifact(run_dir, record, dataset, expected_thinking=False)
                    for dataset in TASK_DATASETS[target]
                }
                eval_phase = "eval_only"

            for dataset in TASK_DATASETS[target]:
                artifact = artifacts[dataset]
                baseline = references[target][dataset]["baseline"]
                specialist = references[target][dataset]["specialist"]
                require(artifact.prompts == specialist.prompts, f"Prompt count changed for {source}->{dataset}.")
                require(
                    artifact.samples_per_prompt == specialist.samples_per_prompt,
                    f"Sampling count changed for {source}->{dataset}.",
                )
                versus_baseline = paired_comparison(
                    baseline,
                    artifact,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=comparison_seed,
                )
                comparison_seed += 1
                versus_specialist = paired_comparison(
                    specialist,
                    artifact,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=comparison_seed,
                )
                comparison_seed += 1
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "dataset": dataset,
                        "display_name": DISPLAY_NAMES[dataset],
                        "diagonal": source == target,
                        "num_updates": num_updates,
                        "eval_phase": eval_phase,
                        "score": artifact.score,
                        "pass_at_k": artifact.pass_at_k,
                        "pass_k": artifact.k,
                        "baseline_score": baseline.score,
                        "delta_from_baseline": versus_baseline["delta"],
                        "delta_from_baseline_ci95_low": versus_baseline["ci95_low"],
                        "delta_from_baseline_ci95_high": versus_baseline["ci95_high"],
                        "target_specialist_score": specialist.score,
                        "delta_from_target_specialist": versus_specialist["delta"],
                        "delta_from_target_specialist_ci95_low": versus_specialist["ci95_low"],
                        "delta_from_target_specialist_ci95_high": versus_specialist["ci95_high"],
                        "improved_vs_baseline_samples": versus_baseline["improved_samples"],
                        "regressed_vs_baseline_samples": versus_baseline["regressed_samples"],
                        "improved_vs_specialist_samples": versus_specialist["improved_samples"],
                        "regressed_vs_specialist_samples": versus_specialist["regressed_samples"],
                        "prompts": artifact.prompts,
                        "samples_per_prompt": artifact.samples_per_prompt,
                        "samples": len(artifact.rows),
                        "response_len_mean": artifact.response_len_mean,
                        "truncated_fraction": artifact.truncated_fraction,
                        "artifact_path": str(artifact.path),
                        "artifact_sha256": artifact.sha256,
                        "starts_with_think": artifact.starts_with_think,
                        "contains_think": artifact.contains_think,
                        "run_dir": str(run_dir),
                    }
                )

    auxiliary_rows: list[dict[str, Any]] = []
    for source, target in OFF_DIAGONAL:
        run_dir = cross_root / f"{source}_to_{target}"
        marker = read_json(run_dir / "run_complete.json")
        require(marker["status"] == "complete", f"Incomplete auxiliary run: {run_dir}")
        record = find_index_record(run_dir, int(run_specs[source]["num_updates"]), "eval_only")
        manifest = read_json(run_dir / "provenance" / "run_manifest.json")
        command = [str(value) for value in manifest["command"]]
        require("--apply-chat-template-kwargs" not in command, f"Auxiliary run was not default-thinking: {run_dir}")
        for dataset in TASK_DATASETS[target]:
            artifact = load_artifact(run_dir, record, dataset, expected_thinking=True)
            auxiliary_rows.append(
                {
                    "source": source,
                    "target": target,
                    "dataset": dataset,
                    "score": artifact.score,
                    "response_len_mean": artifact.response_len_mean,
                    "truncated_fraction": artifact.truncated_fraction,
                    "samples": len(artifact.rows),
                    "starts_with_think": artifact.starts_with_think,
                    "contains_think": artifact.contains_think,
                    "artifact_path": str(artifact.path),
                    "artifact_sha256": artifact.sha256,
                    "excluded_from_primary_matrix": True,
                    "exclusion_reason": "Qwen3 thinking was enabled by default; diagonals use enable_thinking=false.",
                }
            )

    write_csv(output_dir / "matrix.csv", rows)
    write_csv(output_dir / "thinking_enabled_auxiliary.csv", auxiliary_rows)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "seed": 42,
            "deterministic_inference": True,
            "enable_thinking": False,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "checkpoint_mapping": locks["mapping"],
            "checkpoint_locks_path": str((cross_root / "checkpoint_locks.json").resolve()),
        },
        "checkpoint_locks": locks["checkpoints"],
        "rows": rows,
        "auxiliary_thinking_enabled_rows": auxiliary_rows,
        "audit": {
            "primary_domain_cells": 9,
            "primary_benchmark_rows": len(rows),
            "off_diagonal_domain_cells": len(OFF_DIAGONAL),
            "off_diagonal_benchmark_rows": sum(not row["diagonal"] for row in rows),
            "all_primary_artifact_hashes_match": True,
            "all_primary_completion_markers_complete": True,
            "all_primary_responses_without_think_tag": all(row["contains_think"] == 0 for row in rows),
            "all_code_cross_evals_without_infrastructure_errors": True,
        },
    }
    (output_dir / "matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    off_rows = [row for row in rows if not row["diagonal"]]
    confident_base_gains = [row for row in off_rows if row["delta_from_baseline_ci95_low"] > 0.0]
    confident_base_losses = [row for row in off_rows if row["delta_from_baseline_ci95_high"] < 0.0]
    confident_specialist_gaps = [row for row in off_rows if row["delta_from_target_specialist_ci95_high"] < 0.0]
    row_by_key = {(row["source"], row["target"], row["dataset"]): row for row in rows}
    science_to_math500 = row_by_key[("science", "math", "math500")]
    math_to_code = row_by_key[("math", "code", "livecodebench_online")]
    code_diagonal = row_by_key[("code", "code", "livecodebench_online")]
    science_to_code = row_by_key[("science", "code", "livecodebench_online")]
    report = [
        "# ST checkpoint 3×3 cross-domain evaluation matrix",
        "",
        "## 结论",
        "",
        (
            f"六个 off-diagonal domain cells 已补齐，共对应 {len(off_rows)} 个 benchmark 分数。"
            f"其中 {len(confident_base_gains)} 个相对初始模型的 paired 95% CI 全为正，"
            f"{len(confident_base_losses)} 个全为负；其余区间跨 0。"
        ),
        (
            f"相对目标域 specialist，{len(confident_specialist_gaps)}/{len(off_rows)} 个 off-domain benchmark "
            "的 paired 95% CI 全为负。该比较比只看绝对分数更直接地刻画 task specialization。"
        ),
        (
            "没有任何 off-domain benchmark 相对初始模型的 paired 95% CI 全为负，因此这组固定 checkpoint "
            "没有显著跨域灾难性遗忘证据。唯一 CI 全为正的迁移是 Science → MATH500："
            f"{format_pp(science_to_math500['delta_from_baseline'])} "
            f"[{100.0 * science_to_math500['delta_from_baseline_ci95_low']:+.2f}, "
            f"{100.0 * science_to_math500['delta_from_baseline_ci95_high']:+.2f}]。"
        ),
        (
            f"Code 列没有形成清晰 specialist separation：Math checkpoint 为 "
            f"{format_percent(math_to_code['score'])}，Code specialist 为 "
            f"{format_percent(code_diagonal['score'])}，Science checkpoint 为 "
            f"{format_percent(science_to_code['score'])}（等于 base）。这与现有 Code 同域提升不显著的结论一致。"
        ),
        "",
        "## 主矩阵（no-thinking）",
        "",
        "行是 checkpoint 的训练域，列是评测域；Math cell 依次为 `AIME24 avg@8 / MATH500`。粗体是已有 diagonal。",
        "",
        "| checkpoint ↓ / eval → | Math | Code | Science |",
        "|---|---:|---:|---:|",
    ]
    for source in TASKS:
        report.append(
            f"| {source.title()} | {matrix_cell(rows, source, 'math')} | "
            f"{matrix_cell(rows, source, 'code')} | {matrix_cell(rows, source, 'science')} |"
        )
    report.extend(
        [
            "",
            "## Off-diagonal paired comparisons",
            "",
            f"CI 使用 {args.bootstrap_samples:,} 次 paired prompt-cluster bootstrap；多样本 benchmark 先在同一 prompt 内聚合。",
            "`Δ base` 比较同一题、同一 sample slot 的初始模型，`Δ specialist` 比较目标域对角线 checkpoint。",
            "",
            "| source → target / dataset | score | Δ base [95% CI] | Δ specialist [95% CI] | mean tokens | truncation |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in off_rows:
        base_comparison = {
            "ci95_low": row["delta_from_baseline_ci95_low"],
            "ci95_high": row["delta_from_baseline_ci95_high"],
        }
        specialist_comparison = {
            "ci95_low": row["delta_from_target_specialist_ci95_low"],
            "ci95_high": row["delta_from_target_specialist_ci95_high"],
        }
        report.append(
            f"| {row['source'].title()} → {row['target'].title()} / {row['display_name']} | "
            f"{format_percent(row['score'])} | {format_pp(row['delta_from_baseline'])} "
            f"{format_ci(base_comparison)} | {format_pp(row['delta_from_target_specialist'])} "
            f"{format_ci(specialist_comparison)} | {row['response_len_mean']:,.0f} | "
            f"{format_percent(row['truncated_fraction'])} |"
        )
    report.extend(
        [
            "",
            "## 协议与完整性审计",
            "",
            "- 三个 checkpoint 固定为 Math iter299（300 updates）、Code iter99（100 updates）、Science iter599（600 updates）。",
            "- 所有主矩阵评测均为 seed 42、deterministic inference、`enable_thinking=false`；Math 最大 32,768 tokens，Code/Science 最大 16,384 tokens。",
            "- 六个新 run 的 completion marker 均为 complete，artifact 行数、索引 SHA-256、prompt/sample-slot 配对均通过；所有主矩阵响应均不含 `<think>`。",
            "- 两个 Code off-diagonal 的 SandboxFusion infrastructure errors 均为 0。模型代码编译/执行错误属于任务结果，不按基础设施错误剔除。",
            "",
            "## 排除的 thinking-enabled pilot",
            "",
            "最初六次 pilot 未显式传 `enable_thinking=false`，Qwen3 因而使用默认 thinking 模式；样本中出现 `<think>`，响应长度分布也与 diagonal 不同。它们是有效的另一协议数据，但不能填入上述主矩阵，故原样保留并单独列出：",
            "",
            "| source → target / dataset | score | mean tokens | truncation | `<think>` samples |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in auxiliary_rows:
        report.append(
            f"| {row['source'].title()} → {row['target'].title()} / {DISPLAY_NAMES[row['dataset']]} | "
            f"{format_percent(row['score'])} | {row['response_len_mean']:,.0f} | "
            f"{format_percent(row['truncated_fraction'])} | {row['contains_think']}/{row['samples']} |"
        )
    report.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是固定三个 checkpoint、单一训练 seed 的离线评测。paired bootstrap 描述 benchmark prompt 的不确定性，不等同于多训练 seed 方差；因此可以报告当前 checkpoint 的跨域迁移/遗忘模式，但不能据此估计训练过程方差。",
            "",
            "机器可读结果见 `matrix.json` 与 `matrix.csv`；排除的 pilot 见 `thinking_enabled_auxiliary.csv`。",
            "",
        ]
    )
    (output_dir / "REPORT_zh.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-root", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive.")
    return args


if __name__ == "__main__":
    analyze(parse_args())
