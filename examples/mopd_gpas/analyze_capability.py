#!/usr/bin/env python3
"""Summarize the four preregistered final capability scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SEED = 42
CONFIGS = (
    "uniform",
    "gpas",
    "cost_gpas",
    "raw_noise",
    "loss_gap",
    "std_mopd",
    "d3_mopd",
    "open_mopd",
)
NON_FIXED_OBJECTIVE = {"std_mopd", "d3_mopd", "open_mopd"}
REFERENCES = ("initial_student", "teacher_math", "teacher_if", "teacher_qwen3_4b")
TEACHER_BY_DOMAIN = {
    "math": "teacher_math",
    "code": "teacher_qwen3_4b",
    "if": "teacher_if",
    "science": "teacher_qwen3_4b",
}
DOMAINS = {
    "math": ("math500_pass1", 1, 500),
    "code": ("livecodebench_postcutoff_pass1", 1, 128),
    "if": ("ifbench_strict", 1, 300),
    "science": ("gpqa_diamond_avg4", 4, 198),
}
TABLE_TARGETS = REFERENCES + (
    "std_mopd",
    "uniform",
    "raw_noise",
    "loss_gap",
    "gpas",
    "cost_gpas",
    "d3_mopd",
    "open_mopd",
)


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(path: Path, domains: dict[str, tuple[str, int, int]]) -> dict:
    completion = json.loads((path / "run_complete.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError(f"Capability evaluation is incomplete: {path}")
    indices = jsonl(path / "eval_artifacts/index.jsonl")
    if len(indices) != 1:
        raise ValueError(f"Expected exactly one final capability evaluation artifact: {path}")
    index = indices[0]
    scores = {}
    for domain, (dataset, expected_samples_per_prompt, expected_prompts) in domains.items():
        artifact = index["datasets"][dataset]
        artifact_path = Path(artifact["path"])
        if sha256(artifact_path) != artifact["sha256"]:
            raise ValueError(f"Capability artifact hash mismatch: {artifact_path}")
        samples_per_prompt = int(artifact["n_samples_per_prompt"])
        if samples_per_prompt != expected_samples_per_prompt:
            raise ValueError(
                f"Capability dataset {dataset} uses {samples_per_prompt} samples/prompt; "
                f"expected {expected_samples_per_prompt}."
            )
        records = jsonl(artifact_path)
        if len(records) != int(artifact["samples"]):
            raise ValueError(f"Capability artifact sample count mismatch: {artifact_path}")
        groups: dict[int, list[float]] = {}
        for record in records:
            groups.setdefault(int(record["prompt_index"]), []).append(float(record["reward"]))
        if not groups:
            raise ValueError(f"Empty capability dataset {dataset}: {path}")
        if (
            len(groups) != expected_prompts
            or int(artifact["prompts"]) != expected_prompts
            or any(len(values) != samples_per_prompt for values in groups.values())
        ):
            raise ValueError(
                f"Capability prompt grouping mismatch for {dataset}: got {len(groups)}, "
                f"expected {expected_prompts}: {artifact_path}"
            )
        prompt_indices = sorted(groups)
        prompt_scores = [float(np.mean(groups[index])) for index in prompt_indices]
        scores[domain] = {
            "score": float(np.mean(prompt_scores)),
            "prompts": len(prompt_scores),
            "samples": len(records),
            "samples_per_prompt": samples_per_prompt,
            "prompt_indices": prompt_indices,
            "prompt_scores": prompt_scores,
        }
    return {"domains": scores}


def paired_bootstrap(
    configs: dict[str, dict], references: dict[str, dict] | None = None, replicates: int = 1_000
) -> dict:
    baseline = CONFIGS[0]
    rng = np.random.default_rng(SEED)
    targets = dict(configs)
    if references is not None:
        targets.update(references)
    bootstrap_scores: dict[str, dict[str, np.ndarray]] = {target: {} for target in targets}
    for domain in DOMAINS:
        reference_indices = configs[baseline]["domains"][domain]["prompt_indices"]
        count = len(reference_indices)
        resamples = rng.integers(0, count, size=(replicates, count))
        for target in targets:
            values = targets[target]["domains"][domain]
            if values["prompt_indices"] != reference_indices:
                raise ValueError(f"capability prompts are not paired for {domain}: {target}")
            prompt_scores = np.asarray(values["prompt_scores"], dtype=np.float64)
            bootstrap_scores[target][domain] = prompt_scores[resamples].mean(axis=1)

    macro = {config: sum(bootstrap_scores[config].values()) / len(DOMAINS) for config in CONFIGS}
    result = {"replicates": replicates, "seed": SEED, "baseline": baseline, "configs": {}}
    for config in CONFIGS:
        macro_delta = macro[config] - macro[baseline]
        result["configs"][config] = {
            "macro_score_95ci": list(map(float, np.percentile(macro[config], [2.5, 97.5]))),
            "paired_macro_delta_vs_baseline": float(np.mean(macro_delta)),
            "paired_macro_delta_95ci": list(map(float, np.percentile(macro_delta, [2.5, 97.5]))),
            "domains": {
                domain: {
                    "score_95ci": list(map(float, np.percentile(bootstrap_scores[config][domain], [2.5, 97.5]))),
                    "paired_delta_vs_baseline": float(
                        np.mean(bootstrap_scores[config][domain] - bootstrap_scores[baseline][domain])
                    ),
                    "paired_delta_95ci": list(
                        map(
                            float,
                            np.percentile(
                                bootstrap_scores[config][domain] - bootstrap_scores[baseline][domain],
                                [2.5, 97.5],
                            ),
                        )
                    ),
                }
                for domain in DOMAINS
            },
        }
    if references is not None:
        normalized: dict[str, dict[str, np.ndarray]] = {config: {} for config in CONFIGS}
        for domain in DOMAINS:
            initial = references["initial_student"]["domains"][domain]["score"]
            teacher = references[TEACHER_BY_DOMAIN[domain]]["domains"][domain]["score"]
            headroom = float(teacher - initial)
            if headroom == 0:
                raise ValueError(f"zero teacher/student capability headroom for {domain}")
            initial_bootstrap = bootstrap_scores["initial_student"][domain]
            for config in CONFIGS:
                normalized[config][domain] = (bootstrap_scores[config][domain] - initial_bootstrap) / headroom
        normalized_macro = {config: sum(normalized[config].values()) / len(DOMAINS) for config in CONFIGS}
        for config in CONFIGS:
            baseline_delta = normalized_macro[config] - normalized_macro[baseline]
            result["configs"][config]["normalized_gain_95ci"] = list(
                map(float, np.percentile(normalized_macro[config], [2.5, 97.5]))
            )
            result["configs"][config]["paired_normalized_delta_vs_baseline"] = float(np.mean(baseline_delta))
            result["configs"][config]["paired_normalized_delta_95ci"] = list(
                map(float, np.percentile(baseline_delta, [2.5, 97.5]))
            )
    return result


def _write_main_table(path: Path, report: dict[str, Any], mopd: dict[str, Any] | None) -> None:
    fields = [
        "target",
        "kind",
        *(f"score_{domain}" for domain in DOMAINS),
        *(f"normalized_gain_{domain}" for domain in DOMAINS),
        "normalized_gain_mean",
        *(f"score_delta_vs_uniform_{domain}" for domain in DOMAINS),
        "score_delta_vs_uniform_mean",
        *(f"normalized_delta_vs_uniform_{domain}" for domain in DOMAINS),
        "normalized_delta_vs_uniform_mean",
        "final_heldout_F",
        "gpu_hours_to_uniform_final",
    ]
    uniform = report["configs"]["uniform"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for target in TABLE_TARGETS:
            kind = "reference" if target in REFERENCES else "config"
            values = report["references" if kind == "reference" else "configs"][target]
            scores = {domain: float(values["domains"][domain]["score"]) for domain in DOMAINS}
            normalized = {
                domain: (scores[domain] - float(report["headroom"][domain]["initial_score"]))
                / float(report["headroom"][domain]["difference"])
                for domain in DOMAINS
            }
            row: dict[str, Any] = {
                "target": target,
                "kind": kind,
                **{f"score_{domain}": scores[domain] for domain in DOMAINS},
                **{f"normalized_gain_{domain}": normalized[domain] for domain in DOMAINS},
                "normalized_gain_mean": float(np.mean(list(normalized.values()))),
            }
            if kind == "config":
                score_deltas = {
                    domain: scores[domain] - float(uniform["domains"][domain]["score"]) for domain in DOMAINS
                }
                normalized_deltas = {
                    domain: normalized[domain] - float(uniform["domains"][domain]["normalized_gain"])
                    for domain in DOMAINS
                }
                row.update(
                    {
                        **{f"score_delta_vs_uniform_{domain}": score_deltas[domain] for domain in DOMAINS},
                        "score_delta_vs_uniform_mean": float(np.mean(list(score_deltas.values()))),
                        **{f"normalized_delta_vs_uniform_{domain}": normalized_deltas[domain] for domain in DOMAINS},
                        "normalized_delta_vs_uniform_mean": float(np.mean(list(normalized_deltas.values()))),
                    }
                )
                if target not in NON_FIXED_OBJECTIVE and mopd is not None:
                    outcome = mopd["outcomes"][target]
                    row["final_heldout_F"] = float(outcome["weighted_loss"])
                    row["gpu_hours_to_uniform_final"] = outcome["gpu_hours_to_uniform_final"]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mopd-report", type=Path)
    parser.add_argument("--response-budget", type=int, default=32_000)
    args = parser.parse_args()
    configs = {
        config: evaluate(
            args.root / f"{config}-seed{SEED}/capability_eval/response_{args.response_budget}",
            DOMAINS,
        )
        for config in CONFIGS
    }
    references = {
        reference: evaluate(args.root / "capability_references" / reference, DOMAINS) for reference in REFERENCES
    }
    for values in (*configs.values(), *references.values()):
        values["macro_score"] = float(np.mean([values["domains"][domain]["score"] for domain in DOMAINS]))
    headroom = {}
    for domain in DOMAINS:
        initial = references["initial_student"]["domains"][domain]["score"]
        teacher_name = TEACHER_BY_DOMAIN[domain]
        teacher = references[teacher_name]["domains"][domain]["score"]
        headroom[domain] = {
            "initial_score": initial,
            "teacher": teacher_name,
            "teacher_score": teacher,
            "difference": teacher - initial,
            "normalization_reliable": teacher - initial >= 0.03,
        }
    for config, values in configs.items():
        for domain in DOMAINS:
            denominator = headroom[domain]["difference"]
            if denominator == 0:
                raise ValueError(f"zero teacher/student capability headroom for {domain}")
            values["domains"][domain]["normalized_gain"] = (
                values["domains"][domain]["score"] - headroom[domain]["initial_score"]
            ) / denominator
        values["normalized_gain_mean"] = float(
            np.mean([values["domains"][domain]["normalized_gain"] for domain in DOMAINS])
        )
    result = {
        "schema_version": 3,
        "objective": "mopd",
        "seed": SEED,
        "attempted_responses": args.response_budget,
        "references": references,
        "teacher_by_domain": TEACHER_BY_DOMAIN,
        "headroom": headroom,
        "configs": configs,
        "paired_bootstrap": paired_bootstrap(configs, references),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = args.output.with_name("mopd_main_table.csv")
    mopd = None if args.mopd_report is None else json.loads(args.mopd_report.read_text())
    _write_main_table(table, result, mopd)
    result["main_table_csv"] = str(table.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Capability report written to {args.output}")


if __name__ == "__main__":
    main()
