#!/usr/bin/env python3
"""Summarize the four preregistered final capability scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 42
CONFIGS = (
    "uniform_k1_conventional",
    "uniform_k1_taskwise",
    "gpas_k1_taskwise",
    "cost_gpas_k1_taskwise",
    "uniform_k2_taskwise",
    "cost_gpas_k2_taskwise",
    "all_k4_taskwise",
    "all_k4_conventional",
)
DOMAINS = {
    "math": ("math500_pass1", 1, 500),
    "code": ("livecodebench_postcutoff_pass1", 1, 128),
    "if": ("ifbench_strict", 1, 300),
    "science": ("gpqa_diamond_avg4", 4, 198),
}


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


def paired_bootstrap(configs: dict[str, dict], replicates: int = 10_000) -> dict:
    baseline = CONFIGS[0]
    rng = np.random.default_rng(SEED)
    bootstrap_scores: dict[str, dict[str, np.ndarray]] = {config: {} for config in CONFIGS}
    for domain in DOMAINS:
        reference_indices = configs[baseline]["domains"][domain]["prompt_indices"]
        count = len(reference_indices)
        resamples = rng.integers(0, count, size=(replicates, count))
        for config in CONFIGS:
            values = configs[config]["domains"][domain]
            if values["prompt_indices"] != reference_indices:
                raise ValueError(f"capability prompts are not paired for {domain}: {config}")
            prompt_scores = np.asarray(values["prompt_scores"], dtype=np.float64)
            bootstrap_scores[config][domain] = prompt_scores[resamples].mean(axis=1)

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
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-budget", type=int, default=64_000)
    args = parser.parse_args()
    configs = {
        config: evaluate(
            args.root / f"{config}-seed{SEED}/capability_eval/response_{args.response_budget}",
            DOMAINS,
        )
        for config in CONFIGS
    }
    for values in configs.values():
        values["macro_score"] = float(np.mean([values["domains"][domain]["score"] for domain in DOMAINS]))
    result = {
        "schema_version": 2,
        "objective": "mopd",
        "seed": SEED,
        "attempted_responses": args.response_budget,
        "configs": configs,
        "paired_bootstrap": paired_bootstrap(configs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Capability report written to {args.output}")


if __name__ == "__main__":
    main()
