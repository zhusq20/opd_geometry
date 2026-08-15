#!/usr/bin/env python3
"""Validate an optimizer × post-training algorithm experiment before Ray starts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import yaml


OPTIMIZERS = {"adamw": "adam", "adam": "adam", "sgd": "sgd", "muon": "muon", "dist_muon": "dist_muon"}
ALGORITHMS = {"grpo", "ppo", "opd", "sft_opd", "grpo_opd", "ppo_opd"}


def mapping_file(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        text = os.path.expandvars(stream.read())
    unresolved = sorted(set(re.findall(r"\$\{([^}]+)\}", text)))
    if unresolved:
        raise ValueError(f"Unresolved environment variables in {path}: {', '.join(unresolved)}")
    value = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return value


def validate_url(value: str, label: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL, got {value!r}.")


def validate_eval_config(
    path: Path | None,
    reward_config: dict,
    *,
    check_runtime_deps: bool,
    expected_max_response_len: int | None = None,
) -> list[str]:
    """Validate held-out data and code-execution dependencies before model startup."""

    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {path}")
    root = mapping_file(path).get("eval")
    if not isinstance(root, dict):
        raise ValueError(f"{path} requires a top-level `eval` mapping.")
    defaults = root.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("eval.defaults must be a mapping.")
    raw_datasets = root.get("datasets")
    if isinstance(raw_datasets, dict):
        datasets = []
        for name, value in raw_datasets.items():
            entry = dict(value or {})
            entry.setdefault("name", name)
            datasets.append(entry)
    elif isinstance(raw_datasets, list):
        datasets = raw_datasets
    else:
        raise ValueError("eval.datasets must be a non-empty list or mapping.")
    if not datasets:
        raise ValueError("eval.datasets must not be empty.")

    names: list[str] = []
    routes = reward_config.get("routes") or {}
    for raw in datasets:
        if not isinstance(raw, dict):
            raise ValueError("Every eval dataset must be a mapping.")
        entry = {**defaults, **raw}
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError("Every eval dataset needs a non-empty name.")
        names.append(name)
        data_value = entry.get("path")
        if not data_value:
            raise ValueError(f"Evaluation dataset {name!r} is missing path.")
        data_path = Path(os.path.expandvars(str(data_value))).expanduser()
        if not data_path.is_absolute():
            data_path = path.parent / data_path
        if not data_path.is_file():
            raise FileNotFoundError(f"Evaluation dataset {name!r} does not exist: {data_path}")

        n = int(entry.get("n_samples_per_eval_prompt", 1))
        max_response_len = int(entry.get("max_response_len", 1))
        temperature = float(entry.get("temperature", 0))
        top_p = float(entry.get("top_p", 1))
        if n <= 0 or max_response_len <= 0:
            raise ValueError(f"Evaluation dataset {name!r} requires positive sample and response limits.")
        if expected_max_response_len is not None and max_response_len != expected_max_response_len:
            raise ValueError(
                f"Evaluation dataset {name!r} has max_response_len={max_response_len}; "
                f"the frozen experiment requires {expected_max_response_len}."
            )
        if temperature < 0 or not 0 < top_p <= 1:
            raise ValueError(f"Evaluation dataset {name!r} has invalid temperature/top_p.")

        rm_type = str(entry.get("rm_type") or "").strip()
        if rm_type not in {"unit_test", "livecodebench"}:
            continue
        route = routes.get(rm_type) or (reward_config.get("code") if rm_type == "unit_test" else None) or {}
        if not route.get("url"):
            raise ValueError(f"Evaluation dataset {name!r} requires routes.{rm_type}.url.")
        validate_url(str(route["url"]), f"{name} evaluator URL")
        preflight_url = str(route.get("preflight_url") or route["url"])
        validate_url(preflight_url, f"{name} sandbox preflight URL")
        if check_runtime_deps:
            from slime_plugins.m2rl.sandbox_security import validate_preflight_marker

            validate_preflight_marker(route, preflight_url)

    if len(names) != len(set(names)):
        raise ValueError("Evaluation dataset names must be unique.")
    return names


def validate(args: argparse.Namespace) -> dict:
    if args.optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {args.optimizer!r}; choose from {sorted(OPTIMIZERS)}.")
    if args.algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {args.algorithm!r}; choose from {sorted(ALGORITHMS)}.")
    for path, label in (
        (args.manifest, "data manifest"),
        (args.model_config, "model config"),
        (args.hf_checkpoint, "HF checkpoint"),
        (args.load_checkpoint, "Megatron checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not (
        (args.load_checkpoint / "latest_checkpointed_iteration.txt").exists()
        or (args.load_checkpoint / "config.json").exists()
    ):
        raise FileNotFoundError(
            "Megatron checkpoint must contain latest_checkpointed_iteration.txt "
            f"(or be a supported HF directory with config.json): {args.load_checkpoint}"
        )

    manifest = mapping_file(args.manifest)
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Data manifest requires a non-empty sources list.")
    if any(not isinstance(source, dict) or not source.get("path") for source in sources):
        raise ValueError("Every data-manifest source must be a mapping with a non-empty path.")
    names = [source.get("name") for source in sources]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Data-manifest source names must be non-empty and unique.")
    sampling = manifest.get("sampling") or {}
    if sampling.get("strategy", "proportional") not in {
        "uniform",
        "proportional",
        "weighted",
        "stratified",
        "round_robin",
        "sequential",
    }:
        raise ValueError(f"Unknown task sampling strategy: {sampling.get('strategy')!r}.")
    if sampling.get("unit", "prompt") not in {"prompt", "batch"}:
        raise ValueError("Task sampling unit must be `prompt` or `batch`.")

    base = args.manifest.parent
    missing_data = []
    for source in sources:
        source_path = Path(os.path.expandvars(str(source.get("path", "")).split("@[", 1)[0])).expanduser()
        if not source_path.is_absolute():
            source_path = base / source_path
        if not source_path.exists():
            missing_data.append(str(source_path))
    if missing_data:
        raise FileNotFoundError("Manifest source files do not exist: " + ", ".join(missing_data))

    if args.reward_config is None or not args.reward_config.exists():
        raise FileNotFoundError("An existing --reward-config is required for the multi-task matrix.")
    reward_config = mapping_file(args.reward_config)
    if any(source.get("rm_type") == "unit_test" for source in sources):
        code_route = (reward_config.get("routes") or {}).get("unit_test") or reward_config.get("code") or {}
        if not code_route.get("url"):
            raise ValueError("Code tasks require routes.unit_test.url (or code.url) in the reward config.")
        validate_url(str(code_route["url"]), "Code sandbox URL")
        if args.check_runtime_deps:
            from slime_plugins.m2rl.sandbox_security import validate_preflight_marker

            validate_preflight_marker(code_route, str(code_route.get("preflight_url") or code_route["url"]))
    if any(source.get("rm_type") == "workbench" for source in sources):
        validate_url(args.workbench_url, "WorkBench resource-server URL")
    eval_names = validate_eval_config(
        args.eval_config,
        reward_config,
        check_runtime_deps=args.check_runtime_deps,
        expected_max_response_len=args.expected_eval_max_response_len,
    )

    uses_opd = args.algorithm in {"opd", "sft_opd", "grpo_opd", "ppo_opd"}
    uses_ppo = args.algorithm in {"ppo", "ppo_opd"}
    if args.algorithm == "sft_opd":
        modes = [str((source.get("metadata") or {}).get("training_mode", "")) for source in sources]
        if "sft" not in modes or "opd" not in modes:
            raise ValueError(
                "sft_opd requires manifest sources with metadata.training_mode set to both `sft` and `opd`."
            )
        if sampling.get("strategy") != "stratified" or sampling.get("unit", "prompt") != "prompt":
            raise ValueError("sft_opd requires `sampling.strategy: stratified` and `sampling.unit: prompt`.")
    if uses_opd:
        if args.teacher_config is None or not args.teacher_config.exists():
            raise FileNotFoundError("OPD algorithms require an existing --teacher-config.")
        teacher_config = mapping_file(args.teacher_config)
        teachers = teacher_config.get("teachers")
        if not isinstance(teachers, dict) or not teachers:
            raise ValueError("Teacher config requires a non-empty teachers mapping.")
        missing_teachers = []
        for source in sources:
            metadata = source.get("metadata") or {}
            route_keys = (
                source.get("teacher"),
                metadata.get("teacher"),
                metadata.get("task_name"),
                source.get("name"),
                source.get("rm_type"),
            )
            if not any(key is not None and str(key) in teachers for key in route_keys):
                missing_teachers.append(source["name"])
        if missing_teachers and not teacher_config.get("default"):
            raise ValueError(f"Teacher config has no route for tasks: {missing_teachers}")
        for task, route in teachers.items():
            value = route.get("url") if isinstance(route, dict) else route
            validate_url(str(value), f"Teacher URL for {task}")
        if teacher_config.get("default") is not None:
            default = teacher_config["default"]
            value = default.get("url") if isinstance(default, dict) else default
            validate_url(str(value), "Default teacher URL")
    if uses_ppo:
        if args.ppo_config is None or not args.ppo_config.exists():
            raise FileNotFoundError("PPO algorithms require an existing --ppo-config.")
        entries = mapping_file(args.ppo_config).get("megatron")
        if not isinstance(entries, list):
            raise ValueError("PPO config requires a top-level megatron list.")
        actor_entries = [entry for entry in entries if entry.get("role") == "actor"]
        critic_entries = [entry for entry in entries if entry.get("role") == "critic"]
        if len(actor_entries) > 1 or len(critic_entries) != 1:
            raise ValueError("PPO config requires exactly one critic entry and at most one actor entry.")
        actor_overrides = (actor_entries[0].get("overrides") or actor_entries[0].get("args") or {}) if actor_entries else {}
        if "optimizer" in actor_overrides:
            raise ValueError("PPO actor config must inherit the CLI optimizer; remove its optimizer override.")
        critic_overrides = critic_entries[0].get("overrides") or critic_entries[0].get("args") or {}
        if str(critic_overrides.get("optimizer", "")).lower() != "adam":
            raise ValueError("PPO critic optimizer must be explicitly fixed to AdamW (`optimizer: adam`).")
    if OPTIMIZERS[args.optimizer] in {"muon", "dist_muon"} and args.check_runtime_deps:
        if importlib.util.find_spec("emerging_optimizers") is None:
            raise ImportError("Muon requires emerging_optimizers; install the pinned requirements first.")
    if args.check_runtime_deps and any(source.get("rm_type") in {"ifevalg", "ifbench"} for source in sources):
        missing = [
            module
            for module in ("absl", "immutabledict", "langdetect", "nltk")
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            raise ImportError(
                "IFEvalG dependencies are missing: " + ", ".join(missing) + ". Rebuild the pinned environment."
            )
        import nltk

        missing_resources = []
        for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                nltk.data.find(resource)
            except LookupError:
                missing_resources.append(resource.rsplit("/", 1)[-1])
        if missing_resources:
            raise LookupError(
                "Missing NLTK resources: "
                + ", ".join(missing_resources)
                + ". Run `python -m nltk.downloader punkt punkt_tab`."
            )

    return {
        "optimizer": OPTIMIZERS[args.optimizer],
        "algorithm": args.algorithm,
        "tasks": names,
        "sampling": sampling,
        "uses_opd": uses_opd,
        "uses_ppo": uses_ppo,
        "eval_datasets": eval_names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--hf-checkpoint", type=Path, required=True)
    parser.add_argument("--load-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--expected-eval-max-response-len", type=int)
    parser.add_argument("--workbench-url", default="http://127.0.0.1:12000")
    parser.add_argument("--ppo-config", type=Path)
    parser.add_argument("--check-runtime-deps", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2, sort_keys=True))
