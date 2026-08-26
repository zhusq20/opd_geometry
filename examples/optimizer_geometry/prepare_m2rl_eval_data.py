#!/usr/bin/env python3
"""Prepare the independent online-eval datasets used by the M2RL experiments.

The optimizer-geometry study can evaluate math on AIME 2024, MATH-500, or both,
science on GPQA Diamond, and instruction following on official IFEval and
IFBench during training. GPQA is gated on Hugging Face; this script reads
authentication only from ``HF_TOKEN`` (or the normal Hugging Face credential
store) and never writes a credential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import get_token


AIME24_ID = "math-ai/aime24"
AIME24_REVISION = "83a7f387baaa524a8bda0022eac0541582297103"
MATH500_ID = "HuggingFaceH4/MATH-500"
# Pin the benchmark contents used by the paper cells. The output index also
# records this revision and the materialized Parquet SHA-256.
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
GPQA_ID = "Idavidrein/gpqa"
IFEVAL_ID = "google/IFEval"
IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
IFBENCH_ID = "allenai/IFBench_test"
IFBENCH_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
IFBENCH_SCORER_REPOSITORY = "https://github.com/allenai/IFBench.git"
IFBENCH_SCORER_REVISION = "1091c4c3de6c1f6ed12c012ed68f11ea450b0117"
EXPECTED_ROWS = {"aime24": 30, "math500": 500, "gpqa_diamond": 198, "ifeval": 541, "ifbench": 300}
MATH_EVAL_DATASETS = ("aime24", "math500")

MATH_INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: \\boxed{$Answer} (without quotes) where $Answer is the answer to the problem.\n\n"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boxed_answer(solution: str) -> str:
    solution = str(solution).strip()
    if solution.startswith("\\boxed{") and solution.endswith("}"):
        return solution[len("\\boxed{") : -1]
    raise ValueError(f"Unexpected AIME solution format: {solution!r}")


def aime24_rows(dataset: Any) -> list[dict[str, Any]]:
    return [
        {
            "prompt": [{"role": "user", "content": MATH_INSTRUCTION + str(row["problem"])}],
            "label": boxed_answer(str(row["solution"])),
            "data_source": "aime-2024",
            "metadata": {"rm_type": "deepscaler"},
        }
        for row in dataset
    ]


def math500_rows(dataset: Any) -> list[dict[str, Any]]:
    """Convert official MATH-500 rows without deriving labels from solutions."""

    return [
        {
            "prompt": [{"role": "user", "content": MATH_INSTRUCTION + str(row["problem"])}],
            "label": str(row["answer"]).strip(),
            "data_source": "math500",
            "metadata": {
                "rm_type": "deepscaler",
                # Preserve the raw problem so single-task preparation can
                # remove benchmark overlap from the training view.
                "problem": str(row["problem"]),
                "subject": str(row["subject"]),
                "level": int(row["level"]),
                "unique_id": str(row["unique_id"]),
            },
        }
        for row in dataset
    ]


def gpqa_diamond_rows(dataset: Any, seed: int) -> list[dict[str, Any]]:
    instruction = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
    query_template = "{instruction}\n\n{question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
    rng = random.Random(seed)
    output = []
    for row in dataset:
        gold_index = rng.randint(0, 3)
        choices = [
            str(row["Incorrect Answer 1"]).strip(),
            str(row["Incorrect Answer 2"]).strip(),
            str(row["Incorrect Answer 3"]).strip(),
        ]
        choices.insert(gold_index, str(row["Correct Answer"]).strip())
        correct_letter = "ABCD"[gold_index]
        output.append(
            {
                "prompt": query_template.format(
                    instruction=instruction,
                    question=str(row["Question"]).strip(),
                    a=choices[0],
                    b=choices[1],
                    c=choices[2],
                    d=choices[3],
                ),
                "label": correct_letter,
                "data_source": "gpqa",
                "metadata": {
                    "rm_type": "gpqa",
                    "valid_letters": "ABCD",
                    "correct_letter": correct_letter,
                    "choices": choices,
                },
            }
        )
    return output


def ifbench_rows(dataset: Any) -> list[dict[str, Any]]:
    """Convert the pinned official IFBench test split to Slime's eval schema."""

    return [
        {
            "prompt": str(row["prompt"]),
            "label": None,
            "data_source": "ifbench",
            "metadata": {
                "rm_type": "ifbench",
                "record_id": int(row["key"]),
                "prompt_text": str(row["prompt"]),
                "instruction_id_list": list(row["instruction_id_list"]),
                "kwargs": list(row["kwargs"]),
            },
        }
        for row in dataset
    ]


def ifeval_rows(dataset: Any) -> list[dict[str, Any]]:
    """Convert the pinned official IFEval prompts to Slime's eval schema."""

    return [
        {
            "prompt": str(row["prompt"]),
            "label": None,
            "data_source": "ifeval",
            "metadata": {
                "rm_type": "ifevalg",
                "record_id": int(row["key"]),
                "prompt_text": str(row["prompt"]),
                "instruction_id_list": list(row["instruction_id_list"]),
                "kwargs": list(row["kwargs"]),
            },
        }
        for row in dataset
    ]


def atomic_write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.parquet")
    try:
        Dataset.from_list(rows).to_parquet(str(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_math_eval_datasets(values: list[str]) -> list[str]:
    selected = set(values)
    unknown = selected - set(MATH_EVAL_DATASETS)
    if unknown:
        raise ValueError(f"Unknown Math evaluation datasets: {sorted(unknown)}.")
    if not values:
        raise ValueError("At least one Math evaluation dataset must be selected.")
    if len(values) != len(set(values)):
        raise ValueError("Math evaluation dataset names must not be repeated.")
    return [name for name in MATH_EVAL_DATASETS if name in selected]


def math_eval_config(dataset_names: list[str], data_dir: Path) -> dict[str, Any]:
    selected = normalize_math_eval_datasets(dataset_names)
    dataset_configs = {
        "aime24": {
            "name": "aime24",
            "path": str((data_dir / "aime24.parquet").resolve()),
            "rm_type": "deepscaler",
            "n_samples_per_eval_prompt": 8,
            "temperature": 1.0,
            "top_p": 0.7,
        },
        "math500": {
            "name": "math500",
            "path": str((data_dir / "math500.parquet").resolve()),
            "rm_type": "deepscaler",
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }
    for name in selected:
        data_path = Path(dataset_configs[name]["path"])
        if not data_path.is_file():
            raise FileNotFoundError(f"Math evaluation dataset is missing: {data_path}")
    return {
        "eval": {
            "defaults": {
                "max_response_len": 32768,
                "apply_chat_template": True,
                "custom_rm_path": "slime_plugins.m2rl.rewards.reward",
            },
            "datasets": [dataset_configs[name] for name in selected],
        }
    }


def write_math_eval_configs(data_dir: Path, config_dir: Path, active_datasets: list[str]) -> dict[str, Any]:
    """Write single- and combined-benchmark configs plus the active alias."""

    active = normalize_math_eval_datasets(active_datasets)
    variants = {
        "aime24": ["aime24"],
        "math500": ["math500"],
        "aime24_math500": ["aime24", "math500"],
    }
    paths: dict[str, str] = {}
    for name, datasets in variants.items():
        path = config_dir / f"math_eval_{name}.yaml"
        atomic_write_yaml(math_eval_config(datasets, data_dir), path)
        paths[name] = str(path.resolve())

    active_name = "_".join(active)
    active_path = config_dir / "math_eval.yaml"
    atomic_write_yaml(math_eval_config(active, data_dir), active_path)
    return {
        "active_datasets": active,
        "active_config": str(active_path.resolve()),
        "available_configs": paths,
        "active_variant": active_name,
    }


def write_instruction_following_eval_config(data_dir: Path, config_dir: Path) -> dict[str, Any]:
    """Write the paired in-distribution and OOD strict-prompt evaluation config."""

    dataset_configs = [
        {
            "name": "ifeval_strict_prompt",
            "path": str((data_dir / "ifeval.parquet").resolve()),
            "rm_type": "ifevalg",
        },
        {
            "name": "ifbench_strict",
            "path": str((data_dir / "ifbench.parquet").resolve()),
            "rm_type": "ifbench",
        },
    ]
    for dataset in dataset_configs:
        data_path = Path(dataset["path"])
        if not data_path.is_file():
            raise FileNotFoundError(f"Instruction-following evaluation dataset is missing: {data_path}")

    config_path = config_dir / "if_eval.yaml"
    atomic_write_yaml(
        {
            "eval": {
                "defaults": {
                    "max_response_len": 32768,
                    "top_p": 1.0,
                    "n_samples_per_eval_prompt": 1,
                    "apply_chat_template": True,
                    "custom_rm_path": "slime_plugins.m2rl.rewards.reward",
                    "temperature": 0.0,
                },
                "datasets": dataset_configs,
            }
        },
        config_path,
    )
    return {
        "config": str(config_path.resolve()),
        "datasets": [dataset["name"] for dataset in dataset_configs],
        "mode": "strict_prompt_level",
    }


def dataset_entry(name: str, path: Path, *, seed: int) -> dict[str, Any]:
    source = {
        "aime24": {
            "dataset_id": AIME24_ID,
            "config": None,
            "split": "test",
            "revision": AIME24_REVISION,
        },
        "math500": {
            "dataset_id": MATH500_ID,
            "config": None,
            "split": "test",
            "revision": MATH500_REVISION,
        },
        "gpqa_diamond": {"dataset_id": GPQA_ID, "config": "gpqa_diamond", "split": "train"},
        "ifeval": {
            "dataset_id": IFEVAL_ID,
            "config": None,
            "split": "train",
            "revision": IFEVAL_REVISION,
        },
        "ifbench": {
            "dataset_id": IFBENCH_ID,
            "config": None,
            "split": "train",
            "revision": IFBENCH_REVISION,
        },
    }[name]
    entry = {
        **source,
        "path": str(path.resolve()),
        "rows": parquet.ParquetFile(path).metadata.num_rows,
        "sha256": sha256_file(path),
        "seed": seed if name == "gpqa_diamond" else None,
    }
    if name == "ifeval":
        entry["scorer"] = {
            "implementation": "slime_plugins.m2rl.ifevalg",
            "mode": "strict_prompt_level",
        }
    elif name == "ifbench":
        entry["scorer"] = {
            "repository": IFBENCH_SCORER_REPOSITORY,
            "revision": IFBENCH_SCORER_REVISION,
            "mode": "strict_prompt_level",
        }
    return entry


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "eval_data_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"version": 1, "datasets": {}}
    index["purpose"] = "optimizer-geometry independent online evaluation"

    token: str | bool | None = os.environ.get("HF_TOKEN")
    if token is None and get_token() is not None:
        # ``token=True`` tells huggingface_hub to use its credential store
        # without materializing that credential in our arguments or index.
        token = True
    for name in args.datasets:
        path = args.output_dir / f"{name}.parquet"
        expected = EXPECTED_ROWS[name]
        ready = path.exists() and parquet.ParquetFile(path).metadata.num_rows == expected
        if args.force or not ready:
            try:
                if name == "aime24":
                    source = load_dataset(
                        AIME24_ID,
                        split="test",
                        revision=AIME24_REVISION,
                        token=token,
                    )
                    rows = aime24_rows(source)
                elif name == "math500":
                    source = load_dataset(
                        MATH500_ID,
                        split="test",
                        revision=MATH500_REVISION,
                        token=token,
                    )
                    rows = math500_rows(source)
                elif name == "gpqa_diamond":
                    source = load_dataset(GPQA_ID, "gpqa_diamond", split="train", token=token)
                    rows = gpqa_diamond_rows(source, args.seed)
                elif name == "ifeval":
                    source = load_dataset(
                        IFEVAL_ID,
                        split="train",
                        revision=IFEVAL_REVISION,
                        token=token,
                    )
                    rows = ifeval_rows(source)
                elif name == "ifbench":
                    source = load_dataset(
                        IFBENCH_ID,
                        split="train",
                        revision=IFBENCH_REVISION,
                        token=token,
                    )
                    rows = ifbench_rows(source)
                else:
                    raise AssertionError(f"Unhandled evaluation dataset: {name}")
            except Exception as exc:
                if name == "gpqa_diamond":
                    raise RuntimeError("GPQA Diamond is gated. Accept its Hugging Face access terms, export HF_TOKEN in your shell, and rerun; the token is never stored by this script.") from exc
                raise
            if len(rows) != expected:
                raise ValueError(f"Expected {expected} rows for {name}, received {len(rows)}.")
            atomic_write_parquet(rows, path)

        index["datasets"][name] = dataset_entry(name, path, seed=args.seed)
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    math_eval_config_dir = getattr(args, "math_eval_config_dir", None)
    if math_eval_config_dir is not None:
        index["math_eval"] = write_math_eval_configs(
            args.output_dir,
            math_eval_config_dir,
            getattr(args, "math_eval_datasets", ["math500"]),
        )
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if_eval_config_dir = getattr(args, "if_eval_config_dir", None)
    if if_eval_config_dir is not None:
        index["if_eval"] = write_instruction_following_eval_config(args.output_dir, if_eval_config_dir)
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(EXPECTED_ROWS),
        default=["aime24", "math500", "gpqa_diamond", "ifeval", "ifbench"],
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic GPQA option-order seed.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--math-eval-config-dir",
        type=Path,
        help="Write AIME'24-only, MATH-500-only, combined, and active Math eval YAML files here.",
    )
    parser.add_argument(
        "--math-eval-datasets",
        nargs="+",
        choices=MATH_EVAL_DATASETS,
        default=["math500"],
        help="Datasets included in the active math_eval.yaml alias.",
    )
    parser.add_argument(
        "--if-eval-config-dir",
        type=Path,
        help="Write a paired IFEval-strict and IFBench-strict eval config in this directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = prepare(parse_args())
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, indent=2, sort_keys=True))
