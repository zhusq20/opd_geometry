"""Tests for single-task manifest and holdout preparation."""

import argparse
import json
import runpy
from pathlib import Path

import pytest
import yaml


NUM_GPUS = 0
MODULE = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "optimizer_geometry" / "prepare_single_task_data.py"))


@pytest.mark.unit
def test_prepare_writes_disjoint_holdout_and_hybrid_manifest(tmp_path):
    rows = [
        {
            "prompt": [{"role": "user", "content": f"problem {index}"}],
            "label": str(index),
            "metadata": {"rm_type": "deepscaler"},
        }
        for index in range(10)
    ]
    rl_path = tmp_path / "math.jsonl"
    rl_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sft_path = tmp_path / "math_sft.jsonl"
    sft_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "problem"},
                    {"role": "assistant", "content": "solution"},
                ]
            }
        )
        + "\n"
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "math",
                        "path": "math.jsonl",
                        "input_key": "prompt",
                        "label_key": "label",
                        "metadata_key": "metadata",
                        "apply_chat_template": True,
                        "rm_type": "deepscaler",
                    }
                ]
            }
        )
    )
    args = argparse.Namespace(
        rl_manifest=manifest_path,
        output_dir=tmp_path / "out",
        tasks=["math"],
        sft=[f"math={sft_path}"],
        eval=[],
        holdout_count=3,
        sft_ratio=0.25,
        sft_max_samples=100,
        eval_max_response_len=512,
        eval_samples_per_prompt=1,
        seed=9,
    )

    summary = MODULE["prepare"](args)
    assert summary["tasks"]["math"]["train_rows"] == 7
    assert summary["tasks"]["math"]["eval_rows"] == 3

    train_lines = (tmp_path / "out" / "math" / "math_train.jsonl").read_text().splitlines()
    eval_lines = (tmp_path / "out" / "math" / "math_holdout.jsonl").read_text().splitlines()
    assert set(train_lines).isdisjoint(eval_lines)
    assert len(train_lines) + len(eval_lines) == 10

    hybrid = yaml.safe_load((tmp_path / "out" / "math" / "math_sft_opd.yaml").read_text())
    assert hybrid["sampling"] == {
        "strategy": "stratified",
        "unit": "prompt",
        "seed": 9,
        "repeat": True,
    }
    assert [source["weight"] for source in hybrid["sources"]] == [0.75, 0.25]
    assert [source["metadata"]["training_mode"] for source in hybrid["sources"]] == ["opd", "sft"]


@pytest.mark.unit
def test_tail_view_holdout_does_not_duplicate_training_file(tmp_path):
    rows = [{"prompt": f"problem {index}", "label": str(index), "metadata": {"rm_type": "gpqa"}} for index in range(8)]
    source = tmp_path / "science.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"sources": [{"name": "science", "path": source.name, "rm_type": "gpqa"}]}))
    args = argparse.Namespace(
        rl_manifest=manifest,
        output_dir=tmp_path / "out",
        tasks=["science"],
        sft=[],
        eval=[],
        holdout_count=3,
        holdout_mode="tail_view",
        sft_ratio=0.5,
        sft_max_samples=100,
        eval_max_response_len=512,
        eval_samples_per_prompt=1,
        seed=4,
    )

    summary = MODULE["prepare"](args)

    assert summary["tasks"]["science"]["train_rows"] == 5
    assert not (tmp_path / "out" / "science" / "science_train.jsonl").exists()
    holdout = (tmp_path / "out" / "science" / "science_holdout.jsonl").read_text().splitlines()
    assert [json.loads(line)["label"] for line in holdout] == ["5", "6", "7"]
    on_policy = yaml.safe_load((tmp_path / "out" / "science" / "science_on_policy.yaml").read_text())
    assert on_policy["sources"][0]["path"] == f"{source.resolve()}@[0:5]"


@pytest.mark.unit
def test_external_benchmark_keeps_full_train_and_skip_eval_writes_no_config(tmp_path):
    sources = []
    for task, rm_type in (("math", "deepscaler"), ("code", "unit_test")):
        path = tmp_path / f"{task}.jsonl"
        path.write_text("".join(json.dumps({"prompt": f"{task}-{index}", "label": str(index), "metadata": {"rm_type": rm_type}}) + "\n" for index in range(5)))
        sources.append({"name": task, "path": path.name, "rm_type": rm_type})

    benchmark = tmp_path / "aime24.jsonl"
    benchmark.write_text(json.dumps({"prompt": "aime", "label": "1"}) + "\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"sources": sources}))
    args = argparse.Namespace(
        rl_manifest=manifest,
        output_dir=tmp_path / "out",
        tasks=["math", "code"],
        sft=[],
        eval=[f"math={benchmark}"],
        eval_name=["math=aime24"],
        eval_rm_type=["math=deepscaler"],
        eval_samples=["math=8"],
        skip_eval=["code"],
        holdout_count=3,
        holdout_mode="tail_view",
        sft_ratio=0.5,
        sft_max_samples=100,
        eval_max_response_len=16384,
        eval_samples_per_prompt=1,
        eval_top_p=0.7,
        seed=42,
    )

    summary = MODULE["prepare"](args)

    assert summary["tasks"]["math"] == {
        "on_policy_manifest": str(tmp_path / "out" / "math" / "math_on_policy.yaml"),
        "sft_opd_manifest": None,
        "eval_config": str(tmp_path / "out" / "math" / "math_eval.yaml"),
        "eval_kind": "external_benchmark",
        "eval_name": "aime24",
        "eval_rm_type": "deepscaler",
        "eval_samples_per_prompt": 8,
        "train_rows": 5,
        "eval_rows": 1,
    }
    assert summary["tasks"]["code"]["eval_kind"] == "disabled"
    assert summary["tasks"]["code"]["eval_config"] is None
    assert summary["tasks"]["code"]["train_rows"] == 5
    assert not (tmp_path / "out" / "code" / "code_eval.yaml").exists()

    math_manifest = yaml.safe_load((tmp_path / "out" / "math" / "math_on_policy.yaml").read_text())
    code_manifest = yaml.safe_load((tmp_path / "out" / "code" / "code_on_policy.yaml").read_text())
    assert math_manifest["sources"][0]["path"] == str((tmp_path / "math.jsonl").resolve())
    assert code_manifest["sources"][0]["path"] == str((tmp_path / "code.jsonl").resolve())
    eval_config = yaml.safe_load((tmp_path / "out" / "math" / "math_eval.yaml").read_text())
    assert eval_config["eval"]["defaults"]["max_response_len"] == 16384
    assert eval_config["eval"]["defaults"]["n_samples_per_eval_prompt"] == 8
    assert eval_config["eval"]["datasets"][0]["name"] == "aime24"


@pytest.mark.unit
def test_external_benchmark_can_remove_training_overlap_and_override_sampling(tmp_path):
    overlapping_problem = "Find the number of integer values of $k$ for which the equation has one solution."
    train_rows = [
        {
            "prompt": [{"role": "user", "content": "Solve carefully. " + overlapping_problem}],
            "label": "501",
            "metadata": {"rm_type": "deepscaler", "original_dataset": "source-a", "original_index": 7},
        },
        {
            "prompt": [{"role": "user", "content": "A different problem."}],
            "label": "2",
            "metadata": {"rm_type": "deepscaler", "original_dataset": "source-b", "original_index": 8},
        },
    ]
    train_path = tmp_path / "math.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    benchmark_path = tmp_path / "math500.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {
                "prompt": "benchmark prompt",
                "label": "501",
                "metadata": {
                    "problem": overlapping_problem,
                    "unique_id": "test/algebra/overlap.json",
                },
            }
        )
        + "\n"
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "math",
                        "path": train_path.name,
                        "input_key": "prompt",
                        "rm_type": "deepscaler",
                    }
                ]
            }
        )
    )
    args = argparse.Namespace(
        rl_manifest=manifest_path,
        output_dir=tmp_path / "out",
        tasks=["math"],
        sft=[],
        eval=[f"math={benchmark_path}"],
        eval_name=["math=math500"],
        eval_rm_type=["math=deepscaler"],
        eval_samples=["math=1"],
        eval_max_response_len_override=["math=32768"],
        eval_temperature=["math=0"],
        eval_top_p_override=["math=1"],
        exclude_eval_overlap=["math"],
        skip_eval=[],
        holdout_count=1,
        holdout_mode="tail_view",
        sft_ratio=0.5,
        sft_max_samples=100,
        eval_max_response_len=16384,
        eval_samples_per_prompt=1,
        eval_top_p=0.7,
        seed=42,
    )

    summary = MODULE["prepare"](args)

    math = summary["tasks"]["math"]
    assert math["train_rows"] == 1
    assert math["excluded_eval_overlap_rows"] == 1
    assert math["excluded_eval_overlaps"] == [
        {
            "train_row": 0,
            "eval_ids": ["test/algebra/overlap.json"],
            "original_dataset": "source-a",
            "original_index": 7,
        }
    ]
    filtered_path = tmp_path / "out" / "math" / "math_train_eval_disjoint.jsonl"
    assert json.loads(filtered_path.read_text())["label"] == "2"
    on_policy = yaml.safe_load((tmp_path / "out" / "math" / "math_on_policy.yaml").read_text())
    assert on_policy["sources"][0]["path"] == str(filtered_path.resolve())
    eval_config = yaml.safe_load((tmp_path / "out" / "math" / "math_eval.yaml").read_text())
    assert eval_config["eval"]["defaults"]["max_response_len"] == 32768
    assert eval_config["eval"]["defaults"]["temperature"] == 0.0
    assert eval_config["eval"]["defaults"]["top_p"] == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
