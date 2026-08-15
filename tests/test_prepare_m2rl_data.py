"""Tests for faithful Nemotron/M2RL row conversion."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


NUM_GPUS = 0
SCRIPT = Path(__file__).parents[1] / "examples" / "optimizer_geometry" / "prepare_m2rl_data.py"
MODULE = runpy.run_path(str(SCRIPT))
convert_row = MODULE["convert_row"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset, task, rm_type",
    [
        ("nano_v3_sft_profiled_dapo17k", "math", "deepscaler"),
        ("nano_v3_sft_profiled_stem_mcqa", "science", "gpqa"),
        ("nano_v3_sft_profiled_instruction_following", "if", "ifevalg"),
        ("nano_v3_sft_profiled_comp_coding_50tests", "code", "unit_test"),
        ("nano_v3_sft_profiled_workbench", "agent", "workbench"),
    ],
)
def test_convert_row_maps_all_five_domains(dataset, task, rm_type):
    row = {
        "dataset": dataset,
        "responses_create_params": {"input": [{"role": "user", "content": "question"}], "tools": []},
        "expected_answer": "A",
        "prompt": "question",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["x"]}],
        "verifier_metadata": {"unit_tests": {"inputs": ["1"], "outputs": ["1"]}},
        "ground_truth": [{"name": "tool"}],
    }

    actual_task, converted = convert_row(row, 17)

    assert actual_task == task
    assert converted["data_source"] == task
    assert converted["metadata"]["rm_type"] == rm_type
    assert converted["metadata"]["original_index"] == 17
    assert isinstance(converted["prompt"], list)


@pytest.mark.unit
def test_default_preparation_materializes_only_four_non_agent_domains(tmp_path):
    datasets = [
        "nano_v3_sft_profiled_dapo17k",
        "nano_v3_sft_profiled_stem_mcqa",
        "nano_v3_sft_profiled_instruction_following",
        "nano_v3_sft_profiled_comp_coding_50tests",
        "nano_v3_sft_profiled_workbench",
    ]
    input_path = tmp_path / "train_complete.jsonl"
    rows = []
    for dataset in datasets:
        rows.append(
            {
                "dataset": dataset,
                "responses_create_params": {"input": [{"role": "user", "content": "question"}], "tools": []},
                "expected_answer": "A",
                "prompt": "question",
                "instruction_id_list": [],
                "kwargs": [],
                "verifier_metadata": {"unit_tests": {"inputs": ["1"], "outputs": ["1"]}},
                "ground_truth": [{"name": "tool"}],
            }
        )
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output_dir = tmp_path / "prepared"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(input_path), "--output-dir", str(output_dir)],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert {path.name for path in output_dir.glob("*.jsonl")} == {
        "math.jsonl",
        "science.jsonl",
        "if.jsonl",
        "code.jsonl",
    }
    manifest = yaml.safe_load((output_dir / "multitask_manifest.yaml").read_text())
    assert [source["name"] for source in manifest["sources"]] == ["math", "science", "if", "code"]
    info = json.loads((output_dir / "dataset_info.json").read_text())
    assert info["tasks"] == ["math", "science", "if", "code"]
    assert all(entry["records"] == 1 for entry in info["files"].values())
