"""CPU tests for the pinned Logic-RL Knights-and-Knaves preparer."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

NUM_GPUS = 0
MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "examples" / "optimizer_geometry" / "prepare_logic_data.py")
)


def _raw_row(difficulty: int, index: int, *, quiz_suffix: str = ""):
    names = [f"Person{position}" for position in range(difficulty)]
    return {
        "quiz": f"A {difficulty}-person Knights-and-Knaves puzzle {index}{quiz_suffix}",
        "names": names,
        "solution": [position % 2 == 0 for position in range(difficulty)],
        "statements": repr(tuple(("telling-truth", position) for position in range(difficulty))),
        "index": index,
    }


@pytest.mark.unit
def test_convert_logic_row_rebuilds_messages_and_structured_binary_label():
    converted = MODULE["convert_row"](_raw_row(3, 123), difficulty=3, source_split="train")

    assert converted["prompt"][0] == {
        "role": "system",
        "content": MODULE["LOGIC_SYSTEM_PROMPT"],
    }
    assert converted["prompt"][1]["role"] == "user"
    assert "<|im_start|>" not in converted["prompt"][1]["content"]
    assert converted["label"] == {
        "names": ["Person0", "Person1", "Person2"],
        "roles": ["knight", "knave", "knight"],
    }
    assert converted["metadata"]["rm_type"] == "kk"
    assert converted["metadata"]["difficulty"] == 3
    assert converted["metadata"]["source_id"] == "3ppl/train/123"
    assert converted["metadata"]["statements"]


@pytest.mark.unit
def test_validate_logic_splits_checks_balance_uniqueness_and_disjointness():
    train = [
        MODULE["convert_row"](_raw_row(difficulty, difficulty), difficulty=difficulty, source_split="train")
        for difficulty in MODULE["DIFFICULTIES"]
    ]
    validation = [
        MODULE["convert_row"](
            _raw_row(difficulty, 100 + difficulty, quiz_suffix=" validation"),
            difficulty=difficulty,
            source_split="validation",
        )
        for difficulty in MODULE["DIFFICULTIES"]
    ]

    MODULE["validate_splits"](
        train,
        validation,
        expected_output_rows={"train": 5, "validation": 5},
    )

    validation[0]["prompt"][1]["content"] = train[0]["prompt"][1]["content"]
    with pytest.raises(ValueError, match="overlap"):
        MODULE["validate_splits"](
            train,
            validation,
            expected_output_rows={"train": 5, "validation": 5},
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "row, message",
    [
        ({"quiz": "", "names": ["A"] * 3, "solution": [True] * 3, "statements": "x", "index": 1}, "empty"),
        (_raw_row(3, 1) | {"solution": [True, False]}, "solution entries"),
        (_raw_row(3, 1) | {"names": ["Alice", "alice", "Bob"]}, "duplicate names"),
        (_raw_row(3, 1) | {"solution": [1, 0, 1]}, "booleans"),
    ],
)
def test_convert_logic_row_rejects_invalid_upstream_schema(row, message):
    with pytest.raises(ValueError, match=message):
        MODULE["convert_row"](row, difficulty=3, source_split="train")


@pytest.mark.unit
def test_augment_manifest_adds_one_chat_templated_kk_source(tmp_path):
    base = tmp_path / "multitask_manifest.yaml"
    train = tmp_path / "logic_train.jsonl"
    output = tmp_path / "multitask_manifest_with_logic.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sampling": {"strategy": "uniform", "unit": "batch", "seed": 42, "repeat": True},
                "sources": [{"name": "math", "path": "math.jsonl", "rm_type": "deepscaler"}],
            }
        )
    )
    train.write_text("{}\n")

    MODULE["augment_manifest"](base, output, train)

    manifest = yaml.safe_load(output.read_text())
    assert [source["name"] for source in manifest["sources"]] == ["math", "logic"]
    logic = manifest["sources"][-1]
    assert logic["path"] == str(train.resolve())
    assert logic["rm_type"] == "kk"
    assert logic["apply_chat_template"] is True
    assert logic["apply_chat_template_kwargs"] == {"enable_thinking": False}
    assert logic["metadata"]["task_name"] == "logic"


@pytest.mark.unit
def test_prepare_refuses_to_overwrite_pinned_base_manifest(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("sources: []\n")
    args = SimpleNamespace(
        base_manifest=manifest,
        output_manifest=manifest,
        output_dir=tmp_path / "output",
        download_retries=3,
    )

    with pytest.raises(ValueError, match="base is immutable"):
        MODULE["prepare"](args)


@pytest.mark.unit
def test_logic_source_contract_is_fully_pinned():
    hashes = MODULE["SOURCE_SHA256"]

    assert MODULE["LOGIC_RL_REVISION"] == "9d2c457525ec14639e85afa12d49bb16efb053a4"
    assert MODULE["K_AND_K_LICENSE"] == "CC-BY-NC-SA-4.0"
    assert MODULE["EXPECTED_OUTPUT_ROWS"] == {"train": 4_500, "validation": 500}
    assert set(hashes) == {
        (difficulty, split)
        for difficulty in MODULE["DIFFICULTIES"]
        for split in ("train", "test")
    }
    assert all(len(value) == 64 for value in hashes.values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
