"""Unit tests for the M2RL independent online-evaluation converter."""

import runpy
from pathlib import Path

import pytest
import yaml


NUM_GPUS = 0
MODULE = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "optimizer_geometry" / "prepare_m2rl_eval_data.py"))


@pytest.mark.unit
def test_aime24_conversion_uses_m2rl_prompt_and_label_schema():
    rows = MODULE["aime24_rows"]([{"problem": "What is 1+1?", "solution": "\\boxed{2}"}])

    assert rows == [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": ("Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} (without quotes) where $Answer is the answer to the problem.\n\nWhat is 1+1?"),
                }
            ],
            "label": "2",
            "data_source": "aime-2024",
            "metadata": {"rm_type": "deepscaler"},
        }
    ]


@pytest.mark.unit
def test_math500_conversion_preserves_official_answer_and_overlap_metadata():
    rows = MODULE["math500_rows"](
        [
            {
                "problem": "Simplify $1+1$.",
                "solution": "The answer is $\\boxed{2}$.",
                "answer": "2",
                "subject": "Prealgebra",
                "level": 1,
                "unique_id": "test/prealgebra/example.json",
            }
        ]
    )

    assert rows == [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Solve the following math problem step by step. The last line of your response should be "
                        "of the form Answer: \\boxed{$Answer} (without quotes) where $Answer is the answer to "
                        "the problem.\n\nSimplify $1+1$."
                    ),
                }
            ],
            "label": "2",
            "data_source": "math500",
            "metadata": {
                "rm_type": "deepscaler",
                "problem": "Simplify $1+1$.",
                "subject": "Prealgebra",
                "level": 1,
                "unique_id": "test/prealgebra/example.json",
            },
        }
    ]


@pytest.mark.unit
def test_math_eval_configs_support_aime24_math500_and_combined_modes(tmp_path):
    data_dir = tmp_path / "eval_data"
    data_dir.mkdir()
    (data_dir / "aime24.parquet").touch()
    (data_dir / "math500.parquet").touch()
    config_dir = tmp_path / "configs"

    result = MODULE["write_math_eval_configs"](data_dir, config_dir, ["math500", "aime24"])

    assert result["active_datasets"] == ["aime24", "math500"]
    assert set(result["available_configs"]) == {"aime24", "math500", "aime24_math500"}
    active = yaml.safe_load((config_dir / "math_eval.yaml").read_text())
    assert active["eval"]["defaults"]["max_response_len"] == 32768
    assert [dataset["name"] for dataset in active["eval"]["datasets"]] == ["aime24", "math500"]
    aime24, math500 = active["eval"]["datasets"]
    assert (aime24["n_samples_per_eval_prompt"], aime24["temperature"], aime24["top_p"]) == (8, 1.0, 0.7)
    assert (math500["n_samples_per_eval_prompt"], math500["temperature"], math500["top_p"]) == (1, 0.0, 1.0)
    assert [
        dataset["name"]
        for dataset in yaml.safe_load((config_dir / "math_eval_aime24.yaml").read_text())["eval"]["datasets"]
    ] == ["aime24"]
    assert [
        dataset["name"]
        for dataset in yaml.safe_load((config_dir / "math_eval_math500.yaml").read_text())["eval"]["datasets"]
    ] == ["math500"]


@pytest.mark.unit
def test_gpqa_conversion_is_seeded_and_preserves_all_choices():
    source = [
        {
            "Question": "Question?",
            "Correct Answer": "correct",
            "Incorrect Answer 1": "wrong-1",
            "Incorrect Answer 2": "wrong-2",
            "Incorrect Answer 3": "wrong-3",
        }
    ]

    first = MODULE["gpqa_diamond_rows"](source, 42)
    second = MODULE["gpqa_diamond_rows"](source, 42)

    assert first == second
    row = first[0]
    assert row["metadata"]["rm_type"] == "gpqa"
    assert row["label"] == row["metadata"]["correct_letter"]
    assert sorted(row["metadata"]["choices"]) == ["correct", "wrong-1", "wrong-2", "wrong-3"]
    assert "Answer: $LETTER" in row["prompt"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
