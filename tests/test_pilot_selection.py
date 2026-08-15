"""Tests for response-cap and common training-budget preregistration rules."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from examples.optimizer_geometry.select_response_cap import select as select_cap
from examples.optimizer_geometry.select_training_budget import decide


NUM_GPUS = 0
ROOT = Path(__file__).parents[1]
RESPONSE_LAUNCHER = ROOT / "examples" / "optimizer_geometry" / "run_response_cap_pilot.sh"
BUDGET_LAUNCHER = ROOT / "examples" / "optimizer_geometry" / "run_budget_pilot.sh"


def _cap_run(cap: int, seed: int, truncated: float) -> dict:
    return {
        "cap": cap,
        "seed": seed,
        "truncated_ratio": truncated,
        "response_p95": cap if truncated else cap * 0.8,
        "rollout_time_seconds": cap / 10,
        "stable": True,
    }


@pytest.mark.unit
def test_response_cap_rule_chooses_smallest_matched_cap_below_threshold():
    runs = [
        *[_cap_run(8192, seed, value) for seed, value in ((1042, 0.67), (1043, 0.65))],
        *[_cap_run(12288, seed, value) for seed, value in ((1042, 0.08), (1043, 0.10))],
        *[_cap_run(16384, seed, value) for seed, value in ((1042, 0.01), (1043, 0.02))],
    ]

    result = select_cap(runs, required_seeds=2, max_truncation=0.10)

    assert result["selected_cap"] == 12288
    assert result["shared_seed_set"] == [1042, 1043]


@pytest.mark.unit
def test_response_cap_rule_rejects_unmatched_seeds():
    runs = [
        _cap_run(8192, 1042, 0.05),
        _cap_run(8192, 1043, 0.05),
        _cap_run(12288, 1042, 0.01),
        _cap_run(12288, 1044, 0.01),
    ]

    result = select_cap(runs, required_seeds=2, max_truncation=0.10)

    assert result["selected_cap"] is None


def _budget_run(algorithm: str, seed: int, gain: float, slope: float) -> dict:
    return {
        "algorithm": algorithm,
        "seed": seed,
        "eval_steps": list(range(0, 201, 10)),
        "gain": gain,
        "tail_slope_per_update": slope,
        "stable": True,
    }


@pytest.mark.unit
def test_budget_rule_uses_200_only_if_every_algorithm_is_saturated():
    saturated = [
        _budget_run(algorithm, seed, gain, slope)
        for algorithm in ("grpo", "ppo", "opd")
        for seed, gain, slope in ((1044, 0.004, -0.0001), (1045, 0.006, 0.0), (1046, 0.005, 0.0001))
    ]

    result = decide(
        saturated,
        required_seeds=3,
        gain_threshold=0.01,
        required_algorithms=["grpo", "ppo", "opd"],
        target_step=200,
    )

    assert result["recommended_common_steps"] == 200
    assert all(row["recommended_steps"] == 200 for row in result["algorithm_decisions"])

    declining = [dict(run, tail_slope_per_update=-0.001 - 0.00001 * index) for index, run in enumerate(saturated)]
    result = decide(
        declining,
        required_seeds=3,
        gain_threshold=0.01,
        required_algorithms=["grpo", "ppo", "opd"],
        target_step=200,
    )
    assert result["recommended_common_steps"] == 200

    still_learning = [dict(run) for run in saturated]
    for run in still_learning:
        if run["algorithm"] == "ppo":
            run["gain"] = 0.02
    result = decide(
        still_learning,
        required_seeds=3,
        gain_threshold=0.01,
        required_algorithms=["grpo", "ppo", "opd"],
        target_step=200,
    )
    assert result["recommended_common_steps"] == 400
    assert next(row for row in result["algorithm_decisions"] if row["algorithm"] == "ppo")[
        "recommended_steps"
    ] == 400


@pytest.mark.unit
def test_budget_rule_marks_a_missing_required_algorithm_incomplete():
    runs = [
        _budget_run("grpo", seed, 0.001, slope)
        for seed, slope in ((1044, -0.0001), (1045, 0.0), (1046, 0.0001))
    ]

    result = decide(
        runs,
        required_seeds=3,
        gain_threshold=0.01,
        required_algorithms=["grpo", "ppo"],
        target_step=200,
    )

    ppo = next(row for row in result["algorithm_decisions"] if row["algorithm"] == "ppo")
    assert ppo["complete"] is False
    assert result["recommended_common_steps"] == 400


@pytest.mark.unit
def test_response_cap_launcher_expands_matched_profiles(tmp_path):
    data = tmp_path / "tuning_train.yaml"
    data.write_text("sources: []\n")
    fake = tmp_path / "launcher.sh"
    fake.write_text(
        'printf "%s|%s|%s|%s|%s\\n" "$MAX_RESPONSE_LEN" "$SGLANG_MAX_RUNNING_REQUESTS" '
        '"$MAX_TOKENS_PER_GPU" "$SEED" "$NUM_ROLLOUT"\n'
    )
    environment = os.environ.copy()
    environment.update(
        {
            "TUNING_DATA_MANIFEST": str(data),
            "EXPERIMENT_LAUNCHER": str(fake),
            "RESPONSE_PROFILES": "100:7 200:3",
            "PILOT_PROMPT_CAP": "20",
            "PILOT_SEEDS": "11 12",
        }
    )

    result = subprocess.run(["bash", str(RESPONSE_LAUNCHER)], env=environment, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {"100|7|120|11|1", "100|7|120|12|1", "200|3|220|11|1", "200|3|220|12|1"}


@pytest.mark.unit
@pytest.mark.parametrize(("extend", "steps", "fresh"), [("0", "3200", "1"), ("1", "6400", "0")])
def test_budget_launcher_supports_fresh_and_exact_extension(tmp_path, extend, steps, fresh):
    data = tmp_path / "tuning_train.yaml"
    evaluation = tmp_path / "tuning_eval.yaml"
    data.write_text("sources: []\n")
    evaluation.write_text("eval: {}\n")
    output = tmp_path / "output"
    if extend == "1":
        checkpoint = output / "budget_grpo_adamw_responsive16_seed11" / "checkpoints"
        checkpoint.mkdir(parents=True)
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("3200\n")
    fake = tmp_path / "launcher.sh"
    fake.write_text(
        'printf "%s|%s|%s|%s|%s\\n" "$ALGORITHM" "$NUM_ROLLOUT" "$FRESH_START" '
        '"${LOAD_CHECKPOINT:-base}" "$SEED"\n'
    )
    environment = os.environ.copy()
    environment.update(
        {
            "TUNING_DATA_MANIFEST": str(data),
            "TUNING_EVAL_CONFIG": str(evaluation),
            "EXPERIMENT_LAUNCHER": str(fake),
            "PILOT_OUTPUT_ROOT": str(output),
            "PILOT_ALGORITHMS": "grpo",
            "PILOT_SEEDS": "11",
            "EXTEND_TO_400": extend,
        }
    )
    environment.pop("LOAD_CHECKPOINT", None)

    result = subprocess.run(["bash", str(BUDGET_LAUNCHER)], env=environment, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    expected_load = (
        str(output / "budget_grpo_adamw_responsive16_seed11" / "checkpoints") if extend == "1" else "base"
    )
    assert result.stdout.strip() == f"grpo|{steps}|{fresh}|{expected_load}|11"
