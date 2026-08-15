"""Tests for preregistered, equal-budget hyperparameter selection."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from examples.optimizer_geometry.select_tuning_hparams import candidate_rows, inspect_run, selections


NUM_GPUS = 0
ROOT = Path(__file__).parents[1]
OPTIMIZER_TUNING_LAUNCHER = ROOT / "examples" / "optimizer_geometry" / "run_optimizer_tuning.sh"
OPD_COEFFICIENT_LAUNCHER = ROOT / "examples" / "optimizer_geometry" / "run_opd_coefficient_tuning.sh"


def _run(tmp_path, *, lr: float, seed: int, start: float, final: float, clipped: int = 0):
    run = tmp_path / f"lr{lr}_seed{seed}"
    (run / "provenance").mkdir(parents=True)
    (run / "metrics").mkdir()
    command = [
        "python3",
        "train.py",
        "--experiment-condition",
        "grpo",
        "--experiment-optimizer",
        "adamw",
        "--seed",
        str(seed),
        "--lr",
        str(lr),
    ]
    (run / "provenance" / "run_manifest.json").write_text(
        json.dumps({"status": "complete", "command": command}) + "\n"
    )
    (run / "run_complete.json").write_text(
        json.dumps({"status": "complete", "final_num_updates": 50}) + "\n"
    )
    eval_events = [
        {"metrics": {"eval/num_updates": 0, "eval/tuning": start}},
        {"metrics": {"eval/num_updates": 50, "eval/tuning": final}},
    ]
    (run / "metrics" / "eval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in eval_events))
    train_events = [
        {"metrics": {"train/grad_norm": 1.0, "train/grad_clipped": clipped}}
        for _ in range(4)
    ]
    (run / "metrics" / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_events))
    return run


@pytest.mark.unit
def test_tuning_selector_uses_auc_and_one_se_smaller_value_rule(tmp_path):
    specs = [
        (1e-6, 1042, 0.48, 0.50),
        (1e-6, 1043, 0.50, 0.54),
        (3e-6, 1042, 0.49, 0.53),
        (3e-6, 1043, 0.51, 0.55),
    ]
    runs = [inspect_run(_run(tmp_path, lr=lr, seed=seed, start=start, final=final), "eval/tuning", "lr", 1e4) for lr, seed, start, final in specs]

    selected = selections(candidate_rows(runs, required_seeds=2))

    assert selected[0]["selected_value"] == pytest.approx(1e-6)
    assert selected[0]["best_observed_value"] == pytest.approx(3e-6)


@pytest.mark.unit
def test_tuning_selector_marks_frequently_clipped_run_unstable(tmp_path):
    run = _run(tmp_path, lr=1e-6, seed=1042, start=0.1, final=0.2, clipped=1)

    result = inspect_run(run, "eval/tuning", "lr", 1e4)

    assert result["stable"] is False
    assert any("more than 50%" in error for error in result["errors"])


@pytest.mark.unit
def test_tuning_selector_rejects_unmatched_seed_sets(tmp_path):
    runs = [
        inspect_run(_run(tmp_path, lr=1e-6, seed=1042, start=0.1, final=0.2), "eval/tuning", "lr", 1e4),
        inspect_run(_run(tmp_path, lr=1e-6, seed=1043, start=0.1, final=0.2), "eval/tuning", "lr", 1e4),
        inspect_run(_run(tmp_path, lr=3e-6, seed=1042, start=0.1, final=0.3), "eval/tuning", "lr", 1e4),
        inspect_run(_run(tmp_path, lr=3e-6, seed=1044, start=0.1, final=0.3), "eval/tuning", "lr", 1e4),
    ]

    candidates = candidate_rows(runs, required_seeds=2)

    assert selections(candidates) == []
    assert all("shared stable seeds" in row["eligibility_errors"][0] for row in candidates)


@pytest.mark.unit
def test_tuning_selector_rejects_different_eval_grids(tmp_path):
    runs = []
    for lr in (1e-6, 3e-6):
        for seed in (1042, 1043):
            run = _run(tmp_path, lr=lr, seed=seed, start=0.1, final=0.2)
            if lr == 3e-6:
                events = [
                    {"metrics": {"eval/num_updates": 0, "eval/tuning": 0.1}},
                    {"metrics": {"eval/num_updates": 25, "eval/tuning": 0.15}},
                    {"metrics": {"eval/num_updates": 50, "eval/tuning": 0.2}},
                ]
                (run / "metrics" / "eval.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in events)
                )
            runs.append(inspect_run(run, "eval/tuning", "lr", 1e4))

    candidates = candidate_rows(runs, required_seeds=2)

    assert selections(candidates) == []
    assert all("eval step grids differ" in row["eligibility_errors"] for row in candidates)


@pytest.mark.unit
def test_optimizer_tuning_launcher_uses_responsive_budget_and_lr_grids(tmp_path):
    data = tmp_path / "tuning_train.yaml"
    evaluation = tmp_path / "tuning_eval.yaml"
    data.write_text("sources: []\n")
    evaluation.write_text("eval: {}\n")
    fake = tmp_path / "launcher.sh"
    fake.write_text(
        'lr="${ADAMW_LR:-${MUON_LR:-${SGD_LR:-}}}"\n'
        'printf "%s|%s|%s|%s|%s\\n" "$OPTIMIZER" "$lr" "$NUM_ROLLOUT" '
        '"$EVAL_INTERVAL" "$BATCH_PROFILE"\n'
    )
    environment = os.environ.copy()
    environment.update(
        {
            "TUNING_DATA_MANIFEST": str(data),
            "TUNING_EVAL_CONFIG": str(evaluation),
            "EXPERIMENT_LAUNCHER": str(fake),
            "TUNING_ALGORITHMS": "opd",
            "TUNING_SEEDS": "11",
            "ADAPTIVE_LR_CANDIDATES": "2.5e-7",
            "SGD_OPD_PPO_LR_CANDIDATES": "2.5e-3",
        }
    )

    result = subprocess.run(
        ["bash", str(OPTIMIZER_TUNING_LAUNCHER)], env=environment, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {
        "adamw|2.5e-7|800|160|responsive16",
        "muon|2.5e-7|800|160|responsive16",
        "sgd|2.5e-3|800|160|responsive16",
    }


@pytest.mark.unit
def test_optimizer_tuning_launcher_centers_grid_on_responsive8_defaults(tmp_path):
    data = tmp_path / "tuning_train.yaml"
    evaluation = tmp_path / "tuning_eval.yaml"
    data.write_text("sources: []\n")
    evaluation.write_text("eval: {}\n")
    fake = tmp_path / "launcher.sh"
    fake.write_text(
        'lr="${ADAMW_LR:-${MUON_LR:-${SGD_LR:-}}}"\n'
        'printf "%s|%s\\n" "$OPTIMIZER" "$lr"\n'
    )
    environment = os.environ.copy()
    for name in (
        "ADAPTIVE_LR_CANDIDATES",
        "SGD_GRPO_LR_CANDIDATES",
        "SGD_OPD_PPO_LR_CANDIDATES",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "TUNING_DATA_MANIFEST": str(data),
            "TUNING_EVAL_CONFIG": str(evaluation),
            "EXPERIMENT_LAUNCHER": str(fake),
            "BATCH_PROFILE": "responsive8",
            "TUNING_ALGORITHMS": "grpo",
            "TUNING_SEEDS": "11",
        }
    )

    result = subprocess.run(
        ["bash", str(OPTIMIZER_TUNING_LAUNCHER)], env=environment, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {
        *(f"adamw|{lr}" for lr in ("9e-8", "1.8e-7", "3.6e-7", "7.2e-7")),
        *(f"muon|{lr}" for lr in ("9e-8", "1.8e-7", "3.6e-7", "7.2e-7")),
        *(f"sgd|{lr}" for lr in ("9e-3", "1.8e-2", "3.6e-2", "7.2e-2")),
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("profile", "updates", "eval_interval", "lr"),
    [
        ("responsive8", "1600", "320", "1.8e-7"),
        ("responsive16", "800", "160", "2.5e-7"),
        ("reference256", "50", "10", "1e-6"),
    ],
)
def test_opd_coefficient_launcher_scales_with_batch_profile(
    tmp_path, profile, updates, eval_interval, lr
):
    data = tmp_path / "tuning_train.yaml"
    evaluation = tmp_path / "tuning_eval.yaml"
    data.write_text("sources: []\n")
    evaluation.write_text("eval: {}\n")
    fake = tmp_path / "launcher.sh"
    fake.write_text(
        'printf "%s|%s|%s|%s|%s\\n" "$OPD_KL_COEF" "$ADAMW_LR" "$NUM_ROLLOUT" '
        '"$EVAL_INTERVAL" "$BATCH_PROFILE"\n'
    )
    environment = os.environ.copy()
    environment.update(
        {
            "TUNING_DATA_MANIFEST": str(data),
            "TUNING_EVAL_CONFIG": str(evaluation),
            "EXPERIMENT_LAUNCHER": str(fake),
            "BATCH_PROFILE": profile,
            "TUNING_SEEDS": "11",
            "OPD_COEFFICIENTS": "0.3",
        }
    )

    result = subprocess.run(
        ["bash", str(OPD_COEFFICIENT_LAUNCHER)], env=environment, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"0.3|{lr}|{updates}|{eval_interval}|{profile}"
