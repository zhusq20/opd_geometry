import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
FULL = ROOT / "examples/mopd_gpas/open_mopd_full"


def test_full_recipe_pins_paper_algorithm():
    recipe = json.loads((FULL / "recipe.json").read_text(encoding="utf-8"))
    training = recipe["training"]

    assert recipe["official_revision"] == "4809a96cf85a869106ff0ff3f37d0a51e12010ae"
    assert training["train_batch_size"] == 1024
    assert training["mini_batch_size"] == 256
    assert training["inner_updates_per_rollout"] == 4
    assert training["dense_student_top_k"] == 16
    assert training["token_share_target"] == {
        "math": 1 / 3,
        "code": 1 / 3,
        "if": 1 / 3,
    }
    assert training["gap_alpha"] == 1.0
    assert training["gap_factor_bounds"] == [0.05, 20.0]
    assert training["reward_refresh"] is True
    assert recipe["comparability"]["include_in_qwen3_four_task_main_table"] is False


def test_full_train_dry_run_contains_non_degenerate_open_mopd(tmp_path):
    script = FULL / "train.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OPEN_MOPD_FULL_ROOT": str(tmp_path)},
    )
    command = result.stdout

    assert "actor_rollout_ref.actor.ppo_mini_batch_size=256" in command
    assert "data.train_batch_size=1024" in command
    assert "actor_rollout_ref.rollout.log_prob_top_k=16" in command
    assert "actor_rollout_ref.actor.opd_refresh_advantage=True" in command
    assert "mt_opd.reward_scale_direction=multiply" in command
    assert "train_kwargs_by_data_source.nemotron_if_rl.max_tokens=2048" in command
    assert "trainer.total_training_steps=600" in command


def test_full_eval_dry_run_covers_paper_suite(tmp_path):
    script = FULL / "evaluate.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OPEN_MOPD_FULL_ROOT": str(tmp_path)},
    )

    for dataset in ("aime24", "aime25", "livecodebench_v5", "livecodebench_v6", "ifeval", "ifbench_test"):
        assert f"--dataset {dataset}" in result.stdout
    assert "--top-p 0.95" in result.stdout
    assert "--stop-token-ids 128012" in result.stdout


def test_full_summary_macro_averages_domains(tmp_path):
    values = {
        "aime24": 10.0,
        "aime25": 30.0,
        "livecodebench_v5": 20.0,
        "livecodebench_v6": 40.0,
        "ifeval": 50.0,
        "ifbench_test": 70.0,
    }
    for name, value in values.items():
        path = tmp_path / name / "scores.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"pct": value}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(FULL / "summarize.py"), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["domains"] == {"math": 20.0, "code": 30.0, "if": 60.0}
    assert summary["total"] == 110.0 / 3.0
