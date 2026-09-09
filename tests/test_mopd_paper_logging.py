"""Native verifier observation stays separate from the distillation objective."""

import asyncio
import base64
import json
import pickle
import zlib
from types import SimpleNamespace

import pytest

from slime.ray.rollout import _compute_training_reward_metrics
from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.types import Sample
from slime_plugins.mopd import eval as mopd_eval
from slime_plugins.mopd.rollout import _observe_task_rewards
from slime_plugins.m2rl.rewards import _functional_test_program, _test_output_matches, open_mopd_lcb_row
from examples.mopd_gpas.analyze_paper import analyze, teacher_distance_pairs
from train import _eval_only_num_updates

NUM_GPUS = 0


@pytest.mark.parametrize("checkpoint_step,expected", [(None, 0), (0, 0), (250, 250)])
def test_standalone_eval_uses_loaded_checkpoint_clock(checkpoint_step, expected, tmp_path):
    from slime.ray.rollout import _save_eval_artifacts

    args = SimpleNamespace(
        eval_checkpoint_step=checkpoint_step,
        start_rollout_id=0,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
        eval_artifact_dir=str(tmp_path),
        eval_datasets=[],
    )
    step = _eval_only_num_updates(args)
    metrics = {"eval/num_updates": step, "eval/model_version": step, "eval/phase": "eval_only"}
    sample = Sample(prompt="question", response="answer", response_length=1, reward=1.0)
    _save_eval_artifacts(args, 0, {"math": {"samples": [sample], "rewards": [1.0]}}, metrics)
    assert step == expected
    record = json.loads((tmp_path / "index.jsonl").read_text().splitlines()[0])
    assert record["num_updates"] == expected
    assert (tmp_path / "math" / f"updates_{expected:08d}_eval_only.jsonl").is_file()


def test_eval_logs_rewards_as_observations_without_loss_use(monkeypatch, tmp_path):
    from slime.ray import rollout

    args = SimpleNamespace(
        custom_eval_rollout_log_function_path=None,
        eval_artifact_dir=str(tmp_path),
        eval_datasets=[],
        log_passrate=False,
        wandb_always_use_train_step=False,
        log_reward_category=None,
        advantage_estimator="ppo",
        reward_key=None,
        use_opd=False,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
    )
    sample = Sample(prompt="question", response="answer", response_length=1, reward=1.0)
    logged = []
    monkeypatch.setattr(rollout.logging_utils, "log", lambda _args, metrics, **_kwargs: logged.append(metrics))
    metrics = rollout._log_eval_rollout_data(
        0,
        args,
        {"math": {"samples": [sample], "rewards": [1.0]}},
        {"eval/num_updates": 1, "eval/model_version": 1, "eval/phase": "eval_only"},
    )
    assert metrics["eval/math/reward/mean"] == 1.0
    assert metrics["eval/math/task_reward_observed"] == 1
    assert metrics["eval/math/reward_used_in_loss"] == 0
    assert metrics["eval/math/reward_loss_coefficient"] == 0.0
    assert logged[0] == metrics
    index = json.loads((tmp_path / "index.jsonl").read_text().splitlines()[0])
    assert index["metrics"]["eval/math/reward_used_in_loss"] == 0
    assert rollout._compute_training_reward_metrics(args, [sample])["reward_used_in_loss"] == 1


@pytest.mark.parametrize("custom_paper_loss", [False, True])
def test_native_logic_reward_is_logged_without_replacing_teacher_targets(custom_paper_loss):
    payload = {"meta_info": {"input_token_logprobs": [[-1.0, 7]]}}
    sample = Sample(
        response="<answer>Alice is a knight.</answer>",
        label={"names": ["Alice"], "roles": ["knight"]},
        reward=payload,
        metadata={"rm_type": "kk", "task_name": "logic"},
    )
    args = SimpleNamespace(
        use_opd=not custom_paper_loss, mopd_enabled=custom_paper_loss, opd_task_reward_weight=0.0, reward_key=None
    )

    assert asyncio.run(_observe_task_rewards(args, [sample])) >= 0
    metrics = _compute_training_reward_metrics(args, [sample])

    assert sample.reward is payload
    assert metrics["reward/logic/mean"] == 1
    assert metrics["task_reward_observed"] == 1
    assert metrics["reward_used_in_loss"] == 0


def test_paper_eval_reports_native_capability_and_truncation(monkeypatch):
    args = SimpleNamespace(custom_rm_path="teacher.targets", mopd_enabled=True, mopd_loss="student_topk")
    observed = []
    output = RolloutFnEvalOutput(
        data={
            "logic": {
                "samples": [
                    Sample(
                        status=Sample.Status.TRUNCATED,
                        metadata={"generation_max_new_tokens": 31768, "generation_context_length": 32768},
                    )
                ],
                "rewards": [0.0],
            }
        }
    )

    def generate(eval_args, rollout_id, data_source, evaluation):
        from slime_plugins.mopd.topk import uses_student_topk

        assert not uses_student_topk(eval_args)
        observed.append(eval_args.custom_rm_path)
        return output

    monkeypatch.setattr(mopd_eval, "default_generate_rollout", generate)
    result = mopd_eval.generate_capability_eval(args, 2, None, evaluation=True)

    assert args.custom_rm_path == "teacher.targets"
    assert args.mopd_enabled
    assert observed == ["slime_plugins.m2rl.rewards.reward"]
    assert result.metrics["eval/capability/logic"] == 0
    assert result.metrics["eval/capability/logic/truncation_rate"] == 1
    assert result.metrics["eval/capability/logic/generation_max_new_tokens_mean"] == 31768
    assert result.metrics["eval/capability/logic/generation_max_new_tokens_min"] == 31768
    assert result.metrics["eval/capability/logic/generation_context_length"] == 32768


def test_skip_training_verifiers_preserves_teacher_targets_without_fake_accuracy(monkeypatch):
    from slime_plugins.m2rl import rewards

    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not access verifiers or sandbox when observations are disabled")

    monkeypatch.setattr(rewards, "batched_reward", forbidden)
    payload = {"meta_info": {"input_token_logprobs": [[-1.0, 7]]}}
    sample = Sample(reward=payload, metadata={"task_name": "code", "rm_type": "unit_test"})
    args = SimpleNamespace(mopd_skip_task_rewards=True, mopd_enabled=True, use_opd=False,
                           opd_task_reward_weight=0.0, reward_key=None)
    assert asyncio.run(_observe_task_rewards(args, [sample])) == 0
    assert sample.reward is payload
    assert "task_reward_observed" not in sample.metadata
    metrics = _compute_training_reward_metrics(args, [sample])
    assert metrics["task_reward_observed"] == 0
    assert metrics["reward_used_in_loss"] == 0
    assert not any(key.startswith("reward/") for key in metrics)


def test_functional_code_wrapper_matches_taco_call_contract(monkeypatch, capsys):
    from io import StringIO
    import sys

    monkeypatch.setattr(sys, "stdin", StringIO("[1, 2]\n3\n"))
    # This hand-written test program is trusted; generated code runs only at
    # the configured external sandbox endpoint in the production path.
    program = _functional_test_program("class Solution:\n    def add(self, xs, n): return [x+n for x in xs]", "add")
    exec(program, {})
    assert _test_output_matches(capsys.readouterr().out, "[4,5]", functional=True)
    assert not _test_output_matches("[4,6]", "[4,5]", functional=True)


def test_official_lcb_private_cases_are_preserved_in_remote_evaluator_row():
    private = [{"input": "7\n", "output": "14\n", "testtype": "stdin"}]
    problem = {
        "question_id": "released_problem",
        "question_content": "Double the integer",
        "public_test_cases": json.dumps([{"input": "2\n", "output": "4\n", "testtype": "stdin"}]),
        "private_test_cases": base64.b64encode(zlib.compress(pickle.dumps(json.dumps(private)))).decode(),
        "metadata": "{}",
    }
    row = open_mopd_lcb_row({"metadata": json.dumps(problem)})
    tests = json.loads(json.loads(row["test"])["input_output"])
    assert set(row) == {"id", "content", "labels", "test"}
    assert tests == {"inputs": ["2\n", "7\n"], "outputs": ["4\n", "14\n"]}


def test_paper_analysis_exports_measured_clocks_and_all_study_figures(tmp_path):
    run = tmp_path / "runs/M-TK-DR"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)

    def events(name, rows):
        (metrics / f"{name}.jsonl").write_text("".join(json.dumps({"metrics": row}) + "\n" for row in rows))

    events(
        "mopd",
        [
            {"mopd/update": 1, "mopd/valid_response_tokens": 100, "mopd/total_gpu_seconds": 36},
            {"mopd/update": 2, "mopd/valid_response_tokens": 200, "mopd/total_gpu_seconds": 72},
        ],
    )
    events(
        "rollout",
        [
            {"rollout/step": 0, "rollout/reward/math/mean": 0.5},
            {"rollout/step": 0, "rollout/response_lengths": 100},
            {"rollout/step": 1, "rollout/reward/math/mean": 1.0},
            {"rollout/step": 1, "rollout/response_lengths": 200},
        ],
    )
    events("eval", [{"eval/step": 0, "eval/capability/math": 0.25}, {"eval/step": 2, "eval/capability/math": 0.75}])
    rows = [
        {"kind": "sparsity", "step": 2, "quantity": "update", "metrics": {"l2": 2.0, "energy90_fraction": 0.1}},
        {
            "kind": "overlap",
            "step": 2,
            "teacher_a": "math",
            "teacher_b": "code",
            "quantity": "update",
            "support_fraction": 0.05,
            "metrics": {"jaccard": 0.2},
        },
        {"kind": "teacher_distance", "step": 2, "teacher_a": "math", "teacher_b": "code", "metrics": {"js": 0.1}},
    ]
    (run / "paper").mkdir()
    (run / "paper/measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = analyze(tmp_path / "runs", tmp_path / "report")

    assert report["runs"][0]["status"] == "in_progress"
    assert report["curves"][0]["rollout/reward/math/mean"] == 0.5
    assert report["curves"][0]["rollout/response_lengths"] == 100
    assert report["capabilities"][-1]["valid_tokens"] == 300
    assert report["capabilities"][-1]["gpu_hours"] == pytest.approx(0.03)
    assert report["teacher_distance_pairs"][0]["metrics"]["subnetwork_distance"] == pytest.approx(0.8)
    assert set(report["figures"]) == {
        "normalization_capability",
        "rollout_reward",
        "update_sparsity",
        "teacher_overlap_000",
        "teacher_distance_subnetwork",
    }
    assert (tmp_path / "report/figures/update_sparsity.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "report/sparsity.csv").is_file()


def test_teacher_distance_pairs_do_not_match_different_prefix_draws():
    common = {"run": "M-TK-DR", "step": 3, "teacher_a": "math", "teacher_b": "code"}
    rows = [
        {**common, "kind": "teacher_distance", "probe_batch": 0, "metrics": {"js": 0.1}},
        {**common, "kind": "overlap", "probe_batch": 1, "metrics": {"jaccard": 0.5}},
    ]
    assert teacher_distance_pairs(rows) == []
