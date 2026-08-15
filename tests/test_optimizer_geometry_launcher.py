"""CPU smoke tests for every optimizer x algorithm launch configuration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

NUM_GPUS = 0
REPO = Path(__file__).parents[1]
LAUNCHER = REPO / "examples" / "optimizer_geometry" / "run_experiment.sh"
M2RL_4T_LAUNCHER = REPO / "examples" / "optimizer_geometry" / "run_m2rl_4t.sh"
QWEN_PRESET = REPO / "examples" / "optimizer_geometry" / "run-qwen3-1.7B-student-8B-teacher.sh"
EVAL_LAUNCHER = REPO / "examples" / "optimizer_geometry" / "evaluate_single_task.sh"
MIXED_LOSS_SWEEP = REPO / "examples" / "optimizer_geometry" / "run_mixed_loss_sweep.sh"
SINGLE_TASK_RL = REPO / "examples" / "optimizer_geometry" / "run_single_task_rl.sh"
SINGLE_TASK_OPD = REPO / "examples" / "optimizer_geometry" / "run_single_task_opd.sh"
SINGLE_TASK_MATRIX = REPO / "examples" / "optimizer_geometry" / "run_single_task_matrix.sh"
OPD_CELL_SCRIPTS = [
    (task, optimizer, REPO / "examples" / "optimizer_geometry" / f"run_opd_{task}_{optimizer}.sh")
    for task in ("math", "code", "science")
    for optimizer in ("adamw", "sgd", "muon")
]


@pytest.fixture(autouse=True)
def _do_not_forward_ci_gpu_count_to_launchers(monkeypatch):
    """Keep CI's test-resource count separate from launcher GPU configuration."""

    monkeypatch.delenv("NUM_GPUS", raising=False)


def _fixtures(tmp_path: Path) -> dict[str, str]:
    hf = tmp_path / "hf"
    checkpoint = tmp_path / "checkpoint"
    hf.mkdir()
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("1\n")
    model_config = tmp_path / "model.sh"
    model_config.write_text("MODEL_ARGS=(--num-layers 1 --hidden-size 8)\n")
    data = tmp_path / "math.jsonl"
    data.write_text('{"prompt":"1+1?","label":"2","metadata":{"rm_type":"deepscaler"}}\n')
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "sampling": {"strategy": "uniform", "unit": "batch"},
                "sources": [{"name": "math", "path": str(data), "rm_type": "deepscaler"}],
            }
        )
    )
    teachers = tmp_path / "teachers.yaml"
    teachers.write_text(yaml.safe_dump({"teachers": {"math": "http://teacher/generate"}}))
    reward = tmp_path / "rewards.yaml"
    reward.write_text("routes:\n  deepscaler: {}\n")
    return {
        "MODEL_CONFIG": str(model_config),
        "HF_CHECKPOINT": str(hf),
        "LOAD_CHECKPOINT": str(checkpoint),
        "DATA_MANIFEST": str(manifest),
        "TEACHER_CONFIG": str(teachers),
        "REWARD_CONFIG": str(reward),
        "OUTPUT_ROOT": str(tmp_path / "output"),
    }


@pytest.mark.unit
@pytest.mark.parametrize("launcher", [LAUNCHER, EVAL_LAUNCHER])
def test_local_ray_cleanup_targets_only_the_started_supervisor(launcher):
    source = launcher.read_text()

    assert "ray stop --force" not in source
    assert "--block &" in source
    assert "RAY_START_PID=$!" in source
    assert 'kill -TERM "${RAY_START_PID}"' in source
    assert 'ray job list --address "${RAY_ADDRESS}"' in source


@pytest.mark.unit
@pytest.mark.parametrize("optimizer", ["adamw", "sgd", "muon"])
@pytest.mark.parametrize("algorithm", ["grpo", "ppo", "opd", "sft_opd", "grpo_opd", "ppo_opd"])
def test_launcher_dry_run_covers_full_matrix(tmp_path, optimizer, algorithm):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    if algorithm == "sft_opd":
        sft = tmp_path / "sft.jsonl"
        sft.write_text('{"messages":[{"role":"user","content":"1+1?"},{"role":"assistant","content":"2"}]}\n')
        Path(env["DATA_MANIFEST"]).write_text(
            yaml.safe_dump(
                {
                    "sampling": {"strategy": "stratified", "unit": "prompt"},
                    "sources": [
                        {
                            "name": "math_opd",
                            "path": str(tmp_path / "math.jsonl"),
                            "rm_type": "deepscaler",
                            "weight": 0.5,
                            "metadata": {"task_name": "math", "training_mode": "opd"},
                        },
                        {
                            "name": "math_sft",
                            "path": str(sft),
                            "rm_type": "deepscaler",
                            "weight": 0.5,
                            "metadata": {"task_name": "math", "training_mode": "sft"},
                        },
                    ],
                }
            )
        )
    env.update(
        {
            "OPTIMIZER": optimizer,
            "ALGORITHM": algorithm,
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "RUN_NAME": f"{algorithm}_{optimizer}",
        }
    )

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "Experiment command:" in result.stdout
    assert f"--optimizer {('adam' if optimizer == 'adamw' else optimizer)}" in result.stdout
    expected_estimator = "grpo" if algorithm in {"opd", "sft_opd"} else algorithm.removesuffix("_opd")
    assert f"--advantage-estimator {expected_estimator}" in result.stdout
    assert "--geometry-output-dir" in result.stdout
    if algorithm in {"grpo", "ppo"}:
        assert "--rollout-max-response-len 8192" in result.stdout
        assert "--max-tokens-per-gpu 10240" in result.stdout
    assert '--apply-chat-template-kwargs \\{\\"enable_thinking\\":false\\}' in result.stdout
    if algorithm in {"opd", "sft_opd", "grpo_opd", "ppo_opd"}:
        assert "--use-opd --opd-type sglang" in result.stdout
        assert "--kl-coef 0.0 --kl-loss-coef 0.0" in result.stdout
    if algorithm == "sft_opd":
        assert "--custom-loss-function-path slime_plugins.m2rl.hybrid.hybrid_loss_function" in result.stdout
        assert "--loss-mask-type qwen3" in result.stdout
        assert "--n-samples-per-prompt 1" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    ("algorithm", "optimizer", "expected_lr"),
    [
        ("grpo", "adamw", "2.5e-7"),
        ("grpo", "sgd", "2.5e-2"),
        ("grpo", "muon", "2.5e-7"),
        ("opd", "adamw", "2.5e-7"),
        ("opd", "sgd", "2.5e-3"),
        ("opd", "muon", "2.5e-7"),
    ],
)
def test_responsive16_primary_profile(tmp_path, algorithm, optimizer, expected_lr):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update(
        {
            "ALGORITHM": algorithm,
            "OPTIMIZER": optimizer,
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
        }
    )

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert f"--lr {expected_lr}" in command
    assert "--weight-decay 0.0" in command
    assert "--lr-decay-style constant" in command
    assert "--lr-warmup-iters 0" in command
    assert "--clip-grad 1.0" in command
    assert "--num-rollout 3200" in command
    assert "--rollout-batch-size 16" in command
    assert "--n-samples-per-prompt 4" in command
    assert "--global-batch-size 64" in command
    assert "--rollout-max-prompt-len 2048" in command
    assert "--rollout-max-response-len 8192" in command
    assert "--max-tokens-per-gpu 10240" in command
    assert "--rollout-temperature 1.0" in command
    assert "--rollout-top-p 1.0" in command
    assert "--rollout-top-k -1" in command
    assert "--sglang-enable-deterministic-inference" in command
    if optimizer == "sgd":
        assert "--sgd-momentum 0.0" in command
    if optimizer == "muon":
        assert "--adam-beta1 0.9" in command
        assert "--adam-beta2 0.9987381276" in command
        assert "--adam-eps 1e-8" in command
        assert "--muon-momentum 0.95" in command
        assert "--muon-num-ns-steps 5" in command
        assert "--muon-extra-scale-factor 0.2" in command
    if algorithm == "grpo":
        assert "--use-tis" in command
        assert "--tis-clip 2.0" in command
    else:
        assert "--use-tis" not in command
        assert "--eps-clip 0.2" in command
        assert "--eps-clip-high 0.2" in command


@pytest.mark.unit
@pytest.mark.parametrize(
    ("profile", "batch", "updates", "global_batch", "lr", "beta2", "geometry_interval", "eval_interval"),
    [
        ("responsive8", 8, 6400, 32, "1.8e-7", "0.9993688646", 32, 640),
        ("responsive16", 16, 3200, 64, "2.5e-7", "0.9987381276", 16, 320),
        ("reference256", 256, 200, 1024, "1e-6", "0.98", 1, 20),
    ],
)
def test_batch_profiles_keep_prompt_budget_and_scale_adam(
    tmp_path, profile, batch, updates, global_batch, lr, beta2, geometry_interval, eval_interval
):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update(
        {
            "BATCH_PROFILE": profile,
            "ALGORITHM": "grpo",
            "OPTIMIZER": "adamw",
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
        }
    )

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert f"--rollout-batch-size {batch}" in command
    assert f"--num-rollout {updates}" in command
    assert f"--global-batch-size {global_batch}" in command
    assert f"--lr {lr}" in command
    assert f"--adam-beta2 {beta2}" in command
    assert f"--geometry-interval {geometry_interval}" in command
    assert f"--eval-interval {eval_interval}" not in command  # no eval fixture was requested
    assert f"prompt_budget={batch * updates}" in command


@pytest.mark.unit
def test_batch_profile_rejects_mismatched_manual_batch(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update(
        {
            "BATCH_PROFILE": "responsive16",
            "ROLLOUT_BATCH_SIZE": "8",
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
        }
    )

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 2
    assert "requires ROLLOUT_BATCH_SIZE=16" in result.stderr


@pytest.mark.unit
def test_dataset_epoch_profile_uses_64_prompts_once_and_runtime_tail(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update(
        {
            "ALGORITHM": "opd",
            "BATCH_PROFILE": "opd64x1",
            "N_SAMPLES_PER_PROMPT": "1",
            "NUM_EPOCH": "1",
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
        }
    )

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "--num-epoch 1" in result.stdout
    assert "--include-epoch-tail" in result.stdout
    assert "--num-rollout" not in result.stdout
    assert "--rollout-batch-size 64" in result.stdout
    assert "--n-samples-per-prompt 1" in result.stdout
    assert "--global-batch-size 64" in result.stdout
    assert "usable prompt count and final partial batch are derived at runtime" in result.stdout


@pytest.mark.unit
def test_fixed_profile_rejects_multiple_updates_per_rollout(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update(
        {
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "GLOBAL_BATCH_SIZE": "512",
        }
    )

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 2
    assert "requires GLOBAL_BATCH_SIZE" in result.stderr


@pytest.mark.unit
def test_m2rl_4t_preset_uses_prepared_manifest_without_agent(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update({"DRY_RUN": "1", "CHECK_RUNTIME_DEPS": "0", "RUN_NAME": "m2rl_4t_smoke"})

    result = subprocess.run(["bash", str(M2RL_4T_LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "M2RL-4T manifest:" in result.stdout
    assert "--prompt-data" in result.stdout
    assert "workbench" not in Path(env["DATA_MANIFEST"]).read_text().lower()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("FRESH_START", "maybe", "FRESH_START must be 0"),
        ("DRY_RUN", "yes", "DRY_RUN must be 0 or 1"),
    ],
)
def test_launcher_rejects_ambiguous_run_modes(tmp_path, variable, value, message):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update({"DRY_RUN": "1", variable: value})

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.unit
def test_qwen3_1_7b_student_8b_teacher_preset_dry_run(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.pop("MODEL_CONFIG")
    env.pop("TEACHER_CONFIG")
    env.update(
        {
            "OPTIMIZER": "adamw",
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "AVAILABLE_CUDA_DEVICES": "0,1,2,3,4",
            "TRAIN_GPU_COUNT": "4",
            "RUN_NAME": "qwen_preset",
        }
    )

    result = subprocess.run(["bash", str(QWEN_PRESET)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "Student: Qwen3-1.7B" in result.stdout
    assert "Teacher: Qwen3-8B" in result.stdout
    assert "--num-layers 28" in result.stdout
    assert "--hidden-size 2048" in result.stdout
    assert "--opd-type sglang" in result.stdout
    assert '--apply-chat-template-kwargs \\{\\"enable_thinking\\":false\\}' in result.stdout
    assert "--actor-num-gpus-per-node 4" in result.stdout
    assert "--sglang-max-running-requests 44" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["grpo", "ppo", "opd"])
def test_rl_launcher_rejects_thinking_mode_override(tmp_path, algorithm):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update(
        {
            "ALGORITHM": algorithm,
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "APPLY_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":true}',
        }
    )

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 2
    assert "GRPO/PPO and OPD experiments require enable_thinking=false" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize("profile", ["responsive8", "responsive16", "reference256"])
def test_qwen_preset_uses_32k_math_eval_concurrency_cap_across_batch_profiles(tmp_path, profile):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.pop("MODEL_CONFIG")
    env.pop("TEACHER_CONFIG")
    env.update(
        {
            "ALGORITHM": "grpo",
            "TASK": "math",
            "BATCH_PROFILE": profile,
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "AVAILABLE_CUDA_DEVICES": "0,1,2,3",
            "TRAIN_GPU_COUNT": "4",
        }
    )

    result = subprocess.run(["bash", str(QWEN_PRESET)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "--rollout-max-response-len 8192" in result.stdout
    assert "--max-tokens-per-gpu 10240" in result.stdout
    assert "--eval-max-response-len 32768" in result.stdout
    assert "--sglang-max-running-requests 12" in result.stdout


@pytest.mark.unit
def test_qwen_preset_raises_max_running_requests_for_4096_cap(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.pop("MODEL_CONFIG")
    env.pop("TEACHER_CONFIG")
    env.update(
        {
            "ALGORITHM": "grpo",
            "MAX_PROMPT_LEN": "2048",
            "MAX_RESPONSE_LEN": "4096",
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "AVAILABLE_CUDA_DEVICES": "0,1,2,3",
            "TRAIN_GPU_COUNT": "4",
        }
    )

    result = subprocess.run(["bash", str(QWEN_PRESET)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "--sglang-max-running-requests 72" in result.stdout


@pytest.mark.unit
def test_qwen_preset_rejects_invalid_gpu_preflight_mode(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update(
        {
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "AVAILABLE_CUDA_DEVICES": "0,1,2,3,4",
            "GPU_PREFLIGHT": "maybe",
        }
    )

    result = subprocess.run(["bash", str(QWEN_PRESET)], env=env, text=True, capture_output=True)

    assert result.returncode == 2
    assert "GPU_PREFLIGHT must be 0 or 1" in result.stderr


@pytest.mark.unit
def test_qwen3_1_7b_preset_selects_4b_thinking_teacher(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.pop("MODEL_CONFIG")
    env.pop("TEACHER_CONFIG")
    env.update(
        {
            "TEACHER": "qwen3-4b-thinking-2507",
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "AVAILABLE_CUDA_DEVICES": "0,1,2,3,4",
            "TRAIN_GPU_COUNT": "4",
            "RUN_NAME": "qwen_4b_teacher_preset",
        }
    )

    result = subprocess.run(["bash", str(QWEN_PRESET)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "Teacher: Qwen3-4B-Thinking-2507" in result.stdout
    assert "http://127.0.0.1:13142/generate" in result.stdout


@pytest.mark.unit
def test_qwen_preset_rejects_overlapping_student_and_teacher_gpus(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    env.update(
        {
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "AVAILABLE_CUDA_DEVICES": "0,1,2",
            "TRAIN_CUDA_VISIBLE_DEVICES": "0,1",
            "TEACHER_CUDA_VISIBLE_DEVICES": "1",
        }
    )
    env.pop("NUM_GPUS", None)

    result = subprocess.run(["bash", str(QWEN_PRESET)], env=env, text=True, capture_output=True)

    assert result.returncode == 2
    assert "overlap at device 1" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(("task", "eval_max_response_len"), [(None, 16384), ("math", 32768)])
def test_single_task_eval_launcher_dry_run(tmp_path, task, eval_max_response_len):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "defaults": {"max_response_len": eval_max_response_len},
                    "datasets": [
                        {
                            "name": "math",
                            "path": str(tmp_path / "math.jsonl"),
                            "rm_type": "deepscaler",
                        }
                    ],
                }
            }
        )
    )
    env.update(
        {
            "EVAL_CONFIG": str(eval_config),
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
            "NUM_GPUS": "2",
        }
    )
    if task is not None:
        env["TASK"] = task

    result = subprocess.run(["bash", str(EVAL_LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "Evaluation command:" in result.stdout
    assert "--num-rollout 0" in result.stdout
    assert f"--eval-config {eval_config}" in result.stdout
    assert f"--eval-max-response-len {eval_max_response_len}" in result.stdout
    assert "--eval-max-concurrency 48" in result.stdout
    assert "--sglang-max-running-requests 44" in result.stdout


@pytest.mark.unit
def test_training_launcher_passes_frozen_eval_cadence_cap_and_concurrency(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "defaults": {"max_response_len": 16384},
                    "datasets": [
                        {
                            "name": "math",
                            "path": str(tmp_path / "math.jsonl"),
                            "rm_type": "deepscaler",
                        }
                    ],
                }
            }
        )
    )
    env.update(
        {
            "ALGORITHM": "opd",
            "EVAL_CONFIG": str(eval_config),
            "EVAL_INTERVAL": "50",
            "EVAL_MAX_RESPONSE_LEN": "16384",
            "EVAL_MAX_CONCURRENCY": "48",
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
        }
    )

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "--eval-interval 50" in result.stdout
    assert "--eval-max-response-len 16384" in result.stdout
    assert "--eval-max-concurrency 48" in result.stdout


@pytest.mark.unit
def test_training_launcher_rejects_eval_config_outside_frozen_16k_cap(tmp_path):
    env = os.environ.copy()
    env.update(_fixtures(tmp_path))
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "defaults": {"max_response_len": 8192},
                    "datasets": [{"name": "math", "path": str(tmp_path / "math.jsonl")}],
                }
            }
        )
    )
    env.update(
        {
            "EVAL_CONFIG": str(eval_config),
            "EVAL_MAX_RESPONSE_LEN": "16384",
            "DRY_RUN": "1",
            "CHECK_RUNTIME_DEPS": "0",
        }
    )

    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    assert "the frozen experiment requires 16384" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    ("relative_path", "expected_max_response_len"),
    [
        ("math/math_eval.yaml", 32768),
        ("math/math_eval_aime24.yaml", 32768),
        ("math/math_eval_math500.yaml", 32768),
        ("math/math_eval_aime24_math500.yaml", 32768),
        ("code/code_eval.yaml", 16384),
        ("code/code_eval_final.yaml", 16384),
        ("science/science_eval.yaml", 16384),
    ],
)
def test_committed_single_task_eval_configs_use_task_specific_frozen_cap(relative_path, expected_max_response_len):
    config = yaml.safe_load((REPO / "data" / "m2rl" / "single_task" / relative_path).read_text())

    assert config["eval"]["defaults"]["max_response_len"] == expected_max_response_len


@pytest.mark.unit
def test_mixed_loss_sweep_routes_manifests_and_coefficients(tmp_path):
    config_root = tmp_path / "prepared"
    task_dir = config_root / "math"
    task_dir.mkdir(parents=True)
    (task_dir / "math_on_policy.yaml").touch()
    (task_dir / "math_sft_opd.yaml").touch()
    (task_dir / "math_eval.yaml").touch()
    combined_eval = task_dir / "math_eval_aime24_math500.yaml"
    combined_eval.touch()
    fake_launcher = tmp_path / "launcher.sh"
    fake_launcher.write_text(
        'printf \'%s|%s|%s|%s|%s|%s|%s|%s\\n\' "$ALGORITHM" "$DATA_MANIFEST" '
        '"$OPD_KL_COEF" "$SFT_LOSS_COEF" "$HYBRID_OPD_LOSS_COEF" '
        '"$OPD_TASK_REWARD_WEIGHT" "$EVAL_CONFIG" "${MAX_RESPONSE_LEN:-}"\n'
    )
    env = os.environ.copy()
    env.update(
        {
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            "MIXTURE_SETTINGS": "sft_opd:1:2:0.5:0 grpo_opd:0.25:0:0:1 ppo:0:0:0:1",
        }
    )

    result = subprocess.run(["bash", str(MIXED_LOSS_SWEEP)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert f"sft_opd|{task_dir / 'math_sft_opd.yaml'}|1|2|0.5|0" in result.stdout
    assert f"grpo_opd|{task_dir / 'math_on_policy.yaml'}|0.25|0|0|1" in result.stdout
    assert f"ppo|{task_dir / 'math_on_policy.yaml'}|0|0|0|1|{combined_eval}|8192" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    ("script", "expected_algorithm", "expected_teacher", "expected_seeds"),
    [
        (SINGLE_TASK_RL, "grpo", "", {"42"}),
        (SINGLE_TASK_OPD, "opd", "qwen3-8b", {"42"}),
    ],
)
def test_single_task_three_optimizer_wrappers(tmp_path, script, expected_algorithm, expected_teacher, expected_seeds):
    config_root = tmp_path / "single_task"
    task_dir = config_root / "science"
    task_dir.mkdir(parents=True)
    (task_dir / "science_on_policy.yaml").write_text("sources: []\n")
    (task_dir / "science_eval.yaml").write_text("eval: {}\n")
    (config_root / "single_task_index.json").write_text("{}\n")
    fake_launcher = tmp_path / "launcher.sh"
    fake_launcher.write_text(
        'printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n" '
        '"$TASK" "$ALGORITHM" "$OPTIMIZER" "${TEACHER:-}" "$WANDB_ENTITY" '
        '"$EXPERIMENT_DATA_INDEX" "$SEED" "$NUM_EPOCH" "${NUM_ROLLOUT:-}" '
        '"${TARGET_PROMPT_BUDGET:-}"\n'
    )
    env = os.environ.copy()
    env.update(
        {
            "TASK": "Science",
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            "OPTIMIZERS": "adamw sgd muon",
        }
    )

    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    cells = [line for line in result.stdout.splitlines() if line.startswith("science|")]
    assert len(cells) == 3 * len(expected_seeds)
    assert {line.split("|")[2] for line in cells} == {"adamw", "sgd", "muon"}
    assert all(line.split("|")[1] == expected_algorithm for line in cells)
    assert all(line.split("|")[3] == expected_teacher for line in cells)
    assert all(line.split("|")[4] == "zsqzz" for line in cells)
    assert all(line.split("|")[5] == str(config_root / "single_task_index.json") for line in cells)
    assert {line.split("|")[6] for line in cells} == expected_seeds
    assert all(line.split("|")[7:] == ["1", "", ""] for line in cells)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("invalid_env", "expected_error"),
    [
        ({"SEEDS": "42 43"}, "requires exactly SEEDS=42"),
        ({"SEED": "43"}, "requires SEED=42"),
        ({"NUM_EPOCH": "2"}, "exactly one usable dataset epoch"),
        ({"TARGET_PROMPT_BUDGET": "51200"}, "do not accept NUM_ROLLOUT or TARGET_PROMPT_BUDGET"),
        ({"NUM_ROLLOUT": "3200"}, "do not accept NUM_ROLLOUT or TARGET_PROMPT_BUDGET"),
        ({"MAX_RESPONSE_LEN": "32768"}, "require training MAX_RESPONSE_LEN=8192"),
        ({"MAX_TOKENS_PER_GPU": "10239"}, "must be at least prompt+response=10240"),
    ],
)
def test_single_task_rl_rejects_non_frozen_seed_or_run_length(tmp_path, invalid_env, expected_error):
    config_root = tmp_path / "single_task"
    task_dir = config_root / "math"
    task_dir.mkdir(parents=True)
    (task_dir / "math_on_policy.yaml").write_text("sources: []\n")
    (task_dir / "math_eval_aime24_math500.yaml").write_text("eval: {}\n")
    (config_root / "single_task_index.json").write_text("{}\n")
    fake_launcher = tmp_path / "launcher.sh"
    fake_launcher.write_text("exit 0\n")
    env = os.environ.copy()
    for name in (
        "SEED",
        "SEEDS",
        "NUM_EPOCH",
        "NUM_ROLLOUT",
        "TARGET_PROMPT_BUDGET",
        "MAX_RESPONSE_LEN",
        "MAX_TOKENS_PER_GPU",
    ):
        env.pop(name, None)
    env.update(
        {
            "TASK": "math",
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            "OPTIMIZERS": "adamw",
            **invalid_env,
        }
    )

    result = subprocess.run(["bash", str(SINGLE_TASK_RL)], env=env, text=True, capture_output=True)

    assert result.returncode == 2
    assert expected_error in result.stderr


@pytest.mark.unit
def test_single_task_matrix_defaults_to_seed_42_and_one_dataset_epoch(tmp_path):
    config_root = tmp_path / "single_task"
    task_dir = config_root / "math"
    task_dir.mkdir(parents=True)
    (task_dir / "math_on_policy.yaml").write_text("sources: []\n")
    (task_dir / "math_eval.yaml").write_text("eval: {}\n")
    combined_eval = task_dir / "math_eval_aime24_math500.yaml"
    combined_eval.write_text("eval: {}\n")
    fake_launcher = tmp_path / "launcher.sh"
    fake_launcher.write_text(
        'printf "%s|%s|%s|%s|%s|%s|%s|%s\\n" "$TASK" "$ALGORITHM" "$SEED" '
        '"$NUM_EPOCH" "$EVAL_CONFIG" "$MAX_RESPONSE_LEN" "$MAX_TOKENS_PER_GPU" '
        '"$SGLANG_MAX_RUNNING_REQUESTS"\n'
    )
    env = os.environ.copy()
    for name in ("SEED", "SEEDS", "NUM_EPOCH", "NUM_ROLLOUT", "TARGET_PROMPT_BUDGET"):
        env.pop(name, None)
    env.update(
        {
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            "TASKS": "math",
            "TEACHERS": "qwen3-8b",
            "ALGORITHMS": "grpo",
            "OPTIMIZERS": "adamw",
        }
    )

    result = subprocess.run(["bash", str(SINGLE_TASK_MATRIX)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert f"math|grpo|42|1|{combined_eval}|8192|10240|12" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["grpo", "ppo"])
def test_single_task_math_rl_uses_8k_train_and_32k_combined_eval_rollouts(tmp_path, algorithm):
    config_root = tmp_path / "single_task"
    task_dir = config_root / "math"
    task_dir.mkdir(parents=True)
    (task_dir / "math_on_policy.yaml").write_text("sources: []\n")
    combined_eval = task_dir / "math_eval_aime24_math500.yaml"
    combined_eval.write_text("eval: {}\n")
    (config_root / "single_task_index.json").write_text("{}\n")
    fake_launcher = tmp_path / "launcher.sh"
    fake_launcher.write_text(
        'printf "%s|%s|%s|%s|%s|%s|%s\\n" "$ALGORITHM" "$EVAL_CONFIG" '
        '"$MAX_RESPONSE_LEN" "$MAX_TOKENS_PER_GPU" "$SGLANG_MAX_RUNNING_REQUESTS" '
        '"$EVAL_MAX_RESPONSE_LEN" "$EVAL_MAX_CONCURRENCY"\n'
    )
    env = os.environ.copy()
    for name in (
        "EVAL_CONFIG",
        "MAX_RESPONSE_LEN",
        "MAX_TOKENS_PER_GPU",
        "SGLANG_MAX_RUNNING_REQUESTS",
        "EVAL_MAX_RESPONSE_LEN",
        "EVAL_MAX_CONCURRENCY",
    ):
        env.pop(name, None)
    env.update(
        {
            "TASK": "math",
            "RL_ALGORITHM": algorithm,
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            "OPTIMIZERS": "adamw",
        }
    )

    result = subprocess.run(["bash", str(SINGLE_TASK_RL)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == (
        f"{algorithm}|{combined_eval}|8192|10240|12|32768|48"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("selection", "expected_config"),
    [
        ("aime24", "math_eval_aime24.yaml"),
        ("math500", "math_eval_math500.yaml"),
        ("aime24 math500", "math_eval_aime24_math500.yaml"),
        ("math-500,aime-2024", "math_eval_aime24_math500.yaml"),
    ],
)
def test_single_task_opd_selects_optional_math_eval_configs(tmp_path, selection, expected_config):
    config_root = tmp_path / "single_task"
    task_dir = config_root / "math"
    task_dir.mkdir(parents=True)
    (task_dir / "math_on_policy.yaml").write_text("sources: []\n")
    for name in ("math_eval_aime24.yaml", "math_eval_math500.yaml", "math_eval_aime24_math500.yaml"):
        (task_dir / name).write_text("eval: {}\n")
    (config_root / "single_task_index.json").write_text("{}\n")
    fake_launcher = tmp_path / "launcher.sh"
    fake_launcher.write_text('printf "%s|%s|%s\n" "$TASK" "$EVAL_CONFIG" "$EVAL_MAX_RESPONSE_LEN"\n')
    env = os.environ.copy()
    env.pop("EVAL_CONFIG", None)
    env.update(
        {
            "TASK": "math",
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            "OPTIMIZERS": "adamw",
            "MATH_EVAL_DATASETS": selection,
        }
    )

    result = subprocess.run(["bash", str(SINGLE_TASK_OPD)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert f"math|{task_dir / expected_config}|32768" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize(("task", "optimizer", "script"), OPD_CELL_SCRIPTS)
def test_each_frozen_opd_cell_has_its_own_single_seed_script(tmp_path, task, optimizer, script):
    config_root = tmp_path / "single_task"
    for task_name in ("math", "code", "science"):
        task_dir = config_root / task_name
        task_dir.mkdir(parents=True)
        (task_dir / f"{task_name}_on_policy.yaml").write_text("sources: []\n")
        (task_dir / f"{task_name}_eval.yaml").write_text("eval: {}\n")
    (config_root / "single_task_index.json").write_text("{}\n")
    fake_launcher = tmp_path / "launcher.sh"
    fake_launcher.write_text(
        'printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n" '
        '"$TASK" "$ALGORITHM" "$OPTIMIZER" "$SEED" "$BATCH_PROFILE" '
        '"$ROLLOUT_BATCH_SIZE" "$N_SAMPLES_PER_PROMPT" "$NUM_EPOCH" '
        '"$MAX_PROMPT_LEN" "$MAX_RESPONSE_LEN" "$MAX_TOKENS_PER_GPU" '
        '"$EVAL_INTERVAL" "$EVAL_MAX_RESPONSE_LEN" "$EVAL_MAX_CONCURRENCY" '
        '"$SGLANG_MAX_RUNNING_REQUESTS" "$APPLY_CHAT_TEMPLATE_KWARGS"\n'
    )
    env = os.environ.copy()
    env.update(
        {
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            # Each standalone script must neutralize an ambient matrix configuration.
            "TASK": "science",
            "OPTIMIZERS": "adamw sgd muon",
            "SEED": "99",
            "SEEDS": "98 99",
            "NUM_ROLLOUT": "3200",
            "TARGET_PROMPT_BUDGET": "51200",
            "BATCH_PROFILE": "reference256",
            "MAX_RESPONSE_LEN": "8192",
            "EVAL_INTERVAL": "999",
            "EVAL_MAX_RESPONSE_LEN": "8192",
            "EVAL_MAX_CONCURRENCY": "96",
        }
    )

    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    cells = [line for line in result.stdout.splitlines() if line.startswith(f"{task}|")]
    expected_eval_max_response_len = 32768 if task == "math" else 16384
    assert cells == [
        f"{task}|opd|{optimizer}|42|opd64x1|64|1|1|2048|4096|10240|50|{expected_eval_max_response_len}|48|72|"
        '{"enable_thinking":false}'
    ]


@pytest.mark.unit
@pytest.mark.parametrize("script", [SINGLE_TASK_RL, SINGLE_TASK_OPD])
def test_single_task_code_wrapper_requires_independent_eval_by_default(tmp_path, script):
    config_root = tmp_path / "single_task"
    task_dir = config_root / "code"
    task_dir.mkdir(parents=True)
    (task_dir / "code_on_policy.yaml").write_text("sources: []\n")
    (config_root / "single_task_index.json").write_text("{}\n")
    fake_launcher = tmp_path / "launcher.sh"
    fake_launcher.write_text('printf "%s|%s\n" "$TASK" "${EVAL_CONFIG:-unset}"\n')
    env = os.environ.copy()
    env.update(
        {
            "TASK": "code",
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            "OPTIMIZERS": "adamw",
        }
    )

    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)

    assert result.returncode == 2
    assert "No independent evaluation config is ready for TASK=code" in result.stderr
    assert "prepare_livecodebench_eval.py" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize("script", [SINGLE_TASK_RL, SINGLE_TASK_OPD])
def test_single_task_code_wrapper_uses_prepared_livecodebench_eval(tmp_path, script):
    config_root = tmp_path / "single_task"
    task_dir = config_root / "code"
    task_dir.mkdir(parents=True)
    (task_dir / "code_on_policy.yaml").write_text("sources: []\n")
    eval_config = task_dir / "code_eval.yaml"
    eval_config.write_text("eval: {}\n")
    (config_root / "single_task_index.json").write_text("{}\n")
    fake_launcher = tmp_path / "launcher.sh"
    fake_launcher.write_text('printf "%s|%s\n" "$TASK" "$EVAL_CONFIG"\n')
    env = os.environ.copy()
    env.update(
        {
            "TASK": "code",
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            "OPTIMIZERS": "adamw",
            "SEEDS": "42",
        }
    )

    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert f"code|{eval_config}" in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
