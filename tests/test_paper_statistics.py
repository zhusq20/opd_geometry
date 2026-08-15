"""Integration test for paper-ready tables, paired effects, and figures."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.optimizer_geometry.summarize_single_task import fixed_grid_auc

NUM_GPUS = 0
ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "examples" / "optimizer_geometry" / "paper_statistics.py"


@pytest.mark.unit
def test_fixed_grid_auc_reports_raw_and_update_normalized_values():
    raw, normalized = fixed_grid_auc({0: 0.2, 5: 0.6, 10: 0.2})

    assert raw == pytest.approx(4.0)
    assert normalized == pytest.approx(0.4)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _run(root: Path, optimizer: str, seed: int, correct: int) -> Path:
    run = root / f"{optimizer}_seed{seed}"
    (run / "provenance").mkdir(parents=True)
    command = ["train.py", "--geometry-interval", "1"]
    (run / "provenance" / "run_manifest.json").write_text(
        json.dumps({"status": "complete", "command": command}) + "\n"
    )
    (run / "run_complete.json").write_text(json.dumps({"status": "complete", "final_num_updates": 2}) + "\n")
    geometry = []
    for observation_id in range(2):
        scale = observation_id + 1
        global_metrics = {
            "parameter_count": 100,
            "g_raw_to_theta_ratio": 1e-4 * scale,
            "d_data_to_g_opt_ratio": 1.0,
            "delta_wd_to_delta_data_ratio": 0.0,
            "delta_intended_to_theta_ratio": 1e-5 * scale,
            "delta_model_to_theta_ratio": 1e-5 * scale,
            "displacement_to_reference_ratio": 1e-5 * scale,
            "gradient_directional_step": -0.01,
            "cos_g_raw_g_opt": 1.0,
            "cos_g_opt_d_data": 1.0,
            "cos_g_opt_delta_intended_fp32": -0.8,
            "cos_delta_intended_fp32_delta_model": 0.99,
            "cos_theta_before_delta_model": 0.0,
            "cos_delta_model_displacement": 0.9,
            "dot_g_opt_d_data": 0.01,
            "dot_g_opt_delta_intended_fp32": -0.001,
            "dot_g_opt_delta_model": -0.001,
            "model_change_fraction": 1.0,
            "intended_below_half_ulp_fraction": 0.0,
            "energy_survival": 1.0,
            "quantization_residual": 0.0,
            "intended_energy_zeroed_fraction": 0.0,
            "intended_energy_amplified_fraction": 0.0,
            "intended_energy_attenuated_fraction": 0.0,
            "ulp_ratio_bins": {},
        }
        vector_norms = {
            "theta_before": 1.0,
            "theta_reference": 1.0,
            "g_raw": 0.1 * scale,
            "g_opt": 0.1 * scale,
            "d_data": 0.1 * scale,
            "d_wd": 0.0,
            "delta_data_fp32": 0.01 * scale,
            "delta_wd_fp32": 0.0,
            "delta_intended_fp32": 0.01 * scale,
            "delta_model": 0.01 * scale,
            "displacement": 0.01 * scale,
        }
        for vector_name, norm in vector_norms.items():
            global_metrics[f"{vector_name}_l2"] = norm
            global_metrics[f"{vector_name}_rms"] = norm / 10
            global_metrics[f"{vector_name}_linf"] = norm
            global_metrics[f"{vector_name}_exact_zero_fraction"] = float(norm == 0.0)
        branch = "adam" if optimizer == "adamw" else optimizer
        geometry.append(
            {
                "schema_version": 2,
                "run_id": run.name,
                "seed": seed,
                "task": "code",
                "rollout_id": observation_id,
                "observation_id": observation_id,
                "num_updates": scale,
                "model_version": scale,
                "actual_batch_size": 1,
                "effective_token_count": 32,
                "cumulative_prompt_count": scale,
                "cumulative_effective_token_count": 32 * scale,
                "model_dtype_parameter_counts": {"torch.bfloat16": 100},
                "actual_optimizer_branches": {branch: {"learning_rate": 1e-6, "weight_decay": 0.0}},
                "update_successful": True,
                "valid_update_metrics": True,
                "low_frequency_observation": True,
                "grad_norm_raw": 0.1 * scale,
                "clip_threshold": 1.0,
                "clip_scale": 1.0,
                "optimizer_clip_scale": 1.0,
                "grad_clipped": False,
                "run_clip_fraction": 0.0,
                "geometry_observation_wall_time_ms": 2.0,
                "experiment_task": "code",
                "experiment_teacher": "none",
                "experiment_condition": "grpo",
                "optimizer": optimizer,
                "experiment_seed": seed,
                "learning_rate": 1e-6,
                "weight_decay": 0.0,
                "groups": {
                    "global": global_metrics,
                    f"optimizer_branch/{branch}": {
                        "parameter_count": 100,
                        "parameter_fraction": 1.0,
                        "gradient_energy_fraction": 1.0,
                        "intended_update_energy_fraction": 1.0,
                        "realized_update_energy_fraction": 1.0,
                        "weight_decay_metrics_applicability": "not_applicable",
                    },
                },
            }
        )
    _jsonl(run / "geometry" / "actor" / "metrics.jsonl", geometry)
    (run / "geometry" / "actor" / "exact_reference").mkdir(parents=True)
    (run / "geometry" / "actor" / "exact_reference" / "rank_00000.pt").write_bytes(b"fixture")
    (run / "geometry" / "actor" / "support_state").mkdir(parents=True)
    (run / "geometry" / "actor" / "support_state" / "rank_00000.pt").write_bytes(b"fixture")
    _jsonl(
        run / "geometry" / "rollout" / "metrics.jsonl",
        [
            {
                "schema_version": 1,
                "record_type": "rollout_geometry",
                "run_id": run.name,
                "seed": seed,
                "task": "code",
                "rollout_id": 0,
                "num_updates": 0,
                "model_version": 0,
                "actual_batch_size": 1,
                "effective_token_count": 32,
                "cumulative_prompt_count": 1,
                "cumulative_effective_token_count": 32,
                "sampled_reverse_kl_definition": ("log_pi_student_sampled_action_minus_log_pi_teacher_sampled_action"),
                "task_reward_observed": True,
                "reward_used_in_loss": True,
                "reward_loss_coefficient": 1.0,
                "loss_components": {},
                "availability": {"task_reward_observed": True},
                "observation_wall_time_ms": 1.0,
                "metrics": {"sample_count": 1, "valid_token_count": 32},
            }
        ],
    )
    _jsonl(
        run / "geometry" / "rollout" / "samples" / "rollout_00000000.jsonl",
        [
            {
                "schema_version": 1,
                "run_id": run.name,
                "seed": seed,
                "task": "code",
                "rollout_id": 0,
                "num_updates": 0,
                "model_version": 0,
                "prompt_id": 0,
                "sample_id": 0,
                "prompt": "problem",
                "response": "answer",
                "label": "reference",
                "reward": 1.0,
                "passed": True,
                "task_reward_observed": True,
                "reward_used_in_loss": True,
                "reward_loss_coefficient": 1.0,
                "status": "completed",
                "response_length": 1,
                "effective_response_length": 1,
            }
        ],
    )
    _jsonl(
        run / "metrics" / "train.jsonl",
        [
            {
                "metrics": {
                    "train/num_updates": 2,
                    "train/gpu_peak_allocated_mib": 1024 + correct,
                    "train/gpu_peak_reserved_mib": 2048 + correct,
                    "train/grad_clipped": 0,
                }
            }
        ],
    )
    _jsonl(
        run / "metrics" / "rollout.jsonl",
        [
            {
                "metrics": {
                    "rollout/step": 1,
                    "rollout/truncated_ratio": 0.1,
                    "rollout/response_len/p95": 100,
                    "perf/effective_tokens_per_gpu_per_sec": 50,
                    "perf/step_time": 10,
                }
            }
        ],
    )

    samples = []
    for sample_index in range(10):
        samples.append(
            {
                "dataset": "livecodebench_final",
                "num_updates": 2,
                "model_version": 2,
                "eval_phase": "final",
                "prompt_index": 0,
                "sample_within_prompt": sample_index,
                "prompt": "problem",
                "response": f"answer {sample_index}",
                "reward": 1.0 if sample_index < correct else 0.0,
                "status": "completed",
                "response_length": 2,
                "metadata_sha256": "0" * 64,
            }
        )
    sample_path = run / "eval_artifacts" / "livecodebench_final" / "updates_00000002_final.jsonl"
    _jsonl(sample_path, samples)
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    _jsonl(
        run / "eval_artifacts" / "index.jsonl",
        [
            {
                "num_updates": 2,
                "model_version": 2,
                "eval_phase": "final",
                "datasets": {
                    "livecodebench_final": {
                        "path": str(sample_path.resolve()),
                        "samples": 10,
                        "prompts": 1,
                        "n_samples_per_prompt": 10,
                        "sha256": digest,
                    }
                },
            }
        ],
    )
    score = correct / 10
    _jsonl(
        run / "forgetting" / "metrics.jsonl",
        [
            {
                "num_updates": 2,
                "eval_phase": "final",
                "tasks": {
                    "livecodebench_final": {
                        "score": score,
                        "best": score,
                        "forgetting": 0.0,
                        "backward_transfer": score - 0.1,
                    }
                },
            }
        ],
    )
    return run


@pytest.mark.integration
def test_paper_statistics_writes_validated_tables_and_png_pdf_curves(tmp_path):
    runs = [
        _run(tmp_path, "adamw", 42, 1),
        _run(tmp_path, "adamw", 43, 2),
        _run(tmp_path, "sgd", 42, 3),
        _run(tmp_path, "sgd", 43, 4),
    ]
    output = tmp_path / "paper"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT}:{environment.get('PYTHONPATH', '')}"
    environment["MPLBACKEND"] = "Agg"

    subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, runs), "--output-dir", str(output)],
        check=True,
        cwd=ROOT,
        env=environment,
    )

    expected = {
        "aggregate.csv",
        "analysis_manifest.json",
        "curves.csv",
        "final_scores.pdf",
        "final_scores.png",
        "final_scores.tex",
        "geometry_alignment_curves.pdf",
        "geometry_alignment_curves.png",
        "geometry_update_ratio_curves.pdf",
        "geometry_update_ratio_curves.png",
        "learning_curves.pdf",
        "learning_curves.png",
        "paired_effects.csv",
        "paired_effects.tex",
        "per_run.csv",
    }
    assert expected <= {path.name for path in output.iterdir()}
    with (output / "paired_effects.csv").open(newline="") as stream:
        effects = list(csv.DictReader(stream))
    score = next(row for row in effects if row["optimizer"] == "sgd" and row["metric"] == "eval_score")
    assert float(score["difference_mean"]) == pytest.approx(0.2)
    assert score["paired_seeds"] == "42,43"
    with (output / "per_run.csv").open(newline="") as stream:
        per_run = list(csv.DictReader(stream))
    assert {float(row["train_peak_gpu_allocated_mib"]) for row in per_run} == {1025, 1026, 1027, 1028}
    assert all(float(row["rollout_truncated_ratio_mean"]) == pytest.approx(0.1) for row in per_run)
    assert all(float(row["parameter_geometry_observation_ms_total"]) == pytest.approx(4.0) for row in per_run)
    assert all(int(row["final_cumulative_prompt_count"]) == 2 for row in per_run)
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["validated"] is True
    assert all(report["valid"] for report in manifest["validation_reports"])
    assert (
        manifest["outputs"]["paired_effects.csv"]["sha256"]
        == hashlib.sha256((output / "paired_effects.csv").read_bytes()).hexdigest()
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
