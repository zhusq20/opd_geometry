import json
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.mopd_gpas import (
    analyze_capability,
    analyze_heldout_variance,
    analyze_mopd,
    plot_results,
    prepare_mopd,
)

NUM_GPUS = 0


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _feedback(step: int, config_index: int) -> dict:
    task_units = [
        {
            "task": task,
            "microbatches": 4,
            "microbatch_seconds": 0.25,
        }
        for task in analyze_mopd.TASKS
    ]
    wall = 8.0 + config_index / 10
    return {
        "task_units": task_units,
        "fixed_seconds": wall - 4.0,
        "aggregate_grad_norm": 2.0,
        "aggregate_grad_clipped": False,
        "total_wall_seconds": wall,
        "total_gpu_seconds": 2 * wall,
        "valid_response_tokens": 1_000 + step,
        "generated_tokens": 1_100 + step,
        "truncated_responses": 0,
        "invalid_responses": 0,
        "peak_hbm_bytes": 50 * 2**30,
        "teacher_peak_memory_mib": 40 * 1024,
    }


def _build_runs(root: Path) -> Path:
    weights = {task: 0.25 for task in analyze_mopd.TASKS}
    protocol = {
        "schema_version": 4,
        "training": {"configs": list(analyze_mopd.CONFIGS)},
        "objective": {"weights": weights},
        "initial_kl": {"tasks": {task: {"ell0": 1.0} for task in analyze_mopd.TASKS}},
    }
    protocol_path = root / "protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    for config_index, config in enumerate(analyze_mopd.CONFIGS):
        run = root / f"{config}-seed42"
        run.mkdir(parents=True)
        (run / "run_complete.json").write_text(
            json.dumps({"status": "complete", "final_num_updates": 500}), encoding="utf-8"
        )
        rows = []
        wall = 8.0 + config_index / 10
        for index in range(500):
            noise = {task: float(task_index + 1) for task_index, task in enumerate(analyze_mopd.TASKS)}
            rows.append(
                {
                    "operation_index": index,
                    "optimizer_updates_after": index + 1,
                    "attempted_responses_after": (index + 1) * 64,
                    "counts": {task: 4 for task in analyze_mopd.TASKS},
                    "H": 1.0,
                    "raw_noise_after": noise,
                    "scaled_noise_after": noise,
                    "loss_ema_after": {task: 0.8 for task in analyze_mopd.TASKS},
                    "task_seconds_before": {task: None if index == 0 else 0.25 for task in analyze_mopd.TASKS},
                    "task_seconds_after": {task: 0.25 for task in analyze_mopd.TASKS},
                    "fixed_seconds_before": None if index == 0 else wall - 4.0,
                    "fixed_seconds_after": wall - 4.0,
                    "feedback": _feedback(index, config_index),
                }
            )
        _write_jsonl(run / "allocation/allocation.jsonl", rows)
        eval_rows = []
        for step in analyze_mopd.STEPS:
            loss = 1.0 - step / 2_000 + config_index / 100
            metrics = {
                "eval/num_updates": step,
                "eval/weighted_teacher_loss": loss,
                **{f"eval/teacher_loss/{task}": loss for task in analyze_mopd.TASKS},
                **{f"eval/normalized_teacher_loss/{task}": loss for task in analyze_mopd.TASKS},
            }
            eval_rows.append({"metrics": metrics})
        _write_jsonl(run / "metrics/eval.jsonl", eval_rows)
    return protocol_path


def test_main_report_uses_eight_runs_and_the_500_step_clock(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    protocol = _build_runs(root)
    output = tmp_path / "report/mopd.json"
    monkeypatch.setattr(
        analyze_mopd,
        "paired_bootstrap",
        lambda runs, weights: {"replicates": 1_000, "checkpoints": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_mopd.py",
            "--root",
            str(root),
            "--protocol",
            str(protocol),
            "--output",
            str(output),
        ],
    )
    analyze_mopd.main()
    report = json.loads(output.read_text())
    assert tuple(report["configs"]) == analyze_mopd.CONFIGS
    assert report["response_budget"] == 32_000
    assert all(report["curves"][config][-1]["step"] == 500 for config in report["configs"])
    assert report["outcomes"]["uniform"]["gpu_hours"] == pytest.approx(8_000 / 3_600)
    assert report["outcomes"]["std_mopd"]["gpu_hours_to_uniform_final"] is None
    assert report["outcomes"]["d3_mopd"]["gpu_hours_to_uniform_final"] is None
    assert report["outcomes"]["open_mopd"]["gpu_hours_to_uniform_final"] is None
    assert report["mechanism_summary"]["uniform"]["H"] == {
        "q25": 1.0,
        "median": 1.0,
        "q75": 1.0,
    }
    assert report["mechanism_summary"]["uniform"]["raw_scaled_ranking_disagreement_fraction"] == 0.0
    assert output.with_name("mopd_outcomes.csv").is_file()


def test_bootstrap_is_paired_by_prompt_and_excludes_non_fixed_objectives(monkeypatch, tmp_path):
    runs = {config: {"path": str(tmp_path / f"{config}-seed42")} for config in analyze_mopd.CONFIGS}

    def prompt_losses(path: Path, step: int):
        config = path.name.removesuffix("-seed42")
        offset = analyze_mopd.CONFIGS.index(config) / 100 + step / 100_000
        return {
            task: ([f"{task}-{index}" for index in range(128)], np.arange(128) / 128 + offset)
            for task in analyze_mopd.TASKS
        }

    monkeypatch.setattr(analyze_mopd, "_prompt_losses", prompt_losses)
    report = analyze_mopd.paired_bootstrap(runs, {task: 0.25 for task in analyze_mopd.TASKS}, replicates=100)
    final = report["checkpoints"]["500"]
    assert report["replicates"] == 100
    assert final["gpas"]["paired_delta_vs_uniform"] == pytest.approx(0.01)
    assert "weighted_loss_95ci" not in final["std_mopd"]
    assert "weighted_loss_95ci" not in final["d3_mopd"]
    assert "weighted_loss_95ci" not in final["open_mopd"]
    assert final["std_mopd"]["tasks"]["math"]["paired_delta_vs_uniform"] == pytest.approx(0.05)


def _figure_reports():
    curves = {}
    outcomes = {}
    traces = {}
    checkpoints = {}
    for step in analyze_mopd.STEPS:
        checkpoints[str(step)] = {}
        for config_index, config in enumerate(plot_results.CONFIGS):
            loss = 1.0 - step / 2_000 + config_index / 100
            item = {"objective_comparable": config not in analyze_mopd.NON_FIXED_OBJECTIVE, "tasks": {}}
            if config not in analyze_mopd.NON_FIXED_OBJECTIVE:
                item["weighted_loss_95ci"] = [loss - 0.01, loss + 0.01]
            checkpoints[str(step)][config] = item
    for config_index, config in enumerate(plot_results.CONFIGS):
        curves[config] = [
            {
                "step": step,
                "attempted_responses": step * 64,
                "gpu_hours": step / 200 + config_index / 100,
                "weighted_loss": 1.0 - step / 2_000 + config_index / 100,
            }
            for step in analyze_mopd.STEPS
        ]
        outcomes[config] = {
            "raw_losses": {task: 0.5 + task_index / 10 for task_index, task in enumerate(plot_results.TASKS)},
            "weighted_loss": 0.65 + config_index / 100,
            "gpu_hours": 2.5 + config_index / 10,
            "gpu_hours_to_uniform_final": 2.0 + config_index / 10,
            "fixed_to_variable_ratio_median": 1.5,
            "step_time_p50_seconds": 8.0,
            "step_time_p95_seconds": 10.0,
        }
        traces[config] = [
            {
                "step": step,
                "counts": {
                    task: 4 + ((task_index + step) % 3) - 1 for task_index, task in enumerate(plot_results.TASKS)
                },
                "raw_noise": {task: 1.0 + task_index for task_index, task in enumerate(plot_results.TASKS)},
                "scaled_noise": {task: 2.0 + task_index for task_index, task in enumerate(plot_results.TASKS)},
                "task_seconds_ema": {
                    task: 0.2 + task_index / 10 for task_index, task in enumerate(plot_results.TASKS)
                },
                "fixed_to_variable_ratio": 1.5,
                "H": 1.2,
            }
            for step in range(1, 501, 25)
        ]
    mopd = {
        "seed": 42,
        "configs": list(plot_results.CONFIGS),
        "common_threshold": 0.75,
        "curves": curves,
        "outcomes": outcomes,
        "system_traces": traces,
        "paired_bootstrap": {"checkpoints": checkpoints},
    }
    capability = {
        "seed": 42,
        "configs": {
            config: {
                "domains": {
                    task: {
                        "score": 0.4 + task_index / 10 + config_index / 100,
                        "normalized_gain": 0.2 + task_index / 10 + config_index / 100,
                    }
                    for task_index, task in enumerate(plot_results.TASKS)
                }
            }
            for config_index, config in enumerate(plot_results.CONFIGS)
        },
    }
    variance = {
        "checkpoints": {
            str(step): {
                method: {"relative_variance": 1.0 - method_index / 20}
                for method_index, method in enumerate(("uniform", "raw_noise", "loss_gap", "gpas", "cost_gpas"))
            }
            for step in (50, 250, 500)
        }
    }
    return mopd, capability, variance


def test_result_gallery_renders_protocol_figures(tmp_path):
    mopd, capability, variance = _figure_reports()
    manifest = plot_results.render_all(mopd, capability, tmp_path, variance)
    assert manifest["seed"] == 42
    assert set(manifest["figures"]) == {
        "learning_efficiency",
        "task_losses",
        "allocation_dynamics",
        "system_costs",
        "capability",
        "heldout_gradient_variance",
    }
    assert all((tmp_path / name).stat().st_size > 1_000 for name in manifest["figures"].values())


def test_capability_bootstrap_is_single_seed_and_paired_by_prompt():
    configs = {
        config: {
            "domains": {
                domain: {
                    "prompt_indices": list(range(5)),
                    "prompt_scores": [0.1 * index + 0.01 * config_index for index in range(5)],
                }
                for domain in analyze_capability.DOMAINS
            }
        }
        for config_index, config in enumerate(analyze_capability.CONFIGS)
    }
    report = analyze_capability.paired_bootstrap(configs, replicates=100)
    assert report["seed"] == 42
    assert report["configs"]["uniform"]["paired_macro_delta_95ci"] == [0.0, 0.0]
    assert report["configs"]["gpas"]["paired_macro_delta_vs_baseline"] == pytest.approx(0.01)


def test_capability_bootstrap_reports_teacher_normalized_gain():
    configs = {
        config: {
            "domains": {
                domain: {
                    "prompt_indices": list(range(5)),
                    "prompt_scores": [0.4 + 0.01 * config_index] * 5,
                }
                for domain in analyze_capability.DOMAINS
            }
        }
        for config_index, config in enumerate(analyze_capability.CONFIGS)
    }
    references = {
        reference: {
            "domains": {
                domain: {
                    "score": 0.2 if reference == "initial_student" else 0.6,
                    "prompt_indices": list(range(5)),
                    "prompt_scores": [0.2 if reference == "initial_student" else 0.6] * 5,
                }
                for domain in analyze_capability.DOMAINS
            }
        }
        for reference in analyze_capability.REFERENCES
    }
    report = analyze_capability.paired_bootstrap(configs, references, replicates=100)
    assert report["configs"]["uniform"]["normalized_gain_95ci"] == pytest.approx([0.5, 0.5])
    assert report["configs"]["gpas"]["paired_normalized_delta_vs_baseline"] == pytest.approx(0.025)


def test_heldout_variance_crossfits_halves_with_training_time_costs(tmp_path):
    run = tmp_path / "step_050"
    run.mkdir()
    (run / "run_complete.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    units = []
    for index, task in enumerate(analyze_mopd.TASKS, start=1):
        units.append(
            {
                "task": task,
                "microbatches": 32,
                "microbatch_seconds": 100.0,
                "raw_microbatch_sq": [2.0 * index] * 32,
                "scaled_microbatch_sq": [3.0 * index] * 32,
                "teacher_loss_microbatch": [0.1 * index] * 32,
                "raw_half_mean_sq": [float(index), float(index)],
                "scaled_half_mean_sq": [float(index), float(index)],
                "raw_task_mean_sq": float(index),
                "scaled_task_mean_sq": float(index),
                "raw_subset_mean_sq": {"2": [float(index)] * 2, "4": [float(index)] * 2},
                "scaled_subset_mean_sq": {"2": [float(index)] * 2, "4": [float(index)] * 2},
            }
        )
    artifact = {
        "schema_version": 1,
        "checkpoint_step": 50,
        "target_weights": dict.fromkeys(analyze_mopd.TASKS, 0.25),
        "training_controller_state": {
            "checkpoint_step": 50,
            "task_seconds": {task: 1.0 + index for index, task in enumerate(analyze_mopd.TASKS)},
            "fixed_seconds": 7.0,
        },
        "feedback": {"fixed_seconds": 400.0, "task_units": units},
    }
    (run / "heldout_gradient_scalars.json").write_text(json.dumps(artifact), encoding="utf-8")

    report = analyze_heldout_variance.analyze_checkpoint(run, 50)

    assert report["uniform"]["relative_variance"] == pytest.approx(1.0)
    assert report["task_seconds"]["math"] == 1.0
    assert report["fixed_seconds"] == 7.0


def test_analysis_tables_match_the_preregistered_rows_and_columns(tmp_path):
    mopd, capability, variance = _figure_reports()
    references = {}
    for reference_index, reference in enumerate(analyze_capability.REFERENCES):
        references[reference] = {
            "domains": {domain: {"score": 0.2 + 0.2 * (reference_index > 0)} for domain in analyze_capability.DOMAINS}
        }
    capability["references"] = references
    capability["headroom"] = {
        domain: {"initial_score": 0.2, "difference": 0.4} for domain in analyze_capability.DOMAINS
    }
    main_table = tmp_path / "mopd_main_table.csv"
    analyze_capability._write_main_table(main_table, capability, mopd)
    rows = main_table.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + len(analyze_capability.TABLE_TARGETS)
    assert "final_heldout_F" in rows[0]
    assert rows[1].startswith("initial_student,reference,")

    variance_table = tmp_path / "heldout_gradient_variance.csv"
    analyze_heldout_variance._write_table(variance_table, variance["checkpoints"])
    variance_rows = variance_table.read_text(encoding="utf-8").splitlines()
    assert len(variance_rows) == 4
    assert variance_rows[1].startswith("50,")


def test_protocol_preparation_rejects_an_additional_seed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_mopd.py", "--seed", "43"])
    with pytest.raises(SystemExit):
        prepare_mopd.main()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
