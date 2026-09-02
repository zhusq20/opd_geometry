import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.mopd_gpas import (
    analyze_capability,
    analyze_frozen_bank,
    analyze_mopd,
    plot_results,
    prepare_mopd,
)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_frozen_bank_k2_enumerates_six_sets_and_taskwise_target_is_exact():
    raw = np.asarray([1.0, 2.0, 3.0, 4.0])
    marginals, distribution = analyze_frozen_bank._distribution(
        "gpas",
        raw_scores=raw,
        adam_scores=raw,
        width=2,
        task_seconds=np.ones(4),
        switch_seconds=np.zeros(4),
        resident=0,
    )
    assert len(distribution) == 6
    vectors = [np.asarray([i + 1.0, 2 * i - 1.0]) for i in range(4)]
    _mse, conventional, taskwise = analyze_frozen_bank._estimator_error(vectors, marginals, distribution)
    assert conventional > 0
    assert taskwise == pytest.approx(0.0, abs=1e-12)


def test_frozen_bank_fold_uses_rms_unit_scores_and_raw_gradient_second_moment():
    observations = {}
    for task_index, task in enumerate(analyze_frozen_bank.TASKS):
        observations[task] = [
            {
                "raw": np.asarray([1.0 + offset, (-1.0) ** offset * (task_index + 1.0)]),
                "adam": np.asarray([100.0, 100.0]),
                "raw_score": float(offset + 1 + task_index),
                "adam_score": float(2 * (offset + 1 + task_index)),
                "task_seconds": 1.0,
                "switch_seconds": 0.0,
            }
            for offset in range(8)
        ]
    fold = analyze_frozen_bank._fold(observations, slice(0, 4), slice(4, 8), resident=0)
    expected_score = np.sqrt(np.mean(np.square([1.0, 2.0, 3.0, 4.0])))
    assert fold["scores"]["math"]["raw_norm"] == pytest.approx(expected_score)

    method = fold["K"]["1"]["uniform"]
    marginals = np.asarray(list(method["marginals"].values()))
    distribution = {
        tuple(analyze_frozen_bank.TASKS.index(name) for name in names.split("+")): probability
        for names, probability in method["set_distribution"].items()
    }
    expected = []
    for offset in range(4, 8):
        raw_vectors = [observations[task][offset]["raw"] for task in analyze_frozen_bank.TASKS]
        expected.append(analyze_frozen_bank._estimator_error(raw_vectors, marginals, distribution)[1])
    assert method["conventional_second_moment_relative_error"] == pytest.approx(np.mean(expected))


def _frozen_bank_rows():
    rows = []
    for operation in range(32):
        task = analyze_frozen_bank.TASKS[operation % 4]
        rows.append(
            {
                "operation_index": operation,
                "execution_order": [task],
                "feedback": {
                    "task_units": [
                        {
                            "predicted_full_task_seconds": 1.0,
                            "teacher_transfer_tail_seconds": 0.0,
                        }
                    ]
                },
            }
        )
    return rows


def test_frozen_bank_loads_the_single_optimizer_rank_used_by_the_launcher(tmp_path):
    rows = _frozen_bank_rows()
    _write_jsonl(tmp_path / "allocation/allocation.jsonl", rows)
    for row in rows:
        operation = row["operation_index"]
        task = row["execution_order"][0]
        path = tmp_path / "bank" / f"unit_{operation:03d}_{task}_rank_0000.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "metadata": {
                    "operation_index": operation,
                    "task": task,
                    "raw_score": 1.0,
                    "adam_score": 2.0,
                },
                "layout": [{"name": "parameter", "take": 2}],
                "raw": torch.tensor([1.0, 2.0]),
                "scaled": torch.tensor([2.0, 4.0]),
            },
            path,
        )

    observations, _ = analyze_frozen_bank._load_stage(tmp_path)

    assert all(len(units) == 8 for units in observations.values())


def test_frozen_bank_rejects_a_missing_single_rank_capture(tmp_path):
    rows = _frozen_bank_rows()
    _write_jsonl(tmp_path / "allocation/allocation.jsonl", rows)
    with pytest.raises(ValueError, match="requires ranks 0..0"):
        analyze_frozen_bank._load_stage(tmp_path)


def _completion(path: Path, updates=1):
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_complete.json").write_text(
        json.dumps({"status": "complete", "final_num_updates": updates}), encoding="utf-8"
    )


def _eval_metrics(rollout_id, mean):
    tasks = ("math", "code", "if", "science")
    return {
        "eval/rollout_id": rollout_id,
        "eval/num_updates": 1,
        "eval/mean_relative_teacher_loss": mean,
        **{f"eval/relative_teacher_loss/{task}": mean for task in tasks},
        **{f"eval/teacher_loss/{task}": mean / (index + 1) for index, task in enumerate(tasks)},
    }


def _build_synthetic_runs(root: Path):
    def feedback(operation, task, gpu_seconds, valid_tokens=10_000):
        unit = {
            "task": task,
            "switched": True,
            "teacher_switch_seconds": 0.1,
            "teacher_transfer_tail_seconds": 0.1,
            "teacher_memory_mib": 3000.0,
            "student_rollout_seconds": 4.0,
            "teacher_scoring_seconds": 3.0,
            "actor_forward_backward_wall_seconds": 2.0,
            "optimizer_wall_seconds": 1.0,
            "student_peak_hbm_bytes": 4 * 2**30,
        }
        return {
            "operation": operation,
            "total_gpu_seconds": gpu_seconds,
            "active_gpu_seconds": gpu_seconds - 2.0,
            "rollout_gpu_seconds": 8.0,
            "teacher_gpu_seconds": 3.0,
            "actor_forward_backward_gpu_seconds": 4.0,
            "optimizer_gpu_seconds": 1.0,
            "valid_response_tokens": valid_tokens,
            "completed_responses": 64,
            "invalid_responses": 0,
            "truncated_responses": 0,
            "peak_hbm_bytes": 4 * 2**30,
            "teacher_peak_memory_mib": 3000,
            "allocated_gpu_count": 5,
            "total_step_seconds": gpu_seconds / 5,
            "total_wall_seconds": gpu_seconds / 5,
            "rollout_wall_seconds": 4.0,
            "teacher_wall_seconds": 3.0,
            "rollout_and_teacher_wall_seconds": 4.0,
            "actor_forward_backward_wall_seconds": 2.0,
            "optimizer_wall_seconds": 1.0,
            "task_units": [unit],
        }

    warm = root / "warm_start-seed42"
    _completion(warm, 8)
    warm_rows = []
    for operation in range(8):
        task = analyze_mopd.TASKS[operation % 4]
        warm_rows.append(
            {
                "operation_index": operation,
                "rollout_id": operation,
                "attempted_responses_before": 64 * operation,
                "attempted_responses_after": 64 * (operation + 1),
                "operation": "train",
                "selected_set": [task],
                "execution_order": [task],
                "resident_teacher_before": ("math" if operation == 0 else analyze_mopd.TASKS[(operation - 1) % 4]),
                "resident_teacher_after": task,
                "optimizer_updates_after": operation + 1,
                "processed_task_units_after": operation + 1,
                "probe_count_after": 0,
                "inclusion_probabilities": {task: 0.25 for task in analyze_mopd.TASKS},
                "set_distribution": {task: 0.25 for task in analyze_mopd.TASKS},
                "score_ages_after": {task: 0 for task in analyze_mopd.TASKS},
                "raw_gradient_rms_after": {task: 1.0 for task in analyze_mopd.TASKS},
                "adam_gradient_rms_after": {task: 2.0 for task in analyze_mopd.TASKS},
                "predicted_set_seconds": {task: 1.0 for task in analyze_mopd.TASKS},
                "feedback": feedback("train", task, 5.0, valid_tokens=100),
            }
        )
    _write_jsonl(warm / "allocation/allocation.jsonl", warm_rows)
    _write_jsonl(warm / "metrics/eval.jsonl", [{"metrics": _eval_metrics(7, 0.9)}])

    for index, config in enumerate(analyze_mopd.CONFIGS):
        path = root / f"{config}-seed42"
        _completion(path, 2)

        train_probabilities = {"math": 0.1, "code": 0.2, "if": 0.3, "science": 0.4}
        _write_jsonl(
            path / "allocation/allocation.jsonl",
            [
                {
                    "operation_index": 0,
                    "rollout_id": 8,
                    "attempted_responses_before": 512,
                    "attempted_responses_after": 63992,
                    "operation": "train",
                    "selected_set": ["math"],
                    "execution_order": ["math"],
                    "resident_teacher_before": "science",
                    "resident_teacher_after": "math",
                    "feedback": feedback("train", "math", 100.0 + index),
                    "optimizer_updates_after": 9,
                    "processed_task_units_after": 9,
                    "probe_count_after": 0,
                    "score_ages_after": {task: 0 for task in analyze_mopd.TASKS},
                    "inclusion_probabilities": train_probabilities,
                    "set_distribution": train_probabilities,
                    "raw_gradient_rms_after": {task: 1.0 for task in analyze_mopd.TASKS},
                    "adam_gradient_rms_after": {task: 2.0 for task in analyze_mopd.TASKS},
                    "predicted_set_seconds": {task: 1.0 for task in analyze_mopd.TASKS},
                },
                {
                    "operation_index": 1,
                    "rollout_id": 9,
                    "attempted_responses_before": 63992,
                    "attempted_responses_after": 64000,
                    "operation": "probe",
                    "selected_set": ["science"],
                    "execution_order": ["science"],
                    "resident_teacher_before": "math",
                    "resident_teacher_after": "science",
                    "feedback": feedback("probe", "science", 10.0),
                    "optimizer_updates_after": 9,
                    "processed_task_units_after": 9,
                    "probe_count_after": 1,
                    "score_ages_after": {task: 1 for task in analyze_mopd.TASKS},
                    "inclusion_probabilities": {
                        "math": 0.0,
                        "code": 0.0,
                        "if": 0.0,
                        "science": 1.0,
                    },
                    "set_distribution": {"science": 1.0},
                    "raw_gradient_rms_after": {task: 1.0 for task in analyze_mopd.TASKS},
                    "adam_gradient_rms_after": {task: 2.0 for task in analyze_mopd.TASKS},
                    "predicted_set_seconds": {task: 1.0 for task in analyze_mopd.TASKS},
                },
            ],
        )
        _write_jsonl(path / "metrics/eval.jsonl", [{"metrics": _eval_metrics(9, 0.7 + index * 0.01)}])


def test_main_report_contains_all_eight_configs_and_response_axis(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    _build_synthetic_runs(root)
    output = tmp_path / "report.json"
    monkeypatch.setattr(analyze_mopd, "_paired_bootstrap", lambda runs: {"configs": list(runs)})
    monkeypatch.setattr(sys, "argv", ["analyze_mopd.py", "--root", str(root), "--output", str(output)])
    analyze_mopd.main()
    report = json.loads(output.read_text())
    assert output.with_name("mopd_outcomes.csv").is_file()
    assert tuple(report["configs"]) == analyze_mopd.CONFIGS
    assert all(report["curves"][config][-1]["attempted_responses"] == 64000 for config in report["configs"])
    assert report["outcomes"]["uniform_k1_conventional"]["responses_to_threshold"] is not None
    assert report["outcomes"]["uniform_k1_conventional"]["final_train_operation_index"] == 0
    assert report["outcomes"]["uniform_k1_conventional"]["final_inclusion_probabilities"]["math"] == 0.1
    assert len(report["system_traces"]["uniform_k1_conventional"]) == 2


def _synthetic_figure_reports():
    configs = list(plot_results.CONFIGS)
    tasks = plot_results.TASKS

    def system_trace(config_index, config):
        if config == "cost_gpas_k2_taskwise":
            set_distribution = {
                "math+code": 0.20,
                "math+if": 0.15,
                "math+science": 0.15,
                "code+if": 0.15,
                "code+science": 0.20,
                "if+science": 0.15,
            }
            inclusion = {task: 0.5 for task in tasks}
        else:
            set_distribution = {task: probability for task, probability in zip(tasks, (0.1, 0.2, 0.3, 0.4))}
            inclusion = dict(set_distribution)
        rows = []
        for operation in range(5):
            is_probe = operation == 2 and config in plot_results.ADAPTIVE_CONFIGS
            rows.append(
                {
                    "operation": "probe" if is_probe else "train",
                    "attempted_responses_after": 512 + (operation + 1) * 12_000,
                    "total_gpu_seconds": 100.0 + config_index,
                    "score_ages": {task: operation + task_index for task_index, task in enumerate(tasks)},
                    "raw_gradient_rms": {
                        task: float(task_index + 1 + operation / 10) for task_index, task in enumerate(tasks)
                    },
                    "adam_gradient_rms": {
                        task: float((4 - task_index) * (operation + 1)) for task_index, task in enumerate(tasks)
                    },
                    "inclusion_probabilities": inclusion,
                    "set_distribution": set_distribution,
                    "component_wall_seconds": {
                        "rollout_and_teacher_wall_seconds": 10.0 + config_index,
                        "actor_forward_backward_wall_seconds": 2.0,
                        "optimizer_wall_seconds": 0.5,
                    },
                    "student_peak_hbm_gib": 48.0 + config_index / 10,
                    "teacher_peak_hbm_gib": 18.0 + config_index / 10,
                    "task_units": [
                        {
                            "switched": operation > 0,
                            "teacher_transfer_tail_seconds": operation / 10,
                        }
                    ],
                }
            )
        return rows

    outcomes = {}
    bootstraps = {}
    curves = {}
    for index, config in enumerate(configs):
        final = 0.68 + index * 0.015
        relative_losses = {task: final + (task_index - 1.5) * 0.02 for task_index, task in enumerate(tasks)}
        outcomes[config] = {
            "mean_relative_loss": final,
            "relative_losses": relative_losses,
            "responses_to_threshold": None if index == len(configs) - 1 else 40_000 + 1_000 * index,
            "gpu_hours_to_threshold": None if index == len(configs) - 1 else 1.1 + index / 10,
            "gpu_hours": 2.0 + index / 10,
        }
        worst = max(relative_losses.values())
        bootstraps[config] = {
            "mean_relative_loss_95ci": [final - 0.02, final + 0.02],
            "worst_task_loss_95ci": [worst - 0.02, worst + 0.02],
        }
        curves[config] = [
            {
                "attempted_responses": 0,
                "valid_response_tokens": 0,
                "gpu_hours": 0.0,
                "mean_relative_loss": 1.0,
            },
            {
                "attempted_responses": 32_000,
                "valid_response_tokens": 10_000_000 + index * 100_000,
                "gpu_hours": 1.0 + index / 20,
                "mean_relative_loss": 0.82 + index * 0.01,
            },
            {
                "attempted_responses": 64_000,
                "valid_response_tokens": 20_000_000 + index * 200_000,
                "gpu_hours": 2.0 + index / 10,
                "mean_relative_loss": final,
            },
        ]

    mopd = {
        "seed": 42,
        "configs": configs,
        "response_budget": 64_000,
        "common_threshold": 0.75,
        "curves": curves,
        "outcomes": outcomes,
        "paired_bootstrap": {"baseline": configs[0], "configs": bootstraps},
        "system_traces": {config: system_trace(index, config) for index, config in enumerate(configs)},
    }
    capability_configs = {}
    capability_bootstraps = {}
    baseline_macro = float(np.mean([0.45 + 0.02 * task_index for task_index in range(len(tasks))]))
    for index, config in enumerate(configs):
        scores = {task: {"score": 0.45 + 0.02 * task_index + 0.005 * index} for task_index, task in enumerate(tasks)}
        macro = float(np.mean([value["score"] for value in scores.values()]))
        delta = macro - baseline_macro
        capability_configs[config] = {"domains": scores, "macro_score": macro}
        capability_bootstraps[config] = {
            "paired_macro_delta_vs_baseline": delta,
            "paired_macro_delta_95ci": [delta - 0.01, delta + 0.01],
        }
    capability = {
        "seed": 42,
        "configs": capability_configs,
        "paired_bootstrap": {"baseline": configs[0], "configs": capability_bootstraps},
    }
    bank = {
        "stages": {
            stage: {
                "cross_fit": {
                    "K": {
                        str(k): {
                            method: {
                                "relative_adamw_estimator_mse": 0.2
                                + 0.01 * k
                                + 0.02 * stage_index
                                + 0.01 * method_index
                            }
                            for method_index, method in enumerate(("uniform", "raw_norm", "gpas", "cost_gpas"))
                        }
                        for k in (1, 2, 4)
                    }
                }
            }
            for stage_index, stage in enumerate(("warm", "middle", "late"))
        }
    }
    sampling = [
        {
            "adamw_empirical_mse_ratio": str(value),
            "adamw_theory_mse_ratio": str(value * 0.99),
            "cost_empirical_ratio": str(0.8 + value / 10),
            "cost_theory_ratio": str(0.79 + value / 10),
        }
        for value in (1.0, 1.4, 0.7, 0.6)
    ]
    moments = [
        {
            "optimizer": optimizer,
            "moment": moment,
            "mc_relative_bias": str(value),
            "calculated_relative_bias": str(value * 0.98),
        }
        for optimizer, values in (
            ("Standard AdamW", (0.01, 0.8)),
            ("Moment-consistent AdamW", (0.01, 0.002)),
        )
        for moment, value in zip(("First moment", "Second moment"), values)
    ]
    return mopd, capability, bank, sampling, moments


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
    assert report["configs"][analyze_capability.CONFIGS[0]]["paired_macro_delta_95ci"] == [0.0, 0.0]
    assert report["configs"][analyze_capability.CONFIGS[1]]["paired_macro_delta_vs_baseline"] == pytest.approx(0.01)


def test_protocol_preparation_rejects_an_additional_seed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_mopd.py", "--seed", "43"])
    with pytest.raises(SystemExit):
        prepare_mopd.main()


def test_single_seed_result_gallery_renders_nine_rows_and_twenty_seven_panels(tmp_path):
    reports = _synthetic_figure_reports()
    manifest = plot_results.render_all(*reports, tmp_path)
    assert manifest["seed"] == 42
    assert manifest["row_count"] == 9
    assert manifest["panel_count"] == 27
    for relative_path in (*manifest["rows"], *manifest["panels"]):
        assert (tmp_path / relative_path).stat().st_size > 1_000
    assert json.loads((tmp_path / "figure_manifest.json").read_text()) == manifest


def test_result_gallery_rejects_a_second_seed(tmp_path):
    mopd, capability, bank, sampling, moments = _synthetic_figure_reports()
    capability["seed"] = 43
    with pytest.raises(ValueError, match="single training seed 42"):
        plot_results.render_all(mopd, capability, bank, sampling, moments, tmp_path)
