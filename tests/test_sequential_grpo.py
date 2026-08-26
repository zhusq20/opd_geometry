from __future__ import annotations

import csv
import json
import os
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
import yaml

from examples.optimizer_geometry.analyze_sequential_experiment import (
    Boundary,
    EvalDataset,
    PRIMARY_DATASET,
    paired_change,
    resolve_torch_dist_checkpoint,
    transfer_and_forgetting,
)
from examples.optimizer_geometry.summarize_sequential_raw_gradient_probes import (
    read_rollout_metrics,
)
from slime_plugins.geometry.raw_gradient_probe import RawGradientProbeAccumulator
from examples.optimizer_geometry.analyze_checkpoint_updates import TorchDistTensorReader

REPO = Path(__file__).parents[1]
PREPARE = REPO / "examples/optimizer_geometry/prepare_sequential_grpo_data.py"
ANALYZE_GRADIENTS = REPO / "examples/optimizer_geometry/analyze_raw_gradient_probes.py"
ANALYZE_UPDATES = REPO / "examples/optimizer_geometry/analyze_sequential_updates.py"
SEQUENCE_LAUNCHER = REPO / "examples/optimizer_geometry/run_sequential_grpo.sh"
PROBE_LAUNCHER = REPO / "examples/optimizer_geometry/run_raw_gradient_probe.sh"
BOUNDARY_EVAL_LAUNCHER = REPO / "examples/optimizer_geometry/evaluate_sequential_boundaries.sh"
BOUNDARY_PROBE_LAUNCHER = REPO / "examples/optimizer_geometry/run_sequential_boundary_gradient_probes.sh"


@pytest.mark.unit
def test_checkpoint_reader_excludes_optimizer_state(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    metadata = SimpleNamespace(
        state_dict_metadata={
            "decoder.layers.weight": SimpleNamespace(size=(2, 2)),
            "optimizer.distributed.bucket.param": SimpleNamespace(size=(4,)),
            "optimizer.distributed.bucket.exp_avg": SimpleNamespace(size=(4,)),
        },
        storage_data={},
    )
    with (checkpoint / ".metadata").open("wb") as stream:
        pickle.dump(metadata, stream)

    with TorchDistTensorReader(checkpoint) as reader:
        assert reader.tensor_names == ("decoder.layers.weight",)


def _write_single_task_configs(root: Path, rows: int = 8) -> None:
    eval_names = {
        "math": ("math_eval_aime24_math500.yaml", "math500", 32768),
        "code": ("code_eval.yaml", "livecodebench_online", 16384),
        "science": ("science_eval.yaml", "gpqa", 16384),
        "if": ("if_eval.yaml", "ifeval", 32768),
    }
    rm_types = {"math": "deepscaler", "code": "livecodebench", "science": "gpqa", "if": "ifevalg"}
    for task, (eval_file, eval_name, max_response_len) in eval_names.items():
        task_dir = root / task
        task_dir.mkdir(parents=True)
        data = task_dir / f"{task}.jsonl"
        data.write_text(
            "".join(
                json.dumps({"prompt": f"{task}-{index}", "label": str(index), "metadata": {}}) + "\n"
                for index in range(rows)
            )
        )
        manifest = {
            "version": 1,
            "sampling": {"strategy": "weighted", "unit": "batch", "seed": 42, "repeat": True},
            "sources": [
                {
                    "name": task,
                    "path": str(data),
                    "input_key": "prompt",
                    "label_key": "label",
                    "metadata_key": "metadata",
                    "rm_type": rm_types[task],
                }
            ],
        }
        (task_dir / f"{task}_on_policy.yaml").write_text(yaml.safe_dump(manifest))
        eval_data = task_dir / f"{task}_eval.jsonl"
        eval_data.write_text(json.dumps({"prompt": "eval", "label": "1", "metadata": {}}) + "\n")
        eval_config = {
            "eval": {
                "defaults": {
                    "apply_chat_template": True,
                    "custom_rm_path": "slime_plugins.m2rl.rewards.reward",
                    "max_response_len": max_response_len,
                },
                "datasets": [
                    {
                        "name": eval_name,
                        "path": str(eval_data),
                        "rm_type": rm_types[task],
                        "n_samples_per_eval_prompt": 1,
                    }
                ],
            }
        }
        (task_dir / eval_file).write_text(yaml.safe_dump(eval_config))


@pytest.mark.unit
def test_prepare_sequential_data_is_disjoint_and_deterministic(tmp_path):
    config_root = tmp_path / "single_task"
    _write_single_task_configs(config_root)
    outputs = [tmp_path / "prepared_a", tmp_path / "prepared_b"]
    for output in outputs:
        result = subprocess.run(
            [
                "python3",
                str(PREPARE),
                "--config-root",
                str(config_root),
                "--output-dir",
                str(output),
                "--probe-prompts",
                "2",
            ],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr

    for task in ("math", "code", "science", "if"):
        probe = [
            json.loads(line) for line in (outputs[0] / task / f"{task}_gradient_probe.jsonl").read_text().splitlines()
        ]
        assert len(probe) == 2
        assert [row["prompt"] for row in probe] == [f"{task}-6", f"{task}-7"]
        train_manifest = yaml.safe_load((outputs[0] / task / f"{task}_on_policy.yaml").read_text())
        assert train_manifest["sources"][0]["path"].endswith("@[0:-2]")
        assert (outputs[0] / task / f"{task}_gradient_probe.jsonl").read_bytes() == (
            outputs[1] / task / f"{task}_gradient_probe.jsonl"
        ).read_bytes()

    combined = yaml.safe_load((outputs[0] / "all_tasks_eval.yaml").read_text())
    datasets = combined["eval"]["datasets"]
    assert len(datasets) == 4
    assert {entry["metadata_overrides"]["sequential_domain"] for entry in datasets} == {
        "math",
        "code",
        "knowledge",
        "if",
    }
    assert {entry["max_response_len"] for entry in datasets} == {16384, 32768}
    data_index = json.loads((outputs[0] / "sequential_data_index.json").read_text())
    assert data_index["config_root"] == str(config_root.resolve())
    assert data_index["source_index"] is None
    assert data_index["source_index_sha256"] is None


@pytest.mark.unit
def test_prepare_can_limit_each_task_to_a_short_training_prefix(tmp_path):
    config_root = tmp_path / "single_task"
    _write_single_task_configs(config_root, rows=12)
    output = tmp_path / "prepared"

    result = subprocess.run(
        [
            "python3",
            str(PREPARE),
            "--config-root",
            str(config_root),
            "--output-dir",
            str(output),
            "--probe-prompts",
            "2",
            "--train-prompts-per-task",
            "4",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    index = json.loads((output / "sequential_data_index.json").read_text())
    assert index["train_prompts_per_task"] == 4
    for task in ("math", "code", "science", "if"):
        manifest = yaml.safe_load((output / task / f"{task}_on_policy.yaml").read_text())
        assert manifest["sources"][0]["path"].endswith("@[0:4]")
        assert index["tasks"][task]["train_rows"] == 4
        assert index["tasks"][task]["available_train_rows"] == 10


@pytest.mark.unit
def test_code_origin_plan_is_sandbox_free(tmp_path):
    config_root = tmp_path / "single_task"
    _write_single_task_configs(config_root, rows=40)
    code_checkpoint = tmp_path / "code_checkpoints/iter_0000299"
    code_checkpoint.mkdir(parents=True)
    (code_checkpoint / ".metadata").write_bytes(b"synthetic-metadata")
    (code_checkpoint / "common.pt").write_bytes(b"synthetic-common")
    sequence_dir = tmp_path / "sequence"

    result = subprocess.run(
        ["bash", str(SEQUENCE_LAUNCHER)],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "CODE_CHECKPOINT": str(code_checkpoint),
            "SEQUENCE_DIR": str(sequence_dir),
            "SINGLE_TASK_CONFIG_ROOT": str(config_root),
            "PROBE_PROMPTS": "16",
            "TRAIN_PROMPTS_PER_TASK": "16",
            "PLAN_ONLY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "existing Code -> Math -> Knowledge -> IF" in result.stdout
    assert "SandboxFusion is not required" in result.stdout
    manifest = json.loads((sequence_dir / "sequence_manifest.json").read_text())
    assert manifest["sequence"] == ["code_origin", "math", "knowledge", "if"]
    assert manifest["measurement_tasks"] == ["math", "knowledge", "if"]
    assert manifest["sandbox_execution_enabled"] is False
    assert manifest["task_boundary_evaluation"] is False
    assert manifest["same_checkpoint_raw_gradient_probes"] is False
    assert manifest["train_prompts_per_task"] == 16
    assert manifest["new_stage_training"]["checkpoint_interval_updates"] == 100
    prepared = json.loads((sequence_dir / "prepared_data/sequential_data_index.json").read_text())
    assert prepared["tasks_prepared"] == ["math", "science", "if"]
    assert set(prepared["tasks"]) == {"math", "science", "if"}
    eval_config = yaml.safe_load((sequence_dir / "prepared_data/all_tasks_eval.yaml").read_text())
    eval_domains = {entry["metadata_overrides"]["sequential_domain"] for entry in eval_config["eval"]["datasets"]}
    assert eval_domains == {"math", "knowledge", "if"}
    assert not (sequence_dir / "prepared_data/code").exists()
    assert not (sequence_dir / "stages").exists()


@pytest.mark.unit
def test_probe_launcher_can_exclude_code(tmp_path):
    checkpoint = tmp_path / "checkpoints/iter_0000007"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"synthetic-metadata")
    (checkpoint / "common.pt").write_bytes(b"synthetic-common")
    config_root = tmp_path / "prepared"
    for task in ("math", "science", "if"):
        task_dir = config_root / task
        task_dir.mkdir(parents=True)
        (task_dir / f"{task}_gradient_probe.yaml").write_text("sources: [{}]\n")
    launcher_log = tmp_path / "launcher.log"
    fake_launcher = tmp_path / "fake_launcher.sh"
    fake_launcher.write_text(
        'printf "%s:%s:%s\\n" "$TASK" "${NUM_EPOCH-unset}" "$NUM_ROLLOUT" ' '>>"$PROBE_LAUNCHER_LOG"\n'
    )

    result = subprocess.run(
        ["bash", str(PROBE_LAUNCHER)],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "LOAD_CHECKPOINT": str(checkpoint),
            "PROBE_CONFIG_ROOT": str(config_root),
            "OUTPUT_DIR": str(tmp_path / "probe_output"),
            "EXPERIMENT_LAUNCHER": str(fake_launcher),
            "PROBE_LAUNCHER_LOG": str(launcher_log),
            "PROBE_TASKS": "math science if",
            "PROBE_PROMPTS": "16",
            "DRY_RUN": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert launcher_log.read_text().splitlines() == ["math:unset:1", "science:unset:1", "if:unset:1"]


@pytest.mark.unit
def test_probe_summary_requires_one_rollout_quality_record(tmp_path):
    metrics_path = tmp_path / "rollout.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"metrics": {"rollout/step": 0}}),
                json.dumps(
                    {
                        "metrics": {
                            "rollout/step": 0,
                            "rollout/reward/mean": 0.25,
                            "rollout/reward/count": 256,
                        }
                    }
                ),
            ]
        )
        + "\n"
    )

    metrics = read_rollout_metrics(metrics_path)

    assert metrics["rollout/reward/mean"] == pytest.approx(0.25)
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"metrics": {"rollout/reward/mean": 0.5}}) + "\n")
    with pytest.raises(ValueError, match="exactly one rollout reward summary"):
        read_rollout_metrics(metrics_path)


@pytest.mark.unit
def test_prepare_excludes_globally_duplicated_tail_prompts(tmp_path):
    config_root = tmp_path / "single_task"
    _write_single_task_configs(config_root)
    code_data = config_root / "code/code.jsonl"
    prompts = ["code-0", "code-1", "code-2", "code-3", "code-4", "code-5", "code-0", "code-1"]
    code_data.write_text(
        "".join(
            json.dumps({"prompt": prompt, "label": str(index), "metadata": {}}) + "\n"
            for index, prompt in enumerate(prompts)
        )
    )
    output = tmp_path / "prepared"

    result = subprocess.run(
        [
            "python3",
            str(PREPARE),
            "--config-root",
            str(config_root),
            "--output-dir",
            str(output),
            "--probe-prompts",
            "2",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    probe = [
        json.loads(line)["prompt"] for line in (output / "code/code_gradient_probe.jsonl").read_text().splitlines()
    ]
    assert probe == ["code-4", "code-5"]
    manifest = yaml.safe_load((output / "code/code_on_policy.yaml").read_text())
    assert manifest["sources"][0]["path"].endswith("@[0:-4]")
    index = json.loads((output / "sequential_data_index.json").read_text())
    assert index["tasks"]["code"]["source_duplicate_prompt_rows"] == 2


@pytest.mark.unit
def test_prepare_rejects_probe_overlap_with_formal_eval(tmp_path):
    config_root = tmp_path / "single_task"
    _write_single_task_configs(config_root)
    (config_root / "math/math_eval.jsonl").write_text(
        json.dumps({"prompt": "math-7", "label": "1", "metadata": {}}) + "\n"
    )
    output = tmp_path / "prepared"

    result = subprocess.run(
        [
            "python3",
            str(PREPARE),
            "--config-root",
            str(config_root),
            "--output-dir",
            str(output),
            "--probe-prompts",
            "2",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "overlap formal evaluation prompts" in result.stderr


@pytest.mark.unit
def test_raw_gradient_probe_accumulates_equal_batch_mean(tmp_path):
    parameter = torch.nn.Parameter(torch.zeros(3))

    class View:
        name = "weight"
        start = 0
        stop = None
        group_names = ("layer/0000", "optimizer_branch/sgd")
        optimizer_branch = "sgd"

        @staticmethod
        def raw_gradient():
            return parameter.grad

    args = SimpleNamespace(
        geometry_raw_gradient_probe_dir=str(tmp_path / "probe"),
        geometry_raw_gradient_probe_updates=2,
        experiment_task="math",
        load="/checkpoint",
        ckpt_step=None,
        seed=42,
        rollout_seed=42,
        rollout_batch_size=16,
        n_samples_per_prompt=1,
        geometry_raw_gradient_probe_only=True,
    )
    accumulator = RawGradientProbeAccumulator(args, [View()])
    parameter.grad = torch.tensor([1.0, 2.0, 3.0])
    accumulator.add(
        observation_id=0,
        source_names=["math"],
        actual_batch_size=16,
        effective_token_count=100,
    )
    parameter.grad = torch.tensor([3.0, 4.0, 5.0])
    accumulator.add(
        observation_id=1,
        source_names=["math"],
        actual_batch_size=16,
        effective_token_count=120,
    )

    saved = torch.load(tmp_path / "probe/rank_00000/view_00000.pt", weights_only=True)
    torch.testing.assert_close(saved, torch.tensor([2.0, 3.0, 4.0]))
    manifest = json.loads((tmp_path / "probe/manifest.json").read_text())
    assert manifest["expected_updates"] == 2
    assert manifest["actual_batch_size_per_update"] == 16
    assert manifest["effective_token_counts"] == [100, 120]
    assert manifest["effective_token_count_total"] == 220
    assert manifest["probe_prompt_count"] == 32
    assert manifest["optimizer_step_executed"] is False


def _write_probe(root: Path, task: str, gradient: torch.Tensor) -> None:
    rank_dir = root / "rank_00000"
    rank_dir.mkdir(parents=True)
    torch.save(gradient, rank_dir / "view_00000.pt")
    view = {
        "file": "view_00000.pt",
        "group_names": ["layer/0000", "optimizer_branch/adam"],
        "name": "weight",
        "numel": gradient.numel(),
        "optimizer_branch": "adam",
        "start": 0,
        "stop": None,
        "bytes": (rank_dir / "view_00000.pt").stat().st_size,
    }
    rank_manifest = {
        "schema_version": 1,
        "rank": 0,
        "world_size": 1,
        "task": task,
        "checkpoint": "/same/checkpoint",
        "checkpoint_step": 7,
        "seed": 42,
        "expected_updates": 2,
        "observed_updates": 2,
        "actual_batch_size_per_update": 16,
        "rollout_batch_size": 16,
        "n_samples_per_prompt": 1,
        "probe_prompt_count": 32,
        "optimizer_step_executed": False,
        "views": [view],
    }
    (rank_dir / "manifest.json").write_text(json.dumps(rank_manifest))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": task,
                "checkpoint": "/same/checkpoint",
                "checkpoint_step": 7,
                "seed": 42,
                "world_size": 1,
                "expected_updates": 2,
                "actual_batch_size_per_update": 16,
                "rollout_batch_size": 16,
                "n_samples_per_prompt": 1,
                "probe_prompt_count": 32,
                "optimizer_step_executed": False,
                "rank_manifests": ["rank_00000/manifest.json"],
            }
        )
    )


@pytest.mark.unit
def test_exact_raw_gradient_analyzer(tmp_path):
    math_probe = tmp_path / "math"
    code_probe = tmp_path / "code"
    _write_probe(math_probe, "math", torch.tensor([1.0, 0.0]))
    _write_probe(code_probe, "code", torch.tensor([-1.0, 1.0]))
    output = tmp_path / "analysis"

    result = subprocess.run(
        [
            "python3",
            str(ANALYZE_GRADIENTS),
            "--probe",
            f"math={math_probe}",
            "--probe",
            f"code={code_probe}",
            "--output-dir",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["global"]["gradient_norm/math"] == pytest.approx(1.0)
    assert summary["global"]["gradient_norm/code"] == pytest.approx(2**0.5)
    assert summary["global"]["dot/math__code"] == pytest.approx(-1.0)
    assert summary["global"]["cosine/math__code"] == pytest.approx(-(2**-0.5))


@pytest.mark.unit
def test_exact_raw_gradient_analyzer_marks_zero_vector_cosine_undefined(tmp_path):
    zero_probe = tmp_path / "zero"
    math_probe = tmp_path / "math"
    _write_probe(zero_probe, "zero", torch.zeros(2))
    _write_probe(math_probe, "math", torch.tensor([1.0, 0.0]))
    output = tmp_path / "analysis"

    result = subprocess.run(
        [
            "python3",
            str(ANALYZE_GRADIENTS),
            "--probe",
            f"zero={zero_probe}",
            "--probe",
            f"math={math_probe}",
            "--output-dir",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["global"]["gradient_norm/zero"] == 0.0
    assert summary["global"]["gradient_direction_defined/zero"] is False
    assert summary["global"]["cosine/zero__math"] is None
    matrix = list(csv.DictReader((output / "global_cosine_matrix.csv").open()))
    assert matrix[0]["zero"] == ""
    assert matrix[0]["math"] == ""


def _save_checkpoint(path: Path, model: torch.Tensor, optimizer: torch.Tensor) -> None:
    dcp.save(
        {
            "decoder.layers.weight": model.to(torch.bfloat16),
            "optimizer.distributed.bucket.exp_avg": optimizer,
        },
        checkpoint_id=path,
    )
    (path / "common.pt").write_bytes(b"synthetic-common-state")


@pytest.mark.unit
def test_sequential_update_analyzer_uses_only_chained_model_deltas(tmp_path):
    origin = tmp_path / "origin"
    after_code = tmp_path / "after_code"
    after_knowledge = tmp_path / "after_knowledge"
    after_if = tmp_path / "after_if"
    _save_checkpoint(origin, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([100.0]))
    _save_checkpoint(after_code, torch.tensor([[2.0, 2.0], [3.0, 4.0]]), torch.tensor([200.0]))
    _save_checkpoint(after_knowledge, torch.tensor([[2.0, 3.0], [3.0, 4.0]]), torch.tensor([300.0]))
    _save_checkpoint(after_if, torch.tensor([[1.0, 3.0], [3.0, 4.0]]), torch.tensor([400.0]))
    output = tmp_path / "analysis"

    result = subprocess.run(
        [
            "python3",
            str(ANALYZE_UPDATES),
            "--stage",
            f"code={origin}::{after_code}",
            "--stage",
            f"knowledge={after_code}::{after_knowledge}",
            "--stage",
            f"if={after_knowledge}::{after_if}",
            "--output-dir",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    stage = summary["stage_local_global"]
    cumulative = summary["cumulative_global"]
    assert stage["parameter_count"] == 4
    assert stage["update_norm/code"] == pytest.approx(1.0)
    assert stage["cosine/code__knowledge"] == pytest.approx(0.0)
    assert stage["cosine/code__if"] == pytest.approx(-1.0)
    assert stage["visible_sparsity@1e-5/code"] == pytest.approx(0.75)
    assert stage["bf16_aware_sparsity_eta1e-3/code"] == pytest.approx(0.75)
    assert stage["relative_sparsity_tau1e-3/code"] == pytest.approx(0.75)
    assert stage["support_overlap@1e-5/code_to_if"] == pytest.approx(1.0)
    assert stage["support_overlap_lift@1e-5/code_to_if"] == pytest.approx(4.0)
    assert stage["support_sign_conflict_fraction@1e-5/code__if"] == pytest.approx(1.0)
    assert stage["interference_ratio/code_self_from_if"] == pytest.approx(-1.0)
    assert cumulative["cosine/code__knowledge"] == pytest.approx(2**-0.5)
    assert cumulative["cosine/code__if"] == pytest.approx(0.0)
    support_rows = summary["stage_local_support_overlap"]
    code_if = next(row for row in support_rows if row["left"] == "code" and row["right"] == "if")
    assert code_if["jaccard"] == pytest.approx(1.0)
    assert code_if["intersection_sign_conflict_fraction"] == pytest.approx(1.0)


def _eval_dataset(name: str, domain: str, score: float, root: Path) -> EvalDataset:
    passed = round(score * 10)
    rewards = [1.0] * passed + [0.0] * (10 - passed)
    rows = [
        {
            "prompt_index": index,
            "sample_within_prompt": 0,
            "reward": reward,
            "effective_response_length": 1,
            "status": "completed",
            "metadata": {"sequential_domain": domain},
        }
        for index, reward in enumerate(rewards)
    ]
    return EvalDataset(
        name=name,
        domain=domain,
        samples_per_prompt=1,
        rows=rows,
        prompt_rewards={index: [reward] for index, reward in enumerate(rewards)},
        artifact_path=root / f"{name}.jsonl",
        artifact_sha256="synthetic",
    )


@pytest.mark.unit
def test_sequential_behavior_analysis_uses_acquisition_diagonal_and_paired_changes(tmp_path):
    boundary_specs = (
        ("pre_code", None),
        ("code_origin", "code"),
        ("after_math", "math"),
        ("after_knowledge", "knowledge"),
        ("after_if", "if"),
    )
    boundaries = [
        Boundary(index, label, trained_task, tmp_path / label, tmp_path / f"checkpoint-{index}")
        for index, (label, trained_task) in enumerate(boundary_specs)
    ]
    scores = {
        "code": (0.4, 0.8, 0.7, 0.6, 0.5),
        "math": (0.1, 0.2, 0.9, 0.8, 0.7),
        "knowledge": (0.1, 0.1, 0.2, 0.8, 0.6),
        "if": (0.1, 0.1, 0.2, 0.3, 0.9),
    }
    evaluations = {
        label: {
            PRIMARY_DATASET[domain]: _eval_dataset(
                PRIMARY_DATASET[domain], domain, scores[domain][boundary_index], tmp_path
            )
            for domain in scores
        }
        for boundary_index, (label, _) in enumerate(boundary_specs)
    }

    adjacent, primary, forgetting, forward_transfer, base_to_final, continual = transfer_and_forgetting(
        boundaries,
        evaluations,
        bootstrap_samples=1_000,
        bootstrap_seed=42,
    )

    assert len(adjacent) == len(primary) == 16
    knowledge_on_math = next(
        row for row in primary if row["training_stage"] == "knowledge" and row["domain"] == "math"
    )
    assert knowledge_on_math["signed_change"] == pytest.approx(-0.1)
    assert knowledge_on_math["interference_loss"] == pytest.approx(0.1)
    assert knowledge_on_math["is_previously_trained_task"] is True
    by_domain = {row["domain"]: row for row in forgetting}
    assert by_domain["code"]["acquisition_score"] == pytest.approx(0.8)
    assert by_domain["code"]["forgetting"] == pytest.approx(0.3)
    assert by_domain["code"]["forgetting_rate"] == pytest.approx(0.375)
    assert by_domain["math"]["acquisition_score"] == pytest.approx(0.9)
    assert by_domain["if"]["forgetting"] == pytest.approx(0.0)
    assert continual["final_ACC"] == pytest.approx(0.675)
    assert continual["classical_BWT"] == pytest.approx((-0.3 - 0.2 - 0.2) / 3)
    assert continual["code_acquisition_gain"] == pytest.approx(0.4)
    assert continual["pretrained_base_referenced_FWT"] == pytest.approx((0.1 + 0.1 + 0.2) / 3)
    assert len(forward_transfer) == 3
    assert len(base_to_final) == 4


@pytest.mark.unit
def test_resolve_torch_dist_checkpoint_accepts_release_marker(tmp_path):
    checkpoint = tmp_path / "release"
    checkpoint.mkdir()
    (checkpoint / ".metadata").write_bytes(b"metadata")
    (checkpoint / "common.pt").write_bytes(b"common")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release\n")

    assert resolve_torch_dist_checkpoint(tmp_path) == checkpoint.resolve()
    assert resolve_torch_dist_checkpoint(checkpoint) == checkpoint.resolve()


@pytest.mark.unit
def test_paired_behavior_change_counts_binary_sample_transitions(tmp_path):
    before = _eval_dataset("math500", "math", 0.4, tmp_path)
    after = _eval_dataset("math500", "math", 0.6, tmp_path)

    result = paired_change(before, after, bootstrap_samples=1_000, seed=42)

    assert result["signed_change"] == pytest.approx(0.2)
    assert result["sample_pass_to_fail_count"] == 0
    assert result["sample_fail_to_pass_count"] == 2
    assert result["prompt_improved_count"] == 2
    assert result["prompt_regressed_count"] == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "script",
    [SEQUENCE_LAUNCHER, PROBE_LAUNCHER, BOUNDARY_EVAL_LAUNCHER, BOUNDARY_PROBE_LAUNCHER],
)
def test_sequential_shell_launchers_are_strict_and_syntax_valid(script):
    result = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    source = script.read_text()
    assert "set -euo pipefail" in source
