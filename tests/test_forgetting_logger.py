"""Unit test for max-so-far forgetting and backward-transfer persistence."""

import json
from types import SimpleNamespace

import pytest

from slime.utils.types import Sample
from slime_plugins.geometry import forgetting as forgetting_module
from slime_plugins.geometry.forgetting import log_eval_and_forgetting

NUM_GPUS = 0


def test_forgetting_tracks_best_and_baseline(tmp_path):
    args = SimpleNamespace(forgetting_output_dir=str(tmp_path), geometry_output_dir=None)
    assert log_eval_and_forgetting(0, args, {"math": {"rewards": [0.4, 0.6]}}, {}) is False
    log_eval_and_forgetting(1, args, {"math": {"rewards": [0.9]}, "code": {"rewards": [0.2]}}, {})
    log_eval_and_forgetting(2, args, {"math": {"rewards": [0.7]}, "code": {"rewards": [0.5]}}, {})

    records = [json.loads(line) for line in (tmp_path / "forgetting" / "metrics.jsonl").read_text().splitlines()]
    assert records[-1]["tasks"]["math"]["forgetting"] == pytest.approx(0.2)
    assert records[-1]["tasks"]["math"]["backward_transfer"] == pytest.approx(0.2)
    assert records[-1]["tasks"]["code"]["backward_transfer"] == pytest.approx(0.3)
    assert records[-1]["ACC"] == pytest.approx(0.6)
    state = json.loads((tmp_path / "forgetting" / "state.json").read_text())
    assert len(state["performance_matrix"]) == 3
    assert state["performance_matrix"][-1]["scores"] == {"code": 0.5, "math": 0.7}


def test_forgetting_tracks_fixed_sample_pass_fail_transitions(tmp_path):
    args = SimpleNamespace(
        forgetting_output_dir=str(tmp_path),
        geometry_output_dir=None,
        experiment_task="phase_math",
    )
    samples = [
        Sample(index=10, status=Sample.Status.COMPLETED),
        Sample(index=11, status=Sample.Status.COMPLETED),
    ]
    log_eval_and_forgetting(
        0,
        args,
        {"heldout": {"rewards": [1.0, 0.0], "samples": samples}},
        {"eval/num_updates": 0, "eval/model_version": 0},
    )
    log_eval_and_forgetting(
        1,
        args,
        {"heldout": {"rewards": [0.0, 1.0], "samples": samples}},
        {"eval/num_updates": 1, "eval/model_version": 1},
    )

    record = json.loads((tmp_path / "forgetting" / "metrics.jsonl").read_text().splitlines()[-1])
    task = record["tasks"]["heldout"]
    assert task["pass_to_fail_count"] == 1
    assert task["fail_to_pass_count"] == 1
    assert task["matched_sample_count"] == 2
    assert task["pass_rate"] == pytest.approx(0.5)
    assert task["response_error_rate"] == pytest.approx(0.0)
    assert record["probe_metric_availability"]["nll"] == "not_collected_eval_interface_returns_reward_only"


def test_forgetting_rejects_a_replayed_fixed_probe(tmp_path):
    args = SimpleNamespace(forgetting_output_dir=str(tmp_path), geometry_output_dir=None)
    payload = {"math": {"rewards": [1.0]}}
    metrics = {"eval/num_updates": 5, "eval/model_version": 5, "eval/phase": "post_update"}

    log_eval_and_forgetting(4, args, payload, metrics)

    with pytest.raises(ValueError, match="refusing to double-count"):
        log_eval_and_forgetting(4, args, payload, metrics)


def test_forgetting_recovers_a_metric_written_before_state_commit(tmp_path, monkeypatch):
    args = SimpleNamespace(forgetting_output_dir=str(tmp_path), geometry_output_dir=None)

    def post_update(step):
        return {
            "eval/num_updates": step,
            "eval/model_version": step,
            "eval/phase": "post_update",
        }

    log_eval_and_forgetting(0, args, {"math": {"rewards": [0.4]}}, post_update(0))
    log_eval_and_forgetting(1, args, {"math": {"rewards": [0.6]}}, post_update(1))

    original_atomic_json = forgetting_module._atomic_json

    def fail_state_commit(path, payload):
        if path.name == "state.json":
            raise PermissionError("simulated state commit failure")
        original_atomic_json(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(forgetting_module, "_atomic_json", fail_state_commit)
        with pytest.raises(PermissionError, match="simulated state commit failure"):
            log_eval_and_forgetting(2, args, {"math": {"rewards": [0.1]}}, post_update(2))

    metrics_path = tmp_path / "forgetting" / "metrics.jsonl"
    assert len(metrics_path.read_text().splitlines()) == 3

    log_eval_and_forgetting(2, args, {"math": {"rewards": [0.8]}}, post_update(2))

    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(records) == 3
    assert records[-1]["tasks"]["math"]["score"] == pytest.approx(0.8)
    state = json.loads((tmp_path / "forgetting" / "state.json").read_text())
    assert len(state["performance_matrix"]) == 3
    assert state["last_evaluation_key"] == "2:2:post_update"
    recovery_paths = list((tmp_path / "forgetting" / "recovery").glob("uncommitted_metrics.*.jsonl"))
    assert len(recovery_paths) == 1
    recovered = json.loads(recovery_paths[0].read_text())
    assert recovered["tasks"]["math"]["score"] == pytest.approx(0.1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
