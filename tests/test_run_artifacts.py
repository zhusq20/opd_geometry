"""Tests for unambiguous run terminal state and artifact validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.optimizer_geometry.run_provenance import finish, start
from examples.optimizer_geometry.validate_run_artifacts import validate

NUM_GPUS = 0


@pytest.mark.unit
def test_successful_finish_removes_a_stale_failure_marker(tmp_path):
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "run_manifest.json").write_text(json.dumps({"status": "running"}) + "\n")
    (tmp_path / "run_complete.json").write_text(json.dumps({"status": "complete", "final_num_updates": 1}) + "\n")
    (tmp_path / "run_failed.json").write_text(json.dumps({"status": "failed", "exit_code": 1}) + "\n")

    result = finish(SimpleNamespace(run_dir=tmp_path, exit_code=0))

    assert result["status"] == "complete"
    assert not (tmp_path / "run_failed.json").exists()


@pytest.mark.unit
def test_validator_rejects_contradictory_success_and_failure_markers(tmp_path):
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "run_manifest.json").write_text(json.dumps({"status": "complete", "command": []}) + "\n")
    (tmp_path / "run_complete.json").write_text(json.dumps({"status": "complete", "final_num_updates": 0}) + "\n")
    (tmp_path / "run_failed.json").write_text(json.dumps({"status": "failed", "exit_code": 1}) + "\n")

    result = validate(SimpleNamespace(run_dir=tmp_path, expected_updates=0, require_eval=False))

    assert result["valid"] is False
    assert "run_failed.json exists for a purportedly completed run" in result["errors"]


@pytest.mark.unit
def test_resume_archives_old_terminal_markers_and_records_inputs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    input_path = repo / "config.yaml"
    input_path.write_text("value: 1\n")
    (repo / "train.py").write_text("print('unchanged source')\n")
    run = tmp_path / "run"

    initial = SimpleNamespace(
        repo=repo,
        run_dir=run,
        input=[str(input_path)],
        checkpoint=[],
        resume=False,
        allow_source_change=False,
        training_command=["python", "train.py", "--num-rollout", "200"],
    )
    initial_manifest = start(initial)
    assert Path(initial_manifest["inputs"][0]["archived_path"]).read_text() == "value: 1\n"
    (run / "run_complete.json").write_text(json.dumps({"status": "complete"}) + "\n")
    (run / "run_failed.json").write_text(json.dumps({"status": "failed"}) + "\n")

    resumed = SimpleNamespace(
        **{
            **initial.__dict__,
            "resume": True,
            "training_command": ["python", "train.py", "--num-rollout", "400"],
        }
    )
    manifest = start(resumed)

    assert manifest["status"] == "running"
    assert manifest["last_command"][-1] == "400"
    assert not (run / "run_complete.json").exists()
    assert not (run / "run_failed.json").exists()
    event = manifest["resume_events"][-1]
    assert set(event["archived_terminal_markers"]) == {"run_complete.json", "run_failed.json"}
    assert event["inputs"][0]["sha256"]
    for record in event["archived_terminal_markers"].values():
        assert Path(record["path"]).is_file()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
