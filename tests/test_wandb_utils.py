import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from slime.utils import logging_utils, wandb_utils
from slime.utils.wandb_utils import _args_to_config_dict, _log_provenance_artifact, _resolve_wandb_run_id

NUM_GPUS = 0


@pytest.mark.unit
def test_wandb_config_never_contains_cli_key():
    config = _args_to_config_dict(
        SimpleNamespace(wandb_key="must-not-be-logged", experiment_task="math", optimizer="adam")
    )

    assert "wandb_key" not in config
    assert config["experiment_task"] == "math"


@pytest.mark.unit
def test_wandb_run_id_is_durable_and_reused(tmp_path, monkeypatch):
    id_path = tmp_path / "wandb_run_id.txt"
    args = SimpleNamespace(wandb_run_id=None, wandb_run_id_file=str(id_path))
    monkeypatch.setattr(wandb_utils.wandb.util, "generate_id", lambda: "fixed_run_42")

    assert _resolve_wandb_run_id(args) == "fixed_run_42"
    assert id_path.read_text() == "fixed_run_42\n"
    assert _resolve_wandb_run_id(args) == "fixed_run_42"


@pytest.mark.unit
def test_scalar_events_are_durable_even_without_wandb(tmp_path):
    args = SimpleNamespace(metrics_output_dir=str(tmp_path), use_wandb=False, use_tensorboard=False)

    logging_utils.log(args, {"train/step": 3, "train/loss": 0.25}, step_key="train/step")

    record = json.loads((tmp_path / "train.jsonl").read_text())
    assert record["step_key"] == "train/step"
    assert record["metrics"] == {"train/loss": 0.25, "train/step": 3}


@pytest.mark.unit
def test_concurrent_scalar_events_remain_complete_jsonl_records(tmp_path):
    args = SimpleNamespace(metrics_output_dir=str(tmp_path), use_wandb=False, use_tensorboard=False)

    def write_event(step: int) -> None:
        logging_utils.log(args, {"train/step": step, "train/loss": step / 10}, step_key="train/step")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_event, range(24)))

    records = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len(records) == 24
    assert sorted(record["metrics"]["train/step"] for record in records) == list(range(24))


@pytest.mark.unit
def test_wandb_provenance_artifact_contains_manifest_and_source_snapshot(tmp_path, monkeypatch):
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    manifest = provenance / "run_manifest.json"
    snapshot = provenance / "source_snapshot.tar.gz"
    inputs = provenance / "inputs"
    inputs.mkdir()
    reward_config = inputs / "reward.yaml"
    manifest.write_text("{}\n")
    snapshot.write_bytes(b"source")
    reward_config.write_text("routes: {}\n")
    added = []

    class Artifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add_file(self, path, name):
            added.append((path, name))

    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="run42"))
    monkeypatch.setattr(wandb_utils.wandb, "Artifact", Artifact)
    monkeypatch.setattr(wandb_utils.wandb, "log_artifact", lambda artifact: added.append((artifact, "logged")))

    _log_provenance_artifact(SimpleNamespace(run_manifest_path=str(manifest)))

    assert (str(manifest), "run_manifest.json") in added
    assert (str(snapshot), "source_snapshot.tar.gz") in added
    assert (str(reward_config), "inputs/reward.yaml") in added
    assert added[-1][1] == "logged"


@pytest.mark.integration
def test_real_offline_wandb_persists_id_metrics_provenance_and_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    manifest = provenance / "run_manifest.json"
    manifest.write_text(json.dumps({"status": "running", "command": ["train.py"]}) + "\n")
    (provenance / "source_snapshot.tar.gz").write_bytes(b"source snapshot")

    def make_args():
        return SimpleNamespace(
            use_wandb=True,
            wandb_mode="offline",
            wandb_key=None,
            wandb_host=None,
            wandb_run_id=None,
            wandb_run_id_file=str(tmp_path / "wandb_run_id.txt"),
            wandb_group="offline_integration",
            wandb_run_name="offline_integration",
            wandb_random_suffix=False,
            wandb_team="zsqzz",
            wandb_project="iclr2027-opd-geometry-test",
            wandb_dir=str(tmp_path / "wandb"),
            rank=0,
            run_manifest_path=str(manifest),
            metrics_output_dir=str(tmp_path / "metrics"),
            completion_marker_path=str(tmp_path / "run_complete.json"),
            use_tensorboard=False,
            use_critic=False,
            experiment_task="code",
            experiment_condition="grpo",
            experiment_optimizer="adamw",
            num_rollout=1,
            start_rollout_id=0,
            experiment_name="offline_integration",
        )

    first = make_args()
    try:
        logging_utils.init_tracking(first)
        first_id = first.wandb_run_id
        logging_utils.log(first, {"train/step": 1, "train/loss": 0.25}, "train/step")
    finally:
        logging_utils.finish_tracking(first)

    resumed = make_args()
    try:
        logging_utils.init_tracking(resumed)
        assert resumed.wandb_run_id == first_id
        logging_utils.log(resumed, {"rollout/step": 1, "rollout/reward/code": 0.5}, "rollout/step")
        logging_utils.log(resumed, {"eval/step": 1, "eval/livecodebench/pass@1": 0.4}, "eval/step")
        logging_utils.log(resumed, {"geometry/step": 1, "geometry/global/update_norm": 0.01}, "geometry/step")
        logging_utils.log(resumed, {"forgetting/step": 1, "forgetting/code": 0.0}, "forgetting/step")
        logging_utils.mark_run_complete(resumed, final_num_updates=1)
    finally:
        logging_utils.finish_tracking(resumed)

    assert (tmp_path / "wandb_run_id.txt").read_text().strip() == first_id
    marker = json.loads((tmp_path / "run_complete.json").read_text())
    assert marker["wandb_run_id"] == first_id
    assert marker["final_num_updates"] == 1
    assert {path.name for path in (tmp_path / "metrics").glob("*.jsonl")} == {
        "eval.jsonl",
        "forgetting.jsonl",
        "geometry.jsonl",
        "rollout.jsonl",
        "train.jsonl",
    }
    assert any(path.suffix == ".wandb" for path in (tmp_path / "wandb").rglob("*"))
