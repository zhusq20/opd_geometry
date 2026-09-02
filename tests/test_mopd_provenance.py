import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from examples.mopd_gpas import convert_teachers, prepare_mopd
from examples.mopd_gpas.convert_teachers import (
    complete_hf,
    verify_compatibility,
    verify_source_identity,
    write_conversion_manifest,
)
from examples.mopd_gpas.prepare_mopd import SPLITS, sliced
from examples.mopd_gpas.prepare_resume import apply as apply_resume
from examples.mopd_gpas.prepare_resume import inspect as inspect_resume
from examples.mopd_gpas.provenance import checkpoint_record, selected_command_options, source_snapshot

NUM_GPUS = 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_config():
    return {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 151_936,
        "hidden_size": 2_048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "rope_theta": 1_000_000,
    }


def _write_hf_anchors(path: Path, *, with_shard: bool = True) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_model_config()), encoding="utf-8")
    (path / "tokenizer.json").write_text('{"tokenizer":"qwen3"}\n', encoding="utf-8")
    (path / "tokenizer_config.json").write_text('{"eos_token":"<eos>"}\n', encoding="utf-8")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    if with_shard:
        (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    files = [item for item in path.iterdir() if item.name != "conversion_manifest.json"]
    (path / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "output_files": [
                    {"path": item.name, "bytes": item.stat().st_size, "sha256": _sha256(item)} for item in files
                ],
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_manifest_hashes_the_selected_iteration_anchors(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    selected = checkpoint / "iter_0000007"
    selected.mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("7\n", encoding="utf-8")
    (selected / ".metadata").write_bytes(b"metadata")
    (selected / "common.pt").write_bytes(b"common")

    record = checkpoint_record(str(checkpoint))
    assert record["selected_checkpoint"]["selector"] == "7"
    assert record["selected_checkpoint"][".metadata"]["sha256"] == _sha256(selected / ".metadata")
    assert record["selected_checkpoint"]["common.pt"]["sha256"] == _sha256(selected / "common.pt")


def test_converted_teacher_requires_every_shard_named_by_the_index(tmp_path):
    teacher = tmp_path / "teacher"
    _write_hf_anchors(teacher, with_shard=False)
    assert not complete_hf(teacher)


def test_converted_teacher_must_match_student_architecture_and_tokenizer(tmp_path):
    base = tmp_path / "base"
    teacher = tmp_path / "teacher"
    _write_hf_anchors(base)
    _write_hf_anchors(teacher)
    verify_compatibility(base, teacher)

    (teacher / "tokenizer.json").write_text('{"tokenizer":"wrong"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer anchor"):
        verify_compatibility(base, teacher)


def test_published_teachers_can_be_verified_without_original_training_checkpoints(tmp_path, monkeypatch):
    base = tmp_path / "student_hf"
    teachers = tmp_path / "teachers_hf"
    _write_hf_anchors(base)
    teachers.mkdir()
    for task in ("math", "code", "if", "science"):
        _write_hf_anchors(teachers / task)

    monkeypatch.setenv("MOPD_HF_CHECKPOINT", str(base))
    monkeypatch.setenv("MOPD_TEACHER_HF_ROOT", str(teachers))
    monkeypatch.setattr(sys, "argv", ["convert_teachers.py", "--verify-only"])
    convert_teachers.main()


def test_fresh_teacher_conversion_manifest_pins_source_and_every_output(tmp_path):
    base = tmp_path / "base"
    teacher = tmp_path / "teacher"
    checkpoint = tmp_path / "checkpoint/iter_0000007"
    _write_hf_anchors(base)
    _write_hf_anchors(teacher)
    (teacher / "conversion_manifest.json").unlink()
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"metadata")
    (checkpoint / "common.pt").write_bytes(b"common")
    (checkpoint / "__0_0.distcp").write_bytes(b"checkpoint shard")

    manifest_path = write_conversion_manifest(
        teacher,
        task="math",
        step=7,
        checkpoint=checkpoint,
        base_hf=base,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_records = {record["path"]: record for record in manifest["output_files"]}

    shard = teacher / "model-00001-of-00001.safetensors"
    assert output_records[shard.name]["sha256"] == _sha256(shard)
    assert manifest["source"]["task"] == "math"
    assert manifest["source"]["step"] == 7
    assert manifest["source"]["checkpoint_anchors"]["common.pt"]["sha256"] == _sha256(checkpoint / "common.pt")
    verify_compatibility(base, teacher)
    verify_source_identity(teacher, task="math", step=7, checkpoint=checkpoint, base_hf=base)

    (checkpoint / "common.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source anchor"):
        verify_source_identity(teacher, task="math", step=7, checkpoint=checkpoint, base_hf=base)


def test_checkpoint_manifest_records_hf_model_index_and_tokenizer(tmp_path):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    record = checkpoint_record(str(checkpoint))
    assert record["config.json"]["sha256"] == _sha256(checkpoint / "config.json")
    assert record["model.safetensors.index.json"]["sha256"] == _sha256(checkpoint / "model.safetensors.index.json")
    assert record["tokenizer.json"]["sha256"] == _sha256(checkpoint / "tokenizer.json")


def test_source_snapshot_includes_source_and_excludes_generated_files(tmp_path):
    repo = tmp_path / "repo"
    example = repo / "examples/mopd_gpas"
    example.mkdir(parents=True)
    (example / "run.py").write_text("print('exact source')\n", encoding="utf-8")
    generated = example / "generated"
    generated.mkdir()
    (generated / "output.jsonl").write_text("not source\n", encoding="utf-8")

    record = source_snapshot(repo, repo / "run", [str(example)], "source_snapshot.tar.gz")
    assert record is not None
    assert [item["path"] for item in record["files"]] == ["examples/mopd_gpas/run.py"]


def test_training_and_teacher_loss_splits_are_disjoint(tmp_path):
    path = tmp_path / "code.jsonl"
    assert sliced(path, "train").endswith("code.jsonl@[0:16000]")
    assert sliced(path, "probe").endswith("code.jsonl@[16000:16064]")
    assert sliced(path, "bank").endswith("code.jsonl@[16064:16192]")
    assert sliced(path, "teacher_loss").endswith("code.jsonl@[16384:16575]")
    assert SPLITS["train"][1] <= SPLITS["probe"][0]
    assert SPLITS["probe"][1] <= SPLITS["bank"][0]
    assert SPLITS["bank"][1] <= SPLITS["teacher_loss"][0]


def test_protocol_preparation_uses_portable_asset_roots(tmp_path, monkeypatch):
    data_root = tmp_path / "data/m2rl"
    base_hf = tmp_path / "models/student_hf"
    base_megatron = tmp_path / "models/student_megatron"
    teachers = tmp_path / "models/teachers_hf"
    output = tmp_path / "generated"

    base_hf.mkdir(parents=True)
    (base_hf / "config.json").write_text("{}\n", encoding="utf-8")
    base_megatron.mkdir(parents=True)
    (base_megatron / "latest_checkpointed_iteration.txt").write_text("0\n", encoding="utf-8")
    for task in ("math", "code", "if", "science"):
        data_path = data_root / "train" / f"{task}.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text("{}\n" * 16_575, encoding="utf-8")
        teacher = teachers / task
        teacher.mkdir(parents=True)
        (teacher / "config.json").write_text("{}\n", encoding="utf-8")
        (teacher / "model.safetensors").write_bytes(b"weights")
        (teacher / "conversion_manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MOPD_DATA_ROOT", str(data_root))
    monkeypatch.setenv("MOPD_HF_CHECKPOINT", str(base_hf))
    monkeypatch.setenv("MOPD_BASE_MEGATRON", str(base_megatron))
    monkeypatch.setenv("MOPD_TEACHER_HF_ROOT", str(teachers))
    monkeypatch.setattr(sys, "argv", ["prepare_mopd.py", "--repo", str(tmp_path), "--output", str(output)])
    prepare_mopd.main()

    train = yaml.safe_load((output / "train.yaml").read_text(encoding="utf-8"))
    assert train["sources"][0]["path"].startswith(str(data_root.resolve()))
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["teachers"]["math"]["converted_hf"] == str((teachers / "math").resolve())
    assert "checkpoint" not in protocol["teachers"]["math"]
    assert "sha256" not in protocol["datasets"]["math"]


def test_capability_eval_resolves_the_worker_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MOPD_DATA_ROOT", str(tmp_path / "m2rl"))
    config_path = Path("examples/mopd_gpas/configs/capability_eval.yaml")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert config["eval"]["datasets"][0]["path"] == str(tmp_path / "m2rl/eval/m2rl_online/math500.parquet")


def test_complete_worker_run_can_be_packaged_for_analysis(tmp_path):
    config = "uniform_k1_conventional"
    output_root = tmp_path / "outputs"
    run = output_root / f"{config}-seed42"
    capability = run / "capability_eval/response_64000"
    for path in (
        run / "provenance",
        run / "allocation",
        run / "metrics",
        run / "teacher_loss_eval",
        capability,
    ):
        path.mkdir(parents=True, exist_ok=True)
        (path / "artifact.json").write_text("{}\n", encoding="utf-8")
    (run / "run_complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    (run / "allocation/allocation.jsonl").write_text('{"attempted_responses_after":64000}\n', encoding="utf-8")
    (capability / "run_complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")

    package_dir = tmp_path / "packages"
    script = Path(__file__).parents[1] / "examples/mopd_gpas/validate_and_package_run.sh"
    subprocess.run(
        ["bash", str(script), config],
        check=True,
        env={
            **os.environ,
            "MOPD_OUTPUT_ROOT": str(output_root),
            "MOPD_PACKAGE_DIR": str(package_dir),
        },
    )
    archive = package_dir / f"{config}-seed42-analysis.tar.gz"
    with tarfile.open(archive) as stream:
        names = stream.getnames()
    assert f"{config}-seed42/allocation/allocation.jsonl" in names
    assert not any("checkpoints" in name for name in names)


def test_protocol_cli_exposes_optimizer_parallelism_and_evaluation_settings():
    command = [
        "python3",
        "train.py",
        "--optimizer",
        "adam",
        "--clip-grad",
        "1.0",
        "--tensor-model-parallel-size",
        "1",
        "--eval-interval",
        "50",
    ]
    assert selected_command_options(command) == {
        "optimizer": "adam",
        "clip_grad": "1.0",
        "tensor_model_parallel_size": "1",
        "eval_interval": "50",
    }


def test_resume_rewinds_append_only_outputs_to_latest_complete_checkpoint(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/iter_0000009"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"metadata")
    (checkpoint / "common.pt").write_bytes(b"common")
    sampler = run / "checkpoints/rollout/mopd_dataset_state_dict_9.pt"
    sampler.parent.mkdir()
    torch.save(
        {
            "controller": {
                "completed_operations": 2,
                "attempted_responses": 16_384,
                "resident_teacher": 1,
                "pending": None,
            }
        },
        sampler,
    )
    (run / "checkpoints/mopd_checkpoint_index.json").write_text(
        json.dumps(
            [
                {
                    "rollout_id": 9,
                    "operation_index": 1,
                    "attempted_responses": 16_384,
                    "optimizer_updates": 2,
                }
            ]
        ),
        encoding="utf-8",
    )
    allocation = run / "allocation/allocation.jsonl"
    allocation.parent.mkdir()
    allocation.write_text(
        "".join(json.dumps({"operation_index": index, "rollout_id": 8 + index}) + "\n" for index in range(3)),
        encoding="utf-8",
    )
    metrics = run / "metrics"
    metrics.mkdir()
    (metrics / "mopd.jsonl").write_text(
        "".join(json.dumps({"metrics": {"mopd/update": index}}) + "\n" for index in range(3)),
        encoding="utf-8",
    )
    (run / "wandb_run_id.txt").write_text("stale_run\n", encoding="utf-8")
    (run / "run_failed.json").write_text('{"status":"failed"}\n', encoding="utf-8")

    state = inspect_resume(run)
    assert state["rollout_id"] == 9
    assert state["resident_task"] == "code"
    assert state["rewind_required"]
    assert state["eval_on_start"]

    applied = apply_resume(run)
    assert Path(applied["archive"]).is_file()
    assert not (run / "wandb_run_id.txt").exists()
    assert len((run / "allocation/allocation.jsonl").read_text().splitlines()) == 2
    assert len((metrics / "mopd.jsonl").read_text().splitlines()) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
