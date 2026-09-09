import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from examples.mopd_gpas import convert_teachers, measure_initial_kl, prepare_mopd, provenance, verify_hardware
from examples.mopd_gpas.convert_teachers import (
    complete_hf,
    verify_compatibility,
    verify_source_identity,
    write_conversion_manifest,
)
from examples.mopd_gpas.prepare_mopd import (
    HELDOUT_CANDIDATE_SLICE,
    TRAIN_CANDIDATE_SLICE,
    hardware_spec,
    objective_from_measurement,
    sliced,
)
from examples.mopd_gpas.prepare_resume import apply as apply_resume
from examples.mopd_gpas.prepare_resume import inspect as inspect_resume
from examples.mopd_gpas.provenance import checkpoint_record, selected_command_options, source_snapshot
from omegaconf import OmegaConf
from slime_plugins.mopd.prompting import EMPTY_THINKING_SUFFIX, PROMPT_FORMAT

NUM_GPUS = 0


def test_smollm3_mixsft_profile_uses_fresh_paths_and_pinned_weights(monkeypatch):
    from examples.mopd_gpas.profiles import PAPER_RUNS, PAPER_TOPK, profile

    for key in tuple(os.environ):
        if key.startswith("MOPD_"):
            monkeypatch.delenv(key)
    spec = profile("smollm3")
    assert spec["student_repo"] == "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT"
    assert spec["student_revision"] == "c9e7bad031667828656ead188d7e8ea162c048a4"
    assert spec["asset_root"].name == "mopd_smollm3_mixsft_assets"
    assert spec["generated"].name == "mopd_smollm3_mixsft_generated"
    assert PAPER_RUNS["m-intersection64-dr"] == ("topk_intersection", "domain_response", False)
    assert PAPER_TOPK["m-intersection64-dr"] == 64


def test_fetch_refuses_to_overwrite_base_assets_as_mixsft(tmp_path, monkeypatch):
    from examples.mopd_gpas import fetch_profile

    previous = json.dumps({"student": {"repository": "HuggingFaceTB/SmolLM3-3B-Base", "revision": "old"}})
    (tmp_path / "assets.json").write_text(previous)
    monkeypatch.setenv("MOPD_PROFILE", "smollm3")
    monkeypatch.setenv("MOPD_ASSET_ROOT", str(tmp_path))
    monkeypatch.delenv("MOPD_STUDENT_REPO", raising=False)
    monkeypatch.delenv("MOPD_STUDENT_REVISION", raising=False)
    monkeypatch.setattr(fetch_profile, "fetch", lambda *a, **kw: pytest.fail("must fail before downloading"))
    with pytest.raises(ValueError, match="Student identity changed"):
        fetch_profile.main()
    assert (tmp_path / "assets.json").read_text() == previous


def test_smollm3_entry_rejects_prepared_base_before_starting_training(tmp_path):
    identity = {"student": {"repository": "HuggingFaceTB/SmolLM3-3B-Base", "revision": "old"}}
    (tmp_path / "assets.json").write_text(json.dumps(identity))
    (tmp_path / "protocol.json").write_text(json.dumps({"assets": identity}))
    env = {key: value for key, value in os.environ.items() if not key.startswith("MOPD_")}
    env.update(MOPD_ASSET_ROOT=str(tmp_path), MOPD_GENERATED_DIR=str(tmp_path))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "examples/mopd_gpas/run_smollm3.sh"), "pg", "smoke"],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires the pinned MixSFT student" in result.stderr
    assert not (tmp_path / "provenance").exists()


@pytest.mark.parametrize("previous,next_loss", [("teacher_topk", "student_topk"), ("student_topk", "topk_intersection")])
def test_resume_rejects_switching_loss_before_writing(tmp_path, previous, next_loss):
    run = tmp_path / "run"
    manifest = run / "provenance/run_manifest.json"
    manifest.parent.mkdir(parents=True)
    contents = json.dumps({"protocol_cli": {"mopd_loss": previous}, "status": "failed"})
    manifest.write_text(contents)
    with pytest.raises(ValueError, match="Cannot resume.*new run ID"):
        provenance.resume(SimpleNamespace(repo=tmp_path, run_dir=run,
                                           training_command=["train.py", "--mopd-loss", next_loss]))
    assert manifest.read_text() == contents
    assert list(manifest.parent.iterdir()) == [manifest]


@pytest.mark.parametrize("previous_k", [None, 16])
@pytest.mark.parametrize("loss", ["student_topk", "topk_intersection"])
def test_resume_rejects_changing_student_topk_size(tmp_path, previous_k, loss):
    run = tmp_path / "run"
    manifest = run / "provenance/run_manifest.json"
    manifest.parent.mkdir(parents=True)
    options = {"mopd_loss": loss}
    if previous_k is not None:
        options["mopd_topk"] = str(previous_k)
    contents = json.dumps({"protocol_cli": options, "status": "failed"})
    manifest.write_text(contents)
    with pytest.raises(ValueError, match="Cannot resume student Top16 as Top64"):
        provenance.resume(SimpleNamespace(repo=tmp_path, run_dir=run,
                                           training_command=["train.py", "--mopd-loss", loss, "--mopd-topk", "64"]))
    assert manifest.read_text() == contents


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


def test_provenance_records_nvidia_smi_stdout_failure(monkeypatch):
    failure = subprocess.CompletedProcess([], 255, "Failed to initialize NVML: Unknown Error\n", "")
    monkeypatch.setattr(provenance.subprocess, "run", lambda *_args, **_kwargs: failure)

    record = provenance.hardware_record()["nvidia_smi"]

    assert record["available"] is False
    assert record["gpus"] == []
    assert record["error"] == "Failed to initialize NVML: Unknown Error"


def test_hardware_preflight_retries_then_reports_nvidia_smi_stdout(monkeypatch):
    calls = []
    failure = subprocess.CompletedProcess([], 255, "Failed to initialize NVML: Unknown Error\n", "")
    monkeypatch.setattr(verify_hardware, "QUERY_RETRY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(
        verify_hardware.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or failure,
    )

    with pytest.raises(RuntimeError, match="Failed to initialize NVML: Unknown Error"):
        verify_hardware.query()

    assert len(calls) == verify_hardware.QUERY_ATTEMPTS


def test_dual_48gb_profile_requires_two_distinct_48gb_training_gpus():
    inventory = {
        index: {
            "name": "NVIDIA RTX A6000",
            "memory_total_mib": 49_140,
            "uuid": f"GPU-{index}",
            "driver_version": "550.163.01",
        }
        for index in range(3)
    }

    record = verify_hardware.validate("dual-48gb-tp2", [0, 1], 2, inventory)

    assert record["tensor_model_parallel_size"] == 2
    assert [device["index"] for device in record["training_gpus"]] == [0, 1]
    with pytest.raises(ValueError, match="assigned|distinct"):
        verify_hardware.validate("dual-48gb-tp2", [0, 1], 1, inventory)
    with pytest.raises(ValueError, match="requires 2 training GPU"):
        verify_hardware.validate("dual-48gb-tp2", [0], 2, inventory)


def test_hardware_protocol_profiles_change_only_the_declared_topology():
    frozen = hardware_spec("frozen-96gb-tp1")
    a6000 = hardware_spec("dual-48gb-tp2")

    assert (frozen["gpus_per_run"], frozen["tensor_model_parallel_size"]) == (2, 1)
    assert (a6000["gpus_per_run"], a6000["tensor_model_parallel_size"]) == (3, 2)
    assert frozen["teacher_order"] == a6000["teacher_order"] == ["math", "code", "if", "science"]


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


def test_converted_teacher_must_match_source_architecture_and_tokenizer(tmp_path):
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

    monkeypatch.setenv("MOPD_TEACHER_BASE_HF", str(base))
    monkeypatch.setenv("MOPD_TEACHER_HF_ROOT", str(teachers))
    monkeypatch.setattr(sys, "argv", ["convert_teachers.py", "--verify-only"])
    convert_teachers.main()


def test_rl_teacher_requires_the_exact_source_tokenizer_config(tmp_path):
    base = tmp_path / "base"
    teacher = tmp_path / "teacher"
    _write_hf_anchors(base)
    _write_hf_anchors(teacher)
    verify_compatibility(base, teacher)
    (teacher / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        verify_compatibility(base, teacher)


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


def test_source_snapshot_survives_source_truncation_and_hashes_the_archived_bytes(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "run.py"
    original = b"print('the complete source captured at launch')\n"
    source.write_bytes(original)
    original_addfile = tarfile.TarFile.addfile

    def truncate_source_before_archiving(archive, info, fileobj=None):
        source.write_bytes(b"x")
        return original_addfile(archive, info, fileobj)

    monkeypatch.setattr(tarfile.TarFile, "addfile", truncate_source_before_archiving)
    record = source_snapshot(repo, repo / "run", [str(source)], "source_snapshot.tar.gz")
    with tarfile.open(record["path"]) as archive:
        assert archive.extractfile("run.py").read() == original
    assert record["files"] == [
        {"path": "run.py", "bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()}
    ]
    assert source.read_bytes() == b"x"


def test_training_and_teacher_loss_splits_are_disjoint(tmp_path):
    path = tmp_path / "code.jsonl"
    assert sliced(path, TRAIN_CANDIDATE_SLICE).endswith("code.jsonl@[0:16384]")
    assert sliced(path, HELDOUT_CANDIDATE_SLICE).endswith("code.jsonl@[16384:16575]")
    assert TRAIN_CANDIDATE_SLICE[1] <= HELDOUT_CANDIDATE_SLICE[0]


def test_initial_loss_ratio_over_ten_switches_to_equal_weights(tmp_path):
    path = tmp_path / "initial.json"
    value = {
        "schema_version": 2,
        "task_order": ["math", "code", "if", "science"],
        "tasks": {
            task: {"ell0": loss}
            for task, loss in zip(("math", "code", "if", "science"), (0.01, 0.2, 0.3, 0.4), strict=True)
        },
        "weight_rule": "equal_due_to_max_min_ratio_gt_10",
        "target_weights": dict.fromkeys(("math", "code", "if", "science"), 0.25),
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    losses, weights, _ = objective_from_measurement(path)
    assert losses["math"] == 0.01
    assert weights == dict.fromkeys(("math", "code", "if", "science"), 0.25)


@pytest.mark.parametrize("heldout_only", [False, True])
@pytest.mark.parametrize("previous_protocol", ["v5", "unmarked_manifest"])
def test_protocol_preparation_preserves_frozen_old_prompts(tmp_path, monkeypatch, heldout_only, previous_protocol):
    output = tmp_path / "old_generated"
    heldout = output / "heldout/math.jsonl"
    heldout.parent.mkdir(parents=True)
    heldout.write_text("old frozen prompts\n")
    if previous_protocol == "v5":
        (output / "protocol.json").write_text(json.dumps({"schema_version": 5}))
    else:
        (output / "train.yaml").write_text("version: 4\n")
    original = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    command = ["prepare_mopd.py", "--repo", str(tmp_path), "--output", str(output)]
    if heldout_only:
        command.append("--heldout-only")
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(ValueError, match="select a new output directory"):
        prepare_mopd.main()
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()} == original


def test_legacy_initial_kl_rejects_asymmetric_protocol_before_loading_models(tmp_path, monkeypatch):
    (tmp_path / "protocol.json").write_text(json.dumps({"schema_version": 6, "prompt_format": PROMPT_FORMAT}))
    output = tmp_path / "initial_kl.json"
    output.write_text("preserve previous measurement\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["measure_initial_kl.py", "--heldout-dir", str(tmp_path / "heldout"), "--output", str(output)],
    )
    with pytest.raises(ValueError, match="only supports the legacy same-prefix protocol"):
        measure_initial_kl.main()
    assert output.read_text() == "preserve previous measurement\n"


def test_protocol_preparation_uses_portable_asset_roots(tmp_path, monkeypatch):
    data_root = tmp_path / "data/m2rl"
    base_hf = tmp_path / "models/qwen3-1.7b-base"
    base_megatron = tmp_path / "models/qwen3-1.7b-base_torch_dist"
    teachers = tmp_path / "models/teachers_hf"
    output = tmp_path / "generated"

    base_hf.parent.mkdir(parents=True)
    _write_hf_anchors(base_hf)
    base_megatron.mkdir(parents=True)
    (base_megatron / "latest_checkpointed_iteration.txt").write_text("0\n", encoding="utf-8")
    for task in ("math", "code", "if", "science"):
        data_path = data_root / "train" / f"{task}.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text("{}\n" * 16_575, encoding="utf-8")
    teachers.mkdir()
    for task in ("math", "code", "if", "science"):
        _write_hf_anchors(teachers / task)

    initial_path = output / "initial_kl.json"
    initial_path.parent.mkdir()
    # Preparation must not depend on an initial loss measurement or its weights.
    assert not initial_path.exists()

    def materialize(_data_root, generated, _student):
        records = {}
        for task in ("math", "code", "if", "science"):
            path = generated / "heldout" / f"{task}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"prompt":"p"}\n' * 128, encoding="utf-8")
            records[task] = {"selected": 64, "file": prepare_mopd.file_record(path)}
        return records

    monkeypatch.setattr(prepare_mopd, "materialize_heldout", materialize)

    monkeypatch.setenv("MOPD_DATA_ROOT", str(data_root))
    monkeypatch.setenv("MOPD_HF_CHECKPOINT", str(base_hf))
    monkeypatch.setenv("MOPD_BASE_MEGATRON", str(base_megatron))
    monkeypatch.setenv("MOPD_TEACHER_HF_ROOT", str(teachers))
    monkeypatch.setenv("MOPD_INITIAL_KL", str(initial_path))
    monkeypatch.setattr(sys, "argv", ["prepare_mopd.py", "--repo", str(tmp_path), "--output", str(output)])
    prepare_mopd.main()

    train = yaml.safe_load((output / "train.yaml").read_text(encoding="utf-8"))
    assert train["sources"][0]["path"].startswith(str(data_root.resolve()))
    assert train["version"] == 4
    assert train["prompt_format"] == PROMPT_FORMAT
    assert all(source["chat_template_suffix_to_remove"] == EMPTY_THINKING_SUFFIX for source in train["sources"])
    diagnostic = yaml.safe_load((output / "diagnostic.yaml").read_text(encoding="utf-8"))
    assert diagnostic["prompt_format"] == PROMPT_FORMAT
    assert all(source["chat_template_suffix_to_remove"] is None for source in diagnostic["sources"])
    heldout = yaml.safe_load((output / "teacher_loss_eval.yaml").read_text(encoding="utf-8"))
    assert heldout["eval"]["defaults"]["chat_template_suffix_to_remove"] is None
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["teachers"]["math"]["model_path"] == str((teachers / "math").resolve())
    assert protocol["student"]["model"] == "Qwen3-1.7B-Base"
    assert protocol["teachers"]["code"]["model_path"] == str((teachers / "code").resolve())
    assert protocol["teachers"]["science"]["model_path"] == str((teachers / "science").resolve())
    assert protocol["training"]["steps"] == 500
    assert protocol["training"]["configs"] == ["uniform-s1", "gpas-s1", "gpas-raw-s1", "d3-fixed-s1"]
    assert protocol["schema_version"] == 6
    assert protocol["prompt_format"] == PROMPT_FORMAT
    assert protocol["objective"]["loss"] == "teacher_top64_corrected_reverse_kl"
    assert all(not teacher["temporary_substitute"] for teacher in protocol["teachers"].values())
    assert protocol["objective"]["weights"] == dict.fromkeys(("math", "code", "if", "science"), 0.25)
    baselines = protocol["objective"]["paper_baselines"]
    assert baselines["d3_fixed"]["scheduler"] == {
        "update_cadence": 10,
        "window": 10,
        "windows": 3,
        "initial_steps": 5,
        "ema_window": 10,
        "kl_denominator_floor": 0.15,
        "temperature": 0.5,
        "mixture_floor": 0.1,
        "jitter": 0.3,
    }
    assert set(baselines) == {"d3_fixed"}
    assert protocol["common_checkpoint"]["generated_responses"] == 1792


@pytest.mark.parametrize("name", ["capability_eval.yaml", "capability_eval_noncode.yaml"])
def test_capability_eval_resolves_the_worker_data_root(tmp_path, monkeypatch, name):
    monkeypatch.setenv("MOPD_DATA_ROOT", str(tmp_path / "m2rl"))
    config_path = Path("examples/mopd_gpas/configs") / name
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert config["eval"]["datasets"][0]["path"] == str(tmp_path / "m2rl/eval/m2rl_online/math500.parquet")
    assert config["eval"]["defaults"]["chat_template_suffix_to_remove"] == EMPTY_THINKING_SUFFIX


@pytest.mark.parametrize("name", ["teacher_router.yaml", "teacher_router_distributed.yaml"])
def test_teacher_routes_keep_the_empty_thinking_prefix(name):
    config = yaml.safe_load((Path("examples/mopd_gpas/configs") / name).read_text())
    assert set(config["teachers"]) == {"math", "code", "if", "science"}
    assert all(route["prompt_suffix"] == EMPTY_THINKING_SUFFIX for route in config["teachers"].values())


@pytest.mark.parametrize("target", ["initial_student", "teacher_math"])
@pytest.mark.parametrize("protocol_version", [5, 6])
def test_capability_launcher_binds_prefixes_to_the_current_protocol(tmp_path, target, protocol_version):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "train.yaml").write_text("{}\n")
    (generated / "protocol.json").write_text(
        json.dumps({"schema_version": protocol_version, "prompt_format": PROMPT_FORMAT})
    )
    model = tmp_path / "models/math"
    model.mkdir(parents=True)
    for name in ("config.json", "model.safetensors"):
        (model / name).write_text("{}\n")
    data = tmp_path / "data"
    for name in ("eval/m2rl_online/eval_data_index.json", "single_task/code/livecodebench_index_v6.json"):
        path = data / name
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
    commands = tmp_path / "commands.json"
    binary = tmp_path / "bin/python3"
    binary.parent.mkdir()
    # Run the launcher's config generation, capturing its final dry-run CLI
    # before it imports GPU runtime dependencies.
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    pathlib.Path(os.environ['CAPTURE_COMMAND']).write_text(json.dumps(sys.argv[3:]))\n"
        "else:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    binary.chmod(0o755)
    script = Path(__file__).parents[1] / "examples/mopd_gpas/run_capability_eval.sh"
    result = subprocess.run(
        ["bash", str(script), target],
        check=protocol_version == 6,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"],
            "CAPTURE_COMMAND": str(commands),
            "DRY_RUN": "1",
            "MOPD_CAPABILITY_SUITE": "all",
            "MOPD_GENERATED_DIR": str(generated),
            "MOPD_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "MOPD_DATA_ROOT": str(data),
            "MOPD_HF_CHECKPOINT": str(model),
            "MOPD_TEACHER_HF_ROOT": str(model.parent),
            "CAPABILITY_CUDA_VISIBLE_DEVICES": "0",
        },
    )
    if protocol_version == 5:
        assert result.returncode != 0
        assert "Regenerate the MOPD protocol" in result.stderr
        assert not commands.exists()
        assert not list(generated.glob("capability_*.yaml"))
        return
    command = json.loads(commands.read_text())
    config = yaml.safe_load(Path(command[command.index("--eval-config") + 1]).read_text())
    if target == "initial_student":
        assert command[command.index("--chat-template-suffix-to-remove") + 1] == EMPTY_THINKING_SUFFIX
        assert config["eval"]["defaults"]["chat_template_suffix_to_remove"] == EMPTY_THINKING_SUFFIX
    else:
        assert "--chat-template-suffix-to-remove" not in command
        assert config["eval"]["defaults"]["chat_template_suffix_to_remove"] is None
        assert [dataset["name"] for dataset in config["eval"]["datasets"]] == ["math500_pass1"]


def test_heldout_materialization_uses_student_prompts_before_freezing_them(tmp_path, monkeypatch):
    class Dataset:
        def __init__(self, _path, **kwargs):
            assert kwargs["apply_chat_template"] is True
            assert kwargs["apply_chat_template_kwargs"] == {"enable_thinking": False}
            assert kwargs["chat_template_suffix_to_remove"] == EMPTY_THINKING_SUFFIX
            self.origin_samples = [
                SimpleNamespace(prompt=f"student prompt {index}", label="answer", metadata={}) for index in range(128)
            ]

        def __len__(self):
            return len(self.origin_samples)

    monkeypatch.setattr(prepare_mopd, "Dataset", Dataset)
    monkeypatch.setattr(prepare_mopd, "load_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(prepare_mopd, "load_processor", lambda *_args, **_kwargs: None)
    records = prepare_mopd.materialize_heldout(tmp_path / "data", tmp_path / "generated", tmp_path / "model")
    for task, record in records.items():
        reference = [json.loads(line) for line in Path(record["file"]["path"]).read_text().splitlines()]
        diagnostic = [json.loads(line) for line in Path(record["diagnostic"]["file"]["path"]).read_text().splitlines()]
        assert len(reference) == len(diagnostic) == 64
        assert reference[0]["prompt"] == "student prompt 0"
        assert diagnostic[0]["prompt"] == "student prompt 64"
        assert reference[0]["metadata"]["teacher"] == task


def test_complete_worker_run_can_be_packaged_for_analysis(tmp_path):
    config = "uniform-s1"
    output_root = tmp_path / "outputs"
    run = output_root / config
    capability = run / "capability_eval/response_32000"
    for path in (
        run / "provenance",
        run / "allocation",
        run / "metrics",
        run / "fixed_loss",
        capability,
    ):
        path.mkdir(parents=True, exist_ok=True)
        (path / "artifact.json").write_text("{}\n", encoding="utf-8")
    (run / "run_complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    (run / "allocation/allocation.jsonl").write_text('{"attempted_responses_after":32000}\n', encoding="utf-8")
    (capability / "run_complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")

    (run / "checkpoint_costs.jsonl").write_text("{}\n")

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
    archive = package_dir / f"{config}-analysis.tar.gz"
    with tarfile.open(archive) as stream:
        names = stream.getnames()
    assert f"{config}/allocation/allocation.jsonl" in names
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


@pytest.mark.parametrize("fixed_bank_protocol", [False, True])
@pytest.mark.parametrize("allocation_relative_path", ["allocation/allocation.jsonl", "allocation.jsonl"])
def test_resume_rewinds_append_only_outputs_to_latest_complete_checkpoint(
    tmp_path, fixed_bank_protocol, allocation_relative_path
):
    run = tmp_path / "run"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema_version": 6, "prompt_format": PROMPT_FORMAT}))
    updates = 100 if fixed_bank_protocol else 2
    checkpoint = run / "checkpoints/iter_0000009"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"metadata")
    (checkpoint / "common.pt").write_bytes(b"common")
    sampler = run / "checkpoints/rollout/mopd_dataset_state_dict_9.pt"
    sampler.parent.mkdir()
    torch.save(
        {
            "protocol_sha256": _sha256(protocol),
            "controller": {
                "completed_steps": updates,
                "attempted_responses": 128,
                "pending": None,
            },
        },
        sampler,
    )
    (run / "checkpoints/mopd_checkpoint_index.json").write_text(
        json.dumps(
            [
                {
                    "rollout_id": 9,
                    "operation_index": 1,
                    "attempted_responses": 128,
                    "optimizer_updates": updates,
                }
            ]
        ),
        encoding="utf-8",
    )
    allocation = run / allocation_relative_path
    allocation.parent.mkdir(parents=True, exist_ok=True)
    allocation.write_text(
        "".join(json.dumps({"operation_index": index, "rollout_id": 8 + index}) + "\n" for index in range(3)),
        encoding="utf-8",
    )
    metrics = run / "metrics"
    metrics.mkdir()
    (metrics / "mopd.jsonl").write_text(
        "".join(json.dumps({"metrics": {"mopd/update": index}}) + "\n" for index in range(1, 4)),
        encoding="utf-8",
    )
    (run / "wandb_run_id.txt").write_text("stale_run\n", encoding="utf-8")
    (run / "run_failed.json").write_text('{"status":"failed"}\n', encoding="utf-8")
    if fixed_bank_protocol:
        fixed_loss = run / "fixed_loss"
        fixed_loss.mkdir()
        for step in (0, 100, 200):
            (fixed_loss / f"step_{step:04d}.json").write_text(json.dumps({"step": step}))
        (fixed_loss / "fresh_final.json").write_text(json.dumps({"step": 500}))
        (fixed_loss / "fresh_bank.pt").write_bytes(b"responses from an abandoned trajectory")
        (run / "checkpoint_costs.jsonl").write_text(
            "".join(json.dumps({"step": step, "wall_seconds": 1, "occupied_gpus": 2}) + "\n" for step in (100, 200))
        )

    state = inspect_resume(run, protocol_path=protocol)
    assert state["rollout_id"] == 9
    assert state["rewind_required"]
    assert state["eval_on_start"] == (not fixed_bank_protocol)

    applied = apply_resume(run, protocol_path=protocol)
    assert Path(applied["archive"]).is_file()
    assert not (run / "wandb_run_id.txt").exists()
    assert len(allocation.read_text().splitlines()) == 2
    with tarfile.open(applied["archive"]) as archive:
        assert allocation_relative_path in archive.getnames()
    assert len((metrics / "mopd.jsonl").read_text().splitlines()) == (3 if fixed_bank_protocol else 2)
    if fixed_bank_protocol:
        assert {path.name for path in fixed_loss.iterdir()} == {"step_0000.json", "step_0100.json"}
        costs = [json.loads(row) for row in (run / "checkpoint_costs.jsonl").read_text().splitlines()]
        assert [row["step"] for row in costs] == [100]
        with tarfile.open(applied["archive"]) as archive:
            assert "fixed_loss/fresh_bank.pt" in archive.getnames()
            assert "fixed_loss/step_0200.json" in archive.getnames()


@pytest.mark.parametrize("saved_protocol", [None, "old-protocol-hash"])
def test_resume_rejects_old_prompt_protocol_before_rewinding_outputs(tmp_path, saved_protocol):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/iter_0000009"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"metadata")
    (checkpoint / "common.pt").write_bytes(b"common")
    sampler = run / "checkpoints/rollout/mopd_dataset_state_dict_9.pt"
    sampler.parent.mkdir()
    torch.save(
        {
            "protocol_sha256": saved_protocol,
            "controller": {"completed_steps": 2, "attempted_responses": 128, "pending": None},
        },
        sampler,
    )
    (run / "checkpoints/mopd_checkpoint_index.json").write_text(
        json.dumps([{"rollout_id": 9, "operation_index": 1, "attempted_responses": 128, "optimizer_updates": 2}])
    )
    (run / "wandb_run_id.txt").write_text("keep-this-id\n")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema_version": 6, "prompt_format": PROMPT_FORMAT}))
    original = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}

    for resume in (inspect_resume, apply_resume):
        with pytest.raises(ValueError, match="prompt protocol differs"):
            resume(run, protocol_path=protocol)
        assert {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()} == original


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
