#!/usr/bin/env python3
"""Convert the math/IF RL experts and verify the shared pretrained 4B teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

TASKS = ("math", "code", "if", "science")
MODEL_IDENTITY_KEYS = (
    "model_type",
    "architectures",
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "rope_theta",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def load_config(path: Path, root: Path, output_root: Path):
    text = path.read_text(encoding="utf-8")
    replacements = {
        "SLIME_ROOT": str(root),
        "MOPD_HF_CHECKPOINT": os.environ.get("MOPD_HF_CHECKPOINT", "/workspace/dev/checkpoints/Qwen3-1.7B"),
        "MOPD_BASE_MEGATRON": os.environ.get("MOPD_BASE_MEGATRON", "/workspace/dev/checkpoints/Qwen3-1.7B_torch_dist"),
        "MOPD_TEACHER_HF_ROOT": str(output_root),
        "MOPD_QWEN3_4B": os.environ.get("MOPD_QWEN3_4B", str(root / "local/mopd_assets/models/qwen3-4b")),
    }
    for name, value in replacements.items():
        text = text.replace("${" + name + "}", value)
    return yaml.safe_load(os.path.expandvars(text))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    return {
        "path": str(path.relative_to(relative_to) if relative_to is not None else path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_conversion_manifest(
    teacher_hf: Path,
    *,
    task: str,
    step: int,
    checkpoint: Path,
    base_hf: Path,
) -> Path:
    """Pin one freshly converted teacher before it is published for serving."""

    output_files = [
        file_record(path, relative_to=teacher_hf)
        for path in sorted(teacher_hf.rglob("*"))
        if path.is_file() and path.name != "conversion_manifest.json"
    ]
    if not output_files:
        raise ValueError(f"Converted teacher contains no files: {teacher_hf}")
    anchors = {}
    for name in (".metadata", "common.pt"):
        path = checkpoint / name
        anchors[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    checkpoint_files = [
        {"path": str(path.relative_to(checkpoint)), "bytes": path.stat().st_size}
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    ]
    base_assets = [
        file_record(path, relative_to=base_hf)
        for path in sorted(base_hf.rglob("*"))
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json"
    ]
    value = {
        "schema_version": 1,
        "output_files": output_files,
        "source": {
            "task": task,
            "step": int(step),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_anchors": anchors,
            "checkpoint_files": checkpoint_files,
            "base_hf": str(base_hf.resolve()),
            "base_assets": base_assets,
        },
    }
    path = teacher_hf / "conversion_manifest.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def complete_hf(path: Path) -> bool:
    anchors = [path / "config.json", path / "model.safetensors.index.json"]
    anchors.extend(path / name for name in TOKENIZER_FILES)
    if not all(anchor.is_file() and anchor.stat().st_size > 0 for anchor in anchors):
        return False
    try:
        index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards = {path / str(name) for name in weight_map.values()}
    return all(shard.is_file() and shard.stat().st_size > 0 for shard in shards)


def verify_conversion_manifest(teacher_hf: Path) -> None:
    path = teacher_hf / "conversion_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Converted teacher has no conversion manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1 or not manifest.get("output_files"):
        raise ValueError(f"Invalid conversion manifest: {path}")
    for record in manifest["output_files"]:
        output = teacher_hf / str(record["path"])
        if (
            not output.is_file()
            or output.stat().st_size != int(record["bytes"])
            or sha256(output) != str(record["sha256"])
        ):
            raise ValueError(f"Converted teacher file differs from its frozen manifest: {output}")


def verify_source_identity(
    teacher_hf: Path,
    *,
    task: str,
    step: int,
    checkpoint: Path,
    base_hf: Path,
) -> None:
    """Bind a converted teacher to the configured GRPO checkpoint and base assets."""

    manifest = json.loads((teacher_hf / "conversion_manifest.json").read_text(encoding="utf-8"))
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"Converted teacher manifest has no source identity: {teacher_hf}")
    expected = {
        "task": task,
        "step": int(step),
        "checkpoint": str(checkpoint.resolve()),
        "base_hf": str(base_hf.resolve()),
    }
    observed = {key: source.get(key) for key in expected}
    observed["step"] = int(observed["step"]) if observed["step"] is not None else None
    if observed != expected:
        raise ValueError(f"Converted teacher source mismatch: expected {expected}, got {observed}")
    anchors = source.get("checkpoint_anchors")
    if not isinstance(anchors, dict):
        raise ValueError(f"Converted teacher has no checkpoint anchors: {teacher_hf}")
    for name in (".metadata", "common.pt"):
        path = checkpoint / name
        record = anchors.get(name)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256(path) != str(record.get("sha256"))
        ):
            raise ValueError(f"Configured GRPO checkpoint differs from the teacher source anchor: {path}")
    for record in source.get("base_assets") or []:
        path = base_hf / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256(path) != str(record["sha256"]):
            raise ValueError(f"Base HF asset differs from the teacher conversion manifest: {path}")


def verify_compatibility(base_hf: Path, teacher_hf: Path) -> None:
    if not complete_hf(teacher_hf):
        raise FileNotFoundError(f"Incomplete converted teacher: {teacher_hf}")
    base_config = json.loads((base_hf / "config.json").read_text(encoding="utf-8"))
    teacher_config = json.loads((teacher_hf / "config.json").read_text(encoding="utf-8"))
    mismatches = {
        key: (base_config.get(key), teacher_config.get(key))
        for key in MODEL_IDENTITY_KEYS
        if base_config.get(key) != teacher_config.get(key)
    }
    if mismatches:
        raise ValueError(f"Teacher architecture differs from the Qwen3-1.7B student: {mismatches}")
    for name in TOKENIZER_FILES:
        base_path = base_hf / name
        teacher_path = teacher_hf / name
        if not base_path.is_file() or sha256(base_path) != sha256(teacher_path):
            raise ValueError(f"Teacher tokenizer anchor differs from the student: {teacher_path}")
    verify_conversion_manifest(teacher_hf)


def verify_pretrained_teacher(base_hf: Path, teacher_hf: Path) -> None:
    if not complete_hf(teacher_hf):
        raise FileNotFoundError(f"Incomplete pretrained teacher: {teacher_hf}")
    config = json.loads((teacher_hf / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ValueError(f"Expected Qwen3ForCausalLM teacher at {teacher_hf}")
    for name in TOKENIZER_FILES:
        if sha256(base_hf / name) != sha256(teacher_hf / name):
            raise ValueError(f"Pretrained teacher {name} differs from the student tokenizer")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=here.parents[1])
    parser.add_argument("--config", type=Path, default=here / "configs/teacher_checkpoints.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--load-workers", type=int, default=4)
    parser.add_argument("--save-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = args.repo.resolve()
    output_root = (
        args.output_root
        if args.output_root is not None
        else Path(os.environ.get("MOPD_TEACHER_HF_ROOT", root / "local/mopd_assets/models/teachers_hf"))
    ).resolve()
    config = load_config(args.config.resolve(), root, output_root)
    base_hf = Path(config["base_hf"])
    if not (base_hf / "config.json").is_file():
        raise FileNotFoundError(base_hf / "config.json")

    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        item = config["teachers"][task]
        output_dir = Path(item["model_path"])
        if item["kind"] == "pretrained_qwen3_4b":
            verify_pretrained_teacher(base_hf, output_dir)
            print(f"[{task}] verified pretrained Qwen3-4B {output_dir}")
            continue
        step = int(item["step"])
        input_dir = Path(item["checkpoint_root"]) / f"iter_{step:07d}"
        if args.verify_only:
            verify_compatibility(base_hf, output_dir)
            print(f"[{task}] verified {output_dir}")
            continue
        if not (input_dir / ".metadata").is_file() or not (input_dir / "common.pt").is_file():
            raise FileNotFoundError(f"Incomplete {task} checkpoint: {input_dir}")
        if complete_hf(output_dir):
            verify_compatibility(base_hf, output_dir)
            verify_source_identity(
                output_dir,
                task=task,
                step=step,
                checkpoint=input_dir,
                base_hf=base_hf,
            )
            print(f"[{task}] reuse {output_dir}")
            continue
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite incomplete converted teacher {output_dir}; move it aside first"
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = output_dir.with_name(f".{output_dir.name}.convert-{os.getpid()}")
        if staging_dir.exists():
            raise FileExistsError(f"Stale teacher conversion staging directory: {staging_dir}")

        command = [
            sys.executable,
            str(root / "tools/convert_torch_dist_to_hf_parallel.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(staging_dir),
            "--origin-hf-dir",
            str(base_hf),
            "--vocab-size",
            "151936",
            "--load-max-workers",
            str(args.load_workers),
            "--save-max-workers",
            str(args.save_workers),
        ]
        print(f"[{task}] {shlex.join(command)}")
        if not args.dry_run:
            try:
                subprocess.run(command, cwd=root, check=True, env={**os.environ, "PYTHONPATH": str(root)})
                if not complete_hf(staging_dir):
                    raise RuntimeError(f"Conversion did not produce a complete HF teacher: {staging_dir}")
                write_conversion_manifest(
                    staging_dir,
                    task=task,
                    step=step,
                    checkpoint=input_dir,
                    base_hf=base_hf,
                )
                verify_compatibility(base_hf, staging_dir)
                verify_source_identity(
                    staging_dir,
                    task=task,
                    step=step,
                    checkpoint=input_dir,
                    base_hf=base_hf,
                )
                os.replace(staging_dir, output_dir)
            except BaseException:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise
            print(f"[{task}] published {output_dir}")


if __name__ == "__main__":
    main()
