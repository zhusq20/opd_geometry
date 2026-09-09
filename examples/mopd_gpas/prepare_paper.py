#!/usr/bin/env python3
"""Prepare the two model families for normalization, sparsity, and density runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import yaml

if __package__:
    from .profiles import PAPER_RUNS, PAPER_TOPK, profile
else:
    from profiles import PAPER_RUNS, PAPER_TOPK, profile


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with path.open("wb") as stream:
        for row in rows:
            data = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            stream.write(data)
            digest.update(data)
            size += len(data)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def as_mapping(value):
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def normalize_open_row(row, *, evaluation=False):
    """Preserve the released prompts and verifier inputs, including callable tests."""
    raw_domain = row.get("domain")
    if raw_domain is None:
        dataset = str(row.get("dataset", ""))
        raw_domain = "code" if "livecodebench" in dataset else ("if" if dataset.startswith("if") else "math")
    domain = {"instruction": "if", "instruction_following": "if"}.get(raw_domain, raw_domain)
    metadata = as_mapping(row.get("metadata"))
    extra = {key: value for key, value in dict(row.get("extra_info") or {}).items() if value is not None}
    metadata.update(extra)
    metadata.update({"task_name": domain, "teacher": domain, "source_dataset": row.get("data_source", row.get("dataset")),
                     "source_index": row.get("sample_id", extra.get("index"))})
    label = dict(row.get("reward_model") or {}).get("ground_truth", row.get("answer"))
    if domain == "code":
        if not evaluation:
            metadata["unit_tests"] = as_mapping(label)
        rm_type = "unit_test" if not evaluation else "open_mopd_lcb"
    elif domain == "if":
        if "instruction_kwargs_json" in metadata:
            metadata["kwargs"] = [json.loads(value) or {} for value in metadata["instruction_kwargs_json"]]
        label = {key: metadata.get(key) for key in ("instruction_id_list", "kwargs", "prompt")}
        rm_type = "open_mopd_if"
    else:
        rm_type = "deepscaler"
    metadata["rm_type"] = rm_type
    if evaluation:
        metadata["open_mopd_evaluation"] = True
        metadata["metadata"] = as_mapping(row.get("metadata"))
        metadata["evaluator"] = row.get("evaluator")
        metadata["dataset"] = row.get("dataset")
        metadata["eval_prompt_for_scorer"] = row.get("original_prompt")
    return {"prompt": row["prompt"], "label": label, "metadata": metadata}


def data_rows(p):
    rows = {task: [] for task in p["tasks"]}
    if p["name"] == "qwen3":
        paths = [p["data_root"] / "train" / f"{task}.jsonl" for task in p["tasks"]]
        for task, path in zip(p["tasks"], paths, strict=True):
            with path.open() as stream:
                rows[task] = [json.loads(line) for line in stream if line.strip()]
    else:
        import pyarrow.parquet as pq
        paths = [p["data_root"] / "rl_prompt_mix/train.parquet"]
        for batch in pq.ParquetFile(paths[0]).iter_batches(batch_size=512):
            for original in batch.to_pylist():
                row = normalize_open_row(original)
                rows[row["metadata"]["task_name"]].append(row)
    return rows, [record(path) for path in paths]


def prepare(p, *, seed=42, heldout_count=64, diagnostic_count=64):
    from transformers import AutoTokenizer

    output = p["generated"]
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(p["student_hf"])
    rows, data_identity = data_rows(p)
    prompt_format = {"student_chat_template_suffix_to_remove": p["student_suffix_to_remove"],
                     "teacher_prompt_suffix": p["teacher_suffix"], "chat_template_kwargs": p["chat_kwargs"]}
    sources, teachers, split_records = [], {}, {}
    for index, task in enumerate(p["tasks"]):
        # Select disjoint held-out/diagnostic prompts with the same prompt cap as training.
        ordered = list(rows[task])
        random.Random(seed + index * 100003).shuffle(ordered)
        reserved, training = [], []
        for row in ordered:
            metadata = dict(row.get("metadata") or {})
            metadata.update({"task_name": task, "teacher": task})
            row = {**row, "metadata": metadata}
            if len(reserved) < heldout_count + diagnostic_count:
                text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True,
                                                     **p["chat_kwargs"])
                suffix = p["student_suffix_to_remove"]
                if suffix and text.endswith(suffix):
                    text = text[:-len(suffix)]
                if len(tokenizer(text, add_special_tokens=False).input_ids) <= 2048:
                    reserved.append({**row, "prompt": text})
                    continue
            training.append(row)
        if len(reserved) < heldout_count + diagnostic_count:
            raise ValueError(f"Insufficient usable {task} prompts for requested heldout/diagnostic splits")
        split_records[task] = {}
        for split, values in (("train", training), ("heldout", reserved[:heldout_count]),
                              ("diagnostic", reserved[heldout_count:])):
            for row in values:
                row["metadata"]["protocol_split"] = split
            path = output / split / f"{task}.jsonl"
            split_records[task][split] = {**write_jsonl(path, values), "records": len(values)}
        sources.append({"name": task, "path": str((output / "train" / f"{task}.jsonl").resolve()),
                        "input_key": "prompt", "label_key": "label", "metadata_key": "metadata",
                        "apply_chat_template": True, "apply_chat_template_kwargs": p["chat_kwargs"],
                        "chat_template_suffix_to_remove": p["student_suffix_to_remove"],
                        "target_weight": 1 / len(p["tasks"]), "teacher": task,
                        "shuffle_seed": seed + index * 100003})
        model = p["teacher_root"] / task
        # Matching token IDs are essential for shared student-prefix teacher scoring.
        teacher_tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError(f"Student and {task} teacher vocabulary IDs differ")
        teachers[task] = {"kind": "rl", "model_path": str(model.resolve()), "config": record(model / "config.json")}
        conversion = model / "conversion_manifest.json"
        if conversion.is_file():
            teachers[task]["conversion_manifest"] = record(conversion)
    manifest = {"version": 7, "protocol": "mopd_paper", "prompt_format": prompt_format,
                "sampling": {"seed": seed, "repeat": True}, "sources": sources}
    (output / "train.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    diagnostic = {**manifest, "sources": [{**source, "path": str(output / "diagnostic" / f"{source['name']}.jsonl"),
                                          "apply_chat_template": False, "chat_template_suffix_to_remove": None}
                                         for source in sources]}
    (output / "diagnostic.yaml").write_text(yaml.safe_dump(diagnostic, sort_keys=False))
    heldout_manifest = {**diagnostic, "sources": [{**source, "path": str(output / "heldout" / f"{source['name']}.jsonl")}
                                                 for source in diagnostic["sources"]]}
    (output / "heldout.yaml").write_text(yaml.safe_dump(heldout_manifest, sort_keys=False))
    teacher_gpus = {task: os.environ.get('MOPD_TEACHER_' + task.upper() + '_GPU', os.environ.get('MOPD_INFERENCE_GPU', '1'))
                    for task in p["tasks"]}
    router = {"schema_version": 4, "concurrency": 8, "request_timeout": 900,
              "resident_pool": {"gpu_count": len(set(teacher_gpus.values())), "physical_gpus": teacher_gpus},
              "teachers": {task: {"url": f"http://127.0.0.1:{os.environ.get('MOPD_TEACHER_' + task.upper() + '_PORT', 31001 + i)}/generate",
                                   "model_path": teachers[task]["model_path"], "prompt_suffix": p["teacher_suffix"]}
                           for i, task in enumerate(p["tasks"])}}
    (output / "teacher_router.yaml").write_text(yaml.safe_dump(router, sort_keys=False))
    eval_defaults = {"apply_chat_template": False, "chat_template_suffix_to_remove": None,
                     "custom_rm_path": "slime_plugins.mopd.eval.generation_only_reward", "n_samples_per_eval_prompt": 1,
                     "temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_response_len": 32768}
    teacher_eval = {"eval": {"defaults": eval_defaults, "datasets": [
        {"name": task, "path": str(output / "heldout" / f"{task}.jsonl"),
         "input_key": "prompt", "label_key": "label", "metadata_key": "metadata"} for task in p["tasks"]]}}
    (output / "teacher_loss_eval.yaml").write_text(yaml.safe_dump(teacher_eval, sort_keys=False))
    assets_path = p["asset_root"] / "assets.json"
    assets = json.loads(assets_path.read_text()) if assets_path.exists() else None
    student_repository = ((assets or {}).get("student") or {}).get("repository", os.environ.get("MOPD_STUDENT_REPO", p["student_repo"]))
    protocol = {"schema_version": 7, "protocol": "mopd_paper", "profile": p["name"], "seed": seed,
                "task_order": list(p["tasks"]), "prompt_format": prompt_format, "teachers": teachers,
                "student": {"model": student_repository, "hf_config": record(p["student_hf"] / "config.json"),
                            "tokenizer": record(p["student_hf"] / "tokenizer.json"),
                            "tokenizer_config": record(p["student_hf"] / "tokenizer_config.json")},
                "assets": assets,
                "datasets": data_identity, "splits": split_records,
                "training": {"configs": PAPER_RUNS, "steps": 500, "response_cap_tokens": 4096,
                             "topk": {"loss": "student_topk", "k": 16, "selection": "rollout_student",
                                      "supported_k": [16, 64], "k_by_config": PAPER_TOPK,
                                      "advantage": "stopgrad(softmax(selected_student_logp) * (teacher_logp - student_logp))",
                                      "logprob_normalization": "full_vocabulary", "ppo_clipping": False,
                                      "additional_domain_weighting": False},
                             "topk_intersection": {
                                 "k": 64, "selection": "rollout_student_top64_intersect_teacher_top64",
                                 "weights": "softmax over original student Top64, masked without renormalization",
                                 "empty_intersection": "zero loss and zero gradient",
                             },
                             "sampling": "equal prompt quotas per active domain; shuffled repeat after exhaustion"},
                "evaluation": {"response_cap_tokens": 32768}}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    official = Path(os.environ.get("OPEN_MOPD_ROOT", Path(__file__).resolve().parents[3] / "Open-MOPD"))
    sandbox = os.environ.get("SANDBOXFUSION_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
    rewards = {"routes": {"deepscaler": {}, "gpqa": {}, "ifevalg": {}, "ifbench": {},
                          "unit_test": {"url": sandbox + "/run_code", "run_timeout": 10,
                                        "request_timeout": 180, "concurrency": 8},
                          "livecodebench": {"url": sandbox + "/submit", "run_timeout": 6,
                                            "request_timeout": 180, "concurrency": 8},
                          "open_mopd_lcb": {"url": sandbox + "/submit", "run_timeout": 6,
                                             "request_timeout": 180, "concurrency": 8},
                          "open_mopd_if": {"scorer_path": str(official / "training/verl/verl/utils/reward_score/instruction_following.py")}}}
    for name in ("unit_test", "livecodebench", "open_mopd_lcb"):
        rewards["routes"][name].update(
            preflight_marker=os.environ.get("M2RL_SANDBOX_PREFLIGHT_MARKER", "/workspace/sandboxfusion-state/sandboxfusion_preflight.json"),
            require_preflight=True, preflight_max_age_seconds=604800, preflight_url=sandbox + "/run_code")
    (output / "rewards.yaml").write_text(yaml.safe_dump(rewards, sort_keys=False))
    prepare_capability(p)
    print(f"Prepared {p['name']} paper protocol: {output}")


def prepare_capability(p):
    output = p["generated"]
    defaults = {"apply_chat_template": True, "apply_chat_template_kwargs": p["chat_kwargs"],
                "chat_template_suffix_to_remove": p["student_suffix_to_remove"],
                "custom_rm_path": "slime_plugins.m2rl.rewards.reward", "top_k": -1,
                "input_key": "prompt", "label_key": "label", "metadata_key": "metadata", "max_response_len": 32768}
    if p["name"] == "qwen3":
        config = yaml.safe_load((Path(__file__).parent / "configs/capability_eval.yaml").read_text())
        datasets = config["eval"]["datasets"]
        for entry in datasets:
            entry["path"] = entry["path"].replace("${oc.env:MOPD_DATA_ROOT}", str(p["data_root"]))
            entry["max_response_len"] = 32768
    else:
        import pyarrow.parquet as pq
        datasets = []
        for domain, name, count, temperature in (("math", "aime24", 64, 0.6), ("math", "aime25", 64, 0.6),
                                                ("code", "livecodebench_v5", 10, 1.0), ("code", "livecodebench_v6", 10, 1.0),
                                                ("if", "ifeval_aligned", 1, 1.0), ("if", "ifbench_test_aligned", 1, 1.0)):
            path = p["data_root"] / "eval" / domain / f"{name}.parquet"
            rows = [normalize_open_row(row, evaluation=True) for row in pq.read_table(path).to_pylist()]
            target = output / "capability" / f"{name}.jsonl"
            write_jsonl(target, rows)
            datasets.append({"name": name, "path": str(target), "n_samples_per_eval_prompt": count,
                             "temperature": temperature, "top_p": 0.95, "max_response_len": 32768})
    (output / "capability_eval.yaml").write_text(yaml.safe_dump({"eval": {"defaults": defaults, "datasets": datasets}}, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("qwen3", "smollm3"), default=os.environ.get("MOPD_PROFILE", "qwen3"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(profile(args.profile), seed=args.seed)
