#!/usr/bin/env python3
"""Legacy same-prefix initial KL measurement for the 128-prompt protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

TASKS = ("math", "code", "if", "science")


def read_prompts(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 128:
        raise ValueError(f"{path} must contain exactly 128 held-out prompts")
    return [str(row["prompt"]) for row in rows]


@torch.no_grad()
def response_logprobs(model, prompt_ids: list[int], response_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor([prompt_ids + response_ids], device=model.device)
    logits = model(ids, use_cache=False).logits[0]
    start = len(prompt_ids) - 1
    selected = logits[start : start + len(response_ids)]
    targets = torch.tensor(response_ids, device=selected.device)
    chunks = []
    for offset in range(0, len(response_ids), 256):
        stop = min(offset + 256, len(response_ids))
        chunks.append(F.log_softmax(selected[offset:stop].float(), dim=-1).gather(1, targets[offset:stop, None]))
    return torch.cat(chunks).squeeze(1).cpu()


def load_model(path: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student", default=os.environ.get("MOPD_HF_CHECKPOINT", "/workspace/dev/checkpoints/Qwen3-1.7B-Base")
    )
    parser.add_argument(
        "--math-teacher",
        default=os.environ.get("MOPD_TEACHER_HF_ROOT", str(root / "local/mopd_assets/models/teachers_hf")) + "/math",
    )
    parser.add_argument(
        "--if-teacher",
        default=os.environ.get("MOPD_TEACHER_HF_ROOT", str(root / "local/mopd_assets/models/teachers_hf")) + "/if",
    )
    parser.add_argument(
        "--code-teacher",
        default=os.environ.get("MOPD_TEACHER_HF_ROOT", str(root / "local/mopd_assets/models/teachers_hf")) + "/code",
    )
    parser.add_argument(
        "--science-teacher",
        default=os.environ.get("MOPD_TEACHER_HF_ROOT", str(root / "local/mopd_assets/models/teachers_hf"))
        + "/science",
    )
    parser.add_argument(
        "--heldout-dir",
        type=Path,
        default=Path(os.environ.get("MOPD_GENERATED_DIR", root / "local/mopd_generated")) / "heldout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "MOPD_INITIAL_KL",
                Path(os.environ.get("MOPD_GENERATED_DIR", root / "local/mopd_generated")) / "initial_kl.json",
            )
        ),
    )
    parser.add_argument("--max-response-len", type=int, default=4_096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()

    protocol_path = args.heldout_dir.parent / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("schema_version", 0) >= 6 or "prompt_format" in protocol:
            raise ValueError(
                "measure-initial only supports the legacy same-prefix protocol; "
                "asymmetric OPD uses equal task weights and builds its reference bank during training"
            )
    if args.seed != 42 or args.max_response_len != 4_096 or args.temperature != 1.0:
        raise ValueError("the initial-KL protocol freezes seed=42, temperature=1, and max_response_len=4096")
    teacher_paths = {
        "math": str(Path(args.math_teacher).resolve()),
        "code": str(Path(args.code_teacher).resolve()),
        "if": str(Path(args.if_teacher).resolve()),
        "science": str(Path(args.science_teacher).resolve()),
    }
    prompts = {task: read_prompts(args.heldout_dir / f"{task}.jsonl") for task in TASKS}

    from sglang import Engine
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    prompt_ids = {
        task: [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts[task]] for task in TASKS
    }
    generator = Engine(
        model_path=args.student,
        dtype="bfloat16",
        mem_fraction_static=args.gpu_memory_utilization,
        context_length=2_048 + args.max_response_len,
        random_seed=args.seed,
        enable_deterministic_inference=True,
        skip_tokenizer_init=True,
    )
    sampling = {
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": args.max_response_len,
        "sampling_seed": args.seed,
    }
    generated: dict[str, list[dict]] = {}
    try:
        for task in TASKS:
            started = time.time()
            outputs = generator.generate(input_ids=prompt_ids[task], sampling_params=sampling)
            generated[task] = [
                {
                    "prompt_ids": prompt,
                    "response_ids": list(output["output_ids"]),
                    "truncated": output["meta_info"]["finish_reason"]["type"] == "length",
                }
                for prompt, output in zip(prompt_ids[task], outputs, strict=True)
            ]
            if any(not row["response_ids"] for row in generated[task]):
                raise RuntimeError(f"initial student produced an empty {task} response")
            print(f"[generate] {task}: 128 responses in {time.time() - started:.1f}s")
    finally:
        generator.shutdown()
    torch.cuda.empty_cache()

    student = load_model(args.student)
    for task in TASKS:
        for row in generated[task]:
            row["student_logprobs"] = response_logprobs(student, row["prompt_ids"], row["response_ids"])
    del student
    torch.cuda.empty_cache()

    losses: dict[str, float] = {}
    tasks_by_teacher: dict[str, list[str]] = defaultdict(list)
    for task, path in teacher_paths.items():
        tasks_by_teacher[path].append(task)
    for teacher_path, teacher_tasks in tasks_by_teacher.items():
        teacher = load_model(teacher_path)
        for task in teacher_tasks:
            response_losses = []
            for row in generated[task]:
                teacher_logprobs = response_logprobs(teacher, row["prompt_ids"], row["response_ids"])
                response_losses.append(float((row["student_logprobs"] - teacher_logprobs).mean()))
            losses[task] = sum(response_losses) / len(response_losses)
            print(f"[score] {task}: ell0={losses[task]:.8f}")
        del teacher
        torch.cuda.empty_cache()

    if any(value <= 0 or not math.isfinite(value) for value in losses.values()):
        raise ValueError(f"initial teacher losses must be finite and positive: {losses}")
    ratio = max(losses.values()) / min(losses.values())
    inverse = {task: 1.0 / losses[task] for task in TASKS}
    inverse_total = sum(inverse.values())
    inverse_weights = {task: inverse[task] / inverse_total for task in TASKS}
    if ratio > 10:
        weight_rule = "equal_due_to_max_min_ratio_gt_10"
        target_weights = {task: 0.25 for task in TASKS}
    else:
        weight_rule = "inverse_initial_teacher_loss"
        target_weights = inverse_weights

    result = {
        "schema_version": 2,
        "task_order": list(TASKS),
        "student": str(Path(args.student).resolve()),
        "teachers": teacher_paths,
        "heldout_dir": str(args.heldout_dir.resolve()),
        "seed": args.seed,
        "temperature": args.temperature,
        "max_response_len": args.max_response_len,
        "tasks": {
            task: {
                "ell0": losses[task],
                "inverse_loss_weight": inverse_weights[task],
                "responses": len(generated[task]),
                "mean_response_length": sum(len(row["response_ids"]) for row in generated[task]) / 128,
                "truncation_rate": sum(row["truncated"] for row in generated[task]) / 128,
            }
            for task in TASKS
        },
        "max_min_ratio": ratio,
        "weight_rule": weight_rule,
        "target_weights": target_weights,
    }
    atomic_json(args.output, result)
    print(f"weight rule: {weight_rule}; max/min={ratio:.4f}; wrote {args.output}")


if __name__ == "__main__":
    main()
