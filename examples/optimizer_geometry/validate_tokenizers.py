#!/usr/bin/env python3
"""Verify token-ID compatibility required by external-teacher OPD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def tokenizer_signature(tokenizer: Any) -> dict[str, Any]:
    return {
        "vocab": tokenizer.get_vocab(),
        "added_vocab": tokenizer.get_added_vocab(),
        "special_tokens_map": tokenizer.special_tokens_map,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def compare_tokenizers(student: Any, teacher: Any) -> dict[str, Any]:
    student_signature = tokenizer_signature(student)
    teacher_signature = tokenizer_signature(teacher)
    student_vocab = student_signature.pop("vocab")
    teacher_vocab = teacher_signature.pop("vocab")

    missing = sorted(set(student_vocab) - set(teacher_vocab))
    extra = sorted(set(teacher_vocab) - set(student_vocab))
    mismatched = sorted(
        token
        for token in set(student_vocab).intersection(teacher_vocab)
        if student_vocab[token] != teacher_vocab[token]
    )
    compatible = not missing and not extra and not mismatched and student_signature == teacher_signature
    result = {
        "compatible": compatible,
        "student_vocab_size": len(student_vocab),
        "teacher_vocab_size": len(teacher_vocab),
        "missing_token_count": len(missing),
        "extra_token_count": len(extra),
        "mismatched_id_count": len(mismatched),
        "missing_examples": missing[:10],
        "extra_examples": extra[:10],
        "mismatched_examples": [
            {"token": token, "student_id": student_vocab[token], "teacher_id": teacher_vocab[token]}
            for token in mismatched[:10]
        ],
        "student_metadata": student_signature,
        "teacher_metadata": teacher_signature,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True, local_files_only=True)
    teacher = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True, local_files_only=True)
    result = compare_tokenizers(student, teacher)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if not result["compatible"]:
        raise SystemExit(
            "Student and teacher tokenizers are not token-ID compatible; external-teacher OPD would be invalid."
        )


if __name__ == "__main__":
    main()
