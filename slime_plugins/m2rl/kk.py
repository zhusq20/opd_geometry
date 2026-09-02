"""Strict binary verifier for Knights-and-Knaves assignments.

The MOPD experiments use a binary correctness contract: ``1``
means that every named inhabitant has exactly the expected identity and that
the final answer follows the requested structure; every other model response
receives ``0``.  Dataset/schema errors are raised instead of being converted
to a negative example.
"""

from __future__ import annotations

import json
import re
from typing import Any

_OPEN_ANSWER_TAG = re.compile(r"<answer>", re.IGNORECASE)
_CLOSE_ANSWER_TAG = re.compile(r"</answer>", re.IGNORECASE)
_TERMINAL_SPECIAL_TOKENS = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)+\s*\Z")
_ASSIGNMENT_LINE = re.compile(
    r"^\s*"
    r"(?:(?:\(\d+\)|\d+[.)])\s*|[-*\u2022]\s*)?"
    r"(?P<name>.+?)\s+is\s+(?:a\s+)?(?P<role>knight|knave)"
    r"\s*[.!]?\s*$",
    re.IGNORECASE,
)
_ROLE_CLAUSE = re.compile(r"\bis\s+(?:a\s+)?(?:knight|knave)\b", re.IGNORECASE)
_VALID_ROLES = {"knight", "knave"}


def _normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _parse_label(label: Any) -> tuple[list[str], list[str]]:
    """Return validated names and lower-case roles from a structured label."""

    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError as exc:
            raise ValueError("logic_kk labels must be JSON objects, not free-form answer text.") from exc
    if not isinstance(label, dict):
        raise ValueError("logic_kk labels must be mappings with `names` and `roles` lists.")

    names = label.get("names")
    roles = label.get("roles")
    if not isinstance(names, (list, tuple)) or not isinstance(roles, (list, tuple)):
        raise ValueError("logic_kk labels require list-valued `names` and `roles` fields.")
    if not names or len(names) != len(roles):
        raise ValueError("logic_kk labels require equally sized, non-empty `names` and `roles` lists.")

    clean_names = [str(name).strip() for name in names]
    clean_roles = [str(role).strip().casefold() for role in roles]
    if any(not name for name in clean_names):
        raise ValueError("logic_kk label names must be non-empty strings.")
    normalized_names = [_normalize_name(name) for name in clean_names]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("logic_kk label names must be unique ignoring case and whitespace.")
    invalid_roles = sorted(set(clean_roles) - _VALID_ROLES)
    if invalid_roles:
        raise ValueError(f"logic_kk labels contain invalid roles: {invalid_roles}.")
    return clean_names, clean_roles


def evaluate_kk_response(response: str | None, label: Any) -> dict[str, Any]:
    """Parse one response and return JSON-serializable grading diagnostics."""

    expected_names, expected_roles = _parse_label(label)
    expected_by_key = {
        _normalize_name(name): (name, role) for name, role in zip(expected_names, expected_roles, strict=True)
    }
    text = _TERMINAL_SPECIAL_TOKENS.sub("", response or "")
    opening_tags = list(_OPEN_ANSWER_TAG.finditer(text))
    closing_tags = list(_CLOSE_ANSWER_TAG.finditer(text))
    format_errors: list[str] = []
    malformed_lines: list[str] = []
    extra_names: list[str] = []
    duplicate_names: list[str] = []
    conflicting_names: list[str] = []
    assignments: dict[str, str] = {}

    if len(opening_tags) != 1 or len(closing_tags) != 1:
        format_errors.append("answer_tags_must_appear_exactly_once")
    can_extract = len(opening_tags) == 1 and len(closing_tags) == 1
    if can_extract and closing_tags[0].start() < opening_tags[0].end():
        format_errors.append("closing_answer_tag_precedes_opening_tag")
        can_extract = False

    if can_extract:
        opening = opening_tags[0]
        closing = closing_tags[0]
        if text[closing.end() :].strip():
            format_errors.append("text_after_answer_block")
        answer_text = text[opening.end() : closing.start()]
        answer_lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
        if not answer_lines:
            format_errors.append("empty_answer_block")

        for line in answer_lines:
            if len(_ROLE_CLAUSE.findall(line)) != 1:
                malformed_lines.append(line)
                continue
            match = _ASSIGNMENT_LINE.fullmatch(line)
            if match is None:
                malformed_lines.append(line)
                continue
            supplied_name = re.sub(r"\s+", " ", match.group("name")).strip()
            supplied_key = _normalize_name(supplied_name)
            supplied_role = match.group("role").casefold()
            if supplied_key not in expected_by_key:
                extra_names.append(supplied_name)
                continue
            canonical_name = expected_by_key[supplied_key][0]
            if supplied_key in assignments:
                if assignments[supplied_key] == supplied_role:
                    duplicate_names.append(canonical_name)
                else:
                    conflicting_names.append(canonical_name)
                continue
            assignments[supplied_key] = supplied_role

    if malformed_lines:
        format_errors.append("malformed_assignment_line")
    if extra_names:
        format_errors.append("unexpected_name")
    if duplicate_names:
        format_errors.append("duplicate_name")
    if conflicting_names:
        format_errors.append("conflicting_roles_for_name")

    missing_names = [name for key, (name, _role) in expected_by_key.items() if key not in assignments]
    if missing_names:
        format_errors.append("missing_name")
    incorrect_names = [
        name
        for key, (name, expected_role) in expected_by_key.items()
        if key in assignments and assignments[key] != expected_role
    ]
    format_errors = list(dict.fromkeys(format_errors))
    format_valid = not format_errors
    correct = format_valid and not incorrect_names
    assigned_roles = {name: assignments[key] for key, (name, _role) in expected_by_key.items() if key in assignments}
    return {
        "correct": correct,
        "format_valid": format_valid,
        "format_errors": format_errors,
        "expected_names": expected_names,
        "assigned_roles": assigned_roles,
        "incorrect_names": incorrect_names,
        "missing_names": missing_names,
        "extra_names": extra_names,
        "duplicate_names": duplicate_names,
        "conflicting_names": conflicting_names,
        "malformed_lines": malformed_lines,
    }


def compute_kk_reward(response: str | None, label: Any, metadata: dict[str, Any] | None = None) -> float:
    """Return binary exact-match reward and attach non-reward diagnostics."""

    diagnostics = evaluate_kk_response(response, label)
    if metadata is not None:
        metadata["kk_eval"] = diagnostics
        metadata["format_error"] = not diagnostics["format_valid"]
    return float(diagnostics["correct"])
