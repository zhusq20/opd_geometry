"""Identity and runtime checks for asymmetric Base-student OPD prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EMPTY_THINKING_SUFFIX = "<think>\n\n</think>\n\n"
PROMPT_FORMAT = {
    "student_chat_template_suffix_to_remove": EMPTY_THINKING_SUFFIX,
    "teacher_prompt_suffix": EMPTY_THINKING_SUFFIX,
}


def protocol_identity(protocol_path):
    content = Path(protocol_path).read_bytes()
    protocol = json.loads(content)
    modern = protocol.get("schema_version") == 7 and protocol.get("protocol") == "mopd_paper"
    if not modern and (protocol.get("schema_version") != 6 or protocol.get("prompt_format") != PROMPT_FORMAT):
        raise ValueError("Regenerate the MOPD protocol and rendered prompts for asymmetric student/teacher prefixes")
    return hashlib.sha256(content).hexdigest()


def validate_prompt_runtime(args, manifest=None):
    """Bind active student rendering and teacher routes to the frozen protocol."""
    path = getattr(args, "experiment_data_index", None)
    protocol = json.loads(Path(path).read_bytes()) if path else {}
    enabled = (
        getattr(args, "mopd_loss", None) in {"student_topk", "topk_intersection", "teacher_topk"}
        or bool(getattr(args, "chat_template_suffix_to_remove", None))
        or "prompt_format" in protocol
        or (manifest is not None and "prompt_format" in manifest)
    )
    if not enabled:
        return None
    if not path:
        raise ValueError("Asymmetric MOPD prompts require --experiment-data-index with the regenerated protocol")
    identity = protocol_identity(path)
    if protocol.get("schema_version") == 7:
        expected = protocol["prompt_format"]
        if manifest is not None and manifest.get("prompt_format") != expected:
            raise ValueError("Manifest and protocol prompt serialization differ")
        if getattr(args, "chat_template_suffix_to_remove", None) != expected["student_chat_template_suffix_to_remove"]:
            raise ValueError("Student prompt suffix differs from the selected model profile")
        if dict(getattr(args, "apply_chat_template_kwargs", None) or {}) != expected["chat_template_kwargs"]:
            raise ValueError("Student chat-template kwargs differ from the selected model profile")
        from slime_plugins.m2rl.opd import load_teacher_router
        from .sampler import active_tasks

        teachers = load_teacher_router(args.opd_teacher_router_config)["teachers"]
        for task in active_tasks(args):
            if teachers[task].get("prompt_suffix", "") != expected["teacher_prompt_suffix"]:
                raise ValueError(f"Teacher {task} prompt suffix differs from the selected profile")
        return identity
    if getattr(args, "chat_template_suffix_to_remove", None) != EMPTY_THINKING_SUFFIX:
        raise ValueError("Student chat-template suffix removal differs from the frozen MOPD prompt protocol")
    if manifest is not None:
        if manifest.get("prompt_format") != PROMPT_FORMAT:
            raise ValueError("Regenerate the MOPD data manifest and rendered prompts for asymmetric prefixes")
        for source in manifest.get("sources", []):
            applies_template = source.get("apply_chat_template", getattr(args, "apply_chat_template", False))
            removal = source.get("chat_template_suffix_to_remove", args.chat_template_suffix_to_remove)
            if applies_template and removal != EMPTY_THINKING_SUFFIX:
                raise ValueError(f"Student prompt suffix for source {source.get('name')} differs from the protocol")
            if not applies_template and removal:
                raise ValueError(f"Pre-rendered source {source.get('name')} must clear chat_template_suffix_to_remove")
    router_path = getattr(args, "opd_teacher_router_config", None)
    if not router_path:
        raise ValueError("Asymmetric MOPD prompts require a teacher router with the frozen prompt_suffix")
    from slime_plugins.m2rl.opd import load_teacher_router

    from .sampler import TASKS

    teachers = load_teacher_router(router_path)["teachers"]
    for task in TASKS:
        route = teachers.get(task)
        if not isinstance(route, dict) or route.get("prompt_suffix") != EMPTY_THINKING_SUFFIX:
            raise ValueError(f"Teacher {task} prompt_suffix differs from the frozen MOPD prompt protocol")
    return identity
