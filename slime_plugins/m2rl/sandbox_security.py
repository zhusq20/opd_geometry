"""Fail-closed validation for external code-execution services."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDITED_BASE_IMAGE = (
    "volcengine/sandbox-fusion@" "sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9"
)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def validate_preflight_marker(config: dict[str, Any], sandbox_url: str) -> dict[str, Any]:
    """Require a recent successful active isolation probe for this exact endpoint."""

    if not bool(config.get("require_preflight", True)):
        raise ValueError("Sandbox preflight cannot be disabled for external code execution.")

    marker_value = config.get("preflight_marker")
    if not marker_value:
        raise ValueError("Code reward requires routes.<type>.preflight_marker (fail-closed sandbox policy).")
    marker_path = Path(os.path.expandvars(os.path.expanduser(str(marker_value)))).resolve()
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Sandbox preflight marker is missing: {marker_path}. Run sandbox_preflight.py against the endpoint."
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != 2:
        raise RuntimeError("Sandbox marker does not use the required cgroup-v2 attestation schema.")
    if marker.get("url") != sandbox_url:
        raise ValueError(
            f"Sandbox marker endpoint {marker.get('url')!r} does not match configured URL {sandbox_url!r}."
        )
    if marker.get("safe") is not True:
        raise RuntimeError(f"Sandbox marker {marker_path} does not attest a safe endpoint.")
    required_checks = {
        "ping",
        "execution",
        "livecodebench_submit",
        "submit_policy_enforced",
        "control_plane_network_denied",
        "filesystem_separation",
        "service_filesystem_separation",
        "network_denied",
        "cgroup_v2_enforced",
        "memory_limit_enforced",
        "least_privilege",
        "dangerous_syscalls_denied",
        "namespace_isolation",
        "timeout_enforced",
        "deployment_attested",
    }
    checks = marker.get("checks") or {}
    failed = sorted(name for name in required_checks if checks.get(name) is not True)
    if failed:
        raise RuntimeError(f"Sandbox marker is missing successful checks: {', '.join(failed)}")
    deployment = marker.get("deployment") or {}
    if deployment.get("sandbox_config") != "ci":
        raise RuntimeError("Sandbox marker does not attest SANDBOX_CONFIG=ci (lite isolation).")
    image_reference = str(deployment.get("image_reference") or "")
    digest = image_reference.rsplit("@", 1)[-1] if "@" in image_reference else image_reference
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise RuntimeError("Sandbox marker does not attest a digest-pinned container image.")
    if re.fullmatch(r"[0-9a-f]{64}", str(deployment.get("container_id") or "")) is None:
        raise RuntimeError("Sandbox marker does not contain a full container ID.")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", str(deployment.get("image_id") or "")) is None:
        raise RuntimeError("Sandbox marker does not contain a content-addressed image ID.")
    if re.fullmatch(r"[0-9a-f]{64}", str(deployment.get("compose_sha256") or "")) is None:
        raise RuntimeError("Sandbox marker does not contain a valid Compose hash.")
    if deployment.get("cgroup_version") != "2":
        raise RuntimeError("Sandbox marker does not attest host cgroup v2.")
    if deployment.get("control_plane_network_internal") != "true":
        raise RuntimeError("Sandbox marker does not attest an internal control-plane network.")
    if deployment.get("patch_id") != "cgroup2-v1":
        raise RuntimeError("Sandbox marker does not attest the audited cgroup2-v1 patch.")
    if deployment.get("base_image") != AUDITED_BASE_IMAGE:
        raise RuntimeError("Sandbox marker does not attest the audited upstream base digest.")
    if re.fullmatch(r"[0-9a-f]{64}", str(deployment.get("patch_sha256") or "")) is None:
        raise RuntimeError("Sandbox marker does not contain a valid patch source hash.")
    for setting in ("aggregate_memory_max", "aggregate_pids_max"):
        value = str(deployment.get(setting) or "")
        if re.fullmatch(r"[1-9][0-9]*", value) is None:
            raise RuntimeError(f"Sandbox marker does not attest a positive {setting}.")
    tested_at = _parse_time(str(marker["tested_at_utc"]))
    age_seconds = (datetime.now(timezone.utc) - tested_at).total_seconds()
    max_age = float(config.get("preflight_max_age_seconds", 86400))
    if age_seconds < -300 or age_seconds > max_age:
        raise RuntimeError(
            f"Sandbox marker age {age_seconds:.0f}s is outside the allowed window (max {max_age:.0f}s)."
        )
    return marker
