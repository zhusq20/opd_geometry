"""Fail-closed policy tests for generated-code execution."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from examples.optimizer_geometry.validate_experiment import validate_eval_config

from slime_plugins.m2rl.sandbox_security import validate_preflight_marker

NUM_GPUS = 0
URL = "http://127.0.0.1:8080/run_code"


def _marker(tmp_path, *, safe=True, age_seconds=0, image_pinned=True):
    path = tmp_path / "preflight.json"
    checks = {
        "ping": True,
        "execution": True,
        "livecodebench_submit": True,
        "submit_policy_enforced": True,
        "control_plane_network_denied": True,
        "filesystem_separation": True,
        "service_filesystem_separation": True,
        "network_denied": True,
        "cgroup_v2_enforced": True,
        "memory_limit_enforced": True,
        "least_privilege": True,
        "dangerous_syscalls_denied": True,
        "namespace_isolation": True,
        "timeout_enforced": True,
        "deployment_attested": True,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "tested_at_utc": (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat(),
                "url": URL,
                "safe": safe,
                "checks": checks,
                "deployment": {
                    "container_id": "c" * 64,
                    "image_id": "sha256:" + "d" * 64,
                    "compose_sha256": "e" * 64,
                    "sandbox_config": "ci",
                    "image_reference": (
                        "volcengine/sandbox-fusion@sha256:" + "a" * 64
                        if image_pinned
                        else "volcengine/sandbox-fusion:latest"
                    ),
                    "cgroup_version": "2",
                    "control_plane_network_internal": "true",
                    "patch_id": "cgroup2-v1",
                    "patch_sha256": "b" * 64,
                    "aggregate_memory_max": str(32 * 1024**3),
                    "aggregate_pids_max": "4096",
                    "base_image": (
                        "volcengine/sandbox-fusion@sha256:"
                        "dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9"
                    ),
                },
            }
        )
        + "\n"
    )
    return path


@pytest.mark.unit
def test_secure_recent_marker_is_accepted(tmp_path):
    marker = _marker(tmp_path)

    result = validate_preflight_marker({"preflight_marker": str(marker)}, URL)

    assert result["safe"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"safe": False}, "does not attest"),
        ({"url": "http://elsewhere/run_code"}, "does not match"),
    ],
)
def test_marker_rejects_unsafe_or_wrong_endpoint(tmp_path, mutation, error):
    marker = _marker(tmp_path)
    payload = json.loads(marker.read_text())
    payload.update(mutation)
    marker.write_text(json.dumps(payload))

    with pytest.raises((RuntimeError, ValueError), match=error):
        validate_preflight_marker({"preflight_marker": str(marker)}, URL)


@pytest.mark.unit
def test_marker_rejects_stale_probe(tmp_path):
    marker = _marker(tmp_path, age_seconds=120)

    with pytest.raises(RuntimeError, match="outside the allowed window"):
        validate_preflight_marker({"preflight_marker": str(marker), "preflight_max_age_seconds": 60}, URL)


@pytest.mark.unit
def test_marker_rejects_unpinned_image(tmp_path):
    marker = _marker(tmp_path, image_pinned=False)

    with pytest.raises(RuntimeError, match="digest-pinned"):
        validate_preflight_marker({"preflight_marker": str(marker)}, URL)


@pytest.mark.unit
def test_marker_rejects_missing_cgroup_v2_check(tmp_path):
    marker = _marker(tmp_path)
    payload = json.loads(marker.read_text())
    del payload["checks"]["cgroup_v2_enforced"]
    marker.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="cgroup_v2_enforced"):
        validate_preflight_marker({"preflight_marker": str(marker)}, URL)


@pytest.mark.unit
def test_marker_rejects_wrong_upstream_base_digest(tmp_path):
    marker = _marker(tmp_path)
    payload = json.loads(marker.read_text())
    payload["deployment"]["base_image"] = "volcengine/sandbox-fusion@sha256:" + "c" * 64
    marker.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="upstream base digest"):
        validate_preflight_marker({"preflight_marker": str(marker)}, URL)


@pytest.mark.unit
@pytest.mark.parametrize("unsafe_override", [False, True])
def test_disabling_preflight_is_never_allowed(unsafe_override):
    with pytest.raises(ValueError, match="cannot be disabled"):
        validate_preflight_marker(
            {
                "require_preflight": False,
                "allow_unsafe_without_preflight": unsafe_override,
            },
            URL,
        )


@pytest.mark.unit
def test_livecodebench_eval_config_requires_matching_sandbox_attestation(tmp_path):
    marker = _marker(tmp_path)
    data = tmp_path / "livecodebench.parquet"
    data.write_bytes(b"placeholder")
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(
        "eval:\n"
        "  defaults:\n"
        "    n_samples_per_eval_prompt: 10\n"
        "    max_response_len: 8192\n"
        "  datasets:\n"
        "    - name: livecodebench_final\n"
        f"      path: {data}\n"
        "      rm_type: livecodebench\n"
    )
    reward_config = {
        "routes": {
            "livecodebench": {
                "url": "http://127.0.0.1:8080/submit",
                "preflight_url": URL,
                "preflight_marker": str(marker),
            }
        }
    }

    assert validate_eval_config(eval_config, reward_config, check_runtime_deps=True) == ["livecodebench_final"]
