#!/usr/bin/env python3
"""Actively probe a SandboxFusion /run_code endpoint and write a safety marker."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDITED_BASE_IMAGE = (
    "volcengine/sandbox-fusion@" "sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9"
)
MIN_LIVECODEBENCH_UPLOAD_BYTES = 137 * 1024 * 1024


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_json_status(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any]:
    """Return an API response status without making expected rejections fatal."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            decoded: Any = json.loads(body)
        except json.JSONDecodeError:
            decoded = body
        return exc.code, decoded


def stdout(payload: dict[str, Any]) -> str:
    result = payload.get("run_result") or payload.get("result") or {}
    return str(result.get("stdout") or "") if isinstance(result, dict) else ""


def run_code(
    url: str,
    code: str,
    *,
    run_timeout: float = 2,
    request_timeout: float = 10,
    memory_limit_mb: int = 256,
) -> dict[str, Any]:
    return request_json(
        url,
        {
            "language": "python",
            "code": code,
            "stdin": "",
            "compile_timeout": 2,
            "run_timeout": run_timeout,
            "memory_limit_MB": memory_limit_mb,
        },
        request_timeout,
    )


def livecodebench_payload(token: str) -> dict[str, Any]:
    problem_id = f"sandboxfusion-preflight-{token}"
    tests = {
        "input_output": json.dumps(
            {"inputs": [""], "outputs": [f"{token}\n"]},
            separators=(",", ":"),
        )
    }
    row = {
        "id": problem_id,
        "labels": "{}",
        "content": "SandboxFusion isolation preflight",
        "test": json.dumps(tests, separators=(",", ":")),
    }
    return {
        "dataset": "m2rl_livecodebench",
        "id": problem_id,
        "completion": f"```python\nprint({token!r})\n```",
        "config": {
            "dataset_type": "LiveCodeBenchDataset",
            "provided_data": row,
            "run_timeout": 2,
        },
    }


def command_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("run_result") or payload.get("result") or {}
    return result if isinstance(result, dict) else {}


def execution_succeeded(payload: dict[str, Any]) -> bool:
    result = command_result(payload)
    return (
        str(payload.get("status") or "").lower() == "success"
        and str(result.get("status") or "").lower() == "finished"
        and result.get("return_code") == 0
    )


def image_reference_is_pinned(value: str | None) -> bool:
    text = str(value or "")
    digest = text.rsplit("@", 1)[-1] if "@" in text else text
    return (
        len(digest) == 71 and digest.startswith("sha256:") and all(char in "0123456789abcdef" for char in digest[7:])
    )


def is_lower_hex(value: str | None, length: int) -> bool:
    text = str(value or "")
    return len(text) == length and all(char in "0123456789abcdef" for char in text)


def response_diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
    result = command_result(payload)
    return {
        "status": payload.get("status"),
        "message": str(payload.get("message") or "")[:2000],
        "run_status": result.get("status"),
        "return_code": result.get("return_code"),
        "stderr": str(result.get("stderr") or "")[:2000],
    }


def json_stdout(payload: dict[str, Any]) -> dict[str, Any]:
    if not execution_succeeded(payload):
        return {}
    try:
        value = json.loads(stdout(payload).strip())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    atomic_json(
        args.marker,
        {
            "schema_version": 2,
            "tested_at_utc": datetime.now(timezone.utc).isoformat(),
            "url": args.url,
            "safe": False,
            "reason": "active probes are still running",
        },
    )
    base_url = args.url.rsplit("/run_code", 1)[0].rstrip("/")
    ping = request_json(f"{base_url}/v1/ping", None, args.request_timeout)
    token = secrets.token_hex(16)
    execution = run_code(args.url, f"print({token!r})", request_timeout=args.request_timeout)
    submit_url = f"{base_url}/submit"
    submit_request = livecodebench_payload(token)
    livecodebench_submit = request_json(submit_url, submit_request, max(args.request_timeout, 20))

    custom_extract_request = json.loads(json.dumps(submit_request))
    custom_extract_request["config"]["custom_extract_logic"] = "raise RuntimeError('control-plane execution')"
    custom_extract_status, custom_extract_response = request_json_status(
        submit_url,
        custom_extract_request,
        args.request_timeout,
    )
    legacy_pickle_request = json.loads(json.dumps(submit_request))
    legacy_pickle_request["config"]["provided_data"]["test"] = "not-json-and-must-never-be-unpickled"
    legacy_pickle_status, legacy_pickle_response = request_json_status(
        submit_url,
        legacy_pickle_request,
        args.request_timeout,
    )

    descriptor, canary_name = tempfile.mkstemp(prefix="sandboxfusion-host-canary-")
    canary = Path(canary_name)
    try:
        os.write(descriptor, token.encode("utf-8"))
        os.close(descriptor)
        descriptor = -1
        canary.chmod(0o644)
        filesystem = run_code(
            args.url,
            (
                "from pathlib import Path\n"
                f"p=Path({str(canary)!r})\n"
                "print(p.read_text() if p.exists() else 'HOST_PATH_NOT_VISIBLE')\n"
            ),
            request_timeout=args.request_timeout,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        canary.unlink(missing_ok=True)

    service_filesystem = run_code(
        args.url,
        (
            "from pathlib import Path\n"
            f"p=Path({str(args.service_canary_path)!r})\n"
            "print(p.read_text() if p.exists() else 'SERVICE_PATH_NOT_VISIBLE')\n"
        ),
        request_timeout=args.request_timeout,
    )

    network = run_code(
        args.url,
        (
            "import socket\n"
            "try:\n"
            " s=socket.create_connection(('1.1.1.1',443),1); s.close(); print('NETWORK_OPEN')\n"
            "except Exception:\n"
            " print('NETWORK_DENIED')\n"
        ),
        request_timeout=args.request_timeout,
    )
    cgroup = run_code(
        args.url,
        (
            "import json\n"
            "from pathlib import Path\n"
            "line=next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))\n"
            "relative=line.split('::',1)[1]\n"
            "group=Path('/sys/fs/cgroup')\n"
            "pids_max=(group/'pids.max').read_text().strip()\n"
            "writable=False\n"
            "try:\n"
            " (group/'pids.max').write_text(pids_max); writable=True\n"
            "except OSError:\n"
            " pass\n"
            "print(json.dumps({'path':relative,'memory_max':(group/'memory.max').read_text().strip(),"
            "'memory_swap_max':(group/'memory.swap.max').read_text().strip(),"
            "'memory_oom_group':(group/'memory.oom.group').read_text().strip(),"
            "'cpu_max':(group/'cpu.max').read_text().strip(),'pids_max':pids_max,"
            "'writable':writable,'child_groups':sum(p.is_dir() for p in group.iterdir())}))\n"
        ),
        request_timeout=args.request_timeout,
    )
    namespace_code = (
        "import json,os\n"
        "print(json.dumps({name:os.readlink('/proc/self/ns/'+name) "
        "for name in ('mnt','pid','net','ipc','uts')}))\n"
    )
    namespace_first = run_code(args.url, namespace_code, request_timeout=args.request_timeout)
    namespace_second = run_code(args.url, namespace_code, request_timeout=args.request_timeout)
    privilege = run_code(
        args.url,
        (
            "import json,os\n"
            "from pathlib import Path\n"
            "status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)\n"
            "print(json.dumps({'euid':os.geteuid(),'cap_eff':status['CapEff'].strip(),"
            "'no_new_privs':status['NoNewPrivs'].strip(),'seccomp':status['Seccomp'].strip(),"
            "'seccomp_filters':status.get('Seccomp_filters','').strip()}))\n"
        ),
        request_timeout=args.request_timeout,
    )
    dangerous_syscalls = run_code(
        args.url,
        (
            "import ctypes,json\n"
            "libc=ctypes.CDLL(None,use_errno=True)\n"
            "ctypes.set_errno(0)\n"
            "rc=libc.unshare(0x10000000)\n"
            "print(json.dumps({'userns_rc':rc,'userns_errno':ctypes.get_errno()}))\n"
        ),
        request_timeout=args.request_timeout,
    )
    memory = run_code(
        args.url,
        "x=bytearray(256 * 1024 * 1024)\nprint(len(x))\n",
        memory_limit_mb=64,
        request_timeout=args.request_timeout,
    )
    timeout_result = run_code(
        args.url,
        "while True:\n pass\n",
        run_timeout=0.25,
        request_timeout=args.request_timeout,
    )
    run_result = timeout_result.get("run_result") or {}
    timeout_status = str(run_result.get("status") or timeout_result.get("status") or "").lower()
    cgroup_report = json_stdout(cgroup)
    privilege_report = json_stdout(privilege)
    namespace_first_report = json_stdout(namespace_first)
    namespace_second_report = json_stdout(namespace_second)
    dangerous_syscalls_report = json_stdout(dangerous_syscalls)
    try:
        service_namespace_report = json.loads(args.service_namespaces)
    except json.JSONDecodeError:
        service_namespace_report = {}
    if not isinstance(service_namespace_report, dict):
        service_namespace_report = {}
    cgroup_path = str(cgroup_report.get("path") or "")
    expected_cgroup_parent = Path(str(args.cgroup_path or "")).name
    memory_result = command_result(memory)
    seccomp_filter_count = str(privilege_report.get("seccomp_filters") or "")
    memory_limited = (
        str(memory.get("status") or "").lower() == "failed"
        and str(memory_result.get("status") or "").lower() == "finished"
        and memory_result.get("return_code") not in (None, 0)
        and "268435456" not in stdout(memory)
    )
    checks = {
        "ping": ping == "pong",
        "execution": execution_succeeded(execution) and token in stdout(execution),
        "livecodebench_submit": (
            livecodebench_submit.get("accepted") is True
            and livecodebench_submit.get("id") == submit_request["id"]
            and livecodebench_submit.get("extracted_code", "").strip() == f"print({token!r})"
        ),
        "submit_policy_enforced": custom_extract_status == 400 and legacy_pickle_status == 400,
        "control_plane_network_denied": (
            args.control_plane_network_internal == "true" and args.control_plane_network_denied == "true"
        ),
        "filesystem_separation": (
            execution_succeeded(filesystem)
            and token not in stdout(filesystem)
            and "HOST_PATH_NOT_VISIBLE" in stdout(filesystem)
        ),
        "service_filesystem_separation": (
            execution_succeeded(service_filesystem)
            and args.service_canary_token not in stdout(service_filesystem)
            and "SERVICE_PATH_NOT_VISIBLE" in stdout(service_filesystem)
        ),
        "network_denied": (
            execution_succeeded(network)
            and "NETWORK_OPEN" not in stdout(network)
            and "NETWORK_DENIED" in stdout(network)
        ),
        "cgroup_v2_enforced": (
            execution_succeeded(cgroup)
            and f"/{expected_cgroup_parent}/sandboxfusion-" in cgroup_path
            and cgroup_report.get("memory_max") == str(256 * 1024 * 1024)
            and cgroup_report.get("memory_swap_max") == "0"
            and cgroup_report.get("memory_oom_group") == "1"
            and cgroup_report.get("cpu_max") == "100000 100000"
            and cgroup_report.get("pids_max") == "512"
            and cgroup_report.get("writable") is False
            and cgroup_report.get("child_groups") == 0
        ),
        "memory_limit_enforced": memory_limited,
        "least_privilege": (
            execution_succeeded(privilege)
            and privilege_report.get("euid") == 1000
            and privilege_report.get("cap_eff") == "0000000000000000"
            and privilege_report.get("no_new_privs") == "1"
            and privilege_report.get("seccomp") == "2"
            and (not seccomp_filter_count or int(seccomp_filter_count) >= 2)
        ),
        "dangerous_syscalls_denied": (
            execution_succeeded(dangerous_syscalls)
            and dangerous_syscalls_report.get("userns_rc") == -1
            and dangerous_syscalls_report.get("userns_errno") == 1
        ),
        "namespace_isolation": (
            execution_succeeded(namespace_first)
            and execution_succeeded(namespace_second)
            and set(namespace_first_report) == {"mnt", "pid", "net", "ipc", "uts"}
            and set(namespace_second_report) == set(namespace_first_report)
            and set(service_namespace_report) == set(namespace_first_report)
            and all(
                namespace_first_report[name] != service_namespace_report.get(name)
                and namespace_second_report[name] != service_namespace_report.get(name)
                for name in namespace_first_report
            )
        ),
        "timeout_enforced": (
            str(timeout_result.get("status") or "").lower() == "failed" and "timelimit" in timeout_status
        ),
        "deployment_attested": (
            args.sandbox_config == "ci"
            and args.cgroup_version == "2"
            and args.patch_id == "cgroup2-v1"
            and args.base_image == AUDITED_BASE_IMAGE
            and is_lower_hex(args.patch_sha256, 64)
            and is_lower_hex(args.container_id, 64)
            and is_lower_hex(args.proxy_container_id, 64)
            and str(args.image_id or "").startswith("sha256:")
            and is_lower_hex(str(args.image_id or "")[7:], 64)
            and args.proxy_image_id == args.image_id
            and is_lower_hex(args.compose_sha256, 64)
            and str(args.aggregate_memory_max).isdigit()
            and int(args.aggregate_memory_max) > 0
            and str(args.aggregate_pids_max).isdigit()
            and int(args.aggregate_pids_max) > 0
            and image_reference_is_pinned(args.image_reference)
        ),
        "livecodebench_upload_capacity": (
            args.max_upload_bytes >= MIN_LIVECODEBENCH_UPLOAD_BYTES
            and args.livecodebench_max_staged_bytes > 0
            and args.livecodebench_max_staged_bytes <= args.max_upload_bytes
        ),
    }
    marker = {
        "schema_version": 2,
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "safe": all(checks.values()),
        "checks": checks,
        "deployment": {
            "container_id": args.container_id,
            "proxy_container_id": args.proxy_container_id,
            "image_id": args.image_id,
            "proxy_image_id": args.proxy_image_id,
            "image_reference": args.image_reference,
            "compose_sha256": args.compose_sha256,
            "sandbox_config": args.sandbox_config,
            "cgroup_version": args.cgroup_version,
            "cgroup_path": args.cgroup_path,
            "patch_id": args.patch_id,
            "patch_sha256": args.patch_sha256,
            "base_image": args.base_image,
            "aggregate_memory_max": args.aggregate_memory_max,
            "aggregate_pids_max": args.aggregate_pids_max,
            "max_upload_bytes": args.max_upload_bytes,
            "control_plane_network_internal": args.control_plane_network_internal,
        },
        "diagnostics": {
            "execution": response_diagnostic(execution),
            "livecodebench_submit": livecodebench_submit,
            "submit_policy": {
                "custom_extract_status": custom_extract_status,
                "custom_extract_response": custom_extract_response,
                "legacy_pickle_status": legacy_pickle_status,
                "legacy_pickle_response": legacy_pickle_response,
            },
            "filesystem": response_diagnostic(filesystem),
            "service_filesystem": response_diagnostic(service_filesystem),
            "network": response_diagnostic(network),
            "cgroup": {**response_diagnostic(cgroup), "report": cgroup_report},
            "privilege": {**response_diagnostic(privilege), "report": privilege_report},
            "dangerous_syscalls": {
                **response_diagnostic(dangerous_syscalls),
                "report": dangerous_syscalls_report,
            },
            "namespace_first": {
                **response_diagnostic(namespace_first),
                "report": namespace_first_report,
            },
            "namespace_second": {
                **response_diagnostic(namespace_second),
                "report": namespace_second_report,
            },
            "service_namespaces": service_namespace_report,
            "memory": response_diagnostic(memory),
            "timeout": response_diagnostic(timeout_result),
            "livecodebench_upload": {
                "minimum_contract_bytes": MIN_LIVECODEBENCH_UPLOAD_BYTES,
                "max_staged_bytes": args.livecodebench_max_staged_bytes,
                "runtime_limit_bytes": args.max_upload_bytes,
            },
        },
        "notes": {
            "filesystem_probe": "A world-readable client-host canary was not visible to submitted code.",
            "service_filesystem_probe": "A world-readable canary in the service container's /tmp was not visible to submitted code.",
            "network_probe": "Submitted code could not open a TCP connection to 1.1.1.1:443.",
            "control_plane_network_probe": "The API service used one Docker-internal bridge and could not reach 1.1.1.1:443.",
            "resource_probe": "Submitted code ran in a cgroup v2 leaf with CPU, memory, and PID controls; an over-limit allocation was killed.",
            "privilege_probe": "Submitted code ran as uid 1000 without effective capabilities and with no_new_privs.",
            "seccomp_probe": "Submitted code inherited the extra untrusted-code seccomp filter and user namespace creation returned EPERM.",
            "submit_probe": "The training /submit route executed a JSON LiveCodeBench case in the sandbox and rejected control-plane extraction code and legacy pickle input.",
            "upload_probe": "The runtime upload limit covers the API contract and the largest staged row in the configured online LiveCodeBench parquet.",
            "namespace_probe": "Two executions each differed from the service mount, PID, network, IPC, and UTS namespaces.",
            "scope": "Active black-box checks; retain the pinned deployment config, patch hash, and image digest as provenance.",
        },
    }
    atomic_json(args.marker, marker)
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/run_code")
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=15)
    parser.add_argument("--container-id")
    parser.add_argument("--proxy-container-id")
    parser.add_argument("--image-id")
    parser.add_argument("--proxy-image-id")
    parser.add_argument("--image-reference")
    parser.add_argument("--compose-sha256")
    parser.add_argument("--sandbox-config")
    parser.add_argument("--cgroup-version")
    parser.add_argument("--cgroup-path")
    parser.add_argument("--patch-id")
    parser.add_argument("--patch-sha256")
    parser.add_argument("--base-image")
    parser.add_argument("--aggregate-memory-max", required=True)
    parser.add_argument("--aggregate-pids-max", required=True)
    parser.add_argument("--max-upload-bytes", type=int, required=True)
    parser.add_argument("--livecodebench-max-staged-bytes", type=int, required=True)
    parser.add_argument("--service-canary-path", required=True)
    parser.add_argument("--service-canary-token", required=True)
    parser.add_argument("--service-namespaces", required=True)
    parser.add_argument("--control-plane-network-internal", required=True)
    parser.add_argument("--control-plane-network-denied", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = preflight(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["safe"]:
        raise SystemExit("Sandbox isolation preflight failed; marker was written with safe=false.")
