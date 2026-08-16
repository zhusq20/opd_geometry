"""Unit and contract tests for the patched SandboxFusion image layer."""

from __future__ import annotations

import importlib.util
import base64
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "examples/optimizer_geometry/sandboxfusion_patch"
CGROUP_MODULE = PATCH_ROOT / "sandbox/runners/cgroup_v2.py"
PREPARE_MODULE = PATCH_ROOT / "scripts/prepare_cgroup2.py"
FILE_SECURITY_MODULE = PATCH_ROOT / "sandbox/runners/file_security.py"
RESTRICT_EXEC = PATCH_ROOT / "scripts/restrict_exec.py"
PREFLIGHT_MODULE = ROOT / "examples/optimizer_geometry/sandbox_preflight.py"
BUILD_SCRIPT = ROOT / "examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh"
START_SCRIPT = ROOT / "examples/optimizer_geometry/start_sandboxfusion.sh"
COMPOSE_FILE = ROOT / "examples/optimizer_geometry/sandboxfusion-compose.yaml"
OJ_MODULE = PATCH_ROOT / "sandbox/server/online_judge_api.py"


def _load_cgroup_module():
    spec = importlib.util.spec_from_file_location("sandboxfusion_patch_cgroup_v2", CGROUP_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("sandboxfusion_patch_prepare_cgroup2", PREPARE_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_file_security_module():
    spec = importlib.util.spec_from_file_location("sandboxfusion_patch_file_security", FILE_SECURITY_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location("sandboxfusion_active_preflight", PREFLIGHT_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_oj_module(monkeypatch):
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class APIRouter:
        def get(self, *_args, **_kwargs):
            return lambda function: function

        def post(self, *_args, **_kwargs):
            return lambda function: function

    fastapi = types.ModuleType("fastapi")
    fastapi.APIRouter = APIRouter
    fastapi.HTTPException = HTTPException
    dataset_types = types.ModuleType("sandbox.datasets.types")
    for name in (
        "CodingDataset",
        "EvalResult",
        "GetMetricsFunctionRequest",
        "GetMetricsFunctionResult",
        "GetMetricsRequest",
        "GetPromptByIdRequest",
        "GetPromptsRequest",
        "Prompt",
        "SubmitRequest",
        "TestConfig",
    ):
        setattr(dataset_types, name, type(name, (), {}))
    registry = types.ModuleType("sandbox.registry")
    registry.get_all_dataset_ids = lambda: []
    registry.get_coding_class_by_dataset = lambda _dataset: None
    registry.get_coding_class_by_name = lambda _name: None

    for name, module in {
        "fastapi": fastapi,
        "sandbox": types.ModuleType("sandbox"),
        "sandbox.datasets": types.ModuleType("sandbox.datasets"),
        "sandbox.datasets.types": dataset_types,
        "sandbox.registry": registry,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("sandboxfusion_patch_online_judge", OJ_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _livecodebench_submit_request(*, custom_extract_logic=None, test_override=None, run_timeout=2):
    problem_id = "preflight-problem"
    test = json.dumps(
        {
            "input_output": json.dumps(
                {"inputs": ["1\n"], "outputs": ["2\n"]},
                separators=(",", ":"),
            )
        },
        separators=(",", ":"),
    )
    row = {
        "id": problem_id,
        "labels": "{}",
        "content": "Double the input.",
        "test": test if test_override is None else test_override,
    }
    config = SimpleNamespace(
        dataset_type="LiveCodeBenchDataset",
        language=None,
        locale=None,
        is_fewshot=None,
        compile_timeout=None,
        run_timeout=run_timeout,
        custom_extract_logic=custom_extract_logic,
        provided_data=row,
        extra={},
    )
    return SimpleNamespace(id=problem_id, completion="```python\nprint(2)\n```", config=config)


@pytest.mark.unit
def test_online_judge_accepts_only_bounded_json_livecodebench_rows(monkeypatch):
    module = _load_oj_module(monkeypatch)
    dataset = type("LiveCodeBenchDataset", (), {})

    module._validate_livecodebench_submit(_livecodebench_submit_request(), dataset)


@pytest.mark.unit
@pytest.mark.parametrize(
    "submit_request",
    [
        _livecodebench_submit_request(custom_extract_logic="raise RuntimeError('unsafe')"),
        _livecodebench_submit_request(test_override="legacy-pickle-payload"),
        _livecodebench_submit_request(run_timeout=0),
        _livecodebench_submit_request(run_timeout=float("nan")),
    ],
)
def test_online_judge_rejects_control_plane_execution_inputs(monkeypatch, submit_request):
    module = _load_oj_module(monkeypatch)
    dataset = type("LiveCodeBenchDataset", (), {})

    with pytest.raises(module.HTTPException) as error:
        module._validate_livecodebench_submit(submit_request, dataset)

    assert error.value.status_code == 400


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4096, 4096),
        ("256M", 256 * 1024**2),
        ("4GiB", 4 * 1024**3),
        ("1T", 1024**4),
    ],
)
def test_cgroup_v2_byte_limit_parser(value, expected):
    module = _load_cgroup_module()
    assert module.parse_byte_limit(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, -1, "", "unlimited", "0M", "1.5G"])
def test_cgroup_v2_byte_limit_parser_rejects_unsafe_values(value):
    module = _load_cgroup_module()
    with pytest.raises(ValueError):
        module.parse_byte_limit(value)


@pytest.mark.unit
def test_cgroup_v2_manager_writes_kernel_v2_controls(tmp_path, monkeypatch):
    module = _load_cgroup_module()
    (tmp_path / "cgroup.controllers").write_text("cpu memory pids\n")
    (tmp_path / "cgroup.subtree_control").write_text("cpu memory pids\n")

    def fake_mkdir(path):
        path.mkdir()
        for name in (
            "cgroup.procs",
            "cpu.max",
            "memory.max",
            "memory.swap.max",
            "memory.oom.group",
            "pids.max",
            "cgroup.kill",
            "cgroup.events",
        ):
            (path / name).write_text("populated 0\n" if name == "cgroup.events" else "")

    monkeypatch.setattr(module, "_mkdir_cgroup", fake_mkdir)
    monkeypatch.setattr(module.secrets, "token_hex", lambda _size: "a" * 24)
    manager = module.CgroupV2Manager(tmp_path)
    handle = manager.create(memory_limit="256M", cpu_limit=0.5, pids_limit=64)

    assert handle.path.parent == tmp_path
    assert handle.path.name == "sandboxfusion-" + "a" * 24
    assert (handle.path / "memory.max").read_text() == f"{256 * 1024**2}\n"
    assert (handle.path / "memory.swap.max").read_text() == "0\n"
    assert (handle.path / "memory.oom.group").read_text() == "1\n"
    assert (handle.path / "cpu.max").read_text() == "50000 100000\n"
    assert (handle.path / "pids.max").read_text() == "64\n"
    assert handle.command_prefix[-1] == str(handle.path)


@pytest.mark.unit
def test_cgroup_v2_manager_requires_delegated_controllers(tmp_path):
    module = _load_cgroup_module()
    (tmp_path / "cgroup.controllers").write_text("cpu memory pids\n")
    (tmp_path / "cgroup.subtree_control").write_text("cpu\n")
    with pytest.raises(module.CgroupV2Error, match="not delegated"):
        module.CgroupV2Manager(tmp_path)


@pytest.mark.unit
def test_cgroup_v2_leaf_is_searchable_but_not_world_writable(tmp_path):
    module = _load_cgroup_module()
    leaf = tmp_path / "sandboxfusion-test"
    module._mkdir_cgroup(leaf)

    assert leaf.stat().st_mode & 0o777 == 0o755


@pytest.mark.unit
def test_cgroup_exec_wrapper_rejects_paths_outside_delegated_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "sandboxfusion-outside"
    outside.mkdir()
    (outside / "cgroup.procs").write_text("")
    env = {**os.environ, "SANDBOX_CGROUP2_ROOT": str(root)}
    result = subprocess.run(
        [
            sys.executable,
            str(PATCH_ROOT / "scripts/cgroup2_exec.py"),
            str(outside),
            "/bin/true",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing unexpected cgroup path" in result.stderr


@pytest.mark.unit
def test_cgroup_exec_wrapper_joins_group_before_exec(tmp_path):
    root = tmp_path / "root"
    group = root / "sandboxfusion-test"
    group.mkdir(parents=True)
    (group / "cgroup.procs").write_text("")
    env = {**os.environ, "SANDBOX_CGROUP2_ROOT": str(root)}
    result = subprocess.run(
        [
            sys.executable,
            str(PATCH_ROOT / "scripts/cgroup2_exec.py"),
            str(group),
            "/bin/sh",
            "-c",
            "printf EXEC_OK",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "EXEC_OK"
    assert int((group / "cgroup.procs").read_text()) > 0


@pytest.mark.unit
def test_restrict_exec_allows_normal_process_and_threads():
    result = subprocess.run(
        [
            sys.executable,
            str(RESTRICT_EXEC),
            sys.executable,
            "-c",
            (
                "import subprocess,threading; "
                "t=threading.Thread(target=lambda:None); t.start(); t.join(); "
                "subprocess.run(['/bin/true'],check=True); print('RESTRICT_OK')"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "RESTRICT_OK"


@pytest.mark.unit
def test_restrict_exec_denies_user_namespace_creation():
    result = subprocess.run(
        [sys.executable, str(RESTRICT_EXEC), "unshare", "--user", "/bin/true"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Operation not permitted" in result.stderr


@pytest.mark.unit
def test_prepare_cgroup_delegates_only_required_controllers(tmp_path, monkeypatch):
    module = _load_prepare_module()
    delegated = tmp_path / "iclr2027-sandboxfusion"
    delegated.mkdir()
    for directory in (tmp_path, delegated):
        (directory / "cgroup.controllers").write_text("cpu io memory pids\n")
        (directory / "cgroup.subtree_control").write_text("")
        (directory / "cgroup.procs").write_text("")

    def fake_write(path, value):
        if path.name == "cgroup.subtree_control" and str(value).startswith("+"):
            current = set(path.read_text().split())
            current.add(str(value)[1:])
            path.write_text(" ".join(sorted(current)) + "\n")
        else:
            path.write_text(f"{value}\n")

    monkeypatch.setattr(module, "write_control", fake_write)
    assert module.prepare(tmp_path, delegated.name) == delegated
    assert set((tmp_path / "cgroup.subtree_control").read_text().split()) == {
        "cpu",
        "memory",
        "pids",
    }
    assert set((delegated / "cgroup.subtree_control").read_text().split()) == {
        "cpu",
        "memory",
        "pids",
    }
    assert (delegated / "memory.max").read_text() == f"{32 * 1024**3}\n"
    assert (delegated / "pids.max").read_text() == "4096\n"
    assert delegated.stat().st_mode & 0o777 == 0o755


@pytest.mark.unit
def test_prepare_cgroup_rejects_unknown_stale_children(tmp_path, monkeypatch):
    module = _load_prepare_module()
    delegated = tmp_path / "iclr2027-sandboxfusion"
    delegated.mkdir()
    (delegated / "unexpected").mkdir()
    for directory in (tmp_path, delegated):
        (directory / "cgroup.controllers").write_text("cpu memory pids\n")
        (directory / "cgroup.subtree_control").write_text("cpu memory pids\n")
        (directory / "cgroup.procs").write_text("")
    with pytest.raises(RuntimeError, match="unexpected child"):
        module.prepare(tmp_path, delegated.name)


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename",
    ["../escape", "/absolute", "a/../../escape", "a/../escape", "", "."],
)
def test_file_staging_rejects_path_escape(tmp_path, filename):
    module = _load_file_security_module()
    with pytest.raises(ValueError):
        module.safe_relative_path(tmp_path, filename)


@pytest.mark.unit
def test_file_staging_rejects_symlink_traversal(tmp_path):
    module = _load_file_security_module()
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        module.safe_relative_path(tmp_path, "link/payload")
    with pytest.raises(ValueError, match="symbolic link"):
        module.make_tree_world_writable(tmp_path)


@pytest.mark.unit
def test_file_staging_enforces_total_decoded_size(tmp_path):
    module = _load_file_security_module()
    encoded = base64.b64encode(b"12345").decode()
    with pytest.raises(ValueError, match="uploads exceed"):
        module.restore_base64_files(
            tmp_path,
            {"one": encoded, "two": encoded},
            max_bytes=9,
        )


@pytest.mark.unit
def test_file_fetch_reads_only_regular_files_beneath_worktree(tmp_path):
    module = _load_file_security_module()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "result.txt").write_bytes(b"result")

    assert (
        module.read_regular_file_beneath(
            tmp_path,
            "nested/result.txt",
            max_bytes=6,
        )
        == b"result"
    )
    assert module.read_regular_file_beneath(tmp_path, "missing", max_bytes=6) is None
    with pytest.raises(ValueError, match="fetched files exceed"):
        module.read_regular_file_beneath(
            tmp_path,
            "nested/result.txt",
            max_bytes=5,
        )


@pytest.mark.unit
@pytest.mark.parametrize("link_parent", [False, True])
def test_file_fetch_never_follows_symlinks(tmp_path, link_parent):
    module = _load_file_security_module()
    outside = tmp_path.parent / "fetch-secret"
    outside.write_text("secret")
    if link_parent:
        real_directory = tmp_path.parent / "fetch-outside-dir"
        real_directory.mkdir()
        (real_directory / "secret").write_text("secret")
        (tmp_path / "link").symlink_to(real_directory, target_is_directory=True)
        filename = "link/secret"
    else:
        (tmp_path / "link").symlink_to(outside)
        filename = "link"

    with pytest.raises(ValueError, match="unsafe sandbox fetch path"):
        module.read_regular_file_beneath(tmp_path, filename, max_bytes=1024)


@pytest.mark.unit
def test_file_fetch_ignores_fifo_without_blocking(tmp_path):
    module = _load_file_security_module()
    os.mkfifo(tmp_path / "pipe")

    assert module.read_regular_file_beneath(tmp_path, "pipe", max_bytes=1024) is None


@pytest.mark.unit
def test_file_upload_rejects_oversized_encoding_before_decode(tmp_path):
    module = _load_file_security_module()
    with pytest.raises(ValueError, match="uploads exceed"):
        module.restore_base64_files(tmp_path, {"large": "A" * 1024}, max_bytes=1)


@pytest.mark.unit
def test_worktree_is_copied_into_overlay_private_tmp(tmp_path):
    module = _load_file_security_module()
    source = tmp_path / "request-worktree"
    source.mkdir()
    (source / "program.py").write_text("print('ok')\n")
    overlay = tmp_path / "overlay-root"
    (overlay / "tmp").mkdir(parents=True)

    destination = module.copy_worktree_into_overlay(overlay, source)

    assert destination != source
    assert destination.relative_to(overlay) == Path("tmp") / source.relative_to("/tmp")
    assert (destination / "program.py").read_text() == "print('ok')\n"


@pytest.mark.unit
def test_worktree_copy_rejects_source_symlinks(tmp_path):
    module = _load_file_security_module()
    source = tmp_path / "request-with-link"
    source.mkdir()
    (source / "link").symlink_to("/etc/passwd")
    overlay = tmp_path / "overlay-root"
    (overlay / "tmp").mkdir(parents=True)

    with pytest.raises(ValueError, match="symbolic link"):
        module.copy_worktree_into_overlay(overlay, source)


@pytest.mark.unit
def test_worktree_copy_rejects_special_files(tmp_path):
    module = _load_file_security_module()
    source = tmp_path / "request-with-fifo"
    source.mkdir()
    os.mkfifo(source / "pipe")
    overlay = tmp_path / "overlay-root"
    (overlay / "tmp").mkdir(parents=True)

    with pytest.raises(ValueError, match="non-regular"):
        module.copy_worktree_into_overlay(overlay, source)


@pytest.mark.unit
def test_worktree_copy_preserves_only_trusted_runtime_symlink(tmp_path, monkeypatch):
    module = _load_file_security_module()
    trusted_runtime = tmp_path / "trusted-runtime"
    trusted_runtime.mkdir()
    (trusted_runtime / "dependency").write_text("trusted")
    monkeypatch.setattr(module, "TRUSTED_SYMLINK_ROOTS", (trusted_runtime,))
    source = tmp_path / "request-with-runtime-link"
    source.mkdir()
    (source / "dependency").symlink_to(trusted_runtime / "dependency")
    overlay = tmp_path / "overlay-root"
    (overlay / "tmp").mkdir(parents=True)

    destination = module.copy_worktree_into_overlay(overlay, source)

    assert (destination / "dependency").is_symlink()
    assert (destination / "dependency").read_text() == "trusted"


@pytest.mark.unit
def test_preflight_does_not_treat_failed_execution_as_filesystem_isolation():
    module = _load_preflight_module()
    failed = {
        "status": "SandboxError",
        "message": "cgroup setup failed",
        "run_result": None,
    }
    assert module.stdout(failed) == ""
    assert module.execution_succeeded(failed) is False
    assert module.json_stdout(failed) == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("sha256:" + "a" * 64, True),
        ("registry.example/sandbox@sha256:" + "b" * 64, True),
        ("registry.example/sandbox:latest", False),
        ("sha256:not-a-digest", False),
    ],
)
def test_preflight_accepts_only_content_addressed_images(reference, expected):
    module = _load_preflight_module()
    assert module.image_reference_is_pinned(reference) is expected


@pytest.mark.unit
def test_patch_is_based_on_digest_and_forces_offline_unprivileged_execution():
    dockerfile = (PATCH_ROOT / "Dockerfile").read_text()
    base = (PATCH_ROOT / "sandbox/runners/base.py").read_text()
    isolation = (PATCH_ROOT / "sandbox/runners/isolation.py").read_text()
    api = (PATCH_ROOT / "sandbox/server/sandbox_api.py").read_text()
    oj_api = (PATCH_ROOT / "sandbox/server/online_judge_api.py").read_text()
    runners_init = (PATCH_ROOT / "sandbox/runners/__init__.py").read_text()
    runner_types = (PATCH_ROOT / "sandbox/runners/types.py").read_text()
    restrict_exec = (PATCH_ROOT / "scripts/restrict_exec.py").read_text()

    assert "volcengine/sandbox-fusion@sha256:" in dockerfile
    assert "sha256sum -c" in dockerfile
    assert "restrict_exec.py" in dockerfile
    assert "tmp_netns(no_bridge=True)" in base
    assert '"--drop-to"' in base
    assert "PR_CAPBSET_DROP" in restrict_exec
    assert "PR_SET_NO_NEW_PRIVS" in restrict_exec
    assert '"--mount-proc=/proc"' in base
    assert base.index('"chroot",') < base.index('"unshare",')
    assert '["mount", "-t", "proc", "-o", "nosuid,nodev,noexec"' in isolation
    assert 'unmount_fs(f"{merged_dir}/proc")' in isolation
    assert '"--ipc"' in base
    assert '"--uts"' in base
    assert "tmp_cgroup_view(root, cgroup.path)" in base
    assert "read_regular_file_beneath" in base
    assert "MAX_FETCH_BYTES" in base
    assert "copy_worktree_into_overlay" in base
    assert '"HOME": "/home/app"' in base
    assert "sandboxfusion-restrict-exec" in base
    assert "psutil.pid_exists" not in base
    assert 'raise RuntimeError("isolation=none is disabled' in base
    assert 'raise RuntimeError("network-enabled lite sandboxes are disabled' in isolation
    assert '"remount,bind,ro,nosuid,nodev,noexec"' in isolation
    assert '"mode=1777,nosuid,nodev,size=4g"' in isolation
    assert '"mount", "--rbind", "/dev"' not in isolation
    assert '["umount", "-l"' not in isolation
    assert '"mknod"' in isolation
    assert "memory_limit_MB" in api
    assert "le=160" in api
    assert "custom extraction code is disabled" in oj_api
    assert "legacy pickle" in oj_api
    assert "LIVE_CODE_BENCH_ROW_KEYS" in oj_api
    assert "GPU_RUNNERS" not in runners_init
    assert '"python_gpu"' not in runner_types
    assert '"cuda"' not in runner_types


@pytest.mark.unit
def test_build_and_start_contract_is_fail_closed_and_content_addressed():
    dockerfile = (PATCH_ROOT / "Dockerfile").read_text()
    build = BUILD_SCRIPT.read_text()
    start = START_SCRIPT.read_text()
    compose = COMPOSE_FILE.read_text()

    subprocess.run(["bash", "-n", BUILD_SCRIPT, START_SCRIPT], check=True)
    assert dockerfile.startswith("FROM volcengine/sandbox-fusion@sha256:")
    assert "ARG UPSTREAM_IMAGE" not in dockerfile
    assert "/root/miniconda3/bin/python3 -m py_compile" in dockerfile
    assert "/bin/python3 -m py_compile" in dockerfile
    assert "RUN --mount=type=bind,source=upstream-files.sha256" in dockerfile
    assert "AS sandboxfusion-final" in dockerfile
    assert "COPY --chmod=0555" in dockerfile
    assert "docker buildx build" in build
    assert "--target sandboxfusion-verified" in build
    assert "--target sandboxfusion-final" in build
    assert build.index("--target sandboxfusion-verified") < build.index('build_image "${BUILD_TAG}"')
    assert "--output type=docker,rewrite-timestamp=true" in build
    assert "image_runtime_fingerprint" in build
    assert 'stat -c "%n|%f|%a|%u|%g|%s|%Y"' in build
    assert "--no-cache" in build
    assert "Reproducibility check failed" in build
    assert "Docker CLI is required" in build
    assert "'docker info' failed" in build
    assert "SANDBOXFUSION_IMAGE=%s" in build
    assert "cp -R --no-preserve=mode,ownership,timestamps" in build
    assert "cp -a" not in build
    assert start.index("write_unsafe_marker") < start.index("down --remove-orphans")
    assert "--pull never" in start
    assert "cleanup_failed_deployment" in start
    assert "127.0.0.1:${PORT}" in start
    assert 'CGROUP_NAME="${SANDBOXFUSION_CGROUP_NAME:-sandboxfusion}"' in start
    assert "--cap-add DAC_OVERRIDE" in start
    assert "ps -q loopback_proxy" in start
    assert "--proxy-container-id" in start
    assert "validate_livecodebench_upload.py" in start
    assert "--max-upload-bytes" in start
    assert "--livecodebench-max-staged-bytes" in start
    assert "image: ${SANDBOXFUSION_IMAGE:?" in compose
    assert "loopback_proxy:" in compose
    assert "SANDBOXFUSION_PROXY_UPSTREAM_HOST: sandboxfusion" in compose
    assert "cgroup: host" in compose
    assert "privileged: false" in compose
    assert "read_only: true" in compose
    assert "mem_limit: 4g" in compose
    assert "pids_limit: 2048" in compose
    assert "cap_drop:\n      - ALL" in compose
    assert "      - SYS_ADMIN" in compose
    assert "      - NET_ADMIN" in compose
    assert "no-new-privileges=true" in compose
    assert "127.0.0.1:${SANDBOXFUSION_PORT:-8080}:8080" in compose
    assert "internal: true" in compose
    assert "isolation=none" not in compose
    assert 'SANDBOX_MAX_UPLOAD_BYTES: "${SANDBOXFUSION_MAX_UPLOAD_BYTES:-150994944}"' in compose
