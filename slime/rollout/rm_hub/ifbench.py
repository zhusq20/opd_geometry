from __future__ import annotations

import fcntl
import hashlib
import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_IFBENCH_REQUIREMENTS = _WORKSPACE_ROOT / "examples" / "eval_multi_task" / "requirements_ifbench.txt"
_IFBENCH_REPOSITORY = "https://github.com/allenai/IFBench.git"
_IFBENCH_REVISION = "1091c4c3de6c1f6ed12c012ed68f11ea450b0117"
_DEFAULT_IFBENCH_REPO = _WORKSPACE_ROOT / "data" / "m2rl" / "ifbench_scorer" / "IFBench"
_EVALUATION_LIB: Any | None = None


def _git_repo_command(repo_path: Path, *args: str) -> list[str]:
    # Some shared/container workspaces expose cloned files as uid 65534 even
    # though the current process created them. Scope the exception to this
    # exact managed checkout instead of mutating the user's global Git config.
    return ["git", "-c", f"safe.directory={repo_path}", "-C", str(repo_path), *args]


def _ensure_ifbench_repo() -> Path:
    """Prepare the pinned IFBench scorer in a writable, shared cache."""

    configured_path = os.environ.get("SLIME_IFBENCH_REPO")
    repo_path = Path(configured_path).expanduser() if configured_path else _DEFAULT_IFBENCH_REPO
    repo_path = repo_path.resolve()
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    # Ray reward workers can reach this import concurrently. Serialize the
    # one-time clone so all workers observe a complete checkout.
    lock_path = repo_path.parent / ".ifbench_repo.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not repo_path.exists():
            clone_cmd = ["git", "clone", "--filter=blob:none", _IFBENCH_REPOSITORY, str(repo_path)]
            try:
                subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
                subprocess.run(
                    _git_repo_command(repo_path, "checkout", "--detach", _IFBENCH_REVISION),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception as exc:
                raise ImportError(
                    "Unable to prepare the pinned IFBench scorer. Set SLIME_IFBENCH_REPO to an existing "
                    f"checkout of {_IFBENCH_REPOSITORY} at {_IFBENCH_REVISION}."
                ) from exc

    if not (repo_path / "evaluation_lib.py").is_file():
        raise ImportError(f"SLIME_IFBENCH_REPO is not an IFBench checkout: {repo_path}")

    if configured_path is None:
        try:
            revision = subprocess.run(
                _git_repo_command(repo_path, "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception as exc:
            raise ImportError(f"Unable to identify the managed IFBench checkout at {repo_path}.") from exc
        if revision != _IFBENCH_REVISION:
            raise ImportError(
                f"Managed IFBench checkout has revision {revision}, expected {_IFBENCH_REVISION}. "
                "Remove that managed cache directory or set SLIME_IFBENCH_REPO explicitly."
            )

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath is None:
        os.environ["PYTHONPATH"] = repo_str
    elif repo_str not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_str, current_pythonpath])

    return repo_path


def _ensure_ifbench_dependencies(repo_path: Path) -> None:
    """Install IFBench requirements the first time the module is imported."""

    requirements_file = _LOCAL_IFBENCH_REQUIREMENTS

    if not requirements_file.exists():
        logger.debug("Local IFBench requirements file not found at %s; skipping install.", requirements_file)
        return

    signature = hashlib.sha256(requirements_file.read_bytes()).hexdigest()
    sentinel = repo_path / ".deps_installed"
    lock_path = repo_path / ".deps_install.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if sentinel.exists() and sentinel.read_text(encoding="utf-8").strip() == signature:
            return

        install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        try:
            subprocess.run(install_cmd, check=True)
        except Exception as exc:
            logger.warning("Failed to install IFBench dependencies automatically: %s", exc)
        else:
            sentinel.write_text(signature + "\n", encoding="utf-8")


def _load_evaluation_lib():
    global _EVALUATION_LIB
    if _EVALUATION_LIB is not None:
        return _EVALUATION_LIB
    repo_path = _ensure_ifbench_repo()
    importlib.invalidate_caches()
    try:
        _EVALUATION_LIB = importlib.import_module("evaluation_lib")
    except ImportError:
        _ensure_ifbench_dependencies(repo_path)
        importlib.invalidate_caches()
        _EVALUATION_LIB = importlib.import_module("evaluation_lib")
    return _EVALUATION_LIB


JsonDict = dict[str, Any]
KwargsDict = dict[str, str | int | float | None]


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    """Ensure instruction identifiers are clean strings."""

    normalized: list[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _coerce_kwargs_list(
    raw_kwargs: Any,
    num_instructions: int,
) -> list[KwargsDict]:
    """Convert stored kwargs into the list structure expected by IFBench."""

    if isinstance(raw_kwargs, list):
        processed: list[KwargsDict] = []
        for entry in raw_kwargs:
            if isinstance(entry, dict):
                processed.append(dict(entry))
            else:
                processed.append({})
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    # Remove explicit None values to match official preprocessing.
    sanitized: list[KwargsDict] = []
    for entry in processed:
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    return sanitized


def _build_input_example(metadata: JsonDict, evaluation_lib: Any | None = None) -> Any | None:
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list") or [])
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return None

    prompt_text = metadata.get("prompt_text")
    if prompt_text is None:
        prompt_text = ""
    else:
        prompt_text = str(prompt_text)

    raw_kwargs = metadata.get("kwargs")
    kwargs_list = _coerce_kwargs_list(raw_kwargs, len(instruction_ids))

    if evaluation_lib is None:
        evaluation_lib = _load_evaluation_lib()
    return evaluation_lib.InputExample(
        key=int(metadata.get("record_id") or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


def compute_ifbench_reward(response: str, label: Any, metadata: JsonDict | None = None) -> float:
    """Score a model response using the official IFBench rules."""

    if metadata is None:
        logger.debug("No metadata provided for IFBench scoring.")
        return 0.0

    if response is None:
        return 0.0

    evaluation_lib = _load_evaluation_lib()
    inp = _build_input_example(metadata, evaluation_lib)
    if inp is None:
        return 0.0

    prompt_to_response = {inp.prompt: str(response or "")}
    output = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    return 1.0 if output.follow_all_instructions else 0.0
