import fcntl
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    _log_local_jsonl(args, metrics, step_key)

    if getattr(args, "use_wandb", False):
        wandb.log(metrics)

    if getattr(args, "use_tensorboard", False):
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "numel") and value.numel() == 1:
        value = value.detach().cpu().item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _log_local_jsonl(args, metrics: dict[str, Any], step_key: str) -> None:
    """Append one complete scalar event independently of W&B availability.

    Different Ray roles may write concurrently. An advisory file lock plus
    ``O_APPEND`` keeps each complete JSON record intact, while a file per
    metric namespace avoids unnecessary cross-role contention.
    """

    output_value = getattr(args, "metrics_output_dir", None)
    if not output_value:
        return
    output_dir = Path(os.path.expandvars(os.path.expanduser(output_value)))
    output_dir.mkdir(parents=True, exist_ok=True)
    namespace = re.sub(r"[^A-Za-z0-9_.-]+", "_", step_key.split("/", 1)[0]) or "metrics"
    record = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "step_key": step_key,
        "metrics": _json_safe(metrics),
    }
    payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(output_dir / f"{namespace}.jsonl", os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(f"Failed to append durable metrics for namespace {namespace}.")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def mark_run_complete(args, *, final_num_updates: int) -> None:
    """Write a success marker only after all requested work has completed."""

    marker_value = getattr(args, "completion_marker_path", None)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_num_updates": int(final_num_updates),
        "num_rollout": int(getattr(args, "num_rollout", 0)),
        "start_rollout_id": int(getattr(args, "start_rollout_id", 0) or 0),
        "experiment_name": getattr(args, "experiment_name", None),
        "wandb_run_id": getattr(args, "wandb_run_id", None),
    }
    if marker_value:
        marker = Path(os.path.expandvars(os.path.expanduser(marker_value)))
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    if getattr(args, "use_wandb", False) and wandb.run is not None:
        wandb.run.summary["run/status"] = "complete"
        wandb.run.summary["run/final_num_updates"] = int(final_num_updates)
