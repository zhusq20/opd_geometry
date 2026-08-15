import logging
import os
import re
from copy import deepcopy
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    run_id = _resolve_wandb_run_id(args)

    # Prepare wandb init parameters. Keep the historical group-derived run
    # name unless a launcher supplies an explicit experiment name.
    base_group = args.wandb_group or "default"
    base_run_name = getattr(args, "wandb_run_name", None) or base_group
    if args.wandb_random_suffix:
        suffix = wandb.util.generate_id()
        group = f"{base_group}_{suffix}"
        run_name = f"{base_run_name}_{suffix}-RANK_{args.rank}"
    else:
        group = base_group
        run_name = base_run_name

    # Prepare wandb init parameters
    init_kwargs = {
        "id": run_id,
        "resume": "allow",
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id
    _log_provenance_artifact(args)


def _resolve_wandb_run_id(args) -> str:
    """Resolve and durably persist the run id before any distributed worker joins."""

    run_id = getattr(args, "wandb_run_id", None)
    id_file_value = getattr(args, "wandb_run_id_file", None)
    id_file = None
    if id_file_value:
        id_file = Path(os.path.expandvars(os.path.expanduser(id_file_value)))
        if run_id is None and id_file.exists():
            run_id = id_file.read_text(encoding="utf-8").strip()
    if run_id is None:
        run_id = wandb.util.generate_id()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(run_id)):
        raise ValueError(f"Invalid W&B run id {run_id!r}.")
    if id_file is not None:
        id_file.parent.mkdir(parents=True, exist_ok=True)
        if id_file.exists():
            persisted = id_file.read_text(encoding="utf-8").strip()
            if persisted and persisted != run_id:
                raise ValueError(f"W&B id file {id_file} contains {persisted!r}, but {run_id!r} was requested.")
        else:
            temporary = id_file.with_name(f".{id_file.name}.{os.getpid()}.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(f"{run_id}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, id_file)
    return str(run_id)


def _log_provenance_artifact(args) -> None:
    manifest_value = getattr(args, "run_manifest_path", None)
    if not manifest_value or wandb.run is None:
        return
    manifest = Path(os.path.expandvars(os.path.expanduser(manifest_value)))
    if not manifest.is_file():
        logger.warning("Run manifest does not exist; skipping W&B provenance artifact: %s", manifest)
        return
    artifact = wandb.Artifact(
        name=f"{wandb.run.id}-provenance",
        type="run-provenance",
        description="Exact command, configuration hashes, environment and source snapshot for this run.",
    )
    artifact.add_file(str(manifest), name="run_manifest.json")
    for snapshot in sorted(manifest.parent.glob("source_snapshot*.tar.gz")):
        artifact.add_file(str(snapshot), name=snapshot.name)
    for marker in sorted(manifest.parent.glob("run_*_before_resume_*.json")):
        artifact.add_file(str(marker), name=f"resume_history/{marker.name}")
    inputs = manifest.parent / "inputs"
    if inputs.is_dir():
        for path in sorted(item for item in inputs.rglob("*") if item.is_file()):
            artifact.add_file(str(path), name=f"inputs/{path.relative_to(inputs)}")
    wandb.log_artifact(artifact)


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def _args_to_config_dict(args):
    output = deepcopy(args.__dict__)
    # Credentials must never become W&B run configuration. The preferred
    # authentication path is WANDB_API_KEY, which is not part of argparse.
    output.pop("wandb_key", None)
    return output


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, role=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
    wandb.define_metric("geometry/step")
    wandb.define_metric("geometry/*", step_metric="geometry/step")
    wandb.define_metric("forgetting/step")
    wandb.define_metric("forgetting/*", step_metric="forgetting/step")
