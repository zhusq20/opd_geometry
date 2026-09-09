import fcntl
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, mark_run_complete
from slime.utils.metric_utils import num_updates_before_rollout, updates_per_rollout
from slime.utils.misc import should_run_periodic_action
from slime_plugins.mopd.sampler import crossed_response_milestone


def _eval_only_num_updates(args):
    step = getattr(args, "eval_checkpoint_step", None)
    if step is None:
        return num_updates_before_rollout(args, args.start_rollout_id)
    if step < 0:
        raise ValueError("--eval-checkpoint-step must be nonnegative")
    return int(step)


def _evaluate_mopd(args, rollout_manager, actor_model, rollout_id, step):
    if getattr(args, "mopd_profile", None) or getattr(args, "mopd_loss", None) != "teacher_topk":
        return ray.get(
            rollout_manager.eval.remote(rollout_id, num_updates=step, model_version=step, eval_phase="step_clock")
        )
    from slime_plugins.mopd.reference_bank import loss_record, write_json

    started = time.perf_counter()
    bank = ray.get(rollout_manager.prepare_mopd_bank.remote(rollout_id, args.mopd_reference_bank, initial=step == 0))
    initial_scores = Path(bank).with_suffix(".initial.json")
    with initial_scores.with_suffix(".lock").open("a") as lock:
        if step == 0:
            fcntl.flock(lock, fcntl.LOCK_EX)
        if step == 0 and initial_scores.is_file():
            record = json.loads(initial_scores.read_text())
            if record["bank_sha256"] != loss_record(bank, [0.0] * 256)["bank_sha256"]:
                raise ValueError("cached initial scores do not match the reference bank")
        else:
            record = loss_record(bank, actor_model.score_mopd_bank(bank))
            if step == 0:
                write_json(initial_scores, record)
    directory = Path(args.save).parent / "fixed_loss"
    write_json(
        directory / f"step_{step:04d}.json",
        {
            **record,
            "step": step,
            "evaluation_wall_seconds": time.perf_counter() - started,
        },
    )
    if step == args.mopd_total_steps:
        fresh_started = time.perf_counter()
        fresh_bank = ray.get(
            rollout_manager.prepare_mopd_bank.remote(rollout_id, str(directory / "fresh_bank.pt"), initial=False)
        )
        write_json(
            directory / "fresh_final.json",
            {
                **loss_record(fresh_bank, actor_model.score_mopd_bank(fresh_bank)),
                "step": step,
                "evaluation_wall_seconds": time.perf_counter() - fresh_started,
            },
        )


def _save_mopd_actor_checkpoint(args, rollout_manager, actor_model, rollout_id):
    """Save a synchronous MOPD actor and its resumable sampler frontier."""

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())
    try:
        actor_model.save_model(rollout_id, force_sync=True)
    finally:
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
            # Updatable SGLang servers intentionally do not keep a CPU backup
            # of actor weights. Resuming the weights region only reallocates
            # its storage, so repopulate it before generation can resume.
            actor_model.update_weights()
            ray.get(rollout_manager.onload_kv.remote())


def _mark_and_prune_mopd_optimizer_checkpoints(args, entries):
    """Keep the latest full state and Uniform step 250; archive scheduled HF weights."""

    retained_uniform_steps = {250}
    for entry in entries:
        entry["optimizer_state_retained"] = bool(
            getattr(args, "mopd_profile", None)
            or entry is entries[-1]
            or (args.mopd_allocation == "uniform" and int(entry["optimizer_step"]) in retained_uniform_steps)
        )

    index_path = Path(args.save).resolve() / "mopd_checkpoint_index.json"
    temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, index_path)

    for entry in entries[:-1]:
        if not entry["optimizer_state_retained"]:
            checkpoint = Path(args.save).resolve() / f"iter_{int(entry['rollout_id']):07d}"
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint)


def train(args):
    session_started = time.perf_counter()
    configure_logger()
    release_train = args.release_train

    if getattr(args, "mopd_profile", None) and not ray.is_initialized():
        # Each local paper run uses its selected visible GPUs. Starting explicitly
        # avoids reconnecting to a stale cluster or sharing another run's actors.
        actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        local_gpus = (
            args.rollout_num_gpus
            if args.debug_rollout_only
            else (max(actor_gpus, args.rollout_num_gpus) if args.colocate else actor_gpus + args.rollout_num_gpus)
        )
        ray.init(
            address="local",
            include_dashboard=False,
            num_cpus=8,
            num_gpus=local_gpus,
            object_store_memory=2 * 1024**3,
            _temp_dir=tempfile.mkdtemp(prefix="mopd-ray-"),
        )

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if getattr(args, "mopd_common_checkpoint", False):
        from slime_plugins.mopd.diagnostic import run_common_checkpoint

        run_common_checkpoint(
            args, actor_model, rollout_manager, startup_seconds=time.perf_counter() - session_started
        )
        ray.get(rollout_manager.dispose.remote())
        mark_run_complete(args, final_num_updates=20)
        finish_tracking(args)
        return

    last_eval_num_updates = None
    mopd_enabled = bool(getattr(args, "mopd_enabled", False))
    last_rollout_id = args.start_rollout_id - 1
    final_mopd_status = None
    mopd_eval_responses = (
        tuple(int(value) for value in args.mopd_eval_responses.split(",") if value) if mopd_enabled else ()
    )

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        num_updates = _eval_only_num_updates(args)
        ray.get(
            rollout_manager.eval.remote(
                rollout_id=args.start_rollout_id,
                num_updates=num_updates,
                model_version=num_updates,
                eval_phase="eval_only",
            )
        )
        last_eval_num_updates = num_updates

    if mopd_enabled:
        resume_status = ray.get(rollout_manager.mopd_budget_status.remote())
        mopd_startup_seconds = time.perf_counter() - session_started
        if args.eval_interval is not None and (
            int(resume_status["optimizer_updates"]) == 0 or args.mopd_eval_on_start
        ):
            resume_rollout_id = max(0, args.start_rollout_id - 1)
            _evaluate_mopd(
                args, rollout_manager, actor_model, resume_rollout_id, int(resume_status["optimizer_updates"])
            )
            last_eval_num_updates = int(resume_status["optimizer_updates"])

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if mopd_enabled:
            final_mopd_status = ray.get(rollout_manager.mopd_budget_status.remote())
            if final_mopd_status["complete"]:
                break
        last_rollout_id = rollout_id
        if not mopd_enabled and args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(
                rollout_manager.eval.remote(
                    rollout_id,
                    num_updates=0,
                    model_version=0,
                    eval_phase="pre_train",
                )
            )
            last_eval_num_updates = 0

        mopd_step_started = time.perf_counter() if mopd_enabled else None
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                actor_results = ray.get(
                    actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs)
                )
            else:
                ray.get(value_refs)
                actor_results = []
        else:
            actor_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        feedback = None
        if mopd_enabled:
            feedback = next(
                (value for value in actor_results if isinstance(value, dict) and value.get("mopd")),
                None,
            )
            if feedback is None:
                raise RuntimeError(f"No canonical MOPD trainer feedback was returned for rollout {rollout_id}.")
            if feedback.get("failure_reason"):
                raise RuntimeError(f"MOPD trainer rejected rollout {rollout_id}: {feedback['failure_reason']}.")

        num_updates_after = num_updates_before_rollout(args, rollout_id) + updates_per_rollout(args)

        if not mopd_enabled and (
            release_train
            or should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout)
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if mopd_enabled:
            feedback["driver_step_wall_seconds"] = time.perf_counter() - mopd_step_started
            if rollout_id == args.start_rollout_id:
                feedback["startup_wall_seconds"] = mopd_startup_seconds
                feedback["driver_step_wall_seconds"] += mopd_startup_seconds
            record = ray.get(rollout_manager.complete_mopd_update.remote(rollout_id, feedback))
            final_mopd_status = {
                "complete": bool(record["budget_complete"]),
                "attempted_responses": int(record["attempted_responses_after"]),
                "optimizer_updates": int(record["optimizer_updates_after"]),
                "completed_operations": int(record["operation_index"]) + 1,
            }
            save_boundary = bool(
                record.get("operation", "train") == "train" and (record["checkpoint_due"] or record["budget_complete"])
            )
            if save_boundary:
                save_started = time.perf_counter()
                _save_mopd_actor_checkpoint(args, rollout_manager, actor_model, rollout_id)
                if args.rollout_global_dataset:
                    ray.get(rollout_manager.save.remote(rollout_id))
                index_path = Path(args.save).resolve() / "mopd_checkpoint_index.json"
                entries = []
                if index_path.is_file():
                    entries = json.loads(index_path.read_text(encoding="utf-8"))
                entries.append(
                    {
                        "rollout_id": int(rollout_id),
                        "operation_index": int(record["operation_index"]),
                        "attempted_responses": int(record["attempted_responses_after"]),
                        "optimizer_updates": int(record["optimizer_updates_after"]),
                        "optimizer_step": int(record["optimizer_updates_after"]),
                        "hf_checkpoint": str(Path(args.save_hf.format(rollout_id=rollout_id)).resolve()),
                    }
                )
                _mark_and_prune_mopd_optimizer_checkpoints(args, entries)
                with (Path(args.save).parent / "checkpoint_costs.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "step": int(record["optimizer_updates_after"]),
                                "wall_seconds": time.perf_counter() - save_started,
                                "occupied_gpus": getattr(args, "mopd_occupied_gpus", None)
                                or args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus,
                            }
                        )
                        + "\n"
                    )

            attempted = int(record["attempted_responses_after"])
            evaluation_due = args.eval_interval is not None and (
                record["budget_complete"]
                or (
                    getattr(args, "mopd_profile", None)
                    and int(record["optimizer_updates_after"]) % args.eval_interval == 0
                )
                or crossed_response_milestone(
                    int(record["attempted_responses_before"]),
                    attempted,
                    mopd_eval_responses,
                )
            )
            if evaluation_due:
                _evaluate_mopd(args, rollout_manager, actor_model, rollout_id, int(record["optimizer_updates_after"]))
                last_eval_num_updates = int(record["optimizer_updates_after"])
            if record["budget_complete"]:
                break
        elif should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(
                rollout_manager.eval.remote(
                    rollout_id,
                    num_updates=num_updates_after,
                    model_version=num_updates_after,
                    eval_phase="post_update",
                )
            )
            last_eval_num_updates = num_updates_after

    if mopd_enabled:
        final_mopd_status = final_mopd_status or ray.get(rollout_manager.mopd_budget_status.remote())
        if not final_mopd_status["complete"]:
            raise RuntimeError(f"--num-rollout ended before the MOPD {args.mopd_total_steps}-step budget")
        final_num_updates = int(final_mopd_status["optimizer_updates"])
    else:
        final_num_updates = (
            last_eval_num_updates
            if args.num_rollout == 0 and last_eval_num_updates is not None
            else num_updates_before_rollout(args, args.num_rollout)
        )
    # A run whose length is not divisible by eval_interval still needs a final
    # paper-facing measurement of the final checkpoint.
    if args.eval_interval is not None and last_eval_num_updates != final_num_updates:
        final_rollout_id = max(args.start_rollout_id, last_rollout_id)
        if mopd_enabled:
            _evaluate_mopd(args, rollout_manager, actor_model, final_rollout_id, final_num_updates)
        else:
            ray.get(
                rollout_manager.eval.remote(
                    final_rollout_id,
                    num_updates=final_num_updates,
                    model_version=final_num_updates,
                    eval_phase="final",
                )
            )

    ray.get(rollout_manager.dispose.remote())
    mark_run_complete(args, final_num_updates=final_num_updates)
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    finally:
        if getattr(args, "mopd_profile", None):
            ray.shutdown()
