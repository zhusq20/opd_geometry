import json
import os
import time
from pathlib import Path

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, mark_run_complete
from slime.utils.metric_utils import num_updates_before_rollout, updates_per_rollout
from slime.utils.misc import should_run_periodic_action
from slime_plugins.mopd.sampler import crossed_response_milestone


def _save_mopd_actor_checkpoint(args, rollout_manager, actor_model, rollout_id):
    """Save a MOPD actor without overlapping colocated rollout memory."""

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


def train(args):
    configure_logger()
    release_train = args.release_train

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

    last_eval_num_updates = None
    mopd_enabled = bool(getattr(args, "mopd_enabled", False))
    last_rollout_id = args.start_rollout_id - 1
    final_mopd_status = None
    mopd_eval_responses = (
        tuple(int(value) for value in args.mopd_eval_responses.split(",") if value)
        if mopd_enabled
        else ()
    )

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        num_updates = num_updates_before_rollout(args, args.start_rollout_id)
        ray.get(
            rollout_manager.eval.remote(
                rollout_id=args.start_rollout_id,
                num_updates=num_updates,
                model_version=num_updates,
                eval_phase="eval_only",
            )
        )
        last_eval_num_updates = num_updates

    if mopd_enabled and args.mopd_eval_on_start:
        resume_status = ray.get(rollout_manager.mopd_budget_status.remote())
        resume_rollout_id = args.start_rollout_id - 1
        ray.get(
            rollout_manager.eval.remote(
                resume_rollout_id,
                num_updates=int(resume_status["optimizer_updates"]),
                model_version=int(resume_status["optimizer_updates"]),
                eval_phase="response_clock",
            )
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

        if not mopd_enabled and (release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )):
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
        if not mopd_enabled or feedback["operation"] == "train":
            actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if mopd_enabled:
            feedback["driver_step_wall_seconds"] = time.perf_counter() - mopd_step_started
            record = ray.get(rollout_manager.complete_mopd_update.remote(rollout_id, feedback))
            final_mopd_status = {
                "complete": bool(record["budget_complete"]),
                "attempted_responses": int(record["attempted_responses_after"]),
                "optimizer_updates": int(record["optimizer_updates_after"]),
                "completed_operations": int(record["operation_index"]) + 1,
            }
            save_boundary = bool(record["checkpoint_due"] or record["budget_complete"])
            if save_boundary:
                if args.mopd_run_mode != "bank":
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
                        "run_mode": str(record["run_mode"]),
                    }
                )
                index_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
                temporary.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temporary, index_path)

            attempted = int(record["attempted_responses_after"])
            evaluation_due = (
                args.eval_interval is not None
                and args.mopd_run_mode != "bank"
                and (
                    record["budget_complete"]
                    or crossed_response_milestone(
                        int(record["attempted_responses_before"]),
                        attempted,
                        mopd_eval_responses,
                    )
                )
            )
            if evaluation_due:
                ray.get(
                    rollout_manager.eval.remote(
                        rollout_id,
                        num_updates=int(record["optimizer_updates_after"]),
                        model_version=int(record["optimizer_updates_after"]),
                        eval_phase="response_clock",
                    )
                )
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
            raise RuntimeError(
                "--num-rollout ended before the MOPD response budget; increase the launcher upper bound"
            )
        final_num_updates = int(final_mopd_status["optimizer_updates"])
    else:
        final_num_updates = (
            last_eval_num_updates
            if args.num_rollout == 0 and last_eval_num_updates is not None
            else num_updates_before_rollout(args, args.num_rollout)
        )
    # A run whose length is not divisible by eval_interval still needs a final
    # paper-facing measurement of the final checkpoint.
    if (
        args.eval_interval is not None
        and last_eval_num_updates != final_num_updates
        and not (mopd_enabled and args.mopd_run_mode == "bank")
    ):
        final_rollout_id = max(args.start_rollout_id, last_rollout_id)
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
    train(args)
