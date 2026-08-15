import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, mark_run_complete
from slime.utils.metric_utils import num_updates_before_rollout, updates_per_rollout
from slime.utils.misc import should_run_periodic_action


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
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(
                rollout_manager.eval.remote(
                    rollout_id,
                    num_updates=0,
                    model_version=0,
                    eval_phase="pre_train",
                )
            )
            last_eval_num_updates = 0

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        num_updates_after = num_updates_before_rollout(args, rollout_id) + updates_per_rollout(args)

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
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

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(
                rollout_manager.eval.remote(
                    rollout_id,
                    num_updates=num_updates_after,
                    model_version=num_updates_after,
                    eval_phase="post_update",
                )
            )
            last_eval_num_updates = num_updates_after

    final_num_updates = (
        last_eval_num_updates
        if args.num_rollout == 0 and last_eval_num_updates is not None
        else num_updates_before_rollout(args, args.num_rollout)
    )
    # A run whose length is not divisible by eval_interval still needs a final
    # paper-facing measurement of the final checkpoint.
    if args.eval_interval is not None and last_eval_num_updates != final_num_updates:
        final_rollout_id = max(args.start_rollout_id, args.num_rollout - 1)
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
