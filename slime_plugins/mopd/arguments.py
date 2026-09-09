"""Validation of the current paper's configurable, equal-quota OPD runs."""

from __future__ import annotations


def validate_paper_args(args):
    from .sampler import active_tasks

    tasks = active_tasks(args)
    if getattr(args, "mopd_loss", None) in {"student_topk", "topk_intersection"}:
        from .topk import student_topk_size

        student_topk_size(args)
    responses = args.mopd_responses_per_update
    slice_size = args.global_batch_size
    if responses <= 0 or responses % (len(tasks) * slice_size):
        raise ValueError("Responses per update must divide into equal domain quotas and complete backward slices.")
    if args.rollout_batch_size != responses or args.n_samples_per_prompt != 1:
        raise ValueError("Paper OPD requires one response per prompt and rollout batch size equal to responses per update.")
    if args.mopd_total_steps != args.num_rollout:
        raise ValueError("Set num-rollout and mopd-total-steps to the same optimizer-update budget.")
    if args.calculate_per_token_loss != (args.mopd_reduction != "domain_response"):
        raise ValueError("Token reductions require calculate-per-token-loss; response reduction requires sample means.")
    if args.opd_task_reward_weight != 0:
        raise ValueError("Paper OPD uses verifier rewards for observation only; set opd-task-reward-weight to zero.")
    if args.loss_type != "custom_loss" or args.custom_loss_function_path != "slime_plugins.mopd.loss.paper_loss":
        raise ValueError("Paper OPD requires the paper_loss custom loss for both sampled PG and TopK.")
    if args.use_opd or args.compute_advantages_and_returns:
        raise ValueError("Paper OPD differentiates its loss directly; disable the separate PPO/advantage machinery.")
    if args.custom_reward_post_process_path != "slime_plugins.mopd.loss.post_process_rewards":
        raise ValueError("Paper OPD requires the teacher-score postprocessor.")
    if args.optimizer != "adam" or args.use_critic:
        raise ValueError("The paper comparisons use AdamW without a critic.")
    if args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1:
        raise ValueError("Paper gradient measurements currently require PP=CP=1.")
    if args.actor_num_nodes * args.actor_num_gpus_per_node != args.tensor_model_parallel_size:
        raise ValueError("Paper gradient measurements require a single training replica (DP=1).")
    if args.tensor_model_parallel_size != 1 and not getattr(args, "mopd_skip_paper_measurements", False):
        raise ValueError("Portable parameter geometry snapshots require TP=1; use --mopd-skip-paper-measurements for TP>1.")
    if args.partial_rollout or args.dynamic_sampling_filter_path:
        raise ValueError("Equal-domain comparisons retain every attempted response; partial/dynamic filtering is unsupported.")
    if not args.opd_teacher_router_config:
        raise ValueError("Paper OPD requires the profile's teacher router.")
    if not args.save or not args.save_hf:
        raise ValueError("Paper runs need resumable optimizer checkpoints and HF checkpoints for capability evaluation.")
