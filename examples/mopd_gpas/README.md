# Multi-teacher OPD: token balancing, update sparsity, supervision density

SmolLM collaborator guide: [MixSFT + PG / Top64 intersection](README_SMOLLM3_zh.md), including training without a sandbox and separate capability evaluation setup.

This example follows the current [three-study paper plan](../../../Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/MOPD_THREE_CONTRIBUTIONS_2026-09-05_zh.md). The directory name is historical; the default experiment matrix contains six conditions:

| Condition | Teachers | Loss | Reduction |
|---|---|---|---|
| `s-pg` | One routed RL teacher | Sampled-token PG | Domain response mean |
| `s-tk` | Same RL teacher | Student Top16 | Domain response mean |
| `m-pg` | All routed RL teachers | Sampled-token PG | Domain response mean |
| `m-tk-dr` | All routed RL teachers | Student Top16 | Domain response mean |
| `m-tk-dt` | All routed RL teachers | Student Top16 | Domain token mean |
| `m-tk-gt` | All routed RL teachers | Student Top16 | Global token mean |

TopK runs use `--mopd-loss student_topk` with **K=16 by default**; `--mopd-topk` supports 16 and 64. At every rollout prefix, the student selects K IDs and the teacher scores those same IDs. For full-vocabulary log-probabilities `log_p` and `log_q` on this support, the loss per prefix is:

```python
advantage = (log_p.softmax(-1) * (log_q - log_p)).detach()
loss = -(advantage * log_p).sum(-1)
```

Only the student weighting distribution is normalized within each prefix's selected set. The log-ratio uses full-vocabulary probabilities. The current student probabilities and detached advantages are refreshed at each actor forward on the saved rollout IDs. This follows [Open-MOPD's TopK advantage](https://github.com/BytedTsinghua-SIA/Open-MOPD/blob/4809a96cf85a869106ff0ff3f37d0a51e12010ae/training/verl/verl/workers/actor/dp_actor.py#L672) without PPO ratios, PPO clipping, or additional domain weighting; the reductions above remain configurable. SGLang teacher scoring uses per-chunk ID unions because its `token_ids_logprob` API accepts one ID list per request. Chunks cover 32 positions for Top16 and 8 for Top64, keeping each union at most 512 IDs, with strict position/ID checks.

Logs distinguish the signed `student_top16_normalized_logratio` (or `student_top64_normalized_logratio`), the surrogate value, and student/teacher retained mass on the student support. The log-ratio is a diagnostic, not a nonnegative KL divergence. Legacy `teacher_topk` retains the corrected teacher Top64 objective for historical runs. Use a new run ID when changing objectives or K; resuming a run with a different loss or support size is rejected.

Two profiles share this code and measurement format. `qwen3` uses Qwen3-1.7B-Base, four Qwen3-1.7B RL teachers, and the existing math/code/IF/science datasets. `smollm3` uses Open-MOPD's SmolLM3-3B **MixSFT** student, its released math/code/IF RL teachers, and Open-MOPD-Data. Revisions and dataset identities are recorded by preparation. SmolLM uses the released MixSFT weights and tokenizer, with fresh `smollm3_mixsft` asset/output directories to separate historical Base runs.

From the repository root, choose one profile in a fresh shell:

```bash
source examples/mopd_gpas/configs/smollm3.env  # or configs/qwen3.env
bash examples/mopd_gpas/run_stage.sh fetch-assets
bash examples/mopd_gpas/run_stage.sh prepare
bash examples/mopd_gpas/run_stage.sh start-teacher
bash examples/mopd_gpas/run_mopd.sh m-tk-dr
# Run the six conditions when ready:
bash examples/mopd_gpas/run_mopd_matrix.sh
bash examples/mopd_gpas/analyze_all.sh
```

For the student Top64 reverse-KL experiment, use the same profile and site settings:

```bash
bash examples/mopd_gpas/run_student_top64.sh            # m-tk64-dr: multi-teacher, domain response mean
bash examples/mopd_gpas/run_student_top64.sh s-tk64     # single teacher; MOPD_SINGLE_TASK defaults to math
bash examples/mopd_gpas/run_student_top64.sh m-tk64-dt  # multi-teacher, domain token mean
bash examples/mopd_gpas/run_student_top64.sh m-tk64-gt  # multi-teacher, global token mean
```

These conditions change K to 64 while using the same normalized detached loss above. They also work through `run_mopd.sh`. The default run ID is, for example, `m-tk64-dr-s42`, with separate checkpoints and W&B records. `DRY_RUN=1` validates the launch command without training. The default six-condition matrix remains unchanged.

Set site paths and GPU assignments in the sourced environment. `MOPD_SINGLE_TASK=math` selects the representative single teacher. Training responses are capped at **4096 tokens**, capability evaluation at **32768 tokens**, subject to the native context space remaining after the prompt. Qwen3 keeps its native 32768-token context, so its effective evaluation response budget is `min(32768, 32768 - prompt_tokens)`; SmolLM3 uses its native 65536-token context. Per-response artifacts and W&B capability metrics record the effective generation budget. Domain prompt quotas are equal; the three reductions change the loss weighting without changing those quotas. Defaults use 500 updates. Model/data roots, generated configuration, and results are separate for each profile under `local/mopd_<profile>_*` and `outputs/mopd_<profile>`.

W&B is enabled by default (`WANDB_PROJECT=iclr2027-mopd-dynamics`); use the ordinary `WANDB_API_KEY` environment authentication or `WANDB_MODE=offline`. Every scalar event is also preserved in `metrics/*.jsonl`. Native task verifiers run during OPD and populate `rollout/reward/<task>/mean`, quantiles, and pass rate. `rollout/reward_used_in_loss=0` records that these scores observe progress; the teacher distillation objective is separate. Code uses the configured SandboxFusion service and its existing preflight marker; set `SANDBOXFUSION_BASE_URL` and `M2RL_SANDBOX_PREFLIGHT_MARKER` for your deployment.

The figure inputs include response lengths, truncation/completion rates, domain token shares, optimizer updates, token exposure, and measured GPU cost. Capability figures use all three training clocks. Paper probes write long records to each run's `paper/` directory, separating raw gradients, proposed Adam updates, FP32 cumulative parameter changes, and BF16 checkpoint changes. Overlap rows retain the teacher pair, support fraction, layer, checkpoint, and prefix draw; teacher distances are matched to the same checkpoint and prefixes. Full-vocabulary probes describe local supervision; PG/Top16 training runs describe cumulative trajectories.

For a saved checkpoint, run the common-prefix measurements on an available GPU:

```bash
RUN_DIR="${MOPD_OUTPUT_ROOT}/m-tk-dr-s42"
python examples/mopd_gpas/probe_checkpoint.py \
  --snapshot "${RUN_DIR}/paper/checkpoint_step_0250.pt" \
  --manifest "${MOPD_GENERATED_DIR}/diagnostic.yaml" \
  --teachers "${MOPD_TEACHER_ROUTER_CONFIG}" \
  --output "${RUN_DIR}/paper/probe_step_0250" \
  --wandb-project "${WANDB_PROJECT}" --wandb-mode "${WANDB_MODE:-online}"
```

Repeat at the selected early/middle/final checkpoints for single- and multi-teacher runs. Each probe restores the exported FP32 master values and Adam state for every teacher/loss branch, records the exact length–gradient decomposition, and computes teacher JS on shared prefixes. TopK probes read K from the snapshot (16 for older student snapshots) and preserve teacher Top64 for snapshots marked `teacher_topk`; `--topk-loss` and `--topk` provide explicit overrides. The main overlap heatmaps/scatter use top-5% optimizer updates and retain the loss and K; all thresholds, top-1/5/10% supports, losses, layers, and repeated draws remain in the exported CSVs.

`analyze_all.sh` exports CSV tables and measured PDF/PNG figures into the report directory. Missing measurements produce no invented results. `report.json` retains run completion status, original records, and figure inventory. `supervision_comparison.csv` provides the PG/TopK/full-vocabulary measurement table with explicit loss identities. Compact paper measurements and scalar logs are included in the W&B plot-data artifact when training completes.

See [中文说明](README_zh.md). The old GPAS allocation experiments and their analyzers remain available as historical tools and are outside this six-condition matrix.

Site configuration and numerical/runtime evidence are documented in the [implementation validation record](VALIDATION_2026-09-06.md).
