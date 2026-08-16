# Optimizer × Post-training Parameter Geometry

严谨的 optimizer 比较、超参数选择和 51,200/102,400 prompt 预算判定规则见
[`OPTIMIZER_COMPARISON_PROTOCOL_zh.md`](OPTIMIZER_COMPARISON_PROTOCOL_zh.md)。
当前服务器的真实 GPU/ECC、峰值显存、耗时与截断证据见
[`BATCH_FEASIBILITY_2026-08-13_zh.md`](BATCH_FEASIBILITY_2026-08-13_zh.md)。

This experiment suite supports the paper matrix

| experimental axis | values |
| --- | --- |
| actor optimizer | AdamW (`--optimizer adam`), vanilla SGD, Muon (`muon` or layer-wise `dist_muon`) |
| post-training algorithm | GRPO, PPO, pure OPD |

It also accepts the hybrid `sft_opd`, `grpo_opd`, and `ppo_opd` conditions. The same
multi-task rollout, reward, evaluation, geometry, and storage path is used for
every cell. In PPO runs the actor uses the ablated optimizer and the critic is
held fixed to AdamW by `configs/ppo_roles.yaml`. Actor and critic checkpoints
are stored separately (`checkpoints/` and `checkpoints/critic/`) so PPO resume
restores both optimizer states without collisions.

## What is implemented

- AdamW, SGD, Muon and layer-wise distributed Muon dispatch in the Megatron
  backend. Adam is decoupled-weight-decay AdamW in the pinned Megatron version.
- The M2RL-4T study domains: math, science, instruction following, and code.
  WorkBench/Agent is deliberately excluded from downloaded training artifacts
  and launch manifests.
- Deterministic `uniform`, `proportional`, `weighted`, `round_robin`, and
  `sequential` task schedules, with independent source cursors and checkpointed
RNG state. `unit: batch` makes every optimizer step task-homogeneous, which is
useful for cross-task gradient cosine estimates.
- Rule rewards for math/science, a vendored IFEvalG evaluator, and external
  sandbox code rewards.
- Per-task SGLang teacher routing for OPD. Teachers can differ in size,
  checkpoint, and endpoint, but must share the actor's tokenizer/vocabulary
  because token-level OPD scores the actor's token IDs.
- Distributed-safe exact pre/post optimizer geometry on every attempted update:
  raw/clipped gradients, optimizer direction, FP32 intended data/WD update,
  model-dtype realized update, displacement, dots/cosines, ULP realization,
  and actual optimizer-branch energy shares. Muon matrix versus Adam fallback
  comes from optimizer membership, never tensor dimensionality. CountSketch,
  coordinate histograms/support windows, and fixed-matrix spectral diagnostics
  run only at the configured low-frequency cadence; every approximation is
  named with `_sketch`.
- Per-task evaluation curves, max-so-far forgetting, and backward transfer.
- A single-cell launcher, full matrix launcher, validation command, and offline
  geometry analyzer.

For the Qwen3-1.7B math/code/science single-task study (including Qwen3-8B versus
Qwen3-4B-Thinking-2507 teachers and a true mixed-batch SFT+OPD loss), follow
[`SINGLE_TASK_zh.md`](SINGLE_TASK_zh.md). The older multi-task instructions
below remain valid.

The current dataset-epoch pure-OPD paper matrix is nine standalone
`run_opd_{math,code,science}_{adamw,sgd,muon}.sh` cells. Each cell fixes seed 42,
the `opd64x1` training profile (64 distinct prompts, one response each), one
complete task epoch with an exact partial tail, evaluation every 50 optimizer updates, a
task-specific eval response cap (32,768 for Math and 16,384 for Code/Science),
and a separate global eval concurrency limit of 48. Training prompt/response
caps are 2,048/4,096; checkpoints are every 80 updates plus the final tail, and the training
engine max-running ceiling is 72. Every OPD variant explicitly renders Qwen3
with `enable_thinking=false`; the teacher performs scoring only on the exact
student token IDs. The OPD advantage uses one student and teacher scalar
log-prob per sampled response token and rejects full-vocabulary targets.
Pure OPD Math evaluation defaults to the pinned official MATH-500 test split;
plain GRPO/PPO Math evaluation always uses pinned AIME'24 plus MATH-500. Both
paths use a 32,768-token eval response cap. `MATH_EVAL_DATASETS` can still
select AIME'24, MATH-500, or both for explicitly customized OPD evaluation.
Single-task preparation removes the five training rows that contain four
MATH-500 problems, preserving all 500 benchmark rows while keeping every Math
eval mode on the same disjoint training view.

## 1. Environment

Use this clean repository rather than the older M2RL Slime fork. Rebuild the
environment after this change:

```bash
bash build_conda.sh
```

The requirements pin NVIDIA Emerging-Optimizers `v0.1.0`, which is required by
Megatron Muon, and install the fixed IFEvalG/NLTK dependencies. The Dockerfile
contains the same additions. Muon currently requires BF16 or FP32 and disables
Megatron's ordinary distributed optimizer and gradient/parameter overlap. The
`dist_muon` option uses Megatron's layer-wise distributed wrapper; it is not the
ordinary ZeRO optimizer.

Prepare the model checkpoint as in the normal Slime Qwen3 guide. For example,
the launch defaults expect:

- Hugging Face checkpoint: `/root/Qwen3-4B`
- Megatron checkpoint: `/root/Qwen3-4B_torch_dist`
- model definition: `scripts/models/qwen3-4B.sh`

The Megatron checkpoint must contain `latest_checkpointed_iteration.txt`.

## 2. Prepare M2RL data

Download, restore, and convert the official M2RL/Nemotron blend in one command:

```bash
bash examples/optimizer_geometry/prepare_m2rl_dataset.sh
```

By default this writes `data/m2rl/train/{math,science,if,code}.jsonl`,
`data/m2rl/train/multitask_manifest.yaml`, and `dataset_info.json` with row
counts, byte sizes, and SHA-256 checksums. It preserves unit tests, instruction
constraints, labels, and original dataset IDs. Agent rows are never written to
the processed directory. Because the official source blend is one monolithic
JSONL, it is downloaded transiently and removed after successful four-task
conversion; set `CLEAN_SOURCE_BLEND=0` only when the mixed raw source must be
retained. Override the root with `M2RL_DATA_ROOT=/data/m2rl`.

If an already restored `train_complete.jsonl` exists, the converter can be run
directly; the explicit task list is also its default:

```bash
python examples/optimizer_geometry/prepare_m2rl_data.py \
  --input /data/Nemotron-3-Nano-RL-Training-Blend/train_complete.jsonl \
  --output-dir /data/m2rl/train \
  --tasks math science if code \
  --sampling uniform --sampling-unit batch --seed 42
```

For simultaneous multi-task transfer, use `uniform`, `weighted`, or
`proportional`. For the forgetting experiment use a curriculum such as:

```yaml
sampling:
  strategy: sequential
  unit: batch
  seed: 42
  repeat: true
sources:
  - name: math
    path: math.jsonl
    rm_type: deepscaler
    phase_samples: 1024
  - name: code
    path: code.jsonl
    rm_type: unit_test
    phase_samples: 1024
```

`phase_samples` counts prompts, not generated responses. With `unit: batch`, a
phase boundary is rounded up to keep the boundary batch task-homogeneous. The sampler state and
each source's shuffle/offset are saved under the rollout checkpoint, so resume
does not restart a task phase. The launcher overrides the manifest's task seed
with `SEED` so replicates 42--46 get distinct but optimizer/algorithm-matched
task stream.

## 3. Configure rewards and services

Copy and edit `configs/rewards.example.yaml`. Math and science are local rule
rewards. IFEvalG is local. Code is sent to the configured sandbox using:

```json
{
  "code": "...",
  "stdin": "...",
  "language": "python",
  "compile_timeout": 5,
  "run_timeout": 10,
  "memory_limit_MB": 4096
}
```

The reward config option is `memory_limit_mb`; the JSON above shows the actual
SandboxFusion request field. The expected response contains
`status=Success` and `run_result.stdout`. Never point this setting at an
unsandboxed general-purpose execution service.

On a Linux cgroup-v2 Docker host, first build the audited derivative image. The
script validates the patch against the pinned base, builds the final image
twice, and requires either identical image IDs or identical normalized runtime
fingerprints (ignoring only Docker's atime/ctime and directory-mtime metadata).
It then pins the first immutable local image ID or an optional pushed repository
digest. Start the localhost-only profile and export the attested endpoint before
any code training/evaluation cell:

```bash
export SANDBOX_STATE="${SANDBOX_STATE:-$HOME/.local/state/slime-opd-geometry/sandboxfusion}"
mkdir -p "$SANDBOX_STATE"
chmod 700 "$SANDBOX_STATE"

SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh
SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
SANDBOX_PREFLIGHT_MARKER="$SANDBOX_STATE/sandboxfusion_preflight.json" \
  bash examples/optimizer_geometry/start_sandboxfusion.sh
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
```

Bind-mount `SANDBOX_STATE` read-only at `/workspace/sandboxfusion-state` in the
training container, then set
`M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json`.

The patched `lite` runner supports cgroup v2 directly and gives each execution
CPU, memory, and PID controls in a dedicated delegated subtree. Startup probes
execution, host/service-filesystem separation, per-request namespaces, network
denial, a read-only cgroup leaf, actual memory enforcement, privilege dropping,
an additional untrusted-code seccomp filter, user-namespace denial, timeouts,
and teardown, and refuses `isolation=none`. Per-request limits are nested below
a default 32-GiB/4096-PID aggregate cgroup limit. A stale success marker is
invalidated before deployment; any startup/probe failure stops the replacement
service. It requires host Docker access and a narrowly allowlisted set of Linux
capabilities for mounts and namespaces; it explicitly refuses a privileged
service container. The API remains on an internal network and a read-only,
capability-free fixed-upstream relay provides the localhost-only endpoint on
Docker versions that discard published ports for internal networks. The
training container needs neither a Docker socket nor namespace privileges, so
an administrator must build/start this profile on the host; launchers
intentionally fail closed until a valid marker exists. See
`SANDBOXFUSION_CGROUP2_zh.md` for the full host and training-container commands.

For OPD, copy `configs/teachers.example.yaml` and start the listed SGLang
teachers before training. Each endpoint must implement `/generate` with
`return_logprob=true` and use the same token-ID mapping as the actor. Routing
checks, in order, the sample's explicit
`teacher`, task name, source name, and reward type, then an optional `default`.

Pure `opd` assigns zero scalar task reward and learns only from reverse KL to
the teacher. `grpo_opd` and `ppo_opd` combine the task reward with OPD; adjust
`OPD_TASK_REWARD_WEIGHT` and `OPD_KL_COEF` independently.

## 4. Evaluation and forgetting

Copy `configs/eval_tasks.example.yaml`, prepare held-out JSONL files with the
same schemas, and set:

```bash
export M2RL_EVAL_DIR=/data/m2rl_eval
export EVAL_CONFIG=$PWD/examples/optimizer_geometry/configs/eval_tasks.example.yaml
```

At every eval point, `forgetting/metrics.jsonl` stores, for task `i` at time
`t`:

- `score[i,t]`: current mean reward;
- `forgetting[i,t] = max_{s<=t} score[i,s] - score[i,t]`;
- `backward_transfer[i,t] = score[i,t] - score[i,0]`.

The initial evaluation should not be skipped: it defines the backward-transfer
baseline. `forgetting/state.json` persists baseline and max-so-far values.

## 5. Run one cell

```bash
export OUTPUT_ROOT=/data/optimizer_geometry_runs

OPTIMIZER=muon ALGORITHM=grpo SEED=42 \
  bash examples/optimizer_geometry/run_m2rl_4t.sh
```

The preset resolves the local four-task manifest and project checkpoint paths.
Override `DATA_MANIFEST`, `HF_CHECKPOINT`, `LOAD_CHECKPOINT`, or
`MODEL_CONFIG` when those artifacts live elsewhere. OPD conditions additionally
require `TEACHER_CONFIG` and live teacher endpoints.

On the current experiment host, physical GPU 0 has an uncorrectable ECC error.
The Qwen3 preset checks aggregate/volatile ECC plus free memory before every
cell. One measured healthy allocation for a four-GPU student is:

```bash
export CUDA_VISIBLE_DEVICES=1,4,6,8
export NUM_GPUS=4
```

Inside the process, `cuda:0` then maps to physical GPU 1. Keep the same mask
for validation, conversion, and all matrix cells so device assignment is
comparable.

The launcher defaults to the latency-oriented `responsive16` starting profile:

```bash
BATCH_PROFILE=responsive16
ADAMW_LR=2.5e-7
ADAM_BETA2=0.9987381276
SGD_LR=2.5e-2            # GRPO; 2.5e-3 for OPD/PPO
SGD_MOMENTUM=0
MUON_LR=2.5e-7
MUON_EXTRA_SCALE_FACTOR=0.2
WEIGHT_DECAY=0
GEOMETRY_INTERVAL=16
GEOMETRY_PROJECTION_DIM=256
ROLLOUT_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=4
GLOBAL_BATCH_SIZE=64
MAX_RESPONSE_LEN=8192    # plain GRPO/PPO training; frozen OPD uses 4096
MAX_TOKENS_PER_GPU=10240
EVAL_MAX_RESPONSE_LEN=32768  # Math GRPO/PPO evaluation only
SGLANG_MAX_RUNNING_REQUESTS=12  # sized for the longer 32k Math eval phase
APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
NUM_ROLLOUT=3200
FRESH_START=1
```

For the generic fixed-budget profile this is 51,200 prompts, 204,800 responses, and
exactly 3,200 optimizer updates. Train and rollout scalars are logged after
every update, so W&B feedback no longer waits for a 256-prompt rollout.
The equality `GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT` is
checked by default. Weight decay is zero in the primary causal comparison so
that the optimizer-specific learning rates do not also induce radically
different per-step parameter shrinkage. AdamW uses beta1 0.9 and a batch-scaled
beta2 that preserves the reference recipe's second-moment half-life in sampled
responses; Muon uses
momentum 0.95, five Newton--Schulz steps and the Scalable-Muon 0.2 spectral
factor; SGD is the vanilla (zero-momentum) method studied in the direct RL/OPD
comparisons. Select OPD strength and optimizer-specific LR with equal-budget,
disjoint tuning runs before freezing the primary matrix; see
[`OPTIMIZER_COMPARISON_PROTOCOL_zh.md`](OPTIMIZER_COMPARISON_PROTOCOL_zh.md)
for the candidates, stability exclusions, and one-standard-error rule.

Frozen `run_opd_*` does not use that 51,200-prompt default: it exports
`BATCH_PROFILE=opd64x1`, `N_SAMPLES_PER_PROMPT=1`, and `NUM_EPOCH=1`.
After the retained 2,048-token prompt filter, Math/Code/Science train
22,050/19,125/19,668 prompts and take 345/299/308 updates; their final batches
contain 34/53/20 prompts. The Math count is after removing the five raw rows
that overlap MATH-500.

The single-task reward-RL entry point `run_single_task_rl.sh` and the
single-task factorial launcher `run_single_task_matrix.sh` also override the
generic fixed-prompt budget with `NUM_EPOCH=1`, include the final partial
batch, and admit only seed 42. They retain their selected batch profile (the
RL entry point defaults to `responsive16` with four responses per prompt), so
their update counts are derived from each task's usable prompt count rather
than fixed at 3,200. Plain Math GRPO/PPO trains with an 8,192-token rollout cap,
evaluates the pinned AIME'24 plus MATH-500 config with a separate 32,768-token
cap, disables Qwen3 thinking, and records both caps in the run name.

The batch profiles are deliberately explicit:

| profile | prompts/full update | responses/full update | run length | intended use |
| --- | ---: | ---: | ---: | --- |
| `opd64x1` | 64 | 64 (`n=1`) | `ceil(usable_rows/64)` | frozen pure-OPD cells |
| `responsive16` | 16 | 64 (`n=4`) | 3,200 for 51,200 prompts | generic fixed-budget/tuning runs |
| `responsive8` | 8 | 32 (`n=4`) | 6,400 for 51,200 prompts | marginally faster feedback/diagnostics |
| `reference256` | 256 | 1,024 (`n=4`) | 200 for 51,200 prompts | reproduce the original M2RL-derived recipe and feasibility runs |

Use, for example, `BATCH_PROFILE=responsive8`; do not override only
`ROLLOUT_BATCH_SIZE`, because LR, Adam beta2, PPO critic settings, run length,
and expensive logging cadences must move together. On the measured 8192-token
GRPO smoke runs, the 256-prompt reference took 1,539.7 seconds/update, while
`responsive16` took 199.0 seconds (3.32 minutes) and `responsive8` took 190.4
seconds (3.17 minutes). Thus b8 bought only 4.3% lower latency while doubling
the updates needed for the same prompt budget. A one-step extrapolation gives
roughly 7.37 days for b16 and 14.10 days for b8; this excludes startup,
evaluation, checkpointing, OPD teacher work, and policy-dependent response
length changes. Use b16 for legacy fixed-budget runs and b8 for short diagnostics. The
first W&B point also waits for roughly two minutes of model/Ray startup on this
host; subsequent train/rollout scalars arrive after every update.

Use `DRY_RUN=1` to validate paths/configuration and print the exact command
without starting Ray. Use an existing cluster with `RAY_ADDRESS=...`; otherwise
the launcher starts a local Ray head and only stops that head on exit. It does
not `pkill` unrelated Python/SGLang processes.

### Qwen3-1.7B student with selectable Qwen3 teachers

Use the dedicated preset when the student is Qwen3-1.7B. `TEACHER` accepts
`qwen3-8b` (default) and `qwen3-4b-thinking-2507`:

```bash
export DATA_MANIFEST=/data/m2rl/train/multitask_manifest.yaml
export REWARD_CONFIG=$PWD/examples/optimizer_geometry/configs/rewards.example.yaml
export OUTPUT_ROOT=/data/optimizer_geometry_runs
export AVAILABLE_CUDA_DEVICES=1,4,6,8,7

ALGORITHM=opd OPTIMIZER=adamw \
  bash examples/optimizer_geometry/run-qwen3-1.7B-student-teacher.sh
```

The preset looks for the student Hugging Face and `torch_dist` checkpoints in
the shared `../../checkpoints` directory and for the teacher in
`../checkpoints/Qwen3-8B` or `../checkpoints/Qwen3-4B-Thinking-2507`, relative to the repository. Paths remain
overrideable with `HF_CHECKPOINT`, `LOAD_CHECKPOINT`, and
`TEACHER_MODEL_PATH`. The teacher stays in Hugging Face format because it is
served by SGLang; do not pass it through `--opd-teacher-load`, whose Megatron
mode requires the teacher and student to have the same architecture.

Within the explicitly supplied available-device list, the first four GPUs are
exposed to the colocated student train/rollout job and the next GPU is exposed
only to the teacher. Override the split when needed:

```bash
AVAILABLE_CUDA_DEVICES=1,4,6,8,7 \
TRAIN_GPU_COUNT=4 TEACHER_TP_SIZE=1 \
  bash examples/optimizer_geometry/run-qwen3-1.7B-student-teacher.sh
```

Alternatively set `TRAIN_CUDA_VISIBLE_DEVICES` and
`TEACHER_CUDA_VISIBLE_DEVICES` directly. `DRY_RUN=1` prints the expanded
student command without loading either model. To reuse a separately managed
teacher, set `START_TEACHER=0`, `OPD_TEACHER_URL`, and optionally
`TEACHER_HEALTH_URL`.

Run the complete matrix through the same preset with:

```bash
AVAILABLE_CUDA_DEVICES=1,4,6,8,7 \
EXPERIMENT_LAUNCHER=$PWD/examples/optimizer_geometry/run-qwen3-1.7B-student-teacher.sh \
RUN_NAME_PREFIX=qwen3_1.7b_student_8b_teacher_ \
  bash examples/optimizer_geometry/run_matrix.sh
```

For checkpoint resume, point `LOAD_CHECKPOINT` at the run's `checkpoints`
directory, reuse the same `OUTPUT_ROOT` and `RUN_NAME`, and set
`FRESH_START=0`. The geometry baseline and multi-task sampler state will then
continue from their persisted files. The first resumed observation must be
strictly newer than the last committed observation. The forgetting logger uses
`state.json` as its commit boundary and automatically archives a JSONL tail
left by an interrupted state write under `forgetting/recovery/` before retrying
that observation. Fully committed forgetting or geometry observations beyond
the restored checkpoint still require restoring the matching checkpoint or
archiving/truncating the stale records rather than mixing two trajectories.

The repaired Math SGD run has a provenance snapshot from before the recovery
changes. Resume it with the original GPU/service settings plus:

```bash
FRESH_START=0 ALLOW_SOURCE_CHANGE_ON_RESUME=1 \
  bash examples/optimizer_geometry/run_opd_math_sgd.sh
```

`ALLOW_SOURCE_CHANGE_ON_RESUME=1` acknowledges the reviewed recovery and Ray
lifecycle changes; it does not bypass checkpoint or persisted-state validation.

## 6. Run the matrix

The generic matrix launcher defaults to 3 optimizers × 3 algorithms × 5 seeds
(42--46):

```bash
bash examples/optimizer_geometry/run_matrix.sh
```

Subsets are space-separated:

```bash
OPTIMIZERS="adamw sgd muon" \
ALGORITHMS="grpo ppo opd grpo_opd ppo_opd" \
SEEDS="42 43 44 45 46" \
  bash examples/optimizer_geometry/run_matrix.sh
```

For a defensible comparison, keep model initialization, rollout samples,
token budget, batch construction, clipping, weight decay, and seeds fixed.
The default optimizer-specific learning rates are literature starting points;
freeze selected values only after equal-budget tuning on a non-reporting split,
and never select them using results from the reported tasks. Muon updates 2-D non-embedding weights with Muon and
uses AdamW for embeddings, output, norms and biases; report this explicitly
rather than describing the condition as pure Muon over every parameter.

## 7. Geometry records

Each run writes:

```text
RUN_DIR/
  provenance/
    run_manifest.json
    source_snapshot.tar.gz
    inputs/                  # exact data/reward/eval/teacher/role config copies
  metrics/{rollout,train,eval,geometry,forgetting}.jsonl
  geometry/rollout/metrics.jsonl   # exact valid-token OPD/RL distributions
  geometry/rollout/samples/*.jsonl # prompt/response/reward/version per training sample
  geometry/actor/
    initial_projection.pt
    exact_reference/rank_*.pt
    support_state/rank_*.pt
    metrics.jsonl
    vectors/rollout_...pt
  forgetting/
    state.json
    metrics.jsonl
  checkpoints/
    critic/                 # PPO only
  eval_artifacts/
    index.jsonl
    <dataset>/*.jsonl       # response/label/reward/status/model version
  wandb_run_id.txt
  wandb/
  run_complete.json
```

`validate_run_artifacts.py` rejects incomplete update/eval/checkpoint/geometry
records, corrupt per-sample counts or SHA-256, inconsistent W&B identities, and
a completed run that also carries a failure marker. `paper_statistics.py`
produces run/aggregate/paired-effect CSV, LaTeX tables, and PNG/PDF learning
curves. Its run rows include final/best/fixed-grid AUC, failed/skipped updates,
clip fraction, throughput, recorded observation time, and the final optimizer
update/prompt/effective-token axes. Analyses that require same-checkpoint probe
backward passes or checkpoint-clone interventions are explicitly marked
unavailable in `analysis_manifest.json`; online task centroids are never used
as substitutes.

For every attempted optimizer step (with failed/skipped events retained), scalar records contain:

- exact distributed norms/RMS/Linf/zero rates for weights, raw/clipped
  gradients, optimizer directions, intended data/WD updates, realized updates,
  and displacement;
- exact required dots, cosines, ratios, branch energy shares, and FP32-to-model
  dtype realization/ULP metrics;
- low-frequency CountSketch vectors, coordinate histograms/support windows, and
  fixed-matrix diagnostics, all explicitly carrying `_sketch` where approximate;
- cosine of gradient–update, weight–update, gradient–displacement, and
  update–displacement;
- pre/post weight-norm drift, update/gradient ratio, descent alignment, and an
  effective step-size estimate `-<g, update>/||g||^2`;
- optimizer, base advantage estimator, OPD flag, role, rollout/step IDs, and
  the exact task counts in that training step.

`initial_projection.pt` is reused on resume, so displacement retains the
original reference point. Its signature includes the optimizer/algorithm
hyperparameters, dataset/seeds, parameter selection, and parallel topology, so
an incompatible run cannot silently reuse it. Un-suffixed vector norms, dots,
cosines and FP32-to-model-dtype realization metrics are exact distributed
reductions. CountSketch is fixed by parameter name, rank and seed, additive
across tensor/pipeline shards, and stores O(projection_dim) state; its derived
fields carry `_sketch`. Fixed log2 coordinate histograms and deterministic
coordinate samples likewise remain bounded-memory and explicitly carry
`_sketch`.

Exact geometry scans touch every selected optimizer-owned coordinate on every
update, while CountSketch, distribution histograms, support persistence and
sampled-matrix diagnostics run every 16 updates by default (256 prompts in the
reference configuration). Rollout OPD/RL records gather only valid CP-owned
response scalars; the rollout process also atomically saves one deterministic
per-sample JSONL file per rollout for offline reward/error analysis. Normal
training pays none of this serialization, geometry, collective, or file-I/O
cost when `--geometry-output-dir` is absent. Each record stores its observation
wall time. Observation IDs/cadence resume from the durable log, while the
persisted rollout/step frontier rejects checkpoint replay; baselines are likewise
resume-stable even when different rollouts contain different update counts.
Parameter-gather overlap is
intentionally rejected while geometry is enabled so the post-step snapshot
cannot read stale sharded parameters.

Flatten records and calculate adjacent-step plus task-centroid gradient cosine:

```bash
python examples/optimizer_geometry/analyze_geometry.py \
  /data/optimizer_geometry_runs/grpo_adamw_seed42/geometry/actor \
  /data/optimizer_geometry_runs/grpo_muon_seed42/geometry/actor \
  --output-prefix /data/analysis/grpo_geometry
```

This creates `grpo_geometry.csv` and
`grpo_geometry.task_cosines.json`. These CountSketch-derived cosine fields carry
the required `_sketch` suffix. Adjacent-step and task-centroid cosines are
kept separate per input run because different seeds use different projection
coordinates. Task-centroid cosines are only assigned for task-homogeneous
steps; use `sampling.unit: batch` for that analysis.
They compare gradients observed at the successive parameter states visited by
training; they are not same-checkpoint probe gradients. If the paper needs the
latter causal control, evaluate a fixed probe batch from every task at each
saved checkpoint and keep that analysis separate from the online records.
