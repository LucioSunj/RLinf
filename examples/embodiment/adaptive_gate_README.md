# Binary adaptive world-model gate (RLinf side)

The controller makes one binary decision per executed action chunk:

- **UNCOND (0):** reactive FastWAM action inference from the current frame.
- **IDM (1):** complete future generation followed by the IDM action path.

The same frozen dual-regime FastWAM serves both choices. Only the small categorical
gate is optimized. There is no low-step "latent" mode: solver quality is no longer
confounded with the question of whether future prediction is useful.

## Implementation

| file | role |
|---|---|
| `rlinf/models/embodiment/gate_policy/gate_policy.py` | binary policy over UNCOND/IDM; spatial world feature + proprio + task text |
| `rlinf/models/embodiment/gate_policy/__init__.py` | builds and freezes the dual-regime IDM model; checkpoint validation |
| `rlinf/models/embodiment/gate_policy/obs_preprocessor.py` | suite image layout, normalization, cached text features and action conversion |
| `rlinf/models/embodiment/gate_policy/reward.py` | `success - lambda * relative_compute_cost` and lambda schedule |
| `rlinf/workers/env/env_worker.py` | execution-horizon slicing, reward/cost alignment and gate metrics |
| `rlinf/models/embodiment/gate_policy/bc.py` | optional oracle-label BC warm-start |
| `examples/embodiment/config/*_grpo_gate.yaml` | LIBERO and RoboTwin GRPO recipes |

## Prerequisites

1. Train an IDM dual-regime checkpoint with
   `libero_dual_regime_fused_2cam224_1e-4` or
   `robotwin_dual_regime_fused_3cam_384_1e-4`. A vanilla UNCOND-only checkpoint is
   always rejected. `allow_legacy_checkpoint` is only for an older, manually
   verified dual-regime checkpoint that predates provenance metadata.
2. Install FastWAM into the RLinf environment and set `FASTWAM_CONFIGS` to the
   absolute `FastWAM/configs` directory.
3. Precompute FastWAM text embeddings. Online text encoding is disabled by default
   so the large text encoder is not retained on the rollout GPU.
4. Profile the two modes and set `wam.cost_table_path`:

```bash
cd FastWAM
python scripts/profile_wam_modes.py \
  --task libero_dual_regime_fused_2cam224_1e-4 \
  --backbone-kind idm \
  --ckpt /path/to/dual_regime_idm_checkpoint.pt \
  --out configs/adaptive_gate/wam_cost_libero_idm.yaml
```

## Horizon contract

The WAM generates `wam.generation_horizon: 32` robot actions, but the environment
executes only the configured prefix:

- LIBERO: `wam.exec_horizon: 10`
- RoboTwin: `wam.exec_horizon: 24`

There is still exactly one categorical action and one logprob (`[B,1]`) per gate
decision. The environment returns 10 or 24 step rewards/dones; `chunk_level`
reward processing sums those into the return for that one decision. Bootstrap
dones use the same execution horizon, so trajectory tensors remain stackable.
Gate environments must expose per-slot `elapsed_steps`, and all slots in a vector
batch must share one episode clock; the worker fails closed otherwise rather than
silently overrunning a horizon or mixing asynchronous GRPO decisions.

The compute cost is charged to the action that was actually executed, including
the final action in a rollout. Lambda uses the runner's checkpointed global update
step, not a worker-local environment-call counter, so warmup is synchronized and
resume-safe.

`lambda_cost` is an episode-level compute-budget weight. Each decision contributes
`-lambda_cost * cost(mode) / ceil(max_episode_steps / exec_horizon)`, so even an
all-IDM maximum-length episode pays at most `lambda_cost`; reward scale does not
silently change with the replanning horizon. Logs include raw per-decision mean
cost and `episode_normalized_cost` (the sum of those normalized contributions).

## Zero-supervision training

The default objective is pure GRPO on `success - lambda * cost(mode)`. Stabilizers
are the entropy bonus, near-uniform initialization, lambda warmup and uniform-
mixture exploration. With `explore_eps=epsilon`, the training policy itself is
`mu=(1-epsilon) * pi + epsilon * Uniform(2)`. Each sample stores epsilon; actor
replay recomputes the same `mu`, so the probability ratio is exactly one before an
optimizer update rather than treating samples from `mu` as if they came from `pi`.

Finite groups are not guaranteed to contain both modes. Monitor these metrics:
even with the configured `epsilon=0.10`, a fully collapsed policy still produces
an all-primary-mode group with probability `0.95^8`, about 66.3%.

- `env/gate/mode_uncond_usage`, `env/gate/mode_idm_usage`
- `env/gate/mode_cost`, `env/gate/mode_entropy`
- `eval/gate/episode_total_cost`, `eval/gate/episode_normalized_cost`
- `env/gate/group_mode_diversity`, `env/gate/group_single_mode_ratio`
- `env/gate/reward_success`, `env/gate/reward_compute_penalty`
- `env/gate/reward_total`, `env/gate/lambda_cost`

Evaluation logs the two mode-usage rates, mean cost and entropy under the greedy
gate. These metrics use an active-episode mask, so decisions after first success
are not counted even when the environment continues to a fixed horizon. LIBERO-10
uses the official 700-step evaluation horizon. RoboTwin configs use separate
train/eval seed files, enable both wrist cameras, disable the base environment's
center crop, and run nine 24-action decisions for the 200-step limit (the final
decision executes only the remaining eight actions). The environment reward/done
tensors are right-padded back to width 24, preserving RLinf's fixed rollout shape
without sending actions beyond the episode limit.

```bash
cd RLinf
bash examples/embodiment/run_embodiment.sh libero_10_grpo_gate
bash examples/embodiment/run_embodiment.sh robotwin_grpo_gate
```

The configured rollout sizes are exactly divisible by their global batches:
LIBERO yields `480/10 * 8 * 64 = 24576` decisions with batch 2048. RoboTwin yields
`ceil(200/24) * 8 * 64 = 4608` decisions with batch 512.

## Evaluation-only

`rollout.model` contains the complete gate and WAM configuration rather than only
rollout overrides, so `runner.only_eval: true` does not depend on actor-side model
merging. Set `runner.ckpt_path` to the gate state dict, retain `load_wam: true`, and
provide the checkpoint, dataset stats, cached text embeddings and cost table.

RL training writes evaluation weights to
`<checkpoint>/actor/model_state_dict/full_weights.pt` and writes required provenance
beside them as `full_weights.pt.meta.json`. Keep the two files together. Loading is
fail-closed when the sidecar is missing or when its task, WAM checkpoint ID,
dataset-stats SHA, feature layout, dtype, context length, solver depth, or horizons
do not match. `gate.allow_legacy_gate_checkpoint=true` is only an escape hatch for
an old checkpoint that has been verified manually.

Before learned-gate evaluation, run both forced-mode end-to-end smoke tests. The
same setting is shared by `actor.model.gate` and the interpolated
`rollout.model.gate`:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path "$(pwd)/examples/embodiment/config" \
  --config-name libero_10_grpo_gate \
  runner.only_eval=true runner.ckpt_path=/path/to/gate.pt \
  actor.model.gate.eval_policy.kind=forced \
  actor.model.gate.eval_policy.mode=0

python evaluations/eval_embodied_agent.py \
  --config-path "$(pwd)/examples/embodiment/config" \
  --config-name libero_10_grpo_gate \
  runner.only_eval=true runner.ckpt_path=/path/to/gate.pt \
  actor.model.gate.eval_policy.kind=forced \
  actor.model.gate.eval_policy.mode=1
```

Mode `0` is always-UNCOND and mode `1` is always-IDM. Repeat with the RoboTwin
config after the LIBERO smoke.

Matched-budget controls use the same evaluation-only selector with
`episode_mixture`, `bernoulli`, `random_k`, `periodic_k`, `manifest`, or
`phase_heuristic`. For LIBERO, `max_decisions` must remain 70. Random-K and
periodic-K reserve exactly `k` slots over that full horizon; early success only
changes the compute actually spent. Materialize an auditable schedule manifest
before evaluation when a task/phase-matched allocation is prepared offline:

```bash
python examples/embodiment/build_gate_mode_manifest.py \
  --episode-manifest /path/to/libero_plus_episodes.json \
  --checkpoint /path/to/dual_regime_wam.pt \
  --kind random_k --k 35 --max-decisions 70 --seed 0 \
  --out /path/to/random_k35_modes.json
```

Every gate evaluation writes canonical JSONL under the run directory (or
`gate.eval_policy.trace_path`). Each episode record contains immutable task,
factor, level, reset and seed identity; success and success slot; the complete
70-slot reservation; modes and costs actually spent before success; and gate,
WAM, episode-manifest and mode-manifest hashes. Manifest selection fails closed
if the runtime episode manifest or WAM checkpoint hash differs.

A learned evaluation additionally records a fixed 70-slot `reference_modes`
sequence. Rollout inference continues through the registered horizon after
success solely to materialize that ex-post reference budget, while
`actual_*_before_success` remains absorbing and excludes post-success compute.
Trace construction fails instead of padding if any learned reference slot was
not actually inferred.

Build quota-matched random baselines directly from that canonical learned trace:

```bash
python examples/embodiment/build_gate_mode_manifest.py \
  --episode-manifest /path/to/libero_plus_episodes.json \
  --checkpoint /path/to/dual_regime_wam.pt \
  --reference-trace /path/to/learned_step_000000.jsonl \
  --kind reference_random_k --seed 0 \
  --out /path/to/reference_random_k.json
```

`reference_random_k` conserves each episode's exact 70-slot IDM count;
`reference_task_factor` conserves aggregate quotas within each task/factor cell;
and `reference_phase` conserves task/factor/reference-phase quotas. The latter is
explicitly labeled `reference_phase_matching`: phases come from the learned
reference trajectory and are not claimed to remain strictly matched after the
randomized branch changes the closed-loop trajectory. Generation validates the
70-slot shape, exact frozen episode set, WAM and episode-manifest hashes, every
cell quota, and uses order-independent SHA256 ranking for rank-invariant output.

One `LiberoEnv` process can instantiate only one benchmark task suite. A frozen
Plus-Full manifest may span several suites, so passing that parent directly to a
single `libero_10` config is rejected at environment construction instead of
silently omitting episodes. The E2, E5 BC-only, and E6 launchers partition the parent
by `task_suite_name`, override the suite and exact environment count for each
run, and then strictly merge the canonical JSONL shards. Every physical child
manifest validates its complete ordered subset against the parent and exposes
the parent's SHA256 as its logical `episode_manifest_sha256`. The merge refuses
missing or extra suites, missing or duplicate episodes, changed checkpoint or
selector provenance, and traces bound only to a physical partition hash.

The partition and merge contract is also available directly:

```bash
python ../scripts/adaptive_gate/plus_suite_manifest.py partition \
  --manifest /path/to/plus_full.json --out-dir /path/to/suites \
  --out-tsv /path/to/suites.tsv --materialize
python ../scripts/adaptive_gate/plus_suite_manifest.py merge-traces \
  --manifest /path/to/plus_full.json \
  --suite-trace libero_10=/path/to/libero_10.jsonl \
  --suite-trace libero_goal=/path/to/libero_goal.jsonl \
  --out /path/to/plus_full_trace.jsonl
```

Each suite's episode count must still be compatible with the selected RLinf rank
layout. Use a compatible rank count for the exact headline manifest; the runner
does not duplicate or drop frozen episodes merely to make a suite divisible.

Mechanism controls are separate from the production binary mode space. Generate
their independent latency/FLOP profile with FastWAM's
`scripts/profile_wam_controls.py`, then select exactly one evaluation-only
intervention:

```bash
python evaluations/eval_embodied_agent.py \
  --config-path "$(pwd)/examples/embodiment/config" \
  --config-name libero_10_grpo_gate \
  runner.only_eval=true runner.ckpt_path=/path/to/gate.pt \
  actor.model.gate.eval_control.kind=no_read \
  actor.model.gate.eval_control.profile_path=/abs/path/wam_controls.yaml
```

Valid values are `valid_idm`, `no_read`, `repeat_current`, `shuffled`, and
`extra_compute`. `no_read` and `extra_compute` fail closed unless the separate
profile marks them compute-matched; `extra_compute` takes its action solver-step
count from that profile. Any control configured in a training rollout raises an
error.

The shuffled-future bank has an executable capture path rather than a manual
`.pt` prerequisite. Run a frozen manifest evaluation with `valid_idm` and add:

```text
actor.model.gate.eval_control.capture_donor_dir=/abs/path/donor_capture
actor.model.gate.eval_control.wam_seed=0
```

Each decision writes an atomic, hash-recorded donor artifact next to the
canonical trace. It contains `video_latents`, canonical state identity, the
pre-treatment `task/factor/level/phase` cell, WAM and dataset-stats fingerprints,
solver/shape settings, manifest identity, and `wam_seed`. Pack the resulting
artifacts directly:

```bash
cd ../FastWAM
python scripts/build_shuffled_future_bank.py \
  --inputs '/abs/path/donor_capture/donor_*.pt' \
  --profile /abs/path/wam_controls.yaml \
  --shared-ckpt /abs/path/dual_regime_wam.pt \
  --dataset-stats /abs/path/dataset_stats.json \
  --out /abs/path/shuffled_future_bank.pt
```

Then evaluate `shuffled` with
`actor.model.gate.eval_control.donor_bank_path` and
`actor.model.gate.eval_control.expected_donor_wam_seed`. Loading fails on a
WAM, stats, solver, shape, or donor-generation-seed mismatch. Donors are selected
deterministically within the same `task/factor/level/phase` cell and exclude the
recipient state.

The staged launchers under `examples/embodiment/adaptive_gate/` encode these
contracts and support `--dry-run` plus extra arguments after `--`:

- `run_e2_future_controls_plus.sh` runs all five interventions, builds the donor
  bank, merges canonical traces and emits the E2 decision.
- `run_e3_collect_paired_states.sh` partitions a logical multi-task-suite
  training manifest, collects one physical paired-v1 per suite, and then writes
  one validated logical paired-v1 directory. The logical artifact embeds the
  exact parent manifest, copies and rechecks every snapshot, binds a composite
  source fingerprint, and creates a fresh global trajectory-grouped five-fold
  split. Missing/duplicate suites, incomplete episode coverage, cross-suite
  trajectory overlap, or WAM/stats/solver/feature mismatches fail closed;
  `run_e3_analyze_heterogeneity.sh` analyzes success and one-/three-chunk
  progress before emitting the E3 decision.
- `run_e4_train_uplift.sh` cross-fits independent helpfulness and difficulty
  scorers directly from that logical dataset; `run_e4_eval_predictability.sh`
  requires both OOF artifacts and emits the E4 decision. The uplift sidecar
  binds both the canonical inference-solver fingerprint and the logical paired
  dataset/composite fingerprint.

Every launcher begins with its setting, checkpoint lineage, scientific goal,
acceptance rule, required inputs and outputs. Later stages verify the preceding
`decision.json`; `ALLOW_FAILED_GATE=1` is the explicit audit-only bypass.

For an interior budget run, enable `gate_diagnostics.collapse` and set
`target_idm_usage`. Also preregister `LAMBDA_COST` and
`BUDGET_USAGE_TOLERANCE<=0.10`; target usage is not a display-only label. Rollout logs group return variance, zero-advantage groups and
effective samples. Three consecutive evaluations below 5% or above 95% IDM use,
with target error above 0.15, mark the seed collapsed; tracker state is saved and
validated on resume.

Every gate GRPO run also atomically updates
`<log_path>/<experiment_name>/gate_diagnostics.json`; each checkpoint contains
the same file. It records the run seed/step, nonzero-return-variance and
zero-advantage fractions, effective group/sample counts, target usage error, and
current/ever collapse state so multi-seed launchers can make decisions without
scraping TensorBoard.

After an aggregate E5 PASS, the GRPO launchers register each final Gate checkpoint
in an immutable `checkpoint_evidence.json`. The registration hashes the exact
weights and sidecar (including checkpoint step), that checkpoint's sibling
cumulative diagnostics, its resolved training run manifest, and the E5 decision.
The per-run `evidence_run_id` must also agree across the sidecar, diagnostics and
resolved config. The decision itself records the diagnostics path and SHA. E6 sweep specs list only
these registrations (`adaptive-gate-e6-sweep-v3`), and validation recomputes every
hash plus the logger-directory/step relationship. Evidence from another run with
the same seed and target budget therefore fails closed.

Evidence v2 additionally binds lambda cost, usage tolerance, and the checkpoint's
final validation usage. Across an E6 grid, lambda must strictly decrease as target
usage increases and adjacent five-seed mean validation usages must differ by at
least 0.03. Plus-Full reference and actual usage must remain within tolerance;
matched-random comparisons are checked against actual normalized compute rather
than being certified by nominal budget names alone.

## Optional BC and KL prior

Oracle labels remain an optional analysis/warm-start path; they are not required
by GRPO. They are an offline action-agreement proxy based on error against the
dataset action chunk, not a closed-loop success oracle or a performance upper
bound. Generate labels with both modes and the same execution horizon, then train
the identical binary GatePolicy with `train_gate_bc.py`.

When a BC prior is enabled, KL regularization is applied as a differentiable actor
auxiliary loss, not as a detached environment reward:

```yaml
actor:
  model:
    gate:
      bc_init_path: /path/to/gate_bc.pt
      kl_prior:
        enabled: true
        path: /path/to/gate_bc.pt
        beta: 0.05
        beta_end: 0.0
        decay_steps: 200
```

The actor's synchronized global step drives the KL decay. Keep the BC architecture
(`world_feat_dim`, `text_feat_dim`, hidden sizes and activation) identical to the
RL configuration.

## Remaining runtime validation

The CPU contract tests cover horizons, trajectory shapes, final-action cost,
reward alignment, active-eval masking and distributed GRPO group boundaries.
Periodic validation uses 496 LIBERO or 96 RoboTwin environments for clean 1/2/4/8
rank sharding; exact 500/100-trial reporting needs a separate compatible layout.
Before a long run, still perform a real
GPU two-mode smoke test (forced UNCOND and forced IDM) in each simulator, then a
short distributed rollout to validate memory use and measured cost on the target
hardware.
