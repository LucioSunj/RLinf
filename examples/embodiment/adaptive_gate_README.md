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
  actor.model.gate.force_mode=0

python evaluations/eval_embodied_agent.py \
  --config-path "$(pwd)/examples/embodiment/config" \
  --config-name libero_10_grpo_gate \
  runner.only_eval=true runner.ckpt_path=/path/to/gate.pt \
  actor.model.gate.force_mode=1
```

Mode `0` is always-UNCOND and mode `1` is always-IDM. Repeat with the RoboTwin
config after the LIBERO smoke.

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
