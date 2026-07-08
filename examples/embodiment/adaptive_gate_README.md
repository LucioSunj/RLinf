# Adaptive-prediction gate (RLinf side)

An RL-trained controller that, per action-chunk step, picks how much forward-
prediction compute a **frozen** fast-wam world-action model spends:
**SKIP** (reactive) / **LATENT** (few video-denoise steps) / **FULL** (full schedule).
fast-wam stays frozen; **only the gate trains**.

This complements the fast-wam side (`FastWAM/docs/adaptive_gate.md`): `WAMModeAdapter`
+ FLOPs profiler + cost. Here we add the gate policy, reward, and RL configs.

## Pieces (this repo)

| file | role |
|---|---|
| `rlinf/models/embodiment/gate_policy/gate_policy.py` | `GatePolicy` — 3-way categorical over {SKIP,LATENT,FULL} on a frozen `WAMModeAdapter`; `mode_logits` shared by BC+RL; KL-to-BC prior |
| `rlinf/models/embodiment/gate_policy/__init__.py` | `get_model` + `build_wam_adapter` (build/freeze the dual-regime WAM, wrap it); BC init + prior wiring |
| `rlinf/models/embodiment/gate_policy/obs_preprocessor.py` | env_obs → fast-wam inputs (image/proprio/text), per suite |
| `rlinf/models/embodiment/gate_policy/reward.py` | multi-component reward: success, `-λ·cost`, optional agreement, `-β·KL(π‖π_BC)`; λ/β schedules |
| `rlinf/models/embodiment/gate_policy/bc.py` | M3 BC (SFT) warm-start: oracle-label dataset + weighted-CE trainer + ckpt IO |
| `examples/embodiment/train_gate_bc.py` | M3 standalone SFT CLI (no Ray/simulator/WAM needed) |
| `rlinf/models/__init__.py` | registers `gate_policy` |
| `examples/embodiment/config/libero_10_grpo_gate.yaml` | GRPO on LIBERO-10 |
| `examples/embodiment/config/robotwin_grpo_gate.yaml` | GRPO on RoboTwin |
| `tests/unit_tests/test_gate_policy.py` | gate-policy unit tests (stub adapter) |
| `tests/unit_tests/test_gate_bc.py` | BC trainer / KL-prior / reward-KL unit tests |

Same gate works for **both** backbones (`wam.backbone_kind: joint|idm`) and is used
identically in training and rollout (one interface).

## Prerequisites

1. A **dual-regime** fast-wam checkpoint (`MetricAdaptiveFastWAMJoint` or
   `MetricAdaptiveFastWAM`) — the vanilla base checkpoint is SKIP-only. Train one
   with the dual-regime recipe (`FastWAM/docs/metric_adaptive*.md`).
2. `pip install -e FastWAM` so `import fastwam.adaptive_gate` works inside RLinf.
3. Profile the per-mode cost table (fast-wam side):
   ```bash
   cd FastWAM && python scripts/profile_wam_modes.py \
     --task libero_metric_adaptive_joint_2cam224_1e-4 --backbone-kind joint \
     --out configs/adaptive_gate/wam_cost_libero_joint.yaml
   ```
4. Set env vars in the config: `FASTWAM_CONFIGS` → `FastWAM/configs`; fill the
   `wam.ckpt` / `wam.cost_table_path` paths.

## Two integration hooks (validate on-server)

These cross worker boundaries and need a running cluster to validate:

1. **obs preprocessor injection.** The rollout must set the gate's preprocessor so
   it can turn raw env_obs into fast-wam inputs:
   ```python
   from rlinf.models.embodiment.gate_policy.obs_preprocessor import make_gate_obs_preprocessor
   policy.obs_preprocessor = make_gate_obs_preprocessor(policy.wam_adapter.model, suite="libero")
   ```
   Wire this where the rollout worker builds/holds the policy (huggingface rollout
   worker). Verify the image layout/normalization matches the fast-wam eval pipeline.

2. **cost → reward stream.** The env reward is success-only; the gate's per-step
   `mode_cost` is carried in the rollout buffer (`forward_inputs["mode_cost"]`).
   Combine at the reward-assembly point (`EnvWorker.compute_bootstrap_rewards` or the
   embodied runner) using `gate_policy.reward.gate_reward_components`, anneal λ with
   `lambda_cost_schedule`, and log each component + the **mode-usage distribution**
   separately (`env/<component>`, `rollout/mode_usage`).

3. **shape-reconcile (STEP-0 #8).** The trained action is the discrete mode (`[B,1]`,
   `action_dim=1`, `num_action_chunks=1`) while `chunk_actions` is the robot chunk.
   Confirm the categorical logprob flows through the GRPO actor unchanged; if not,
   register a small categorical `policy_loss`.

## Training the gate: ZERO supervision (default path)

The gate needs no labels, no demonstrations, no annotation — plain GRPO on
`success − λ·cost(mode)`. Stability without a warm-start comes from four
label-free mechanisms (all on by default in the gate configs):

1. `algorithm.entropy_bonus > 0` — keeps the per-state mode distribution alive;
2. `gate_reward.lambda_warmup_steps` — λ anneals 0 → λ_max (act well first,
   economize later; prevents early all-SKIP collapse);
3. small logits-head init — near-uniform mode prior at step 0;
4. `gate.explore_eps` — w.p. ε a TRAIN rollout samples a uniform-random mode.
   The mixture behavior logprob `(1−ε)·π + ε/3` is recorded as `prev_logprobs`,
   so the GRPO ratio `π_θ/μ` stays an exact importance weight. This guarantees
   each GRPO group (same reset init) keeps contrast across modes even if π
   momentarily collapses. Anneal via `GatePolicy.set_explore_eps` (runner hook)
   or leave a small constant.

```bash
bash examples/embodiment/run_embodiment.sh libero_10_grpo_gate   # that's all
```

## Optional: BC warm-start + KL prior (ablation / accelerator — OFF by default)

If you want a faster start or an "oracle vs learned" ablation, a supervised
warm-start can be built WITHOUT annotation: the "which mode was necessary
here?" labels are self-generated by comparing each mode's action against the
raw VLA dataset's ground-truth chunk (cheapest sufficient mode; see
`FastWAM/docs/adaptive_gate.md` §oracle). These labels are also the project's
offline ANALYSIS tool — their distribution over states is direct evidence of
when prediction helps, independent of any training use. The pure-RL gate above
never consumes them.

```bash
# 1. (fast-wam, GPU, heavy — once) oracle-label shards from the raw VLA dataset
cd FastWAM && python scripts/generate_gate_oracle_labels.py \
  --task libero_metric_adaptive_joint_2cam224_1e-4 --backbone-kind joint \
  --ckpt /path/to/dual_regime_joint.pt --dataset-stats /path/to/dataset_stats.json \
  --stride 20 --exec-horizon 10 --out data/gate_oracle/libero_joint

# 2. (here, minutes, standalone) BC warm-start of the SAME GatePolicy the RL uses
cd RLinf && python examples/embodiment/train_gate_bc.py \
  --labels 'FastWAM/data/gate_oracle/libero_joint/shard_*.pt' \
  --out ckpts/gate_bc_libero_joint.pt --epochs 30 --device cuda
# watch: val accuracy, per-mode recall, pred-vs-oracle mean cost

# 3. GRPO from the BC init (+ optional KL-to-BC prior, decays to 0):
#    runner.ckpt_path:                 ckpts/gate_bc_libero_joint.pt
#    actor.model.gate.bc_init_path:    ckpts/gate_bc_libero_joint.pt
#    actor.model.gate.kl_prior.enabled: True
#    gate_reward.beta_kl_prior: 0.05   # beta_kl_prior_decay_steps anneals it away
```

The reward-side `-β·KL(π‖π_BC)` keeps early exploration near the BC prior (the
embodied actor has no reference-model KL path, so the prior rides the reward
stream — exact categorical KL, computed in the rollout and carried in
`forward_inputs["kl_to_prior"]`); β decays so RL can overrule the oracle where
the environment disagrees.

## Launch stages

```bash
# 0. (fast-wam) profile cost table   -> see Prerequisites
# 0.5 oracle labels + BC warm-start  -> see M3 recipe above
# 1. GRPO on LIBERO
bash examples/embodiment/run_embodiment.sh libero_10_grpo_gate
# 2. GRPO on RoboTwin
bash examples/embodiment/run_embodiment.sh robotwin_grpo_gate
# 3. multi-setting eval (in-domain + held-out/OOD): per-setting success + mode-usage
#    (use runner.val_check_interval / evaluations/; log mode-usage per split)
```

## Collapse-prevention (configured; all label-free unless marked)

- entropy bonus on the mode distribution (`algorithm.entropy_bonus`, set > 0).
- λ-annealing (`gate_reward.lambda_warmup_steps`): act well first, then economize.
- small logits-head init (near-uniform mode prior at start) — in `GatePolicy`.
- `gate.explore_eps` uniform-mixture rollout sampling (exact behavior logprobs).
- OPTIONAL (supervised by self-generated labels; off by default): BC warm-start
  (`train_gate_bc.py`) + decaying KL-to-BC prior (`gate.kl_prior` +
  `gate_reward.beta_kl_prior`). Hook left in `reward.py` for a
  budget-constrained (E[cost] ≤ B) variant.

## Status

- M1 (fast-wam): WAMModeAdapter + profiler + cost — done.
- M2 part 1: gate policy + registration + unit tests — done.
- M2 part 2: obs preprocessor + reward module + GRPO configs (LIBERO+RoboTwin)
  + README — done; the two integration hooks above need on-server wiring.
- M3: zero-supervision default path hardened (`explore_eps`); PLUS optional
  oracle labels (fast-wam side, doubling as offline analysis) + BC warm-start +
  KL-to-BC prior + unit tests — done (code-complete; runs need data + GPU).
- Next: forced-mode smoke test (always-SKIP / always-FULL end-to-end), then M4
  (GRPO training + collapse/OOD checks; PPO config).
```
