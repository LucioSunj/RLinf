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
| `rlinf/models/embodiment/gate_policy/gate_policy.py` | `GatePolicy` — 3-way categorical over {SKIP,LATENT,FULL} on a frozen `WAMModeAdapter` |
| `rlinf/models/embodiment/gate_policy/__init__.py` | `get_model` + `build_wam_adapter` (build/freeze the dual-regime WAM, wrap it) |
| `rlinf/models/embodiment/gate_policy/obs_preprocessor.py` | env_obs → fast-wam inputs (image/proprio/text), per suite |
| `rlinf/models/embodiment/gate_policy/reward.py` | multi-component reward: success, `-λ·cost`, optional agreement; λ schedule |
| `rlinf/models/__init__.py` | registers `gate_policy` |
| `examples/embodiment/config/libero_10_grpo_gate.yaml` | GRPO on LIBERO-10 |
| `examples/embodiment/config/robotwin_grpo_gate.yaml` | GRPO on RoboTwin |
| `tests/unit_tests/test_gate_policy.py` | gate-policy unit tests (stub adapter) |

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

## Launch stages

```bash
# 0. (fast-wam) profile cost table   -> see Prerequisites
# 1. GRPO on LIBERO
bash examples/embodiment/run_embodiment.sh libero_10_grpo_gate
# 2. GRPO on RoboTwin
bash examples/embodiment/run_embodiment.sh robotwin_grpo_gate
# 3. multi-setting eval (in-domain + held-out/OOD): per-setting success + mode-usage
#    (use runner.val_check_interval / evaluations/; log mode-usage per split)
```

## Collapse-prevention (configured / planned)

- entropy bonus on the mode distribution (`algorithm.entropy_bonus`, set > 0).
- λ-annealing (`gate_reward.lambda_warmup_steps`): act well first, then economize.
- small logits-head init (near-uniform mode prior at start) — in `GatePolicy`.
- **M3 (planned):** oracle-label generation + BC warm-start (`--bc-warmstart`) +
  KL-to-BC-prior regularizer. Hook left in `reward.py` for a budget-constrained
  (E[cost] ≤ B) variant.

## Status

- M1 (fast-wam): WAMModeAdapter + profiler + cost — done.
- M2 part 1: gate policy + registration + unit tests — done.
- M2 part 2 (this): obs preprocessor + reward module + GRPO configs (LIBERO+RoboTwin)
  + README — done; the two integration hooks above need on-server wiring.
- Next: forced-mode smoke test (always-SKIP / always-FULL end-to-end), then M3 (BC
  warm-start), then M4 (GRPO training + collapse/OOD checks; PPO config).
```
