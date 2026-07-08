# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Multi-component reward for the adaptive-prediction gate.

Each term is computed and returned SEPARATELY (for per-component logging); the
caller sums `total`. Terms (all configurable):
  - success         : terminal task success (sparse), from the env reward stream.
  - compute_penalty : -lambda_cost * cost(mode), cost(FULL)=1 (from WAMModeAdapter).
  - agreement       : OPTIONAL dense shaping = similarity between the chosen mode's
                      action chunk and the FULL-mode action chunk at that step
                      (reduces variance). Off by default.
  - kl_prior        : OPTIONAL M3 term = -beta * KL(pi || pi_BC) per gate decision,
                      keeping early RL near the BC warm-start (the embodied actor
                      has no reference-model KL path, so the prior rides the reward
                      stream). `kl_to_prior` is emitted by GatePolicy's rollout when
                      `gate.kl_prior.enabled`; beta DECAYS over steps (trust the
                      prior early, wean off as RL takes over).

WIRING (the one integration hook; validate on-server):
  The gate's per-step `mode_cost` is surfaced by GatePolicy.predict_action_batch in
  `result["forward_inputs"]["mode_cost"]`, so it persists in the rollout buffer.
  Combine it with the env's success reward where the reward stream is assembled
  (EnvWorker.compute_bootstrap_rewards or the embodied runner's reward assembly):

      comps = gate_reward_components(success=env_rewards, mode_cost=buf["mode_cost"],
                                     lambda_cost=lam, agreement=optional_agreement, ...)
      rewards = comps["total"]
      # log comps["success"], comps["compute_penalty"], comps["agreement"] separately

  `lam` should follow `lambda_cost_schedule(global_step, ...)` (anneal up: learn to
  act well first, then to economize).
"""
from __future__ import annotations

from typing import Optional

import torch


def gate_reward_components(
    *,
    success,
    mode_cost,
    lambda_cost: float,
    agreement=None,
    w_success: float = 1.0,
    w_agreement: float = 0.0,
    kl_to_prior=None,
    beta_kl_prior: float = 0.0,
):
    """Return {success, compute_penalty, [agreement], [kl_prior], total}. Works with
    floats or torch tensors / numpy arrays (broadcasting). Keep each component for
    logging; do NOT collapse upstream."""
    success_term = w_success * success
    compute_penalty = -float(lambda_cost) * mode_cost
    components = {"success": success_term, "compute_penalty": compute_penalty}
    total = success_term + compute_penalty
    if agreement is not None and w_agreement != 0.0:
        agreement_term = float(w_agreement) * agreement
        components["agreement"] = agreement_term
        total = total + agreement_term
    if kl_to_prior is not None and beta_kl_prior != 0.0:
        kl_term = -float(beta_kl_prior) * kl_to_prior
        components["kl_prior"] = kl_term
        total = total + kl_term
    components["total"] = total
    return components


def lambda_cost_schedule(
    step: int,
    *,
    lambda_max: float,
    warmup_steps: int,
    start: float = 0.0,
) -> float:
    """Linear anneal of the compute-penalty weight from `start` to `lambda_max`
    over `warmup_steps`, then hold. Start small so the gate first learns to act
    well, then to economize (collapse-prevention, decision #collapse)."""
    if warmup_steps <= 0:
        return float(lambda_max)
    frac = min(max(step / float(warmup_steps), 0.0), 1.0)
    return float(start) + (float(lambda_max) - float(start)) * frac


def kl_prior_schedule(
    step: int,
    *,
    beta_start: float,
    beta_end: float = 0.0,
    decay_steps: int = 0,
) -> float:
    """Linear DECAY of the KL-to-BC weight from `beta_start` to `beta_end` over
    `decay_steps`, then hold. Mirror image of `lambda_cost_schedule`: trust the
    BC prior early (when the policy would otherwise collapse or thrash), then
    wean off so RL can depart from the oracle where the env says otherwise."""
    if decay_steps <= 0:
        return float(beta_start)
    frac = min(max(step / float(decay_steps), 0.0), 1.0)
    return float(beta_start) + (float(beta_end) - float(beta_start)) * frac


def spread_mode_cost_over_reward_steps(
    *,
    mode_cost: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Align per-decision mode cost to reward shape without multiplying by chunk length.

    The embodied GRPO path with `reward_type=chunk_level` later sums rewards over the
    last dimension. FastWAM gate emits one `mode_cost` per policy decision, while the
    env returns one reward per executed robot action. Spread the scalar cost evenly
    across those env rewards so the later sum contributes exactly one cost penalty.
    """
    cost = mode_cost.to(device=rewards.device, dtype=rewards.dtype)
    while cost.ndim < rewards.ndim:
        cost = cost.unsqueeze(-1)
    if cost.shape == rewards.shape:
        return cost
    if cost.shape[:-1] == rewards.shape[:-1] and cost.shape[-1] == 1:
        return cost.expand_as(rewards) / max(int(rewards.shape[-1]), 1)
    return cost.expand_as(rewards)


def apply_gate_reward(
    *,
    rewards: torch.Tensor,
    mode_cost: torch.Tensor,
    step: int,
    lambda_cost: float,
    lambda_warmup_steps: int,
    lambda_start: float = 0.0,
    w_success: float = 1.0,
    w_agreement: float = 0.0,
    agreement: Optional[torch.Tensor] = None,
    kl_to_prior: Optional[torch.Tensor] = None,
    beta_kl_prior: float = 0.0,
    beta_kl_prior_end: float = 0.0,
    beta_kl_prior_decay_steps: int = 0,
) -> dict[str, torch.Tensor]:
    """Combine env rewards with the gate compute penalty (+ optional KL-to-BC).

    `kl_to_prior` is the per-decision KL carried in the rollout buffer
    (`forward_inputs["kl_to_prior"]`, [B,1]); like `mode_cost` it is spread over
    the env reward steps so the later chunk-level sum applies it exactly once.
    """
    lam = lambda_cost_schedule(
        step,
        lambda_max=float(lambda_cost),
        warmup_steps=int(lambda_warmup_steps),
        start=float(lambda_start),
    )
    aligned_cost = spread_mode_cost_over_reward_steps(
        mode_cost=mode_cost,
        rewards=rewards,
    )
    aligned_kl = None
    beta = 0.0
    if kl_to_prior is not None and float(beta_kl_prior) != 0.0:
        beta = kl_prior_schedule(
            step,
            beta_start=float(beta_kl_prior),
            beta_end=float(beta_kl_prior_end),
            decay_steps=int(beta_kl_prior_decay_steps),
        )
        aligned_kl = spread_mode_cost_over_reward_steps(
            mode_cost=kl_to_prior,
            rewards=rewards,
        )
    return gate_reward_components(
        success=rewards,
        mode_cost=aligned_cost,
        lambda_cost=lam,
        agreement=agreement,
        w_success=w_success,
        w_agreement=w_agreement,
        kl_to_prior=aligned_kl,
        beta_kl_prior=beta,
    )


# TODO(budget-constrained variant, clean hook — do NOT fully implement here):
#   For a target E[cost] <= B, replace the fixed-lambda penalty with a dual update
#   lambda_{t+1} = clip(lambda_t + eta * (E_hat[cost] - B), 0, inf), estimating
#   E_hat[cost] from the batch mean of `mode_cost`. Plug the updated lambda into
#   `gate_reward_components(lambda_cost=...)` each step.
def budgeted_lambda_update(*args, **kwargs):  # pragma: no cover - intentional stub
    raise NotImplementedError(
        "Budget-constrained (E[cost] <= B) dual-ascent on lambda is a documented "
        "hook; implement when needed (see TODO above)."
    )
