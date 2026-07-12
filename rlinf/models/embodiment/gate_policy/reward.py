# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Multi-component reward for the adaptive-prediction gate.

Each term is computed and returned SEPARATELY (for per-component logging); the
caller sums `total`. Terms (all configurable):
  - success         : terminal task success (sparse), from the env reward stream.
  - compute_penalty : -lambda_cost * normalized_cost(mode), with the EnvWorker
                      dividing relative mode cost by maximum decisions per episode.

The optional BC-prior KL is not part of this reward. GatePolicy computes an exact,
differentiable KL and EmbodiedFSDPActor adds it directly to the actor objective.

WIRING:
  The gate's per-decision `mode_cost` is surfaced by
  GatePolicy.predict_action_batch in `result["forward_inputs"]["mode_cost"]` and
  retained in the rollout buffer. EnvWorker aligns that decision with the reward
  produced by its executed robot chunk, normalizes it by the maximum decisions per
  episode, and applies:

      comps = apply_gate_reward(rewards=env_rewards, mode_cost=normalized_cost,
                                step=global_step, ...)
      rewards = comps["total"]

  `apply_gate_reward` uses `lambda_cost_schedule(global_step, ...)`, so the gate
  learns task performance before the compute penalty reaches full strength.
"""
from __future__ import annotations

import torch


def gate_reward_components(
    *,
    success,
    mode_cost,
    lambda_cost: float,
    w_success: float = 1.0,
):
    """Return {success, compute_penalty, total}. Works with
    floats or torch tensors / numpy arrays (broadcasting). Keep each component for
    logging; do NOT collapse upstream."""
    success_term = w_success * success
    compute_penalty = -float(lambda_cost) * mode_cost
    components = {"success": success_term, "compute_penalty": compute_penalty}
    total = success_term + compute_penalty
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
) -> dict[str, torch.Tensor]:
    """Combine environment rewards with one compute penalty per gate decision."""
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
    return gate_reward_components(
        success=rewards,
        mode_cost=aligned_cost,
        lambda_cost=lam,
        w_success=w_success,
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
