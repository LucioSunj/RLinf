"""Reward helpers shared by LIBERO chunk execution paths."""

from __future__ import annotations

import torch


def mask_rewards_after_first_done(
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    """Keep rewards through the first terminal primitive and zero the rest."""

    if rewards.shape != dones.shape:
        raise ValueError(
            "LIBERO chunk rewards and dones must have identical shapes, got "
            f"{tuple(rewards.shape)} and {tuple(dones.shape)}."
        )
    if rewards.ndim != 2:
        raise ValueError("LIBERO chunk rewards must have shape [batch, actions].")
    if dones.dtype != torch.bool:
        raise TypeError("LIBERO chunk dones must use torch.bool.")
    prior_done = torch.zeros_like(dones)
    if dones.shape[1] > 1:
        prior_done[:, 1:] = dones[:, :-1].to(torch.int64).cumsum(dim=1) > 0
    return torch.where(prior_done, torch.zeros_like(rewards), rewards)
