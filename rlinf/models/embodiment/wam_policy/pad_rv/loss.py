# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD-Frozen same-chunk alignment and Gate-only policy loss."""

from __future__ import annotations

import torch

from rlinf.algorithms.advantages import FastWAMPolicyAlignment
from rlinf.algorithms.fastwam_dual_ppo import compute_gate_ppo_loss
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    WAMRoute,
)


def align_current_step_advantages(
    *,
    advantages: torch.Tensor,
    route: ChunkRouteRecord,
    emitted: GateDecisionRecord,
    loss_mask: torch.Tensor | None,
) -> FastWAMPolicyAlignment:
    """Align each PAD decision with the advantage of its executed same chunk."""

    if route.route_used.shape != emitted.next_route.shape:
        raise ValueError("PAD route and Gate records must share [time,batch] shape.")
    if advantages.shape[:2] != route.route_used.shape:
        raise ValueError("PAD advantages must begin with [time,batch].")
    valid = torch.ones_like(route.route_used, dtype=torch.bool)
    if loss_mask is not None:
        if loss_mask.shape[:2] != valid.shape:
            raise ValueError("PAD loss mask must begin with [time,batch].")
        valid &= loss_mask.bool().reshape(*valid.shape, -1).all(dim=-1)
    mismatch = emitted.valid & (
        route.route_was_forced
        | (route.route_source_chunk_ids != route.chunk_ids)
        | (route.route_used != emitted.next_route)
    )
    if bool(mismatch.any().item()):
        index = tuple(
            int(item) for item in mismatch.nonzero(as_tuple=False)[0].tolist()
        )
        raise ValueError(
            "PAD current-step decision does not control its source chunk; "
            f"first mismatch at {index}."
        )
    gate_valid = valid & emitted.valid
    scalar_advantages = advantages[..., 0]
    gate_advantages = torch.where(
        gate_valid,
        scalar_advantages,
        torch.zeros_like(scalar_advantages),
    )
    return FastWAMPolicyAlignment(
        flow_advantages=torch.zeros_like(advantages),
        flow_valid_mask=torch.zeros_like(valid),
        gate_advantages=gate_advantages,
        gate_valid_mask=gate_valid,
    )


def compute_pad_frozen_policy_loss(
    *,
    gate_logprobs: torch.Tensor,
    gate_old_logprobs: torch.Tensor,
    gate_advantages: torch.Tensor,
    gate_valid_mask: torch.Tensor,
    gate_clip_ratio_low: float,
    gate_clip_ratio_high: float,
    gate_base_probabilities: torch.Tensor | None,
    gate_behavior_probabilities: torch.Tensor | None,
    gate_entropy_coefficient: float,
    gate_loss_coefficient: float,
    selected_loss_scale: float | torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Gate PPO with no action-policy or Flow replay term."""

    if gate_loss_coefficient < 0:
        raise ValueError("PAD Gate loss coefficient must be non-negative.")
    gate_loss, metrics = compute_gate_ppo_loss(
        logprobs=gate_logprobs,
        old_logprobs=gate_old_logprobs,
        advantages=gate_advantages,
        valid_mask=gate_valid_mask,
        clip_ratio_low=gate_clip_ratio_low,
        clip_ratio_high=gate_clip_ratio_high,
        base_probabilities=gate_base_probabilities,
        behavior_probabilities=gate_behavior_probabilities,
        entropy_coefficient=gate_entropy_coefficient,
        selected_loss_scale=selected_loss_scale,
    )
    total = gate_loss_coefficient * gate_loss
    return total, {**metrics, "pad_frozen/total_policy_loss": total.detach()}


def absent_uncond_flow_metrics(
    *,
    route_used: torch.Tensor,
    valid_chunk_mask: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Report an explicitly absent Flow branch without creating a Flow loss."""

    if route_used.shape != valid_chunk_mask.shape:
        raise ValueError("PAD route and valid-chunk masks must have the same shape.")
    if valid_chunk_mask.dtype != torch.bool:
        raise TypeError("PAD valid-chunk mask must use torch.bool.")
    if route_used.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("PAD routes must use an integer dtype.")
    invalid_route = (route_used != int(WAMRoute.UNCOND)) & (
        route_used != int(WAMRoute.IDM)
    )
    if bool(invalid_route.any().item()):
        raise ValueError("PAD routes contain a value outside WAMRoute.")

    zero = reference.detach().sum() * 0.0
    sample_count = (valid_chunk_mask & (route_used == int(WAMRoute.UNCOND))).sum(
        dtype=zero.dtype
    )
    one = zero + 1.0
    return {
        "uncond_flow/policy_loss": zero,
        "uncond_flow/total_loss": zero,
        "uncond_flow/sample_count": sample_count,
        "uncond_flow/ratio": one,
        "uncond_flow/ratio_abs": zero,
        "uncond_flow/log_ratio_max_abs": zero,
        "uncond_flow/approx_kl": zero,
        "uncond_flow/clip_fraction": zero,
        "uncond_flow/entropy": zero,
    }
