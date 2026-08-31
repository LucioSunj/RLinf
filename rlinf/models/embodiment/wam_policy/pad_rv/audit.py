# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD-Frozen rollout audits for condition-only, non-Action-K/V replay."""

from __future__ import annotations

import torch

from rlinf.algorithms.advantages import FastWAMRolloutStateAudit
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    WAMRoute,
)
from rlinf.utils.checkpoint_state import checkpoint_state_sha256


def _chunk_mask(
    value: torch.Tensor | None,
    *,
    shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if value.dtype != torch.bool or value.shape[: len(shape)] != shape:
        raise ValueError("PAD valid_mask must be boolean and start with route shape.")
    if value.ndim == len(shape):
        return value
    return value.reshape(*shape, -1).any(dim=-1)


def _probability_summary(
    value: torch.Tensor,
    *,
    eligible: torch.Tensor,
) -> dict[str, float]:
    selected = value[eligible].to(torch.float64)
    if not bool(torch.isfinite(selected).all().item()):
        raise ValueError("PAD eligible Gate probability contains non-finite values.")
    quantiles = torch.quantile(
        selected,
        torch.tensor([0.10, 0.50, 0.90], dtype=torch.float64, device=selected.device),
    )
    centered = selected - selected.mean()
    second_moment = centered.square().mean()
    if float(second_moment.item()) == 0.0:
        bimodality_score = 0.0
    else:
        third_moment = centered.pow(3).mean()
        fourth_moment = centered.pow(4).mean()
        pearson = (third_moment.square() / second_moment.pow(3) + 1.0) / (
            fourth_moment / second_moment.square()
        )
        bimodality_score = float((pearson * (quantiles[2] - quantiles[0])).item())
    return {
        "minimum": float(selected.min().item()),
        "maximum": float(selected.max().item()),
        "mean": float(selected.mean().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
        "bimodality_score": bimodality_score,
        "outside": float(
            ((selected < 0.2) | (selected > 0.8)).to(torch.float64).mean().item()
        ),
    }


def summarize_pad_frozen_rollout_state(
    *,
    route: ChunkRouteRecord,
    emitted: GateDecisionRecord,
    eligible_gate_mask: torch.Tensor,
    valid_mask: torch.Tensor | None,
    kv_replay_backend: str,
    max_bytes_per_sample: int | None,
) -> FastWAMRolloutStateAudit:
    """Summarize current-step routing while proving Action K/V is absent."""

    del max_bytes_per_sample
    if str(kv_replay_backend).strip().lower() != "condition":
        raise ValueError("PAD-Frozen rollout audit requires condition replay.")
    if emitted.kv_metadata is not None:
        raise ValueError(
            "PAD-Frozen condition replay cannot carry Action K/V metadata."
        )
    if len(route.shape) != 2 or emitted.shape != route.shape:
        raise ValueError("PAD rollout audit requires matching [time, batch] records.")
    if (
        eligible_gate_mask.dtype != torch.bool
        or eligible_gate_mask.shape != route.shape
    ):
        raise ValueError("PAD eligible Gate mask must be boolean and match routes.")
    chunk_mask = _chunk_mask(
        valid_mask,
        shape=route.shape,
        device=route.route_used.device,
    )
    if bool((eligible_gate_mask & (~emitted.valid | ~chunk_mask)).any().item()):
        raise ValueError("PAD eligible decisions must be emitted valid chunks.")
    valid_count = int(chunk_mask.sum().item())
    eligible_count = int(eligible_gate_mask.sum().item())
    if valid_count < 1 or eligible_count < 1:
        raise ValueError("PAD rollout audit requires valid and eligible decisions.")

    base = _probability_summary(emitted.base_probability, eligible=eligible_gate_mask)
    behavior = _probability_summary(
        emitted.behavior_probability,
        eligible=eligible_gate_mask,
    )
    emitted_count = int(emitted.valid.sum().item())
    valid_idm_count = int(
        (chunk_mask & (route.route_used == int(WAMRoute.IDM))).sum().item()
    )
    valid_uncond_count = int(
        (chunk_mask & (route.route_used == int(WAMRoute.UNCOND))).sum().item()
    )
    eligible_idm_count = int(
        (eligible_gate_mask & (emitted.next_route == int(WAMRoute.IDM))).sum().item()
    )
    return FastWAMRolloutStateAudit(
        decision_shape=tuple(int(size) for size in route.shape),
        total_decision_count=int(route.route_used.numel()),
        valid_chunk_count=valid_count,
        valid_idm_chunk_count=valid_idm_count,
        valid_uncond_chunk_count=valid_uncond_count,
        forced_route_count=int((chunk_mask & route.route_was_forced).sum().item()),
        emitted_decision_count=emitted_count,
        eligible_gate_decision_count=eligible_count,
        eligible_idm_decision_count=eligible_idm_count,
        unused_emitted_decision_count=emitted_count - eligible_count,
        route_decision_sha256=checkpoint_state_sha256(
            {
                "route_used": route.route_used,
                "route_was_forced": route.route_was_forced,
                "chunk_ids": route.chunk_ids,
                "episode_ids": route.episode_ids,
                "route_source_chunk_ids": route.route_source_chunk_ids,
                "actor_versions": route.actor_versions,
                "emitted_next_route": emitted.next_route,
                "emitted_valid": emitted.valid,
                "emitted_source_chunk_ids": emitted.source_chunk_ids,
            }
        ),
        base_probability_min=base["minimum"],
        base_probability_max=base["maximum"],
        base_probability_mean=base["mean"],
        base_probability_p10=base["p10"],
        base_probability_p50=base["p50"],
        base_probability_p90=base["p90"],
        base_probability_bimodality_score=base["bimodality_score"],
        base_probability_outside_0p2_0p8_fraction=base["outside"],
        behavior_probability_min=behavior["minimum"],
        behavior_probability_max=behavior["maximum"],
        behavior_probability_mean=behavior["mean"],
        kv_replay_backend="condition",
        kv_storage_dtype="none",
        kv_layer_indices=(),
        kv_denoise_tap_count=0,
        kv_configured_max_bytes_per_sample=None,
        kv_all_emitted_sample_count=0,
        kv_all_emitted_nonzero_sample_count=0,
        kv_all_emitted_total_bytes=0,
        kv_all_emitted_maximum_bytes_per_sample=0,
        kv_eligible_sample_count=0,
        kv_eligible_nonzero_sample_count=0,
        kv_eligible_total_bytes=0,
        kv_eligible_maximum_bytes_per_sample=0,
    )
