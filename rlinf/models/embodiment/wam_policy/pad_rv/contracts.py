# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD-RV data contracts kept outside the legacy adaptive policy."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from fastwam.models.wan22.adaptive_action import CachedActionCondition

from rlinf.envs.action_contract import ActionExecutionTrace
from rlinf.models.embodiment.wam_policy.critic import FastWAMValueFeatures


@dataclass(frozen=True)
class PreparedRouteContext:
    """One encoded current observation shared by Gate and selected expert."""

    images: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    first_frame_latents: tuple[torch.Tensor, ...]
    current_conditions: tuple[CachedActionCondition, ...]
    gate_features: FastWAMValueFeatures
    action_noise_seeds: tuple[int, ...] | None = None
    idm_noise_seeds: tuple[int, ...] | None = None
    action_initial_noise: torch.Tensor | None = None
    idm_initial_latents: torch.Tensor | None = None

    def __post_init__(self) -> None:
        batch_size = int(self.images.shape[0])
        if (
            self.context.shape[0] != batch_size
            or self.context_mask.shape[0] != batch_size
        ):
            raise ValueError("PAD context batch does not match images.")
        if (
            len(self.first_frame_latents) != batch_size
            or len(self.current_conditions) != batch_size
        ):
            raise ValueError("PAD per-sample condition state does not match images.")
        if self.gate_features.batch_size != batch_size:
            raise ValueError("PAD Gate features do not match images.")
        for name in ("action_noise_seeds", "idm_noise_seeds"):
            seeds = getattr(self, name)
            if seeds is not None and len(seeds) != batch_size:
                raise ValueError(f"PAD {name} does not match images.")
        for name in ("action_initial_noise", "idm_initial_latents"):
            tensor = getattr(self, name)
            if tensor is not None and tensor.shape[0] != batch_size:
                raise ValueError(f"PAD {name} does not match images.")

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])


@dataclass(frozen=True)
class PadFrozenChunkSample:
    """Frozen expert action with Gate/value replay and no Flow chain."""

    actions: torch.Tensor
    gate_features: FastWAMValueFeatures
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    critic_features: FastWAMValueFeatures | torch.Tensor | None = None
    action_execution_trace: ActionExecutionTrace | None = None

    def __post_init__(self) -> None:
        batch_size = int(self.actions.shape[0])
        if self.gate_features.batch_size != batch_size:
            raise ValueError("PAD Gate feature batch must match actions.")
        if self.critic_features is not None:
            feature_batch = (
                self.critic_features.batch_size
                if isinstance(self.critic_features, FastWAMValueFeatures)
                else int(self.critic_features.shape[0])
            )
            if feature_batch != batch_size:
                raise ValueError("PAD critic feature batch must match actions.")
        if (
            self.action_execution_trace is not None
            and self.action_execution_trace.batch_size != batch_size
        ):
            raise ValueError("PAD Action trace batch must match actions.")
