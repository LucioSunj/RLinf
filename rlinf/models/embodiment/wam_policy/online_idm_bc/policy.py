# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adaptive FastWAM policy subclass that exposes online IDM BC outputs."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.models.embodiment.wam_policy.adaptive_policy import FastWAMAdaptivePolicy
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
)

from .config import OnlineIDMBCConfig
from .runtime import OnlineIDMTeacherLiberoRuntime


class OnlineIDMBCFastWAMPolicy(FastWAMAdaptivePolicy):
    """Add one LoRA-only BC graph while preserving the base policy surface."""

    def __init__(
        self,
        *,
        online_idm_bc_config: OnlineIDMBCConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not online_idm_bc_config.enabled:
            raise ValueError("Online IDM BC policy requires `enabled: true`.")
        if not isinstance(self.runtime, OnlineIDMTeacherLiberoRuntime):
            raise TypeError(
                "Online IDM BC policy requires OnlineIDMTeacherLiberoRuntime."
            )
        self.online_idm_bc_config = online_idm_bc_config

    @classmethod
    def from_base_policy(
        cls,
        policy: FastWAMAdaptivePolicy,
        *,
        config: OnlineIDMBCConfig,
    ) -> "OnlineIDMBCFastWAMPolicy":
        """Reuse every constructed module from the standard adaptive policy."""

        if type(policy) is not FastWAMAdaptivePolicy:
            raise TypeError(
                "Online IDM BC can wrap only the standard FastWAMAdaptivePolicy, "
                f"got {type(policy).__name__}."
            )
        result = cls(
            actor=policy.actor,
            runtime=policy.runtime,
            lora_adapter=policy.lora_adapter,
            gate=policy.gate,
            critic=policy.critic,
            config=policy.config,
            online_idm_bc_config=config,
        )
        result.route_tracker = policy.route_tracker
        result.actor_version = int(policy.actor_version)
        result.train(policy.training)
        return result

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        *,
        route_info: ChunkRouteRecord,
        emitted_gate: GateDecisionRecord,
        compute_values: bool = True,
        compute_base_logprobs: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run unchanged RL replay, then attach the online BC loss numerator."""

        result = super().default_forward(
            forward_inputs,
            route_info=route_info,
            emitted_gate=emitted_gate,
            compute_values=compute_values,
            compute_base_logprobs=compute_base_logprobs,
            **kwargs,
        )
        online_bc = self.runtime.compute_online_idm_bc_loss(
            forward_inputs=forward_inputs,
            route_info=route_info,
        )
        overlap = sorted(set(result).intersection(online_bc.as_forward_outputs()))
        if overlap:
            raise KeyError(
                f"Online IDM BC output keys collide with base policy: {overlap}."
            )
        result.update(online_bc.as_forward_outputs())
        return result
