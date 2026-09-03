# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Additive builder for route-neutral Gate + trainable UNCOND LoRA."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.models.embodiment.wam_policy import get_model as build_adaptive_model
from rlinf.models.embodiment.wam_policy.adaptive_policy import FastWAMAdaptivePolicy
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import OnlineIDMBCConfig
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PadRouteNeutralCurrentStepGate,
    PadRouteNeutralGateConfig,
)

from .policy import RouteNeutralOnlineIDMBCFastWAMPolicy
from .runtime import RouteNeutralOnlineIDMTeacherLiberoRuntime


def build_route_neutral_online_idm_bc_model(cfg: Any, torch_dtype):
    """Reuse the production adaptive builder and replace only Gate/policy APIs."""

    base = build_adaptive_model(cfg, torch_dtype)
    if type(base) is not FastWAMAdaptivePolicy:
        raise TypeError(
            "Route-neutral builder expected the standard adaptive policy, got "
            f"{type(base).__name__}."
        )
    runtime = base.runtime
    if not isinstance(runtime, RouteNeutralOnlineIDMTeacherLiberoRuntime):
        raise TypeError("Route-neutral builder received the wrong runtime.")
    gate = PadRouteNeutralCurrentStepGate(
        PadRouteNeutralGateConfig(
            visual=runtime.route_neutral_visual,
            language_dim=int(base.actor.text_dim),
            state_dim=runtime.route_neutral_input.state_dim,
            history_length_chunks=(runtime.route_neutral_input.history_length_chunks),
        )
    ).to(dtype=torch.float32)
    profile = cfg.route_neutral_online
    return RouteNeutralOnlineIDMBCFastWAMPolicy(
        actor=base.actor,
        runtime=runtime,
        lora_adapter=base.lora_adapter,
        gate=gate,
        critic=base.critic,
        config=base.config,
        online_idm_bc_config=OnlineIDMBCConfig.from_mapping(profile.online_idm_bc),
        critic_warmup=profile.critic_warmup,
    )


__all__ = ["build_route_neutral_online_idm_bc_model"]
