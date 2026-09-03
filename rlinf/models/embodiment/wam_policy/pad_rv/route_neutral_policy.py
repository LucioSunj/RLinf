# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""PAD policy with route-neutral replay and critic-only warm-up routing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from fastwam.models.wan22.gate_transformer import epsilon_mixture_bernoulli

from .policy import PadFrozenPolicy
from .route_neutral_contracts import PadCriticWarmupConfig
from .route_neutral_gate import (
    PhysicalStateHistoryTracker,
    deserialize_route_neutral_features,
)
from .route_neutral_runtime import PadRouteNeutralLiberoRuntime
from .routing_state import CurrentStepRouteTracker


class RouteNeutralRoutingState:
    """Checkpoint route bookkeeping and physical history without mixing inputs."""

    _SCHEMA = "pad-route-neutral-runtime-state-v1"

    def __init__(
        self,
        *,
        physical_history: PhysicalStateHistoryTracker,
    ) -> None:
        self.current_step = CurrentStepRouteTracker()
        self.physical_history = physical_history

    def __getattr__(self, name: str):
        return getattr(self.current_step, name)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self._SCHEMA,
            "current_step": self.current_step.state_dict(),
            "physical_history": self.physical_history.state_dict(),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        expected = {"schema", "current_step", "physical_history"}
        if set(payload) != expected or payload.get("schema") != self._SCHEMA:
            raise ValueError("Unsupported route-neutral runtime checkpoint state.")
        self.current_step.load_state_dict(payload["current_step"])
        self.physical_history.load_state_dict(payload["physical_history"])


class PadRouteNeutralPolicy(PadFrozenPolicy):
    """Reuse PAD same-chunk routing while replacing only Gate-visible state."""

    def __init__(
        self,
        *,
        runtime: PadRouteNeutralLiberoRuntime,
        critic_warmup,
        **kwargs: Any,
    ) -> None:
        if not isinstance(runtime, PadRouteNeutralLiberoRuntime):
            raise TypeError("Route-neutral policy requires its dedicated runtime.")
        self.critic_warmup = PadCriticWarmupConfig.from_mapping(critic_warmup)
        super().__init__(runtime=runtime, **kwargs)
        self.route_tracker = RouteNeutralRoutingState(
            physical_history=runtime.physical_history
        )

    def _warmup_active(self) -> bool:
        return self.actor_version < self.critic_warmup.runner_updates

    def _training_gate_decision(
        self,
        *,
        logits: torch.Tensor,
        sampling_seeds: torch.Tensor | None = None,
    ):
        if not self._warmup_active():
            return super()._training_gate_decision(
                logits=logits,
                sampling_seeds=sampling_seeds,
            )
        behavior = epsilon_mixture_bernoulli(
            logits,
            temperature=self.config.gate_temperature,
            epsilon=1.0,
        )
        if sampling_seeds is None:
            routes = behavior.sample()
        else:
            seeds = torch.as_tensor(sampling_seeds, device="cpu", dtype=torch.long)
            if seeds.shape != logits.shape:
                raise ValueError("Warm-up Gate seeds must match Gate logits.")
            sampled = []
            for probability, seed in zip(
                behavior.behavior_idm_probability.reshape(-1),
                seeds.reshape(-1),
                strict=True,
            ):
                generator = torch.Generator(device=probability.device)
                generator.manual_seed(int(seed.item()))
                sampled.append(
                    torch.bernoulli(probability.reshape(1), generator=generator)
                )
            routes = torch.cat(sampled).reshape_as(logits).to(dtype=torch.long)
        exploration = (
            torch.ones_like(routes, dtype=torch.bool)
            if self.config.decision_telemetry_enabled
            else None
        )
        return (
            routes,
            behavior.base_idm_probability,
            behavior.behavior_idm_probability,
            behavior.log_prob(routes),
            exploration,
        )

    def _training_gate_epsilon(self, logits: torch.Tensor) -> torch.Tensor:
        if self._warmup_active():
            return torch.ones_like(logits)
        return super()._training_gate_epsilon(logits)

    def _gate_features_from_forward_inputs(
        self,
        forward_inputs: Mapping[str, torch.Tensor],
        *,
        layer_indices: tuple[int, ...],
    ):
        return deserialize_route_neutral_features(
            forward_inputs,
            layer_indices=layer_indices,
        )

    def load_eval_checkpoint(self, *args, **kwargs) -> int:
        version = super().load_eval_checkpoint(*args, **kwargs)
        physical_history = PhysicalStateHistoryTracker(self.runtime.route_neutral_input)
        self.runtime.physical_history = physical_history
        self.route_tracker = RouteNeutralRoutingState(physical_history=physical_history)
        return version


__all__ = ["PadRouteNeutralPolicy", "RouteNeutralRoutingState"]
