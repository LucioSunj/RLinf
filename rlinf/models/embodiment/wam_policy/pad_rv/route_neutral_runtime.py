# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Route-neutral feature adapter over the existing PAD current-step runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from rlinf.models.embodiment.wam_policy.critic import FastWAMValueFeatures

from .contracts import PreparedRouteContext
from .route_neutral_contracts import RouteNeutralGateInputContract
from .route_neutral_gate import (
    PhysicalStateHistoryTracker,
    RouteNeutralGateFeatures,
    RouteNeutralVisualFeatures,
    serialize_route_neutral_features,
)
from .runtime import PadFrozenLiberoRuntime


class PadRouteNeutralLiberoRuntime(PadFrozenLiberoRuntime):
    """Narrow canonical condition features before they cross the Gate API."""

    def __init__(
        self,
        *,
        gate_input_contract,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        state_dim = int(getattr(self.actor, "proprio_dim", 0) or 0)
        self.route_neutral_input = RouteNeutralGateInputContract.from_mapping(
            gate_input_contract,
            state_dim=state_dim,
        )
        if tuple(self.gate_feature_config.sources) != ("current_frame_video",):
            raise ValueError(
                "Route-neutral runtime requires current-frame-only visual features."
            )
        if getattr(self.actor, "proprio_encoder", None) is None:
            raise ValueError("Route-neutral runtime requires FastWAM proprio encoding.")
        self.physical_history = PhysicalStateHistoryTracker(self.route_neutral_input)

    @staticmethod
    def _history_metadata(
        env_obs: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = env_obs.get("_fastwam_env_ids")
        reset_mask = env_obs.get("_fastwam_reset_mask")
        env_ids = (
            torch.arange(batch_size, device=device, dtype=torch.long)
            if env_ids is None
            else torch.as_tensor(env_ids, device=device, dtype=torch.long)
        )
        reset_mask = (
            torch.zeros(batch_size, device=device, dtype=torch.bool)
            if reset_mask is None
            else torch.as_tensor(reset_mask, device=device, dtype=torch.bool)
        )
        if env_ids.shape != (batch_size,) or reset_mask.shape != (batch_size,):
            raise ValueError("Route-neutral history metadata must have shape [B].")
        return env_ids, reset_mask

    @torch.no_grad()
    def prepare_route_context(self, *, env_obs: dict[str, Any]) -> PreparedRouteContext:
        """Reuse PAD condition production, then expose only neutral Gate inputs."""

        prepared = super().prepare_route_context(env_obs=env_obs)
        if not isinstance(prepared.gate_features, FastWAMValueFeatures):
            raise TypeError("Canonical PAD feature producer changed its output type.")
        state = self._normalized_proprio(env_obs["states"]).detach()
        if state.shape != (
            prepared.batch_size,
            self.route_neutral_input.state_dim,
        ):
            raise ValueError("Route-neutral normalized proprio shape changed.")
        if prepared.context.shape[1] < 2 or not bool(
            prepared.context_mask[:, -1].all().item()
        ):
            raise ValueError(
                "Route-neutral runtime could not isolate the appended proprio token."
            )
        language = prepared.context[:, :-1].detach()
        language_mask = prepared.context_mask[:, :-1].detach().to(dtype=torch.bool)
        env_ids, reset_mask = self._history_metadata(
            env_obs,
            batch_size=prepared.batch_size,
            device=state.device,
        )
        history = self.physical_history.features_and_append(
            env_ids=env_ids,
            reset_mask=reset_mask,
            current_state=state,
        )
        gate_features = RouteNeutralGateFeatures(
            visual=RouteNeutralVisualFeatures.from_value_features(
                prepared.gate_features
            ),
            language=language,
            language_mask=language_mask,
            state=state,
            physical_history=history,
        )
        return replace(prepared, gate_features=gate_features)

    def _gate_replay_inputs(
        self,
        prepared: PreparedRouteContext,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(prepared.gate_features, RouteNeutralGateFeatures):
            raise TypeError("Route-neutral runtime received legacy Gate features.")
        return serialize_route_neutral_features(prepared.gate_features)

    def _critic_features_for_prepared(
        self,
        prepared: PreparedRouteContext,
    ) -> FastWAMValueFeatures | None:
        """Keep critic replay separate from the narrowed Gate feature object."""

        if self.critic_feature_config is None:
            return None
        return FastWAMValueFeatures.cat(
            [
                self._condition_features(condition, config=self.critic_feature_config)
                for condition in prepared.current_conditions
            ]
        )


__all__ = ["PadRouteNeutralLiberoRuntime"]
