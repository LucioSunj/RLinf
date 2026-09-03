# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Route-neutral decision features over the trainable online-BC runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from fastwam.adapters import PolicyRegime

from rlinf.envs.action_contract import ActionExecutionTrace
from rlinf.models.embodiment.wam_policy.adaptive_policy import FastWAMChunkSample
from rlinf.models.embodiment.wam_policy.critic import (
    FastWAMValueFeatures,
    FastWAMValueTransformerConfig,
    extract_fastwam_value_features,
)
from rlinf.models.embodiment.wam_policy.kv_replay import GateKVReplayBackend
from rlinf.models.embodiment.wam_policy.online_idm_bc.runtime import (
    OnlineIDMTeacherLiberoRuntime,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_contracts import (
    RouteNeutralGateInputContract,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PhysicalStateHistoryTracker,
    RouteNeutralGateFeatures,
    RouteNeutralVisualFeatures,
)


@dataclass(frozen=True, slots=True)
class RouteNeutralTrainableChunkSample:
    """Trainable routed action payload with no actor-facing Gate snapshot."""

    actions: torch.Tensor
    old_flow_logprobs: torch.Tensor
    flow_chains: torch.Tensor
    denoise_indices: torch.Tensor
    forward_inputs: dict[str, torch.Tensor]
    critic_features: FastWAMValueFeatures | torch.Tensor | None
    action_execution_trace: ActionExecutionTrace | None

    @classmethod
    def without_route_snapshot(
        cls,
        sample: FastWAMChunkSample,
    ) -> "RouteNeutralTrainableChunkSample":
        """Drop action/regime-derived Gate K/V at the runtime boundary."""

        return cls(
            actions=sample.actions,
            old_flow_logprobs=sample.old_flow_logprobs,
            flow_chains=sample.flow_chains,
            denoise_indices=sample.denoise_indices,
            forward_inputs=dict(sample.forward_inputs),
            critic_features=sample.critic_features,
            action_execution_trace=sample.action_execution_trace,
        )


class RouteNeutralOnlineIDMTeacherLiberoRuntime(OnlineIDMTeacherLiberoRuntime):
    """Produce neutral Gate inputs, then reuse trainable UNCOND + IDM teacher."""

    def __init__(
        self,
        *,
        route_neutral_input,
        route_neutral_visual,
        gate_replay_backend="recompute",
        **kwargs: Any,
    ) -> None:
        # The orchestration-level backend is ``recompute`` so RLinf creates no
        # Action-K/V handle store. Internally the inherited sampler is told
        # ``stored`` only to avoid materializing unused IDM latent replay; its
        # snapshots are discarded by ``sample_routed_action_batch`` below.
        orchestration_backend = GateKVReplayBackend(gate_replay_backend)
        if orchestration_backend is not GateKVReplayBackend.RECOMPUTE:
            raise ValueError(
                "Route-neutral trainable runtime requires inactive recompute "
                "orchestration."
            )
        super().__init__(
            gate_replay_backend=GateKVReplayBackend.STORED,
            **kwargs,
        )
        state_dim = int(getattr(self.actor, "proprio_dim", 0) or 0)
        self.route_neutral_input = RouteNeutralGateInputContract.from_mapping(
            route_neutral_input,
            state_dim=state_dim,
        )
        self.route_neutral_visual = FastWAMValueTransformerConfig.materialize(
            route_neutral_visual
        )
        if tuple(self.route_neutral_visual.sources) != ("current_frame_video",):
            raise ValueError(
                "Route-neutral visual producer must be current-frame-only."
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
    def prepare_route_neutral_gate_features(
        self,
        *,
        env_obs: dict[str, Any],
    ) -> RouteNeutralGateFeatures:
        """Read base-regime, no-action current conditions before route choice."""

        images, context, context_mask = self._encode_condition(env_obs)
        batch_size = int(images.shape[0])
        state = self._normalized_proprio(env_obs["states"]).detach()
        if state.shape != (batch_size, self.route_neutral_input.state_dim):
            raise ValueError("Route-neutral normalized proprio shape changed.")
        if context.shape[1] < 2 or not bool(context_mask[:, -1].all().item()):
            raise ValueError("Could not isolate FastWAM's appended proprio token.")

        visual_features: list[FastWAMValueFeatures] = []
        for index in range(batch_size):
            condition, replay_noise = self._prepare_action_condition(
                image=images[index : index + 1],
                context=context[index : index + 1],
                context_mask=context_mask[index : index + 1],
                regime=PolicyRegime.UNCOND,
            )
            if replay_noise is not None:
                raise AssertionError("Current-frame condition created future noise.")
            visual_features.append(
                extract_fastwam_value_features(
                    condition,
                    mot=self.actor.mot,
                    action_expert=self.actor.action_expert,
                    config=self.route_neutral_visual,
                    regime_context=None,
                )
            )
        env_ids, reset_mask = self._history_metadata(
            env_obs,
            batch_size=batch_size,
            device=state.device,
        )
        history = self.physical_history.features_and_append(
            env_ids=env_ids,
            reset_mask=reset_mask,
            current_state=state,
        )
        return RouteNeutralGateFeatures(
            visual=RouteNeutralVisualFeatures.from_value_features(
                FastWAMValueFeatures.cat(visual_features)
            ),
            language=context[:, :-1].detach(),
            language_mask=context_mask[:, :-1].detach().to(dtype=torch.bool),
            state=state,
            physical_history=history,
        )

    def sample_routed_action_batch(
        self,
        *,
        env_obs: dict[str, Any],
        routes: torch.Tensor,
        mode: Literal["train", "eval"],
        actor_version: int,
        collect_replay: bool,
    ) -> RouteNeutralTrainableChunkSample:
        """Reuse Flow-SDE/teacher execution and seal legacy Gate snapshots."""

        sample = super().sample_action_batch(
            env_obs=env_obs,
            routes=routes,
            mode=mode,
            actor_version=actor_version,
            collect_replay=collect_replay,
        )
        return RouteNeutralTrainableChunkSample.without_route_snapshot(sample)


__all__ = [
    "RouteNeutralOnlineIDMTeacherLiberoRuntime",
    "RouteNeutralTrainableChunkSample",
]
