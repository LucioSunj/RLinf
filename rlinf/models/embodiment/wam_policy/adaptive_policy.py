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

"""Composite FastWAM actor, delayed Gate, and pi0.5 critic policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch
import torch.nn as nn
from fastwam.models.wan22.gate_transformer import epsilon_mixture_bernoulli
from fastwam.models.wan22.kv_tap import GateKVSnapshot

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    GateKVMetadata,
)
from .evaluation import (
    EvaluationRouteSelection,
    EvaluationRoutingConfig,
    EvaluationRoutingMode,
    select_evaluation_routes,
)
from .kv_replay import (
    GateKVReplayBackend,
    GateKVReplayConfig,
    PackedGateKVTaps,
    pack_gate_kv,
)
from .routing_state import PendingRouteTracker


@dataclass(frozen=True)
class FastWAMChunkSample:
    """Runtime output for one mixed-route action chunk."""

    actions: torch.Tensor
    old_flow_logprobs: torch.Tensor
    flow_chains: torch.Tensor
    denoise_indices: torch.Tensor
    gate_snapshots: tuple[GateKVSnapshot, ...]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        batch_size = self.actions.shape[0]
        if self.old_flow_logprobs.shape[0] != batch_size:
            raise ValueError("Flow log-prob batch must match actions.")
        if self.flow_chains.shape[0] != batch_size:
            raise ValueError("Flow chain batch must match actions.")
        if self.denoise_indices.shape != (batch_size,):
            raise ValueError("Denoising indices must have shape [B].")
        if not self.gate_snapshots:
            raise ValueError("Gate snapshots are required after every chunk.")
        if any(snapshot.batch_size != batch_size for snapshot in self.gate_snapshots):
            raise ValueError("Gate snapshot batch must match actions.")


class FastWAMPolicyRuntime(Protocol):
    """Model-specific preprocessing and branch execution used by the wrapper."""

    def sample_action_batch(
        self,
        *,
        env_obs: dict[str, Any],
        routes: torch.Tensor,
        mode: Literal["train", "eval"],
        actor_version: int,
        collect_replay: bool = True,
    ) -> FastWAMChunkSample: ...

    def replay_action_batch(
        self,
        *,
        forward_inputs: dict[str, torch.Tensor],
        route_info: ChunkRouteRecord,
        compute_base_logprobs: bool = False,
    ) -> dict[str, torch.Tensor]: ...

    def critic_observation(
        self,
        *,
        env_obs: dict[str, Any] | None = None,
        forward_inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]: ...

    def recompute_gate_snapshots(
        self,
        *,
        forward_inputs: dict[str, torch.Tensor],
        route_info: ChunkRouteRecord,
    ) -> tuple[GateKVSnapshot, ...]: ...


@dataclass(frozen=True)
class FastWAMAdaptivePolicyConfig:
    """Behavior-policy and replay configuration."""

    gate_epsilon: float = 0.1
    gate_temperature: float = 1.0
    eval_routing_mode: EvaluationRoutingMode | str = (
        EvaluationRoutingMode.LEARNED_THRESHOLD
    )
    eval_idm_threshold: float = 0.5
    eval_random_idm_probability: float | None = None
    eval_routing_seed: int = 0
    eval_microbatch_size: int = 1
    kv_replay: GateKVReplayConfig = field(default_factory=GateKVReplayConfig)

    def __post_init__(self) -> None:
        if not math.isfinite(self.gate_epsilon) or not 0 <= self.gate_epsilon <= 1:
            raise ValueError("`gate_epsilon` must lie in [0, 1].")
        if not math.isfinite(self.gate_temperature) or self.gate_temperature <= 0:
            raise ValueError("`gate_temperature` must be positive.")
        if self.eval_microbatch_size < 1:
            raise ValueError("`eval_microbatch_size` must be positive.")
        evaluation = self.evaluation_routing
        object.__setattr__(self, "eval_routing_mode", evaluation.mode)
        object.__setattr__(self, "eval_idm_threshold", evaluation.idm_threshold)
        object.__setattr__(
            self,
            "eval_random_idm_probability",
            evaluation.random_idm_probability,
        )
        object.__setattr__(self, "eval_routing_seed", evaluation.routing_seed)

    @property
    def evaluation_routing(self) -> EvaluationRoutingConfig:
        """Return the validated pure evaluation-routing configuration."""

        return EvaluationRoutingConfig(
            mode=self.eval_routing_mode,
            idm_threshold=self.eval_idm_threshold,
            random_idm_probability=self.eval_random_idm_probability,
            routing_seed=self.eval_routing_seed,
        )


def _bytes_per_sample(packed: PackedGateKVTaps) -> torch.Tensor:
    result = torch.zeros(packed.batch_size, dtype=torch.long)
    for name, tensor in packed.as_forward_inputs().items():
        if name.endswith("_layer_indices"):
            continue
        if tensor.ndim == 0 or tensor.shape[0] != packed.batch_size:
            continue
        per_sample = tensor[0].numel() * tensor.element_size()
        result += per_sample
    return result


def _column_values(values: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if values.shape == (batch_size,):
        return values[:, None]
    if values.shape == (batch_size, 1):
        return values
    raise ValueError(
        "FastWAM critic values must have shape [B] or [B, 1], got "
        f"{tuple(values.shape)} for batch size {batch_size}."
    )


class FastWAMAdaptivePolicy(nn.Module, BasePolicy):
    """RLinf policy with separate Gate and UNCOND Flow-SDE likelihoods."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        runtime: FastWAMPolicyRuntime,
        lora_adapter: Any,
        gate: nn.Module,
        critic: nn.Module | None,
        config: FastWAMAdaptivePolicyConfig | None = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.runtime = runtime
        self.lora_adapter = lora_adapter
        self.gate = gate
        self.critic = critic
        self.config = config or FastWAMAdaptivePolicyConfig()
        self.route_tracker = PendingRouteTracker()
        self.actor_version = 0
        self._enforce_frozen_actor()
        self.actor.eval()

    def _require_critic(self) -> nn.Module:
        if self.critic is None:
            raise RuntimeError(
                "The pi0.5 critic is intentionally absent from standalone evaluation."
            )
        return self.critic

    def _enforce_frozen_actor(self) -> None:
        """Keep only the injected UNCOND LoRA trainable inside FastWAM."""

        lora_parameters = tuple(self.lora_adapter.lora_parameters())
        if not lora_parameters:
            raise ValueError(
                "FastWAM adaptive policy requires trainable LoRA parameters."
            )
        lora_parameter_ids = {id(parameter) for parameter in lora_parameters}
        for parameter in self.actor.parameters():
            parameter.requires_grad_(id(parameter) in lora_parameter_ids)
        unexpected = [
            name
            for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad and id(parameter) not in lora_parameter_ids
        ]
        if unexpected:
            raise RuntimeError(
                f"FastWAM actor has trainable non-LoRA parameters: {unexpected}."
            )

    def train(self, mode: bool = True) -> FastWAMAdaptivePolicy:
        """Train adaptive modules while keeping the frozen actor in eval mode."""

        super().train(mode)
        self.actor.eval()
        self.gate.train(mode)
        if self.critic is not None:
            self.critic.train(mode)
        return self

    def forward(
        self,
        forward_type: ForwardType = ForwardType.DEFAULT,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Dispatch through ``BasePolicy`` instead of ``nn.Module.forward``."""

        return BasePolicy.forward(self, forward_type=forward_type, **kwargs)

    @property
    def value_head(self) -> nn.Module:
        """Expose the colocated critic head to existing rollout bootstrap code."""

        return self._require_critic().value_head

    @staticmethod
    def _critic_backbone_no_split_values(
        critic: nn.Module | None,
        attribute: str,
    ) -> list[str]:
        """Return deduplicated FSDP metadata from the nested critic backbone."""

        backbone = getattr(critic, "backbone", None)
        values = getattr(backbone, attribute, None)
        if values is None:
            return []
        if isinstance(values, str):
            values = (values,)
        return list(dict.fromkeys(values))

    @property
    def _no_split_modules(self) -> list[str]:
        """Expose nested pi0.5 module classes to FSDP wrapping."""

        return self._critic_backbone_no_split_values(
            self.critic,
            "_no_split_modules",
        )

    @property
    def _no_split_names(self) -> list[str]:
        """Expose nested pi0.5 backbone parameter names to FSDP wrapping."""

        return self._critic_backbone_no_split_values(
            self.critic,
            "_no_split_names",
        )

    def set_global_step(self, version: int) -> None:
        if version < 0:
            raise ValueError("Actor version must be non-negative.")
        version = int(version)
        if version != self.actor_version:
            self.route_tracker.force_idm_after_actor_update()
        self.actor_version = version

    def capture_gate_recompute_reference(self) -> None:
        """Freeze this rollout's behavior LoRA for opt-in K/V recomputation."""

        if self.config.kv_replay.backend is not GateKVReplayBackend.RECOMPUTE:
            return
        self.lora_adapter.capture_replay_reference(
            actor_version=self.actor_version,
        )

    @staticmethod
    def _routing_metadata(
        env_obs: dict[str, Any],
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = env_obs.get("_fastwam_env_ids")
        reset_mask = env_obs.get("_fastwam_reset_mask")
        if env_ids is None:
            env_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        if reset_mask is None:
            # Only unseen IDs are interpreted as new below. Existing IDs must
            # receive an explicit reset mask from the environment worker.
            reset_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            reset_mask = torch.as_tensor(reset_mask, device=device, dtype=torch.bool)
        return env_ids, reset_mask

    def _training_gate_decision(
        self,
        *,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        behavior = epsilon_mixture_bernoulli(
            logits,
            temperature=self.config.gate_temperature,
            epsilon=self.config.gate_epsilon,
        )
        route = behavior.sample()
        logprob = behavior.log_prob(route)
        return (
            route,
            behavior.base_idm_probability,
            behavior.behavior_idm_probability,
            logprob,
        )

    def _evaluation_gate_decision(
        self,
        *,
        logits: torch.Tensor,
        env_ids: torch.Tensor,
        episode_ids: torch.Tensor,
        source_chunk_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        EvaluationRouteSelection,
    ]:
        behavior = epsilon_mixture_bernoulli(
            logits,
            temperature=self.config.gate_temperature,
            epsilon=0.0,
        )
        selection = select_evaluation_routes(
            self.config.evaluation_routing,
            gate_idm_probabilities=behavior.base_idm_probability,
            env_ids=env_ids,
            episode_ids=episode_ids,
            source_chunk_ids=source_chunk_ids,
        )
        route = selection.effective_next_route
        return (
            route,
            behavior.base_idm_probability,
            behavior.behavior_idm_probability,
            behavior.log_prob(route),
            selection,
        )

    @staticmethod
    def _slice_env_obs(
        env_obs: dict[str, Any],
        *,
        start: int,
        end: int,
        batch_size: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in env_obs.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                result[key] = value[start:end]
            elif isinstance(value, list) and len(value) == batch_size:
                result[key] = value[start:end]
            elif isinstance(value, tuple) and len(value) == batch_size:
                result[key] = value[start:end]
            else:
                result[key] = value
        return result

    @torch.no_grad()
    def _predict_eval_action_batch(
        self,
        *,
        env_obs: dict[str, Any],
        route_info: ChunkRouteRecord,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run deterministic evaluation without constructing PPO replay."""

        batch_size = route_info.route_used.shape[0]
        actions = []
        decisions = []
        microbatch = self.config.eval_microbatch_size
        for start in range(0, batch_size, microbatch):
            end = min(start + microbatch, batch_size)
            sample = self.runtime.sample_action_batch(
                env_obs=self._slice_env_obs(
                    env_obs,
                    start=start,
                    end=end,
                    batch_size=batch_size,
                ),
                routes=route_info.route_used[start:end],
                mode="eval",
                actor_version=self.actor_version,
                collect_replay=False,
            )
            logits = self.gate(sample.gate_snapshots)
            decisions.append(
                self._evaluation_gate_decision(
                    logits=logits,
                    env_ids=env_ids[start:end],
                    episode_ids=route_info.episode_ids[start:end],
                    source_chunk_ids=route_info.chunk_ids[start:end],
                )
            )
            actions.append(sample.actions)

        next_route = torch.cat([item[0] for item in decisions], dim=0)
        base_probability = torch.cat([item[1] for item in decisions], dim=0)
        behavior_probability = torch.cat([item[2] for item in decisions], dim=0)
        gate_logprob = torch.cat([item[3] for item in decisions], dim=0)
        evaluation_selection = EvaluationRouteSelection.cat(
            [item[4] for item in decisions]
        )
        self.route_tracker.emit(
            env_ids=env_ids,
            routes=next_route,
            source_chunk_ids=route_info.chunk_ids,
            episode_ids=route_info.episode_ids,
            actor_version=self.actor_version,
        )
        emitted_gate = GateDecisionRecord(
            next_route=next_route,
            base_probability=base_probability,
            behavior_probability=behavior_probability,
            old_logprob=gate_logprob,
            epsilon=torch.zeros_like(base_probability),
            temperature=torch.full_like(
                base_probability,
                self.config.gate_temperature,
            ),
            valid=torch.ones_like(next_route, dtype=torch.bool),
            source_chunk_ids=route_info.chunk_ids.to(next_route.device),
            episode_ids=route_info.episode_ids.to(next_route.device),
            actor_versions=torch.full_like(next_route, self.actor_version),
            kv_metadata=None,
        )
        return torch.cat(actions, dim=0), {
            "prev_logprobs": torch.empty(
                batch_size,
                0,
                device=route_info.route_used.device,
                dtype=torch.float32,
            ),
            "prev_values": torch.zeros(
                batch_size,
                1,
                device=route_info.route_used.device,
                dtype=torch.float32,
            ),
            "forward_inputs": {},
            "route_info": route_info,
            "emitted_gate": emitted_gate,
            "evaluation_selection": evaluation_selection,
        }

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
        **_kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported FastWAM policy mode {mode!r}.")
        batch_size = int(env_obs["states"].shape[0])
        device = env_obs["states"].device
        env_ids, reset_mask = self._routing_metadata(env_obs, batch_size, device)
        route_info = self.route_tracker.consume(
            env_ids=env_ids,
            reset_mask=reset_mask,
            actor_version=self.actor_version,
        )

        if mode == "eval":
            return self._predict_eval_action_batch(
                env_obs=env_obs,
                route_info=route_info,
                env_ids=env_ids,
            )

        sample = self.runtime.sample_action_batch(
            env_obs=env_obs,
            routes=route_info.route_used,
            mode=mode,
            actor_version=self.actor_version,
            collect_replay=True,
        )
        logits = self.gate(sample.gate_snapshots)
        next_route, base_probability, behavior_probability, gate_logprob = (
            self._training_gate_decision(logits=logits)
        )
        self.route_tracker.emit(
            env_ids=env_ids,
            routes=next_route,
            source_chunk_ids=route_info.chunk_ids,
            episode_ids=route_info.episode_ids,
            actor_version=self.actor_version,
        )

        packed_kv = None
        if self.config.kv_replay.backend is GateKVReplayBackend.STORED:
            packed_kv = pack_gate_kv(sample.gate_snapshots, self.config.kv_replay)
            packed_bytes = _bytes_per_sample(packed_kv)
            byte_limit = self.config.kv_replay.max_bytes_per_sample
            if byte_limit is not None and bool((packed_bytes > byte_limit).any()):
                raise MemoryError(
                    "Stored Gate K/V exceeds `max_bytes_per_sample`: "
                    f"max={int(packed_bytes.max())}, limit={byte_limit}. "
                    "Select fewer Gate layers or explicitly test recompute."
                )
            kv_metadata = GateKVMetadata(
                layer_indices=tuple(int(item) for item in packed_kv.layer_indices),
                denoise_timesteps=packed_kv.denoise_timesteps,
                total_bytes=packed_bytes,
                storage_dtype=self.config.kv_replay.storage_dtype,
                tensor_shapes=tuple(
                    tuple(tensor.shape[1:])
                    for name, tensor in packed_kv.as_forward_inputs().items()
                    if not name.endswith("_layer_indices")
                    if tensor.ndim > 0 and tensor.shape[0] == batch_size
                ),
            )
        else:
            first_snapshot = sample.gate_snapshots[0]
            kv_metadata = GateKVMetadata(
                layer_indices=first_snapshot.layer_indices,
                denoise_timesteps=torch.stack(
                    [
                        snapshot.layers[0].denoise_timestep.detach().cpu()
                        for snapshot in sample.gate_snapshots
                    ],
                    dim=1,
                ),
                total_bytes=torch.zeros(batch_size, dtype=torch.long),
                storage_dtype=self.config.kv_replay.storage_dtype,
                tensor_shapes=(),
            )
        emitted_gate = GateDecisionRecord(
            next_route=next_route,
            base_probability=base_probability,
            behavior_probability=behavior_probability,
            old_logprob=gate_logprob,
            epsilon=torch.full_like(
                logits,
                self.config.gate_epsilon if mode == "train" else 0.0,
            ),
            temperature=torch.full_like(logits, self.config.gate_temperature),
            valid=torch.ones_like(next_route, dtype=torch.bool),
            source_chunk_ids=route_info.chunk_ids.to(next_route.device),
            episode_ids=route_info.episode_ids.to(next_route.device),
            actor_versions=torch.full_like(next_route, self.actor_version),
            kv_metadata=kv_metadata,
        )

        if compute_values:
            critic_obs = self.runtime.critic_observation(env_obs=env_obs)
            critic_result = self._require_critic().predict_value_batch(
                critic_obs,
                return_prefix=True,
            )
            if isinstance(critic_result, tuple):
                values, critic_prefix = critic_result
            else:
                values = critic_result
                critic_prefix = None
        else:
            values = torch.zeros(batch_size, device=device, dtype=torch.float32)
            critic_prefix = None
        values = _column_values(values, batch_size=batch_size)
        forward_inputs = {
            **sample.forward_inputs,
            "flow_chains": sample.flow_chains,
            "denoise_indices": sample.denoise_indices,
        }
        if packed_kv is not None:
            packed_inputs = packed_kv.as_forward_inputs()
            # Layer indices are static schema metadata, not batch data. Keeping
            # them in forward_inputs would make rollout sharding split [L] by B.
            packed_inputs.pop("gate_kv_layer_indices", None)
            forward_inputs.update(packed_inputs)
        if critic_prefix is not None:
            forward_inputs["critic_prefix"] = critic_prefix
        result = {
            "prev_logprobs": sample.old_flow_logprobs,
            "prev_values": values,
            "forward_inputs": forward_inputs,
            "route_info": route_info,
            "emitted_gate": emitted_gate,
        }
        return sample.actions, result

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        *,
        route_info: ChunkRouteRecord,
        emitted_gate: GateDecisionRecord,
        compute_values: bool = True,
        compute_base_logprobs: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        gate_device = next(self.gate.parameters()).device
        gate_dtype = next(self.gate.parameters()).dtype
        if self.config.kv_replay.backend is GateKVReplayBackend.STORED:
            packed_inputs = forward_inputs
            if "gate_kv_layer_indices" not in packed_inputs:
                if emitted_gate.kv_metadata is None:
                    raise ValueError("Stored Gate K/V replay requires metadata.")
                packed_inputs = dict(packed_inputs)
                packed_inputs["gate_kv_layer_indices"] = torch.tensor(
                    emitted_gate.kv_metadata.layer_indices,
                    dtype=torch.long,
                )
            packed = PackedGateKVTaps.from_forward_inputs(packed_inputs)
            snapshots = packed.materialize(device=gate_device, dtype=gate_dtype)
        else:
            snapshots = self.runtime.recompute_gate_snapshots(
                forward_inputs=forward_inputs,
                route_info=route_info,
            )
            snapshots = tuple(
                snapshot.detached().to(device=gate_device, dtype=gate_dtype)
                for snapshot in snapshots
            )
            if not snapshots:
                raise RuntimeError("Gate K/V recomputation returned no snapshots.")
        gate_logits = self.gate(snapshots)
        behavior = epsilon_mixture_bernoulli(
            gate_logits,
            temperature=emitted_gate.temperature.to(gate_logits.device),
            epsilon=emitted_gate.epsilon.to(gate_logits.device),
        )
        gate_routes = emitted_gate.next_route.to(gate_logits.device)
        gate_logprobs = behavior.log_prob(gate_routes)
        gate_entropy = behavior.distribution.entropy()

        # Recompute may temporarily restore the behavior LoRA. Finish that
        # no-grad observation reconstruction before building the live LoRA
        # autograd graph for Flow PPO.
        replay_kwargs = {
            "forward_inputs": forward_inputs,
            "route_info": route_info,
        }
        if compute_base_logprobs:
            replay_kwargs["compute_base_logprobs"] = True
        replay = self.runtime.replay_action_batch(**replay_kwargs)

        if compute_values:
            critic = self._require_critic()
            if "critic_prefix" in forward_inputs:
                values = critic.value_from_prefix(forward_inputs["critic_prefix"])
            else:
                critic_obs = self.runtime.critic_observation(
                    forward_inputs=forward_inputs
                )
                values = critic.predict_value_batch(critic_obs)
        else:
            values = torch.zeros(
                (gate_logits.shape[0], 1),
                device=gate_logits.device,
                dtype=torch.float32,
            )
        values = _column_values(values, batch_size=gate_logits.shape[0])
        result = {
            "logprobs": replay["flow_logprobs"],
            "flow_logprobs": replay["flow_logprobs"],
            "flow_entropy": replay["flow_entropy"],
            "gate_logprobs": gate_logprobs,
            "gate_entropy": gate_entropy,
            "gate_base_probabilities": behavior.base_idm_probability,
            "gate_behavior_probabilities": behavior.behavior_idm_probability,
            "values": values,
        }
        if compute_base_logprobs:
            if "base_uncond_kl" not in replay:
                raise KeyError(
                    "The FastWAM runtime did not return frozen-base transition KL."
                )
            result["base_uncond_kl"] = replay["base_uncond_kl"]
        return result

    def optimizer_parameter_groups(
        self,
        *,
        gate_lr: float,
        lora_lr: float,
        value_lr: float,
    ) -> list[dict[str, Any]]:
        """Return disjoint Gate, LoRA, and fresh-value-head optimizer groups."""

        critic = self._require_critic()
        groups = [
            {
                "name": "gate",
                "params": [
                    parameter
                    for parameter in self.gate.parameters()
                    if parameter.requires_grad
                ],
                "lr": gate_lr,
            },
            {
                "name": "uncond_lora",
                "params": [
                    parameter
                    for parameter in self.lora_adapter.lora_parameters()
                    if parameter.requires_grad
                ],
                "lr": lora_lr,
            },
            {
                "name": "value_head",
                "params": [
                    parameter
                    for parameter in critic.value_head.parameters()
                    if parameter.requires_grad
                ],
                "lr": value_lr,
            },
        ]
        empty_groups = [group["name"] for group in groups if not group["params"]]
        if empty_groups:
            raise RuntimeError(f"FastWAM optimizer groups are empty: {empty_groups}.")
        parameter_ids = [
            id(parameter) for group in groups for parameter in group["params"]
        ]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("FastWAM optimizer parameter groups overlap.")
        return groups

    def trainable_state_dict(self) -> dict[str, Any]:
        """Save only adaptive state plus delayed-route schedules."""

        critic = self._require_critic()
        return {
            "schema": "fastwam-adaptive-policy-v1",
            "actor_version": self.actor_version,
            "gate": self.gate.state_dict(),
            "lora": self.lora_adapter.lora_state_dict(),
            "value_head": critic.value_head.state_dict(),
            "route_tracker": self.route_tracker.state_dict(),
        }

    def load_trainable_state_dict(self, payload: dict[str, Any]) -> None:
        expected_keys = {
            "schema",
            "actor_version",
            "gate",
            "lora",
            "value_head",
            "route_tracker",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                f"FastWAM adaptive-policy checkpoint keys changed: {sorted(payload)}."
            )
        if payload.get("schema") != "fastwam-adaptive-policy-v1":
            raise ValueError("Unsupported FastWAM adaptive-policy checkpoint.")
        self.gate.load_state_dict(payload["gate"], strict=True)
        self.lora_adapter.load_lora_state_dict(payload["lora"], strict=True)
        if self.critic is not None:
            self.critic.value_head.load_state_dict(payload["value_head"], strict=True)
        self.route_tracker.load_state_dict(payload["route_tracker"])
        actor_version = int(payload["actor_version"])
        if actor_version < 0:
            raise ValueError("Checkpoint actor version must be non-negative.")
        # Exact resume restores a pending decision produced by these same
        # weights, so it must not be invalidated as a live policy update.
        self.actor_version = actor_version

    def rollout_runtime_state_dict(self) -> dict[str, Any]:
        """Serialize only delayed-route runtime state owned by rollout workers."""

        return {
            "schema": "fastwam-adaptive-rollout-policy-runtime-v1",
            "actor_version": self.actor_version,
            "route_tracker": self.route_tracker.state_dict(),
        }

    def load_rollout_runtime_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore rollout-owned delayed routes without duplicating trainables."""

        expected_keys = {"schema", "actor_version", "route_tracker"}
        if set(payload) != expected_keys:
            raise ValueError(
                f"FastWAM rollout-runtime policy keys changed: {sorted(payload)}."
            )
        if payload.get("schema") != ("fastwam-adaptive-rollout-policy-runtime-v1"):
            raise ValueError("Unsupported FastWAM rollout-runtime policy schema.")
        actor_version = int(payload["actor_version"])
        if actor_version < 0:
            raise ValueError("Rollout-runtime actor version must be non-negative.")
        self.route_tracker.load_state_dict(payload["route_tracker"])
        # The saved pending decision was emitted by this exact rollout version.
        # A subsequent actor-version change still invalidates it through
        # set_global_step before the first resumed chunk.
        self.actor_version = actor_version

    def load_eval_checkpoint(
        self,
        payload: Mapping[str, Any],
        *,
        expected_parent_checkpoint_sha256: str,
        expected_critic_parent_checkpoint_sha256: str | None = None,
    ) -> int:
        """Restore adaptive policy state from one actor-rank project checkpoint."""

        if payload.get("schema") != "fastwam-adaptive-rl-checkpoint-v1":
            raise ValueError("Unsupported FastWAM adaptive evaluation checkpoint.")
        expected_parent = str(expected_parent_checkpoint_sha256).strip().lower()
        if len(expected_parent) != 64 or any(
            character not in "0123456789abcdef" for character in expected_parent
        ):
            raise ValueError("Expected FastWAM parent SHA-256 is invalid.")
        if payload.get("parent_checkpoint_sha256") != expected_parent:
            raise ValueError(
                "FastWAM evaluation checkpoint parent hash mismatch: "
                f"expected {expected_parent}, got "
                f"{payload.get('parent_checkpoint_sha256')}."
            )

        contract = payload.get("contract")
        contract_model = (
            contract.get("model") if isinstance(contract, Mapping) else None
        )
        if not isinstance(contract_model, Mapping) or (
            str(contract_model.get("actor_checkpoint_sha256", "")).lower()
            != expected_parent
        ):
            raise ValueError(
                "FastWAM evaluation checkpoint contract has the wrong parent hash."
            )

        if self.critic is not None:
            expected_critic_parent = (
                str(expected_critic_parent_checkpoint_sha256 or "").strip().lower()
            )
            if len(expected_critic_parent) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_critic_parent
            ):
                raise ValueError("Expected pi0.5 critic parent SHA-256 is invalid.")
            if payload.get("critic_parent_checkpoint_sha256") != expected_critic_parent:
                raise ValueError(
                    "pi0.5 evaluation checkpoint parent hash mismatch: "
                    f"expected {expected_critic_parent}, got "
                    f"{payload.get('critic_parent_checkpoint_sha256')}."
                )
            contract_critic = contract_model.get("critic")
            if not isinstance(contract_critic, Mapping) or (
                str(contract_critic.get("backbone_checkpoint_sha256", "")).lower()
                != expected_critic_parent
            ):
                raise ValueError(
                    "FastWAM evaluation checkpoint contract has the wrong critic "
                    "parent hash."
                )

        policy_payload = payload.get("policy")
        if not isinstance(policy_payload, Mapping):
            raise ValueError(
                "FastWAM evaluation checkpoint is missing its policy payload."
            )
        outer_step = payload.get("step")
        inner_step = policy_payload.get("actor_version")
        if (
            outer_step is None
            or inner_step is None
            or isinstance(outer_step, bool)
            or isinstance(inner_step, bool)
            or int(outer_step) != int(inner_step)
        ):
            raise ValueError(
                "FastWAM evaluation checkpoint step does not match its policy version."
            )
        checkpoint_gate = policy_payload.get("gate")
        if not isinstance(checkpoint_gate, Mapping):
            raise ValueError(
                "FastWAM evaluation checkpoint is missing its Gate state mapping."
            )
        current_gate = self.gate.state_dict()
        current_keys = set(current_gate)
        checkpoint_keys = set(checkpoint_gate)
        missing_keys = sorted(current_keys - checkpoint_keys)
        unexpected_keys = sorted(checkpoint_keys - current_keys)
        shape_mismatches = sorted(
            key
            for key in current_keys & checkpoint_keys
            if getattr(current_gate[key], "shape", None)
            != getattr(checkpoint_gate[key], "shape", None)
        )
        if missing_keys or unexpected_keys or shape_mismatches:
            def preview(values: list[str]) -> list[str]:
                return values[:8]

            raise ValueError(
                "FastWAM evaluation checkpoint Gate architecture mismatch: "
                f"current_keys={len(current_keys)}, "
                f"checkpoint_keys={len(checkpoint_keys)}, "
                f"missing={preview(missing_keys)}, "
                f"unexpected={preview(unexpected_keys)}, "
                f"shape_mismatches={preview(shape_mismatches)}. "
                "Refusing to synthesize, replicate, or drop Gate weights."
            )

        self.load_trainable_state_dict(dict(policy_payload))
        if self.actor_version != int(outer_step):
            raise RuntimeError(
                "FastWAM evaluation checkpoint restored the wrong version."
            )
        return self.actor_version
