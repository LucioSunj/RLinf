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

"""Composite FastWAM actor, delayed Gate, and configurable critic policy."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

import torch
import torch.nn as nn
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.gate_transformer import epsilon_mixture_bernoulli
from fastwam.models.wan22.kv_tap import GateKVSnapshot

from rlinf.envs.action_contract import ActionExecutionTrace
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    GateKVMetadata,
)
from .critic import (
    CriticKind,
    FastWAMValueFeatures,
    critic_parent_checkpoint_sha256,
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
    critic_features: FastWAMValueFeatures | torch.Tensor | None = None
    action_execution_trace: ActionExecutionTrace | None = None

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
        if self.critic_features is not None:
            if isinstance(self.critic_features, FastWAMValueFeatures):
                feature_batch = self.critic_features.batch_size
            else:
                if self.critic_features.ndim != 2:
                    raise ValueError("Tensor critic features must have shape [B, D].")
                feature_batch = int(self.critic_features.shape[0])
            if feature_batch != batch_size:
                raise ValueError("Critic feature batch must match actions.")
        if (
            self.action_execution_trace is not None
            and self.action_execution_trace.batch_size != batch_size
        ):
            raise ValueError("Action trace batch must match actions.")


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
    ) -> dict[str, Any]: ...

    def critic_observation(
        self,
        *,
        env_obs: dict[str, Any] | None = None,
        forward_inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]: ...

    def critic_features(self, *, env_obs: dict[str, Any]) -> Any: ...

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
    eval_random_lag1_autocorrelation: float | None = None
    eval_routing_seed: int = 0
    eval_microbatch_size: int = 1
    eval_timing_cuda_synchronize: bool = False
    training_rollout_microbatch_size: int | None = None
    formal_training_sampling_seed: int | None = None
    decision_telemetry_enabled: bool = False
    kv_replay: GateKVReplayConfig = field(default_factory=GateKVReplayConfig)

    def __post_init__(self) -> None:
        if not math.isfinite(self.gate_epsilon) or not 0 <= self.gate_epsilon <= 1:
            raise ValueError("`gate_epsilon` must lie in [0, 1].")
        if not math.isfinite(self.gate_temperature) or self.gate_temperature <= 0:
            raise ValueError("`gate_temperature` must be positive.")
        if self.eval_microbatch_size < 1:
            raise ValueError("`eval_microbatch_size` must be positive.")
        if not isinstance(self.eval_timing_cuda_synchronize, bool):
            raise TypeError("`eval_timing_cuda_synchronize` must be a boolean.")
        if not isinstance(self.decision_telemetry_enabled, bool):
            raise TypeError("`decision_telemetry_enabled` must be a boolean.")
        training_microbatch = self.training_rollout_microbatch_size
        if training_microbatch is not None:
            if isinstance(training_microbatch, bool) or not isinstance(
                training_microbatch, int
            ):
                raise TypeError(
                    "`training_rollout_microbatch_size` must be an integer or null."
                )
            if training_microbatch < 1:
                raise ValueError(
                    "`training_rollout_microbatch_size` must be positive or null."
                )
        sampling_seed = self.formal_training_sampling_seed
        if sampling_seed is not None:
            if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
                raise TypeError(
                    "`formal_training_sampling_seed` must be an integer or null."
                )
            if sampling_seed < 0:
                raise ValueError(
                    "`formal_training_sampling_seed` must be non-negative or null."
                )
        evaluation = self.evaluation_routing
        object.__setattr__(self, "eval_routing_mode", evaluation.mode)
        object.__setattr__(self, "eval_idm_threshold", evaluation.idm_threshold)
        object.__setattr__(
            self,
            "eval_random_idm_probability",
            evaluation.random_idm_probability,
        )
        object.__setattr__(
            self,
            "eval_random_lag1_autocorrelation",
            evaluation.random_lag1_autocorrelation,
        )
        object.__setattr__(self, "eval_routing_seed", evaluation.routing_seed)

    @property
    def evaluation_routing(self) -> EvaluationRoutingConfig:
        """Return the validated pure evaluation-routing configuration."""

        return EvaluationRoutingConfig(
            mode=self.eval_routing_mode,
            idm_threshold=self.eval_idm_threshold,
            random_idm_probability=self.eval_random_idm_probability,
            random_lag1_autocorrelation=(self.eval_random_lag1_autocorrelation),
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


def _flip_gate_snapshot_current_modes(
    snapshots: tuple[GateKVSnapshot, ...],
) -> tuple[GateKVSnapshot, ...]:
    """Flip only the current-regime bit in detached Gate inputs."""

    flipped = []
    for snapshot in snapshots:
        layers = []
        for layer in snapshot.layers:
            modes = tuple(
                PolicyRegime.UNCOND if mode is PolicyRegime.IDM else PolicyRegime.IDM
                for mode in layer.current_mode
            )
            layers.append(replace(layer, current_mode=modes))
        flipped.append(replace(snapshot, layers=tuple(layers)))
    return tuple(flipped)


def _sample_epsilon_mixture_with_component(
    *,
    base_probability: torch.Tensor,
    behavior_probability: torch.Tensor,
    epsilon: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one epsilon mixture and expose its exploration component.

    One uniform draw is partitioned into base-IDM, exploration-IDM,
    base-UNCOND, and exploration-UNCOND mass. The resulting route therefore
    has exactly ``behavior_probability`` while the returned boolean states
    whether the draw came from the epsilon component.
    """

    if not (base_probability.shape == behavior_probability.shape == epsilon.shape):
        raise ValueError("Gate mixture tensors must have identical shapes.")
    draw = torch.rand(
        base_probability.shape,
        dtype=base_probability.dtype,
        device=base_probability.device,
        generator=generator,
    )
    route_is_idm = draw < behavior_probability
    base_idm_upper = (1 - epsilon) * base_probability
    base_uncond_upper = behavior_probability + (1 - epsilon) * (1 - base_probability)
    exploration_forced = torch.where(
        route_is_idm,
        draw >= base_idm_upper,
        draw >= base_uncond_upper,
    )
    return route_is_idm.to(dtype=torch.long), exploration_forced


def _column_values(values: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if values.shape == (batch_size,):
        return values[:, None]
    if values.shape == (batch_size, 1):
        return values
    raise ValueError(
        "FastWAM critic values must have shape [B] or [B, 1], got "
        f"{tuple(values.shape)} for batch size {batch_size}."
    )


def _formal_training_sample_seed(
    *,
    base_seed: int,
    domain: str,
    environment_id: int,
    episode_id: int,
    chunk_id: int,
    actor_version: int,
) -> int:
    """Derive one stage-independent local seed for formal training sampling."""

    payload = b"\0".join(
        (
            b"fastwam-formal-training-sampling-v1",
            str(int(base_seed)).encode("ascii"),
            str(domain).encode("utf-8"),
            str(int(environment_id)).encode("ascii"),
            str(int(episode_id)).encode("ascii"),
            str(int(chunk_id)).encode("ascii"),
            str(int(actor_version)).encode("ascii"),
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


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
                "The critic is intentionally absent from standalone evaluation."
            )
        return self.critic

    def _critic_kind(self) -> CriticKind:
        """Return the configured critic kind, preserving legacy test doubles."""

        critic = self._require_critic()
        return CriticKind.parse(
            getattr(critic, "kind", CriticKind.PI0_5_VALUE_AFTER_VLM)
        )

    def predict_value_batch(self, env_obs: dict[str, Any]) -> torch.Tensor:
        """Predict bootstrap values through the selected critic backend."""

        critic = self._require_critic()
        if self._critic_kind() is CriticKind.FASTWAM_CURRENT_FRAME_VALUE:
            features = self.runtime.critic_features(env_obs=env_obs)
            values = critic.value_from_features(features)
        else:
            critic_obs = self.runtime.critic_observation(env_obs=env_obs)
            values = critic.predict_value_batch(critic_obs)
        batch_size = int(env_obs["states"].shape[0])
        return _column_values(values, batch_size=batch_size)

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

    @staticmethod
    def _decision_identity_metadata(
        env_obs: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor | None]:
        """Return optional environment identity carried with each decision."""

        source_keys = {
            "task_ids": "_fastwam_task_ids",
            "trial_ids": "_fastwam_trial_ids",
            "reset_state_ids": "_fastwam_reset_state_ids",
        }
        present = {
            field_name: env_obs.get(source_key)
            for field_name, source_key in source_keys.items()
        }
        if any(value is not None for value in present.values()) and any(
            value is None for value in present.values()
        ):
            raise ValueError(
                "FastWAM decision identity requires task, trial, and reset-state IDs."
            )
        result: dict[str, torch.Tensor | None] = {}
        for field_name, value in present.items():
            if value is None:
                result[field_name] = None
                continue
            tensor = torch.as_tensor(value, device=device, dtype=torch.long)
            if tensor.shape != (batch_size,):
                raise ValueError(
                    f"FastWAM {field_name} must have shape [{batch_size}], got "
                    f"{tuple(tensor.shape)}."
                )
            if bool((tensor < 0).any().item()):
                raise ValueError(f"FastWAM {field_name} must be non-negative.")
            result[field_name] = tensor
        return result

    def _mode_flip_delta(
        self,
        *,
        snapshots: tuple[GateKVSnapshot, ...],
        base_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Return flipped-current-mode minus observed base IDM probability."""

        with torch.no_grad():
            flipped_logits = self.gate(_flip_gate_snapshot_current_modes(snapshots))
            flipped = epsilon_mixture_bernoulli(
                flipped_logits,
                temperature=self.config.gate_temperature,
                epsilon=0.0,
            ).base_idm_probability
        if flipped.shape != base_probability.shape:
            raise ValueError("Mode-flip Gate output changed the decision shape.")
        return (flipped - base_probability.detach()).detach()

    def _training_gate_decision(
        self,
        *,
        logits: torch.Tensor,
        sampling_seeds: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        behavior = epsilon_mixture_bernoulli(
            logits,
            temperature=self.config.gate_temperature,
            epsilon=self.config.gate_epsilon,
        )
        if sampling_seeds is None:
            if self.config.decision_telemetry_enabled:
                epsilon = torch.full_like(logits, self.config.gate_epsilon)
                route, exploration_forced = _sample_epsilon_mixture_with_component(
                    base_probability=behavior.base_idm_probability,
                    behavior_probability=behavior.behavior_idm_probability,
                    epsilon=epsilon,
                )
            else:
                # Preserve the established sampler and RNG stream unless the
                # explicit telemetry profile requests mixture attribution.
                route = behavior.sample()
                exploration_forced = None
        else:
            seeds = torch.as_tensor(sampling_seeds, device="cpu", dtype=torch.long)
            if seeds.shape != logits.shape:
                raise ValueError(
                    "Formal Gate sampling seeds must match Gate logits: "
                    f"{tuple(seeds.shape)} != {tuple(logits.shape)}."
                )
            routes = []
            exploration_flags = []
            epsilon = torch.full_like(logits, self.config.gate_epsilon)
            for index, (probability, seed) in enumerate(
                zip(
                    behavior.behavior_idm_probability.reshape(-1),
                    seeds.reshape(-1),
                    strict=True,
                )
            ):
                generator = torch.Generator(device=probability.device)
                generator.manual_seed(int(seed.item()))
                if self.config.decision_telemetry_enabled:
                    sampled_route, sampled_exploration = (
                        _sample_epsilon_mixture_with_component(
                            base_probability=behavior.base_idm_probability.reshape(-1)[
                                index
                            ].reshape(1),
                            behavior_probability=probability.reshape(1),
                            epsilon=epsilon.reshape(-1)[index].reshape(1),
                            generator=generator,
                        )
                    )
                    exploration_flags.append(sampled_exploration)
                else:
                    sampled_route = torch.bernoulli(
                        probability.reshape(1),
                        generator=generator,
                    )
                routes.append(sampled_route)
            route = torch.cat(routes).reshape_as(logits).to(dtype=torch.long)
            exploration_forced = (
                torch.cat(exploration_flags).reshape_as(logits)
                if self.config.decision_telemetry_enabled
                else None
            )
        logprob = behavior.log_prob(route)
        return (
            route,
            behavior.base_idm_probability,
            behavior.behavior_idm_probability,
            logprob,
            exploration_forced,
        )

    def _formal_training_sampling_seeds(
        self,
        *,
        env_ids: torch.Tensor,
        route_info: ChunkRouteRecord,
    ) -> dict[str, torch.Tensor] | None:
        """Build per-sample RNG streams that are invariant to stage sharding."""

        base_seed = self.config.formal_training_sampling_seed
        if base_seed is None:
            return None
        metadata = (
            env_ids.detach().cpu().reshape(-1),
            route_info.episode_ids.detach().cpu().reshape(-1),
            route_info.chunk_ids.detach().cpu().reshape(-1),
        )
        if len({int(values.numel()) for values in metadata}) != 1:
            raise ValueError("Formal training sampling metadata batch sizes differ.")
        environments, episodes, chunks = metadata
        result: dict[str, torch.Tensor] = {}
        for domain in ("action", "idm", "gate"):
            result[domain] = torch.tensor(
                [
                    _formal_training_sample_seed(
                        base_seed=base_seed,
                        domain=domain,
                        environment_id=int(environment_id),
                        episode_id=int(episode_id),
                        chunk_id=int(chunk_id),
                        actor_version=self.actor_version,
                    )
                    for environment_id, episode_id, chunk_id in zip(
                        environments.tolist(),
                        episodes.tolist(),
                        chunks.tolist(),
                        strict=True,
                    )
                ],
                dtype=torch.long,
            )
        return result

    def _evaluation_gate_decision(
        self,
        *,
        logits: torch.Tensor,
        env_ids: torch.Tensor,
        episode_ids: torch.Tensor,
        source_chunk_ids: torch.Tensor,
        current_routes: torch.Tensor,
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
            current_routes=current_routes,
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

    @staticmethod
    def _cat_training_tensors(values: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate one sharding-invariant training tensor batch."""

        if not values or any(value.ndim == 0 for value in values):
            raise ValueError("Training rollout tensors must be batch-first.")
        reference = values[0]
        if any(
            value.device != reference.device or value.dtype != reference.dtype
            for value in values[1:]
        ):
            raise ValueError("Training rollout tensor shards disagree on placement.")
        if reference.device.type == "cpu" and all(
            value.is_pinned() for value in values
        ):
            output_shape = list(reference.shape)
            output_shape[0] = sum(int(value.shape[0]) for value in values)
            output = torch.empty(
                output_shape,
                dtype=reference.dtype,
                device="cpu",
                pin_memory=True,
            )
            return torch.cat(values, dim=0, out=output)
        return torch.cat(values, dim=0)

    def _merge_training_microbatch_results(
        self,
        actions: list[torch.Tensor],
        results: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Restore global-environment order after invariant rollout microbatches."""

        if not actions or len(actions) != len(results):
            raise ValueError("Training rollout microbatch results are incomplete.")
        forward_key_sets = [set(result["forward_inputs"]) for result in results]
        if any(keys != forward_key_sets[0] for keys in forward_key_sets[1:]):
            raise ValueError("Training rollout microbatches changed replay fields.")
        forward_inputs = {
            key: self._cat_training_tensors(
                [result["forward_inputs"][key] for result in results]
            )
            for key in sorted(forward_key_sets[0])
        }
        traces = [result.get("action_execution_trace") for result in results]
        if any(trace is None for trace in traces) and not all(
            trace is None for trace in traces
        ):
            raise ValueError("Training rollout microbatches returned partial traces.")
        return self._cat_training_tensors(actions), {
            "prev_logprobs": self._cat_training_tensors(
                [result["prev_logprobs"] for result in results]
            ),
            "prev_values": self._cat_training_tensors(
                [result["prev_values"] for result in results]
            ),
            "forward_inputs": forward_inputs,
            "route_info": ChunkRouteRecord.cat(
                [result["route_info"] for result in results]
            ),
            "emitted_gate": GateDecisionRecord.cat(
                [result["emitted_gate"] for result in results]
            ),
            "action_execution_trace": (
                None
                if all(trace is None for trace in traces)
                else ActionExecutionTrace.cat(
                    [trace for trace in traces if trace is not None], dim=0
                )
            ),
        }

    def _predict_training_microbatches(
        self,
        *,
        env_obs: dict[str, Any],
        batch_size: int,
        compute_values: bool,
        microbatch_size: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run training in stable global-environment order across stage shards."""

        actions: list[torch.Tensor] = []
        results: list[dict[str, Any]] = []
        for start in range(0, batch_size, microbatch_size):
            end = min(start + microbatch_size, batch_size)
            action, result = self.predict_action_batch(
                self._slice_env_obs(
                    env_obs,
                    start=start,
                    end=end,
                    batch_size=batch_size,
                ),
                mode="train",
                compute_values=compute_values,
            )
            actions.append(action)
            results.append(result)
        return self._merge_training_microbatch_results(actions, results)

    @torch.no_grad()
    def _predict_eval_action_batch(
        self,
        *,
        env_obs: dict[str, Any],
        route_info: ChunkRouteRecord,
        env_ids: torch.Tensor,
        identity_metadata: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run deterministic evaluation without constructing PPO replay."""

        batch_size = route_info.route_used.shape[0]
        actions = []
        decisions = []
        action_traces: list[ActionExecutionTrace | None] = []
        microbatch = self.config.eval_microbatch_size
        measure_gate_latency = self.config.eval_timing_cuda_synchronize
        gate_latencies: list[torch.Tensor] = []
        gate_h2d_latencies: list[torch.Tensor] = []
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
            gate_parameter = next(self.gate.parameters(), None)
            timing_device = (
                sample.actions.device
                if gate_parameter is None
                else gate_parameter.device
            )
            if measure_gate_latency and timing_device.type == "cuda":
                torch.cuda.synchronize(timing_device)
            gate_started_at = time.perf_counter()
            logits = self.gate(sample.gate_snapshots)
            decision = self._evaluation_gate_decision(
                logits=logits,
                env_ids=env_ids[start:end],
                episode_ids=route_info.episode_ids[start:end],
                source_chunk_ids=route_info.chunk_ids[start:end],
                current_routes=route_info.route_used[start:end],
            )
            mode_flip_delta = (
                self._mode_flip_delta(
                    snapshots=sample.gate_snapshots,
                    base_probability=decision[1],
                )
                if self.config.decision_telemetry_enabled
                else None
            )
            if measure_gate_latency:
                if logits.device.type == "cuda":
                    torch.cuda.synchronize(logits.device)
                elapsed = time.perf_counter() - gate_started_at
                gate_latencies.append(
                    torch.full((end - start,), elapsed, dtype=torch.float64)
                )
                gate_h2d_latencies.append(torch.zeros(end - start, dtype=torch.float64))
            decisions.append((*decision, mode_flip_delta))
            actions.append(sample.actions)
            action_traces.append(sample.action_execution_trace)

        next_route = torch.cat([item[0] for item in decisions], dim=0)
        base_probability = torch.cat([item[1] for item in decisions], dim=0)
        behavior_probability = torch.cat([item[2] for item in decisions], dim=0)
        gate_logprob = torch.cat([item[3] for item in decisions], dim=0)
        evaluation_selection = EvaluationRouteSelection.cat(
            [item[4] for item in decisions]
        )
        mode_flip_delta = (
            torch.cat([item[5] for item in decisions if item[5] is not None], dim=0)
            if self.config.decision_telemetry_enabled
            else None
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
            exploration_forced=(
                torch.zeros_like(next_route, dtype=torch.bool)
                if self.config.decision_telemetry_enabled
                else None
            ),
            mode_flip_delta=mode_flip_delta,
            environment_ids=(
                env_ids.to(next_route.device)
                if self.config.decision_telemetry_enabled
                else None
            ),
            task_ids=(
                identity_metadata["task_ids"].to(next_route.device)
                if self.config.decision_telemetry_enabled
                and identity_metadata["task_ids"] is not None
                else None
            ),
            trial_ids=(
                identity_metadata["trial_ids"].to(next_route.device)
                if self.config.decision_telemetry_enabled
                and identity_metadata["trial_ids"] is not None
                else None
            ),
            reset_state_ids=(
                identity_metadata["reset_state_ids"].to(next_route.device)
                if self.config.decision_telemetry_enabled
                and identity_metadata["reset_state_ids"] is not None
                else None
            ),
            kv_metadata=None,
        )
        if any(trace is None for trace in action_traces) and not all(
            trace is None for trace in action_traces
        ):
            raise ValueError(
                "FastWAM eval microbatches returned partial Action traces."
            )
        action_execution_trace = (
            None
            if all(trace is None for trace in action_traces)
            else ActionExecutionTrace.cat(
                [trace for trace in action_traces if trace is not None], dim=0
            )
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
            "gate_latency_seconds": (
                torch.cat(gate_latencies) if measure_gate_latency else None
            ),
            "gate_h2d_seconds": (
                torch.cat(gate_h2d_latencies) if measure_gate_latency else None
            ),
            "action_execution_trace": action_execution_trace,
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
        training_microbatch = self.config.training_rollout_microbatch_size
        if (
            mode == "train"
            and training_microbatch is not None
            and batch_size > training_microbatch
        ):
            return self._predict_training_microbatches(
                env_obs=env_obs,
                batch_size=batch_size,
                compute_values=compute_values,
                microbatch_size=training_microbatch,
            )
        device = env_obs["states"].device
        env_ids, reset_mask = self._routing_metadata(env_obs, batch_size, device)
        identity_metadata = (
            self._decision_identity_metadata(
                env_obs,
                batch_size=batch_size,
                device=device,
            )
            if self.config.decision_telemetry_enabled
            else {"task_ids": None, "trial_ids": None, "reset_state_ids": None}
        )
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
                identity_metadata=identity_metadata,
            )

        sampling_seeds = self._formal_training_sampling_seeds(
            env_ids=env_ids,
            route_info=route_info,
        )
        runtime_env_obs = env_obs
        if sampling_seeds is not None:
            seed_fields = {
                "_fastwam_action_noise_seeds": sampling_seeds["action"],
                "_fastwam_idm_noise_seeds": sampling_seeds["idm"],
            }
            collisions = sorted(set(env_obs).intersection(seed_fields))
            if collisions:
                raise ValueError(
                    "Formal training refuses caller-supplied sampling seeds: "
                    f"{collisions}."
                )
            runtime_env_obs = {**env_obs, **seed_fields}
        sample = self.runtime.sample_action_batch(
            env_obs=runtime_env_obs,
            routes=route_info.route_used,
            mode=mode,
            actor_version=self.actor_version,
            collect_replay=True,
        )
        logits = self.gate(sample.gate_snapshots)
        (
            next_route,
            base_probability,
            behavior_probability,
            gate_logprob,
            exploration_forced,
        ) = self._training_gate_decision(
            logits=logits,
            sampling_seeds=(None if sampling_seeds is None else sampling_seeds["gate"]),
        )
        mode_flip_delta = (
            self._mode_flip_delta(
                snapshots=sample.gate_snapshots,
                base_probability=base_probability,
            )
            if self.config.decision_telemetry_enabled
            else None
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
            exploration_forced=exploration_forced,
            mode_flip_delta=mode_flip_delta,
            environment_ids=(
                env_ids.to(next_route.device)
                if self.config.decision_telemetry_enabled
                else None
            ),
            task_ids=(
                identity_metadata["task_ids"].to(next_route.device)
                if self.config.decision_telemetry_enabled
                and identity_metadata["task_ids"] is not None
                else None
            ),
            trial_ids=(
                identity_metadata["trial_ids"].to(next_route.device)
                if self.config.decision_telemetry_enabled
                and identity_metadata["trial_ids"] is not None
                else None
            ),
            reset_state_ids=(
                identity_metadata["reset_state_ids"].to(next_route.device)
                if self.config.decision_telemetry_enabled
                and identity_metadata["reset_state_ids"] is not None
                else None
            ),
            kv_metadata=kv_metadata,
        )

        if compute_values:
            critic = self._require_critic()
            if self._critic_kind() is CriticKind.FASTWAM_CURRENT_FRAME_VALUE:
                if sample.critic_features is None:
                    raise RuntimeError(
                        "FastWAM current-frame critic rollout returned no features."
                    )
                critic_features = sample.critic_features
                values = critic.value_from_features(critic_features)
            else:
                critic_obs = self.runtime.critic_observation(env_obs=env_obs)
                critic_result = critic.predict_value_batch(
                    critic_obs,
                    return_prefix=True,
                )
                if isinstance(critic_result, tuple):
                    values, critic_features = critic_result
                else:
                    values = critic_result
                    critic_features = None
        else:
            values = torch.zeros(batch_size, device=device, dtype=torch.float32)
            critic_features = None
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
        if (
            critic_features is not None
            and self._critic_kind() is not CriticKind.FASTWAM_CURRENT_FRAME_VALUE
        ):
            replay_key = getattr(critic, "replay_feature_key", "critic_prefix")
            forward_inputs[replay_key] = critic_features
        result = {
            "prev_logprobs": sample.old_flow_logprobs,
            "prev_values": values,
            "forward_inputs": forward_inputs,
            "route_info": route_info,
            "emitted_gate": emitted_gate,
            "action_execution_trace": sample.action_execution_trace,
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
        # Materialize Gate K/V in the declared storage dtype, not in the Gate's
        # parameter dtype. The Gate holds FP32 master weights, but upcasting
        # every layer's K/V to FP32 here would double replay memory and break
        # the declared per-sample and tier byte limits. `DirectKVAttention`
        # casts each layer's bank to the query dtype transiently instead.
        gate_dtype = self.config.kv_replay.torch_dtype
        batch_size = int(route_info.route_used.shape[0])
        if self.config.kv_replay.backend is GateKVReplayBackend.STORED:
            from .tiered_kv_store import GATE_KV_BATCH_INDICES

            packed_inputs = forward_inputs
            batch_indices = packed_inputs.get(GATE_KV_BATCH_INDICES)
            if batch_indices is None:
                batch_indices = torch.arange(
                    batch_size,
                    device=gate_device,
                    dtype=torch.long,
                )
            else:
                batch_indices = batch_indices.to(
                    device=gate_device,
                    dtype=torch.long,
                )
            if batch_indices.ndim != 1:
                raise ValueError("Sparse Gate K/V batch indices must be 1D.")
            if (
                bool(((batch_indices < 0) | (batch_indices >= batch_size)).any())
                or batch_indices.unique().numel() != batch_indices.numel()
            ):
                raise ValueError("Sparse Gate K/V batch indices are invalid.")
            if batch_indices.numel() and "gate_kv_layer_indices" not in packed_inputs:
                if emitted_gate.kv_metadata is None:
                    raise ValueError("Stored Gate K/V replay requires metadata.")
                packed_inputs = dict(packed_inputs)
                packed_inputs["gate_kv_layer_indices"] = torch.tensor(
                    emitted_gate.kv_metadata.layer_indices,
                    dtype=torch.long,
                )
            if batch_indices.numel():
                packed = PackedGateKVTaps.from_forward_inputs(packed_inputs)
                if packed.batch_size != batch_indices.numel():
                    raise ValueError(
                        "Sparse Gate K/V payload count does not match batch indices."
                    )
                snapshots = packed.materialize(device=gate_device, dtype=gate_dtype)
                selected_logits = self.gate(snapshots)
            else:
                selected_logits = torch.empty(
                    0,
                    device=gate_device,
                    dtype=gate_dtype,
                )
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
            batch_indices = torch.arange(
                batch_size,
                device=gate_device,
                dtype=torch.long,
            )
            selected_logits = self.gate(snapshots)

        selected_temperature = emitted_gate.temperature.to(gate_device)[batch_indices]
        selected_epsilon = emitted_gate.epsilon.to(gate_device)[batch_indices]
        selected_routes = emitted_gate.next_route.to(gate_device)[batch_indices]
        if batch_indices.numel():
            behavior = epsilon_mixture_bernoulli(
                selected_logits,
                temperature=selected_temperature,
                epsilon=selected_epsilon,
            )
            selected_logprobs = behavior.log_prob(selected_routes)
            selected_entropy = behavior.distribution.entropy()
            selected_base = behavior.base_idm_probability
            selected_behavior = behavior.behavior_idm_probability
        else:
            selected_logprobs = selected_entropy = selected_logits
            selected_base = selected_behavior = selected_logits

        def scatter_selected(values: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                batch_size,
                device=gate_device,
                dtype=values.dtype,
            ).index_copy(0, batch_indices, values)

        gate_logits = scatter_selected(selected_logits)
        gate_logprobs = scatter_selected(selected_logprobs)
        gate_entropy = scatter_selected(selected_entropy)
        gate_base_probabilities = scatter_selected(selected_base)
        gate_behavior_probabilities = scatter_selected(selected_behavior)

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
            critic_kind = self._critic_kind()
            if critic_kind is CriticKind.FASTWAM_CURRENT_FRAME_VALUE:
                if "critic_features" not in replay:
                    raise KeyError(
                        "FastWAM critic replay returned no reconstructed value K/V."
                    )
                values = critic.value_from_features(replay["critic_features"])
            else:
                replay_key = getattr(critic, "replay_feature_key", "critic_prefix")
                if replay_key in forward_inputs:
                    features = forward_inputs[replay_key]
                    if hasattr(critic, "value_from_features"):
                        values = critic.value_from_features(features)
                    else:
                        values = critic.value_from_prefix(features)
                else:
                    critic_obs = self.runtime.critic_observation(
                        forward_inputs=forward_inputs
                    )
                    values = critic.predict_value_batch(critic_obs)
        else:
            values = torch.zeros(
                (batch_size, 1),
                device=gate_logits.device,
                dtype=torch.float32,
            )
        values = _column_values(values, batch_size=batch_size)
        result = {
            "logprobs": replay["flow_logprobs"],
            "flow_logprobs": replay["flow_logprobs"],
            "flow_entropy": replay["flow_entropy"],
            "gate_logprobs": gate_logprobs,
            "gate_entropy": gate_entropy,
            "gate_base_probabilities": gate_base_probabilities,
            "gate_behavior_probabilities": gate_behavior_probabilities,
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
            critic_kind = self._critic_kind()
            expected_critic_parent = expected_critic_parent_checkpoint_sha256
            if expected_critic_parent is not None:
                expected_critic_parent = str(expected_critic_parent).strip().lower()
            if critic_kind is CriticKind.PI0_5_VALUE_AFTER_VLM and (
                expected_critic_parent is None
                or len(expected_critic_parent) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_critic_parent
                )
            ):
                raise ValueError("Expected pi0.5 critic parent SHA-256 is invalid.")
            if payload.get("critic_parent_checkpoint_sha256") != expected_critic_parent:
                raise ValueError(
                    "FastWAM evaluation critic parent mismatch: "
                    f"expected {expected_critic_parent}, got "
                    f"{payload.get('critic_parent_checkpoint_sha256')}."
                )
            contract_critic = contract_model.get("critic")
            if not isinstance(contract_critic, Mapping):
                raise ValueError(
                    "FastWAM evaluation checkpoint contract has no critic config."
                )
            contract_kind = CriticKind.parse(
                contract_critic.get("kind", CriticKind.PI0_5_VALUE_AFTER_VLM)
            )
            if contract_kind is not critic_kind or (
                critic_parent_checkpoint_sha256(contract_critic)
                != expected_critic_parent
            ):
                raise ValueError(
                    "FastWAM evaluation checkpoint critic contract mismatch."
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
