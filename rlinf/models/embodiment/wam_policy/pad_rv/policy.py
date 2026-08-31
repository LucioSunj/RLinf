# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD-Frozen policy subclass with same-chunk routing and no actor replay."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn as nn
from fastwam.models.wan22.gate_transformer import epsilon_mixture_bernoulli

from rlinf.envs.action_contract import ActionExecutionTrace
from rlinf.models.embodiment.wam_policy.adaptive_policy import (
    FastWAMAdaptivePolicy,
    FastWAMAdaptivePolicyConfig,
    _column_values,
)
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
)
from rlinf.models.embodiment.wam_policy.critic import CriticKind

from .gate import deserialize_condition_features
from .routing_state import CurrentStepRouteTracker
from .runtime import PadFrozenLiberoRuntime


class PadFrozenPolicy(FastWAMAdaptivePolicy):
    """A config-selected policy whose only trainables are Gate and value heads."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        uncond_action_expert: nn.Module,
        runtime: PadFrozenLiberoRuntime,
        gate: nn.Module,
        critic: nn.Module | None,
        config: FastWAMAdaptivePolicyConfig | None = None,
    ) -> None:
        # Deliberately do not call the legacy constructor: it enforces dynamic
        # LoRA ownership and delayed-route state. We inherit its stable helper
        # methods while owning a separate Stage 1 initialization contract.
        nn.Module.__init__(self)
        if not isinstance(runtime, PadFrozenLiberoRuntime):
            raise TypeError("PAD-Frozen requires PadFrozenLiberoRuntime.")
        self.actor = actor
        self.uncond_action_expert = uncond_action_expert
        self.runtime = runtime
        self.lora_adapter = None
        self.gate = gate
        self.critic = critic
        self.config = config or FastWAMAdaptivePolicyConfig()
        self.route_tracker = CurrentStepRouteTracker()
        self.actor_version = 0
        self.actor.requires_grad_(False)
        self.uncond_action_expert.requires_grad_(False)
        self.actor.eval()
        self.uncond_action_expert.eval()

    def train(self, mode: bool = True) -> "PadFrozenPolicy":
        nn.Module.train(self, mode)
        self.actor.eval()
        self.uncond_action_expert.eval()
        self.gate.train(mode)
        if self.critic is not None:
            self.critic.train(mode)
        return self

    def capture_gate_recompute_reference(self) -> None:
        """PAD condition replay has no behavior-LoRA reference."""

    def set_global_step(self, version: int) -> None:
        """Advance current-step policy version without forcing a future route."""

        version = int(version)
        if version < 0:
            raise ValueError("PAD actor version must be non-negative.")
        self.actor_version = version

    def _merge_training_microbatch_results(
        self,
        actions: list[torch.Tensor],
        results: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Merge PAD shards without inventing an Action-PPO logprob field."""

        if not actions or len(actions) != len(results):
            raise ValueError("PAD training rollout microbatches are incomplete.")
        if any(result.get("prev_logprobs") is not None for result in results):
            raise ValueError("PAD-Frozen cannot return Action-PPO logprobs.")
        forward_key_sets = [set(result["forward_inputs"]) for result in results]
        if any(keys != forward_key_sets[0] for keys in forward_key_sets[1:]):
            raise ValueError("PAD training microbatches changed replay fields.")
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
            raise ValueError("PAD training microbatches returned partial traces.")
        return self._cat_training_tensors(actions), {
            "prev_logprobs": None,
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

    def _predict_current_step(
        self,
        *,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"],
        compute_values: bool,
        env_ids: torch.Tensor,
        reset_mask: torch.Tensor,
        identity_metadata: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        identity = self.route_tracker.prepare(
            env_ids=env_ids,
            reset_mask=reset_mask,
            actor_version=self.actor_version,
        )
        sampling_seeds = (
            self._formal_training_sampling_seeds(
                env_ids=env_ids,
                route_info=identity,
            )
            if mode == "train"
            else None
        )
        runtime_obs = env_obs
        if sampling_seeds is not None:
            seed_fields = {
                "_fastwam_action_noise_seeds": sampling_seeds["action"],
                "_fastwam_idm_noise_seeds": sampling_seeds["idm"],
            }
            collisions = sorted(set(env_obs).intersection(seed_fields))
            if collisions:
                raise ValueError(f"PAD refuses caller-supplied seeds: {collisions}.")
            runtime_obs = {**env_obs, **seed_fields}

        prepared = self.runtime.prepare_route_context(env_obs=runtime_obs)
        gate_parameter = next(self.gate.parameters())
        gate_features = prepared.gate_features.to(device=gate_parameter.device)
        measure_latency = self.config.eval_timing_cuda_synchronize and mode == "eval"
        if measure_latency and gate_parameter.device.type == "cuda":
            torch.cuda.synchronize(gate_parameter.device)
        started = time.perf_counter()
        logits = self.gate(gate_features)
        selection = None
        if mode == "train":
            routes, base, behavior, logprob, exploration = self._training_gate_decision(
                logits=logits,
                sampling_seeds=(
                    None if sampling_seeds is None else sampling_seeds["gate"]
                ),
            )
            epsilon = torch.full_like(logits, self.config.gate_epsilon)
        else:
            routes, base, behavior, logprob, selection = self._evaluation_gate_decision(
                logits=logits,
                env_ids=env_ids,
                episode_ids=identity.episode_ids,
                source_chunk_ids=identity.chunk_ids,
                current_routes=self.route_tracker.previous_routes(identity),
            )
            exploration = (
                torch.zeros_like(routes, dtype=torch.bool)
                if self.config.decision_telemetry_enabled
                else None
            )
            epsilon = torch.zeros_like(logits)
        gate_latency = None
        if measure_latency:
            if logits.device.type == "cuda":
                torch.cuda.synchronize(logits.device)
            gate_latency = torch.full(
                (prepared.batch_size,),
                time.perf_counter() - started,
                dtype=torch.float64,
            )
        route_info = self.route_tracker.commit(identity=identity, routes=routes)
        sample = self.runtime.sample_prepared_action_batch(
            prepared=prepared,
            env_obs=runtime_obs,
            routes=route_info.route_used,
            mode=mode,
            actor_version=self.actor_version,
        )
        emitted = GateDecisionRecord(
            next_route=route_info.route_used,
            base_probability=base,
            behavior_probability=behavior,
            old_logprob=logprob,
            epsilon=epsilon,
            temperature=torch.full_like(logits, self.config.gate_temperature),
            valid=torch.ones_like(routes, dtype=torch.bool),
            source_chunk_ids=route_info.chunk_ids,
            episode_ids=route_info.episode_ids,
            actor_versions=route_info.actor_versions,
            exploration_forced=exploration,
            mode_flip_delta=None,
            environment_ids=(
                env_ids.to(routes.device)
                if self.config.decision_telemetry_enabled
                else None
            ),
            task_ids=(
                identity_metadata["task_ids"].to(routes.device)
                if self.config.decision_telemetry_enabled
                and identity_metadata["task_ids"] is not None
                else None
            ),
            trial_ids=(
                identity_metadata["trial_ids"].to(routes.device)
                if self.config.decision_telemetry_enabled
                and identity_metadata["trial_ids"] is not None
                else None
            ),
            reset_state_ids=(
                identity_metadata["reset_state_ids"].to(routes.device)
                if self.config.decision_telemetry_enabled
                and identity_metadata["reset_state_ids"] is not None
                else None
            ),
            kv_metadata=None,
        )
        critic_features = None
        if compute_values:
            critic = self._require_critic()
            if self._critic_kind() is CriticKind.FASTWAM_CURRENT_FRAME_VALUE:
                if sample.critic_features is None:
                    raise RuntimeError("PAD rollout produced no critic features.")
                critic_features = sample.critic_features
                values = critic.value_from_features(critic_features)
            else:
                critic_result = critic.predict_value_batch(
                    self.runtime.critic_observation(env_obs=env_obs),
                    return_prefix=True,
                )
                if isinstance(critic_result, tuple):
                    values, critic_features = critic_result
                else:
                    values = critic_result
        else:
            values = torch.zeros(
                prepared.batch_size, device=routes.device, dtype=torch.float32
            )
        values = _column_values(values, batch_size=prepared.batch_size)
        forward_inputs = dict(sample.forward_inputs)
        if (
            critic_features is not None
            and self._critic_kind() is not CriticKind.FASTWAM_CURRENT_FRAME_VALUE
        ):
            replay_key = getattr(
                self._require_critic(), "replay_feature_key", "critic_prefix"
            )
            forward_inputs[replay_key] = critic_features
        result: dict[str, Any] = {
            "prev_logprobs": None,
            "prev_values": values,
            "forward_inputs": forward_inputs,
            "route_info": route_info,
            "emitted_gate": emitted,
            "action_execution_trace": sample.action_execution_trace,
        }
        if selection is not None:
            result["evaluation_selection"] = selection
        if gate_latency is not None:
            result["gate_latency_seconds"] = gate_latency
            result["gate_h2d_seconds"] = torch.zeros_like(gate_latency)
        return sample.actions, result

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported PAD policy mode {mode!r}.")
        if mode == "eval":
            compute_values = False
        batch_size = int(env_obs["states"].shape[0])
        microbatch = self.config.training_rollout_microbatch_size
        if mode == "train" and microbatch is not None and batch_size > microbatch:
            return self._predict_training_microbatches(
                env_obs=env_obs,
                batch_size=batch_size,
                compute_values=compute_values,
                microbatch_size=microbatch,
            )
        device = env_obs["states"].device
        env_ids, reset_mask = self._routing_metadata(env_obs, batch_size, device)
        identity_metadata = (
            self._decision_identity_metadata(
                env_obs, batch_size=batch_size, device=device
            )
            if self.config.decision_telemetry_enabled
            else {"task_ids": None, "trial_ids": None, "reset_state_ids": None}
        )
        return self._predict_current_step(
            env_obs=env_obs,
            mode=mode,
            compute_values=compute_values,
            env_ids=env_ids,
            reset_mask=reset_mask,
            identity_metadata=identity_metadata,
        )

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        *,
        route_info: ChunkRouteRecord,
        emitted_gate: GateDecisionRecord,
        compute_values: bool = True,
        compute_base_logprobs: bool = False,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if compute_base_logprobs:
            raise ValueError("PAD-Frozen has no action base-logprob replay.")
        gate_parameter = next(self.gate.parameters())
        gate_config = getattr(self.gate, "config", None)
        layer_indices = getattr(gate_config, "layer_indices", None)
        if layer_indices is None:
            raise TypeError("PAD Gate must expose condition layer indices.")
        features = deserialize_condition_features(
            forward_inputs,
            prefix="route_condition",
            layer_indices=layer_indices,
        ).to(device=gate_parameter.device)
        logits = self.gate(features)
        if logits.shape != route_info.route_used.shape:
            raise ValueError("PAD Gate replay changed route batch shape.")
        behavior = epsilon_mixture_bernoulli(
            logits,
            temperature=emitted_gate.temperature.to(logits.device),
            epsilon=emitted_gate.epsilon.to(logits.device),
        )
        routes = route_info.route_used.to(logits.device)
        if not torch.equal(routes, emitted_gate.next_route.to(logits.device)):
            raise ValueError("PAD replay route differs from executed route.")
        result = {
            "gate_logits": logits,
            "gate_logprobs": behavior.log_prob(routes),
            "gate_entropy": behavior.distribution.entropy(),
            "gate_base_probabilities": behavior.base_idm_probability,
            "gate_behavior_probabilities": behavior.behavior_idm_probability,
        }
        if compute_values:
            critic = self._require_critic()
            if self._critic_kind() is CriticKind.FASTWAM_CURRENT_FRAME_VALUE:
                critic_config = getattr(critic, "config", None)
                critic_indices = getattr(critic_config, "layer_indices", None)
                if critic_indices is None:
                    raise TypeError("PAD critic must expose condition layer indices.")
                prefix = (
                    "route_condition"
                    if critic_config == gate_config
                    else "critic_condition"
                )
                critic_features = deserialize_condition_features(
                    forward_inputs,
                    prefix=prefix,
                    layer_indices=critic_indices,
                )
                values = critic.value_from_features(critic_features)
            else:
                replay_key = getattr(critic, "replay_feature_key", "critic_prefix")
                if replay_key not in forward_inputs:
                    raise KeyError(f"PAD critic replay is missing {replay_key!r}.")
                values = critic.value_from_features(forward_inputs[replay_key])
            result["values"] = _column_values(values, batch_size=int(routes.shape[0]))
        return result

    def optimizer_groups(
        self,
        *,
        gate_lr: float,
        value_lr: float,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        critic = self._require_critic()
        groups = [
            {
                "name": "gate",
                "params": [p for p in self.gate.parameters() if p.requires_grad],
                "lr": gate_lr,
            },
            {
                "name": "value_head",
                "params": [
                    p for p in critic.value_head.parameters() if p.requires_grad
                ],
                "lr": value_lr,
            },
        ]
        if any(not group["params"] for group in groups):
            raise RuntimeError("PAD Gate/value optimizer group is empty.")
        return groups

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "pad-frozen-policy-v1",
            "actor_version": self.actor_version,
            "gate": self.gate.state_dict(),
            "value_head": self._require_critic().value_head.state_dict(),
            "route_tracker": self.route_tracker.state_dict(),
        }

    def load_trainable_state_dict(self, payload: dict[str, Any]) -> None:
        expected = {"schema", "actor_version", "gate", "value_head", "route_tracker"}
        if set(payload) != expected or payload.get("schema") != "pad-frozen-policy-v1":
            raise ValueError("Unsupported PAD-Frozen policy checkpoint.")
        self.gate.load_state_dict(payload["gate"], strict=True)
        self._require_critic().value_head.load_state_dict(
            payload["value_head"], strict=True
        )
        self.route_tracker.load_state_dict(payload["route_tracker"])
        self.actor_version = int(payload["actor_version"])
        if self.actor_version < 0:
            raise ValueError("PAD actor version must be non-negative.")

    def load_eval_checkpoint(
        self,
        payload: Mapping[str, Any],
        *,
        expected_parent_checkpoint_sha256: str,
        expected_critic_parent_checkpoint_sha256: str | None = None,
    ) -> int:
        """Load Gate weights while starting evaluation with fresh route state."""

        del expected_critic_parent_checkpoint_sha256
        if payload.get("schema") != "fastwam-gate-only-frozen-pair-v1":
            raise ValueError("Unsupported PAD-Frozen evaluation checkpoint.")
        policy = payload.get("policy")
        expected = {"schema", "actor_version", "gate", "value_head", "route_tracker"}
        if (
            not isinstance(policy, Mapping)
            or set(policy) != expected
            or policy.get("schema") != "pad-frozen-policy-v1"
        ):
            raise ValueError("Unsupported PAD-Frozen evaluation policy payload.")
        artifacts = payload.get("stage_contract", {}).get("artifact_identities", {})
        if (
            artifacts.get("idm_parent_checkpoint_sha256")
            != str(expected_parent_checkpoint_sha256).lower()
        ):
            raise ValueError("PAD-Frozen evaluation IDM parent identity differs.")
        self.gate.load_state_dict(policy["gate"], strict=True)
        if self.critic is not None:
            self.critic.value_head.load_state_dict(policy["value_head"], strict=True)
        self.route_tracker = CurrentStepRouteTracker()
        self.actor_version = int(policy["actor_version"])
        if self.actor_version != int(payload.get("step", -1)):
            raise ValueError("PAD-Frozen evaluation actor version differs from step.")
        return self.actor_version

    def rollout_runtime_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "pad-frozen-rollout-runtime-v1",
            "actor_version": self.actor_version,
            "route_tracker": self.route_tracker.state_dict(),
        }

    def load_rollout_runtime_state_dict(self, payload: dict[str, Any]) -> None:
        expected = {"schema", "actor_version", "route_tracker"}
        if (
            set(payload) != expected
            or payload.get("schema") != "pad-frozen-rollout-runtime-v1"
        ):
            raise ValueError("Unsupported PAD-Frozen rollout checkpoint.")
        self.route_tracker.load_state_dict(payload["route_tracker"])
        self.actor_version = int(payload["actor_version"])
