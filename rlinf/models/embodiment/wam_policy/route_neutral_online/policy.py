# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Current-step route-neutral policy with trainable BC-initialized UNCOND LoRA."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Literal

import torch
from fastwam.models.wan22.gate_transformer import epsilon_mixture_bernoulli

from rlinf.models.embodiment.wam_policy.adaptive_policy import _column_values
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    WAMRoute,
)
from rlinf.models.embodiment.wam_policy.critic import CriticKind
from rlinf.models.embodiment.wam_policy.online_idm_bc.policy import (
    OnlineIDMBCFastWAMPolicy,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_contracts import (
    PadCriticWarmupConfig,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PhysicalStateHistoryTracker,
    deserialize_route_neutral_features,
    serialize_route_neutral_features,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_policy import (
    RouteNeutralRoutingState,
)

from .inference import InferenceAccelerationConfig, RouteNeutralInference
from .runtime import (
    ROUTE_NEUTRAL_ROLLOUT_IDM_BATCH_SIZE,
    ROUTE_NEUTRAL_ROLLOUT_UNCOND_BATCH_SIZE,
    ROUTE_NEUTRAL_TEACHER_BATCH_SIZE,
    RouteNeutralOnlineIDMTeacherLiberoRuntime,
)


class RouteNeutralOnlineIDMBCFastWAMPolicy(OnlineIDMBCFastWAMPolicy):
    """Select the current chunk before action generation from neutral features."""

    def __init__(
        self,
        *,
        runtime: RouteNeutralOnlineIDMTeacherLiberoRuntime,
        critic_warmup,
        pure_uncond_num_envs: int = 0,
        inference_acceleration: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(runtime, RouteNeutralOnlineIDMTeacherLiberoRuntime):
            raise TypeError("Route-neutral policy requires its dedicated runtime.")
        self.critic_warmup = PadCriticWarmupConfig.from_mapping(critic_warmup)
        self.pure_uncond_num_envs = pure_uncond_num_envs
        super().__init__(runtime=runtime, **kwargs)
        self.route_tracker = RouteNeutralRoutingState(
            physical_history=runtime.physical_history
        )
        self.inference_acceleration = (
            None
            if inference_acceleration is None
            else InferenceAccelerationConfig(**dict(inference_acceleration))
        )
        self._inference_engine = None

    def enable_inference_acceleration(
        self,
        *,
        merge_lora: bool = True,
        compile: bool = True,
        mode: str = "default",
        backend: str = "inductor",
    ) -> None:
        """Prepare resident experts lazily on the next B1 evaluation call."""

        self.inference_acceleration = InferenceAccelerationConfig(
            merge_lora=merge_lora, compile=compile, mode=mode, backend=backend
        )
        self._inference_engine = None

    def enable_torch_compile(self, mode: str = "default", **kwargs) -> None:
        """Support RLinf's rollout compile switch for pure evaluation only."""

        self.enable_inference_acceleration(compile=True, mode=mode, **kwargs)

    @torch.no_grad()
    def warmup_inference(
        self, env_obs: dict[str, Any], *, repetitions: int = 3
    ) -> dict[str, list[float]]:
        """Warm both route kernels before control, retaining history and RNG."""

        if env_obs["states"].shape[0] != 1 or repetitions < 1:
            raise ValueError("Inference warmup requires B1 and positive repetitions.")
        config = self.config
        history = copy.deepcopy(self.route_tracker.state_dict())
        cpu_rng = torch.get_rng_state()
        device = self.runtime.device
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        times = {}
        try:
            for routing in ("forced_uncond", "forced_idm"):
                self.config = replace(config, eval_routing_mode=routing)
                times[routing] = []
                for _ in range(repetitions):
                    self.route_tracker.load_state_dict(copy.deepcopy(history))
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, device)
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    self.predict_action_batch(env_obs, mode="eval")
                    if cuda_rng is not None:
                        torch.cuda.synchronize(device)
                    times[routing].append(time.perf_counter() - started)
        finally:
            self.config = config
            self.route_tracker.load_state_dict(history)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        return times

    def _apply(self, fn, recurse=True):
        # Inference views share frozen tensor storage. Rebuild after .to() so
        # their plain modules and compiled graphs follow the actual device/dtype.
        self._inference_engine = None
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True):
        if mode:
            self._inference_engine = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._inference_engine = None
        return super()._load_from_state_dict(*args, **kwargs)

    def load_trainable_state_dict(self, payload):
        self._inference_engine = None
        return super().load_trainable_state_dict(payload)

    def set_global_step(self, version: int) -> None:
        # RLinf's native actor-to-rollout weight transfer finishes with this
        # hook, including transfers which copy tensors without load_state_dict.
        if int(version) != self.actor_version:
            self._inference_engine = None
        super().set_global_step(version)

    def capture_gate_recompute_reference(self) -> None:
        """Condition replay never reconstructs route-derived Gate snapshots."""

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
        return (
            torch.ones_like(logits)
            if self._warmup_active()
            else torch.full_like(logits, self.config.gate_epsilon)
        )

    def _gate_features_from_forward_inputs(
        self,
        forward_inputs: Mapping[str, torch.Tensor],
    ):
        return deserialize_route_neutral_features(
            forward_inputs,
            layer_indices=self.runtime.route_neutral_visual.layer_indices,
        )

    @staticmethod
    def _critic_replay_values(
        critic: torch.nn.Module,
        prefix: torch.Tensor,
    ) -> torch.Tensor:
        """Keep only the route-neutral replay value head on its B1 path."""

        rowwise_head = getattr(critic, "value_from_features_rowwise_head", None)
        if callable(rowwise_head):
            return rowwise_head(prefix)
        if hasattr(critic, "value_from_features"):
            return critic.value_from_features(prefix)
        return critic.value_from_prefix(prefix)

    def _replay_gate_logits(self, features) -> torch.Tensor:
        """Evaluate complete replay Gate rows in their original B1 order."""

        serialized = serialize_route_neutral_features(features)
        logits = []
        for index in range(features.batch_size):
            row = {name: value[index : index + 1] for name, value in serialized.items()}
            logits.append(
                self.gate(
                    deserialize_route_neutral_features(
                        row,
                        layer_indices=features.visual.layer_indices,
                    )
                )
            )
        return torch.cat(logits, dim=0)

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
        runtime = self.runtime
        gate_forward = self.gate
        acceleration = self.inference_acceleration
        if (
            mode == "eval"
            and env_obs["states"].shape[0] == 1
            and acceleration is not None
            and acceleration.merge_lora
        ):
            if self._inference_engine is None:
                self._inference_engine = RouteNeutralInference(
                    self.runtime, self.gate, acceleration
                )
            self._inference_engine.begin_chunk()
            runtime = self._inference_engine.runtime
            gate_forward = self._inference_engine.gate_logits
        identity = self.route_tracker.prepare(
            env_ids=env_ids,
            reset_mask=reset_mask,
            actor_version=self.actor_version,
            episode_ids=(
                env_obs.get("_fastwam_evaluation_episode_ids")
                if mode == "eval"
                else None
            ),
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
                raise ValueError(
                    f"Route-neutral policy refuses caller seeds: {collisions}."
                )
            runtime_obs = {**env_obs, **seed_fields}

        prepared = runtime.prepare_route_neutral_step(
            env_obs=runtime_obs,
            mode=mode,
            include_critic_features=(
                compute_values
                and self._critic_kind() is CriticKind.FASTWAM_CURRENT_FRAME_VALUE
            ),
        )
        gate_features = prepared.gate_features
        gate_parameter = next(self.gate.parameters())
        measure_latency = self.config.eval_timing_cuda_synchronize and mode == "eval"
        if measure_latency and gate_parameter.device.type == "cuda":
            torch.cuda.synchronize(gate_parameter.device)
        started = time.perf_counter()
        logits = gate_forward(gate_features.to(device=gate_parameter.device))
        selection = None
        if mode == "train":
            routes, base, behavior, logprob, exploration = self._training_gate_decision(
                logits=logits,
                sampling_seeds=(
                    None if sampling_seeds is None else sampling_seeds["gate"]
                ),
            )
            epsilon = self._training_gate_epsilon(logits)
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
        # Global environment slots retain their behavior across chunks, resets,
        # rollout microbatches and actor updates. Evaluation has its own routing.
        pure_uncond = None
        if mode == "train" and self.pure_uncond_num_envs:
            pure_uncond = env_ids.to(routes.device) < self.pure_uncond_num_envs
            routes = torch.where(pure_uncond, int(WAMRoute.UNCOND), routes)
            behavior = torch.where(pure_uncond, 0.0, behavior)
            logprob = torch.where(pure_uncond, 0.0, logprob)
        gate_latency = None
        if measure_latency:
            if logits.device.type == "cuda":
                torch.cuda.synchronize(logits.device)
            gate_latency = torch.full(
                (gate_features.batch_size,),
                time.perf_counter() - started,
                dtype=torch.float64,
            )
        route_info = self.route_tracker.commit(identity=identity, routes=routes)
        if pure_uncond is not None:
            route_info = replace(
                route_info,
                route_was_forced=pure_uncond,
                route_source_chunk_ids=torch.where(
                    pure_uncond, -1, route_info.route_source_chunk_ids
                ),
            )
        sample = runtime.sample_routed_action_batch(
            env_obs=runtime_obs,
            routes=route_info.route_used,
            mode=mode,
            actor_version=self.actor_version,
            collect_replay=mode == "train",
            prepared=prepared,
        )
        emitted = GateDecisionRecord(
            next_route=route_info.route_used,
            base_probability=base,
            behavior_probability=behavior,
            old_logprob=logprob,
            epsilon=epsilon,
            temperature=torch.full_like(logits, self.config.gate_temperature),
            valid=~route_info.route_was_forced,
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
                    raise RuntimeError("Route-neutral rollout lacks critic features.")
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
                gate_features.batch_size,
                device=routes.device,
                dtype=torch.float32,
            )
        values = _column_values(values, batch_size=gate_features.batch_size)
        forward_inputs = (
            {
                **sample.forward_inputs,
                **serialize_route_neutral_features(gate_features),
                "flow_chains": sample.flow_chains,
                "denoise_indices": sample.denoise_indices,
            }
            if mode == "train"
            else {}
        )
        if (
            critic_features is not None
            and self._critic_kind() is not CriticKind.FASTWAM_CURRENT_FRAME_VALUE
        ):
            replay_key = getattr(
                self._require_critic(), "replay_feature_key", "critic_prefix"
            )
            forward_inputs[replay_key] = critic_features
        result: dict[str, Any] = {
            "prev_logprobs": sample.old_flow_logprobs,
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
            raise ValueError(f"Unsupported route-neutral mode {mode!r}.")
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
                env_obs,
                batch_size=batch_size,
                device=device,
            )
            if self.config.decision_telemetry_enabled
            else {"task_ids": None, "trial_ids": None, "reset_state_ids": None}
        )
        with torch.no_grad() if mode == "eval" else nullcontext():
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
        gate_parameter = next(self.gate.parameters())
        features = self._gate_features_from_forward_inputs(forward_inputs).to(
            device=gate_parameter.device
        )
        logits = self._replay_gate_logits(features)
        routes = route_info.route_used.to(logits.device)
        if not torch.equal(routes, emitted_gate.next_route.to(logits.device)):
            raise ValueError("Current-step replay route differs from executed route.")
        behavior = epsilon_mixture_bernoulli(
            logits,
            temperature=emitted_gate.temperature.to(logits.device),
            epsilon=emitted_gate.epsilon.to(logits.device),
        )
        replay_kwargs: dict[str, Any] = {
            "forward_inputs": forward_inputs,
            "route_info": route_info,
        }
        if compute_base_logprobs:
            replay_kwargs["compute_base_logprobs"] = True
        replay = self.runtime.replay_action_batch(**replay_kwargs)

        if compute_values:
            critic = self._require_critic()
            if self._critic_kind() is CriticKind.FASTWAM_CURRENT_FRAME_VALUE:
                if "critic_features" not in replay:
                    raise KeyError("Route-neutral critic replay lacks value features.")
                values = critic.value_from_features(replay["critic_features"])
            else:
                replay_key = getattr(critic, "replay_feature_key", "critic_prefix")
                if replay_key in forward_inputs:
                    prefix = forward_inputs[replay_key]
                    values = self._critic_replay_values(critic, prefix)
                else:
                    values = critic.predict_value_batch(
                        self.runtime.critic_observation(forward_inputs=forward_inputs)
                    )
        else:
            values = torch.zeros(
                (routes.shape[0], 1),
                device=logits.device,
                dtype=torch.float32,
            )
        result = {
            "logprobs": replay["flow_logprobs"],
            "flow_logprobs": replay["flow_logprobs"],
            "flow_entropy": replay["flow_entropy"],
            "gate_logprobs": behavior.log_prob(routes),
            "gate_entropy": behavior.distribution.entropy(),
            "gate_base_probabilities": behavior.base_idm_probability,
            "gate_behavior_probabilities": behavior.behavior_idm_probability,
            "values": _column_values(values, batch_size=int(routes.shape[0])),
        }
        if self.pure_uncond_num_envs:
            forced = route_info.route_was_forced.to(logits.device)
            for key in ("gate_logprobs", "gate_entropy", "gate_behavior_probabilities"):
                result[key] = torch.where(forced, 0.0, result[key])
        if compute_base_logprobs:
            if "base_uncond_kl" not in replay:
                raise KeyError("UNCOND replay did not return base KL.")
            result["base_uncond_kl"] = replay["base_uncond_kl"]
        online_bc = self.runtime.compute_online_idm_bc_loss(
            forward_inputs=forward_inputs,
            route_info=route_info,
        )
        result.update(online_bc.as_forward_outputs())
        for forward_key, metric_name in (
            (ROUTE_NEUTRAL_ROLLOUT_IDM_BATCH_SIZE, "perf/rollout_idm_batch_size"),
            (
                ROUTE_NEUTRAL_ROLLOUT_UNCOND_BATCH_SIZE,
                "perf/rollout_uncond_batch_size",
            ),
            (ROUTE_NEUTRAL_TEACHER_BATCH_SIZE, "perf/teacher_batch_size"),
        ):
            if forward_key in forward_inputs:
                result[metric_name] = forward_inputs[forward_key].float().max()
        return result

    def load_eval_checkpoint(self, *args, **kwargs) -> int:
        self._inference_engine = None
        version = super().load_eval_checkpoint(*args, **kwargs)
        history = PhysicalStateHistoryTracker(self.runtime.route_neutral_input)
        self.runtime.physical_history = history
        self.route_tracker = RouteNeutralRoutingState(physical_history=history)
        if self.inference_acceleration is None:
            # Standalone native-checkpoint evaluation uses the accelerated
            # path by default. Training restores call load_trainable_state_dict
            # and never opt into these evaluation-only implementation choices.
            self.enable_inference_acceleration()
        return version


__all__ = ["RouteNeutralOnlineIDMBCFastWAMPolicy"]
