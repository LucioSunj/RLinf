# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""BC-initialized UNCOND policy using the native FastWAM PPO/checkpoint adapter."""

from __future__ import annotations

import copy
from typing import Any

import torch

from .adaptive_policy import (
    FastWAMAdaptivePolicy,
    _column_values,
    _formal_training_sample_seed,
)
from .contracts import ChunkRouteRecord, GateDecisionRecord, WAMRoute
from .evaluation import EvaluationRouteSelection
from .libero_runtime import _validate_noise_seeds


class UncondSamplingState:
    """Track episode/chunk identities for sampling and native rollout resume."""

    def __init__(self) -> None:
        self.episode_chunks: dict[int, tuple[int, int]] = {}

    def state_dict(self) -> dict[str, Any]:
        """Return sampling positions without any pending routing decisions."""

        return {"episode_chunks": copy.deepcopy(self.episode_chunks)}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore per-environment sampling positions."""

        self.episode_chunks = copy.deepcopy(payload["episode_chunks"])


class FastWAMUncondRLPolicy(FastWAMAdaptivePolicy):
    """Train only UNCOND LoRA and a fresh value head, with no routing network."""

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("gate") is not None:
            raise ValueError("UNCOND RL must not construct a Gate.")
        super().__init__(**kwargs)
        # These counters identify action RNG streams and resume positions only.
        # No pending route, physical history, or branch decision is retained.
        self.route_tracker = UncondSamplingState()
        self._compile_mode: str | None = None
        self._compiled_inference = None

    def enable_torch_compile(self, mode: str = "default", **_kwargs: Any) -> None:
        """Select a read-only compiled Action view for B1 evaluation."""

        if self.training:
            raise ValueError("UNCOND compilation is restricted to evaluation.")
        self._compile_mode = mode
        self._compiled_inference = None

    def set_global_step(self, version: int) -> None:
        """Advance the policy version without changing the action regime."""

        if version < 0:
            raise ValueError("Actor version must be non-negative.")
        self.actor_version = int(version)

    def capture_gate_recompute_reference(self) -> None:
        """UNCOND replay uses the live adapters and needs no Gate reference."""

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "train",
        compute_values: bool = True,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute UNCOND at every chunk, including reset and weight-sync boundaries."""

        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported UNCOND policy mode {mode!r}.")
        batch_size = int(env_obs["states"].shape[0])
        device = env_obs["states"].device
        env_ids, resets = self._routing_metadata(env_obs, batch_size, device)
        episodes, chunks = [], []
        for env_id, reset in zip(env_ids.tolist(), resets.tolist(), strict=True):
            episode, chunk = self.route_tracker.episode_chunks.get(env_id, (-1, 0))
            if reset or episode < 0:
                episode, chunk = episode + 1, 0
            episodes.append(episode)
            chunks.append(chunk)
            self.route_tracker.episode_chunks[env_id] = (episode, chunk + 1)
        route = torch.full(
            (batch_size,), int(WAMRoute.UNCOND), device=device, dtype=torch.long
        )
        route_info = ChunkRouteRecord(
            route_used=route,
            route_was_forced=torch.ones_like(route, dtype=torch.bool),
            chunk_ids=torch.tensor(chunks, device=device),
            episode_ids=torch.tensor(episodes, device=device),
            route_source_chunk_ids=torch.full_like(route, -1),
            actor_versions=torch.full_like(route, self.actor_version),
        )
        seeds = None
        if mode == "eval" and "_fastwam_action_noise_seeds" in env_obs:
            seeds = _validate_noise_seeds(
                env_obs["_fastwam_action_noise_seeds"],
                batch_size=batch_size,
                name="action",
            )
        if mode == "train" and self.config.formal_training_sampling_seed is not None:
            seeds = torch.tensor(
                [
                    _formal_training_sample_seed(
                        base_seed=self.config.formal_training_sampling_seed,
                        domain="action",
                        environment_id=env_id,
                        episode_id=episode,
                        chunk_id=chunk,
                        actor_version=self.actor_version,
                    )
                    for env_id, episode, chunk in zip(
                        env_ids.tolist(), episodes, chunks, strict=True
                    )
                ],
                dtype=torch.long,
            )
        runtime = self.runtime
        if mode == "eval" and self._compile_mode is not None:
            if batch_size != 1:
                raise ValueError("Compiled UNCOND evaluation requires B1.")
            if self._compiled_inference is None:
                from .uncond_rl_inference import UncondRLCompiledInference

                self._compiled_inference = UncondRLCompiledInference(
                    self.runtime, mode=self._compile_mode
                )
            runtime = self._compiled_inference.runtime
        actions, result = runtime.sample_uncond_actions(
            env_obs, mode=mode, actor_version=self.actor_version, action_seeds=seeds
        )
        if compute_values and mode == "train":
            # Keep the original critic batch geometry, including finished rows.
            # Only heavy action inference uses the active-row selection.
            critic = self._require_critic()
            values, features = critic.predict_value_batch(
                self.runtime.critic_observation(env_obs=env_obs), return_prefix=True
            )
            result["forward_inputs"][critic.replay_feature_key] = features.detach()
        else:
            values = torch.zeros(batch_size, device=device)
        result["prev_values"] = _column_values(values, batch_size=batch_size)
        result["route_info"] = route_info
        # The existing trajectory transport requires these typed fields. They
        # are constant invalid placeholders, with no Gate computation or credit.
        zeros = torch.zeros(batch_size, device=device)
        result["emitted_gate"] = GateDecisionRecord(
            next_route=route,
            base_probability=zeros,
            behavior_probability=zeros,
            old_logprob=zeros,
            epsilon=zeros,
            temperature=torch.ones_like(zeros),
            valid=torch.zeros_like(route, dtype=torch.bool),
            source_chunk_ids=route_info.chunk_ids,
            episode_ids=route_info.episode_ids,
            actor_versions=route_info.actor_versions,
        )
        if mode == "eval":
            result["evaluation_selection"] = EvaluationRouteSelection(
                mode="forced_uncond",
                effective_next_route=route,
                counterfactual_next_route=route,
            )
            result["prev_logprobs"] = torch.empty(batch_size, 0, device=device)
            result["forward_inputs"] = {}
        return actions, result

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        *,
        compute_values: bool = True,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Replay Flow-SDE and the detached critic features without a teacher loss."""

        result = self.runtime.replay_uncond_actions(
            forward_inputs, actor_version=self.actor_version
        )
        result["logprobs"] = result["flow_logprobs"]
        batch_size = result["logprobs"].shape[0]
        if compute_values:
            critic = self._require_critic()
            values = critic.value_from_features(
                forward_inputs[critic.replay_feature_key]
            )
        else:
            values = result["logprobs"].new_zeros((batch_size, 1))
        result["values"] = _column_values(values, batch_size=batch_size)
        return result

    def optimizer_parameter_groups(
        self, *, lora_lr: float, value_lr: float, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        """Expose only the two trainable owners to optimizer tooling."""

        return [
            {
                "name": "uncond_lora",
                "params": list(self.lora_parameters()),
                "lr": lora_lr,
            },
            {
                "name": "value_head",
                "params": list(self._require_critic().value_head.parameters()),
                "lr": value_lr,
            },
        ]

    def rollout_runtime_state_dict(self) -> dict[str, Any]:
        """Save the policy version and per-environment sampling positions."""

        return {
            "schema": "fastwam-uncond-rollout-v1",
            "actor_version": self.actor_version,
            "route_tracker": self.route_tracker.state_dict(),
        }

    def load_rollout_runtime_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore sampling positions for exact native continuation."""

        if payload.get("schema") != "fastwam-uncond-rollout-v1":
            raise ValueError("Unsupported UNCOND rollout checkpoint.")
        self.set_global_step(int(payload["actor_version"]))
        self.route_tracker.load_state_dict(payload["route_tracker"])

    def trainable_state_dict(self) -> dict[str, Any]:
        """Save both adapters and the value head without frozen base weights."""

        return {
            "schema": "fastwam-uncond-policy-v1",
            "lora": self.lora_adapter.lora_state_dict(),
            "video_lora": (
                None
                if self.video_lora_adapter is None
                else self.video_lora_adapter.lora_state_dict()
            ),
            "value_head": self._require_critic().value_head.state_dict(),
            "actor_version": self.actor_version,
            "route_tracker": self.route_tracker.state_dict(),
        }

    def load_trainable_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore the native UNCOND schema; mixed-policy checkpoints are rejected."""

        if payload.get("schema") != "fastwam-uncond-policy-v1":
            raise ValueError("Unsupported UNCOND policy checkpoint.")
        if (payload["video_lora"] is None) != (self.video_lora_adapter is None):
            raise ValueError("UNCOND checkpoint Video LoRA configuration differs.")
        self.lora_adapter.load_lora_state_dict(payload["lora"], strict=True)
        if self.video_lora_adapter is not None:
            self.video_lora_adapter.load_lora_state_dict(
                payload["video_lora"], strict=True
            )
        if self.critic is not None:
            self.critic.value_head.load_state_dict(payload["value_head"], strict=True)
        self.route_tracker.load_state_dict(payload["route_tracker"])
        self.set_global_step(int(payload["actor_version"]))


def compute_uncond_rl_loss(
    *,
    cfg: Any,
    actor_version: int,
    micro_batch: dict[str, Any],
    output_dict: dict[str, torch.Tensor],
    selected_loss_scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Use the existing joint-chunk Flow PPO and clipped value losses only."""

    from rlinf.algorithms.fastwam_dual_ppo import compute_uncond_flow_ppo_loss
    from rlinf.algorithms.losses import compute_ppo_critic_loss

    flow = cfg.algorithm.uncond_flow_ppo
    policy_loss, metrics = compute_uncond_flow_ppo_loss(
        logprobs=output_dict["flow_logprobs"].float(),
        old_logprobs=micro_batch["prev_logprobs"].float(),
        advantages=micro_batch["flow_advantages"].float(),
        route_used=micro_batch["route_info"].route_used,
        valid_mask=micro_batch["flow_valid_mask"].bool(),
        clip_ratio_low=float(flow.clip_ratio_low),
        clip_ratio_high=float(flow.clip_ratio_high),
        entropy=output_dict["flow_entropy"],
        entropy_coefficient=float(flow.entropy_coefficient),
        selected_loss_scale=(selected_loss_scales or {}).get("flow"),
    )
    warmup = actor_version < int(cfg.actor.model.uncond_rl.critic_warmup_updates)
    if warmup:
        # No LoRA gradients or Adam state advances during critic-only warm-up.
        policy_loss = policy_loss.detach() * 0.0
    critic = cfg.algorithm.critic_loss
    value_loss, value_metrics = compute_ppo_critic_loss(
        values=output_dict["values"].float(),
        returns=micro_batch["returns"].float(),
        prev_values=micro_batch["prev_values"].float(),
        value_clip=float(critic.value_clip),
        huber_delta=float(critic.huber_delta),
        loss_mask=micro_batch.get("loss_mask"),
        loss_mask_sum=micro_batch.get("loss_mask_sum"),
        max_episode_steps=cfg.env.train.max_episode_steps,
    )
    loss = (
        float(flow.loss_weight) * policy_loss + float(critic.loss_weight) * value_loss
    )
    metrics.update(value_metrics)
    metrics["uncond_flow/critic_warmup"] = float(warmup)
    metrics["fastwam/total_loss"] = loss.detach()
    return loss, {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in metrics.items()
    }
