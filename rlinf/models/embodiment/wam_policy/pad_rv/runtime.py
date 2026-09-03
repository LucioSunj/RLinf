# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Frozen dual-expert, current-step LIBERO runtime for PAD-RV."""

from __future__ import annotations

from typing import Any, Literal

import torch
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    StaticCachedActionVelocity,
)
from fastwam.models.wan22.adaptive_sampler import sample_action_flow_sde

from rlinf.envs.libero.action_protocol import select_executed_action_prefix
from rlinf.models.embodiment.wam_policy.contracts import WAMRoute
from rlinf.models.embodiment.wam_policy.critic import (
    FastWAMValueFeatures,
    FastWAMValueTransformerConfig,
    extract_fastwam_value_features,
)
from rlinf.models.embodiment.wam_policy.libero_runtime import (
    LiberoFastWAMRuntime,
    _seeded_randn,
    _validate_noise_seeds,
)

from .contracts import PadFrozenChunkSample, PreparedRouteContext
from .gate import serialize_condition_features


class PadFrozenLiberoRuntime(LiberoFastWAMRuntime):
    """Reuse legacy observation/action utilities while replacing route execution."""

    def __init__(
        self,
        *,
        actor,
        uncond_action_expert,
        gate_feature_config: FastWAMValueTransformerConfig | dict,
        gate_replay_backend: str = "condition",
        lora_adapter=None,
        **kwargs: Any,
    ) -> None:
        if lora_adapter is not None:
            raise ValueError("PAD-Frozen must not instantiate a dynamic LoRA adapter.")
        if str(gate_replay_backend) != "condition":
            raise ValueError("PAD-Frozen requires condition-only Gate replay.")
        # The parent runtime owns common preprocessing. Its replay backend is
        # unreachable because this subclass never calls its Flow-replay path.
        super().__init__(
            actor=actor,
            lora_adapter=None,
            gate_replay_backend="recompute",
            **kwargs,
        )
        self.uncond_action_expert = uncond_action_expert
        self.gate_feature_config = FastWAMValueTransformerConfig.materialize(
            gate_feature_config
        )
        self.actor.requires_grad_(False)
        self.uncond_action_expert.requires_grad_(False)
        self.actor.eval()
        self.uncond_action_expert.eval()

    @torch.no_grad()
    def _prepare_from_first_frame(
        self,
        *,
        first_frame: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        regime: PolicyRegime,
        idm_initial_latents: torch.Tensor | None = None,
        idm_noise_seed: int | None = None,
    ) -> tuple[CachedActionCondition, torch.Tensor | None]:
        """Build one route condition while encoding the current frame once."""

        if (
            first_frame.ndim != 5
            or first_frame.shape[0] != 1
            or first_frame.shape[2] != 1
        ):
            raise ValueError("PAD first-frame latent must have shape [1,C,1,H,W].")
        fuse_flag = bool(
            getattr(self.actor.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        video_latents = first_frame
        replay_initial_latents = None
        if regime is PolicyRegime.IDM:
            latent_t = (
                self.num_video_frames - 1
            ) // self.actor.vae.temporal_downsample_factor + 1
            expected = (
                1,
                self.actor.vae.model.z_dim,
                latent_t,
                int(first_frame.shape[-2]),
                int(first_frame.shape[-1]),
            )
            if idm_initial_latents is not None and idm_noise_seed is not None:
                raise ValueError("Specify PAD IDM latents or seed, not both.")
            if idm_initial_latents is None:
                video_latents = (
                    torch.randn(expected, device=self.device, dtype=torch.float32).to(
                        dtype=self.dtype
                    )
                    if idm_noise_seed is None
                    else _seeded_randn(
                        idm_noise_seed,
                        expected,
                        device=self.device,
                        dtype=self.dtype,
                        rand_device=self.seeded_noise_device,
                    )
                )
            else:
                if tuple(idm_initial_latents.shape) != expected:
                    raise ValueError(
                        f"PAD IDM latents have {tuple(idm_initial_latents.shape)}, "
                        f"expected {expected}."
                    )
                video_latents = idm_initial_latents.to(
                    device=self.device, dtype=self.dtype
                ).clone()
            replay_initial_latents = video_latents.detach().clone()
            video_latents[:, :, :1] = first_frame
            timesteps, deltas = (
                self.actor.infer_video_scheduler.build_inference_schedule(
                    num_inference_steps=self.num_inference_steps,
                    device=self.device,
                    dtype=self.dtype,
                    shift_override=self.sigma_shift,
                )
            )
            for timestep, delta in zip(timesteps, deltas, strict=True):
                velocity = self.actor._video_denoise_step_compiled(
                    latents_video=video_latents,
                    timestep_video=timestep.expand(1).to(dtype=self.dtype),
                    context=context,
                    context_mask=context_mask,
                    fuse_flag=fuse_flag,
                )
                video_latents = self.actor.infer_video_scheduler.step(
                    velocity, delta, video_latents
                )
                video_latents[:, :, :1] = first_frame

        video_pre = self.actor.video_expert.pre_dit(
            x=video_latents,
            timestep=torch.zeros(1, device=self.device, dtype=self.dtype),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_mask = self.actor._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=self.action_protocol.generation_horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        video_cache = self.actor.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            gate_current_frame_video_tokens=tokens_per_frame,
        )
        return (
            CachedActionCondition(
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                current_frame_video_tokens=tokens_per_frame,
            ),
            replay_initial_latents,
        )

    def _condition_features(
        self,
        condition: CachedActionCondition,
        *,
        config: FastWAMValueTransformerConfig,
    ) -> FastWAMValueFeatures:
        return extract_fastwam_value_features(
            condition,
            mot=self.actor.mot,
            action_expert=self.actor.action_expert,
            config=config,
            regime_context=None,
        )

    def _static_velocity(
        self,
        condition: CachedActionCondition,
        *,
        regime: PolicyRegime,
        actor_version: int,
    ) -> StaticCachedActionVelocity:
        expert = (
            self.actor.action_expert
            if regime is PolicyRegime.IDM
            else self.uncond_action_expert
        )
        return StaticCachedActionVelocity(
            action_expert=expert,
            mot=self.actor.mot,
            condition=condition,
            regime=regime,
            gate_layer_indices=self.gate_feature_config.layer_indices,
            capture_gate_kv=False,
            actor_version=actor_version,
        )

    def _critic_features_for_prepared(
        self,
        prepared: PreparedRouteContext,
    ) -> FastWAMValueFeatures | None:
        """Build legacy PAD critic features behind an overridable boundary."""

        if self.critic_feature_config is None:
            return None
        if (
            isinstance(prepared.gate_features, FastWAMValueFeatures)
            and self.critic_feature_config == self.gate_feature_config
        ):
            return prepared.gate_features
        return FastWAMValueFeatures.cat(
            [
                self._condition_features(condition, config=self.critic_feature_config)
                for condition in prepared.current_conditions
            ]
        )

    def _gate_replay_inputs(
        self,
        prepared: PreparedRouteContext,
    ) -> dict[str, torch.Tensor]:
        """Serialize the established PAD condition contract by default."""

        if not isinstance(prepared.gate_features, FastWAMValueFeatures):
            raise TypeError("PAD-Frozen replay requires FastWAMValueFeatures.")
        return {
            **serialize_condition_features(
                prepared.gate_features, prefix="route_condition"
            ),
            "fastwam_images": prepared.images,
            "fastwam_context": prepared.context,
            "fastwam_context_mask": prepared.context_mask,
        }

    @torch.no_grad()
    def prepare_route_context(self, *, env_obs: dict[str, Any]) -> PreparedRouteContext:
        images, context, context_mask = self._encode_condition(env_obs)
        batch_size = int(images.shape[0])
        action_noise = env_obs.get("_fastwam_action_initial_noise")
        if action_noise is not None:
            action_noise = torch.as_tensor(
                action_noise, device=self.device, dtype=self.dtype
            )
            expected = (
                batch_size,
                self.action_protocol.generation_horizon,
                self.actor.action_expert.action_dim,
            )
            if tuple(action_noise.shape) != expected:
                raise ValueError(f"PAD action noise shape must be {expected}.")
        action_seeds = env_obs.get("_fastwam_action_noise_seeds")
        if action_seeds is not None:
            if action_noise is not None:
                raise ValueError("Specify PAD action noise or seeds, not both.")
            action_seeds = _validate_noise_seeds(
                action_seeds, batch_size=batch_size, name="action noise"
            )
        idm_latents = env_obs.get("_fastwam_idm_initial_latents")
        if idm_latents is not None:
            idm_latents = torch.as_tensor(
                idm_latents, device=self.device, dtype=self.dtype
            )
            if idm_latents.shape[0] != batch_size:
                raise ValueError("PAD IDM latent batch must match observations.")
        idm_seeds = env_obs.get("_fastwam_idm_noise_seeds")
        if idm_seeds is not None:
            if idm_latents is not None:
                raise ValueError("Specify PAD IDM latents or seeds, not both.")
            idm_seeds = _validate_noise_seeds(
                idm_seeds, batch_size=batch_size, name="IDM video noise"
            )

        first_frames: list[torch.Tensor] = []
        conditions: list[CachedActionCondition] = []
        features: list[FastWAMValueFeatures] = []
        for index in range(batch_size):
            first_frame = self.actor._encode_input_image_latents_tensor(
                images[index : index + 1], tiled=self.tiled_vae
            )
            condition, replay_noise = self._prepare_from_first_frame(
                first_frame=first_frame,
                context=context[index : index + 1],
                context_mask=context_mask[index : index + 1],
                regime=PolicyRegime.UNCOND,
            )
            if replay_noise is not None:
                raise AssertionError("Current-only PAD condition created IDM noise.")
            first_frames.append(first_frame.detach())
            conditions.append(condition)
            features.append(
                self._condition_features(condition, config=self.gate_feature_config)
            )
        return PreparedRouteContext(
            images=images.detach(),
            context=context.detach(),
            context_mask=context_mask.detach(),
            first_frame_latents=tuple(first_frames),
            current_conditions=tuple(conditions),
            gate_features=FastWAMValueFeatures.cat(features),
            action_noise_seeds=(
                None if action_seeds is None else tuple(map(int, action_seeds.tolist()))
            ),
            idm_noise_seeds=(
                None if idm_seeds is None else tuple(map(int, idm_seeds.tolist()))
            ),
            action_initial_noise=None
            if action_noise is None
            else action_noise.detach(),
            idm_initial_latents=None if idm_latents is None else idm_latents.detach(),
        )

    @torch.no_grad()
    def sample_prepared_action_batch(
        self,
        *,
        prepared: PreparedRouteContext,
        env_obs: dict[str, Any],
        routes: torch.Tensor,
        mode: Literal["train", "eval"],
        actor_version: int,
    ) -> PadFrozenChunkSample:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported PAD runtime mode {mode!r}.")
        if routes.shape != (prepared.batch_size,):
            raise ValueError("PAD routes must have shape [B].")
        if bool(
            ((routes != int(WAMRoute.UNCOND)) & (routes != int(WAMRoute.IDM))).any()
        ):
            raise ValueError("PAD routes contain an invalid WAMRoute.")
        timesteps, deltas = self._action_schedule()
        rollouts = []
        for index, route in enumerate(routes.tolist()):
            regime = (
                PolicyRegime.IDM
                if int(route) == int(WAMRoute.IDM)
                else PolicyRegime.UNCOND
            )
            condition = prepared.current_conditions[index]
            if regime is PolicyRegime.IDM:
                condition, _ = self._prepare_from_first_frame(
                    first_frame=prepared.first_frame_latents[index],
                    context=prepared.context[index : index + 1],
                    context_mask=prepared.context_mask[index : index + 1],
                    regime=regime,
                    idm_initial_latents=(
                        None
                        if prepared.idm_initial_latents is None
                        else prepared.idm_initial_latents[index : index + 1]
                    ),
                    idm_noise_seed=(
                        None
                        if prepared.idm_noise_seeds is None
                        else prepared.idm_noise_seeds[index]
                    ),
                )
            shape = (
                1,
                self.action_protocol.generation_horizon,
                self.actor.action_expert.action_dim,
            )
            if prepared.action_initial_noise is not None:
                initial_noise = prepared.action_initial_noise[index : index + 1]
            elif prepared.action_noise_seeds is not None:
                initial_noise = _seeded_randn(
                    prepared.action_noise_seeds[index],
                    shape,
                    device=self.device,
                    dtype=self.dtype,
                    rand_device=self.seeded_noise_device,
                )
            else:
                initial_noise = torch.randn(
                    shape, device=self.device, dtype=torch.float32
                ).to(dtype=self.dtype)
            rollouts.append(
                sample_action_flow_sde(
                    initial_noise,
                    velocity_fn=self._static_velocity(
                        condition, regime=regime, actor_version=actor_version
                    ),
                    timesteps=timesteps,
                    scheduler_deltas=deltas,
                    num_train_timesteps=self.actor.infer_action_scheduler.num_train_timesteps,
                    noise_level=0.0,
                    gate_last_n=1,
                    ignore_last_transition=False,
                    stochastic=False,
                    collect_replay=False,
                )
            )
        normalized = torch.cat([rollout.actions for rollout in rollouts], dim=0)
        executed = select_executed_action_prefix(
            normalized, protocol=self.action_protocol
        )
        actions, trace = self._denormalize_action_stages(executed, env_obs=env_obs)
        critic_features = self._critic_features_for_prepared(prepared)
        replay_inputs = self._gate_replay_inputs(prepared)
        if (
            isinstance(critic_features, FastWAMValueFeatures)
            and critic_features is not prepared.gate_features
        ):
            replay_inputs.update(
                serialize_condition_features(critic_features, prefix="critic_condition")
            )
        return PadFrozenChunkSample(
            actions=actions,
            gate_features=prepared.gate_features,
            forward_inputs=replay_inputs,
            critic_features=critic_features,
            action_execution_trace=trace,
        )
