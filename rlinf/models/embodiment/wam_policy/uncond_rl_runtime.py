# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Current-observation action sampling and Flow-SDE replay for UNCOND PPO."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    CachedActionVelocity,
)
from fastwam.models.wan22.adaptive_sampler import (
    replay_action_flow_sde_transition,
    sample_action_flow_sde,
    sample_denoise_indices,
)
from fastwam.models.wan22.batch_linear import install_batch_invariant_linears

from rlinf.envs.libero.action_protocol import (
    select_executed_action_prefix,
    select_executed_flow_statistics,
)
from rlinf.utils.nested_dict_process import map_nested_tensors

from .libero_runtime import (
    LiberoFastWAMRuntime,
    _domain_separated_noise_seed,
    _seeded_randn,
)


class UncondRLLiberoRuntime(LiberoFastWAMRuntime):
    """Reuse LIBERO preprocessing with only current-frame Video and Action LoRA."""

    def __init__(self, *, video_lora_adapter=None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.video_lora_adapter = video_lora_adapter
        self._inactive_rollout_template = None
        self.batch_linear_context = install_batch_invariant_linears(self.actor)
        if self.critic_feature_config is not None:
            raise ValueError("UNCOND RL uses the existing independent pi0.5 critic.")

    @contextmanager
    def _video_scope(self, batch_size: int) -> Iterator[None]:
        previous_training = self.actor.mot.training
        self.actor.mot.training = torch.is_grad_enabled()
        try:
            with (
                self.lora_adapter.use_regime(PolicyRegime.UNCOND),
                self.batch_linear_context.use(batch_size),
            ):
                yield
        finally:
            self.actor.mot.training = previous_training

    def _video_checkpoint_contexts(self, batch_size: int):
        return self._video_scope(batch_size), self._video_scope(batch_size)

    def _current_condition(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> CachedActionCondition:
        # VAE/text/proprio stay frozen. Only the optional Video adapter needs
        # gradients through the current-frame prefill during actor replay.
        batch_size = int(latents.shape[0])
        with (
            torch.set_grad_enabled(
                torch.is_grad_enabled() and self.video_lora_adapter is not None
            ),
            self._video_scope(batch_size),
        ):
            return self._prefill_video_condition(
                video_latents=latents.detach(),
                context=context.detach(),
                context_mask=context_mask,
                fuse_flag=bool(
                    getattr(
                        self.actor.video_expert, "fuse_vae_embedding_in_latents", False
                    )
                ),
                checkpoint_context_fn=partial(
                    self._video_checkpoint_contexts, batch_size
                ),
            )

    def _uncond_velocity(
        self, condition: CachedActionCondition, actor_version: int
    ) -> CachedActionVelocity:
        return CachedActionVelocity(
            action_expert=self.actor.action_expert,
            mot=self.actor.mot,
            condition=condition,
            regime=PolicyRegime.UNCOND,
            regime_context=self.lora_adapter.regime_context,
            batch_linear_context=self.batch_linear_context,
            capture_gate_kv=False,
            actor_version=actor_version,
        )

    @torch.no_grad()
    def sample_uncond_actions(
        self,
        env_obs: dict[str, Any],
        *,
        mode: str,
        actor_version: int,
        action_seeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Batch active training rows while preserving the full rollout time axis."""

        batch_size = int(env_obs["states"].shape[0])
        active = env_obs.get("_uncond_rl_active") if mode == "train" else None
        if active is None or bool(active.all()):
            output = self._sample_uncond_rows(
                env_obs,
                mode=mode,
                actor_version=actor_version,
                action_seeds=action_seeds,
            )
        else:
            active = active.bool().reshape(-1)
            if active.numel() != batch_size:
                raise ValueError("UNCOND active mask and observation batches differ.")
            indices = active.nonzero(as_tuple=False).reshape(-1)
            if indices.numel():
                rows = indices.tolist()
                selected = {}
                for key, value in env_obs.items():
                    if key == "_uncond_rl_active":
                        continue
                    if (
                        torch.is_tensor(value)
                        and value.ndim > 0
                        and value.shape[0] == batch_size
                    ):
                        selected[key] = value.index_select(0, indices.to(value.device))
                    elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                        selected[key] = [value[row] for row in rows]
                    else:
                        selected[key] = value
                live_output = self._sample_uncond_rows(
                    selected,
                    mode=mode,
                    actor_version=actor_version,
                    action_seeds=(
                        None
                        if action_seeds is None
                        else action_seeds.index_select(
                            0, indices.to(action_seeds.device)
                        )
                    ),
                )
                output = map_nested_tensors(
                    live_output,
                    lambda tensor: tensor.new_zeros(
                        (batch_size, *tensor.shape[1:])
                    ).index_copy_(0, indices.to(tensor.device), tensor),
                )
            else:
                if self._inactive_rollout_template is None:
                    raise RuntimeError(
                        "A UNCOND rollout must start with active environments."
                    )
                output = map_nested_tensors(
                    self._inactive_rollout_template,
                    lambda tensor: tensor.expand(batch_size, *tensor.shape[1:]),
                )
        if mode == "train":
            # Scalar-backed zeros retain geometry only. Dead rows keep a valid
            # dummy transition index (zero), so padded replay remains well formed.
            # Action traces have zero counts: no model actions were generated.
            self._inactive_rollout_template = map_nested_tensors(
                output,
                lambda tensor: torch.zeros(
                    (), dtype=tensor.dtype, device=tensor.device
                ).expand(1, *tensor.shape[1:]),
            )
        return output

    def _sample_uncond_rows(
        self,
        env_obs: dict[str, Any],
        *,
        mode: str,
        actor_version: int,
        action_seeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reuse per-row noise streams for batched current-frame/action inference."""

        images, context, context_mask = self._encode_condition(env_obs)
        timesteps, deltas = self._action_schedule()
        # Keep VAE geometry unchanged; only the expensive Video prefill and
        # Action denoising become batched during training. Evaluation stays B1.
        latents = torch.cat(
            [
                self.actor._encode_input_image_latents_tensor(
                    images[index : index + 1], tiled=self.tiled_vae
                )
                for index in range(images.shape[0])
            ]
        )
        action_shape = (
            1,
            self.action_protocol.generation_horizon,
            self.actor.action_expert.action_dim,
        )
        initial_noise, denoise_indices, transition_noise = [], [], []
        for index in range(images.shape[0]):
            generator = None
            if action_seeds is None:
                noise = torch.randn(
                    action_shape, device=self.device, dtype=torch.float32
                ).to(self.dtype)
            else:
                seed = int(action_seeds[index])
                noise = _seeded_randn(
                    seed,
                    action_shape,
                    device=self.device,
                    dtype=self.dtype,
                    rand_device=self.seeded_noise_device,
                )
                generator = torch.Generator(device=self.device).manual_seed(
                    _domain_separated_noise_seed(seed, domain="flow-sde")
                )
            initial_noise.append(noise)
            if mode == "train":
                # Match the serial sampler's exact generator order: selected
                # index, then one transition-noise draw for this sample.
                denoise_indices.append(
                    sample_denoise_indices(
                        1,
                        int(timesteps.numel()),
                        device=self.device,
                        generator=generator,
                        ignore_last=self.flow_sde_ignore_last_transition,
                    )
                )
                transition_noise.append(
                    torch.randn(
                        action_shape,
                        generator=generator,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
        initial_noise = torch.cat(initial_noise)
        denoise_indices = torch.cat(denoise_indices) if denoise_indices else None
        transition_noise = torch.cat(transition_noise) if transition_noise else None
        rollouts = []
        sampling_batch_size = images.shape[0] if mode == "train" else 1
        for start in range(0, images.shape[0], sampling_batch_size):
            rows = slice(start, start + sampling_batch_size)
            condition = self._current_condition(
                latents[rows], context[rows], context_mask[rows]
            )
            rollouts.append(
                sample_action_flow_sde(
                    initial_noise[rows],
                    velocity_fn=self._uncond_velocity(condition, actor_version),
                    timesteps=timesteps,
                    scheduler_deltas=deltas,
                    num_train_timesteps=self.actor.infer_action_scheduler.num_train_timesteps,
                    noise_level=self.flow_sde_noise_level,
                    denoise_indices=None
                    if denoise_indices is None
                    else denoise_indices[rows],
                    transition_noise=None
                    if transition_noise is None
                    else transition_noise[rows],
                    ignore_last_transition=self.flow_sde_ignore_last_transition,
                    stochastic=mode == "train",
                )
            )
        actions, trace = self._denormalize_action_stages(
            select_executed_action_prefix(
                torch.cat([item.actions for item in rollouts]),
                protocol=self.action_protocol,
            ),
            env_obs=env_obs,
        )
        return actions, {
            "prev_logprobs": select_executed_flow_statistics(
                torch.cat([item.old_log_probs for item in rollouts]),
                protocol=self.action_protocol,
            ),
            "forward_inputs": {
                "fastwam_first_frame_latents": latents,
                "fastwam_context": context,
                "fastwam_context_mask": context_mask,
                "flow_chains": torch.cat([item.chains for item in rollouts]),
                "denoise_indices": torch.cat(
                    [item.denoise_indices for item in rollouts]
                ),
            },
            "action_execution_trace": trace,
        }

    def replay_uncond_actions(
        self, forward_inputs: dict[str, torch.Tensor], *, actor_version: int
    ) -> dict[str, torch.Tensor]:
        """Rebuild the selected transition with gradients into both LoRA branches."""

        condition = self._current_condition(
            forward_inputs["fastwam_first_frame_latents"],
            forward_inputs["fastwam_context"],
            forward_inputs["fastwam_context_mask"],
        )
        timesteps, deltas = self._action_schedule()
        replay = replay_action_flow_sde_transition(
            forward_inputs["flow_chains"],
            forward_inputs["denoise_indices"],
            velocity_fn=self._uncond_velocity(condition, actor_version),
            timesteps=timesteps,
            scheduler_deltas=deltas,
            num_train_timesteps=self.actor.infer_action_scheduler.num_train_timesteps,
            noise_level=self.flow_sde_noise_level,
        )
        entropy = torch.broadcast_to(
            replay.std.float().log() + 0.5 * math.log(2.0 * math.pi * math.e),
            replay.mean.shape,
        )
        return {
            "flow_logprobs": select_executed_action_prefix(
                replay.log_prob, protocol=self.action_protocol
            ),
            "flow_entropy": select_executed_action_prefix(
                entropy, protocol=self.action_protocol
            ),
        }
