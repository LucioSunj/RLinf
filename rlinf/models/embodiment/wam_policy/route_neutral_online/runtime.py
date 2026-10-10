# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Route-neutral decision features over the trainable online-BC runtime."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import torch
from fastwam.adapters import PolicyRegime, VideoBCDiTLoRAAdapter
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
from fastwam.models.wan22.kv_tap import KeyValueBank, KVSource
from fastwam.models.wan22.mot import _GATE_CURRENT_FRAME_PROVENANCE_KEY
from fastwam.uncond_bc import (
    compute_action_flow_matching_bc_loss,
    stateless_validation_flow_inputs,
)

from rlinf.envs.action_contract import ActionExecutionTrace
from rlinf.envs.libero.action_protocol import (
    select_executed_action_prefix,
    select_executed_flow_statistics,
)
from rlinf.models.embodiment.wam_policy.adaptive_policy import FastWAMChunkSample
from rlinf.models.embodiment.wam_policy.contracts import ChunkRouteRecord, WAMRoute
from rlinf.models.embodiment.wam_policy.critic import (
    FastWAMValueFeatures,
    FastWAMValueTransformerConfig,
)
from rlinf.models.embodiment.wam_policy.kv_replay import GateKVReplayBackend
from rlinf.models.embodiment.wam_policy.libero_runtime import (
    _domain_separated_noise_seed,
    _format_fastwam_prompts,
    _load_cached_text_contexts,
    _seeded_randn,
    _validate_flow_sde_sampling,
    _validate_noise_seeds,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import (
    ONLINE_IDM_BC_FLOW_VALID,
    ONLINE_IDM_BC_FORWARD_KEYS,
    ONLINE_IDM_BC_SAMPLE_IDENTITIES,
    ONLINE_IDM_BC_TEACHER_ACTIONS,
    ONLINE_IDM_BC_TEACHER_BYTES,
    ONLINE_IDM_BC_TEACHER_PRESENT,
    ONLINE_IDM_BC_TEACHER_SECONDS,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.runtime import (
    OnlineIDMBCLossBatch,
    OnlineIDMTeacherLiberoRuntime,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_contracts import (
    RouteNeutralGateInputContract,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PhysicalStateHistoryTracker,
    RouteNeutralGateFeatures,
    RouteNeutralVisualFeatures,
    RouteNeutralVisualLayer,
)

ROUTE_NEUTRAL_ROLLOUT_IDM_BATCH_SIZE = "route_neutral_rollout_idm_batch_size"
ROUTE_NEUTRAL_ROLLOUT_UNCOND_BATCH_SIZE = "route_neutral_rollout_uncond_batch_size"
ROUTE_NEUTRAL_TEACHER_BATCH_SIZE = "route_neutral_teacher_batch_size"


@dataclass(frozen=True, slots=True)
class RouteNeutralPreparedStep:
    """Canonical current-only state shared by Gate, action, and critic calls."""

    images: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    current_condition: CachedActionCondition
    gate_features: RouteNeutralGateFeatures
    critic_features: FastWAMValueFeatures | None
    first_frame_latents: torch.Tensor | None = None


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

    video_lora_adapter: VideoBCDiTLoRAAdapter | None = None

    def __init__(
        self,
        *,
        route_neutral_input,
        route_neutral_visual,
        gate_replay_backend="recompute",
        video_lora_adapter: VideoBCDiTLoRAAdapter | None = None,
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
        self.video_lora_adapter = video_lora_adapter
        self.batch_linear_context = install_batch_invariant_linears(self.actor)
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
        # One immutable language payload; visual/proprio/history state is rebuilt
        # every chunk. This cache is neither a module buffer nor checkpoint state.
        self._evaluation_text_context = None

    def _prepare_action_condition(
        self,
        *,
        image: torch.Tensor | None,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        regime: PolicyRegime,
        idm_initial_latents: torch.Tensor | None = None,
        idm_noise_seed: int | None = None,
        first_frame_latents: torch.Tensor | None = None,
    ) -> tuple[CachedActionCondition, torch.Tensor | None]:
        if self.video_lora_adapter is not None and regime is PolicyRegime.UNCOND:
            with torch.no_grad():
                first_frame = (
                    self.actor._encode_input_image_latents_tensor(
                        image, tiled=self.tiled_vae
                    )
                    if first_frame_latents is None
                    else first_frame_latents.detach()
                )
            batch_size = int(context.shape[0])
            with self._uncond_video_scope(batch_size):
                condition = self._prefill_video_condition(
                    video_latents=first_frame,
                    context=context.detach(),
                    context_mask=context_mask,
                    fuse_flag=bool(
                        getattr(
                            self.actor.video_expert,
                            "fuse_vae_embedding_in_latents",
                            False,
                        )
                    ),
                    checkpoint_context_fn=partial(
                        self._video_checkpoint_contexts, batch_size
                    ),
                )
            return condition, None
        with (
            self.batch_linear_context.use(int(image.shape[0])),
            self.lora_adapter.use_regime(PolicyRegime.IDM)
            if self.video_lora_adapter is not None
            else nullcontext(),
        ):
            return super()._prepare_action_condition(
                image=image,
                context=context,
                context_mask=context_mask,
                regime=regime,
                idm_initial_latents=idm_initial_latents,
                idm_noise_seed=idm_noise_seed,
                first_frame_latents=first_frame_latents,
            )

    @contextmanager
    def _uncond_video_scope(self, batch_size: int) -> Iterator[None]:
        """Retain LoRA/batch contexts through Video checkpoint recomputation."""

        mot_training = self.actor.mot.training
        # MoT uses this flag only to select activation checkpointing. Keep the
        # frozen experts in eval mode, including dropout and normalization.
        self.actor.mot.training = torch.is_grad_enabled()
        try:
            with (
                self.lora_adapter.use_regime(PolicyRegime.UNCOND),
                self.batch_linear_context.use(batch_size),
            ):
                yield
        finally:
            self.actor.mot.training = mot_training

    def _video_checkpoint_contexts(
        self, batch_size: int
    ) -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
        return self._uncond_video_scope(batch_size), self._uncond_video_scope(
            batch_size
        )

    @torch.no_grad()
    def _prepare_parent_current_condition(
        self, **kwargs: torch.Tensor | None
    ) -> tuple[CachedActionCondition, torch.Tensor | None]:
        """Build canonical parent K/V for the route-neutral Gate and critic."""

        with (
            self.lora_adapter.use_regime(PolicyRegime.IDM),
            self.batch_linear_context.use(int(kwargs["context"].shape[0])),
        ):
            if kwargs.get("image") is None:
                return self._prefill_video_condition(
                    video_latents=kwargs["first_frame_latents"],
                    context=kwargs["context"],
                    context_mask=kwargs["context_mask"],
                    fuse_flag=bool(
                        getattr(
                            self.actor.video_expert,
                            "fuse_vae_embedding_in_latents",
                            False,
                        )
                    ),
                ), None
            return super()._prepare_action_condition(
                **kwargs, regime=PolicyRegime.UNCOND
            )

    def _uncond_condition_from_prepared(
        self, prepared: RouteNeutralPreparedStep, indices: torch.Tensor | None = None
    ) -> CachedActionCondition:
        """Use adapted Video K/V only for the UNCOND action expert."""

        if self.video_lora_adapter is None:
            return (
                prepared.current_condition
                if indices is None
                else prepared.current_condition.index_select(indices)
            )

        def select(value):
            return (
                value
                if value is None or indices is None
                else value.index_select(0, indices.to(value.device))
            )

        condition, _ = self._prepare_action_condition(
            image=select(prepared.images),
            context=select(prepared.context),
            context_mask=select(prepared.context_mask),
            regime=PolicyRegime.UNCOND,
            first_frame_latents=select(prepared.first_frame_latents),
        )
        return condition

    def _prefill_parent_gate_kv(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict[str, torch.Tensor],
        video_attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compute parent Video K/V only through the final Gate tap.

        This tensor-only loop is separately compilable in the inference view.
        The final tapped layer needs its input K/V but no attention or FFN.
        """

        mot = self.actor.mot
        expert = self.actor.video_expert
        last_layer = self.route_neutral_visual.layer_indices[-1]
        x = video_tokens
        keys, values = [], []
        for layer_index in range(last_layer + 1):
            block = expert.blocks[layer_index]
            q, k, v, residual, gate_msa, shift, scale, gate_mlp, _ = (
                mot._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=video_freqs,
                    t_mod=video_t_mod,
                )
            )
            keys.append(k)
            values.append(v)
            if layer_index == last_layer:
                break
            mixed = mot._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            x = mot._apply_expert_post_block(
                block=block,
                residual_x=residual,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift,
                scale_mlp=scale,
                gate_mlp=gate_mlp,
                context_payload=video_context_payload,
            )
        return keys, values

    @torch.no_grad()
    def _prepare_parent_gate_condition(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> CachedActionCondition:
        """Build a Gate-only prefix; dual routes prepare their own action cache."""

        with (
            self.lora_adapter.use_regime(PolicyRegime.IDM),
            self.batch_linear_context.use(1),
        ):
            video_pre = self.actor.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=torch.zeros(1, device=self.device, dtype=self.dtype),
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    getattr(
                        self.actor.video_expert, "fuse_vae_embedding_in_latents", False
                    )
                ),
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
            attention_mask = self.actor._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=self.action_protocol.generation_horizon,
                video_tokens_per_frame=tokens_per_frame,
                device=self.device,
            )
            video_mask = attention_mask[:video_seq_len, :video_seq_len]
            self.actor.mot._validate_current_frame_video_mask(
                attention_mask=video_mask,
                current_frame_video_tokens=tokens_per_frame,
                video_seq_len=video_seq_len,
            )
            keys, values = self._prefill_parent_gate_kv(
                video_pre["tokens"],
                video_pre["freqs"],
                video_pre["t_mod"],
                {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                video_mask,
            )
        return CachedActionCondition(
            context=context,
            context_mask=context_mask,
            video_kv_cache=[
                {
                    "k": key,
                    "v": value,
                    _GATE_CURRENT_FRAME_PROVENANCE_KEY: tokens_per_frame,
                }
                for key, value in zip(keys, values, strict=True)
            ],
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            current_frame_video_tokens=tokens_per_frame,
        )

    def _current_frame_gate_features(
        self, condition: CachedActionCondition
    ) -> RouteNeutralVisualFeatures:
        """Read current Video K/V without projecting unused Action context."""

        layers = []
        current_length = condition.current_frame_video_tokens
        for layer_index in self.route_neutral_visual.layer_indices:
            cache = condition.video_kv_cache[layer_index]
            self.actor.mot._validate_condition_cache_provenance(
                layer_index=layer_index,
                layer_cache=cache,
                current_frame_video_tokens=current_length,
            )
            key = cache["k"][:, :current_length].detach()
            layers.append(
                RouteNeutralVisualLayer(
                    layer_index=layer_index,
                    current_frame_video=KeyValueBank(
                        source=KVSource.CURRENT_FRAME_VIDEO,
                        key=key,
                        value=cache["v"][:, :current_length].detach(),
                        valid_mask=torch.ones(
                            key.shape[:2], dtype=torch.bool, device=key.device
                        ),
                        contains_generated_future_video=False,
                    ),
                )
            )
        features = RouteNeutralVisualFeatures(tuple(layers))
        if features.feature_dim != self.route_neutral_visual.source_dim:
            raise ValueError(
                "Route-neutral visual K/V width differs from configuration."
            )
        return features

    @torch.no_grad()
    def critic_features(self, *, env_obs: dict[str, Any]) -> FastWAMValueFeatures:
        """Keep bootstrap values on the same frozen features used in replay."""

        if self.video_lora_adapter is None:
            return super().critic_features(env_obs=env_obs)
        images, context, context_mask = self._encode_condition(env_obs)
        condition, _ = self._prepare_parent_current_condition(
            image=images, context=context, context_mask=context_mask
        )
        return self._critic_features_from_condition(condition)

    def _velocity(
        self,
        condition: CachedActionCondition,
        *,
        regime: PolicyRegime,
        capture_gate_kv: bool,
        actor_version: int,
    ) -> CachedActionVelocity:
        velocity = super()._velocity(
            condition,
            regime=regime,
            capture_gate_kv=capture_gate_kv,
            actor_version=actor_version,
        )
        velocity.batch_linear_context = self.batch_linear_context
        return velocity

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
    def _encode_evaluation_condition(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep the latest instruction resident, then append fresh proprio."""

        prompts = _format_fastwam_prompts(
            env_obs["task_descriptions"], prompt_template=self.prompt_template
        )
        key = (
            tuple(prompts),
            self.text_embedding_cache_dir,
            self.text_embedding_context_len,
            self.device,
            self.dtype,
        )
        cached = self._evaluation_text_context
        if cached is None or cached[0] != key:
            if self.text_embedding_cache_dir is None:
                context, mask = self.actor.encode_prompt(prompts)
            else:
                context, mask = _load_cached_text_contexts(
                    prompts,
                    cache_dir=self.text_embedding_cache_dir,
                    context_len=self.text_embedding_context_len,
                    expected_dim=int(self.actor.text_dim),
                    device=self.device,
                    dtype=self.dtype,
                    text_padding=getattr(self.actor, "text_padding", "legacy_visible"),
                )
            cached = (key, (context.detach(), mask.detach()))
            self._evaluation_text_context = cached
        return self._encode_condition(env_obs, text_context=cached[1])

    @torch.no_grad()
    def prepare_route_neutral_step(
        self,
        *,
        env_obs: dict[str, Any],
        include_critic_features: bool = True,
        mode: Literal["train", "eval"] = "train",
    ) -> RouteNeutralPreparedStep:
        """Build one batched current-only condition before route choice."""

        single_eval = mode == "eval" and env_obs["states"].shape[0] == 1
        images, context, context_mask = (
            self._encode_evaluation_condition(env_obs)
            if single_eval
            else self._encode_condition(env_obs)
        )
        batch_size = int(images.shape[0])
        state = self._normalized_proprio(env_obs["states"]).detach()
        if state.shape != (batch_size, self.route_neutral_input.state_dim):
            raise ValueError("Route-neutral normalized proprio shape changed.")
        if context.shape[1] < 2 or not bool(context_mask[:, -1].all().item()):
            raise ValueError("Could not isolate FastWAM's appended proprio token.")

        first_frame_latents = (
            self.actor._encode_input_image_latents_tensor(images, tiled=self.tiled_vae)
            if single_eval or self.video_lora_adapter is not None
            else None
        )
        condition_kwargs = (
            {"first_frame_latents": first_frame_latents}
            if first_frame_latents is not None
            else {}
        )
        need_critic = include_critic_features and self.critic_feature_config is not None
        if single_eval and self.video_lora_adapter is not None and not need_critic:
            condition = self._prepare_parent_gate_condition(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
            )
            replay_noise = None
        elif self.video_lora_adapter is not None:
            condition, replay_noise = self._prepare_parent_current_condition(
                image=images,
                context=context,
                context_mask=context_mask,
                **condition_kwargs,
            )
        else:
            condition, replay_noise = self._prepare_action_condition(
                image=images,
                context=context,
                context_mask=context_mask,
                regime=PolicyRegime.UNCOND,
                **condition_kwargs,
            )
        if replay_noise is not None:
            raise AssertionError("Current-frame condition created future noise.")
        visual_features = self._current_frame_gate_features(condition)
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
        gate_features = RouteNeutralGateFeatures(
            visual=visual_features,
            language=context[:, :-1].detach(),
            language_mask=context_mask[:, :-1].detach().to(dtype=torch.bool),
            state=state,
            physical_history=history,
        )
        critic_features = (
            self._critic_features_from_condition(condition) if need_critic else None
        )
        return RouteNeutralPreparedStep(
            images=images,
            context=context,
            context_mask=context_mask,
            current_condition=condition,
            gate_features=gate_features,
            critic_features=critic_features,
            first_frame_latents=first_frame_latents,
        )

    @torch.no_grad()
    def prepare_route_neutral_gate_features(
        self,
        *,
        env_obs: dict[str, Any],
    ) -> RouteNeutralGateFeatures:
        """Retain the legacy feature-only entry point for evaluation callers."""

        return self.prepare_route_neutral_step(
            env_obs=env_obs,
            include_critic_features=False,
        ).gate_features

    def _seeded_idm_latents(
        self,
        *,
        images: torch.Tensor,
        seeds: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize independent per-sample IDM latents before batching."""

        latent_t = (
            self.num_video_frames - 1
        ) // self.actor.vae.temporal_downsample_factor + 1
        shape = (
            1,
            self.actor.vae.model.z_dim,
            latent_t,
            images.shape[-2] // self.actor.vae.upsampling_factor,
            images.shape[-1] // self.actor.vae.upsampling_factor,
        )
        return torch.cat(
            [
                _seeded_randn(
                    int(seed.item()),
                    shape,
                    device=self.device,
                    dtype=self.dtype,
                    rand_device=self.seeded_noise_device,
                )
                for seed in seeds.reshape(-1)
            ],
            dim=0,
        )

    def _sample_idm_teacher_batch(
        self,
        *,
        prepared: RouteNeutralPreparedStep,
        indices: torch.Tensor,
        initial_action_noise: torch.Tensor,
        idm_seeds: torch.Tensor,
        actor_version: int,
    ) -> tuple[torch.Tensor, float]:
        """Run one deterministic IDM teacher batch and return device time."""

        start_event = end_event = None
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))
        else:
            started = time.perf_counter()
        selected_images = prepared.images.index_select(
            0,
            indices.to(prepared.images.device),
        )
        selected_context = prepared.context.index_select(
            0,
            indices.to(prepared.context.device),
        )
        selected_context_mask = prepared.context_mask.index_select(
            0,
            indices.to(prepared.context_mask.device),
        )
        selected_seeds = idm_seeds.index_select(0, indices.cpu())
        initial_latents = self._seeded_idm_latents(
            images=selected_images,
            seeds=selected_seeds,
        )

        condition, _ = self._prepare_action_condition(
            image=selected_images,
            context=selected_context,
            context_mask=selected_context_mask,
            regime=PolicyRegime.IDM,
            idm_initial_latents=initial_latents,
            first_frame_latents=(
                None
                if prepared.first_frame_latents is None
                else prepared.first_frame_latents.index_select(0, indices)
            ),
        )
        timesteps, deltas = self._action_schedule()
        rollout = sample_action_flow_sde(
            initial_action_noise,
            velocity_fn=self._velocity(
                condition,
                regime=PolicyRegime.IDM,
                capture_gate_kv=False,
                actor_version=int(actor_version),
            ),
            timesteps=timesteps,
            scheduler_deltas=deltas,
            num_train_timesteps=self.actor.infer_action_scheduler.num_train_timesteps,
            noise_level=self.flow_sde_noise_level,
            gate_last_n=1,
            ignore_last_transition=self.flow_sde_ignore_last_transition,
            stochastic=False,
            collect_replay=False,
        )
        if end_event is not None and start_event is not None:
            end_event.record(torch.cuda.current_stream(self.device))
            end_event.synchronize()
            elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
        else:
            elapsed = time.perf_counter() - started
        return rollout.actions, elapsed

    @torch.no_grad()
    def _sample_seeded_training_batch(
        self,
        *,
        env_obs: dict[str, Any],
        routes: torch.Tensor,
        actor_version: int,
        prepared: RouteNeutralPreparedStep,
    ) -> RouteNeutralTrainableChunkSample:
        """Vectorize one formal-training batch without merging RNG streams."""

        batch_size = int(routes.numel())
        routes = routes.reshape(-1).to(device=self.device, dtype=torch.long)
        action_seeds = _validate_noise_seeds(
            env_obs["_fastwam_action_noise_seeds"],
            batch_size=batch_size,
            name="action noise",
        )
        idm_seeds = _validate_noise_seeds(
            env_obs["_fastwam_idm_noise_seeds"],
            batch_size=batch_size,
            name="IDM video noise",
        )
        action_shape = (
            1,
            self.action_protocol.generation_horizon,
            self.actor.action_expert.action_dim,
        )
        initial_noise = torch.cat(
            [
                _seeded_randn(
                    int(seed.item()),
                    action_shape,
                    device=self.device,
                    dtype=self.dtype,
                    rand_device=self.seeded_noise_device,
                )
                for seed in action_seeds
            ],
            dim=0,
        )
        timesteps, deltas = self._action_schedule()
        uncond_indices = (
            (routes == int(WAMRoute.UNCOND)).nonzero(as_tuple=False).reshape(-1)
        )
        idm_indices = (routes == int(WAMRoute.IDM)).nonzero(as_tuple=False).reshape(-1)
        if int(uncond_indices.numel() + idm_indices.numel()) != batch_size:
            raise ValueError("Route-neutral batch contains an unknown route value.")

        actions = torch.empty_like(initial_noise)
        chains = torch.empty(
            (batch_size, timesteps.numel() + 1, *initial_noise.shape[1:]),
            device=self.device,
            dtype=self.dtype,
        )
        old_logprobs = torch.zeros_like(initial_noise, dtype=torch.float32)
        denoise_indices = torch.full(
            (batch_size,),
            -1,
            device=self.device,
            dtype=torch.long,
        )

        if uncond_indices.numel():
            selected_denoise = []
            selected_transition_noise = []
            for raw_index in uncond_indices.cpu():
                generator = torch.Generator(device=self.device)
                generator.manual_seed(
                    _domain_separated_noise_seed(
                        int(action_seeds[int(raw_index)].item()),
                        domain="flow-sde",
                    )
                )
                selected_denoise.append(
                    sample_denoise_indices(
                        1,
                        int(timesteps.numel()),
                        device=self.device,
                        generator=generator,
                        ignore_last=self.flow_sde_ignore_last_transition,
                    )
                )
                selected_transition_noise.append(
                    torch.randn(
                        action_shape,
                        generator=generator,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
            uncond_rollout = sample_action_flow_sde(
                initial_noise.index_select(0, uncond_indices),
                velocity_fn=self._velocity(
                    self._uncond_condition_from_prepared(prepared, uncond_indices),
                    regime=PolicyRegime.UNCOND,
                    capture_gate_kv=False,
                    actor_version=int(actor_version),
                ),
                timesteps=timesteps,
                scheduler_deltas=deltas,
                num_train_timesteps=(
                    self.actor.infer_action_scheduler.num_train_timesteps
                ),
                noise_level=self.flow_sde_noise_level,
                denoise_indices=torch.cat(selected_denoise),
                transition_noise=torch.cat(selected_transition_noise),
                gate_last_n=self.gate_denoise_last_n,
                ignore_last_transition=self.flow_sde_ignore_last_transition,
                stochastic=True,
                collect_replay=True,
            )
            actions.index_copy_(0, uncond_indices, uncond_rollout.actions)
            chains.index_copy_(0, uncond_indices, uncond_rollout.chains)
            old_logprobs.index_copy_(
                0,
                uncond_indices,
                uncond_rollout.old_log_probs,
            )
            denoise_indices.index_copy_(
                0,
                uncond_indices,
                uncond_rollout.denoise_indices,
            )

        if idm_indices.numel():
            selected_images = prepared.images.index_select(
                0,
                idm_indices.to(prepared.images.device),
            )
            selected_context = prepared.context.index_select(
                0,
                idm_indices.to(prepared.context.device),
            )
            selected_context_mask = prepared.context_mask.index_select(
                0,
                idm_indices.to(prepared.context_mask.device),
            )
            idm_latents = self._seeded_idm_latents(
                images=selected_images,
                seeds=idm_seeds.index_select(0, idm_indices.cpu()),
            )
            idm_condition, _ = self._prepare_action_condition(
                image=selected_images,
                context=selected_context,
                context_mask=selected_context_mask,
                regime=PolicyRegime.IDM,
                idm_initial_latents=idm_latents,
                first_frame_latents=(
                    None
                    if prepared.first_frame_latents is None
                    else prepared.first_frame_latents.index_select(0, idm_indices)
                ),
            )
            idm_rollout = sample_action_flow_sde(
                initial_noise.index_select(0, idm_indices),
                velocity_fn=self._velocity(
                    idm_condition,
                    regime=PolicyRegime.IDM,
                    capture_gate_kv=False,
                    actor_version=int(actor_version),
                ),
                timesteps=timesteps,
                scheduler_deltas=deltas,
                num_train_timesteps=(
                    self.actor.infer_action_scheduler.num_train_timesteps
                ),
                noise_level=self.flow_sde_noise_level,
                gate_last_n=self.gate_denoise_last_n,
                ignore_last_transition=self.flow_sde_ignore_last_transition,
                stochastic=False,
                collect_replay=True,
            )
            actions.index_copy_(0, idm_indices, idm_rollout.actions)
            chains.index_copy_(0, idm_indices, idm_rollout.chains)
            old_logprobs.index_copy_(0, idm_indices, idm_rollout.old_log_probs)
            denoise_indices.index_copy_(0, idm_indices, idm_rollout.denoise_indices)

        executed_actions = select_executed_action_prefix(
            actions,
            protocol=self.action_protocol,
        )
        processed_actions, action_execution_trace = self._denormalize_action_stages(
            executed_actions,
            env_obs=env_obs,
        )
        forward_inputs = {
            "fastwam_context": prepared.context.detach(),
            "fastwam_context_mask": prepared.context_mask.detach(),
        }
        if prepared.first_frame_latents is None:
            forward_inputs["fastwam_images"] = prepared.images.detach()
        else:
            forward_inputs["fastwam_first_frame_latents"] = (
                prepared.first_frame_latents.detach()
            )

        teacher_actions = torch.zeros_like(
            chains[:, -1],
            dtype=torch.bfloat16,
        )
        teacher_present = torch.zeros(
            batch_size,
            device=self.device,
            dtype=torch.bool,
        )
        teacher_seconds = torch.zeros(
            batch_size,
            device=self.device,
            dtype=torch.float32,
        )
        teacher_bytes = torch.zeros(
            batch_size,
            device=self.device,
            dtype=torch.long,
        )
        if uncond_indices.numel():
            target, elapsed = self._sample_idm_teacher_batch(
                prepared=prepared,
                indices=uncond_indices,
                initial_action_noise=chains.index_select(0, uncond_indices)[:, 0],
                idm_seeds=idm_seeds,
                actor_version=actor_version,
            )
            if not bool(torch.isfinite(target.float()).all().item()):
                raise FloatingPointError("IDM teacher produced a non-finite action.")
            teacher_actions.index_copy_(
                0,
                uncond_indices,
                target.to(dtype=torch.bfloat16),
            )
            teacher_present[uncond_indices] = True
            teacher_seconds[uncond_indices] = float(elapsed) / float(
                uncond_indices.numel()
            )
            bytes_per_target = int(
                teacher_actions[0].numel() * teacher_actions.element_size()
            )
            teacher_bytes[uncond_indices] = bytes_per_target

        collisions = sorted(
            set(forward_inputs).intersection(ONLINE_IDM_BC_FORWARD_KEYS)
        )
        if collisions:
            raise KeyError(f"Online IDM BC replay fields already exist: {collisions}.")
        forward_inputs.update(
            {
                ONLINE_IDM_BC_TEACHER_ACTIONS: teacher_actions.detach(),
                ONLINE_IDM_BC_TEACHER_PRESENT: teacher_present.detach(),
                ONLINE_IDM_BC_SAMPLE_IDENTITIES: action_seeds.to(self.device),
                ONLINE_IDM_BC_TEACHER_SECONDS: teacher_seconds.detach(),
                ONLINE_IDM_BC_TEACHER_BYTES: teacher_bytes.detach(),
                ROUTE_NEUTRAL_ROLLOUT_IDM_BATCH_SIZE: torch.full(
                    (batch_size,),
                    float(idm_indices.numel()),
                    device=self.device,
                ),
                ROUTE_NEUTRAL_ROLLOUT_UNCOND_BATCH_SIZE: torch.full(
                    (batch_size,),
                    float(uncond_indices.numel()),
                    device=self.device,
                ),
                ROUTE_NEUTRAL_TEACHER_BATCH_SIZE: torch.full(
                    (batch_size,),
                    float(uncond_indices.numel()),
                    device=self.device,
                ),
            }
        )
        return RouteNeutralTrainableChunkSample(
            actions=processed_actions,
            old_flow_logprobs=select_executed_flow_statistics(
                old_logprobs,
                protocol=self.action_protocol,
            ),
            flow_chains=chains,
            denoise_indices=denoise_indices,
            forward_inputs=forward_inputs,
            critic_features=prepared.critic_features,
            action_execution_trace=action_execution_trace,
        )

    @torch.no_grad()
    def _sample_prepared_evaluation(
        self,
        *,
        env_obs: dict[str, Any],
        routes: torch.Tensor,
        actor_version: int,
        prepared: RouteNeutralPreparedStep,
    ) -> RouteNeutralTrainableChunkSample:
        """Run one deterministic action chunk using its existing Gate condition.

        Keep batch one: the legacy evaluation branch uses serial B1 kernels,
        whereas a batched Gate prefill can have different numerical behavior.
        """

        if routes.shape != (1,):
            raise ValueError(
                "Prepared single-environment evaluation requires one route."
            )
        route = int(routes.item())
        if route not in (int(WAMRoute.IDM), int(WAMRoute.UNCOND)):
            raise ValueError("Evaluation route must be IDM or UNCOND.")
        regime = PolicyRegime.IDM if route == int(WAMRoute.IDM) else PolicyRegime.UNCOND
        action_shape = (
            1,
            self.action_protocol.generation_horizon,
            self.actor.action_expert.action_dim,
        )
        action_noise = env_obs.get("_fastwam_action_initial_noise")
        if action_noise is not None:
            action_noise = torch.as_tensor(
                action_noise, device=self.device, dtype=self.dtype
            )
            if tuple(action_noise.shape) != action_shape:
                raise ValueError(
                    f"Injected FastWAM action noise must have shape {action_shape}."
                )
        action_seeds = env_obs.get("_fastwam_action_noise_seeds")
        if action_seeds is not None:
            if action_noise is not None:
                raise ValueError(
                    "Specify either injected action noise or action seeds, not both."
                )
            action_seeds = _validate_noise_seeds(
                action_seeds, batch_size=1, name="action noise"
            )
        video_noise = env_obs.get("_fastwam_idm_initial_latents")
        if video_noise is not None:
            video_noise = torch.as_tensor(
                video_noise, device=self.device, dtype=self.dtype
            )
            if video_noise.shape[0] != 1:
                raise ValueError(
                    "Injected FastWAM IDM video noise batch must match routes."
                )
        idm_seeds = env_obs.get("_fastwam_idm_noise_seeds")
        if idm_seeds is not None:
            if video_noise is not None:
                raise ValueError(
                    "Specify either injected IDM latents or IDM seeds, not both."
                )
            idm_seeds = _validate_noise_seeds(
                idm_seeds, batch_size=1, name="IDM video noise"
            )

        condition = prepared.current_condition
        if regime is PolicyRegime.IDM:
            condition, _ = self._prepare_action_condition(
                image=prepared.images,
                context=prepared.context,
                context_mask=prepared.context_mask,
                regime=regime,
                idm_initial_latents=video_noise,
                idm_noise_seed=None if idm_seeds is None else int(idm_seeds[0]),
                first_frame_latents=prepared.first_frame_latents,
            )
        elif self.video_lora_adapter is not None:
            condition = self._uncond_condition_from_prepared(prepared)
        # Preserve the legacy order: IDM video noise precedes action noise.
        if action_noise is None:
            action_noise = (
                torch.randn(action_shape, device=self.device, dtype=torch.float32).to(
                    dtype=self.dtype
                )
                if action_seeds is None
                else _seeded_randn(
                    int(action_seeds[0]),
                    action_shape,
                    device=self.device,
                    dtype=self.dtype,
                    rand_device=self.seeded_noise_device,
                )
            )
        timesteps, deltas = self._action_schedule()
        rollout = sample_action_flow_sde(
            action_noise,
            velocity_fn=self._velocity(
                condition,
                regime=regime,
                capture_gate_kv=False,
                actor_version=actor_version,
            ),
            timesteps=timesteps,
            scheduler_deltas=deltas,
            num_train_timesteps=self.actor.infer_action_scheduler.num_train_timesteps,
            noise_level=self.flow_sde_noise_level,
            gate_last_n=1,
            ignore_last_transition=self.flow_sde_ignore_last_transition,
            stochastic=False,
            collect_replay=False,
        )
        executed = select_executed_action_prefix(
            rollout.actions, protocol=self.action_protocol
        )
        actions, trace = self._denormalize_action_stages(executed, env_obs=env_obs)
        return RouteNeutralTrainableChunkSample(
            actions=actions,
            old_flow_logprobs=select_executed_flow_statistics(
                rollout.old_log_probs, protocol=self.action_protocol
            ),
            flow_chains=rollout.chains,
            denoise_indices=rollout.denoise_indices,
            forward_inputs={},
            critic_features=None,
            action_execution_trace=trace,
        )

    def sample_routed_action_batch(
        self,
        *,
        env_obs: dict[str, Any],
        routes: torch.Tensor,
        mode: Literal["train", "eval"],
        actor_version: int,
        collect_replay: bool,
        prepared: RouteNeutralPreparedStep | None = None,
    ) -> RouteNeutralTrainableChunkSample:
        """Reuse B1 evaluation conditions; retain the existing training path."""

        _validate_flow_sde_sampling(
            mode=mode,
            routes=routes,
            noise_level=self.flow_sde_noise_level,
        )
        if (
            mode == "eval"
            and not collect_replay
            and prepared is not None
            and prepared.images.shape[0] == 1
        ):
            return self._sample_prepared_evaluation(
                env_obs=env_obs,
                routes=routes,
                actor_version=actor_version,
                prepared=prepared,
            )
        can_batch = (
            mode == "train"
            and collect_replay
            and prepared is not None
            and env_obs.get("_fastwam_action_noise_seeds") is not None
            and env_obs.get("_fastwam_idm_noise_seeds") is not None
            and env_obs.get("_fastwam_action_initial_noise") is None
            and env_obs.get("_fastwam_idm_initial_latents") is None
        )
        if can_batch:
            return self._sample_seeded_training_batch(
                env_obs=env_obs,
                routes=routes,
                actor_version=actor_version,
                prepared=prepared,
            )
        sample = super().sample_action_batch(
            env_obs=env_obs,
            routes=routes,
            mode=mode,
            actor_version=actor_version,
            collect_replay=collect_replay,
        )
        return RouteNeutralTrainableChunkSample.without_route_snapshot(sample)

    @staticmethod
    def _replay_condition_inputs(
        forward_inputs: dict[str, torch.Tensor],
        indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Use the frozen VAE result already computed by the dual rollout."""

        result = {
            "image": forward_inputs.get("fastwam_images"),
            "context": forward_inputs["fastwam_context"],
            "context_mask": forward_inputs["fastwam_context_mask"],
        }
        if "fastwam_first_frame_latents" in forward_inputs:
            result["first_frame_latents"] = forward_inputs[
                "fastwam_first_frame_latents"
            ]
        if indices is not None:
            result = {
                name: value.index_select(0, indices.to(value.device))
                if value is not None
                else None
                for name, value in result.items()
            }
        return result

    def replay_action_batch(
        self,
        *,
        forward_inputs: dict[str, torch.Tensor],
        route_info: ChunkRouteRecord,
        compute_base_logprobs: bool = False,
    ) -> dict[str, Any]:
        """Replay every UNCOND row in one batch and build critic features once."""

        _validate_flow_sde_sampling(
            mode="train",
            routes=route_info.route_used,
            noise_level=self.flow_sde_noise_level,
        )
        chains = forward_inputs["flow_chains"]
        indices = forward_inputs["denoise_indices"]
        condition_inputs = self._replay_condition_inputs(forward_inputs)
        if self.video_lora_adapter is None:
            current_condition, replay_noise = self._prepare_action_condition(
                **condition_inputs, regime=PolicyRegime.UNCOND
            )
        elif self.critic_feature_config is not None or compute_base_logprobs:
            current_condition, replay_noise = self._prepare_parent_current_condition(
                **condition_inputs
            )
        else:
            # The pi0.5 critic has its own frozen observation encoder. It does
            # not consume Video K/V, so replay needs only the adapted prefill.
            current_condition, replay_noise = None, None
        if replay_noise is not None:
            raise AssertionError("Current-only replay created future video noise.")
        timesteps, deltas = self._action_schedule()
        logprobs = torch.zeros_like(chains[:, 0], dtype=torch.float32)
        entropies = torch.zeros_like(chains[:, 0], dtype=torch.float32)
        base_kl = (
            torch.zeros_like(chains[:, 0], dtype=torch.float32)
            if compute_base_logprobs
            else None
        )
        routes = route_info.route_used.reshape(-1).to(self.device)
        selected = (routes == int(WAMRoute.UNCOND)).nonzero(as_tuple=False).reshape(-1)
        selected_condition = None
        if selected.numel():
            actor_versions = route_info.actor_versions.reshape(-1)
            actor_version = int(actor_versions[selected[0]].item())
            if self.video_lora_adapter is not None:
                selected_condition, _ = self._prepare_action_condition(
                    **self._replay_condition_inputs(forward_inputs, selected),
                    regime=PolicyRegime.UNCOND,
                )
            else:
                selected_condition = current_condition.index_select(selected)
            current_replay = replay_action_flow_sde_transition(
                chains.index_select(0, selected),
                indices.index_select(0, selected),
                velocity_fn=self._velocity(
                    selected_condition,
                    regime=PolicyRegime.UNCOND,
                    capture_gate_kv=False,
                    actor_version=actor_version,
                ),
                timesteps=timesteps,
                scheduler_deltas=deltas,
                num_train_timesteps=(
                    self.actor.infer_action_scheduler.num_train_timesteps
                ),
                noise_level=self.flow_sde_noise_level,
            )
            logprobs = logprobs.index_copy(0, selected, current_replay.log_prob)
            entropy = torch.broadcast_to(
                current_replay.std.float().log()
                + 0.5 * math.log(2.0 * math.pi * math.e),
                current_replay.mean.shape,
            )
            entropies = entropies.index_copy(0, selected, entropy)
            if base_kl is not None:
                with torch.no_grad():
                    base_replay = replay_action_flow_sde_transition(
                        chains.index_select(0, selected),
                        indices.index_select(0, selected),
                        velocity_fn=self._velocity(
                            current_condition.index_select(selected),
                            regime=PolicyRegime.IDM,
                            capture_gate_kv=False,
                            actor_version=actor_version,
                        ),
                        timesteps=timesteps,
                        scheduler_deltas=deltas,
                        num_train_timesteps=(
                            self.actor.infer_action_scheduler.num_train_timesteps
                        ),
                        noise_level=self.flow_sde_noise_level,
                    )
                selected_kl = (
                    0.5
                    * (
                        (current_replay.mean.float() - base_replay.mean.float())
                        / current_replay.std.float()
                    ).square()
                )
                base_kl = base_kl.index_copy(0, selected, selected_kl)
        result = {
            # This graph belongs only to the current forward. BC consumes a
            # subset of it before the combined loss is backpropagated.
            "uncond_condition": (selected, selected_condition),
            "flow_logprobs": select_executed_action_prefix(
                logprobs,
                protocol=self.action_protocol,
            ),
            "flow_entropy": select_executed_action_prefix(
                entropies,
                protocol=self.action_protocol,
            ),
        }
        if base_kl is not None:
            result["base_uncond_kl"] = select_executed_action_prefix(
                base_kl,
                protocol=self.action_protocol,
            )
        if self.critic_feature_config is not None:
            result["critic_features"] = self._critic_features_from_condition(
                current_condition
            )
        return result

    def compute_online_idm_bc_loss(
        self,
        *,
        forward_inputs: dict[str, torch.Tensor],
        route_info: ChunkRouteRecord,
        uncond_condition: tuple[torch.Tensor, CachedActionCondition | None]
        | None = None,
    ) -> OnlineIDMBCLossBatch:
        """Compute the online BC numerator with one selected-row forward."""

        required = set(ONLINE_IDM_BC_FORWARD_KEYS) | {
            ONLINE_IDM_BC_FLOW_VALID,
            "flow_chains",
            "fastwam_context",
            "fastwam_context_mask",
        }
        required.add(
            "fastwam_first_frame_latents"
            if "fastwam_first_frame_latents" in forward_inputs
            else "fastwam_images"
        )
        missing = sorted(required - set(forward_inputs))
        if missing:
            raise KeyError(f"Online IDM BC actor replay is missing fields: {missing}.")

        routes = route_info.route_used.reshape(-1).to(self.device)
        batch_size = int(routes.numel())
        flow_valid = forward_inputs[ONLINE_IDM_BC_FLOW_VALID].bool().reshape(-1)
        present = forward_inputs[ONLINE_IDM_BC_TEACHER_PRESENT].bool().reshape(-1)
        identities = forward_inputs[ONLINE_IDM_BC_SAMPLE_IDENTITIES].long().reshape(-1)
        teacher_actions = forward_inputs[ONLINE_IDM_BC_TEACHER_ACTIONS]
        teacher_seconds = (
            forward_inputs[ONLINE_IDM_BC_TEACHER_SECONDS].float().reshape(-1)
        )
        teacher_bytes = forward_inputs[ONLINE_IDM_BC_TEACHER_BYTES].long().reshape(-1)
        expected_shape = (
            batch_size,
            self.action_protocol.generation_horizon,
            self.actor.action_expert.action_dim,
        )
        if tuple(teacher_actions.shape) != expected_shape:
            raise ValueError(
                "Online IDM BC teacher actions must have shape "
                f"{expected_shape}, got {tuple(teacher_actions.shape)}."
            )
        for name, value in {
            "flow-valid": flow_valid,
            "teacher-present": present,
            "sample identities": identities,
            "teacher seconds": teacher_seconds,
            "teacher bytes": teacher_bytes,
        }.items():
            if value.shape != (batch_size,):
                raise ValueError(
                    f"Online IDM BC {name} must have shape ({batch_size},)."
                )
        if teacher_actions.dtype is not torch.bfloat16:
            raise TypeError("Online IDM BC teacher actions must use BF16 transport.")
        if bool((teacher_seconds < 0).any().item()) or bool(
            (teacher_bytes < 0).any().item()
        ):
            raise ValueError(
                "Online IDM BC teacher time/byte metrics must be nonnegative."
            )

        is_uncond = routes == int(WAMRoute.UNCOND)
        expected = flow_valid.to(self.device) & is_uncond
        present = present.to(self.device)
        missing_teacher = is_uncond & ~present
        if bool(missing_teacher.any().item()):
            indices = missing_teacher.nonzero(as_tuple=False).reshape(-1).tolist()
            raise RuntimeError(
                f"UNCOND rows lack IDM teacher targets at indices {indices}."
            )
        unexpected_teacher = present & ~is_uncond
        if bool(unexpected_teacher.any().item()):
            indices = unexpected_teacher.nonzero(as_tuple=False).reshape(-1).tolist()
            raise RuntimeError(
                f"IDM-routed rows carry teacher targets at indices {indices}."
            )
        selected = expected & present
        selected_indices = selected.nonzero(as_tuple=False).reshape(-1)
        selected_count = selected.sum().to(device=self.device, dtype=torch.float32)

        lora_parameter = next(self.lora_adapter.lora_parameters())
        differentiable_zero = lora_parameter.reshape(-1)[0] * 0.0
        action_dim = int(teacher_actions.shape[-1])
        timestep_bins = 10
        zero_dimensions = torch.zeros(
            action_dim,
            device=self.device,
            dtype=torch.float32,
        )
        zero_bins = torch.zeros(
            timestep_bins,
            device=self.device,
            dtype=torch.float32,
        )
        zero_bin_counts = torch.zeros(
            timestep_bins,
            device=self.device,
            dtype=torch.long,
        )
        zero_action_count = torch.zeros((), device=self.device, dtype=torch.long)
        per_sample_loss = (
            torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            if "multitask_task_id" in forward_inputs
            else None
        )

        if selected_indices.numel():
            action = teacher_actions.index_select(0, selected_indices).to(
                device=self.device,
                dtype=self.dtype,
            )
            selected_identities = identities.index_select(
                0,
                selected_indices.to(identities.device),
            )
            timestep, noise = stateless_validation_flow_inputs(
                sample_identities=[
                    int(value) for value in selected_identities.tolist()
                ],
                action_shape=tuple(action.shape),
                scheduler=self.actor.train_action_scheduler,
                seed=0,
                device=self.device,
                dtype=self.dtype,
            )
            noisy_action = self.actor.train_action_scheduler.add_noise(
                action,
                noise,
                timestep,
            )
            velocity_target = self.actor.train_action_scheduler.training_target(
                action,
                noise,
                timestep,
            )
            if uncond_condition is None:
                condition, replay_noise = self._prepare_action_condition(
                    **self._replay_condition_inputs(forward_inputs, selected_indices),
                    regime=PolicyRegime.UNCOND,
                )
                if replay_noise is not None:
                    raise AssertionError(
                        "Online BC current condition created future noise."
                    )
            else:
                replay_indices, replay_condition = uncond_condition
                # Both index vectors follow the original microbatch row order.
                if selected_indices.numel() == replay_indices.numel():
                    condition = replay_condition
                else:
                    bc_in_replay = torch.searchsorted(replay_indices, selected_indices)
                    condition = replay_condition.index_select(bc_in_replay)
            actor_versions = route_info.actor_versions.reshape(-1)
            prediction = self._velocity(
                condition,
                regime=PolicyRegime.UNCOND,
                capture_gate_kv=False,
                actor_version=int(actor_versions[selected_indices[0]].item()),
            )(noisy_action, timestep).velocity
            bc_result = compute_action_flow_matching_bc_loss(
                prediction=prediction,
                target=velocity_target,
                timestep=timestep,
                action_is_pad=None,
                scheduler=self.actor.train_action_scheduler,
                gripper_dimension=6,
                timestep_bins=timestep_bins,
            )
            loss_sum = bc_result.loss_action_bc * selected_count
            if per_sample_loss is not None:
                # Record the same full-action, scheduler-weighted row objective.
                # This detached diagnostic does not change the trained reduction.
                with torch.no_grad():
                    row_mse = (
                        prediction.float() - velocity_target.float()
                    ).square().mean(dim=2).sum(dim=1) / prediction.shape[1]
                    per_sample_loss[selected_indices] = (
                        row_mse
                        * self.actor.train_action_scheduler.training_weight(
                            timestep
                        ).float()
                    )
            raw_loss = bc_result.loss_action_bc.detach()
            mse_per_dimension = bc_result.mse_per_dimension.detach()
            mse_pose = bc_result.mse_pose.detach()
            mse_gripper = bc_result.mse_gripper.detach()
            mse_by_timestep_bin = bc_result.mse_by_timestep_bin.detach()
            timestep_bin_count = bc_result.timestep_bin_count.detach()
            valid_action_count = bc_result.valid_action_count.detach()
        else:
            loss_sum = differentiable_zero
            raw_loss = differentiable_zero.detach()
            mse_per_dimension = zero_dimensions
            mse_pose = differentiable_zero.detach()
            mse_gripper = differentiable_zero.detach()
            mse_by_timestep_bin = zero_bins
            timestep_bin_count = zero_bin_counts
            valid_action_count = zero_action_count

        student_actions = forward_inputs["flow_chains"][:, -1].float()
        action_error = (student_actions - teacher_actions.float()).square()
        if selected_indices.numel():
            selected_error = action_error[selected]
            full_action_mse = selected_error.mean()
            executed_prefix_mse = selected_error[
                :, : self.action_protocol.execution_horizon
            ].mean()
        else:
            full_action_mse = differentiable_zero.detach()
            executed_prefix_mse = differentiable_zero.detach()

        return OnlineIDMBCLossBatch(
            loss_sum=loss_sum.float(),
            raw_loss=raw_loss.float(),
            selected_count=selected_count,
            expected_count=expected.sum().to(self.device, dtype=torch.float32),
            present_count=present.sum().to(self.device, dtype=torch.float32),
            mse_per_dimension=mse_per_dimension.float(),
            mse_pose=mse_pose.float(),
            mse_gripper=mse_gripper.float(),
            mse_by_timestep_bin=mse_by_timestep_bin.float(),
            timestep_bin_count=timestep_bin_count,
            valid_action_count=valid_action_count,
            full_action_mse=full_action_mse.detach().float(),
            executed_prefix_mse=executed_prefix_mse.detach().float(),
            teacher_seconds_sum=teacher_seconds[present.to(teacher_seconds.device)]
            .sum()
            .to(self.device),
            teacher_bytes_sum=teacher_bytes[present.to(teacher_bytes.device)]
            .sum()
            .to(self.device),
            per_sample_loss=per_sample_loss,
        )


__all__ = [
    "ROUTE_NEUTRAL_ROLLOUT_IDM_BATCH_SIZE",
    "ROUTE_NEUTRAL_ROLLOUT_UNCOND_BATCH_SIZE",
    "ROUTE_NEUTRAL_TEACHER_BATCH_SIZE",
    "RouteNeutralOnlineIDMTeacherLiberoRuntime",
    "RouteNeutralPreparedStep",
    "RouteNeutralTrainableChunkSample",
]
