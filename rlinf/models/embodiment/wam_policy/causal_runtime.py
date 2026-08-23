# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Same-chunk causal runtime, isolated from adaptive delayed routing."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, TypeVar

import torch
from fastwam.causal_prediction import (
    CausalComputeMode,
    CausalControlKind,
    CausalInterventionSpecV2,
    compact_current_condition,
    compose_controlled_video_latents,
    derive_generic_proposal_seed,
    normalized_action_medoid,
    pool_current_kv_layers,
    splice_exact_current_prefix,
)
from fastwam.causal_prediction.cached_action import (
    SharedCachedActionVelocity,
    assert_current_prefix_cache_equal,
)
from fastwam.models.wan22.adaptive_action import CachedActionCondition
from fastwam.models.wan22.adaptive_sampler import sample_action_flow_sde

from rlinf.envs.action_contract import ActionExecutionTrace
from rlinf.envs.libero.action_protocol import select_executed_action_prefix

from .libero_runtime import (
    LiberoFastWAMRuntime,
    _format_fastwam_prompts,
    _load_cached_text_contexts,
    _seeded_randn,
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class CausalConditionContract:
    """Logical nine-frame interface plus the physically materialized K/V bank."""

    mode: CausalComputeMode
    logical_input_frames: int
    logical_future_frames: int
    current_frame_video_tokens: int
    physical_video_tokens: int
    physical_future_tokens: int
    attention_contract: str = "current-prefix-causal"
    control: CausalControlKind = CausalControlKind.STANDARD

    def __post_init__(self) -> None:
        if self.logical_input_frames != 9 or self.logical_future_frames != 8:
            raise ValueError("Causal v1 keeps the logical 9-frame/8-future interface.")
        expected_future = self.physical_video_tokens - self.current_frame_video_tokens
        if expected_future != self.physical_future_tokens or expected_future < 0:
            raise ValueError("Physical causal-condition token counts are inconsistent.")
        if not self.mode.reads_future_condition and self.physical_future_tokens != 0:
            raise ValueError(
                "A no-read action condition physically contains future K/V."
            )
        object.__setattr__(self, "control", CausalControlKind.parse(self.control))


@dataclass(frozen=True)
class CausalChunkSample:
    """One current-chunk action treatment and its measured critical path."""

    mode: CausalComputeMode
    actions: torch.Tensor
    normalized_actions: torch.Tensor
    action_execution_trace: ActionExecutionTrace | None
    condition_contract: CausalConditionContract
    video_denoise_calls: int
    latency_ms: dict[str, float]
    control: CausalControlKind = CausalControlKind.STANDARD
    proposal_seeds: tuple[int, ...] = ()
    medoid_index: int | None = None
    flow_statistics: dict[str, float] | None = None
    action_denoise_calls: int = 0


@dataclass(frozen=True)
class CausalPreGateCaptureV1:
    """Detached online features plus separately reported extraction cost."""

    features: dict[str, torch.Tensor]
    proposal_latency_ms: float
    proposal_count: int


class CausalLiberoFastWAMRuntime(LiberoFastWAMRuntime):
    """Execute current-chunk causal treatments with one shared ActionDiT LoRA.

    The class reuses image, text, action-normalization, and scheduler utilities
    from the production runtime. It does not use ``PolicyRegime``, the delayed
    route tracker, Gate replay, or Flow-SDE policy ratios.
    """

    def _timed(self, operation: Callable[[], _T]) -> tuple[_T, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        result = operation()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return result, (time.perf_counter() - start) * 1000.0

    def _encode_pre_gate_condition(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return image, language, proprio, and action context before prediction."""

        images = self._model_images(env_obs)
        prompts = _format_fastwam_prompts(
            env_obs["task_descriptions"],
            prompt_template=self.prompt_template,
        )
        if self.text_embedding_cache_dir is None:
            language, language_mask = self.actor.encode_prompt(prompts)
        else:
            language, language_mask = _load_cached_text_contexts(
                prompts,
                cache_dir=self.text_embedding_cache_dir,
                context_len=self.text_embedding_context_len,
                expected_dim=int(self.actor.text_dim),
                device=self.device,
                dtype=self.dtype,
            )
        proprio = self._normalized_proprio(env_obs["states"])
        context, context_mask = self.actor._append_proprio_to_context(
            context=language,
            context_mask=language_mask,
            proprio=proprio,
        )
        return images, language, language_mask, proprio, context, context_mask

    def _condition_from_video_latents(
        self,
        *,
        video_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_flag: bool,
    ) -> CachedActionCondition:
        batch_size = int(video_latents.shape[0])
        video_pre = self.actor.video_expert.pre_dit(
            x=video_latents,
            timestep=torch.zeros(
                batch_size,
                device=self.device,
                dtype=self.dtype,
            ),
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
        cache = self.actor.mot.prefill_video_cache(
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
        return CachedActionCondition(
            context=context,
            context_mask=context_mask,
            video_kv_cache=cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            current_frame_video_tokens=tokens_per_frame,
        )

    def _full_shape_null_current_condition(
        self,
        *,
        first_frame: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_flag: bool,
    ) -> CachedActionCondition:
        """Prefill logical null future slots, then retain only current K/V."""

        temporal_factor = int(self.actor.vae.temporal_downsample_factor)
        latent_t = (self.num_video_frames - 1) // temporal_factor + 1
        null_latents = first_frame.new_zeros(
            (
                first_frame.shape[0],
                first_frame.shape[1],
                latent_t,
                first_frame.shape[3],
                first_frame.shape[4],
            )
        )
        null_latents[:, :, :1] = first_frame
        full_condition = self._condition_from_video_latents(
            video_latents=null_latents,
            context=context,
            context_mask=context_mask,
            fuse_flag=fuse_flag,
        )
        return compact_current_condition(full_condition)

    @torch.no_grad()
    def _generate_c2_video_latents(
        self,
        *,
        image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float]]:
        """Generate a full C2 latent trajectory while preserving its prefix."""

        first_frame, vae_ms = self._timed(
            lambda: self.actor._encode_input_image_latents_tensor(
                image,
                tiled=self.tiled_vae,
            )
        )
        fuse_flag = bool(
            getattr(self.actor.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        latent_t = (
            self.num_video_frames - 1
        ) // self.actor.vae.temporal_downsample_factor + 1
        expected_shape = (
            int(image.shape[0]),
            self.actor.vae.model.z_dim,
            latent_t,
            image.shape[-2] // self.actor.vae.upsampling_factor,
            image.shape[-1] // self.actor.vae.upsampling_factor,
        )
        video_latents = _seeded_randn(
            int(video_seed),
            expected_shape,
            device=self.device,
            dtype=self.dtype,
            rand_device=self.seeded_noise_device,
        )
        video_latents[:, :, :1] = first_frame
        timesteps, deltas = self.actor.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=self.device,
            dtype=self.dtype,
            shift_override=self.sigma_shift,
        )

        def denoise() -> torch.Tensor:
            latents = video_latents
            for timestep, delta in zip(timesteps, deltas):
                velocity = self.actor._video_denoise_step_compiled(
                    latents_video=latents,
                    timestep_video=timestep.expand(image.shape[0]).to(dtype=self.dtype),
                    context=context,
                    context_mask=context_mask,
                    fuse_flag=fuse_flag,
                )
                latents = self.actor.infer_video_scheduler.step(
                    velocity,
                    delta,
                    latents,
                )
                latents[:, :, :1] = first_frame
            return latents

        full, video_ms = self._timed(denoise)
        return (
            first_frame,
            full,
            int(timesteps.numel()),
            {"vae": vae_ms, "video_denoise": video_ms},
        )

    @torch.no_grad()
    def _prepare_causal_condition(
        self,
        *,
        image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        mode: CausalComputeMode,
        video_seed: int | None,
    ) -> tuple[
        CachedActionCondition,
        int,
        dict[str, float],
        CausalConditionContract,
    ]:
        mode = CausalComputeMode.parse(mode)
        if mode is CausalComputeMode.C1_ONE_PASS:
            if video_seed is None:
                raise ValueError("C1 one-pass conditioning requires a video seed.")
            fusion = getattr(self.actor, "causal_c1_interval_fusion", None)
            if fusion is None:
                raise RuntimeError(
                    "C1_ONE_PASS requires an accepted tri-mode checkpoint with "
                    "attached interval-fusion weights."
                )
            first_frame, vae_ms = self._timed(
                lambda: self.actor._encode_input_image_latents_tensor(
                    image,
                    tiled=self.tiled_vae,
                )
            )
            temporal_factor = int(self.actor.vae.temporal_downsample_factor)
            latent_t = (self.num_video_frames - 1) // temporal_factor + 1
            slots = _seeded_randn(
                int(video_seed),
                (
                    int(image.shape[0]),
                    int(first_frame.shape[1]),
                    latent_t,
                    int(first_frame.shape[3]),
                    int(first_frame.shape[4]),
                ),
                device=self.device,
                dtype=self.dtype,
                rand_device=self.seeded_noise_device,
            )
            slots[:, :, :1] = first_frame
            fuse_flag = bool(
                getattr(
                    self.actor.video_expert,
                    "fuse_vae_embedding_in_latents",
                    False,
                )
            )

            def one_pass() -> CachedActionCondition:
                video_pre = self.actor.video_expert.pre_dit(
                    x=slots,
                    timestep=torch.ones(
                        image.shape[0], device=self.device, dtype=self.dtype
                    ),
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
                cache = self.actor.mot.prefill_video_cache(
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
                cache = fusion(cache, current_token_count=tokens_per_frame)
                full_condition = CachedActionCondition(
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    current_frame_video_tokens=tokens_per_frame,
                )
                current_condition = self._condition_from_video_latents(
                    video_latents=first_frame,
                    context=context,
                    context_mask=context_mask,
                    fuse_flag=fuse_flag,
                )
                return splice_exact_current_prefix(
                    current_condition,
                    full_condition,
                )

            condition, one_pass_ms = self._timed(one_pass)
            return (
                condition,
                1,
                {
                    "vae": vae_ms,
                    "video_denoise": one_pass_ms,
                    "video_prefill": 0.0,
                },
                CausalConditionContract(
                    mode=mode,
                    logical_input_frames=self.num_video_frames,
                    logical_future_frames=self.num_video_frames - 1,
                    current_frame_video_tokens=condition.current_frame_video_tokens,
                    physical_video_tokens=condition.video_seq_len,
                    physical_future_tokens=(
                        condition.video_seq_len - condition.current_frame_video_tokens
                    ),
                ),
            )
        fuse_flag = bool(
            getattr(self.actor.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        if not mode.runs_future_prediction:
            first_frame, vae_ms = self._timed(
                lambda: self.actor._encode_input_image_latents_tensor(
                    image,
                    tiled=self.tiled_vae,
                )
            )
            condition, prefill_ms = self._timed(
                lambda: self._full_shape_null_current_condition(
                    first_frame=first_frame,
                    context=context,
                    context_mask=context_mask,
                    fuse_flag=fuse_flag,
                )
            )
            return (
                condition,
                0,
                {
                    "vae": vae_ms,
                    "video_denoise": 0.0,
                    "video_prefill": prefill_ms,
                },
                CausalConditionContract(
                    mode=mode,
                    logical_input_frames=self.num_video_frames,
                    logical_future_frames=self.num_video_frames - 1,
                    current_frame_video_tokens=condition.current_frame_video_tokens,
                    physical_video_tokens=condition.video_seq_len,
                    physical_future_tokens=0,
                ),
            )

        if video_seed is None:
            raise ValueError("C2 and G interventions require an explicit video seed.")
        _first_frame, video_latents, video_calls, generated_latency = (
            self._generate_c2_video_latents(
                image=image,
                context=context,
                context_mask=context_mask,
                video_seed=video_seed,
            )
        )
        vae_ms = generated_latency["vae"]
        video_ms = generated_latency["video_denoise"]

        def prefill_condition() -> CachedActionCondition:
            full_condition = self._condition_from_video_latents(
                video_latents=video_latents,
                context=context,
                context_mask=context_mask,
                fuse_flag=fuse_flag,
            )
            return (
                compact_current_condition(full_condition)
                if mode is CausalComputeMode.G_NO_READ
                else full_condition
            )

        condition, prefill_ms = self._timed(prefill_condition)
        return (
            condition,
            video_calls,
            {
                "vae": vae_ms,
                "video_denoise": video_ms,
                "video_prefill": prefill_ms,
            },
            CausalConditionContract(
                mode=mode,
                logical_input_frames=self.num_video_frames,
                logical_future_frames=self.num_video_frames - 1,
                current_frame_video_tokens=condition.current_frame_video_tokens,
                physical_video_tokens=condition.video_seq_len,
                physical_future_tokens=(
                    condition.video_seq_len - condition.current_frame_video_tokens
                ),
            ),
        )

    @torch.no_grad()
    def _prepare_intervention_condition(
        self,
        *,
        image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        spec: CausalInterventionSpecV2,
        donor_future_latents: torch.Tensor | None,
        prediction_context: tuple[torch.Tensor, torch.Tensor] | None,
        ground_truth_future_latents: torch.Tensor | None,
    ) -> tuple[
        CachedActionCondition,
        int,
        dict[str, float],
        CausalConditionContract,
    ]:
        """Prepare v2 action K/V after the specified information intervention."""

        if spec.control is CausalControlKind.STANDARD:
            return self._prepare_causal_condition(
                image=image,
                context=context,
                context_mask=context_mask,
                mode=spec.mode,
                video_seed=spec.video_seed,
            )
        if spec.control is CausalControlKind.GENERIC_MEDOID:
            raise ValueError(
                "Generic medoid is prepared as multiple complete C0 calls."
            )

        prediction_context_value, prediction_mask = (
            prediction_context
            if prediction_context is not None
            else (context, context_mask)
        )
        first, generated, calls, latency = self._generate_c2_video_latents(
            image=image,
            context=prediction_context_value,
            context_mask=prediction_mask,
            video_seed=int(spec.video_seed),
        )
        supported = {
            CausalControlKind.NO_READ,
            CausalControlKind.REPEAT_CURRENT,
            CausalControlKind.SHUFFLED_WRONG_STATE,
            CausalControlKind.TEMPORAL_SHIFT,
            CausalControlKind.INSTRUCTION_MISMATCH,
            CausalControlKind.GT_FUTURE_OFFLINE,
        }
        if spec.control not in supported:
            raise ValueError(f"Unsupported causal v2 control {spec.control.value}.")
        selected = compose_controlled_video_latents(
            first_frame=first,
            generated_full=generated,
            control=spec.control,
            donor_future=donor_future_latents,
            ground_truth_future=ground_truth_future_latents,
        )

        fuse_flag = bool(
            getattr(self.actor.video_expert, "fuse_vae_embedding_in_latents", False)
        )

        def prefill_condition() -> CachedActionCondition:
            full_condition = self._condition_from_video_latents(
                video_latents=selected,
                context=context,
                context_mask=context_mask,
                fuse_flag=fuse_flag,
            )
            return (
                compact_current_condition(full_condition)
                if spec.control is CausalControlKind.NO_READ
                else full_condition
            )

        condition, prefill_ms = self._timed(prefill_condition)
        latency = {
            **latency,
            "video_prefill": prefill_ms,
        }
        return (
            condition,
            calls,
            latency,
            CausalConditionContract(
                mode=spec.mode,
                logical_input_frames=self.num_video_frames,
                logical_future_frames=self.num_video_frames - 1,
                current_frame_video_tokens=condition.current_frame_video_tokens,
                physical_video_tokens=condition.video_seq_len,
                physical_future_tokens=(
                    condition.video_seq_len - condition.current_frame_video_tokens
                ),
                control=spec.control,
            ),
        )

    @torch.no_grad()
    def _sample_from_condition(
        self,
        *,
        env_obs: dict[str, Any],
        mode: CausalComputeMode,
        control: CausalControlKind,
        condition: CachedActionCondition,
        condition_contract: CausalConditionContract,
        action_seed: int,
        video_calls: int,
        latency: dict[str, float],
        collect_flow_statistics: bool,
    ) -> CausalChunkSample:
        """Generate one action proposal from a prepared causal condition."""

        initial_noise = _seeded_randn(
            int(action_seed),
            (
                1,
                self.action_protocol.generation_horizon,
                self.actor.action_expert.action_dim,
            ),
            device=self.device,
            dtype=self.dtype,
            rand_device=self.seeded_noise_device,
        )
        timesteps, deltas = self._action_schedule()
        rollout, action_ms = self._timed(
            lambda: sample_action_flow_sde(
                initial_noise,
                velocity_fn=SharedCachedActionVelocity(
                    action_expert=self.actor.action_expert,
                    mot=self.actor.mot,
                    condition=condition,
                ),
                timesteps=timesteps,
                scheduler_deltas=deltas,
                num_train_timesteps=self.actor.infer_action_scheduler.num_train_timesteps,
                noise_level=0.0,
                stochastic=False,
                collect_replay=collect_flow_statistics,
            )
        )
        executed = select_executed_action_prefix(
            rollout.actions,
            protocol=self.action_protocol,
        )
        actions, trace = self._denormalize_action_stages(executed, env_obs=env_obs)
        latency = {**latency, "action_dit": action_ms}
        latency["critical_path"] = sum(latency.values())
        flow_statistics = None
        if collect_flow_statistics:
            chains = rollout.chains.detach().float()
            deltas_chain = chains[1:] - chains[:-1]
            flow_statistics = {
                "initial_noise_mean": float(chains[0].mean().item()),
                "initial_noise_std": float(chains[0].std(unbiased=False).item()),
                "path_l2": float(deltas_chain.square().sum().sqrt().item()),
                "terminal_mean": float(chains[-1].mean().item()),
                "terminal_std": float(chains[-1].std(unbiased=False).item()),
            }
        return CausalChunkSample(
            mode=mode,
            actions=actions,
            normalized_actions=executed,
            action_execution_trace=trace,
            condition_contract=condition_contract,
            video_denoise_calls=video_calls,
            latency_ms=latency,
            control=control,
            proposal_seeds=(int(action_seed),),
            flow_statistics=flow_statistics,
            action_denoise_calls=int(timesteps.numel()),
        )

    @torch.no_grad()
    def audit_current_prefix(
        self,
        *,
        image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_seed: int,
    ) -> None:
        """Execute the exact C0/C2 prefix acceptance gate on one input."""

        current, calls, _, current_contract = self._prepare_causal_condition(
            image=image,
            context=context,
            context_mask=context_mask,
            mode=CausalComputeMode.C0_CURRENT,
            video_seed=None,
        )
        if calls != 0:
            raise AssertionError("C0 unexpectedly executed video denoising.")
        if current_contract.physical_future_tokens != 0:
            raise AssertionError("C0 unexpectedly materialized future K/V.")
        full, calls, _, full_contract = self._prepare_causal_condition(
            image=image,
            context=context,
            context_mask=context_mask,
            mode=CausalComputeMode.C2_FULL,
            video_seed=video_seed,
        )
        if calls != self.num_inference_steps:
            raise AssertionError("C2 video-denoise call count is incorrect.")
        if full_contract.physical_future_tokens <= 0:
            raise AssertionError("C2 did not materialize future K/V.")
        assert_current_prefix_cache_equal(current, full)

    @torch.no_grad()
    def sample_causal_action(
        self,
        *,
        env_obs: dict[str, Any],
        mode: CausalComputeMode | str,
        action_seed: int,
        video_seed: int | None,
    ) -> CausalChunkSample:
        """Apply one deterministic current-chunk intervention.

        Per-sample CPU-seeded noise makes C0/C2 common-random pairing
        independent of branch execution order and of C2 video RNG consumption.
        """

        selected = CausalComputeMode.parse(mode)
        images, context, context_mask = self._encode_condition(env_obs)
        if images.shape[0] != 1:
            raise ValueError(
                "Causal snapshot interventions currently require batch size 1."
            )
        condition, video_calls, latency, condition_contract = (
            self._prepare_causal_condition(
                image=images,
                context=context,
                context_mask=context_mask,
                mode=selected,
                video_seed=video_seed,
            )
        )
        return self._sample_from_condition(
            env_obs=env_obs,
            mode=selected,
            control=(
                CausalControlKind.NO_READ
                if selected is CausalComputeMode.G_NO_READ
                else CausalControlKind.STANDARD
            ),
            condition=condition,
            condition_contract=condition_contract,
            action_seed=action_seed,
            video_calls=video_calls,
            latency=latency,
            collect_flow_statistics=False,
        )

    @torch.no_grad()
    def sample_causal_intervention(
        self,
        *,
        env_obs: dict[str, Any],
        spec: CausalInterventionSpecV2,
        donor_future_latents: torch.Tensor | None = None,
        prediction_context: tuple[torch.Tensor, torch.Tensor] | None = None,
        ground_truth_future_latents: torch.Tensor | None = None,
        collect_flow_statistics: bool = False,
    ) -> CausalChunkSample:
        """Apply one v2 expert/control intervention to the current chunk."""

        images, context, context_mask = self._encode_condition(env_obs)
        if images.shape[0] != 1:
            raise ValueError("Causal v2 interventions require batch size one.")
        if spec.control is CausalControlKind.GENERIC_MEDOID:
            proposals = []
            seeds = []
            for index in range(int(spec.generic_proposal_count)):
                seed = derive_generic_proposal_seed(spec.action_seed, index)
                seeds.append(seed)
                proposals.append(
                    self.sample_causal_action(
                        env_obs=env_obs,
                        mode=CausalComputeMode.C0_CURRENT,
                        action_seed=seed,
                        video_seed=None,
                    )
                )
            stacked = torch.cat(
                [sample.normalized_actions for sample in proposals],
                dim=0,
            )
            medoid_index, _ = normalized_action_medoid(stacked)
            selected = proposals[medoid_index]
            combined_latency: dict[str, float] = {}
            for sample in proposals:
                for name, value in sample.latency_ms.items():
                    combined_latency[name] = combined_latency.get(name, 0.0) + float(
                        value
                    )
            combined_latency["proposal"] = combined_latency.get("critical_path", 0.0)
            return CausalChunkSample(
                mode=CausalComputeMode.C0_CURRENT,
                actions=selected.actions,
                normalized_actions=selected.normalized_actions,
                action_execution_trace=selected.action_execution_trace,
                condition_contract=replace(
                    selected.condition_contract,
                    control=CausalControlKind.GENERIC_MEDOID,
                ),
                video_denoise_calls=0,
                latency_ms=combined_latency,
                control=CausalControlKind.GENERIC_MEDOID,
                proposal_seeds=tuple(seeds),
                medoid_index=medoid_index,
                flow_statistics=selected.flow_statistics,
                action_denoise_calls=sum(
                    sample.action_denoise_calls for sample in proposals
                ),
            )
        condition, calls, latency, contract = self._prepare_intervention_condition(
            image=images,
            context=context,
            context_mask=context_mask,
            spec=spec,
            donor_future_latents=donor_future_latents,
            prediction_context=prediction_context,
            ground_truth_future_latents=ground_truth_future_latents,
        )
        return self._sample_from_condition(
            env_obs=env_obs,
            mode=spec.mode,
            control=spec.control,
            condition=condition,
            condition_contract=contract,
            action_seed=spec.action_seed,
            video_calls=calls,
            latency=latency,
            collect_flow_statistics=collect_flow_statistics,
        )

    @torch.no_grad()
    def extract_pre_treatment_gate_features(
        self,
        *,
        env_obs: dict[str, Any],
        action_seed: int,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        remaining_budget: float,
        previous_mode: CausalComputeMode | str | None,
        steps_to_go: float,
        two_proposals: bool,
        gate_modes: tuple[CausalComputeMode | str, ...] = (
            CausalComputeMode.C0_CURRENT,
            CausalComputeMode.C2_FULL,
        ),
    ) -> CausalPreGateCaptureV1:
        """Extract the frozen Gate feature surface before any future prediction."""

        images, language, language_mask, proprio, context, context_mask = (
            self._encode_pre_gate_condition(env_obs)
        )
        if images.shape[0] != 1 or proprio.shape != (1, 8):
            raise ValueError("Gate feature extraction requires batch-one 8-D proprio.")
        if history.ndim != 3 or history.shape[:2] != (1, 4):
            raise ValueError("Gate history must have shape [1,4,H].")
        if history_mask.shape != (1, 4) or history_mask.dtype is not torch.bool:
            raise ValueError("Gate history mask must be bool [1,4].")
        condition, calls, latency, contract = self._prepare_causal_condition(
            image=images,
            context=context,
            context_mask=context_mask,
            mode=CausalComputeMode.C0_CURRENT,
            video_seed=None,
        )
        if calls != 0:
            raise AssertionError("Pre-Gate feature extraction executed prediction.")
        first = self._sample_from_condition(
            env_obs=env_obs,
            mode=CausalComputeMode.C0_CURRENT,
            control=CausalControlKind.STANDARD,
            condition=condition,
            condition_contract=contract,
            action_seed=action_seed,
            video_calls=0,
            latency=latency,
            collect_flow_statistics=True,
        )
        flow = first.flow_statistics or {}
        flow_values = torch.tensor(
            [
                flow["initial_noise_mean"],
                flow["initial_noise_std"],
                flow["path_l2"],
                flow["terminal_mean"],
                flow["terminal_std"],
            ],
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        proposal = torch.cat(
            (first.normalized_actions.detach().flatten(start_dim=1), flow_values),
            dim=-1,
        )
        proposal_latency = float(first.latency_ms["critical_path"])
        disagreement = torch.zeros(1, 1, device=self.device, dtype=self.dtype)
        proposal_count = 1
        if two_proposals:
            second = self.sample_causal_action(
                env_obs=env_obs,
                mode=CausalComputeMode.C0_CURRENT,
                action_seed=derive_generic_proposal_seed(action_seed, 1),
                video_seed=None,
            )
            disagreement = (
                (
                    first.normalized_actions.detach().float()
                    - second.normalized_actions.detach().float()
                )
                .square()
                .mean(dim=(1, 2), keepdim=False)[:, None]
                .to(dtype=self.dtype)
            )
            proposal_latency += float(second.latency_ms["critical_path"])
            proposal_count = 2
        parsed_gate_modes = tuple(CausalComputeMode.parse(mode) for mode in gate_modes)
        if any(not mode.is_routable for mode in parsed_gate_modes) or len(
            set(parsed_gate_modes)
        ) != len(parsed_gate_modes):
            raise ValueError("Gate feature modes must be unique routable experts.")
        previous_width = len(parsed_gate_modes) + 1
        if previous_mode is None:
            previous_index = 0
        else:
            parsed_previous = CausalComputeMode.parse(previous_mode)
            if parsed_previous not in parsed_gate_modes:
                raise ValueError("Previous mode is absent from the Gate expert set.")
            previous_index = parsed_gate_modes.index(parsed_previous) + 1
        previous = torch.nn.functional.one_hot(
            torch.tensor([previous_index], device=self.device),
            num_classes=previous_width,
        ).to(dtype=self.dtype)
        features = {
            "current_video_kv": pool_current_kv_layers(
                condition.video_kv_cache,
                current_token_count=condition.current_frame_video_tokens,
            ),
            "current_video_mask": torch.ones(
                1, 30, dtype=torch.bool, device=self.device
            ),
            "language": language.detach(),
            "language_mask": language_mask.detach(),
            "proprio": proprio.detach(),
            "history": history.detach().to(device=self.device, dtype=self.dtype),
            "history_mask": history_mask.detach().to(device=self.device),
            "action_proposal": proposal.detach(),
            "proposal_disagreement": disagreement.detach(),
            "remaining_budget": torch.tensor(
                [[remaining_budget]], device=self.device, dtype=self.dtype
            ),
            "previous_mode": previous,
            "steps_to_go": torch.tensor(
                [[steps_to_go]], device=self.device, dtype=self.dtype
            ),
        }
        return CausalPreGateCaptureV1(
            features=features,
            proposal_latency_ms=proposal_latency,
            proposal_count=proposal_count,
        )
