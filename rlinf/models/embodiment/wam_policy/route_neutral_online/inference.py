# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Resident merged UNCOND and compiled kernels for batch-one evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from fastwam.adapters import PolicyRegime, RegimeLoRALinear
from fastwam.models.wan22.adaptive_sampler import VelocityOutput
from fastwam.models.wan22.batch_linear import BatchInvariantLinear
from fastwam.models.wan22.mot import _GATE_CURRENT_FRAME_PROVENANCE_KEY
from fastwam.models.wan22.wan_video_dit import (
    flash_attention,
    modulate,
    sinusoidal_embedding_1d,
)
from torch import nn


@dataclass(frozen=True)
class InferenceAccelerationConfig:
    """Implementation choices for a fixed-checkpoint inference session.

    ``video_backend`` overrides compilation of Video and observation-condition
    kernels, including the parent Gate prefill and text/proprio preparation.
    Action and Gate projections retain ``backend``. ``compile_vae=False`` keeps
    the native image encoder even when the other tensor kernels are compiled.
    """

    merge_lora: bool = True
    compile: bool = True
    mode: str = "default"
    backend: str = "inductor"
    video_backend: str | None = None
    compile_vae: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(f"Unknown inference compilation mode {self.mode!r}.")
        if self.compile and not self.merge_lora:
            raise ValueError("Compiled route-neutral inference requires merged LoRA.")


@torch.no_grad()
def _inference_view(
    module: nn.Module,
    *,
    merge_lora: bool = False,
    memo: dict[int, nn.Module] | None = None,
) -> nn.Module:
    """Copy module structure, sharing read-only tensors except merged weights.

    The training model retains its modules, Parameters, flags and state keys.
    Removing training-only context-variable linears also makes the B1 kernels
    capturable by Dynamo. No random initialization or whole-model tensor copy.
    """

    if memo is None:
        memo = {}
    if id(module) in memo:
        return memo[id(module)]
    if isinstance(module, (RegimeLoRALinear, BatchInvariantLinear)):
        result = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
        )
        result.weight = module.weight
        result.bias = module.bias
        if merge_lora and isinstance(module, RegimeLoRALinear):
            # Match the existing offline merge formula. CPU FP32 matmul avoids
            # inheriting the inference process's TF32 setting for B @ A. Only
            # one projection's temporary weight is materialized at a time.
            delta = module.lora_B.detach().to(device="cpu", dtype=torch.float32) @ (
                module.lora_A.detach().to(device="cpu", dtype=torch.float32)
            )
            merged = module.weight.detach().to(device="cpu", dtype=torch.float32)
            merged = merged + delta * float(module.scaling)
            if not bool(torch.isfinite(merged).all()):
                raise FloatingPointError("Non-finite merged UNCOND projection.")
            result.weight = nn.Parameter(
                merged.to(device=module.weight.device, dtype=module.weight.dtype),
                requires_grad=False,
            )
    else:
        result = copy.copy(module)
        result._parameters = module._parameters.copy()
        result._buffers = module._buffers.copy()
        result._modules = {}
        memo[id(module)] = result
        result._modules = {
            name: (
                None
                if child is None
                else _inference_view(child, merge_lora=merge_lora, memo=memo)
            )
            for name, child in module._modules.items()
        }
    result.training = False
    memo[id(module)] = result
    return result


class RouteNeutralInference:
    """An evaluation runtime view; never registered in the training policy."""

    def __init__(self, runtime, gate: nn.Module, config: InferenceAccelerationConfig):
        if not config.merge_lora:
            raise ValueError("An inference view requires merged route experts.")
        self.config = config
        self.runtime = copy.copy(runtime)
        self.runtime.actor = _inference_view(runtime.actor)
        self.runtime._evaluation_text_context = None
        self.gate = gate
        self.idm_action_expert = self.runtime.actor.action_expert
        self.uncond_action_expert = _inference_view(
            runtime.actor.action_expert, merge_lora=True
        )
        self.uncond_runtime = None
        self.uncond_video_expert = None
        adapted_experts = [runtime.actor.action_expert]
        if runtime.video_lora_adapter is not None:
            # Gate and IDM retain the parent's Video weights. Only UNCOND
            # current-frame prefill sees this separately materialized expert.
            self.uncond_video_expert = _inference_view(
                runtime.actor.video_expert, merge_lora=True
            )
            self.uncond_runtime = copy.copy(runtime)
            self.uncond_runtime.actor = _inference_view(
                runtime.actor,
                memo={
                    id(runtime.actor.action_expert): self.uncond_action_expert,
                    id(runtime.actor.video_expert): self.uncond_video_expert,
                },
            )
            adapted_experts.append(runtime.actor.video_expert)
        self._parent_prepare_action_condition = self.runtime._prepare_action_condition
        self.runtime._prepare_action_condition = self._prepare_action_condition
        adapted = [
            module
            for expert in adapted_experts
            for module in expert.modules()
            if isinstance(module, RegimeLoRALinear)
        ]
        self.merged_projection_count = len(adapted)
        self.additional_weight_bytes = sum(
            layer.weight.numel() * layer.weight.element_size() for layer in adapted
        )
        actor = self.runtime.actor
        # These tensors are plain attributes, so Module.to() never moves them.
        # Keep their immutable values resident instead of transferring them on
        # each denoising step (and introducing CPU inputs into CUDA graphs).
        self.idm_action_expert.freqs = self.idm_action_expert.freqs.to(runtime.device)
        self.uncond_action_expert.freqs = self.idm_action_expert.freqs
        actor.video_expert.freqs = tuple(
            value.to(runtime.device) for value in actor.video_expert.freqs
        )
        if self.uncond_video_expert is not None:
            self.uncond_video_expert.freqs = actor.video_expert.freqs
        # Reuse native patchification, per-token timestep/fuse handling, masks,
        # and RoPE construction with an already projected context. This view
        # shares parent tensors and is never used for Gate or UNCOND prefill.
        self._video_preprocessor = _inference_view(runtime.actor.video_expert)
        self._video_preprocessor.text_embedding = nn.Identity()
        self._video_preprocessor.freqs = actor.video_expert.freqs
        if isinstance(actor.vae, nn.Module):
            actor.vae.scale = [value.to(runtime.device) for value in actor.vae.scale]
        # Validate the two fixed experts once. The generic resolver enumerates
        # Parameters, which Dynamo 2.7 cannot capture with next(..., default).
        actor.mot._resolve_action_expert(self.idm_action_expert)
        actor.mot._resolve_action_expert(self.uncond_action_expert)
        actor.mot._resolve_action_expert = self._resolve_action_expert
        action_experts = (
            (PolicyRegime.IDM, self.idm_action_expert),
            (PolicyRegime.UNCOND, self.uncond_action_expert),
        )
        self._action_steps = {
            regime: self._compile(partial(self._action_step, expert), fullgraph=True)
            for regime, expert in action_experts
        }
        self._action_contexts = {
            regime: self._compile(
                partial(self._prepare_action_context, expert), fullgraph=True
            )
            for regime, expert in action_experts
        }
        self.runtime._velocity = self.velocity
        self.runtime._prefill_parent_gate_kv = self._compile(
            self.runtime._prefill_parent_gate_kv,
            fullgraph=True,
            backend=config.video_backend,
        )
        self._install_video_prefill(actor)
        if self.uncond_runtime is not None:
            self._install_video_prefill(self.uncond_runtime.actor)
        actor._video_denoise_step_compiled = self._compile(
            actor._video_denoise_step_compiled,
            fullgraph=True,
            backend=config.video_backend,
        )
        self._video_context = self._compile(
            self._prepare_video_context, fullgraph=True, backend=config.video_backend
        )
        self._video_step_compiled = self._compile(
            self._video_step, fullgraph=True, backend=config.video_backend
        )
        # The VAE clears and updates its temporal convolution caches. Allow
        # graph boundaries here, preserving the native encoder/cache lifecycle.
        if config.compile_vae:
            actor._encode_input_image_latents_tensor = self._compile(
                actor._encode_input_image_latents_tensor, fullgraph=False
            )
        actor._append_proprio_to_context = self._compile(
            actor._append_proprio_to_context,
            fullgraph=True,
            backend=config.video_backend,
        )
        self._gate_inputs = self._compile(self._project_gate_inputs, fullgraph=True)
        self._gate_output = self._compile(self._fuse_gate, fullgraph=True)

    def _prepare_action_condition(self, *, regime, first_frame_latents=None, **kwargs):
        """Select merged Video only after the canonical parent Gate decision."""

        if regime is PolicyRegime.IDM:
            # The native runtime still owns noise, scheduler updates, first-frame
            # replacement, and final K/V prefill. Only its ten Video forwards
            # share this call's immutable text/proprio cross-attention tensors.
            projected, keys, values = self._video_context(kwargs["context"])
            actor = self.runtime.actor
            original_step = actor._video_denoise_step_compiled
            actor._video_denoise_step_compiled = partial(
                self._video_step_compiled,
                projected_context=projected,
                context_keys=keys,
                context_values=values,
            )
            try:
                return self._parent_prepare_action_condition(
                    regime=regime, first_frame_latents=first_frame_latents, **kwargs
                )
            finally:
                actor._video_denoise_step_compiled = original_step
        if self.uncond_runtime is None:
            return self._parent_prepare_action_condition(
                regime=regime, first_frame_latents=first_frame_latents, **kwargs
            )
        if first_frame_latents is None:
            first_frame_latents = self.runtime.actor._encode_input_image_latents_tensor(
                kwargs["image"], tiled=self.runtime.tiled_vae
            )
        return self.uncond_runtime._prefill_video_condition(
            video_latents=first_frame_latents,
            context=kwargs["context"],
            context_mask=kwargs["context_mask"],
            fuse_flag=bool(
                getattr(
                    self.uncond_video_expert, "fuse_vae_embedding_in_latents", False
                )
            ),
        ), None

    def _prepare_video_context(self, context):
        """Project parent Video text/proprio K/V once for one IDM chunk."""

        expert = self.runtime.actor.video_expert
        projected = expert.text_embedding(context)
        keys, values = [], []
        for block in expert.blocks:
            keys.append(block.cross_attn.norm_k(block.cross_attn.k(projected)))
            values.append(block.cross_attn.v(projected))
        return projected, keys, values

    def _video_step(
        self,
        latents_video,
        timestep_video,
        context,
        context_mask,
        fuse_flag,
        *,
        projected_context,
        context_keys,
        context_values,
    ):
        del context
        expert = self.runtime.actor.video_expert
        pre = self._video_preprocessor.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=projected_context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        tokens = pre["tokens"]
        self_mask = (
            expert.build_video_to_video_mask(
                video_seq_len=tokens.shape[1],
                video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]),
                device=tokens.device,
            )
            if expert.video_attention_mask_mode != "bidirectional"
            else None
        )
        cross_mask = pre["context_mask"]
        if cross_mask is not None and cross_mask.dim() == 3:
            cross_mask = cross_mask.unsqueeze(1)
        for index, block in enumerate(expert.blocks):
            (
                query,
                key,
                value,
                residual,
                gate_attention,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                _,
            ) = self.runtime.actor.mot._build_expert_attention_io(
                expert=expert,
                block=block,
                x=tokens,
                freqs=pre["freqs"],
                t_mod=pre["t_mod"],
            )
            mixed = flash_attention(
                q=query, k=key, v=value, num_heads=block.num_heads, ctx_mask=self_mask
            )
            tokens = block.gate(residual, gate_attention, block.self_attn.o(mixed))
            cross = block.cross_attn
            attended = flash_attention(
                q=cross.norm_q(cross.q(block.norm3(tokens))),
                k=context_keys[index],
                v=context_values[index],
                num_heads=cross.num_heads,
                ctx_mask=cross_mask,
            )
            tokens = tokens + cross.o(attended)
            mlp_input = modulate(block.norm2(tokens), shift_mlp, scale_mlp)
            tokens = block.gate(tokens, gate_mlp, block.ffn(mlp_input))
        return expert.post_dit(tokens, pre)

    def _install_video_prefill(self, actor) -> None:
        prefill_inner = self._compile(
            partial(self._prefill_step, actor.mot),
            fullgraph=True,
            backend=self.config.video_backend,
        )
        actor.mot.prefill_video_cache = partial(
            self.prefill_video_cache, actor.mot, prefill_inner
        )
        actor.video_expert.pre_dit = self._compile(
            actor.video_expert.pre_dit,
            fullgraph=True,
            backend=self.config.video_backend,
        )

    def _resolve_action_expert(self, action_expert, *, action_tokens=None):
        return self.idm_action_expert if action_expert is None else action_expert

    def begin_chunk(self) -> None:
        backends = {self.config.backend, self.config.video_backend}
        if self.config.compile and (
            "cudagraphs" in backends
            or (
                "inductor" in backends
                and self.config.mode in {"reduce-overhead", "max-autotune"}
            )
        ):
            torch.compiler.cudagraph_mark_step_begin()

    def _prefill_step(self, mot, tokens, freqs, modulation, context, mask):
        _, keys, values = mot._prefill_video_cache_inner(
            tokens, freqs, modulation, context, mask
        )
        # The final post-attention video tokens are unused. Returning only K/V
        # lets Inductor eliminate that last layer's unused post-block work.
        return keys, values

    def _compile(self, function, *, fullgraph: bool, backend: str | None = None):
        if not self.config.compile:
            return function
        if backend is None:
            backend = self.config.backend
        kwargs: dict[str, Any] = {
            "backend": backend,
            "fullgraph": fullgraph,
            "dynamic": False,
        }
        if backend == "inductor":
            kwargs["mode"] = self.config.mode
        return torch.compile(function, **kwargs)

    def prefill_video_cache(
        self,
        mot,
        prefill_inner,
        video_tokens,
        video_freqs,
        video_t_mod,
        video_context_payload,
        video_attention_mask,
        gate_current_frame_video_tokens=None,
    ):
        """Keep causal provenance checks outside the compiled tensor loop."""

        if gate_current_frame_video_tokens is not None:
            mot._validate_current_frame_video_mask(
                attention_mask=video_attention_mask,
                current_frame_video_tokens=int(gate_current_frame_video_tokens),
                video_seq_len=int(video_tokens.shape[1]),
            )
        keys, values = prefill_inner(
            video_tokens,
            video_freqs,
            video_t_mod,
            video_context_payload,
            video_attention_mask,
        )
        return [
            {
                "k": key,
                "v": value,
                **(
                    {}
                    if gate_current_frame_video_tokens is None
                    else {
                        _GATE_CURRENT_FRAME_PROVENANCE_KEY: int(
                            gate_current_frame_video_tokens
                        )
                    }
                ),
            }
            for key, value in zip(keys, values, strict=True)
        ]

    def _prepare_action_context(self, expert, context):
        """Project this chunk's context once with the selected fixed expert."""

        projected = expert.text_embedding(context)
        keys, values = [], []
        for block in expert.blocks:
            keys.append(block.cross_attn.norm_k(block.cross_attn.k(projected)))
            values.append(block.cross_attn.v(projected))
        return keys, values

    def _action_step(
        self,
        expert,
        actions,
        timestep,
        context_keys,
        context_values,
        mask,
        keys,
        values,
        rows,
    ):
        # Keep the native ActionDiT/MoT operations and masks. Only context
        # projection moves out of the denoising loop, as in EasyWAM's inference
        # cross-attention cache; action/time-dependent tensors remain fresh.
        seq_len = actions.shape[1]
        time_embedding = expert.time_embedding(
            sinusoidal_embedding_1d(expert.freq_dim, timestep)
        )
        modulation = expert.time_projection(time_embedding).unflatten(
            1, (6, expert.hidden_dim)
        )
        tokens = expert.action_encoder(actions)
        freqs = expert.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)
        context_mask = mask.unsqueeze(1).expand(-1, seq_len, -1).unsqueeze(1)
        mot = self.runtime.actor.mot
        for index, block in enumerate(expert.blocks):
            (
                query,
                key,
                value,
                residual,
                gate_attention,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                _,
            ) = mot._build_expert_attention_io(
                expert=expert, block=block, x=tokens, freqs=freqs, t_mod=modulation
            )
            mixed = mot._mixed_attention(
                q_cat=query,
                k_cat=torch.cat([keys[index], key], dim=1),
                v_cat=torch.cat([values[index], value], dim=1),
                attention_mask=rows,
            )
            tokens = block.gate(residual, gate_attention, block.self_attn.o(mixed))
            cross = block.cross_attn
            cross_query = cross.norm_q(cross.q(block.norm3(tokens)))
            attended = flash_attention(
                q=cross_query,
                k=context_keys[index],
                v=context_values[index],
                num_heads=cross.num_heads,
                ctx_mask=context_mask,
            )
            tokens = tokens + cross.o(attended)
            mlp_input = modulate(block.norm2(tokens), shift_mlp, scale_mlp)
            tokens = block.gate(tokens, gate_mlp, block.ffn(mlp_input))
        return expert.post_dit(tokens, {})

    def velocity(self, condition, *, regime, capture_gate_kv, actor_version):
        """Bind observation tensors without compiling route/version metadata."""

        if capture_gate_kv:
            raise ValueError("Pure inference does not collect action Gate taps.")
        keys = [layer["k"] for layer in condition.video_kv_cache]
        values = [layer["v"] for layer in condition.video_kv_cache]
        rows = condition.attention_mask[condition.video_seq_len :]
        step = self._action_steps[regime]
        context_keys, context_values = self._action_contexts[regime](condition.context)

        def call(actions, timestep):
            return VelocityOutput(
                velocity=step(
                    actions,
                    timestep,
                    context_keys,
                    context_values,
                    condition.context_mask,
                    keys,
                    values,
                    rows,
                ),
                gate_tap=None,
            )

        return call

    def _project_gate_inputs(self, visual, language, mask, state, history):
        gate = self.gate
        visual = gate.visual(visual)
        mask = mask.unsqueeze(-1)
        language = gate.language_norm(language)
        language = (language * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        language = gate.language_projection(language)
        state = gate.state_projection(gate.state_norm(state))
        history = gate.history_projection(history)
        return visual, language, state, history

    def _fuse_gate(self, visual, language, state, history):
        return self.gate.fusion(torch.cat([visual, language, state, history], dim=-1))[
            :, 0
        ]

    def gate_logits(self, features):
        parameter = next(self.gate.parameters())
        features = features.detached().to(
            device=parameter.device, dtype=parameter.dtype
        )
        visual, language, state, history_input = self._gate_inputs(
            features.visual,
            features.language,
            features.language_mask,
            features.state,
            features.physical_history,
        )
        # PyTorch 2.7 deliberately graph-breaks on nn.GRU. Keep the established
        # native recurrent kernel; compile its projections and fusion separately.
        _, hidden = self.gate.history_encoder(history_input)
        return self._gate_output(visual, language, state, hidden[-1])


__all__ = ["InferenceAccelerationConfig", "RouteNeutralInference"]
