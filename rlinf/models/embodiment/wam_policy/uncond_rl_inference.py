# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only merged Action compilation with native Video LoRA evaluation."""

from __future__ import annotations

import copy

import torch
from fastwam.adapters import RegimeLoRALinear
from fastwam.models.wan22.adaptive_sampler import VelocityOutput
from torch import nn

from .route_neutral_online.inference import _inference_view


class UncondRLCompiledInference:
    """Compile B1 Action denoising without changing the checkpointed policy."""

    def __init__(self, runtime, *, mode: str) -> None:
        actor = runtime.actor
        for branch in (actor.action_expert, actor.video_expert):
            if not any(
                isinstance(layer, RegimeLoRALinear) for layer in branch.modules()
            ):
                raise ValueError(
                    "Compiled UNCOND evaluation requires both LoRA branches."
                )

        self.runtime = copy.copy(runtime)
        self.runtime.actor = _inference_view(actor)
        self.runtime._evaluation_text_context = None
        merged = self.runtime.actor
        # Action denoising needs static linears for Dynamo, whereas Video
        # prefill stays eager under the existing UNCOND LoRA context. Sharing
        # that Video expert avoids a second 5B-parameter projection inventory.
        merged.action_expert = _inference_view(actor.action_expert, merge_lora=True)
        merged.video_expert = actor.video_expert
        merged.mot.mixtures["action"] = merged.action_expert
        merged.mot.mixtures["video"] = merged.video_expert
        merged.action_expert.freqs = merged.action_expert.freqs.to(runtime.device)
        merged.video_expert.freqs = tuple(
            frequency.to(runtime.device) for frequency in merged.video_expert.freqs
        )
        if isinstance(merged.vae, nn.Module):
            merged.vae.scale = [scale.to(runtime.device) for scale in merged.vae.scale]
        merged.mot._resolve_action_expert(merged.action_expert)
        merged.mot._resolve_action_expert = self._resolve_action_expert
        self._compiled_action_step = torch.compile(
            self._action_step,
            backend="inductor",
            mode=mode,
            fullgraph=True,
            dynamic=False,
        )
        self.runtime._uncond_velocity = self.velocity

    def _resolve_action_expert(self, action_expert, *, action_tokens=None):
        return (
            self.runtime.actor.action_expert if action_expert is None else action_expert
        )

    def _action_step(self, actions, timestep, context, mask, keys, values, rows):
        actor = self.runtime.actor
        expert = actor.action_expert
        pre = expert.pre_dit(
            action_tokens=actions,
            timestep=timestep,
            context=context,
            context_mask=mask,
        )
        tokens = actor.mot._forward_action_with_video_cache_inner(
            action_tokens=pre["tokens"],
            action_freqs=pre["freqs"],
            action_t_mod=pre["t_mod"],
            action_context_payload={
                "context": pre["context"],
                "mask": pre["context_mask"],
            },
            video_cache_k=keys,
            video_cache_v=values,
            action_attention_mask=rows,
            action_expert=expert,
        )
        return expert.post_dit(tokens, pre)

    def velocity(self, condition, actor_version):
        """Bind current-frame K/V to the compiled Action tensor step."""

        del actor_version
        keys = [layer["k"] for layer in condition.video_kv_cache]
        values = [layer["v"] for layer in condition.video_kv_cache]
        rows = condition.attention_mask[condition.video_seq_len :]

        def call(actions, timestep):
            return VelocityOutput(
                velocity=self._compiled_action_step(
                    actions,
                    timestep,
                    condition.context,
                    condition.context_mask,
                    keys,
                    values,
                    rows,
                ),
                gate_tap=None,
            )

        return call


__all__ = ["UncondRLCompiledInference"]
