# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen native VAE/T5 encoding with separate current-frame VAE state."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ObservationEncoder(nn.Module):
    def __init__(self, vae, text_encoder, tokenizer, config):
        super().__init__()
        from wan_va.modules.utils import WanVAEStreamingWrapper

        self.vae = vae.requires_grad_(False).eval()
        self.text_encoder = text_encoder.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.config = config
        self.stream = WanVAEStreamingWrapper(self.vae)
        self.prompt_cache = {}

    def reset(self):
        self.stream.clear_cache()

    @torch.no_grad()
    def text(self, instruction):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        instruction = prompt_clean(instruction)
        if instruction not in self.prompt_cache:
            tokens = self.tokenizer(
                instruction,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            device = next(self.text_encoder.parameters()).device
            mask = tokens.attention_mask.to(device)
            values = self.text_encoder(
                tokens.input_ids.to(device), mask
            ).last_hidden_state
            values = values * mask[..., None]
            self.prompt_cache[instruction] = (values.detach().cpu(), mask.bool().cpu())
        return self.prompt_cache[instruction]

    @torch.no_grad()
    def images(self, observations, *, current_only=False):
        from wan_va.modules.utils import WanVAEStreamingWrapper

        if current_only and len(observations) != 1:
            raise ValueError("Gate encoding accepts exactly the current image pair.")
        videos = np.stack(
            [
                np.stack([obs["images"][key] for obs in observations])
                for key in self.config.camera_keys
            ]
        )
        if videos.dtype != np.uint8 or videos.shape[-1] != 3:
            raise ValueError("Camera images must be RGB uint8 HWC arrays.")
        v, t, h, w, c = videos.shape
        images = (
            torch.from_numpy(videos)
            .permute(0, 1, 4, 2, 3)
            .reshape(v * t, c, h, w)
            .float()
        )
        images = F.interpolate(
            images,
            (self.config.height, self.config.width),
            mode="bilinear",
            align_corners=False,
        )
        images = images.reshape(v, t, c, self.config.height, self.config.width).permute(
            0, 2, 1, 3, 4
        )
        parameter = next(self.vae.parameters())
        images = (images / 255 * 2 - 1).to(parameter)
        # A fresh wrapper shares weights, never the episode's temporal state.
        wrapper = WanVAEStreamingWrapper(self.vae) if current_only else self.stream
        encoded = torch.cat(
            [wrapper.encode_chunk(chunk) for chunk in images.split(4, dim=2)], dim=2
        )
        mu, _ = encoded.chunk(2, dim=1)
        mean = torch.tensor(self.vae.config.latents_mean, device=mu.device)[
            None, :, None, None, None
        ]
        std = torch.tensor(self.vae.config.latents_std, device=mu.device)[
            None, :, None, None, None
        ]
        # Match native normalize_latents: FP32 affine arithmetic, then cast back.
        normalized = ((mu.float() - mean) * (1.0 / std)).to(mu)
        return torch.cat(normalized.split(1, dim=0), dim=-1)
