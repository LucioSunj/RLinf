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

"""Differentiable native Wan blocks with explicit adapters and read-only K/V.

The native module owns all parent weights. This forward preserves its block
arithmetic but never writes to WanAttention.attn_caches. Only the current action
tokens use LoRA; historical projections always come from the frozen parent.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .cache import Cache, LayerKV


class LowRankDelta(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.a = nn.Parameter(
            torch.empty(rank, linear.in_features, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.zeros(linear.out_features, rank, dtype=torch.float32)
        )
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(F.linear(x.float(), self.a), self.b) * self.scale


def rotate(x, freqs):
    complex_x = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(complex_x * freqs).flatten(-2).to(x.dtype)


def mesh(frames, height, width, *, start, action, device):
    """Native Wan grid, with integer physical frames kept separately from RoPE."""
    f, h, w = torch.meshgrid(
        torch.arange(start, start + frames, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    physical = f.flatten().long()
    if action:
        f = f + torch.arange(1, height + 1, device=device)[None, :, None] / (height + 1)
        h, w = torch.full_like(h, -1), torch.full_like(w, -1)
    grid = torch.stack((f, h, w, torch.full_like(f, int(action)))).flatten(1)
    return grid, physical


class FunctionalVA(nn.Module):
    def __init__(self, parent: nn.Module, *, rank=16, alpha=16.0):
        super().__init__()
        self.parent = parent.requires_grad_(False).eval()
        self.adapters = nn.ModuleDict()
        for index, block in enumerate(parent.blocks):
            for attn in ("attn1", "attn2"):
                module = getattr(block, attn)
                for proj in ("to_q", "to_k", "to_v"):
                    self.adapters[f"{index}_{attn}_{proj}"] = LowRankDelta(
                        getattr(module, proj), rank, alpha
                    )
                self.adapters[f"{index}_{attn}_out"] = LowRankDelta(
                    module.to_out[0], rank, alpha
                )
            self.adapters[f"{index}_ffn_in"] = LowRankDelta(
                block.ffn.net[0].proj, rank, alpha
            )
            self.adapters[f"{index}_ffn_out"] = LowRankDelta(
                block.ffn.net[2], rank, alpha
            )
        self.adapters.to(next(parent.parameters()).device)

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def linear(self, linear, x, key, enabled):
        result = linear(x)
        if enabled:
            result = result + self.adapters[key](x).to(result.dtype)
        return result

    def attention(
        self,
        module,
        q_input,
        kv_input,
        *,
        prefix,
        enabled,
        freqs=None,
        history=None,
        frames=None,
    ):
        q = self.linear(module.to_q, q_input, prefix + "_to_q", enabled)
        k = self.linear(module.to_k, kv_input, prefix + "_to_k", enabled)
        v = self.linear(module.to_v, kv_input, prefix + "_to_v", enabled)
        q = module.norm_q(q).unflatten(-1, (module.heads, -1))
        k = module.norm_k(k).unflatten(-1, (module.heads, -1))
        v = v.unflatten(-1, (module.heads, -1))
        if freqs is not None:
            q, k = rotate(q, freqs), rotate(k, freqs)
        current = None if frames is None else LayerKV(k, v, frames)
        if history is not None:
            k = torch.cat((history.key.to(k), k), dim=1)
            v = torch.cat((history.value.to(v), v), dim=1)
        result = module.attn_op(q, k, v).flatten(2).type_as(q)
        result = self.linear(module.to_out[0], result, prefix + "_out", enabled)
        return module.to_out[1](result), current

    def forward(
        self,
        latents,
        timesteps,
        text,
        *,
        action=False,
        uncond=False,
        history: Cache = (),
        start=0,
    ):
        """Return native-shaped velocities and current-token K/V only."""
        if uncond and not action:
            raise ValueError("UNCOND adapters cannot be active on video tokens.")
        model = self.parent
        frames, height, width = latents.shape[-3:]
        x = model._input_embed(latents, input_type="action" if action else "latent")
        p_h, p_w = (1, 1) if action else tuple(model.patch_size[1:])
        grid, physical = mesh(
            frames,
            height // p_h,
            width // p_w,
            start=start,
            action=action,
            device=latents.device,
        )
        freqs = model.rope(grid[None].expand(len(latents), -1, -1))[:, :, None]
        if not torch.is_tensor(timesteps):
            timesteps = torch.full(
                (len(latents), frames), float(timesteps), device=latents.device
            )
        elif timesteps.ndim == 0:
            timesteps = timesteps.expand(len(latents), frames)
        temb, timestep_proj = model._time_embed(
            timesteps, height, width, x.dtype, action_mode=action
        )
        text_hidden = model.condition_embedder.text_embedder(text)
        current_layers = []
        for index, block in enumerate(model.blocks):
            values = block.scale_shift_table[None] + timestep_proj.float()
            shift, scale, gate, fshift, fscale, fgate = values.unbind(2)
            norm = (block.norm1(x.float()) * (1 + scale) + shift).type_as(x)
            result, current = self.attention(
                block.attn1,
                norm,
                norm,
                prefix=f"{index}_attn1",
                enabled=uncond,
                freqs=freqs,
                history=history[index] if history else None,
                frames=physical,
            )
            current_layers.append(current)
            x = (x.float() + result * gate).type_as(x)
            norm = block.norm2(x.float()).type_as(x)
            result, _ = self.attention(
                block.attn2, norm, text_hidden, prefix=f"{index}_attn2", enabled=uncond
            )
            x = x + result
            norm = (block.norm3(x.float()) * (1 + fscale) + fshift).type_as(x)
            if uncond:
                hidden = self.linear(
                    block.ffn.net[0].proj, norm, f"{index}_ffn_in", True
                )
                hidden = F.gelu(hidden, approximate="tanh")
                hidden = block.ffn.net[1](hidden)
                result = self.linear(block.ffn.net[2], hidden, f"{index}_ffn_out", True)
                for module in block.ffn.net[3:]:
                    result = module(result)
            else:
                result = block.ffn(norm)
            x = (x.float() + result.float() * fgate).type_as(x)
        shift, scale = (model.scale_shift_table[None] + temb[:, :, None]).unbind(2)
        x = (model.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        if action:
            output = model.action_proj_out(x)
            output = rearrange(
                output, "b (f h w) c -> b c f h w", f=frames, h=height, w=width
            )
        else:
            output = model.proj_out(x)
            output = rearrange(
                output,
                "b (f h w) (p q c) -> b c f (h p) (w q)",
                f=frames,
                h=height // p_h,
                w=width // p_w,
                p=p_h,
                q=p_w,
            )
        return output, tuple(current_layers)
