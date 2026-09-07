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

"""Immutable reads and real-time-only eviction for LingBot-VA attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor  # [B,S,H,D], including separate positive/negative CFG rows
    value: torch.Tensor
    frames: torch.Tensor  # integer physical latent frame, independent of route

    def detach(self):
        return LayerKV(self.key.detach(), self.value.detach(), self.frames.detach())

    def to(self, device):
        return LayerKV(
            self.key.to(device), self.value.to(device), self.frames.to(device)
        )

    def select(self, selected):
        return LayerKV(
            self.key[:, selected], self.value[:, selected], self.frames[selected]
        )


Cache = tuple[LayerKV, ...]


def append_cache(history: Cache, new: Cache, *, max_frames: int | None = None) -> Cache:
    """Return a new cache; readers and teacher snapshots retain their inputs."""
    if history and len(history) != len(new):
        raise ValueError("Cache layer count changed.")
    result = []
    for i, layer in enumerate(new):
        if history:
            old = history[i]
            layer = LayerKV(
                torch.cat((old.key, layer.key), dim=1),
                torch.cat((old.value, layer.value), dim=1),
                torch.cat((old.frames, layer.frames)),
            )
        if max_frames is not None and layer.frames.numel():
            layer = layer.select(layer.frames >= layer.frames.max() - max_frames + 1)
        result.append(layer.detach())
    return tuple(result)


def positive_cache(cache: Cache) -> Cache:
    return tuple(LayerKV(x.key[:1], x.value[:1], x.frames) for x in cache)
