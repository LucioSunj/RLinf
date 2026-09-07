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

"""Physical action, temporal alignment and experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch


class Route(str, Enum):
    IDM = "idm"
    UNCOND = "uncond"


@dataclass(frozen=True)
class RoutingConfig:
    height: int = 256
    width: int = 256
    camera_keys: tuple[str, ...] = ("external", "wrist")
    action_channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 28)
    frame_chunk_size: int = 4
    action_per_frame: int = 4
    execution_horizon: int = 8
    history_frames: int = 56
    action_hz: int = 20
    video_steps: int = 5
    action_steps: int = 10
    video_shift: float = 5.0
    action_shift: float = 1.0
    guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    lora_rank: int = 16
    lora_alpha: float = 16.0
    gate_layers: tuple[int, ...] = (14, 15, 16, 17, 18, 19)
    gate_hidden: int = 256
    gate_queries: int = 4
    physical_history: int = 4
    state_dim: int = 15
    epsilon: float = 0.1
    temperature: float = 1.0
    flow_noise: float = 0.5

    def __post_init__(self):
        for name in ("camera_keys", "action_channels", "gate_layers"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.camera_keys) != 2 or len(set(self.camera_keys)) != 2:
            raise ValueError("The single-arm profile requires two named cameras.")
        if self.action_channels != (0, 1, 2, 3, 4, 5, 6, 28):
            raise ValueError("Actions must be left TCP xyzw plus channel-28 gripper.")
        if self.action_per_frame != 4 or self.frame_chunk_size != 4:
            raise ValueError(
                "This profile uses four actions/frame and four frames/chunk."
            )
        if self.execution_horizon != 8:
            raise ValueError("The executed prefix is exactly eight actions.")
        if self.height % 32 or self.width % 32:
            raise ValueError("Image size must be divisible by VAE16 times patch2.")
        if self.action_steps < 2 or self.video_steps < 1:
            raise ValueError("Flow replay needs at least two action solver steps.")
        if self.state_dim != 15 or self.physical_history < 1:
            raise ValueError("State is joint7 + TCP7 + gripper1.")
        if self.history_frames < 4 or not 0 <= self.epsilon <= 1:
            raise ValueError("Invalid history window or exploration probability.")
        if self.temperature <= 0 or self.lora_rank < 1 or self.flow_noise <= 0:
            raise ValueError(
                "Temperature, adapter rank and Flow noise must be positive."
            )

    def to_dict(self):
        return asdict(self)


@dataclass
class ActionNormalizer:
    """Quantiles belong to the new training split, never to LIBERO."""

    q01: torch.Tensor
    q99: torch.Tensor
    state_q01: torch.Tensor
    state_q99: torch.Tensor

    def __post_init__(self):
        for name, size in (
            ("q01", 30),
            ("q99", 30),
            ("state_q01", 15),
            ("state_q99", 15),
        ):
            value = torch.as_tensor(getattr(self, name), dtype=torch.float32).cpu()
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain {size} finite values.")
            setattr(self, name, value)
        if (self.q99 < self.q01).any() or (self.state_q99 < self.state_q01).any():
            raise ValueError("Upper quantiles must not be below lower quantiles.")

    @classmethod
    def load(cls, path: str | Path):
        import json

        data = json.loads(Path(path).read_text())
        return cls(
            **{name: data[name] for name in ("q01", "q99", "state_q01", "state_q99")}
        )

    def to_dict(self):
        return {
            name: getattr(self, name).tolist()
            for name in ("q01", "q99", "state_q01", "state_q99")
        }

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != 8 or not torch.isfinite(actions).all():
            raise ValueError("Expected finite physical TCP8 actions.")
        channels = [0, 1, 2, 3, 4, 5, 6, 28]
        lo, hi = self.q01.to(actions)[channels], self.q99.to(actions)[channels]
        result = actions.new_zeros((*actions.shape[:-1], 30))
        result[..., channels] = 2 * (actions - lo) / (hi - lo + 1e-6) - 1
        return result

    def decode(self, actions: torch.Tensor) -> torch.Tensor:
        channels = [0, 1, 2, 3, 4, 5, 6, 28]
        lo, hi = self.q01.to(actions), self.q99.to(actions)
        return ((actions + 1) * 0.5 * (hi - lo + 1e-6) + lo)[..., channels]

    def state(self, state: torch.Tensor) -> torch.Tensor:
        lo, hi = self.state_q01.to(state), self.state_q99.to(state)
        return 2 * (state - lo) / (hi - lo + 1e-6) - 1


def action_tensor(actions: torch.Tensor, action_per_frame: int = 4) -> torch.Tensor:
    """Map [N,30] to the native [1,30,F,4,1] convention."""
    if actions.ndim != 2 or actions.shape[1] != 30 or len(actions) % action_per_frame:
        raise ValueError("Native actions require complete four-action time groups.")
    return actions.reshape(-1, action_per_frame, 30).permute(2, 0, 1)[None, ..., None]


def flatten_actions(actions: torch.Tensor) -> torch.Tensor:
    return actions[0, ..., 0].permute(1, 2, 0).reshape(-1, actions.shape[1])


def valid_action_mask(actions: torch.Tensor, first_chunk: bool) -> torch.Tensor:
    mask = torch.zeros_like(actions, dtype=torch.bool)
    mask[:, [0, 1, 2, 3, 4, 5, 6, 28]] = True
    if first_chunk:
        mask[:, :, :1] = False
    return mask


def canonical_quaternions(
    actions: np.ndarray, previous: np.ndarray | None = None
) -> np.ndarray:
    """Unit xyzw quaternions with temporal sign continuity."""
    result = np.asarray(actions, dtype=np.float64).copy()
    norms = np.linalg.norm(result[..., 3:7], axis=-1)
    if not np.isfinite(result).all() or np.any(norms < 1e-8):
        raise ValueError("Physical poses require finite nonzero quaternions.")
    result[..., 3:7] /= norms[..., None]
    flat = result.reshape(-1, result.shape[-1])
    for i, row in enumerate(flat):
        sign = (
            (row[6] if previous is None else np.dot(row[3:7], previous))
            if i == 0
            else np.dot(row[3:7], flat[i - 1, 3:7])
        )
        if sign < 0:
            row[3:7] *= -1
    return result.astype(np.float32)
