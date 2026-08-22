# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configurable critic contracts for the adaptive FastWAM policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.value_head import ValueHead


class CriticKind(str, Enum):
    """Built-in critic feature backends supported by adaptive FastWAM."""

    PI0_5_VALUE_AFTER_VLM = "pi0_5_value_after_vlm"
    FASTWAM_CURRENT_FRAME_VALUE = "fastwam_current_frame_value"

    @classmethod
    def parse(cls, value: CriticKind | str) -> CriticKind:
        """Normalize one configured critic kind."""

        try:
            return value if isinstance(value, cls) else cls(str(value))
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported FastWAM critic kind {value!r}; expected one of "
                f"{supported}."
            ) from exc


class CurrentFramePooling(str, Enum):
    """Token-pooling modes for the FastWAM current-frame critic."""

    MEAN_TOKEN = "mean_token"
    FIRST_TOKEN = "first_token"
    LAST_TOKEN = "last_token"

    @classmethod
    def parse(cls, value: CurrentFramePooling | str) -> CurrentFramePooling:
        """Normalize one configured current-frame pooling mode."""

        try:
            return value if isinstance(value, cls) else cls(str(value))
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported FastWAM critic pooling {value!r}; expected one of "
                f"{supported}."
            ) from exc


@dataclass(frozen=True)
class FastWAMCurrentFrameFeatureConfig:
    """Read-only current-frame Video-V feature selection."""

    input_dim: int
    layer_index: int = 14
    pooling: CurrentFramePooling | str = CurrentFramePooling.MEAN_TOKEN
    source: str = "current_frame_video_value"

    def __post_init__(self) -> None:
        if self.source != "current_frame_video_value":
            raise ValueError(
                "FastWAM current-frame critic supports only "
                "`source: current_frame_video_value`."
            )
        if (
            isinstance(self.input_dim, bool)
            or not isinstance(self.input_dim, int)
            or self.input_dim < 1
        ):
            raise ValueError(
                "FastWAM critic `input_dim` must be a positive integer."
            )
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError(
                "FastWAM critic `layer_index` must be a non-negative integer."
            )
        object.__setattr__(self, "pooling", CurrentFramePooling.parse(self.pooling))


def pool_current_frame_video_values(
    condition: Any,
    config: FastWAMCurrentFrameFeatureConfig,
) -> torch.Tensor:
    """Pool detached current-frame Video-V tokens from one cached condition.

    Args:
        condition: A ``CachedActionCondition``-compatible object.
        config: Layer, width, and pooling selection.

    Returns:
        A detached FP32 tensor with shape ``[B, input_dim]``.
    """

    video_kv_cache = condition.video_kv_cache
    if config.layer_index >= len(video_kv_cache):
        raise ValueError(
            "FastWAM critic layer index is outside the Video K/V cache: "
            f"index={config.layer_index}, layers={len(video_kv_cache)}."
        )
    layer_cache = video_kv_cache[config.layer_index]
    if "v" not in layer_cache:
        raise KeyError(
            f"FastWAM Video K/V cache layer {config.layer_index} has no `v` tensor."
        )
    values = layer_cache["v"]
    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        shape = None if not isinstance(values, torch.Tensor) else tuple(values.shape)
        raise ValueError(
            "FastWAM current-frame Video-V values must have shape [B, S, D], "
            f"got {shape}."
        )
    current_tokens = int(condition.current_frame_video_tokens)
    if not 1 <= current_tokens <= values.shape[1]:
        raise ValueError(
            "FastWAM current-frame token count is outside the Video-V sequence: "
            f"tokens={current_tokens}, sequence={values.shape[1]}."
        )
    if values.shape[2] != config.input_dim:
        raise ValueError(
            "FastWAM critic input width does not match Video-V values: "
            f"configured={config.input_dim}, observed={values.shape[2]}."
        )

    current_values = values[:, :current_tokens]
    if config.pooling is CurrentFramePooling.MEAN_TOKEN:
        pooled = current_values.mean(dim=1)
    elif config.pooling is CurrentFramePooling.FIRST_TOKEN:
        pooled = current_values[:, 0]
    else:
        pooled = current_values[:, -1]
    return pooled.detach().to(dtype=torch.float32).contiguous()


class FastWAMCurrentFrameValueCritic(nn.Module):
    """A fresh value head over detached FastWAM current-frame features."""

    kind = CriticKind.FASTWAM_CURRENT_FRAME_VALUE
    replay_feature_key = "fastwam_critic_features"

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (1024, 512, 256),
        activation: str = "relu",
        bias_last: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.value_head = ValueHead(
            input_dim=self.input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
            bias_last=bias_last,
        )
        self.value_head.register_forward_pre_hook(self._match_input_dtype)

    @staticmethod
    def _match_input_dtype(
        module: nn.Module,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Move features to the live value-head parameter placement."""

        parameter = next(module.parameters())
        return tuple(
            value.to(device=parameter.device, dtype=parameter.dtype)
            if isinstance(value, torch.Tensor)
            else value
            for value in args
        )

    def value_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Predict scalar values without propagating into FastWAM features."""

        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "FastWAM critic features must have shape [B, input_dim], got "
                f"{tuple(features.shape)} for input_dim={self.input_dim}."
            )
        return self.value_head(features.detach().to(dtype=torch.float32))[:, 0]

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the value-head parameters and assert exclusive ownership."""

        trainable = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        expected = list(self.value_head.parameters())
        if {id(parameter) for parameter in trainable} != {
            id(parameter) for parameter in expected
        }:
            raise RuntimeError(
                "FastWAM current-frame critic has trainable parameters outside "
                "the fresh value head."
            )
        return trainable


def critic_parent_checkpoint_sha256(critic_cfg: Any) -> str | None:
    """Return the external critic parent identity required by one backend."""

    def get(name: str, default: Any = None) -> Any:
        if hasattr(critic_cfg, "get"):
            return critic_cfg.get(name, default)
        return getattr(critic_cfg, name, default)

    kind = CriticKind.parse(get("kind", CriticKind.PI0_5_VALUE_AFTER_VLM))
    if kind is CriticKind.FASTWAM_CURRENT_FRAME_VALUE:
        return None
    value = str(get("backbone_checkpoint_sha256", "")).strip().lower()
    return value


__all__ = [
    "CriticKind",
    "CurrentFramePooling",
    "FastWAMCurrentFrameFeatureConfig",
    "FastWAMCurrentFrameValueCritic",
    "critic_parent_checkpoint_sha256",
    "pool_current_frame_video_values",
]
