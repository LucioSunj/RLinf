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

"""Frozen pi0.5 prefix backbone with a freshly initialized value head."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.value_head import ValueHead

VALUE_HEAD_KEY_FRAGMENT = "value_head."


def filter_pretrained_value_head(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove all pretrained RL value-head tensors from a checkpoint."""

    return {
        key: value
        for key, value in state_dict.items()
        if VALUE_HEAD_KEY_FRAGMENT not in key
    }


def _configure_value_after_vlm(config: Any) -> Any:
    """Return a config with the literal pi0.5 critic fields enabled."""

    updates = {
        "add_value_head": True,
        "value_after_vlm": True,
        "value_vlm_mode": "mean_token",
        "detach_critic_input": True,
    }
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.replace(config, **updates)
    for name, value in updates.items():
        setattr(config, name, value)
    return config


class Pi05ValueAfterVLMCritic(nn.Module):
    """Use a frozen pi0.5 PaliGemma prefix and train only a new MLP head.

    The wrapped object is the existing RLinf OpenPi policy model.  Keeping the
    wrapper around that model preserves its exact observation preprocessing and
    ``value_after_vlm`` token-selection behavior.  The action expert remains
    present in v0 but is frozen and is never called for value prediction.
    """

    kind = "pi0_5_value_after_vlm"
    replay_feature_key = "critic_prefix"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        input_dim: int = 2048,
        hidden_sizes: tuple[int, ...] = (1024, 512, 256),
        activation: str = "relu",
        bias_last: bool = True,
    ) -> None:
        """Initialize the frozen backbone and replace any existing value head."""

        super().__init__()
        if not hasattr(backbone, "config"):
            raise TypeError("The pi0.5 critic backbone must expose `config`.")
        config_name = str(getattr(backbone.config, "config_name", ""))
        if "pi05_" not in config_name:
            raise ValueError(
                "The literal value_after_vlm critic requires a pi0.5 config, "
                f"got {config_name!r}."
            )
        if not hasattr(backbone, "get_value_from_vlm"):
            raise TypeError(
                "The pi0.5 critic backbone must implement `get_value_from_vlm`."
            )

        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

        value_head = ValueHead(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
            bias_last=bias_last,
        )
        backbone.value_head = value_head
        value_head.register_forward_pre_hook(self._match_value_head_input_dtype)
        backbone.config = _configure_value_after_vlm(backbone.config)
        for parameter in value_head.parameters():
            parameter.requires_grad_(True)

        self.backbone = backbone

    @staticmethod
    def _match_value_head_input_dtype(
        module: nn.Module,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Match the live FSDP-cast head dtype after pi0.5 forces FP32 features."""

        parameter = next(module.parameters())
        return tuple(
            value.to(device=parameter.device, dtype=parameter.dtype)
            if isinstance(value, torch.Tensor)
            else value
            for value in args
        )

    @property
    def value_head(self) -> nn.Module:
        """Return the newly initialized trainable value head."""

        return self.backbone.value_head

    def train(self, mode: bool = True) -> Pi05ValueAfterVLMCritic:
        """Keep the frozen backbone in eval mode while training the value head."""

        super().train(mode)
        self.backbone.eval()
        self.value_head.train(mode)
        return self

    def value_from_prefix(self, prefix_output: torch.Tensor) -> torch.Tensor:
        """Predict values from detached pi0.5 prefix features."""

        return self.backbone.get_value_from_vlm(prefix_output.detach())

    def value_from_features(self, prefix_output: torch.Tensor) -> torch.Tensor:
        """Predict values through the common critic replay-feature interface."""

        return self.value_from_prefix(prefix_output)

    def encode_features(self, env_obs: dict[str, Any]) -> torch.Tensor:
        """Run exact pi0.5 preprocessing and return detached prefix features."""

        _values, prefix_output = self.predict_value_batch(
            env_obs,
            return_prefix=True,
        )
        return prefix_output

    def predict_value_batch(
        self,
        env_obs: dict[str, Any],
        *,
        return_prefix: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run exact pi0.5 preprocessing and prefix inference for environment data."""

        try:
            from openpi.models import model as openpi_model
        except ImportError as exc:
            raise ImportError(
                "pi0.5 critic inference requires the OpenPI dependencies."
            ) from exc

        processed_obs = self.backbone.obs_processor(env_obs)
        processed_obs = self.backbone.input_transform(
            processed_obs,
            transpose=False,
        )
        processed_obs = self.backbone.precision_processor(processed_obs)
        observation = openpi_model.Observation.from_dict(processed_obs)
        images, image_masks, language_tokens, language_masks, _state = (
            self.backbone._preprocess_observation(observation, train=False)
        )
        prefix_output, _prefix_masks, _past_key_values = (
            self.backbone._build_prefix_cache(
                images,
                image_masks,
                language_tokens,
                language_masks,
            )
        )
        values = self.value_from_prefix(prefix_output)
        if return_prefix:
            return values, prefix_output.detach()
        return values

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the value-head parameters and assert backbone freezing."""

        trainable = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        expected = list(self.value_head.parameters())
        if {id(parameter) for parameter in trainable} != {
            id(parameter) for parameter in expected
        }:
            raise RuntimeError(
                "pi0.5 critic has trainable parameters outside the fresh value head."
            )
        return trainable
