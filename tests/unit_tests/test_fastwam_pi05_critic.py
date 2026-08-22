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

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


def _load_critic_module():
    """Load the focused module without importing RLinf's optional Ray runtime."""

    repo_root = Path(__file__).resolve().parents[2]
    modules_path = repo_root / "rlinf/models/embodiment/modules/value_head.py"
    modules_spec = importlib.util.spec_from_file_location(
        "rlinf.models.embodiment.modules.value_head",
        modules_path,
    )
    if modules_spec.name not in sys.modules:
        modules = importlib.util.module_from_spec(modules_spec)
        sys.modules[modules_spec.name] = modules
        modules_spec.loader.exec_module(modules)

    critic_path = repo_root / "rlinf/models/embodiment/wam_policy/pi05_critic.py"
    critic_spec = importlib.util.spec_from_file_location(
        "fastwam_pi05_critic_under_test",
        critic_path,
    )
    critic = importlib.util.module_from_spec(critic_spec)
    critic_spec.loader.exec_module(critic)
    return critic


_critic = _load_critic_module()
Pi05ValueAfterVLMCritic = _critic.Pi05ValueAfterVLMCritic
filter_pretrained_value_head = _critic.filter_pretrained_value_head


class _DummyPi05(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            config_name="pi05_libero",
            num_images_in_input=1,
            add_value_head=True,
            value_after_vlm=True,
            value_vlm_mode="mean_token",
            detach_critic_input=False,
        )
        self.encoder = nn.Linear(4, 4)
        self.value_head = nn.Linear(4, 1)

    def get_value_from_vlm(self, prefix_output):
        pooled = prefix_output.mean(dim=1)
        return self.value_head(pooled)[:, 0]


@dataclass(frozen=True)
class _FrozenPi05Config:
    config_name: str = "pi05_libero"
    num_images_in_input: int = 1
    add_value_head: bool = False
    value_after_vlm: bool = False
    value_vlm_mode: str = "last_token"
    detach_critic_input: bool = False


class _FrozenConfigDummyPi05(_DummyPi05):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FrozenPi05Config()


def test_filter_pretrained_value_head_removes_nested_keys():
    state = {
        "encoder.weight": torch.ones(1),
        "value_head.mlp.0.weight": torch.ones(1),
        "module.value_head.mlp.2.bias": torch.ones(1),
    }
    assert set(filter_pretrained_value_head(state)) == {"encoder.weight"}


def test_critic_replaces_value_head_and_freezes_backbone():
    torch.manual_seed(3)
    backbone = _DummyPi05()
    mutable_config = backbone.config
    old_head = backbone.value_head
    critic = Pi05ValueAfterVLMCritic(
        backbone,
        input_dim=4,
        hidden_sizes=(3, 2),
    )

    assert critic.value_head is not old_head
    assert all(
        not parameter.requires_grad for parameter in backbone.encoder.parameters()
    )
    assert all(parameter.requires_grad for parameter in critic.value_head.parameters())
    assert critic.trainable_parameters() == list(critic.value_head.parameters())
    assert backbone.config is mutable_config
    assert backbone.config.detach_critic_input is True


def test_critic_replaces_frozen_config_and_backpropagates():
    backbone = _FrozenConfigDummyPi05()
    original_config = backbone.config
    critic = Pi05ValueAfterVLMCritic(
        backbone,
        input_dim=4,
        hidden_sizes=(3, 2),
    )

    assert backbone.config is not original_config
    assert original_config == _FrozenPi05Config()
    assert backbone.config.add_value_head is True
    assert backbone.config.value_after_vlm is True
    assert backbone.config.value_vlm_mode == "mean_token"
    assert backbone.config.detach_critic_input is True

    prefix = torch.randn(2, 5, 4, requires_grad=True)
    values = critic.value_from_prefix(prefix)
    assert values.shape == (2,)
    values.sum().backward()

    assert prefix.grad is None
    assert all(parameter.grad is None for parameter in backbone.encoder.parameters())
    assert all(
        parameter.grad is not None for parameter in critic.value_head.parameters()
    )


def test_critic_gradient_stops_at_prefix_and_updates_only_value_head():
    backbone = _DummyPi05()
    critic = Pi05ValueAfterVLMCritic(
        backbone,
        input_dim=4,
        hidden_sizes=(3, 2),
    )
    prefix = torch.randn(2, 5, 4, requires_grad=True)
    critic.value_from_prefix(prefix).sum().backward()

    assert prefix.grad is None
    assert all(parameter.grad is None for parameter in backbone.encoder.parameters())
    assert all(
        parameter.grad is not None for parameter in critic.value_head.parameters()
    )


def test_value_head_hook_matches_live_parameter_dtype():
    critic = Pi05ValueAfterVLMCritic(
        _DummyPi05(),
        input_dim=4,
        hidden_sizes=(3, 2),
    )
    critic.value_head.to(dtype=torch.float64)

    values = critic.value_from_prefix(torch.randn(2, 5, 4, dtype=torch.float32))

    assert values.dtype == torch.float64


def test_non_pi05_backbone_is_rejected():
    backbone = _DummyPi05()
    backbone.config.config_name = "pi0_libero"
    try:
        Pi05ValueAfterVLMCritic(backbone, input_dim=4)
    except ValueError as exc:
        assert "pi0.5" in str(exc)
    else:
        raise AssertionError("Expected a ValueError")
