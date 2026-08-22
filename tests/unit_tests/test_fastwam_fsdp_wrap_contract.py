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

"""FSDP wrap contracts for the composite FastWAM policy."""

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models.embodiment.wam_policy.adaptive_policy import (
    FastWAMAdaptivePolicy,
)
from rlinf.models.embodiment.wam_policy.critic import (
    FastWAMCurrentFrameValueCritic,
    FastWAMValueTransformerConfig,
)


class RegimeLoRALinear(nn.Linear):
    """Stand-in for the mixed frozen/trainable ActionDiT projection."""

    def __init__(self) -> None:
        super().__init__(2, 2)
        self.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.ones(1, 2))
        self.lora_B = nn.Parameter(torch.ones(2, 1))


class GateTransformer(nn.Module):
    """Stand-in for the independently wrapped Gate sidecar."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 1)


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapted = RegimeLoRALinear()


class GemmaRMSNorm(nn.Module):
    """Small stand-in whose class name matches OpenPI's no-split contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _Backbone(nn.Module):
    _no_split_modules = ["GemmaRMSNorm"]
    _no_split_names = ["action_in_proj"]

    def __init__(self) -> None:
        super().__init__()
        self.norm = GemmaRMSNorm()
        self.action_in_proj = nn.Linear(2, 2)
        self.action_in_proj._fsdp_wrap_name = "action_in_proj"
        self.other = nn.Linear(2, 2)


class _Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.value_head = nn.Linear(2, 1)


def _policy_shell() -> FastWAMAdaptivePolicy:
    policy = FastWAMAdaptivePolicy.__new__(FastWAMAdaptivePolicy)
    nn.Module.__init__(policy)
    policy.actor = _Actor()
    policy.gate = GateTransformer()
    policy.critic = _Critic()
    return policy


def test_only_nested_critic_no_split_metadata_is_consumed() -> None:
    policy = _policy_shell()
    fsdp_config = OmegaConf.create({"use_orig_params": True})
    wrap_policy = get_fsdp_wrap_policy(
        module=policy,
        config=fsdp_config,
        is_lora=False,
        model_type="fastwam_adaptive",
    )

    assert wrap_policy is not None
    assert not wrap_policy(
        module=policy.actor.adapted,
        recurse=False,
        nonwrapped_numel=0,
    )
    assert not wrap_policy(
        module=policy.gate,
        recurse=False,
        nonwrapped_numel=0,
    )
    assert wrap_policy(
        module=policy.critic.backbone.norm,
        recurse=False,
        nonwrapped_numel=0,
    )
    assert wrap_policy(
        module=policy.critic.backbone.action_in_proj,
        recurse=False,
        nonwrapped_numel=0,
    )
    assert not wrap_policy(
        module=policy.critic.backbone.other,
        recurse=False,
        nonwrapped_numel=0,
    )


def test_current_frame_critic_wraps_only_its_value_head() -> None:
    policy = _policy_shell()
    policy.critic = FastWAMCurrentFrameValueCritic(
        config=FastWAMValueTransformerConfig(
            num_mot_layers=1,
            source_num_heads=1,
            source_head_dim=2,
            layer_indices=(0,),
            hidden_dim=2,
            num_query_tokens=1,
        ),
        hidden_sizes=(4,),
    )
    fsdp_config = OmegaConf.create({"use_orig_params": True})

    wrap_policy = get_fsdp_wrap_policy(
        module=policy,
        config=fsdp_config,
        is_lora=False,
        model_type="fastwam_adaptive",
    )

    assert policy._no_split_modules == []
    assert policy._no_split_names == []
    assert wrap_policy(
        module=policy.critic.value_head,
        recurse=False,
        nonwrapped_numel=0,
    )
    assert not hasattr(policy.critic, "actor")
