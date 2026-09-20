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

"""CPU checks for the standalone dual-BC inference branch."""

import copy

import torch
from fastwam.adapters.regime_lora import (
    RegimeLoRAConfig,
    inject_action_dit_lora,
    inject_video_bc_dit_lora,
)
from fastwam.models.wan22.wan_video_dit import DiTBlock
from fastwam.uncond_bc_checkpoint import save_uncond_bc_sidecar
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.wam_policy import get_model, uncond_bc_policy


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim=8, attn_head_dim=4, num_heads=2, ffn_dim=16)
                for _ in range(2)
            ]
        )


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_expert = _Expert()
        self.action_expert = _Expert()
        self.mot = nn.ModuleDict(
            {"video": self.video_expert, "action": self.action_expert}
        )
        self.proprio_encoder = None

    def load_checkpoint(self, _path):
        return {"mot": self.mot.state_dict()}

    def infer_action(self, *, input_image):
        video = self.video_expert.blocks[0].self_attn.k(input_image)
        return {"action": self.action_expert.blocks[0].self_attn.k(video)}

    def infer_joint(self, **kwargs):
        raise AssertionError("BC evaluation must not predict future video")


def test_registered_builder_loads_and_activates_both_bc_branches(tmp_path, monkeypatch):
    torch.manual_seed(42)
    actor = _Actor()
    pristine = copy.deepcopy(actor)
    config = RegimeLoRAConfig(rank=128, alpha=128)
    action = inject_action_dit_lora(actor.action_expert, config=config)
    video = inject_video_bc_dit_lora(
        actor.video_expert, config=config, regime_context=action.regime_context
    )
    for adapter in (action, video):
        for parameter in adapter.lora_parameters():
            parameter.data.fill_(0.025)
    path = tmp_path / "dual.pt"
    save_uncond_bc_sidecar(
        path,
        adapter=action,
        video_adapter=video,
        parent_checkpoint_sha256="a" * 64,
        extra_metadata={"bc_step": 1942},
    )
    monkeypatch.setattr(
        uncond_bc_policy, "instantiate", lambda *args, **kwargs: pristine
    )
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "actor_checkpoint": "parent.pt",
            "actor_checkpoint_sha256": "a" * 64,
            "fastwam": {
                "_target_": "fastwam.runtime.create_fastwam",
                "load_text_encoder": False,
            },
            "uncond_bc_eval": {
                "sidecar": str(path),
                "rank": 128,
                "alpha": 128,
                "video_lora": True,
            },
        }
    )
    policy = get_model(cfg, torch.float32)
    assert policy.adapter_branches == ["action", "video"]
    assert policy.video_adapter.regime_context is policy.action_adapter.regime_context
    assert not any(parameter.requires_grad for parameter in policy.parameters())
    for source, restored in (
        (action, policy.action_adapter),
        (video, policy.video_adapter),
    ):
        for key, tensor in source.lora_state_dict().items():
            assert torch.equal(tensor, restored.lora_state_dict()[key])
    pixels = torch.ones(3, 8)
    prediction, _ = policy.predict_action_batch(input_image=pixels)
    base = policy.actor.infer_action(input_image=pixels)["action"]
    assert not torch.allclose(prediction[0], base)
    with action.regime_context.use("uncond"):
        assert torch.equal(
            prediction[0], actor.infer_action(input_image=pixels)["action"]
        )
