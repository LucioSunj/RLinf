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

"""Current-frame inference for independently trained UNCOND BC sidecars."""

from typing import Any

import torch
from fastwam.adapters.regime_lora import (
    PolicyRegime,
    RegimeLoRAConfig,
    inject_action_dit_lora,
    inject_video_bc_dit_lora,
)
from fastwam.uncond_bc_checkpoint import load_uncond_bc_sidecar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy


class FastWAMUncondBCEvalPolicy(nn.Module, BasePolicy):
    """Run the current-frame action solver with all loaded BC branches active.

    Inputs are the preprocessed FastWAM image, proprioception and T5 context.
    This evaluation-only adapter owns neither a Gate nor a value network.
    """

    def __init__(self, actor: nn.Module, action_adapter, video_adapter, metadata):
        super().__init__()
        self.actor = actor
        self.action_adapter = action_adapter
        self.video_adapter = video_adapter
        self.sidecar_metadata = metadata
        self.adapter_branches = ["action"]
        if video_adapter is not None:
            self.adapter_branches.append("video")
        self.requires_grad_(False)
        self.eval()

    def forward(self, **kwargs):
        """Reject training through the standalone evaluation adapter."""
        return self.default_forward(**kwargs)

    def default_forward(self, **kwargs):
        """Use the separate BC trainer for loss computation."""
        raise NotImplementedError("UNCOND BC evaluation has no training forward.")

    @torch.no_grad()
    def predict_action_batch(self, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return one normalized action batch without future-video prediction."""
        with self.action_adapter.regime_context.use(PolicyRegime.UNCOND):
            prediction = self.actor.infer_action(**kwargs)
        return prediction["action"].unsqueeze(0), {}


def get_uncond_bc_model(cfg: DictConfig, torch_dtype: torch.dtype):
    """Build a frozen parent plus strictly loaded Action/optional Video LoRA."""
    from . import _load_strict_fastwam_parent

    if cfg.fastwam.get("_target_") != "fastwam.runtime.create_fastwam":
        raise ValueError("UNCOND BC evaluation requires the current-frame FastWAM.")
    if torch_dtype is None or cfg.fastwam.get("load_text_encoder") is not False:
        raise ValueError(
            "UNCOND BC evaluation requires explicit precision and T5 cache."
        )
    settings = cfg.uncond_bc_eval
    actor = instantiate(cfg.fastwam, model_dtype=torch_dtype, device=cfg.device)
    _load_strict_fastwam_parent(actor, str(cfg.actor_checkpoint))
    lora_config = RegimeLoRAConfig(
        rank=int(settings.rank), alpha=float(settings.alpha), dropout=0.0
    )
    action_adapter = inject_action_dit_lora(actor.action_expert, config=lora_config)
    video_adapter = None
    if settings.video_lora:
        video_adapter = inject_video_bc_dit_lora(
            actor.video_expert,
            config=lora_config,
            regime_context=action_adapter.regime_context,
        )
    metadata = load_uncond_bc_sidecar(
        str(settings.sidecar),
        adapter=action_adapter,
        video_adapter=video_adapter,
        expected_parent_checkpoint_sha256=str(cfg.actor_checkpoint_sha256),
    )
    return FastWAMUncondBCEvalPolicy(actor, action_adapter, video_adapter, metadata)
