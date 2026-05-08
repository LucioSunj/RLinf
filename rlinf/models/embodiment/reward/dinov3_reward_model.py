# Copyright 2025 The RLinf Authors.
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

"""DINOv3-based similarity reward model for embodied RL.

This reward model compares current observation images against a pre-loaded
expert reference image using DINOv3 [CLS] token embeddings.

Reward formula: R = exp(-||f_cur - f_exp||^2 / temperature)
"""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from rlinf.models.embodiment.reward.base_image_reward_model import BaseImageRewardModel


class DINoV3RewardModel(BaseImageRewardModel):
    """DINOv3-based reward model using image similarity to an expert reference.

    Computes reward as R = exp(-||f_cur - f_exp||^2 / temperature),
    where f_cur and f_exp are L2-normalized DINOv3 [CLS] embeddings.

    The expert reference image is loaded once at initialization from
    ``cfg.expert_ref_image_path``.

    Args:
        cfg: Configuration containing:
            - model_name: HuggingFace DINOv3 checkpoint (default: vitb16).
            - expert_ref_image_path: Path to the expert reference image (required).
            - temperature: RBF kernel bandwidth for similarity scaling.
            - image_size: Expected [C, H, W] (used by base class, DINOv3 processor
              handles its own resize).
            - normalize: Whether to apply ImageNet normalization in base class
              (should be ``False`` since DINOv3 processor normalizes internally).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.model_name = cfg.get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.temperature = cfg.get("temperature", 1.0)
        self.expert_ref_image_path = cfg.get("expert_ref_image_path", None)

        # Load DINOv3 processor and backbone
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.backbone = AutoModel.from_pretrained(self.model_name)

        # Freeze backbone — we only use pretrained features
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Precompute expert reference embedding at init time
        if self.expert_ref_image_path is not None:
            ref_img = Image.open(self.expert_ref_image_path).convert("RGB")
            ref_emb = self._get_embedding([ref_img])  # (1, D)
            self.register_buffer("expert_embedding", ref_emb)
        else:
            self.register_buffer("expert_embedding", torch.empty(0))

    def _get_embedding(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract L2-normalized DINOv3 [CLS] embeddings from PIL images."""
        inputs = self.processor(images=images, return_tensors="pt")
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.backbone(**inputs)
            emb = outputs.last_hidden_state[:, 0]  # CLS token
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def _tensor_to_pil(self, images: torch.Tensor) -> list[Image.Image]:
        """Convert observation tensor to list of PIL Images for DINOv3 processor.

        Handles both NHWC (uint8) and NCHW (float) formats.
        """
        # NHWC uint8 from env worker
        if images.dim() == 4 and images.shape[-1] in [1, 3]:
            images = images.cpu().numpy()
        # NCHW float from worker after dtype conversion
        elif images.dim() == 4 and images.shape[1] in [1, 3]:
            images = images.permute(0, 2, 3, 1).cpu().numpy()
        else:
            raise ValueError(f"Unexpected image tensor shape: {images.shape}")

        # Normalize to [0, 1] then uint8
        if images.dtype != np.uint8:
            if images.max() > 1.0:
                images = images / 255.0
            images = np.clip(images * 255, 0, 255).astype(np.uint8)

        return [Image.fromarray(img) for img in images]

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward pass for inference (called by EmbodiedRewardWorker).

        Args:
            input_data: Image tensor from env worker, typically ``(B, H, W, C)``
                uint8 or ``(B, C, H, W)`` float.

        Returns:
            Dictionary with keys ``loss``, ``accuracy``, ``logits``,
            ``probabilities``.  ``probabilities`` holds the continuous
            similarity reward in ``[0, 1]``.
        """
        if self.expert_embedding.numel() == 0:
            raise RuntimeError(
                "No expert reference image loaded. "
                "Set 'expert_ref_image_path' in reward model config."
            )

        # Convert to PIL — DINOv3 processor expects raw uint8 images
        pil_images = self._tensor_to_pil(input_data)

        # Extract embeddings
        embeddings = self._get_embedding(pil_images)  # (B, D)

        # Similarity reward
        diff = embeddings - self.expert_embedding  # broadcasting
        l2_dist_sq = (diff ** 2).sum(dim=-1)  # (B,)
        rewards = torch.exp(-l2_dist_sq / self.temperature)  # (B,)

        # Interface-compatible dummy metrics
        device = rewards.device
        loss = torch.tensor(0.0, device=device)
        accuracy = torch.tensor(0.0, device=device)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "logits": rewards,
            "probabilities": rewards,
        }

    def compute_reward(
        self,
        observations: Any,
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Compute rewards from observations (alternative inference path)."""
        images = observations["images"] if isinstance(observations, dict) else observations
        pil_images = self._tensor_to_pil(images)
        embeddings = self._get_embedding(pil_images)

        diff = embeddings - self.expert_embedding
        l2_dist_sq = (diff ** 2).sum(dim=-1)
        rewards = torch.exp(-l2_dist_sq / self.temperature)
        return rewards
