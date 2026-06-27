# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Map RLinf embodied env_obs -> fast-wam inputs for the GatePolicy.

env_obs (LIBERO/RoboTwin) provides:
    {main_images, wrist_images, states, task_descriptions}
GatePolicy.obs_preprocessor must return batched:
    {input_image [B,3,H,W], proprio [B,P], context [B,L,D], context_mask [B,L]}

- proprio   : `states` directly (LIBERO: 8-dim eef_pos+axisangle+gripper).
- context   : text embedding of `task_descriptions`, cached per unique instruction
              (instructions are fixed per task). Requires the frozen WAM to expose
              `encode_prompt` (build it with load_text_encoder=true for RL, or
              precompute embeddings offline). No GT leakage (text only).
- input_image: assembled to the fast-wam layout for the suite:
    LIBERO  : resize main & wrist to 224x224, hconcat -> [3,224,448].
    RoboTwin: head 320x256, two wrists 160x128 stacked below -> [3,384,320].
  # TODO(verify-on-server): match the EXACT resize/normalization fast-wam eval uses
  #   (experiments/libero/eval_libero_single.py + libero_utils; robotwin deploy_policy).
  #   Wrong layout/normalization fails silently — diff against the eval pipeline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_bchw(images: torch.Tensor) -> torch.Tensor:
    """Accept [B,H,W,3] or [B,3,H,W] -> [B,3,H,W] float."""
    if images.ndim != 4:
        raise ValueError(f"expected 4D image batch, got {tuple(images.shape)}")
    if images.shape[1] != 3 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    return images.float()


def _resize(images: torch.Tensor, hw) -> torch.Tensor:
    return F.interpolate(images, size=hw, mode="bilinear", align_corners=False)


def _normalize(images: torch.Tensor) -> torch.Tensor:
    # fast-wam VAE expects [-1,1]. If inputs are uint8/[0,255] or [0,1], map to [-1,1].
    # TODO(verify-on-server): confirm against the eval pipeline's normalization.
    x = images
    if x.max() > 1.5:
        x = x / 255.0
    return x * 2.0 - 1.0


def _libero_image(env_obs) -> torch.Tensor:
    main = _to_bchw(env_obs["main_images"])
    wrist = _to_bchw(env_obs["wrist_images"])
    main = _resize(main, (224, 224))
    wrist = _resize(wrist, (224, 224))
    img = torch.cat([main, wrist], dim=-1)  # hconcat -> [B,3,224,448]
    return _normalize(img)


def _robotwin_image(env_obs) -> torch.Tensor:
    # head 320x256; two wrists 160x128 stacked below the head -> [3,384,320].
    main = _resize(_to_bchw(env_obs["main_images"]), (256, 320))
    wrist = _to_bchw(env_obs["wrist_images"])
    # TODO(verify-on-server): RoboTwin has TWO wrist cams; split/concat to 2x [128,160]
    #   side by side -> [128,320], stacked under head -> [384,320]. Adjust to the
    #   exact deploy_policy layout.
    wrist = _resize(wrist, (128, 320))
    img = torch.cat([main, wrist], dim=-2)  # vstack -> [B,3,384,320]
    return _normalize(img)


_IMAGE_FN = {"libero": _libero_image, "robotwin": _robotwin_image}


class GateObsPreprocessor:
    """Callable env_obs -> fast-wam inputs, with a per-instruction text cache."""

    def __init__(self, wam_model, suite: str, device=None):
        suite = str(suite)
        if suite not in _IMAGE_FN:
            raise ValueError(f"unknown suite `{suite}`, expected one of {sorted(_IMAGE_FN)}")
        self.suite = suite
        self.wam_model = wam_model  # frozen WAM (must expose encode_prompt for text)
        self.device = device
        self._image_fn = _IMAGE_FN[suite]
        self._text_cache: dict[str, tuple] = {}

    @torch.no_grad()
    def _encode_text(self, descriptions):
        ctxs, masks = [], []
        for desc in descriptions:
            key = str(desc)
            if key not in self._text_cache:
                if not hasattr(self.wam_model, "encode_prompt"):
                    raise RuntimeError(
                        "WAM has no `encode_prompt`; build it with load_text_encoder=true "
                        "for RL, or precompute task-instruction embeddings offline."
                    )
                ctx, mask = self.wam_model.encode_prompt(key)  # [1,L,D],[1,L]
                self._text_cache[key] = (ctx.detach(), mask.detach())
            ctx, mask = self._text_cache[key]
            ctxs.append(ctx)
            masks.append(mask)
        return torch.cat(ctxs, dim=0), torch.cat(masks, dim=0)

    @torch.no_grad()
    def __call__(self, env_obs):
        input_image = self._image_fn(env_obs)
        proprio = env_obs["states"].float()
        context, context_mask = self._encode_text(env_obs["task_descriptions"])
        if self.device is not None:
            input_image = input_image.to(self.device)
            proprio = proprio.to(self.device)
            context = context.to(self.device)
            context_mask = context_mask.to(self.device)
        return {
            "input_image": input_image,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
        }


def make_gate_obs_preprocessor(wam_model, suite: str, device=None) -> GateObsPreprocessor:
    return GateObsPreprocessor(wam_model, suite, device=device)
