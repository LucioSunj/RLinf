# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Map RLinf embodied observations to frozen FastWAM inputs.

env_obs (LIBERO/RoboTwin) provides:
    {main_images, wrist_images, states, task_descriptions}
GatePolicy.obs_preprocessor must return batched:
    {input_image [B,3,H,W], proprio [B,P], context [B,L,D], context_mask [B,L]}

- proprio   : `states` normalized with the FastWAM dataset stats.
- context   : precomputed FastWAM text embedding of `task_descriptions`, cached per
              unique instruction. Online encoding is an explicit fallback only, so
              the large Wan text encoder is not resident on the rollout GPU.
- input_image: assembled to the fast-wam layout for the suite:
    LIBERO  : resize main & wrist to 224x224, hconcat -> [3,224,448].
    RoboTwin: head 320x256, two wrists 160x128 stacked below -> [3,384,320].
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _to_bchw(images: torch.Tensor) -> torch.Tensor:
    """Accept [B,H,W,3] or [B,3,H,W] -> [B,3,H,W] float."""
    if images.ndim != 4:
        raise ValueError(f"expected 4D image batch, got {tuple(images.shape)}")
    if images.shape[1] == 3:
        pass
    elif images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"image batch has no RGB channel dimension: {tuple(images.shape)}")
    return images.float()


def _first_wrist_image(images: torch.Tensor) -> torch.Tensor:
    """Accept [B,H,W,3], [B,3,H,W], or [B,N,H,W,3] and return one wrist cam."""
    if images is None:
        raise ValueError("LIBERO gate preprocessing requires a wrist camera image")
    if images.ndim == 5:
        images = images[:, 0]
    return _to_bchw(images)


def _robotwin_wrist_pair(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left/right wrist batches from RoboTwin [B,N,H,W,3] observations."""
    if images is None:
        raise ValueError(
            "RoboTwin gate preprocessing requires two wrist cameras, but "
            "`wrist_images` is None. Set collect_wrist_camera=true."
        )
    if images.ndim == 5:
        if images.shape[1] < 2:
            raise ValueError(
                f"RoboTwin gate preprocessing requires two wrist cameras, got {tuple(images.shape)}"
            )
        return _to_bchw(images[:, 0]), _to_bchw(images[:, 1])
    raise ValueError(
        "RoboTwin gate preprocessing requires [B,2,H,W,3] wrist images; "
        f"got {tuple(images.shape)}"
    )


def _resize(images: torch.Tensor, hw) -> torch.Tensor:
    # Match torchvision/PIL antialiased bilinear resizing used by FastWAM's
    # training dataset and official deployment adapters. Without antialiasing,
    # downsampling injects a large distribution shift into the frozen VAE.
    return F.interpolate(
        images, size=hw, mode="bilinear", align_corners=False, antialias=True
    )


def _normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize uint8/[0,255] or [0,1], while preserving existing [-1,1]."""
    x = images.float()
    x_min, x_max = float(x.min()), float(x.max())
    if x_max > 1.5:
        x = x / 255.0
        x_min, x_max = float(x.min()), float(x.max())
    if x_min >= 0.0 and x_max <= 1.0:
        return x * 2.0 - 1.0
    if x_min >= -1.0 and x_max <= 1.0:
        return x
    raise ValueError(
        f"image values must be uint8/[0,255], [0,1], or [-1,1], got [{x_min}, {x_max}]"
    )


def _fallback_pool_text_context(
    context: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    output_dim: int = 64,
) -> torch.Tensor:
    squeeze = context.ndim == 2
    if squeeze:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError(
            "context/context_mask must be [B,L,D]/[B,L], got "
            f"{tuple(context.shape)}/{tuple(context_mask.shape)}"
        )
    if context.shape[:2] != context_mask.shape:
        raise ValueError("context_mask must match the first two context dimensions")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")
    mask = context_mask.to(device=context.device, dtype=torch.bool)
    counts = mask.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise ValueError("context_mask must contain at least one valid token per sample")
    pooled = (
        context.float() * mask.unsqueeze(-1).to(dtype=torch.float32)
    ).sum(dim=1) / counts.to(dtype=torch.float32)
    compressed = F.adaptive_avg_pool1d(pooled.unsqueeze(1), output_dim).squeeze(1)
    return compressed[0] if squeeze else compressed


def pool_text_context(
    context: torch.Tensor,
    context_mask: torch.Tensor,
    output_dim: int = 64,
) -> torch.Tensor:
    """Deterministically compress masked WAM text context without learned weights."""
    try:
        from fastwam.adaptive_gate.features import pool_text_context as fastwam_pool
    except ImportError:
        return _fallback_pool_text_context(
            context, context_mask, output_dim=output_dim
        )
    return fastwam_pool(context, context_mask, output_dim=output_dim)


def _libero_image(env_obs) -> torch.Tensor:
    main = _to_bchw(env_obs["main_images"])
    wrist = _first_wrist_image(env_obs.get("wrist_images"))
    main = _resize(main, (224, 224))
    wrist = _resize(wrist, (224, 224))
    img = torch.cat([main, wrist], dim=-1)  # hconcat -> [B,3,224,448]
    return _normalize(img)


def _robotwin_image(env_obs) -> torch.Tensor:
    # head 320x256; two wrists 160x128 stacked below the head -> [3,384,320].
    main = _resize(_to_bchw(env_obs["main_images"]), (256, 320))
    left, right = _robotwin_wrist_pair(env_obs.get("wrist_images"))
    left = _resize(left, (128, 160))
    right = _resize(right, (128, 160))
    bottom = torch.cat([left, right], dim=-1)  # [B,3,128,320]
    img = torch.cat([main, bottom], dim=-2)  # vstack -> [B,3,384,320]
    return _normalize(img)


_IMAGE_FN = {"libero": _libero_image, "robotwin": _robotwin_image}


class GateObsPreprocessor:
    """Callable env_obs -> fast-wam inputs, with a per-instruction text cache."""

    def __init__(
        self,
        wam_model,
        suite: str,
        *,
        processor=None,
        prompt_template: str | None = None,
        device=None,
        text_feat_dim: int = 64,
        text_embedding_cache_dir: str | Path | None = None,
        context_len: int = 128,
        text_encoder_id: str = "wan22ti2v5b",
        allow_online_text_encoding: bool = False,
        binarize_libero_gripper: bool = True,
    ):
        suite = str(suite)
        if suite not in _IMAGE_FN:
            raise ValueError(f"unknown suite `{suite}`, expected one of {sorted(_IMAGE_FN)}")
        self.suite = suite
        self.wam_model = wam_model  # frozen WAM (must expose encode_prompt for text)
        self.processor = processor
        self.prompt_template = (
            prompt_template
            or "A video recorded from a robot's point of view executing the "
            "following instruction: {task}"
        )
        self.device = device
        self.text_feat_dim = int(text_feat_dim)
        self.text_embedding_cache_dir = (
            None
            if text_embedding_cache_dir is None
            else Path(text_embedding_cache_dir).expanduser().resolve()
        )
        self.context_len = int(context_len)
        self.text_encoder_id = str(text_encoder_id)
        self.allow_online_text_encoding = bool(allow_online_text_encoding)
        self.binarize_libero_gripper = bool(binarize_libero_gripper)
        self._image_fn = _IMAGE_FN[suite]
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _format_prompt(self, desc: Any) -> str:
        text = str(desc)
        if "{task}" in self.prompt_template:
            return self.prompt_template.format(task=text)
        return self.prompt_template + text

    @torch.no_grad()
    def _encode_text(self, descriptions):
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        ctxs, masks, text_feats = [], [], []
        for desc in descriptions:
            key = self._format_prompt(desc)
            if key not in self._text_cache:
                ctx, raw_mask = self._load_cached_text(key)
                if ctx is None:
                    if not self.allow_online_text_encoding:
                        cache_hint = self.text_embedding_cache_dir or "<unset>"
                        raise FileNotFoundError(
                            f"Missing FastWAM text embedding for prompt in {cache_hint}. "
                            "Run scripts/precompute_text_embeds.py. Online text encoding "
                            "is disabled to avoid keeping the Wan text encoder on rollout GPU."
                        )
                    if not hasattr(self.wam_model, "encode_prompt"):
                        raise RuntimeError(
                            "allow_online_text_encoding=True but WAM has no encode_prompt"
                        )
                    ctx, raw_mask = self.wam_model.encode_prompt(key)
                    ctx, raw_mask = ctx.detach().cpu(), raw_mask.detach().cpu()
                if ctx.ndim == 2:
                    ctx = ctx.unsqueeze(0)
                if raw_mask.ndim == 1:
                    raw_mask = raw_mask.unsqueeze(0)
                # Match FastWAM training: padded context is zero, then cross-attn
                # sees an all-true mask.
                model_ctx = ctx.clone()
                model_ctx[~raw_mask.bool()] = 0.0
                model_mask = torch.ones_like(raw_mask, dtype=torch.bool)
                text_feat = pool_text_context(
                    model_ctx, model_mask, output_dim=self.text_feat_dim
                ).cpu()
                if self.device is not None:
                    model_ctx = model_ctx.to(self.device)
                    model_mask = model_mask.to(self.device)
                    text_feat = text_feat.to(self.device)
                self._text_cache[key] = (model_ctx, model_mask, text_feat)
            ctx, mask, text_feat = self._text_cache[key]
            ctxs.append(ctx)
            masks.append(mask)
            text_feats.append(text_feat)
        return (
            torch.cat(ctxs, dim=0),
            torch.cat(masks, dim=0),
            torch.cat(text_feats, dim=0),
        )

    def _load_cached_text(
        self, prompt: str
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.text_embedding_cache_dir is None:
            return None, None
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = self.text_embedding_cache_dir / (
            f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        )
        if not cache_path.exists():
            return None, None
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not {"context", "mask"} <= payload.keys():
            raise ValueError(f"Malformed text embedding cache: {cache_path}")
        context, mask = payload["context"], payload["mask"].bool()
        if context.ndim != 2 or mask.ndim != 1 or context.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Malformed context/mask shapes in {cache_path}: "
                f"{tuple(context.shape)}/{tuple(mask.shape)}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Expected context_len={self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        return context, mask

    def normalize_proprio(self, states: torch.Tensor) -> torch.Tensor:
        states = states.float()
        if self.processor is None:
            return states
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("GatePolicy expects one merged FastWAM state key.")
        state_key = state_meta[0]["key"]
        batch = {"state": {state_key: states}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][state_key]

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert FastWAM normalized actions [B,T,D] back to simulator space."""
        if self.processor is None:
            return actions
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(actions.shape)}")
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("GatePolicy expects one merged FastWAM action key.")
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        device = actions.device
        denorm = normalizer.backward(actions.to(dtype=torch.float32, device="cpu"))
        return denorm.to(device=device, dtype=torch.float32)

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions and convert to the target simulator convention."""
        actions = self.denormalize_actions(actions)
        if self.suite == "libero":
            actions = actions.clone()
            # FastWAM data uses 0=close, 1=open; LIBERO executes -1=open,+1=close.
            actions[..., -1] = -(actions[..., -1] * 2.0 - 1.0)
            if self.binarize_libero_gripper:
                actions[..., -1] = torch.sign(actions[..., -1])
        return actions.float()

    @torch.no_grad()
    def __call__(self, env_obs):
        input_image = self._image_fn(env_obs)
        proprio = self.normalize_proprio(env_obs["states"])
        context, context_mask, text_feat = self._encode_text(
            env_obs["task_descriptions"]
        )
        if self.device is not None:
            input_image = input_image.to(self.device)
            proprio = proprio.to(self.device)
            context = context.to(self.device)
            context_mask = context_mask.to(self.device)
            text_feat = text_feat.to(self.device)
        return {
            "input_image": input_image,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "text_feat": text_feat,
        }


def make_gate_obs_preprocessor(
    wam_model,
    suite: str,
    *,
    processor=None,
    prompt_template: str | None = None,
    device=None,
    text_feat_dim: int = 64,
    text_embedding_cache_dir: str | Path | None = None,
    context_len: int = 128,
    text_encoder_id: str = "wan22ti2v5b",
    allow_online_text_encoding: bool = False,
    binarize_libero_gripper: bool = True,
) -> GateObsPreprocessor:
    return GateObsPreprocessor(
        wam_model,
        suite,
        processor=processor,
        prompt_template=prompt_template,
        device=device,
        text_feat_dim=text_feat_dim,
        text_embedding_cache_dir=text_embedding_cache_dir,
        context_len=context_len,
        text_encoder_id=text_encoder_id,
        allow_online_text_encoding=allow_online_text_encoding,
        binarize_libero_gripper=binarize_libero_gripper,
    )
