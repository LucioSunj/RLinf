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

"""Single-robot FR3 inference using EasyWAM's native FastWAM-IDM model."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.base_policy import BasePolicy

from .fr3_processing import (
    ACTION_HORIZON,
    CAMERA_ROLES,
    NUM_VIDEO_FRAMES,
    Fr3Processor,
    _easywam_source,
    validate_training_config,
)


class FastWAMFr3Policy(BasePolicy):
    """Own one native IDM model and one request's raw joint anchor at a time.

    Args:
        model: Native EasyWAM model; production construction uses from_config.
        processor: The training-matched FR3 data processor.
        checkpoint_step: Step returned by the strict native checkpoint loader.
        action_dt: Confirmed time interval between training action targets.
        num_inference_steps: Video and action denoising steps.
        seed: Explicit per-request sampling seed.
        rand_device: Device used to generate initial noise.
        cached_context: Optional prompt-bound precomputed text embedding.
        preview_video: Decode and return the video used by the IDM action branch.
    """

    def __init__(
        self,
        model: Any,
        processor: Fr3Processor,
        *,
        checkpoint_step: int,
        action_dt: float = 0.1,
        num_inference_steps: int = 10,
        seed: int = 42,
        rand_device: str = "cpu",
        cached_context: Mapping[str, Any] | None = None,
        preview_video: bool = False,
    ) -> None:
        if not math.isfinite(action_dt) or action_dt <= 0:
            raise ValueError("action_dt must be finite and positive.")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        self.model = model
        self.processor = processor
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.rand_device = rand_device
        self.cached_context = cached_context
        self.preview_video = preview_video
        self._prompt: str | None = None
        self._context: tuple[torch.Tensor, torch.Tensor] | None = None
        self.metadata = {
            "protocol": "fastwam_fr3_v1",
            "model_variant": "fastwam_idm",
            "checkpoint_step": int(checkpoint_step),
            "camera_roles": list(CAMERA_ROLES),
            "state_dim": 8,
            "action_dim": 8,
            "action_horizon": ACTION_HORIZON,
            "num_video_frames": NUM_VIDEO_FRAMES,
            "action_dt": float(action_dt),
            "action_space": "absolute_joint_position",
            "gripper_encoding": "closed_0_1",
            "preview_video": preview_video,
        }

    @classmethod
    def from_config(cls, config: DictConfig) -> FastWAMFr3Policy:
        """Load the native model only when the caller explicitly builds a policy."""
        training = OmegaConf.load(Path(config.training_config).expanduser())
        validate_training_config(training)
        source = _easywam_source(config.easywam_root)
        checkpoint = Path(config.checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing FR3 checkpoint: {checkpoint}")
        processor = Fr3Processor(training, config.statistics, config.easywam_root)
        model_config = OmegaConf.create(
            OmegaConf.to_container(training.model, resolve=True)
        )
        context_path = config.get("context_path")
        if config.get("backbone_dir"):
            model_config.backbone.model_id = str(Path(config.backbone_dir).expanduser())
        if config.get("tokenizer_dir"):
            model_config.backbone.tokenizer_model_id = str(
                Path(config.tokenizer_dir).expanduser()
            )
        model_config.backbone.load_text_encoder = context_path is None
        model_config.skip_dit_load_from_pretrain = True
        model_config.action_dit_pretrained_path = None
        assets = Path(model_config.backbone.model_id)
        required = [assets / "Wan2.2_VAE.pth"]
        if context_path is None:
            required += [assets / "models_t5_umt5-xxl-enc-bf16.pth"]
            tokenizer = Path(model_config.backbone.tokenizer_model_id)
            if not tokenizer.is_dir():
                raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer}")
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Missing Wan2.2 inference asset: {path}")

        cached = None
        if context_path is not None:
            cached = torch.load(
                Path(context_path).expanduser(), map_location="cpu", weights_only=True
            )
            _validate_cached_context(
                cached, int(model_config.backbone.tokenizer_max_len)
            )
        import runtime

        if Path(runtime.__file__).resolve() != source / "runtime.py":
            raise RuntimeError(
                "The process imported a different runtime; launch FR3 in a fresh EasyWAM process."
            )
        model = instantiate(
            model_config, model_dtype=torch.bfloat16, device=str(config.device)
        )
        from model.helpers.inference import configure_model_execution

        model = configure_model_execution(
            model, vae_micro_batch_size=1, inference_cross_kv_reuse=True
        )
        payload = model.load_checkpoint(str(checkpoint))
        for key, expected in (
            ("model_variant", "fastwam_idm"),
            ("backbone_name", "wan22"),
            ("state_position", "context"),
            ("text_padding", "masked"),
        ):
            if payload.get(key) != expected:
                raise ValueError(
                    f"FR3 checkpoint {key}: expected {expected!r}, got {payload.get(key)!r}."
                )
        model.eval().requires_grad_(False)
        return cls(
            model,
            processor,
            checkpoint_step=int(payload["step"]),
            action_dt=float(config.get("action_dt", 0.1)),
            num_inference_steps=int(config.get("num_inference_steps", 10)),
            seed=int(config.get("seed", 42)),
            rand_device=str(config.get("rand_device", "cpu")),
            cached_context=cached,
            preview_video=bool(config.get("preview_video", False)),
        )

    def _text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cached_context is not None:
            payload = self.cached_context
            if "prompt" in payload:
                matches = payload["prompt"] == prompt
            else:
                from data.lerobot.text_embedding_cache import prompt_hash

                matches = payload["prompt_hash"] == prompt_hash(prompt)
            if not matches:
                raise ValueError(
                    "The request task does not match the precomputed context prompt."
                )
            context = payload["context"]
            mask = payload.get("context_mask", payload.get("mask"))
            return context.unsqueeze(0) if context.ndim == 2 else context, (
                mask.unsqueeze(0) if mask.ndim == 1 else mask
            )
        if prompt != self._prompt:
            self._context = self.model.encode_prompt(prompt)
            self._prompt = prompt
        return self._context

    @torch.inference_mode()
    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Return physical actions and echo the request identity without actuation."""
        episode_id, chunk_id = observation["episode_id"], observation["chunk_id"]
        if (
            not isinstance(episode_id, str)
            or not isinstance(chunk_id, int)
            or isinstance(chunk_id, bool)
        ):
            raise ValueError("episode_id must be a string and chunk_id an integer.")
        started = time.perf_counter()
        inputs = self.processor.prepare(observation)
        prepared = time.perf_counter()
        context, context_mask = self._text_context(inputs["prompt"])
        encoded = time.perf_counter()
        # Both native entrypoints use the same video-then-action denoising path.
        # Joint inference additionally decodes those same video latents afterwards.
        infer = (
            self.model.infer_joint_batch
            if self.preview_video
            else self.model.infer_action_batch
        )
        prediction = infer(
            prompt=None,
            context=context,
            context_mask=context_mask,
            input_image=inputs["input_image"],
            proprio=inputs["proprio"],
            action_horizon=ACTION_HORIZON,
            num_video_frames=NUM_VIDEO_FRAMES,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            rand_device=self.rand_device,
            **({"decode_video": True} if self.preview_video else {}),
        )
        inferred = time.perf_counter()
        actions = self.processor.decode(prediction["action"], inputs["raw_state"])
        result = {
            "actions": actions,
            "episode_id": episode_id,
            "chunk_id": chunk_id,
            "action_space": "absolute_joint_position",
            "normalized_actions": prediction["action"][0]
            .detach()
            .float()
            .cpu()
            .numpy(),
            "timing": {
                "preprocess_s": prepared - started,
                "text_s": encoded - prepared,
                "inference_s": inferred - encoded,
            },
        }
        if self.preview_video:
            packing_started = time.perf_counter()
            frames = prediction["video"][0]
            frame_dt = self.metadata["action_dt"] * 4
            buffer = BytesIO()
            frames[0].save(
                buffer,
                format="WEBP",
                save_all=True,
                append_images=frames[1:],
                duration=round(frame_dt * 1000),
                loop=0,
                quality=85,
                method=1,
            )
            result["predicted_video"] = {
                "data": buffer.getvalue(),
                "format": "webp",
                "num_frames": len(frames),
                "width": frames[0].width,
                "height": frames[0].height,
                "frame_dt": frame_dt,
                "relative_times_s": [
                    round(i * frame_dt, 6) for i in range(len(frames))
                ],
                "camera_roles": list(CAMERA_ROLES),
                "first_frame": "observation_reconstruction",
            }
            result["timing"]["video_pack_s"] = time.perf_counter() - packing_started
        result["timing"]["total_s"] = time.perf_counter() - started
        return result

    def default_forward(
        self, observation: Mapping[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Use the same single-request inference contract as the service."""
        return self.infer(observation)

    def predict_action_batch(
        self, env_obs: Mapping[str, Any], mode: str = "eval", **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return a batch of one for BasePolicy callers; training is unsupported."""
        if mode != "eval":
            raise ValueError("The FR3 IDM service policy only supports mode='eval'.")
        result = self.infer(env_obs)
        return torch.from_numpy(result["actions"]).unsqueeze(0), result


def _validate_cached_context(payload: Any, max_length: int) -> None:
    if not isinstance(payload, Mapping) or not (
        "prompt" in payload or "prompt_hash" in payload
    ):
        raise ValueError(
            "Precomputed context requires prompt (or native prompt_hash) binding."
        )
    context, mask = (
        payload.get("context"),
        payload.get("context_mask", payload.get("mask")),
    )
    if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError(
            "Precomputed context requires context and context_mask/mask tensors."
        )
    if context.ndim == 3 and context.shape[0] == 1:
        context = context[0]
    if mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask[0]
    if (
        context.ndim != 2
        or context.shape[1] != 4096
        or not 0 < context.shape[0] <= max_length
    ):
        raise ValueError(f"Expected text context [L,4096] with 0<L<={max_length}.")
    if mask.shape != (context.shape[0],) or mask.dtype != torch.bool or not mask.any():
        raise ValueError("Expected a nonempty boolean valid-token mask [L].")
    if not torch.is_floating_point(context) or not torch.isfinite(context).all():
        raise ValueError("Text context must contain finite floating point values.")
    if payload.get("encoder_id", "wan22ti2v5b") != "wan22ti2v5b":
        raise ValueError("Precomputed context must use the wan22ti2v5b text encoder.")


def get_model(config: DictConfig, torch_dtype: Any = None) -> FastWAMFr3Policy:
    """Build the registered inference policy with the checkpoint's BF16 dtype."""
    if torch_dtype is not None and torch_dtype != torch.bfloat16:
        raise ValueError("FR3 FastWAM-IDM requires precision='bf16'.")
    return FastWAMFr3Policy.from_config(config)
