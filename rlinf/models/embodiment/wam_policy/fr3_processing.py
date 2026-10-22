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

"""Training-matched FR3 preprocessing and action decoding with raw anchors."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

CAMERA_ROLES = ("left", "wrist")
ACTION_HORIZON = 16
NUM_VIDEO_FRAMES = 5


def _easywam_source(root: str | Path) -> Path:
    source = Path(root).expanduser().resolve() / "src"
    if not (source / "model/fastwam.py").is_file():
        raise FileNotFoundError(f"Expected EasyWAM source at {source}.")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def validate_training_config(config: DictConfig) -> None:
    """Reject a recipe that changes the FR3 checkpoint's observation contract."""
    model, data = config.model, config.data.train
    processor = data.processor
    required = {
        "model._target_": (model.get("_target_"), "runtime.create_fastwam_idm"),
        "backbone": (model.backbone.name, "wan22"),
        "state_position": (model.state_position, "context"),
        "state_dim": (model.state_dim, 8),
        "action_dim": (model.action_dit_config.action_dim, 8),
        "num_frames": (data.num_frames, 17),
        "action_video_freq_ratio": (data.action_video_freq_ratio, 4),
        "global_sample_stride": (data.global_sample_stride, 1),
        "concat_multi_camera": (data.concat_multi_camera, "horizontal"),
        "video_size": (list(data.video_size), [224, 448]),
        "normalization": (processor.norm_default_mode, "z-score"),
        "stepwise normalization": (processor.use_stepwise_action_norm, False),
        "normalization exceptions": (processor.get("norm_exception_mode"), None),
        "observation steps": (processor.num_obs_steps, 17),
        "delta_action_dim_mask": (processor.get("delta_action_dim_mask"), None),
        "action_output_dim": (processor.action_output_dim, 8),
        "proprio_output_dim": (processor.proprio_output_dim, 8),
        "num_output_cameras": (processor.num_output_cameras, 2),
    }
    for name, (actual, expected) in required.items():
        if actual != expected:
            raise ValueError(f"FR3 {name}: expected {expected!r}, got {actual!r}.")
    cameras = processor.shape_meta.images
    if [item.key for item in cameras] != list(CAMERA_ROLES):
        raise ValueError("FR3 camera order must be ['left', 'wrist'].")
    for camera in cameras:
        if list(camera.raw_shape) != [3, 480, 640] or list(camera.shape) != [
            3,
            224,
            224,
        ]:
            raise ValueError("FR3 cameras must map RGB [3,480,640] to [3,224,224].")
    for field in ("state", "action"):
        meta = processor.shape_meta[field]
        if len(meta) != 1 or meta[0].key != "default" or int(meta[0].shape) != 8:
            raise ValueError(f"FR3 {field} requires one 8-D default field.")
    transforms = processor.action_state_transforms
    if (
        len(transforms) != 1
        or transforms[0].get("_target_")
        != "data.lerobot.transforms.relative_action.RelativeJointTransform"
        or list(transforms[0]["keys"]) != ["default"]
        or transforms[0].get("joint_dim") != 7
    ):
        raise ValueError(
            "FR3 requires RelativeJointTransform(keys=[default], joint_dim=7)."
        )

    expected_images = [
        {"_target_": "data.lerobot.transforms.image.ToTensor"},
        {"_target_": "torchvision.transforms.Resize", "size": [168, 224]},
        {"_target_": "data.lerobot.transforms.image.Pad", "padding": [0, 28, 0, 28]},
    ]
    for mode in ("train_transforms", "val_transforms"):
        if OmegaConf.to_container(processor[mode], resolve=True) != expected_images:
            raise ValueError("FR3 cameras require resize 168x224 then vertical pad 28.")


class Fr3Processor:
    """Apply training transforms and recover all sixteen absolute joint targets.

    Args:
        training_config: Resolved EasyWAM training configuration.
        statistics: Dataset statistics JSON from the same training run.
        easywam_root: EasyWAM checkout containing the native data transforms.
    """

    def __init__(
        self,
        training_config: DictConfig,
        statistics: str | Path,
        easywam_root: str | Path,
    ) -> None:
        validate_training_config(training_config)
        _easywam_source(easywam_root)
        from data.dataset_utils import (
            CenterCrop,
            Normalize,
            ResizeSmallestSideAspectPreserving,
        )
        from data.lerobot.prompts import DEFAULT_PROMPT
        from data.lerobot.utils.normalizer import load_dataset_stats_from_json

        self.processor = instantiate(training_config.data.train.processor).eval()
        stats = load_dataset_stats_from_json(str(Path(statistics).expanduser()))
        for field in ("state", "action"):
            for name in ("global_mean", "global_std"):
                values = stats[field]["default"][name]
                if values.shape != (8,) or not torch.isfinite(values).all():
                    raise ValueError(
                        f"Expected finite {field}.{name} statistics with shape [8]."
                    )
                if name == "global_std" and (values < 0).any():
                    raise ValueError(
                        f"Expected nonnegative {field} standard deviations."
                    )
        self.processor.set_normalizer_from_stats(stats)
        image_size = {"img_w": 448, "img_h": 224}
        self.resize = ResizeSmallestSideAspectPreserving(image_size)
        self.crop = CenterCrop(image_size)
        self.normalize_image = Normalize({"mean": 0.5, "std": 0.5})
        self.prompt_template = DEFAULT_PROMPT

    def prepare(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare a single request while retaining its unmodified state anchor."""
        state = np.asarray(observation["observation.state"])
        if state.shape != (8,) or not np.issubdtype(state.dtype, np.floating):
            raise ValueError("observation.state must be a floating point vector [8].")
        if not np.isfinite(state).all() or not 0 <= state[7] <= 1:
            raise ValueError("FR3 state must be finite with gripper_closed in [0,1].")
        raw_state = torch.from_numpy(
            np.array(state, dtype=np.float32, copy=True)
        ).reshape(1, 1, 8)
        state_batch = {"state": {"default": raw_state[0].clone()}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        state_batch = self.processor.action_state_merger.forward(state_batch)

        images = []
        for role in CAMERA_ROLES:
            image = np.asarray(observation[f"observation.images.{role}"])
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"observation.images.{role} must be RGB uint8 [480,640,3]."
                )
            tensor = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            transforms = self.processor.val_transforms
            if isinstance(transforms, Mapping):
                transforms = transforms[role]
            for transform in transforms:
                tensor = transform(tensor)
            if tensor.shape != (1, 3, 224, 224):
                raise ValueError(
                    f"Expected transformed {role} image [1,3,224,224], got {tuple(tensor.shape)}."
                )
            images.append(tensor)
        image = torch.cat(images, dim=-1)
        image = self.normalize_image(self.crop(self.resize(image)))
        task = observation["task"]
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must contain the training task instruction.")
        return {
            "input_image": image,
            "proprio": state_batch["state"],
            "raw_state": raw_state,
            "prompt": self.prompt_template.format(task=task),
        }

    def decode(
        self, normalized_action: torch.Tensor, raw_state: torch.Tensor
    ) -> np.ndarray:
        """Undo normalization and relative joints without the training time crop."""
        action = normalized_action.detach().to(device="cpu", dtype=torch.float32)
        if action.shape != (1, ACTION_HORIZON, 8) or not torch.isfinite(action).all():
            raise ValueError("Expected finite normalized actions [1,16,8].")
        raw_state = raw_state.detach().to(device="cpu", dtype=torch.float32)
        if raw_state.shape != (1, 1, 8) or not torch.isfinite(raw_state).all():
            raise ValueError("Expected a finite raw state anchor [1,1,8].")
        batch = self.processor.action_state_merger.backward(
            {"action": action, "state": raw_state}
        )
        batch["action"]["default"] = self.processor.normalizer.normalizers["action"][
            "default"
        ].backward(batch["action"]["default"])
        # The state is already raw: inversion of clamped proprio would move q0.
        for transform in reversed(self.processor.action_state_transforms):
            batch = transform.backward(batch)
        result = batch["action"]["default"][0].numpy().astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Action inverse transform produced non-finite targets.")
        return result

    def prepare_batch(self, observations: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Prepare rows independently; each row keeps its own immutable q0."""
        if not observations:
            raise ValueError("FR3 observation batch must not be empty.")
        rows = [self.prepare(observation) for observation in observations]
        return {
            "input_image": torch.cat([row["input_image"] for row in rows]),
            "proprio": torch.stack([row["proprio"] for row in rows]),
            "raw_state": torch.cat([row["raw_state"] for row in rows]),
            "prompts": [row["prompt"] for row in rows],
        }

    def decode_batch(
        self, normalized_action: torch.Tensor, raw_state: torch.Tensor
    ) -> np.ndarray:
        """Decode B independent chunks without storing mutable processor anchors."""
        if normalized_action.ndim != 3 or normalized_action.shape[1:] != (16, 8):
            raise ValueError("Expected normalized actions [B,16,8].")
        if raw_state.shape != (normalized_action.shape[0], 1, 8):
            raise ValueError("Expected a separate raw state [B,1,8] for each chunk.")
        return np.stack(
            [
                self.decode(action[None], state[None])
                for action, state in zip(normalized_action, raw_state, strict=True)
            ]
        )
