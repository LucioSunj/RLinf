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

"""Exact FastWAM official camera preprocessing for standard LIBERO."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

OFFICIAL_LIBERO_CAMERA_RESIZE_MODE = "official_pil_center_crop"


def _positive_dimension(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _official_center_crop_resize(
    image: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    pil_image = Image.fromarray(np.ascontiguousarray(image))
    source_width, source_height = pil_image.size
    scale = max(width / source_width, height / source_height)
    resized = pil_image.resize(
        (
            round(source_width * scale),
            round(source_height * scale),
        ),
        resample=Image.BILINEAR,
    )
    resized_width, resized_height = resized.size
    left = max((resized_width - width) // 2, 0)
    top = max((resized_height - height) // 2, 0)
    return np.asarray(
        resized.crop((left, top, left + width, top + height)),
        dtype=np.uint8,
    )


def prepare_libero_camera_batch(
    images: Any,
    *,
    height: int,
    width: int,
    resize_mode: str,
) -> torch.Tensor:
    """Apply the official PIL resize/crop and return uint8 BCHW images.

    Args:
        images: Batched uint8 camera frames in BHWC or BCHW layout.
        height: Per-camera output height.
        width: Per-camera output width.
        resize_mode: Explicit preprocessing implementation identifier.

    Returns:
        A CPU uint8 tensor with the official evaluator's exact pixels.
    """

    height = _positive_dimension(height, name="Camera height")
    width = _positive_dimension(width, name="Camera width")
    if str(resize_mode) != OFFICIAL_LIBERO_CAMERA_RESIZE_MODE:
        raise ValueError(f"Unsupported LIBERO camera resize mode: {resize_mode!r}.")
    if isinstance(images, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(images))
    else:
        tensor = torch.as_tensor(images).detach().cpu()
    if tensor.ndim != 4:
        raise ValueError(f"LIBERO images must be rank four, got {tensor.shape}.")
    if tensor.shape[-1] != 3:
        if tensor.shape[1] != 3:
            raise ValueError(f"Cannot identify RGB channel in shape {tensor.shape}.")
        tensor = tensor.permute(0, 2, 3, 1)
    if tensor.dtype != torch.uint8:
        raise TypeError("Official LIBERO camera preprocessing requires uint8 input.")
    frames = tensor.contiguous().numpy()
    resized = np.stack(
        [
            _official_center_crop_resize(frame, height=height, width=width)
            for frame in frames
        ],
        axis=0,
    )
    return torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous()
