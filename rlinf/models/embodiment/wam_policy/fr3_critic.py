# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FR3 joint-state and camera inputs for the frozen pi0.5 prefix critic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class Fr3Pi05Inputs:
    """Use FR3 statistics and raw RGB, then the existing pi0.5 model transforms."""

    def __init__(self, statistics: str | Path):
        stats = json.loads(Path(statistics).read_text())["state"]["default"]
        self.mean = np.asarray(stats["global_mean"], dtype=np.float32)
        self.std = np.asarray(stats["global_std"], dtype=np.float32)
        if (
            self.mean.shape != (8,)
            or self.std.shape != (8,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or (self.std < 0).any()
        ):
            raise ValueError("FR3 critic requires finite 8-D state z-score statistics.")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("FR3 critic expects seven raw joints and gripper_closed.")
        images = [
            np.asarray(data[key])
            for key in ("observation/image", "observation/wrist_image")
        ]
        for image in images:
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(
                    "FR3 critic cameras must be raw RGB uint8 [480,640,3]."
                )
        return {
            "state": (state - self.mean) / np.maximum(self.std, 1e-6),
            "image": {
                "base_0_rgb": images[0],
                "left_wrist_0_rgb": images[1],
                "right_wrist_0_rgb": np.zeros_like(images[0]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "prompt": data["prompt"],
        }
