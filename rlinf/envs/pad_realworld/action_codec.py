# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical metric actions and the existing Franka left Euler convention."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import RobotCommand


def canonical_delta_from_target(
    target: np.ndarray, anchor: np.ndarray, opened: float
) -> np.ndarray:
    """Recover the commanded increment, never a difference of measured states."""
    target = np.asarray(target, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if target.shape != (7,) or anchor.shape != (7,):
        raise ValueError("Recorded target and command anchor must be TCP poses.")
    rotation = Rotation.from_quat(target[3:]) * Rotation.from_quat(anchor[3:]).inv()
    return np.r_[target[:3] - anchor[:3], rotation.as_euler("xyz"), opened]


class ActionCodec:
    """One parameter-independent execution mapping shared by every method."""

    def __init__(
        self, *, delta_limits, workspace_min, workspace_max, rotation_min, rotation_max
    ) -> None:
        self.delta_limits = np.asarray(delta_limits, dtype=np.float64)
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64)
        self.rotation_min = np.asarray(rotation_min, dtype=np.float64)
        self.rotation_max = np.asarray(rotation_max, dtype=np.float64)
        if self.delta_limits.shape != (6,) or np.any(self.delta_limits <= 0):
            raise ValueError(
                "Six positive metric/radian command delta limits are required."
            )
        for low, high in (
            (self.workspace_min, self.workspace_max),
            (self.rotation_min, self.rotation_max),
        ):
            if low.shape != (3,) or high.shape != (3,) or not np.all(low < high):
                raise ValueError(
                    "Explicit three-dimensional workspace/orientation bounds are required."
                )
        if not all(
            np.isfinite(v).all()
            for v in (
                self.delta_limits,
                self.workspace_min,
                self.workspace_max,
                self.rotation_min,
                self.rotation_max,
            )
        ):
            raise ValueError("Execution limits must be finite.")

    def command(self, action: np.ndarray, anchor_pose: np.ndarray) -> RobotCommand:
        action = np.asarray(action, dtype=np.float64)
        anchor = np.asarray(anchor_pose, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Canonical action must contain seven finite values.")
        if anchor.shape != (7,) or not np.isfinite(anchor).all():
            raise ValueError("Command anchor must be a measured finite TCP pose.")
        limited = action.copy()
        limited[:6] = np.clip(limited[:6], -self.delta_limits, self.delta_limits)
        limited[6] = np.clip(limited[6], 0.0, 1.0)
        target = anchor.copy()
        target[:3] = np.clip(
            anchor[:3] + limited[:3], self.workspace_min, self.workspace_max
        )
        rotation = Rotation.from_euler("xyz", limited[3:6]) * Rotation.from_quat(
            anchor[3:]
        )
        euler = np.clip(rotation.as_euler("xyz"), self.rotation_min, self.rotation_max)
        target[3:] = Rotation.from_euler("xyz", euler).as_quat()
        limited = canonical_delta_from_target(target, anchor, limited[6])
        return RobotCommand(
            action.copy(), limited, anchor.copy(), target, 2 * limited[6] - 1
        )

    @staticmethod
    def scaled_franka_action(
        command: RobotCommand, action_scale: np.ndarray
    ) -> np.ndarray:
        """Adapt to FrankaEnv's scaled-delta API with positive=open gripper."""
        scale = np.asarray(action_scale, dtype=np.float64)
        if scale.shape != (3,) or not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError(
                "Franka scale must specify positive translation/rotation/gripper scales."
            )
        result = command.limited_action.copy()
        result[:3] /= scale[0]
        result[3:6] /= scale[1]
        result[6] = command.gripper_signal / scale[2]
        return result
