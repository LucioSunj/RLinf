# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The same explicit image/state conversion for recording and policy inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .contracts import CameraSpec, ObservationSnapshot


class ObservationAdapter:
    """Select measured TCP + gripper state, without sorting or padding raw fields."""

    def __init__(
        self,
        cameras: tuple[CameraSpec, ...],
        *,
        quaternion_order: str = "xyzw",
        pose_key: str = "tcp_pose",
        gripper_key: str = "gripper_position",
        gripper_closed: float = 0.0,
        gripper_open: float = 1.0,
        max_age_seconds: float = 1.0,
        max_camera_skew_seconds: float = 0.05,
        clock_domain: str = "robot_monotonic",
    ) -> None:
        if not cameras or len({camera.name for camera in cameras}) != len(cameras):
            raise ValueError("Specify at least one uniquely named camera.")
        if quaternion_order not in {"xyzw", "wxyz"}:
            raise ValueError("Quaternion order must be explicit xyzw or wxyz.")
        if (
            not np.isfinite([gripper_closed, gripper_open]).all()
            or gripper_closed == gripper_open
        ):
            raise ValueError("Measured gripper open/closed calibration is required.")
        if max_age_seconds <= 0 or max_camera_skew_seconds < 0:
            raise ValueError("Freshness and camera skew limits are invalid.")
        self.cameras = cameras
        self.quaternion_order = quaternion_order
        self.pose_key = pose_key
        self.gripper_key = gripper_key
        self.gripper_closed = gripper_closed
        self.gripper_open = gripper_open
        self.max_age_seconds = max_age_seconds
        self.max_camera_skew_seconds = max_camera_skew_seconds
        self.clock_domain = clock_domain
        self._previous_quaternion: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ObservationAdapter:
        camera_cfg = cfg["cameras"]
        cameras = tuple(
            CameraSpec(
                name=item["name"],
                source_color=item["source_color"],
                resize=tuple(item["resize"]),
                crop=None if item.get("crop") is None else tuple(item["crop"]),
            )
            for item in camera_cfg
        )
        return cls(cameras, **cfg["state"], **cfg["freshness"])

    def reset(self) -> None:
        """Reset only quaternion sign continuity at an explicit episode boundary."""
        self._previous_quaternion = None

    def canonical_pose(self, state: dict[str, np.ndarray]) -> np.ndarray:
        pose = np.asarray(state[self.pose_key], dtype=np.float64).copy()
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(
                "Measured TCP pose must contain 3 meters + 4 quaternion values."
            )
        if self.quaternion_order == "wxyz":
            pose[3:] = pose[[4, 5, 6, 3]]
        norm = np.linalg.norm(pose[3:])
        if norm < 1e-8:
            raise ValueError("Measured TCP quaternion cannot be zero.")
        pose[3:] /= norm
        return pose

    def canonical_state(self, state: dict[str, np.ndarray]) -> np.ndarray:
        pose = self.canonical_pose(state)
        q = pose[3:]
        previous = self._previous_quaternion
        if (previous is None and q[-1] < 0) or (
            previous is not None and q @ previous < 0
        ):
            pose[3:] *= -1
        self._previous_quaternion = pose[3:].copy()
        measured = np.asarray(state[self.gripper_key]).reshape(-1)
        if measured.size != 1 or not np.isfinite(measured).all():
            raise ValueError("Measured scalar gripper state is required.")
        opened = (float(measured[0]) - self.gripper_closed) / (
            self.gripper_open - self.gripper_closed
        )
        if not -1e-4 <= opened <= 1.0001:
            raise ValueError("Measured gripper state is outside its calibrated range.")
        return np.r_[pose, np.clip(opened, 0.0, 1.0)].astype(np.float32)

    def validate_freshness(self, snapshot: ObservationSnapshot, now: float) -> None:
        if snapshot.clock_domain != self.clock_domain:
            raise ValueError("Uncalibrated clock domains cannot be subtracted.")
        times = []
        for camera in self.cameras:
            received = snapshot.receive_times[camera.name]
            captured = snapshot.capture_times[camera.name]
            reference = received if captured is None else captured
            if (
                not np.isfinite([received, reference, now]).all()
                or not 0 <= now - reference <= self.max_age_seconds
            ):
                raise ValueError(f"Stale or invalid observation from {camera.name}.")
            times.append(reference)
        # Capture timestamps are compared only when all views have them.
        known = [snapshot.capture_times[c.name] is not None for c in self.cameras]
        if any(known) and not all(known):
            raise ValueError(
                "Mixed capture/receive clocks cannot establish camera alignment."
            )
        if max(times) - min(times) > self.max_camera_skew_seconds:
            raise ValueError("Camera views exceed the configured synchronization skew.")

    def images(self, snapshot: ObservationSnapshot) -> tuple[np.ndarray, ...]:
        output = []
        for spec in self.cameras:
            image = snapshot.images[spec.name]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Camera {spec.name} must supply HWC uint8.")
            if spec.source_color == "BGR":
                image = image[..., ::-1]
            if spec.crop is not None:
                top, left, bottom, right = spec.crop
                if bottom > image.shape[0] or right > image.shape[1]:
                    raise ValueError(f"Crop exceeds camera {spec.name}'s actual frame.")
                image = image[top:bottom, left:right]
            image = Image.fromarray(np.ascontiguousarray(image)).resize(
                (spec.resize[1], spec.resize[0]), Image.Resampling.BILINEAR
            )
            output.append(np.asarray(image).copy())
        return tuple(output)
