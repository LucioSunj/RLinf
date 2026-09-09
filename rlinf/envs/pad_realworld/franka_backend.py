# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Explicit binding to an already connected RLinf Franka controller and cameras.

Import and construction perform no RPC, connection, camera open, reset or error
recovery. The caller owns startup and must obtain motion authorization separately.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import replace

import numpy as np

from .action_codec import canonical_delta_from_target
from .contracts import ObservationSnapshot


class MonotonicClock:
    def now(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(max(0.0, seconds))


class FrankaControllerBackend:
    """Reuse native worker RPCs without FrankaEnv.step/reset/reward side effects.

    Acknowledgement means that the controller RPC submitted the target. Native
    Worker ``wait()`` has no timeout/cancellation parameter; Python stop only
    prevents subsequent submissions. Independent hardware stopping remains with
    the existing controller/site operator, including its contact limits.
    """

    is_mock = False
    rpc_cancel_supported = False
    acknowledgement = "controller_submission_then_state_read"

    def __init__(
        self,
        *,
        controller,
        cameras: dict,
        score_source,
        clip_position,
        gripper_wait_seconds: float,
        camera_timeout_seconds: float,
        calibration_id: str,
        clock=None,
    ):
        if (
            not cameras
            or gripper_wait_seconds < 0
            or camera_timeout_seconds <= 0
            or not calibration_id
        ):
            raise ValueError(
                "Explicit camera mapping, gripper wait and calibration are required."
            )
        self.controller = controller
        self.cameras = dict(cameras)
        self.score_source = score_source
        self.clip_position = clip_position
        self.gripper_wait_seconds = gripper_wait_seconds
        self.camera_timeout_seconds = camera_timeout_seconds
        self.calibration_id = calibration_id
        self.clock = clock or MonotonicClock()

    @classmethod
    def from_connected_env(
        cls,
        env,
        *,
        cameras,
        score_source,
        gripper_wait_seconds,
        camera_timeout_seconds,
        calibration_id,
        clock=None,
    ):
        """Bind an explicitly provided existing FrankaEnv, preserving its safety box.

        Do not construct FrankaEnv here: its initialization opens cameras and
        launches controller workers. Cameras map model names to its already
        opened BaseCamera objects; get_frame returns raw views before env crop/flip.
        The observation config declares each camera's actual source color.
        """
        return cls(
            controller=env._controller,
            cameras=cameras,
            score_source=score_source,
            clip_position=env._clip_position_to_safety_box,
            gripper_wait_seconds=gripper_wait_seconds,
            camera_timeout_seconds=camera_timeout_seconds,
            calibration_id=calibration_id,
            clock=clock,
        )

    def begin_episode(self, episode_id, layout_id):
        """Only score-source bookkeeping; no reset, go_to_rest, or clear_errors."""
        self.score_source.begin_episode(episode_id, layout_id)

    def read_state(self):
        state = self._rpc(self.controller.get_state)[0]
        gripper = state.gripper_position
        if gripper is None:
            raise ValueError(
                "The PAD 7-D profile requires a measured calibrated gripper."
            )
        return {
            "tcp_pose": np.asarray(state.tcp_pose).copy(),
            "gripper_position": np.asarray(gripper).reshape(1).copy(),
        }

    def observe(self, **identity):
        images, received = {}, {}
        for name, camera in self.cameras.items():
            try:
                images[name] = np.asarray(
                    camera.get_frame(timeout=self.camera_timeout_seconds)
                ).copy()
            except Exception as error:
                raise ConnectionError(
                    f"Camera {name} failed; observation is unavailable."
                ) from error
            received[name] = self.clock.now()
        state = self.read_state()
        return ObservationSnapshot(
            **identity,
            images=images,
            state=state,
            capture_times=dict.fromkeys(images, None),
            receive_times=received,
            receive_time=self.clock.now(),
            clock_domain="robot_monotonic",
        )

    def send(self, command):
        """Submit gripper and bounded pose directly, retaining native wait latency."""
        changed = self._rpc(
            self.controller.command_end_effector, np.array([command.gripper_signal])
        )[0]
        if changed:
            self.clock.sleep(self.gripper_wait_seconds)
        self._rpc(self.controller.move_arm, command.target_pose)
        return True

    def limit_command(self, command):
        target = self.clip_position(command.target_pose.copy())
        limited = canonical_delta_from_target(
            target, command.anchor_pose, command.limited_action[-1]
        )
        return replace(command, target_pose=target, limited_action=limited)

    @staticmethod
    def _rpc(method, *args):
        try:
            return method(*args).wait()
        except Exception as error:
            raise ConnectionError(
                "Franka RPC failed; execution/state may be unknown."
            ) from error

    def poll_outcome(self, episode_id):
        return self.score_source.poll_outcome(episode_id)


class ManualEpisodeScores:
    """Minimal model-blind event surface for an operator UI or teleoperation client."""

    def __init__(self):
        self.episode_id = None
        self.outcome = None
        self._confirmed = threading.Event()

    def begin_episode(self, episode_id, layout_id):
        self.episode_id = episode_id
        self.outcome = None
        self._confirmed.clear()

    def submit(self, outcome):
        if outcome.episode_id != self.episode_id or self.outcome is not None:
            raise ValueError("Score must target the active episode exactly once.")
        self.outcome = deepcopy(outcome)
        self._confirmed.set()

    def poll_outcome(self, episode_id):
        if episode_id != self.episode_id:
            raise ValueError("Scoring episode changed.")
        return self.outcome

    def wait_for_outcome(self, episode_id):
        """Wait for the scoring UI after stop without issuing policy commands."""
        if episode_id != self.episode_id:
            raise ValueError("Scoring episode changed.")
        self._confirmed.wait()
        return self.poll_outcome(episode_id)
