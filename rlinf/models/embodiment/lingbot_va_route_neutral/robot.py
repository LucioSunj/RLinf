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

"""Absolute TCP actions, feedback and immediate termination for one Franka."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic, sleep, time

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import canonical_quaternions


def absolute_to_delta(
    target, measured_pose, scales, *, gripper_open=0.08, gripper_closed=0.0
):
    """Franka applies left-multiplied XYZ Euler increments and binary grip."""
    target = canonical_quaternions(np.asarray(target)[None])[0]
    current = canonical_quaternions(np.asarray(measured_pose)[None])[0]
    scales = np.asarray(scales)
    delta = np.zeros(7, dtype=np.float32)
    delta[:3] = (target[:3] - current[:3]) / scales[0]
    delta[3:6] = (
        Rotation.from_quat(target[3:7]) * Rotation.from_quat(current[3:7]).inv()
    ).as_euler("xyz") / scales[1]
    opened = target[7] >= (gripper_open + gripper_closed) / 2
    delta[6] = (1.0 if opened else -1.0) / scales[2]
    submitted = target.copy()
    submitted[7] = gripper_open if opened else gripper_closed
    return delta, submitted


@dataclass
class ExecutionFeedback:
    observed_frames: list = field(default_factory=list)
    executed_actions: list = field(default_factory=list)
    submitted_commands: list = field(default_factory=list)
    action_timestamps: list = field(default_factory=list)
    observation_timestamps: list = field(default_factory=list)
    terminated: bool = False
    success: bool = False
    autonomous: bool = True
    reason: str = "prefix_complete"
    stage_progress: float = 0.0

    def wire(self):
        return {
            "observed_frames": self.observed_frames,
            "executed_actions": np.asarray(
                self.executed_actions, dtype=np.float32
            ).reshape(-1, 8),
            "terminated": self.terminated,
            "success": self.success,
            "autonomous": self.autonomous,
            "reason": self.reason,
            "stage_progress": self.stage_progress,
            "submitted_commands": np.asarray(
                self.submitted_commands, dtype=np.float32
            ).reshape(-1, 7),
            "action_timestamps": self.action_timestamps,
            "observation_timestamps": self.observation_timestamps,
        }


class ChunkExecutor:
    """No queued command survives terminal success, takeover or an exception."""

    def __init__(self, driver, *, hz=20):
        self.driver, self.hz = driver, hz

    def execute(self, actions, *, remaining_steps, stop_status=lambda: {}):
        if len(actions) != 8:
            raise ValueError(
                "The policy must provide the eight-action executable prefix."
            )
        result = ExecutionFeedback()
        deadline = monotonic()
        for target in np.asarray(actions)[:remaining_steps]:
            status = stop_status()
            if status.get("success") or status.get("failure") or status.get("takeover"):
                result.terminated = True
                result.success = bool(status.get("success"))
                result.autonomous = not status.get("takeover", False)
                result.reason = (
                    "takeover"
                    if not result.autonomous
                    else ("success" if result.success else "failure")
                )
                result.stage_progress = float(status.get("stage_progress", 0))
                break
            try:
                started = time()
                observation, submitted, command, info = self.driver.submit(target)
                result.executed_actions.append(submitted)
                result.submitted_commands.append(command)
                result.action_timestamps.append(started)
                result.observation_timestamps.append(observation["timestamp"])
                result.observed_frames.append(observation)
                status = {**info, **stop_status()}
                result.stage_progress = float(
                    status.get("stage_progress", result.stage_progress)
                )
                result.autonomous = result.autonomous and not status.get(
                    "takeover", False
                )
                result.success = bool(status.get("success", False))
                result.terminated = (
                    result.success
                    or bool(status.get("failure"))
                    or not result.autonomous
                )
                if result.terminated:
                    result.reason = (
                        "takeover"
                        if not result.autonomous
                        else ("success" if result.success else "failure")
                    )
                    break
            except Exception as exc:
                # A driver exception can follow a partly applied command. Mark the
                # entire current decision invalid and never invent its feedback.
                result.terminated, result.autonomous = True, False
                result.reason = f"{type(exc).__name__}: {exc}"
                break
            deadline += 1 / self.hz
            sleep(max(0, deadline - monotonic()))
        if not result.terminated and len(result.executed_actions) >= remaining_steps:
            result.terminated, result.reason = True, "step_limit"
        return result


class FrankaDriver:
    """Reuse an initialized RLinf FrankaEnv and its workspace/controller limits."""

    def __init__(
        self,
        env,
        camera_names,
        *,
        frame_color="BGR",
        gripper_open=0.08,
        gripper_closed=0.0,
    ):
        if frame_color not in {"BGR", "RGB"} or len(camera_names) != 2:
            raise ValueError("Specify the two camera names and their RGB/BGR format.")
        if env.config.step_frequency != 20:
            raise ValueError("Franka's controller step_frequency must be 20 Hz.")
        self.env, self.camera_names, self.frame_color = env, camera_names, frame_color
        self.gripper_open, self.gripper_closed = gripper_open, gripper_closed
        self.previous_pose, self.previous_action = None, None

    def _observation(self, raw):
        measured = self.env._franka_state
        pose = canonical_quaternions(
            np.asarray(measured.tcp_pose)[None], self.previous_pose
        )[0]
        self.previous_pose = pose[3:7].copy()
        state = np.concatenate(
            (measured.arm_joint_position, pose, [measured.gripper_position])
        ).astype(np.float32)
        images = {}
        for key, name in zip(("external", "wrist"), self.camera_names):
            value = np.asarray(raw["frames"][name])
            images[key] = (
                value[..., ::-1] if self.frame_color == "BGR" else value
            ).copy()
        return {"state": state, "images": images, "timestamp": time()}

    def observe(self):
        self.env.get_tcp_pose()  # refresh measured joints and TCP from the controller
        return self._observation(self.env._get_observation())

    def reset(self):
        self.previous_pose, self.previous_action = None, None
        raw, _ = self.env.reset()
        return self._observation(raw)

    def submit(self, target):
        measured = self.env.get_tcp_pose()
        command, submitted = absolute_to_delta(
            target,
            measured,
            self.env.get_action_scale(),
            gripper_open=self.gripper_open,
            gripper_closed=self.gripper_closed,
        )
        bounded = np.clip(
            command, self.env.action_space.low, self.env.action_space.high
        )
        safe_pose = self.env._clip_position_to_safety_box(submitted[:7].copy())
        rotation_error = (
            Rotation.from_quat(safe_pose[3:]) * Rotation.from_quat(submitted[3:7]).inv()
        ).magnitude()
        if (
            not np.allclose(bounded, command, atol=1e-6, rtol=0)
            or not np.allclose(safe_pose[:3], submitted[:3], atol=1e-6, rtol=0)
            or rotation_error > 1e-6
        ):
            raise ValueError(
                "Target exceeds the existing Franka action/workspace limits."
            )
        raw, _reward, terminated, truncated, info = self.env.step(command)
        submitted = canonical_quaternions(submitted[None], self.previous_action)[0]
        self.previous_action = submitted[3:7].copy()
        # Human labels define success; the environment's shaped reward is ignored.
        info = {**info, "failure": bool(terminated or truncated)}
        return self._observation(raw), submitted, command, info


def build_franka_driver(config):
    """Construct the existing single-arm driver on the robot controller machine."""
    from omegaconf import OmegaConf

    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    from rlinf.envs.realworld.franka.franka_env import FrankaEnv
    from rlinf.scheduler import Cluster, FrankaHWInfo
    from rlinf.scheduler.hardware.robots.franka import FrankaConfig

    Cluster(cluster_cfg=config["cluster"])
    hardware = FrankaHWInfo(
        type="Franka", model="Franka", config=FrankaConfig(**dict(config["hardware"]))
    )

    class ManualFrankaEnv(FrankaEnv):
        def _calc_step_reward(self, observation, is_gripper_action_effective=False):
            return 0.0

    env = ManualFrankaEnv(
        dict(config["environment"]), worker_info=None, hardware_info=hardware, env_idx=0
    )
    return FrankaDriver(
        env,
        tuple(config["camera_names"]),
        frame_color=config.get("frame_color", "BGR"),
        gripper_open=config.get("gripper_open", 0.08),
        gripper_closed=config.get("gripper_closed", 0.0),
    )
