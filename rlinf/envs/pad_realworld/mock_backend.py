# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Traceable CPU robot dynamics and deterministic terminal-event fixtures."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .contracts import EpisodeOutcome, ObservationSnapshot, RobotCommand


class FakeClock:
    """Advance modeled command time without blocking the test process."""

    def __init__(self, start: float = 0.0) -> None:
        self.time = start

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.time += max(0.0, seconds)


class MockRobotBackend:
    """Metric pose dynamics; default scoring depends on the attained position.

    Explicit outcomes can be injected to test third-step success, interruption
    and uncertain execution. They are fixtures, not learned-policy scores.
    """

    is_mock = True

    def __init__(
        self,
        *,
        camera_names=("main",),
        episode_steps=(13, 23),
        scripted_outcomes: tuple[str | None, ...] | None = None,
        target_x: float = 0.48,
        clock: FakeClock | None = None,
        gripper_wait_seconds: float = 0.6,
    ) -> None:
        if not episode_steps or min(episode_steps) <= 0:
            raise ValueError("Mock attempts need positive step limits.")
        self.clock = clock or FakeClock()
        self.camera_names = tuple(camera_names)
        self.episode_steps = tuple(episode_steps)
        self.scripted_outcomes = scripted_outcomes
        self.target_x = target_x
        self.gripper_wait_seconds = gripper_wait_seconds
        self.commands: list[RobotCommand] = []
        self.episode_index = -1
        self.steps = 0
        self.observe_calls = 0
        self.state = self._initial_state()

    @staticmethod
    def _initial_state() -> dict[str, np.ndarray]:
        return {
            "tcp_pose": np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
            "gripper_position": np.array([1.0]),
            "tcp_force": np.zeros(3),
        }

    def begin_episode(self, episode_id: str, layout_id: str) -> None:
        """Simulate the operator's completed reset, only for this mock object."""
        self.episode_index += 1
        self.steps = 0
        self.state = self._initial_state()

    def read_state(self) -> dict[str, np.ndarray]:
        return deepcopy(self.state)

    def observe(self, **identity) -> ObservationSnapshot:
        self.observe_calls += 1
        # Images are state-dependent; they are not a constant successful fixture.
        level = int(np.clip(self.state["tcp_pose"][0] * 255, 0, 255))
        images = {}
        for index, name in enumerate(self.camera_names):
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[..., 0] = level
            image[..., 1] = (self.steps * 7 + index * 61) % 256
            image[..., 2] = np.arange(32, dtype=np.uint8)[None, :] * 8
            images[name] = image
        now = self.clock.now()
        return ObservationSnapshot(
            **identity,
            images=images,
            state=self.read_state(),
            capture_times=dict.fromkeys(self.camera_names, now),
            receive_times=dict.fromkeys(self.camera_names, now),
            receive_time=now,
            clock_domain="robot_monotonic",
        )

    def send(self, command: RobotCommand) -> bool:
        self.commands.append(deepcopy(command))
        self.steps += 1
        limit = self.episode_steps[self.episode_index % len(self.episode_steps)]
        scripted = self._scripted()
        if scripted == "censored" and self.steps == limit:
            return False
        # A lagging plant makes commanded targets distinct from measured deltas.
        self.state["tcp_pose"][:3] += 0.8 * (
            command.target_pose[:3] - self.state["tcp_pose"][:3]
        )
        self.state["tcp_pose"][3:] = command.target_pose[3:]
        opened = float(self.state["gripper_position"][0])
        if command.gripper_signal >= 0.5:
            opened = 1.0
        elif command.gripper_signal <= -0.5:
            opened = 0.0
        if opened != self.state["gripper_position"][0]:
            self.clock.sleep(self.gripper_wait_seconds)
        self.state["gripper_position"][0] = opened
        self.clock.sleep(0.005)
        return True

    def limit_command(self, command: RobotCommand) -> RobotCommand:
        return command

    def _scripted(self) -> str | None:
        return (
            None
            if self.scripted_outcomes is None
            else self.scripted_outcomes[
                self.episode_index % len(self.scripted_outcomes)
            ]
        )

    def poll_outcome(self, episode_id: str) -> EpisodeOutcome | None:
        limit = self.episode_steps[self.episode_index % len(self.episode_steps)]
        if self.steps < limit:
            return None
        kind = self._scripted()
        if kind is None:
            kind = (
                "success" if self.state["tcp_pose"][0] >= self.target_x else "failure"
            )
        stage = float(np.clip((self.state["tcp_pose"][0] - 0.3) / 0.3, 0, 1))
        return EpisodeOutcome(episode_id, kind, stage_score=stage)
