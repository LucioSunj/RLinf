# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Physical observations, commands and receipts, independent of the base policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    """One ordered source view; crop is [top, left, bottom, right] in pixels."""

    name: str
    source_color: str
    resize: tuple[int, int]
    crop: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.source_color not in {"RGB", "BGR"}:
            raise ValueError("Camera source_color must be RGB or BGR.")
        if len(self.resize) != 2 or min(self.resize) <= 0:
            raise ValueError("Camera resize must specify positive height and width.")
        if self.crop is not None:
            top, left, bottom, right = self.crop
            if min(top, left) < 0 or bottom <= top or right <= left:
                raise ValueError("Camera crop must be a nonempty pixel rectangle.")


@dataclass(frozen=True)
class RealActionProtocol:
    """Physical sampling grid without simulator reset or episode divisibility."""

    generation_horizon: int = 32
    execution_horizon: int = 10
    prediction_video_frames: int = 9
    sample_period: float = 0.05
    video_offsets: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28, 32)

    def __post_init__(self) -> None:
        if not 0 < self.execution_horizon <= self.generation_horizon:
            raise ValueError("Action protocol requires 0 < execution <= generation.")
        if not np.isfinite(self.sample_period) or self.sample_period <= 0:
            raise ValueError("Action sample_period must be finite and positive.")
        if (
            len(self.video_offsets) != self.prediction_video_frames
            or self.video_offsets[0] != 0
            or any(b <= a for a, b in zip(self.video_offsets, self.video_offsets[1:]))
        ):
            raise ValueError("Video offsets must be ordered and start at zero.")


@dataclass(frozen=True)
class ObservationSnapshot:
    """Saved sensor data; a missing exposure timestamp remains explicitly unknown."""

    run_id: str
    episode_id: str
    chunk_id: int
    task_id: str
    instruction: str
    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    capture_times: dict[str, float | None]
    receive_times: dict[str, float]
    receive_time: float
    clock_domain: str
    layout_id: str
    calibration_id: str


@dataclass(frozen=True)
class ActionProposal:
    """Unmodified model sample and physical actions before common execution limits."""

    proposal_id: str
    snapshot: ObservationSnapshot
    actor_version: int
    route: int
    normalized_actions: np.ndarray
    canonical_actions: np.ndarray
    replay: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    noise: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotCommand:
    """One command in meters/radians, anchored to a fresh measured TCP pose."""

    requested_action: np.ndarray
    limited_action: np.ndarray
    anchor_pose: np.ndarray
    target_pose: np.ndarray
    gripper_signal: float


@dataclass(frozen=True)
class EpisodeOutcome:
    """One terminal score bound to one attempt; stages never feed the policy."""

    episode_id: str
    kind: str
    stage_score: float = 0.0
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {
            "success",
            "failure",
            "timeout",
            "abort",
            "intervention",
            "censored",
        }:
            raise ValueError(f"Unknown episode outcome {self.kind!r}.")

    @property
    def trainable(self) -> bool:
        return self.kind in {"success", "failure", "timeout"}


@dataclass(frozen=True)
class StepReceipt:
    """A request and its acknowledgement, including uncertain execution."""

    index: int
    command: RobotCommand
    send_time: float
    acknowledge_time: float
    acknowledged: bool
    state_after: dict[str, np.ndarray] | None


@dataclass(frozen=True)
class ExecutionReceipt:
    """Physical execution mask is separate from the chunk's learning validity."""

    proposal_id: str
    steps: tuple[StepReceipt, ...]
    execution_horizon: int
    outcome: EpisodeOutcome | None
    stopped_reason: str | None
    start_time: float
    end_time: float
    final_state: dict[str, np.ndarray] | None

    @property
    def executed_count(self) -> int:
        return sum(step.acknowledged for step in self.steps)

    @property
    def executed_prefix_mask(self) -> np.ndarray:
        mask = np.zeros(self.execution_horizon, dtype=bool)
        for step in self.steps:
            mask[step.index] = step.acknowledged
        return mask

    @property
    def chunk_valid(self) -> bool:
        return (
            self.executed_count > 0
            and all(step.acknowledged for step in self.steps)
            and (self.outcome is None or self.outcome.trainable)
            and self.stopped_reason in {None, "success", "failure", "timeout"}
        )


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class RobotBackend(Protocol):
    """The reusable physical boundary; no method implies reset or recovery."""

    clock: Clock
    is_mock: bool

    def read_state(self) -> dict[str, np.ndarray]: ...

    def observe(self, **identity: Any) -> ObservationSnapshot: ...

    def send(self, command: RobotCommand) -> bool: ...

    def limit_command(self, command: RobotCommand) -> RobotCommand: ...

    def poll_outcome(self, episode_id: str) -> EpisodeOutcome | None: ...

    def begin_episode(self, episode_id: str, layout_id: str) -> None: ...


class ChunkPolicy(Protocol):
    """Collector interface for PAD or another base policy's observation adapter."""

    actor_version: int

    def begin_episode(self) -> None: ...

    def propose(
        self, snapshot: ObservationSnapshot, **kwargs: Any
    ) -> ActionProposal: ...
