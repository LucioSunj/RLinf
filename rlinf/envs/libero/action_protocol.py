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

"""Explicit temporal protocol for FastWAM evaluation and training in LIBERO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

TensorLike = TypeVar("TensorLike")


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"LIBERO {name} must be a positive integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class LiberoActionProtocol:
    """Pinned FastWAM generation, replanning, reset, and episode horizons."""

    generation_horizon: int
    execution_horizon: int
    prediction_video_frames: int
    reset_wait_steps: int
    max_episode_steps: int

    def __post_init__(self) -> None:
        generation = _positive_integer(
            self.generation_horizon,
            name="generation_horizon",
        )
        execution = _positive_integer(
            self.execution_horizon,
            name="execution_horizon",
        )
        prediction_frames = _positive_integer(
            self.prediction_video_frames,
            name="prediction_video_frames",
        )
        if (
            isinstance(self.reset_wait_steps, bool)
            or int(self.reset_wait_steps) != (self.reset_wait_steps)
            or int(self.reset_wait_steps) < 0
        ):
            raise ValueError("LIBERO reset_wait_steps must be a non-negative integer.")
        max_steps = _positive_integer(
            self.max_episode_steps,
            name="max_episode_steps",
        )
        if execution > generation:
            raise ValueError(
                "LIBERO execution_horizon cannot exceed generation_horizon."
            )
        if max_steps % execution:
            raise ValueError(
                "LIBERO max_episode_steps must be divisible by execution_horizon."
            )
        object.__setattr__(self, "generation_horizon", generation)
        object.__setattr__(self, "execution_horizon", execution)
        object.__setattr__(self, "prediction_video_frames", prediction_frames)
        object.__setattr__(self, "reset_wait_steps", int(self.reset_wait_steps))
        object.__setattr__(self, "max_episode_steps", max_steps)


def select_executed_action_prefix(
    values: TensorLike,
    *,
    protocol: LiberoActionProtocol,
) -> TensorLike:
    """Select exactly the official replan prefix without clipping or mutation."""

    shape = getattr(values, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("FastWAM generated Actions must have shape [B, H, ...].")
    if int(shape[1]) != protocol.generation_horizon:
        raise ValueError(
            "FastWAM generated horizon differs from the declared LIBERO protocol: "
            f"{int(shape[1])} != {protocol.generation_horizon}."
        )
    return values[:, : protocol.execution_horizon]


def select_executed_flow_statistics(
    values: TensorLike,
    *,
    protocol: LiberoActionProtocol,
) -> TensorLike:
    """Select the executed prefix while preserving disabled eval replay state."""

    shape = getattr(values, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("FastWAM Flow statistics must have shape [B, H, ...].")
    if int(shape[1]) == 0:
        return values
    return select_executed_action_prefix(values, protocol=protocol)
