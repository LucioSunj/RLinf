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

"""Focused parity tests for the pinned FastWAM LIBERO execution protocol."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.libero.action_protocol import (
    LiberoActionProtocol,
    select_executed_action_prefix,
    select_executed_flow_statistics,
)

ROOT = Path(__file__).resolve().parents[2]


def test_official_fastwam_libero_protocol_is_explicit_in_hydra() -> None:
    model = OmegaConf.load(
        ROOT / "examples/embodiment/config/model/fastwam_adaptive.yaml"
    )
    env = OmegaConf.load(ROOT / "examples/embodiment/config/env/libero_10.yaml")

    assert model.num_action_chunks == 10
    assert model.runtime.generation_horizon == 32
    assert model.runtime.execution_horizon == 10
    assert model.runtime.num_video_frames == 9
    assert env.reset_wait_steps == 30
    assert env.max_episode_steps == 700
    assert env.max_steps_per_rollout_epoch == 700


def test_execution_selects_exact_prefix_without_changing_generation() -> None:
    actions = torch.arange(2 * 32 * 7).reshape(2, 32, 7)
    protocol = LiberoActionProtocol(
        generation_horizon=32,
        execution_horizon=10,
        prediction_video_frames=9,
        reset_wait_steps=30,
        max_episode_steps=700,
    )

    selected = select_executed_action_prefix(actions, protocol=protocol)

    assert selected.shape == (2, 10, 7)
    assert torch.equal(selected, actions[:, :10])
    assert actions.shape == (2, 32, 7)


def test_disabled_eval_flow_statistics_remain_empty() -> None:
    protocol = LiberoActionProtocol(
        generation_horizon=32,
        execution_horizon=10,
        prediction_video_frames=9,
        reset_wait_steps=30,
        max_episode_steps=700,
    )
    empty = torch.empty(2, 0, dtype=torch.float32)

    selected = select_executed_flow_statistics(empty, protocol=protocol)

    assert selected is empty
    with pytest.raises(ValueError, match="31 != 32"):
        select_executed_flow_statistics(
            torch.zeros(2, 31, 7),
            protocol=protocol,
        )


@pytest.mark.parametrize(
    ("generation", "execution", "video_frames"),
    [(0, 10, 9), (32, 0, 9), (10, 11, 9), (32, 10, 0)],
)
def test_protocol_fails_closed_on_malformed_horizons(
    generation: int,
    execution: int,
    video_frames: int,
) -> None:
    with pytest.raises(ValueError):
        LiberoActionProtocol(
            generation_horizon=generation,
            execution_horizon=execution,
            prediction_video_frames=video_frames,
            reset_wait_steps=30,
            max_episode_steps=700,
        )


def test_prefix_selection_rejects_a_mismatched_generated_horizon() -> None:
    protocol = LiberoActionProtocol(
        generation_horizon=32,
        execution_horizon=10,
        prediction_video_frames=9,
        reset_wait_steps=30,
        max_episode_steps=700,
    )

    with pytest.raises(ValueError, match="generated horizon"):
        select_executed_action_prefix(
            torch.zeros(1, 16, 7),
            protocol=protocol,
        )
