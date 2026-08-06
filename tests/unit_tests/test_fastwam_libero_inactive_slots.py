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

from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.libero.action_contract import LiberoActionContract
from rlinf.envs.libero.libero_env import LiberoEnv


class _SubsetVectorEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, tuple[int, ...]]] = []

    def step(self, actions, ids=None):
        indices = tuple(int(index) for index in ids)
        values = np.asarray(actions).copy()
        self.calls.append((values, indices))
        observations = np.asarray(
            [{"token": 100 * len(self.calls) + index} for index in indices],
            dtype=object,
        )
        rewards = np.zeros(len(indices), dtype=np.float32)
        terminations = np.zeros(len(indices), dtype=bool)
        infos = np.asarray([{} for _ in indices], dtype=object)
        return observations, rewards, terminations, infos


def _libero_env() -> LiberoEnv:
    env = object.__new__(LiberoEnv)
    env.num_envs = 3
    env.env = _SubsetVectorEnv()
    env.current_raw_obs = [{"token": 0}, {"token": 1}, {"token": 2}]
    env._elapsed_steps = np.zeros(3, dtype=np.int32)
    env.prev_step_reward = np.zeros(3, dtype=np.float32)
    env.success_once = np.zeros(3, dtype=bool)
    env.fail_once = np.zeros(3, dtype=bool)
    env.returns = np.zeros(3, dtype=np.float32)
    env.success_episode_len = np.zeros(3, dtype=np.int32)
    env.cfg = OmegaConf.create(
        {
            "max_episode_steps": 1000,
            "reward_coef": 1.0,
        }
    )
    env.auto_reset = False
    env.ignore_terminations = False
    env.use_rel_reward = False
    env.use_step_penalty = False
    env._action_submission_capture = None
    env._wrap_obs = MethodType(
        lambda _self, observations: {
            "states": torch.tensor(
                [[observation["token"]] for observation in observations],
                dtype=torch.float32,
            ),
            "task_descriptions": ["task"] * 3,
        },
        env,
    )
    return env


def _contract() -> LiberoActionContract:
    return LiberoActionContract(
        low=(-1.0,) * 7,
        high=(1.0,) * 7,
        dimension_names=(
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_axis_angle_x",
            "delta_axis_angle_y",
            "delta_axis_angle_z",
            "gripper",
        ),
        gripper_dimension_index=6,
        outer_environment_classes=("unit.OffScreenRenderEnv",),
        underlying_environment_classes=("unit.LiberoTask",),
        robot_class="unit.SingleArm",
        robot_model="OnTheGroundPanda",
        controller_class="unit.OperationalSpaceController",
        controller_name="OSC_POSE",
        controller_input_low=(-1.0,) * 6,
        controller_input_high=(1.0,) * 6,
        controller_output_low=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        controller_output_high=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        gripper_class="unit.PandaGripper",
        gripper_dof=1,
        gripper_speed=0.01,
        control_frequency_hz=20,
        environment_horizon=1000,
        dependency_versions=(("robosuite_version", "1.4.0"),),
    )


def test_chunk_step_never_steps_inactive_ledger_slots_and_keeps_batch_shape() -> None:
    env = _libero_env()
    actions = np.zeros((3, 2, 7), dtype=np.float32)
    actions[1] = 9.0

    observations, rewards, terminations, truncations, _infos = env.chunk_step(
        actions,
        active_mask=np.asarray([True, False, True]),
    )

    assert [indices for _actions, indices in env.env.calls] == [(0, 2), (0, 2)]
    assert all(call_actions.shape == (2, 7) for call_actions, _ids in env.env.calls)
    assert len(observations) == 2
    assert observations[-1]["states"].shape == (3, 1)
    assert observations[-1]["states"][:, 0].tolist() == [200.0, 1.0, 202.0]
    assert rewards.shape == (3, 2)
    assert rewards[1].tolist() == [0.0, 0.0]
    assert not terminations[1].any()
    assert truncations[1].tolist() == [False, True]
    assert env.elapsed_steps.tolist() == [2, 0, 2]


def test_all_inactive_chunk_skips_every_subprocess_step_and_trace_has_no_values() -> (
    None
):
    env = _libero_env()
    actions = np.full((3, 2, 7), 4.0, dtype=np.float32)

    result, submitted = env.chunk_step_with_action_trace(
        actions,
        _contract(),
        active_mask=torch.zeros(3, dtype=torch.bool),
    )

    _observations, rewards, terminations, truncations, _infos = result
    assert env.env.calls == []
    assert rewards.eq(0).all()
    assert not terminations.any()
    assert truncations[:, -1].all()
    assert submitted.finite_count.eq(0).all()
    assert submitted.total_value_count.eq(0).all()
    assert submitted.below_low_count.eq(0).all()
    assert submitted.above_high_count.eq(0).all()


def test_traced_chunk_rejects_active_out_of_contract_action_before_env_step() -> None:
    env = _libero_env()
    actions = np.zeros((3, 2, 7), dtype=np.float32)
    actions[1, 0, 2] = -1.125

    with pytest.raises(
        ValueError,
        match=r"dimension_name': 'delta_z'.*No clamp was applied",
    ):
        env.chunk_step_with_action_trace(actions, _contract())

    assert env.env.calls == []


def test_traced_chunk_ignores_invalid_values_in_inactive_slots() -> None:
    env = _libero_env()
    actions = np.zeros((3, 2, 7), dtype=np.float32)
    actions[1] = np.nan

    _result, submitted = env.chunk_step_with_action_trace(
        actions,
        _contract(),
        active_mask=np.asarray([True, False, True]),
    )

    assert [indices for _actions, indices in env.env.calls] == [(0, 2), (0, 2)]
    assert submitted.total_value_count[1].eq(0).all()
    assert submitted.finite_count[1].eq(0).all()


@pytest.mark.parametrize(
    "bad_mask",
    (
        [True, False],
        [[True, False, True]],
        [1, 0, 1],
    ),
)
def test_inactive_slot_mask_fails_closed_when_malformed(bad_mask) -> None:
    env = _libero_env()

    with pytest.raises((TypeError, ValueError), match="active mask"):
        env.chunk_step(np.zeros((3, 2, 7), dtype=np.float32), active_mask=bad_mask)
