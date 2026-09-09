# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import importlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.pad_realworld.action_codec import (
    ActionCodec,
    canonical_delta_from_target,
)
from rlinf.envs.pad_realworld.contracts import (
    ActionProposal,
    CameraSpec,
    RealActionProtocol,
)
from rlinf.envs.pad_realworld.env import SynchronousRobotEnv
from rlinf.envs.pad_realworld.mock_backend import MockRobotBackend
from rlinf.envs.pad_realworld.observation import ObservationAdapter
from rlinf.models.embodiment.wam_policy.real_robot.config import load_config

ROOT = Path(__file__).resolve().parents[3]


def make_env(steps=(3,), outcomes=("success",)):
    cfg = load_config(ROOT / "configs/real_robot/pad_mock.yaml")
    backend = MockRobotBackend(episode_steps=steps, scripted_outcomes=outcomes)
    env = SynchronousRobotEnv(
        backend,
        ObservationAdapter.from_config(cfg["observation"]),
        ActionCodec(**cfg["limits"]),
        RealActionProtocol(),
        run_id="test",
        calibration_id="test",
        episode_timeout_seconds=120,
    )
    env.begin_episode(
        episode_id="a",
        task_id="test",
        instruction="move",
        layout_id="x",
        operator_ready=True,
    )
    return env


def proposal(env, action=None):
    actions = (
        np.zeros((32, 7)) if action is None else np.broadcast_to(action, (32, 7)).copy()
    )
    return ActionProposal("a-0", env.observe(0), 0, 1, actions.copy(), actions.copy())


def test_third_step_success_keeps_chunk_and_stops_remaining_seven():
    env = make_env()
    receipt = env.execute_prefix(proposal(env))
    assert len(env.backend.commands) == receipt.executed_count == 3
    assert receipt.chunk_valid
    assert receipt.executed_prefix_mask.tolist() == [True] * 3 + [False] * 7
    assert receipt.outcome.kind == "success"
    with pytest.raises(RuntimeError, match="twice"):
        env.execute_prefix(
            ActionProposal(
                "a-0",
                proposal_snapshot(receipt, env),
                0,
                1,
                np.zeros((32, 7)),
                np.zeros((32, 7)),
            )
        )


def proposal_snapshot(receipt, env):
    return env.backend.observe(**env.identity, chunk_id=0)


@pytest.mark.parametrize("outcome", ["intervention", "abort", "censored"])
def test_censored_and_intervened_chunks_are_not_ppo_samples(outcome):
    env = make_env(outcomes=(outcome,))
    receipt = env.execute_prefix(proposal(env))
    assert len(env.backend.commands) == 3
    assert not receipt.chunk_valid
    assert receipt.executed_count == (2 if outcome == "censored" else 3)
    assert receipt.steps[-1].state_after is None if outcome == "censored" else True


def test_stale_proposal_never_sends_a_command():
    env = make_env()
    action = proposal(env)
    env.backend.clock.sleep(3)
    receipt = env.execute_prefix(action)
    assert not env.backend.commands
    assert receipt.outcome.kind == "censored"
    assert "Stale" in receipt.outcome.detail


def test_stop_before_send_and_manual_scoring_are_separate():
    from rlinf.envs.pad_realworld.contracts import EpisodeOutcome

    env = make_env()
    action = proposal(env)
    env.request_stop("operator_stop")
    receipt = env.execute_prefix(action)
    assert receipt.executed_count == 0
    assert receipt.outcome is None
    env.confirm_outcome(EpisodeOutcome("a", "failure"))
    with pytest.raises(ValueError, match="exactly one"):
        env.confirm_outcome(EpisodeOutcome("a", "success"))
    assert env.finish_episode().kind == "failure"


def test_limits_record_commands_without_mutating_model_samples():
    env = make_env()
    action = proposal(env, np.array([2, 0, 0, 0, 0, 0, 1]))
    original = action.normalized_actions.copy()
    receipt = env.execute_prefix(action)
    assert receipt.steps[0].command.limited_action[0] == pytest.approx(0.02)
    np.testing.assert_array_equal(action.normalized_actions, original)
    assert receipt.steps[1].command.anchor_pose[0] == pytest.approx(0.516)


def test_named_state_quaternion_and_color_conversion():
    env = make_env()
    snapshot = env.observe(0)
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[..., 0] = 255
    snapshot = replace(snapshot, images={"main": image})
    obs = ObservationAdapter(
        (CameraSpec("main", "BGR", (4, 4), (0, 2, 8, 10)),), quaternion_order="wxyz"
    )
    state = {
        "tcp_force": np.ones(3) * 999,
        "gripper_position": np.array([0.25]),
        "tcp_pose": np.array([0.1, 0.2, 0.3, 2, 0, 0, 0]),
    }
    first = obs.canonical_state(state)
    second = obs.canonical_state(dict(reversed(list(state.items()))))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, [0.1, 0.2, 0.3, 0, 0, 0, 1, 0.25])
    np.testing.assert_array_equal(obs.images(snapshot)[0][0, 0], [0, 0, 255])
    with pytest.raises(KeyError, match="gripper_position"):
        obs.canonical_state({"tcp_pose": state["tcp_pose"]})


def test_rotation_is_left_euler_and_franka_gripper_direction():
    env = make_env()
    anchor = np.r_[[0.5, 0, 0.3], Rotation.from_euler("z", 0.7).as_quat()]
    action = np.array([0.01, -0.01, 0.01, 0.02, -0.03, 0.04, 1.0])
    command = env.codec.command(action, anchor)
    expected = Rotation.from_euler("xyz", action[3:6]) * Rotation.from_quat(anchor[3:])
    np.testing.assert_allclose(
        Rotation.from_quat(command.target_pose[3:]).as_matrix(), expected.as_matrix()
    )
    np.testing.assert_allclose(
        canonical_delta_from_target(command.target_pose, anchor, 1), action, atol=1e-15
    )
    scaled = env.codec.scaled_franka_action(command, [0.1, 0.2, 1])
    np.testing.assert_allclose(scaled[:3], action[:3] / 0.1)
    assert scaled[-1] == 1
    closed = env.codec.command(np.r_[np.zeros(6), 0.0], anchor)
    assert closed.gripper_signal == -1


def test_explicit_time_protocol_does_not_require_divisible_episodes():
    assert RealActionProtocol().execution_horizon == 10
    with pytest.raises(ValueError, match="execution"):
        RealActionProtocol(execution_horizon=33)
    env = make_env(steps=(13,), outcomes=("failure",))
    first = env.execute_prefix(proposal(env))
    assert first.executed_count == 10 and first.outcome is None


def test_import_attempts_no_process_control(monkeypatch):
    import subprocess

    import psutil

    def forbidden(*args, **kwargs):
        raise AssertionError("Import attempted process control")

    monkeypatch.setattr(psutil, "process_iter", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    sys.modules.pop("rlinf.envs.realworld", None)
    module = importlib.import_module("rlinf.envs.realworld")
    assert "FrankaEnv" in module.__all__
    assert "rlinf.envs.realworld.franka.franka_controller" not in sys.modules


def test_live_template_lists_missing_fields_without_allocating_assets():
    with pytest.raises(ValueError, match="Missing required configuration") as error:
        load_config(ROOT / "configs/real_robot/pad_live.example.yaml")
    assert "hardware.robot_ip" in str(error.value)
    assert "execution.command_hz" in str(error.value)
    assert "model.parent_checkpoint" in str(error.value)
