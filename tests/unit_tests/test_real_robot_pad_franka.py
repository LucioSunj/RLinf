# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""No-connection Franka worker/camera contract and existing registration tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.pad_realworld.action_codec import ActionCodec
from rlinf.envs.pad_realworld.contracts import (
    ActionProposal,
    EpisodeOutcome,
    RealActionProtocol,
)
from rlinf.envs.pad_realworld.env import SynchronousRobotEnv
from rlinf.envs.pad_realworld.franka_backend import (
    FrankaControllerBackend,
    ManualEpisodeScores,
)
from rlinf.envs.pad_realworld.mock_backend import FakeClock
from rlinf.envs.pad_realworld.observation import ObservationAdapter
from rlinf.models.embodiment.wam_policy.real_robot.config import load_config

ROOT = Path(__file__).resolve().parents[3]


class RPCResult:
    def __init__(self, value, clock):
        self.value, self.clock = value, clock

    def wait(self):
        self.clock.sleep(0.005)
        return [self.value]


class ControllerFixture:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.state = SimpleNamespace(
            tcp_pose=np.array([0.5, 0, 0.3, 0, 0, 0, 1.0]), gripper_position=0.08
        )

    def get_state(self):
        self.calls.append("get_state")
        return RPCResult(self.state, self.clock)

    def command_end_effector(self, action):
        self.calls.append(("gripper", float(action[0])))
        changed = action[0] <= -0.5 and self.state.gripper_position > 0
        if changed:
            self.state.gripper_position = 0.0
        return RPCResult(changed, self.clock)

    def move_arm(self, position):
        self.calls.append(("move_arm", position.copy()))
        self.state.tcp_pose = position.copy()
        return RPCResult(None, self.clock)

    def clear_errors(self):
        pytest.fail("No implicit error recovery is authorized.")


def test_explicit_native_binding_has_no_startup_motion_and_preserves_wait_and_limits():
    cfg = load_config(ROOT / "configs/real_robot/pad_mock.yaml")
    cfg["observation"]["state"]["gripper_open"] = 0.08
    clock = FakeClock()
    controller = ControllerFixture(clock)
    camera = SimpleNamespace(
        get_frame=lambda timeout: np.full((32, 32, 3), [255, 0, 0], np.uint8)
    )
    scores = ManualEpisodeScores()

    def native_clip(position):
        position[0] = min(position[0], 0.505)
        return position

    native_env = SimpleNamespace(
        _controller=controller, _clip_position_to_safety_box=native_clip
    )
    backend = FrankaControllerBackend.from_connected_env(
        native_env,
        cameras={"main": camera},
        score_source=scores,
        gripper_wait_seconds=0.6,
        camera_timeout_seconds=0.1,
        calibration_id="fixture",
        clock=clock,
    )
    assert controller.calls == []
    assert not backend.rpc_cancel_supported
    env = SynchronousRobotEnv(
        backend,
        ObservationAdapter.from_config(cfg["observation"]),
        ActionCodec(**cfg["limits"]),
        RealActionProtocol(),
        run_id="fake_rpc",
        calibration_id="fixture",
        allow_motion=True,
    )
    env.begin_episode(
        episode_id="one",
        task_id="task",
        instruction="Move.",
        layout_id="layout",
        operator_ready=True,
    )
    assert controller.calls == []
    snapshot = env.observe(0)
    np.testing.assert_array_equal(snapshot.images["main"][0, 0], [255, 0, 0])
    assert snapshot.capture_times["main"] is None
    command = env.codec.command(
        np.array([0.02, 0, 0, 0, 0, 0, 0]), snapshot.state["tcp_pose"]
    )
    final = backend.limit_command(command)
    assert final.target_pose[0] == 0.505
    assert final.limited_action[0] == pytest.approx(0.005)
    started = clock.now()
    backend.send(final)
    assert clock.now() - started == pytest.approx(0.61)
    assert ("gripper", -1.0) in [
        item
        for item in controller.calls
        if isinstance(item, tuple) and item[0] == "gripper"
    ]
    env.request_stop("operator")
    scores.submit(EpisodeOutcome("one", "failure"))
    env.confirm_outcome(scores.poll_outcome("one"))
    before = len(controller.calls)
    proposal = ActionProposal(
        "one:0", snapshot, 0, 1, np.zeros((32, 7)), np.zeros((32, 7))
    )
    assert env.execute_prefix(proposal).executed_count == 0
    assert len(controller.calls) == before
    with pytest.raises(ValueError, match="exactly once"):
        scores.submit(EpisodeOutcome("one", "success"))

    # Native queue timeout must propagate once, without camera reopening or RPC.
    import queue

    timeouts = []

    def no_frame(timeout):
        timeouts.append(timeout)
        raise queue.Empty

    camera.get_frame = no_frame
    before = len(controller.calls)
    with pytest.raises(ConnectionError, match="Camera main failed"):
        backend.observe()
    assert timeouts == [0.1]
    assert len(controller.calls) == before


def test_existing_realworld_exports_and_explicit_task_registration(monkeypatch):
    import subprocess

    import gymnasium as gym
    import psutil

    import rlinf.envs.realworld as module

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Import/registration attempted process control or hardware startup."
        )

    monkeypatch.setattr(psutil, "process_iter", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    module.register_realworld_tasks()
    assert "FrankaEnv-v1" in gym.registry
    assert "DualFrankaJointEnv-v1" in gym.registry
    assert module.FrankaEnv.__name__ == "FrankaEnv"
    assert module.RealWorldEnv.__name__ == "RealWorldEnv"
    module.RealWorldEnv.realworld_setup()
