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

"""Runner-level coverage for FastWAM actor and rollout checkpoint pairing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.runners.fastwam_fair_cost import FastWAMFairCostController
from rlinf.runners.fastwam_training_guard import FastWAMTrainingGuard


class _Handle:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self.value


class _WorkerGroup:
    def __init__(self, *, loaded_steps=None):
        self.loaded_steps = loaded_steps
        self.calls = []

    def init_worker(self):
        self.calls.append(("init_worker",))
        return _Handle()

    def load_checkpoint(self, path):
        self.calls.append(("load_checkpoint", path))
        return _Handle(self.loaded_steps)

    def save_checkpoint(self, path, step):
        self.calls.append(("save_checkpoint", path, step))
        return _Handle()

    def set_fastwam_idm_cost(self, idm_cost, runner_step):
        self.calls.append(("set_fastwam_idm_cost", idm_cost, runner_step))
        return _Handle()


def _runner_cfg(
    tmp_path: Path,
    *,
    resume_dir: str | None = None,
    guard_enabled: bool = False,
    fair_cost_enabled: bool = False,
):
    return OmegaConf.create(
        {
            "actor": {"model": {"model_type": "fastwam_adaptive"}},
            "algorithm": {
                "fixed_branch_cost": {
                    "enabled": True,
                    "idm_cost": 0.01,
                    "uncond_cost": 0.0,
                    "fair_cost": {
                        "enabled": fair_cost_enabled,
                        "window_size": 5,
                        "pi": {"enabled": False},
                    },
                }
            },
            "runner": {
                "resume_dir": resume_dir,
                "fastwam_training_guard": {
                    "enabled": guard_enabled,
                    "zero_success_patience": 3,
                    "window_size": 3,
                    "eligible_idm_fraction_min": 0.05,
                    "eligible_idm_fraction_max": 0.95,
                    "gate_entropy_min": 0.35,
                    "gate_kl_median_max": 0.05,
                    "gate_kl_single_max": 0.1,
                    "gate_clip_median_max": 0.6,
                    "gate_clip_single_max": 0.8,
                },
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "l12",
                },
            },
        }
    )


def _bare_runner(cfg, *, actor, rollout, env=None):
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = cfg
    runner.actor = actor
    runner.rollout = rollout
    runner.env = env or _WorkerGroup()
    runner.reward = None
    runner.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    runner.global_step = 0
    runner.fastwam_training_guard = FastWAMTrainingGuard(
        cfg.runner.fastwam_training_guard
    )
    runner.fastwam_fair_cost_controller = (
        FastWAMFairCostController.from_branch_cost_config(
            cfg.algorithm.fixed_branch_cost
        )
    )
    return runner


def test_fastwam_save_pairs_actor_and_rollout_checkpoints(tmp_path: Path) -> None:
    actor = _WorkerGroup()
    rollout = _WorkerGroup()
    runner = _bare_runner(_runner_cfg(tmp_path), actor=actor, rollout=rollout)
    runner.global_step = 3

    runner._save_checkpoint()

    checkpoint = tmp_path / "l12/checkpoints/global_step_3"
    assert actor.calls == [("save_checkpoint", str(checkpoint / "actor"), 3)]
    assert rollout.calls == [("save_checkpoint", str(checkpoint / "rollout"), 3)]
    assert (checkpoint / "actor").is_dir()
    assert (checkpoint / "rollout").is_dir()


def test_worker_global_step_is_awaited_before_training_continues(
    tmp_path: Path,
) -> None:
    events = []

    class _StepHandle:
        def __init__(self, owner: str) -> None:
            self.owner = owner

        def wait(self):
            events.append(("wait", self.owner))

    class _StepWorkerGroup(_WorkerGroup):
        def __init__(self, owner: str) -> None:
            super().__init__()
            self.owner = owner

        def set_global_step(self, step: int):
            events.append(("dispatch", self.owner, step))
            return _StepHandle(self.owner)

    runner = _bare_runner(
        _runner_cfg(tmp_path),
        actor=_StepWorkerGroup("actor"),
        rollout=_StepWorkerGroup("rollout"),
    )
    runner.global_step = 7

    runner._set_worker_global_step()

    assert events == [
        ("dispatch", "actor", 7),
        ("dispatch", "rollout", 7),
        ("wait", "actor"),
        ("wait", "rollout"),
    ]


def test_lagged_fair_cost_is_published_after_global_step(tmp_path: Path) -> None:
    actor = _WorkerGroup()
    runner = _bare_runner(
        _runner_cfg(
            tmp_path,
            guard_enabled=True,
            fair_cost_enabled=True,
        ),
        actor=actor,
        rollout=_WorkerGroup(),
    )
    runner.global_step = 0

    runner._set_fastwam_runtime_cost()

    assert actor.calls == [("set_fastwam_idm_cost", 0.01, 0)]


def test_fastwam_resume_restores_paired_actor_and_rollout_steps(tmp_path: Path) -> None:
    checkpoint = tmp_path / "global_step_7"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "rollout").mkdir()
    actor = _WorkerGroup(loaded_steps=[7, 7])
    rollout = _WorkerGroup(loaded_steps=[7, 7])
    runner = _bare_runner(
        _runner_cfg(tmp_path, resume_dir=str(checkpoint)),
        actor=actor,
        rollout=rollout,
    )

    runner.init_workers()

    assert runner.global_step == 7
    assert actor.calls[-1] == ("load_checkpoint", str(checkpoint / "actor"))
    assert rollout.calls[-1] == ("load_checkpoint", str(checkpoint / "rollout"))


def test_fastwam_resume_rejects_missing_rollout_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "global_step_7"
    (checkpoint / "actor").mkdir(parents=True)
    runner = _bare_runner(
        _runner_cfg(tmp_path, resume_dir=str(checkpoint)),
        actor=_WorkerGroup(loaded_steps=[7, 7]),
        rollout=_WorkerGroup(loaded_steps=[7, 7]),
    )

    with pytest.raises(FileNotFoundError, match="requires rollout runtime checkpoints"):
        runner.init_workers()


def test_fastwam_resume_rejects_missing_actor_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "global_step_7"
    checkpoint.mkdir()
    runner = _bare_runner(
        _runner_cfg(tmp_path, resume_dir=str(checkpoint)),
        actor=_WorkerGroup(loaded_steps=[7, 7]),
        rollout=_WorkerGroup(loaded_steps=[7, 7]),
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        runner.init_workers()


def test_fastwam_guard_state_is_saved_and_restored_with_runner_step(
    tmp_path: Path,
) -> None:
    cfg = _runner_cfg(tmp_path, guard_enabled=True)
    source = _bare_runner(cfg, actor=_WorkerGroup(), rollout=_WorkerGroup())
    source.fastwam_training_guard.observe_rollout(
        [
            {
                "fastwam/raw_positive_success_signal_count": 0.0,
                "fastwam/successful_trajectory_count": 0.0,
                "fastwam/eligible_idm_fraction": 0.5,
                "fastwam/eligible_gate_decision_count": 10.0,
                "fastwam/eligible_idm_decision_count": 5.0,
                "fastwam/valid_uncond_chunk_count": 4.0,
                "rewards": 0.0,
                "advantages_max": 1.0,
                "advantages_mean": 0.0,
                "advantages_min": -1.0,
                "returns_max": 1.0,
                "returns_mean": 0.5,
                "returns_min": 0.0,
                "values_max": 0.5,
                "values_mean": 0.25,
                "values_min": 0.0,
            }
        ]
    )
    source.global_step = 3
    source._save_checkpoint()

    checkpoint = tmp_path / "l12/checkpoints/global_step_3"
    assert (checkpoint / "training_guard.json").is_file()
    assert (
        json.loads((checkpoint / "training_guard.json").read_text())["schema"]
        == "fastwam-training-guard-checkpoint-v1"
    )

    resumed = _bare_runner(
        _runner_cfg(
            tmp_path,
            resume_dir=str(checkpoint),
            guard_enabled=True,
        ),
        actor=_WorkerGroup(loaded_steps=[3]),
        rollout=_WorkerGroup(loaded_steps=[3]),
    )
    resumed.init_workers()

    assert resumed.fastwam_training_guard.state_dict() == (
        source.fastwam_training_guard.state_dict()
    )


def test_fastwam_fair_cost_state_is_saved_and_restored_with_runner_step(
    tmp_path: Path,
) -> None:
    cfg = _runner_cfg(
        tmp_path,
        guard_enabled=True,
        fair_cost_enabled=True,
    )
    source = _bare_runner(cfg, actor=_WorkerGroup(), rollout=_WorkerGroup())
    source.fastwam_fair_cost_controller.observe_rollout(
        runner_step=0,
        break_even_idm_cost=0.03,
        idm_fraction=0.6,
    )
    source.global_step = 1
    source._save_checkpoint()
    checkpoint = tmp_path / "l12/checkpoints/global_step_1"
    assert (
        json.loads((checkpoint / "training_guard.json").read_text())["schema"]
        == "fastwam-training-guard-checkpoint-v2"
    )

    resumed = _bare_runner(
        _runner_cfg(
            tmp_path,
            resume_dir=str(checkpoint),
            guard_enabled=True,
            fair_cost_enabled=True,
        ),
        actor=_WorkerGroup(loaded_steps=[1]),
        rollout=_WorkerGroup(loaded_steps=[1]),
    )
    resumed.init_workers()

    assert resumed.fastwam_fair_cost_controller.state_dict() == (
        source.fastwam_fair_cost_controller.state_dict()
    )
    assert resumed.fastwam_fair_cost_controller.decision_for_step(
        1
    ).applied_idm_cost == pytest.approx(0.03)


def test_fastwam_guard_resume_fails_closed_when_state_is_missing(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "global_step_3"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "rollout").mkdir()
    runner = _bare_runner(
        _runner_cfg(
            tmp_path,
            resume_dir=str(checkpoint),
            guard_enabled=True,
        ),
        actor=_WorkerGroup(loaded_steps=[3]),
        rollout=_WorkerGroup(loaded_steps=[3]),
    )

    with pytest.raises(FileNotFoundError, match="training_guard.json"):
        runner.init_workers()
