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

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner


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


def _runner_cfg(tmp_path: Path, *, resume_dir: str | None = None):
    return OmegaConf.create(
        {
            "actor": {"model": {"model_type": "fastwam_adaptive"}},
            "runner": {
                "resume_dir": resume_dir,
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
