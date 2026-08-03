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

"""Focused tests for native all-layer step-zero checkpoint export."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.runners.fastwam_checkpoint_export import (
    export_initial_actor_checkpoint,
    validate_initial_checkpoint_export_config,
)


class _Handle:
    def wait(self):
        return None


class _ActorGroup:
    def __init__(self) -> None:
        self.calls = []

    def init_worker(self):
        self.calls.append(("init_worker",))
        return _Handle()

    def save_checkpoint(self, path: str, step: int):
        self.calls.append(("save_checkpoint", path, step))
        return _Handle()


def _cfg(output_dir: Path):
    return OmegaConf.create(
        {
            "runner": {
                "resume_dir": None,
                "bootstrap_project_checkpoint_dir": str(output_dir),
            },
            "actor": {
                "model": {
                    "model_type": "fastwam_adaptive",
                    "fastwam": {"action_dit_config": {"num_layers": 30}},
                    "gate": {
                        "share_blocks": False,
                        "denoise_last_n": 1,
                        "layer_taps": {
                            "mode": "all",
                            "last_n": None,
                            "indices": None,
                        },
                    },
                }
            },
        }
    )


def test_export_initial_actor_checkpoint_uses_only_production_step_zero_save(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "native-all-layer-step0"
    cfg = _cfg(output_dir)
    actor = _ActorGroup()

    actor_dir = export_initial_actor_checkpoint(
        cfg,
        actor_group=actor,
        actor_world_size=1,
    )

    assert actor_dir == output_dir / "actor"
    assert actor.calls == [
        ("init_worker",),
        ("save_checkpoint", str(output_dir / "actor"), 0),
    ]


def test_initial_checkpoint_export_rejects_nonempty_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        export_initial_actor_checkpoint(
            _cfg(output_dir),
            actor_group=_ActorGroup(),
            actor_world_size=1,
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("runner.resume_dir", "/resume", "resume_dir"),
        ("actor.model.model_type", "openvla", "fastwam_adaptive"),
        ("actor.model.gate.layer_taps.mode", "indices", "all-layer"),
        ("actor.model.gate.share_blocks", True, "independent"),
        ("actor.model.gate.denoise_last_n", 2, "denoise_last_n"),
        ("actor.model.fastwam.action_dit_config.num_layers", 29, "30"),
    ],
)
def test_initial_checkpoint_export_rejects_noncanonical_config(
    tmp_path: Path,
    path: str,
    value,
    message: str,
) -> None:
    cfg = _cfg(tmp_path / "checkpoint")
    OmegaConf.update(cfg, path, value, merge=False)

    with pytest.raises(ValueError, match=message):
        validate_initial_checkpoint_export_config(cfg, actor_world_size=1)


def test_initial_checkpoint_export_requires_one_actor_rank(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one actor rank"):
        validate_initial_checkpoint_export_config(
            _cfg(tmp_path / "checkpoint"),
            actor_world_size=2,
        )
