# Copyright 2025 The RLinf Authors.
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

from pathlib import Path

import pytest
import torch.utils.tensorboard
from omegaconf import OmegaConf

import rlinf.utils.metric_logger as metric_logger_module
from rlinf.utils.metric_logger import FASTWAM_TENSORBOARD_LAYOUT, MetricLogger


def test_tensorboard_output_is_scoped_by_experiment_name(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "project_name": "rlinf",
                    "experiment_name": "short-canary-v1",
                    "logger_backends": ["tensorboard"],
                },
                "per_worker_log": False,
            }
        }
    )

    logger = MetricLogger(cfg)
    logger.log({"gate/loss": 0.25}, step=1)
    logger.finish()

    run_dir = tmp_path / "short-canary-v1" / "tensorboard"
    assert (run_dir / "config.yaml").is_file()
    assert list(run_dir.glob("events.out.tfevents.*"))
    assert not (tmp_path / "tensorboard").exists()


def test_fastwam_tensorboard_registers_project_dashboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class _Writer:
        def __init__(self, log_path, *, flush_secs):
            calls.append(("init", Path(log_path), flush_secs))

        def add_custom_scalars(self, layout):
            calls.append(("layout", layout))

        def add_scalar(self, key, value, step):
            calls.append(("scalar", key, value, step))

        def flush(self):
            calls.append(("flush",))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", _Writer)
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "fastwam_adaptive"}},
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "project_name": "rlinf",
                    "experiment_name": "fastwam-dashboard",
                    "logger_backends": ["tensorboard"],
                    "fastwam_observability": {
                        "enabled": True,
                        "tensorboard": {
                            "custom_scalars": True,
                            "flush_every_step": True,
                            "flush_secs": 3,
                        },
                        "static_plots": {"enabled": False},
                    },
                },
                "per_worker_log": False,
            },
        }
    )

    logger = MetricLogger(cfg)
    logger.log({"rollout/returns_mean": 0.25}, step=3)
    logger.commit_step(3)
    logger.finish()

    assert ("init", tmp_path / "fastwam-dashboard" / "tensorboard", 3) in calls
    assert ("layout", FASTWAM_TENSORBOARD_LAYOUT) in calls
    assert ("scalar", "rollout/returns_mean", 0.25, 3) in calls
    assert ("flush",) in calls


def test_fastwam_commit_step_publishes_wandb_metrics() -> None:
    calls = []

    class _Wandb:
        def log(self, *, data, step, commit):
            calls.append((data, step, commit))

    logger = object.__new__(MetricLogger)
    logger.tensorboard_flush_every_step = True
    logger._all_loggers = [{"wandb": _Wandb()}]
    logger._fastwam_plotter = None
    logger._finished = True

    logger.commit_step(7)

    assert calls == [({}, 7, True)]


def test_fastwam_static_plotting_is_switchable_and_committed_per_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class _Plotter:
        def __init__(self, output_dir, **kwargs):
            calls.append(("init", Path(output_dir), kwargs))

        def record(self, data, step):
            calls.append(("record", data, step))

        def maybe_render(self, step):
            calls.append(("render", step))

        def finish(self):
            calls.append(("finish",))

    monkeypatch.setattr(metric_logger_module, "FastWAMTrainingPlotter", _Plotter)
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "fastwam_adaptive"}},
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "live-plots",
                    "logger_backends": None,
                    "fastwam_observability": {
                        "enabled": True,
                        "static_plots": {
                            "enabled": True,
                            "interval_steps": 2,
                        },
                    },
                },
                "per_worker_log": False,
            },
        }
    )

    logger = MetricLogger(cfg)
    logger.log({"env/return": 1.0}, step=4)
    logger.commit_step(4)
    logger.finish()

    assert calls[0][0:2] == (
        "init",
        (tmp_path / "live-plots" / "training_curves").resolve(),
    )
    assert ("record", {"env/return": 1.0}, 4) in calls
    assert ("render", 4) in calls
    assert calls[-1] == ("finish",)


def test_fastwam_observability_master_switch_disables_project_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class _Writer:
        def __init__(self, log_path, *, flush_secs):
            calls.append(("init", Path(log_path), flush_secs))

        def add_custom_scalars(self, layout):
            calls.append(("layout", layout))

        def add_scalar(self, key, value, step):
            calls.append(("scalar", key, value, step))

        def flush(self):
            calls.append(("flush",))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", _Writer)
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "fastwam_adaptive"}},
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "disabled-views",
                    "logger_backends": ["tensorboard"],
                    "fastwam_observability": {"enabled": False},
                },
                "per_worker_log": False,
            },
        }
    )

    logger = MetricLogger(cfg)
    logger.log({"env/return": 1.0}, step=0)
    logger.commit_step(0)
    logger.finish()

    assert not any(call[0] == "layout" for call in calls)
    assert logger._fastwam_plotter is None
