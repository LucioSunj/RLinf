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

from omegaconf import OmegaConf

from rlinf.utils.metric_logger import MetricLogger


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
