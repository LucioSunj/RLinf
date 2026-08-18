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

"""Tests for live FastWAM training-curve exports."""

import csv
import json
from pathlib import Path

from rlinf.utils.fastwam_training_plotter import FastWAMTrainingPlotter


def test_plotter_refreshes_live_overview_and_final_exports(tmp_path: Path) -> None:
    plotter = FastWAMTrainingPlotter(
        tmp_path,
        title="FastWAM test",
        interval_steps=2,
        smoothing=0.5,
        dpi=72,
    )
    plotter.record(
        {
            "env/return": 1.0,
            "rollout/returns_mean": 0.5,
            "rollout/returns_min": 0.25,
            "rollout/returns_max": 0.75,
            "rollout/values_mean": 0.4,
            "rollout/fastwam/gate/base_idm_probability_mean": 0.6,
            "train/gate/policy_loss": 0.1,
            "time/step": 2.0,
        },
        step=0,
    )

    assert plotter.maybe_render(0)
    assert (tmp_path / "overview.png").is_file()
    live_summary = json.loads((tmp_path / "summary.json").read_text())
    assert live_summary["finalized"] is False
    assert live_summary["maximum_step"] == 0

    plotter.record({"env/return": 2.0}, step=1)
    assert not plotter.maybe_render(1)
    plotter.finish()

    for name in (
        "overview.png",
        "overview.pdf",
        "all-scalars.pdf",
        "scalars.csv",
        "summary.json",
    ):
        assert (tmp_path / name).is_file()
        assert (tmp_path / name).stat().st_size > 0
    final_summary = json.loads((tmp_path / "summary.json").read_text())
    assert final_summary["finalized"] is True
    assert final_summary["maximum_step"] == 1
    with (tmp_path / "scalars.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(
        row["tag"] == "env/return" and row["step"] == "1" and row["value"] == "2.0"
        for row in rows
    )
