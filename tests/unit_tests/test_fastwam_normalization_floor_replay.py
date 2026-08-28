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

import statistics

import pytest

from rlinf.utils.fastwam_normalization_floor_replay import (
    infer_legacy_normalization_std,
    replay_normalization_floor,
)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "finite_count": len(values),
        "sum": sum(values),
        "sum_of_squares": sum(value * value for value in values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _record(*, runner_step: int, batch_std: float) -> dict:
    raw = [-0.3, -0.1, 0.2, 0.4]
    mean = statistics.mean(raw)
    divisor = batch_std + 1.0e-5
    normalized = [(value - mean) / divisor for value in raw]
    return {
        "runner_step": runner_step,
        "configured_idm_cost": 0.015,
        "entries": [
            {
                "idm_cost": 0.015,
                "gate_advantage": {
                    "unnormalized": _summary(raw),
                    "normalized": _summary(normalized),
                },
            }
        ],
    }


def test_infers_legacy_batch_standard_deviation() -> None:
    entry = _record(runner_step=0, batch_std=0.1)["entries"][0]

    assert infer_legacy_normalization_std(entry) == pytest.approx(
        0.1,
        abs=1.0e-12,
    )


def test_replays_floor_hit_fraction_per_update() -> None:
    replay = replay_normalization_floor(
        [
            _record(runner_step=0, batch_std=0.1),
            _record(runner_step=1, batch_std=0.2),
        ],
        std_floor=0.15,
    )

    assert replay["floor_hit_count"] == 1
    assert replay["floor_hit_fraction"] == 0.5
    assert [row["floor_hit_fraction"] for row in replay["records"]] == [1.0, 0.0]
    assert replay["batch_standard_deviation_distribution"] == pytest.approx(
        {
            "count": 2,
            "minimum": 0.1,
            "p10": 0.11,
            "p25": 0.125,
            "p50": 0.15,
            "p75": 0.175,
            "p90": 0.19,
            "maximum": 0.2,
        }
    )
    assert replay["floor_divisor_over_batch_standard_deviation_distribution"][
        "p50"
    ] == pytest.approx(1.5)
    assert replay["records"][0][
        "floored_advantage_amplitude_fraction_vs_legacy"
    ] == pytest.approx((0.1 + 1.0e-5) / 0.15)
