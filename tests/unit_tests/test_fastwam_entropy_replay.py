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

"""Tests for the fifth-run Gate entropy replay helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlinf.utils.fastwam_entropy_replay import (
    FASTWAM_ROLLOUT_STATE_SENTINEL,
    binary_entropy,
    build_entropy_replay_rows,
    entropy_bounds_from_range_mean,
    entropy_guard_trigger_intervals,
    load_rollout_probability_summaries,
    summarize_entropy_guard,
)


def test_entropy_bounds_contain_mean_entropy_for_supported_distribution() -> None:
    probabilities = [0.2, 0.4, 0.8]
    mean = sum(probabilities) / len(probabilities)
    observed = sum(binary_entropy(value) for value in probabilities) / len(
        probabilities
    )

    lower, upper = entropy_bounds_from_range_mean(
        minimum=min(probabilities),
        mean=mean,
        maximum=max(probabilities),
    )

    assert lower <= observed <= upper


def test_rollout_summary_parser_and_entropy_rows(tmp_path: Path) -> None:
    record = {
        "schema": "fastwam-rollout-state-audit-v1",
        "eligible_gate_decision_count": 10,
        "base_probability": {
            "minimum": 0.7,
            "mean": 0.8,
            "maximum": 0.9,
        },
        "behavior_probability": {
            "minimum": 0.68,
            "mean": 0.77,
            "maximum": 0.86,
        },
    }
    source = tmp_path / "training.stdout.log"
    source.write_text(
        f"prefix {FASTWAM_ROLLOUT_STATE_SENTINEL} {json.dumps(record)}\n",
        encoding="utf-8",
    )

    rows = build_entropy_replay_rows(
        load_rollout_probability_summaries(source),
        [0.5],
    )

    assert len(rows) == 1
    assert rows[0]["base_entropy_of_mean"] == pytest.approx(binary_entropy(0.8))
    assert rows[0]["behavior_entropy_of_mean"] == pytest.approx(binary_entropy(0.77))
    assert rows[0]["observed_old_train_behavior_entropy"] == 0.5


def test_base_entropy_guard_intervals_do_not_depend_on_exploration_epsilon() -> None:
    intervals = entropy_guard_trigger_intervals(0.35)
    rows = [
        {
            "update": 1.0,
            "base_entropy_lower_bound": 0.34,
            "base_entropy_of_mean": 0.36,
        },
        {
            "update": 2.0,
            "base_entropy_lower_bound": 0.30,
            "base_entropy_of_mean": 0.32,
        },
    ]

    summary = summarize_entropy_guard(rows)

    assert intervals[0]["maximum"] == pytest.approx(0.11167352323075633)
    assert intervals[1]["minimum"] == pytest.approx(0.8883264767692437)
    assert summary["first_possible_same_rollout_trigger_update"] == 1
    assert summary["first_guaranteed_same_rollout_trigger_update"] == 2
    assert summary["indeterminate_same_rollout_updates"] == [1]
    assert summary["trigger_intervals_are_epsilon_independent"] is True
    assert (
        len(
            {
                json.dumps(value, sort_keys=True)
                for value in summary[
                    "base_probability_trigger_intervals_by_epsilon"
                ].values()
            }
        )
        == 1
    )
