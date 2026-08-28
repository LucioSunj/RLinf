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

"""Tests for historical fair-cost replay and definition reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlinf.algorithms.advantages import (
    FASTWAM_COUNTERFACTUAL_COST_AUDIT_SENTINEL,
)
from rlinf.utils.fastwam_fair_cost_replay import (
    load_counterfactual_cost_audits,
    replay_fair_costs,
)


def _entry(cost: float, *, gap: float) -> dict:
    return {
        "idm_cost": cost,
        "idm_destination_gate_advantage": {
            "unnormalized": {"sum": gap, "finite_count": 1}
        },
        "uncond_destination_gate_advantage": {
            "unnormalized": {"sum": 0.0, "finite_count": 1}
        },
    }


def _audit(*, gap_at_zero: float, gap_at_tenth: float) -> dict:
    break_even = (
        None
        if gap_at_zero <= 0.0 or gap_at_tenth >= gap_at_zero
        else gap_at_zero / -((gap_at_tenth - gap_at_zero) / 0.1)
    )
    return {
        "schema": "fastwam-counterfactual-cost-audit-v1",
        "configured_idm_cost": 0.015,
        "break_even_idm_cost": break_even,
        "eligible_gate_decision_count": 10,
        "eligible_idm_decision_count": 6,
        "entries": [
            _entry(0.0, gap=gap_at_zero),
            _entry(0.1, gap=gap_at_tenth),
        ],
    }


def test_replay_loads_stdout_sentinels_and_uses_only_prior_steps(
    tmp_path: Path,
) -> None:
    records = [
        _audit(gap_at_zero=0.3, gap_at_tenth=-0.1),
        _audit(gap_at_zero=-0.1, gap_at_tenth=-0.2),
        _audit(gap_at_zero=0.2, gap_at_tenth=-0.2),
    ]
    source = tmp_path / "training.stdout.log"
    source.write_text(
        "\n".join(
            f"actor-prefix {FASTWAM_COUNTERFACTUAL_COST_AUDIT_SENTINEL} "
            + json.dumps(record)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )

    replay = replay_fair_costs(load_counterfactual_cost_audits(source))

    assert replay["record_count"] == 3
    assert replay["undefined_break_even_count"] == 1
    costs = [
        record["fair_cost_applied"]["applied_idm_cost"] for record in replay["records"]
    ]
    assert costs == pytest.approx([0.015, 0.075, 0.075])
    assert replay["records"][1]["carried_break_even_idm_cost"] == pytest.approx(0.075)
    assert replay["records"][1]["undefined_break_even_carried_forward"] is True
    assert replay["undefined_break_even_runner_steps"] == [1]
    assert replay["stability"]["maximum_adjacent_change_factor"] == pytest.approx(5.0)
    assert replay["stability"]["adjacent_change_exceeds_twofold"] is True
    assert replay["stability"]["closed_loop_stability_claim"] is False


def test_replay_stops_when_serialized_break_even_disagrees_with_production() -> None:
    record = _audit(gap_at_zero=0.3, gap_at_tenth=-0.1)
    record["break_even_idm_cost"] = 0.01

    with pytest.raises(ValueError, match="disagrees with advantages.py"):
        replay_fair_costs([record])
