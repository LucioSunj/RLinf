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

"""Focused contracts for lagged fair-price and opt-in PI cost control."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlinf.runners.fastwam_fair_cost import (
    FastWAMFairCostController,
    append_fastwam_fair_cost_control_jsonl,
)


def _config(*, pi_enabled: bool = False) -> dict:
    return {
        "enabled": True,
        "window_size": 5,
        "pi": {
            "enabled": pi_enabled,
            "target_idm_fraction": 0.5,
            "integral_gain": 0.05,
            "proportional_gain": 0.6,
        },
    }


def test_fair_cost_is_lagged_five_observation_median_without_missing_duplicates() -> (
    None
):
    controller = FastWAMFairCostController(_config(), bootstrap_idm_cost=0.015)
    observations = [0.01, 0.02, 0.03, None, 0.05, 0.09]
    expected_applied = [0.015, 0.01, 0.015, 0.02, 0.02, 0.025]
    expected_next = [0.01, 0.015, 0.02, 0.02, 0.025, 0.03]

    for step, (break_even, applied, next_cost) in enumerate(
        zip(observations, expected_applied, expected_next, strict=True)
    ):
        decision = controller.decision_for_step(step)
        assert decision.applied_idm_cost == pytest.approx(applied)
        record = controller.observe_rollout(
            runner_step=step,
            break_even_idm_cost=break_even,
            idm_fraction=0.5,
        )
        assert record["applied"]["applied_idm_cost"] == pytest.approx(applied)
        assert record["next"]["applied_idm_cost"] == pytest.approx(next_cost)

    # The undefined fourth observation carries the price but does not add a
    # duplicate 0.03 observation to the rolling window.
    assert controller.decision_for_step(6).lagged_break_even_window == pytest.approx(
        (0.01, 0.02, 0.03, 0.05, 0.09)
    )


def test_consecutive_undefined_break_evens_leave_observation_window_unchanged() -> None:
    controller = FastWAMFairCostController(_config(), bootstrap_idm_cost=0.015)
    controller.observe_rollout(
        runner_step=0,
        break_even_idm_cost=0.03,
        idm_fraction=0.5,
    )

    for step in range(1, 4):
        record = controller.observe_rollout(
            runner_step=step,
            break_even_idm_cost=None,
            idm_fraction=0.5,
        )
        assert record["carried_break_even_idm_cost"] == pytest.approx(0.03)
        assert record["next"]["lagged_break_even_window"] == pytest.approx([0.03])
        assert record["next"]["applied_idm_cost"] == pytest.approx(0.03)


def test_calibration_break_evens_replay_without_undefined_duplicates() -> None:
    controller = FastWAMFairCostController(_config(), bootstrap_idm_cost=0.015)
    observations = (
        0.017424965185956716,
        0.0016565770936277546,
        0.049700312536790206,
        None,
        0.012956835035992553,
        0.0372904342249167,
        None,
        0.0011283695805662658,
    )
    expected_applied = (
        0.015,
        0.017424965185956716,
        0.009540771139792235,
        0.017424965185956716,
        0.017424965185956716,
        0.015190900110974635,
        0.017424965185956716,
        0.017424965185956716,
    )

    for step, (break_even, applied) in enumerate(
        zip(observations, expected_applied, strict=True)
    ):
        assert controller.decision_for_step(step).applied_idm_cost == pytest.approx(
            applied
        )
        controller.observe_rollout(
            runner_step=step,
            break_even_idm_cost=break_even,
            idm_fraction=0.5,
        )

    assert controller.decision_for_step(8).applied_idm_cost == pytest.approx(
        0.012956835035992553
    )


def test_first_undefined_break_even_keeps_bootstrap_until_valid_data() -> None:
    controller = FastWAMFairCostController(_config(), bootstrap_idm_cost=0.015)

    record = controller.observe_rollout(
        runner_step=0,
        break_even_idm_cost=None,
        idm_fraction=0.5,
    )

    assert record["carried_break_even_idm_cost"] is None
    assert record["next"]["lagged_break_even_window"] == []
    assert record["next"]["applied_idm_cost"] == pytest.approx(0.015)


def test_pi_is_disabled_by_default_and_adds_no_multiplier() -> None:
    controller = FastWAMFairCostController(
        {"enabled": True, "window_size": 5},
        bootstrap_idm_cost=0.02,
    )

    record = controller.observe_rollout(
        runner_step=0,
        break_even_idm_cost=0.03,
        idm_fraction=0.9,
    )

    assert controller.pi_enabled is False
    assert record["next"]["lagrange_multiplier"] == 0.0
    assert record["next"]["applied_idm_cost"] == pytest.approx(0.03)


def test_pi_converges_on_analytic_one_state_bandit_without_collapse() -> None:
    controller = FastWAMFairCostController(
        _config(pi_enabled=True),
        bootstrap_idm_cost=0.0,
    )
    sensitivity = 0.35
    uncontrolled_fraction = 0.8
    target_fraction = 0.5
    analytic_multiplier = (uncontrolled_fraction - target_fraction) / sensitivity
    fractions = []

    for step in range(500):
        multiplier = controller.decision_for_step(step).lagrange_multiplier
        idm_fraction = max(
            0.0,
            min(1.0, uncontrolled_fraction - sensitivity * multiplier),
        )
        fractions.append(idm_fraction)
        controller.observe_rollout(
            runner_step=step,
            break_even_idm_cost=0.0,
            idm_fraction=idm_fraction,
        )

    final = controller.decision_for_step(500)
    assert final.lagrange_multiplier == pytest.approx(
        analytic_multiplier,
        abs=6.0e-4,
    )
    assert max(fractions[-50:]) - min(fractions[-50:]) < 2.0e-4
    assert all(target_fraction <= value <= uncontrolled_fraction for value in fractions)


def test_fair_cost_state_round_trip_preserves_next_decision() -> None:
    source = FastWAMFairCostController(
        _config(pi_enabled=True), bootstrap_idm_cost=0.01
    )
    for step, break_even in enumerate((0.02, None, 0.04)):
        source.observe_rollout(
            runner_step=step,
            break_even_idm_cost=break_even,
            idm_fraction=0.6,
        )
    state = source.state_dict()
    assert state["break_even_history"][-1] == state["last_valid_break_even_idm_cost"]

    restored = FastWAMFairCostController(
        _config(pi_enabled=True),
        bootstrap_idm_cost=0.01,
    )
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    assert restored.decision_for_step(3) == source.decision_for_step(3)


def test_fair_cost_jsonl_records_applied_and_next_decisions(tmp_path: Path) -> None:
    controller = FastWAMFairCostController(_config(), bootstrap_idm_cost=0.01)
    record = controller.observe_rollout(
        runner_step=0,
        break_even_idm_cost=0.03,
        idm_fraction=0.6,
    )
    destination = tmp_path / "run/audits/fair_cost_control.jsonl"

    append_fastwam_fair_cost_control_jsonl(destination, record)

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == record
    assert loaded["applied"]["applied_idm_cost"] == pytest.approx(0.01)
    assert loaded["next"]["applied_idm_cost"] == pytest.approx(0.03)


def test_pi_cannot_be_enabled_without_fair_cost_control() -> None:
    config = _config(pi_enabled=True)
    config["enabled"] = False

    with pytest.raises(ValueError, match="requires fair-cost"):
        FastWAMFairCostController(config, bootstrap_idm_cost=0.01)
