# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import pytest

from rlinf.models.embodiment.wam_policy.pad_rv.budget import (
    PAD_PREDICTION_BUDGET_CONTROLLER_TARGET,
    PadPredictionBudgetController,
)


def _controller() -> PadPredictionBudgetController:
    return PadPredictionBudgetController.from_configs(
        branch_cost={
            "idm_cost": 0.015,
            "fair_cost": {
                "enabled": True,
                "window_size": 5,
                "pi": {
                    "enabled": False,
                    "target_idm_fraction": 0.5,
                    "integral_gain": 0.05,
                    "proportional_gain": 0.0,
                },
            },
        },
        prediction_budget={
            "enabled": True,
            "controller_target": PAD_PREDICTION_BUDGET_CONTROLLER_TARGET,
            "target_idm_fraction": 0.5,
            "dual_lr": 0.05,
            "proportional_gain": 0.0,
        },
    )


def test_pad_budget_uses_fair_cost_only_as_initial_dual_and_diagnostic() -> None:
    controller = _controller()
    assert controller.decision_for_step(0).applied_idm_cost == pytest.approx(0.015)

    first = controller.observe_rollout(
        runner_step=0,
        break_even_idm_cost=0.006,
        idm_fraction=0.4,
    )
    assert first["next"]["fair_idm_cost"] == pytest.approx(0.006)
    assert first["next"]["applied_idm_cost"] == pytest.approx(0.01)

    controller.observe_rollout(
        runner_step=1,
        break_even_idm_cost=0.008,
        idm_fraction=0.2,
    )
    decision = controller.decision_for_step(2)
    assert decision.fair_idm_cost == pytest.approx(0.007)
    assert decision.applied_idm_cost == 0.0


def test_pad_budget_multiplier_increases_only_when_constraint_is_exceeded() -> None:
    controller = _controller()
    record = controller.observe_rollout(
        runner_step=0,
        break_even_idm_cost=None,
        idm_fraction=0.7,
    )
    assert record["dual_update"] == pytest.approx(0.01)
    assert record["next"]["applied_idm_cost"] == pytest.approx(0.025)


def test_pad_budget_state_round_trip_preserves_next_decision() -> None:
    source = _controller()
    source.observe_rollout(
        runner_step=0,
        break_even_idm_cost=0.004,
        idm_fraction=0.3,
    )
    restored = _controller()
    restored.load_state_dict(source.state_dict())
    assert restored.state_dict() == source.state_dict()
    assert restored.decision_for_step(1) == source.decision_for_step(1)


def test_pad_budget_rejects_legacy_proportional_controller() -> None:
    with pytest.raises(ValueError, match="no proportional term"):
        PadPredictionBudgetController.from_configs(
            branch_cost={
                "idm_cost": 0.015,
                "fair_cost": {
                    "enabled": True,
                    "window_size": 5,
                    "pi": {"enabled": False},
                },
            },
            prediction_budget={
                "enabled": True,
                "controller_target": PAD_PREDICTION_BUDGET_CONTROLLER_TARGET,
                "target_idm_fraction": 0.5,
                "dual_lr": 0.05,
                "proportional_gain": 0.6,
            },
        )
