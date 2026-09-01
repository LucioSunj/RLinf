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

"""Pure-CPU contracts for generic and two-sided FastWAM branch costs."""

from __future__ import annotations

import copy

import pytest

from rlinf.runners.fastwam_branch_cost_control import (
    DiagnosticBranchCostController,
    LegacyIDMCostControllerAdapter,
    SignedBandPriceController,
)
from rlinf.runners.fastwam_cost_diagnostics import (
    FairBreakEvenDiagnosticAdapter,
)
from rlinf.runners.fastwam_idm_cost_control import (
    FastWAMIDMCostObservation,
    ProjectedDualIDMCostController,
)


def _config(
    *,
    learning_rate: float = 1.0,
    ema_beta: float = 0.0,
    update_interval: int = 1,
    maximum: float = 0.2,
    max_delta: float = 0.2,
) -> dict:
    return {
        "type": "band_price",
        "constraint": "two_sided_band",
        "rate": {
            "scope": "eligible_gate_decisions",
            "feedback": "expected_behavior_probability",
            "target_idm_fraction": 0.5,
            "half_width": 0.03,
        },
        "charge_scope": "eligible_nonforced",
        "signed_price": {
            "initial_value": 0.0,
            "learning_rate": learning_rate,
            "ema_beta": ema_beta,
            "update_interval": update_interval,
            "max_abs_value": maximum,
            "max_delta_per_update": max_delta,
        },
    }


def _observation(
    step: int, rate: float, *, break_even: float | None = None
) -> FastWAMIDMCostObservation:
    eligible = 100
    eligible_idm = round(rate * eligible)
    valid = 120
    valid_idm = round(rate * valid)
    return FastWAMIDMCostObservation(
        runner_step=step,
        eligible_gate_decision_count=eligible,
        eligible_idm_decision_count=eligible_idm,
        eligible_realized_fraction=eligible_idm / eligible,
        eligible_expected_fraction=rate,
        valid_chunk_count=valid,
        valid_idm_chunk_count=valid_idm,
        executed_realized_fraction=valid_idm / valid,
        forced_fraction=0.1,
        break_even_idm_cost=break_even,
        configured_idm_cost=None,
    )


def _advance(
    controller: SignedBandPriceController,
    step: int,
    rate: float,
) -> tuple[object, dict]:
    decision = controller.decision_for_step(step)
    return decision, controller.observe_rollout(_observation(step, rate))


def _assert_branch_identity(decision) -> None:
    assert min(decision.idm_cost, decision.uncond_cost) == 0.0
    assert max(decision.idm_cost, decision.uncond_cost) == pytest.approx(
        abs(decision.components["signed_price"])
    )


def test_band_price_penalizes_only_the_overused_branch_with_one_rollout_lag() -> None:
    controller = SignedBandPriceController(_config())
    first, first_record = _advance(controller, 0, 0.70)
    second = controller.decision_for_step(1)
    assert first.idm_cost == first.uncond_cost == 0.0
    assert first_record["next"]["idm_cost"] > 0.0
    assert first_record["next"]["uncond_cost"] == 0.0
    assert second.idm_cost == pytest.approx(first_record["next"]["idm_cost"])
    _assert_branch_identity(second)
    second_record = controller.observe_rollout(_observation(1, 0.0))
    assert second_record["next"]["uncond_cost"] > 0.0
    assert second_record["next"]["idm_cost"] == 0.0


def test_band_price_inside_band_ema_interval_projection_and_max_delta() -> None:
    inside = SignedBandPriceController(_config())
    _, record = _advance(inside, 0, 0.51)
    assert record["update"]["inside_band"]
    assert record["next"]["idm_cost"] == 0.0
    assert record["next"]["uncond_cost"] == 0.0

    interval = SignedBandPriceController(_config(update_interval=2))
    _, first = _advance(interval, 0, 0.9)
    _, second = _advance(interval, 1, 0.9)
    assert first["update"]["applied_delta"] == 0.0
    assert second["update"]["applied_delta"] > 0.0

    ema = SignedBandPriceController(_config(ema_beta=0.5))
    _advance(ema, 0, 0.8)
    _, ema_record = _advance(ema, 1, 0.2)
    assert ema.rate_ema == pytest.approx(0.5)
    assert ema_record["update"]["inside_band"]

    clipped = SignedBandPriceController(
        _config(learning_rate=10.0, maximum=0.05, max_delta=0.02)
    )
    _, clipped_record = _advance(clipped, 0, 1.0)
    assert clipped_record["update"]["applied_delta"] == pytest.approx(0.02)
    assert clipped.max_delta_clip_count == 1
    _advance(clipped, 1, 1.0)
    _, projected = _advance(clipped, 2, 1.0)
    assert projected["next"]["idm_cost"] == pytest.approx(0.05)
    assert clipped.positive_projection_count == 1


def test_band_price_signed_price_crosses_zero_without_dual_branch_overlap() -> None:
    controller = SignedBandPriceController(_config())
    sequence = (0.70, 0.0, 0.0, 0.49, 0.80)
    signs = []
    for step, rate in enumerate(sequence):
        decision, _ = _advance(controller, step, rate)
        _assert_branch_identity(decision)
        signs.append(decision.components["signed_price"])
    assert any(value > 0.0 for value in signs)
    assert any(value < 0.0 for value in signs)


def test_band_price_checkpoint_round_trip() -> None:
    source = SignedBandPriceController(_config())
    _advance(source, 0, 0.8)
    state = copy.deepcopy(source.state_dict())
    restored = SignedBandPriceController(_config())
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.decision_for_step(1) == source.decision_for_step(1)


def test_legacy_idm_adapter_preserves_delegate_state_and_decision() -> None:
    dual_config = {
        "type": "budget_dual",
        "constraint": "upper_bound",
        "rate": {
            "scope": "eligible_gate_decisions",
            "feedback": "eligible_realized",
            "target_idm_fraction": 0.5,
        },
        "charge_scope": "eligible_nonforced_idm",
        "initializer": {"type": "constant", "idm_cost": 0.02},
        "dual": {
            "learning_rate": 0.1,
            "ema_beta": 0.0,
            "deadband": 0.0,
            "update_interval": 1,
            "min_idm_cost": 0.0,
            "max_idm_cost": 0.2,
            "max_delta_per_update": 0.05,
        },
    }
    delegate = ProjectedDualIDMCostController(dual_config, bootstrap_idm_cost=0.0)
    adapter = LegacyIDMCostControllerAdapter(delegate, uncond_cost=0.0)
    branch = adapter.decision_for_step(0)
    assert branch.idm_cost == 0.02
    assert branch.uncond_cost == 0.0
    adapter.observe_rollout(_observation(0, 0.8))
    assert adapter.state_dict() == delegate.state_dict()


def test_fair_diagnostic_cannot_change_band_prices_and_restores() -> None:
    delegate = SignedBandPriceController(_config())
    diagnostic = FairBreakEvenDiagnosticAdapter(
        {
            "diagnostic_only": True,
            "window_size": 3,
            "bootstrap_value_for_display": 0.07,
        },
        bootstrap_idm_cost=0.07,
    )
    controller = DiagnosticBranchCostController(delegate, (diagnostic,))
    decision = controller.decision_for_step(0)
    assert decision.idm_cost == decision.uncond_cost == 0.0
    record = controller.observe_rollout(_observation(0, 0.8, break_even=0.09))
    assert record["next"]["idm_cost"] == pytest.approx(0.2)
    assert record["next"]["uncond_cost"] == 0.0
    assert record["diagnostic_invariants"]["branch_costs_unchanged"] is True
    metrics = controller.record_metrics(record)
    assert metrics["fastwam/idm_cost_control/diagnostic_fair_cost"] == 0.09

    state = controller.state_dict()
    restored = DiagnosticBranchCostController(
        SignedBandPriceController(_config()),
        (
            FairBreakEvenDiagnosticAdapter(
                {
                    "diagnostic_only": True,
                    "window_size": 3,
                    "bootstrap_value_for_display": 0.07,
                },
                bootstrap_idm_cost=0.07,
            ),
        ),
    )
    restored.load_state_dict(copy.deepcopy(state))
    assert restored.state_dict() == state
