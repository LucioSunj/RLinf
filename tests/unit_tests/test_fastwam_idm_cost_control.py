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

"""Pure-CPU contracts for config-selected FastWAM IDM cost control."""

from __future__ import annotations

import copy

import pytest

from rlinf.runners.fastwam_cost_diagnostics import DiagnosticIDMCostController
from rlinf.runners.fastwam_fair_cost import FastWAMFairCostController
from rlinf.runners.fastwam_idm_cost_control import (
    FastWAMIDMCostObservation,
    LegacyFairCostAdapter,
    ProjectedDualIDMCostController,
    aggregate_fastwam_idm_cost_observation,
    build_fastwam_idm_cost_controller,
)

B50_SLACK_RATES = (
    0.419917,
    0.461307,
    0.392830,
    0.422474,
    0.413275,
    0.322959,
    0.263518,
    0.218749,
    0.176070,
    0.136817,
)


def _dual_config(
    *,
    initial: float = 0.1,
    learning_rate: float = 0.1,
    ema_beta: float = 0.0,
    deadband: float = 0.0,
    update_interval: int = 1,
    minimum: float = 0.0,
    maximum: float = 0.2,
    max_delta: float = 0.05,
) -> dict:
    return {
        "type": "budget_dual",
        "constraint": "upper_bound",
        "rate": {
            "scope": "eligible_gate_decisions",
            "feedback": "eligible_realized",
            "target_idm_fraction": 0.5,
        },
        "charge_scope": "eligible_nonforced_idm",
        "initializer": {"type": "constant", "idm_cost": initial},
        "dual": {
            "learning_rate": learning_rate,
            "ema_beta": ema_beta,
            "deadband": deadband,
            "update_interval": update_interval,
            "min_idm_cost": minimum,
            "max_idm_cost": maximum,
            "max_delta_per_update": max_delta,
        },
    }


def _observation(
    step: int,
    rate: float,
    *,
    expected_rate: float | None = None,
    break_even: float | None = None,
) -> FastWAMIDMCostObservation:
    eligible = 100
    eligible_idm = round(rate * eligible)
    return FastWAMIDMCostObservation(
        runner_step=step,
        eligible_gate_decision_count=eligible,
        eligible_idm_decision_count=eligible_idm,
        eligible_realized_fraction=eligible_idm / eligible,
        eligible_expected_fraction=(rate if expected_rate is None else expected_rate),
        valid_chunk_count=120,
        valid_idm_chunk_count=round(rate * 120),
        executed_realized_fraction=round(rate * 120) / 120,
        forced_fraction=0.1,
        break_even_idm_cost=break_even,
        configured_idm_cost=None,
    )


def _advance(
    controller: ProjectedDualIDMCostController,
    step: int,
    rate: float,
    *,
    break_even: float | None = None,
) -> tuple[float, dict]:
    applied = controller.decision_for_step(step).applied_idm_cost
    record = controller.observe_rollout(_observation(step, rate, break_even=break_even))
    return applied, record


def test_legacy_auto_detection_preserves_fixed_and_fair_sequences() -> None:
    fixed, explicit, digest = build_fastwam_idm_cost_controller(
        {
            "enabled": True,
            "idm_cost": 0.015,
            "fair_cost": {"enabled": False},
        }
    )
    assert not fixed.enabled
    assert not explicit
    assert digest == ""

    disabled, explicit, _ = build_fastwam_idm_cost_controller(
        {
            "enabled": False,
            "idm_cost": 0.015,
            "fair_cost": {"enabled": True, "pi": {"enabled": True}},
        }
    )
    assert not disabled.enabled
    assert not explicit

    fair_config = {
        "enabled": True,
        "window_size": 5,
        "pi": {"enabled": False},
    }
    branch = {
        "enabled": True,
        "idm_cost": 0.015,
        "fair_cost": fair_config,
    }
    adapted, explicit, _ = build_fastwam_idm_cost_controller(branch)
    direct = FastWAMFairCostController(fair_config, bootstrap_idm_cost=0.015)
    assert isinstance(adapted, LegacyFairCostAdapter)
    assert not explicit
    for step, break_even in enumerate((0.01, 0.03, None, 0.02)):
        assert adapted.decision_for_step(step).applied_idm_cost == pytest.approx(
            direct.decision_for_step(step).applied_idm_cost
        )
        adapted_record = adapted.observe_rollout(
            _observation(step, 0.6, break_even=break_even)
        )
        direct_record = direct.observe_rollout(
            runner_step=step,
            break_even_idm_cost=break_even,
            idm_fraction=0.6,
        )
        assert adapted_record["next"]["applied_idm_cost"] == pytest.approx(
            direct_record["next"]["applied_idm_cost"]
        )


def test_projected_dual_direction_projection_and_exact_one_rollout_lag() -> None:
    controller = ProjectedDualIDMCostController(_dual_config(), bootstrap_idm_cost=0.0)

    applied0, record0 = _advance(controller, 0, 0.8)
    assert applied0 == pytest.approx(0.1)
    assert record0["applied"]["applied_idm_cost"] == pytest.approx(0.1)
    assert record0["next"]["applied_idm_cost"] == pytest.approx(0.13)
    assert controller.decision_for_step(1).applied_idm_cost == pytest.approx(0.13)
    record1 = controller.observe_rollout(_observation(1, 0.0))
    assert record1["next"]["applied_idm_cost"] == pytest.approx(0.08)

    maximum = ProjectedDualIDMCostController(
        _dual_config(initial=0.19, learning_rate=1.0),
        bootstrap_idm_cost=0.0,
    )
    _, maximum_record = _advance(maximum, 0, 1.0)
    assert maximum_record["next"]["applied_idm_cost"] == pytest.approx(0.2)
    assert maximum.max_projection_count == 1


def test_projected_dual_max_delta_deadband_ema_and_update_interval() -> None:
    clipped = ProjectedDualIDMCostController(
        _dual_config(learning_rate=1.0, max_delta=0.02),
        bootstrap_idm_cost=0.0,
    )
    _, record = _advance(clipped, 0, 1.0)
    assert record["update"]["raw_delta"] == pytest.approx(0.5)
    assert record["update"]["applied_delta"] == pytest.approx(0.02)
    assert clipped.max_delta_clip_count == 1

    deadband = ProjectedDualIDMCostController(
        _dual_config(deadband=0.05), bootstrap_idm_cost=0.0
    )
    _, record = _advance(deadband, 0, 0.53)
    assert record["next"]["applied_idm_cost"] == pytest.approx(0.1)

    ema = ProjectedDualIDMCostController(
        _dual_config(ema_beta=0.5), bootstrap_idm_cost=0.0
    )
    _advance(ema, 0, 1.0)
    _, record = _advance(ema, 1, 0.0)
    assert record["observed"]["feedback_rate"] == pytest.approx(0.0)
    assert ema.rate_ema == pytest.approx(0.5)
    assert record["update"]["applied_delta"] == pytest.approx(0.0)

    interval = ProjectedDualIDMCostController(
        _dual_config(update_interval=2), bootstrap_idm_cost=0.0
    )
    _, first = _advance(interval, 0, 1.0)
    _, second = _advance(interval, 1, 1.0)
    assert first["update"]["applied_delta"] == 0.0
    assert second["update"]["applied_delta"] == pytest.approx(0.05)

    invalid_interval = _dual_config()
    invalid_interval["dual"]["update_interval"] = 1.5
    with pytest.raises(ValueError, match="positive integer"):
        ProjectedDualIDMCostController(
            invalid_interval,
            bootstrap_idm_cost=0.0,
        )


def test_projected_dual_has_no_negative_cost_or_integral_windup() -> None:
    controller = ProjectedDualIDMCostController(
        _dual_config(initial=0.0), bootstrap_idm_cost=0.0
    )
    for step in range(10):
        applied, record = _advance(controller, step, 0.0)
        assert applied == 0.0
        assert record["next"]["applied_idm_cost"] == 0.0
    _, recovery = _advance(controller, 10, 1.0)
    assert recovery["next"]["applied_idm_cost"] == pytest.approx(0.05)
    assert controller.min_projection_count == 10


def test_projected_dual_state_round_trip_and_monotone_plant_convergence() -> None:
    config = _dual_config(initial=0.02, learning_rate=0.1, max_delta=0.02)
    uninterrupted = ProjectedDualIDMCostController(config, bootstrap_idm_cost=0.0)
    _advance(uninterrupted, 0, 0.8)
    restored = ProjectedDualIDMCostController(config, bootstrap_idm_cost=0.0)
    restored.load_state_dict(copy.deepcopy(uninterrupted.state_dict()))
    assert restored.state_dict() == uninterrupted.state_dict()
    assert restored.decision_for_step(1) == uninterrupted.decision_for_step(1)

    plant = ProjectedDualIDMCostController(
        _dual_config(initial=0.0, learning_rate=0.1, max_delta=0.02),
        bootstrap_idm_cost=0.0,
    )
    rates = []
    for step in range(40):
        cost = plant.decision_for_step(step).applied_idm_cost
        rate = max(0.0, min(1.0, 0.8 - 2.0 * cost))
        rates.append(rate)
        plant.observe_rollout(_observation(step, rate))
    assert rates[-1] == pytest.approx(0.5, abs=0.01)
    assert max(rates[-10:]) - min(rates[-10:]) < 0.02


def test_fair_warmstart_transitions_next_rollout_without_double_pricing() -> None:
    config = _dual_config(initial=0.0)
    config["initializer"] = {
        "type": "break_even_median",
        "bootstrap_idm_cost": 0.015,
        "window_size": 5,
        "warmup_rollouts": 2,
        "minimum_valid_observations": 2,
        "insufficient_data_policy": "keep_bootstrap",
        "monitor_after_warmup": True,
    }
    controller = ProjectedDualIDMCostController(config, bootstrap_idm_cost=0.015)

    applied0, first = _advance(controller, 0, 0.8, break_even=0.02)
    applied1, transition = _advance(controller, 1, 0.8, break_even=0.04)
    assert applied0 == pytest.approx(0.015)
    assert first["next"]["applied_idm_cost"] == pytest.approx(0.02)
    assert applied1 == pytest.approx(0.02)
    assert transition["update"]["transitioned"]
    assert transition["update"]["applied_delta"] == 0.0
    assert transition["next"]["phase"] == "dual"
    assert transition["next"]["applied_idm_cost"] == pytest.approx(0.03)
    assert transition["next"]["components"]["dual_multiplier"] == pytest.approx(0.03)

    applied2, monitored = _advance(controller, 2, 0.8, break_even=0.09)
    assert applied2 == pytest.approx(0.03)
    assert monitored["applied"]["applied_idm_cost"] == pytest.approx(
        monitored["applied"]["components"]["dual_multiplier"]
    )
    assert monitored["applied"]["applied_idm_cost"] != pytest.approx(
        monitored["applied"]["components"]["dual_multiplier"]
        + monitored["applied"]["components"]["fair_estimate"]
    )


def test_fair_warmstart_undefined_data_keeps_bootstrap_and_does_not_duplicate() -> None:
    config = _dual_config()
    config["initializer"] = {
        "type": "break_even_median",
        "bootstrap_idm_cost": 0.015,
        "window_size": 3,
        "warmup_rollouts": 1,
        "minimum_valid_observations": 2,
        "insufficient_data_policy": "keep_bootstrap",
        "monitor_after_warmup": False,
    }
    controller = ProjectedDualIDMCostController(config, bootstrap_idm_cost=0.015)
    for step in range(3):
        applied, record = _advance(controller, step, 0.5, break_even=None)
        assert applied == pytest.approx(0.015)
        assert record["next"]["phase"] == "warmstart"
    assert controller.state_dict()["initializer_state"]["break_even_history"] == []


def test_worker_rates_are_count_weighted_and_reconciled() -> None:
    workers = [
        {
            "fastwam/eligible_gate_decision_count": 10.0,
            "fastwam/eligible_idm_decision_count": 2.0,
            "fastwam/gate/behavior_idm_probability_mean": 0.2,
            "fastwam/route/valid_chunk_count": 12.0,
            "fastwam/route/valid_idm_chunk_count": 6.0,
            "fastwam/route/forced_count": 2.0,
        },
        {
            "fastwam/eligible_gate_decision_count": 90.0,
            "fastwam/eligible_idm_decision_count": 54.0,
            "fastwam/gate/behavior_idm_probability_mean": 0.6,
            "fastwam/route/valid_chunk_count": 100.0,
            "fastwam/route/valid_idm_chunk_count": 60.0,
            "fastwam/route/forced_count": 10.0,
        },
    ]
    guard = {
        "eligible_gate_decision_count": 100,
        "eligible_idm_decision_count": 56,
        "eligible_idm_fraction": 0.56,
        "valid_chunk_count": 112,
        "valid_idm_chunk_count": 66,
        "forced_route_count": 12,
    }
    observation = aggregate_fastwam_idm_cost_observation(
        runner_step=0,
        actor_rollout_metrics=workers,
        guard_result=guard,
    )
    assert observation.eligible_expected_fraction == pytest.approx(0.56)
    assert observation.eligible_realized_fraction == pytest.approx(0.56)
    assert observation.executed_realized_fraction == pytest.approx(66 / 112)
    assert observation.forced_fraction == pytest.approx(12 / 112)
    assert observation.executed_realized_fraction != pytest.approx(
        observation.eligible_realized_fraction
    )

    mismatched = dict(guard, eligible_idm_decision_count=55)
    with pytest.raises(ValueError, match="do not reconcile"):
        aggregate_fastwam_idm_cost_observation(
            runner_step=0,
            actor_rollout_metrics=workers,
            guard_result=mismatched,
        )

    missing_expected = [dict(worker) for worker in workers]
    for worker in missing_expected:
        worker.pop("fastwam/gate/behavior_idm_probability_mean")
    observation_without_expected = aggregate_fastwam_idm_cost_observation(
        runner_step=0,
        actor_rollout_metrics=missing_expected,
        guard_result=guard,
    )
    expected_config = _dual_config()
    expected_config["rate"]["feedback"] = "expected_behavior_probability"
    expected_controller = ProjectedDualIDMCostController(
        expected_config,
        bootstrap_idm_cost=0.0,
    )
    expected_controller.decision_for_step(0)
    with pytest.raises(ValueError, match="unavailable"):
        expected_controller.observe_rollout(observation_without_expected)


def _zero_init_diagnostic_branch() -> dict:
    config = _dual_config(initial=0.0, learning_rate=0.1)
    config["profile"] = "upper_bound_zero_init"
    config["rate"]["feedback"] = "expected_behavior_probability"
    config["dual"]["ema_beta"] = 0.0
    config["dual"]["deadband"] = 0.0
    config["diagnostics"] = [
        {
            "type": "fair_break_even",
            "enabled": True,
            "diagnostic_only": True,
            "window_size": 5,
            "bootstrap_value_for_display": 0.03,
        }
    ]
    return {
        "enabled": True,
        "idm_cost": 0.015,
        "uncond_cost": 0.0,
        "fair_cost": {"enabled": False},
        "controller": config,
    }


def test_b50_slack_sequence_stays_zero_with_nonzero_fair_diagnostic() -> None:
    controller, explicit, _ = build_fastwam_idm_cost_controller(
        _zero_init_diagnostic_branch()
    )
    assert explicit
    for step, rate in enumerate(B50_SLACK_RATES):
        decision = controller.decision_for_step(step)
        assert decision.applied_idm_cost == 0.0
        record = controller.observe_rollout(
            _observation(step, rate, break_even=0.01 + step * 0.001)
        )
        assert record["applied"]["applied_idm_cost"] == 0.0
        assert record["next"]["applied_idm_cost"] == 0.0
        metrics = controller.record_metrics(record)
        assert metrics["fastwam/idm_cost_control/diagnostic_fair_cost"] > 0.0
        assert metrics["fastwam/idm_cost_control/applied_cost"] == 0.0
        assert metrics["fastwam/idm_cost_control/dual_multiplier"] == 0.0
        assert metrics["fastwam/idm_cost_control/applied_minus_dual"] == 0.0
        assert metrics["fastwam/idm_cost_control/diagnostic_only_fair"] == 1.0
        assert metrics["fastwam/idm_cost_control/profile_id"] == 1.0
        assert record["diagnostic_invariants"]["applied_equals_dual"] is True


def test_zero_init_positive_violation_branch_and_lag() -> None:
    config = _dual_config(initial=0.0, learning_rate=0.1, max_delta=0.05)
    config["rate"]["target_idm_fraction"] = 0.25
    controller = ProjectedDualIDMCostController(config, bootstrap_idm_cost=0.0)
    applied = []
    proposed = []
    for step, rate in enumerate((0.42, 0.38, 0.31, 0.27, 0.24)):
        decision = controller.decision_for_step(step)
        record = controller.observe_rollout(_observation(step, rate))
        applied.append(decision.applied_idm_cost)
        proposed.append(record["next"]["applied_idm_cost"])
    assert applied[0] == 0.0
    assert applied[1:] == pytest.approx(proposed[:-1])
    assert proposed[0] > applied[0]
    assert proposed[1] > proposed[0]
    assert proposed[-1] < proposed[-2]
    for step in range(5, 8):
        _, record = _advance(controller, step, 0.0)
    assert record["next"]["applied_idm_cost"] == 0.0
    _, recovery = _advance(controller, 8, 0.5)
    assert recovery["next"]["applied_idm_cost"] == pytest.approx(0.025)


def test_multiple_diagnostics_cannot_change_applied_or_next_cost() -> None:
    class StaticDiagnostic:
        def __init__(self, diagnostic_type: str, value: float) -> None:
            self.diagnostic_type = diagnostic_type
            self.value = value

        def decision_metadata_for_step(self, runner_step: int) -> dict[str, float]:
            return {"value": self.value + runner_step}

        def observe_rollout(self, observation) -> dict:
            return {"value": self.value, "runner_step": observation.runner_step}

        def record_metrics(self, record) -> dict[str, float]:
            return {f"diagnostic/{self.diagnostic_type}": float(record["value"])}

        def state_dict(self) -> dict:
            return {"value": self.value}

        def load_state_dict(self, state) -> None:
            self.value = float(state["value"])

    delegate = ProjectedDualIDMCostController(
        _dual_config(initial=0.0),
        bootstrap_idm_cost=0.0,
    )
    controller = DiagnosticIDMCostController(
        delegate,
        (StaticDiagnostic("first", 0.2), StaticDiagnostic("second", 0.7)),
    )
    decision = controller.decision_for_step(0)
    record = controller.observe_rollout(_observation(0, 0.8))
    assert decision.applied_idm_cost == 0.0
    assert record["applied"]["applied_idm_cost"] == 0.0
    assert record["next"]["applied_idm_cost"] == pytest.approx(0.03)


def test_diagnostic_state_round_trip_and_undefined_break_even_not_duplicated() -> None:
    branch = _zero_init_diagnostic_branch()
    source, _, _ = build_fastwam_idm_cost_controller(branch)
    source.decision_for_step(0)
    source.observe_rollout(_observation(0, 0.4, break_even=0.02))
    source.decision_for_step(1)
    source.observe_rollout(_observation(1, 0.4, break_even=None))
    state = copy.deepcopy(source.state_dict())
    diagnostic_state = state["diagnostics"]["fair_break_even"]["payload"]
    assert diagnostic_state["break_even_history"] == [0.02]

    restored, _, _ = build_fastwam_idm_cost_controller(branch)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.decision_for_step(2) == source.decision_for_step(2)


def test_duplicate_diagnostic_type_fails_fast() -> None:
    branch = _zero_init_diagnostic_branch()
    branch["controller"]["diagnostics"] *= 2
    with pytest.raises(ValueError, match="Duplicate FastWAM cost diagnostic"):
        build_fastwam_idm_cost_controller(branch)
