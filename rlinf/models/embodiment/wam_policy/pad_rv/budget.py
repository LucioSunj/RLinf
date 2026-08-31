# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD-only prediction-budget dual with fair price as diagnostics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from rlinf.runners.fastwam_fair_cost import (
    FASTWAM_FAIR_COST_CONTROL_SCHEMA,
    FASTWAM_LAGRANGE_MULTIPLIER_MAX,
    FASTWAM_LAGRANGE_MULTIPLIER_MIN,
    FastWAMFairCostController,
    FastWAMFairCostDecision,
)

PAD_PREDICTION_BUDGET_CONTROLLER_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.budget.PadPredictionBudgetController"
)
PAD_PREDICTION_BUDGET_STATE_SCHEMA = "pad-prediction-budget-controller-state-v1"


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None or not hasattr(value, "items"):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


class PadPredictionBudgetController(FastWAMFairCostController):
    """Implement the Stage-1 constrained objective without a permanent fair cost.

    The inherited FastWAM controller remains untouched for legacy experiments.
    PAD uses the plan's projected dual update directly; the lagged break-even
    estimate is retained as telemetry and as the initial multiplier only.
    """

    def __init__(self, *, branch_cost: Any, prediction_budget: Any) -> None:
        branch = _mapping(branch_cost, name="PAD fixed branch cost")
        budget = _mapping(prediction_budget, name="PAD prediction budget")
        fair = _mapping(branch.get("fair_cost"), name="PAD fair-cost diagnostics")
        super().__init__(fair, bootstrap_idm_cost=branch.get("idm_cost", 0.0))
        if not self.enabled:
            raise ValueError("PAD prediction-budget diagnostics must be enabled.")
        if bool(_mapping(fair.get("pi"), name="PAD legacy PI config").get("enabled")):
            raise ValueError("PAD disables the inherited fair-cost PI controller.")
        if str(budget.get("controller_target", "")) != (
            PAD_PREDICTION_BUDGET_CONTROLLER_TARGET
        ):
            raise ValueError("PAD prediction-budget controller target changed.")
        if not bool(budget.get("enabled", False)):
            raise ValueError("PAD prediction-budget dual must be enabled.")

        self.target_idm_fraction = _finite(
            budget.get("target_idm_fraction"),
            name="PAD target IDM fraction",
        )
        self.integral_gain = _finite(
            budget.get("dual_lr"),
            name="PAD dual learning rate",
        )
        self.proportional_gain = _finite(
            budget.get("proportional_gain"),
            name="PAD proportional gain",
        )
        if not 0.0 < self.target_idm_fraction < 1.0:
            raise ValueError("PAD target IDM fraction must lie in (0, 1).")
        if self.integral_gain < 0.0:
            raise ValueError("PAD dual learning rate must be non-negative.")
        if self.proportional_gain != 0.0:
            raise ValueError("PAD's projected budget dual has no proportional term.")

        # Section 6.4 permits fair cost as a dual initial value. It must not be
        # added again after the multiplier reaches zero below budget.
        self.pi_enabled = False
        self._lagrange_multiplier = min(
            FASTWAM_LAGRANGE_MULTIPLIER_MAX,
            max(FASTWAM_LAGRANGE_MULTIPLIER_MIN, self.bootstrap_idm_cost),
        )
        self._integral_term = self._lagrange_multiplier
        self._config = {
            "controller_target": PAD_PREDICTION_BUDGET_CONTROLLER_TARGET,
            "bootstrap_idm_cost": self.bootstrap_idm_cost,
            "fair_cost_window_size": self.window_size,
            "target_idm_fraction": self.target_idm_fraction,
            "dual_lr": self.integral_gain,
            "proportional_gain": self.proportional_gain,
            "lagrange_multiplier_min": FASTWAM_LAGRANGE_MULTIPLIER_MIN,
            "lagrange_multiplier_max": FASTWAM_LAGRANGE_MULTIPLIER_MAX,
        }

    @classmethod
    def from_configs(
        cls,
        *,
        branch_cost: Any,
        prediction_budget: Any,
    ) -> "PadPredictionBudgetController":
        return cls(branch_cost=branch_cost, prediction_budget=prediction_budget)

    def decision_for_step(self, runner_step: int) -> FastWAMFairCostDecision:
        if isinstance(runner_step, bool) or int(runner_step) != runner_step:
            raise TypeError("PAD budget runner_step must be an integer.")
        runner_step = int(runner_step)
        if runner_step != self._observed_runner_steps:
            raise ValueError(
                "PAD budget step does not match controller state: "
                f"requested {runner_step}, expected {self._observed_runner_steps}."
            )
        return FastWAMFairCostDecision(
            runner_step=runner_step,
            fair_idm_cost=self._fair_idm_cost,
            lagrange_multiplier=self._lagrange_multiplier,
            applied_idm_cost=self._lagrange_multiplier,
            lagged_break_even_window=tuple(self._break_even_history),
            last_valid_break_even_idm_cost=self._last_valid_break_even_idm_cost,
            pi_enabled=False,
        )

    def observe_rollout(
        self,
        *,
        runner_step: int,
        break_even_idm_cost: float | None,
        idm_fraction: float,
    ) -> dict[str, Any]:
        applied = self.decision_for_step(runner_step)
        idm_fraction = _finite(idm_fraction, name="PAD observed IDM fraction")
        if not 0.0 <= idm_fraction <= 1.0:
            raise ValueError("PAD observed IDM fraction must lie in [0, 1].")

        carried_break_even = self._last_valid_break_even_idm_cost
        if break_even_idm_cost is not None:
            carried_break_even = _finite(
                break_even_idm_cost,
                name="PAD observed break-even IDM cost",
            )
            if carried_break_even < 0.0:
                raise ValueError("PAD observed break-even IDM cost is negative.")
            self._last_valid_break_even_idm_cost = carried_break_even
            self._break_even_history.append(carried_break_even)
            del self._break_even_history[
                : max(0, len(self._break_even_history) - self.window_size)
            ]
            self._fair_idm_cost = float(statistics.median(self._break_even_history))

        budget_error = idm_fraction - self.target_idm_fraction
        previous_multiplier = self._lagrange_multiplier
        self._lagrange_multiplier = min(
            FASTWAM_LAGRANGE_MULTIPLIER_MAX,
            max(
                FASTWAM_LAGRANGE_MULTIPLIER_MIN,
                previous_multiplier + self.integral_gain * budget_error,
            ),
        )
        self._integral_term = self._lagrange_multiplier
        self._observed_runner_steps += 1
        next_decision = self.decision_for_step(self._observed_runner_steps)
        return {
            "schema": FASTWAM_FAIR_COST_CONTROL_SCHEMA,
            "controller_mode": "pad_prediction_budget_dual",
            "runner_step": int(runner_step),
            "applied": applied.to_artifact(),
            "observed_break_even_idm_cost": break_even_idm_cost,
            "carried_break_even_idm_cost": carried_break_even,
            "observed_idm_fraction": idm_fraction,
            "pi_error": budget_error,
            "dual_update": self._lagrange_multiplier - previous_multiplier,
            "integral_term": self._integral_term,
            "next": next_decision.to_artifact(),
        }

    @staticmethod
    def record_metrics(record: Mapping[str, Any]) -> dict[str, float]:
        metrics = FastWAMFairCostController.record_metrics(record)
        metrics.update(
            {
                "fastwam/prediction_budget/error": float(record["pi_error"]),
                "fastwam/prediction_budget/dual_update": float(record["dual_update"]),
            }
        )
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": PAD_PREDICTION_BUDGET_STATE_SCHEMA,
            "config": dict(self._config),
            "observed_runner_steps": self._observed_runner_steps,
            "break_even_history": list(self._break_even_history),
            "last_valid_break_even_idm_cost": self._last_valid_break_even_idm_cost,
            "fair_idm_cost": self._fair_idm_cost,
            "lagrange_multiplier": self._lagrange_multiplier,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != PAD_PREDICTION_BUDGET_STATE_SCHEMA:
            raise ValueError("PAD prediction-budget state schema mismatch.")
        if state.get("config") != self._config:
            raise ValueError("PAD prediction-budget controller config mismatch.")
        observed = int(state.get("observed_runner_steps", -1))
        history = [
            _finite(item, name="PAD break-even history")
            for item in state.get("break_even_history", [])
        ]
        last_valid = state.get("last_valid_break_even_idm_cost")
        if last_valid is not None:
            last_valid = _finite(last_valid, name="PAD last valid break-even cost")
        fair = _finite(state.get("fair_idm_cost"), name="PAD fair IDM diagnostic")
        multiplier = _finite(
            state.get("lagrange_multiplier"),
            name="PAD budget multiplier",
        )
        if observed < 0 or len(history) > self.window_size:
            raise ValueError("PAD prediction-budget history is invalid.")
        if any(item < 0.0 for item in history):
            raise ValueError("PAD break-even history contains a negative value.")
        expected_fair = (
            float(statistics.median(history)) if history else self.bootstrap_idm_cost
        )
        if fair != expected_fair:
            raise ValueError("PAD fair IDM diagnostic is inconsistent.")
        if history and (last_valid is None or history[-1] != last_valid):
            raise ValueError("PAD carried break-even value is inconsistent.")
        if not (
            FASTWAM_LAGRANGE_MULTIPLIER_MIN
            <= multiplier
            <= FASTWAM_LAGRANGE_MULTIPLIER_MAX
        ):
            raise ValueError("PAD budget multiplier is out of range.")

        self._observed_runner_steps = observed
        self._break_even_history = history
        self._last_valid_break_even_idm_cost = last_valid
        self._fair_idm_cost = fair
        self._lagrange_multiplier = multiplier
        self._integral_term = multiplier


__all__ = [
    "PAD_PREDICTION_BUDGET_CONTROLLER_TARGET",
    "PAD_PREDICTION_BUDGET_STATE_SCHEMA",
    "PadPredictionBudgetController",
]
