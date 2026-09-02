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

"""Generic non-negative FastWAM branch-cost controller contracts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from rlinf.runners.fastwam_cost_diagnostics import FastWAMCostDiagnostic

FASTWAM_BRANCH_COST_CONTROL_SCHEMA = "fastwam-branch-cost-control-v1"


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMBranchCostDecision:
    """Immutable IDM and UNCOND costs selected before one rollout."""

    runner_step: int
    controller_type: str
    phase: str
    idm_cost: float
    uncond_cost: float
    components: dict[str, float]

    def __post_init__(self) -> None:
        if isinstance(self.runner_step, bool) or self.runner_step < 0:
            raise ValueError("Branch-cost runner_step must be non-negative.")
        if not self.controller_type or not self.phase:
            raise ValueError("Branch-cost controller type and phase are required.")
        for name in ("idm_cost", "uncond_cost"):
            if _finite(getattr(self, name), name=name) < 0.0:
                raise ValueError("FastWAM branch costs must be non-negative.")
        for name, value in self.components.items():
            _finite(value, name=f"branch-cost component {name}")

    def to_artifact(self) -> dict[str, Any]:
        return {
            "runner_step": self.runner_step,
            "controller_type": self.controller_type,
            "phase": self.phase,
            "idm_cost": self.idm_cost,
            "uncond_cost": self.uncond_cost,
            "components": dict(self.components),
        }

    @property
    def applied_idm_cost(self) -> float:
        """Compatibility alias for existing IDM-only runner tests and tools."""

        return self.idm_cost


class FastWAMBranchCostController(Protocol):
    """Controller interface used by the generic runner runtime."""

    enabled: bool
    controller_type: str
    requires_rollout_feedback: bool
    requires_break_even_audit: bool
    observed_runner_steps: int

    def decision_for_step(self, runner_step: int) -> FastWAMBranchCostDecision: ...

    def observe_rollout(self, observation: Any) -> dict[str, Any]: ...

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class LegacyIDMCostControllerAdapter:
    """Expose an existing IDM-only controller as a branch-cost controller."""

    def __init__(self, delegate: Any, *, uncond_cost: float) -> None:
        self.delegate = delegate
        self.uncond_cost = _finite(uncond_cost, name="configured UNCOND cost")
        if self.uncond_cost < 0.0:
            raise ValueError("Configured UNCOND cost must be non-negative.")

    @property
    def enabled(self) -> bool:
        return bool(self.delegate.enabled)

    @property
    def controller_type(self) -> str:
        return str(self.delegate.controller_type)

    @property
    def requires_rollout_feedback(self) -> bool:
        return bool(self.delegate.requires_rollout_feedback)

    @property
    def requires_break_even_audit(self) -> bool:
        return bool(self.delegate.requires_break_even_audit)

    @property
    def observed_runner_steps(self) -> int:
        return int(self.delegate.observed_runner_steps)

    def decision_for_step(self, runner_step: int) -> FastWAMBranchCostDecision:
        decision = self.delegate.decision_for_step(runner_step)
        return FastWAMBranchCostDecision(
            runner_step=decision.runner_step,
            controller_type=decision.controller_type,
            phase=decision.phase,
            idm_cost=decision.applied_idm_cost,
            uncond_cost=self.uncond_cost,
            components=dict(decision.components),
        )

    def observe_rollout(self, observation: Any) -> dict[str, Any]:
        # Preserve the legacy unified IDM JSONL and metric schemas exactly.
        return self.delegate.observe_rollout(observation)

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        return self.delegate.record_metrics(record)

    def state_dict(self) -> dict[str, Any]:
        # Existing v3 checkpoints retain their exact controller payload.
        return self.delegate.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.delegate.load_state_dict(state)


class LaggedBranchCostControllerBase:
    """Enforce decision-before-observation and contiguous runner steps."""

    enabled = True
    requires_rollout_feedback = True

    def __init__(self) -> None:
        self.observed_runner_steps = 0
        self._pending: FastWAMBranchCostDecision | None = None

    def _build_decision(self, runner_step: int) -> FastWAMBranchCostDecision:
        raise NotImplementedError

    def _update_after_rollout(self, observation: Any) -> dict[str, Any]:
        raise NotImplementedError

    def decision_for_step(self, runner_step: int) -> FastWAMBranchCostDecision:
        if runner_step != self.observed_runner_steps:
            raise ValueError(
                f"{self.controller_type} step mismatch: requested {runner_step}, "
                f"expected {self.observed_runner_steps}."
            )
        if self._pending is not None:
            raise RuntimeError(
                f"{self.controller_type} already has a pending decision."
            )
        self._pending = self._build_decision(runner_step)
        return self._pending

    def observe_rollout(self, observation: Any) -> dict[str, Any]:
        if self._pending is None:
            raise RuntimeError(
                f"{self.controller_type} observation has no pending decision."
            )
        if int(observation.runner_step) != self._pending.runner_step:
            raise ValueError("Branch-cost observation step does not match decision.")
        applied = self._pending
        transition = self._update_after_rollout(observation)
        self.observed_runner_steps += 1
        self._pending = None
        next_decision = self._build_decision(self.observed_runner_steps)
        record = {
            "schema": FASTWAM_BRANCH_COST_CONTROL_SCHEMA,
            "controller_type": self.controller_type,
            "runner_step": int(observation.runner_step),
            "applied": applied.to_artifact(),
            "observed": {
                "eligible_realized_fraction": float(
                    observation.eligible_realized_fraction
                ),
                "eligible_expected_fraction": observation.eligible_expected_fraction,
                "executed_realized_fraction": float(
                    observation.executed_realized_fraction
                ),
                "forced_fraction": float(observation.forced_fraction),
                **dict(transition.pop("observed", {})),
            },
            "update": dict(transition.pop("update", {})),
            "next": next_decision.to_artifact(),
        }
        record.update(transition)
        return record


class SignedBandPriceController(LaggedBranchCostControllerBase):
    """Track an IDM-rate band with one projected signed route price."""

    controller_type = "band_price"
    requires_break_even_audit = False

    _FEEDBACK_ALIASES: ClassVar[dict[str, str]] = {
        "expected_behavior_probability": "eligible_expected",
        "eligible_expected": "eligible_expected",
        "eligible_realized": "eligible_realized",
        "realized_gate_decisions": "eligible_realized",
        "executed_realized": "executed_realized",
    }

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self._config = {str(key): value for key, value in config.items()}
        rate = dict(self._config.get("rate", {}))
        signed = dict(self._config.get("signed_price", {}))
        self.target_fraction = _finite(
            rate.get("target_idm_fraction"), name="target IDM fraction"
        )
        self.half_width = _finite(rate.get("half_width"), name="band half-width")
        feedback = str(rate.get("feedback", "expected_behavior_probability")).lower()
        if feedback not in self._FEEDBACK_ALIASES:
            raise ValueError(f"Unsupported FastWAM band feedback {feedback!r}.")
        self.feedback = self._FEEDBACK_ALIASES[feedback]
        self.learning_rate = _finite(
            signed.get("learning_rate"), name="signed-price learning rate"
        )
        self.ema_beta = _finite(signed.get("ema_beta", 0.0), name="EMA beta")
        self.update_interval = _positive_integer(
            signed.get("update_interval", 1), name="signed-price update interval"
        )
        self.max_abs_value = _finite(
            signed.get("max_abs_value"), name="maximum signed price"
        )
        self.max_delta = _finite(
            signed.get("max_delta_per_update"), name="maximum signed-price delta"
        )
        self.signed_price = _finite(
            signed.get("initial_value", 0.0), name="initial signed price"
        )
        self._validate_config()
        self.rate_ema: float | None = None
        self.updates_since_last_update = 0
        self.positive_projection_count = 0
        self.negative_projection_count = 0
        self.max_delta_clip_count = 0
        self.last_feedback_rate: float | None = None
        self.last_band_error = 0.0
        self.last_applied_delta = 0.0

    @property
    def lower_bound(self) -> float:
        return self.target_fraction - self.half_width

    @property
    def upper_bound(self) -> float:
        return self.target_fraction + self.half_width

    def _validate_config(self) -> None:
        if not 0.0 < self.target_fraction < 1.0:
            raise ValueError("Band target IDM fraction must lie in (0, 1).")
        if self.half_width <= 0.0 or not (
            0.0 <= self.lower_bound < self.upper_bound <= 1.0
        ):
            raise ValueError("FastWAM target-rate band is invalid.")
        if self.learning_rate <= 0.0:
            raise ValueError("Signed-price learning rate must be positive.")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("Signed-price EMA beta must lie in [0, 1).")
        if self.max_abs_value <= 0.0:
            raise ValueError("Maximum signed price must be positive.")
        if not 0.0 < self.max_delta <= 2.0 * self.max_abs_value:
            raise ValueError("Maximum signed-price delta is invalid.")
        if not -self.max_abs_value <= self.signed_price <= self.max_abs_value:
            raise ValueError("Initial signed price is outside projection bounds.")

    def _build_decision(self, runner_step: int) -> FastWAMBranchCostDecision:
        idm_cost = max(self.signed_price, 0.0)
        uncond_cost = max(-self.signed_price, 0.0)
        if min(idm_cost, uncond_cost) != 0.0 or not math.isclose(
            max(idm_cost, uncond_cost),
            abs(self.signed_price),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError("FastWAM band-price branch-cost identity failed.")
        return FastWAMBranchCostDecision(
            runner_step=runner_step,
            controller_type=self.controller_type,
            phase="band_tracking",
            idm_cost=idm_cost,
            uncond_cost=uncond_cost,
            components={
                "signed_price": self.signed_price,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
            },
        )

    def _feedback_rate(self, observation: Any) -> float:
        if self.feedback == "eligible_expected":
            if observation.eligible_expected_fraction is None:
                raise ValueError("Expected band-price feedback is unavailable.")
            return float(observation.eligible_expected_fraction)
        if self.feedback == "eligible_realized":
            return float(observation.eligible_realized_fraction)
        return float(observation.executed_realized_fraction)

    def _base_price_for_error(
        self,
        error: float,
    ) -> tuple[float, dict[str, float | bool]]:
        """Return the price base used by one scheduled controller update."""

        del error
        return self.signed_price, {}

    def _update_after_rollout(self, observation: Any) -> dict[str, Any]:
        feedback_rate = self._feedback_rate(observation)
        self.last_feedback_rate = feedback_rate
        self.rate_ema = (
            feedback_rate
            if self.rate_ema is None
            else self.ema_beta * self.rate_ema + (1.0 - self.ema_beta) * feedback_rate
        )
        if self.rate_ema > self.upper_bound:
            error = self.rate_ema - self.upper_bound
        elif self.rate_ema < self.lower_bound:
            error = self.rate_ema - self.lower_bound
        else:
            error = 0.0
        self.last_band_error = error
        raw_delta = 0.0
        applied_delta = 0.0
        clipped = False
        update_metadata: dict[str, float | bool] = {}
        self.updates_since_last_update += 1
        if self.updates_since_last_update >= self.update_interval:
            self.updates_since_last_update = 0
            base_price, update_metadata = self._base_price_for_error(error)
            feedback_delta = self.learning_rate * error
            raw_delta = base_price - self.signed_price + feedback_delta
            delta = min(self.max_delta, max(-self.max_delta, raw_delta))
            if delta != raw_delta:
                clipped = True
                self.max_delta_clip_count += 1
            unprojected = self.signed_price + delta
            projected = min(
                self.max_abs_value,
                max(-self.max_abs_value, unprojected),
            )
            if projected == self.max_abs_value and unprojected > self.max_abs_value:
                clipped = True
                self.positive_projection_count += 1
            if projected == -self.max_abs_value and unprojected < -self.max_abs_value:
                clipped = True
                self.negative_projection_count += 1
            applied_delta = projected - self.signed_price
            self.signed_price = projected
        self.last_applied_delta = applied_delta
        return {
            "observed": {
                "target_fraction": self.target_fraction,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "feedback_rate": feedback_rate,
                "rate_ema": self.rate_ema,
            },
            "update": {
                "band_error": error,
                "raw_delta": raw_delta,
                "applied_delta": applied_delta,
                "clipped": clipped,
                "inside_band": error == 0.0,
                **update_metadata,
            },
        }

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        applied = record["applied"]
        observed = record["observed"]
        update = record["update"]
        return {
            "fastwam/branch_cost_control/signed_price": float(
                applied["components"]["signed_price"]
            ),
            "fastwam/branch_cost_control/idm_cost": float(applied["idm_cost"]),
            "fastwam/branch_cost_control/uncond_cost": float(applied["uncond_cost"]),
            "fastwam/branch_cost_control/target_fraction": self.target_fraction,
            "fastwam/branch_cost_control/lower_bound": self.lower_bound,
            "fastwam/branch_cost_control/upper_bound": self.upper_bound,
            "fastwam/branch_cost_control/feedback_rate": float(
                observed["feedback_rate"]
            ),
            "fastwam/branch_cost_control/band_error": float(update["band_error"]),
            "fastwam/branch_cost_control/inside_band": float(update["inside_band"]),
            "fastwam/branch_cost_control/projection_hit": float(update["clipped"]),
        }

    def state_dict(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Cannot checkpoint a pending band-price decision.")
        return {
            "schema": "fastwam-band-price-controller-state-v1",
            "config": self._config,
            "observed_runner_steps": self.observed_runner_steps,
            "signed_price": self.signed_price,
            "rate_ema": self.rate_ema,
            "updates_since_last_update": self.updates_since_last_update,
            "positive_projection_count": self.positive_projection_count,
            "negative_projection_count": self.negative_projection_count,
            "max_delta_clip_count": self.max_delta_clip_count,
            "last_feedback_rate": self.last_feedback_rate,
            "last_band_error": self.last_band_error,
            "last_applied_delta": self.last_applied_delta,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "fastwam-band-price-controller-state-v1":
            raise ValueError("Band-price controller state schema mismatch.")
        if state.get("config") != self._config:
            raise ValueError("Band-price controller config mismatch.")
        observed = int(state.get("observed_runner_steps", -1))
        signed_price = _finite(state.get("signed_price"), name="restored signed price")
        rate_ema = state.get("rate_ema")
        if rate_ema is not None:
            rate_ema = _finite(rate_ema, name="restored rate EMA")
        interval_count = int(state.get("updates_since_last_update", -1))
        counters = [
            int(state.get(name, -1))
            for name in (
                "positive_projection_count",
                "negative_projection_count",
                "max_delta_clip_count",
            )
        ]
        if (
            observed < 0
            or not -self.max_abs_value <= signed_price <= self.max_abs_value
            or rate_ema is not None
            and not 0.0 <= rate_ema <= 1.0
            or not 0 <= interval_count < self.update_interval
            or any(value < 0 for value in counters)
        ):
            raise ValueError("Restored band-price controller state is invalid.")
        self.observed_runner_steps = observed
        self.signed_price = signed_price
        self.rate_ema = rate_ema
        self.updates_since_last_update = interval_count
        (
            self.positive_projection_count,
            self.negative_projection_count,
            self.max_delta_clip_count,
        ) = counters
        last_feedback_rate = state.get("last_feedback_rate")
        if last_feedback_rate is not None:
            last_feedback_rate = _finite(
                last_feedback_rate,
                name="restored feedback rate",
            )
            if not 0.0 <= last_feedback_rate <= 1.0:
                raise ValueError("Restored feedback rate lies outside [0, 1].")
        self.last_feedback_rate = last_feedback_rate
        self.last_band_error = _finite(
            state.get("last_band_error"), name="restored band error"
        )
        self.last_applied_delta = _finite(
            state.get("last_applied_delta"), name="restored applied delta"
        )
        self._pending = None


class ReversalDampedBandPriceController(SignedBandPriceController):
    """Release an opposite-sign historical price before applying band feedback."""

    controller_type = "band_price_reversal_damped"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        signed = dict(self._config.get("signed_price", {}))
        reversal = signed.get("reversal")
        if not isinstance(reversal, Mapping):
            raise TypeError("Reversal-damped band-price requires reversal config.")
        self.reversal_mode = str(reversal.get("mode", "")).lower()
        if self.reversal_mode != "opposing_decay":
            raise ValueError("Reversal-damped band-price requires mode=opposing_decay.")
        self.reversal_decay_factor = _finite(
            reversal.get("factor"),
            name="opposing signed-price decay factor",
        )
        if not 0.0 <= self.reversal_decay_factor < 1.0:
            raise ValueError("Opposing signed-price decay factor must lie in [0, 1).")
        self.reversal_decay_count = 0

    def _base_price_for_error(
        self,
        error: float,
    ) -> tuple[float, dict[str, float | bool]]:
        opposing = error * self.signed_price < 0.0
        base_price = self.signed_price
        if opposing:
            base_price *= self.reversal_decay_factor
            self.reversal_decay_count += 1
        return base_price, {
            "opposing_decay_applied": opposing,
            "reversal_decay_factor": self.reversal_decay_factor,
            "pre_decay_signed_price": self.signed_price,
            "post_decay_signed_price": base_price,
            "reversal_decay_delta": base_price - self.signed_price,
            "feedback_delta": self.learning_rate * error,
        }

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        metrics = super().record_metrics(record)
        update = record["update"]
        metrics.update(
            {
                "fastwam/branch_cost_control/opposing_decay_applied": float(
                    update.get("opposing_decay_applied", False)
                ),
                "fastwam/branch_cost_control/reversal_decay_factor": (
                    self.reversal_decay_factor
                ),
                "fastwam/branch_cost_control/reversal_decay_delta": float(
                    update.get("reversal_decay_delta", 0.0)
                ),
                "fastwam/branch_cost_control/reversal_decay_count": float(
                    self.reversal_decay_count
                ),
            }
        )
        return metrics

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["schema"] = "fastwam-reversal-damped-band-price-controller-state-v1"
        state["reversal_decay_count"] = self.reversal_decay_count
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema")
            != "fastwam-reversal-damped-band-price-controller-state-v1"
        ):
            raise ValueError(
                "Reversal-damped band-price controller state schema mismatch."
            )
        reversal_decay_count = int(state.get("reversal_decay_count", -1))
        if reversal_decay_count < 0:
            raise ValueError("Restored reversal decay count is invalid.")
        base_state = dict(state)
        base_state["schema"] = "fastwam-band-price-controller-state-v1"
        super().load_state_dict(base_state)
        self.reversal_decay_count = reversal_decay_count


class DiagnosticBranchCostController:
    """Attach diagnostics to a branch controller without changing either cost."""

    def __init__(
        self,
        delegate: FastWAMBranchCostController,
        diagnostics: Sequence[FastWAMCostDiagnostic],
    ) -> None:
        types = [diagnostic.diagnostic_type for diagnostic in diagnostics]
        if not diagnostics or len(types) != len(set(types)):
            raise ValueError("Branch diagnostics must be non-empty and unique.")
        self.delegate = delegate
        self.diagnostics = tuple(diagnostics)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    @property
    def requires_break_even_audit(self) -> bool:
        return True

    def decision_for_step(self, runner_step: int) -> FastWAMBranchCostDecision:
        decision = self.delegate.decision_for_step(runner_step)
        components = dict(decision.components)
        for diagnostic in self.diagnostics:
            for name, value in diagnostic.decision_metadata_for_step(
                runner_step
            ).items():
                components[f"diagnostic/{diagnostic.diagnostic_type}/{name}"] = float(
                    value
                )
        return FastWAMBranchCostDecision(
            runner_step=decision.runner_step,
            controller_type=decision.controller_type,
            phase=decision.phase,
            idm_cost=decision.idm_cost,
            uncond_cost=decision.uncond_cost,
            components=components,
        )

    def observe_rollout(self, observation: Any) -> dict[str, Any]:
        record = self.delegate.observe_rollout(observation)
        applied_idm = float(record["applied"]["idm_cost"])
        applied_uncond = float(record["applied"]["uncond_cost"])
        next_idm = float(record["next"]["idm_cost"])
        next_uncond = float(record["next"]["uncond_cost"])
        record["diagnostics"] = {
            diagnostic.diagnostic_type: diagnostic.observe_rollout(observation)
            for diagnostic in self.diagnostics
        }
        if (
            float(record["applied"]["idm_cost"]) != applied_idm
            or float(record["applied"]["uncond_cost"]) != applied_uncond
            or float(record["next"]["idm_cost"]) != next_idm
            or float(record["next"]["uncond_cost"]) != next_uncond
        ):
            raise RuntimeError("FastWAM diagnostics changed a branch-cost decision.")
        record["diagnostic_invariants"] = {
            "branch_costs_unchanged": True,
            "diagnostic_only": True,
        }
        return record

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        metrics = self.delegate.record_metrics(record)
        records = record.get("diagnostics")
        if not isinstance(records, Mapping):
            raise TypeError("Branch diagnostic records are missing.")
        for diagnostic in self.diagnostics:
            diagnostic_record = records.get(diagnostic.diagnostic_type)
            if not isinstance(diagnostic_record, Mapping):
                raise TypeError("Branch diagnostic record is malformed.")
            metrics.update(diagnostic.record_metrics(diagnostic_record))
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "fastwam-diagnostic-branch-cost-controller-state-v1",
            "controller_type": self.controller_type,
            "delegate": self.delegate.state_dict(),
            "diagnostics": {
                diagnostic.diagnostic_type: diagnostic.state_dict()
                for diagnostic in self.diagnostics
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "fastwam-diagnostic-branch-cost-controller-state-v1":
            raise ValueError("Diagnostic branch controller state schema mismatch.")
        if state.get("controller_type") != self.controller_type:
            raise ValueError("Diagnostic branch controller type mismatch.")
        delegate = state.get("delegate")
        diagnostics = state.get("diagnostics")
        if not isinstance(delegate, Mapping) or not isinstance(diagnostics, Mapping):
            raise TypeError("Diagnostic branch controller state is malformed.")
        if set(diagnostics) != {item.diagnostic_type for item in self.diagnostics}:
            raise ValueError("Diagnostic branch controller diagnostic set mismatch.")
        self.delegate.load_state_dict(delegate)
        for diagnostic in self.diagnostics:
            payload = diagnostics[diagnostic.diagnostic_type]
            if not isinstance(payload, Mapping):
                raise TypeError("Branch diagnostic state is malformed.")
            diagnostic.load_state_dict(payload)
        for diagnostic in self.diagnostics:
            payload = diagnostic.state_dict().get("payload", {})
            if int(payload.get("observed_runner_steps", -1)) != int(
                self.observed_runner_steps
            ):
                raise ValueError("Branch diagnostic and controller steps disagree.")


_BRANCH_CONTROLLER_REGISTRY: dict[
    str, Callable[[Mapping[str, Any]], FastWAMBranchCostController]
] = {}


def register_fastwam_branch_cost_controller(name: str):
    """Register a native branch-cost controller factory."""

    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("FastWAM branch-cost controller name cannot be empty.")

    def decorator(factory):
        if normalized in _BRANCH_CONTROLLER_REGISTRY:
            raise ValueError(f"FastWAM branch-cost controller {normalized!r} exists.")
        _BRANCH_CONTROLLER_REGISTRY[normalized] = factory
        return factory

    return decorator


@register_fastwam_branch_cost_controller("band_price")
def _build_band_price(config: Mapping[str, Any]) -> FastWAMBranchCostController:
    return SignedBandPriceController(config)


@register_fastwam_branch_cost_controller("band_price_reversal_damped")
def _build_reversal_damped_band_price(
    config: Mapping[str, Any],
) -> FastWAMBranchCostController:
    return ReversalDampedBandPriceController(config)


def is_fastwam_branch_cost_controller(name: str) -> bool:
    """Return whether a native branch-cost controller is registered."""

    return str(name).strip().lower() in _BRANCH_CONTROLLER_REGISTRY


def get_fastwam_branch_cost_controller(name: str):
    """Return a registered native branch-cost controller factory."""

    normalized = str(name).strip().lower()
    try:
        return _BRANCH_CONTROLLER_REGISTRY[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unsupported FastWAM branch-cost controller {normalized!r}."
        ) from error


def append_fastwam_branch_cost_control_jsonl(
    path: str | Path,
    record: Mapping[str, Any],
) -> None:
    """Append one native branch-cost transition to its run-scoped audit."""

    if record.get("schema") != FASTWAM_BRANCH_COST_CONTROL_SCHEMA:
        raise ValueError("FastWAM branch-cost control record schema mismatch.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(record), sort_keys=True, allow_nan=False)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "DiagnosticBranchCostController",
    "FASTWAM_BRANCH_COST_CONTROL_SCHEMA",
    "FastWAMBranchCostController",
    "FastWAMBranchCostDecision",
    "LegacyIDMCostControllerAdapter",
    "ReversalDampedBandPriceController",
    "SignedBandPriceController",
    "append_fastwam_branch_cost_control_jsonl",
    "get_fastwam_branch_cost_controller",
    "is_fastwam_branch_cost_controller",
    "register_fastwam_branch_cost_controller",
]
