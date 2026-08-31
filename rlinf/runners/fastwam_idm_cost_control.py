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

"""Config-selected, lagged IDM cost control for FastWAM training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from omegaconf import OmegaConf

from rlinf.runners.fastwam_fair_cost import (
    FASTWAM_FAIR_COST_CONTROL_SCHEMA,
    FastWAMFairCostController,
    append_fastwam_fair_cost_control_jsonl,
)

FASTWAM_IDM_COST_CONTROL_SCHEMA = "fastwam-idm-cost-control-v1"
FASTWAM_IDM_COST_CONTROLLER_STATE_SCHEMA = "fastwam-idm-cost-controller-state-v1"


def _resolved_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    resolved = (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )
    if not isinstance(resolved, Mapping):
        raise TypeError(f"{name} must resolve to a mapping.")
    return {str(key): item for key, item in resolved.items()}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMIDMCostObservation:
    """Count-reconciled route and price statistics from one rollout."""

    runner_step: int
    eligible_gate_decision_count: int
    eligible_idm_decision_count: int
    eligible_realized_fraction: float
    eligible_expected_fraction: float | None
    valid_chunk_count: int
    valid_idm_chunk_count: int
    executed_realized_fraction: float
    forced_fraction: float
    break_even_idm_cost: float | None
    configured_idm_cost: float | None

    def __post_init__(self) -> None:
        _nonnegative_int(self.runner_step, name="runner_step")
        eligible = _nonnegative_int(
            self.eligible_gate_decision_count,
            name="eligible_gate_decision_count",
        )
        eligible_idm = _nonnegative_int(
            self.eligible_idm_decision_count,
            name="eligible_idm_decision_count",
        )
        valid = _nonnegative_int(self.valid_chunk_count, name="valid_chunk_count")
        valid_idm = _nonnegative_int(
            self.valid_idm_chunk_count,
            name="valid_idm_chunk_count",
        )
        if eligible < 1 or eligible_idm > eligible:
            raise ValueError("Eligible FastWAM route counts are invalid.")
        if valid < 1 or valid_idm > valid:
            raise ValueError("Executed FastWAM route counts are invalid.")
        fractions = {
            "eligible_realized_fraction": self.eligible_realized_fraction,
            "executed_realized_fraction": self.executed_realized_fraction,
            "forced_fraction": self.forced_fraction,
        }
        if self.eligible_expected_fraction is not None:
            fractions["eligible_expected_fraction"] = self.eligible_expected_fraction
        for name, value in fractions.items():
            numeric = _finite_float(value, name=name)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if not math.isclose(
            self.eligible_realized_fraction,
            eligible_idm / eligible,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Eligible FastWAM count/fraction values disagree.")
        if not math.isclose(
            self.executed_realized_fraction,
            valid_idm / valid,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Executed FastWAM count/fraction values disagree.")
        for name in ("break_even_idm_cost", "configured_idm_cost"):
            value = getattr(self, name)
            if value is not None and _finite_float(value, name=name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMIDMCostDecision:
    """The immutable cost selected before one rollout."""

    runner_step: int
    controller_type: str
    phase: str
    applied_idm_cost: float
    components: dict[str, float]

    def __post_init__(self) -> None:
        _nonnegative_int(self.runner_step, name="runner_step")
        if not self.controller_type or not self.phase:
            raise ValueError("Controller type and phase must be non-empty.")
        if _finite_float(self.applied_idm_cost, name="applied_idm_cost") < 0.0:
            raise ValueError("Applied IDM cost must be non-negative.")
        for name, value in self.components.items():
            _finite_float(value, name=f"decision component {name}")

    def to_artifact(self) -> dict[str, Any]:
        return {
            "runner_step": self.runner_step,
            "controller_type": self.controller_type,
            "phase": self.phase,
            "applied_idm_cost": self.applied_idm_cost,
            "components": dict(self.components),
        }


class FastWAMIDMCostController(Protocol):
    """Controller interface consumed by the generic runner facade."""

    enabled: bool
    controller_type: str
    requires_rollout_feedback: bool
    requires_break_even_audit: bool
    observed_runner_steps: int

    def decision_for_step(self, runner_step: int) -> FastWAMIDMCostDecision: ...

    def observe_rollout(
        self, observation: FastWAMIDMCostObservation
    ) -> dict[str, Any]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]: ...


class LaggedIDMCostControllerBase:
    """Enforce the one-decision/one-observation lifecycle for feedback control."""

    enabled = True
    requires_rollout_feedback = True

    def __init__(self) -> None:
        self.observed_runner_steps = 0
        self._pending: FastWAMIDMCostDecision | None = None

    def _build_decision(self, runner_step: int) -> FastWAMIDMCostDecision:
        raise NotImplementedError

    def _update_after_rollout(
        self,
        observation: FastWAMIDMCostObservation,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def decision_for_step(self, runner_step: int) -> FastWAMIDMCostDecision:
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

    def observe_rollout(
        self,
        observation: FastWAMIDMCostObservation,
    ) -> dict[str, Any]:
        if self._pending is None:
            raise RuntimeError(
                f"{self.controller_type} observation has no pending decision."
            )
        if self._pending.runner_step != observation.runner_step:
            raise ValueError(
                f"{self.controller_type} observation step mismatch: got "
                f"{observation.runner_step}, expected {self._pending.runner_step}."
            )
        applied = self._pending
        transition = self._update_after_rollout(observation)
        self.observed_runner_steps += 1
        self._pending = None
        next_decision = self._build_decision(self.observed_runner_steps)
        record = {
            "schema": FASTWAM_IDM_COST_CONTROL_SCHEMA,
            "controller_type": self.controller_type,
            "runner_step": observation.runner_step,
            "applied": applied.to_artifact(),
            "observed": {
                **_observation_artifact(observation),
                **dict(transition.pop("observed", {})),
            },
            "update": dict(transition.pop("update", {})),
            "next": next_decision.to_artifact(),
        }
        record.update(transition)
        return record


class _DisabledController:
    enabled = False
    controller_type = "disabled"
    requires_rollout_feedback = False
    requires_break_even_audit = False
    observed_runner_steps = 0

    def decision_for_step(self, runner_step: int) -> FastWAMIDMCostDecision:
        raise RuntimeError("Disabled FastWAM cost control has no decisions.")

    def observe_rollout(self, observation: FastWAMIDMCostObservation) -> dict[str, Any]:
        raise RuntimeError("Disabled FastWAM cost control has no observations.")

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError("Disabled FastWAM cost control has no state.")

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        return {}


class FixedIDMCostController:
    """Explicit fixed-cost controller with checkpoint-aligned step tracking."""

    enabled = True
    controller_type = "fixed"
    requires_rollout_feedback = False
    requires_break_even_audit = False

    def __init__(self, config: Mapping[str, Any], *, idm_cost: float) -> None:
        del config
        self.idm_cost = _finite_float(idm_cost, name="fixed IDM cost")
        if self.idm_cost < 0.0:
            raise ValueError("Fixed IDM cost must be non-negative.")
        self.observed_runner_steps = 0

    def decision_for_step(self, runner_step: int) -> FastWAMIDMCostDecision:
        if runner_step != self.observed_runner_steps:
            raise ValueError(
                "Fixed IDM cost step mismatch: "
                f"requested {runner_step}, expected {self.observed_runner_steps}."
            )
        self.observed_runner_steps += 1
        return FastWAMIDMCostDecision(
            runner_step=runner_step,
            controller_type=self.controller_type,
            phase="fixed",
            applied_idm_cost=self.idm_cost,
            components={"fixed_cost": self.idm_cost},
        )

    def observe_rollout(self, observation: FastWAMIDMCostObservation) -> dict[str, Any]:
        raise RuntimeError("Fixed IDM cost control does not consume feedback.")

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "fastwam-fixed-idm-cost-controller-state-v1",
            "idm_cost": self.idm_cost,
            "observed_runner_steps": self.observed_runner_steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "fastwam-fixed-idm-cost-controller-state-v1":
            raise ValueError("Fixed IDM cost controller state schema mismatch.")
        if float(state.get("idm_cost", math.nan)) != self.idm_cost:
            raise ValueError("Fixed IDM cost controller config mismatch.")
        observed = int(state.get("observed_runner_steps", -1))
        if observed < 0:
            raise ValueError("Fixed IDM cost observed step count is invalid.")
        self.observed_runner_steps = observed

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        return {}


class LegacyFairCostAdapter:
    """Thin protocol adapter preserving the legacy fair/PI implementation."""

    enabled = True
    requires_rollout_feedback = True
    requires_break_even_audit = True

    def __init__(
        self,
        legacy_controller: FastWAMFairCostController,
        *,
        controller_type: str | None = None,
    ) -> None:
        if not legacy_controller.enabled:
            raise ValueError("Legacy fair-cost adapter requires an enabled controller.")
        self.legacy_controller = legacy_controller
        self.controller_type = controller_type or (
            "legacy_fair_pi" if legacy_controller.pi_enabled else "legacy_fair"
        )
        self._pending: FastWAMIDMCostDecision | None = None

    @property
    def observed_runner_steps(self) -> int:
        return int(self.legacy_controller.state_dict()["observed_runner_steps"])

    def decision_for_step(self, runner_step: int) -> FastWAMIDMCostDecision:
        if self._pending is not None:
            raise RuntimeError("Legacy fair-cost decision is already pending.")
        legacy = self.legacy_controller.decision_for_step(runner_step)
        decision = FastWAMIDMCostDecision(
            runner_step=runner_step,
            controller_type=self.controller_type,
            phase="legacy_fair_pi" if legacy.pi_enabled else "legacy_fair",
            applied_idm_cost=legacy.applied_idm_cost,
            components={
                "fair_estimate": legacy.fair_idm_cost,
                "lagrange_multiplier": legacy.lagrange_multiplier,
            },
        )
        self._pending = decision
        return decision

    def observe_rollout(self, observation: FastWAMIDMCostObservation) -> dict[str, Any]:
        if (
            self._pending is None
            or self._pending.runner_step != observation.runner_step
        ):
            raise RuntimeError("Legacy fair-cost observation has no pending decision.")
        legacy_record = self.legacy_controller.observe_rollout(
            runner_step=observation.runner_step,
            break_even_idm_cost=observation.break_even_idm_cost,
            idm_fraction=observation.eligible_realized_fraction,
        )
        applied = self._pending
        self._pending = None
        next_legacy = legacy_record["next"]
        next_decision = FastWAMIDMCostDecision(
            runner_step=observation.runner_step + 1,
            controller_type=self.controller_type,
            phase=("legacy_fair_pi" if next_legacy["pi_enabled"] else "legacy_fair"),
            applied_idm_cost=float(next_legacy["applied_idm_cost"]),
            components={
                "fair_estimate": float(next_legacy["fair_idm_cost"]),
                "lagrange_multiplier": float(next_legacy["lagrange_multiplier"]),
            },
        )
        return {
            "schema": FASTWAM_IDM_COST_CONTROL_SCHEMA,
            "controller_type": self.controller_type,
            "runner_step": observation.runner_step,
            "applied": applied.to_artifact(),
            "observed": _observation_artifact(observation),
            "update": {
                "rate_error": float(legacy_record["pi_error"]),
                "raw_delta": 0.0,
                "applied_delta": (
                    next_decision.applied_idm_cost - applied.applied_idm_cost
                ),
                "clipped": False,
            },
            "next": next_decision.to_artifact(),
            "legacy_record": legacy_record,
        }

    def state_dict(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Cannot checkpoint a pending legacy fair-cost decision.")
        return {
            "schema": "fastwam-legacy-idm-cost-adapter-state-v1",
            "controller_type": self.controller_type,
            "legacy": self.legacy_controller.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "fastwam-legacy-idm-cost-adapter-state-v1":
            raise ValueError("Legacy IDM cost adapter state schema mismatch.")
        if state.get("controller_type") != self.controller_type:
            raise ValueError("Legacy IDM cost adapter type mismatch.")
        legacy = state.get("legacy")
        if not isinstance(legacy, Mapping):
            raise TypeError("Legacy fair-cost adapter state is malformed.")
        self.legacy_controller.load_state_dict(legacy)
        self._pending = None

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        legacy = record.get("legacy_record")
        if not isinstance(legacy, Mapping):
            raise TypeError("Legacy fair-cost record is missing.")
        metrics = self.legacy_controller.record_metrics(legacy)
        metrics.update(_common_record_metrics(record))
        return metrics


class ProjectedDualIDMCostController(LaggedIDMCostControllerBase):
    """Projected dual ascent with constant or lagged fair-price initialization."""

    enabled = True
    controller_type = "budget_dual"
    requires_rollout_feedback = True

    _FEEDBACK_ALIASES: ClassVar[dict[str, str]] = {
        "expected_behavior_probability": "eligible_expected",
        "eligible_expected": "eligible_expected",
        "eligible_realized": "eligible_realized",
        "realized_gate_decisions": "eligible_realized",
        "executed_realized": "executed_realized",
    }

    def __init__(self, config: Mapping[str, Any], *, bootstrap_idm_cost: float) -> None:
        super().__init__()
        self._config = _resolved_mapping(config, name="budget-dual controller config")
        rate = _resolved_mapping(self._config.get("rate"), name="rate config")
        dual = _resolved_mapping(self._config.get("dual"), name="dual config")
        initializer = _resolved_mapping(
            self._config.get("initializer"), name="initializer config"
        )
        self.target_fraction = _finite_float(
            rate.get("target_idm_fraction"), name="target IDM fraction"
        )
        feedback = str(rate.get("feedback", "expected_behavior_probability")).lower()
        if feedback not in self._FEEDBACK_ALIASES:
            raise ValueError(f"Unsupported FastWAM dual feedback {feedback!r}.")
        self.feedback = self._FEEDBACK_ALIASES[feedback]
        self.learning_rate = _finite_float(
            dual.get("learning_rate"), name="dual learning rate"
        )
        self.ema_beta = _finite_float(dual.get("ema_beta", 0.0), name="EMA beta")
        self.deadband = _finite_float(dual.get("deadband", 0.0), name="deadband")
        self.update_interval = _positive_int(
            dual.get("update_interval", 1),
            name="dual update interval",
        )
        self.min_cost = _finite_float(
            dual.get("min_idm_cost", 0.0), name="minimum IDM cost"
        )
        self.max_cost = _finite_float(dual.get("max_idm_cost"), name="maximum IDM cost")
        self.max_delta = _finite_float(
            dual.get("max_delta_per_update"), name="maximum IDM cost delta"
        )
        self.initializer_type = str(initializer.get("type", "constant")).lower()
        self.requires_break_even_audit = self.initializer_type == "break_even_median"
        self._validate_config()
        self.rate_ema: float | None = None
        self.updates_since_last_dual_update = 0
        self.min_projection_count = 0
        self.max_projection_count = 0
        self.max_delta_clip_count = 0
        self.transition_runner_step: int | None = None
        self._fair_estimator: FastWAMFairCostController | None = None
        self._warmup_rollouts = 0
        self._minimum_valid_observations = 0
        self._monitor_fair = False
        self._insufficient_data_policy = "keep_bootstrap"

        if self.initializer_type == "constant":
            initial = _finite_float(
                initializer.get("idm_cost", bootstrap_idm_cost),
                name="constant initial IDM cost",
            )
            self.current_dual_cost = min(self.max_cost, max(self.min_cost, initial))
            self.phase = "dual"
        elif self.initializer_type == "break_even_median":
            bootstrap = _finite_float(
                initializer.get("bootstrap_idm_cost", bootstrap_idm_cost),
                name="fair warm-start bootstrap IDM cost",
            )
            window_size = _positive_int(
                initializer.get("window_size", 5),
                name="fair initializer window size",
            )
            self._warmup_rollouts = _positive_int(
                initializer.get("warmup_rollouts", 5),
                name="fair initializer warmup_rollouts",
            )
            self._minimum_valid_observations = _positive_int(
                initializer.get("minimum_valid_observations", 3),
                name="fair initializer minimum_valid_observations",
            )
            self._monitor_fair = bool(initializer.get("monitor_after_warmup", True))
            self._insufficient_data_policy = str(
                initializer.get("insufficient_data_policy", "keep_bootstrap")
            ).lower()
            if not 1 <= self._minimum_valid_observations <= window_size:
                raise ValueError(
                    "Fair initializer minimum_valid_observations must lie in "
                    "[1, window_size]."
                )
            if self._insufficient_data_policy != "keep_bootstrap":
                raise ValueError(
                    "Only insufficient_data_policy=keep_bootstrap is supported."
                )
            self._fair_estimator = FastWAMFairCostController(
                {
                    "enabled": True,
                    "window_size": window_size,
                    "pi": {"enabled": False},
                },
                bootstrap_idm_cost=bootstrap,
            )
            self.current_dual_cost = min(self.max_cost, max(self.min_cost, bootstrap))
            self.phase = "warmstart"
        else:
            raise ValueError(
                f"Unsupported FastWAM dual initializer {self.initializer_type!r}."
            )

    def _validate_config(self) -> None:
        if not 0.0 < self.target_fraction < 1.0:
            raise ValueError("Target IDM fraction must lie in (0, 1).")
        if self.learning_rate <= 0.0:
            raise ValueError("Dual learning rate must be positive.")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("EMA beta must lie in [0, 1).")
        if not 0.0 <= self.deadband < 1.0:
            raise ValueError("Dual deadband must lie in [0, 1).")
        if self.update_interval < 1:
            raise ValueError("Dual update interval must be positive.")
        if not 0.0 <= self.min_cost < self.max_cost:
            raise ValueError("Dual projection bounds are invalid.")
        if not 0.0 < self.max_delta <= self.max_cost - self.min_cost:
            raise ValueError("Dual maximum delta is invalid.")

    def _fair_decision(self, runner_step: int) -> FastWAMIDMCostDecision:
        if self._fair_estimator is None:
            raise RuntimeError("Fair warm-start estimator is missing.")
        fair = self._fair_estimator.decision_for_step(runner_step)
        return FastWAMIDMCostDecision(
            runner_step=runner_step,
            controller_type=self.controller_type,
            phase="warmstart",
            applied_idm_cost=fair.applied_idm_cost,
            components={
                "fair_estimate": fair.fair_idm_cost,
                "bootstrap_cost": self._fair_estimator.bootstrap_idm_cost,
                "dual_multiplier": self.current_dual_cost,
            },
        )

    def _peek_dual_decision(self, runner_step: int) -> FastWAMIDMCostDecision:
        fair_estimate = 0.0
        if self._fair_estimator is not None:
            fair_estimate = float(self._fair_estimator.state_dict()["fair_idm_cost"])
        return FastWAMIDMCostDecision(
            runner_step=runner_step,
            controller_type=self.controller_type,
            phase="dual",
            applied_idm_cost=self.current_dual_cost,
            components={
                "dual_multiplier": self.current_dual_cost,
                "fair_estimate": fair_estimate,
            },
        )

    def _build_decision(self, runner_step: int) -> FastWAMIDMCostDecision:
        return (
            self._fair_decision(runner_step)
            if self.phase == "warmstart"
            else self._peek_dual_decision(runner_step)
        )

    def _feedback_rate(self, observation: FastWAMIDMCostObservation) -> float:
        if self.feedback == "eligible_expected":
            if observation.eligible_expected_fraction is None:
                raise ValueError(
                    "Expected-behavior feedback was selected but is unavailable."
                )
            return observation.eligible_expected_fraction
        if self.feedback == "eligible_realized":
            return observation.eligible_realized_fraction
        return observation.executed_realized_fraction

    def _observe_fair(
        self, observation: FastWAMIDMCostObservation
    ) -> dict[str, Any] | None:
        if self._fair_estimator is None:
            return None
        if self.phase != "warmstart" and not self._monitor_fair:
            return None
        return self._fair_estimator.observe_rollout(
            runner_step=observation.runner_step,
            break_even_idm_cost=observation.break_even_idm_cost,
            idm_fraction=observation.eligible_realized_fraction,
        )

    def _update_after_rollout(
        self, observation: FastWAMIDMCostObservation
    ) -> dict[str, Any]:
        fair_record = self._observe_fair(observation)
        feedback_rate = self._feedback_rate(observation)
        raw_error = feedback_rate - self.target_fraction
        feedback_error = raw_error
        raw_delta = 0.0
        applied_delta = 0.0
        update_clipped = False
        transitioned = False

        if self.phase == "warmstart":
            if fair_record is None or self._fair_estimator is None:
                raise RuntimeError("Fair warm-start observation was not recorded.")
            fair_state = self._fair_estimator.state_dict()
            valid_count = len(fair_state["break_even_history"])
            completed = observation.runner_step + 1
            if (
                completed >= self._warmup_rollouts
                and valid_count >= self._minimum_valid_observations
            ):
                next_fair = float(fair_record["next"]["fair_idm_cost"])
                self.current_dual_cost = min(
                    self.max_cost, max(self.min_cost, next_fair)
                )
                self.phase = "dual"
                self.transition_runner_step = observation.runner_step + 1
                transitioned = True
        else:
            self.rate_ema = (
                feedback_rate
                if self.rate_ema is None
                else self.ema_beta * self.rate_ema
                + (1.0 - self.ema_beta) * feedback_rate
            )
            raw_error = self.rate_ema - self.target_fraction
            feedback_error = 0.0 if abs(raw_error) <= self.deadband else raw_error
            self.updates_since_last_dual_update += 1
            if self.updates_since_last_dual_update >= self.update_interval:
                self.updates_since_last_dual_update = 0
                raw_delta = self.learning_rate * feedback_error
                applied_delta = min(self.max_delta, max(-self.max_delta, raw_delta))
                if applied_delta != raw_delta:
                    self.max_delta_clip_count += 1
                    update_clipped = True
                unprojected = self.current_dual_cost + applied_delta
                projected = min(self.max_cost, max(self.min_cost, unprojected))
                if projected == self.min_cost and unprojected < self.min_cost:
                    self.min_projection_count += 1
                    update_clipped = True
                if projected == self.max_cost and unprojected > self.max_cost:
                    self.max_projection_count += 1
                    update_clipped = True
                applied_delta = projected - self.current_dual_cost
                self.current_dual_cost = projected

        transition: dict[str, Any] = {
            "observed": {
                "target_fraction": self.target_fraction,
                "feedback_rate": feedback_rate,
            },
            "update": {
                "rate_error": raw_error,
                "feedback_error": feedback_error,
                "raw_delta": raw_delta,
                "applied_delta": applied_delta,
                "clipped": update_clipped,
                "transitioned": transitioned,
            },
        }
        if fair_record is not None:
            transition["fair_diagnostic"] = fair_record
        return transition

    def state_dict(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Cannot checkpoint a pending budget-dual decision.")
        return {
            "schema": "fastwam-budget-dual-controller-state-v1",
            "config": self._config,
            "observed_runner_steps": self.observed_runner_steps,
            "phase": self.phase,
            "current_dual_cost": self.current_dual_cost,
            "rate_ema": self.rate_ema,
            "updates_since_last_dual_update": self.updates_since_last_dual_update,
            "transition_runner_step": self.transition_runner_step,
            "min_projection_count": self.min_projection_count,
            "max_projection_count": self.max_projection_count,
            "max_delta_clip_count": self.max_delta_clip_count,
            "initializer_state": (
                None
                if self._fair_estimator is None
                else self._fair_estimator.state_dict()
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "fastwam-budget-dual-controller-state-v1":
            raise ValueError("Budget-dual controller state schema mismatch.")
        if state.get("config") != self._config:
            raise ValueError("Budget-dual controller config mismatch.")
        observed = int(state.get("observed_runner_steps", -1))
        phase = str(state.get("phase", ""))
        current = _finite_float(
            state.get("current_dual_cost"), name="restored dual multiplier"
        )
        rate_ema = state.get("rate_ema")
        if rate_ema is not None:
            rate_ema = _finite_float(rate_ema, name="restored rate EMA")
            if not 0.0 <= rate_ema <= 1.0:
                raise ValueError("Restored rate EMA lies outside [0, 1].")
        if observed < 0 or phase not in {"warmstart", "dual"}:
            raise ValueError("Budget-dual controller counters are invalid.")
        if not self.min_cost <= current <= self.max_cost:
            raise ValueError("Restored dual multiplier is outside projection bounds.")
        counters = {
            name: int(state.get(name, -1))
            for name in (
                "updates_since_last_dual_update",
                "min_projection_count",
                "max_projection_count",
                "max_delta_clip_count",
            )
        }
        if any(value < 0 for value in counters.values()) or (
            counters["updates_since_last_dual_update"] >= self.update_interval
        ):
            raise ValueError("Budget-dual update counters are invalid.")
        initializer_state = state.get("initializer_state")
        if self._fair_estimator is None:
            if initializer_state is not None or phase != "dual":
                raise ValueError("Constant initializer state is inconsistent.")
        else:
            if not isinstance(initializer_state, Mapping):
                raise TypeError("Fair initializer state is malformed.")
            self._fair_estimator.load_state_dict(initializer_state)
            fair_observed = int(initializer_state["observed_runner_steps"])
            fair_must_be_current = phase == "warmstart" or self._monitor_fair
            if fair_must_be_current and fair_observed != observed:
                raise ValueError("Fair initializer and dual steps disagree.")
            if not fair_must_be_current and fair_observed > observed:
                raise ValueError("Fair initializer is ahead of dual steps.")
        transition = state.get("transition_runner_step")
        if transition is not None:
            transition = int(transition)
            if transition < 1 or transition > observed:
                raise ValueError("Budget-dual transition step is invalid.")
        self.observed_runner_steps = observed
        self.phase = phase
        self.current_dual_cost = current
        self.rate_ema = rate_ema
        self.transition_runner_step = transition
        for name, value in counters.items():
            setattr(self, name, value)
        self._pending = None

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        metrics = _common_record_metrics(record)
        observed = record["observed"]
        update = record["update"]
        next_decision = record["next"]
        components = next_decision["components"]
        metrics.update(
            {
                "fastwam/idm_cost_control/dual_multiplier": float(
                    components["dual_multiplier"]
                ),
                "fastwam/idm_cost_control/dual_delta": float(update["applied_delta"]),
                "fastwam/idm_cost_control/target_fraction": self.target_fraction,
                "fastwam/idm_cost_control/raw_error": float(update["rate_error"]),
                "fastwam/idm_cost_control/feedback_error": float(
                    update["feedback_error"]
                ),
                "fastwam/idm_cost_control/rate_ema": float(
                    observed["feedback_rate"]
                    if self.rate_ema is None
                    else self.rate_ema
                ),
                "fastwam/idm_cost_control/at_min_cost": float(
                    self.current_dual_cost == self.min_cost
                ),
                "fastwam/idm_cost_control/at_max_cost": float(
                    self.current_dual_cost == self.max_cost
                ),
                "fastwam/idm_cost_control/update_clipped": float(update["clipped"]),
                "fastwam/idm_cost_control/phase_id": float(
                    next_decision["phase"] == "dual"
                ),
            }
        )
        fair = record.get("fair_diagnostic")
        if isinstance(fair, Mapping):
            next_fair = fair["next"]
            metrics["fastwam/idm_cost_control/diagnostic_fair_cost"] = float(
                next_fair["fair_idm_cost"]
            )
            metrics["fastwam/idm_cost_control/warmstart_valid_count"] = float(
                len(next_fair["lagged_break_even_window"])
            )
        return metrics


_CONTROLLER_REGISTRY: dict[
    str, Callable[[Mapping[str, Any], Mapping[str, Any]], FastWAMIDMCostController]
] = {}


def register_fastwam_idm_cost_controller(name: str):
    """Register an explicit FastWAM IDM cost controller factory."""

    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("FastWAM IDM cost controller name cannot be empty.")

    def decorator(factory):
        if normalized in _CONTROLLER_REGISTRY:
            raise ValueError(f"FastWAM IDM cost controller {normalized!r} exists.")
        _CONTROLLER_REGISTRY[normalized] = factory
        return factory

    return decorator


def get_fastwam_idm_cost_controller(name: str):
    normalized = str(name).strip().lower()
    try:
        return _CONTROLLER_REGISTRY[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unsupported FastWAM IDM cost controller {normalized!r}."
        ) from error


@register_fastwam_idm_cost_controller("fixed")
def _build_fixed(config, branch_cost):
    return FixedIDMCostController(config, idm_cost=branch_cost.get("idm_cost", 0.0))


def _build_explicit_legacy(config, branch_cost, *, pi_enabled: bool):
    fair_config = {
        "enabled": True,
        "window_size": int(config.get("window_size", 5)),
        "pi": {
            "enabled": pi_enabled,
            "target_idm_fraction": config.get("target_idm_fraction", 0.5),
            "integral_gain": config.get("integral_gain", 0.05),
            "proportional_gain": config.get("proportional_gain", 0.6),
        },
    }
    legacy = FastWAMFairCostController(
        fair_config,
        bootstrap_idm_cost=branch_cost.get("idm_cost", 0.0),
    )
    return LegacyFairCostAdapter(
        legacy,
        controller_type="legacy_fair_pi" if pi_enabled else "legacy_fair",
    )


@register_fastwam_idm_cost_controller("legacy_fair")
def _build_legacy_fair(config, branch_cost):
    return _build_explicit_legacy(config, branch_cost, pi_enabled=False)


@register_fastwam_idm_cost_controller("legacy_fair_pi")
def _build_legacy_fair_pi(config, branch_cost):
    return _build_explicit_legacy(config, branch_cost, pi_enabled=True)


@register_fastwam_idm_cost_controller("budget_dual")
def _build_budget_dual(config, branch_cost):
    return ProjectedDualIDMCostController(
        config,
        bootstrap_idm_cost=branch_cost.get("idm_cost", 0.0),
    )


def build_fastwam_idm_cost_controller(
    branch_cost_config: Any,
) -> tuple[FastWAMIDMCostController, bool, str]:
    """Build an explicit controller or auto-detect the unchanged legacy path."""

    branch_cost = _resolved_mapping(
        branch_cost_config, name="FastWAM fixed_branch_cost config"
    )
    explicit = "controller" in branch_cost and branch_cost["controller"] is not None
    if explicit:
        config = _resolved_mapping(
            branch_cost["controller"], name="FastWAM controller config"
        )
        controller_type = str(config.get("type", "")).lower()
        controller = get_fastwam_idm_cost_controller(controller_type)(
            config, branch_cost
        )
        return controller, True, _canonical_sha256(config)

    if not bool(branch_cost.get("enabled", False)):
        return _DisabledController(), False, ""
    fair = FastWAMFairCostController.from_branch_cost_config(branch_cost)
    if fair.enabled:
        adapter = LegacyFairCostAdapter(fair)
        return adapter, False, ""
    return _DisabledController(), False, ""


def validate_fastwam_idm_cost_control_config(cfg: Any) -> None:
    """Fail before worker startup when an explicit cost controller is invalid."""

    branch_cost = _resolved_mapping(
        OmegaConf.select(cfg, "algorithm.fixed_branch_cost", default={}),
        name="FastWAM fixed_branch_cost config",
    )
    controller_value = branch_cost.get("controller")
    if controller_value is None:
        return
    controller_config = _resolved_mapping(
        controller_value,
        name="FastWAM IDM cost controller config",
    )
    controller_type = str(controller_config.get("type", "")).lower()
    if bool(
        _resolved_mapping(
            branch_cost.get("fair_cost"), name="legacy fair-cost config"
        ).get("enabled", False)
    ):
        raise ValueError(
            "Explicit FastWAM IDM cost control cannot be combined with "
            "legacy fair_cost.enabled=true."
        )
    if not bool(branch_cost.get("enabled", False)):
        raise ValueError("Explicit FastWAM IDM cost control requires cost shaping.")
    if float(branch_cost.get("uncond_cost", 0.0)) != 0.0:
        raise ValueError("FastWAM IDM cost control requires uncond_cost=0.")
    controller, _, _ = build_fastwam_idm_cost_controller(branch_cost)
    dynamic = controller.requires_rollout_feedback
    guard = _resolved_mapping(
        OmegaConf.select(cfg, "runner.fastwam_training_guard", default={}),
        name="FastWAM training guard config",
    )
    cost_audit = _resolved_mapping(
        guard.get("cost_audit"), name="FastWAM cost audit config"
    )
    if dynamic:
        if not bool(guard.get("enabled", False)):
            raise ValueError(
                "Dynamic FastWAM IDM cost control requires the scientific guard."
            )
        if bool(OmegaConf.select(cfg, "runner.use_training_pipeline", default=False)):
            raise ValueError(
                "Dynamic FastWAM IDM cost control requires "
                "runner.use_training_pipeline=false."
            )
        weight_sync_interval = OmegaConf.select(
            cfg,
            "runner.weight_sync_interval",
            default=1,
        )
        if (
            isinstance(weight_sync_interval, bool)
            or not isinstance(weight_sync_interval, int)
            or weight_sync_interval != 1
        ):
            raise ValueError(
                "Dynamic FastWAM IDM cost control requires "
                "runner.weight_sync_interval=1."
            )

    charge_scope = str(controller_config.get("charge_scope", "all_valid_idm")).lower()
    if charge_scope not in {"all_valid_idm", "eligible_nonforced_idm"}:
        raise ValueError(f"Unsupported FastWAM charge scope {charge_scope!r}.")
    if controller_type == "fixed":
        if charge_scope != "all_valid_idm":
            raise ValueError(
                "Fixed FastWAM cost control requires charge_scope=all_valid_idm."
            )
        return
    if controller_type in {"legacy_fair", "legacy_fair_pi"}:
        if charge_scope != "all_valid_idm" and not bool(
            controller_config.get("allow_scope_mismatch", False)
        ):
            raise ValueError(
                "Legacy FastWAM cost controllers require charge_scope=all_valid_idm."
            )
        if not bool(cost_audit.get("enabled", False)):
            raise ValueError("Legacy fair-cost control requires the cost audit.")
        if bool(cost_audit.get("break_even_guard_enabled", True)):
            raise ValueError(
                "Legacy fair-cost control requires break_even_guard_enabled=false."
            )
        return
    if controller_type != "budget_dual":
        raise ValueError(
            f"Unsupported FastWAM IDM cost controller {controller_type!r}."
        )
    if str(controller_config.get("constraint", "")).lower() != "upper_bound":
        raise ValueError("FastWAM budget-dual v1 supports only an upper bound.")

    rate = _resolved_mapping(controller_config.get("rate"), name="rate config")
    rate_scope = str(rate.get("scope", "")).lower()
    feedback = str(rate.get("feedback", "")).lower()
    compatible = {
        "eligible_gate_decisions": {
            "expected_behavior_probability",
            "eligible_expected",
            "eligible_realized",
            "realized_gate_decisions",
        },
        "executed_valid_chunks": {"executed_realized"},
    }
    if rate_scope not in compatible or feedback not in compatible[rate_scope]:
        raise ValueError("FastWAM rate scope and feedback source are incompatible.")
    expected_charge_scope = {
        "eligible_gate_decisions": "eligible_nonforced_idm",
        "executed_valid_chunks": "all_valid_idm",
    }[rate_scope]
    if charge_scope != expected_charge_scope:
        raise ValueError("FastWAM rate scope and charge scope are incompatible.")

    target = _finite_float(rate.get("target_idm_fraction"), name="target IDM fraction")
    actor_epsilon = _finite_float(
        OmegaConf.select(cfg, "actor.model.gate_epsilon"),
        name="actor Gate epsilon",
    )
    rollout_epsilon = _finite_float(
        OmegaConf.select(cfg, "rollout.model.gate_epsilon"),
        name="rollout Gate epsilon",
    )
    if actor_epsilon != rollout_epsilon:
        raise ValueError("FastWAM actor and rollout Gate epsilon differ.")
    if not 0.0 <= actor_epsilon <= 1.0:
        raise ValueError("FastWAM Gate epsilon must lie in [0, 1].")
    reachable_min = actor_epsilon / 2.0
    reachable_max = 1.0 - actor_epsilon / 2.0
    if not reachable_min <= target <= reachable_max:
        raise ValueError(
            "FastWAM target IDM fraction is outside the epsilon-mixture "
            "reachable interval."
        )

    initializer = _resolved_mapping(
        controller_config.get("initializer"), name="initializer config"
    )
    if str(initializer.get("type", "")).lower() != "break_even_median":
        return
    if not bool(guard.get("enabled", False)) or not bool(
        cost_audit.get("enabled", False)
    ):
        raise ValueError(
            "FastWAM fair warm-start requires the scientific guard and cost audit."
        )
    if bool(cost_audit.get("break_even_guard_enabled", True)):
        raise ValueError(
            "FastWAM fair warm-start requires break_even_guard_enabled=false."
        )
    grid = [float(item) for item in cost_audit.get("counterfactual_idm_costs", [])]
    if (
        len(grid) < 2
        or grid != sorted(set(grid))
        or grid[0] != 0.0
        or any(not math.isfinite(item) or item < 0.0 for item in grid)
    ):
        raise ValueError(
            "FastWAM fair warm-start requires at least two unique, sorted, "
            "finite, non-negative counterfactual costs beginning at zero."
        )


def _metric_value(metrics: Mapping[str, Any], key: str, *, worker: int) -> float:
    if key not in metrics:
        raise ValueError(f"FastWAM worker {worker} is missing metric {key!r}.")
    return _finite_float(metrics[key], name=f"worker {worker} metric {key}")


def _metric_count(metrics: Mapping[str, Any], key: str, *, worker: int) -> int:
    value = _metric_value(metrics, key, worker=worker)
    if value < 0.0 or not value.is_integer():
        raise ValueError(f"FastWAM worker {worker} metric {key!r} is not a count.")
    return int(value)


def aggregate_fastwam_idm_cost_observation(
    *,
    runner_step: int,
    actor_rollout_metrics: Sequence[Mapping[str, Any]],
    guard_result: Mapping[str, Any],
) -> FastWAMIDMCostObservation:
    """Build count-weighted global route rates and reconcile the guard."""

    if not actor_rollout_metrics:
        raise ValueError("FastWAM IDM cost control received no worker metrics.")
    eligible_count = 0
    eligible_idm_count = 0
    valid_count = 0
    valid_idm_count = 0
    forced_count = 0
    expected_numerator = 0.0
    expected_available = True
    for worker, metrics in enumerate(actor_rollout_metrics):
        local_eligible = _metric_count(
            metrics,
            "fastwam/eligible_gate_decision_count",
            worker=worker,
        )
        local_eligible_idm = _metric_count(
            metrics,
            "fastwam/eligible_idm_decision_count",
            worker=worker,
        )
        local_valid = _metric_count(
            metrics,
            "fastwam/route/valid_chunk_count",
            worker=worker,
        )
        local_valid_idm = _metric_count(
            metrics,
            "fastwam/route/valid_idm_chunk_count",
            worker=worker,
        )
        local_forced = _metric_count(
            metrics,
            "fastwam/route/forced_count",
            worker=worker,
        )
        if not 0 <= local_eligible_idm <= local_eligible:
            raise ValueError("Worker eligible route counts are invalid.")
        if (
            not 0 <= local_valid_idm <= local_valid
            or not 0 <= local_forced <= local_valid
        ):
            raise ValueError("Worker executed route counts are invalid.")
        eligible_count += local_eligible
        eligible_idm_count += local_eligible_idm
        valid_count += local_valid
        valid_idm_count += local_valid_idm
        forced_count += local_forced
        probability = metrics.get("fastwam/gate/behavior_idm_probability_mean")
        if probability is None:
            expected_available = False
        else:
            probability = _finite_float(
                probability,
                name=f"worker {worker} behavior IDM probability",
            )
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Worker behavior IDM probability is outside [0, 1].")
            expected_numerator += local_eligible * probability
    if eligible_count < 1 or valid_count < 1:
        raise ValueError("FastWAM cost control requires eligible and valid chunks.")

    reconciliations = {
        "eligible_gate_decision_count": eligible_count,
        "eligible_idm_decision_count": eligible_idm_count,
    }
    for key, expected in reconciliations.items():
        if int(guard_result.get(key, -1)) != expected:
            raise ValueError(f"FastWAM runtime and guard {key} do not reconcile.")
    eligible_realized = eligible_idm_count / eligible_count
    if not math.isclose(
        float(guard_result.get("eligible_idm_fraction", math.nan)),
        eligible_realized,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("FastWAM runtime and guard eligible fraction disagree.")
    for key, expected in (
        ("valid_chunk_count", valid_count),
        ("valid_idm_chunk_count", valid_idm_count),
        ("forced_route_count", forced_count),
    ):
        if key in guard_result and int(guard_result[key]) != expected:
            raise ValueError(f"FastWAM runtime and guard {key} do not reconcile.")

    return FastWAMIDMCostObservation(
        runner_step=runner_step,
        eligible_gate_decision_count=eligible_count,
        eligible_idm_decision_count=eligible_idm_count,
        eligible_realized_fraction=eligible_realized,
        eligible_expected_fraction=(
            expected_numerator / eligible_count if expected_available else None
        ),
        valid_chunk_count=valid_count,
        valid_idm_chunk_count=valid_idm_count,
        executed_realized_fraction=valid_idm_count / valid_count,
        forced_fraction=forced_count / valid_count,
        break_even_idm_cost=guard_result.get("break_even_idm_cost"),
        configured_idm_cost=guard_result.get("configured_idm_cost"),
    )


def _observation_artifact(
    observation: FastWAMIDMCostObservation,
) -> dict[str, Any]:
    return {
        "eligible_gate_decision_count": observation.eligible_gate_decision_count,
        "eligible_idm_decision_count": observation.eligible_idm_decision_count,
        "eligible_realized_fraction": observation.eligible_realized_fraction,
        "eligible_expected_fraction": observation.eligible_expected_fraction,
        "valid_chunk_count": observation.valid_chunk_count,
        "valid_idm_chunk_count": observation.valid_idm_chunk_count,
        "executed_realized_fraction": observation.executed_realized_fraction,
        "forced_fraction": observation.forced_fraction,
        "break_even_idm_cost": observation.break_even_idm_cost,
        "configured_idm_cost": observation.configured_idm_cost,
    }


def _common_record_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    if record.get("schema") != FASTWAM_IDM_COST_CONTROL_SCHEMA:
        raise ValueError("FastWAM IDM cost control record schema mismatch.")
    applied = record["applied"]
    observed = record["observed"]
    next_decision = record["next"]
    break_even = observed.get("break_even_idm_cost")
    controller_type_ids = {
        "legacy_fair": 1.0,
        "legacy_fair_pi": 2.0,
        "budget_dual": 3.0,
    }
    metrics = {
        "fastwam/idm_cost_control/controller_type_id": controller_type_ids[
            str(record["controller_type"])
        ],
        "fastwam/idm_cost_control/applied_cost": float(applied["applied_idm_cost"]),
        "fastwam/idm_cost_control/next_cost": float(next_decision["applied_idm_cost"]),
        "fastwam/idm_cost_control/eligible_realized_fraction": float(
            observed["eligible_realized_fraction"]
        ),
        "fastwam/idm_cost_control/executed_realized_fraction": float(
            observed["executed_realized_fraction"]
        ),
        "fastwam/idm_cost_control/forced_fraction": float(observed["forced_fraction"]),
        "fastwam/idm_cost_control/break_even_defined": float(break_even is not None),
    }
    expected = observed.get("eligible_expected_fraction")
    if expected is not None:
        metrics["fastwam/idm_cost_control/eligible_expected_fraction"] = float(expected)
    if break_even is not None:
        metrics["fastwam/idm_cost_control/diagnostic_break_even"] = float(break_even)
    return metrics


def append_fastwam_idm_cost_control_jsonl(
    path: str | Path,
    record: Mapping[str, Any],
) -> None:
    """Append one controller transition to the unified run-scoped audit."""

    if record.get("schema") != FASTWAM_IDM_COST_CONTROL_SCHEMA:
        raise ValueError("FastWAM IDM cost control record schema mismatch.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(record), sort_keys=True, allow_nan=False)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class FastWAMIDMCostControlRuntime:
    """Generic runner facade for cost publication, feedback, and state."""

    def __init__(
        self,
        *,
        controller: FastWAMIDMCostController,
        explicit: bool,
        config_sha256: str,
        audit_root: Path,
    ) -> None:
        self.controller = controller
        self.explicit = explicit
        self.config_sha256 = config_sha256
        self.audit_root = audit_root

    @classmethod
    def from_config(cls, cfg: Any) -> FastWAMIDMCostControlRuntime:
        algorithm = cfg.get("algorithm", {})
        branch_cost = algorithm.get("fixed_branch_cost", {})
        controller, explicit, config_sha256 = build_fastwam_idm_cost_controller(
            branch_cost
        )
        logger = cfg.runner.get("logger", {})
        audit_root = (
            Path(str(logger.get("log_path", ".")))
            / str(logger.get("experiment_name", "run"))
            / "audits"
        )
        return cls(
            controller=controller,
            explicit=explicit,
            config_sha256=config_sha256,
            audit_root=audit_root,
        )

    @classmethod
    def from_legacy_controller(
        cls,
        cfg: Any,
        controller: FastWAMFairCostController,
    ) -> FastWAMIDMCostControlRuntime:
        wrapped: FastWAMIDMCostController = (
            LegacyFairCostAdapter(controller)
            if controller.enabled
            else _DisabledController()
        )
        logger = cfg.runner.get("logger", {})
        return cls(
            controller=wrapped,
            explicit=False,
            config_sha256="",
            audit_root=(
                Path(str(logger.get("log_path", ".")))
                / str(logger.get("experiment_name", "run"))
                / "audits"
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.controller.enabled

    @property
    def requires_rollout_feedback(self) -> bool:
        return self.controller.requires_rollout_feedback

    @property
    def legacy_fair_controller(self) -> FastWAMFairCostController | None:
        if isinstance(self.controller, LegacyFairCostAdapter):
            return self.controller.legacy_controller
        return None

    @property
    def checkpoint_schema(self) -> str:
        if self.explicit:
            return "fastwam-training-guard-checkpoint-v3"
        if self.legacy_fair_controller is not None:
            return "fastwam-training-guard-checkpoint-v2"
        return "fastwam-training-guard-checkpoint-v1"

    def before_rollout(
        self,
        *,
        actor: Any,
        runner_step: int,
    ) -> FastWAMIDMCostDecision | None:
        if not self.enabled:
            return None
        decision = self.controller.decision_for_step(runner_step)
        actor.set_fastwam_idm_cost(
            decision.applied_idm_cost,
            runner_step,
        ).wait()
        return decision

    def after_rollout(
        self,
        *,
        runner_step: int,
        actor_rollout_metrics: list[dict],
        guard_result: dict,
    ) -> dict[str, Any] | None:
        if not self.enabled or not self.requires_rollout_feedback:
            return None
        if guard_result.get("status") != "PASS":
            raise RuntimeError(
                "Feedback IDM cost control requires an enabled scientific guard."
            )
        observation = aggregate_fastwam_idm_cost_observation(
            runner_step=runner_step,
            actor_rollout_metrics=actor_rollout_metrics,
            guard_result=guard_result,
        )
        record = self.controller.observe_rollout(observation)
        metrics = self.controller.record_metrics(record)
        for worker_metrics in actor_rollout_metrics:
            worker_metrics.update(metrics)
        append_fastwam_idm_cost_control_jsonl(
            self.audit_root / "idm_cost_control.jsonl",
            record,
        )
        legacy = record.get("legacy_record")
        if isinstance(legacy, Mapping) and (
            legacy.get("schema") == FASTWAM_FAIR_COST_CONTROL_SCHEMA
        ):
            append_fastwam_fair_cost_control_jsonl(
                self.audit_root / "fair_cost_control.jsonl",
                legacy,
            )
        return record

    def state_dict(self) -> dict[str, Any]:
        if not self.explicit:
            raise RuntimeError("Only explicit FastWAM controllers use v3 state.")
        return {
            "schema": FASTWAM_IDM_COST_CONTROLLER_STATE_SCHEMA,
            "controller_type": self.controller.controller_type,
            "config_sha256": self.config_sha256,
            "payload": self.controller.state_dict(),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        global_step: int,
    ) -> None:
        if not self.explicit:
            raise RuntimeError("Only explicit FastWAM controllers load v3 state.")
        if state.get("schema") != FASTWAM_IDM_COST_CONTROLLER_STATE_SCHEMA:
            raise ValueError("FastWAM IDM cost controller state schema mismatch.")
        if state.get("controller_type") != self.controller.controller_type:
            raise ValueError("FastWAM IDM cost controller type mismatch.")
        if state.get("config_sha256") != self.config_sha256:
            raise ValueError("FastWAM IDM cost controller config hash mismatch.")
        payload = state.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("FastWAM IDM cost controller payload is malformed.")
        self.controller.load_state_dict(payload)
        if self.controller.observed_runner_steps != int(global_step):
            raise ValueError(
                "FastWAM IDM cost controller observed steps do not match global_step."
            )
