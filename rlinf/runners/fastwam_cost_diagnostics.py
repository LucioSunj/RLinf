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

"""Diagnostic-only price estimators for FastWAM cost controllers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from rlinf.runners.fastwam_fair_cost import FastWAMFairCostController

FASTWAM_COST_DIAGNOSTIC_STATE_SCHEMA = "fastwam-cost-diagnostic-state-v1"


class FastWAMCostDiagnostic(Protocol):
    """Read-only telemetry composed around an applied cost controller."""

    diagnostic_type: str

    def decision_metadata_for_step(self, runner_step: int) -> dict[str, float]: ...

    def observe_rollout(self, observation: Any) -> dict[str, Any]: ...

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class FairBreakEvenDiagnosticAdapter:
    """Wrap the PI-disabled legacy fair estimator without applying its price."""

    diagnostic_type = "fair_break_even"

    def __init__(self, config: Mapping[str, Any], *, bootstrap_idm_cost: float) -> None:
        if not bool(config.get("diagnostic_only", True)):
            raise ValueError("Fair break-even diagnostics must be diagnostic_only.")
        window_size = int(config.get("window_size", 5))
        display_bootstrap = float(
            config.get("bootstrap_value_for_display", bootstrap_idm_cost)
        )
        self._config = {
            "diagnostic_only": True,
            "window_size": window_size,
            "bootstrap_value_for_display": display_bootstrap,
        }
        self._estimator = FastWAMFairCostController(
            {
                "enabled": True,
                "window_size": window_size,
                "pi": {"enabled": False},
            },
            bootstrap_idm_cost=display_bootstrap,
        )

    def decision_metadata_for_step(self, runner_step: int) -> dict[str, float]:
        decision = self._estimator.decision_for_step(runner_step)
        return {
            "diagnostic_fair_cost": decision.fair_idm_cost,
            "dual_scale_reference": decision.fair_idm_cost,
            "diagnostic_only_fair": 1.0,
        }

    def observe_rollout(self, observation: Any) -> dict[str, Any]:
        fair_record = self._estimator.observe_rollout(
            runner_step=int(observation.runner_step),
            break_even_idm_cost=observation.break_even_idm_cost,
            idm_fraction=float(observation.eligible_realized_fraction),
        )
        return {
            "diagnostic_type": self.diagnostic_type,
            "diagnostic_only": True,
            "fair_record": fair_record,
        }

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        fair_record = record.get("fair_record")
        if not isinstance(fair_record, Mapping):
            raise TypeError("Fair diagnostic record is malformed.")
        next_decision = fair_record.get("next")
        if not isinstance(next_decision, Mapping):
            raise TypeError("Fair diagnostic next decision is malformed.")
        metrics = {
            "fastwam/idm_cost_control/diagnostic_fair_cost": float(
                next_decision["fair_idm_cost"]
            ),
            "fastwam/idm_cost_control/diagnostic_only_fair": 1.0,
            "fastwam/idm_cost_control/diagnostic_fair_valid_count": float(
                len(next_decision["lagged_break_even_window"])
            ),
        }
        observed = fair_record.get("observed_break_even_idm_cost")
        if observed is not None:
            metrics["fastwam/idm_cost_control/diagnostic_break_even"] = float(observed)
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": FASTWAM_COST_DIAGNOSTIC_STATE_SCHEMA,
            "diagnostic_type": self.diagnostic_type,
            "diagnostic_only": True,
            "config": dict(self._config),
            "payload": self._estimator.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != FASTWAM_COST_DIAGNOSTIC_STATE_SCHEMA:
            raise ValueError("FastWAM cost diagnostic state schema mismatch.")
        if state.get("diagnostic_type") != self.diagnostic_type:
            raise ValueError("FastWAM cost diagnostic type mismatch.")
        if state.get("diagnostic_only") is not True:
            raise ValueError("Restored fair diagnostic is not diagnostic-only.")
        if state.get("config") != self._config:
            raise ValueError("FastWAM cost diagnostic config mismatch.")
        payload = state.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("FastWAM cost diagnostic payload is malformed.")
        self._estimator.load_state_dict(payload)


_DIAGNOSTIC_REGISTRY: dict[
    str, Callable[[Mapping[str, Any], float], FastWAMCostDiagnostic]
] = {}


def register_fastwam_cost_diagnostic(name: str):
    """Register a diagnostic factory used by controller composition."""

    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("FastWAM cost diagnostic name cannot be empty.")

    def decorator(factory):
        if normalized in _DIAGNOSTIC_REGISTRY:
            raise ValueError(f"FastWAM cost diagnostic {normalized!r} exists.")
        _DIAGNOSTIC_REGISTRY[normalized] = factory
        return factory

    return decorator


@register_fastwam_cost_diagnostic("fair_break_even")
def _build_fair_break_even(
    config: Mapping[str, Any], bootstrap_idm_cost: float
) -> FastWAMCostDiagnostic:
    return FairBreakEvenDiagnosticAdapter(
        config,
        bootstrap_idm_cost=bootstrap_idm_cost,
    )


def build_fastwam_cost_diagnostics(
    configs: Sequence[Mapping[str, Any]],
    *,
    bootstrap_idm_cost: float,
) -> tuple[FastWAMCostDiagnostic, ...]:
    """Build a duplicate-free ordered diagnostic collection."""

    diagnostics: list[FastWAMCostDiagnostic] = []
    types: set[str] = set()
    for raw in configs:
        config = {str(key): value for key, value in raw.items()}
        if not bool(config.get("enabled", True)):
            continue
        diagnostic_type = str(config.get("type", "")).strip().lower()
        if diagnostic_type in types:
            raise ValueError(f"Duplicate FastWAM cost diagnostic {diagnostic_type!r}.")
        try:
            factory = _DIAGNOSTIC_REGISTRY[diagnostic_type]
        except KeyError as error:
            raise ValueError(
                f"Unsupported FastWAM cost diagnostic {diagnostic_type!r}."
            ) from error
        diagnostic = factory(config, bootstrap_idm_cost)
        types.add(diagnostic_type)
        diagnostics.append(diagnostic)
    return tuple(diagnostics)


class DiagnosticIDMCostController:
    """Decorate an IDM controller while preserving its exact applied decision."""

    def __init__(self, delegate: Any, diagnostics: Sequence[FastWAMCostDiagnostic]):
        if not diagnostics:
            raise ValueError("Diagnostic controller requires at least one diagnostic.")
        types = [diagnostic.diagnostic_type for diagnostic in diagnostics]
        if len(types) != len(set(types)):
            raise ValueError("Duplicate FastWAM cost diagnostic type.")
        self.delegate = delegate
        self.diagnostics = tuple(diagnostics)

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
        return True

    @property
    def observed_runner_steps(self) -> int:
        return int(self.delegate.observed_runner_steps)

    def decision_for_step(self, runner_step: int) -> Any:
        decision = self.delegate.decision_for_step(runner_step)
        components = dict(decision.components)
        for diagnostic in self.diagnostics:
            for name, value in diagnostic.decision_metadata_for_step(
                runner_step
            ).items():
                key = f"diagnostic/{diagnostic.diagnostic_type}/{name}"
                if key in components:
                    raise ValueError(f"Duplicate diagnostic component {key!r}.")
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError("Diagnostic decision metadata must be finite.")
                components[key] = numeric
        return type(decision)(
            runner_step=decision.runner_step,
            controller_type=decision.controller_type,
            phase=decision.phase,
            applied_idm_cost=decision.applied_idm_cost,
            components=components,
        )

    def observe_rollout(self, observation: Any) -> dict[str, Any]:
        record = self.delegate.observe_rollout(observation)
        applied_before = float(record["applied"]["applied_idm_cost"])
        next_before = float(record["next"]["applied_idm_cost"])
        diagnostic_records = {
            diagnostic.diagnostic_type: diagnostic.observe_rollout(observation)
            for diagnostic in self.diagnostics
        }
        if (
            float(record["applied"]["applied_idm_cost"]) != applied_before
            or float(record["next"]["applied_idm_cost"]) != next_before
        ):
            raise RuntimeError("FastWAM diagnostics changed an applied cost decision.")
        dual_multiplier = float(
            record["applied"]["components"].get(
                "dual_multiplier",
                applied_before,
            )
        )
        if not math.isclose(
            applied_before,
            dual_multiplier,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("Diagnostic-only controller violated applied=dual.")
        record["diagnostics"] = diagnostic_records
        record["diagnostic_invariants"] = {
            "applied_cost_unchanged": True,
            "next_cost_unchanged": True,
            "diagnostic_only": True,
            "applied_equals_dual": True,
        }
        return record

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        metrics = self.delegate.record_metrics(record)
        records = record.get("diagnostics")
        if not isinstance(records, Mapping):
            raise TypeError("FastWAM diagnostic records are missing.")
        for diagnostic in self.diagnostics:
            diagnostic_record = records.get(diagnostic.diagnostic_type)
            if not isinstance(diagnostic_record, Mapping):
                raise TypeError("FastWAM diagnostic record is malformed.")
            metrics.update(diagnostic.record_metrics(diagnostic_record))
        applied = float(record["applied"]["applied_idm_cost"])
        components = record["applied"]["components"]
        dual = float(components.get("dual_multiplier", applied))
        if not math.isclose(applied, dual, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("Diagnostic-only controller violated applied=dual.")
        fair = metrics.get("fastwam/idm_cost_control/diagnostic_fair_cost")
        metrics["fastwam/idm_cost_control/dual_multiplier"] = dual
        metrics["fastwam/idm_cost_control/applied_minus_dual"] = applied - dual
        if fair is not None:
            metrics["fastwam/idm_cost_control/fair_minus_dual"] = fair - dual
            metrics["fastwam/idm_cost_control/dual_scale_reference"] = fair
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "fastwam-diagnostic-idm-cost-controller-state-v1",
            "controller_type": self.controller_type,
            "delegate": self.delegate.state_dict(),
            "diagnostics": {
                diagnostic.diagnostic_type: diagnostic.state_dict()
                for diagnostic in self.diagnostics
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "fastwam-diagnostic-idm-cost-controller-state-v1":
            raise ValueError("Diagnostic IDM controller state schema mismatch.")
        if state.get("controller_type") != self.controller_type:
            raise ValueError("Diagnostic IDM controller type mismatch.")
        delegate = state.get("delegate")
        diagnostics = state.get("diagnostics")
        if not isinstance(delegate, Mapping) or not isinstance(diagnostics, Mapping):
            raise TypeError("Diagnostic IDM controller state is malformed.")
        expected_types = {item.diagnostic_type for item in self.diagnostics}
        if set(diagnostics) != expected_types:
            raise ValueError("Diagnostic IDM controller diagnostic set mismatch.")
        self.delegate.load_state_dict(delegate)
        for diagnostic in self.diagnostics:
            payload = diagnostics[diagnostic.diagnostic_type]
            if not isinstance(payload, Mapping):
                raise TypeError("FastWAM diagnostic state is malformed.")
            diagnostic.load_state_dict(payload)
        for diagnostic in self.diagnostics:
            payload = diagnostic.state_dict().get("payload", {})
            if int(payload.get("observed_runner_steps", -1)) != int(
                self.observed_runner_steps
            ):
                raise ValueError("Diagnostic and IDM controller steps disagree.")


__all__ = [
    "DiagnosticIDMCostController",
    "FairBreakEvenDiagnosticAdapter",
    "FastWAMCostDiagnostic",
    "build_fastwam_cost_diagnostics",
    "register_fastwam_cost_diagnostic",
]
