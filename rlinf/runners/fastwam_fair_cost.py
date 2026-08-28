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

"""Lagged fair-price and optional PI control for the FastWAM IDM cost."""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FASTWAM_FAIR_COST_CONTROL_SCHEMA = "fastwam-fair-cost-control-v1"
FASTWAM_FAIR_COST_STATE_SCHEMA = "fastwam-fair-cost-controller-state-v1"
FASTWAM_LAGRANGE_MULTIPLIER_MIN = 0.0
FASTWAM_LAGRANGE_MULTIPLIER_MAX = 2.0


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    raise TypeError("FastWAM fair-cost config must be a mapping.")


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMFairCostDecision:
    """The cost fixed before one rollout begins."""

    runner_step: int
    fair_idm_cost: float
    lagrange_multiplier: float
    applied_idm_cost: float
    lagged_break_even_window: tuple[float, ...]
    last_valid_break_even_idm_cost: float | None
    pi_enabled: bool

    def to_artifact(self) -> dict[str, Any]:
        return {
            "runner_step": self.runner_step,
            "fair_idm_cost": self.fair_idm_cost,
            "lagrange_multiplier": self.lagrange_multiplier,
            "applied_idm_cost": self.applied_idm_cost,
            "lagged_break_even_window": list(self.lagged_break_even_window),
            "last_valid_break_even_idm_cost": (self.last_valid_break_even_idm_cost),
            "pi_enabled": self.pi_enabled,
        }


class FastWAMFairCostController:
    """Turn prior-rollout break-even prices into the next rollout's IDM cost."""

    def __init__(self, config: Any, *, bootstrap_idm_cost: float) -> None:
        raw = _plain_mapping(config)
        pi = _plain_mapping(raw.get("pi", None))
        self.enabled = bool(raw.get("enabled", False))
        self.window_size = int(raw.get("window_size", 5))
        self.bootstrap_idm_cost = _finite_float(
            bootstrap_idm_cost,
            name="FastWAM fair-cost bootstrap IDM cost",
        )
        self.pi_enabled = bool(pi.get("enabled", False))
        self.target_idm_fraction = _finite_float(
            pi.get("target_idm_fraction", 0.5),
            name="FastWAM PI target IDM fraction",
        )
        self.integral_gain = _finite_float(
            pi.get("integral_gain", 0.05),
            name="FastWAM PI integral gain",
        )
        self.proportional_gain = _finite_float(
            pi.get("proportional_gain", 0.6),
            name="FastWAM PI proportional gain",
        )
        self._validate_config()
        self._config = {
            "enabled": self.enabled,
            "window_size": self.window_size,
            "bootstrap_idm_cost": self.bootstrap_idm_cost,
            "pi_enabled": self.pi_enabled,
            "target_idm_fraction": self.target_idm_fraction,
            "integral_gain": self.integral_gain,
            "proportional_gain": self.proportional_gain,
            "lagrange_multiplier_min": FASTWAM_LAGRANGE_MULTIPLIER_MIN,
            "lagrange_multiplier_max": FASTWAM_LAGRANGE_MULTIPLIER_MAX,
        }
        self._observed_runner_steps = 0
        self._break_even_history: list[float] = []
        self._last_valid_break_even_idm_cost: float | None = None
        self._fair_idm_cost = self.bootstrap_idm_cost
        self._integral_term = 0.0
        self._lagrange_multiplier = 0.0

    @classmethod
    def from_branch_cost_config(cls, config: Any) -> FastWAMFairCostController:
        branch_cost = _plain_mapping(config)
        return cls(
            branch_cost.get("fair_cost", None),
            bootstrap_idm_cost=branch_cost.get("idm_cost", 0.0),
        )

    def _validate_config(self) -> None:
        if self.bootstrap_idm_cost < 0.0:
            raise ValueError("FastWAM fair-cost bootstrap must be non-negative.")
        if self.window_size < 1:
            raise ValueError("FastWAM fair-cost window_size must be positive.")
        if self.pi_enabled and not self.enabled:
            raise ValueError("FastWAM PI control requires fair-cost control.")
        if not 0.0 < self.target_idm_fraction < 1.0:
            raise ValueError("FastWAM PI target IDM fraction must lie in (0, 1).")
        if self.integral_gain < 0.0 or self.proportional_gain < 0.0:
            raise ValueError("FastWAM PI gains must be non-negative.")

    def decision_for_step(self, runner_step: int) -> FastWAMFairCostDecision:
        """Return a read-only decision derived only from earlier rollouts."""

        if isinstance(runner_step, bool) or int(runner_step) != runner_step:
            raise TypeError("FastWAM fair-cost runner_step must be an integer.")
        runner_step = int(runner_step)
        if runner_step != self._observed_runner_steps:
            raise ValueError(
                "FastWAM fair-cost step does not match lagged controller state: "
                f"requested {runner_step}, expected {self._observed_runner_steps}."
            )
        applied = self._fair_idm_cost + self._lagrange_multiplier
        return FastWAMFairCostDecision(
            runner_step=runner_step,
            fair_idm_cost=self._fair_idm_cost,
            lagrange_multiplier=self._lagrange_multiplier,
            applied_idm_cost=applied,
            lagged_break_even_window=tuple(self._break_even_history),
            last_valid_break_even_idm_cost=(self._last_valid_break_even_idm_cost),
            pi_enabled=self.pi_enabled,
        )

    def observe_rollout(
        self,
        *,
        runner_step: int,
        break_even_idm_cost: float | None,
        idm_fraction: float,
    ) -> dict[str, Any]:
        """Consume step ``t`` only after its preselected cost has been used."""

        if not self.enabled:
            raise RuntimeError("FastWAM fair-cost control is disabled.")
        applied = self.decision_for_step(runner_step)
        idm_fraction = _finite_float(
            idm_fraction,
            name="FastWAM observed IDM fraction",
        )
        if not 0.0 <= idm_fraction <= 1.0:
            raise ValueError("FastWAM observed IDM fraction must lie in [0, 1].")

        if break_even_idm_cost is None:
            carried_break_even = self._last_valid_break_even_idm_cost
        else:
            carried_break_even = _finite_float(
                break_even_idm_cost,
                name="FastWAM observed break-even IDM cost",
            )
            if carried_break_even < 0.0:
                raise ValueError(
                    "FastWAM observed break-even IDM cost must be non-negative."
                )
            self._last_valid_break_even_idm_cost = carried_break_even
            # Undefined break-even values carry the current price forward, but
            # are not observations. Re-appending the last valid value would
            # duplicate evidence and bias the rolling median.
            self._break_even_history.append(carried_break_even)
            del self._break_even_history[
                : max(0, len(self._break_even_history) - self.window_size)
            ]
            self._fair_idm_cost = float(statistics.median(self._break_even_history))

        pi_error = idm_fraction - self.target_idm_fraction
        if self.pi_enabled:
            self._integral_term += self.integral_gain * pi_error
            self._lagrange_multiplier = min(
                FASTWAM_LAGRANGE_MULTIPLIER_MAX,
                max(
                    FASTWAM_LAGRANGE_MULTIPLIER_MIN,
                    self._integral_term + self.proportional_gain * pi_error,
                ),
            )
        else:
            self._integral_term = 0.0
            self._lagrange_multiplier = 0.0

        self._observed_runner_steps += 1
        next_decision = self.decision_for_step(self._observed_runner_steps)
        return {
            "schema": FASTWAM_FAIR_COST_CONTROL_SCHEMA,
            "runner_step": int(runner_step),
            "applied": applied.to_artifact(),
            "observed_break_even_idm_cost": break_even_idm_cost,
            "carried_break_even_idm_cost": carried_break_even,
            "observed_idm_fraction": idm_fraction,
            "pi_error": pi_error,
            "integral_term": self._integral_term,
            "next": next_decision.to_artifact(),
        }

    @staticmethod
    def record_metrics(record: Mapping[str, Any]) -> dict[str, float]:
        if record.get("schema") != FASTWAM_FAIR_COST_CONTROL_SCHEMA:
            raise ValueError("FastWAM fair-cost control record schema mismatch.")
        applied = record["applied"]
        observed = record.get("observed_break_even_idm_cost")
        return {
            "fastwam/fair_cost/fair_idm_cost": float(applied["fair_idm_cost"]),
            "fastwam/fair_cost/lagrange_multiplier": float(
                applied["lagrange_multiplier"]
            ),
            "fastwam/fair_cost/applied_idm_cost": float(applied["applied_idm_cost"]),
            "fastwam/fair_cost/lagged_window_count": float(
                len(applied["lagged_break_even_window"])
            ),
            "fastwam/fair_cost/break_even_defined": float(observed is not None),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": FASTWAM_FAIR_COST_STATE_SCHEMA,
            "config": dict(self._config),
            "observed_runner_steps": self._observed_runner_steps,
            "break_even_history": list(self._break_even_history),
            "last_valid_break_even_idm_cost": (self._last_valid_break_even_idm_cost),
            "fair_idm_cost": self._fair_idm_cost,
            "integral_term": self._integral_term,
            "lagrange_multiplier": self._lagrange_multiplier,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != FASTWAM_FAIR_COST_STATE_SCHEMA:
            raise ValueError("FastWAM fair-cost controller state schema mismatch.")
        if state.get("config") != self._config:
            raise ValueError("FastWAM fair-cost controller config mismatch.")
        observed = int(state.get("observed_runner_steps", -1))
        if observed < 0:
            raise ValueError("FastWAM fair-cost observed step count is invalid.")
        history = [
            _finite_float(item, name="FastWAM break-even history")
            for item in state.get("break_even_history", [])
        ]
        if len(history) > self.window_size or any(item < 0.0 for item in history):
            raise ValueError("FastWAM break-even history is invalid.")
        last_valid = state.get("last_valid_break_even_idm_cost")
        if last_valid is not None:
            last_valid = _finite_float(
                last_valid,
                name="FastWAM last valid break-even IDM cost",
            )
            if last_valid < 0.0:
                raise ValueError("FastWAM last valid break-even cost is negative.")
        if history and (last_valid is None or history[-1] != last_valid):
            raise ValueError("FastWAM carried break-even history is inconsistent.")
        fair = _finite_float(
            state.get("fair_idm_cost"),
            name="FastWAM restored fair IDM cost",
        )
        expected_fair = (
            float(statistics.median(history)) if history else self.bootstrap_idm_cost
        )
        if fair != expected_fair:
            raise ValueError("FastWAM restored fair IDM cost is inconsistent.")
        integral = _finite_float(
            state.get("integral_term"),
            name="FastWAM restored PI integral term",
        )
        multiplier = _finite_float(
            state.get("lagrange_multiplier"),
            name="FastWAM restored Lagrange multiplier",
        )
        if not (
            FASTWAM_LAGRANGE_MULTIPLIER_MIN
            <= multiplier
            <= FASTWAM_LAGRANGE_MULTIPLIER_MAX
        ):
            raise ValueError("FastWAM restored Lagrange multiplier is out of range.")
        if not self.pi_enabled and (integral != 0.0 or multiplier != 0.0):
            raise ValueError("Disabled FastWAM PI state must remain zero.")

        self._observed_runner_steps = observed
        self._break_even_history = history
        self._last_valid_break_even_idm_cost = last_valid
        self._fair_idm_cost = fair
        self._integral_term = integral
        self._lagrange_multiplier = multiplier


def append_fastwam_fair_cost_control_jsonl(
    path: str | Path,
    record: Mapping[str, Any],
) -> None:
    """Persist one central control record after its rollout is observed."""

    if record.get("schema") != FASTWAM_FAIR_COST_CONTROL_SCHEMA:
        raise ValueError("FastWAM fair-cost control record schema mismatch.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(record), sort_keys=True, allow_nan=False)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
