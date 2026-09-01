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

"""Hydra adapters for existing FastWAM standalone evaluation route modes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rlinf.models.embodiment.wam_policy.evaluation import (
    autocorrelated_transition_probabilities,
)

RoutingParameter = float | int | str | None


def _routing_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("FastWAM evaluation routing seed must be non-negative.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMEvalRoutingArmSpec:
    """Config-resolved identity and fixed overrides for one evaluation arm."""

    name: str
    arm_type: str
    parameter_name: str | None
    static_overrides: Mapping[str, object]


class FastWAMEvalRoutingArm(Protocol):
    """Translate calibration parameters into existing evaluator overrides."""

    arm_type: str
    parameter_name: str

    def parameter_domain(
        self, config: Mapping[str, object]
    ) -> Sequence[RoutingParameter]: ...

    def hydra_overrides(
        self,
        *,
        parameter: RoutingParameter,
        routing_seed: int,
    ) -> tuple[str, ...]: ...

    def extract_rate(self, result: Any, *, rate_scope: str) -> float: ...


class _BaseRoutingArm:
    parameter_name = "parameter"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = {str(key): value for key, value in config.items()}

    def parameter_domain(
        self, config: Mapping[str, object]
    ) -> Sequence[RoutingParameter]:
        candidates = config.get("candidates", self.config.get("candidates", ()))
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise TypeError("Routing arm candidates must be a sequence.")
        if not candidates:
            raise ValueError("Routing arm candidate domain cannot be empty.")
        return tuple(candidates)

    def extract_rate(self, result: Any, *, rate_scope: str) -> float:
        if rate_scope == "eligible_gate_decisions":
            return float(result.eligible_realized_fraction)
        if rate_scope == "executed_valid_chunks":
            return float(result.executed_realized_fraction)
        raise ValueError(f"Unsupported FastWAM evaluation rate scope {rate_scope!r}.")


class LearnedThresholdArmAdapter(_BaseRoutingArm):
    """Map a threshold candidate to the existing learned-threshold mode."""

    arm_type = "learned_threshold"
    parameter_name = "threshold"

    def hydra_overrides(
        self, *, parameter: RoutingParameter, routing_seed: int
    ) -> tuple[str, ...]:
        threshold = float(parameter)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Learned routing threshold must lie in [0, 1].")
        return (
            "rollout.model.eval_routing_mode=learned_threshold",
            f"rollout.model.eval_idm_threshold={threshold:.17g}",
            f"rollout.model.eval_routing_seed={_routing_seed(routing_seed)}",
        )


class IndependentRandomArmAdapter(_BaseRoutingArm):
    """Map an IDM probability to the existing stateless random mode."""

    arm_type = "independent_random"
    parameter_name = "idm_probability"

    def hydra_overrides(
        self, *, parameter: RoutingParameter, routing_seed: int
    ) -> tuple[str, ...]:
        probability = float(parameter)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Random IDM probability must lie in [0, 1].")
        return (
            "rollout.model.eval_routing_mode=matched_random",
            f"rollout.model.eval_random_idm_probability={probability:.17g}",
            "rollout.model.eval_random_lag1_autocorrelation=null",
            f"rollout.model.eval_routing_seed={_routing_seed(routing_seed)}",
        )


class AutocorrelationMatchedRandomArmAdapter(_BaseRoutingArm):
    """Calibrate stationary rate while preserving a declared autocorrelation."""

    arm_type = "autocorrelation_matched_random"
    parameter_name = "idm_probability"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.lag1_autocorrelation = float(config.get("lag1_autocorrelation"))
        if not -1.0 <= self.lag1_autocorrelation <= 1.0:
            raise ValueError("Random lag-1 autocorrelation must lie in [-1, 1].")

    def hydra_overrides(
        self, *, parameter: RoutingParameter, routing_seed: int
    ) -> tuple[str, ...]:
        probability = float(parameter)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Random IDM probability must lie in [0, 1].")
        autocorrelated_transition_probabilities(
            probability,
            self.lag1_autocorrelation,
        )
        return (
            "rollout.model.eval_routing_mode=autocorrelation_matched_random",
            f"rollout.model.eval_random_idm_probability={probability:.17g}",
            "rollout.model.eval_random_lag1_autocorrelation="
            f"{self.lag1_autocorrelation:.17g}",
            f"rollout.model.eval_routing_seed={_routing_seed(routing_seed)}",
        )


class PeriodicArmAdapter(_BaseRoutingArm):
    """Map a ``period:on_count:phase`` candidate to periodic routing."""

    arm_type = "periodic"
    parameter_name = "periodic_pattern"

    @staticmethod
    def _parse(parameter: RoutingParameter) -> tuple[int, int, int]:
        if not isinstance(parameter, str):
            raise TypeError("Periodic routing parameters use period:on_count:phase.")
        try:
            period, on_count, phase = (int(item) for item in parameter.split(":"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Periodic routing parameters use period:on_count:phase."
            ) from error
        if period < 1 or not 0 <= on_count <= period or not 0 <= phase < period:
            raise ValueError("Periodic routing pattern is invalid.")
        return period, on_count, phase

    def hydra_overrides(
        self, *, parameter: RoutingParameter, routing_seed: int
    ) -> tuple[str, ...]:
        period, on_count, phase = self._parse(parameter)
        return (
            "rollout.model.eval_routing_mode=periodic",
            f"rollout.model.eval_period={period}",
            f"rollout.model.eval_periodic_on_count={on_count}",
            f"rollout.model.eval_periodic_phase={phase}",
            f"rollout.model.eval_routing_seed={_routing_seed(routing_seed)}",
        )


_ARM_REGISTRY: dict[str, Callable[[Mapping[str, Any]], FastWAMEvalRoutingArm]] = {}


def register_fastwam_eval_routing_arm(name: str):
    """Register one config-selected evaluation routing arm adapter."""

    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("FastWAM evaluation routing arm name cannot be empty.")

    def decorator(factory):
        if normalized in _ARM_REGISTRY:
            raise ValueError(f"FastWAM evaluation routing arm {normalized!r} exists.")
        _ARM_REGISTRY[normalized] = factory
        return factory

    return decorator


@register_fastwam_eval_routing_arm("learned_threshold")
def _build_learned(config: Mapping[str, Any]) -> FastWAMEvalRoutingArm:
    return LearnedThresholdArmAdapter(config)


@register_fastwam_eval_routing_arm("independent_random")
def _build_independent(config: Mapping[str, Any]) -> FastWAMEvalRoutingArm:
    return IndependentRandomArmAdapter(config)


@register_fastwam_eval_routing_arm("autocorrelation_matched_random")
def _build_autocorrelation(config: Mapping[str, Any]) -> FastWAMEvalRoutingArm:
    return AutocorrelationMatchedRandomArmAdapter(config)


@register_fastwam_eval_routing_arm("periodic")
def _build_periodic(config: Mapping[str, Any]) -> FastWAMEvalRoutingArm:
    return PeriodicArmAdapter(config)


def get_fastwam_eval_routing_arm(name: str):
    """Return a registered evaluation arm adapter factory."""

    normalized = str(name).strip().lower()
    try:
        return _ARM_REGISTRY[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unsupported FastWAM evaluation routing arm {normalized!r}."
        ) from error


def build_fastwam_eval_routing_arm(
    config: Mapping[str, Any],
) -> FastWAMEvalRoutingArm:
    """Build an arm from its Hydra-resolved mapping."""

    arm_type = str(config.get("type", "")).lower()
    return get_fastwam_eval_routing_arm(arm_type)(config)


__all__ = [
    "AutocorrelationMatchedRandomArmAdapter",
    "FastWAMEvalRoutingArm",
    "FastWAMEvalRoutingArmSpec",
    "IndependentRandomArmAdapter",
    "LearnedThresholdArmAdapter",
    "PeriodicArmAdapter",
    "RoutingParameter",
    "build_fastwam_eval_routing_arm",
    "get_fastwam_eval_routing_arm",
    "register_fastwam_eval_routing_arm",
]
