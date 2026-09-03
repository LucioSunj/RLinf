# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Configuration contracts for the route-neutral current-step Gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields changed: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def _require_false(value: Any, *, name: str) -> None:
    if value is not False:
        raise ValueError(f"Route-neutral Gate requires `{name}: false`.")


@dataclass(frozen=True, slots=True)
class RouteNeutralGateInputContract:
    """Exact producer and exclusion boundary for actor-facing Gate features."""

    history_length_chunks: int
    state_dim: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        state_dim: int,
    ) -> "RouteNeutralGateInputContract":
        root = _mapping(value, name="route-neutral Gate input contract")
        _exact_keys(
            root,
            {
                "schema",
                "visual",
                "language",
                "state",
                "physical_history",
                "forbidden",
            },
            name="route-neutral Gate input contract",
        )
        if root["schema"] != "route-neutral-current-step-v1":
            raise ValueError("Unsupported route-neutral Gate input schema.")

        visual = _mapping(root["visual"], name="route-neutral visual input")
        _exact_keys(
            visual,
            {
                "producer",
                "canonical_regime",
                "sources",
                "include_action_kv",
                "include_generated_future",
                "include_projected_context_kv",
            },
            name="route-neutral visual input",
        )
        if visual["producer"] != "fastwam_value_features":
            raise ValueError(
                "Route-neutral visual input must reuse FastWAMValueFeatures."
            )
        if visual["canonical_regime"] != "idm_parent":
            raise ValueError(
                "Route-neutral visual features require the parent IDM regime."
            )
        if tuple(visual["sources"]) != ("current_frame_video",):
            raise ValueError(
                "Route-neutral visual features must contain current-frame K/V only."
            )
        for field in (
            "include_action_kv",
            "include_generated_future",
            "include_projected_context_kv",
        ):
            _require_false(visual[field], name=f"gate.input_contract.visual.{field}")

        language = _mapping(root["language"], name="route-neutral language input")
        _exact_keys(language, {"source"}, name="route-neutral language input")
        if language["source"] != "pre_action_dit_text_embedding":
            raise ValueError(
                "Route-neutral language must be read before ActionDiT projection."
            )

        state = _mapping(root["state"], name="route-neutral state input")
        _exact_keys(state, {"source"}, name="route-neutral state input")
        if state["source"] != "normalized_proprio":
            raise ValueError("Route-neutral state must use normalized proprioception.")

        history = _mapping(
            root["physical_history"], name="route-neutral physical history"
        )
        _exact_keys(
            history,
            {
                "source",
                "length_chunks",
                "padding",
                "positional_encoding",
                "include_actions",
            },
            name="route-neutral physical history",
        )
        if history["source"] != "normalized_proprio":
            raise ValueError("Physical history must contain normalized proprioception.")
        length = history["length_chunks"]
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError("Physical-history length must be a positive integer.")
        if history["padding"] != "repeat_oldest_or_current":
            raise ValueError("Unsupported route-neutral physical-history padding.")
        _require_false(
            history["positional_encoding"],
            name="gate.input_contract.physical_history.positional_encoding",
        )
        _require_false(
            history["include_actions"],
            name="gate.input_contract.physical_history.include_actions",
        )

        forbidden = _mapping(root["forbidden"], name="route-neutral exclusions")
        expected_forbidden = {
            "current_mode",
            "action_kv",
            "regime_specific_context_kv",
            "route_history",
            "budget_count",
            "chunk_parity",
        }
        _exact_keys(forbidden, expected_forbidden, name="route-neutral exclusions")
        for field in expected_forbidden:
            _require_false(
                forbidden[field], name=f"gate.input_contract.forbidden.{field}"
            )

        if (
            isinstance(state_dim, bool)
            or not isinstance(state_dim, int)
            or state_dim < 1
        ):
            raise ValueError("Route-neutral state dimension must be positive.")
        return cls(history_length_chunks=length, state_dim=state_dim)


@dataclass(frozen=True, slots=True)
class PadCriticWarmupConfig:
    """Runner-update warm-up shared by routing, loss, and cost control."""

    runner_updates: int
    idm_probability: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PadCriticWarmupConfig":
        raw = _mapping(value, name="PAD critic warm-up")
        _exact_keys(
            raw,
            {
                "runner_updates",
                "route_behavior",
                "idm_probability",
                "freeze_gate",
                "freeze_cost_controller",
            },
            name="PAD critic warm-up",
        )
        updates = raw["runner_updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
            raise ValueError("PAD critic warm-up runner_updates must be positive.")
        if raw["route_behavior"] != "independent_random":
            raise ValueError("PAD critic warm-up requires independent_random routing.")
        probability = raw["idm_probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError("PAD warm-up IDM probability must be numeric.")
        probability = float(probability)
        if not math.isfinite(probability) or probability != 0.5:
            raise ValueError(
                "PAD warm-up uses Bernoulli(0.5) so epsilon=1 replay remains exact."
            )
        if raw["freeze_gate"] is not True:
            raise ValueError("PAD critic warm-up must freeze the Gate optimizer owner.")
        if raw["freeze_cost_controller"] is not True:
            raise ValueError("PAD critic warm-up must freeze branch-cost control.")
        return cls(runner_updates=updates, idm_probability=probability)


__all__ = ["PadCriticWarmupConfig", "RouteNeutralGateInputContract"]
