# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pre-treatment LIBERO phase callback contract for adaptive-gate controls."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


UNKNOWN_PHASE = "unknown"
GATE_PHASES = (
    "approach",
    "contact_alignment",
    "transport_completion",
    UNKNOWN_PHASE,
)

_ALIASES = {
    "approach": "approach",
    "free_motion": "approach",
    "approach_free_motion": "approach",
    "contact": "contact_alignment",
    "alignment": "contact_alignment",
    "contact_alignment": "contact_alignment",
    "transport": "transport_completion",
    "completion": "transport_completion",
    "transport_completion": "transport_completion",
    "unknown": UNKNOWN_PHASE,
}


def load_gate_phase_callback(spec: str | None):
    """Resolve a configured ``module:function`` callback, or return ``None``."""
    if spec is None or str(spec).strip().lower() in {"", "none", "null"}:
        return None
    value = str(spec).strip()
    if ":" not in value:
        raise ValueError("gate_phase_fn must have the form python.module:function")
    module_name, function_name = value.rsplit(":", 1)
    callback = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(callback):
        raise ValueError(f"gate_phase_fn={value!r} does not resolve to a callable")
    return callback


def evaluate_gate_phase(callback, **pre_treatment_context: Any) -> tuple[str, bool]:
    """Run a phase predicate fail-closed against only the current state.

    The callback may return ``(phase, reliable)`` or a mapping containing
    ``phase`` and ``reliable``/``phase_reliable``. Exceptions, invalid labels,
    and unknown/unreliable decisions all become ``("unknown", False)``.
    """
    if callback is None:
        return UNKNOWN_PHASE, False
    try:
        result = callback(**pre_treatment_context)
        if isinstance(result, Mapping):
            phase = result.get("phase", UNKNOWN_PHASE)
            reliable = result.get(
                "phase_reliable", result.get("reliable", False)
            )
        elif isinstance(result, (tuple, list)) and len(result) == 2:
            phase, reliable = result
        else:
            return UNKNOWN_PHASE, False
        normalized = str(phase).strip().lower().replace("/", "_").replace("-", "_")
        normalized = _ALIASES.get(normalized, UNKNOWN_PHASE)
        reliable = bool(reliable) and normalized != UNKNOWN_PHASE
        return (normalized, True) if reliable else (UNKNOWN_PHASE, False)
    except Exception:
        return UNKNOWN_PHASE, False


def evaluate_worker_gate_phase(
    env: Any, callback_spec: str, context: Mapping[str, Any]
) -> tuple[str, bool]:
    """Evaluate a callback beside the real worker env and current simulator state."""
    from rlinf.envs.libero.gate_snapshot import (
        _get_observation,
        capture_worker_snapshot,
        restore_worker_snapshot,
    )

    try:
        snapshot = capture_worker_snapshot(env)
    except Exception:
        # Phase is analysis metadata, so unsupported workers degrade to UNKNOWN
        # rather than running a potentially stateful predicate unprotected.
        return UNKNOWN_PHASE, False
    try:
        callback = load_gate_phase_callback(callback_spec)
        raw_observation = _get_observation(env)
        return evaluate_gate_phase(
            callback,
            env=env,
            raw_observation=raw_observation,
            **dict(context),
        )
    except Exception:
        return UNKNOWN_PHASE, False
    finally:
        # Treat even project-owned callbacks as untrusted observers.  A phase
        # predicate may inspect contacts through APIs that mutate MuJoCo,
        # controller targets, wrapper counters, or RNG streams.
        restore_worker_snapshot(env, snapshot)
