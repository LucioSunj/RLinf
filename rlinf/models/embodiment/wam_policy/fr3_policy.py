# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FR3 builder for the shared current-step mixed policy."""

from __future__ import annotations

from typing import Any

from .fr3_runtime import Fr3RouteNeutralRuntime
from .route_neutral_online.builder import build_route_neutral_online_idm_bc_model


def build_fr3_mixed_model(cfg: Any, torch_dtype: Any):
    """Construct the shared frozen parent, dual adapters, Gate and optional value."""
    from .fr3_config import validate_fr3_model_config

    validate_fr3_model_config(cfg)
    policy = build_route_neutral_online_idm_bc_model(cfg, torch_dtype)
    if not isinstance(policy.runtime, Fr3RouteNeutralRuntime):
        raise TypeError("FR3 builder requires Fr3RouteNeutralRuntime.")
    return policy
