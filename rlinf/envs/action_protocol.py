# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Environment-independent action-prefix operations for model replay."""

from __future__ import annotations

from typing import Protocol, TypeVar

TensorLike = TypeVar("TensorLike")


class ActionHorizonProtocol(Protocol):
    """Generated and requested execution lengths, before hardware feedback."""

    generation_horizon: int
    execution_horizon: int


def select_executed_action_prefix(
    values: TensorLike, *, protocol: ActionHorizonProtocol
) -> TensorLike:
    """Select the declared execution prefix without clipping or mutation."""
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("FastWAM generated Actions must have shape [B, H, ...].")
    if int(shape[1]) != protocol.generation_horizon:
        raise ValueError(
            "FastWAM generated horizon differs from the declared action protocol: "
            f"{int(shape[1])} != {protocol.generation_horizon}."
        )
    return values[:, : protocol.execution_horizon]


def select_executed_flow_statistics(
    values: TensorLike, *, protocol: ActionHorizonProtocol
) -> TensorLike:
    """Select a planned prefix while preserving disabled evaluation replay."""
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("FastWAM Flow statistics must have shape [B, H, ...].")
    if int(shape[1]) == 0:
        return values
    return select_executed_action_prefix(values, protocol=protocol)
