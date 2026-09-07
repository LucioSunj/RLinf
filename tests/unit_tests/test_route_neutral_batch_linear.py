# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import torch
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.batch_linear import (
    BatchInvariantLinear,
    BatchLinearContext,
)
from torch import nn

from rlinf.models.embodiment.wam_policy.online_idm_bc.runtime import (
    OnlineIDMTeacherLiberoRuntime,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online import (
    runtime as runtime_module,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.runtime import (
    RouteNeutralOnlineIDMTeacherLiberoRuntime,
)


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proprio_dim = 2
        self.proprio_encoder = object()
        self.video_expert = nn.Module()
        self.video_expert.time_embedding = nn.Sequential(nn.Linear(3, 3))
        self.action_expert = nn.Sequential(nn.Linear(3, 3))


def test_route_runtime_installs_batch_linear_context(monkeypatch) -> None:
    actor = _Actor()

    def initialize_base(self, **kwargs) -> None:
        self.actor = actor

    monkeypatch.setattr(OnlineIDMTeacherLiberoRuntime, "__init__", initialize_base)
    monkeypatch.setattr(
        runtime_module.RouteNeutralGateInputContract,
        "from_mapping",
        lambda value, *, state_dim: SimpleNamespace(state_dim=state_dim),
    )
    monkeypatch.setattr(
        runtime_module.FastWAMValueTransformerConfig,
        "materialize",
        lambda value: SimpleNamespace(sources=("current_frame_video",)),
    )
    monkeypatch.setattr(
        runtime_module,
        "PhysicalStateHistoryTracker",
        lambda value: SimpleNamespace(contract=value),
    )

    runtime = RouteNeutralOnlineIDMTeacherLiberoRuntime(
        route_neutral_input={},
        route_neutral_visual={},
    )

    assert isinstance(runtime.batch_linear_context, BatchLinearContext)
    assert isinstance(actor.action_expert[0], BatchInvariantLinear)


def test_condition_preparation_binds_real_image_batch(monkeypatch) -> None:
    runtime = RouteNeutralOnlineIDMTeacherLiberoRuntime.__new__(
        RouteNeutralOnlineIDMTeacherLiberoRuntime
    )
    runtime.batch_linear_context = BatchLinearContext()
    expected = object()

    def prepare_base(self, **kwargs):
        assert self.batch_linear_context.batch_size == 4
        return expected, None

    monkeypatch.setattr(
        OnlineIDMTeacherLiberoRuntime,
        "_prepare_action_condition",
        prepare_base,
    )
    condition, replay = runtime._prepare_action_condition(
        image=torch.empty(4, 3, 8, 8),
        context=torch.empty(4, 2, 3),
        context_mask=torch.ones(4, 2, dtype=torch.bool),
        regime=PolicyRegime.UNCOND,
    )

    assert condition is expected
    assert replay is None
    assert runtime.batch_linear_context.batch_size is None


def test_all_route_velocity_calls_receive_batch_linear_context(monkeypatch) -> None:
    runtime = RouteNeutralOnlineIDMTeacherLiberoRuntime.__new__(
        RouteNeutralOnlineIDMTeacherLiberoRuntime
    )
    runtime.batch_linear_context = BatchLinearContext()
    velocity = SimpleNamespace(batch_linear_context=None)

    monkeypatch.setattr(
        OnlineIDMTeacherLiberoRuntime,
        "_velocity",
        lambda self, condition, **kwargs: velocity,
    )
    result = runtime._velocity(
        SimpleNamespace(),
        regime=PolicyRegime.UNCOND,
        capture_gate_kv=False,
        actor_version=0,
    )

    assert result is velocity
    assert result.batch_linear_context is runtime.batch_linear_context
