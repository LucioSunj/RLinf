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

"""Contracts for FastWAM selective FSDP rollout synchronization."""

from __future__ import annotations

from types import MethodType

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import rlinf.workers.actor.fastwam_selective_sync as selective_sync
import rlinf.workers.actor.fsdp_actor_worker as actor_worker
from rlinf.config import SupportedModel
from rlinf.workers.actor.fastwam_selective_sync import (
    CapturedSyncTensor,
    capture_fastwam_sync_tensors,
    materialize_fastwam_sync_state,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class _SelectiveModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.frozen = nn.Parameter(torch.ones(3), requires_grad=False)
        self.register_buffer("persistent", torch.tensor([7.0]))
        self.register_buffer("ephemeral", torch.tensor([8.0]), persistent=False)


def test_capture_matches_trainable_and_persistent_sync_contract() -> None:
    module = _SelectiveModule()

    captured = capture_fastwam_sync_tensors(module)

    assert list(captured) == ["trainable", "persistent"]
    assert captured["trainable"].tensor is module.trainable
    assert captured["trainable"].is_parameter
    assert captured["persistent"].tensor is module.persistent
    assert not captured["persistent"].is_parameter


def test_capture_includes_visual_router_but_excludes_frozen_dino() -> None:
    module = nn.Module()
    module.visual_reader = nn.Linear(3, 2, bias=False)
    module.visual_encoder = nn.Linear(3, 2, bias=False).requires_grad_(False)

    captured = capture_fastwam_sync_tensors(module)

    assert list(captured) == ["visual_reader.weight"]
    assert "visual_encoder.weight" not in captured


def test_single_rank_materialization_keeps_only_selected_live_storage() -> None:
    module = _SelectiveModule()
    captured = capture_fastwam_sync_tensors(module)

    state = materialize_fastwam_sync_state(
        captured,
        ["trainable", "persistent"],
    )

    assert list(state) == ["trainable", "persistent"]
    assert state["trainable"].data_ptr() == module.trainable.data_ptr()
    assert state["persistent"].data_ptr() == module.persistent.data_ptr()
    assert "frozen" not in state


def test_materialization_rejects_key_drift() -> None:
    captured = capture_fastwam_sync_tensors(_SelectiveModule())

    with pytest.raises(RuntimeError, match="keys changed"):
        materialize_fastwam_sync_state(captured, ["trainable"])


def test_materialization_reconstructs_only_selected_two_rank_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {
        "trainable": CapturedSyncTensor(
            tensor=torch.tensor([1.0, 2.0]),
            original_shape=torch.Size([4]),
            is_parameter=True,
        )
    }
    calls = 0

    def fake_all_gather(outputs, value, group=None):
        nonlocal calls
        del group
        calls += 1
        if calls == 1:
            outputs[0].copy_(torch.tensor([2], dtype=value.dtype))
            outputs[1].copy_(torch.tensor([2], dtype=value.dtype))
        else:
            outputs[0].copy_(torch.tensor([1.0, 2.0], dtype=value.dtype))
            outputs[1].copy_(torch.tensor([3.0, 4.0], dtype=value.dtype))

    monkeypatch.setattr(selective_sync.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(selective_sync.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(selective_sync.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(selective_sync.dist, "all_gather", fake_all_gather)

    state = materialize_fastwam_sync_state(captured, ["trainable"])

    torch.testing.assert_close(state["trainable"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert calls == 2


def test_materialization_rejects_incomplete_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {
        "trainable": CapturedSyncTensor(
            tensor=torch.tensor([1.0]),
            original_shape=torch.Size([4]),
            is_parameter=True,
        )
    }

    def fake_all_gather(outputs, value, group=None):
        del group
        outputs[0].copy_(torch.tensor([1], dtype=value.dtype))
        outputs[1].copy_(torch.tensor([1], dtype=value.dtype))

    monkeypatch.setattr(selective_sync.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(selective_sync.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(selective_sync.dist, "all_gather", fake_all_gather)

    with pytest.raises(RuntimeError, match="do not reconstruct"):
        materialize_fastwam_sync_state(captured, ["trainable"])


def test_fastwam_actor_rollout_state_never_calls_full_state_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {"actor": {"model": {"model_type": SupportedModel.FASTWAM_ADAPTIVE.value}}}
    )
    captured = {"trainable": object()}
    actor._fastwam_rollout_sync_tensors = captured
    actor.param_names_need_sync = ["trainable"]

    def forbidden_full_state_dict(self, *, cpu_offload, full_state_dict):
        del self, cpu_offload, full_state_dict
        raise AssertionError("FastWAM must not build the full frozen state dict")

    actor.get_model_state_dict = MethodType(forbidden_full_state_dict, actor)
    expected = {"trainable": torch.tensor([1.0])}

    def fake_materialize(received, names):
        assert received is captured
        assert names == ["trainable"]
        return expected

    monkeypatch.setattr(
        actor_worker, "materialize_fastwam_sync_state", fake_materialize
    )

    assert actor.get_rollout_state_dict() is expected
