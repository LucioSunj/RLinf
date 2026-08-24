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

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import rlinf.workers.actor.fastwam_selective_sync as selective_sync
import rlinf.workers.actor.fsdp_actor_worker as actor_worker
from rlinf.config import SupportedModel
from rlinf.models.embodiment.wam_policy.critic import (
    FastWAMCurrentFrameValueCritic,
    FastWAMValueTransformerConfig,
)
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


class _CurrentFrameCriticSyncModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Linear(2, 2)
        self.actor.requires_grad_(False)
        self.gate = nn.Linear(2, 1)
        self.lora_A = nn.Parameter(torch.ones(1, 2))
        self.lora_B = nn.Parameter(torch.ones(2, 1))
        self.critic = FastWAMCurrentFrameValueCritic(
            config=FastWAMValueTransformerConfig(
                num_mot_layers=1,
                source_num_heads=1,
                source_head_dim=2,
                layer_indices=(0,),
                hidden_dim=2,
                num_query_tokens=1,
            ),
            hidden_sizes=(4,),
        )


def test_gate_parameter_cpu_snapshot_audit_uses_runner_update_interval() -> None:
    due = EmbodiedFSDPActor._fastwam_gate_parameter_audit_due

    observed = [
        version + 1
        for version in range(25)
        if due(actor_version=version, interval_updates=10)
    ]

    assert observed == [10, 20]
    assert due(actor_version=0, interval_updates=1)
    with pytest.raises(ValueError, match="interval"):
        due(actor_version=0, interval_updates=0)


def test_capture_matches_trainable_and_persistent_sync_contract() -> None:
    module = _SelectiveModule()

    captured = capture_fastwam_sync_tensors(module)

    assert list(captured) == ["trainable", "persistent"]
    assert captured["trainable"].tensor is module.trainable
    assert captured["trainable"].is_parameter
    assert captured["persistent"].tensor is module.persistent
    assert not captured["persistent"].is_parameter


def test_current_frame_critic_sync_has_one_value_head_and_no_actor_copy() -> None:
    module = _CurrentFrameCriticSyncModule()

    captured = capture_fastwam_sync_tensors(module)

    assert len(captured) == len(set(captured))
    assert not any(name.startswith("actor.") for name in captured)
    assert not any(name.startswith("critic.actor.") for name in captured)
    value_names = [name for name in captured if name.startswith("critic.value_head.")]
    assert len(value_names) == len(list(module.critic.value_head.parameters()))


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


def test_handle_replay_initializes_fsdp_before_rebinding_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.fsdp import FullyShardedDataParallel, _runtime_utils

    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.model = object.__new__(FullyShardedDataParallel)
    handle = object()
    events = []

    def fake_lazy_init(model, root):
        assert model is actor.model
        assert root is actor.model
        events.append("lazy_init")

    actor._strategy = SimpleNamespace(
        _iter_fsdp_handles=lambda model: [handle],
        _rebind_handle_views=lambda received: events.append(("rebind", received)),
    )
    monkeypatch.setattr(_runtime_utils, "_lazy_init", fake_lazy_init)

    actor._initialize_fastwam_fsdp_for_handle_replay()
    actor._initialize_fastwam_fsdp_for_handle_replay()

    assert events == [
        "lazy_init",
        ("rebind", handle),
    ]


def _unused_no_shard_handle_fixture():
    flat_parameter = nn.Parameter(torch.arange(6, dtype=torch.float32))
    original_parameter = nn.Parameter(flat_parameter.view(2, 3))
    owner = nn.Module()
    owner.register_parameter("weight", original_parameter)
    flat_parameter._params = [original_parameter]
    flat_parameter._shard_param_infos = [SimpleNamespace(in_shard=True)]
    flat_parameter._param_infos = [("weight", owner, "owner")]
    handle = SimpleNamespace(
        flat_param=flat_parameter,
        uses_sharded_strategy=False,
    )
    return handle, owner, original_parameter, flat_parameter


def test_restore_same_storage_tensor_view_after_unused_fsdp_handle() -> None:
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.model = object()
    handle, owner, original_parameter, flat_parameter = (
        _unused_no_shard_handle_fixture()
    )
    owner._parameters["weight"] = flat_parameter.view(2, 3)
    rebound = []

    def rebind(received) -> None:
        assert received is handle
        owner._parameters["weight"] = original_parameter
        original_parameter.data = flat_parameter.view(2, 3)
        rebound.append(received)

    actor._strategy = SimpleNamespace(
        _iter_fsdp_handles=lambda model: [handle],
        _rebind_handle_views=rebind,
    )

    assert actor._restore_fastwam_fsdp_parameter_views_after_backward() == 1
    assert owner.weight is original_parameter
    assert rebound == [handle]
    assert actor._restore_fastwam_fsdp_parameter_views_after_backward() == 0


def test_restore_rejects_non_fsdp_parameter_replacement() -> None:
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.model = object()
    handle, owner, _, _ = _unused_no_shard_handle_fixture()
    owner._parameters["weight"] = nn.Parameter(torch.zeros(2, 3))
    actor._strategy = SimpleNamespace(
        _iter_fsdp_handles=lambda model: [handle],
        _rebind_handle_views=lambda received: None,
    )

    with pytest.raises(RuntimeError, match="non-recoverable parameter replacement"):
        actor._restore_fastwam_fsdp_parameter_views_after_backward()
