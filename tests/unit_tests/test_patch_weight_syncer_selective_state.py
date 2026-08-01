# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contracts for exact selective sender state in patch weight sync."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

import pytest
import torch

from rlinf.hybrid_engines.weight_syncer.patch_syncer import (
    CPUSnapshotPatchBuilder,
    EmptyWeightPatch,
    GPUSnapshotPatchBuilder,
    PatchWeightSyncer,
)
from rlinf.scheduler import Worker


def _full_state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        frozen=torch.zeros(2),
        trainable=torch.ones(2),
        persistent_buffer=torch.full((1,), 2.0),
    )


def _selective_state() -> OrderedDict[str, torch.Tensor]:
    full_state = _full_state()
    return OrderedDict(
        (key, full_state[key]) for key in ("trainable", "persistent_buffer")
    )


def _receiver_metadata() -> dict[str, object]:
    full_state = _full_state()
    return {
        "ordered_keys": list(full_state),
        "original_shapes": {key: value.shape for key, value in full_state.items()},
        "receiver_dtypes": {key: value.dtype for key, value in full_state.items()},
    }


@pytest.mark.parametrize(
    "builder_cls", [CPUSnapshotPatchBuilder, GPUSnapshotPatchBuilder]
)
def test_patch_builder_accepts_exact_selective_sender_state(builder_cls) -> None:
    full_state = _full_state()
    builder = builder_cls(
        snapshot=None,
        ordered_keys=list(full_state),
        param_names_need_sync=["trainable", "persistent_buffer"],
        original_shapes={key: value.shape for key, value in full_state.items()},
        transport_device=torch.device("cpu"),
        delta_encoding=True,
    )

    patch = builder.create_patch(_selective_state(), version=3)

    assert isinstance(patch, EmptyWeightPatch)
    assert int(patch.version.item()) == 3


@pytest.mark.parametrize(
    "builder_cls", [CPUSnapshotPatchBuilder, GPUSnapshotPatchBuilder]
)
def test_patch_builder_rejects_inexact_partial_sender_state(builder_cls) -> None:
    full_state = _full_state()
    builder = builder_cls(
        snapshot=None,
        ordered_keys=list(full_state),
        param_names_need_sync=["trainable", "persistent_buffer"],
        original_shapes={key: value.shape for key, value in full_state.items()},
        transport_device=torch.device("cpu"),
        delta_encoding=True,
    )

    with pytest.raises(ValueError, match="State dict keys do not match snapshot keys"):
        builder.create_patch(
            OrderedDict(trainable=torch.ones(2)),
            version=3,
        )


def test_patch_syncer_sender_accepts_exact_selective_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        compression_algorithm="none",
    )

    async def recv():
        return _receiver_metadata()

    async def send(_payload):
        raise AssertionError("init sync is disabled")

    async def run() -> None:
        await syncer.init_sender(
            state_dict=_selective_state(),
            param_names_need_sync=["trainable", "persistent_buffer"],
            send=send,
            recv=recv,
            is_sender=False,
        )

    # The inactive-sender path performs no device copy. Treating CPU as the
    # worker accelerator keeps this validation-only test hardware independent.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    asyncio.run(run())

    assert syncer.sender_initialized()
    assert syncer.snapshot is None
    patch = syncer.create_patch(_selective_state(), version=5)
    assert isinstance(patch, EmptyWeightPatch)


def test_patch_syncer_sender_rejects_inexact_partial_state() -> None:
    syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        compression_algorithm="none",
    )

    async def recv():
        return _receiver_metadata()

    async def send(_payload):
        raise AssertionError("init sync is disabled")

    async def run() -> None:
        await syncer.init_sender(
            state_dict=OrderedDict(trainable=torch.ones(2)),
            param_names_need_sync=["trainable", "persistent_buffer"],
            send=send,
            recv=recv,
            is_sender=False,
        )

    with pytest.raises(ValueError, match="exactly param_names_need_sync"):
        asyncio.run(run())
