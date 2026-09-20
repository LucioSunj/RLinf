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
import copy
import weakref
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from rlinf.hybrid_engines.weight_syncer.patch_syncer import (
    CPUSnapshotPatchBuilder,
    EmptyWeightPatch,
    GPUSnapshotPatchBuilder,
    PatchWeightSyncer,
    WeightPatch,
    WeightPatchSequence,
    _PrefetchedCPUSnapshot,
    as_coo_2d_view,
)
from rlinf.scheduler import Worker
from rlinf.utils.utils import collect_param_names_need_sync


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


@pytest.mark.parametrize("delta_encoding", [False, True])
@pytest.mark.parametrize(
    "builder_cls", [CPUSnapshotPatchBuilder, GPUSnapshotPatchBuilder]
)
def test_chunked_patch_preserves_seven_receiver_states_and_releases_payloads(
    monkeypatch: pytest.MonkeyPatch, builder_cls, delta_encoding: bool
) -> None:
    """Exercise dense, empty and sparse updates with bounded live transfers."""
    stream = SimpleNamespace(wait_event=lambda _event: None, synchronize=lambda: None)
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    monkeypatch.setattr(
        Worker,
        "torch_platform",
        SimpleNamespace(
            current_stream=lambda *_args: stream, is_initialized=lambda: True
        ),
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_args: None)

    def cpu_prefetch(self, state, ordinal):
        key = self.param_names_need_sync[ordinal]
        return _PrefetchedCPUSnapshot(
            ordinal=ordinal,
            global_ordinal=self.param_names_need_sync_ordinals[key],
            key=key,
            state_2dview=as_coo_2d_view(state[key])[0],
            snapshot_value=self.snapshot[key],
            snapshot_on_state_device=self.snapshot[key].clone(),
            copy_done=None,
        )

    # CPU tests exercise the actual comparison/index/snapshot code. Only the
    # accelerator copy stream is replaced; production CUDA is checked by the run.
    monkeypatch.setattr(CPUSnapshotPatchBuilder, "_prefetch_snapshot", cpu_prefetch)

    model = torch.nn.Module()
    for name, rows in (
        ("gate", 4),
        ("action_lora", 4),
        ("video_lora", 5),
        ("value_head", 2),
    ):
        model.register_parameter(
            name,
            torch.nn.Parameter(torch.arange(rows * 4).float().reshape(rows, 4)),
        )
    model.register_parameter(
        "frozen", torch.nn.Parameter(torch.ones(8, 8), requires_grad=False)
    )
    model.register_buffer("persistent", torch.tensor([2**40, 9], dtype=torch.int64))
    receivers = [copy.deepcopy(model) for _ in range(7)]
    selected = collect_param_names_need_sync(model)
    ordered = list(reversed(model.state_dict()))
    shapes = {key: value.shape for key, value in model.state_dict().items()}
    snapshot = {
        key: as_coo_2d_view(value.detach().clone())[0]
        for key, value in model.state_dict().items()
        if key in selected
    }
    sender = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=delta_encoding,
        init_sync_bucket_size=64,
    )
    sender.snapshot = snapshot
    sender.ordered_keys = ordered
    sender.original_shapes = shapes
    sender.param_names_need_sync = selected
    sender.patch_builder = builder_cls(
        snapshot=snapshot,
        ordered_keys=ordered,
        param_names_need_sync=selected,
        original_shapes=shapes,
        transport_device=torch.device("cpu"),
        delta_encoding=delta_encoding,
    )
    receiver_syncers = []
    for _ in receivers:
        syncer = PatchWeightSyncer(
            snapshot_device="cpu",
            transport_device="cpu",
            delta_encoding=delta_encoding,
        )
        syncer.ordered_keys = ordered
        syncer.original_shapes = shapes
        receiver_syncers.append(syncer)

    async def transfer(version):
        queues = [asyncio.Queue(maxsize=1) for _ in receivers]
        previous_payload_refs = []
        payload_sizes = []
        headers = []

        async def send(payload):
            assert all(ref() is None for ref in previous_payload_refs)
            previous_payload_refs.clear()
            if isinstance(payload, WeightPatchSequence):
                headers.append(int(payload.patch_count))
            else:
                previous_payload_refs.extend(weakref.ref(t) for t in payload.tensors())
                if isinstance(payload, WeightPatch):
                    payload_sizes.append(int(payload.values.numel()))
            for queue in queues:
                await queue.put(copy.deepcopy(payload))
            await asyncio.gather(*(queue.join() for queue in queues))

        async def receive(syncer, receiver, queue):
            async def recv():
                payload = await queue.get()
                queue.task_done()
                return payload

            return await syncer.apply(receiver, recv)

        state = OrderedDict((key, model.state_dict()[key]) for key in selected)
        results = await asyncio.gather(
            sender.sync(state, send, version),
            *(
                receive(syncer, receiver, queue)
                for syncer, receiver, queue in zip(
                    receiver_syncers, receivers, queues, strict=True
                )
            ),
        )
        assert results[1:] == [version] * 7
        assert len(headers) == 1 and headers[0] > 1
        assert max(payload_sizes, default=0) <= 80  # One 80-byte tensor exceeds64.
        assert all(ref() is None for ref in previous_payload_refs)
        for receiver in receivers:
            for key, expected in model.state_dict().items():
                actual = receiver.state_dict()[key]
                assert actual.dtype == expected.dtype and torch.equal(actual, expected)
        for key in selected:
            assert torch.equal(
                snapshot[key].reshape(shapes[key]), model.state_dict()[key]
            )
        return payload_sizes

    async def run():
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.25)
            model.persistent.add_(3)
        rng_before = torch.get_rng_state().clone()
        assert await transfer(6)
        assert await transfer(7) == []
        with torch.no_grad():
            model.video_lora[2, 1] = -7.5
        assert await transfer(8) == [4]
        assert torch.equal(torch.get_rng_state(), rng_before)

    asyncio.run(run())
