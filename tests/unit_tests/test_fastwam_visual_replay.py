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

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.visual_contracts import (
    NativePatchMemory,
    PreparedCameraBatch,
)

from rlinf.models.embodiment.wam_policy import visual_replay
from rlinf.models.embodiment.wam_policy.libero_runtime import LiberoFastWAMRuntime
from rlinf.models.embodiment.wam_policy.visual_replay import (
    DualVisualReplayBackend,
    DualVisualReplayConfig,
    NativeMemoryIdentity,
    PackedDualVisualReplay,
    pack_dual_visual_replay,
    pin_dual_visual_forward_inputs,
    validate_dual_visual_aggregate_bytes,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _cameras() -> PreparedCameraBatch:
    return PreparedCameraBatch(
        pixels=torch.full((2, 2, 3, 224, 224), 127, dtype=torch.uint8),
        camera_ids=("main", "wrist"),
        camera_valid_mask=torch.tensor([[True, False], [True, True]]),
        input_contract_sha256=_hash("input"),
    )


def _identity() -> NativeMemoryIdentity:
    return NativeMemoryIdentity(
        camera_ids=("main", "wrist"),
        source_revision="revision",
        weights_sha256=_hash("weights"),
        input_contract_sha256=_hash("input"),
        preprocess_sha256=_hash("preprocess"),
        output_contract_sha256=_hash("output"),
        memory_contract_sha256=_hash("memory"),
    )


def _memory(camera_mask: torch.Tensor) -> NativePatchMemory:
    tokens = torch.randn(1, 2, 196, 384)
    tokens[:, ~camera_mask].zero_()
    identity = _identity()
    return NativePatchMemory(
        tokens=tokens,
        patch_valid_mask=camera_mask[None, :, None].expand(-1, -1, 196),
        camera_valid_mask=camera_mask[None],
        camera_ids=identity.camera_ids,
        grid=(14, 14),
        source_revision=identity.source_revision,
        weights_sha256=identity.weights_sha256,
        input_contract_sha256=identity.input_contract_sha256,
        preprocess_sha256=identity.preprocess_sha256,
        output_contract_sha256=identity.output_contract_sha256,
        memory_contract_sha256=identity.memory_contract_sha256,
    )


def _config(backend: str, *, sample_cap: int = 2_000_000):
    return DualVisualReplayConfig(
        backend=backend,
        storage_dtype="bfloat16",
        pin_memory=True,
        max_bytes_per_sample=sample_cap,
        max_bytes_aggregate=4_000_000,
        fail_closed=True,
    )


def test_recompute_native_record_round_trips_typed_inputs() -> None:
    packed = pack_dual_visual_replay(
        config=_config("recompute_native"),
        cameras=_cameras(),
        present_mask=torch.tensor([False, True]),
        target_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        memory_contract_sha256=_hash("memory"),
        transport_sha256=_hash("transport"),
        actor_version=7,
    )

    restored = PackedDualVisualReplay.from_forward_inputs(
        {
            "fastwam_p7_visual_proprio": torch.ones(2, 8),
            **packed.as_forward_inputs(),
        }
    )

    assert restored.backend is DualVisualReplayBackend.RECOMPUTE_NATIVE
    assert restored.camera_pixels.shape == (2, 2, 3, 224, 224)
    assert restored.effective_transport_hash.shape == (2, 32)
    assert not torch.equal(
        restored.effective_transport_hash[0],
        restored.effective_transport_hash[1],
    )

    runtime = object.__new__(LiberoFastWAMRuntime)
    runtime.visual_reader = object()
    runtime.visual_replay_config = _config("recompute_native")
    runtime.visual_memory_identity = _identity()
    runtime.visual_geometry = SimpleNamespace(transport_sha256=_hash("transport"))
    runtime_record = runtime._visual_replay_record(
        {
            "fastwam_p7_visual_proprio": torch.ones(2, 8),
            **packed.as_forward_inputs(),
        },
        expected_actor_versions=torch.full((2,), 7, dtype=torch.long),
        expected_present_mask=torch.tensor([False, True]),
    )
    assert runtime_record is not None
    with pytest.raises(ValueError, match="presence differs"):
        runtime._visual_replay_record(
            {
                "fastwam_p7_visual_proprio": torch.ones(2, 8),
                **packed.as_forward_inputs(),
            },
            expected_present_mask=torch.tensor([True, False]),
        )

    runtime.visual_reader = None
    with pytest.raises(ValueError, match="Disabled P7"):
        runtime._visual_replay_record({"fastwam_p7_visual_proprio": torch.ones(2, 8)})


def test_stored_native_keeps_idm_empty_and_materializes_uncond() -> None:
    memory = _memory(torch.tensor([True, True]))
    packed = pack_dual_visual_replay(
        config=_config("stored_native"),
        cameras=_cameras(),
        present_mask=torch.tensor([False, True]),
        target_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        memory_contract_sha256=_hash("memory"),
        transport_sha256=_hash("transport"),
        actor_version=7,
        native_memories=(None, memory),
    )

    assert torch.count_nonzero(packed.native_tokens[0]) == 0
    assert not packed.patch_valid_mask[0].any()
    restored = packed.native_memory(
        1,
        identity=_identity(),
        device="cpu",
        dtype=torch.float32,
    )
    assert restored.tokens.shape == (1, 2, 196, 384)
    assert torch.equal(restored.camera_valid_mask, torch.tensor([[True, True]]))
    with pytest.raises(ValueError, match="IDM rows"):
        packed.native_memory(
            0,
            identity=_identity(),
            device="cpu",
            dtype=torch.float32,
        )


def test_visual_replay_caps_and_wan_v_backend_fail_closed() -> None:
    with pytest.raises(ValueError, match="capacity PASS"):
        DualVisualReplayConfig(
            backend="stored_native_and_wan_v",
            storage_dtype="bfloat16",
            pin_memory=True,
            max_bytes_per_sample=1,
            max_bytes_aggregate=1,
            fail_closed=True,
        )
    with pytest.raises(MemoryError, match="max_bytes_per_sample"):
        pack_dual_visual_replay(
            config=_config("recompute_native", sample_cap=128),
            cameras=_cameras(),
            present_mask=torch.tensor([False, True]),
            target_valid_mask=torch.ones(2, 4, dtype=torch.bool),
            memory_contract_sha256=_hash("memory"),
            transport_sha256=_hash("transport"),
            actor_version=7,
        )


def test_pin_failure_is_permitted_only_on_cpu_only_hosts(monkeypatch) -> None:
    tensor = torch.zeros(1)

    def _fail_pin(_tensor):
        raise RuntimeError("pinned allocator failed")

    monkeypatch.setattr(torch.Tensor, "pin_memory", _fail_pin)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert visual_replay._pin(tensor) is tensor

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="pinned allocator failed"):
        visual_replay._pin(tensor)


def test_repin_helper_touches_only_p7_packed_fields(monkeypatch) -> None:
    calls = []

    def _record_pin(tensor):
        calls.append(tensor)
        return tensor

    monkeypatch.setattr(visual_replay, "_pin", _record_pin)
    p7_tensor = torch.zeros(2)
    proprio = torch.ones(2, 8)
    result = pin_dual_visual_forward_inputs(
        {
            "p7_visual_present_mask": p7_tensor,
            "fastwam_p7_visual_proprio": proprio,
        }
    )

    assert len(calls) == 1 and calls[0] is p7_tensor
    assert result["fastwam_p7_visual_proprio"] is proprio
    assert validate_dual_visual_aggregate_bytes(
        result,
        max_bytes_aggregate=128,
    ) == (
        p7_tensor.numel() * p7_tensor.element_size()
        + proprio.numel() * proprio.element_size()
    )
    with pytest.raises(MemoryError, match="actor replay"):
        validate_dual_visual_aggregate_bytes(
            result,
            max_bytes_aggregate=1,
        )


def test_visual_replay_rejects_route_memory_disagreement() -> None:
    memory = _memory(torch.tensor([True, True]))
    with pytest.raises(ValueError, match="route presence"):
        pack_dual_visual_replay(
            config=_config("stored_native"),
            cameras=_cameras(),
            present_mask=torch.tensor([False, True]),
            target_valid_mask=torch.ones(2, 4, dtype=torch.bool),
            memory_contract_sha256=_hash("memory"),
            transport_sha256=_hash("transport"),
            actor_version=7,
            native_memories=(memory, memory),
        )


def test_visual_replay_rejects_mask_and_patch_validity_tampering() -> None:
    packed = pack_dual_visual_replay(
        config=_config("recompute_native"),
        cameras=_cameras(),
        present_mask=torch.tensor([False, True]),
        target_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        memory_contract_sha256=_hash("memory"),
        transport_sha256=_hash("transport"),
        actor_version=7,
    )
    packed.target_valid_mask[1, 0, 0] = False
    with pytest.raises(ValueError, match="effective transport provenance"):
        packed.validate_contract(
            backend="recompute_native",
            memory_contract_sha256=_hash("memory"),
            transport_sha256=_hash("transport"),
        )

    camera_tampered = pack_dual_visual_replay(
        config=_config("recompute_native"),
        cameras=_cameras(),
        present_mask=torch.tensor([False, True]),
        target_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        memory_contract_sha256=_hash("memory"),
        transport_sha256=_hash("transport"),
        actor_version=7,
    )
    camera_tampered.camera_valid_mask[0, 1] = True
    with pytest.raises(ValueError, match="active-camera count"):
        camera_tampered.validate_contract(
            backend="recompute_native",
            memory_contract_sha256=_hash("memory"),
            transport_sha256=_hash("transport"),
        )

    memory = _memory(torch.tensor([True, True]))
    stored = pack_dual_visual_replay(
        config=_config("stored_native"),
        cameras=_cameras(),
        present_mask=torch.tensor([False, True]),
        target_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        memory_contract_sha256=_hash("memory"),
        transport_sha256=_hash("transport"),
        actor_version=7,
        native_memories=(None, memory),
    )
    fields = {
        **stored.__dict__,
        "patch_valid_mask": stored.patch_valid_mask.clone(),
    }
    fields["patch_valid_mask"][1, 1].zero_()
    with pytest.raises(ValueError, match="patch validity disagrees"):
        PackedDualVisualReplay(**fields)


def test_runtime_idm_route_does_not_call_dino_encoder() -> None:
    class _Encoder:
        def __init__(self):
            self.calls = 0

        def prepare_memory(self, regime, cameras):
            del regime, cameras
            self.calls += 1
            return object()

    runtime = object.__new__(LiberoFastWAMRuntime)
    runtime.visual_reader = object()
    runtime.visual_encoder = _Encoder()
    cameras = _cameras()

    assert (
        runtime._prepare_visual_memory(regime=PolicyRegime.IDM, cameras=cameras) is None
    )
    assert runtime.visual_encoder.calls == 0
    assert (
        runtime._prepare_visual_memory(
            regime=PolicyRegime.UNCOND,
            cameras=cameras,
        )
        is not None
    )
    assert runtime.visual_encoder.calls == 1
