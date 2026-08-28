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

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

FASTWAM_SRC = Path(__file__).resolve().parents[3] / "FastWAM/src"
sys.path.insert(0, str(FASTWAM_SRC))

from fastwam.adapters import PolicyRegime  # noqa: E402
from fastwam.models.wan22.kv_tap import (  # noqa: E402
    GateKVSnapshot,
    GateLayerKV,
    KeyValueBank,
    KVSource,
)

KV_REPLAY_PATH = (
    Path(__file__).resolve().parents[2]
    / "rlinf/models/embodiment/wam_policy/kv_replay.py"
)
_spec = importlib.util.spec_from_file_location(
    "fastwam_kv_replay_under_test", KV_REPLAY_PATH
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
GateKVReplayBackend = _module.GateKVReplayBackend
GateKVReplayConfig = _module.GateKVReplayConfig
GateKVReplayRecord = _module.GateKVReplayRecord
offload_gate_kv = _module.offload_gate_kv
pack_gate_kv = _module.pack_gate_kv
pin_gate_kv_forward_inputs = _module.pin_gate_kv_forward_inputs
PackedGateKVTaps = _module.PackedGateKVTaps


def _bank(source, values):
    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 2)
    return KeyValueBank(
        source=source,
        key=tensor,
        value=tensor + 1,
        valid_mask=torch.ones(tensor.shape[:2], dtype=torch.bool),
    )


def _snapshot(action_offset, timestep):
    return GateKVSnapshot(
        (
            GateLayerKV(
                layer_index=0,
                denoise_timestep=torch.tensor([timestep]),
                current_mode=(PolicyRegime.UNCOND,),
                current_frame_video=_bank(KVSource.CURRENT_FRAME_VIDEO, [1.0, 2.0]),
                action=_bank(
                    KVSource.ACTION,
                    [3.0 + action_offset, 4.0 + action_offset],
                ),
                context=_bank(KVSource.TEXT_STATE_CONTEXT, [5.0, 6.0]),
                actor_version=7,
            ),
        )
    )


def test_default_config_is_stored_bfloat16():
    config = GateKVReplayConfig()
    assert config.backend is GateKVReplayBackend.STORED
    assert config.torch_dtype is torch.bfloat16
    assert config.gate_kv_sample_budget is None
    assert config.gate_kv_sample_seed == 0


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5])
def test_gate_kv_sample_budget_rejects_invalid_values(budget):
    with pytest.raises(ValueError, match="gate_kv_sample_budget"):
        GateKVReplayConfig(gate_kv_sample_budget=budget)


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_gate_kv_sample_seed_rejects_invalid_values(seed):
    with pytest.raises(ValueError, match="gate_kv_sample_seed"):
        GateKVReplayConfig(gate_kv_sample_seed=seed)


def test_stored_config_rejects_fake_deduplication_toggle():
    with pytest.raises(ValueError, match="must remain true"):
        GateKVReplayConfig(
            backend=GateKVReplayBackend.STORED,
            deduplicate_static_banks=False,
        )


def test_offload_round_trip_and_static_deduplication():
    source = (_snapshot(0.0, 900.0), _snapshot(1.0, 100.0))
    stored = offload_gate_kv(
        source,
        GateKVReplayConfig(pin_memory=False),
    )

    assert stored.storage_dtype is torch.bfloat16
    assert stored.deduplicated_banks == 2
    assert (
        stored.snapshots[0].layers[0].current_frame_video
        is stored.snapshots[1].layers[0].current_frame_video
    )
    restored = stored.materialize(device="cpu", dtype=torch.float32)
    torch.testing.assert_close(
        restored[1].layers[0].action.key,
        source[1].layers[0].action.key,
    )
    assert restored[0].layers[0].actor_version == 7


def test_recompute_backend_is_explicit_and_callable():
    record = GateKVReplayRecord(
        backend=GateKVReplayBackend.RECOMPUTE,
        recompute_inputs={"seed": 4},
    )
    calls = []

    def recompute(inputs):
        calls.append(inputs["seed"])
        return (_snapshot(0.0, 900.0),)

    materialized = record.materialize(
        device="cpu",
        dtype=torch.float32,
        recompute_fn=recompute,
    )
    assert len(materialized) == 1
    assert calls == [4]


def test_packed_kv_is_batch_first_and_round_trips_through_forward_inputs():
    source = (_snapshot(0.0, 900.0), _snapshot(1.0, 100.0))
    packed = pack_gate_kv(source, GateKVReplayConfig(pin_memory=False))
    fields = packed.as_forward_inputs()
    restored_packed = PackedGateKVTaps.from_forward_inputs(fields)
    restored = restored_packed.materialize(device="cpu", dtype=torch.float32)

    assert packed.video_key.shape[:2] == (1, 1)
    assert packed.action_key.shape[:3] == (1, 2, 1)
    torch.testing.assert_close(
        restored[1].layers[0].action.key,
        source[1].layers[0].action.key,
    )
    assert restored[0].layers[0].current_mode == (PolicyRegime.UNCOND,)


def test_packed_kv_rejects_mixed_actor_versions():
    packed = pack_gate_kv(
        (_snapshot(0.0, 900.0),),
        GateKVReplayConfig(pin_memory=False),
    )
    batch_two = {
        name: (
            tensor
            if name == "layer_indices"
            else tensor.repeat((2,) + (1,) * (tensor.ndim - 1))
        )
        for name, tensor in packed.__dict__.items()
    }
    batch_two["actor_versions"] = torch.tensor([7, 8])

    with pytest.raises(ValueError, match="exactly one actor version"):
        replace(packed, **batch_two)


def test_gate_kv_is_repinned_after_collation(monkeypatch):
    calls = []

    def record_pin(tensor):
        calls.append(tensor)
        return tensor

    monkeypatch.setattr(_module, "_pin_tensor", record_pin)
    inputs = {
        "gate_kv_video_key": torch.zeros(2, 3),
        "gate_kv_action_value": torch.ones(2, 3),
        "flow_chains": torch.zeros(2, 3),
    }

    result = pin_gate_kv_forward_inputs(inputs)

    assert len(calls) == 2
    assert result["flow_chains"] is inputs["flow_chains"]
