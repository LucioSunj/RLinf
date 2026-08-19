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

import asyncio
import sys
from pathlib import Path

import pytest
import torch

FASTWAM_SRC = Path(__file__).resolve().parents[3] / "FastWAM/src"
sys.path.insert(0, str(FASTWAM_SRC))

from rlinf.models.embodiment.wam_policy.kv_replay import (  # noqa: E402
    GateKVReplayConfig,
)
from rlinf.models.embodiment.wam_policy.tiered_kv_store import (  # noqa: E402
    GATE_KV_FORWARD_KEYS,
    GATE_KV_RESPONSE_HANDLES,
    GateKVStoreRequest,
    TieredGateKVStore,
    decode_gate_kv_handle,
    encode_gate_kv_handle,
)
from rlinf.runners.embodied_runner import EmbodiedRunner  # noqa: E402
from rlinf.scheduler.cluster.utils import (  # noqa: E402
    extract_dataclass_tensor_fields,
)
from rlinf.workers.rollout.hf.huggingface_worker import (  # noqa: E402
    MultiStepRolloutWorker,
)


def _packed_inputs(batch_size: int = 3) -> dict[str, torch.Tensor]:
    result = {}
    for index, key in enumerate(GATE_KV_FORWARD_KEYS):
        if key.endswith("_mask"):
            value = torch.ones(batch_size, 2, dtype=torch.bool)
        elif key.endswith("current_modes") or key.endswith("actor_versions"):
            value = torch.full((batch_size,), index, dtype=torch.long)
        else:
            value = (
                torch.arange(
                    batch_size * 4,
                    dtype=torch.float32,
                ).reshape(batch_size, 2, 2)
                + index * 100
            )
        result[key] = value
    result["flow_chains"] = torch.arange(batch_size)
    return result


def test_handle_encoding_round_trips_owner_and_generation():
    handle = encode_gate_kv_handle(source_rank=3, generation=91, local_id=17)
    assert decode_gate_kv_handle(handle) == (3, 91, 17)


def test_stop_request_empty_handles_are_not_classified_as_tensor_field():
    request = GateKVStoreRequest("stop", actor_rank=0, request_id=3)

    tensor_fields, flat_tensors, metadata = extract_dataclass_tensor_fields(request)

    assert tensor_fields == {}
    assert flat_tensors == []
    assert metadata == []


def test_cpu_tier_register_retain_fetch_release_round_trip():
    config = GateKVReplayConfig(
        pin_memory=False,
        hot_capacity_bytes_per_rollout_rank=0,
        cold_capacity_bytes_per_rollout_rank=1024 * 1024,
    )
    store = TieredGateKVStore(source_rank=2, device="cpu", config=config)
    store.begin_generation(7)
    source = _packed_inputs()

    compact, references = store.register_forward_inputs(source)

    assert set(compact) == {"flow_chains"}
    assert torch.equal(compact["flow_chains"], source["flow_chains"])
    assert references.shape == (3,)
    assert all(decode_gate_kv_handle(handle)[:2] == (2, 7) for handle in references)
    keep = tuple(int(value) for value in references[[0, 2]])
    store.retain(keep)
    fetched = store.fetch((keep[1], keep[0]))
    assert fetched[GATE_KV_RESPONSE_HANDLES].tolist() == [keep[1], keep[0]]
    for key in GATE_KV_FORWARD_KEYS:
        torch.testing.assert_close(fetched[key][0], source[key][2])
        torch.testing.assert_close(fetched[key][1], source[key][0])

    store.release(keep)
    assert store.entry_count == 0
    metrics = store.metrics()
    assert metrics["emitted_samples"] == 3
    assert metrics["eligible_samples"] == 2
    assert metrics["discarded_samples"] == 1
    assert metrics["cpu_bytes"] == 0


def test_tier_capacity_fails_without_dropping_samples():
    store = TieredGateKVStore(
        source_rank=0,
        device="cpu",
        config=GateKVReplayConfig(
            hot_capacity_bytes_per_rollout_rank=0,
            cold_capacity_bytes_per_rollout_rank=1,
        ),
    )
    store.begin_generation(0)

    with pytest.raises(MemoryError, match="without dropping a sample"):
        store.register_forward_inputs(_packed_inputs(batch_size=1))


def test_nvme_tier_round_trips_exact_payload_and_removes_shard(tmp_path):
    store = TieredGateKVStore(
        source_rank=1,
        device="cpu",
        config=GateKVReplayConfig(
            pin_memory=False,
            hot_capacity_bytes_per_rollout_rank=0,
            cold_capacity_bytes_per_rollout_rank=0,
            nvme_capacity_bytes_per_rollout_rank=1024 * 1024,
            nvme_path=str(tmp_path),
        ),
    )
    store.begin_generation(4)
    source = {
        key: value.to(torch.bfloat16) if value.is_floating_point() else value
        for key, value in _packed_inputs(batch_size=2).items()
    }

    _, references = store.register_forward_inputs(source)
    handles = tuple(int(value) for value in references)
    assert len(list(tmp_path.glob("*.bin"))) == 1
    store.retain(handles)
    fetched = store.fetch((handles[1], handles[0]))
    for key in GATE_KV_FORWARD_KEYS:
        assert torch.equal(fetched[key][0], source[key][1])
        assert torch.equal(fetched[key][1], source[key][0])

    store.release(handles)
    metrics = store.metrics()
    assert metrics["nvme_bytes"] == 0
    assert metrics["peak_nvme_bytes"] > 0
    assert metrics["nvme_read_bytes"] == metrics["peak_nvme_bytes"]
    assert metrics["nvme_write_bytes"] == metrics["peak_nvme_bytes"]
    assert list(tmp_path.glob("*.bin")) == []


def test_released_generation_cannot_be_fetched_after_store_reuse():
    store = TieredGateKVStore(
        source_rank=0,
        device="cpu",
        config=GateKVReplayConfig(
            pin_memory=False,
            hot_capacity_bytes_per_rollout_rank=0,
            cold_capacity_bytes_per_rollout_rank=1024 * 1024,
        ),
    )
    store.begin_generation(8)
    _, first_references = store.register_forward_inputs(_packed_inputs(batch_size=1))
    stale_handle = int(first_references[0])
    store.retain((stale_handle,))
    store.release((stale_handle,))

    store.begin_generation(9)
    _, second_references = store.register_forward_inputs(_packed_inputs(batch_size=1))
    live_handle = int(second_references[0])

    with pytest.raises(KeyError, match="does not own handles"):
        store.fetch((stale_handle,))
    store.retain((live_handle,))
    store.fetch((live_handle,))
    store.release((live_handle,))


def test_runner_emits_required_per_rank_cache_metric_aliases():
    actor_metrics = [{"kv_cache/prefetch_wait_time": 0.25}]
    service_metrics = [
        {
            "gpu_bytes": 0.0,
            "cpu_bytes": 0.0,
            "nvme_bytes": 0.0,
            "peak_gpu_bytes": 11.0,
            "peak_cpu_bytes": 13.0,
            "peak_nvme_bytes": 0.0,
            "eligible_bytes": 17.0,
            "discarded_bytes": 19.0,
            "fetched_samples": 4.0,
            "gpu_hit_samples": 3.0,
            "transfer_seconds": 0.5,
            "mig_min_free_bytes": 23.0,
            "mig_peak_used_bytes": 29.0,
            "node_min_available_bytes": 31.0,
        },
        {
            "gpu_bytes": 0.0,
            "cpu_bytes": 0.0,
            "nvme_bytes": 0.0,
            "peak_gpu_bytes": 37.0,
            "peak_cpu_bytes": 41.0,
            "peak_nvme_bytes": 43.0,
            "eligible_bytes": 47.0,
            "discarded_bytes": 53.0,
            "fetched_samples": 6.0,
            "gpu_hit_samples": 2.0,
            "transfer_seconds": 0.75,
            "mig_min_free_bytes": 7.0,
            "mig_peak_used_bytes": 59.0,
            "node_min_available_bytes": 5.0,
        },
    ]

    EmbodiedRunner._merge_gate_kv_service_metrics(actor_metrics, service_metrics)

    metrics = actor_metrics[0]
    assert metrics["kv_cache/gpu_bytes_rank_0"] == 11.0
    assert metrics["kv_cache/cpu_bytes_rank_1"] == 41.0
    assert metrics["kv_cache/nvme_bytes_rank_1"] == 43.0
    assert metrics["kv_cache/eligible_bytes"] == 64.0
    assert metrics["kv_cache/discarded_ineligible_bytes"] == 72.0
    assert metrics["kv_cache/hit_fraction"] == 0.5
    assert metrics["kv_cache/transfer_time"] == 0.75
    assert metrics["kv_cache/mig_min_free_bytes"] == 7.0
    assert metrics["kv_cache/mig_peak_used_bytes"] == 59.0
    assert metrics["kv_cache/node_physical_min_available_bytes"] == 5.0


class _ImmediateAsyncValue:
    def __init__(self, value):
        self.value = value

    async def async_wait(self):
        return self.value


class _RequestChannel:
    def __init__(self, requests):
        self.requests = list(requests)

    def get(self, *, key, async_op):
        assert key == "gate_kv_request_rollout_0"
        assert async_op
        return _ImmediateAsyncValue(self.requests.pop(0))


class _ResponseChannel:
    def __init__(self):
        self.items = []
        self.transports = []

    def put(self, item, *, key, async_op):
        assert key.startswith("gate_kv_response_actor_0_")
        assert not async_op
        self.items.append(item)
        self.transports.append("worker")

    def put_via_ray(self, item, *, key, async_op):
        assert key.startswith("gate_kv_response_actor_0_")
        assert not async_op
        self.items.append(item)
        self.transports.append("ray")


def test_rollout_service_obeys_retain_fetch_release_lifecycle():
    store = TieredGateKVStore(
        source_rank=0,
        device="cpu",
        config=GateKVReplayConfig(
            hot_capacity_bytes_per_rollout_rank=0,
            cold_capacity_bytes_per_rollout_rank=1024 * 1024,
        ),
    )
    store.begin_generation(0)
    _, references = store.register_forward_inputs(_packed_inputs(batch_size=2))
    handles = tuple(int(value) for value in references)
    requests = [
        GateKVStoreRequest("retain", 0, 0, handles),
        GateKVStoreRequest("fetch", 0, 1, (handles[1],)),
        GateKVStoreRequest("release", 0, 2, handles),
        GateKVStoreRequest("stop", 0, 3),
    ]
    request_channel = _RequestChannel(requests)
    response_channel = _ResponseChannel()
    worker = object.__new__(MultiStepRolloutWorker)
    worker._rank = 0
    worker._fastwam_kv_store = store
    service = MultiStepRolloutWorker.serve_gate_kv_requests
    while hasattr(service, "__wrapped__"):
        service = service.__wrapped__

    metrics = asyncio.run(service(worker, request_channel, response_channel))

    assert response_channel.items[0] == {"retained": 2}
    assert response_channel.items[1][GATE_KV_RESPONSE_HANDLES].tolist() == [handles[1]]
    assert len(response_channel.items) == 2
    assert response_channel.transports == ["ray", "worker"]
    assert metrics["eligible_samples"] == 2
    assert store.entry_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_hot_tier_fetches_through_pinned_host_staging():
    store = TieredGateKVStore(
        source_rank=1,
        device="cuda",
        config=GateKVReplayConfig(
            hot_capacity_bytes_per_rollout_rank=1024 * 1024,
            cold_capacity_bytes_per_rollout_rank=1024 * 1024,
            hot_min_free_bytes=0,
        ),
    )
    store.begin_generation(2)
    source = {
        key: value.cuda() if torch.is_tensor(value) else value
        for key, value in _packed_inputs(batch_size=2).items()
    }
    _, references = store.register_forward_inputs(source)
    handles = tuple(int(value) for value in references)

    before_fetch = store.metrics()
    assert before_fetch["gpu_bytes"] > 0
    assert before_fetch["cpu_bytes"] == 0
    store.retain(handles)
    fetched = store.fetch(handles)
    for key in GATE_KV_FORWARD_KEYS:
        torch.testing.assert_close(fetched[key], source[key].cpu())
    assert store.metrics()["gpu_hit_fraction"] == 1.0
    store.release(handles)
