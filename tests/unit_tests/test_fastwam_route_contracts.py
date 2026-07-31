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

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_contracts_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "rlinf/models/embodiment/wam_policy/contracts.py"
    name = "rlinf.models.embodiment.wam_policy.contracts"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contracts = _load_contracts_module()
ChunkRouteRecord = contracts.ChunkRouteRecord
GateDecisionRecord = contracts.GateDecisionRecord
GateKVMetadata = contracts.GateKVMetadata
WAMRoute = contracts.WAMRoute
shift_emitted_gate_decisions = contracts.shift_emitted_gate_decisions


def _route_record(
    *,
    route_used,
    forced,
    chunk_ids,
    episode_ids,
    source_ids,
    actor_versions=None,
):
    route_tensor = torch.as_tensor(route_used, dtype=torch.int64).clone()
    return ChunkRouteRecord(
        route_used=route_tensor,
        route_was_forced=torch.as_tensor(forced, dtype=torch.bool).clone(),
        chunk_ids=torch.as_tensor(chunk_ids, dtype=torch.int64).clone(),
        episode_ids=torch.as_tensor(episode_ids, dtype=torch.int64).clone(),
        route_source_chunk_ids=torch.as_tensor(source_ids, dtype=torch.int64).clone(),
        actor_versions=(
            torch.full_like(route_tensor, 3)
            if actor_versions is None
            else torch.as_tensor(actor_versions, dtype=torch.int64).clone()
        ),
    )


def _gate_record(route: ChunkRouteRecord, *, actor_versions=None):
    next_route = torch.tensor(
        [
            [WAMRoute.UNCOND, WAMRoute.IDM],
            [WAMRoute.IDM, WAMRoute.UNCOND],
            [WAMRoute.UNCOND, WAMRoute.UNCOND],
            [WAMRoute.IDM, WAMRoute.IDM],
        ],
        dtype=torch.int64,
    )
    shape = next_route.shape
    behavior_probability = torch.full(shape, 0.6, dtype=torch.float32)
    old_logprob = torch.where(
        next_route == int(WAMRoute.IDM),
        behavior_probability.log(),
        torch.log1p(-behavior_probability),
    )
    total_bytes = torch.arange(100, 108, dtype=torch.int64).reshape(shape)
    return GateDecisionRecord(
        next_route=next_route,
        base_probability=torch.full(shape, 0.625, dtype=torch.float32),
        behavior_probability=behavior_probability,
        old_logprob=old_logprob,
        epsilon=torch.full(shape, 0.2, dtype=torch.float32),
        temperature=torch.ones(shape, dtype=torch.float32),
        valid=torch.ones(shape, dtype=torch.bool),
        source_chunk_ids=route.chunk_ids.clone(),
        episode_ids=route.episode_ids.clone(),
        actor_versions=(
            route.actor_versions.clone()
            if actor_versions is None
            else torch.as_tensor(actor_versions, dtype=torch.int64).clone()
        ),
        kv_metadata=GateKVMetadata(
            layer_indices=(0, 3),
            denoise_timesteps=torch.arange(16, dtype=torch.float32).reshape(4, 2, 2),
            total_bytes=total_bytes,
            storage_dtype="bfloat16",
            tensor_shapes=((2, 8, 16), (2, 8, 16)),
            payload_reference_ids=torch.arange(8, dtype=torch.int64).reshape(shape),
        ),
    )


def _asynchronous_reset_fixture():
    route = _route_record(
        route_used=[
            [WAMRoute.IDM, WAMRoute.IDM],
            [WAMRoute.UNCOND, WAMRoute.IDM],
            [WAMRoute.IDM, WAMRoute.IDM],
            [WAMRoute.UNCOND, WAMRoute.UNCOND],
        ],
        forced=[
            [True, True],
            [False, False],
            [False, True],
            [False, False],
        ],
        chunk_ids=[[0, 0], [1, 1], [2, 0], [3, 1]],
        episode_ids=[[10, 20], [10, 20], [10, 21], [10, 21]],
        source_ids=[[-1, -1], [0, 0], [1, -1], [2, 0]],
    )
    emitted = _gate_record(route)
    dones = torch.tensor(
        [[False, False], [False, True], [False, False], [True, True]],
        dtype=torch.bool,
    )
    reset_mask = torch.tensor(
        [[True, True], [False, False], [False, True], [False, False]],
        dtype=torch.bool,
    )
    return route, emitted, dones, reset_mask


def test_route_enum_has_checkpoint_stable_integer_values():
    assert int(WAMRoute.UNCOND) == 0
    assert int(WAMRoute.IDM) == 1
    assert contracts.WAMMode is WAMRoute


def test_shift_gate_decisions_handles_asynchronous_vector_resets():
    route, emitted, dones, reset_mask = _asynchronous_reset_fixture()
    aligned = shift_emitted_gate_decisions(
        route=route,
        emitted=emitted,
        dones=dones,
        reset_mask=reset_mask,
    )

    expected_valid = torch.tensor(
        [[False, False], [True, True], [True, False], [True, True]],
        dtype=torch.bool,
    )
    assert torch.equal(aligned.valid, expected_valid)
    assert torch.equal(
        aligned.source_time_indices,
        torch.tensor([[-1, -1], [0, 0], [1, -1], [2, 2]]),
    )
    assert torch.equal(
        aligned.decisions.source_chunk_ids,
        torch.tensor([[-1, -1], [0, 0], [1, -1], [2, 0]]),
    )
    assert torch.equal(
        aligned.decisions.next_route[expected_valid],
        route.route_used[expected_valid],
    )
    assert torch.equal(
        aligned.decisions.old_logprob[1, 0],
        emitted.old_logprob[0, 0],
    )
    assert aligned.decisions.old_logprob[2, 1].item() == 0.0
    assert aligned.decisions.kv_metadata.total_bytes[3, 1].item() == 105
    assert aligned.decisions.kv_metadata.total_bytes[2, 1].item() == 0
    assert aligned.decisions.kv_metadata.payload_reference_ids[3, 1].item() == 5


def test_shift_rejects_cross_episode_route_without_reset_marker():
    route, emitted, dones, reset_mask = _asynchronous_reset_fixture()
    reset_mask[2, 1] = False

    with pytest.raises(ValueError, match="episode boundary"):
        shift_emitted_gate_decisions(
            route=route,
            emitted=emitted,
            dones=dones,
            reset_mask=reset_mask,
        )


def test_shift_rejects_actor_version_mismatch():
    route, _, dones, reset_mask = _asynchronous_reset_fixture()
    emitted = _gate_record(
        route,
        actor_versions=[[3, 3], [3, 3], [3, 3], [4, 3]],
    )

    with pytest.raises(ValueError, match="actor version"):
        shift_emitted_gate_decisions(
            route=route,
            emitted=emitted,
            dones=dones,
            reset_mask=reset_mask,
        )


def test_shift_rejects_off_by_one_route_source_id():
    route, emitted, dones, reset_mask = _asynchronous_reset_fixture()
    bad_route = _route_record(
        route_used=route.route_used,
        forced=route.route_was_forced,
        chunk_ids=route.chunk_ids,
        episode_ids=route.episode_ids,
        source_ids=[[-1, -1], [0, 0], [1, -1], [1, 0]],
    )

    with pytest.raises(ValueError, match="route_source_chunk_ids"):
        shift_emitted_gate_decisions(
            route=bad_route,
            emitted=emitted,
            dones=dones,
            reset_mask=reset_mask,
        )


def test_reset_destination_must_be_forced_idm():
    route, emitted, dones, reset_mask = _asynchronous_reset_fixture()
    bad_route = _route_record(
        route_used=route.route_used,
        forced=[
            [True, True],
            [False, False],
            [False, False],
            [False, False],
        ],
        chunk_ids=route.chunk_ids,
        episode_ids=route.episode_ids,
        source_ids=[[-1, -1], [0, 0], [1, 1], [2, 0]],
    )

    with pytest.raises(ValueError, match="first chunk after reset"):
        shift_emitted_gate_decisions(
            route=bad_route,
            emitted=emitted,
            dones=dones,
            reset_mask=reset_mask,
        )


def test_gate_record_rejects_invalid_probability():
    route, emitted, _, _ = _asynchronous_reset_fixture()
    base_probability = emitted.base_probability.clone()
    base_probability[0, 0] = 1.1

    with pytest.raises(ValueError, match="base_probability"):
        GateDecisionRecord(
            next_route=emitted.next_route,
            base_probability=base_probability,
            behavior_probability=emitted.behavior_probability,
            old_logprob=emitted.old_logprob,
            epsilon=emitted.epsilon,
            temperature=emitted.temperature,
            valid=emitted.valid,
            source_chunk_ids=route.chunk_ids,
            episode_ids=route.episode_ids,
            actor_versions=route.actor_versions,
        )
