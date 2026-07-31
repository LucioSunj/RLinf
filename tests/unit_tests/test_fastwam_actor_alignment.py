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
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contracts = _load_module(
    "rlinf.models.embodiment.wam_policy.contracts",
    "rlinf/models/embodiment/wam_policy/contracts.py",
)


def _load_advantages_module():
    stubs = {
        "rlinf.algorithms.registry": types.SimpleNamespace(
            register_advantage=lambda _name: lambda function: function
        ),
        "rlinf.algorithms.utils": types.SimpleNamespace(
            kl_penalty=lambda *_args, **_kwargs: None,
            safe_normalize=lambda value, loss_mask=None: value,
        ),
        "rlinf.utils.utils": types.SimpleNamespace(
            masked_mean=lambda value, mask=None: value.mean()
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        return _load_module(
            "fastwam_advantages_under_test",
            "rlinf/algorithms/advantages.py",
        )
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


advantages = _load_advantages_module()
nested = _load_module(
    "fastwam_nested_dict_process_under_test",
    "rlinf/utils/nested_dict_process.py",
)
ChunkRouteRecord = contracts.ChunkRouteRecord
GateDecisionRecord = contracts.GateDecisionRecord
GateKVMetadata = contracts.GateKVMetadata
WAMRoute = contracts.WAMRoute


def _alignment_records():
    route_used = torch.tensor(
        [
            [WAMRoute.IDM, WAMRoute.IDM, WAMRoute.UNCOND, WAMRoute.IDM],
            [WAMRoute.UNCOND, WAMRoute.UNCOND, WAMRoute.UNCOND, WAMRoute.UNCOND],
            [WAMRoute.UNCOND, WAMRoute.IDM, WAMRoute.UNCOND, WAMRoute.UNCOND],
        ],
        dtype=torch.long,
    )
    forced = torch.tensor(
        [
            [True, True, False, True],
            [False, False, False, False],
            [False, True, False, False],
        ]
    )
    chunk_ids = torch.tensor(
        [
            [0, 0, 3, 0],
            [1, 1, 4, 1],
            [2, 0, 5, 2],
        ]
    )
    episode_ids = torch.tensor(
        [
            [10, 20, 10, 22],
            [10, 20, 10, 22],
            [10, 21, 10, 22],
        ]
    )
    source_ids = torch.tensor(
        [
            [-1, -1, 2, -1],
            [0, 0, 3, 0],
            [1, -1, 4, 1],
        ]
    )
    route = ChunkRouteRecord(
        route_used=route_used,
        route_was_forced=forced,
        chunk_ids=chunk_ids,
        episode_ids=episode_ids,
        route_source_chunk_ids=source_ids,
        actor_versions=torch.full_like(route_used, 7),
    )
    shape = route.shape
    behavior_probability = torch.full(shape, 0.4)
    emitted = GateDecisionRecord(
        next_route=torch.full(shape, int(WAMRoute.UNCOND), dtype=torch.long),
        base_probability=torch.full(shape, 0.375),
        behavior_probability=behavior_probability,
        old_logprob=torch.log1p(-behavior_probability),
        epsilon=torch.full(shape, 0.1),
        temperature=torch.ones(shape),
        valid=torch.ones(shape, dtype=torch.bool),
        source_chunk_ids=chunk_ids.clone(),
        episode_ids=episode_ids.clone(),
        actor_versions=torch.full_like(route_used, 7),
        kv_metadata=GateKVMetadata(
            layer_indices=(0, 2),
            denoise_timesteps=torch.ones(*shape, 1),
            total_bytes=torch.full(shape, 64, dtype=torch.long),
        ),
    )
    dones = torch.zeros(4, 4, 1, dtype=torch.bool)
    dones[2, 1] = True
    dones[3, 1] = True
    return route, emitted, dones


def test_gate_advantage_stays_at_source_and_crosses_rollout_epoch():
    route, emitted, dones = _alignment_records()
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)

    result = advantages.align_fastwam_policy_advantages(
        advantages=values,
        route=route,
        emitted=emitted,
        dones=dones,
        rollout_epoch=2,
        carry_pending_across_epochs=True,
    )

    assert result.gate_valid_mask.sum().item() == 8
    assert result.gate_advantages[0, 0].item() == values[1, 0, 0].item()
    assert result.gate_advantages[2, 0].item() == values[0, 2, 0].item()
    assert not result.gate_valid_mask[1, 1]
    assert not result.gate_valid_mask[2, 1]
    assert emitted.kv_metadata.total_bytes[2, 0].item() == 64


def test_non_auto_reset_does_not_pair_across_rollout_epochs():
    route, emitted, dones = _alignment_records()
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)

    result = advantages.align_fastwam_policy_advantages(
        advantages=values,
        route=route,
        emitted=emitted,
        dones=dones,
        rollout_epoch=2,
        carry_pending_across_epochs=False,
    )

    assert result.gate_valid_mask.sum().item() == 7
    assert not result.gate_valid_mask[2, 0]


def test_actor_version_mismatch_fails_closed():
    route, emitted, dones = _alignment_records()
    actor_versions = route.actor_versions.clone()
    actor_versions[0, 2] = 8
    route = replace(route, actor_versions=actor_versions)
    emitted_versions = emitted.actor_versions.clone()
    emitted_versions[0, 2] = 8
    emitted = replace(emitted, actor_versions=emitted_versions)

    with pytest.raises(ValueError, match="actor-version boundary"):
        advantages.align_fastwam_policy_advantages(
            advantages=torch.ones(3, 4, 1),
            route=route,
            emitted=emitted,
            dones=dones,
            rollout_epoch=2,
            carry_pending_across_epochs=True,
        )


def test_fixed_route_cost_is_subtracted_once_after_reward_aggregation():
    rewards = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        ]
    )
    routes = torch.tensor(
        [[WAMRoute.IDM, WAMRoute.UNCOND], [WAMRoute.UNCOND, WAMRoute.IDM]]
    )

    result = advantages.apply_fastwam_chunk_cost(
        environment_rewards=rewards,
        route_used=routes,
        idm_cost=2.0,
        uncond_cost=0.25,
    )

    assert result.rewards.shape == (2, 2, 1)
    assert torch.equal(
        result.costs[..., 0], torch.tensor([[2.0, 0.25], [0.25, 2.0]])
    )
    assert torch.equal(
        result.rewards[..., 0], torch.tensor([[4.0, 14.75], [1.25, 1.0]])
    )


def test_route_and_gate_dataclasses_survive_mapping_device_and_chunking():
    route, emitted, _ = _alignment_records()
    batch = {"route_info": route, "emitted_gate": emitted}

    mapped = nested.map_nested_tensors(batch, lambda tensor: tensor.clone())
    moved = nested.put_tensor_device(mapped, "cpu")
    chunks = nested.split_dict_to_chunk(moved, 2, dim=1)

    assert isinstance(chunks[0]["route_info"], ChunkRouteRecord)
    assert isinstance(chunks[0]["emitted_gate"], GateDecisionRecord)
    assert chunks[0]["route_info"].shape == (3, 2)
    assert chunks[1]["emitted_gate"].kv_metadata.batch_shape == (3, 2)
    assert chunks[0]["emitted_gate"].kv_metadata.layer_indices == (0, 2)


def test_route_records_survive_epoch_fold_and_train_flatten():
    route, emitted, _ = _alignment_records()

    def unfold_epoch_batch(tensor):
        return tensor.reshape(3, 2, 2, *tensor.shape[2:]).transpose(0, 1).reshape(
            6, 2, *tensor.shape[2:]
        )

    raw = nested.map_nested_tensors(
        {"route_info": route, "emitted_gate": emitted}, unfold_epoch_batch
    )
    merged = nested.merge_rollout_epoch_batch(raw, rollout_epoch=2)
    shuffle = torch.arange(11, -1, -1)
    flattened = nested.flatten_time_batch(merged, shuffle, field_name="routes")

    assert isinstance(flattened["route_info"], ChunkRouteRecord)
    assert flattened["route_info"].shape == (12,)
    assert flattened["emitted_gate"].shape == (12,)
    assert flattened["emitted_gate"].kv_metadata.batch_shape == (12,)
    assert torch.equal(
        flattened["route_info"].chunk_ids,
        route.chunk_ids.reshape(-1)[shuffle],
    )
