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

import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    RolloutResult,
    convert_trajectories_to_batch,
)


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
WAMRoute = contracts.WAMRoute


def _step_records(chunk_id: int, batch_size: int = 2):
    route = ChunkRouteRecord(
        route_used=torch.full((batch_size,), WAMRoute.IDM, dtype=torch.int64),
        route_was_forced=torch.full((batch_size,), chunk_id == 0, dtype=torch.bool),
        chunk_ids=torch.full((batch_size,), chunk_id, dtype=torch.int64),
        episode_ids=torch.zeros(batch_size, dtype=torch.int64),
        route_source_chunk_ids=torch.full(
            (batch_size,), -1 if chunk_id == 0 else chunk_id - 1, dtype=torch.int64
        ),
        actor_versions=torch.zeros(batch_size, dtype=torch.int64),
    )
    probabilities = torch.full((batch_size,), 0.75, dtype=torch.float32)
    emitted = GateDecisionRecord(
        next_route=torch.full((batch_size,), WAMRoute.IDM, dtype=torch.int64),
        base_probability=probabilities,
        behavior_probability=probabilities,
        old_logprob=probabilities.log(),
        epsilon=torch.zeros(batch_size, dtype=torch.float32),
        temperature=torch.ones(batch_size, dtype=torch.float32),
        valid=torch.ones(batch_size, dtype=torch.bool),
        source_chunk_ids=route.chunk_ids,
        episode_ids=route.episode_ids,
        actor_versions=route.actor_versions,
    )
    return route, emitted


def test_existing_rollout_result_constructor_remains_backward_compatible():
    result = RolloutResult(actions=torch.ones(2, 3))
    assert result.route_info is None
    assert result.emitted_gate is None


def test_rollout_merge_concatenates_route_and_gate_records():
    first_route, first_gate = _step_records(0, batch_size=1)
    second_route, second_gate = _step_records(0, batch_size=1)
    merged = RolloutResult.merge_rollout_results(
        [
            RolloutResult(
                actions=torch.ones(1, 2),
                route_info=first_route,
                emitted_gate=first_gate,
            ),
            RolloutResult(
                actions=torch.zeros(1, 2),
                route_info=second_route,
                emitted_gate=second_gate,
            ),
        ]
    )

    assert merged.actions.shape == (2, 2)
    assert merged.route_info.shape == torch.Size([2])
    assert merged.emitted_gate.shape == torch.Size([2])


def test_embodied_rollout_stacks_splits_and_batches_route_records():
    rollout = EmbodiedRolloutResult(max_episode_length=2)
    for chunk_id in range(2):
        route, emitted = _step_records(chunk_id)
        rollout.append_step_result(
            ChunkStepResult(
                actions=torch.full((2, 2), float(chunk_id)),
                rewards=torch.ones(2),
                dones=torch.zeros(2, dtype=torch.bool),
                route_info=route,
                emitted_gate=emitted,
            )
        )

    trajectory = rollout.to_trajectory()
    assert trajectory.route_info.shape == torch.Size([2, 2])
    assert trajectory.emitted_gate.shape == torch.Size([2, 2])

    split = rollout.to_splited_trajectories(2)
    assert [item.route_info.shape for item in split] == [
        torch.Size([2, 1]),
        torch.Size([2, 1]),
    ]
    batch = convert_trajectories_to_batch(split)
    assert batch["route_info"].shape == torch.Size([2, 2])
    assert batch["emitted_gate"].shape == torch.Size([2, 2])


def test_consuming_handoff_preserves_values_and_releases_sources():
    rollout = EmbodiedRolloutResult(max_episode_length=2)
    expected = []
    for chunk_id in range(2):
        route, emitted = _step_records(chunk_id)
        replay = torch.full((2, 3), float(chunk_id + 1))
        expected.append(replay)
        rollout.append_step_result(
            ChunkStepResult(
                actions=torch.full((2, 2), float(chunk_id)),
                rewards=torch.ones(2),
                dones=torch.zeros(2, dtype=torch.bool),
                forward_inputs={"gate_kv_action_key": replay},
                route_info=route,
                emitted_gate=emitted,
            )
        )

    trajectories = rollout.to_splited_trajectories_by_sizes([1, 1], consume=True)
    assert rollout.forward_inputs == []
    assert rollout.actions == []
    assert rollout.route_info == []
    assert torch.equal(
        torch.cat(
            [item.forward_inputs["gate_kv_action_key"] for item in trajectories],
            dim=1,
        ),
        torch.stack(expected, dim=0),
    )

    batch = convert_trajectories_to_batch(trajectories, consume=True)
    assert trajectories == []
    assert torch.equal(
        batch["forward_inputs"]["gate_kv_action_key"],
        torch.stack(expected, dim=0),
    )
    assert batch["route_info"].shape == torch.Size([2, 2])
    assert batch["emitted_gate"].shape == torch.Size([2, 2])
