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

import asyncio
import inspect
from types import SimpleNamespace

import pytest
import torch

import rlinf.workers.actor.fsdp_actor_worker as actor_worker_module
from rlinf.data.embodied_io_struct import ACTOR_TRAJECTORY_CHANNEL_TAG
from rlinf.scheduler.worker.routing import CommMapper
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    summarize_fastwam_gate_kv_episode_contributions,
)
from rlinf.workers.env.env_worker import EnvWorker


class _Placement:
    def __init__(self, *, env_world_size: int, actor_world_size: int):
        self._world_sizes = {
            "env": int(env_world_size),
            "actor": int(actor_world_size),
        }

    def get_world_size(self, name: str) -> int:
        return self._world_sizes[name]


def test_gate_kv_episode_contributions_do_not_merge_environment_columns():
    contributions = summarize_fastwam_gate_kv_episode_contributions(
        episode_ids=torch.tensor([[0, 0], [0, 0], [1, 0]]),
        gate_valid_mask=torch.tensor(
            [[True, True], [True, False], [False, True]],
        ),
        gate_kv_sample_mask=torch.tensor(
            [[True, True], [False, True], [True, True]],
        ),
    )

    assert contributions == [
        {
            "trajectory_column": 0,
            "initial_episode_id": 0,
            "observed_episode_id_count": 2,
            "emitted_chunk_count": 3,
            "sampled_kv_count": 2,
            "sampled_eligible_gate_count": 1,
        },
        {
            "trajectory_column": 1,
            "initial_episode_id": 0,
            "observed_episode_id_count": 1,
            "emitted_chunk_count": 3,
            "sampled_kv_count": 3,
            "sampled_eligible_gate_count": 2,
        },
    ]


def test_gate_gradient_cosine_uses_all_parameter_tensors() -> None:
    cosine, reference_norm, estimate_norm = EmbodiedFSDPActor._fastwam_gradient_cosine(
        (torch.tensor([1.0, 0.0]), torch.tensor([0.0])),
        (torch.tensor([1.0, 1.0]), torch.tensor([0.0])),
    )

    assert reference_norm == pytest.approx(1.0)
    assert estimate_norm == pytest.approx(2.0**0.5)
    assert cosine == pytest.approx(2.0**-0.5)


class _AsyncValue:
    def __init__(self, value):
        self.value = value
        self.waited = False

    async def async_wait(self):
        self.waited = True
        return self.value


class _Channel:
    def __init__(self):
        self.puts = []
        self.gets = []
        self.values = {}

    def put(self, item, *, key, async_op):
        assert async_op is True
        work = _AsyncValue(None)
        self.puts.append((key, item, work))
        return work

    def get(self, *, key, async_op):
        assert async_op is True
        self.gets.append(key)
        return _AsyncValue(self.values[key])


class _RolloutResult:
    def __init__(self):
        self.split_sizes = None
        self.consumed = False
        self.cleared = False

    def to_splited_trajectories_by_sizes(self, split_sizes, *, consume=False):
        self.split_sizes = list(split_sizes)
        self.consumed = bool(consume)
        return [f"trajectory_{index}" for index in range(len(split_sizes))]

    def clear(self):
        self.cleared = True


def _cfg(total_num_envs: int):
    return SimpleNamespace(
        env=SimpleNamespace(train=SimpleNamespace(total_num_envs=int(total_num_envs)))
    )


def test_env_trajectory_send_uses_logical_stage_rank_and_keyed_fanout() -> None:
    worker = SimpleNamespace(
        _component_placement=_Placement(env_world_size=1, actor_world_size=4),
        _rank=0,
        stage_num=2,
        cfg=_cfg(total_num_envs=8),
    )
    channel = _Channel()
    rollout_result = _RolloutResult()

    asyncio.run(
        inspect.unwrap(EnvWorker.send_rollout_trajectories)(
            worker,
            rollout_result,
            channel,
            stage_id=1,
        )
    )

    assert rollout_result.split_sizes == [2, 2]
    assert rollout_result.consumed is True
    assert rollout_result.cleared is True
    assert [key for key, _, _ in channel.puts] == [
        f"1_2_{ACTOR_TRAJECTORY_CHANNEL_TAG}",
        f"1_3_{ACTOR_TRAJECTORY_CHANNEL_TAG}",
    ]
    assert [item for _, item, _ in channel.puts] == [
        "trajectory_0",
        "trajectory_1",
    ]
    assert all(work.waited for _, _, work in channel.puts)


def test_actor_trajectory_receive_uses_stable_logical_source_order(monkeypatch) -> None:
    worker = SimpleNamespace(
        _component_placement=_Placement(env_world_size=2, actor_world_size=2),
        _rank=0,
        stage_num=2,
        cfg=_cfg(total_num_envs=8),
    )
    worker._process_received_rollout_batch = lambda batch: {
        "processed": batch["received"]
    }
    channel = _Channel()
    expected_keys = [
        f"0_0_{ACTOR_TRAJECTORY_CHANNEL_TAG}",
        f"1_0_{ACTOR_TRAJECTORY_CHANNEL_TAG}",
    ]
    channel.values = {
        expected_keys[0]: "logical_env_0",
        expected_keys[1]: "logical_env_1",
    }
    monkeypatch.setattr(actor_worker_module, "clear_memory", lambda **_: None)
    consume_calls = []

    def convert(trajectories, *, consume=False):
        consume_calls.append(bool(consume))
        return {"received": list(trajectories)}

    monkeypatch.setattr(
        actor_worker_module,
        "convert_trajectories_to_batch",
        convert,
    )

    asyncio.run(
        inspect.unwrap(EmbodiedFSDPActor.recv_rollout_trajectories)(worker, channel)
    )

    assert channel.gets == expected_keys
    assert consume_calls == [True]
    assert worker.rollout_batch == {"processed": ["logical_env_0", "logical_env_1"]}


@pytest.mark.parametrize(
    ("logical_env_world_size", "actor_world_size", "total_num_envs"),
    [(2, 3, 12), (3, 2, 12)],
)
def test_actor_trajectory_send_receive_routes_form_exact_round_trip(
    logical_env_world_size: int,
    actor_world_size: int,
    total_num_envs: int,
) -> None:
    sent_routes = []
    for logical_env_rank in range(logical_env_world_size):
        routes = CommMapper.get_dst_ranks(
            batch_size=total_num_envs,
            src_world_size=logical_env_world_size,
            dst_world_size=actor_world_size,
            src_rank=logical_env_rank,
        )
        assert sum(size for _, size in routes) == (
            total_num_envs // logical_env_world_size
        )
        sent_routes.extend(
            (
                CommMapper.build_channel_key(
                    logical_env_rank,
                    actor_rank,
                    ACTOR_TRAJECTORY_CHANNEL_TAG,
                ),
                size,
            )
            for actor_rank, size in routes
        )

    received_routes = []
    for actor_rank in range(actor_world_size):
        routes = CommMapper.get_src_ranks(
            batch_size=total_num_envs,
            src_world_size=logical_env_world_size,
            dst_world_size=actor_world_size,
            dst_rank=actor_rank,
        )
        assert sum(size for _, size in routes) == total_num_envs // actor_world_size
        received_routes.extend(
            (
                CommMapper.build_channel_key(
                    logical_env_rank,
                    actor_rank,
                    ACTOR_TRAJECTORY_CHANNEL_TAG,
                ),
                size,
            )
            for logical_env_rank, size in routes
        )

    assert received_routes == sorted(
        sent_routes,
        key=lambda route: tuple(int(rank) for rank in route[0].split("_", 2)[:2]),
    )
