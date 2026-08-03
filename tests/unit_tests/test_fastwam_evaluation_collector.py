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
import json
from pathlib import Path

import torch

from rlinf.data.embodied_io_struct import EnvOutput, RolloutResult
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
)
from rlinf.models.embodiment.wam_policy.evaluation import EvaluationRouteSelection
from rlinf.runners.fastwam_libero_eval_collector import (
    FastWAMLiberoEvalCollector,
)


def _canonical_sha(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ledger(path: Path) -> dict:
    identity_fields = {
        "task_suite": "libero_10",
        "task_id": 0,
        "trial_id": 0,
        "reset_state_id": 0,
        "environment_seed": 7,
        "action_noise_seed": 101,
        "idm_video_noise_seed": 202,
        "max_primitive_steps": 32,
        "action_horizon": 16,
    }
    payload = {
        "schema": "fastwam-libero-eval-ledger-v1",
        "kind": "preflight",
        "task_suite": "libero_10",
        "entries": [
            {
                "episode_index": 0,
                **identity_fields,
                "episode_identity": _canonical_sha(identity_fields),
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


class _IdentityEnv:
    task_ids = [0]
    trial_ids = [0]
    reset_state_ids = [0]


def _rollout(
    *,
    chunk_id: int,
    route_used: int,
    forced: bool,
    source_chunk_id: int,
    terminal: bool,
) -> RolloutResult:
    route = ChunkRouteRecord(
        route_used=torch.tensor([route_used]),
        route_was_forced=torch.tensor([forced]),
        chunk_ids=torch.tensor([chunk_id]),
        episode_ids=torch.tensor([0]),
        route_source_chunk_ids=torch.tensor([source_chunk_id]),
        actor_versions=torch.tensor([3]),
    )
    probability = torch.tensor([0.25])
    emitted = GateDecisionRecord(
        next_route=torch.tensor([0]),
        base_probability=probability,
        behavior_probability=probability,
        old_logprob=torch.log1p(-probability),
        epsilon=torch.tensor([0.0]),
        temperature=torch.tensor([1.0]),
        valid=torch.tensor([not terminal]),
        source_chunk_ids=torch.tensor([chunk_id]),
        episode_ids=torch.tensor([0]),
        actor_versions=torch.tensor([3]),
        kv_metadata=None,
    )
    selection = EvaluationRouteSelection(
        mode="forced_uncond",
        effective_next_route=torch.tensor([0]),
        counterfactual_next_route=torch.tensor([0]),
    )
    return RolloutResult(
        actions=torch.tensor([[[0.1] * 7, [1.1] * 7]]),
        route_info=route,
        emitted_gate=emitted,
        evaluation_selection=selection,
        gate_latency_seconds=torch.tensor([0.004], dtype=torch.float64),
        gate_h2d_seconds=torch.tensor([0.0], dtype=torch.float64),
    )


def _outcome(*, terminal: bool, success: bool = False) -> EnvOutput:
    terminations = torch.tensor([[False, success]])
    truncations = torch.tensor([[False, terminal and not success]])
    return EnvOutput(
        obs={"states": torch.zeros(1, 3)},
        dones=torch.tensor([terminal]),
        terminations=terminations,
        truncations=truncations,
        rewards=torch.tensor([[0.0, float(success)]]),
    )


def _collector(tmp_path: Path) -> FastWAMLiberoEvalCollector:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger_path = tmp_path / "ledger.json"
    if not ledger_path.exists():
        _ledger(ledger_path)
    return FastWAMLiberoEvalCollector(
        output_dir=str(tmp_path),
        ledger_path=str(ledger_path),
        run_id="collector-unit",
        rank=0,
        routing_mode="forced_uncond",
        idm_threshold=0.5,
        random_idm_probability=None,
        routing_seed=0,
        fixed_idm_cost=0.01,
    )


def test_collector_records_aligned_chunks_episode_and_atomic_shards(tmp_path) -> None:
    collector = _collector(tmp_path)
    env = _IdentityEnv()
    env_ids = torch.tensor([1 << 50])

    first_snapshot = collector.snapshot_before_step(0, env, env_ids)
    first_input = collector.augment_rollout_input(
        {"obs": {"states": torch.zeros(1, 3)}},
        first_snapshot,
    )
    collector.record_chunk(
        snapshot=first_snapshot,
        rollout_result=_rollout(
            chunk_id=0,
            route_used=1,
            forced=True,
            source_chunk_id=-1,
            terminal=False,
        ),
        env_output=_outcome(terminal=False),
        environment_latency_seconds=0.02,
    )

    second_snapshot = collector.snapshot_before_step(0, env, env_ids)
    second_input = collector.augment_rollout_input(
        {"obs": {"states": torch.zeros(1, 3)}},
        second_snapshot,
    )
    collector.record_chunk(
        snapshot=second_snapshot,
        rollout_result=_rollout(
            chunk_id=1,
            route_used=0,
            forced=False,
            source_chunk_id=0,
            terminal=True,
        ),
        env_output=_outcome(terminal=True),
        environment_latency_seconds=0.03,
    )
    shard = collector.finalize()

    assert first_input["obs"]["_fastwam_action_noise_seeds"].shape == (1,)
    assert first_input["obs"]["_fastwam_idm_noise_seeds"].shape == (1,)
    assert not torch.equal(
        first_input["obs"]["_fastwam_action_noise_seeds"],
        second_input["obs"]["_fastwam_action_noise_seeds"],
    )
    assert shard.chunk_record_count == 2
    assert shard.episode_record_count == 1
    assert not (tmp_path / "chunks.rank-0.jsonl.tmp").exists()
    assert not (tmp_path / "episodes.rank-0.jsonl.tmp").exists()

    chunks = [
        json.loads(line)
        for line in (tmp_path / "chunks.rank-0.jsonl").read_text().splitlines()
    ]
    episodes = [
        json.loads(line)
        for line in (tmp_path / "episodes.rank-0.jsonl").read_text().splitlines()
    ]
    assert chunks[0]["route"] == "idm"
    assert chunks[0]["route_was_forced"] is True
    assert chunks[0]["emitted_decision_consumed"] is True
    assert chunks[1]["route"] == "uncond"
    assert chunks[1]["emitted_decision_discarded"] is True
    assert chunks[1]["terminal"] is True
    assert episodes[0]["executed_chunk_count"] == 2
    assert episodes[0]["forced_initial_idm_count"] == 1
    assert episodes[0]["eligible_chunk_count"] == 1
    assert episodes[0]["eligible_idm_count"] == 0
    assert episodes[0]["idm_chunk_count_total"] == 1
    assert episodes[0]["normalized_prediction_compute"] == 0.5
    assert episodes[0]["fixed_prediction_cost"] == 0.01
    assert [chunk["gate_latency_seconds"] for chunk in chunks] == [0.004, 0.004]
    assert [chunk["gate_h2d_seconds"] for chunk in chunks] == [0.0, 0.0]

    serialized = json.dumps({"chunks": chunks, "episodes": episodes}).lower()
    for forbidden in ("gate_kv", "observation", "main_images", "model_weights"):
        assert forbidden not in serialized


def test_collector_seed_augmentation_is_reproducible(tmp_path) -> None:
    first = _collector(tmp_path / "first")
    second = _collector(tmp_path / "second")
    env_ids = torch.tensor([1 << 50])
    first_snapshot = first.snapshot_before_step(0, _IdentityEnv(), env_ids)
    second_snapshot = second.snapshot_before_step(0, _IdentityEnv(), env_ids)

    first_data = first.augment_rollout_input(
        {"obs": {"states": torch.zeros(1, 3)}},
        first_snapshot,
    )
    second_data = second.augment_rollout_input(
        {"obs": {"states": torch.zeros(1, 3)}},
        second_snapshot,
    )

    assert torch.equal(
        first_data["obs"]["_fastwam_action_noise_seeds"],
        second_data["obs"]["_fastwam_action_noise_seeds"],
    )
    assert torch.equal(
        first_data["obs"]["_fastwam_idm_noise_seeds"],
        second_data["obs"]["_fastwam_idm_noise_seeds"],
    )


def test_collector_fails_closed_on_actual_ledger_identity_mismatch(tmp_path) -> None:
    collector = _collector(tmp_path)
    env = _IdentityEnv()
    env.task_ids = [1]

    try:
        collector.snapshot_before_step(0, env, torch.tensor([1 << 50]))
    except ValueError as exc:
        assert "ledger identity mismatch" in str(exc)
    else:
        raise AssertionError("collector accepted a mismatched task/reset identity")


def test_collector_writes_parallel_episodes_in_frozen_ledger_order(tmp_path) -> None:
    entries = []
    for episode_index in range(2):
        identity_fields = {
            "task_suite": "libero_10",
            "task_id": 0,
            "trial_id": episode_index,
            "reset_state_id": episode_index,
            "environment_seed": 7,
            "action_noise_seed": 101 + episode_index,
            "idm_video_noise_seed": 202 + episode_index,
            "max_primitive_steps": 512,
            "action_horizon": 16,
        }
        entries.append(
            {
                "episode_index": episode_index,
                **identity_fields,
                "episode_identity": _canonical_sha(identity_fields),
            }
        )
    identities = [entry["episode_identity"] for entry in entries]
    assert identities != sorted(identities)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema": "fastwam-libero-eval-ledger-v1",
                "kind": "preflight",
                "task_suite": "libero_10",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    collector = FastWAMLiberoEvalCollector(
        output_dir=str(tmp_path),
        ledger_path=str(ledger_path),
        run_id="parallel-order-unit",
        rank=0,
        routing_mode="forced_uncond",
        idm_threshold=0.5,
        random_idm_probability=None,
        routing_seed=0,
        fixed_idm_cost=0.01,
    )
    env = _IdentityEnv()
    env.task_ids = [0, 0]
    env.trial_ids = [0, 1]
    env.reset_state_ids = [0, 1]
    snapshot = collector.snapshot_before_step(
        0,
        env,
        torch.tensor([1 << 50, (1 << 50) + 1]),
    )
    probability = torch.tensor([0.25, 0.25])
    collector.record_chunk(
        snapshot=snapshot,
        rollout_result=RolloutResult(
            actions=torch.zeros(2, 1, 7),
            route_info=ChunkRouteRecord(
                route_used=torch.tensor([1, 1]),
                route_was_forced=torch.tensor([True, True]),
                chunk_ids=torch.tensor([0, 0]),
                episode_ids=torch.tensor([0, 0]),
                route_source_chunk_ids=torch.tensor([-1, -1]),
                actor_versions=torch.tensor([3, 3]),
            ),
            emitted_gate=GateDecisionRecord(
                next_route=torch.tensor([0, 0]),
                base_probability=probability,
                behavior_probability=probability,
                old_logprob=torch.log1p(-probability),
                epsilon=torch.tensor([0.0, 0.0]),
                temperature=torch.tensor([1.0, 1.0]),
                valid=torch.tensor([False, False]),
                source_chunk_ids=torch.tensor([0, 0]),
                episode_ids=torch.tensor([0, 0]),
                actor_versions=torch.tensor([3, 3]),
                kv_metadata=None,
            ),
            evaluation_selection=EvaluationRouteSelection(
                mode="forced_uncond",
                effective_next_route=torch.tensor([0, 0]),
                counterfactual_next_route=torch.tensor([0, 0]),
            ),
        ),
        env_output=EnvOutput(
            obs={"states": torch.zeros(2, 3)},
            dones=torch.tensor([True, True]),
            terminations=torch.tensor([[False], [False]]),
            truncations=torch.tensor([[True], [True]]),
            rewards=torch.zeros(2, 1),
        ),
        environment_latency_seconds=0.02,
    )

    collector.finalize()

    episodes = [
        json.loads(line)
        for line in (tmp_path / "episodes.rank-0.jsonl").read_text().splitlines()
    ]
    chunks = [
        json.loads(line)
        for line in (tmp_path / "chunks.rank-0.jsonl").read_text().splitlines()
    ]
    assert [episode["episode_identity"] for episode in episodes] == identities
    assert [chunk["episode_identity"] for chunk in chunks] == identities
