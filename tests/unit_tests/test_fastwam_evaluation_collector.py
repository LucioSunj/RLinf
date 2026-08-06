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
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rlinf.data.embodied_io_struct import EnvOutput, RolloutResult
from rlinf.envs.action_contract import (
    DENORMALIZED_ACTION_STAGE,
    GRIPPER_CONVERTED_ACTION_STAGE,
    NORMALIZED_ACTION_STAGE,
    PREPARED_LIBERO_ACTION_STAGE,
    SUBMITTED_LIBERO_ACTION_STAGE,
    ActionExecutionTrace,
    ActionStageStatistics,
)
from rlinf.envs.libero.action_contract import LiberoActionContract
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
)
from rlinf.models.embodiment.wam_policy.evaluation import EvaluationRouteSelection
from rlinf.runners.fastwam_libero_eval_collector import (
    FastWAMLiberoEvalCollector,
    _load_ledger,
)


def _canonical_sha(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _action_contract() -> LiberoActionContract:
    return LiberoActionContract(
        low=(-1.0,) * 7,
        high=(1.0,) * 7,
        dimension_names=(
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_axis_angle_x",
            "delta_axis_angle_y",
            "delta_axis_angle_z",
            "gripper",
        ),
        gripper_dimension_index=6,
        outer_environment_classes=("unit.OffScreenRenderEnv",),
        underlying_environment_classes=("unit.LiberoTask",),
        robot_class="unit.SingleArm",
        robot_model="OnTheGroundPanda",
        controller_class="unit.OperationalSpaceController",
        controller_name="OSC_POSE",
        controller_input_low=(-1.0,) * 6,
        controller_input_high=(1.0,) * 6,
        controller_output_low=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        controller_output_high=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        gripper_class="unit.PandaGripper",
        gripper_dof=1,
        gripper_speed=0.01,
        control_frequency_hz=20,
        environment_horizon=1000,
        dependency_versions=(("robosuite_version", "1.4.0"),),
    )


def _action_trace(
    stages: tuple[str, ...],
    *,
    batch_size: int = 1,
    contract: LiberoActionContract | None = None,
) -> ActionExecutionTrace:
    contract = _action_contract() if contract is None else contract
    values = torch.zeros(batch_size, 2, 7)
    return ActionExecutionTrace(
        stages=tuple(
            ActionStageStatistics.from_values(
                stage=stage,
                values=values,
                low=contract.low,
                high=contract.high,
                gripper_dimension_index=contract.gripper_dimension_index,
                action_contract_sha256=contract.canonical_sha256,
            )
            for stage in stages
        )
    )


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
    action_contract = _action_contract()


def _rollout(
    *,
    chunk_id: int,
    route_used: int,
    forced: bool,
    source_chunk_id: int,
    terminal: bool,
    contract: LiberoActionContract | None = None,
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
        actions=torch.tensor([[[0.1] * 7, [0.9] * 7]]),
        route_info=route,
        emitted_gate=emitted,
        evaluation_selection=selection,
        action_execution_trace=_action_trace(
            (
                NORMALIZED_ACTION_STAGE,
                DENORMALIZED_ACTION_STAGE,
                GRIPPER_CONVERTED_ACTION_STAGE,
            ),
            contract=contract,
        ),
        gate_latency_seconds=torch.tensor([0.004], dtype=torch.float64),
        gate_h2d_seconds=torch.tensor([0.0], dtype=torch.float64),
    )


def _outcome(
    *,
    terminal: bool,
    success: bool = False,
    contract: LiberoActionContract | None = None,
) -> EnvOutput:
    terminations = torch.tensor([[False, success]])
    truncations = torch.tensor([[False, terminal and not success]])
    return EnvOutput(
        obs={"states": torch.zeros(1, 3)},
        dones=torch.tensor([terminal]),
        terminations=terminations,
        truncations=truncations,
        rewards=torch.tensor([[0.0, float(success)]]),
        action_execution_trace=_action_trace(
            (
                PREPARED_LIBERO_ACTION_STAGE,
                SUBMITTED_LIBERO_ACTION_STAGE,
            ),
            contract=contract,
        ),
    )


def _collector(
    tmp_path: Path,
    *,
    noise_seed_mode: str = "stateless_per_chunk",
) -> FastWAMLiberoEvalCollector:
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
        noise_seed_mode=noise_seed_mode,
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
    assert collector.is_complete is False

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
    assert collector.is_complete is True
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


def test_collector_writes_each_equivalent_live_contract(tmp_path) -> None:
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
        run_id="multi-contract-unit",
        rank=0,
        routing_mode="forced_idm",
        idm_threshold=0.5,
        random_idm_probability=None,
        routing_seed=0,
        fixed_idm_cost=0.01,
    )
    first_contract = _action_contract()
    second_contract = replace(
        first_contract,
        robot_model="MountedPanda",
        robot_models=("MountedPanda",),
        underlying_environment_classes=("unit.KitchenTask",),
    )
    env = _IdentityEnv()
    env_ids = torch.tensor([1 << 50])
    for episode_index, contract in enumerate((first_contract, second_contract)):
        env.trial_ids = [episode_index]
        env.reset_state_ids = [episode_index]
        env.action_contract = contract
        snapshot = collector.snapshot_before_step(0, env, env_ids)
        collector.record_chunk(
            snapshot=snapshot,
            rollout_result=_rollout(
                chunk_id=0,
                route_used=1,
                forced=True,
                source_chunk_id=-1,
                terminal=True,
                contract=contract,
            ),
            env_output=_outcome(terminal=True, contract=contract),
            environment_latency_seconds=0.02,
        )

    shard = collector.finalize()

    assert len(shard.action_contract_paths) == 2
    assert len(shard.action_contract_file_sha256s) == 2
    assert set(shard.action_contract_canonical_sha256s) == {
        first_contract.canonical_sha256,
        second_contract.canonical_sha256,
    }
    payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in shard.action_contract_paths
    ]
    assert {payload["canonical_sha256"] for payload in payloads} == set(
        shard.action_contract_canonical_sha256s
    )
    chunks = [
        json.loads(line)
        for line in (tmp_path / "chunks.rank-0.jsonl").read_text().splitlines()
    ]
    assert {chunk["action_contract_sha256"] for chunk in chunks} == set(
        shard.action_contract_canonical_sha256s
    )


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


def test_collector_fixed_episode_noise_matches_official_reseeding(tmp_path) -> None:
    collector = _collector(tmp_path, noise_seed_mode="fixed_per_episode")
    env = _IdentityEnv()
    env_ids = torch.tensor([1 << 50])

    first_snapshot = collector.snapshot_before_step(0, env, env_ids)
    first = collector.augment_rollout_input(
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
    second = collector.augment_rollout_input(
        {"obs": {"states": torch.zeros(1, 3)}},
        second_snapshot,
    )

    assert first["obs"]["_fastwam_action_noise_seeds"].tolist() == [101]
    assert first["obs"]["_fastwam_idm_noise_seeds"].tolist() == [202]
    assert torch.equal(
        first["obs"]["_fastwam_action_noise_seeds"],
        second["obs"]["_fastwam_action_noise_seeds"],
    )
    assert torch.equal(
        first["obs"]["_fastwam_idm_noise_seeds"],
        second["obs"]["_fastwam_idm_noise_seeds"],
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


def test_collector_fails_closed_when_ledger_episode_never_starts(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.json"
    payload = _ledger(ledger_path)
    second_identity = {
        **{
            key: value
            for key, value in payload["entries"][0].items()
            if key not in {"episode_index", "episode_identity"}
        },
        "trial_id": 1,
        "reset_state_id": 1,
    }
    payload["entries"].append(
        {
            "episode_index": 1,
            **second_identity,
            "episode_identity": _canonical_sha(second_identity),
        }
    )
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")
    collector = _collector(tmp_path)
    env = _IdentityEnv()
    snapshot = collector.snapshot_before_step(0, env, torch.tensor([1 << 50]))
    collector.record_chunk(
        snapshot=snapshot,
        rollout_result=_rollout(
            chunk_id=0,
            route_used=1,
            forced=True,
            source_chunk_id=-1,
            terminal=True,
        ),
        env_output=_outcome(terminal=True),
        environment_latency_seconds=0.02,
    )

    assert collector.is_complete is False
    with pytest.raises(RuntimeError, match="without starting ledger episodes"):
        collector.finalize()


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
            action_execution_trace=_action_trace(
                (
                    NORMALIZED_ACTION_STAGE,
                    DENORMALIZED_ACTION_STAGE,
                    GRIPPER_CONVERTED_ACTION_STAGE,
                ),
                batch_size=2,
            ),
        ),
        env_output=EnvOutput(
            obs={"states": torch.zeros(2, 3)},
            dones=torch.tensor([True, True]),
            terminations=torch.tensor([[False], [False]]),
            truncations=torch.tensor([[True], [True]]),
            rewards=torch.zeros(2, 1),
            action_execution_trace=_action_trace(
                (
                    PREPARED_LIBERO_ACTION_STAGE,
                    SUBMITTED_LIBERO_ACTION_STAGE,
                ),
                batch_size=2,
            ),
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


def _ledger_v2(path: Path, *, max_primitive_steps: int = 700) -> dict:
    identity_fields = {
        "task_suite": "libero_10",
        "task_id": 0,
        "trial_id": 0,
        "reset_state_id": 0,
        "environment_seed": 7,
        "action_noise_seed": 101,
        "idm_video_noise_seed": 202,
        "max_primitive_steps": max_primitive_steps,
        "generation_horizon": 32,
        "execution_horizon": 10,
        "prediction_video_frames": 9,
        "reset_wait_steps": 30,
    }
    payload = {
        "schema": "fastwam-libero-eval-ledger-v2",
        "kind": "validation",
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


def test_collector_loads_explicit_ledger_v2_protocol(tmp_path) -> None:
    path = tmp_path / "ledger-v2.json"
    expected = _ledger_v2(path)

    assert _load_ledger(path) == expected


def test_collector_rejects_malformed_ledger_v2_protocol(tmp_path) -> None:
    path = tmp_path / "ledger-v2-invalid.json"
    _ledger_v2(path, max_primitive_steps=701)

    with pytest.raises(ValueError, match="inconsistent"):
        _load_ledger(path)
