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

import json

import pytest
import torch

from rlinf.models.embodiment.wam_policy.contracts import GateDecisionRecord
from rlinf.runners.fastwam_decision_telemetry import (
    FASTWAM_DECISION_TELEMETRY_SCHEMA,
    append_fastwam_decision_telemetry_jsonl,
    build_fastwam_decision_telemetry_record,
    build_fastwam_training_decision_records,
)


def _emitted() -> GateDecisionRecord:
    shape = (2, 2)
    base = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
    behavior = 0.9 * base + 0.05
    return GateDecisionRecord(
        next_route=torch.tensor([[0, 1], [1, 0]]),
        base_probability=base,
        behavior_probability=behavior,
        old_logprob=torch.zeros(shape),
        epsilon=torch.full(shape, 0.1),
        temperature=torch.ones(shape),
        valid=torch.ones(shape, dtype=torch.bool),
        source_chunk_ids=torch.tensor([[0, 0], [1, 1]]),
        episode_ids=torch.tensor([[4, 7], [4, 7]]),
        actor_versions=torch.full(shape, 9),
        exploration_forced=torch.tensor([[False, True], [False, False]]),
        mode_flip_delta=torch.tensor([[0.1, -0.2], [0.3, -0.4]]),
        environment_ids=torch.tensor([[10, 20], [10, 20]]),
        task_ids=torch.tensor([[1, 2], [1, 2]]),
        trial_ids=torch.tensor([[3, 4], [3, 4]]),
        reset_state_ids=torch.tensor([[13, 24], [13, 24]]),
    )


def test_training_decision_records_preserve_per_decision_values(tmp_path) -> None:
    records = build_fastwam_training_decision_records(
        emitted=_emitted(),
        gate_valid_mask=torch.tensor([[True, False], [False, True]]),
        unnormalized_gate_advantages=torch.tensor([[1.25, 0.0], [0.0, -2.5]]),
        normalized_gate_advantages=torch.tensor([[0.5, 0.0], [0.0, -1.0]]),
        runner_step=12,
        rank=3,
        run_id="hostb-fix6",
        task_suite="libero_10",
        configured_idm_cost=0.015,
    )

    assert len(records) == 2
    assert records[0] == {
        "schema": FASTWAM_DECISION_TELEMETRY_SCHEMA,
        "phase": "training",
        "run_id": "hostb-fix6",
        "rank": 3,
        "trajectory_id": "10:4",
        "env_id": 10,
        "episode_id": 4,
        "task_suite": "libero_10",
        "task_id": 1,
        "trial_id": 3,
        "reset_state_id": 13,
        "record_id": "10:4:0",
        "cycle_index": 0,
        "destination_cycle_index": 1,
        "update_step": 12,
        "actor_version": 9,
        "route": "uncond",
        "gate_idm_probability": pytest.approx(0.2),
        "gate_behavior_idm_probability": pytest.approx(0.23),
        "forced_exploration": False,
        "configured_idm_cost": 0.015,
        "destination_advantage_unnormalized": 1.25,
        "destination_advantage_normalized": 0.5,
        "mode_flip_delta": pytest.approx(0.1),
        "eligible_decision": True,
    }
    assert records[1]["trajectory_id"] == "20:7"
    assert records[1]["cycle_index"] == 1
    assert records[1]["destination_advantage_unnormalized"] == -2.5

    path = tmp_path / "audits/training_decisions.rank-3.jsonl"
    append_fastwam_decision_telemetry_jsonl(path, records)
    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert written == records


def test_evaluation_and_training_use_the_same_decision_schema() -> None:
    evaluation = build_fastwam_decision_telemetry_record(
        phase="evaluation",
        run_id="eval",
        rank=0,
        trajectory_id="episode-1",
        env_id=5,
        episode_id=1,
        task_suite="libero_10",
        task_id=2,
        trial_id=8,
        reset_state_id=108,
        cycle_index=4,
        update_step=30,
        actor_version=30,
        route="idm",
        base_probability=0.8,
        behavior_probability=0.8,
        forced_exploration=False,
        mode_flip_delta=-0.1,
        configured_idm_cost=None,
        destination_advantage_unnormalized=None,
        destination_advantage_normalized=None,
        eligible_decision=True,
    )

    assert evaluation["schema"] == FASTWAM_DECISION_TELEMETRY_SCHEMA
    assert evaluation["phase"] == "evaluation"
    assert evaluation["destination_advantage_normalized"] is None


def test_training_decision_telemetry_requires_rollout_identity() -> None:
    emitted = _emitted()
    emitted = GateDecisionRecord(
        **{
            name: getattr(emitted, name)
            for name in (
                "next_route",
                "base_probability",
                "behavior_probability",
                "old_logprob",
                "epsilon",
                "temperature",
                "valid",
                "source_chunk_ids",
                "episode_ids",
                "actor_versions",
                "exploration_forced",
                "mode_flip_delta",
            )
        }
    )

    with pytest.raises(ValueError, match="missing rollout metadata"):
        build_fastwam_training_decision_records(
            emitted=emitted,
            gate_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            unnormalized_gate_advantages=torch.ones(2, 2),
            normalized_gate_advantages=torch.ones(2, 2),
            runner_step=1,
            rank=0,
            run_id="missing",
            task_suite="libero_10",
            configured_idm_cost=0.01,
        )
