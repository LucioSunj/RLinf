# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.envs.libero.causal_snapshot import CausalSnapshotV1
from rlinf.envs.libero.stage_c_audit import (
    StageCInterleavingAuditExecutorV1,
    StageCSourceReplayExecutorV1,
)


class _SubmittedAudit:
    def __init__(self, value: float) -> None:
        self.value = value

    def record_for_batch_index(self, index: int):
        assert index == 0
        return {"schema": "fake-action-audit", "value": self.value}


class _Runtime:
    def sample_causal_action(
        self,
        *,
        env_obs,
        mode,
        action_seed: int,
        video_seed: int,
    ):
        mode_offset = 0.0 if mode.value == "c0_current" else 0.25
        video_term = 0.0 if video_seed is None else video_seed / 1e9
        value = (
            float(env_obs["chunk"]) + mode_offset + action_seed / 1_000_000 + video_term
        )
        return SimpleNamespace(actions=torch.full((1, 10, 7), value))


class _Env:
    num_envs = 1
    action_contract = object()

    def __init__(self, *, corrupt_replay: bool = False) -> None:
        self.corrupt_replay = corrupt_replay
        self.restore_count = 0
        self.task_ids = np.array([0])
        self.trial_ids = np.array([0])
        self.success_once = np.array([False])
        self.returns = np.array([0.0])
        self._elapsed_steps = np.array([0])
        self.chunk = 0
        self.sim_value = 0
        self.current_raw_obs = [self._raw()]

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _raw(self):
        value = self.chunk
        if self.corrupt_replay and self.restore_count and self.chunk == 2:
            value = 200
        return {
            "chunk": np.array([value], dtype=np.int64),
            "sim": np.array([self.sim_value], dtype=np.int64),
        }

    def _wrapped(self):
        return {"chunk": self.chunk, "sim": self.sim_value}

    def observe_causal_task_state(self):
        contact = self.chunk in {6, 7}
        predicate = self.chunk >= 11
        return {
            "schema": "causal-libero-task-observation-v1",
            "predicate_vector": (predicate,),
            "predicate_progress": float(predicate),
            "contact_by_object": {"target": contact},
            "contact_active": contact,
        }

    def capture_causal_snapshot(
        self,
        *,
        snapshot_id,
        recent_history,
        policy_runtime_state,
        source_policy,
        previous_mode,
        chunk_index,
        remaining_budget,
    ):
        assert chunk_index == self.chunk
        return CausalSnapshotV1(
            snapshot_id=snapshot_id,
            worker_state={
                "schema": "causal-snapshot-v1",
                "sim": {
                    "flattened": np.array([self.sim_value], dtype=np.int64),
                },
                "chunk": self.chunk,
                "controller": {"goal": self.chunk},
            },
            wrapper_state={
                "elapsed": int(self._elapsed_steps[0]),
                "success": bool(self.success_once[0]),
                "return": float(self.returns[0]),
            },
            current_raw_observation=copy.deepcopy(self.current_raw_obs[0]),
            recent_history=tuple(copy.deepcopy(recent_history)),
            policy_runtime_state=copy.deepcopy(policy_runtime_state),
            driver_rng_state={},
            source_policy=source_policy,
            previous_mode=previous_mode,
            chunk_index=chunk_index,
            remaining_budget=remaining_budget,
        )

    def restore_causal_snapshot(self, snapshot):
        self.restore_count += 1
        self.chunk = int(snapshot.worker_state["chunk"])
        self.sim_value = int(snapshot.worker_state["sim"]["flattened"][0])
        self._elapsed_steps[0] = int(snapshot.wrapper_state["elapsed"])
        self.success_once[0] = bool(snapshot.wrapper_state["success"])
        self.returns[0] = float(snapshot.wrapper_state["return"])
        self.current_raw_obs = [copy.deepcopy(snapshot.current_raw_observation)]
        return self._wrapped()

    def restore_causal_simulator_only_for_audit(self, snapshot):
        self.sim_value = int(snapshot.worker_state["sim"]["flattened"][0])

    def chunk_step_with_action_trace(self, actions, action_contract):
        assert action_contract is self.action_contract
        value = float(actions[0, 0, 0])
        self.chunk += 1
        self.sim_value += 1
        self._elapsed_steps[0] += 10
        if self.chunk >= 12:
            self.success_once[0] = True
            self.returns[0] = 1.0
        self.current_raw_obs = [self._raw()]
        result = (
            [self._wrapped()],
            torch.ones(1, 10),
            torch.zeros(1, 10, dtype=torch.bool),
            torch.zeros(1, 10, dtype=torch.bool),
            [{"chunk": self.chunk}],
        )
        return result, _SubmittedAudit(value)


def _executor(env: _Env, restored: list[dict]):
    return StageCSourceReplayExecutorV1(
        env=env,
        runtime=_Runtime(),
        seed_schedule=lambda index: (42 + 10_007 * index, 73 + 10_009 * index),
        capture_policy_runtime=lambda: {"chunk": env.chunk},
        restore_policy_runtime=lambda state: restored.append(dict(state)),
    )


def test_stage_c_two_pass_source_replay_captures_frozen_targets() -> None:
    env = _Env()
    restored = []
    result = _executor(env, restored).run(initial_observation=env._wrapped())

    assert [(target.phase, target.source.chunk_index) for target in result.targets] == [
        ("early", 0),
        ("mid", 3),
        ("contact", 6),
    ]
    assert [len(target.snapshot.recent_history) for target in result.targets] == [
        0,
        3,
        4,
    ]
    assert restored == [{"chunk": 0}]
    assert result.final_outcome["final_success"] is True
    artifact = result.to_artifact()
    assert artifact["status"] == "PASS"
    assert artifact["scientific_results"] == "NOT-RUN"
    assert all("raw_observation" not in row for row in artifact["source_chunks"])
    assert all("submitted_actions" not in row for row in artifact["source_chunks"])


def test_stage_c_second_pass_rejects_raw_observation_drift() -> None:
    env = _Env(corrupt_replay=True)
    with pytest.raises(AssertionError, match=r"stage_c_source\.raw\[2\]"):
        _executor(env, []).run(initial_observation=env._wrapped())


def test_stage_c_interleaving_executor_runs_all_three_exact_audits() -> None:
    env = _Env()
    source = _executor(env, []).run(initial_observation=env._wrapped())
    auditor = StageCInterleavingAuditExecutorV1(
        env=env,
        runtime=_Runtime(),
        seed_schedule=lambda index: (42 + 10_007 * index, 73 + 10_009 * index),
        capture_policy_runtime=lambda: {"chunk": env.chunk},
        restore_policy_runtime=lambda _state: None,
    )

    targets = auditor.run_targets(source.targets)

    assert [target["phase"] for target in targets] == ["early", "mid", "contact"]
    assert [target["chunk_index"] for target in targets] == [0, 3, 6]
    assert all(target["audit"]["status"] == "PASS" for target in targets)
    assert all(
        target["audit"]["simulator_only_negative_control"] == "EXPECTED-MISMATCH"
        for target in targets
    )
    assert all(target["audit"]["scientific_results"] == "NOT-RUN" for target in targets)
