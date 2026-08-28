# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastwam.causal_prediction import CausalComputeMode

from rlinf.envs.action_contract import (
    NORMALIZED_ACTION_STAGE,
    SUBMITTED_LIBERO_ACTION_STAGE,
    ActionExecutionTrace,
    ActionStageStatistics,
)
from rlinf.envs.libero.action_contract import LiberoActionContract
from rlinf.envs.libero.causal_collection import (
    SourceChunkTraceV2,
    select_stage_c_snapshot_audit_targets,
)
from rlinf.envs.libero.causal_snapshot import (
    CausalSnapshotV1,
    audit_interleaved_snapshot_restore,
    capture_process_rng_state,
    capture_worker_causal_state,
    observe_worker_causal_determinism_state,
    observe_worker_causal_task_state,
    restore_worker_causal_state,
    restore_worker_simulator_only_for_audit,
)
from rlinf.models.embodiment.wam_policy.causal_experiment import (
    PairedCausalForkRunner,
)
from rlinf.models.embodiment.wam_policy.causal_runtime import (
    CausalConditionContract,
)


class _Sim:
    def __init__(self) -> None:
        contact = SimpleNamespace(
            geom1=3,
            geom2=7,
            dist=-0.002,
            pos=np.array([0.1, 0.2, 0.3]),
            frame=np.arange(9, dtype=np.float64),
        )
        self.data = SimpleNamespace(
            time=1.5,
            qpos=np.array([1.0, 2.0]),
            qvel=np.array([3.0, 4.0]),
            act=np.array([5.0]),
            mocap_pos=np.array([[6.0, 7.0, 8.0]]),
            mocap_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
            ctrl=np.array([0.3]),
            qacc_warmstart=np.array([0.4, 0.5]),
            ncon=1,
            contact=(contact,),
        )

    def set_state_from_flattened(self, state) -> None:
        state = np.asarray(state)
        self.data.time = float(state[0])
        self.data.qpos[...] = state[1:3]
        self.data.qvel[...] = state[3:5]

    def forward(self) -> None:
        pass


class _Env:
    def __init__(self) -> None:
        self.sim = _Sim()
        interpolator = SimpleNamespace(
            dim=3,
            ori_interpolate=False,
            order=1,
            step=2,
            total_steps=5,
            use_delta_goal=False,
            start=np.array([0.1, 0.2, 0.3]),
            goal=np.array([0.4, 0.5, 0.6]),
        )
        controller = SimpleNamespace(
            goal_pos=np.array([1.0, 2.0, 3.0]),
            goal_ori=np.eye(3),
            relative_ori=np.eye(3),
            ori_ref=np.eye(3),
            interpolator_pos=interpolator,
            interpolator_ori=copy.deepcopy(interpolator),
            new_update=True,
        )
        gripper = SimpleNamespace(
            current_action=np.array([0.2]), init_qpos=np.array([0.0])
        )
        self.robots = [
            SimpleNamespace(
                controller=controller,
                gripper=gripper,
                recent_actions=np.array([0.6, 0.7]),
            )
        ]
        self._observables = {
            "eef": SimpleNamespace(
                _time_since_last_sample=0.01,
                _current_delay=0.02,
                _current_observed_value=np.array([9.0]),
                _sampled=True,
            )
        }
        self.np_random = np.random.default_rng(31)
        self.parsed_problem = {"goal_state": (("goal", "target"),)}
        self.obj_of_interest = ("target",)
        self.objects_dict = {"target": object()}
        self.fixtures_dict = {}
        self.predicate_active = False
        self.contact_active = False
        self.timestep = 11
        self.cur_time = 0.55
        self.done = False

    def get_sim_state(self):
        data = self.sim.data
        # Match robosuite's real MjSimState: actuator activation is not part of
        # the flattened state and must be restored separately.
        return np.concatenate([[data.time], data.qpos, data.qvel])

    def _eval_predicate(self, _goal):
        return self.predicate_active

    def check_contact(self, gripper, object_model):
        assert gripper is self.robots[0].gripper
        assert object_model is self.objects_dict["target"]
        return self.contact_active


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
        outer_environment_classes=("unit.Outer",),
        underlying_environment_classes=("unit.Task",),
        robot_class="unit.SingleArm",
        robot_model="Panda",
        controller_class="unit.OSC",
        controller_name="OSC_POSE",
        controller_input_low=(-1.0,) * 6,
        controller_input_high=(1.0,) * 6,
        controller_output_low=(-0.05,) * 6,
        controller_output_high=(0.05,) * 6,
        gripper_class="unit.Gripper",
        gripper_dof=1,
        gripper_speed=0.01,
        control_frequency_hz=20,
        environment_horizon=700,
        dependency_versions=(("robosuite_version", "1.4.0"),),
    )


def test_worker_snapshot_round_trip_restores_all_supported_state() -> None:
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    env = _Env()
    snapshot = capture_worker_causal_state(env)
    expected_rng = (random.random(), np.random.random(), torch.rand(()).item())
    expected_local_rng = env.np_random.random()

    env.sim.data.qpos[:] = -1
    env.sim.data.qvel[:] = -2
    env.sim.data.act[:] = -3
    env.sim.data.ctrl[:] = -3
    env.sim.data.qacc_warmstart[:] = -3
    env.robots[0].controller.goal_pos[:] = -4
    env.robots[0].controller.interpolator_pos.goal[:] = -5
    env.robots[0].gripper.current_action[:] = -6
    env.robots[0].recent_actions[:] = -6
    env.timestep = 99
    env.cur_time = 99.0
    env.done = True
    env._observables["eef"]._current_observed_value[:] = -7
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    env.np_random = np.random.default_rng(99)

    restore_worker_causal_state(env, snapshot)
    assert np.array_equal(env.sim.data.qpos, [1.0, 2.0])
    assert np.array_equal(env.sim.data.qvel, [3.0, 4.0])
    assert np.array_equal(env.robots[0].controller.goal_pos, [1.0, 2.0, 3.0])
    assert np.array_equal(
        env.robots[0].controller.interpolator_pos.goal,
        [0.4, 0.5, 0.6],
    )
    assert np.array_equal(env.robots[0].gripper.current_action, [0.2])
    assert np.array_equal(env.robots[0].recent_actions, [0.6, 0.7])
    assert np.array_equal(env.sim.data.ctrl, [0.3])
    assert np.array_equal(env.sim.data.qacc_warmstart, [0.4, 0.5])
    assert (env.timestep, env.cur_time, env.done) == (11, 0.55, False)
    assert np.array_equal(env._observables["eef"]._current_observed_value, [9.0])
    assert (random.random(), np.random.random(), torch.rand(()).item()) == expected_rng
    assert env.np_random.random() == expected_local_rng


def test_simulator_only_negative_control_leaves_controller_state_mutated() -> None:
    env = _Env()
    snapshot = capture_worker_causal_state(env)
    env.sim.data.qpos[:] = -1
    env.robots[0].controller.goal_pos[:] = -2

    restore_worker_simulator_only_for_audit(env, snapshot)

    assert np.array_equal(env.sim.data.qpos, [1.0, 2.0])
    assert np.array_equal(env.robots[0].controller.goal_pos, [-2.0, -2.0, -2.0])


def test_restore_accepts_one_ulp_mujoco_qpos_normalization() -> None:
    env = _Env()
    snapshot = capture_worker_causal_state(env)

    def normalize_qpos_like_mujoco_forward() -> None:
        env.sim.data.qpos[0] = np.nextafter(env.sim.data.qpos[0], np.inf)

    env.sim.forward = normalize_qpos_like_mujoco_forward
    restore_worker_causal_state(env, snapshot)

    assert env.sim.data.qpos[0] != snapshot["sim"]["qpos"][0]
    assert abs(env.sim.data.qpos[0] - snapshot["sim"]["qpos"][0]) <= 1e-15


def test_native_task_observation_reads_predicate_and_gripper_contact() -> None:
    env = _Env()
    first = observe_worker_causal_task_state(env)
    env.predicate_active = True
    env.contact_active = True
    second = observe_worker_causal_task_state(env)

    assert first == {
        "schema": "causal-libero-task-observation-v1",
        "predicate_vector": (False,),
        "predicate_progress": 0.0,
        "contact_by_object": {"target": False},
        "contact_active": False,
    }
    assert second["predicate_vector"] == (True,)
    assert second["predicate_progress"] == 1.0
    assert second["contact_by_object"] == {"target": True}
    assert second["contact_active"] is True


def test_determinism_observation_reads_full_physics_and_contacts() -> None:
    env = _Env()
    observed = observe_worker_causal_determinism_state(env)

    assert observed["schema"] == "causal-libero-determinism-observation-v1"
    assert observed["simulator"]["time"] == 1.5
    assert np.array_equal(observed["simulator"]["qpos"], [1.0, 2.0])
    assert np.array_equal(observed["simulator"]["dynamic"]["ctrl"], [0.3])
    assert len(observed["contacts"]) == 1
    assert observed["contacts"][0]["geom1"] == 3
    assert observed["contacts"][0]["geom2"] == 7
    assert observed["contacts"][0]["dist"] == -0.002
    assert np.array_equal(observed["contacts"][0]["pos"], [0.1, 0.2, 0.3])
    assert observed["task"]["contact_active"] is False


def test_outer_snapshot_contract_carries_history_policy_and_identity() -> None:
    env = _Env()
    worker = capture_worker_causal_state(env)
    snapshot = CausalSnapshotV1(
        snapshot_id="spatial-3-trial-2-chunk-4",
        worker_state=worker,
        wrapper_state={"task_id": 3},
        current_raw_observation={"robot0_eef_pos": np.zeros(3)},
        recent_history=({"action": [0.0] * 7},),
        policy_runtime_state={"cache": torch.ones(2)},
        driver_rng_state=capture_process_rng_state(),
        source_policy="always_c0",
        previous_mode=None,
        chunk_index=4,
        remaining_budget=0.5,
    )
    assert snapshot.schema == "causal-snapshot-v1"
    assert snapshot.recent_history[0]["action"] == [0.0] * 7


def test_c0_condition_is_logically_full_but_physically_compact() -> None:
    contract = CausalConditionContract(
        mode=CausalComputeMode.C0_CURRENT,
        logical_input_frames=9,
        logical_future_frames=8,
        current_frame_video_tokens=16,
        physical_video_tokens=16,
        physical_future_tokens=0,
    )
    assert contract.logical_future_frames == 8
    assert contract.physical_future_tokens == 0


def test_paired_runner_uses_live_submitted_action_audit() -> None:
    contract = _action_contract()

    class Env:
        num_envs = 1
        action_contract = contract

        def restore_causal_snapshot(self, _snapshot):
            self.task_ids = np.array([3])
            self.trial_ids = np.array([2])
            self.reset_state_ids = np.array([152])
            self.success_once = np.array([False])
            self.returns = np.array([0.0])
            self.elapsed_steps = np.array([40])
            return {"task_descriptions": ["test"]}

        def chunk_step_with_action_trace(self, actions, live_contract):
            submitted = ActionStageStatistics.from_values(
                stage=SUBMITTED_LIBERO_ACTION_STAGE,
                values=actions,
                low=live_contract.low,
                high=live_contract.high,
                gripper_dimension_index=live_contract.gripper_dimension_index,
                action_contract_sha256=live_contract.canonical_sha256,
            )
            assert not bool((submitted.below_low_count > 0).any())
            assert not bool((submitted.above_high_count > 0).any())
            self.success_once[:] = True
            self.returns[:] = 1.0
            self.elapsed_steps += actions.shape[1]
            return ([{"task_descriptions": ["test"]}], None), submitted

    class Runtime:
        action_protocol = SimpleNamespace(max_episode_steps=700)

        def sample_causal_action(self, *, env_obs, mode, action_seed, video_seed):
            del env_obs, action_seed, video_seed
            actions = torch.zeros(1, 10, 7)
            trace = ActionExecutionTrace(
                (
                    ActionStageStatistics.from_values(
                        stage=NORMALIZED_ACTION_STAGE,
                        values=actions,
                        low=contract.low,
                        high=contract.high,
                        gripper_dimension_index=contract.gripper_dimension_index,
                        action_contract_sha256=contract.canonical_sha256,
                    ),
                )
            )
            return SimpleNamespace(
                mode=CausalComputeMode.parse(mode),
                actions=actions,
                action_execution_trace=trace,
                latency_ms={"critical_path": 1.0},
            )

    snapshot = CausalSnapshotV1(
        snapshot_id="state-1",
        worker_state={"schema": "causal-snapshot-v1"},
        wrapper_state={},
        current_raw_observation={},
        recent_history=(),
        policy_runtime_state={},
        driver_rng_state={},
        source_policy="always_c0",
        previous_mode=None,
        chunk_index=4,
        remaining_budget=0.5,
    )
    runner = PairedCausalForkRunner(
        env=Env(),
        runtime=Runtime(),
        restore_policy_runtime=lambda _state: None,
        restore_history=lambda _history: None,
    )
    records = runner.run_snapshot(
        snapshot=snapshot,
        state_index=1,
        modes=(CausalComputeMode.C0_CURRENT, CausalComputeMode.C2_FULL),
        replicates=1,
        inclusion_probability=0.1,
    )
    assert len(records) == 2
    for record in records:
        stages = record.secondary_outcomes["submitted_action_audit"]["stages"]
        assert SUBMITTED_LIBERO_ACTION_STAGE in stages


def test_interleaved_audit_compares_complete_branch_trace_exactly() -> None:
    state = {"sim": 3, "controller": 10}

    def restore():
        state.update(sim=3, controller=10)

    def restore_simulator_only():
        state["sim"] = 3

    def run(mode):
        before = dict(state)
        increment = 1 if mode == "A" else 2
        state["sim"] += increment
        state["controller"] += increment
        return {
            "raw_observation": np.array(
                [before["sim"], before["controller"]],
                dtype=np.float64,
            ),
            "submitted_actions": torch.tensor(
                [[before["sim"], before["controller"]]],
                dtype=torch.float32,
            ),
            "next_simulator_state": np.array([state["sim"]]),
            "reward": float(mode == "A"),
            "success": mode == "A",
            "metrics": {"before": before},
            "continuation_outcome": (mode, before),
        }

    report = audit_interleaved_snapshot_restore(
        restore=restore,
        restore_simulator_only=restore_simulator_only,
        run_branch=run,
        mode_a="A",
        mode_b="B",
        phase="contact",
    )
    assert report["status"] == "PASS"
    assert report["scientific_results"] == "NOT-RUN"
    assert report["phase"] == "contact"
    assert report["simulator_only_negative_control"] == "EXPECTED-MISMATCH"


def test_interleaved_audit_rejects_incomplete_trace() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        audit_interleaved_snapshot_restore(
            restore=lambda: None,
            restore_simulator_only=lambda: None,
            run_branch=lambda _mode: {"raw_observation": np.zeros(1)},
            mode_a="c0_current",
            mode_b="c2_full",
            phase="early",
        )


def _source_trace(chunk: int, *, contact: bool = False) -> SourceChunkTraceV2:
    return SourceChunkTraceV2(
        chunk_index=chunk,
        elapsed_steps=chunk * 10,
        eligible=True,
        terminated=False,
        remaining_action_capacity=700 - chunk * 10,
        raw_observation={"chunk": chunk},
        submitted_actions=((0.0,) * 7,),
        predicate_vector=(False,),
        predicate_progress=0.0,
        criticality_components={
            "action_curvature": 0.0,
            "action_precision": 0.0,
            "contact_proximity": float(contact),
            "gripper_transition": 0.0,
            "predicate_transition": 0.0,
        },
        contact_active=contact,
        previous_contact_active=False,
        gripper_closing=False,
        nearest_task_object_distance_m=None,
        predicate_changed=False,
        lightweight_summary={},
    )


def test_stage_c_targets_are_first_midpoint_and_first_contact() -> None:
    targets = select_stage_c_snapshot_audit_targets(
        [_source_trace(index, contact=index in {6, 7}) for index in range(8)]
    )
    assert [(target.phase, target.trace.chunk_index) for target in targets] == [
        ("early", 0),
        ("mid", 3),
        ("contact", 6),
    ]


def test_stage_c_target_selection_does_not_reselect_missing_contact() -> None:
    with pytest.raises(ValueError, match="do not reselect"):
        select_stage_c_snapshot_audit_targets(
            [_source_trace(index) for index in range(4)]
        )
