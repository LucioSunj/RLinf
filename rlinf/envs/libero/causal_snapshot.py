# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exact LIBERO snapshot contracts for same-state causal forks."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch

if TYPE_CHECKING:
    from fastwam.causal_prediction import (
        CausalSamplingMetadataV2,
        CausalStateIdentityV2,
    )

CAUSAL_SNAPSHOT_SCHEMA = "causal-snapshot-v1"
CAUSAL_SNAPSHOT_SCHEMA_V2 = "causal-snapshot-v2"
CAUSAL_SNAPSHOT_AUDIT_SCHEMA = "causal-snapshot-interleaving-audit-v2"
CAUSAL_SNAPSHOT_AUDIT_PHASES = ("early", "mid", "contact")
CAUSAL_SNAPSHOT_AUDIT_ORDERS = ("A-restore-A", "A-B-restore-A")
CAUSAL_SNAPSHOT_AUDIT_REQUIRED_FIELDS = (
    "raw_observation",
    "submitted_actions",
    "next_simulator_state",
    "reward",
    "success",
    "metrics",
    "continuation_outcome",
)

_CONTROLLER_FIELDS = (
    "goal_pos",
    "goal_ori",
    "relative_ori",
    "ori_ref",
    "initial_joint",
    "joint_pos",
    "joint_vel",
    "torque_compensation",
    "torques",
    "new_update",
    "ee_pos",
    "ee_ori_mat",
    "ee_pos_vel",
    "ee_ori_vel",
    "J_pos",
    "J_ori",
    "J_full",
    "mass_matrix",
    "initial_ee_pos",
    "initial_ee_ori_mat",
)
_INTERPOLATOR_FIELDS = (
    "dim",
    "ori_interpolate",
    "order",
    "step",
    "total_steps",
    "use_delta_goal",
    "start",
    "goal",
)
_GRIPPER_FIELDS = (
    "current_action",
    "_current_action",
    "action",
    "init_qpos",
)
_ROBOT_FIELDS = (
    "torques",
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_vel_buffer",
    "recent_ee_acc",
)
_ENVIRONMENT_FIELDS = ("timestep", "cur_time", "done")
_SIM_DYNAMIC_FIELDS = (
    "ctrl",
    "qacc",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "userdata",
)
_OBSERVABLE_FIELDS = (
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
)


def _clone(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return copy.deepcopy(value)


def capture_process_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and initialized Torch RNG streams."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_process_rng_state(state: Mapping[str, Any]) -> None:
    """Restore exactly the RNG streams captured by this schema."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Causal snapshot RNG state is missing: {missing}.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8))
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore a CUDA RNG snapshot without CUDA.")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8) for item in cuda_state]
        )


def _wrapper_chain(env: Any) -> tuple[Any, ...]:
    chain = []
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = getattr(current, "env", None)
    return tuple(chain)


def _find_sim(env: Any) -> Any:
    for wrapper in _wrapper_chain(env):
        sim = getattr(wrapper, "sim", None)
        if sim is not None:
            return sim
    raise RuntimeError("LIBERO causal snapshot could not locate the MuJoCo simulator.")


def _find_robots(env: Any) -> tuple[Any, ...]:
    for wrapper in _wrapper_chain(env):
        robots = getattr(wrapper, "robots", None)
        if robots is not None:
            return tuple(robots)
    raise RuntimeError("LIBERO causal snapshot could not locate robot controllers.")


def _find_task_environment(env: Any) -> Any:
    for wrapper in reversed(_wrapper_chain(env)):
        if isinstance(getattr(wrapper, "parsed_problem", None), Mapping) and callable(
            getattr(wrapper, "_eval_predicate", None)
        ):
            return wrapper
    raise RuntimeError("LIBERO causal observation could not locate the task domain.")


def _capture_fields(owner: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: _clone(getattr(owner, name)) for name in fields if hasattr(owner, name)
    }


def _restore_fields(owner: Any, state: Mapping[str, Any]) -> None:
    for name, value in state.items():
        if not hasattr(owner, name):
            raise RuntimeError(
                f"Snapshot field {name!r} disappeared from {type(owner).__name__}."
            )
        current = getattr(owner, name)
        if isinstance(current, np.ndarray):
            current[...] = value
        elif isinstance(current, torch.Tensor):
            current.copy_(value.to(device=current.device, dtype=current.dtype))
        else:
            setattr(owner, name, _clone(value))


def _capture_controller(controller: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "class": type(controller).__name__,
        "fields": _capture_fields(controller, _CONTROLLER_FIELDS),
        "interpolators": {},
    }
    for name in ("interpolator_pos", "interpolator_ori"):
        interpolator = getattr(controller, name, None)
        if interpolator is not None:
            state["interpolators"][name] = _capture_fields(
                interpolator,
                _INTERPOLATOR_FIELDS,
            )
    return state


def _restore_controller(controller: Any, state: Mapping[str, Any]) -> None:
    if state.get("class") != type(controller).__name__:
        raise RuntimeError(
            "Controller type changed between capture and restore: "
            f"{state.get('class')!r} != {type(controller).__name__!r}."
        )
    _restore_fields(controller, state["fields"])
    for name, interpolator_state in state["interpolators"].items():
        interpolator = getattr(controller, name, None)
        if interpolator is None:
            raise RuntimeError(f"Controller interpolator {name!r} disappeared.")
        _restore_fields(interpolator, interpolator_state)


def _capture_observables(env: Any) -> dict[str, dict[str, Any]]:
    for wrapper in reversed(_wrapper_chain(env)):
        observables = getattr(wrapper, "_observables", None)
        if isinstance(observables, Mapping):
            return {
                str(name): _capture_fields(observable, _OBSERVABLE_FIELDS)
                for name, observable in observables.items()
            }
    return {}


def _restore_observables(env: Any, state: Mapping[str, Mapping[str, Any]]) -> None:
    if not state:
        return
    for wrapper in reversed(_wrapper_chain(env)):
        observables = getattr(wrapper, "_observables", None)
        if isinstance(observables, Mapping):
            missing = sorted(set(state) - set(observables))
            if missing:
                raise RuntimeError(f"LIBERO observables disappeared: {missing}.")
            for name, observable_state in state.items():
                _restore_fields(observables[name], observable_state)
            return
    raise RuntimeError("Snapshot contains observables but restore found none.")


def _capture_local_generators(env: Any) -> list[dict[str, Any]]:
    states = []
    for depth, wrapper in enumerate(_wrapper_chain(env)):
        for name in ("np_random", "_generator", "rng"):
            generator = getattr(wrapper, name, None)
            if isinstance(generator, np.random.Generator):
                states.append(
                    {
                        "depth": depth,
                        "name": name,
                        "kind": "generator",
                        "state": _clone(generator.bit_generator.state),
                    }
                )
            elif isinstance(generator, np.random.RandomState):
                states.append(
                    {
                        "depth": depth,
                        "name": name,
                        "kind": "random_state",
                        "state": _clone(generator.get_state()),
                    }
                )
    return states


def _restore_local_generators(env: Any, states: list[Mapping[str, Any]]) -> None:
    chain = _wrapper_chain(env)
    for state in states:
        owner = chain[int(state["depth"])]
        generator = getattr(owner, state["name"], None)
        if state["kind"] == "generator" and isinstance(generator, np.random.Generator):
            generator.bit_generator.state = _clone(state["state"])
        elif state["kind"] == "random_state" and isinstance(
            generator, np.random.RandomState
        ):
            generator.set_state(state["state"])
        else:
            raise RuntimeError("A LIBERO local RNG changed type before restore.")


def _capture_environment_fields(env: Any) -> list[dict[str, Any]]:
    states = []
    for depth, wrapper in enumerate(_wrapper_chain(env)):
        fields = _capture_fields(wrapper, _ENVIRONMENT_FIELDS)
        if fields:
            states.append(
                {
                    "depth": depth,
                    "class": type(wrapper).__name__,
                    "fields": fields,
                }
            )
    return states


def _restore_environment_fields(env: Any, states: list[Mapping[str, Any]]) -> None:
    chain = _wrapper_chain(env)
    for state in states:
        owner = chain[int(state["depth"])]
        if state["class"] != type(owner).__name__:
            raise RuntimeError("A LIBERO wrapper changed type before restore.")
        _restore_fields(owner, state["fields"])


def capture_worker_causal_state(env: Any) -> dict[str, Any]:
    """Capture simulator, controller, wrapper-observable, and worker RNG state."""

    sim = _find_sim(env)
    robots = _find_robots(env)
    return {
        "schema": CAUSAL_SNAPSHOT_SCHEMA,
        "sim": {
            "flattened": np.asarray(env.get_sim_state()).copy(),
            "time": float(sim.data.time),
            "qpos": np.asarray(sim.data.qpos).copy(),
            "qvel": np.asarray(sim.data.qvel).copy(),
            "act": np.asarray(sim.data.act).copy(),
            "mocap_pos": np.asarray(sim.data.mocap_pos).copy(),
            "mocap_quat": np.asarray(sim.data.mocap_quat).copy(),
            "dynamic": _capture_fields(sim.data, _SIM_DYNAMIC_FIELDS),
        },
        "environment_fields": _capture_environment_fields(env),
        "robots": [
            {
                "fields": _capture_fields(robot, _ROBOT_FIELDS),
                "controller": _capture_controller(robot.controller),
                "gripper": _capture_fields(robot.gripper, _GRIPPER_FIELDS),
            }
            for robot in robots
        ],
        "observables": _capture_observables(env),
        "local_rng": _capture_local_generators(env),
        "process_rng": capture_process_rng_state(),
    }


def observe_worker_causal_task_state(env: Any) -> dict[str, Any]:
    """Read native goal predicates and gripper-to-task-object contacts."""

    task = _find_task_environment(env)
    goals = tuple(task.parsed_problem.get("goal_state", ()))
    if not goals:
        raise RuntimeError("LIBERO causal observation found no BDDL goal predicates.")
    predicate_vector = tuple(bool(task._eval_predicate(goal)) for goal in goals)
    object_names = tuple(str(name) for name in getattr(task, "obj_of_interest", ()))
    if not object_names:
        raise RuntimeError("LIBERO causal observation found no objects of interest.")
    object_models = {
        str(name): model
        for collection_name in ("objects_dict", "fixtures_dict")
        for name, model in getattr(task, collection_name, {}).items()
    }
    missing = sorted(set(object_names) - set(object_models))
    if missing:
        raise RuntimeError(f"LIBERO task objects of interest disappeared: {missing}.")
    robots = _find_robots(env)
    contact_by_object = {
        name: any(
            bool(task.check_contact(robot.gripper, object_models[name]))
            for robot in robots
        )
        for name in object_names
    }
    return {
        "schema": "causal-libero-task-observation-v1",
        "predicate_vector": predicate_vector,
        "predicate_progress": sum(predicate_vector) / len(predicate_vector),
        "contact_by_object": contact_by_object,
        "contact_active": any(contact_by_object.values()),
    }


def restore_worker_causal_state(env: Any, state: Mapping[str, Any]) -> None:
    """Restore every worker-owned state component before another branch."""

    if state.get("schema") != CAUSAL_SNAPSHOT_SCHEMA:
        raise ValueError(f"Unsupported causal snapshot schema {state.get('schema')!r}.")
    sim = _find_sim(env)
    sim_state = state["sim"]
    sim.set_state_from_flattened(np.asarray(sim_state["flattened"]))
    sim.data.mocap_pos[...] = sim_state["mocap_pos"]
    sim.data.mocap_quat[...] = sim_state["mocap_quat"]
    sim.forward()
    _restore_fields(sim.data, sim_state["dynamic"])
    if not (
        float(sim.data.time) == float(sim_state["time"])
        and np.array_equal(sim.data.qpos, sim_state["qpos"])
        and np.array_equal(sim.data.qvel, sim_state["qvel"])
        and np.array_equal(sim.data.act, sim_state["act"])
    ):
        raise RuntimeError("MuJoCo state did not restore exactly.")
    _restore_environment_fields(env, state["environment_fields"])
    robots = _find_robots(env)
    if len(robots) != len(state["robots"]):
        raise RuntimeError("Robot count changed between capture and restore.")
    for robot, robot_state in zip(robots, state["robots"]):
        _restore_fields(robot, robot_state["fields"])
        _restore_controller(robot.controller, robot_state["controller"])
        _restore_fields(robot.gripper, robot_state["gripper"])
    _restore_observables(env, state["observables"])
    _restore_local_generators(env, state["local_rng"])
    restore_process_rng_state(state["process_rng"])


def restore_worker_simulator_only_for_audit(
    env: Any,
    state: Mapping[str, Any],
) -> None:
    """Restore only MuJoCo state for the preregistered negative control."""

    if state.get("schema") != CAUSAL_SNAPSHOT_SCHEMA:
        raise ValueError(f"Unsupported causal snapshot schema {state.get('schema')!r}.")
    sim = _find_sim(env)
    sim_state = state["sim"]
    sim.set_state_from_flattened(np.asarray(sim_state["flattened"]))
    sim.data.mocap_pos[...] = sim_state["mocap_pos"]
    sim.data.mocap_quat[...] = sim_state["mocap_quat"]
    sim.forward()
    _restore_fields(sim.data, sim_state["dynamic"])
    if not (
        float(sim.data.time) == float(sim_state["time"])
        and np.array_equal(sim.data.qpos, sim_state["qpos"])
        and np.array_equal(sim.data.qvel, sim_state["qvel"])
        and np.array_equal(sim.data.act, sim_state["act"])
    ):
        raise RuntimeError("MuJoCo-only audit state did not restore exactly.")


@dataclass(frozen=True)
class CausalSnapshotV1:
    """Complete one-environment snapshot crossing worker and policy boundaries."""

    snapshot_id: str
    worker_state: Mapping[str, Any]
    wrapper_state: Mapping[str, Any]
    current_raw_observation: Mapping[str, Any]
    recent_history: tuple[Mapping[str, Any], ...]
    policy_runtime_state: Mapping[str, Any]
    driver_rng_state: Mapping[str, Any]
    source_policy: str
    previous_mode: str | None
    chunk_index: int
    remaining_budget: float
    schema: str = CAUSAL_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("Causal snapshot identity must be non-empty.")
        if len(self.recent_history) > 4:
            raise ValueError("Causal snapshot history is limited to four chunks.")
        if self.chunk_index < 0 or self.remaining_budget < 0:
            raise ValueError("Chunk index and remaining budget must be non-negative.")
        if self.worker_state.get("schema") != self.schema:
            raise ValueError("Worker and outer causal snapshot schemas disagree.")


@dataclass(frozen=True)
class CausalSnapshotV2:
    """V2 scientific identity wrapped around the exact v1 runtime snapshot."""

    runtime_snapshot: CausalSnapshotV1
    identity: CausalStateIdentityV2
    sampling: CausalSamplingMetadataV2
    source_route: str
    previous_mode: str | None
    remaining_budget: float
    predicate_before: tuple[bool, ...]
    source_trace_summary: Mapping[str, Any]
    parent_checkpoint_identity: str
    statistics_identity: str
    schema: str = CAUSAL_SNAPSHOT_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.runtime_snapshot.snapshot_id != self.identity.snapshot_id:
            raise ValueError("V2 scientific and runtime snapshot identities differ.")
        if self.runtime_snapshot.source_policy != self.sampling.source_policy:
            raise ValueError("V2 snapshot source-policy fields disagree.")
        if self.runtime_snapshot.chunk_index != self.identity.chunk_index:
            raise ValueError("V2 snapshot chunk identities disagree.")
        if not self.source_route or not self.parent_checkpoint_identity:
            raise ValueError("V2 snapshot provenance fields must be non-empty.")
        if not self.statistics_identity or self.remaining_budget < 0:
            raise ValueError("V2 snapshot statistics/budget fields are invalid.")

    @property
    def snapshot_id(self) -> str:
        """Expose the canonical v2 state identity."""

        return self.identity.snapshot_id


def _assert_exact_nested(left: Any, right: Any, *, path: str) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left, right)
        ):
            raise AssertionError(f"Exact snapshot audit mismatch at {path}.")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right)
        ):
            raise AssertionError(f"Exact snapshot audit mismatch at {path}.")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise AssertionError(f"Exact snapshot audit keys differ at {path}.")
        for key in left:
            _assert_exact_nested(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"Exact snapshot audit sequence differs at {path}.")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_exact_nested(
                left_item,
                right_item,
                path=f"{path}[{index}]",
            )
        return
    if type(left) is not type(right) or left != right:
        raise AssertionError(f"Exact snapshot audit value differs at {path}.")


def assert_exact_causal_replay(left: Any, right: Any, *, path: str) -> None:
    """Require recursive exact equality for source replay or audit evidence."""

    _assert_exact_nested(left, right, path=path)


def _validate_audit_trace(trace: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    if not isinstance(trace, Mapping):
        raise TypeError(f"Snapshot audit trace {path} must be a mapping.")
    missing = sorted(set(CAUSAL_SNAPSHOT_AUDIT_REQUIRED_FIELDS) - set(trace))
    if missing:
        raise ValueError(f"Snapshot audit trace {path} is missing fields: {missing}.")
    return dict(trace)


def audit_interleaved_snapshot_restore(
    *,
    restore: Callable[[], None],
    restore_simulator_only: Callable[[], None],
    run_branch: Callable[[str], Mapping[str, Any]],
    mode_a: str,
    mode_b: str,
    phase: str,
) -> dict[str, Any]:
    """Run and compare the required A-restore-A and A-B-restore-A orders.

    ``run_branch`` must return raw observation, submitted actions, next
    simulator state, reward, success, metrics, and fixed-continuation outcome.
    The function compares the entire mapping recursively with exact equality
    and requires the MuJoCo-only negative control to diverge.
    """

    if phase not in CAUSAL_SNAPSHOT_AUDIT_PHASES:
        raise ValueError(f"Unknown snapshot audit phase {phase!r}.")
    if not mode_a or not mode_b or mode_a == mode_b:
        raise ValueError("Snapshot audit modes A and B must be distinct.")

    restore()
    first_a = _validate_audit_trace(run_branch(mode_a), path="first_a")
    restore()
    restored_a = _validate_audit_trace(run_branch(mode_a), path="restored_a")
    _assert_exact_nested(first_a, restored_a, path="a_restore_a")

    restore()
    baseline_a = _validate_audit_trace(run_branch(mode_a), path="baseline_a")
    _validate_audit_trace(run_branch(mode_b), path="interleaved_b")
    restore_simulator_only()
    simulator_only_a = _validate_audit_trace(
        run_branch(mode_a),
        path="simulator_only_a",
    )
    try:
        _assert_exact_nested(
            baseline_a,
            simulator_only_a,
            path="simulator_only_negative_control",
        )
    except AssertionError:
        simulator_only_status = "EXPECTED-MISMATCH"
    else:
        raise AssertionError(
            "MuJoCo-only restore unexpectedly reproduced the complete branch trace."
        )

    restore()
    post_interleave_a = _validate_audit_trace(
        run_branch(mode_a),
        path="post_interleave_a",
    )
    _assert_exact_nested(baseline_a, post_interleave_a, path="a_b_restore_a")
    return {
        "schema": CAUSAL_SNAPSHOT_AUDIT_SCHEMA,
        "status": "PASS",
        "scientific_results": "NOT-RUN",
        "phase": phase,
        "mode_a": str(mode_a),
        "mode_b": str(mode_b),
        "orders": list(CAUSAL_SNAPSHOT_AUDIT_ORDERS),
        "simulator_only_negative_control": simulator_only_status,
        "exact_fields": sorted(first_a),
    }
