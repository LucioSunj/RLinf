# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed, mid-episode LIBERO snapshots for paired gate experiments.

Reset states are not sufficient for a counterfactual branch: robosuite
controllers keep targets outside MuJoCo and both the worker and environment own
random-number generators.  This module deliberately uses a small, explicit
protocol and refuses to collect data when the installed LIBERO wrapper cannot
provide every required component.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
from collections.abc import Mapping
from numbers import Number
from typing import Any

import numpy as np
import torch


WORKER_SNAPSHOT_SCHEMA = "libero-worker-snapshot-v1"

_CONTROLLER_FIELD_HINTS = (
    "goal",
    "target",
    "reference",
    "ref_",
    "qpos",
    "qvel",
    "joint_pos",
    "joint_vel",
    "ee_pos",
    "ee_ori",
    "gripper",
)

_RUNTIME_FIELD_HINTS = (
    "timestep",
    "time_step",
    "elapsed",
    "step_count",
    "episode_step",
    "cur_time",
    "success",
    "stage",
    "phase",
    "counter",
    "done",
)


def _object_chain(env: Any) -> list[Any]:
    """Return wrapper/core objects without following arbitrary object graphs."""
    result: list[Any] = []
    queue = [env]
    while queue:
        obj = queue.pop(0)
        if obj is None or any(obj is existing for existing in result):
            continue
        result.append(obj)
        for name in ("env", "unwrapped"):
            try:
                child = getattr(obj, name, None)
            except Exception:
                child = None
            if child is not None and child is not obj:
                queue.append(child)
    return result


def _find_attr(env: Any, name: str) -> Any:
    for obj in _object_chain(env):
        try:
            value = getattr(obj, name, None)
        except Exception:
            continue
        if value is not None:
            return value
    return None


def _copy_numeric(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Number, bool, str)) or value is None:
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        copied = [_copy_numeric(item) for item in value]
        return tuple(copied) if all(item is not _UNSUPPORTED for item in copied) else _UNSUPPORTED
    if isinstance(value, list):
        copied = [_copy_numeric(item) for item in value]
        return copied if all(item is not _UNSUPPORTED for item in copied) else _UNSUPPORTED
    if isinstance(value, Mapping):
        copied = {str(key): _copy_numeric(item) for key, item in value.items()}
        return copied if all(item is not _UNSUPPORTED for item in copied.values()) else _UNSUPPORTED
    return _UNSUPPORTED


_UNSUPPORTED = object()


def _restore_field(obj: Any, name: str, value: Any) -> None:
    current = getattr(obj, name)
    if (
        isinstance(current, np.ndarray)
        and isinstance(value, np.ndarray)
        and current.shape == value.shape
        and current.dtype == value.dtype
    ):
        np.copyto(current, value)
    else:
        setattr(obj, name, copy.deepcopy(value))


def _controllers(env: Any) -> list[Any]:
    controllers: list[Any] = []
    for obj in _object_chain(env):
        robots = getattr(obj, "robots", None)
        if robots is None:
            continue
        for robot in robots:
            controller = getattr(robot, "controller", None)
            if controller is not None and not any(
                controller is existing for existing in controllers
            ):
                controllers.append(controller)
    return controllers


def _capture_controller_state(env: Any) -> list[dict[str, Any]]:
    controllers = _controllers(env)
    if not controllers:
        raise RuntimeError(
            "LIBERO snapshot requires robosuite robot controllers, but none were "
            "reachable through env(.env/.unwrapped).robots."
        )
    states = []
    for index, controller in enumerate(controllers):
        fields: dict[str, Any] = {}
        for name, value in vars(controller).items():
            lower = name.lower()
            if not any(hint in lower for hint in _CONTROLLER_FIELD_HINTS):
                continue
            copied = _copy_numeric(value)
            if copied is not _UNSUPPORTED:
                fields[name] = copied
        if not fields:
            raise RuntimeError(
                f"controller {index} ({type(controller).__name__}) exposes no "
                "numeric goal/target/joint state; snapshot would be incomplete."
            )
        states.append(
            {
                "index": index,
                "class": f"{type(controller).__module__}.{type(controller).__qualname__}",
                "fields": fields,
            }
        )
    return states


def _restore_controller_state(env: Any, states: list[dict[str, Any]]) -> None:
    controllers = _controllers(env)
    if len(controllers) != len(states):
        raise RuntimeError(
            "controller count changed across snapshot restore: "
            f"{len(states)} -> {len(controllers)}"
        )
    for controller, state in zip(controllers, states):
        actual_class = f"{type(controller).__module__}.{type(controller).__qualname__}"
        if actual_class != state["class"]:
            raise RuntimeError(
                "controller class changed across restore: "
                f"{state['class']!r} -> {actual_class!r}"
            )
        for name, value in state["fields"].items():
            if not hasattr(controller, name):
                raise RuntimeError(f"controller field {name!r} disappeared before restore")
            _restore_field(controller, name, value)


def _capture_runtime_state(env: Any) -> list[dict[str, Any]]:
    """Capture mutable episode clocks/counters outside MuJoCo state."""
    states = []
    for index, obj in enumerate(_object_chain(env)):
        fields = {}
        for name, value in vars(obj).items():
            lower = name.lower()
            if not any(hint in lower for hint in _RUNTIME_FIELD_HINTS):
                continue
            copied = _copy_numeric(value)
            if copied is not _UNSUPPORTED:
                fields[name] = copied
        if fields:
            states.append(
                {
                    "object_index": index,
                    "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
                    "fields": fields,
                }
            )
    if not states:
        raise RuntimeError(
            "LIBERO snapshot found no mutable timestep/stage/success counters "
            "outside MuJoCo; the worker wrapper contract is unsupported"
        )
    return states


def _restore_runtime_state(env: Any, states: list[dict[str, Any]]) -> None:
    chain = _object_chain(env)
    for state in states:
        index = int(state["object_index"])
        if index >= len(chain):
            raise RuntimeError("environment wrapper chain changed before runtime restore")
        obj = chain[index]
        actual_class = f"{type(obj).__module__}.{type(obj).__qualname__}"
        if actual_class != state["class"]:
            raise RuntimeError(
                "environment wrapper class changed across restore: "
                f"{state['class']!r} -> {actual_class!r}"
            )
        for name, value in state["fields"].items():
            if not hasattr(obj, name):
                raise RuntimeError(f"runtime field {name!r} disappeared before restore")
            _restore_field(obj, name, value)


def capture_process_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy and initialized Torch CPU/CUDA streams."""
    cuda_initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda_initialized": cuda_initialized,
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if cuda_initialized
            else []
        ),
    }


def restore_process_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.random.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8))
    if bool(state.get("torch_cuda_initialized", False)):
        if not torch.cuda.is_available():
            raise RuntimeError("snapshot contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(value, dtype=torch.uint8) for value in state["torch_cuda"]]
        )


def _capture_rng_state(env: Any) -> dict[str, Any]:
    object_states = []
    for index, obj in enumerate(_object_chain(env)):
        fields = dict(vars(obj))
        # Gym wrappers occasionally expose ``np_random`` through a property rather
        # than ``__dict__``. Include it without assuming it is the only simulator
        # RNG stream.
        if "np_random" not in fields:
            try:
                fields["np_random"] = getattr(obj, "np_random", None)
            except Exception:
                pass
        for name, rng in fields.items():
            item = {"object_index": index, "field": str(name)}
            if isinstance(rng, np.random.Generator):
                item.update(
                    kind="numpy_generator",
                    state=copy.deepcopy(rng.bit_generator.state),
                )
            elif isinstance(rng, np.random.RandomState):
                item.update(
                    kind="numpy_randomstate",
                    state=copy.deepcopy(rng.get_state()),
                )
            elif isinstance(rng, random.Random):
                item.update(kind="python_random", state=copy.deepcopy(rng.getstate()))
            elif isinstance(rng, torch.Generator):
                item.update(
                    kind="torch_generator",
                    state=rng.get_state().clone(),
                    device=str(rng.device),
                )
            else:
                continue
            object_states.append(item)
    return {**capture_process_rng_state(), "objects": object_states}


def _restore_rng_state(env: Any, state: Mapping[str, Any]) -> None:
    restore_process_rng_state(state)
    chain = _object_chain(env)
    for item in state["objects"]:
        index = int(item["object_index"])
        if index >= len(chain):
            raise RuntimeError("environment wrapper chain changed before RNG restore")
        field = str(item["field"])
        rng = getattr(chain[index], field, None)
        if item["kind"] == "numpy_generator" and isinstance(
            rng, np.random.Generator
        ):
            rng.bit_generator.state = copy.deepcopy(item["state"])
        elif item["kind"] == "numpy_randomstate" and isinstance(
            rng, np.random.RandomState
        ):
            rng.set_state(copy.deepcopy(item["state"]))
        elif item["kind"] == "python_random" and isinstance(rng, random.Random):
            rng.setstate(copy.deepcopy(item["state"]))
        elif item["kind"] == "torch_generator" and isinstance(
            rng, torch.Generator
        ):
            if str(rng.device) != str(item["device"]):
                raise RuntimeError(
                    f"environment Torch RNG device changed at wrapper index {index}"
                )
            rng.set_state(torch.as_tensor(item["state"], dtype=torch.uint8))
        else:
            raise RuntimeError(
                f"environment RNG {field!r} changed at wrapper index {index}"
            )


def _bddl_identity(env: Any) -> dict[str, Any]:
    path = None
    for obj in _object_chain(env):
        for name in ("bddl_file_name", "bddl_file", "_bddl_file_name"):
            candidate = getattr(obj, name, None)
            if isinstance(candidate, (str, os.PathLike)) and str(candidate):
                path = os.path.realpath(os.fspath(candidate))
                break
        if path is not None:
            break
    if path is None or not os.path.isfile(path):
        raise RuntimeError(
            "LIBERO snapshot cannot identify a readable BDDL file; paired data "
            "must be bound to an immutable task definition."
        )
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"bddl_path": path, "bddl_sha256": digest.hexdigest()}


def _get_observation(env: Any) -> dict[str, Any]:
    for obj in _object_chain(env):
        getter = getattr(obj, "_get_observations", None)
        if callable(getter):
            obs = getter()
            if isinstance(obs, Mapping):
                return copy.deepcopy(dict(obs))
    raise RuntimeError(
        "LIBERO snapshot requires `_get_observations()` so restored RGB and "
        "proprio can be verified before branching."
    )


def _numeric_observation(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
            result[str(key)] = value.copy()
        elif isinstance(value, Number):
            result[str(key)] = np.asarray(value)
    if not result:
        raise RuntimeError("LIBERO observation contains no numeric values to verify")
    if not any("image" in key.lower() or "rgb" in key.lower() for key in result):
        raise RuntimeError("LIBERO observation contains no RGB/image field to verify")
    return result


def _verify_observation(
    expected: Mapping[str, np.ndarray],
    actual_obs: Mapping[str, Any],
    *,
    proprio_atol: float,
) -> None:
    actual = _numeric_observation(actual_obs)
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise RuntimeError(f"restored observation is missing fields {missing}")
    for key, before in expected.items():
        after = actual[key]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise RuntimeError(
                f"restored observation field {key!r} changed shape/dtype: "
                f"{before.shape}/{before.dtype} -> {after.shape}/{after.dtype}"
            )
        is_pixel = "image" in key.lower() or "rgb" in key.lower()
        equal = (
            np.array_equal(before, after)
            if is_pixel
            else np.allclose(before, after, rtol=0.0, atol=float(proprio_atol))
        )
        if not bool(equal):
            max_error = float(np.max(np.abs(before.astype(float) - after.astype(float))))
            kind = "pixel" if is_pixel else "proprio/state"
            raise RuntimeError(
                f"snapshot restore {kind} verification failed for {key!r}; "
                f"max_abs_error={max_error:g}"
            )


def capture_worker_snapshot(env: Any) -> dict[str, Any]:
    """Capture one OffScreenRenderEnv at an action-chunk boundary."""
    getter = getattr(env, "get_sim_state", None)
    if not callable(getter):
        raise RuntimeError("LIBERO environment has no get_sim_state()")
    sim_state = copy.deepcopy(getter())
    if sim_state is None:
        raise RuntimeError("LIBERO get_sim_state() returned None")
    controllers = _capture_controller_state(env)
    runtime = _capture_runtime_state(env)
    rng = _capture_rng_state(env)
    try:
        verification_obs = _numeric_observation(_get_observation(env))
    finally:
        # Rendering an observation may update wrapper counters, controller caches,
        # or random streams. Snapshot capture itself must be observational: the
        # live reference rollout and a restored branch must start from one state.
        _restore_sim_state(env, sim_state)
        _restore_controller_state(env, controllers)
        _restore_runtime_state(env, runtime)
        _restore_rng_state(env, rng)
    return {
        "schema": WORKER_SNAPSHOT_SCHEMA,
        "identity": _bddl_identity(env),
        "sim_state": sim_state,
        "controllers": controllers,
        "runtime": runtime,
        "rng": rng,
        "verification_obs": verification_obs,
    }


def _restore_sim_state(env: Any, sim_state: Any) -> None:
    sim = _find_attr(env, "sim")
    setter = getattr(sim, "set_state_from_flattened", None)
    if callable(setter):
        setter(copy.deepcopy(sim_state))
        forward = getattr(sim, "forward", None)
        if callable(forward):
            forward()
        return
    setter = getattr(env, "set_init_state", None)
    if callable(setter):
        setter(copy.deepcopy(sim_state))
        return
    raise RuntimeError(
        "LIBERO environment exposes neither sim.set_state_from_flattened() nor "
        "set_init_state(); mid-episode restore is unavailable."
    )


def restore_worker_snapshot(
    env: Any,
    snapshot: Mapping[str, Any],
    *,
    proprio_atol: float = 1.0e-6,
) -> dict[str, Any]:
    """Restore and verify a worker snapshot, returning the rerendered raw obs."""
    if snapshot.get("schema") != WORKER_SNAPSHOT_SCHEMA:
        raise ValueError(
            f"unsupported LIBERO worker snapshot schema {snapshot.get('schema')!r}"
        )
    if _bddl_identity(env) != snapshot.get("identity"):
        raise RuntimeError("task/BDDL identity changed before snapshot restore")
    _restore_sim_state(env, snapshot["sim_state"])
    _restore_controller_state(env, list(snapshot["controllers"]))
    _restore_runtime_state(env, list(snapshot["runtime"]))
    actual_obs = None
    try:
        actual_obs = _get_observation(env)
        _verify_observation(
            snapshot["verification_obs"], actual_obs, proprio_atol=proprio_atol
        )
    finally:
        # Rerendering is not guaranteed to be observational in robosuite.  Put
        # every captured component back after verification so the first branch
        # action sees the exact snapshot, including controller targets/caches.
        _restore_sim_state(env, snapshot["sim_state"])
        _restore_controller_state(env, list(snapshot["controllers"]))
        _restore_runtime_state(env, list(snapshot["runtime"]))
        _restore_rng_state(env, snapshot["rng"])
    assert actual_obs is not None
    return actual_obs
