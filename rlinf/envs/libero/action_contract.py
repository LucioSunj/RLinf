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

"""Exact live LIBERO / robosuite Action contract inspection."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np

LIBERO_ACTION_CONTRACT_SCHEMA = "fastwam-libero-action-contract-v1"
_OSC_POSE_DIMENSION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
)


def _qualified_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _float_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite.")
    return tuple(float(item) for item in array)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dependency_versions() -> dict[str, str]:
    result = {"numpy_version": np.__version__}
    for module_name in ("libero", "robosuite"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        module_path = getattr(module, "__file__", None)
        if version is not None:
            result[f"{module_name}_version"] = str(version)
        if module_path is not None:
            result[f"{module_name}_module_path"] = str(module_path)
    return result


@dataclass(frozen=True, slots=True)
class LiberoActionContract:
    """Canonical exact Action contract obtained from a live LIBERO instance."""

    low: tuple[float, ...]
    high: tuple[float, ...]
    dimension_names: tuple[str, ...]
    gripper_dimension_index: int
    outer_environment_classes: tuple[str, ...]
    underlying_environment_classes: tuple[str, ...]
    robot_class: str
    robot_model: str
    controller_class: str
    controller_name: str
    controller_input_low: tuple[float, ...]
    controller_input_high: tuple[float, ...]
    controller_output_low: tuple[float, ...]
    controller_output_high: tuple[float, ...]
    gripper_class: str
    gripper_dof: int
    gripper_speed: float
    control_frequency_hz: int
    environment_horizon: int
    dependency_versions: tuple[tuple[str, str], ...]
    source: str = "underlying_env.action_spec"

    def __post_init__(self) -> None:
        action_dim = len(self.low)
        if action_dim < 1 or len(self.high) != action_dim:
            raise ValueError("LIBERO Action low/high dimensions differ.")
        if len(self.dimension_names) != action_dim:
            raise ValueError("LIBERO Action dimension names do not match bounds.")
        if len(set(self.dimension_names)) != action_dim:
            raise ValueError("LIBERO Action dimension names must be unique.")
        if not 0 <= self.gripper_dimension_index < action_dim:
            raise ValueError("LIBERO gripper index is outside the Action dimension.")
        if any(
            not math.isfinite(low) or not math.isfinite(high) or low >= high
            for low, high in zip(self.low, self.high)
        ):
            raise ValueError("LIBERO Action bounds must be finite and ordered.")
        if (
            not self.outer_environment_classes
            or not self.underlying_environment_classes
        ):
            raise ValueError("LIBERO environment class provenance is required.")
        if len(self.controller_input_low) != len(self.controller_input_high):
            raise ValueError("Controller input bound dimensions differ.")
        if len(self.controller_output_low) != len(self.controller_output_high):
            raise ValueError("Controller output bound dimensions differ.")
        if self.gripper_dof < 1 or self.control_frequency_hz < 1:
            raise ValueError("LIBERO gripper/control metadata is invalid.")
        if self.environment_horizon < 1 or not math.isfinite(self.gripper_speed):
            raise ValueError("LIBERO horizon/gripper metadata is invalid.")
        if tuple(sorted(self.dependency_versions)) != self.dependency_versions:
            raise ValueError("Dependency version entries must be sorted.")

    @property
    def action_dim(self) -> int:
        """Return the exact environment Action dimension."""

        return len(self.low)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": LIBERO_ACTION_CONTRACT_SCHEMA,
            "source": self.source,
            "action_dim": self.action_dim,
            "dimension_names": list(self.dimension_names),
            "gripper_dimension_index": self.gripper_dimension_index,
            "low": list(self.low),
            "high": list(self.high),
            "environment": {
                "outer_classes": list(self.outer_environment_classes),
                "underlying_classes": list(self.underlying_environment_classes),
                "control_frequency_hz": self.control_frequency_hz,
                "horizon": self.environment_horizon,
            },
            "robot": {
                "class": self.robot_class,
                "model": self.robot_model,
                "action_low": list(self.low),
                "action_high": list(self.high),
            },
            "controller": {
                "class": self.controller_class,
                "name": self.controller_name,
                "input_low": list(self.controller_input_low),
                "input_high": list(self.controller_input_high),
                "output_low": list(self.controller_output_low),
                "output_high": list(self.controller_output_high),
                "input_clipping_source": (
                    "robosuite.controllers.base_controller.Controller.scale_action"
                ),
            },
            "gripper": {
                "class": self.gripper_class,
                "dof": self.gripper_dof,
                "speed": self.gripper_speed,
                "sign_and_saturation_source": (
                    "robosuite.models.grippers.panda_gripper.PandaGripper.format_action"
                ),
            },
            "dependency_versions": dict(self.dependency_versions),
        }

    @property
    def canonical_sha256(self) -> str:
        """Return the hash of the contract excluding its self-declared hash."""

        return _canonical_sha256(self._payload())

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-safe contract with a self-verifying hash."""

        payload = self._payload()
        payload["canonical_sha256"] = self.canonical_sha256
        return payload

    @classmethod
    def from_artifact(cls, payload: Mapping[str, Any]) -> "LiberoActionContract":
        """Parse and verify a serialized live contract."""

        if payload.get("schema") != LIBERO_ACTION_CONTRACT_SCHEMA:
            raise ValueError("Unsupported LIBERO Action contract schema.")
        environment = payload.get("environment")
        robot = payload.get("robot")
        controller = payload.get("controller")
        gripper = payload.get("gripper")
        dependencies = payload.get("dependency_versions")
        if not all(
            isinstance(item, Mapping)
            for item in (environment, robot, controller, gripper, dependencies)
        ):
            raise TypeError("LIBERO Action contract provenance sections are required.")
        contract = cls(
            low=_float_tuple(payload.get("low"), name="Action low"),
            high=_float_tuple(payload.get("high"), name="Action high"),
            dimension_names=tuple(
                str(item) for item in payload.get("dimension_names", ())
            ),
            gripper_dimension_index=int(payload.get("gripper_dimension_index", -1)),
            outer_environment_classes=tuple(
                str(item) for item in environment.get("outer_classes", ())
            ),
            underlying_environment_classes=tuple(
                str(item) for item in environment.get("underlying_classes", ())
            ),
            robot_class=str(robot.get("class", "")),
            robot_model=str(robot.get("model", "")),
            controller_class=str(controller.get("class", "")),
            controller_name=str(controller.get("name", "")),
            controller_input_low=_float_tuple(
                controller.get("input_low"), name="controller input low"
            ),
            controller_input_high=_float_tuple(
                controller.get("input_high"), name="controller input high"
            ),
            controller_output_low=_float_tuple(
                controller.get("output_low"), name="controller output low"
            ),
            controller_output_high=_float_tuple(
                controller.get("output_high"), name="controller output high"
            ),
            gripper_class=str(gripper.get("class", "")),
            gripper_dof=int(gripper.get("dof", -1)),
            gripper_speed=float(gripper.get("speed", float("nan"))),
            control_frequency_hz=int(environment.get("control_frequency_hz", 0)),
            environment_horizon=int(environment.get("horizon", 0)),
            dependency_versions=tuple(
                sorted((str(key), str(value)) for key, value in dependencies.items())
            ),
            source=str(payload.get("source", "")),
        )
        if int(payload.get("action_dim", -1)) != contract.action_dim:
            raise ValueError("Serialized LIBERO Action dimension does not reconcile.")
        if (
            tuple(robot.get("action_low", ())) != contract.low
            or tuple(robot.get("action_high", ())) != contract.high
        ):
            raise ValueError("Serialized robot Action limits do not reconcile.")
        if payload.get("canonical_sha256") != contract.canonical_sha256:
            raise ValueError("LIBERO Action contract hash mismatch.")
        return contract


def inspect_libero_action_contract(
    env: Any,
    *,
    dependency_versions: Mapping[str, Any] | None = None,
) -> LiberoActionContract:
    """Inspect the underlying robosuite environment and fail closed on mismatch."""

    underlying = getattr(env, "env", None)
    if underlying is None:
        raise AttributeError(
            "LIBERO wrapper does not expose its underlying environment."
        )
    if not hasattr(underlying, "action_spec"):
        raise AttributeError("Underlying LIBERO environment has no action_spec.")
    low_raw, high_raw = underlying.action_spec
    low = _float_tuple(low_raw, name="environment Action low")
    high = _float_tuple(high_raw, name="environment Action high")
    action_dim = int(getattr(underlying, "action_dim", len(low)))
    if action_dim != len(low) or len(high) != action_dim:
        raise ValueError(
            "Underlying LIBERO Action dimension does not match action_spec."
        )

    robots = list(getattr(underlying, "robots", ()))
    if len(robots) != 1:
        raise ValueError("FastWAM LIBERO requires exactly one robot.")
    robot = robots[0]
    robot_low_raw, robot_high_raw = robot.action_limits
    robot_low = _float_tuple(robot_low_raw, name="robot Action low")
    robot_high = _float_tuple(robot_high_raw, name="robot Action high")
    if robot_low != low or robot_high != high:
        raise ValueError(
            "LIBERO robot action limits disagree with environment action_spec."
        )

    controller = getattr(robot, "controller", None)
    gripper = getattr(robot, "gripper", None)
    if controller is None or gripper is None:
        raise ValueError("LIBERO robot must expose controller and gripper.")
    controller_low = _float_tuple(controller.input_min, name="controller input low")
    controller_high = _float_tuple(controller.input_max, name="controller input high")
    controller_dim = int(getattr(controller, "control_dim", len(controller_low)))
    if (
        controller_dim != len(controller_low)
        or len(controller_high) != controller_dim
        or low[:controller_dim] != controller_low
        or high[:controller_dim] != controller_high
    ):
        raise ValueError("Controller input limits disagree with robot action limits.")
    gripper_dof = int(getattr(gripper, "dof", 0))
    if controller_dim + gripper_dof != action_dim or gripper_dof != 1:
        raise ValueError(
            "FastWAM LIBERO requires six controller and one gripper action."
        )

    controller_name = str(getattr(controller, "name", ""))
    if controller_name != "OSC_POSE" or controller_dim != 6:
        raise ValueError(
            "FastWAM LIBERO requires the six-dimensional OSC_POSE controller."
        )
    dimension_names = (*_OSC_POSE_DIMENSION_NAMES, "gripper")
    versions = (
        _dependency_versions()
        if dependency_versions is None
        else {str(key): str(value) for key, value in dependency_versions.items()}
    )
    robot_model = getattr(robot, "robot_model", None)
    return LiberoActionContract(
        low=low,
        high=high,
        dimension_names=dimension_names,
        gripper_dimension_index=action_dim - 1,
        outer_environment_classes=(_qualified_name(env),),
        underlying_environment_classes=(_qualified_name(underlying),),
        robot_class=_qualified_name(robot),
        robot_model=type(robot_model).__name__,
        controller_class=_qualified_name(controller),
        controller_name=controller_name,
        controller_input_low=controller_low,
        controller_input_high=controller_high,
        controller_output_low=_float_tuple(
            controller.output_min, name="controller output low"
        ),
        controller_output_high=_float_tuple(
            controller.output_max, name="controller output high"
        ),
        gripper_class=_qualified_name(gripper),
        gripper_dof=gripper_dof,
        gripper_speed=float(getattr(gripper, "speed", float("nan"))),
        control_frequency_hz=int(getattr(underlying, "control_freq", 0)),
        environment_horizon=int(getattr(underlying, "horizon", 0)),
        dependency_versions=tuple(sorted(versions.items())),
    )


def merge_libero_action_contracts(
    contracts: Iterable[LiberoActionContract | Mapping[str, Any]],
) -> LiberoActionContract:
    """Merge task-specific environment class provenance for one shared spec."""

    parsed = tuple(
        item
        if isinstance(item, LiberoActionContract)
        else LiberoActionContract.from_artifact(item)
        for item in contracts
    )
    if not parsed:
        raise ValueError("No live LIBERO Action contracts were provided.")
    first = parsed[0]
    comparable = first._payload()
    comparable["environment"]["outer_classes"] = []
    comparable["environment"]["underlying_classes"] = []
    for item in parsed[1:]:
        candidate = item._payload()
        candidate["environment"]["outer_classes"] = []
        candidate["environment"]["underlying_classes"] = []
        if candidate != comparable:
            raise ValueError(
                "Vector LIBERO environments expose different Action contracts."
            )
    return replace(
        first,
        outer_environment_classes=tuple(
            sorted({name for item in parsed for name in item.outer_environment_classes})
        ),
        underlying_environment_classes=tuple(
            sorted(
                {
                    name
                    for item in parsed
                    for name in item.underlying_environment_classes
                }
            )
        ),
    )
