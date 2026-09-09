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

"""Lazy realworld exports; imports do not connect devices or manage processes."""

from importlib import import_module

_EXPORTS = {
    "RealWorldEnv": ("realworld_env", "RealWorldEnv"),
    "FrankaEnv": ("franka", "FrankaEnv"),
    "FrankaRobotConfig": ("franka", "FrankaRobotConfig"),
    "FrankaRobotState": ("franka", "FrankaRobotState"),
    "DualFrankaEnv": ("franka.dual_franka_env", "DualFrankaEnv"),
    "DualFrankaRobotConfig": ("franka.dual_franka_env", "DualFrankaRobotConfig"),
    "DualFrankaJointEnv": ("franka.tasks.dual_franka_joint_env", "DualFrankaJointEnv"),
    "DualFrankaJointRobotConfig": (
        "franka.tasks.dual_franka_joint_env",
        "DualFrankaJointRobotConfig",
    ),
    "DualFrankaTCPEnv": ("franka.tasks.dual_franka_tcp_env", "DualFrankaTCPEnv"),
    "DualFrankaTCPRobotConfig": (
        "franka.tasks.dual_franka_tcp_env",
        "DualFrankaTCPRobotConfig",
    ),
    "DOSW1Config": ("dosw1", "DOSW1Config"),
    "DOSW1Env": ("dosw1", "DOSW1Env"),
    "GimArmEnv": ("gim_arm", "GimArmEnv"),
    "GimArmRobotConfig": ("gim_arm", "GimArmRobotConfig"),
    "GimArmRobotState": ("gim_arm", "GimArmRobotState"),
    "Turtle2Env": ("xsquare", "Turtle2Env"),
    "Turtle2RobotConfig": ("xsquare", "Turtle2RobotConfig"),
    "Turtle2RobotState": ("xsquare", "Turtle2RobotState"),
    "franka_tasks": ("franka.tasks", None),
    "dosw1_tasks": ("dosw1.tasks", None),
    "gim_arm_tasks": ("gim_arm.tasks", None),
    "xsquare_tasks": ("xsquare.tasks", None),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    loaded = import_module(f"{__name__}.{module}")
    result = loaded if symbol is None else getattr(loaded, symbol)
    globals()[name] = result
    return result


def register_realworld_tasks() -> None:
    """Register existing Gym tasks at explicit environment construction time."""
    for name in ("franka.tasks", "dosw1.tasks", "gim_arm.tasks", "xsquare.tasks"):
        import_module(f"{__name__}.{name}")


__all__ = list(_EXPORTS) + ["register_realworld_tasks"]
