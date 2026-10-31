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

"""Manual-reset single-robot proxy for the shared Polymetis chunk executor."""

import math
import uuid
from typing import Any

import numpy as np

TASK_IDS = {"drawer": 0, "insert_tubes": 1}


class Fr3PolymetisEnv:
    """Exchange chunks and execution evidence with the robot host over WebSocket RPC.

    One physical robot is one environment. Reset requires an explicit operator
    scene confirmation at the bridge or ``options['scene_ready']``. Each RPC is
    sent once; an uncertain response invalidates the episode until manual reset.
    """

    def __init__(self, cfg: Any, num_envs: int = 1, **_: Any) -> None:
        if num_envs != 1 or int(cfg.get("num_envs", 1)) != 1:
            raise ValueError("FR3 Polymetis requires num_envs=1")
        if cfg.get("auto_reset", False) or cfg.get("ignore_terminations", False):
            raise ValueError(
                "FR3 requires auto_reset=false and ignore_terminations=false"
            )
        self.cfg = cfg
        self.num_envs = 1
        self.task_name = cfg["task_name"]
        if self.task_name not in TASK_IDS:
            raise ValueError("task_name must be drawer or insert_tubes")
        self.task_id = TASK_IDS[self.task_name]
        self.execution_horizon = cfg.get("execution_horizon", 16)
        if self.execution_horizon not in (16, 32):
            raise ValueError("FR3 execution_horizon must be 16 or 32")
        self.url = cfg.get("bridge_url", "ws://127.0.0.1:8003")
        self.timeout = float(cfg.get("request_timeout_s", 60.0))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("request_timeout_s must be finite and positive")
        self.episode_id: str | None = None
        self.policy_version = 0
        self.chunk_id = 0
        self.closed = False
        self.needs_reset = True
        self.last_observation: dict[str, Any] | None = None
        self.last_info: dict[str, Any] = {}

    def _rpc(self, operation: str, **payload: Any) -> dict[str, Any]:
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        if self.closed:
            raise RuntimeError("FR3 environment is closed")
        with connect(
            self.url,
            compression=None,
            max_size=None,
            proxy=None,
            open_timeout=self.timeout,
            close_timeout=1,
        ) as connection:
            connection.send(msgpack_numpy.packb({"op": operation, **payload}))
            reply = msgpack_numpy.unpackb(connection.recv(timeout=self.timeout))
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "invalid bridge response"))
        return reply["result"]

    def status(self) -> dict[str, Any]:
        """Read episode readiness without taking device ownership."""
        result = self._rpc("status")
        if result["task_name"] != self.task_name or result["task_id"] != self.task_id:
            raise ValueError("bridge task differs from configured environment task")
        if result["execution_horizon"] != self.execution_horizon or not math.isclose(
            result["action_dt"], float(self.cfg.get("action_dt", 0.1)), abs_tol=1e-9
        ):
            raise ValueError("bridge horizon/action_dt differs from training contract")
        for key in ("max_episode_actions", "max_episode_seconds"):
            if self.cfg.get(key) is not None and result[key] != self.cfg[key]:
                raise ValueError(
                    f"bridge {key} differs from configured training budget"
                )
        return result

    def reset(
        self, *, options: dict[str, Any] | None = None, seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new physical episode after explicit manual scene readiness."""
        del seed  # Physical state is measured, never reconstructed from an RNG seed.
        options = options or {}
        self.status()
        episode_id = options.get("episode_id", uuid.uuid4().hex)
        version = options.get("policy_version", self.policy_version)
        result = self._rpc(
            "reset",
            episode_id=episode_id,
            policy_version=version,
            scene_ready=options.get("scene_ready", False),
        )
        self.episode_id, self.policy_version, self.chunk_id = episode_id, version, 0
        self.needs_reset = False
        self.last_observation = result.pop("observation")
        self.last_info = result
        return self.last_observation, result

    def observe(self) -> dict[str, Any]:
        """Read fresh raw RGB images and explicit seven-joint/one-gripper state."""
        result = self._rpc("observe")
        self.last_observation = result["observation"]
        return self.last_observation

    def step_chunk(
        self,
        actions: Any,
        *,
        policy_version: int | None = None,
        chunk_id: int | None = None,
        route: str | int | None = None,
        eligible: bool = True,
        forced: bool = False,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute at most one physical prefix, preserving the full execution mask."""
        if self.needs_reset or self.episode_id is None:
            raise RuntimeError("explicit manual reset is required before step_chunk")
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape == (1, self.execution_horizon, 8):
            actions = actions[0]
        if (
            actions.shape != (self.execution_horizon, 8)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                f"expected finite physical actions[{self.execution_horizon},8]"
            )
        version = self.policy_version if policy_version is None else policy_version
        current = self.chunk_id if chunk_id is None else chunk_id
        if version != self.policy_version or current != self.chunk_id:
            raise ValueError(
                "policy version/chunk identity differs from active episode"
            )
        try:
            result = self._rpc(
                "execute_chunk",
                episode_id=self.episode_id,
                chunk_id=current,
                policy_version=version,
                actions=actions,
                action_space="absolute_joint_position",
                route=route,
                eligible=eligible,
                forced=forced,
            )
        except Exception:
            self.needs_reset = True
            # A stop is a distinct control request, never a retry of physical actions.
            try:
                self._rpc(
                    "outcome", episode_id=self.episode_id, outcome="technical_interrupt"
                )
            except Exception:
                pass
            raise
        self.chunk_id += 1
        if (
            result["episode_id"] != self.episode_id
            or result["chunk_id"] != current
            or result["policy_version"] != version
        ):
            self.needs_reset = True
            raise RuntimeError(
                "bridge receipt identity mismatch; execution must not be retried"
            )
        self.last_observation = result.pop("observation")
        self.last_info = result
        self.needs_reset = bool(result["terminated"] or result["truncated"])
        return (
            self.last_observation,
            float(result["reward_env"]),
            bool(result["terminated"]),
            bool(result["truncated"]),
            result,
        )

    def step(self, actions: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Use chunk stepping; zero-filled wait actions are never sent."""
        return self.step_chunk(actions)

    def seal(self) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Seal an idle outcome and attach it to the last transition, with no action."""
        result = self._rpc("seal", episode_id=self.episode_id)
        self.last_observation = result.pop("observation")
        self.last_info = result
        self.needs_reset = True
        return (
            self.last_observation,
            float(result["reward_env"]),
            bool(result["terminated"]),
            bool(result["truncated"]),
            result,
        )

    def outcome(self, value: str) -> dict[str, Any]:
        """Report an explicit task result or intervention for the current episode."""
        result = self._rpc("outcome", episode_id=self.episode_id, outcome=value)
        self.needs_reset = True
        return result

    def stop(self) -> dict[str, Any]:
        """Interrupt the current prefix without releasing or retrying the gripper."""
        result = self._rpc("stop", episode_id=self.episode_id)
        self.needs_reset = True
        return result

    def close(self) -> None:
        """Stop this episode and close the proxy, leaving bridge cleanup to its owner."""
        if not self.closed:
            try:
                if self.episode_id is not None and not self.needs_reset:
                    self.stop()
            finally:
                self.closed = True
