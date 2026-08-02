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

"""Per-environment delayed Gate route state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import ChunkRouteRecord, WAMRoute


@dataclass
class _PendingDecision:
    route: WAMRoute
    source_chunk_id: int
    episode_id: int
    actor_version: int


@dataclass
class _EnvironmentRouteState:
    episode_id: int
    chunk_id: int
    pending: _PendingDecision | None = None
    force_next_idm: bool = False


class PendingRouteTracker:
    """Consume chunk routes now and store Gate decisions for the next chunk."""

    def __init__(self) -> None:
        self._states: dict[int, _EnvironmentRouteState] = {}
        self._next_episode_ids: dict[int, int] = {}

    def _start_episode(self, env_id: int) -> _EnvironmentRouteState:
        episode_id = self._next_episode_ids.get(env_id, 0)
        state = _EnvironmentRouteState(
            episode_id=episode_id,
            chunk_id=0,
        )
        self._next_episode_ids[env_id] = episode_id + 1
        self._states[env_id] = state
        return state

    def consume(
        self,
        *,
        env_ids: torch.Tensor,
        reset_mask: torch.Tensor,
        actor_version: int,
    ) -> ChunkRouteRecord:
        """Return the routes for this chunk, forcing IDM on every new episode."""

        if env_ids.ndim != 1:
            raise ValueError("`env_ids` must be one-dimensional.")
        if reset_mask.shape != env_ids.shape or reset_mask.dtype != torch.bool:
            raise ValueError("`reset_mask` must be bool and match `env_ids`.")
        if len({int(item) for item in env_ids.tolist()}) != env_ids.numel():
            raise ValueError("`env_ids` must be unique within a policy batch.")
        if actor_version < 0:
            raise ValueError("`actor_version` must be non-negative.")

        routes = []
        forced = []
        chunk_ids = []
        episode_ids = []
        source_ids = []
        versions = []
        for env_id_tensor, reset_tensor in zip(env_ids, reset_mask):
            env_id = int(env_id_tensor)
            reset = bool(reset_tensor)
            state = self._states.get(env_id)
            if state is None or reset:
                state = self._start_episode(env_id)
                route = WAMRoute.IDM
                is_forced = True
                source_chunk_id = -1
                route_actor_version = actor_version
            elif state.force_next_idm:
                # The actor weights changed after the pending decision was
                # emitted. Execute a documented forced-IDM boundary chunk
                # instead of attributing an old-policy route to new weights.
                route = WAMRoute.IDM
                is_forced = True
                source_chunk_id = -1
                route_actor_version = actor_version
                state.force_next_idm = False
                state.pending = None
            else:
                if state.pending is None:
                    raise RuntimeError(
                        f"Environment {env_id} has no Gate decision for chunk "
                        f"{state.chunk_id}."
                    )
                pending = state.pending
                if pending.episode_id != state.episode_id:
                    raise RuntimeError("Pending Gate decision crossed an episode boundary.")
                if pending.source_chunk_id != state.chunk_id - 1:
                    raise RuntimeError(
                        "Pending Gate decision is not from the immediately "
                        "preceding chunk."
                    )
                route = pending.route
                is_forced = False
                source_chunk_id = pending.source_chunk_id
                route_actor_version = pending.actor_version
                state.pending = None

            routes.append(int(route))
            forced.append(is_forced)
            chunk_ids.append(state.chunk_id)
            episode_ids.append(state.episode_id)
            source_ids.append(source_chunk_id)
            versions.append(route_actor_version)
            state.chunk_id += 1

        device = env_ids.device
        return ChunkRouteRecord(
            route_used=torch.tensor(routes, device=device, dtype=torch.long),
            route_was_forced=torch.tensor(forced, device=device, dtype=torch.bool),
            chunk_ids=torch.tensor(chunk_ids, device=device, dtype=torch.long),
            episode_ids=torch.tensor(episode_ids, device=device, dtype=torch.long),
            route_source_chunk_ids=torch.tensor(
                source_ids, device=device, dtype=torch.long
            ),
            actor_versions=torch.tensor(versions, device=device, dtype=torch.long),
        )

    def emit(
        self,
        *,
        env_ids: torch.Tensor,
        routes: torch.Tensor,
        source_chunk_ids: torch.Tensor,
        episode_ids: torch.Tensor,
        actor_version: int,
    ) -> None:
        """Store one post-chunk Gate route for each environment."""

        shape = env_ids.shape
        if env_ids.ndim != 1 or any(
            tensor.shape != shape
            for tensor in (routes, source_chunk_ids, episode_ids)
        ):
            raise ValueError("All emitted Gate fields must have matching [B] shape.")
        if actor_version < 0:
            raise ValueError("`actor_version` must be non-negative.")

        for env_id_value, route_value, source_value, episode_value in zip(
            env_ids.tolist(),
            routes.tolist(),
            source_chunk_ids.tolist(),
            episode_ids.tolist(),
        ):
            env_id = int(env_id_value)
            state = self._states.get(env_id)
            if state is None:
                raise RuntimeError(f"Environment {env_id} has not consumed a route.")
            if state.pending is not None:
                raise RuntimeError(
                    f"Environment {env_id} already has an unused Gate decision."
                )
            source_chunk_id = int(source_value)
            if source_chunk_id != state.chunk_id - 1:
                raise ValueError(
                    f"Gate source chunk {source_chunk_id} does not match the "
                    f"just-executed chunk {state.chunk_id - 1}."
                )
            if int(episode_value) != state.episode_id:
                raise ValueError("Gate decision episode does not match route state.")
            try:
                route = WAMRoute(int(route_value))
            except ValueError as exc:
                raise ValueError(f"Invalid emitted WAM route {route_value}.") from exc
            state.pending = _PendingDecision(
                route=route,
                source_chunk_id=source_chunk_id,
                episode_id=state.episode_id,
                actor_version=actor_version,
            )

    def discard(self, env_ids: torch.Tensor) -> None:
        """Discard terminal-unused decisions before an explicit reset."""

        for env_id in env_ids.tolist():
            state = self._states.get(int(env_id))
            if state is not None:
                state.pending = None

    def force_idm_after_actor_update(self) -> None:
        """Invalidate pending decisions and force one IDM chunk per live env."""

        for state in self._states.values():
            state.pending = None
            state.force_next_idm = True

    def state_dict(self) -> dict:
        """Return checkpointable route state with no tensor/device dependency."""

        return {
            # Retain the scalar for legacy readers while the per-environment map
            # makes episode identities independent of asynchronous reset order.
            "next_episode_id": sum(self._next_episode_ids.values()),
            "next_episode_ids": dict(self._next_episode_ids),
            "states": {
                env_id: {
                    "episode_id": state.episode_id,
                    "chunk_id": state.chunk_id,
                    "force_next_idm": state.force_next_idm,
                    "pending": (
                        None
                        if state.pending is None
                        else {
                            "route": int(state.pending.route),
                            "source_chunk_id": state.pending.source_chunk_id,
                            "episode_id": state.pending.episode_id,
                            "actor_version": state.pending.actor_version,
                        }
                    ),
                }
                for env_id, state in self._states.items()
            },
        }

    def load_state_dict(self, payload: dict) -> None:
        """Restore checkpointed route schedules and reject malformed state."""

        legacy_next_episode_id = int(payload.get("next_episode_id", 0))
        if legacy_next_episode_id < 0:
            raise ValueError("next_episode_id must be non-negative.")
        self._states = {}
        for env_id_value, raw in payload["states"].items():
            pending_raw = raw["pending"]
            pending = (
                None
                if pending_raw is None
                else _PendingDecision(
                    route=WAMRoute(int(pending_raw["route"])),
                    source_chunk_id=int(pending_raw["source_chunk_id"]),
                    episode_id=int(pending_raw["episode_id"]),
                    actor_version=int(pending_raw["actor_version"]),
                )
            )
            self._states[int(env_id_value)] = _EnvironmentRouteState(
                episode_id=int(raw["episode_id"]),
                chunk_id=int(raw["chunk_id"]),
                pending=pending,
                force_next_idm=bool(raw.get("force_next_idm", False)),
            )

        raw_next_ids = payload.get("next_episode_ids")
        if raw_next_ids is None:
            # Legacy checkpoints used one process-global counter. Active state
            # still gives a safe per-environment lower bound for future resets.
            self._next_episode_ids = {
                env_id: state.episode_id + 1
                for env_id, state in self._states.items()
            }
        else:
            self._next_episode_ids = {
                int(env_id): int(next_id)
                for env_id, next_id in raw_next_ids.items()
            }
            if any(next_id < 0 for next_id in self._next_episode_ids.values()):
                raise ValueError("next_episode_ids must be non-negative.")
            for env_id, state in self._states.items():
                if self._next_episode_ids.get(env_id, -1) <= state.episode_id:
                    raise ValueError(
                        "next_episode_ids must advance past every active episode."
                    )
