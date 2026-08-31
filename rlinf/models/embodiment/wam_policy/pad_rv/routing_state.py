# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Same-chunk routing state used only by PAD-RV."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rlinf.models.embodiment.wam_policy.contracts import ChunkRouteRecord, WAMRoute


@dataclass(frozen=True)
class CurrentStepRouteIdentity:
    """Episode/chunk identity prepared before a same-chunk Gate decision."""

    env_ids: torch.Tensor
    chunk_ids: torch.Tensor
    episode_ids: torch.Tensor
    actor_versions: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.env_ids.shape
        values = (
            self.env_ids,
            self.chunk_ids,
            self.episode_ids,
            self.actor_versions,
        )
        if self.env_ids.ndim != 1 or any(value.shape != shape for value in values):
            raise ValueError("Current-step identity fields must share [B] shape.")
        if any(value.dtype not in (torch.int32, torch.int64) for value in values):
            raise TypeError("Current-step identity fields must be integers.")
        if any(bool((value < 0).any()) for value in values):
            raise ValueError("Current-step identity fields must be non-negative.")


@dataclass
class _EnvironmentState:
    episode_id: int
    chunk_id: int
    prepared: bool = False
    last_route: WAMRoute | None = None


class CurrentStepRouteTracker:
    """Track Gate decisions that control the action chunk with the same id."""

    def __init__(self) -> None:
        self._states: dict[int, _EnvironmentState] = {}
        self._next_episode_ids: dict[int, int] = {}

    def _start_episode(self, env_id: int) -> _EnvironmentState:
        episode_id = self._next_episode_ids.get(env_id, 0)
        state = _EnvironmentState(episode_id=episode_id, chunk_id=0)
        self._next_episode_ids[env_id] = episode_id + 1
        self._states[env_id] = state
        return state

    def prepare(
        self,
        *,
        env_ids: torch.Tensor,
        reset_mask: torch.Tensor,
        actor_version: int,
    ) -> CurrentStepRouteIdentity:
        if env_ids.ndim != 1:
            raise ValueError("`env_ids` must be one-dimensional.")
        if reset_mask.shape != env_ids.shape or reset_mask.dtype != torch.bool:
            raise ValueError("`reset_mask` must be bool and match `env_ids`.")
        if len(set(map(int, env_ids.tolist()))) != env_ids.numel():
            raise ValueError("`env_ids` must be unique within one policy batch.")
        if actor_version < 0:
            raise ValueError("`actor_version` must be non-negative.")
        chunks: list[int] = []
        episodes: list[int] = []
        for raw_env, raw_reset in zip(
            env_ids.tolist(), reset_mask.tolist(), strict=True
        ):
            env_id = int(raw_env)
            state = self._states.get(env_id)
            if state is None or bool(raw_reset):
                if state is not None and state.prepared:
                    raise RuntimeError("Cannot reset an uncommitted PAD route.")
                state = self._start_episode(env_id)
            if state.prepared:
                raise RuntimeError(f"Environment {env_id} already has a PAD route.")
            state.prepared = True
            chunks.append(state.chunk_id)
            episodes.append(state.episode_id)
        return CurrentStepRouteIdentity(
            env_ids=env_ids.to(dtype=torch.long),
            chunk_ids=torch.tensor(chunks, device=env_ids.device, dtype=torch.long),
            episode_ids=torch.tensor(episodes, device=env_ids.device, dtype=torch.long),
            actor_versions=torch.full_like(env_ids, actor_version, dtype=torch.long),
        )

    def commit(
        self,
        *,
        identity: CurrentStepRouteIdentity,
        routes: torch.Tensor,
    ) -> ChunkRouteRecord:
        if routes.shape != identity.chunk_ids.shape:
            raise ValueError("PAD routes must match the prepared batch shape.")
        try:
            normalized = tuple(WAMRoute(int(value)) for value in routes.tolist())
        except ValueError as error:
            raise ValueError("PAD route is outside WAMRoute.") from error
        for env, chunk, episode, route in zip(
            identity.env_ids.tolist(),
            identity.chunk_ids.tolist(),
            identity.episode_ids.tolist(),
            normalized,
            strict=True,
        ):
            state = self._states.get(int(env))
            if state is None or not state.prepared:
                raise RuntimeError("PAD route has no matching preparation.")
            if state.chunk_id != int(chunk) or state.episode_id != int(episode):
                raise RuntimeError("PAD route identity changed before commit.")
            state.prepared = False
            state.last_route = route
            state.chunk_id += 1
        route_tensor = torch.tensor(
            tuple(map(int, normalized)), device=routes.device, dtype=torch.long
        )
        return ChunkRouteRecord(
            route_used=route_tensor,
            route_was_forced=torch.zeros_like(route_tensor, dtype=torch.bool),
            chunk_ids=identity.chunk_ids.to(route_tensor.device),
            episode_ids=identity.episode_ids.to(route_tensor.device),
            route_source_chunk_ids=identity.chunk_ids.to(route_tensor.device),
            actor_versions=identity.actor_versions.to(route_tensor.device),
        )

    def previous_routes(self, identity: CurrentStepRouteIdentity) -> torch.Tensor:
        routes: list[int] = []
        for raw_env in identity.env_ids.tolist():
            state = self._states.get(int(raw_env))
            if state is None or not state.prepared:
                raise RuntimeError("PAD route identity is not prepared.")
            routes.append(
                int(WAMRoute.UNCOND if state.last_route is None else state.last_route)
            )
        return torch.tensor(routes, device=identity.env_ids.device, dtype=torch.long)

    def discard(self, env_ids: torch.Tensor) -> None:
        for raw_env in env_ids.tolist():
            state = self._states.get(int(raw_env))
            if state is not None and state.prepared:
                raise RuntimeError("Cannot discard an uncommitted PAD route.")

    def force_idm_after_actor_update(self) -> None:
        if any(state.prepared for state in self._states.values()):
            raise RuntimeError("Actor version changed during PAD route preparation.")

    def state_dict(self) -> dict:
        if any(state.prepared for state in self._states.values()):
            raise RuntimeError("Cannot checkpoint an uncommitted PAD route.")
        return {
            "routing_semantics": "current_step",
            "next_episode_ids": dict(self._next_episode_ids),
            "states": {
                env_id: {
                    "episode_id": state.episode_id,
                    "chunk_id": state.chunk_id,
                    "last_route": (
                        None if state.last_route is None else int(state.last_route)
                    ),
                }
                for env_id, state in self._states.items()
            },
        }

    def load_state_dict(self, payload: dict) -> None:
        if payload.get("routing_semantics") != "current_step":
            raise ValueError("PAD tracker requires current_step checkpoint state.")
        self._states = {
            int(env_id): _EnvironmentState(
                episode_id=int(raw["episode_id"]),
                chunk_id=int(raw["chunk_id"]),
                last_route=(
                    None
                    if raw.get("last_route") is None
                    else WAMRoute(int(raw["last_route"]))
                ),
            )
            for env_id, raw in payload["states"].items()
        }
        self._next_episode_ids = {
            int(env_id): int(next_id)
            for env_id, next_id in payload["next_episode_ids"].items()
        }
        if any(
            state.episode_id < 0 or state.chunk_id < 0
            for state in self._states.values()
        ) or any(next_id < 0 for next_id in self._next_episode_ids.values()):
            raise ValueError("PAD route checkpoint contains negative state.")
        for env_id, state in self._states.items():
            if self._next_episode_ids.get(env_id, -1) <= state.episode_id:
                raise ValueError("PAD next episode id did not advance.")
