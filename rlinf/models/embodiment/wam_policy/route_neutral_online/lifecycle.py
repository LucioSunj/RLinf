# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bounded host-memory lifecycle for the seven-GPU online profile."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult
from rlinf.models.embodiment.wam_policy.pad_rv.memory import release_pad_host_memory
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_runner import (
    PadRouteNeutralRunner,
)
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _lifecycle_cfg(cfg: Any) -> Any:
    profile = cfg.route_neutral_online_implementation
    required = (
        "release_host_memory_after_rollout_init",
        "release_host_memory_after_trajectory_send",
        "release_host_memory_after_trajectory_receive",
    )
    disabled = [name for name in required if not bool(profile.get(name, False))]
    if disabled:
        raise ValueError(f"Route-neutral host-memory releases disabled: {disabled}.")
    return profile


class RouteNeutralOnlineRolloutWorker(MultiStepRolloutWorker):
    """Retain standard trainable replay while releasing build temporaries."""

    def init_worker(self) -> None:
        super().init_worker()
        _lifecycle_cfg(self.cfg)
        report = release_pad_host_memory(
            schema="route-neutral-online-rollout-host-memory-release-v1",
            rank=int(self._rank),
            phase="post_model_initialization",
        )
        print(
            "ROUTE_NEUTRAL_ONLINE_ROLLOUT_HOST_MEMORY_RELEASE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )


class RouteNeutralOnlineEnvWorker(EnvWorker):
    """Serialize large rank payloads and release them after channel transfer."""

    async def send_rollout_trajectories(
        self,
        rollout_result: EmbodiedRolloutResult,
        channel: Channel,
        *,
        stage_id: int,
    ) -> None:
        profile = _lifecycle_cfg(self.cfg)
        mode = str(profile.trajectory_send_mode)
        if mode == "concurrent":
            await self._send_and_release(
                rollout_result,
                channel,
                stage_id=stage_id,
            )
            return
        if mode != "serialized":
            raise ValueError(f"Unsupported route-neutral trajectory_send_mode: {mode}.")

        lock_root = Path(str(self.cfg.runner.logger.log_path))
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / ".route-neutral-online-trajectory-send.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        wait_started = time.perf_counter()
        await asyncio.to_thread(fcntl.flock, lock_fd, fcntl.LOCK_EX)
        wait_seconds = time.perf_counter() - wait_started
        try:
            await self._send_and_release(
                rollout_result,
                channel,
                stage_id=stage_id,
            )
        finally:
            await asyncio.to_thread(fcntl.flock, lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        print(
            "ROUTE_NEUTRAL_ONLINE_TRAJECTORY_SEND_SERIALIZATION_AUDIT="
            + json.dumps(
                {
                    "schema": "route-neutral-online-trajectory-send-v1",
                    "status": "PASS",
                    "rank": int(self._rank),
                    "stage_id": int(stage_id),
                    "wait_seconds": wait_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    async def _send_and_release(
        self,
        rollout_result: EmbodiedRolloutResult,
        channel: Channel,
        *,
        stage_id: int,
    ) -> None:
        await super().send_rollout_trajectories(
            rollout_result,
            channel,
            stage_id=stage_id,
        )
        report = release_pad_host_memory(
            schema="route-neutral-online-env-host-memory-release-v1",
            rank=int(self._rank),
            phase="post_trajectory_send",
        )
        print(
            "ROUTE_NEUTRAL_ONLINE_ENV_HOST_MEMORY_RELEASE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )


class RouteNeutralOnlineRunner(PadRouteNeutralRunner):
    """Reuse generic damped control with rank-serial rollout initialization."""

    def _init_rollout_workers_serially(self) -> None:
        profile = _lifecycle_cfg(self.cfg)
        if str(profile.rollout_init_mode) != "serial_rank":
            raise ValueError(
                "Route-neutral online training requires serial_rank initialization."
            )
        ranks = [item.rank for item in self.rollout.worker_info_list]
        if ranks != list(range(len(ranks))):
            raise ValueError(
                f"Route-neutral rollout ranks are not contiguous: {ranks}."
            )
        for rank in ranks:
            self.logger.info(
                "Initializing route-neutral online rollout rank %s/%s with "
                "bounded host memory.",
                rank,
                len(ranks) - 1,
            )
            self.rollout.execute_on(rank).init_worker().wait()


__all__ = [
    "RouteNeutralOnlineEnvWorker",
    "RouteNeutralOnlineRolloutWorker",
    "RouteNeutralOnlineRunner",
]
