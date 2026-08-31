# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD-Frozen environment worker with a no-Flow audit contract."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import (
    EnvWorker,
    build_fastwam_action_failure_audit,
)

from .memory import release_pad_host_memory


class PadFrozenEnvWorker(EnvWorker):
    """Keep the guarded Action audit while omitting Flow-only metadata."""

    def _build_fastwam_training_action_failure_audit(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record rejected static merged-U Actions without Flow provenance."""

        return build_fastwam_action_failure_audit(
            **kwargs,
            require_uncond_denoise_index=False,
        )

    async def send_rollout_trajectories(
        self,
        rollout_result: EmbodiedRolloutResult,
        channel: Channel,
        *,
        stage_id: int,
    ) -> None:
        """Return consumed trajectory pages after their channel send completes."""

        mode = str(self.cfg.pad_rv_implementation.trajectory_send_mode)
        if mode == "concurrent":
            await self._send_rollout_trajectories_and_release(
                rollout_result,
                channel,
                stage_id=stage_id,
            )
            return
        if mode != "serialized":
            raise ValueError(f"Unsupported PAD trajectory_send_mode: {mode}.")

        lock_root = Path(str(self.cfg.runner.logger.log_path))
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / ".pad-trajectory-send.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        wait_started = time.perf_counter()
        await asyncio.to_thread(fcntl.flock, lock_fd, fcntl.LOCK_EX)
        wait_seconds = time.perf_counter() - wait_started
        try:
            await self._send_rollout_trajectories_and_release(
                rollout_result,
                channel,
                stage_id=stage_id,
            )
        finally:
            await asyncio.to_thread(fcntl.flock, lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        print(
            "PAD_ENV_TRAJECTORY_SEND_SERIALIZATION_AUDIT="
            + json.dumps(
                {
                    "schema": "pad-env-trajectory-send-serialization-v1",
                    "status": "PASS",
                    "rank": int(self._rank),
                    "stage_id": int(stage_id),
                    "wait_seconds": wait_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    async def _send_rollout_trajectories_and_release(
        self,
        rollout_result: EmbodiedRolloutResult,
        channel: Channel,
        *,
        stage_id: int,
    ) -> None:
        """Send one rank payload and release its consumed host pages."""

        await super().send_rollout_trajectories(
            rollout_result,
            channel,
            stage_id=stage_id,
        )
        if not bool(
            self.cfg.pad_rv_implementation.release_host_memory_after_trajectory_send
        ):
            raise ValueError("PAD trajectory-send host-memory release was disabled.")
        report = release_pad_host_memory(
            schema="pad-env-trajectory-host-memory-release-v1",
            rank=int(self._rank),
            phase="post_trajectory_send",
        )
        print(
            "PAD_ENV_TRAJECTORY_HOST_MEMORY_RELEASE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )

    def _record_fastwam_training_policy_metadata(
        self,
        rollout_result: Any,
        destination: list[Any],
    ) -> None:
        if rollout_result.forward_inputs.get("denoise_indices") is not None:
            raise ValueError("PAD-Frozen cannot collect Flow-SDE denoise indices.")
        if destination:
            raise RuntimeError("PAD-Frozen policy metadata stream must remain empty.")

    def _build_fastwam_training_policy_metadata_audit(
        self,
        *,
        streams: list[Any],
        traces: list[Any],
        environment_count: int,
        global_environment_offset: int,
    ) -> dict[str, Any]:
        if streams:
            raise RuntimeError("PAD-Frozen unexpectedly collected Flow metadata.")
        if not traces or environment_count <= 0:
            raise RuntimeError("PAD-Frozen Action audit has no executed trace data.")
        if global_environment_offset < 0:
            raise RuntimeError("PAD-Frozen environment offset cannot be negative.")
        return {
            "flow_sde_enabled": False,
            "denoise_index_stream_sha256_by_global_environment": [],
            "flow_sde_denoise_indices": None,
        }
