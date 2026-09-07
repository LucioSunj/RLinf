# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bounded host-memory lifecycle for the multi-rank online profile."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult
from rlinf.envs.libero.task_sampler import BalancedLiberoTaskSampler
from rlinf.envs.utils import get_env_attr
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

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        shared_rank = cfg.route_neutral_online_implementation.get(
            "shared_gpu_rollout_rank"
        )
        if shared_rank is not None:
            self.enable_offload = int(self._rank) == int(shared_rank)

    def audit_shared_gpu_roundtrip(self) -> dict[str, Any]:
        """Validate the shared replica's first restored CPU/GPU/CPU cycle."""

        from .shared_gpu import audit_residency_roundtrip

        report = audit_residency_roundtrip(
            model=self.hf_model,
            onload=self.reload_model,
            offload=self.offload_model,
            device=self.device,
            state=lambda: {
                "version": self.version,
                "runtime": self.hf_model.rollout_runtime_state_dict(),
            },
        )
        self.log_info("ROUTE_NEUTRAL_SHARED_GPU_ROUNDTRIP=" + json.dumps(report))
        return report

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

    def describe_task_pools(self) -> dict[str, Any]:
        """Expose actual reset-pool sizes and language labels to the sampler."""

        environment = self.env_list[0]
        bins = np.asarray(get_env_attr(environment, "cumsum_trial_id_bins"))
        suite = get_env_attr(environment, "task_suite")
        return {
            "reset_pool_sizes": np.diff(np.concatenate(([0], bins))).tolist(),
            "task_names": [suite.get_task(i).language for i in range(10)],
        }

    def set_global_task_plan(self, plan: dict[str, Any]) -> None:
        """Receive one globally generated plan; workers never draw task quotas."""

        if (
            self.stage_num != 1
            or self.cfg.env.train.get("task_sampling") != "global_balanced"
        ):
            raise ValueError(
                "Global task plans require the synchronous balanced profile."
            )
        slots = plan["ranks"][int(self._rank)]
        setter = get_env_attr(self.env_list[0], "set_global_task_plan")
        setter(slots, plan["runner_step"])
        self._global_task_plan = plan

    def _attach_task_metadata(
        self, result: EmbodiedRolloutResult, stage_id: int
    ) -> None:
        """Attach episode identity after inference, without adding policy features."""

        if self.cfg.env.train.get("task_sampling") != "global_balanced":
            return
        slots = self._global_task_plan["ranks"][int(self._rank)]
        environment = self.env_list[stage_id]
        actual_tasks = torch.as_tensor(
            get_env_attr(environment, "task_ids"), dtype=torch.long
        )
        expected_tasks = torch.tensor([slot["task_id"] for slot in slots])
        actual_trials = torch.as_tensor(
            get_env_attr(environment, "trial_ids"), dtype=torch.long
        )
        if not torch.equal(actual_tasks, expected_tasks) or not torch.equal(
            actual_trials, torch.tensor([slot["trial_id"] for slot in slots])
        ):
            raise ValueError(
                "Executed task/reset identities differ from the global plan."
            )
        success = torch.as_tensor(
            get_env_attr(environment, "success_once"), dtype=torch.bool
        ).reshape(-1)
        terminated = (
            torch.stack(result.terminations)
            .reshape(-1, len(slots), self.model_cfg.num_action_chunks)
            .any(dim=(0, 2))
        )
        truncated = (
            torch.stack(result.truncations)
            .reshape(-1, len(slots), self.model_cfg.num_action_chunks)
            .any(dim=(0, 2))
        )
        metadata = {
            "multitask_task_id": expected_tasks,
            "multitask_reset_state_id": torch.tensor(
                [slot["reset_state_id"] for slot in slots]
            ),
            "multitask_episode_slot_id": torch.tensor(
                [slot["episode_slot_id"] for slot in slots]
            ),
            "multitask_episode_success": success,
            "multitask_episode_failed": terminated & ~success,
            "multitask_episode_truncated": truncated & ~success,
        }
        for forward_inputs in result.forward_inputs:
            forward_inputs.update(
                {key: value.clone() for key, value in metadata.items()}
            )

    async def send_rollout_trajectories(
        self,
        rollout_result: EmbodiedRolloutResult,
        channel: Channel,
        *,
        stage_id: int,
    ) -> None:
        self._attach_task_metadata(rollout_result, stage_id)
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

    def init_workers(self) -> None:
        super().init_workers()
        if self.cfg.env.train.get("task_sampling") == "global_balanced":
            descriptions = self.env.describe_task_pools().wait()
            if any(item != descriptions[0] for item in descriptions):
                raise ValueError("LIBERO task/reset pools differ across ranks.")
            self.task_sampler = BalancedLiberoTaskSampler(
                total_envs=int(self.cfg.env.train.total_num_envs),
                reset_pool_sizes=descriptions[0]["reset_pool_sizes"],
                seed=int(self.cfg.env.train.seed),
            )
            self.task_names = descriptions[0]["task_names"]
            resume_dir = self.cfg.runner.get("resume_dir")
            if resume_dir is not None:
                self.task_sampler.load_state_dict(
                    json.loads((Path(resume_dir) / "task_sampler.json").read_text()),
                    runner_step=self.global_step,
                )
            self._task_audit_dir.mkdir(parents=True, exist_ok=True)
            (self._task_audit_dir / "task_definition.json").write_text(
                json.dumps(
                    {**descriptions[0], "sampler": self.task_sampler.state_dict()},
                    indent=2,
                )
                + "\n"
            )
        shared_rank = self.cfg.route_neutral_online_implementation.get(
            "shared_gpu_rollout_rank"
        )
        if shared_rank is not None:
            actor_report = self.actor.audit_shared_gpu_roundtrip().wait()
            rollout_report = (
                self.rollout.execute_on(int(shared_rank))
                .audit_shared_gpu_roundtrip()
                .wait()
            )
            self.logger.info(
                "Shared GPU residency round trips passed: actor=%s rollout=%s",
                actor_report,
                rollout_report,
            )

    @property
    def _task_audit_dir(self) -> Path:
        return (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "audits"
        )

    def _set_worker_global_step(self) -> None:
        self._complete_update_started = time.perf_counter()
        self._checkpoint_seconds = 0.0
        if hasattr(self, "task_sampler"):
            plan = self.task_sampler.next_plan(self.global_step)
            self.env.set_global_task_plan(plan).wait()
            with (self._task_audit_dir / "task_plans.jsonl").open("a") as stream:
                stream.write(json.dumps(plan) + "\n")
        super()._set_worker_global_step()

    def _save_checkpoint(self) -> None:
        started = time.perf_counter()
        super()._save_checkpoint()
        if hasattr(self, "task_sampler"):
            directory = (
                self._task_audit_dir.parent
                / "checkpoints"
                / f"global_step_{self.global_step}"
            )
            temporary = directory / ".task_sampler.json.tmp"
            temporary.write_text(
                json.dumps(self.task_sampler.state_dict(), indent=2) + "\n"
            )
            temporary.replace(directory / "task_sampler.json")
        self._checkpoint_seconds = time.perf_counter() - started

    def _log_step_metrics(self, **kwargs) -> None:
        if hasattr(self, "task_sampler"):
            duration = time.perf_counter() - self._complete_update_started
            checkpoint_seconds = self._checkpoint_seconds
            metrics = {
                "time/complete_update": duration,
                "time/checkpoint": checkpoint_seconds,
                "time/update_without_checkpoint": duration - checkpoint_seconds,
            }
            self.metric_logger.log(data=metrics, step=kwargs["step"])
            report = {
                "runner_step": self.global_step,
                "timing": metrics,
                "rollout": kwargs["actor_rollout_metrics"],
                "training": kwargs["actor_training_metrics"],
            }
            with (self._task_audit_dir / "multitask_updates.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(report, default=lambda value: float(value)) + "\n"
                )
        super()._log_step_metrics(**kwargs)

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
