# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Balanced task execution and native resume for pure UNCOND PPO."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.envs.libero.task_sampler import BalancedLiberoTaskSampler
from rlinf.envs.utils import get_env_attr
from rlinf.models.embodiment.wam_policy.pad_rv.memory import release_pad_host_memory
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class UncondRLRolloutWorker(MultiStepRolloutWorker):
    """Release construction temporaries before initializing the next replica."""

    def init_worker(self) -> None:
        super().init_worker()
        release_pad_host_memory(
            schema="fastwam-uncond-host-memory-release-v1",
            rank=int(self._rank),
            phase="post_model_initialization",
        )

    def serve_actor_training(self, **kwargs: Any) -> dict[str, Any]:
        """Use the resident policy as an optimizer-free gradient helper."""
        from .uncond_rl_helpers import serve_actor_training

        return serve_actor_training(self, **kwargs)


class UncondRLEnvWorker(EnvWorker):
    """Apply runner-owned task/reset plans and retain executed episode outcomes."""

    def _build_rollout_input_data(
        self,
        env_batch: dict[str, Any],
        *,
        stage_id: int,
        eval_mode: bool = False,
        force_reset: bool = False,
    ) -> dict[str, Any]:
        """Mark post-terminal rows while retaining the original rollout time axis."""
        data = super()._build_rollout_input_data(
            env_batch,
            stage_id=stage_id,
            eval_mode=eval_mode,
            force_reset=force_reset,
        )
        if (
            eval_mode
            or self.cfg.env.train.auto_reset
            or self.cfg.env.train.ignore_terminations
        ):
            return data
        if not hasattr(self, "_uncond_rl_finished"):
            self._uncond_rl_finished = {}
        if force_reset or stage_id not in self._uncond_rl_finished:
            self._uncond_rl_finished[stage_id] = torch.zeros_like(
                data["fastwam_reset_mask"]
            )
        elif env_batch.get("dones") is not None:
            finished = self._uncond_rl_finished[stage_id]
            finished |= env_batch["dones"].bool().reshape(finished.numel(), -1).any(1)
        data["obs"] = {
            **data["obs"],
            "_uncond_rl_active": ~self._uncond_rl_finished[stage_id],
        }
        return data

    def describe_task_pools(self) -> dict[str, Any]:
        """Expose the actual standard-LIBERO reset pools without model inputs."""
        environment = self.env_list[0]
        bins = np.asarray(get_env_attr(environment, "cumsum_trial_id_bins"))
        suite = get_env_attr(environment, "task_suite")
        return {
            "reset_pool_sizes": np.diff(np.concatenate(([0], bins))).tolist(),
            "task_names": [suite.get_task(i).language for i in range(10)],
        }

    def set_global_task_plan(self, plan: dict[str, Any]) -> None:
        """Assign this worker's share of one globally balanced update."""
        if self.stage_num != 1:
            raise ValueError("UNCOND task plans require one synchronous stage.")
        slots = plan["ranks"][int(self._rank)]
        get_env_attr(self.env_list[0], "set_global_task_plan")(
            slots, plan["runner_step"]
        )
        self._global_task_plan = plan

    def capture_task_outcomes(self, stage_id: int) -> None:
        """Record success before environment cleanup can reset its counters."""
        slots = self._global_task_plan["ranks"][int(self._rank)]
        environment = self.env_list[stage_id]
        for name, key in (("task_ids", "task_id"), ("trial_ids", "trial_id")):
            if not np.array_equal(
                get_env_attr(environment, name), [slot[key] for slot in slots]
            ):
                raise ValueError("Executed task/reset differs from the UNCOND plan.")
        success = np.asarray(get_env_attr(environment, "success_once"), dtype=bool)
        self._task_outcomes = {
            "runner_step": self._global_task_plan["runner_step"],
            "rank": int(self._rank),
            "episodes": [
                {**slot, "success": bool(won)}
                for slot, won in zip(slots, success, strict=True)
            ],
        }

    def task_outcomes(self) -> dict[str, Any]:
        """Return the last completed rollout's identities and success flags."""
        return self._task_outcomes

    async def send_rollout_trajectories(
        self, rollout_result, channel, *, stage_id: int
    ) -> None:
        """Record outcomes, finish transport, then return freed replay pages."""

        self.capture_task_outcomes(stage_id)
        await super().send_rollout_trajectories(
            rollout_result, channel, stage_id=stage_id
        )
        report = release_pad_host_memory(
            schema="uncond-env-host-memory-release-v1",
            rank=int(self._rank),
            phase="post_trajectory_send",
        )
        report["runner_step"] = int(self._global_task_plan["runner_step"])
        self.log_info(f"UNCOND_ENV_HOST_MEMORY_RELEASE {json.dumps(report)}")


class UncondRLRunner(EmbodiedRunner):
    """Use the native PPO runner with balanced, checkpointed Long task sampling."""

    def _init_rollout_workers(self) -> None:
        for worker in self.rollout.worker_info_list:
            self.rollout.execute_on(worker.rank).init_worker().wait()

    def _run_actor_training(self, *, next_step: bool):
        """Dispatch original microbatches to idle rollout replicas between syncs."""
        from .uncond_rl_config import uncond_training_helper_ranks
        from .uncond_rl_helpers import pack_helper_message

        helpers = uncond_training_helper_ranks(self.cfg)
        if not helpers:
            return super()._run_actor_training(next_step=next_step)
        available = {int(worker.rank) for worker in self.rollout.worker_info_list}
        if not set(helpers).issubset(available):
            raise ValueError("UNCOND helper rank is absent from the rollout group.")
        if not hasattr(self, "_training_helper_commands"):
            self._training_helper_commands = Channel.create(
                "UncondTrainingCommands", maxsize=2
            )
            self._training_helper_results = Channel.create(
                "UncondTrainingResults", maxsize=2
            )
            self.actor.configure_training_helpers(
                helper_ranks=helpers,
                command_channel=self._training_helper_commands,
                result_channel=self._training_helper_results,
            ).wait()
        handle = self.rollout.execute_on(*helpers).serve_actor_training(
            helper_ranks=helpers,
            command_channel=self._training_helper_commands,
            result_channel=self._training_helper_results,
            version=int(self.global_step),
        )
        result = super()._run_actor_training(next_step=next_step)
        opportunities = (
            int(self.cfg.env.train.total_num_envs)
            * int(self.cfg.env.train.max_steps_per_rollout_epoch)
            // int(self.cfg.actor.model.runtime.execution_horizon)
            // int(self.cfg.actor.global_batch_size)
            * int(self.cfg.algorithm.update_epoch)
        )
        for rank in helpers:
            self._training_helper_commands.put(
                pack_helper_message(
                    "finish_update",
                    {"optimizer_opportunities": opportunities},
                    version=int(self.global_step),
                    optimizer_step=-1,
                ),
                key=rank,
            )
        reports = handle.wait()
        self.logger.info("UNCOND helper phase complete: %s", reports)
        return result

    @property
    def _task_audit_dir(self) -> Path:
        return (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "audits"
        )

    def init_workers(self) -> None:
        super().init_workers()
        descriptions = self.env.describe_task_pools().wait()
        if any(item != descriptions[0] for item in descriptions):
            raise ValueError("LIBERO task/reset pools differ across workers.")
        self.task_sampler = BalancedLiberoTaskSampler(
            total_envs=int(self.cfg.env.train.total_num_envs),
            reset_pool_sizes=descriptions[0]["reset_pool_sizes"],
            seed=int(self.cfg.env.train.seed),
            num_ranks=len(descriptions),
        )
        resume_dir = self.cfg.runner.get("resume_dir")
        if resume_dir is not None:
            self.task_sampler.load_state_dict(
                json.loads((Path(resume_dir) / "task_sampler.json").read_text()),
                runner_step=self.global_step,
            )
        self._task_audit_dir.mkdir(parents=True, exist_ok=True)
        (self._task_audit_dir / "task_definition.json").write_text(
            json.dumps(descriptions[0], indent=2) + "\n"
        )
        if resume_dir is None:
            # Publish BC adapters and the fresh critic to every rollout owner
            # before writing a native step-zero initialization receipt.
            self.update_rollout_weights()
            self._save_checkpoint()

    def _set_worker_global_step(self) -> None:
        self._update_started = time.perf_counter()
        self._checkpoint_seconds = 0.0
        plan = self.task_sampler.next_plan(self.global_step)
        self.env.set_global_task_plan(plan).wait()
        with (self._task_audit_dir / "task_plans.jsonl").open("a") as stream:
            stream.write(json.dumps(plan) + "\n")
        super()._set_worker_global_step()

    def _save_checkpoint(self) -> None:
        started = time.perf_counter()
        super()._save_checkpoint()
        directory = (
            self._task_audit_dir.parent
            / "checkpoints"
            / f"global_step_{self.global_step}"
        )
        temporary = directory / ".task_sampler.json.tmp"
        temporary.write_text(json.dumps(self.task_sampler.state_dict()) + "\n")
        temporary.replace(directory / "task_sampler.json")
        self._checkpoint_seconds = time.perf_counter() - started

    def _log_step_metrics(self, **kwargs) -> None:
        # The environment RPC completes before reading outcomes, including the
        # final trajectory send and its captured pre-cleanup success flags.
        kwargs["env_handle"].wait()
        outcomes = self.env.task_outcomes().wait()
        episodes = [episode for rank in outcomes for episode in rank["episodes"]]
        if (
            any(rank["runner_step"] != self.global_step - 1 for rank in outcomes)
            or len(episodes) != self.cfg.env.train.total_num_envs
        ):
            raise ValueError("UNCOND episode report does not match this update.")
        metrics = {
            "time/complete_update": time.perf_counter() - self._update_started,
            "time/checkpoint": self._checkpoint_seconds,
            "task_global/success_count": sum(row["success"] for row in episodes),
        }
        for task in range(10):
            selected = [row for row in episodes if row["task_id"] == task]
            metrics[f"task/{task}/success_count"] = sum(
                row["success"] for row in selected
            )
            metrics[f"task/{task}/episode_count"] = len(selected)
        self.metric_logger.log(data=metrics, step=kwargs["step"])
        report = {
            "runner_step": self.global_step,
            "metrics": metrics,
            "episodes": episodes,
            "rollout": kwargs["actor_rollout_metrics"],
            "training": kwargs["actor_training_metrics"],
        }
        with (self._task_audit_dir / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(report, default=float) + "\n")
        super()._log_step_metrics(**kwargs)
