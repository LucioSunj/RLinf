# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD runner with config-selected, rank-serial rollout initialization."""

from __future__ import annotations

import os

from rlinf.config_contracts import validate_fastwam_resume_steps
from rlinf.runners.embodied_runner import EmbodiedRunner

from .budget import PadPredictionBudgetController


class PadFrozenRunner(EmbodiedRunner):
    """Keep six rollout replicas while bounding their CPU construction peak."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fastwam_fair_cost_controller = PadPredictionBudgetController.from_configs(
            branch_cost=self.cfg.algorithm.fixed_branch_cost,
            prediction_budget=self.cfg.algorithm.prediction_budget,
        )

    def _init_rollout_workers_serially(self) -> None:
        if str(self.cfg.pad_rv_implementation.rollout_init_mode) != "serial_rank":
            raise ValueError("PAD-Frozen requires serial_rank rollout initialization.")
        ranks = [item.rank for item in self.rollout.worker_info_list]
        if ranks != list(range(len(ranks))):
            raise ValueError(f"PAD rollout ranks are not contiguous: {ranks}.")
        for rank in ranks:
            self.logger.info(
                "Initializing PAD rollout rank %s/%s with bounded host memory.",
                rank,
                len(ranks) - 1,
            )
            self.rollout.execute_on(rank).init_worker().wait()

    def init_workers(self) -> None:
        """Initialize rollout replicas one rank at a time, then env and actor."""

        self._init_rollout_workers_serially()
        self.env.init_worker().wait()
        if self.reward is not None:
            self.reward.init_worker().wait()
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info("Resuming training from checkpoint directory %s.", resume_dir)
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        if not os.path.exists(actor_checkpoint_path):
            raise FileNotFoundError(
                f"resume_dir {actor_checkpoint_path} does not exist."
            )
        actor_step = validate_fastwam_resume_steps(
            self.actor.load_checkpoint(actor_checkpoint_path).wait(),
            resume_dir,
        )
        rollout_checkpoint_path = os.path.join(resume_dir, "rollout")
        if not os.path.exists(rollout_checkpoint_path):
            raise FileNotFoundError(
                "PAD-Frozen resume requires rollout runtime checkpoints at "
                f"{rollout_checkpoint_path}."
            )
        rollout_step = validate_fastwam_resume_steps(
            self.rollout.load_checkpoint(rollout_checkpoint_path).wait(),
            resume_dir,
        )
        if rollout_step != actor_step:
            raise ValueError(
                "PAD-Frozen actor and rollout checkpoints disagree on step: "
                f"actor={actor_step}, rollout={rollout_step}."
            )
        self.global_step = actor_step
        self._load_fastwam_training_guard(resume_dir)
