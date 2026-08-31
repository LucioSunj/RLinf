# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD-Frozen evaluation runner with bounded rank-serial model construction."""

from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner


class PadFrozenEvalRunner(EmbodiedEvalRunner):
    """Initialize seven frozen-pair replicas without overlapping CPU builds."""

    def init_workers(self) -> None:
        if str(self.cfg.pad_rv_implementation.rollout_init_mode) != "serial_rank":
            raise ValueError("PAD-Frozen evaluation requires serial_rank init.")
        ranks = [item.rank for item in self.rollout.worker_info_list]
        if ranks != list(range(len(ranks))):
            raise ValueError(
                f"PAD evaluation rollout ranks are not contiguous: {ranks}."
            )
        for rank in ranks:
            self.logger.info(
                "Initializing PAD evaluation rollout rank %s/%s.",
                rank,
                len(ranks) - 1,
            )
            self.rollout.execute_on(rank).init_worker().wait()
        self.env.init_worker().wait()
