# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Frozen-ledger evaluation for the native pure UNCOND policy checkpoint."""

from typing import Any

from rlinf.models.embodiment.wam_policy.contracts import ChunkRouteRecord, WAMRoute
from rlinf.models.embodiment.wam_policy.evaluation import (
    EvaluationRouteSelection,
    EvaluationRoutingMode,
)
from rlinf.runners.fastwam_libero_eval_collector import FastWAMLiberoEvalCollector


class UncondRLEvalCollector(FastWAMLiberoEvalCollector):
    """Reuse episode/action evidence while requiring zero routing decisions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(fixed_idm_cost=0.0, **kwargs)
        if self.routing_mode != EvaluationRoutingMode.FORCED_UNCOND:
            raise ValueError("Pure UNCOND evaluation requires forced_uncond.")

    def _expected_gate_validity(
        self,
        route: ChunkRouteRecord,
        selection: EvaluationRouteSelection,
        *,
        index: int,
        current_step: bool,
        terminal: bool,
    ) -> bool:
        if (
            int(route.route_used[index]) != int(WAMRoute.UNCOND)
            or not bool(route.route_was_forced[index])
            or int(route.route_source_chunk_ids[index]) != -1
            or selection.mode != EvaluationRoutingMode.FORCED_UNCOND
            or int(selection.effective_next_route[index]) != int(WAMRoute.UNCOND)
        ):
            raise ValueError("Pure UNCOND evaluation cannot emit routing decisions.")
        return False

    def _persist_completed_episode(self, *, episode, chunks) -> None:
        # The shared legacy aggregate counts a forced initial chunk under this
        # field even for UNCOND. Pure UNCOND never executes an initial IDM.
        episode["forced_initial_idm_count"] = 0
        super()._persist_completed_episode(episode=episode, chunks=chunks)
