# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Same-chunk closed-loop evaluation loop for clean and zero-shot domains."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastwam.causal_prediction import (
    CausalComputeMode,
    CausalControlKind,
    CausalInterventionSpecV2,
)

from rlinf.envs.action_contract import ActionExecutionTrace
from rlinf.envs.libero.action_contract import LiberoActionContract
from rlinf.models.embodiment.wam_policy.causal_routing_v2 import (
    EpisodeComputeBudgetV2,
)
from rlinf.models.embodiment.wam_policy.causal_runtime import (
    CausalLiberoFastWAMRuntime,
)


@dataclass(frozen=True)
class SameChunkRouteContextV2:
    """Information available before the current action chunk is generated."""

    observation: Mapping[str, Any]
    chunk_index: int
    previous_mode: CausalComputeMode | None
    remaining_budget: float
    steps_to_go: int
    history: tuple[Mapping[str, Any], ...]

    @property
    def no_history(self) -> bool:
        """Return whether this is the explicit first-chunk history state."""

        return not self.history

    @property
    def no_previous_route(self) -> bool:
        """Return whether no earlier route exists in this episode."""

        return self.previous_mode is None


@dataclass(frozen=True)
class SameChunkRouteDecisionV2:
    """A pre-action route plus measured overhead and frozen mode costs."""

    desired_mode: CausalComputeMode
    mode_costs: Mapping[CausalComputeMode, float]
    proposal_cost: float = 0.0
    gate_cost: float = 0.0
    proposal_calls: int = 0
    gate_calls: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = CausalComputeMode.parse(self.desired_mode)
        object.__setattr__(self, "desired_mode", mode)
        costs = {
            CausalComputeMode.parse(key): float(value)
            for key, value in self.mode_costs.items()
        }
        object.__setattr__(self, "mode_costs", costs)
        if not mode.is_routable or mode not in costs:
            raise ValueError("Closed-loop route must select a costed formal expert.")
        if CausalComputeMode.C0_CURRENT not in costs:
            raise ValueError("Closed-loop costs must include the fastest C0 endpoint.")
        if any(value < 0 or not np.isfinite(value) for value in costs.values()):
            raise ValueError("Closed-loop normalized mode costs must be non-negative.")
        if min(self.proposal_cost, self.gate_cost) < 0:
            raise ValueError("Closed-loop routing overhead must be non-negative.")
        if min(self.proposal_calls, self.gate_calls) < 0:
            raise ValueError("Closed-loop routing call counts must be non-negative.")


@dataclass(frozen=True)
class ClosedLoopChunkRecordV2:
    """One route decision that controls the same indexed action chunk."""

    chunk_index: int
    desired_mode: CausalComputeMode
    executed_mode: CausalComputeMode
    previous_mode: CausalComputeMode | None
    action_seed: int
    video_seed: int | None
    proposal_cost: float
    gate_cost: float
    proposal_calls: int
    gate_calls: int
    mode_budget_cost: float
    remaining_budget: float
    critical_path_latency_ms: float
    video_denoise_calls: int
    action_denoise_calls: int
    submitted_action_count: int
    switched: bool
    route_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ClosedLoopEpisodeResultV2:
    """Typed result for one resumable clean or Plus episode cell."""

    chunks: tuple[ClosedLoopChunkRecordV2, ...]
    final_success: bool
    final_return: float
    completion_step: int
    total_critical_path_latency_ms: float
    episode_gpu_seconds: float
    proposal_cost: float
    gate_cost: float
    proposal_calls: int
    gate_calls: int
    prediction_calls: int
    action_denoise_calls: int
    submitted_action_count: int
    switch_count: int
    budget_remaining: float


RouteDecisionCallbackV2 = Callable[[SameChunkRouteContextV2], SameChunkRouteDecisionV2]
SeedScheduleV2 = Callable[[int], tuple[int, int]]


class SameChunkCausalEvaluationRunnerV2:
    """Execute each route decision on the chunk for which it was computed."""

    def __init__(
        self,
        *,
        env: Any,
        runtime: CausalLiberoFastWAMRuntime,
        action_contract: LiberoActionContract,
        history_chunks: int = 4,
    ) -> None:
        if int(getattr(env, "num_envs", 0)) != 1:
            raise ValueError("Causal closed-loop evaluation requires batch one.")
        if history_chunks != 4:
            raise ValueError("Causal v2 closed-loop history is frozen to four chunks.")
        if action_contract.low != (-1.0,) * 7 or action_contract.high != (1.0,) * 7:
            raise RuntimeError("Causal v2 evaluation requires live [-1,1]^7 actions.")
        self.env = env
        self.runtime = runtime
        self.action_contract = action_contract
        self.history_chunks = history_chunks

    @staticmethod
    def _latest_observation(step_result: Any) -> Mapping[str, Any]:
        observations = step_result[0]
        if not isinstance(observations, Sequence) or not observations:
            raise RuntimeError("Closed-loop LIBERO step returned no observation.")
        return dict(observations[-1])

    def run_episode(
        self,
        *,
        initial_observation: Mapping[str, Any],
        route_decision: RouteDecisionCallbackV2,
        seed_schedule: SeedScheduleV2,
        budget: EpisodeComputeBudgetV2,
    ) -> ClosedLoopEpisodeResultV2:
        """Run one episode, including a real route decision for chunk zero."""

        observation = dict(initial_observation)
        history: list[Mapping[str, Any]] = []
        previous_mode = None
        chunks = []
        max_steps = int(self.runtime.action_protocol.max_episode_steps)
        chunk_index = 0
        while (
            not bool(self.env.success_once[0])
            and int(self.env.elapsed_steps[0]) < max_steps
        ):
            context = SameChunkRouteContextV2(
                observation=observation,
                chunk_index=chunk_index,
                previous_mode=previous_mode,
                remaining_budget=budget.remaining_cost,
                steps_to_go=max_steps - int(self.env.elapsed_steps[0]),
                history=tuple(history),
            )
            decision = route_decision(context)
            budget.debit_overhead(
                proposal_cost=decision.proposal_cost,
                gate_cost=decision.gate_cost,
            )
            executed_mode, mode_cost = budget.debit_desired_mode(
                desired_mode=decision.desired_mode,
                mode_costs=decision.mode_costs,
            )
            action_seed, video_seed = seed_schedule(chunk_index)
            spec = CausalInterventionSpecV2(
                mode=executed_mode,
                control=CausalControlKind.STANDARD,
                treatment_chunks=1,
                continuation_mode=executed_mode,
                replicate=0,
                action_seed=action_seed,
                video_seed=video_seed if executed_mode.runs_future_prediction else None,
            )
            sample = self.runtime.sample_causal_intervention(
                env_obs=observation,
                spec=spec,
            )
            if sample.action_execution_trace is None:
                raise RuntimeError(
                    "Closed-loop action conversion produced no audit trace."
                )
            step_result, submitted = self.env.chunk_step_with_action_trace(
                sample.actions,
                self.action_contract,
            )
            trace = ActionExecutionTrace.combine(
                sample.action_execution_trace,
                ActionExecutionTrace((submitted,)),
            )
            submitted_record = trace.record_for_batch_index(0)
            submitted_count = int(sample.actions.shape[1])
            switched = previous_mode is not None and previous_mode is not executed_mode
            chunks.append(
                ClosedLoopChunkRecordV2(
                    chunk_index=chunk_index,
                    desired_mode=decision.desired_mode,
                    executed_mode=executed_mode,
                    previous_mode=previous_mode,
                    action_seed=action_seed,
                    video_seed=(
                        video_seed if executed_mode.runs_future_prediction else None
                    ),
                    proposal_cost=float(decision.proposal_cost),
                    gate_cost=float(decision.gate_cost),
                    proposal_calls=int(decision.proposal_calls),
                    gate_calls=int(decision.gate_calls),
                    mode_budget_cost=mode_cost,
                    remaining_budget=budget.remaining_cost,
                    critical_path_latency_ms=float(sample.latency_ms["critical_path"]),
                    video_denoise_calls=int(sample.video_denoise_calls),
                    action_denoise_calls=int(sample.action_denoise_calls),
                    submitted_action_count=submitted_count,
                    switched=switched,
                    route_metadata={
                        **dict(decision.metadata),
                        "submitted_action_audit": submitted_record,
                    },
                )
            )
            observation = self._latest_observation(step_result)
            history.append(
                {
                    "chunk_index": chunk_index,
                    "executed_mode": executed_mode.value,
                    "submitted_action_count": submitted_count,
                    "success": bool(self.env.success_once[0]),
                }
            )
            history = history[-self.history_chunks :]
            previous_mode = executed_mode
            chunk_index += 1
        return ClosedLoopEpisodeResultV2(
            chunks=tuple(chunks),
            final_success=bool(self.env.success_once[0]),
            final_return=float(self.env.returns[0]),
            completion_step=int(self.env.elapsed_steps[0]),
            total_critical_path_latency_ms=float(
                sum(chunk.critical_path_latency_ms for chunk in chunks)
            ),
            episode_gpu_seconds=float(
                sum(chunk.critical_path_latency_ms for chunk in chunks) / 1000.0
            ),
            proposal_cost=float(sum(chunk.proposal_cost for chunk in chunks)),
            gate_cost=float(sum(chunk.gate_cost for chunk in chunks)),
            proposal_calls=sum(chunk.proposal_calls for chunk in chunks),
            gate_calls=sum(chunk.gate_calls for chunk in chunks),
            prediction_calls=sum(chunk.video_denoise_calls for chunk in chunks),
            action_denoise_calls=sum(chunk.action_denoise_calls for chunk in chunks),
            submitted_action_count=sum(
                chunk.submitted_action_count for chunk in chunks
            ),
            switch_count=sum(int(chunk.switched) for chunk in chunks),
            budget_remaining=float(budget.remaining_cost),
        )
