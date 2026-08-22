# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Typed v2 same-state branch execution with segmented outcomes and costs."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from fastwam.causal_prediction import (
    CausalControlKind,
    CausalCostV2,
    CausalInterventionSpecV2,
    CausalOutcomeV2,
    CausalTerminationType,
    PairedInterventionRecordV2,
)

from rlinf.envs.action_contract import ActionExecutionTrace
from rlinf.envs.libero.action_contract import LiberoActionContract
from rlinf.envs.libero.causal_snapshot import CausalSnapshotV2

from .causal_runtime import CausalChunkSample, CausalLiberoFastWAMRuntime


@dataclass(frozen=True)
class BranchObservationV2:
    """Directly observable causal outcomes at one branch boundary."""

    predicate_vector: tuple[bool, ...]
    predicate_progress: float
    contact_events: Mapping[str, int]

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.predicate_progress) <= 1.0:
            raise ValueError("Predicate progress must lie in [0, 1].")
        if any(int(value) < 0 for value in self.contact_events.values()):
            raise ValueError("Contact-event counts must be non-negative.")


def _sum_metrics(target: dict[str, float], values: Mapping[str, float]) -> None:
    for name, value in values.items():
        target[name] = target.get(name, 0.0) + float(value)


def _sum_calls(target: dict[str, int], sample: CausalChunkSample) -> None:
    target["video_prediction_calls"] = target.get("video_prediction_calls", 0) + int(
        sample.video_denoise_calls > 0
    )
    target["video_denoise_steps"] = target.get("video_denoise_steps", 0) + int(
        sample.video_denoise_calls
    )
    target["action_denoise_steps"] = target.get("action_denoise_steps", 0) + int(
        sample.action_denoise_calls
    )
    target["proposal_calls"] = target.get("proposal_calls", 0) + max(
        1, len(sample.proposal_seeds)
    )
    target["submitted_action_count"] = target.get("submitted_action_count", 0) + int(
        sample.actions.shape[1]
    )
    target.setdefault("gate_calls", 0)


class PairedCausalForkRunnerV2:
    """Execute v2 intervention specs after an exact restore before every branch."""

    def __init__(
        self,
        *,
        env,
        runtime: CausalLiberoFastWAMRuntime,
        restore_policy_runtime: Callable[[Mapping[str, Any]], None],
        restore_history: Callable[[Sequence[Mapping[str, Any]]], None],
        observe_branch: Callable[[Any], BranchObservationV2],
        secondary_outcomes: Callable[[Any], Mapping[str, Any]] | None = None,
    ) -> None:
        if int(env.num_envs) != 1:
            raise ValueError("Paired causal v2 forks require one environment.")
        if not isinstance(env.action_contract, LiberoActionContract):
            raise TypeError("Causal v2 forks require the live LIBERO Action contract.")
        if (
            env.action_contract.low != (-1.0,) * 7
            or env.action_contract.high != (1.0,) * 7
        ):
            raise RuntimeError("Causal v2 requires the live [-1,1]^7 contract.")
        self.env = env
        self.runtime = runtime
        self.action_contract = env.action_contract
        self.restore_policy_runtime = restore_policy_runtime
        self.restore_history = restore_history
        self.observe_branch = observe_branch
        self.secondary_outcomes = secondary_outcomes or (lambda _env: {})

    def _restore(self, snapshot: CausalSnapshotV2) -> dict[str, Any]:
        runtime_snapshot = snapshot.runtime_snapshot
        observation = self.env.restore_causal_snapshot(snapshot)
        self.restore_policy_runtime(
            copy.deepcopy(runtime_snapshot.policy_runtime_state)
        )
        self.restore_history(copy.deepcopy(runtime_snapshot.recent_history))
        return self._instrument_observation(observation)

    def _instrument_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(observation)
        result["_fastwam_action_contract_low"] = torch.tensor(
            self.action_contract.low, dtype=torch.float32
        ).unsqueeze(0)
        result["_fastwam_action_contract_high"] = torch.tensor(
            self.action_contract.high, dtype=torch.float32
        ).unsqueeze(0)
        result["_fastwam_action_gripper_indices"] = torch.tensor(
            [self.action_contract.gripper_dimension_index], dtype=torch.long
        )
        result["_fastwam_action_contract_sha256"] = [
            self.action_contract.canonical_sha256
        ]
        return result

    @staticmethod
    def _actions_as_tuple(actions: torch.Tensor) -> tuple[tuple[float, ...], ...]:
        array = actions.detach().float().cpu().numpy()
        if array.ndim != 3 or array.shape[0] != 1 or array.shape[-1] != 7:
            raise ValueError("Submitted actions must have shape [1,T,7].")
        if not np.isfinite(array).all():
            raise ValueError("Submitted actions must be finite.")
        return tuple(tuple(float(value) for value in row) for row in array[0])

    def _step(self, sample: CausalChunkSample):
        if sample.action_execution_trace is None:
            raise RuntimeError("Causal action conversion produced no Action trace.")
        result, submitted = self.env.chunk_step_with_action_trace(
            sample.actions,
            self.action_contract,
        )
        trace = ActionExecutionTrace.combine(
            sample.action_execution_trace,
            ActionExecutionTrace((submitted,)),
        )
        return result, trace

    @staticmethod
    def _derived_chunk_spec(
        spec: CausalInterventionSpecV2,
        chunk_offset: int,
    ) -> CausalInterventionSpecV2:
        return replace(
            spec,
            action_seed=spec.action_seed + 10_007 * chunk_offset,
            video_seed=(
                None
                if spec.video_seed is None
                else spec.video_seed + 10_009 * chunk_offset
            ),
            generic_medoid_index=None,
        )

    @staticmethod
    def _trace_artifact(traces: Sequence[ActionExecutionTrace]) -> Mapping[str, Any]:
        if not traces:
            return {}
        return ActionExecutionTrace.merge_time(traces).record_for_batch_index(0)

    def _execute_branch(
        self,
        *,
        snapshot: CausalSnapshotV2,
        spec: CausalInterventionSpecV2,
        donor_future_latents: torch.Tensor | None,
        prediction_context: tuple[torch.Tensor, torch.Tensor] | None,
        ground_truth_future_latents: torch.Tensor | None,
    ) -> PairedInterventionRecordV2:
        observation = self._restore(snapshot)
        before = self.observe_branch(self.env)
        treatment_actions: list[tuple[float, ...]] = []
        continuation_actions: list[tuple[float, ...]] = []
        treatment_latency: dict[str, float] = {}
        continuation_latency: dict[str, float] = {}
        treatment_calls: dict[str, int] = {}
        continuation_calls: dict[str, int] = {}
        treatment_traces: list[ActionExecutionTrace] = []
        continuation_traces: list[ActionExecutionTrace] = []
        max_steps = int(self.runtime.action_protocol.max_episode_steps)
        if (
            bool(self.env.success_once[0])
            or int(self.env.elapsed_steps[0]) >= max_steps
        ):
            raise ValueError("Causal treatment snapshots must precede termination.")
        treatment_medoid_index = None

        for chunk_offset in range(spec.treatment_chunks):
            if (
                bool(self.env.success_once[0])
                or int(self.env.elapsed_steps[0]) >= max_steps
            ):
                break
            chunk_spec = self._derived_chunk_spec(spec, chunk_offset)
            sample = self.runtime.sample_causal_intervention(
                env_obs=observation,
                spec=chunk_spec,
                donor_future_latents=donor_future_latents,
                prediction_context=prediction_context,
                ground_truth_future_latents=ground_truth_future_latents,
            )
            treatment_actions.extend(self._actions_as_tuple(sample.actions))
            treatment_medoid_index = sample.medoid_index
            _sum_metrics(treatment_latency, sample.latency_ms)
            _sum_calls(treatment_calls, sample)
            chunk_result, trace = self._step(sample)
            treatment_traces.append(trace)
            observations, *_ = chunk_result
            observation = self._instrument_observation(observations[-1])

        after_treatment = self.observe_branch(self.env)
        continuation_offset = 0
        while (
            not bool(self.env.success_once[0])
            and int(self.env.elapsed_steps[0]) < max_steps
        ):
            action_seed = spec.action_seed + 1_000_003 + 10_007 * continuation_offset
            video_seed = (
                (spec.video_seed if spec.video_seed is not None else 42_000_001)
                + 1_000_033
                + 10_009 * continuation_offset
            )
            continuation_spec = CausalInterventionSpecV2(
                mode=spec.continuation_mode,
                control=CausalControlKind.STANDARD,
                treatment_chunks=1,
                continuation_mode=spec.continuation_mode,
                replicate=spec.replicate,
                action_seed=action_seed,
                video_seed=(
                    video_seed
                    if spec.continuation_mode.runs_future_prediction
                    else None
                ),
            )
            sample = self.runtime.sample_causal_intervention(
                env_obs=observation,
                spec=continuation_spec,
            )
            continuation_actions.extend(self._actions_as_tuple(sample.actions))
            _sum_metrics(continuation_latency, sample.latency_ms)
            _sum_calls(continuation_calls, sample)
            chunk_result, trace = self._step(sample)
            continuation_traces.append(trace)
            observations, *_ = chunk_result
            observation = self._instrument_observation(observations[-1])
            continuation_offset += 1

        terminal = self.observe_branch(self.env)
        total_latency = {
            key: treatment_latency.get(key, 0.0) + continuation_latency.get(key, 0.0)
            for key in set(treatment_latency) | set(continuation_latency)
        }
        success = bool(self.env.success_once[0])
        elapsed = int(self.env.elapsed_steps[0])
        if success:
            termination = CausalTerminationType.SUCCESS
        elif elapsed >= max_steps:
            termination = CausalTerminationType.TIME_LIMIT
        elif bool(getattr(self.env, "done", [False])[0]):
            termination = CausalTerminationType.ENV_TERMINATION
        else:
            termination = CausalTerminationType.UNKNOWN
        first_success = None
        if success:
            values = np.asarray(self.env.success_episode_len)
            if values.size:
                first_success = int(values[0])
        outcome = CausalOutcomeV2(
            predicate_before=before.predicate_vector,
            predicate_after_treatment=after_treatment.predicate_vector,
            predicate_terminal=terminal.predicate_vector,
            progress_before=before.predicate_progress,
            progress_after_treatment=after_treatment.predicate_progress,
            progress_terminal=terminal.predicate_progress,
            final_success=success,
            final_return=float(self.env.returns[0]),
            first_success_step=first_success,
            completion_step=elapsed,
            termination_type=termination,
            contact_events=terminal.contact_events,
            treatment_submitted_action_count=len(treatment_actions),
            continuation_submitted_action_count=len(continuation_actions),
            treatment_action_audit=self._trace_artifact(treatment_traces),
            continuation_action_audit=self._trace_artifact(continuation_traces),
        )
        cost = CausalCostV2(
            treatment_latency_ms=treatment_latency,
            continuation_latency_ms=continuation_latency,
            total_latency_ms=total_latency,
            treatment_calls=treatment_calls,
            continuation_calls=continuation_calls,
            episode_gpu_seconds=float(total_latency.get("critical_path", 0.0)) / 1000.0,
        )
        secondary = dict(self.secondary_outcomes(self.env))
        forbidden = {
            "final_success",
            "final_return",
            "predicate_progress",
            "submitted_action_audit",
            "latency",
        }
        if forbidden & set(secondary):
            raise ValueError("Secondary outcomes duplicate a typed v2 core field.")
        recorded_spec = (
            replace(spec, generic_medoid_index=treatment_medoid_index)
            if spec.control is CausalControlKind.GENERIC_MEDOID
            else spec
        )
        return PairedInterventionRecordV2(
            identity=snapshot.identity,
            sampling=snapshot.sampling,
            intervention=recorded_spec,
            outcome=outcome,
            cost=cost,
            treatment_submitted_actions=tuple(treatment_actions),
            continuation_submitted_actions=tuple(continuation_actions),
            secondary_outcomes=secondary,
        )

    def run_snapshot(
        self,
        *,
        snapshot: CausalSnapshotV2,
        state_index: int,
        specs: Sequence[CausalInterventionSpecV2],
        intervention_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[PairedInterventionRecordV2]:
        """Run randomized v2 branches, restoring the snapshot before each one."""

        if state_index < 0 or not specs:
            raise ValueError("V2 state index/specs are invalid.")
        keys = [
            (
                spec.mode.value,
                spec.control.value,
                spec.treatment_chunks,
                spec.replicate,
                spec.continuation_mode.value,
            )
            for spec in specs
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("V2 branch specifications contain duplicates.")
        ordered = list(specs)
        random.Random(42_073 + state_index).shuffle(ordered)
        inputs = intervention_inputs or {}
        records = []
        for spec in ordered:
            branch_inputs = inputs.get(spec.control.value, {})
            records.append(
                self._execute_branch(
                    snapshot=snapshot,
                    spec=spec,
                    donor_future_latents=branch_inputs.get("donor_future_latents"),
                    prediction_context=branch_inputs.get("prediction_context"),
                    ground_truth_future_latents=branch_inputs.get(
                        "ground_truth_future_latents"
                    ),
                )
            )
        return records
