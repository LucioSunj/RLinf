# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Snapshot-to-record orchestration for causal LIBERO branch execution."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from fastwam.causal_prediction import (
    CausalComputeMode,
    PairedInterventionRecordV1,
)

from rlinf.envs.action_contract import ActionExecutionTrace
from rlinf.envs.libero.action_contract import LiberoActionContract
from rlinf.envs.libero.causal_snapshot import CausalSnapshotV1

from .causal_runtime import CausalLiberoFastWAMRuntime


class PairedCausalForkRunner:
    """Run randomized restored branches with common-random action/video seeds."""

    def __init__(
        self,
        *,
        env,
        runtime: CausalLiberoFastWAMRuntime,
        restore_policy_runtime: Callable[[Mapping[str, Any]], None],
        restore_history: Callable[[Sequence[Mapping[str, Any]]], None],
        secondary_outcomes: Callable[[Any], Mapping[str, Any]] | None = None,
    ) -> None:
        if int(env.num_envs) != 1:
            raise ValueError("Paired causal forks require one LIBERO environment.")
        self.env = env
        self.action_contract = env.action_contract
        if not isinstance(self.action_contract, LiberoActionContract):
            raise TypeError("Causal forks require a typed live LIBERO Action contract.")
        if (
            self.action_contract.low != (-1.0,) * 7
            or self.action_contract.high != (1.0,) * 7
        ):
            raise RuntimeError(
                "Causal v1 requires the live LIBERO Action contract [-1,1]^7."
            )
        self.runtime = runtime
        self.restore_policy_runtime = restore_policy_runtime
        self.restore_history = restore_history
        self.secondary_outcomes = secondary_outcomes or (lambda _env: {})

    @staticmethod
    def treatment_seeds(state_index: int, replicate: int) -> tuple[int, int]:
        """Return the frozen action/video common-random seed pair."""

        if state_index < 0 or replicate < 0:
            raise ValueError("State and replicate indices must be non-negative.")
        return (
            42 + 1_000_003 * state_index + 101 * replicate,
            42_000_001 + 1_000_033 * state_index + 103 * replicate,
        )

    @staticmethod
    def continuation_seeds(
        state_index: int,
        replicate: int,
        chunk_offset: int,
    ) -> tuple[int, int]:
        """Return a branch-independent continuation seed schedule."""

        action, video = PairedCausalForkRunner.treatment_seeds(
            state_index,
            replicate,
        )
        return action + 10_007 * (chunk_offset + 1), video + 10_009 * (chunk_offset + 1)

    def _restore(self, snapshot: CausalSnapshotV1) -> dict[str, Any]:
        observation = self.env.restore_causal_snapshot(snapshot)
        self.restore_policy_runtime(copy.deepcopy(snapshot.policy_runtime_state))
        self.restore_history(copy.deepcopy(snapshot.recent_history))
        return observation

    def _instrument_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Attach the exact live Action contract before policy conversion."""

        result = dict(observation)
        result["_fastwam_action_contract_low"] = torch.tensor(
            self.action_contract.low,
            dtype=torch.float32,
        ).unsqueeze(0)
        result["_fastwam_action_contract_high"] = torch.tensor(
            self.action_contract.high,
            dtype=torch.float32,
        ).unsqueeze(0)
        result["_fastwam_action_gripper_indices"] = torch.tensor(
            [self.action_contract.gripper_dimension_index],
            dtype=torch.long,
        )
        result["_fastwam_action_contract_sha256"] = [
            self.action_contract.canonical_sha256
        ]
        return result

    def _step_audited_chunk(self, sample):
        """Submit one chunk through the fail-closed live bounds audit."""

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
    def _actions_as_tuple(actions: torch.Tensor) -> tuple[tuple[float, ...], ...]:
        array = actions.detach().float().cpu().numpy()
        if array.shape[0] != 1 or array.shape[-1] != 7:
            raise ValueError("Recorded submitted actions must be [1,T,7].")
        if not np.isfinite(array).all():
            raise ValueError("Refusing to record non-finite submitted actions.")
        return tuple(tuple(float(value) for value in row) for row in array[0])

    def _execute_branch(
        self,
        *,
        snapshot: CausalSnapshotV1,
        state_index: int,
        mode: CausalComputeMode,
        replicate: int,
        continuation: CausalComputeMode,
        inclusion_probability: float,
    ) -> PairedInterventionRecordV1:
        observation = self._instrument_observation(self._restore(snapshot))
        action_seed, video_seed = self.treatment_seeds(state_index, replicate)
        sample = self.runtime.sample_causal_action(
            env_obs=observation,
            mode=mode,
            action_seed=action_seed,
            video_seed=video_seed if mode.runs_future_prediction else None,
        )
        submitted = list(self._actions_as_tuple(sample.actions))
        latency = dict(sample.latency_ms)
        chunk_result, trace = self._step_audited_chunk(sample)
        chunk_traces = [trace]
        observations, *_ = chunk_result
        observation = self._instrument_observation(observations[-1])
        chunk_offset = 0
        max_steps = int(self.runtime.action_protocol.max_episode_steps)
        while (
            not bool(self.env.success_once[0])
            and int(self.env.elapsed_steps[0]) < max_steps
        ):
            next_action_seed, next_video_seed = self.continuation_seeds(
                state_index,
                replicate,
                chunk_offset,
            )
            next_sample = self.runtime.sample_causal_action(
                env_obs=observation,
                mode=continuation,
                action_seed=next_action_seed,
                video_seed=(
                    next_video_seed if continuation.runs_future_prediction else None
                ),
            )
            submitted.extend(self._actions_as_tuple(next_sample.actions))
            for name, value in next_sample.latency_ms.items():
                latency[name] = latency.get(name, 0.0) + float(value)
            chunk_result, trace = self._step_audited_chunk(next_sample)
            chunk_traces.append(trace)
            observations, *_ = chunk_result
            observation = self._instrument_observation(observations[-1])
            chunk_offset += 1
        secondary = dict(self.secondary_outcomes(self.env))
        if "submitted_action_audit" in secondary:
            raise ValueError("Secondary outcomes cannot replace the live Action audit.")
        secondary["submitted_action_audit"] = ActionExecutionTrace.merge_time(
            chunk_traces
        ).record_for_batch_index(0)
        return PairedInterventionRecordV1(
            snapshot_id=snapshot.snapshot_id,
            environment="libero",
            task_id=int(self.env.task_ids[0]),
            trial_id=int(self.env.trial_ids[0]),
            reset_id=int(self.env.reset_state_ids[0]),
            chunk_index=int(snapshot.chunk_index),
            mode=mode,
            replicate=int(replicate),
            action_seed=action_seed,
            video_seed=video_seed if mode.runs_future_prediction else None,
            continuation=f"always_{continuation.value.split('_')[0]}",
            source_policy=snapshot.source_policy,
            inclusion_probability=float(inclusion_probability),
            final_success=bool(self.env.success_once[0]),
            final_return=float(self.env.returns[0]),
            remaining_steps=max(0, max_steps - int(self.env.elapsed_steps[0])),
            submitted_action_count=len(submitted),
            submitted_actions=tuple(submitted),
            latency_ms=latency,
            secondary_outcomes=secondary,
        )

    def run_snapshot(
        self,
        *,
        snapshot: CausalSnapshotV1,
        state_index: int,
        modes: Sequence[CausalComputeMode | str],
        replicates: int,
        continuation: CausalComputeMode | str = CausalComputeMode.C2_FULL,
        inclusion_probability: float,
    ) -> list[PairedInterventionRecordV1]:
        """Execute all branches after randomized ordering and per-branch restore."""

        if replicates < 1:
            raise ValueError("Causal branch replicate count must be positive.")
        parsed_modes = tuple(CausalComputeMode.parse(mode) for mode in modes)
        if len(set(parsed_modes)) != len(parsed_modes):
            raise ValueError("Causal treatment modes must be unique.")
        selected_continuation = CausalComputeMode.parse(continuation)
        if not selected_continuation.is_routable:
            raise ValueError("A negative control cannot be a continuation policy.")
        branch_specs = [
            (mode, replicate)
            for replicate in range(replicates)
            for mode in parsed_modes
        ]
        order_rng = random.Random(42_073 + int(state_index))
        order_rng.shuffle(branch_specs)
        return [
            self._execute_branch(
                snapshot=snapshot,
                state_index=state_index,
                mode=mode,
                replicate=replicate,
                continuation=selected_continuation,
                inclusion_probability=inclusion_probability,
            )
            for mode, replicate in branch_specs
        ]
