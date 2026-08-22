# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Two-pass source replay and exact snapshot capture for Stage C."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from fastwam.causal_prediction import CausalComputeMode

from rlinf.envs.libero.causal_collection import (
    StageCSnapshotAuditTargetV1,
    select_stage_c_snapshot_audit_targets,
)
from rlinf.envs.libero.causal_snapshot import (
    CausalSnapshotV1,
    assert_exact_causal_replay,
    audit_interleaved_snapshot_restore,
)


@dataclass(frozen=True)
class StageCSourceChunkV1:
    """One pre-action source chunk used only for mechanical target selection."""

    chunk_index: int
    elapsed_steps: int
    eligible: bool
    contact_active: bool
    raw_observation: Mapping[str, Any]
    task_observation: Mapping[str, Any]
    submitted_actions: tuple[tuple[float, ...], ...]
    submitted_action_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.chunk_index < 0 or self.elapsed_steps < 0:
            raise ValueError("Stage-C source counters must be non-negative.")
        if not self.submitted_actions:
            raise ValueError(
                "Stage-C source chunk must submit one complete action block."
            )
        if bool(self.task_observation.get("contact_active")) != self.contact_active:
            raise ValueError("Stage-C source contact fields disagree.")

    def to_lightweight_artifact(self) -> dict[str, Any]:
        """Return source mechanics without raw observations or Action values."""

        return {
            "schema": "causal-stage-c-source-chunk-v1",
            "chunk_index": self.chunk_index,
            "elapsed_steps": self.elapsed_steps,
            "eligible": self.eligible,
            "contact_active": self.contact_active,
            "predicate_vector": list(self.task_observation["predicate_vector"]),
            "predicate_progress": float(self.task_observation["predicate_progress"]),
            "submitted_action_count": len(self.submitted_actions),
            "submitted_action_audit": dict(self.submitted_action_audit),
        }


@dataclass(frozen=True)
class StageCCapturedTargetV1:
    """One selected phase bound to its exact replayed runtime snapshot."""

    phase: str
    source: StageCSourceChunkV1
    snapshot: CausalSnapshotV1

    def __post_init__(self) -> None:
        if self.snapshot.chunk_index != self.source.chunk_index:
            raise ValueError("Stage-C target snapshot and source chunk differ.")


@dataclass(frozen=True)
class StageCSourceReplayResultV1:
    """Complete first-pass trace plus three exact second-pass snapshots."""

    source_chunks: tuple[StageCSourceChunkV1, ...]
    targets: tuple[StageCCapturedTargetV1, ...]
    final_outcome: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.targets) != 3:
            raise ValueError("Stage-C source replay requires exactly three targets.")

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible replay summary without snapshot payloads."""

        return {
            "schema": "causal-stage-c-source-replay-v1",
            "status": "PASS",
            "scientific_results": "NOT-RUN",
            "source_chunk_count": len(self.source_chunks),
            "source_chunks": [
                chunk.to_lightweight_artifact() for chunk in self.source_chunks
            ],
            "targets": [
                {
                    "phase": target.phase,
                    "snapshot_id": target.snapshot.snapshot_id,
                    "chunk_index": target.source.chunk_index,
                    "contact_active": target.source.contact_active,
                }
                for target in self.targets
            ],
            "final_outcome": dict(self.final_outcome),
        }


class StageCSourceReplayExecutorV1:
    """Run one Always-C2 source episode and exact-replay three target snapshots."""

    def __init__(
        self,
        *,
        env: Any,
        runtime: Any,
        seed_schedule: Callable[[int], tuple[int, int]],
        capture_policy_runtime: Callable[[], Mapping[str, Any]],
        restore_policy_runtime: Callable[[Mapping[str, Any]], None],
        execution_horizon: int = 10,
        max_steps: int = 700,
    ) -> None:
        if int(getattr(env, "num_envs", 0)) != 1:
            raise ValueError("Stage-C source replay requires one environment.")
        if execution_horizon != 10 or max_steps != 700:
            raise ValueError("Stage-C action horizon or episode limit changed.")
        self.env = env
        self.runtime = runtime
        self.seed_schedule = seed_schedule
        self.capture_policy_runtime = capture_policy_runtime
        self.restore_policy_runtime = restore_policy_runtime
        self.execution_horizon = execution_horizon
        self.max_steps = max_steps

    @staticmethod
    def _actions_as_tuple(actions: torch.Tensor) -> tuple[tuple[float, ...], ...]:
        array = actions.detach().float().cpu().numpy()
        if array.shape != (1, 10, 7) or not np.isfinite(array).all():
            raise ValueError("Stage-C submitted Actions must be finite [1,10,7].")
        return tuple(tuple(float(value) for value in row) for row in array[0])

    @staticmethod
    def _latest_observation(step_result: Any) -> dict[str, Any]:
        observations = step_result[0]
        if not isinstance(observations, Sequence) or not observations:
            raise RuntimeError("Stage-C LIBERO chunk returned no observation.")
        return dict(observations[-1])

    def _raw_observation(self) -> Mapping[str, Any]:
        current = getattr(self.env, "current_raw_obs", None)
        if not isinstance(current, Sequence) or len(current) != 1:
            raise RuntimeError("Stage-C environment lacks one current raw observation.")
        return copy.deepcopy(current[0])

    def _final_outcome(self) -> dict[str, Any]:
        task = dict(self.env.observe_causal_task_state())
        return {
            "final_success": bool(self.env.success_once[0]),
            "final_return": float(self.env.returns[0]),
            "elapsed_steps": int(self.env.elapsed_steps[0]),
            "task_observation": task,
        }

    @staticmethod
    def _history_row(chunk_index: int, *, success: bool) -> dict[str, Any]:
        return {
            "chunk_index": chunk_index,
            "executed_mode": CausalComputeMode.C2_FULL.value,
            "submitted_action_count": 10,
            "success": success,
        }

    def _sample_and_step(
        self,
        observation: Mapping[str, Any],
        *,
        chunk_index: int,
    ) -> tuple[
        dict[str, Any],
        tuple[tuple[float, ...], ...],
        Mapping[str, Any],
    ]:
        action_seed, video_seed = self.seed_schedule(chunk_index)
        sample = self.runtime.sample_causal_action(
            env_obs=dict(observation),
            mode=CausalComputeMode.C2_FULL,
            action_seed=action_seed,
            video_seed=video_seed,
        )
        actions = self._actions_as_tuple(sample.actions)
        step_result, submitted = self.env.chunk_step_with_action_trace(
            sample.actions,
            self.env.action_contract,
        )
        return (
            self._latest_observation(step_result),
            actions,
            submitted.record_for_batch_index(0),
        )

    def _capture_snapshot(
        self,
        *,
        snapshot_id: str,
        history: Sequence[Mapping[str, Any]],
        chunk_index: int,
    ) -> CausalSnapshotV1:
        return self.env.capture_causal_snapshot(
            snapshot_id=snapshot_id,
            recent_history=tuple(copy.deepcopy(tuple(history[-4:]))),
            policy_runtime_state=copy.deepcopy(self.capture_policy_runtime()),
            source_policy="always_c2",
            previous_mode=(
                None if chunk_index == 0 else CausalComputeMode.C2_FULL.value
            ),
            chunk_index=chunk_index,
            remaining_budget=float(
                max(0, self.max_steps - int(self.env.elapsed_steps[0]))
            ),
        )

    def _run_source_pass(
        self,
        initial_observation: Mapping[str, Any],
    ) -> tuple[tuple[StageCSourceChunkV1, ...], Mapping[str, Any]]:
        observation = dict(initial_observation)
        chunks = []
        chunk_index = 0
        while (
            not bool(self.env.success_once[0])
            and int(self.env.elapsed_steps[0]) < self.max_steps
        ):
            elapsed = int(self.env.elapsed_steps[0])
            raw = self._raw_observation()
            task = dict(self.env.observe_causal_task_state())
            observation, actions, action_audit = self._sample_and_step(
                observation,
                chunk_index=chunk_index,
            )
            chunks.append(
                StageCSourceChunkV1(
                    chunk_index=chunk_index,
                    elapsed_steps=elapsed,
                    eligible=(self.max_steps - elapsed) >= self.execution_horizon,
                    contact_active=bool(task["contact_active"]),
                    raw_observation=raw,
                    task_observation=task,
                    submitted_actions=actions,
                    submitted_action_audit=action_audit,
                )
            )
            chunk_index += 1
        return tuple(chunks), self._final_outcome()

    def _restore(self, snapshot: CausalSnapshotV1) -> dict[str, Any]:
        observation = self.env.restore_causal_snapshot(snapshot)
        self.restore_policy_runtime(copy.deepcopy(snapshot.policy_runtime_state))
        return dict(observation)

    def _replay_and_capture(
        self,
        *,
        initial_snapshot: CausalSnapshotV1,
        source_chunks: Sequence[StageCSourceChunkV1],
        selected: Sequence[StageCSnapshotAuditTargetV1],
        expected_final: Mapping[str, Any],
    ) -> tuple[StageCCapturedTargetV1, ...]:
        observation = self._restore(initial_snapshot)
        history: list[Mapping[str, Any]] = []
        selected_by_chunk = {
            target.trace.chunk_index: target.phase for target in selected
        }
        captured = []
        for source in source_chunks:
            raw = self._raw_observation()
            task = dict(self.env.observe_causal_task_state())
            assert_exact_causal_replay(
                source.raw_observation,
                raw,
                path=f"stage_c_source.raw[{source.chunk_index}]",
            )
            assert_exact_causal_replay(
                source.task_observation,
                task,
                path=f"stage_c_source.task[{source.chunk_index}]",
            )
            phase = selected_by_chunk.get(source.chunk_index)
            if phase is not None:
                snapshot = self._capture_snapshot(
                    snapshot_id=(
                        f"stage-c-task{int(self.env.task_ids[0])}-"
                        f"trial{int(self.env.trial_ids[0])}-{phase}-"
                        f"chunk{source.chunk_index}"
                    ),
                    history=history,
                    chunk_index=source.chunk_index,
                )
                captured.append(
                    StageCCapturedTargetV1(
                        phase=phase,
                        source=source,
                        snapshot=snapshot,
                    )
                )
            observation, actions, action_audit = self._sample_and_step(
                observation,
                chunk_index=source.chunk_index,
            )
            assert_exact_causal_replay(
                source.submitted_actions,
                actions,
                path=f"stage_c_source.actions[{source.chunk_index}]",
            )
            assert_exact_causal_replay(
                source.submitted_action_audit,
                action_audit,
                path=f"stage_c_source.action_audit[{source.chunk_index}]",
            )
            history.append(
                self._history_row(
                    source.chunk_index,
                    success=bool(self.env.success_once[0]),
                )
            )
            history = history[-4:]
        assert_exact_causal_replay(
            expected_final,
            self._final_outcome(),
            path="stage_c_source.final_outcome",
        )
        phase_order = {
            phase: index for index, phase in enumerate(("early", "mid", "contact"))
        }
        return tuple(sorted(captured, key=lambda item: phase_order[item.phase]))

    def run(
        self,
        *,
        initial_observation: Mapping[str, Any],
    ) -> StageCSourceReplayResultV1:
        """Run, select, exact-replay, and capture the three frozen targets."""

        initial_snapshot = self._capture_snapshot(
            snapshot_id="stage-c-source-start",
            history=(),
            chunk_index=0,
        )
        source_chunks, final_outcome = self._run_source_pass(initial_observation)
        selected = select_stage_c_snapshot_audit_targets(source_chunks)
        targets = self._replay_and_capture(
            initial_snapshot=initial_snapshot,
            source_chunks=source_chunks,
            selected=selected,
            expected_final=final_outcome,
        )
        return StageCSourceReplayResultV1(
            source_chunks=source_chunks,
            targets=targets,
            final_outcome=final_outcome,
        )


class StageCInterleavingAuditExecutorV1:
    """Execute the two frozen restore orders on captured Stage-C targets.

    Each branch executes one C0/C2 treatment chunk, captures its direct trace,
    runs a fixed C2 continuation to completion, and then restores the exact
    post-treatment state. Leaving the environment at that post-treatment state
    makes the subsequent B branch in ``A-B-restore-A`` a real interleaving,
    while retaining the complete continuation as part of the A trace.
    """

    def __init__(
        self,
        *,
        env: Any,
        runtime: Any,
        seed_schedule: Callable[[int], tuple[int, int]],
        capture_policy_runtime: Callable[[], Mapping[str, Any]],
        restore_policy_runtime: Callable[[Mapping[str, Any]], None],
        execution_horizon: int = 10,
        max_steps: int = 700,
    ) -> None:
        if int(getattr(env, "num_envs", 0)) != 1:
            raise ValueError("Stage-C interleaving audit requires one environment.")
        if execution_horizon != 10 or max_steps != 700:
            raise ValueError("Stage-C audit action horizon or episode limit changed.")
        self.env = env
        self.runtime = runtime
        self.seed_schedule = seed_schedule
        self.capture_policy_runtime = capture_policy_runtime
        self.restore_policy_runtime = restore_policy_runtime
        self.execution_horizon = execution_horizon
        self.max_steps = max_steps
        self._target: StageCCapturedTargetV1 | None = None
        self._observation: dict[str, Any] | None = None
        self._history: list[Mapping[str, Any]] = []
        self._chunk_index = 0
        self._previous_mode: CausalComputeMode | None = None

    @staticmethod
    def _batch_row(value: Any, *, label: str) -> tuple[Any, ...]:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.ndim < 1 or int(array.shape[0]) != 1:
            raise ValueError(f"Stage-C {label} must preserve batch size one.")
        return tuple(array[0].tolist())

    def _raw_observation(self) -> Mapping[str, Any]:
        current = getattr(self.env, "current_raw_obs", None)
        if not isinstance(current, Sequence) or len(current) != 1:
            raise RuntimeError("Stage-C audit lacks one current raw observation.")
        return copy.deepcopy(current[0])

    def _restore_full(self, snapshot: CausalSnapshotV1) -> None:
        self._observation = dict(self.env.restore_causal_snapshot(snapshot))
        self.restore_policy_runtime(copy.deepcopy(snapshot.policy_runtime_state))
        self._history = list(copy.deepcopy(snapshot.recent_history))
        self._chunk_index = int(snapshot.chunk_index)
        self._previous_mode = (
            None
            if snapshot.previous_mode is None
            else CausalComputeMode.parse(snapshot.previous_mode)
        )

    def _restore_target(self) -> None:
        if self._target is None:
            raise RuntimeError("Stage-C audit has no active target.")
        self._restore_full(self._target.snapshot)

    def _restore_target_simulator_only(self) -> None:
        if self._target is None:
            raise RuntimeError("Stage-C audit has no active target.")
        self.env.restore_causal_simulator_only_for_audit(self._target.snapshot)

    def _capture_post_treatment(
        self,
        *,
        mode: CausalComputeMode,
    ) -> CausalSnapshotV1:
        target = self._target
        if target is None:
            raise RuntimeError("Stage-C audit has no active target.")
        return self.env.capture_causal_snapshot(
            snapshot_id=(
                f"{target.snapshot.snapshot_id}-audit-{mode.value}-"
                f"post-chunk{self._chunk_index - 1}"
            ),
            recent_history=tuple(copy.deepcopy(self._history[-4:])),
            policy_runtime_state=copy.deepcopy(self.capture_policy_runtime()),
            source_policy=target.snapshot.source_policy,
            previous_mode=mode.value,
            chunk_index=self._chunk_index,
            remaining_budget=float(
                max(0, self.max_steps - int(self.env.elapsed_steps[0]))
            ),
        )

    def _execute_chunk(
        self,
        mode: CausalComputeMode,
    ) -> tuple[dict[str, Any], CausalSnapshotV1]:
        if self._observation is None:
            raise RuntimeError("Stage-C audit target has not been restored.")
        if (
            bool(self.env.success_once[0])
            or int(self.env.elapsed_steps[0]) >= self.max_steps
        ):
            raise RuntimeError("Stage-C audit cannot execute from a terminal state.")

        raw_observation = self._raw_observation()
        action_seed, video_seed = self.seed_schedule(self._chunk_index)
        sample = self.runtime.sample_causal_action(
            env_obs=dict(self._observation),
            mode=mode,
            action_seed=action_seed,
            video_seed=video_seed if mode.runs_future_prediction else None,
        )
        actions = StageCSourceReplayExecutorV1._actions_as_tuple(sample.actions)
        step_result, submitted = self.env.chunk_step_with_action_trace(
            sample.actions,
            self.env.action_contract,
        )
        self._observation = StageCSourceReplayExecutorV1._latest_observation(
            step_result
        )
        self._history.append(
            StageCSourceReplayExecutorV1._history_row(
                self._chunk_index,
                success=bool(self.env.success_once[0]),
            )
        )
        self._history = self._history[-4:]
        self._previous_mode = mode
        self._chunk_index += 1
        post_treatment = self._capture_post_treatment(mode=mode)
        worker_state = dict(post_treatment.worker_state)
        if "sim" not in worker_state:
            raise RuntimeError("Stage-C worker snapshot has no simulator state.")
        metrics = {
            "submitted_action_audit": submitted.record_for_batch_index(0),
            "terminations": self._batch_row(
                step_result[2],
                label="termination trace",
            ),
            "truncations": self._batch_row(
                step_result[3],
                label="truncation trace",
            ),
            "infos": copy.deepcopy(step_result[4]),
            "task_observation": copy.deepcopy(self.env.observe_causal_task_state()),
            "post_raw_observation": self._raw_observation(),
            "wrapper_state": copy.deepcopy(post_treatment.wrapper_state),
            "worker_non_simulator_state": {
                key: copy.deepcopy(value)
                for key, value in worker_state.items()
                if key != "sim"
            },
            "recent_history": copy.deepcopy(post_treatment.recent_history),
            "policy_runtime_state": copy.deepcopy(post_treatment.policy_runtime_state),
            "driver_rng_state": copy.deepcopy(post_treatment.driver_rng_state),
        }
        trace = {
            "raw_observation": raw_observation,
            "submitted_actions": actions,
            "next_simulator_state": copy.deepcopy(worker_state["sim"]),
            "reward": self._batch_row(step_result[1], label="reward trace"),
            "success": bool(self.env.success_once[0]),
            "metrics": metrics,
        }
        return trace, post_treatment

    def _run_fixed_continuation(self) -> dict[str, Any]:
        chunks = []
        while (
            not bool(self.env.success_once[0])
            and int(self.env.elapsed_steps[0]) < self.max_steps
        ):
            trace, _post = self._execute_chunk(CausalComputeMode.C2_FULL)
            chunks.append(trace)
        return {
            "mode": CausalComputeMode.C2_FULL.value,
            "chunks": tuple(chunks),
            "final_raw_observation": self._raw_observation(),
            "final_task_observation": copy.deepcopy(
                self.env.observe_causal_task_state()
            ),
            "final_success": bool(self.env.success_once[0]),
            "final_return": float(self.env.returns[0]),
            "elapsed_steps": int(self.env.elapsed_steps[0]),
            "recent_history": copy.deepcopy(tuple(self._history)),
            "policy_runtime_state": copy.deepcopy(self.capture_policy_runtime()),
        }

    def _run_branch(self, mode: str) -> dict[str, Any]:
        selected = CausalComputeMode.parse(mode)
        if selected not in {
            CausalComputeMode.C0_CURRENT,
            CausalComputeMode.C2_FULL,
        }:
            raise ValueError("Stage-C audit branches are frozen to C0 and C2.")
        treatment, post_treatment = self._execute_chunk(selected)
        continuation = self._run_fixed_continuation()
        self._restore_full(post_treatment)
        return {**treatment, "continuation_outcome": continuation}

    def run_target(self, target: StageCCapturedTargetV1) -> dict[str, Any]:
        """Return the exact interleaving audit for one frozen phase target."""

        if target.phase not in {"early", "mid", "contact"}:
            raise ValueError(f"Unknown Stage-C phase {target.phase!r}.")
        self._target = target
        try:
            audit = audit_interleaved_snapshot_restore(
                restore=self._restore_target,
                restore_simulator_only=self._restore_target_simulator_only,
                run_branch=self._run_branch,
                mode_a=CausalComputeMode.C0_CURRENT.value,
                mode_b=CausalComputeMode.C2_FULL.value,
                phase=target.phase,
            )
            self._restore_target()
            return audit
        finally:
            self._target = None

    def run_targets(
        self,
        targets: Sequence[StageCCapturedTargetV1],
    ) -> tuple[dict[str, Any], ...]:
        """Return three report-ready target rows in frozen phase order."""

        if tuple(target.phase for target in targets) != ("early", "mid", "contact"):
            raise ValueError("Stage-C audit requires early/mid/contact target order.")
        return tuple(
            {
                "phase": target.phase,
                "snapshot_id": target.snapshot.snapshot_id,
                "chunk_index": target.source.chunk_index,
                "contact_active": target.source.contact_active,
                "audit": self.run_target(target),
            }
            for target in targets
        )
