# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Source-trace selection, exact replay, and donor matching for causal v2."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from fastwam.causal_prediction import (
    CausalPhase,
    CausalSamplingMetadataV2,
    CausalSamplingStratum,
    CausalStateIdentityV2,
    validate_pre_prediction_feature_payload,
)

from rlinf.envs.libero.causal_snapshot import CausalSnapshotV2

CRITICALITY_COMPONENT_NAMES = (
    "action_curvature",
    "action_precision",
    "contact_proximity",
    "gripper_transition",
    "predicate_transition",
)
STAGE_C_SNAPSHOT_AUDIT_PHASES = ("early", "mid", "contact")


class StageCSnapshotAuditTrace(Protocol):
    """Minimum source-trace surface used by the mechanical Stage-C selector."""

    chunk_index: int
    eligible: bool
    contact_active: bool


@dataclass(frozen=True)
class SourceChunkTraceV2:
    """One in-memory source chunk plus its lightweight persisted summary."""

    chunk_index: int
    elapsed_steps: int
    eligible: bool
    terminated: bool
    remaining_action_capacity: int
    raw_observation: Mapping[str, Any]
    submitted_actions: tuple[tuple[float, ...], ...]
    predicate_vector: tuple[bool, ...]
    predicate_progress: float
    criticality_components: Mapping[str, float]
    contact_active: bool
    previous_contact_active: bool
    gripper_closing: bool
    nearest_task_object_distance_m: float | None
    predicate_changed: bool
    lightweight_summary: Mapping[str, Any]
    failure_adjacent: bool = False
    recovery: bool = False

    def __post_init__(self) -> None:
        if self.chunk_index < 0 or self.elapsed_steps < 0:
            raise ValueError("Source chunk counters must be non-negative.")
        if self.remaining_action_capacity < 0:
            raise ValueError("Source remaining action capacity must be non-negative.")
        expected_eligible = not self.terminated and self.remaining_action_capacity >= 10
        if self.eligible != expected_eligible:
            raise ValueError(
                "Eligible source chunks must be live with one complete treatment block."
            )
        if set(self.criticality_components) != set(CRITICALITY_COMPONENT_NAMES):
            raise ValueError("Source criticality component names changed.")
        if any(
            not np.isfinite(float(value))
            for value in self.criticality_components.values()
        ):
            raise ValueError("Source criticality components must be finite.")
        if not 0.0 <= float(self.predicate_progress) <= 1.0:
            raise ValueError("Predicate progress must lie in [0, 1].")
        if self.nearest_task_object_distance_m is not None and (
            not np.isfinite(float(self.nearest_task_object_distance_m))
            or float(self.nearest_task_object_distance_m) < 0
        ):
            raise ValueError("Object proximity must be finite and non-negative.")

    def to_lightweight_artifact(self) -> dict[str, Any]:
        """Return the source trace without raw images or simulator state."""

        return {
            "schema": "causal-source-chunk-trace-v2",
            "chunk_index": self.chunk_index,
            "elapsed_steps": self.elapsed_steps,
            "eligible": self.eligible,
            "terminated": self.terminated,
            "remaining_action_capacity": self.remaining_action_capacity,
            "submitted_actions": [list(action) for action in self.submitted_actions],
            "predicate_vector": list(self.predicate_vector),
            "predicate_progress": self.predicate_progress,
            "criticality_components": dict(self.criticality_components),
            "contact_active": self.contact_active,
            "previous_contact_active": self.previous_contact_active,
            "gripper_closing": self.gripper_closing,
            "nearest_task_object_distance_m": self.nearest_task_object_distance_m,
            "predicate_changed": self.predicate_changed,
            "failure_adjacent": self.failure_adjacent,
            "recovery": self.recovery,
            "summary": dict(self.lightweight_summary),
        }


@dataclass(frozen=True)
class SelectedSourceStateV2:
    """Selected chunk, replayed snapshot, and detached Gate feature payload."""

    trace: SourceChunkTraceV2
    snapshot: CausalSnapshotV2
    gate_features: Mapping[str, Any]
    source_traces: tuple[SourceChunkTraceV2, ...]


@dataclass(frozen=True)
class SourceEpisodeTraceV2:
    """Complete lightweight source trajectory and its directly observed result."""

    source_policy: str
    final_success: bool
    traces: tuple[SourceChunkTraceV2, ...]

    def __post_init__(self) -> None:
        if not self.source_policy or not self.traces:
            raise ValueError("Source episode policy and traces must be non-empty.")


@dataclass(frozen=True)
class StageCSnapshotAuditTargetV1:
    """One deterministically selected clean Stage-C replay target."""

    phase: str
    trace: StageCSnapshotAuditTrace

    def __post_init__(self) -> None:
        if self.phase not in STAGE_C_SNAPSHOT_AUDIT_PHASES:
            raise ValueError(f"Unknown Stage-C audit phase {self.phase!r}.")


@dataclass(frozen=True)
class FutureLatentEntryV1:
    """Metadata used to match a native future-latent donor."""

    state_id: str
    suite: str
    global_task_uid: str
    source_episode_id: str
    chunk_index: int
    criticality_bin: int
    instruction_id: str
    tensor_shard: str
    tensor_row: int


def _percentile_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("Percentile inputs must be finite and non-empty.")
    order = np.argsort(array, kind="stable")
    result = np.empty_like(array)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        result[order[start:end]] = ((start + end - 1) / 2 + 0.5) / array.size
        start = end
    return result


def classify_causal_phase(trace: SourceChunkTraceV2) -> CausalPhase:
    """Assign the preregistered mutually exclusive online phase."""

    if trace.predicate_changed:
        return CausalPhase.SUBGOAL_TRANSITION
    if trace.contact_active:
        return CausalPhase.CONTACT
    if trace.previous_contact_active:
        return CausalPhase.POST_CONTACT
    if (
        trace.nearest_task_object_distance_m is not None
        and trace.nearest_task_object_distance_m <= 0.08
        and trace.gripper_closing
    ):
        return CausalPhase.PRE_CONTACT
    return CausalPhase.TRANSIT


def select_stage_c_snapshot_audit_targets(
    traces: Sequence[StageCSnapshotAuditTrace],
) -> tuple[StageCSnapshotAuditTargetV1, ...]:
    """Select early, temporal-midpoint, and first-contact targets without outcomes."""

    eligible = [trace for trace in traces if trace.eligible]
    indices = [trace.chunk_index for trace in eligible]
    if indices != sorted(set(indices)):
        raise ValueError("Stage-C source traces must have unique increasing indices.")
    if not eligible:
        raise ValueError("Stage-C source episode has no eligible chunk.")
    early = eligible[0]
    contacts = [trace for trace in eligible if trace.contact_active]
    if not contacts:
        raise ValueError(
            "Frozen Stage-C source episode has no eligible contact state; do not "
            "reselect another task, trial, seed, or route."
        )
    contact = contacts[0]
    interior = [
        trace
        for trace in eligible
        if early.chunk_index < trace.chunk_index < contact.chunk_index
    ]
    if not interior:
        raise ValueError(
            "Frozen Stage-C source episode lacks a distinct mid state before contact."
        )
    midpoint = (early.chunk_index + contact.chunk_index) / 2.0
    mid = min(
        interior,
        key=lambda trace: (abs(trace.chunk_index - midpoint), trace.chunk_index),
    )
    return (
        StageCSnapshotAuditTargetV1(phase="early", trace=early),
        StageCSnapshotAuditTargetV1(phase="mid", trace=mid),
        StageCSnapshotAuditTargetV1(phase="contact", trace=contact),
    )


def select_source_chunk(
    traces: Sequence[SourceChunkTraceV2],
    *,
    stratum: CausalSamplingStratum | str,
    seed: int,
    source_policy: str,
    source_final_success: bool | None,
) -> tuple[SourceChunkTraceV2, CausalSamplingMetadataV2]:
    """Select one eligible state and return its exact design probability."""

    selected_stratum = CausalSamplingStratum(stratum)
    eligible = [trace for trace in traces if trace.eligible]
    if not eligible:
        raise ValueError("Source episode contains no eligible treatment chunk.")
    component_values = {
        name: [float(trace.criticality_components[name]) for trace in eligible]
        for name in CRITICALITY_COMPONENT_NAMES
    }
    percentiles = {
        name: _percentile_ranks(values) for name, values in component_values.items()
    }
    scores = np.asarray(
        [
            0.05 + float(np.mean([percentiles[name][index] for name in percentiles]))
            for index in range(len(eligible))
        ],
        dtype=np.float64,
    )
    conditional = (
        np.full(len(eligible), 1.0 / len(eligible), dtype=np.float64)
        if selected_stratum is CausalSamplingStratum.UNIFORM
        else scores / scores.sum()
    )
    rng = random.Random(int(seed))
    draw = rng.random()
    cumulative = 0.0
    selected_index = len(eligible) - 1
    for index, probability in enumerate(conditional):
        cumulative += float(probability)
        if draw < cumulative:
            selected_index = index
            break
    trace = eligible[selected_index]
    component_percentiles = {
        name: float(values[selected_index]) for name, values in percentiles.items()
    }
    probability = float(conditional[selected_index])
    metadata = CausalSamplingMetadataV2(
        source_policy=source_policy,
        source_final_success=source_final_success,
        sampling_stratum=selected_stratum,
        criticality_components=dict(trace.criticality_components),
        criticality_percentiles=component_percentiles,
        criticality_score=float(scores[selected_index]),
        eligible_chunk_count=len(eligible),
        conditional_selection_probability=probability,
        joint_inclusion_probability=0.5 * probability,
        phase=classify_causal_phase(trace),
        failure_adjacent=trace.failure_adjacent,
        recovery=trace.recovery,
    )
    return trace, metadata


def assert_source_replay_exact(
    source: SourceChunkTraceV2,
    replay: Mapping[str, Any],
) -> None:
    """Require exact selected observation, actions, and predicates on replay."""

    expected = {
        "raw_observation": source.raw_observation,
        "submitted_actions": source.submitted_actions,
        "predicate_vector": source.predicate_vector,
    }
    if set(replay) != set(expected):
        raise AssertionError("Source replay verification fields changed.")
    _assert_nested_equal(expected, replay, path="source_replay")


def _assert_nested_equal(left: Any, right: Any, *, path: str) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left, right)
        ):
            raise AssertionError(f"Replay tensor differs at {path}.")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right)
        ):
            raise AssertionError(f"Replay array differs at {path}.")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise AssertionError(f"Replay mapping keys differ at {path}.")
        for key in left:
            _assert_nested_equal(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise AssertionError(f"Replay sequence length differs at {path}.")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_equal(left_item, right_item, path=f"{path}[{index}]")
        return
    if type(left) is not type(right) or left != right:
        raise AssertionError(f"Replay value differs at {path}.")


class CausalSourceCollectorV2:
    """Execute one source episode, select one state, and exact-replay it."""

    def __init__(
        self,
        *,
        run_source_episode: Callable[[str], SourceEpisodeTraceV2],
        replay_to_snapshot: Callable[
            [SourceChunkTraceV2, CausalStateIdentityV2, CausalSamplingMetadataV2],
            tuple[CausalSnapshotV2, Mapping[str, Any]],
        ],
        extract_gate_features: Callable[[CausalSnapshotV2], Mapping[str, Any]],
    ) -> None:
        self.run_source_episode = run_source_episode
        self.replay_to_snapshot = replay_to_snapshot
        self.extract_gate_features = extract_gate_features

    def collect_one(
        self,
        *,
        source_policy: str,
        stratum: CausalSamplingStratum | str,
        selection_seed: int,
        build_identity: Callable[[SourceChunkTraceV2], CausalStateIdentityV2],
    ) -> SelectedSourceStateV2:
        """Return one verified snapshot and its pre-prediction Gate features."""

        episode = self.run_source_episode(source_policy)
        if episode.source_policy != source_policy:
            raise ValueError("Source episode policy differs from its assigned quota.")
        traces = episode.traces
        trace, sampling = select_source_chunk(
            traces,
            stratum=stratum,
            seed=selection_seed,
            source_policy=source_policy,
            source_final_success=episode.final_success,
        )
        identity = build_identity(trace)
        snapshot, replay = self.replay_to_snapshot(trace, identity, sampling)
        assert_source_replay_exact(trace, replay)
        features = dict(self.extract_gate_features(snapshot))
        validate_pre_prediction_feature_payload(features)
        return SelectedSourceStateV2(
            trace=trace,
            snapshot=snapshot,
            gate_features=features,
            source_traces=traces,
        )


def assign_future_donors(
    entries: Sequence[FutureLatentEntryV1],
) -> dict[str, dict[str, FutureLatentEntryV1]]:
    """Freeze shuffled, temporal, and instruction donors before outcomes exist."""

    by_task_bin: dict[tuple[str, str, int], list[FutureLatentEntryV1]] = defaultdict(
        list
    )
    by_episode: dict[str, list[FutureLatentEntryV1]] = defaultdict(list)
    by_suite: dict[str, list[FutureLatentEntryV1]] = defaultdict(list)
    for entry in entries:
        by_task_bin[(entry.suite, entry.global_task_uid, entry.criticality_bin)].append(
            entry
        )
        by_episode[entry.source_episode_id].append(entry)
        by_suite[entry.suite].append(entry)
    result: dict[str, dict[str, FutureLatentEntryV1]] = {}
    for entry in sorted(entries, key=lambda item: item.state_id):
        shuffled = sorted(
            (
                item
                for item in by_task_bin[
                    (entry.suite, entry.global_task_uid, entry.criticality_bin)
                ]
                if item.source_episode_id != entry.source_episode_id
            ),
            key=lambda item: item.state_id,
        )
        temporal = sorted(
            (
                item
                for item in by_episode[entry.source_episode_id]
                if abs(item.chunk_index - entry.chunk_index) >= 2
            ),
            key=lambda item: (abs(item.chunk_index - entry.chunk_index), item.state_id),
        )
        instruction = sorted(
            (
                item
                for item in by_suite[entry.suite]
                if item.instruction_id != entry.instruction_id
            ),
            key=lambda item: item.state_id,
        )
        if not shuffled or not temporal or not instruction:
            continue
        result[entry.state_id] = {
            "shuffled_wrong_state": shuffled[0],
            "temporal_shift": temporal[0],
            "instruction_mismatch": instruction[0],
        }
    return result
