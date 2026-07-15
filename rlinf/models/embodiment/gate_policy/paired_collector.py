# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Simulator-agnostic orchestration for LIBERO paired counterfactual branches."""

from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from rlinf.models.embodiment.gate_policy.paired_data import (
    reference_assignment_sha256,
    write_paired_dataset,
)


UNCOND = 0
IDM = 1


class PairedCollectorDriver(Protocol):
    """Heavy LIBERO/FastWAM adapter required by :class:`PairedStateCollector`."""

    paired_metadata: Mapping[str, Any]

    def reset_episode(self, episode: Mapping[str, Any]) -> Any: ...

    def capture_snapshot(self) -> Mapping[str, Any]: ...

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> Any: ...

    def context(self, observation: Any) -> Mapping[str, Any]: ...

    def features(self, observation: Any) -> Mapping[str, torch.Tensor]: ...

    def action(self, observation: Any, *, mode: int, seed: int) -> Any: ...

    def step_chunk(self, action: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class BranchOutcome:
    success: bool
    progress_1: float
    progress_3: float
    decisions: int


def _row_feature(value: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).detach().cpu().float()
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"driver feature {name!r} must be [D] or [1,D]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"driver feature {name!r} contains non-finite values")
    return value


def _step_fields(result: Mapping[str, Any]) -> tuple[Any, bool, bool, float]:
    if not isinstance(result, Mapping):
        raise TypeError("driver.step_chunk() must return a mapping")
    missing = [key for key in ("observation", "done", "success", "progress") if key not in result]
    if missing:
        raise ValueError(f"driver.step_chunk() result is missing {missing}")
    progress = float(result["progress"])
    if not np.isfinite(progress):
        raise ValueError("driver progress metric must be finite")
    return result["observation"], bool(result["done"]), bool(result["success"]), progress


class PairedStateCollector:
    """Collect U-vs-I one-chunk treatments with common random numbers."""

    def __init__(
        self,
        driver: PairedCollectorDriver,
        *,
        collector_seed: int,
        max_reference_decisions: int,
        max_branch_decisions: int,
        sensitivity_fraction: float = 0.2,
        snapshot_dir: str | os.PathLike[str] | None = None,
    ):
        self.driver = driver
        self.collector_seed = int(collector_seed)
        self.max_reference_decisions = int(max_reference_decisions)
        self.max_branch_decisions = int(max_branch_decisions)
        self.sensitivity_fraction = float(sensitivity_fraction)
        if self.collector_seed < 0:
            raise ValueError("collector_seed must be non-negative")
        if min(self.max_reference_decisions, self.max_branch_decisions) <= 0:
            raise ValueError("collector decision limits must be positive")
        if not 0.0 <= self.sensitivity_fraction <= 1.0:
            raise ValueError("sensitivity_fraction must be in [0,1]")
        self.snapshot_dir = None if snapshot_dir is None else Path(snapshot_dir).resolve()
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _seed(self, episode_uid: str, decision: int, stream: int) -> int:
        token = f"{self.collector_seed}|{episode_uid}|{decision}|{stream}"
        return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little")

    def _run_branch(
        self,
        snapshot: Mapping[str, Any],
        *,
        first_mode: int,
        continuation_mode: int,
        episode_uid: str,
        decision_index: int,
        stream: int,
    ) -> BranchOutcome:
        observation = self.driver.restore_snapshot(snapshot)
        progress_1 = None
        progress_3 = None
        success = False
        decisions = 0
        for offset in range(self.max_branch_decisions):
            mode = first_mode if offset == 0 else continuation_mode
            action = self.driver.action(
                observation,
                mode=mode,
                seed=self._seed(episode_uid, decision_index, stream + offset),
            )
            observation, done, step_success, progress = _step_fields(
                self.driver.step_chunk(action)
            )
            decisions += 1
            success = success or step_success
            if offset == 0:
                progress_1 = progress
            if offset == 2:
                progress_3 = progress
            if done or success:
                break
        if progress_1 is None:
            raise RuntimeError("paired branch executed no decisions")
        if progress_3 is None:
            progress_3 = progress
        return BranchOutcome(success, progress_1, progress_3, decisions)

    def _reference_mode(self, episode_index: int) -> int | None:
        # Fallback for injected tests or deliberately tiny noncanonical pilots.
        return (UNCOND, IDM, None)[episode_index % 3]

    def _reference_assignments(self, count: int) -> list[int | None]:
        if count <= 0:
            return []
        if count % 3:
            return [self._reference_mode(index) for index in range(count)]
        assignments: list[int | None] = (
            [UNCOND] * (count // 3)
            + [IDM] * (count // 3)
            + [None] * (count // 3)
        )
        # Do not align source policies with a factor/task-sorted manifest. This
        # stream is separate from branch/sensitivity randomness and is fully
        # determined by the recorded collector seed.
        rng = np.random.default_rng(self.collector_seed ^ 0x5A17C0DE)
        order = rng.permutation(count).tolist()
        return [assignments[index] for index in order]

    def reference_assignment_map(
        self, episodes: Sequence[Mapping[str, Any]]
    ) -> dict[str, str]:
        """Build one deterministic assignment over the complete logical manifest."""
        assignments = self._reference_assignments(len(episodes))
        result: dict[str, str] = {}
        for episode, assignment in zip(episodes, assignments):
            episode_uid = str(episode.get("episode_id", ""))
            if not episode_uid or episode_uid in result:
                raise ValueError(
                    "reference assignment episodes require unique non-empty episode_id"
                )
            result[episode_uid] = (
                "random_0.5"
                if assignment is None
                else "uncond" if assignment == UNCOND else "idm"
            )
        return result

    def collect(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        reference_assignments: Mapping[str, str] | None = None,
        reference_assignment_manifest_sha256: str | None = None,
    ) -> dict[str, Any]:
        rows: dict[str, list[torch.Tensor]] = {}
        records: list[dict[str, Any]] = []

        def append(name: str, value: Any, *, dtype=None) -> None:
            tensor = torch.as_tensor(value, dtype=dtype).detach().cpu()
            rows.setdefault(name, []).append(tensor)

        if reference_assignments is None:
            reference_assignments = self.reference_assignment_map(episodes)
        else:
            reference_assignments = {
                str(key): str(value) for key, value in reference_assignments.items()
            }
        if set(reference_assignments.values()) - {"uncond", "idm", "random_0.5"}:
            raise ValueError("reference assignment map contains an unsupported policy")
        assignment_manifest_sha256 = str(
            reference_assignment_manifest_sha256
            or self.driver.paired_metadata.get("episode_manifest_sha256", "")
        )
        for trajectory_id, episode in enumerate(episodes):
            observation = self.driver.reset_episode(episode)
            requested_episode_uid = str(episode.get("episode_id", ""))
            if requested_episode_uid not in reference_assignments:
                raise ValueError(
                    f"episode {requested_episode_uid!r} is missing from the parent "
                    "reference assignment contract"
                )
            assignment_name = reference_assignments[requested_episode_uid]
            reference_mode = {
                "uncond": UNCOND,
                "idm": IDM,
                "random_0.5": None,
            }[assignment_name]
            for local_decision in range(self.max_reference_decisions):
                context = dict(self.driver.context(observation))
                episode_uid = str(context.get("episode_uid", ""))
                if not episode_uid:
                    raise ValueError("driver context must provide a stable episode_uid")
                if episode_uid != requested_episode_uid:
                    raise ValueError(
                        "driver episode_uid differs from the frozen manifest episode_id"
                    )
                decision_index = int(context.get("decision_index", local_decision))
                if decision_index != local_decision:
                    raise ValueError(
                        "driver decision_index must start at zero and increase once per chunk"
                    )
                phase = str(context.get("phase", "unknown"))
                phase_reliable = bool(context.get("phase_reliable", False))
                if phase not in {
                    "approach",
                    "contact_alignment",
                    "transport_completion",
                    "unknown",
                }:
                    raise ValueError(f"unsupported pre-treatment phase {phase!r}")
                if not phase_reliable:
                    phase = "unknown"

                snapshot = self.driver.capture_snapshot()
                snapshot_schema = str(snapshot.get("schema", ""))
                expected_snapshot_schema = str(
                    self.driver.paired_metadata.get("snapshot_schema", "")
                )
                if not snapshot_schema or snapshot_schema != expected_snapshot_schema:
                    raise ValueError(
                        "driver snapshot schema does not match paired_metadata.snapshot_schema"
                    )
                features = dict(self.driver.features(observation))
                feature_rows = {
                    key: _row_feature(features[key], key)
                    for key in ("world_feat", "proprio", "text_feat")
                }
                state_id = hashlib.sha256(
                    f"{episode_uid}|{decision_index}".encode()
                ).hexdigest()[:32]
                snapshot_path = ""
                buffer = io.BytesIO()
                torch.save(dict(snapshot), buffer)
                snapshot_bytes = buffer.getvalue()
                snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
                if self.snapshot_dir is not None:
                    path = self.snapshot_dir / f"{state_id}.pt"
                    path.write_bytes(snapshot_bytes)
                    snapshot_path = str(path)

                branch_seed = self._seed(episode_uid, decision_index, 0)
                uncond = self._run_branch(
                    snapshot,
                    first_mode=UNCOND,
                    continuation_mode=UNCOND,
                    episode_uid=episode_uid,
                    decision_index=decision_index,
                    stream=0,
                )
                idm = self._run_branch(
                    snapshot,
                    first_mode=IDM,
                    continuation_mode=UNCOND,
                    episode_uid=episode_uid,
                    decision_index=decision_index,
                    stream=0,
                )

                # State-addressed randomness keeps collection invariant to the
                # physical task-suite partition and process scheduling.
                sensitivity = (
                    self._seed(episode_uid, decision_index, 30_000) / float(2**32)
                    < self.sensitivity_fraction
                )
                if sensitivity:
                    sens_uncond = self._run_branch(
                        snapshot,
                        first_mode=UNCOND,
                        continuation_mode=IDM,
                        episode_uid=episode_uid,
                        decision_index=decision_index,
                        stream=10_000,
                    )
                    sens_idm = self._run_branch(
                        snapshot,
                        first_mode=IDM,
                        continuation_mode=IDM,
                        episode_uid=episode_uid,
                        decision_index=decision_index,
                        stream=10_000,
                    )
                else:
                    sens_uncond = BranchOutcome(False, 0.0, 0.0, 0)
                    sens_idm = BranchOutcome(False, 0.0, 0.0, 0)

                chosen_source_mode = (
                    self._seed(episode_uid, decision_index, 40_000) % 2
                    if reference_mode is None
                    else reference_mode
                )
                for key, value in feature_rows.items():
                    append(key, value)
                append("trajectory_id", trajectory_id, dtype=torch.int64)
                append("decision_index", decision_index, dtype=torch.int64)
                append("task_id", int(context["task_id"]), dtype=torch.int64)
                append("source_mode", chosen_source_mode, dtype=torch.int64)
                append("branch_seed", branch_seed, dtype=torch.int64)
                append("success_uncond", uncond.success, dtype=torch.bool)
                append("success_idm", idm.success, dtype=torch.bool)
                append("progress_1_uncond", uncond.progress_1, dtype=torch.float32)
                append("progress_1_idm", idm.progress_1, dtype=torch.float32)
                append("progress_3_uncond", uncond.progress_3, dtype=torch.float32)
                append("progress_3_idm", idm.progress_3, dtype=torch.float32)
                append("sensitivity_mask", sensitivity, dtype=torch.bool)
                append("sensitivity_success_uncond", sens_uncond.success, dtype=torch.bool)
                append("sensitivity_success_idm", sens_idm.success, dtype=torch.bool)
                append("sensitivity_progress_3_uncond", sens_uncond.progress_3, dtype=torch.float32)
                append("sensitivity_progress_3_idm", sens_idm.progress_3, dtype=torch.float32)
                records.append(
                    {
                        "state_id": state_id,
                        "episode_uid": episode_uid,
                        "base_task": str(context.get("base_task", "unknown")),
                        "task_suite_name": str(
                            context.get("task_suite_name", "unknown")
                        ),
                        "task_description": str(
                            context.get("task_description", "unknown")
                        ),
                        "trial_id": int(context.get("trial_id", 0)),
                        "reset_state_id": int(context.get("reset_state_id", 0)),
                        "env_seed": int(context.get("env_seed", 0)),
                        "factor": str(context.get("factor", "unknown")),
                        "level": str(context.get("level", "unknown")),
                        "perturbation_id": str(
                            context.get("perturbation_id", "unknown")
                        ),
                        "phase": phase,
                        "phase_reliable": phase_reliable,
                        "snapshot_path": snapshot_path,
                        "snapshot_sha256": snapshot_sha256,
                        "asset_ids": list(context.get("asset_ids", [])),
                    }
                )

                observation = self.driver.restore_snapshot(snapshot)
                source_action = self.driver.action(
                    observation,
                    mode=chosen_source_mode,
                    seed=self._seed(episode_uid, decision_index, 20_000),
                )
                observation, done, success, _ = _step_fields(
                    self.driver.step_chunk(source_action)
                )
                if done or success:
                    break

        if not records:
            raise RuntimeError("paired collector produced no decision states")
        data = {key: torch.stack(values, dim=0) for key, values in rows.items()}
        meta = {
            **dict(self.driver.paired_metadata),
            "collector_seed": self.collector_seed,
            "continuation_mode": "uncond",
            "max_reference_decisions": self.max_reference_decisions,
            "max_branch_decisions": self.max_branch_decisions,
            "sensitivity_fraction": self.sensitivity_fraction,
            "reference_policy_mix": ["uncond", "idm", "random_0.5"],
            "reference_policy_assignment": "balanced_shuffled_v1",
            "reference_assignment_manifest_sha256": assignment_manifest_sha256,
            "reference_assignment_sha256": reference_assignment_sha256(
                reference_assignments
            ),
            "reference_policy_episode_assignments": dict(
                sorted(reference_assignments.items())
            ),
            "num_samples": len(records),
        }
        return {"data": data, "records": records, "meta": meta}

    def collect_to_path(
        self,
        episodes: Sequence[Mapping[str, Any]],
        path: str | os.PathLike[str],
        *,
        reference_assignments: Mapping[str, str] | None = None,
        reference_assignment_manifest_sha256: str | None = None,
    ) -> str:
        result = self.collect(
            episodes,
            reference_assignments=reference_assignments,
            reference_assignment_manifest_sha256=(
                reference_assignment_manifest_sha256
            ),
        )
        return write_paired_dataset(path, **result)
