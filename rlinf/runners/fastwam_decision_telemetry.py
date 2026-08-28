# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-decision FastWAM routing telemetry shared by training and evaluation."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from rlinf.models.embodiment.wam_policy.contracts import (
    GateDecisionRecord,
    WAMRoute,
)

FASTWAM_DECISION_TELEMETRY_SCHEMA = "fastwam-routing-decision-v1"


def _finite_optional(value: float | int | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric or null.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite or null.")
    return result


def build_fastwam_decision_telemetry_record(
    *,
    phase: str,
    run_id: str,
    rank: int,
    trajectory_id: str,
    env_id: int,
    episode_id: int,
    task_suite: str,
    task_id: int,
    trial_id: int,
    reset_state_id: int,
    cycle_index: int,
    update_step: int,
    actor_version: int,
    route: WAMRoute | int | str,
    base_probability: float,
    behavior_probability: float,
    forced_exploration: bool,
    mode_flip_delta: float,
    configured_idm_cost: float | None,
    destination_advantage_unnormalized: float | None,
    destination_advantage_normalized: float | None,
    eligible_decision: bool,
) -> dict[str, Any]:
    """Build one JSON-safe routing decision with a phase-independent schema."""

    phase = str(phase).strip().lower()
    if phase not in {"training", "evaluation"}:
        raise ValueError("FastWAM decision phase must be training or evaluation.")
    run_id = str(run_id).strip()
    trajectory_id = str(trajectory_id).strip()
    task_suite = str(task_suite).strip()
    if not run_id or not trajectory_id or not task_suite:
        raise ValueError("FastWAM decision string identities must be non-empty.")
    integer_fields = {
        "rank": rank,
        "env_id": env_id,
        "episode_id": episode_id,
        "task_id": task_id,
        "trial_id": trial_id,
        "reset_state_id": reset_state_id,
        "cycle_index": cycle_index,
        "update_step": update_step,
        "actor_version": actor_version,
    }
    if any(
        isinstance(value, bool) or int(value) < 0 for value in integer_fields.values()
    ):
        raise ValueError("FastWAM decision integer identities must be non-negative.")
    if isinstance(route, str):
        route_name = route.strip().lower()
        if route_name not in {"idm", "uncond"}:
            raise ValueError(f"Unknown FastWAM decision route {route!r}.")
    else:
        route_name = WAMRoute(int(route)).name.lower()
    base_probability = _finite_optional(base_probability, name="base_probability")
    behavior_probability = _finite_optional(
        behavior_probability, name="behavior_probability"
    )
    if not 0.0 <= base_probability <= 1.0:
        raise ValueError("base_probability must lie in [0, 1].")
    if not 0.0 <= behavior_probability <= 1.0:
        raise ValueError("behavior_probability must lie in [0, 1].")
    if not isinstance(forced_exploration, bool):
        raise TypeError("forced_exploration must be boolean.")
    if not isinstance(eligible_decision, bool):
        raise TypeError("eligible_decision must be boolean.")
    mode_flip_delta = _finite_optional(mode_flip_delta, name="mode_flip_delta")
    configured_idm_cost = _finite_optional(
        configured_idm_cost, name="configured_idm_cost"
    )
    if configured_idm_cost is not None and configured_idm_cost < 0.0:
        raise ValueError("configured_idm_cost must be non-negative.")
    unnormalized = _finite_optional(
        destination_advantage_unnormalized,
        name="destination_advantage_unnormalized",
    )
    normalized = _finite_optional(
        destination_advantage_normalized,
        name="destination_advantage_normalized",
    )
    if (
        phase == "training"
        and eligible_decision
        and (unnormalized is None or normalized is None)
    ):
        raise ValueError(
            "Eligible training decisions require both destination advantages."
        )
    cycle_index = int(cycle_index)
    return {
        "schema": FASTWAM_DECISION_TELEMETRY_SCHEMA,
        "phase": phase,
        "run_id": run_id,
        "rank": int(rank),
        "trajectory_id": trajectory_id,
        "env_id": int(env_id),
        "episode_id": int(episode_id),
        "task_suite": task_suite,
        "task_id": int(task_id),
        "trial_id": int(trial_id),
        "reset_state_id": int(reset_state_id),
        "record_id": f"{trajectory_id}:{cycle_index}",
        "cycle_index": cycle_index,
        "destination_cycle_index": cycle_index + 1,
        "update_step": int(update_step),
        "actor_version": int(actor_version),
        "route": route_name,
        "gate_idm_probability": base_probability,
        "gate_behavior_idm_probability": behavior_probability,
        "forced_exploration": forced_exploration,
        "configured_idm_cost": configured_idm_cost,
        "destination_advantage_unnormalized": unnormalized,
        "destination_advantage_normalized": normalized,
        "mode_flip_delta": mode_flip_delta,
        "eligible_decision": eligible_decision,
    }


def build_fastwam_training_decision_records(
    *,
    emitted: GateDecisionRecord,
    gate_valid_mask: torch.Tensor,
    unnormalized_gate_advantages: torch.Tensor,
    normalized_gate_advantages: torch.Tensor,
    runner_step: int,
    rank: int,
    run_id: str,
    task_suite: str,
    configured_idm_cost: float,
) -> list[dict[str, Any]]:
    """Materialize each consumed training decision in deterministic order."""

    if len(emitted.shape) != 2:
        raise ValueError("Training decision telemetry requires [time, batch] data.")
    for name, value in (
        ("gate_valid_mask", gate_valid_mask),
        ("unnormalized_gate_advantages", unnormalized_gate_advantages),
        ("normalized_gate_advantages", normalized_gate_advantages),
    ):
        if value.shape != emitted.shape:
            raise ValueError(f"{name} must match emitted Gate decisions.")
    if gate_valid_mask.dtype != torch.bool:
        raise TypeError("gate_valid_mask must use torch.bool.")
    if bool((gate_valid_mask & ~emitted.valid).any().item()):
        raise ValueError("Telemetry cannot select an invalid Gate emission.")
    required_metadata = {
        "exploration_forced": emitted.exploration_forced,
        "mode_flip_delta": emitted.mode_flip_delta,
        "environment_ids": emitted.environment_ids,
        "task_ids": emitted.task_ids,
        "trial_ids": emitted.trial_ids,
        "reset_state_ids": emitted.reset_state_ids,
    }
    missing = [name for name, value in required_metadata.items() if value is None]
    if missing:
        raise ValueError(
            f"Training decision telemetry is missing rollout metadata: {missing}."
        )

    records = []
    for time_index, batch_index in gate_valid_mask.nonzero(as_tuple=False).tolist():
        env_id = int(emitted.environment_ids[time_index, batch_index].item())
        episode_id = int(emitted.episode_ids[time_index, batch_index].item())
        trajectory_id = f"{env_id}:{episode_id}"
        records.append(
            build_fastwam_decision_telemetry_record(
                phase="training",
                run_id=run_id,
                rank=rank,
                trajectory_id=trajectory_id,
                env_id=env_id,
                episode_id=episode_id,
                task_suite=task_suite,
                task_id=int(emitted.task_ids[time_index, batch_index].item()),
                trial_id=int(emitted.trial_ids[time_index, batch_index].item()),
                reset_state_id=int(
                    emitted.reset_state_ids[time_index, batch_index].item()
                ),
                cycle_index=int(
                    emitted.source_chunk_ids[time_index, batch_index].item()
                ),
                update_step=runner_step,
                actor_version=int(
                    emitted.actor_versions[time_index, batch_index].item()
                ),
                route=int(emitted.next_route[time_index, batch_index].item()),
                base_probability=float(
                    emitted.base_probability[time_index, batch_index].item()
                ),
                behavior_probability=float(
                    emitted.behavior_probability[time_index, batch_index].item()
                ),
                forced_exploration=bool(
                    emitted.exploration_forced[time_index, batch_index].item()
                ),
                mode_flip_delta=float(
                    emitted.mode_flip_delta[time_index, batch_index].item()
                ),
                configured_idm_cost=configured_idm_cost,
                destination_advantage_unnormalized=float(
                    unnormalized_gate_advantages[time_index, batch_index].item()
                ),
                destination_advantage_normalized=float(
                    normalized_gate_advantages[time_index, batch_index].item()
                ),
                eligible_decision=True,
            )
        )
    return records


def append_fastwam_decision_telemetry_jsonl(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Append one update's decision records and make them immediately durable."""

    if not records:
        raise ValueError("FastWAM decision telemetry has no consumed decisions.")
    encoded = []
    for record in records:
        payload = dict(record)
        if payload.get("schema") != FASTWAM_DECISION_TELEMETRY_SCHEMA:
            raise ValueError("FastWAM decision telemetry schema mismatch.")
        if payload.get("phase") != "training":
            raise ValueError("The training JSONL accepts only training decisions.")
        encoded.append(json.dumps(payload, sort_keys=True, allow_nan=False))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(encoded) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
