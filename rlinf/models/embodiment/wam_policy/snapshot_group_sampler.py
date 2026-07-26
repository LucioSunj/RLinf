# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Snapshot-grouped rollout planning: one group = K rollouts of ONE snapshot.

Existing GRPO grouping is "G mode sequences from one fixed reset init"; stage-2
needs groups anchored on a mid-episode snapshot state instead.  This module
plans those groups deterministically and rank-invariantly:

- :func:`plan_snapshot_groups` is a pure function of ``(snapshots, group_size,
  seed)``.  Rank count never enters the plan, so every rank layout sees the
  identical grouping; :func:`shard_plan` hands whole groups to ranks in
  contiguous blocks, mirroring how env slots keep groups contiguous per rank.
- Snapshot counts that do not fill every supported rank layout are handled by
  the DROP contract: the deterministic tail of the seed-keyed hash ordering is
  dropped and recorded on the plan (``dropped_snapshot_ids``) - never silently
  truncated.  Zero usable groups is an error, not an empty plan.  Ordering is
  ``sha256(f"{seed}:{snapshot_id}")``, not a numpy Generator stream: NEP 19
  permits Generator bit-streams to change across numpy feature releases, which
  would silently break cross-machine determinism of the kept/dropped split.
- :func:`collect_group_rollouts` executes a plan against a
  ``PairedCollectorDriver``-shaped adapter, restoring the snapshot BEFORE EVERY
  member rollout (a rollout mutates simulator state, so members restore
  independently rather than share one restore).

Data flow: snapshot -> K rollouts (this module) -> buffer rows
(``buffer_schema``, with ``(tau, eps, cfm_loss_old)`` attached by the WS5
actor) -> per-trajectory scores reshaped ``[-1, group_size]`` -> GRPO
advantages.  This module deliberately stops at rollout records: the frozen
``PairedCollectorDriver`` protocol returns actions only, so CFM quantities
cannot originate here.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DEFAULT_RANK_LAYOUTS = (1, 2, 4, 8)


class SnapshotGroupError(ValueError):
    """Raised when a grouping request violates the snapshot-group contract."""


@dataclass(frozen=True)
class GroupSpec:
    """One planned group: ``group_size`` rollouts of one snapshot."""

    group_id: int
    snapshot_id: str
    episode_id: str


@dataclass(frozen=True)
class SnapshotGroupPlan:
    """Deterministic, rank-independent grouping over a snapshot batch.

    ``group_id`` values are PLAN-SCOPED (``0..N-1`` within this plan).  Rows
    from different plans must not share one buffer without the caller
    renumbering group ids per rollout round; ``serialize_samples`` will
    otherwise refuse them as duplicate or snapshot-mixing groups.
    """

    group_size: int
    seed: int
    groups: tuple[GroupSpec, ...]
    dropped_snapshot_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroupRolloutRecord:
    """Outcome of one member rollout, ready to become a buffer row."""

    snapshot_id: str
    group_id: int
    group_member_index: int
    episode_id: str
    member_seed: int
    reward: float
    decisions_executed: int
    terminated: bool


def _require_divisible(name: str, total: int, divisor_name: str, divisor: int) -> None:
    if divisor <= 0 or total % divisor != 0:
        raise SnapshotGroupError(
            f"{name} ({total}) must be divisible by {divisor_name} ({divisor})"
        )


def member_seed(
    base_seed: int, snapshot_id: str, member_index: int, decision: int
) -> int:
    """Deterministic per-(member, decision) seed, invariant to rank layout."""
    digest = hashlib.sha256(
        f"{int(base_seed)}:{snapshot_id}:{int(member_index)}:{int(decision)}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def plan_snapshot_groups(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    group_size: int,
    seed: int,
    rank_layouts: Sequence[int] = DEFAULT_RANK_LAYOUTS,
) -> SnapshotGroupPlan:
    """Plan one group per usable snapshot, deterministically.

    Args:
        snapshots: Mappings each carrying at least ``snapshot_id`` and
            ``episode_id``.  Snapshot payloads stay with the caller; the plan
            references snapshots by id only.
        group_size: Rollouts per snapshot (K).  Must be >= 2 - a singleton
            group has no within-group contrast for a GRPO baseline.
        seed: Drives the permutation AND which snapshots are dropped.
        rank_layouts: Every world size the plan must divide cleanly into.

    Returns:
        A plan whose group count is divisible by every entry of
        ``rank_layouts``, with the dropped tail recorded explicitly.

    Raises:
        SnapshotGroupError: On duplicate ids, malformed descriptors,
            ``group_size < 2``, or zero usable groups after rounding down.
    """
    if int(group_size) < 2:
        raise SnapshotGroupError(f"group_size must be >= 2, got {group_size}")
    if not rank_layouts or any(int(r) < 1 for r in rank_layouts):
        raise SnapshotGroupError(f"rank_layouts must be positive, got {rank_layouts!r}")
    ids: list[str] = []
    episodes: list[str] = []
    for index, descriptor in enumerate(snapshots):
        snapshot_id = descriptor.get("snapshot_id")
        episode_id = descriptor.get("episode_id")
        for name, value in (("snapshot_id", snapshot_id), ("episode_id", episode_id)):
            if not isinstance(value, str) or not value:
                raise SnapshotGroupError(
                    f"snapshots[{index}]: {name} must be a non-empty string, "
                    f"got {value!r}"
                )
        ids.append(snapshot_id)
        episodes.append(episode_id)
    if len(set(ids)) != len(ids):
        raise SnapshotGroupError("snapshot_id values must be unique")

    layout_lcm = math.lcm(*(int(r) for r in rank_layouts))
    usable = (len(ids) // layout_lcm) * layout_lcm
    if usable == 0:
        raise SnapshotGroupError(
            f"{len(ids)} snapshots cannot fill even one group per rank for "
            f"layouts {tuple(rank_layouts)} (need a multiple of {layout_lcm})"
        )
    order = sorted(
        range(len(ids)),
        key=lambda i: hashlib.sha256(f"{int(seed)}:{ids[i]}".encode()).hexdigest(),
    )
    kept = order[:usable]
    dropped = tuple(ids[int(i)] for i in order[usable:])
    groups = tuple(
        GroupSpec(
            group_id=position,
            snapshot_id=ids[int(source)],
            episode_id=episodes[int(source)],
        )
        for position, source in enumerate(kept)
    )
    return SnapshotGroupPlan(
        group_size=int(group_size),
        seed=int(seed),
        groups=groups,
        dropped_snapshot_ids=dropped,
    )


def shard_plan(
    plan: SnapshotGroupPlan, *, rank: int, num_ranks: int
) -> tuple[GroupSpec, ...]:
    """Contiguous whole-group shard for one rank; never splits a group."""
    if not 0 <= int(rank) < int(num_ranks):
        raise SnapshotGroupError(f"rank {rank} outside [0, {num_ranks})")
    _require_divisible("planned groups", len(plan.groups), "num_ranks", int(num_ranks))
    per_rank = len(plan.groups) // int(num_ranks)
    start = int(rank) * per_rank
    return plan.groups[start : start + per_rank]


def collect_group_rollouts(
    driver: Any,
    groups: Sequence[GroupSpec],
    *,
    group_size: int,
    base_seed: int,
    snapshots_by_id: Mapping[str, Mapping[str, Any]],
    max_decisions: int,
) -> list[GroupRolloutRecord]:
    """Run K independent rollouts per planned group through a paired driver.

    ``driver`` follows the ``PairedCollectorDriver`` protocol
    (``restore_snapshot`` / ``action`` / ``step_chunk``).  Every member starts
    from its own ``restore_snapshot`` call; rewards are summed over executed
    decisions.  ``mode=0`` (UNCOND) is fixed: the whole point of the snapshot
    group is contrasting UNCOND rollouts from an identical state.
    """
    if int(max_decisions) < 1:
        raise SnapshotGroupError(f"max_decisions must be >= 1, got {max_decisions}")
    # Mirror the planner's contract instead of trusting the caller: group_size=0
    # would silently execute nothing, and a K that disagrees with the plan
    # produces a buffer that lies about its groups.
    if int(group_size) < 2:
        raise SnapshotGroupError(f"group_size must be >= 2, got {group_size}")
    if not groups:
        raise SnapshotGroupError("refusing to collect over an empty group sequence")
    records: list[GroupRolloutRecord] = []
    for spec in groups:
        snapshot = snapshots_by_id.get(spec.snapshot_id)
        if snapshot is None:
            raise SnapshotGroupError(
                f"group {spec.group_id}: snapshot {spec.snapshot_id!r} missing "
                "from snapshots_by_id"
            )
        for member in range(int(group_size)):
            observation = driver.restore_snapshot(snapshot)
            first_seed = member_seed(base_seed, spec.snapshot_id, member, 0)
            total_reward = 0.0
            terminated = False
            decisions = 0
            while not terminated and decisions < int(max_decisions):
                seed = member_seed(base_seed, spec.snapshot_id, member, decisions)
                action = driver.action(observation, mode=0, seed=seed)
                result = driver.step_chunk(action)
                total_reward += float(result["reward"])
                terminated = bool(result["done"])
                decisions += 1
                if not terminated and decisions < int(max_decisions):
                    # Fail closed on a missing/None observation: silently
                    # reusing the restore-time observation would make the
                    # policy act blind on stale state for the whole rollout.
                    observation = result.get("observation")
                    if observation is None:
                        raise SnapshotGroupError(
                            f"group {spec.group_id} member {member} decision "
                            f"{decisions}: step_chunk returned no observation "
                            "for a non-terminal step"
                        )
            records.append(
                GroupRolloutRecord(
                    snapshot_id=spec.snapshot_id,
                    group_id=spec.group_id,
                    group_member_index=member,
                    episode_id=spec.episode_id,
                    member_seed=first_seed,
                    reward=total_reward,
                    decisions_executed=decisions,
                    terminated=terminated,
                )
            )
    return records
