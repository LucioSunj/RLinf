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

import dataclasses
import hashlib
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _namespace(name: str, path: Path) -> None:
    """Fabricate an empty namespace package so heavy __init__ files never run.

    Same mechanism as _gate_test_imports.load_gate_modules: rlinf/__init__ and
    rlinf/models/__init__ pull omegaconf/torch-heavy registration at import
    time, which these unit tests do not need. setdefault keeps compatibility
    with test files that import the real ``rlinf`` package first.
    """
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules.setdefault(name, module)


_namespace("rlinf", ROOT / "rlinf")
_namespace("rlinf.models", ROOT / "rlinf" / "models")
_namespace("rlinf.models.embodiment", ROOT / "rlinf" / "models" / "embodiment")
_namespace(
    "rlinf.models.embodiment.wam_policy",
    ROOT / "rlinf" / "models" / "embodiment" / "wam_policy",
)

buffer_schema = importlib.import_module(
    "rlinf.models.embodiment.wam_policy.buffer_schema"
)
sampler = importlib.import_module(
    "rlinf.models.embodiment.wam_policy.snapshot_group_sampler"
)

CONFIG = buffer_schema.WamBufferConfig(action_horizon=4, action_dim=3, group_size=2)


def _sample(**overrides) -> "buffer_schema.WamRolloutSample":
    base = {
        "snapshot_id": "snap-a",
        "group_id": 0,
        "group_member_index": 0,
        "episode_id": "ep-1",
        "decision_index": 0,
        "tau": torch.tensor(0.37, dtype=torch.float32),
        "eps": torch.zeros(4, 3, dtype=torch.float32),
        "cfm_loss_old": 1.25,
        "reward": 1.0,
        "advantage": None,
    }
    base.update(overrides)
    return buffer_schema.WamRolloutSample(**base)


def _full_group(group_id: int = 0, snapshot_id: str = "snap-a"):
    return [
        _sample(
            group_id=group_id,
            group_member_index=k,
            snapshot_id=snapshot_id,
            eps=torch.full((4, 3), float(k), dtype=torch.float32),
            advantage=0.5 if k == 0 else -0.5,
        )
        for k in range(CONFIG.group_size)
    ]


def _snapshots(count: int):
    return [
        {"snapshot_id": f"snap-{i}", "episode_id": f"ep-{i}", "payload": {"i": i}}
        for i in range(count)
    ]


class FakePairedDriver:
    """PairedCollectorDriver-shaped adapter over a deterministic toy world.

    Reward is a pure function of (snapshot_id, seed); an episode terminates
    after ``horizon`` decisions. Every protocol member exists so the shape of
    the seam matches gate_policy.paired_collector.PairedCollectorDriver.
    """

    def __init__(self, horizon: int = 3):
        self.horizon = horizon
        self.paired_metadata = {"driver": "fake"}
        self.restore_calls: list[str] = []
        self.action_seeds: list[int] = []
        self._steps = 0
        self._snapshot_id = ""

    def reset_episode(self, episode):
        return {"episode": dict(episode)}

    def capture_snapshot(self):
        return {"snapshot_id": self._snapshot_id}

    def restore_snapshot(self, snapshot):
        self._snapshot_id = snapshot["snapshot_id"]
        self.restore_calls.append(self._snapshot_id)
        self._steps = 0
        return {"snapshot_id": self._snapshot_id, "step": 0}

    def context(self, observation):
        return {"snapshot_id": observation["snapshot_id"]}

    def features(self, observation):
        return {"world_feat": torch.zeros(2, dtype=torch.float32)}

    def action(self, observation, *, mode: int, seed: int):
        assert mode == 0, "snapshot groups contrast UNCOND rollouts only"
        self.action_seeds.append(seed)
        return {"seed": seed, "snapshot_id": observation["snapshot_id"]}

    def step_chunk(self, action):
        self._steps += 1
        digest = hashlib.sha256(
            f"{action['snapshot_id']}:{action['seed']}".encode()
        ).digest()
        return {
            "reward": digest[0] / 255.0,
            "done": self._steps >= self.horizon,
            "observation": {
                "snapshot_id": action["snapshot_id"],
                "step": self._steps,
            },
        }


# --------------------------------------------------------------------------- #
# buffer schema
# --------------------------------------------------------------------------- #
def test_roundtrip_is_exact():
    samples = _full_group(0, "snap-a") + _full_group(1, "snap-b")
    payload = buffer_schema.serialize_samples(samples, CONFIG)
    config, restored = buffer_schema.deserialize_samples(payload)
    assert config == CONFIG
    assert len(restored) == len(samples)
    for before, after in zip(samples, restored):
        assert torch.equal(before.tau, after.tau) and after.tau.dtype == torch.float32
        assert torch.equal(before.eps, after.eps) and after.eps.dtype == torch.float32
        for field in (
            "snapshot_id",
            "group_id",
            "group_member_index",
            "episode_id",
            "decision_index",
            "cfm_loss_old",
            "reward",
            "advantage",
        ):
            assert getattr(before, field) == getattr(after, field)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"snapshot_id": ""}, "non-empty string"),
        ({"group_id": -1}, "non-negative"),
        ({"group_member_index": 2}, "group_member_index"),
        ({"tau": torch.tensor([0.1, 0.2])}, "shape"),
        ({"tau": torch.tensor(0.1, dtype=torch.float64)}, "float32"),
        ({"eps": torch.zeros(4, 2)}, "shape"),
        ({"eps": torch.zeros(4, 3, dtype=torch.float64)}, "float32"),
        ({"eps": torch.full((4, 3), float("nan"))}, "non-finite"),
        ({"cfm_loss_old": -0.1}, ">= 0"),
        ({"cfm_loss_old": float("inf")}, "finite"),
        ({"reward": float("nan")}, "finite"),
        ({"advantage": float("inf")}, "finite"),
    ],
)
def test_invalid_samples_are_refused(overrides, match):
    with pytest.raises(buffer_schema.WamBufferError, match=match):
        buffer_schema.validate_sample(_sample(**overrides), CONFIG, context="unit")


def test_buffer_level_invariants():
    with pytest.raises(buffer_schema.WamBufferError, match="empty"):
        buffer_schema.serialize_samples([], CONFIG)

    duplicated = _full_group() + [_full_group()[0]]
    with pytest.raises(buffer_schema.WamBufferError, match="duplicate"):
        buffer_schema.serialize_samples(duplicated, CONFIG)

    incomplete = _full_group()[:1]
    with pytest.raises(buffer_schema.WamBufferError, match="incomplete"):
        buffer_schema.serialize_samples(incomplete, CONFIG)

    mixed = _full_group()
    mixed[1] = dataclasses.replace(mixed[1], snapshot_id="snap-other")
    with pytest.raises(buffer_schema.WamBufferError, match="mixes snapshots"):
        buffer_schema.serialize_samples(mixed, CONFIG)


def test_payload_tampering_is_refused():
    payload = buffer_schema.serialize_samples(_full_group(), CONFIG)

    wrong_schema = dict(payload)
    wrong_schema["schema"] = "wam-rollout-buffer-v0"
    with pytest.raises(buffer_schema.WamBufferError, match="schema"):
        buffer_schema.deserialize_samples(wrong_schema)

    missing = dict(payload)
    del missing["eps"]
    with pytest.raises(buffer_schema.WamBufferError, match="missing="):
        buffer_schema.deserialize_samples(missing)

    extra = dict(payload)
    extra["invented"] = 1
    with pytest.raises(buffer_schema.WamBufferError, match="unexpected="):
        buffer_schema.deserialize_samples(extra)

    truncated = dict(payload)
    truncated["reward"] = truncated["reward"][:-1]
    with pytest.raises(buffer_schema.WamBufferError, match="entries"):
        buffer_schema.deserialize_samples(truncated)


def test_group_size_below_two_is_refused():
    with pytest.raises(buffer_schema.WamBufferError, match="group_size"):
        buffer_schema.WamBufferConfig(action_horizon=4, action_dim=3, group_size=1)


# --------------------------------------------------------------------------- #
# grouping plan
# --------------------------------------------------------------------------- #
def test_plan_is_seed_reproducible_and_rank_independent():
    snaps = _snapshots(16)
    plan_a = sampler.plan_snapshot_groups(snaps, group_size=4, seed=7)
    plan_b = sampler.plan_snapshot_groups(snaps, group_size=4, seed=7)
    assert plan_a == plan_b
    assert sampler.plan_snapshot_groups(snaps, group_size=4, seed=8) != plan_a

    # The plan never saw a rank count; shards of ANY layout reassemble to the
    # identical plan, whole groups only, no overlap.
    for num_ranks in (1, 2, 4, 8):
        shards = [
            sampler.shard_plan(plan_a, rank=r, num_ranks=num_ranks)
            for r in range(num_ranks)
        ]
        flattened = [spec for shard in shards for spec in shard]
        assert flattened == list(plan_a.groups)
        assert len({spec.group_id for spec in flattened}) == len(plan_a.groups)


def test_drop_contract_is_explicit_and_deterministic():
    snaps = _snapshots(13)  # lcm(1,2,4,8) == 8 -> keep 8, drop 5
    plan = sampler.plan_snapshot_groups(snaps, group_size=2, seed=3)
    assert len(plan.groups) == 8
    assert len(plan.dropped_snapshot_ids) == 5
    kept = {spec.snapshot_id for spec in plan.groups}
    assert kept.isdisjoint(plan.dropped_snapshot_ids)
    assert kept | set(plan.dropped_snapshot_ids) == {s["snapshot_id"] for s in snaps}
    again = sampler.plan_snapshot_groups(snaps, group_size=2, seed=3)
    assert again.dropped_snapshot_ids == plan.dropped_snapshot_ids


def test_plan_contract_violations():
    with pytest.raises(sampler.SnapshotGroupError, match="cannot fill"):
        sampler.plan_snapshot_groups(_snapshots(7), group_size=2, seed=0)
    with pytest.raises(sampler.SnapshotGroupError, match="group_size"):
        sampler.plan_snapshot_groups(_snapshots(8), group_size=1, seed=0)
    duplicated = _snapshots(8)
    duplicated[3] = dict(duplicated[0])
    with pytest.raises(sampler.SnapshotGroupError, match="unique"):
        sampler.plan_snapshot_groups(duplicated, group_size=2, seed=0)
    plan = sampler.plan_snapshot_groups(_snapshots(8), group_size=2, seed=0)
    with pytest.raises(sampler.SnapshotGroupError, match="divisible"):
        sampler.shard_plan(plan, rank=0, num_ranks=3)
    with pytest.raises(sampler.SnapshotGroupError, match="rank"):
        sampler.shard_plan(plan, rank=8, num_ranks=8)


# --------------------------------------------------------------------------- #
# rollout collection through the fake driver
# --------------------------------------------------------------------------- #
def test_collect_runs_k_rollouts_per_snapshot_with_fresh_restores():
    snaps = _snapshots(8)
    by_id = {s["snapshot_id"]: s for s in snaps}
    plan = sampler.plan_snapshot_groups(snaps, group_size=3, seed=1)
    driver = FakePairedDriver(horizon=2)
    records = sampler.collect_group_rollouts(
        driver,
        plan.groups,
        group_size=3,
        base_seed=11,
        snapshots_by_id=by_id,
        max_decisions=5,
    )
    assert len(records) == len(plan.groups) * 3
    # Every member restored its own snapshot: one restore per record, in order.
    assert len(driver.restore_calls) == len(records)
    by_group: dict[int, list] = {}
    for record in records:
        by_group.setdefault(record.group_id, []).append(record)
    for spec in plan.groups:
        members = by_group[spec.group_id]
        assert {r.snapshot_id for r in members} == {spec.snapshot_id}
        assert sorted(r.group_member_index for r in members) == [0, 1, 2]
        assert len({r.member_seed for r in members}) == 3  # distinct member seeds
        assert all(r.decisions_executed == 2 and r.terminated for r in members)

    # Deterministic replay: same plan + seed + fresh driver => identical records.
    replay = sampler.collect_group_rollouts(
        FakePairedDriver(horizon=2),
        plan.groups,
        group_size=3,
        base_seed=11,
        snapshots_by_id=by_id,
        max_decisions=5,
    )
    assert replay == records


def test_collect_respects_max_decisions_and_missing_snapshot():
    snaps = _snapshots(8)
    by_id = {s["snapshot_id"]: s for s in snaps}
    plan = sampler.plan_snapshot_groups(snaps, group_size=2, seed=2)
    driver = FakePairedDriver(horizon=10)
    records = sampler.collect_group_rollouts(
        driver,
        plan.groups,
        group_size=2,
        base_seed=0,
        snapshots_by_id=by_id,
        max_decisions=4,
    )
    assert all(r.decisions_executed == 4 and not r.terminated for r in records)

    with pytest.raises(sampler.SnapshotGroupError, match="missing"):
        sampler.collect_group_rollouts(
            driver,
            plan.groups,
            group_size=2,
            base_seed=0,
            snapshots_by_id={},
            max_decisions=1,
        )


def test_records_bridge_into_the_buffer_schema():
    """records -> buffer rows -> [-1, group_size] scores: the GRPO input shape."""
    snaps = _snapshots(8)
    by_id = {s["snapshot_id"]: s for s in snaps}
    config = buffer_schema.WamBufferConfig(action_horizon=4, action_dim=3, group_size=2)
    plan = sampler.plan_snapshot_groups(snaps, group_size=2, seed=5)
    records = sampler.collect_group_rollouts(
        FakePairedDriver(horizon=1),
        plan.groups,
        group_size=2,
        base_seed=9,
        snapshots_by_id=by_id,
        max_decisions=1,
    )
    samples = [
        buffer_schema.WamRolloutSample(
            snapshot_id=r.snapshot_id,
            group_id=r.group_id,
            group_member_index=r.group_member_index,
            episode_id=r.episode_id,
            decision_index=0,
            tau=torch.tensor(0.5, dtype=torch.float32),
            eps=torch.zeros(4, 3, dtype=torch.float32),
            cfm_loss_old=1.0,
            reward=r.reward,
            advantage=None,
        )
        for r in records
    ]
    payload = buffer_schema.serialize_samples(samples, config)
    _, restored = buffer_schema.deserialize_samples(payload)
    ordered = sorted(restored, key=lambda s: (s.group_id, s.group_member_index))
    scores = torch.tensor([s.reward for s in ordered]).view(-1, config.group_size)
    assert scores.shape == (len(plan.groups), config.group_size)


# --------------------------------------------------------------------------- #
# adversarial-review hardening (2026-07-26 pre-push review)
# --------------------------------------------------------------------------- #
def _chunk_rows(group_id, snapshot_id, member, decisions, reward_each=0.5):
    return [
        _sample(
            group_id=group_id,
            snapshot_id=snapshot_id,
            group_member_index=member,
            decision_index=d,
            reward=reward_each,
        )
        for d in decisions
    ]


def test_multi_decision_rollouts_serialize_and_roundtrip():
    """One row per executed chunk: the schema's own stated unit.

    The pre-push review confirmed the original uniqueness key omitted
    decision_index, refusing every multi-chunk rollout; this locks the fix.
    Members may differ in decision count (early termination).
    """
    samples = (
        _chunk_rows(0, "snap-a", 0, [0, 1, 2])
        + _chunk_rows(0, "snap-a", 1, [0, 1])
        + _chunk_rows(1, "snap-b", 0, [0])
        + _chunk_rows(1, "snap-b", 1, [0, 1, 2])
    )
    payload = buffer_schema.serialize_samples(samples, CONFIG)
    _, restored = buffer_schema.deserialize_samples(payload)
    assert len(restored) == 9

    # Per-member reduction to one score, then the GRPO [-1, K] view.
    totals: dict[tuple[int, int], float] = {}
    for row in restored:
        key = (row.group_id, row.group_member_index)
        totals[key] = totals.get(key, 0.0) + row.reward
    scores = torch.tensor([totals[k] for k in sorted(totals)]).view(
        -1, CONFIG.group_size
    )
    assert scores.shape == (2, 2)


def test_duplicate_chunk_and_non_contiguous_decisions_are_refused():
    dup = _chunk_rows(0, "snap-a", 0, [0, 1]) + _chunk_rows(0, "snap-a", 1, [0])
    dup.append(dup[0])
    with pytest.raises(buffer_schema.WamBufferError, match="duplicate"):
        buffer_schema.serialize_samples(dup, CONFIG)

    gappy = (
        _chunk_rows(0, "snap-a", 0, [0, 2])  # missing decision 1
        + _chunk_rows(0, "snap-a", 1, [0])
    )
    with pytest.raises(buffer_schema.WamBufferError, match="non-contiguous"):
        buffer_schema.serialize_samples(gappy, CONFIG)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"action_horizon": 0, "action_dim": 3, "group_size": 2},
        {"action_horizon": 4.0, "action_dim": 3, "group_size": 2},
        {"action_horizon": 4, "action_dim": 3, "group_size": 2.0},
        {"action_horizon": 4, "action_dim": 3, "group_size": "2"},
        {"action_horizon": "abc", "action_dim": 3, "group_size": 2},
        {"action_horizon": 4, "action_dim": 3, "group_size": True},
    ],
)
def test_malformed_config_raises_wam_buffer_error(config_kwargs):
    """Every malformed config fails closed as WamBufferError, never TypeError."""
    with pytest.raises(buffer_schema.WamBufferError):
        buffer_schema.WamBufferConfig(**config_kwargs)


def test_malformed_config_inside_payload_is_refused():
    payload = buffer_schema.serialize_samples(_full_group(), CONFIG)
    tampered = dict(payload)
    tampered["config"] = {"action_horizon": 4.0, "action_dim": 3.0, "group_size": 2}
    with pytest.raises(buffer_schema.WamBufferError, match="int"):
        buffer_schema.deserialize_samples(tampered)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"cfm_loss_old": 1}, "finite float"),  # int is not a float here
        ({"group_id": 1.0}, "non-negative int"),
        ({"group_id": True}, "non-negative int"),
        ({"decision_index": True}, "non-negative int"),
        ({"group_member_index": True}, "group_member_index"),
    ],
)
def test_bool_and_numeric_lookalikes_are_refused(overrides, match):
    with pytest.raises(buffer_schema.WamBufferError, match=match):
        buffer_schema.validate_sample(_sample(**overrides), CONFIG, context="unit")


def test_tensor_payload_tampering_is_refused():
    payload = buffer_schema.serialize_samples(_full_group(), CONFIG)

    truncated = dict(payload)
    truncated["tau"] = truncated["tau"][:-1]
    with pytest.raises(buffer_schema.WamBufferError, match="rows"):
        buffer_schema.deserialize_samples(truncated)

    listified = dict(payload)
    listified["eps"] = listified["eps"].tolist()
    with pytest.raises(buffer_schema.WamBufferError, match="stacked tensor"):
        buffer_schema.deserialize_samples(listified)

    wrong_dtype = dict(payload)
    wrong_dtype["eps"] = wrong_dtype["eps"].to(torch.float64)
    with pytest.raises(buffer_schema.WamBufferError, match="float32"):
        buffer_schema.deserialize_samples(wrong_dtype)


def test_collect_fails_closed_on_missing_observation():
    """A driver omitting the observation on a non-terminal step must raise,
    not silently replay the restore-time observation for the whole rollout."""

    class BlindDriver(FakePairedDriver):
        def step_chunk(self, action):
            result = super().step_chunk(action)
            del result["observation"]
            return result

    snaps = _snapshots(8)
    by_id = {s["snapshot_id"]: s for s in snaps}
    plan = sampler.plan_snapshot_groups(snaps, group_size=2, seed=4)
    with pytest.raises(sampler.SnapshotGroupError, match="no observation"):
        sampler.collect_group_rollouts(
            BlindDriver(horizon=5),
            plan.groups,
            group_size=2,
            base_seed=0,
            snapshots_by_id=by_id,
            max_decisions=3,
        )
    # Terminal steps may omit the observation: horizon == max_decisions works.
    records = sampler.collect_group_rollouts(
        BlindDriver(horizon=1),
        plan.groups,
        group_size=2,
        base_seed=0,
        snapshots_by_id=by_id,
        max_decisions=1,
    )
    assert all(r.terminated for r in records)


def test_collect_validates_group_size_and_empty_groups():
    snaps = _snapshots(8)
    by_id = {s["snapshot_id"]: s for s in snaps}
    plan = sampler.plan_snapshot_groups(snaps, group_size=2, seed=0)
    for bad_k in (0, 1):
        with pytest.raises(sampler.SnapshotGroupError, match="group_size"):
            sampler.collect_group_rollouts(
                FakePairedDriver(),
                plan.groups,
                group_size=bad_k,
                base_seed=0,
                snapshots_by_id=by_id,
                max_decisions=1,
            )
    with pytest.raises(sampler.SnapshotGroupError, match="empty group"):
        sampler.collect_group_rollouts(
            FakePairedDriver(),
            [],
            group_size=2,
            base_seed=0,
            snapshots_by_id=by_id,
            max_decisions=1,
        )


def test_plan_ordering_is_hash_based_not_numpy():
    """The kept/dropped split must not depend on numpy's Generator stream
    (NEP 19 allows it to change across feature releases). Recompute the
    ordering from first principles and require exact agreement."""
    import hashlib as _hashlib

    snaps = _snapshots(13)
    plan = sampler.plan_snapshot_groups(snaps, group_size=2, seed=3)
    ids = [s["snapshot_id"] for s in snaps]
    expected = sorted(ids, key=lambda i: _hashlib.sha256(f"3:{i}".encode()).hexdigest())
    assert [spec.snapshot_id for spec in plan.groups] == expected[:8]
    assert list(plan.dropped_snapshot_ids) == expected[8:]
