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

    Same mechanism as test_snapshot_group_sampler.py: rlinf/__init__ and
    rlinf/models/__init__ pull omegaconf/torch-heavy registration at import
    time, which these unit tests do not need.
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
reward = importlib.import_module("rlinf.models.embodiment.wam_policy.reward")

CONFIG = buffer_schema.WamBufferConfig(action_horizon=4, action_dim=3, group_size=2)


def _bools(rows):
    return torch.tensor(rows, dtype=torch.bool)


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
        "reward": 0.0,
        "advantage": None,
    }
    base.update(overrides)
    return buffer_schema.WamRolloutSample(**base)


def _snapshots(count: int):
    return [
        {"snapshot_id": f"snap-{i}", "episode_id": f"ep-{i}", "payload": {"i": i}}
        for i in range(count)
    ]


class FakePairedDriver:
    """PairedCollectorDriver-shaped adapter over a deterministic toy world.

    Mirrors the fixture in test_snapshot_group_sampler.py so the integration
    test exercises the real sampler -> buffer -> reward pipeline seam.
    """

    def __init__(self, horizon: int = 3):
        self.horizon = horizon
        self.paired_metadata = {"driver": "fake"}
        self.restore_calls: list[str] = []
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
# terminal_success_return - semantics
# --------------------------------------------------------------------------- #
def test_success_before_done_scores_one():
    returns = reward.terminal_success_return(
        _bools([[False, True, False]]), _bools([[False, False, True]])
    )
    assert torch.equal(returns, torch.tensor([1.0]))
    assert returns.dtype == torch.float32


def test_success_at_the_done_step_scores_one():
    returns = reward.terminal_success_return(
        _bools([[False, False, True]]), _bools([[False, False, True]])
    )
    assert torch.equal(returns, torch.tensor([1.0]))


def test_never_success_scores_zero_with_and_without_done():
    returns = reward.terminal_success_return(
        _bools([[False, False, False], [False, False, False]]),
        _bools([[False, False, True], [False, False, False]]),  # truncated row
    )
    assert torch.equal(returns, torch.tensor([0.0, 0.0]))


def test_sticky_done_padding_is_tolerated():
    # Done padded True after termination; success AT the first done is fine.
    returns = reward.terminal_success_return(
        _bools([[False, True, False]]), _bools([[False, True, True]])
    )
    assert torch.equal(returns, torch.tensor([1.0]))


def test_gamma_discounting_hand_computed():
    successes = _bools(
        [
            [True, False, False],  # success at step 0 -> gamma**0 == 1.0
            [False, False, True],  # success at step 2 -> 0.5**2 == 0.25
            [False, False, False],  # never -> 0.0
        ]
    )
    dones = _bools(
        [
            [True, False, False],
            [False, False, True],
            [False, False, True],
        ]
    )
    returns = reward.terminal_success_return(successes, dones, gamma=0.5)
    assert torch.equal(returns, torch.tensor([1.0, 0.25, 0.0]))


def test_success_after_done_is_refused():
    with pytest.raises(reward.WamRewardError, match=r"success-after-done"):
        reward.terminal_success_return(
            _bools([[False, False, True]]), _bools([[True, False, False]])
        )
    # The offending row index is reported.
    with pytest.raises(reward.WamRewardError, match=r"rows \[1\]"):
        reward.terminal_success_return(
            _bools([[False, True], [False, True]]),
            _bools([[False, True], [True, True]]),
        )


@pytest.mark.parametrize(
    "successes, dones, match",
    [
        (torch.zeros(1, 3), _bools([[False, False, True]]), "bool tensor"),
        (_bools([[False, True]]), torch.zeros(1, 2), "bool tensor"),
        (_bools([[False, True]]), _bools([[False, False, True]]), "shape"),
        (_bools([False, True]), _bools([False, True]), "2-D"),
        (
            torch.zeros(0, 3, dtype=torch.bool),
            torch.zeros(0, 3, dtype=torch.bool),
            "non-empty",
        ),
        ([[False, True]], _bools([[False, True]]), "torch.Tensor"),
    ],
)
def test_malformed_success_inputs_are_refused(successes, dones, match):
    with pytest.raises(reward.WamRewardError, match=match):
        reward.terminal_success_return(successes, dones)


@pytest.mark.parametrize("gamma", [0.0, -0.5, 1.5, float("nan"), float("inf"), True])
def test_bad_gamma_is_refused(gamma):
    with pytest.raises(reward.WamRewardError, match="gamma"):
        reward.terminal_success_return(
            _bools([[False, True]]), _bools([[False, True]]), gamma=gamma
        )


# --------------------------------------------------------------------------- #
# group_normalized_advantage
# --------------------------------------------------------------------------- #
def test_two_group_hand_computed_case_is_exact():
    # Group 0: [3, 1] -> mean 2, population std 1 -> (x-2)/(1+1) = +/-0.5
    # Group 1: [10, 4] -> mean 7, population std 3 -> (x-7)/(3+1) = +/-0.75
    # Every quantity is exactly representable in float32 with eps=1.0.
    returns = torch.tensor([3.0, 1.0, 10.0, 4.0])
    advantages, diag = reward.group_normalized_advantage(
        returns, [0, 0, 1, 1], eps=1.0
    )
    assert torch.equal(advantages, torch.tensor([0.5, -0.5, 0.75, -0.75]))
    assert diag["group_mean"] == {0: 2.0, 1: 7.0}
    assert diag["group_std"] == {0: 1.0, 1: 3.0}
    assert diag["num_samples"] == 4 and diag["num_groups"] == 2
    assert diag["zero_variance_group_count"] == 0
    assert diag["zero_variance_group_ids"] == ()
    assert diag["effective_group_count"] == 2
    assert diag["effective_sample_count"] == 4


def test_permutation_of_samples_permutes_advantages_identically():
    advantages, _ = reward.group_normalized_advantage(
        torch.tensor([1.0, 10.0, 3.0, 4.0]), [0, 1, 0, 1], eps=1.0
    )
    assert torch.equal(advantages, torch.tensor([-0.5, 0.75, 0.5, -0.75]))


def test_zero_variance_group_is_exactly_zero_and_counted():
    # 0.1 is NOT exactly representable: the float mean of three copies can be
    # a hair off 0.1, so a std==0 check could leak eps-scaled noise. The
    # all-equal rule must still produce EXACT zeros.
    returns = torch.tensor([0.1, 0.1, 0.1, 2.0, 6.0])
    advantages, diag = reward.group_normalized_advantage(
        returns, [5, 5, 5, 9, 9], eps=1.0
    )
    assert torch.equal(advantages[:3], torch.zeros(3))
    # Group 9: mean 4, population std 2 -> (x-4)/(2+1) = -2/3, +2/3.
    assert torch.allclose(advantages[3:], torch.tensor([-2.0 / 3.0, 2.0 / 3.0]))
    assert diag["zero_variance_group_ids"] == (5,)
    assert diag["zero_variance_group_count"] == 1
    assert diag["effective_group_count"] == 1
    assert diag["effective_sample_count"] == 2
    assert diag["num_samples"] == 5 and diag["num_groups"] == 2


def test_eps_guards_near_zero_variance_and_is_validated():
    tiny = reward.group_normalized_advantage(
        torch.tensor([1.0, 1.0 + 1e-6]), [0, 0]
    )[0]
    assert bool(torch.isfinite(tiny).all())
    for bad_eps in (0.0, -1e-8, float("nan")):
        with pytest.raises(reward.WamRewardError, match="eps"):
            reward.group_normalized_advantage(
                torch.tensor([1.0, 2.0]), [0, 0], eps=bad_eps
            )


def test_group_ids_accepted_as_int_tensor():
    advantages, _ = reward.group_normalized_advantage(
        torch.tensor([3.0, 1.0]), torch.tensor([4, 4], dtype=torch.int64), eps=1.0
    )
    assert torch.equal(advantages, torch.tensor([0.5, -0.5]))


@pytest.mark.parametrize(
    "returns, group_ids, match",
    [
        (torch.tensor([1.0, 2.0]), [0], "2 returns"),
        (torch.tensor([1.0, 2.0, 3.0]), [0, 0, 1], "group 1 has 1"),
        (torch.tensor([]), [], "empty"),
        (torch.tensor([[1.0], [2.0]]), [0, 0], "1-D"),
        (torch.tensor([1, 2]), [0, 0], "floating"),
        (torch.tensor([1.0, float("nan")]), [0, 0], "non-finite"),
        (torch.tensor([1.0, 2.0]), [0, True], "int"),
        (torch.tensor([1.0, 2.0]), [-1, -1], ">= 0"),
        (torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]), "int32/int64"),
        ([1.0, 2.0], [0, 0], "torch.Tensor"),
    ],
)
def test_malformed_advantage_inputs_are_refused(returns, group_ids, match):
    with pytest.raises(reward.WamRewardError, match=match):
        reward.group_normalized_advantage(returns, group_ids)


# --------------------------------------------------------------------------- #
# demo anchor schedule
# --------------------------------------------------------------------------- #
def test_constant_schedule_is_pinned():
    schedule = reward.AnchorSchedule.constant(0.05)
    for step in (0, 1, 17, 10**6):
        assert reward.demo_anchor_weight(step, schedule=schedule) == 0.05


def test_linear_decay_schedule_is_pinned():
    schedule = reward.AnchorSchedule.linear_decay(start=1.0, end=0.25, decay_steps=4)
    expected = {0: 1.0, 1: 0.8125, 2: 0.625, 3: 0.4375, 4: 0.25, 9: 0.25, 100: 0.25}
    for step, value in expected.items():
        assert reward.demo_anchor_weight(step, schedule=schedule) == value


def test_decay_to_zero_switches_the_anchor_off():
    schedule = reward.AnchorSchedule.linear_decay(start=0.5, end=0.0, decay_steps=2)
    assert reward.demo_anchor_weight(0, schedule=schedule) == 0.5
    assert reward.demo_anchor_weight(2, schedule=schedule) == 0.0
    assert reward.demo_anchor_weight(50, schedule=schedule) == 0.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"kind": "constant", "start": -0.1}, ">= 0"),
        ({"kind": "linear_decay", "start": -1.0, "end": 0.0, "decay_steps": 2}, ">= 0"),
        ({"kind": "linear_decay", "start": 1.0, "end": -0.5, "decay_steps": 2}, ">= 0"),
        (
            {"kind": "linear_decay", "start": 0.1, "end": 0.5, "decay_steps": 2},
            "increase",
        ),
        (
            {"kind": "linear_decay", "start": 1.0, "end": 0.0, "decay_steps": 0},
            "decay_steps",
        ),
        ({"kind": "cosine", "start": 1.0}, "kind"),
        ({"kind": "constant", "start": 1.0, "decay_steps": 5}, "defaults"),
        ({"kind": "constant", "start": 1.0, "end": 0.5}, "defaults"),
        ({"kind": "constant", "start": float("nan")}, "finite"),
        ({"kind": "constant", "start": True}, "real number"),
    ],
)
def test_malformed_schedules_are_refused(kwargs, match):
    with pytest.raises(reward.WamRewardError, match=match):
        reward.AnchorSchedule(**kwargs)


def test_demo_anchor_weight_validates_step_and_schedule():
    schedule = reward.AnchorSchedule.constant(0.1)
    with pytest.raises(reward.WamRewardError, match="non-negative int"):
        reward.demo_anchor_weight(-1, schedule=schedule)
    with pytest.raises(reward.WamRewardError, match="non-negative int"):
        reward.demo_anchor_weight(True, schedule=schedule)
    with pytest.raises(reward.WamRewardError, match="AnchorSchedule"):
        reward.demo_anchor_weight(0, schedule={"kind": "constant", "start": 0.1})


# --------------------------------------------------------------------------- #
# trajectory_value_map
# --------------------------------------------------------------------------- #
def test_trajectory_value_map_zips_and_fails_closed():
    keys = [(0, 0), (0, 1)]
    mapping = reward.trajectory_value_map(keys, torch.tensor([1.0, 0.0]))
    assert mapping == {(0, 0): 1.0, (0, 1): 0.0}

    with pytest.raises(reward.WamRewardError, match="duplicate"):
        reward.trajectory_value_map([(0, 0), (0, 0)], torch.tensor([1.0, 0.0]))
    with pytest.raises(reward.WamRewardError, match="keys for"):
        reward.trajectory_value_map(keys, torch.tensor([1.0]))
    with pytest.raises(reward.WamRewardError, match="non-finite"):
        reward.trajectory_value_map(keys, torch.tensor([1.0, float("inf")]))
    with pytest.raises(reward.WamRewardError, match="tuple"):
        reward.trajectory_value_map([(0, 0), "0-1"], torch.tensor([1.0, 0.0]))


# --------------------------------------------------------------------------- #
# fill_reward_fields
# --------------------------------------------------------------------------- #
def _two_group_rows():
    rows = []
    for group_id, snapshot_id in ((0, "snap-a"), (1, "snap-b")):
        for member in range(CONFIG.group_size):
            for decision in range(3 if member == 0 else 2):
                rows.append(
                    _sample(
                        group_id=group_id,
                        snapshot_id=snapshot_id,
                        group_member_index=member,
                        decision_index=decision,
                    )
                )
    return rows


def test_fill_broadcasts_per_trajectory_values_onto_every_chunk_row():
    rows = _two_group_rows()
    returns = {(0, 0): 1.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}
    advantages = {(0, 0): 0.5, (0, 1): -0.5, (1, 0): 0.0, (1, 1): 0.0}
    filled = reward.fill_reward_fields(rows, returns, advantages, config=CONFIG)
    assert len(filled) == len(rows)
    for row in filled:
        key = (row.group_id, row.group_member_index)
        assert row.reward == returns[key]
        assert row.advantage == advantages[key]
    # Inputs are untouched (frozen dataclasses, new list).
    assert all(row.advantage is None for row in rows)


def test_fill_rejects_injected_and_missing_trajectories():
    rows = _two_group_rows()
    returns = {(0, 0): 1.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}
    advantages = {(0, 0): 0.5, (0, 1): -0.5, (1, 0): 0.0, (1, 1): 0.0}

    injected = dict(returns)
    injected[(99, 0)] = 1.0  # group id from some other plan
    with pytest.raises(reward.WamRewardError, match="absent from the samples"):
        reward.fill_reward_fields(rows, injected, advantages, config=CONFIG)

    short = {key: value for key, value in advantages.items() if key != (1, 1)}
    with pytest.raises(reward.WamRewardError, match="missing trajectories"):
        reward.fill_reward_fields(rows, returns, short, config=CONFIG)

    poisoned = dict(advantages)
    poisoned[(0, 0)] = float("inf")
    with pytest.raises(reward.WamRewardError, match="finite"):
        reward.fill_reward_fields(rows, returns, poisoned, config=CONFIG)

    with pytest.raises(reward.WamRewardError, match="empty"):
        reward.fill_reward_fields([], returns, advantages, config=CONFIG)


def test_fill_reuses_buffer_schema_validation():
    # Drop every row of member 1 in group 1: the maps and samples agree, but
    # the buffer contract (complete groups) must still refuse the batch.
    rows = [
        row
        for row in _two_group_rows()
        if not (row.group_id == 1 and row.group_member_index == 1)
    ]
    returns = {(0, 0): 1.0, (0, 1): 0.0, (1, 0): 0.0}
    advantages = {(0, 0): 0.5, (0, 1): -0.5, (1, 0): 0.0}
    with pytest.raises(buffer_schema.WamBufferError, match="incomplete"):
        reward.fill_reward_fields(rows, returns, advantages, config=CONFIG)


# --------------------------------------------------------------------------- #
# integration: sampler -> returns -> advantages -> filled, validated buffer
# --------------------------------------------------------------------------- #
def test_end_to_end_rollouts_to_validated_buffer():
    snaps = _snapshots(8)
    by_id = {s["snapshot_id"]: s for s in snaps}
    plan = sampler.plan_snapshot_groups(snaps, group_size=CONFIG.group_size, seed=7)
    records = sampler.collect_group_rollouts(
        FakePairedDriver(horizon=2),
        plan.groups,
        group_size=CONFIG.group_size,
        base_seed=11,
        snapshots_by_id=by_id,
        max_decisions=3,
    )
    assert all(r.decisions_executed == 2 for r in records)

    # Synthetic outcome labels: in groups 0..5 member 0 succeeds at its final
    # chunk and member 1 fails; groups 6 and 7 fail entirely (zero variance).
    zero_variance_groups = {6, 7}
    keys = [(r.group_id, r.group_member_index) for r in records]
    successes = torch.zeros(len(records), 2, dtype=torch.bool)
    dones = torch.zeros(len(records), 2, dtype=torch.bool)
    for row, record in enumerate(records):
        dones[row, 1] = True
        if (
            record.group_member_index == 0
            and record.group_id not in zero_variance_groups
        ):
            successes[row, 1] = True

    returns = reward.terminal_success_return(successes, dones)
    advantages, diag = reward.group_normalized_advantage(
        returns, [key[0] for key in keys], eps=0.5
    )
    # Contrastive groups: [1, 0] -> mean 0.5, population std 0.5,
    # (x - 0.5) / (0.5 + 0.5) = +/-0.5 exactly. Zero-variance groups: 0.0.
    expected = torch.tensor(
        [
            0.0 if key[0] in zero_variance_groups else (0.5 if key[1] == 0 else -0.5)
            for key in keys
        ]
    )
    assert torch.equal(advantages, expected)
    assert diag["num_groups"] == 8
    assert diag["zero_variance_group_count"] == 2
    assert sorted(diag["zero_variance_group_ids"]) == [6, 7]
    assert diag["effective_sample_count"] == 12

    rows = [
        _sample(
            snapshot_id=record.snapshot_id,
            group_id=record.group_id,
            group_member_index=record.group_member_index,
            episode_id=record.episode_id,
            decision_index=decision,
            reward=0.123,  # per-chunk env reward, superseded by the return
        )
        for record in records
        for decision in range(record.decisions_executed)
    ]
    filled = reward.fill_reward_fields(
        rows,
        reward.trajectory_value_map(keys, returns),
        reward.trajectory_value_map(keys, advantages),
        config=CONFIG,
    )

    # End-to-end: the filled batch passes the buffer contract and round-trips.
    payload = buffer_schema.serialize_samples(filled, CONFIG)
    _, restored = buffer_schema.deserialize_samples(payload)
    by_key = {
        (r.group_id, r.group_member_index): (float(ret), float(adv))
        for r, ret, adv in zip(records, returns, advantages)
    }
    for row in restored:
        expected_return, expected_advantage = by_key[
            (row.group_id, row.group_member_index)
        ]
        assert row.reward == expected_return
        assert row.advantage == expected_advantage

    # Injecting a value keyed by a foreign group id must be refused.
    tampered = reward.trajectory_value_map(keys, returns)
    tampered[(999, 0)] = 1.0
    with pytest.raises(reward.WamRewardError, match="injected group ids"):
        reward.fill_reward_fields(
            rows,
            tampered,
            reward.trajectory_value_map(keys, advantages),
            config=CONFIG,
        )
