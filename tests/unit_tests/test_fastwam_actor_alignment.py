# Copyright 2026 The RLinf Authors.
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

import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contracts = _load_module(
    "rlinf.models.embodiment.wam_policy.contracts",
    "rlinf/models/embodiment/wam_policy/contracts.py",
)


def _load_advantages_module():
    stubs = {
        "rlinf.algorithms.registry": types.SimpleNamespace(
            register_advantage=lambda _name: lambda function: function
        ),
        "rlinf.algorithms.utils": types.SimpleNamespace(
            kl_penalty=lambda *_args, **_kwargs: None,
            safe_normalize=lambda value, loss_mask=None: value,
        ),
        "rlinf.utils.utils": types.SimpleNamespace(
            masked_mean=lambda value, mask=None: value.mean()
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        return _load_module(
            "fastwam_advantages_under_test",
            "rlinf/algorithms/advantages.py",
        )
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


advantages = _load_advantages_module()
nested = _load_module(
    "fastwam_nested_dict_process_under_test",
    "rlinf/utils/nested_dict_process.py",
)
ChunkRouteRecord = contracts.ChunkRouteRecord
GateDecisionRecord = contracts.GateDecisionRecord
GateKVMetadata = contracts.GateKVMetadata
WAMRoute = contracts.WAMRoute


def _alignment_records():
    route_used = torch.tensor(
        [
            [WAMRoute.IDM, WAMRoute.IDM, WAMRoute.UNCOND, WAMRoute.IDM],
            [WAMRoute.UNCOND, WAMRoute.UNCOND, WAMRoute.UNCOND, WAMRoute.UNCOND],
            [WAMRoute.UNCOND, WAMRoute.IDM, WAMRoute.UNCOND, WAMRoute.UNCOND],
        ],
        dtype=torch.long,
    )
    forced = torch.tensor(
        [
            [True, True, False, True],
            [False, False, False, False],
            [False, True, False, False],
        ]
    )
    chunk_ids = torch.tensor(
        [
            [0, 0, 3, 0],
            [1, 1, 4, 1],
            [2, 0, 5, 2],
        ]
    )
    episode_ids = torch.tensor(
        [
            [10, 20, 10, 22],
            [10, 20, 10, 22],
            [10, 21, 10, 22],
        ]
    )
    source_ids = torch.tensor(
        [
            [-1, -1, 2, -1],
            [0, 0, 3, 0],
            [1, -1, 4, 1],
        ]
    )
    route = ChunkRouteRecord(
        route_used=route_used,
        route_was_forced=forced,
        chunk_ids=chunk_ids,
        episode_ids=episode_ids,
        route_source_chunk_ids=source_ids,
        actor_versions=torch.full_like(route_used, 7),
    )
    shape = route.shape
    behavior_probability = torch.full(shape, 0.4)
    emitted = GateDecisionRecord(
        next_route=torch.full(shape, int(WAMRoute.UNCOND), dtype=torch.long),
        base_probability=torch.full(shape, 0.375),
        behavior_probability=behavior_probability,
        old_logprob=torch.log1p(-behavior_probability),
        epsilon=torch.full(shape, 0.1),
        temperature=torch.ones(shape),
        valid=torch.ones(shape, dtype=torch.bool),
        source_chunk_ids=chunk_ids.clone(),
        episode_ids=episode_ids.clone(),
        actor_versions=torch.full_like(route_used, 7),
        kv_metadata=GateKVMetadata(
            layer_indices=(0, 2),
            denoise_timesteps=torch.ones(*shape, 1),
            total_bytes=torch.full(shape, 64, dtype=torch.long),
        ),
    )
    dones = torch.zeros(4, 4, 1, dtype=torch.bool)
    dones[2, 1] = True
    dones[3, 1] = True
    return route, emitted, dones


def test_gate_advantage_stays_at_source_and_crosses_rollout_epoch():
    route, emitted, dones = _alignment_records()
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)

    result = advantages.align_fastwam_policy_advantages(
        advantages=values,
        route=route,
        emitted=emitted,
        dones=dones,
        rollout_epoch=2,
        carry_pending_across_epochs=True,
    )

    assert result.gate_valid_mask.sum().item() == 8
    assert result.gate_advantages[0, 0].item() == values[1, 0, 0].item()
    assert result.gate_advantages[2, 0].item() == values[0, 2, 0].item()
    assert not result.gate_valid_mask[1, 1]
    assert not result.gate_valid_mask[2, 1]
    assert emitted.kv_metadata.total_bytes[2, 0].item() == 64


def test_non_auto_reset_does_not_pair_across_rollout_epochs():
    route, emitted, dones = _alignment_records()
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4, 1)

    result = advantages.align_fastwam_policy_advantages(
        advantages=values,
        route=route,
        emitted=emitted,
        dones=dones,
        rollout_epoch=2,
        carry_pending_across_epochs=False,
    )

    assert result.gate_valid_mask.sum().item() == 7
    assert not result.gate_valid_mask[2, 0]


def test_actor_version_mismatch_fails_closed():
    route, emitted, dones = _alignment_records()
    actor_versions = route.actor_versions.clone()
    actor_versions[0, 2] = 8
    route = replace(route, actor_versions=actor_versions)
    emitted_versions = emitted.actor_versions.clone()
    emitted_versions[0, 2] = 8
    emitted = replace(emitted, actor_versions=emitted_versions)

    with pytest.raises(ValueError, match="actor-version boundary"):
        advantages.align_fastwam_policy_advantages(
            advantages=torch.ones(3, 4, 1),
            route=route,
            emitted=emitted,
            dones=dones,
            rollout_epoch=2,
            carry_pending_across_epochs=True,
        )


def test_fixed_route_cost_is_subtracted_once_after_reward_aggregation():
    rewards = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        ]
    )
    routes = torch.tensor(
        [[WAMRoute.IDM, WAMRoute.UNCOND], [WAMRoute.UNCOND, WAMRoute.IDM]]
    )

    result = advantages.apply_fastwam_chunk_cost(
        environment_rewards=rewards,
        route_used=routes,
        idm_cost=2.0,
        uncond_cost=0.25,
    )

    assert result.rewards.shape == (2, 2, 1)
    assert torch.equal(result.costs[..., 0], torch.tensor([[2.0, 0.25], [0.25, 2.0]]))
    assert torch.equal(
        result.rewards[..., 0], torch.tensor([[4.0, 14.75], [1.25, 1.0]])
    )


def test_chunk_cost_audit_reconciles_forced_eligible_and_padded_routes():
    route, _, _ = _alignment_records()
    rewards = torch.arange(36, dtype=torch.float32).reshape(3, 4, 3) / 10
    valid_mask = torch.ones(3, 4, 1, dtype=torch.bool)
    valid_mask[2, 3] = False
    result = advantages.apply_fastwam_chunk_cost(
        environment_rewards=rewards,
        route_used=route.route_used,
        idm_cost=0.25,
        valid_mask=valid_mask,
    )

    audit = advantages.summarize_fastwam_chunk_cost(
        environment_rewards=rewards,
        route=route,
        cost_result=result,
        idm_cost=0.25,
        valid_mask=valid_mask,
    )
    artifact = audit.to_artifact()

    assert artifact["schema"] == "fastwam-chunk-cost-audit-v1"
    assert artifact["valid_chunk_count"] == 11
    assert artifact["valid_idm_chunk_count"] == 4
    assert artifact["forced_idm_chunk_count"] == 4
    assert artifact["eligible_idm_chunk_count"] == 0
    assert artifact["valid_uncond_chunk_count"] == 7
    assert artifact["expected_cost_sum"] == 1.0
    assert artifact["actual_branch_costs"]["sum"] == 1.0
    assert artifact["shaped_reward_identity_max_abs_error"] == 0.0
    assert artifact["raw_primitive_rewards"]["count"] == 33
    metrics = audit.to_metrics()
    assert metrics["fastwam/cost/actual_chunk/sum"] == pytest.approx(1.0)
    assert metrics["fastwam/cost/actual_chunk/mean"] == pytest.approx(1 / 11)
    assert metrics["fastwam/reward/shaped_chunk/count"] == 11.0
    assert metrics["fastwam/cost/identity_max_abs_error"] == 0.0
    assert metrics["fastwam/branch_cost_sum"] == pytest.approx(1.0)
    json.dumps(artifact, sort_keys=True)


def _counterfactual_records():
    routes = torch.tensor(
        [
            [WAMRoute.IDM, WAMRoute.IDM],
            [WAMRoute.IDM, WAMRoute.UNCOND],
            [WAMRoute.UNCOND, WAMRoute.IDM],
        ],
        dtype=torch.long,
    )
    forced = torch.tensor([[True, True], [False, False], [False, False]])
    chunks = torch.tensor([[0, 0], [1, 1], [2, 2]])
    episodes = torch.tensor([[10, 20], [10, 20], [10, 20]])
    route = ChunkRouteRecord(
        route_used=routes,
        route_was_forced=forced,
        chunk_ids=chunks,
        episode_ids=episodes,
        route_source_chunk_ids=torch.tensor([[-1, -1], [0, 0], [1, 1]]),
        actor_versions=torch.full_like(routes, 3),
    )
    next_route = torch.tensor(
        [
            [WAMRoute.IDM, WAMRoute.UNCOND],
            [WAMRoute.UNCOND, WAMRoute.IDM],
            [WAMRoute.IDM, WAMRoute.UNCOND],
        ],
        dtype=torch.long,
    )
    behavior = torch.full(route.shape, 0.5)
    emitted = GateDecisionRecord(
        next_route=next_route,
        base_probability=behavior,
        behavior_probability=behavior,
        old_logprob=torch.full(route.shape, -torch.log(torch.tensor(2.0))),
        epsilon=torch.full(route.shape, 0.1),
        temperature=torch.ones(route.shape),
        valid=torch.ones(route.shape, dtype=torch.bool),
        source_chunk_ids=chunks.clone(),
        episode_ids=episodes.clone(),
        actor_versions=torch.full_like(routes, 3),
        kv_metadata=GateKVMetadata(
            layer_indices=(0,),
            denoise_timesteps=torch.ones(*route.shape, 1),
            total_bytes=torch.full(route.shape, 16, dtype=torch.long),
        ),
    )
    dones = torch.zeros(4, 2, 1, dtype=torch.bool)
    return route, emitted, dones


def test_counterfactual_cost_audit_is_read_only_and_lowers_idm_advantage():
    route, emitted, dones = _counterfactual_records()
    rewards = torch.zeros(3, 2, 2)
    rewards[-1, :, -1] = 1.0
    values = torch.zeros(4, 2, 1)
    mask = torch.ones(3, 2, 1, dtype=torch.bool)
    configured_cost = 0.025
    configured_rewards = advantages.apply_fastwam_chunk_cost(
        environment_rewards=rewards,
        route_used=route.route_used,
        idm_cost=configured_cost,
        valid_mask=mask,
    ).rewards
    configured_advantages, _ = advantages.compute_gae_advantages_and_returns(
        rewards=configured_rewards[..., 0],
        values=values[..., 0],
        dones=dones[..., 0],
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        loss_mask=mask[..., 0],
    )
    configured_alignment = advantages.align_fastwam_policy_advantages(
        advantages=configured_advantages.unsqueeze(-1),
        route=route,
        emitted=emitted,
        dones=dones,
        rollout_epoch=1,
        carry_pending_across_epochs=False,
        loss_mask=mask,
    )
    input_clones = {
        "rewards": rewards.clone(),
        "values": values.clone(),
        "routes": route.route_used.clone(),
    }
    rng_before = torch.random.get_rng_state().clone()

    audit = advantages.summarize_fastwam_counterfactual_costs(
        environment_rewards=rewards,
        route=route,
        emitted=emitted,
        dones=dones,
        values=values,
        valid_mask=mask,
        idm_costs=(0.0, 0.025, 0.05, 0.1),
        configured_idm_cost=configured_cost,
        configured_gate_advantages=configured_alignment.gate_advantages,
        gamma=0.99,
        gae_lambda=0.95,
        rollout_epoch=1,
        carry_pending_across_epochs=False,
    )
    artifact = audit.to_artifact()

    assert artifact["schema"] == "fastwam-counterfactual-cost-audit-v1"
    assert artifact["configured_alignment_max_abs_error"] == 0.0
    assert artifact["eligible_gate_decision_count"] == 4
    assert artifact["eligible_idm_decision_count"] == 2
    assert [entry["idm_cost"] for entry in artifact["entries"]] == [
        0.0,
        0.025,
        0.05,
        0.1,
    ]
    assert (
        artifact["entries"][0]["idm_destination_delta_from_zero"]["unnormalized"]["sum"]
        == 0.0
    )
    assert all(
        entry["idm_destination_delta_from_zero"]["unnormalized"]["sum"] < 0.0
        for entry in artifact["entries"][1:]
    )
    metrics = audit.to_metrics()
    assert audit.break_even_idm_cost is None
    assert artifact["break_even_idm_cost"] is None
    assert "fastwam/counterfactual/break_even_idm_cost" not in metrics
    assert metrics["fastwam/counterfactual/configured_idm_cost"] == 0.025
    assert metrics["fastwam/counterfactual/alignment_max_abs_error"] == 0.0
    assert metrics["fastwam/counterfactual/gate_advantage_normalized/count"] == 4.0
    assert metrics["fastwam/counterfactual/idm_delta_from_zero_unnormalized/sum"] < 0
    assert torch.equal(rewards, input_clones["rewards"])
    assert torch.equal(values, input_clones["values"])
    assert torch.equal(route.route_used, input_clones["routes"])
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    json.dumps(artifact, sort_keys=True)


def test_environment_reward_audit_counts_only_valid_success_signals():
    rewards = torch.zeros(3, 2, 4)
    rewards[1, 0, 2] = 1.0
    rewards[2, 1, 1] = 1.0
    rewards[0, 1, 0] = -0.5
    routes = torch.tensor(
        [
            [WAMRoute.IDM, WAMRoute.UNCOND],
            [WAMRoute.UNCOND, WAMRoute.IDM],
            [WAMRoute.IDM, WAMRoute.UNCOND],
        ]
    )
    valid_mask = torch.ones(3, 2, 1, dtype=torch.bool)
    valid_mask[2, 1] = False

    audit = advantages.summarize_fastwam_environment_rewards(
        environment_rewards=rewards,
        route_used=routes,
        valid_mask=valid_mask,
    )
    artifact = audit.to_artifact()

    assert artifact["schema"] == "fastwam-environment-reward-audit-v1"
    assert artifact["reward_shape"] == [3, 2, 4]
    assert artifact["reward_dtype"] == "torch.float32"
    assert artifact["total_value_count"] == 24
    assert artifact["valid_value_count"] == 20
    assert artifact["finite_value_count"] == 24
    assert artifact["nonfinite_value_count"] == 0
    assert artifact["positive_success_signal_count"] == 1
    assert artifact["successful_trajectory_count"] == 1
    assert artifact["total_chunk_count"] == 6
    assert artifact["valid_chunk_count"] == 5
    assert artifact["valid_idm_chunk_count"] == 3
    assert artifact["valid_uncond_chunk_count"] == 2
    assert artifact["valid_reward_min"] == -0.5
    assert artifact["valid_reward_max"] == 1.0
    assert artifact["valid_reward_sum"] == 0.5
    metrics = audit.to_metrics()
    assert metrics["fastwam/reward/raw_sum"] == pytest.approx(0.5)
    assert metrics["fastwam/reward/raw_mean"] == pytest.approx(0.025)
    assert metrics["fastwam/reward/raw_min"] == pytest.approx(-0.5)
    assert metrics["fastwam/reward/raw_max"] == pytest.approx(1.0)
    assert metrics["fastwam/reward/successful_trajectory_count"] == 1.0
    assert metrics["fastwam/successful_trajectory_count"] == 1.0
    json.dumps(artifact, sort_keys=True)
    audit.require_success_signal()


def test_rollout_state_audit_records_route_probabilities_and_stored_kv_bytes():
    route, emitted, dones = _alignment_records()
    alignment = advantages.align_fastwam_policy_advantages(
        advantages=torch.ones(3, 4, 1),
        route=route,
        emitted=emitted,
        dones=dones,
        rollout_epoch=2,
        carry_pending_across_epochs=True,
    )

    audit = advantages.summarize_fastwam_rollout_state(
        route=route,
        emitted=emitted,
        eligible_gate_mask=alignment.gate_valid_mask,
        valid_mask=torch.ones(3, 4, 1, dtype=torch.bool),
        kv_replay_backend="stored",
        max_bytes_per_sample=256,
    )
    artifact = audit.to_artifact()

    assert artifact["schema"] == "fastwam-rollout-state-audit-v1"
    assert artifact["decision_shape"] == [3, 4]
    assert artifact["total_decision_count"] == 12
    assert artifact["valid_chunk_count"] == 12
    assert artifact["valid_idm_chunk_count"] == 4
    assert artifact["valid_uncond_chunk_count"] == 8
    assert artifact["forced_route_count"] == 4
    assert artifact["executed_idm_fraction"] == pytest.approx(1 / 3)
    assert artifact["emitted_decision_count"] == 12
    assert artifact["eligible_gate_decision_count"] == 8
    assert artifact["eligible_idm_decision_count"] == 0
    assert artifact["eligible_idm_fraction"] == 0.0
    assert artifact["unused_emitted_decision_count"] == 4
    assert artifact["base_probability"] == {
        "count": 8,
        "minimum": pytest.approx(0.375),
        "maximum": pytest.approx(0.375),
        "mean": pytest.approx(0.375),
    }
    assert artifact["behavior_probability"] == {
        "count": 8,
        "minimum": pytest.approx(0.4),
        "maximum": pytest.approx(0.4),
        "mean": pytest.approx(0.4),
    }
    assert artifact["kv_replay_backend"] == "stored"
    assert artifact["kv_storage_dtype"] == "bfloat16"
    assert artifact["kv_layer_indices"] == [0, 2]
    assert artifact["kv_denoise_tap_count"] == 1
    assert artifact["kv_configured_max_bytes_per_sample"] == 256
    assert artifact["kv_all_emitted"] == {
        "sample_count": 12,
        "nonzero_sample_count": 12,
        "total_bytes": 768,
        "maximum_bytes_per_sample": 64,
    }
    assert artifact["kv_eligible"] == {
        "sample_count": 8,
        "nonzero_sample_count": 8,
        "total_bytes": 512,
        "maximum_bytes_per_sample": 64,
    }
    metrics = audit.to_metrics()
    assert metrics["fastwam/route/executed_idm_fraction"] == pytest.approx(1 / 3)
    assert metrics["fastwam/route/eligible_idm_fraction"] == 0.0
    assert metrics["fastwam/route/forced_fraction"] == pytest.approx(1 / 3)
    assert metrics["fastwam/gate/base_idm_probability_mean"] == pytest.approx(0.375)
    assert metrics["fastwam/gate/behavior_idm_probability_mean"] == pytest.approx(0.4)
    assert metrics["fastwam/kv/eligible_total_bytes"] == 512.0
    assert metrics["fastwam/eligible_idm_fraction"] == 0.0
    json.dumps(artifact, sort_keys=True)


@pytest.mark.parametrize("missing_metadata", [True, False])
def test_rollout_state_audit_fails_closed_for_missing_or_zero_stored_kv(
    missing_metadata: bool,
) -> None:
    route, emitted, _ = _alignment_records()
    metadata = emitted.kv_metadata
    if not missing_metadata:
        metadata = replace(metadata, total_bytes=torch.zeros_like(metadata.total_bytes))
    emitted = replace(emitted, kv_metadata=None if missing_metadata else metadata)

    with pytest.raises(ValueError, match="stored K/V"):
        advantages.summarize_fastwam_rollout_state(
            route=route,
            emitted=emitted,
            eligible_gate_mask=torch.ones_like(emitted.valid),
            valid_mask=None,
            kv_replay_backend="stored",
            max_bytes_per_sample=256,
        )


def test_environment_reward_audit_zero_success_fails_closed():
    audit = advantages.summarize_fastwam_environment_rewards(
        environment_rewards=torch.zeros(2, 1, 3),
        route_used=torch.tensor([[WAMRoute.IDM], [WAMRoute.UNCOND]]),
    )

    with pytest.raises(RuntimeError, match="zero positive sparse-success"):
        audit.require_success_signal()


def test_environment_reward_audit_nonfinite_fails_closed_after_reporting():
    rewards = torch.tensor([[[float("nan"), 1.0]]])
    audit = advantages.summarize_fastwam_environment_rewards(
        environment_rewards=rewards,
        route_used=torch.tensor([[WAMRoute.IDM]]),
    )

    assert audit.nonfinite_value_count == 1
    assert audit.positive_success_signal_count == 1
    with pytest.raises(RuntimeError, match="non-finite"):
        audit.require_success_signal()


@pytest.mark.parametrize(
    ("rewards", "routes", "valid_mask", "message"),
    [
        (torch.ones(1, 1, 1, dtype=torch.long), torch.ones(1, 1), None, "floating"),
        (torch.ones(1, 1, 1), torch.ones(2, 1), None, "match"),
        (
            torch.ones(1, 1, 1),
            torch.ones(1, 1, dtype=torch.long),
            torch.ones(1, 1),
            "torch.bool",
        ),
    ],
)
def test_environment_reward_audit_rejects_malformed_inputs(
    rewards: torch.Tensor,
    routes: torch.Tensor,
    valid_mask: torch.Tensor | None,
    message: str,
):
    with pytest.raises((TypeError, ValueError), match=message):
        advantages.summarize_fastwam_environment_rewards(
            environment_rewards=rewards,
            route_used=routes,
            valid_mask=valid_mask,
        )


def test_route_and_gate_dataclasses_survive_mapping_device_and_chunking():
    route, emitted, _ = _alignment_records()
    batch = {"route_info": route, "emitted_gate": emitted}

    mapped = nested.map_nested_tensors(batch, lambda tensor: tensor.clone())
    moved = nested.put_tensor_device(mapped, "cpu")
    chunks = nested.split_dict_to_chunk(moved, 2, dim=1)

    assert isinstance(chunks[0]["route_info"], ChunkRouteRecord)
    assert isinstance(chunks[0]["emitted_gate"], GateDecisionRecord)
    assert chunks[0]["route_info"].shape == (3, 2)
    assert chunks[1]["emitted_gate"].kv_metadata.batch_shape == (3, 2)
    assert chunks[0]["emitted_gate"].kv_metadata.layer_indices == (0, 2)


def test_route_records_survive_epoch_fold_and_train_flatten():
    route, emitted, _ = _alignment_records()

    def unfold_epoch_batch(tensor):
        return (
            tensor.reshape(3, 2, 2, *tensor.shape[2:])
            .transpose(0, 1)
            .reshape(6, 2, *tensor.shape[2:])
        )

    raw = nested.map_nested_tensors(
        {"route_info": route, "emitted_gate": emitted}, unfold_epoch_batch
    )
    merged = nested.merge_rollout_epoch_batch(raw, rollout_epoch=2)
    shuffle = torch.arange(11, -1, -1)
    flattened = nested.flatten_time_batch(merged, shuffle, field_name="routes")

    assert isinstance(flattened["route_info"], ChunkRouteRecord)
    assert flattened["route_info"].shape == (12,)
    assert flattened["emitted_gate"].shape == (12,)
    assert flattened["emitted_gate"].kv_metadata.batch_shape == (12,)
    assert torch.equal(
        flattened["route_info"].chunk_ids,
        route.chunk_ids.reshape(-1)[shuffle],
    )


def test_consuming_flatten_matches_regular_and_releases_sources():
    route, emitted, _ = _alignment_records()
    replay = torch.arange(12 * 8, dtype=torch.bfloat16).reshape(3, 4, 8)
    batch = {
        "forward_inputs": {
            "gate_kv_action_key": replay,
            "nested": {"gate_kv_action_value": replay + 1},
        },
        "route_info": route,
        "emitted_gate": emitted,
    }
    shuffle = torch.arange(11, -1, -1)
    regular_source = nested.map_nested_tensors(batch, torch.Tensor.clone)
    consuming_source = nested.map_nested_tensors(batch, torch.Tensor.clone)

    expected = nested.flatten_time_batch(
        regular_source,
        shuffle,
        field_name="batch",
    )
    actual = nested.flatten_time_batch_consuming(
        consuming_source,
        shuffle,
        field_name="batch",
    )

    assert consuming_source == {}
    assert torch.equal(
        actual["forward_inputs"]["gate_kv_action_key"],
        expected["forward_inputs"]["gate_kv_action_key"],
    )
    assert torch.equal(
        actual["forward_inputs"]["nested"]["gate_kv_action_value"],
        expected["forward_inputs"]["nested"]["gate_kv_action_value"],
    )
    assert torch.equal(actual["route_info"].chunk_ids, expected["route_info"].chunk_ids)
    assert torch.equal(
        actual["emitted_gate"].old_logprob,
        expected["emitted_gate"].old_logprob,
    )
