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

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import (
    EvaluationRolloutControl,
    RolloutResult,
)
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
)
from rlinf.models.embodiment.wam_policy.evaluation import (
    EvaluationRouteSelection,
    EvaluationRoutingMode,
)
from rlinf.models.embodiment.wam_policy.routing_state import PendingRouteTracker
from rlinf.runners.fastwam_libero_eval_collector import EvaluationArtifactShard
from rlinf.workers.env import env_worker as env_worker_module
from rlinf.workers.env.env_worker import (
    EnvWorker,
    _mark_terminal_gate_unused,
    _merge_evaluation_rollout_results,
)
from rlinf.workers.rollout.hf.huggingface_worker import (
    MultiStepRolloutWorker,
    _build_evaluation_rollout_result,
)


def _records(
    *,
    routes=(1, 0),
    forced=(True, False),
    chunk_ids=(0, 1),
    episode_ids=(0, 0),
    source_chunk_ids=(-1, 0),
) -> tuple[ChunkRouteRecord, GateDecisionRecord, EvaluationRouteSelection]:
    route_tensor = torch.tensor(routes, dtype=torch.long)
    route = ChunkRouteRecord(
        route_used=route_tensor,
        route_was_forced=torch.tensor(forced, dtype=torch.bool),
        chunk_ids=torch.tensor(chunk_ids, dtype=torch.long),
        episode_ids=torch.tensor(episode_ids, dtype=torch.long),
        route_source_chunk_ids=torch.tensor(source_chunk_ids, dtype=torch.long),
        actor_versions=torch.zeros(len(routes), dtype=torch.long),
    )
    probability = torch.tensor([0.25, 0.75], dtype=torch.float32)[: len(routes)]
    emitted = GateDecisionRecord(
        next_route=torch.zeros(len(routes), dtype=torch.long),
        base_probability=probability,
        behavior_probability=probability,
        old_logprob=torch.log1p(-probability),
        epsilon=torch.zeros(len(routes), dtype=torch.float32),
        temperature=torch.ones(len(routes), dtype=torch.float32),
        valid=torch.ones(len(routes), dtype=torch.bool),
        source_chunk_ids=route.chunk_ids.clone(),
        episode_ids=route.episode_ids.clone(),
        actor_versions=route.actor_versions.clone(),
        kv_metadata=None,
    )
    selection = EvaluationRouteSelection(
        mode=EvaluationRoutingMode.FORCED_UNCOND,
        effective_next_route=emitted.next_route.clone(),
        counterfactual_next_route=(probability >= 0.5).to(torch.long),
    )
    return route, emitted, selection


def test_compact_eval_result_keeps_only_actions_and_typed_route_metadata() -> None:
    route, emitted, selection = _records()
    actions = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    result = {
        "prev_logprobs": torch.ones(2, 2, 3),
        "prev_values": torch.ones(2, 1),
        "forward_inputs": {
            "flow_chains": torch.ones(2, 3, 2, 3),
            "gate_kv_action_keys": torch.ones(2, 8),
            "critic_prefix": torch.ones(2, 8),
        },
        "route_info": route,
        "emitted_gate": emitted,
        "evaluation_selection": selection,
        "gate_latency_seconds": torch.tensor([0.01, 0.02], dtype=torch.float64),
        "gate_h2d_seconds": torch.zeros(2, dtype=torch.float64),
    }

    compact = _build_evaluation_rollout_result(actions, result)

    assert torch.equal(compact.actions, actions)
    assert compact.route_info is not None
    assert compact.emitted_gate is not None
    assert compact.evaluation_selection is not None
    assert compact.prev_logprobs is None
    assert compact.prev_values is None
    assert compact.bootstrap_values is None
    assert compact.versions is None
    assert compact.forward_inputs == {}
    assert compact.emitted_gate.kv_metadata is None
    assert torch.equal(
        compact.gate_latency_seconds,
        torch.tensor([0.01, 0.02], dtype=torch.float64),
    )
    assert torch.equal(compact.gate_h2d_seconds, torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("split_sizes", ([4, 2], [2, 4]))
def test_typed_eval_result_split_merge_preserves_all_records(split_sizes) -> None:
    route = ChunkRouteRecord(
        route_used=torch.tensor([1, 0, 1, 0, 1, 0]),
        route_was_forced=torch.tensor([True, False, True, False, True, False]),
        chunk_ids=torch.tensor([0, 1, 0, 1, 0, 1]),
        episode_ids=torch.tensor([0, 0, 0, 0, 0, 0]),
        route_source_chunk_ids=torch.tensor([-1, 0, -1, 0, -1, 0]),
        actor_versions=torch.zeros(6, dtype=torch.long),
    )
    probability = torch.linspace(0.1, 0.6, 6)
    emitted = GateDecisionRecord(
        next_route=(probability >= 0.5).to(torch.long),
        base_probability=probability,
        behavior_probability=probability,
        old_logprob=torch.log(
            torch.where(probability >= 0.5, probability, 1 - probability)
        ),
        epsilon=torch.zeros(6),
        temperature=torch.ones(6),
        valid=torch.ones(6, dtype=torch.bool),
        source_chunk_ids=route.chunk_ids.clone(),
        episode_ids=route.episode_ids.clone(),
        actor_versions=route.actor_versions.clone(),
    )
    selection = EvaluationRouteSelection(
        mode="matched_random",
        effective_next_route=emitted.next_route.clone(),
        counterfactual_next_route=(probability >= 0.5).to(torch.long),
        random_draws=torch.linspace(0.05, 0.55, 6, dtype=torch.float64),
    )
    result = RolloutResult(
        actions=torch.arange(12, dtype=torch.float32).view(6, 2),
        route_info=route,
        emitted_gate=emitted,
        evaluation_selection=selection,
        gate_latency_seconds=torch.linspace(0.01, 0.06, 6, dtype=torch.float64),
        gate_h2d_seconds=torch.zeros(6, dtype=torch.float64),
    )
    worker = object.__new__(MultiStepRolloutWorker)

    shards = worker._split_rollout_result(result, list(split_sizes))
    merged = RolloutResult.merge_rollout_results(shards)

    assert torch.equal(merged.actions, result.actions)
    assert torch.equal(merged.route_info.route_used, route.route_used)
    assert torch.equal(merged.emitted_gate.valid, emitted.valid)
    assert torch.equal(
        merged.evaluation_selection.effective_next_route,
        selection.effective_next_route,
    )
    assert torch.equal(
        merged.evaluation_selection.random_draws,
        selection.random_draws,
    )
    assert torch.equal(
        merged.gate_latency_seconds,
        result.gate_latency_seconds,
    )
    assert torch.equal(merged.gate_h2d_seconds, result.gate_h2d_seconds)


def test_eval_merge_accepts_typed_or_legacy_tensor_payloads_but_not_mixed() -> None:
    route, emitted, selection = _records()
    first = RolloutResult(
        actions=torch.ones(1, 3),
        route_info=route.split([1, 1])[0],
        emitted_gate=emitted.split([1, 1])[0],
        evaluation_selection=selection.split([1, 1])[0],
    )
    second = RolloutResult(
        actions=torch.zeros(1, 3),
        route_info=route.split([1, 1])[1],
        emitted_gate=emitted.split([1, 1])[1],
        evaluation_selection=selection.split([1, 1])[1],
    )

    typed = _merge_evaluation_rollout_results([first, second])
    legacy_tensor = _merge_evaluation_rollout_results(
        [torch.ones(1, 3), torch.zeros(1, 3)]
    )
    legacy_array = _merge_evaluation_rollout_results(
        [np.ones((1, 3)), np.zeros((1, 3))]
    )

    assert isinstance(typed, RolloutResult)
    assert typed.actions.shape == (2, 3)
    assert legacy_tensor.shape == (2, 3)
    assert legacy_array.shape == (2, 3)
    with pytest.raises(TypeError, match="mixed typed and legacy"):
        _merge_evaluation_rollout_results([first, torch.zeros(1, 3)])


def test_terminal_emission_is_retained_but_marked_unused() -> None:
    route, emitted, selection = _records()
    result = RolloutResult(
        actions=torch.ones(2, 3),
        route_info=route,
        emitted_gate=emitted,
        evaluation_selection=selection,
    )

    aligned = _mark_terminal_gate_unused(
        result,
        torch.tensor([False, True]),
    )

    assert emitted.valid.tolist() == [True, True]
    assert aligned.emitted_gate.valid.tolist() == [True, False]
    assert torch.equal(
        aligned.emitted_gate.base_probability,
        emitted.base_probability,
    )
    assert aligned.emitted_gate.source_chunk_ids.tolist() == [0, 1]
    assert torch.equal(
        aligned.evaluation_selection.effective_next_route,
        selection.effective_next_route,
    )


class _EvalEnv:
    def __init__(self) -> None:
        self.rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.terminations = torch.tensor(
            [[False, False], [False, True]],
            dtype=torch.bool,
        )
        self.truncations = torch.zeros(2, 2, dtype=torch.bool)

    def chunk_step(self, actions):
        del actions
        obs = {
            "states": torch.zeros(2, 3),
            "task_descriptions": ["first", "second"],
        }
        return (
            [obs],
            self.rewards,
            self.terminations,
            self.truncations,
            [{}],
        )


def test_env_evaluate_step_propagates_outcomes_into_next_reset_mask(
    monkeypatch,
) -> None:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "env_type": "libero",
                    "auto_reset": True,
                }
            }
        }
    )
    worker.model_cfg = OmegaConf.create(
        {
            "model_type": "fastwam_adaptive",
            "num_action_chunks": 2,
            "action_dim": 1,
        }
    )
    worker.eval_env_list = [_EvalEnv()]
    worker.eval_prev_done = [torch.zeros(2, dtype=torch.bool)]
    worker.use_external_reward_model = False
    worker.stage_num = 1
    worker.eval_num_envs_per_stage = 2
    worker.train_num_envs_per_stage = 2
    worker._rank = 0
    worker.enable_rlt = False
    monkeypatch.setattr(
        env_worker_module,
        "prepare_actions",
        lambda raw_chunk_actions, **_kwargs: raw_chunk_actions,
    )

    env_output, _ = worker.env_evaluate_step(torch.zeros(2, 2, 1), stage_id=0)
    next_input = worker._build_rollout_input_data(
        env_output.to_dict(),
        stage_id=0,
        eval_mode=True,
    )

    assert env_output.dones.tolist() == [False, True]
    assert torch.equal(env_output.terminations, worker.eval_env_list[0].terminations)
    assert torch.equal(env_output.truncations, worker.eval_env_list[0].truncations)
    assert torch.equal(env_output.rewards, worker.eval_env_list[0].rewards)
    assert next_input["fastwam_reset_mask"].tolist() == [False, True]


def _emit_uncond(
    tracker: PendingRouteTracker,
    env_ids: torch.Tensor,
    route: ChunkRouteRecord,
) -> None:
    tracker.emit(
        env_ids=env_ids,
        routes=torch.zeros_like(env_ids),
        source_chunk_ids=route.chunk_ids,
        episode_ids=route.episode_ids,
        actor_version=0,
    )


def test_two_env_staggered_resets_match_exact_delayed_route_timeline() -> None:
    tracker = PendingRouteTracker()
    env_ids = torch.tensor([10, 20])
    reset_masks = [
        torch.tensor([True, True]),
        torch.tensor([False, True]),
        torch.tensor([True, False]),
    ]
    expected_routes = [[1, 1], [0, 1], [1, 0]]
    expected_forced = [[True, True], [False, True], [True, False]]
    expected_sources = [[-1, -1], [0, -1], [-1, 0]]
    expected_episodes = [[0, 0], [0, 1], [1, 1]]

    for index, reset_mask in enumerate(reset_masks):
        route = tracker.consume(
            env_ids=env_ids,
            reset_mask=reset_mask,
            actor_version=0,
        )
        assert route.route_used.tolist() == expected_routes[index]
        assert route.route_was_forced.tolist() == expected_forced[index]
        assert route.route_source_chunk_ids.tolist() == expected_sources[index]
        assert route.episode_ids.tolist() == expected_episodes[index]
        _emit_uncond(tracker, env_ids, route)


def test_episode_ids_are_per_environment_and_reset_order_invariant() -> None:
    def collect(reset_order):
        tracker = PendingRouteTracker()
        observed = {10: [], 20: []}
        for env_id in reset_order:
            route = tracker.consume(
                env_ids=torch.tensor([env_id]),
                reset_mask=torch.tensor([True]),
                actor_version=0,
            )
            observed[env_id].append(route.episode_ids.item())
        return observed

    forward = collect([10, 20, 10, 20])
    reordered = collect([20, 10, 20, 10])

    assert forward == {10: [0, 1], 20: [0, 1]}
    assert reordered == forward


class _ResetOnlyEvalEnv:
    def reset(self):
        return (
            {
                "states": torch.zeros(2, 3),
                "task_descriptions": ["first", "second"],
            },
            {},
        )


def test_coupled_env_eval_validates_typed_rollout_batch_size() -> None:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "rollout": {"group_name": "rollout"},
            "env": {"eval": {"auto_reset": True}},
        }
    )
    worker.eval_rollout_epoch = 1
    worker.n_eval_chunk_steps = 1
    worker.stage_num = 1
    worker.eval_batch_size = 2
    worker.eval_num_envs_per_stage = 2
    worker.eval_prev_done = [torch.zeros(2, dtype=torch.bool)]
    worker.env_decoupled_mode = False
    worker.eval_enable_offload = False
    worker.evaluation_collector = None
    worker.eval_env_list = [_ResetOnlyEvalEnv()]
    worker._build_rollout_input_data = lambda _env_batch, **_kwargs: {
        "fastwam_env_ids": torch.tensor([10, 20])
    }
    worker.send_to = lambda **_kwargs: None
    typed_result = RolloutResult(actions=torch.ones(2, 2, 3))

    def recv_from(**kwargs):
        infer_batch_size_fn = kwargs["infer_batch_size_fn"]
        assert infer_batch_size_fn is not None
        assert infer_batch_size_fn(typed_result) == 2
        return typed_result

    worker.recv_from = recv_from
    worker.env_evaluate_step = lambda _actions, _stage_id, **_kwargs: (
        env_worker_module.EnvOutput(
            obs={
                "states": torch.zeros(2, 3),
                "task_descriptions": ["first", "second"],
            },
            dones=torch.zeros(2, dtype=torch.bool),
        ),
        {},
    )
    worker.finish_rollout = lambda mode: None

    evaluate_impl = EnvWorker.evaluate
    while hasattr(evaluate_impl, "__wrapped__"):
        evaluate_impl = evaluate_impl.__wrapped__
    metrics = evaluate_impl(worker, input_channel=object(), rollout_channel=object())

    assert metrics == {}


class _SnapshotWithActiveSlots:
    active_mask = (True, True)


class _CompleteAfterOneCollector:
    def __init__(self) -> None:
        self.is_complete = False
        self.finalized = False

    def snapshot_before_step(self, stage_id, env, env_ids):
        del stage_id, env, env_ids
        return _SnapshotWithActiveSlots()

    def augment_rollout_input(self, data, snapshot):
        del snapshot
        return data

    def record_chunk(self, **_kwargs) -> None:
        self.is_complete = True

    def build_rollout_stop_control(
        self, *, logical_batch_size: int
    ) -> EvaluationRolloutControl:
        return EvaluationRolloutControl(
            logical_batch_size=logical_batch_size,
            completed_episode_count=1,
            ledger_episode_count=1,
            ledger_sha256="a" * 64,
        )

    def finalize(self) -> EvaluationArtifactShard:
        self.finalized = True
        return EvaluationArtifactShard(
            rank=0,
            chunk_path="chunks.jsonl",
            episode_path="episodes.jsonl",
            action_contract_paths=("action-contract.json",),
            chunk_record_count=1,
            episode_record_count=1,
            chunk_sha256="a" * 64,
            episode_sha256="b" * 64,
            action_contract_file_sha256s=("c" * 64,),
            action_contract_canonical_sha256s=("d" * 64,),
            executable_action_contract_sha256="e" * 64,
            canonical_content_sha256="f" * 64,
        )


class _SnapshotWithInactiveSlot:
    active_mask = (True, False)


class _InactiveSlotCollector(_CompleteAfterOneCollector):
    def snapshot_before_step(self, stage_id, env, env_ids):
        del stage_id, env, env_ids
        return _SnapshotWithInactiveSlot()


def test_coupled_env_eval_passes_ledger_active_mask_to_environment_step() -> None:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "rollout": {"group_name": "rollout"},
            "env": {"eval": {"auto_reset": True}},
        }
    )
    worker.eval_rollout_epoch = 1
    worker.n_eval_chunk_steps = 1
    worker.stage_num = 1
    worker.eval_batch_size = 2
    worker.eval_num_envs_per_stage = 2
    worker.eval_prev_done = [torch.zeros(2, dtype=torch.bool)]
    worker.env_decoupled_mode = False
    worker.eval_enable_offload = False
    worker.evaluation_collector = _InactiveSlotCollector()
    worker.eval_env_list = [_ResetOnlyEvalEnv()]
    worker._build_rollout_input_data = lambda _env_batch, **_kwargs: {
        "fastwam_env_ids": torch.tensor([10, 20])
    }
    worker.send_to = lambda **_kwargs: None
    worker.recv_from = lambda **_kwargs: RolloutResult(actions=torch.ones(2, 2, 3))
    received_masks = []

    def env_evaluate_step(_actions, _stage_id, *, active_mask=None):
        received_masks.append(tuple(active_mask))
        return (
            env_worker_module.EnvOutput(
                obs={
                    "states": torch.zeros(2, 3),
                    "task_descriptions": ["first", "second"],
                },
                dones=torch.ones(2, dtype=torch.bool),
            ),
            {},
        )

    worker.env_evaluate_step = env_evaluate_step
    worker.finish_rollout = lambda mode: None

    evaluate_impl = EnvWorker.evaluate
    while hasattr(evaluate_impl, "__wrapped__"):
        evaluate_impl = evaluate_impl.__wrapped__
    evaluate_impl(worker, input_channel=object(), rollout_channel=object())

    assert received_masks == [(True, False)]


def test_coupled_env_eval_stops_when_frozen_ledger_is_complete() -> None:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "rollout": {"group_name": "rollout"},
            "env": {"eval": {"auto_reset": True}},
        }
    )
    worker.eval_rollout_epoch = 1
    worker.n_eval_chunk_steps = 4
    worker.stage_num = 1
    worker.eval_batch_size = 2
    worker.eval_num_envs_per_stage = 2
    worker.eval_prev_done = [torch.zeros(2, dtype=torch.bool)]
    worker.env_decoupled_mode = False
    worker.eval_enable_offload = False
    collector = _CompleteAfterOneCollector()
    worker.evaluation_collector = collector
    worker.eval_env_list = [_ResetOnlyEvalEnv()]
    worker._build_rollout_input_data = lambda _env_batch, **_kwargs: {
        "fastwam_env_ids": torch.tensor([10, 20])
    }
    sends = []
    worker.send_to = lambda **kwargs: sends.append(kwargs)
    receives = []
    typed_result = RolloutResult(actions=torch.ones(2, 2, 3))
    worker.recv_from = lambda **kwargs: receives.append(kwargs) or typed_result
    worker.env_evaluate_step = lambda _actions, _stage_id, **_kwargs: (
        env_worker_module.EnvOutput(
            obs={
                "states": torch.zeros(2, 3),
                "task_descriptions": ["first", "second"],
            },
            dones=torch.ones(2, dtype=torch.bool),
        ),
        {},
    )
    worker.finish_rollout = lambda mode: None

    evaluate_impl = EnvWorker.evaluate
    while hasattr(evaluate_impl, "__wrapped__"):
        evaluate_impl = evaluate_impl.__wrapped__
    result = evaluate_impl(worker, input_channel=object(), rollout_channel=object())

    assert len(sends) == 2
    assert isinstance(sends[1]["data"], EvaluationRolloutControl)
    assert sends[1]["data"].completed_episode_count == 1
    assert len(receives) == 1
    assert collector.finalized is True
    assert result["evaluation_artifact_shard"]["episode_record_count"] == 1


class _ImmediateAsyncValue:
    def __init__(self, value) -> None:
        self.value = value

    async def async_wait(self):
        return self.value


@pytest.mark.parametrize(
    ("model_type", "expect_typed"),
    [("fastwam_adaptive", True), ("mlp_policy", False)],
)
def test_coupled_eval_loop_sends_typed_fastwam_and_legacy_generic_payloads(
    model_type,
    expect_typed,
) -> None:
    worker = object.__new__(MultiStepRolloutWorker)
    worker.enable_offload = False
    worker.env_decoupled_mode = False
    worker.eval_rollout_epoch = 1
    worker.n_eval_chunk_steps = 1
    worker.num_pipeline_stages = 1
    worker.eval_batch_size = 2
    worker._rank = 0
    worker.cfg = OmegaConf.create({"env": {"group_name": "env"}})
    worker.model_cfg = OmegaConf.create({"model_type": model_type})
    env_output = {
        "obs": {"states": torch.ones(2, 3)},
        "final_obs": None,
        "fastwam_env_ids": torch.tensor([1, 2]),
        "fastwam_reset_mask": torch.tensor([True, True]),
    }
    worker.recv_from = lambda **_kwargs: _ImmediateAsyncValue(env_output)
    route, emitted, selection = _records()
    model_result = {
        "route_info": route,
        "emitted_gate": emitted,
        "evaluation_selection": selection,
    }
    worker._predict_rollout_actions = lambda *_args, **_kwargs: (
        torch.ones(2, 2, 3),
        model_result,
    )
    sends = []
    worker.send_to = lambda **kwargs: sends.append(kwargs)

    evaluate_impl = MultiStepRolloutWorker.evaluate
    while hasattr(evaluate_impl, "__wrapped__"):
        evaluate_impl = evaluate_impl.__wrapped__
    asyncio.run(
        evaluate_impl(
            worker,
            input_channel=object(),
            output_channel=object(),
        )
    )

    assert len(sends) == 1
    payload = sends[0]["data"]
    if expect_typed:
        assert isinstance(payload, RolloutResult)
        assert payload.route_info is not None
        assert payload.emitted_gate is not None
        assert payload.evaluation_selection is not None
        assert sends[0]["split_fn"] == worker._split_rollout_result
        assert payload.bootstrap_values is None
        assert payload.forward_inputs == {}
    else:
        assert isinstance(payload, torch.Tensor)
        assert payload.shape == (2, 2, 3)
        assert sends[0]["split_fn"] is None


def test_evaluation_rollout_control_split_merge_is_strict() -> None:
    control = EvaluationRolloutControl(
        logical_batch_size=3,
        completed_episode_count=30,
        ledger_episode_count=30,
        ledger_sha256="a" * 64,
    )

    shards = control.split([1, 2])
    merged = EvaluationRolloutControl.merge(shards)

    assert [shard.logical_batch_size for shard in shards] == [1, 2]
    assert merged == control
    with pytest.raises(ValueError, match="split sizes"):
        control.split([1, 1])
    with pytest.raises(ValueError, match="different provenance"):
        EvaluationRolloutControl.merge(
            [
                shards[0],
                EvaluationRolloutControl(
                    logical_batch_size=2,
                    completed_episode_count=30,
                    ledger_episode_count=30,
                    ledger_sha256="b" * 64,
                ),
            ]
        )
    with pytest.raises(ValueError, match="ledger-complete stop"):
        EvaluationRolloutControl(
            logical_batch_size=1,
            completed_episode_count=1,
            ledger_episode_count=1,
            ledger_sha256="a" * 64,
            command="continue",
        )


def test_coupled_rollout_eval_consumes_typed_ledger_stop() -> None:
    worker = object.__new__(MultiStepRolloutWorker)
    worker.enable_offload = False
    worker.env_decoupled_mode = False
    worker.eval_rollout_epoch = 1
    worker.n_eval_chunk_steps = 4
    worker.num_pipeline_stages = 1
    worker.eval_batch_size = 2
    worker._rank = 0
    worker.cfg = OmegaConf.create({"env": {"group_name": "env"}})
    worker.model_cfg = OmegaConf.create({"model_type": "fastwam_adaptive"})
    env_output = {
        "obs": {"states": torch.ones(2, 3)},
        "final_obs": None,
        "fastwam_env_ids": torch.tensor([1, 2]),
        "fastwam_reset_mask": torch.tensor([True, True]),
    }
    stop = EvaluationRolloutControl(
        logical_batch_size=2,
        completed_episode_count=30,
        ledger_episode_count=30,
        ledger_sha256="a" * 64,
    )
    inputs = iter((env_output, stop))
    receives = []

    def recv_from(**kwargs):
        receives.append(kwargs)
        return _ImmediateAsyncValue(next(inputs))

    worker.recv_from = recv_from
    route, emitted, selection = _records()
    predictions = []

    def predict(*_args, **_kwargs):
        predictions.append(True)
        return (
            torch.ones(2, 2, 3),
            {
                "route_info": route,
                "emitted_gate": emitted,
                "evaluation_selection": selection,
            },
        )

    worker._predict_rollout_actions = predict
    sends = []
    worker.send_to = lambda **kwargs: sends.append(kwargs)

    evaluate_impl = MultiStepRolloutWorker.evaluate
    while hasattr(evaluate_impl, "__wrapped__"):
        evaluate_impl = evaluate_impl.__wrapped__
    asyncio.run(
        evaluate_impl(
            worker,
            input_channel=object(),
            output_channel=object(),
        )
    )

    assert len(receives) == 2
    assert len(predictions) == 1
    assert len(sends) == 1
