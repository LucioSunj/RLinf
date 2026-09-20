# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_route_neutral_online import _compose, _replay_gate_features

from rlinf.algorithms.advantages import apply_fastwam_chunk_cost
from rlinf.algorithms.fastwam_dual_ppo import compute_fastwam_dual_ppo_loss
from rlinf.config_contracts import (
    build_fastwam_checkpoint_contract,
    validate_fastwam_eval_model_contract,
    validate_fastwam_training_checkpoint_contract,
)
from rlinf.models.embodiment.wam_policy.adaptive_policy import (
    FastWAMAdaptivePolicyConfig,
)
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import OnlineIDMBCConfig
from rlinf.models.embodiment.wam_policy.pad_rv.audit import (
    summarize_pad_frozen_rollout_state,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_contracts import (
    RouteNeutralGateInputContract,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PhysicalStateHistoryTracker,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.actor import (
    align_current_step_trainable_advantages,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.config import (
    validate_route_neutral_online_idm_bc_training_config,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.policy import (
    RouteNeutralOnlineIDMBCFastWAMPolicy,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.runtime import (
    RouteNeutralOnlineIDMTeacherLiberoRuntime,
)
from rlinf.runners.fastwam_idm_cost_control import (
    aggregate_fastwam_idm_cost_observation,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _compose_extra(monkeypatch):
    return _compose(monkeypatch, overrides=["+pure_uncond_rollouts=extra6"])


def test_extra_uncond_preset_adds_six_slots_to_thirty_mixed(monkeypatch) -> None:
    cfg = _compose_extra(monkeypatch)
    validate_route_neutral_online_idm_bc_training_config(cfg)
    assert cfg.env.train.total_num_envs == 36
    assert cfg.actor.model.route_neutral_online.pure_uncond_num_envs == 6
    assert cfg.rollout.model.route_neutral_online.pure_uncond_num_envs == 6
    assert cfg.actor.global_batch_size == 252
    assert cfg.algorithm.uncond_flow_ppo.loss_weight == 1.0
    assert cfg.algorithm.uncond_idm_bc.loss_weight == 0.2
    base = _compose(monkeypatch)
    assert "pure_uncond_num_envs" not in base.actor.model.route_neutral_online
    assert base.env.train.total_num_envs == 30


@pytest.mark.parametrize("count", [-1, True, 1.5, 36])
def test_invalid_pure_uncond_counts_fail_before_model_loading(monkeypatch, count):
    cfg = _compose_extra(monkeypatch)
    for owner in (cfg.actor, cfg.rollout):
        owner.model.route_neutral_online.pure_uncond_num_envs = count
    with pytest.raises(ValueError, match="pure_uncond_num_envs"):
        validate_route_neutral_online_idm_bc_training_config(cfg)


def test_actor_and_rollout_must_agree_on_pure_slots(monkeypatch) -> None:
    cfg = _compose_extra(monkeypatch)
    cfg.rollout.model.route_neutral_online.pure_uncond_num_envs = 5
    with pytest.raises(ValueError, match="profiles differ"):
        validate_route_neutral_online_idm_bc_training_config(cfg)


@pytest.mark.parametrize("balanced", [False, True])
def test_extra_slots_preserve_shared_gpu_mixed_geometry(monkeypatch, balanced):
    cfg = _compose(
        monkeypatch,
        (
            "libero_10_ppo_fastwam_route_neutral_online_all"
            if balanced
            else "libero_10_ppo_fastwam_route_neutral_online_formal"
        ),
        ([] if balanced else ["+route_neutral_online_perfopt=task3_28"])
        + [
            f"env.train.total_num_envs={56 if balanced else 42}",
            "+actor.model.route_neutral_online.pure_uncond_num_envs=14",
            "+rollout.model.route_neutral_online.pure_uncond_num_envs=14",
        ],
    )
    validate_route_neutral_online_idm_bc_training_config(cfg)
    assert cfg.actor.global_batch_size == 196
    assert cfg.actor.micro_batch_size == 4


def test_training_resume_binds_pure_slots_but_evaluation_does_not(monkeypatch):
    cfg = _compose_extra(monkeypatch)
    saved = build_fastwam_checkpoint_contract(cfg, world_size=1)
    validate_fastwam_training_checkpoint_contract(
        saved, saved, allow_n4_to_three_rollout_expansion=False, owner="actor"
    )
    for owner in (cfg.actor, cfg.rollout):
        owner.model.route_neutral_online.pure_uncond_num_envs = 5
    changed = build_fastwam_checkpoint_contract(cfg, world_size=1)
    with pytest.raises(ValueError):
        validate_fastwam_training_checkpoint_contract(
            saved, changed, allow_n4_to_three_rollout_expansion=False, owner="actor"
        )
    base = _compose(monkeypatch)
    validate_fastwam_eval_model_contract(
        cfg.actor.model, base.rollout.model, load_critic=False
    )


class _Gate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, features):
        return self.logit.expand(features.batch_size)


class _Runtime(RouteNeutralOnlineIDMTeacherLiberoRuntime):
    """Replace pretrained computation while exercising real policy transport."""

    def __init__(self, actor) -> None:
        self.actor = actor
        self.physical_history = PhysicalStateHistoryTracker(
            RouteNeutralGateInputContract(2, 2)
        )
        self.route_neutral_visual = SimpleNamespace(layer_indices=(14,))

    def prepare_route_neutral_step(self, *, env_obs, **kwargs):
        return SimpleNamespace(
            gate_features=_replay_gate_features(env_obs["states"].shape[0])
        )

    def sample_routed_action_batch(self, *, routes, **kwargs):
        batch_size = routes.numel()
        return SimpleNamespace(
            actions=routes.float()[:, None],
            old_flow_logprobs=torch.zeros(batch_size, 1),
            flow_chains=torch.zeros(batch_size, 2, 1),
            denoise_indices=torch.zeros(batch_size, dtype=torch.long),
            forward_inputs={},
            action_execution_trace=None,
        )

    def replay_action_batch(self, *, route_info, **kwargs):
        return {
            "flow_logprobs": self.actor.weight.expand(route_info.shape[0], 1),
            "flow_entropy": torch.zeros(route_info.shape[0], 1),
        }

    def compute_online_idm_bc_loss(self, **kwargs):
        # Actual BC numerator/gradient coverage is in test_route_neutral_online.
        return SimpleNamespace(as_forward_outputs=lambda: {})


def _policy(count=2, microbatch=None, version=5):
    actor = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(actor.weight)
    policy = RouteNeutralOnlineIDMBCFastWAMPolicy(
        actor=actor,
        runtime=_Runtime(actor),
        lora_adapter=SimpleNamespace(lora_parameters=actor.parameters),
        gate=_Gate(),
        critic=None,
        config=FastWAMAdaptivePolicyConfig(
            formal_training_sampling_seed=42,
            training_rollout_microbatch_size=microbatch,
        ),
        online_idm_bc_config=OnlineIDMBCConfig(enabled=True, loss_weight=0.2),
        critic_warmup={
            "runner_updates": 5,
            "route_behavior": "independent_random",
            "idm_probability": 0.5,
            "freeze_gate": True,
            "freeze_cost_controller": True,
        },
        pure_uncond_num_envs=count,
    )
    policy.actor_version = version
    return policy


def _predict(policy, env_ids, *, reset=True, mode="train"):
    return policy.predict_action_batch(
        {
            "states": torch.zeros(len(env_ids), 2),
            "_fastwam_env_ids": torch.tensor(env_ids),
            "_fastwam_reset_mask": torch.full((len(env_ids),), reset),
        },
        mode=mode,
        compute_values=False,
    )


@pytest.mark.parametrize("version", [0, 5])
@pytest.mark.parametrize("microbatch", [None, 1, 2])
def test_whole_pure_trajectories_use_global_ids_across_resets(version, microbatch):
    policy = _policy(microbatch=microbatch, version=version)
    control = _policy(count=0, version=version)
    pure = torch.tensor([False, True, False, True])
    for reset in (True, False, True):
        actions, result = _predict(policy, [11, 0, 8, 1], reset=reset)
        _, baseline = _predict(control, [11, 0, 8, 1], reset=reset)
        route, emitted = result["route_info"], result["emitted_gate"]
        assert torch.equal(actions[:, 0], route.route_used.float())
        assert route.route_used[pure].tolist() == [0, 0]
        assert torch.equal(
            route.route_used[~pure], baseline["route_info"].route_used[~pure]
        )
        assert torch.equal(route.route_was_forced, pure)
        assert route.route_source_chunk_ids[pure].tolist() == [-1, -1]
        assert torch.equal(emitted.valid, ~pure)
        assert emitted.old_logprob[pure].tolist() == [0.0, 0.0]
        assert emitted.behavior_probability[pure].tolist() == [0.0, 0.0]


def test_evaluation_uses_its_requested_routing_for_every_slot() -> None:
    policy = _policy()
    policy.config = replace(policy.config, eval_routing_mode="forced_idm")
    actions, result = _predict(policy, [0, 1, 8], mode="eval")
    assert actions[:, 0].tolist() == [1, 1, 1]
    assert not result["route_info"].route_was_forced.any()


def test_route_state_restore_keeps_pure_and_mixed_trajectories() -> None:
    policy = _policy()
    _predict(policy, [0, 3])
    restored = _policy()
    restored.route_tracker.load_state_dict(
        copy.deepcopy(policy.route_tracker.state_dict())
    )
    left_action, left = _predict(policy, [0, 3], reset=False)
    right_action, right = _predict(restored, [0, 3], reset=False)
    assert torch.equal(left_action, right_action)
    for name in ("route_used", "route_was_forced", "chunk_ids", "episode_ids"):
        assert torch.equal(
            getattr(left["route_info"], name), getattr(right["route_info"], name)
        )
    assert torch.equal(
        left["emitted_gate"].old_logprob, right["emitted_gate"].old_logprob
    )


def test_pure_rows_train_flow_without_gate_gradients() -> None:
    policy = _policy()
    _, result = _predict(policy, [0, 1])
    route, emitted = result["route_info"], result["emitted_gate"]
    valid = torch.tensor([[[True], [False]]])
    alignment = align_current_step_trainable_advantages(
        advantages=torch.ones(1, 2, 1),
        route=ChunkRouteRecord.stack([route]),
        emitted=GateDecisionRecord.stack([emitted]),
        loss_mask=valid,
    )
    assert alignment.flow_valid_mask.tolist() == [[True, False]]
    assert not alignment.gate_valid_mask.any()
    output = policy.default_forward(
        result["forward_inputs"],
        route_info=route,
        emitted_gate=emitted,
        compute_values=False,
    )
    assert torch.equal(output["gate_logprobs"], emitted.old_logprob)
    loss, metrics = compute_fastwam_dual_ppo_loss(
        gate_logprobs=output["gate_logprobs"],
        gate_old_logprobs=emitted.old_logprob,
        gate_advantages=alignment.gate_advantages[0],
        gate_valid_mask=alignment.gate_valid_mask[0],
        gate_clip_ratio_low=0.2,
        gate_clip_ratio_high=0.2,
        gate_base_probabilities=output["gate_base_probabilities"],
        gate_behavior_probabilities=output["gate_behavior_probabilities"],
        gate_entropy_coefficient=0.01,
        flow_logprobs=output["flow_logprobs"],
        flow_old_logprobs=result["prev_logprobs"],
        flow_advantages=alignment.flow_advantages[0],
        route_used=route.route_used,
        flow_valid_mask=alignment.flow_valid_mask[0],
        flow_clip_ratio_low=0.2,
        flow_clip_ratio_high=0.2,
    )
    loss.backward()
    assert torch.count_nonzero(policy.actor.weight.grad) == 1
    assert policy.gate.logit.grad.item() == 0.0
    assert metrics["gate/sample_count"].item() == 0
    assert metrics["uncond_flow/sample_count"].item() == 1


def test_pure_rows_are_excluded_from_cost_and_controller_feedback() -> None:
    policy = _policy()
    policy.gate.logit.data.fill_(30)
    policy.config = replace(policy.config, gate_epsilon=0.0)
    _, result = _predict(policy, [0, 1, 2, 3])
    route = ChunkRouteRecord.stack([result["route_info"]])
    emitted = GateDecisionRecord.stack([result["emitted_gate"]])
    valid = torch.ones(1, 4, 1, dtype=torch.bool)
    alignment = align_current_step_trainable_advantages(
        advantages=torch.ones(1, 4, 1), route=route, emitted=emitted, loss_mask=valid
    )
    charge = EmbodiedFSDPActor._fastwam_charge_mask(
        charge_scope="eligible_nonforced", route=route, valid_mask=valid
    )
    cost = apply_fastwam_chunk_cost(
        environment_rewards=torch.ones(1, 4, 1),
        route_used=route.route_used,
        idm_cost=0.2,
        uncond_cost=0.3,
        valid_mask=valid,
        charge_mask=charge,
    )
    torch.testing.assert_close(
        cost.rewards, torch.tensor([[[1.0], [1.0], [0.8], [0.8]]])
    )
    audit = summarize_pad_frozen_rollout_state(
        route=route,
        emitted=emitted,
        eligible_gate_mask=alignment.gate_valid_mask,
        valid_mask=valid,
        kv_replay_backend="condition",
        max_bytes_per_sample=None,
    )
    assert audit.valid_uncond_chunk_count == 2
    assert audit.forced_route_count == 2
    assert audit.eligible_gate_decision_count == 2
    observation = aggregate_fastwam_idm_cost_observation(
        runner_step=5,
        actor_rollout_metrics=[audit.to_metrics()],
        guard_result={
            "eligible_gate_decision_count": 2,
            "eligible_idm_decision_count": 2,
            "eligible_idm_fraction": 1.0,
        },
    )
    assert observation.eligible_realized_fraction == 1.0
    assert observation.eligible_expected_fraction == pytest.approx(1.0)
    assert observation.executed_realized_fraction == 0.5
