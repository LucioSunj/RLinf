# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastwam.models.wan22.condition_kv import ConditionLayerKV
from fastwam.models.wan22.kv_tap import KeyValueBank, KVSource
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config_contracts import build_fastwam_checkpoint_contract
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
)
from rlinf.models.embodiment.wam_policy.critic import (
    FastWAMValueFeatures,
    FastWAMValueTransformerConfig,
)
from rlinf.models.embodiment.wam_policy.pad_rv.config import (
    PAD_ROUTE_NEUTRAL_BUILDER_TARGET,
    validate_pad_route_neutral_training_config,
)
from rlinf.models.embodiment.wam_policy.pad_rv.optimizer import (
    assert_pad_frozen_update_resolution,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_actor import (
    PadRouteNeutralFSDPActor,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_budget import (
    PAD_WARMUP_DAMPED_CONTROLLER_TYPE,
    PadCriticWarmupReversalDampedController,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_contracts import (
    PadCriticWarmupConfig,
    RouteNeutralGateInputContract,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PadRouteNeutralCurrentStepGate,
    PadRouteNeutralGateConfig,
    PhysicalStateHistoryTracker,
    RouteNeutralGateFeatures,
    RouteNeutralVisualFeatures,
    deserialize_route_neutral_features,
    serialize_route_neutral_features,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_policy import (
    PadRouteNeutralPolicy,
)
from rlinf.runners.fastwam_idm_cost_control import FastWAMIDMCostObservation


def _set_config_environment(monkeypatch) -> Path:
    config_dir = Path(__file__).parents[2] / "examples" / "embodiment" / "config"
    environment = {
        "EMBODIED_PATH": str(config_dir.parent),
        "FASTWAM_CHECKPOINT": "/parent.pt",
        "FASTWAM_CHECKPOINT_SHA256": "a" * 64,
        "FASTWAM_DATASET_STATS": "/stats.json",
        "FASTWAM_MERGED_WARM_U": "/merged.pt",
        "FASTWAM_MERGED_WARM_U_SHA256": "b" * 64,
        "FASTWAM_MERGED_WARM_U_SOURCE_LORA_SHA256": "c" * 64,
        "FASTWAM_TEXT_CACHE": "/text-cache",
        "PI05_CRITIC_CHECKPOINT": "/critic",
        "PI05_CRITIC_CHECKPOINT_SHA256": "d" * 64,
        "FASTWAM_UNCOND_BC_SIDECAR": "/unused.pt",
        "FASTWAM_UNCOND_BC_SIDECAR_SHA256": "e" * 64,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return config_dir


def _compose_route_neutral(monkeypatch, overrides: list[str] | None = None):
    config_dir = _set_config_environment(monkeypatch)
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        return compose(
            config_name="libero_10_ppo_fastwam_pad_route_neutral_formal",
            overrides=overrides or [],
        )


def _bank(source: KVSource, values: torch.Tensor) -> KeyValueBank:
    return KeyValueBank(
        source=source,
        key=values.clone(),
        value=values.clone() + 0.25,
        valid_mask=torch.ones(values.shape[:2], dtype=torch.bool),
    )


def _source_features(context_offset: float = 0.0) -> FastWAMValueFeatures:
    return FastWAMValueFeatures(
        (
            ConditionLayerKV(
                layer_index=0,
                current_frame_video=_bank(
                    KVSource.CURRENT_FRAME_VIDEO,
                    torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
                ),
                context=_bank(
                    KVSource.TEXT_STATE_CONTEXT,
                    torch.tensor([[[0.5, 0.6], [0.7, 0.8]]]) + context_offset,
                ),
            ),
        )
    )


def _neutral_features(context_offset: float = 0.0) -> RouteNeutralGateFeatures:
    return RouteNeutralGateFeatures(
        visual=RouteNeutralVisualFeatures.from_value_features(
            _source_features(context_offset)
        ),
        language=torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
        language_mask=torch.tensor([[True, True]]),
        state=torch.tensor([[0.2, -0.1]]),
        physical_history=torch.tensor([[[0.0, 0.0], [0.1, -0.1]]]),
    )


def _visual_config() -> FastWAMValueTransformerConfig:
    return FastWAMValueTransformerConfig(
        num_mot_layers=1,
        source_num_heads=1,
        source_head_dim=2,
        layer_indices=(0,),
        sources=("current_frame_video",),
        hidden_dim=4,
        num_query_tokens=2,
        ffn_multiplier=2,
        share_blocks=False,
        layer_index_embedding=False,
        pooling="mean_token",
    )


def test_route_neutral_config_composes_and_allows_direct_experiment_overrides(
    monkeypatch,
) -> None:
    overrides = [
        "actor.model.gate.input_contract.physical_history.length_chunks=6",
        "rollout.model.gate.input_contract.physical_history.length_chunks=6",
        "actor.model.critic_warmup.runner_updates=3",
        "rollout.model.critic_warmup.runner_updates=3",
        "algorithm.fixed_branch_cost.controller.critic_warmup.runner_updates=3",
        "algorithm.fixed_branch_cost.controller.signed_price.reversal.factor=0.25",
    ]
    cfg = _compose_route_neutral(monkeypatch, overrides)
    validate_pad_route_neutral_training_config(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    checkpoint_contract = build_fastwam_checkpoint_contract(cfg, world_size=1)

    assert cfg.actor.model.builder_target == PAD_ROUTE_NEUTRAL_BUILDER_TARGET
    assert isinstance(resolved, dict)
    assert checkpoint_contract["model"]["critic_warmup"]["runner_updates"] == 3
    assert checkpoint_contract["algorithm"]["prediction_budget"] is None
    assert cfg.algorithm.prediction_budget is None
    assert cfg.actor.model.gate.input_contract.physical_history.length_chunks == 6
    assert cfg.actor.model.critic_warmup.runner_updates == 3
    assert cfg.algorithm.fixed_branch_cost.controller.type == (
        PAD_WARMUP_DAMPED_CONTROLLER_TYPE
    )
    assert cfg.algorithm.fixed_branch_cost.controller.signed_price.reversal.factor == (
        0.25
    )


def test_route_neutral_config_rejects_mode_input(monkeypatch) -> None:
    cfg = _compose_route_neutral(monkeypatch)
    cfg.actor.model.gate.input_contract.forbidden.current_mode = True
    with pytest.raises(ValueError, match="current_mode"):
        validate_pad_route_neutral_training_config(cfg)


def test_route_neutral_config_rejects_actor_rollout_gate_drift(monkeypatch) -> None:
    cfg = _compose_route_neutral(monkeypatch)
    cfg.actor.model.gate.hidden_dim = 512
    with pytest.raises(ValueError, match="Gate configs differ"):
        validate_pad_route_neutral_training_config(cfg)


def test_route_neutral_config_rejects_legacy_critic_warmup(monkeypatch) -> None:
    cfg = _compose_route_neutral(monkeypatch)
    cfg.algorithm.critic_loss.warmup_steps = 3
    with pytest.raises(ValueError, match="critic_loss.warmup_steps"):
        validate_pad_route_neutral_training_config(cfg)


def test_route_neutral_codec_discards_context_and_gate_is_context_invariant() -> None:
    torch.manual_seed(7)
    config = PadRouteNeutralGateConfig(
        visual=_visual_config(),
        language_dim=3,
        state_dim=2,
        history_length_chunks=2,
    )
    gate = PadRouteNeutralCurrentStepGate(config).eval()
    left = _neutral_features(context_offset=0.0)
    right = _neutral_features(context_offset=1000.0)

    assert not hasattr(left.visual.layers[0], "context")
    assert torch.equal(
        left.visual.layers[0].current_frame_video.key,
        right.visual.layers[0].current_frame_video.key,
    )
    assert torch.equal(gate(left), gate(right))

    replay = serialize_route_neutral_features(left)
    forbidden_fragments = ("action", "mode", "context", "budget", "parity")
    assert not any(
        fragment in key for key in replay for fragment in forbidden_fragments
    )
    restored = deserialize_route_neutral_features(replay, layer_indices=(0,))
    assert torch.equal(gate(left), gate(restored))


def test_physical_history_is_fixed_width_resettable_and_route_free() -> None:
    contract = RouteNeutralGateInputContract(
        history_length_chunks=3,
        state_dim=2,
    )
    tracker = PhysicalStateHistoryTracker(contract)
    env_ids = torch.tensor([4])

    first = tracker.features_and_append(
        env_ids=env_ids,
        reset_mask=torch.tensor([True]),
        current_state=torch.tensor([[1.0, 2.0]]),
    )
    second = tracker.features_and_append(
        env_ids=env_ids,
        reset_mask=torch.tensor([False]),
        current_state=torch.tensor([[3.0, 4.0]]),
    )
    assert torch.equal(first, torch.tensor([[[1.0, 2.0]] * 3]))
    assert torch.equal(second, torch.tensor([[[1.0, 2.0]] * 3]))

    state = tracker.state_dict()
    assert not any(
        fragment in key
        for key in state
        for fragment in ("route", "mode", "budget", "chunk_id", "parity")
    )
    restored = PhysicalStateHistoryTracker(contract)
    restored.load_state_dict(state)
    reset = restored.features_and_append(
        env_ids=env_ids,
        reset_mask=torch.tensor([True]),
        current_state=torch.tensor([[9.0, 8.0]]),
    )
    assert torch.equal(reset, torch.tensor([[[9.0, 8.0]] * 3]))


def test_warmup_routing_is_seeded_half_random_and_logit_independent() -> None:
    policy = object.__new__(PadRouteNeutralPolicy)
    policy.actor_version = 0
    policy.critic_warmup = PadCriticWarmupConfig(
        runner_updates=5,
        idm_probability=0.5,
    )
    policy.config = SimpleNamespace(
        gate_temperature=1.0,
        gate_epsilon=0.1,
        decision_telemetry_enabled=True,
    )
    seeds = torch.tensor([11, 12])
    low_high = torch.tensor([-100.0, 100.0])
    high_low = -low_high

    left = policy._training_gate_decision(logits=low_high, sampling_seeds=seeds)
    right = policy._training_gate_decision(logits=high_low, sampling_seeds=seeds)
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[2], torch.full((2,), 0.5))
    assert torch.equal(right[2], torch.full((2,), 0.5))
    assert torch.allclose(left[3], torch.full((2,), math.log(0.5)))
    assert torch.equal(left[4], torch.ones(2, dtype=torch.bool))
    assert torch.equal(policy._training_gate_epsilon(low_high), torch.ones(2))


def _controller_config(*, warmup_updates: int = 2) -> dict:
    return {
        "type": PAD_WARMUP_DAMPED_CONTROLLER_TYPE,
        "constraint": "two_sided_band",
        "rate": {
            "scope": "eligible_gate_decisions",
            "feedback": "eligible_expected",
            "target_idm_fraction": 0.5,
            "half_width": 0.03,
        },
        "charge_scope": "eligible_nonforced",
        "signed_price": {
            "initial_value": 0.0,
            "learning_rate": 0.1,
            "ema_beta": 0.0,
            "update_interval": 1,
            "max_abs_value": 1.0,
            "max_delta_per_update": 1.0,
            "reversal": {"mode": "opposing_decay", "factor": 0.5},
        },
        "critic_warmup": {
            "runner_updates": warmup_updates,
            "route_behavior": "independent_random",
            "idm_probability": 0.5,
            "freeze_gate": True,
            "freeze_cost_controller": True,
        },
    }


def _observation(step: int, expected_fraction: float) -> FastWAMIDMCostObservation:
    eligible_idm = int(round(10 * expected_fraction))
    return FastWAMIDMCostObservation(
        runner_step=step,
        eligible_gate_decision_count=10,
        eligible_idm_decision_count=eligible_idm,
        eligible_realized_fraction=eligible_idm / 10,
        eligible_expected_fraction=expected_fraction,
        valid_chunk_count=10,
        valid_idm_chunk_count=eligible_idm,
        executed_realized_fraction=eligible_idm / 10,
        forced_fraction=0.0,
        break_even_idm_cost=None,
        configured_idm_cost=None,
    )


def test_cost_controller_freezes_history_then_reuses_opposing_decay() -> None:
    controller = PadCriticWarmupReversalDampedController(_controller_config())
    for step in (0, 1):
        decision = controller.decision_for_step(step)
        assert decision.phase == "critic_warmup"
        assert decision.idm_cost == decision.uncond_cost == 0.0
        controller.observe_rollout(_observation(step, 0.9))
        assert controller.signed_price == 0.0
        assert controller.rate_ema is None

    controller.decision_for_step(2)
    high_record = controller.observe_rollout(_observation(2, 0.9))
    assert high_record["update"]["applied_delta"] > 0.0
    positive_price = controller.signed_price

    controller.decision_for_step(3)
    low_record = controller.observe_rollout(_observation(3, 0.1))
    assert low_record["update"]["opposing_decay_applied"] is True
    assert low_record["update"]["post_decay_signed_price"] == pytest.approx(
        positive_price * 0.5
    )


def test_critic_only_warmup_has_no_gate_gradient() -> None:
    actor = object.__new__(PadRouteNeutralFSDPActor)
    actor.critic_warmup = PadCriticWarmupConfig(
        runner_updates=5,
        idm_probability=0.5,
    )
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "gate_ppo": {
                    "clip_ratio_low": 0.2,
                    "clip_ratio_high": 0.2,
                    "entropy_coefficient": 0.0,
                    "entropy_metric_source": "base",
                },
                "critic_loss": {
                    "value_clip": 0.2,
                    "huber_delta": 10.0,
                    "loss_weight": 1.0,
                },
            },
            "env": {"train": {"max_episode_steps": 10}},
        }
    )
    gate_weight = torch.nn.Parameter(torch.tensor(0.25))
    value_weight = torch.nn.Parameter(torch.tensor(0.5))
    logits = gate_weight.expand(2)
    behavior = torch.sigmoid(logits)
    values = value_weight.expand(2, 1)
    route = ChunkRouteRecord(
        route_used=torch.tensor([0, 1]),
        route_was_forced=torch.tensor([False, False]),
        chunk_ids=torch.tensor([0, 0]),
        episode_ids=torch.tensor([0, 1]),
        route_source_chunk_ids=torch.tensor([0, 0]),
        actor_versions=torch.tensor([0, 0]),
    )
    half = torch.full((2,), 0.5)
    replayed_behavior = gate_weight * 0.0 + half
    gate_logprobs = torch.where(
        route.route_used.bool(),
        torch.log(replayed_behavior),
        torch.log1p(-replayed_behavior),
    )
    emitted = GateDecisionRecord(
        next_route=route.route_used,
        base_probability=behavior.detach(),
        behavior_probability=half,
        old_logprob=torch.full((2,), math.log(0.5)),
        epsilon=torch.ones(2),
        temperature=torch.ones(2),
        valid=torch.ones(2, dtype=torch.bool),
        source_chunk_ids=route.chunk_ids,
        episode_ids=route.episode_ids,
        actor_versions=route.actor_versions,
    )
    loss, metrics = actor._compute_fastwam_loss(
        micro_batch={
            "route_info": route,
            "emitted_gate": emitted,
            "gate_advantages": torch.tensor([1.0, -1.0]),
            "gate_valid_mask": torch.tensor([True, True]),
            "returns": torch.tensor([[1.0], [0.0]]),
            "prev_values": torch.zeros(2, 1),
        },
        output_dict={
            "values": values,
            "gate_logprobs": gate_logprobs,
            "gate_base_probabilities": behavior,
            "gate_behavior_probabilities": replayed_behavior,
        },
    )
    loss.backward()

    assert gate_weight.grad is None
    assert value_weight.grad is not None
    assert metrics["fastwam/critic_warmup/active"] == 1.0
    assert metrics["fastwam/regularized_policy_loss"] == 0.0
    assert metrics["gate/sample_count"] == 2.0
    assert metrics["gate/policy_loss"] == 0.0
    assert metrics["gate/ratio"] == 1.0
    assert metrics["gate/log_ratio_max_abs"] == 0.0


def test_update_resolution_can_audit_only_the_active_warmup_owner() -> None:
    gate = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    value = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [
            {"name": "gate", "params": [gate], "lr": 1e-3},
            {"name": "value_head", "params": [value], "lr": 1e-3},
        ],
        weight_decay=0.0,
    )
    value.grad = torch.tensor([1.0])
    report = assert_pad_frozen_update_resolution(
        optimizer,
        minimum_half_ulp_ratio=1.0,
        group_names=("value_head",),
    )
    assert set(report) == {"value_head"}
    with pytest.raises(RuntimeError, match="gate"):
        assert_pad_frozen_update_resolution(
            optimizer,
            minimum_half_ulp_ratio=1.0,
        )


def test_warmup_optimizer_step_skips_gate_weight_decay_and_state() -> None:
    gate = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    value = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [
            {"name": "gate", "params": [gate], "lr": 1e-3},
            {"name": "value_head", "params": [value], "lr": 1e-3},
        ],
        weight_decay=0.1,
    )
    gate.grad = torch.zeros_like(gate)
    value.grad = torch.ones_like(value)

    class _Scaler:
        @staticmethod
        def unscale_(_optimizer) -> None:
            return None

        @staticmethod
        def step(*, optimizer) -> None:
            optimizer.step()

        @staticmethod
        def update() -> None:
            return None

    actor = object.__new__(PadRouteNeutralFSDPActor)
    actor.optimizer = optimizer
    actor.grad_scaler = _Scaler()
    actor._strategy = SimpleNamespace(
        clip_grad_norm_=lambda **_kwargs: torch.tensor(1.0)
    )
    actor.model = SimpleNamespace()
    actor._logger = SimpleNamespace(
        info=lambda *_args: None, warning=lambda *_args: None
    )
    actor._cfg = SimpleNamespace(
        optim=SimpleNamespace(update_resolution_min_half_ulp_ratio=1.0)
    )
    actor._pad_critic_warmup_active = True
    actor._pad_resolution_groups_checked = set()
    actor._fastwam_update_resolution_checked = False
    actor.optimizer_steps = 0
    actor.gate_optimizer_steps = 0
    actor.critic_optimizer_steps = 0
    initial_gate = gate.detach().clone()

    actor.optimizer_step()

    assert torch.equal(gate, initial_gate)
    assert gate not in optimizer.state
    assert not torch.equal(value, torch.ones_like(value))
    assert actor.optimizer_steps == 1
    assert actor.gate_optimizer_steps == 0
    assert actor.critic_optimizer_steps == 1
