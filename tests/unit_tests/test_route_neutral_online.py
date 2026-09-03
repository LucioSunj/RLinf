# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir

from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    WAMRoute,
)
from rlinf.models.embodiment.wam_policy.kv_replay import GateKVReplayBackend
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_runner import (
    PadRouteNeutralRunner,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.actor import (
    RouteNeutralOnlineIDMBCFSDPActor,
    align_current_step_trainable_advantages,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.config import (
    validate_route_neutral_online_idm_bc_training_config,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.lifecycle import (
    RouteNeutralOnlineRunner,
)


def _compose(monkeypatch):
    config_dir = Path(__file__).parents[2] / "examples" / "embodiment" / "config"
    environment = {
        "EMBODIED_PATH": str(config_dir.parent),
        "FASTWAM_CHECKPOINT": "/parent.pt",
        "FASTWAM_CHECKPOINT_SHA256": "a" * 64,
        "FASTWAM_DATASET_STATS": "/stats.json",
        "FASTWAM_UNCOND_BC_SIDECAR": "/bc.pt",
        "FASTWAM_UNCOND_BC_SIDECAR_SHA256": "b" * 64,
        "FASTWAM_TEXT_CACHE": "/text-cache",
        "PI05_CRITIC_CHECKPOINT": "/critic",
        "PI05_CRITIC_CHECKPOINT_SHA256": "c" * 64,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        return compose(config_name="libero_10_ppo_fastwam_route_neutral_online_formal")


def test_config_selects_bc_initialized_trainable_uncond(monkeypatch) -> None:
    cfg = _compose(monkeypatch)
    validate_route_neutral_online_idm_bc_training_config(cfg)

    assert cfg.actor.model.uncond_lora.rank == 16
    assert cfg.algorithm.uncond_flow_ppo.loss_weight == 1.0
    assert cfg.algorithm.uncond_idm_bc.loss_weight == 0.2
    assert cfg.actor.model.kv_replay.backend == "recompute"
    assert cfg.actor.model.gate.current_mode_embedding is False
    assert cfg.actor.model.gate.denoise_timestep_embedding is False
    assert cfg.route_neutral_online_implementation.rollout_init_mode == "serial_rank"
    assert cfg.route_neutral_online_implementation.trajectory_send_mode == (
        "serialized"
    )
    assert (
        cfg.route_neutral_online_implementation.consume_rollout_batch_during_train_preparation
        is True
    )
    assert (
        cfg.route_neutral_online_implementation.release_host_memory_after_train_preparation
        is True
    )
    assert issubclass(RouteNeutralOnlineRunner, PadRouteNeutralRunner)


def test_recompute_backend_enum_preserves_runtime_boundary() -> None:
    assert (
        GateKVReplayBackend(GateKVReplayBackend.RECOMPUTE)
        is GateKVReplayBackend.RECOMPUTE
    )
    assert GateKVReplayBackend("recompute") is GateKVReplayBackend.RECOMPUTE


def test_route_neutral_train_preparation_consumes_recompute_replay() -> None:
    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    actor.cfg = SimpleNamespace(
        route_neutral_online_implementation=SimpleNamespace(
            consume_rollout_batch_during_train_preparation=True
        )
    )

    assert actor._consume_rollout_batch_during_train_preparation() is True


def test_current_step_alignment_preserves_flow_and_gate_credit() -> None:
    route = ChunkRouteRecord(
        route_used=torch.tensor(
            [[WAMRoute.IDM, WAMRoute.UNCOND], [WAMRoute.UNCOND, WAMRoute.IDM]]
        ),
        route_was_forced=torch.zeros(2, 2, dtype=torch.bool),
        chunk_ids=torch.tensor([[0, 0], [1, 1]]),
        episode_ids=torch.zeros(2, 2, dtype=torch.long),
        route_source_chunk_ids=torch.tensor([[0, 0], [1, 1]]),
        actor_versions=torch.zeros(2, 2, dtype=torch.long),
    )
    half = torch.full((2, 2), 0.5)
    emitted = GateDecisionRecord(
        next_route=route.route_used,
        base_probability=half,
        behavior_probability=half,
        old_logprob=torch.full((2, 2), -0.6931471805599453),
        epsilon=torch.ones(2, 2),
        temperature=torch.ones(2, 2),
        valid=torch.ones(2, 2, dtype=torch.bool),
        source_chunk_ids=route.chunk_ids,
        episode_ids=route.episode_ids,
        actor_versions=route.actor_versions,
    )
    advantages = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    alignment = align_current_step_trainable_advantages(
        advantages=advantages,
        route=route,
        emitted=emitted,
        loss_mask=torch.ones(2, 2, 1, dtype=torch.bool),
    )

    assert torch.equal(alignment.flow_advantages, advantages)
    assert bool(alignment.flow_valid_mask.all())
    assert torch.equal(alignment.gate_advantages, advantages[..., 0])
    assert bool(alignment.gate_valid_mask.all())


def test_warmup_optimizer_does_not_create_gate_or_lora_adam_state() -> None:
    gate = torch.nn.Parameter(torch.tensor([1.0]))
    lora = torch.nn.Parameter(torch.tensor([1.0]))
    value = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW(
        [
            {"name": "gate", "params": [gate], "lr": 1e-3},
            {"name": "uncond_lora", "params": [lora], "lr": 1e-3},
            {"name": "value_head", "params": [value], "lr": 1e-3},
        ],
        weight_decay=0.1,
    )
    gate.grad = torch.zeros_like(gate)
    lora.grad = torch.zeros_like(lora)
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

    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    actor.optimizer = optimizer
    actor.optimizer_steps = 0
    actor.grad_scaler = _Scaler()
    actor._strategy = SimpleNamespace(
        clip_grad_norm_=lambda **_kwargs: torch.tensor(1.0)
    )
    actor.model = SimpleNamespace()
    actor._logger = SimpleNamespace(
        warning=lambda *_args: None, info=lambda *_args: None
    )
    actor._cfg = SimpleNamespace(
        optim=SimpleNamespace(update_resolution_min_half_ulp_ratio=1.0)
    )
    actor._route_neutral_warmup_active = True
    actor._fastwam_update_resolution_checked = False
    actor._online_idm_bc_gradient_audit_complete = True
    actor._online_idm_bc_audit_micro_batch = None
    before_gate = gate.detach().clone()
    before_lora = lora.detach().clone()

    actor.optimizer_step()

    assert torch.equal(gate, before_gate)
    assert torch.equal(lora, before_lora)
    assert gate not in optimizer.state
    assert lora not in optimizer.state
    assert value in optimizer.state
    assert actor.optimizer_steps == 1
