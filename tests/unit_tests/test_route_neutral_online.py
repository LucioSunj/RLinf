# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.adaptive_action import CachedActionCondition
from fastwam.models.wan22.adaptive_sampler import VelocityOutput
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from hydra import compose, initialize_config_dir

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.envs.libero.action_protocol import LiberoActionProtocol
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    WAMRoute,
)
from rlinf.models.embodiment.wam_policy.kv_replay import GateKVReplayBackend
from rlinf.models.embodiment.wam_policy.online_idm_bc.actor import (
    OnlineIDMBCFSDPActor,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import (
    ONLINE_IDM_BC_FLOW_VALID,
    ONLINE_IDM_BC_SAMPLE_IDENTITIES,
    ONLINE_IDM_BC_TEACHER_ACTIONS,
    ONLINE_IDM_BC_TEACHER_BYTES,
    ONLINE_IDM_BC_TEACHER_PRESENT,
    ONLINE_IDM_BC_TEACHER_SECONDS,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.runtime import (
    OnlineIDMTeacherLiberoRuntime,
)
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
from rlinf.models.embodiment.wam_policy.route_neutral_online.policy import (
    RouteNeutralOnlineIDMBCFastWAMPolicy,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.runtime import (
    ROUTE_NEUTRAL_ROLLOUT_IDM_BATCH_SIZE,
    ROUTE_NEUTRAL_ROLLOUT_UNCOND_BATCH_SIZE,
    ROUTE_NEUTRAL_TEACHER_BATCH_SIZE,
    RouteNeutralOnlineIDMTeacherLiberoRuntime,
    RouteNeutralPreparedStep,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _compose(
    monkeypatch,
    config_name="libero_10_ppo_fastwam_route_neutral_online_formal",
    overrides=None,
):
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
        return compose(config_name=config_name, overrides=overrides or [])


def _compose_eval(monkeypatch):
    config_dir = Path(__file__).parents[2]
    environment = {
        "EMBODIED_PATH": str(config_dir / "examples" / "embodiment"),
        "FASTWAM_CHECKPOINT": "/parent.pt",
        "FASTWAM_CHECKPOINT_SHA256": "a" * 64,
        "FASTWAM_DATASET_STATS": "/stats.json",
        "FASTWAM_TEXT_CACHE": "/text-cache",
        "PI05_CRITIC_CHECKPOINT": "/critic",
        "PI05_CRITIC_CHECKPOINT_SHA256": "c" * 64,
        "FASTWAM_EVAL_OUTPUT_DIR": "/output",
        "FASTWAM_EVAL_RUN_ID": "route-neutral-eval-test",
        "FASTWAM_EVAL_LEDGER": "/ledger.json",
        "FASTWAM_PROJECT_CHECKPOINT": "/project-checkpoint.pt",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with initialize_config_dir(
        version_base="1.1", config_dir=str(config_dir / "evaluations" / "libero")
    ):
        return compose(config_name="libero_plus_long_fastwam_route_neutral_online_eval")


def _cached_condition(
    batch_size: int, *, context_dim: int = 3
) -> CachedActionCondition:
    return CachedActionCondition(
        context=torch.arange(batch_size * 2 * context_dim, dtype=torch.float32).reshape(
            batch_size,
            2,
            context_dim,
        ),
        context_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        video_kv_cache=[
            {
                "k": torch.zeros(batch_size, 1, 2),
                "v": torch.ones(batch_size, 1, 2),
            }
        ],
        attention_mask=torch.ones(3, 3, dtype=torch.bool),
        video_seq_len=1,
        current_frame_video_tokens=1,
    )


def _route_record(routes: torch.Tensor) -> ChunkRouteRecord:
    batch_size = int(routes.numel())
    chunk_ids = torch.arange(batch_size)
    return ChunkRouteRecord(
        route_used=routes,
        route_was_forced=torch.zeros(batch_size, dtype=torch.bool),
        chunk_ids=chunk_ids,
        episode_ids=torch.zeros(batch_size, dtype=torch.long),
        route_source_chunk_ids=chunk_ids,
        actor_versions=torch.zeros(batch_size, dtype=torch.long),
    )


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


def test_config_accepts_five_rollout_rank_training_placement(monkeypatch) -> None:
    cfg = _compose(monkeypatch)
    cfg.cluster.component_placement.env = "1-5"
    cfg.cluster.component_placement.rollout = "1-5"

    validate_route_neutral_online_idm_bc_training_config(cfg)

    assert cfg.cluster.component_placement.actor == "0-0"
    assert cfg.cluster.component_placement.env == "1-5"
    assert cfg.cluster.component_placement.rollout == "1-5"


def test_config_accepts_seven_rollout_rank_training_placement(monkeypatch) -> None:
    cfg = _compose(monkeypatch)
    cfg.cluster.component_placement.env = "1-7"
    cfg.cluster.component_placement.rollout = "1-7"
    cfg.env.train.total_num_envs = 28
    cfg.actor.global_batch_size = 196

    validate_route_neutral_online_idm_bc_training_config(cfg)

    assert cfg.cluster.component_placement.actor == "0-0"
    assert cfg.cluster.component_placement.env == "1-7"
    assert cfg.cluster.component_placement.rollout == "1-7"


def test_config_rejects_rollout_size_not_divisible_by_global_batch(
    monkeypatch,
) -> None:
    cfg = _compose(monkeypatch)
    cfg.cluster.component_placement.env = "1-7"
    cfg.cluster.component_placement.rollout = "1-7"
    cfg.env.train.total_num_envs = 28

    with pytest.raises(
        ValueError,
        match="rollout size 1960 must be divisible by actor global batch size 210",
    ):
        validate_route_neutral_online_idm_bc_training_config(cfg)


@pytest.mark.parametrize(
    ("overlay", "total_envs", "optimizer_minibatches"),
    [
        (
            "task6_28",
            28,
            10,
        ),
        (
            "task6_42_engineering",
            42,
            15,
        ),
    ],
)
def test_perfopt_configs_select_mb4_geometry(
    monkeypatch,
    overlay,
    total_envs,
    optimizer_minibatches,
) -> None:
    cfg = _compose(
        monkeypatch,
        overrides=[f"+route_neutral_online_perfopt={overlay}"],
    )

    validate_route_neutral_online_idm_bc_training_config(cfg)

    rollout_size = cfg.env.train.total_num_envs * (
        cfg.env.train.max_steps_per_rollout_epoch
        // cfg.actor.model.runtime.execution_horizon
    )
    assert cfg.env.train.task_id_filter == [6]
    assert cfg.env.train.total_num_envs == total_envs
    assert cfg.actor.global_batch_size == 196
    assert cfg.actor.micro_batch_size == 4
    assert rollout_size // cfg.actor.global_batch_size == optimizer_minibatches
    assert cfg.algorithm.regularization.base_uncond_kl.enabled is False
    assert cfg.algorithm.regularization.base_uncond_kl.coefficient == 0.0
    assert cfg.algorithm.regularization.base_uncond_kl.log_metric is False
    assert (
        cfg.algorithm.fixed_branch_cost.controller.signed_price.reversal.factor == 0.0
    )
    assert cfg.runner.max_steps == (6 if total_envs == 42 else 50)


def test_perfopt_config_rejects_hot_path_base_kl_logging(monkeypatch) -> None:
    cfg = _compose(
        monkeypatch,
        overrides=["+route_neutral_online_perfopt=task6_28"],
    )
    cfg.algorithm.regularization.base_uncond_kl.log_metric = True

    with pytest.raises(ValueError, match="disabled base UNCOND KL"):
        validate_route_neutral_online_idm_bc_training_config(cfg)


def test_perfopt_config_rejects_non_divisible_35_env_geometry(monkeypatch) -> None:
    cfg = _compose(
        monkeypatch,
        overrides=["+route_neutral_online_perfopt=task6_28"],
    )
    cfg.env.train.total_num_envs = 35

    with pytest.raises(ValueError, match="rollout size 2450"):
        validate_route_neutral_online_idm_bc_training_config(cfg)


def test_resume_preserves_completed_first_joint_update_audit(monkeypatch) -> None:
    def _load_checkpoint(actor, _load_path):
        actor.optimizer_steps = 300
        return 30

    monkeypatch.setattr(OnlineIDMBCFSDPActor, "load_checkpoint", _load_checkpoint)
    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    actor.optimizer_steps = 0
    actor.critic_warmup = SimpleNamespace(runner_updates=5)
    actor._fastwam_update_resolution_checked = False
    messages = []
    actor._logger = SimpleNamespace(info=messages.append)

    assert actor.load_checkpoint("/checkpoint/global_step_30") == 30
    assert actor._fastwam_update_resolution_checked is True
    assert messages == [
        "[FSDP] Preserving completed first joint route-neutral update "
        "resolution audit from resumed step 30."
    ]


def test_warmup_boundary_resume_keeps_first_joint_update_audit_due(monkeypatch) -> None:
    def _load_checkpoint(actor, _load_path):
        actor.optimizer_steps = 50
        return 5

    monkeypatch.setattr(OnlineIDMBCFSDPActor, "load_checkpoint", _load_checkpoint)
    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    actor.optimizer_steps = 0
    actor.critic_warmup = SimpleNamespace(runner_updates=5)
    actor._fastwam_update_resolution_checked = False
    actor._logger = SimpleNamespace(info=lambda _message: None)

    assert actor.load_checkpoint("/checkpoint/global_step_5") == 5
    assert actor._fastwam_update_resolution_checked is False


def test_eval_config_selects_current_step_gate_without_critic(monkeypatch) -> None:
    cfg = _compose_eval(monkeypatch)
    online = validate_route_neutral_online_idm_bc_training_config(cfg, only_eval=True)

    assert online.enabled is True
    assert cfg.rollout.model.eval_routing_mode == "learned_threshold"
    assert cfg.rollout.model.formal_training_sampling_seed == 42
    assert cfg.rollout.model.eval_without_critic is True
    assert cfg.rollout.model.fastwam.load_text_encoder is False
    assert cfg.rollout.model.route_neutral_online.visual.layer_indices == [
        14,
        15,
        16,
        17,
        18,
        19,
    ]


def test_eval_prediction_disables_value_computation(monkeypatch) -> None:
    policy = object.__new__(RouteNeutralOnlineIDMBCFastWAMPolicy)
    policy.config = SimpleNamespace(
        training_rollout_microbatch_size=None,
        decision_telemetry_enabled=False,
    )
    observed = {}
    monkeypatch.setattr(
        RouteNeutralOnlineIDMBCFastWAMPolicy,
        "_routing_metadata",
        lambda _self, _env_obs, batch_size, device: (
            torch.arange(batch_size, device=device),
            torch.ones(batch_size, device=device, dtype=torch.bool),
        ),
    )

    def _predict_current_step(_self, **kwargs):
        observed.update(kwargs)
        return torch.zeros(2, 1), {}

    monkeypatch.setattr(
        RouteNeutralOnlineIDMBCFastWAMPolicy,
        "_predict_current_step",
        _predict_current_step,
    )

    policy.predict_action_batch(
        {"states": torch.zeros(2, 1)},
        mode="eval",
        compute_values=True,
    )

    assert observed["compute_values"] is False


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


def test_mb4_compaction_keeps_union_and_minimal_padding() -> None:
    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    actor.cfg = SimpleNamespace(actor=SimpleNamespace(micro_batch_size=4))
    routes = torch.tensor(
        [
            WAMRoute.IDM,
            WAMRoute.IDM,
            WAMRoute.UNCOND,
            WAMRoute.IDM,
            WAMRoute.IDM,
            WAMRoute.IDM,
            WAMRoute.IDM,
            WAMRoute.IDM,
        ]
    )
    gate_valid = torch.zeros(8, dtype=torch.bool)
    gate_valid[0] = True
    flow_valid = torch.zeros(8, dtype=torch.bool)
    flow_valid[2] = True
    loss_mask = torch.zeros(8, 2, dtype=torch.bool)
    loss_mask[5, 0] = True
    batch = {
        "prev_logprobs": torch.arange(8, dtype=torch.float32).reshape(8, 1),
        "route_info": _route_record(routes),
        "gate_valid_mask": gate_valid,
        "flow_valid_mask": flow_valid,
        "loss_mask": loss_mask,
        "forward_inputs": {
            "payload": torch.arange(8, dtype=torch.float32).reshape(8, 1)
        },
    }

    compacted, metrics = actor._prepare_train_global_batch_for_microbatches(batch)

    assert compacted["prev_logprobs"].reshape(-1).tolist() == [0.0, 1.0, 2.0, 5.0]
    assert compacted["forward_inputs"]["payload"].reshape(-1).tolist() == [
        0.0,
        1.0,
        2.0,
        5.0,
    ]
    assert compacted["route_info"].route_used.tolist() == [
        WAMRoute.IDM,
        WAMRoute.IDM,
        WAMRoute.UNCOND,
        WAMRoute.IDM,
    ]
    assert metrics["perf/actor_rows_original"] == 8.0
    assert metrics["perf/actor_rows_active"] == 3.0
    assert metrics["perf/actor_rows_padded"] == 1.0
    assert metrics["perf/actor_rows_forwarded"] == 4.0
    assert metrics["perf/actor_microbatches_executed"] == 1.0


def test_mb4_compaction_keeps_one_dummy_microbatch_when_all_rows_inactive() -> None:
    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    actor.cfg = SimpleNamespace(actor=SimpleNamespace(micro_batch_size=4))
    batch = {
        "prev_logprobs": torch.arange(8, dtype=torch.float32).reshape(8, 1),
        "route_info": _route_record(torch.full((8,), int(WAMRoute.IDM))),
        "gate_valid_mask": torch.zeros(8, dtype=torch.bool),
        "flow_valid_mask": torch.zeros(8, dtype=torch.bool),
        "loss_mask": torch.zeros(8, 1, dtype=torch.bool),
        "loss_mask_sum": torch.zeros(8, 1, dtype=torch.long),
    }

    compacted, metrics = actor._prepare_train_global_batch_for_microbatches(batch)

    assert compacted["prev_logprobs"].reshape(-1).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert torch.equal(compacted["loss_mask_sum"], torch.ones(4, 1, dtype=torch.long))
    values = torch.ones(4, 1, requires_grad=True)
    critic_loss, _ = compute_ppo_critic_loss(
        values=values,
        returns=torch.zeros_like(values),
        prev_values=torch.zeros_like(values),
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=compacted["loss_mask"],
        loss_mask_sum=compacted["loss_mask_sum"],
        max_episode_steps=700,
    )
    critic_loss.backward()
    assert critic_loss.item() == 0.0
    assert torch.equal(values.grad, torch.zeros_like(values))
    assert metrics["perf/actor_rows_active"] == 0.0
    assert metrics["perf/actor_rows_padded"] == 4.0
    assert metrics["perf/actor_rows_forwarded"] == 4.0
    assert metrics["perf/actor_microbatches_executed"] == 1.0


def test_actor_syncs_on_actual_final_compacted_microbatch() -> None:
    source = inspect.getsource(EmbodiedFSDPActor.run_training)

    assert "is_last=(idx + 1) == len(train_micro_batch)" in source


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


def test_seeded_route_runtime_batches_each_branch_and_teacher_once() -> None:
    runtime = object.__new__(RouteNeutralOnlineIDMTeacherLiberoRuntime)
    runtime.seeded_noise_device = "cpu"
    runtime.flow_sde_noise_level = 0.5
    runtime.flow_sde_ignore_last_transition = False
    runtime.gate_denoise_last_n = 1
    runtime.action_protocol = LiberoActionProtocol(
        generation_horizon=3,
        execution_horizon=2,
        prediction_video_frames=3,
        reset_wait_steps=0,
        max_episode_steps=6,
    )
    runtime.actor = torch.nn.Module()
    runtime.actor.register_parameter(
        "anchor",
        torch.nn.Parameter(torch.zeros(()), requires_grad=False),
    )
    runtime.actor.action_expert = SimpleNamespace(action_dim=3)
    runtime.actor.infer_action_scheduler = SimpleNamespace(num_train_timesteps=1000)
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    runtime._action_schedule = MethodType(
        lambda _self: (timesteps, deltas),
        runtime,
    )
    runtime._seeded_idm_latents = MethodType(
        lambda _self, *, images, seeds: torch.zeros(
            images.shape[0],
            1,
            1,
            1,
            1,
        ),
        runtime,
    )
    condition_calls = []

    def _prepare(_self, *, image, context, context_mask, regime, **_kwargs):
        condition_calls.append((regime, int(image.shape[0])))
        return _cached_condition(
            int(image.shape[0]),
            context_dim=int(context.shape[-1]),
        ), None

    runtime._prepare_action_condition = MethodType(_prepare, runtime)
    velocity_calls = []

    def _velocity(
        _self,
        condition,
        *,
        regime,
        capture_gate_kv,
        actor_version,
    ):
        del capture_gate_kv, actor_version
        velocity_calls.append((regime, int(condition.context.shape[0])))
        scale = 0.125 if regime is PolicyRegime.UNCOND else 0.25
        return lambda action, timestep: action * scale

    runtime._velocity = MethodType(_velocity, runtime)
    runtime._denormalize_action_stages = MethodType(
        lambda _self, actions, *, env_obs: (actions, None),
        runtime,
    )
    batch_size = 4
    prepared = RouteNeutralPreparedStep(
        images=torch.zeros(batch_size, 3, 8, 8),
        context=torch.zeros(batch_size, 2, 3),
        context_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        current_condition=_cached_condition(batch_size),
        gate_features=None,
        critic_features=None,
    )
    routes = torch.tensor(
        [WAMRoute.IDM, WAMRoute.UNCOND, WAMRoute.IDM, WAMRoute.UNCOND]
    )
    env_obs = {
        "_fastwam_action_noise_seeds": torch.tensor([101, 102, 103, 104]),
        "_fastwam_idm_noise_seeds": torch.tensor([201, 202, 203, 204]),
    }

    first = runtime._sample_seeded_training_batch(
        env_obs=env_obs,
        routes=routes,
        actor_version=7,
        prepared=prepared,
    )
    second = runtime._sample_seeded_training_batch(
        env_obs=env_obs,
        routes=routes,
        actor_version=7,
        prepared=prepared,
    )

    torch.testing.assert_close(first.actions, second.actions, rtol=0, atol=0)
    torch.testing.assert_close(first.flow_chains, second.flow_chains, rtol=0, atol=0)
    torch.testing.assert_close(
        first.old_flow_logprobs,
        second.old_flow_logprobs,
        rtol=0,
        atol=0,
    )
    assert first.actions.shape == (batch_size, 2, 3)
    assert first.flow_chains.shape == (batch_size, 4, 3, 3)
    assert first.forward_inputs[ONLINE_IDM_BC_TEACHER_PRESENT].tolist() == [
        False,
        True,
        False,
        True,
    ]
    assert (
        first.forward_inputs[ROUTE_NEUTRAL_ROLLOUT_IDM_BATCH_SIZE].tolist()
        == [2.0] * batch_size
    )
    assert (
        first.forward_inputs[ROUTE_NEUTRAL_ROLLOUT_UNCOND_BATCH_SIZE].tolist()
        == [2.0] * batch_size
    )
    assert (
        first.forward_inputs[ROUTE_NEUTRAL_TEACHER_BATCH_SIZE].tolist()
        == [2.0] * batch_size
    )
    assert velocity_calls[:3] == [
        (PolicyRegime.UNCOND, 2),
        (PolicyRegime.IDM, 2),
        (PolicyRegime.IDM, 2),
    ]
    assert condition_calls[:2] == [
        (PolicyRegime.IDM, 2),
        (PolicyRegime.IDM, 2),
    ]
    serial = []
    for index in range(batch_size):
        serial.append(
            runtime._sample_seeded_training_batch(
                env_obs={
                    name: value[index : index + 1] for name, value in env_obs.items()
                },
                routes=routes[index : index + 1],
                actor_version=7,
                prepared=RouteNeutralPreparedStep(
                    images=prepared.images[index : index + 1],
                    context=prepared.context[index : index + 1],
                    context_mask=prepared.context_mask[index : index + 1],
                    current_condition=prepared.current_condition.index_select(
                        torch.tensor([index])
                    ),
                    gate_features=None,
                    critic_features=None,
                ),
            )
        )
    for name in ("actions", "flow_chains", "old_flow_logprobs", "denoise_indices"):
        torch.testing.assert_close(
            getattr(first, name),
            torch.cat([getattr(sample, name) for sample in serial]),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        first.forward_inputs[ONLINE_IDM_BC_TEACHER_ACTIONS],
        torch.cat(
            [sample.forward_inputs[ONLINE_IDM_BC_TEACHER_ACTIONS] for sample in serial]
        ),
        rtol=0,
        atol=0,
    )


def test_batched_online_bc_matches_serial_loss_and_gradient() -> None:
    runtime = object.__new__(RouteNeutralOnlineIDMTeacherLiberoRuntime)
    runtime.action_protocol = LiberoActionProtocol(
        generation_horizon=3,
        execution_horizon=2,
        prediction_video_frames=3,
        reset_wait_steps=0,
        max_episode_steps=6,
    )
    train_scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    runtime.actor = torch.nn.Module()
    runtime.actor.register_parameter(
        "anchor",
        torch.nn.Parameter(torch.zeros(()), requires_grad=False),
    )
    runtime.actor.action_expert = SimpleNamespace(action_dim=7)
    runtime.actor.train_action_scheduler = train_scheduler
    lora_parameter = torch.nn.Parameter(torch.tensor(0.125))
    runtime.lora_adapter = SimpleNamespace(
        lora_parameters=lambda: iter((lora_parameter,))
    )
    prepare_batch_sizes = []

    def _prepare(_self, *, image, context, context_mask, regime, **_kwargs):
        del context_mask, regime
        prepare_batch_sizes.append(int(image.shape[0]))
        return _cached_condition(
            int(image.shape[0]),
            context_dim=int(context.shape[-1]),
        ), None

    runtime._prepare_action_condition = MethodType(_prepare, runtime)

    def _velocity(
        _self,
        condition,
        *,
        regime,
        capture_gate_kv,
        actor_version,
    ):
        del condition, regime, capture_gate_kv, actor_version
        return lambda action, timestep: VelocityOutput(
            velocity=action * lora_parameter + timestep.reshape(-1, 1, 1) / 1000.0
        )

    runtime._velocity = MethodType(_velocity, runtime)
    routes = torch.tensor(
        [WAMRoute.UNCOND, WAMRoute.IDM, WAMRoute.UNCOND, WAMRoute.UNCOND]
    )
    route_info = _route_record(routes)
    teacher_actions = torch.linspace(-1.0, 1.0, 4 * 3 * 7).reshape(4, 3, 7)
    teacher_present = torch.tensor([True, False, True, True])
    forward_inputs = {
        ONLINE_IDM_BC_FLOW_VALID: torch.tensor([True, True, True, False]),
        ONLINE_IDM_BC_TEACHER_ACTIONS: teacher_actions.to(torch.bfloat16),
        ONLINE_IDM_BC_TEACHER_PRESENT: teacher_present,
        ONLINE_IDM_BC_SAMPLE_IDENTITIES: torch.tensor([11, 12, 13, 14]),
        ONLINE_IDM_BC_TEACHER_SECONDS: torch.tensor([0.1, 0.0, 0.2, 0.3]),
        ONLINE_IDM_BC_TEACHER_BYTES: torch.tensor([42, 0, 42, 42]),
        "flow_chains": torch.zeros(4, 2, 3, 7),
        "fastwam_images": torch.zeros(4, 3, 8, 8),
        "fastwam_context": torch.zeros(4, 2, 3),
        "fastwam_context_mask": torch.ones(4, 2, dtype=torch.bool),
    }

    serial = OnlineIDMTeacherLiberoRuntime.compute_online_idm_bc_loss(
        runtime,
        forward_inputs=forward_inputs,
        route_info=route_info,
    )
    serial_gradient = torch.autograd.grad(
        serial.loss_sum,
        lora_parameter,
        retain_graph=True,
    )[0]
    assert prepare_batch_sizes == [1, 1]
    prepare_batch_sizes.clear()

    batched = runtime.compute_online_idm_bc_loss(
        forward_inputs=forward_inputs,
        route_info=route_info,
    )
    batched_gradient = torch.autograd.grad(batched.loss_sum, lora_parameter)[0]

    assert prepare_batch_sizes == [2]
    torch.testing.assert_close(batched.loss_sum, serial.loss_sum)
    torch.testing.assert_close(batched.raw_loss, serial.raw_loss)
    torch.testing.assert_close(
        batched.mse_per_dimension,
        serial.mse_per_dimension,
    )
    torch.testing.assert_close(
        batched.mse_by_timestep_bin,
        serial.mse_by_timestep_bin,
    )
    torch.testing.assert_close(batched_gradient, serial_gradient)
    assert batched.selected_count.item() == 2.0
    assert batched.expected_count.item() == 2.0
    assert batched.present_count.item() == 3.0


def test_route_replay_batches_all_uncond_rows_once() -> None:
    runtime = object.__new__(RouteNeutralOnlineIDMTeacherLiberoRuntime)
    runtime.flow_sde_noise_level = 0.5
    runtime.action_protocol = LiberoActionProtocol(
        generation_horizon=3,
        execution_horizon=2,
        prediction_video_frames=3,
        reset_wait_steps=0,
        max_episode_steps=6,
    )
    runtime.actor = torch.nn.Module()
    runtime.actor.register_parameter(
        "anchor",
        torch.nn.Parameter(torch.zeros(()), requires_grad=False),
    )
    runtime.actor.action_expert = SimpleNamespace(action_dim=3)
    runtime.actor.infer_action_scheduler = SimpleNamespace(num_train_timesteps=1000)
    runtime.critic_feature_config = None
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    runtime._action_schedule = MethodType(
        lambda _self: (timesteps, deltas),
        runtime,
    )
    prepare_batch_sizes = []

    def _prepare(_self, *, image, context, context_mask, regime, **_kwargs):
        del context_mask, regime
        prepare_batch_sizes.append(int(image.shape[0]))
        return _cached_condition(
            int(image.shape[0]),
            context_dim=int(context.shape[-1]),
        ), None

    runtime._prepare_action_condition = MethodType(_prepare, runtime)
    replay_parameter = torch.nn.Parameter(torch.tensor(0.2))
    velocity_batch_sizes = []

    def _velocity(
        _self,
        condition,
        *,
        regime,
        capture_gate_kv,
        actor_version,
    ):
        del regime, capture_gate_kv, actor_version
        velocity_batch_sizes.append(int(condition.context.shape[0]))
        return lambda action, timestep: VelocityOutput(
            action * replay_parameter
            + timestep.reshape(-1, 1, 1).to(action.dtype) / 1000.0
        )

    runtime._velocity = MethodType(_velocity, runtime)
    routes = torch.tensor(
        [WAMRoute.UNCOND, WAMRoute.IDM, WAMRoute.UNCOND, WAMRoute.IDM]
    )
    chains = torch.randn(4, 4, 3, 3, generator=torch.Generator().manual_seed(9))
    forward_inputs = {
        "flow_chains": chains,
        "denoise_indices": torch.tensor([0, -1, 1, -1]),
        "fastwam_images": torch.zeros(4, 3, 8, 8),
        "fastwam_context": torch.zeros(4, 2, 3),
        "fastwam_context_mask": torch.ones(4, 2, dtype=torch.bool),
    }

    replay = runtime.replay_action_batch(
        forward_inputs=forward_inputs,
        route_info=_route_record(routes),
    )

    assert prepare_batch_sizes == [4]
    assert velocity_batch_sizes == [2]
    assert replay["flow_logprobs"].shape == (4, 2, 3)
    assert replay["flow_entropy"].shape == (4, 2, 3)
    assert torch.count_nonzero(replay["flow_logprobs"][[1, 3]]) == 0
    gradient = torch.autograd.grad(
        replay["flow_logprobs"].sum(),
        replay_parameter,
    )[0]
    assert torch.isfinite(gradient)
    assert gradient != 0


def test_mb4_compaction_preserves_original_global_batch_denominator() -> None:
    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    actor.cfg = SimpleNamespace(actor=SimpleNamespace(micro_batch_size=4))
    batch_size = 196
    active_count = 67
    gate_valid = torch.zeros(batch_size, dtype=torch.bool)
    gate_valid[:active_count] = True
    row_loss = torch.linspace(0.25, 2.0, batch_size)
    batch = {
        "prev_logprobs": row_loss.reshape(batch_size, 1),
        "route_info": _route_record(torch.full((batch_size,), int(WAMRoute.IDM))),
        "gate_valid_mask": gate_valid,
        "flow_valid_mask": torch.zeros(batch_size, dtype=torch.bool),
        "loss_mask": torch.zeros(batch_size, 1, dtype=torch.bool),
        "forward_inputs": {"row_loss": row_loss.reshape(batch_size, 1)},
    }

    compacted, metrics = actor._prepare_train_global_batch_for_microbatches(batch)
    compacted_loss = compacted["forward_inputs"]["row_loss"].reshape(-1)
    compacted_mask = compacted["gate_valid_mask"].float().reshape(-1)
    baseline = (row_loss * gate_valid.float()).sum() / float(batch_size)
    accumulated_mb4 = (compacted_loss * compacted_mask).reshape(-1, 4).mean(
        dim=1
    ).sum() / 49.0

    torch.testing.assert_close(accumulated_mb4, baseline)
    assert metrics["perf/actor_rows_active"] == 67.0
    assert metrics["perf/actor_rows_padded"] == 1.0
    assert metrics["perf/actor_rows_forwarded"] == 68.0
    assert metrics["perf/actor_microbatches_executed"] == 17.0


def test_mb4_compaction_reports_original_rollout_batch_means() -> None:
    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    actor.cfg = SimpleNamespace(
        actor=SimpleNamespace(global_batch_size=8, micro_batch_size=4)
    )
    actor._world_size = 1
    actor.gradient_accumulation = 2
    batch = {
        "prev_logprobs": torch.zeros(8, 1),
        "route_info": _route_record(torch.full((8,), int(WAMRoute.UNCOND))),
        "gate_valid_mask": torch.ones(8, dtype=torch.bool),
        "flow_valid_mask": torch.ones(8, dtype=torch.bool),
        "loss_mask": torch.ones(8, 1, dtype=torch.bool),
        "forward_inputs": {
            ROUTE_NEUTRAL_ROLLOUT_IDM_BATCH_SIZE: torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
            ),
            ROUTE_NEUTRAL_ROLLOUT_UNCOND_BATCH_SIZE: torch.tensor(
                [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]
            ),
            ROUTE_NEUTRAL_TEACHER_BATCH_SIZE: torch.tensor(
                [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]
            ),
        },
    }

    _, preparation_metrics = actor._prepare_train_global_batch_for_microbatches(batch)
    metrics = {name: [value] for name, value in preparation_metrics.items()}
    for idm_max, uncond_max in ((4.0, 3.0), (3.0, 4.0)):
        microbatch_metrics = {
            "perf/rollout_idm_batch_size": idm_max,
            "perf/rollout_uncond_batch_size": uncond_max,
            "perf/teacher_batch_size": uncond_max,
        }
        actor._append_route_neutral_metric_numerators(microbatch_metrics, {})
        for name, value in microbatch_metrics.items():
            metrics.setdefault(name, []).append(value)

    actor._finalize_train_metrics_before_reduction(metrics)

    assert metrics["perf/rollout_idm_batch_size"] == [2.0]
    assert metrics["perf/rollout_uncond_batch_size"] == [2.0]
    assert metrics["perf/teacher_batch_size"] == [2.0]
    assert all(not name.startswith("_route_neutral_compaction/") for name in metrics)


@pytest.mark.parametrize("full_teacher_metrics", [False, True])
def test_mb4_compaction_restores_mb1_metric_denominators(full_teacher_metrics) -> None:
    prefix = "_route_neutral_compaction/"
    actor = SimpleNamespace(
        cfg=SimpleNamespace(
            actor=SimpleNamespace(global_batch_size=8, micro_batch_size=4)
        ),
        gradient_accumulation=2,
        _world_size=1,
        _finalize_online_bc_compaction_metrics=(
            RouteNeutralOnlineIDMBCFSDPActor._finalize_online_bc_compaction_metrics
        ),
    )
    metrics = {
        f"{prefix}global_batch": [0.0, 0.0, 1.0],
        f"{prefix}loss/critic/value_loss": [2.0, 4.0, 6.0],
        f"{prefix}loss/fastwam/regularized_policy_loss": [1.0, 3.0, 4.0],
        f"{prefix}loss/fastwam/total_loss": [3.0, 7.0, 10.0],
        "actor/total_loss": [0.5, 1.0, 1.5],
        f"{prefix}scale/gate": [7.0, 7.0, 9.0],
        f"{prefix}scale/uncond_flow": [2.0, 2.0, 4.0],
        f"{prefix}scale/online_idm_bc": [5.0, 5.0, 7.0],
        f"{prefix}online/selected": [1.0, 1.0, 2.0],
        f"{prefix}online/loss_sum": [2.0, 4.0, 10.0],
        f"{prefix}online/expected": [1.0, 1.0, 2.0],
        f"{prefix}online/present": [1.0, 0.0, 1.0],
        f"{prefix}online/valid_action_count": [3.0, 5.0, 8.0],
        f"{prefix}online/teacher_seconds": [0.1, 0.2, 0.3],
        f"{prefix}online/teacher_bytes": [10.0, 20.0, 30.0],
        f"{prefix}online/mse_pose_sum": [1.0, 3.0, 8.0],
        f"{prefix}online/mse_gripper_sum": [2.0, 2.0, 4.0],
        f"{prefix}online/full_action_mse_sum": [4.0, 4.0, 8.0],
        f"{prefix}online/executed_prefix_mse_sum": [5.0, 3.0, 8.0],
        "online_idm_bc/loss_weight": [0.2],
    }
    if full_teacher_metrics:
        metrics.update(
            {
                f"{prefix}rollout_teacher/count": [3.0, 3.0],
                f"{prefix}rollout_teacher/seconds": [0.7, 1.1],
                f"{prefix}rollout_teacher/bytes": [300.0, 300.0],
            }
        )
    for index in range(7):
        value = float(index + 1)
        metrics[f"{prefix}online/mse_dimension_sum_{index}"] = [
            value,
            value,
            2.0 * value,
        ]
    for index in range(10):
        metrics[f"{prefix}online/timestep_count_{index}"] = (
            [1.0, 0.0, 2.0] if index == 0 else [0.0, 0.0, 0.0]
        )
        metrics[f"{prefix}online/timestep_mse_sum_{index}"] = (
            [2.0, 0.0, 8.0] if index == 0 else [0.0, 0.0, 0.0]
        )

    RouteNeutralOnlineIDMBCFSDPActor._finalize_train_metrics_before_reduction(
        actor,
        metrics,
    )

    assert metrics["critic/value_loss"] == pytest.approx([3.0])
    assert metrics["fastwam/regularized_policy_loss"] == pytest.approx([2.0])
    assert metrics["fastwam/total_loss"] == pytest.approx([5.0])
    assert metrics["actor/total_loss"] == pytest.approx([0.1875])
    assert metrics["gate/selected_loss_scale_compacted"] == pytest.approx([8.0])
    assert metrics["gate/selected_loss_scale"] == pytest.approx([32.0])
    assert metrics["uncond_flow/selected_loss_scale"] == pytest.approx([12.0])
    assert metrics["online_idm_bc/selected_loss_scale"] == pytest.approx([24.0])
    assert metrics["online_idm_bc/raw_loss"] == pytest.approx([4.0])
    assert metrics["online_idm_bc/weighted_loss"] == pytest.approx([0.8])
    assert metrics["online_idm_bc/selected_count"] == pytest.approx([2.0])
    assert metrics["online_idm_bc/teacher_call_count"] == pytest.approx(
        [3.0 if full_teacher_metrics else 1.0]
    )
    assert metrics["online_idm_bc/teacher_seconds_per_call"] == pytest.approx([0.3])
    assert metrics["online_idm_bc/teacher_bytes_per_call"] == pytest.approx(
        [100.0 if full_teacher_metrics else 30.0]
    )
    assert metrics["online_idm_bc/valid_action_count"] == pytest.approx([4.0])
    assert metrics["online_idm_bc/mse_pose"] == pytest.approx([3.0])
    assert metrics["online_idm_bc/mse_timestep_bin_0"] == pytest.approx([10.0 / 3.0])
    assert metrics["online_idm_bc/timestep_bin_count_0"] == [1.0]
    assert metrics[
        "online_idm_bc/timestep_bin_selected_count_compacted_0"
    ] == pytest.approx([1.5])
    assert all(not key.startswith(prefix) for key in metrics)

    reduced = {
        "online_idm_bc/weighted_loss": metrics["online_idm_bc/weighted_loss"][0],
        "uncond_flow/total_loss": 0.5,
    }
    RouteNeutralOnlineIDMBCFSDPActor._finalize_train_metrics_after_reduction(
        actor,
        reduced,
    )
    assert reduced["online_idm_bc/weighted_to_flow_loss_ratio"] == pytest.approx(1.6)


def test_mb4_dummy_batch_consumes_zero_selected_metric_state() -> None:
    prefix = "_route_neutral_compaction/"
    actor = SimpleNamespace(
        cfg=SimpleNamespace(
            actor=SimpleNamespace(global_batch_size=196, micro_batch_size=4)
        ),
        gradient_accumulation=49,
        _world_size=1,
        _finalize_online_bc_compaction_metrics=(
            RouteNeutralOnlineIDMBCFSDPActor._finalize_online_bc_compaction_metrics
        ),
    )
    metrics = {
        f"{prefix}global_batch": [0.0],
        f"{prefix}online/selected": [0.0],
        f"{prefix}online/loss_sum": [0.0],
        f"{prefix}online/expected": [0.0],
        f"{prefix}online/present": [0.0],
        f"{prefix}online/valid_action_count": [0.0],
        f"{prefix}online/teacher_seconds": [0.0],
        f"{prefix}online/teacher_bytes": [0.0],
        f"{prefix}online/mse_pose_sum": [0.0],
        f"{prefix}online/mse_gripper_sum": [0.0],
        f"{prefix}online/full_action_mse_sum": [0.0],
        f"{prefix}online/executed_prefix_mse_sum": [0.0],
        "online_idm_bc/loss_weight": [0.2],
    }
    for index in range(7):
        metrics[f"{prefix}online/mse_dimension_sum_{index}"] = [0.0]
    for index in range(10):
        metrics[f"{prefix}online/timestep_count_{index}"] = [0.0]
        metrics[f"{prefix}online/timestep_mse_sum_{index}"] = [0.0]

    RouteNeutralOnlineIDMBCFSDPActor._finalize_train_metrics_before_reduction(
        actor,
        metrics,
    )

    assert metrics["online_idm_bc/raw_loss"] == [0.0]
    assert metrics["online_idm_bc/selected_count"] == [0.0]
    assert "online_idm_bc/mse_pose" not in metrics
    assert all(not key.startswith(prefix) for key in metrics)
