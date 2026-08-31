# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.algorithms.advantages import (
    apply_fastwam_chunk_cost,
    compute_gae_advantages_and_returns,
)
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    WAMRoute,
)
from rlinf.models.embodiment.wam_policy.pad_rv.actor import PadFrozenFSDPActor
from rlinf.models.embodiment.wam_policy.pad_rv.audit import (
    summarize_pad_frozen_rollout_state,
)
from rlinf.models.embodiment.wam_policy.pad_rv.checkpoint import (
    PAD_FROZEN_CHECKPOINT_SCHEMA,
    build_pad_frozen_checkpoint_contract,
    pad_frozen_artifact_identities,
    validate_pad_frozen_eval_checkpoint,
)
from rlinf.models.embodiment.wam_policy.pad_rv.config import (
    PAD_FROZEN_EGL_INSTANTIATION_TARGET,
    PadFrozenConfig,
    PadRVStage,
    validate_pad_frozen_training_config,
)
from rlinf.models.embodiment.wam_policy.pad_rv.loss import (
    absent_uncond_flow_metrics,
    align_current_step_advantages,
    compute_pad_frozen_policy_loss,
)
from rlinf.models.embodiment.wam_policy.pad_rv.optimizer import (
    assert_pad_frozen_update_resolution,
    partition_pad_frozen_parameters,
)
from rlinf.models.embodiment.wam_policy.pad_rv.routing_state import (
    CurrentStepRouteTracker,
)


def _compose_pad_config(monkeypatch, *, profile: str = "frozen") -> OmegaConf:
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
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        return compose(
            config_name="libero_10_ppo_fastwam_pad_frozen_formal",
            overrides=[f"pad_rv={profile}"],
        )


def _compose_pad_eval_config(monkeypatch) -> OmegaConf:
    repo_root = Path(__file__).parents[2]
    config_dir = repo_root / "evaluations" / "libero"
    environment = {
        "EMBODIED_PATH": str(repo_root / "examples" / "embodiment"),
        "FASTWAM_CHECKPOINT": "/parent.pt",
        "FASTWAM_CHECKPOINT_SHA256": "a" * 64,
        "FASTWAM_DATASET_STATS": "/stats.json",
        "FASTWAM_MERGED_WARM_U": "/merged.pt",
        "FASTWAM_MERGED_WARM_U_SHA256": "b" * 64,
        "FASTWAM_MERGED_WARM_U_SOURCE_LORA_SHA256": "c" * 64,
        "FASTWAM_TEXT_EMBEDDING_CACHE": "/text-cache",
        "PI05_CRITIC_CHECKPOINT": "/critic",
        "PI05_CRITIC_CHECKPOINT_SHA256": "d" * 64,
        "FASTWAM_PROJECT_CHECKPOINT": "/project/actor",
        "FASTWAM_PROJECT_CHECKPOINT_SHA256": "e" * 64,
        "FASTWAM_EVAL_OUTPUT_DIR": "/eval",
        "FASTWAM_EVAL_LEDGER": "/ledger.json",
        "FASTWAM_EVAL_RUN_ID": "pad-eval-unit",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        return compose(config_name="libero_10_fastwam_pad_frozen_eval")


def _gate_record(route: ChunkRouteRecord) -> GateDecisionRecord:
    shape = route.route_used.shape
    probability = torch.full(shape, 0.4)
    return GateDecisionRecord(
        next_route=route.route_used.clone(),
        base_probability=probability,
        behavior_probability=probability,
        old_logprob=torch.log(
            torch.where(route.route_used.bool(), probability, 1 - probability)
        ),
        epsilon=torch.zeros(shape),
        temperature=torch.ones(shape),
        valid=torch.ones(shape, dtype=torch.bool),
        source_chunk_ids=route.chunk_ids.clone(),
        episode_ids=route.episode_ids.clone(),
        actor_versions=route.actor_versions.clone(),
    )


def test_pad_config_is_explicit_and_coadapt_stays_locked() -> None:
    config = PadFrozenConfig.from_mapping(
        {
            "enabled": True,
            "stage": "gate_only_frozen_pair",
            "routing_semantics": "current_step",
        }
    )
    assert config.enabled
    with pytest.raises(NotImplementedError, match="G3"):
        PadFrozenConfig.from_mapping(
            {
                "enabled": True,
                "stage": "coadaptive_uncond_delta",
                "routing_semantics": "current_step",
            }
        )


def test_pad_primary_config_is_the_only_stage_selection_surface(monkeypatch) -> None:
    cfg = _compose_pad_config(monkeypatch)

    validate_pad_frozen_training_config(cfg)
    assert cfg.algorithm.loss_type == "fastwam_gate_only_ppo"
    assert cfg.algorithm.pad_rv.stage == "gate_only_frozen_pair"
    assert cfg.algorithm.prediction_budget.target_idm_fraction == 0.5
    assert cfg.algorithm.prediction_budget.controller_target.endswith(
        ".PadPredictionBudgetController"
    )
    assert cfg.algorithm.prediction_budget.proportional_gain == 0.0
    assert cfg.algorithm.fixed_branch_cost.fair_cost.pi.target_idm_fraction == 0.5
    assert cfg.algorithm.fixed_branch_cost.fair_cost.pi.enabled is False
    assert cfg.actor.model.uncond_lora is None
    assert cfg.rollout.model.uncond_lora is None
    assert cfg.actor.model.kv_replay.backend == "condition"
    assert cfg.rollout.model.kv_replay.backend == "condition"
    assert cfg.pad_rv_implementation.rollout_init_mode == "serial_rank"
    assert cfg.pad_rv_implementation.trajectory_send_mode == "concurrent"
    assert cfg.pad_rv_implementation.release_host_memory_after_rollout_init
    assert cfg.pad_rv_implementation.release_host_memory_after_trajectory_send
    assert cfg.pad_rv_implementation.release_host_memory_after_trajectory_receive
    assert cfg.pad_rv_implementation.rollout_worker_target.endswith(
        ".PadFrozenRolloutWorker"
    )
    assert cfg.pad_rv_implementation.runner_target.endswith(".PadFrozenRunner")
    assert cfg.pad_rv_implementation.env_worker_target.endswith(".PadFrozenEnvWorker")
    assert cfg.pad_rv_implementation.text_cache_preflight_target.endswith(
        ".validate_pad_text_cache_coverage"
    )
    assert cfg.env.train.egl_instantiation_target == PAD_FROZEN_EGL_INSTANTIATION_TARGET
    assert cfg.env.eval.egl_instantiation_target == PAD_FROZEN_EGL_INSTANTIATION_TARGET
    with open_dict(cfg):
        cfg.actor.model.flow_sde.enabled = True
    with pytest.raises(ValueError, match="disables Flow-SDE"):
        validate_pad_frozen_training_config(cfg)


def test_pad_low_gain_budget_profile_is_config_selected(monkeypatch) -> None:
    cfg = _compose_pad_config(monkeypatch, profile="frozen_dual_lr_0p01")

    validate_pad_frozen_training_config(cfg)
    assert cfg.algorithm.prediction_budget.dual_lr == 0.01
    assert cfg.algorithm.fixed_branch_cost.fair_cost.pi.integral_gain == 0.01
    assert cfg.algorithm.prediction_budget.controller_target.endswith(
        ".PadPredictionBudgetController"
    )


def test_pad_low_gain_serial_send_profile_is_config_selected(monkeypatch) -> None:
    cfg = _compose_pad_config(
        monkeypatch,
        profile="frozen_dual_lr_0p01_serial_send",
    )

    validate_pad_frozen_training_config(cfg)
    assert cfg.algorithm.prediction_budget.dual_lr == 0.01
    assert cfg.pad_rv_implementation.trajectory_send_mode == "serialized"


def test_pad_high_entropy_calibration_profile_is_config_selected(monkeypatch) -> None:
    cfg = _compose_pad_config(
        monkeypatch,
        profile="frozen_dual_lr_0p01_serial_send_entropy_0p05",
    )

    validate_pad_frozen_training_config(cfg)
    assert cfg.algorithm.prediction_budget.dual_lr == 0.01
    assert cfg.algorithm.gate_ppo.entropy_coefficient == 0.05
    assert cfg.actor.optim.gate_lr == 3e-5
    assert cfg.pad_rv_implementation.trajectory_send_mode == "serialized"


def test_pad_eval_config_selects_incremental_workers_and_no_actor(monkeypatch) -> None:
    cfg = _compose_pad_eval_config(monkeypatch)

    validate_pad_frozen_training_config(cfg, only_eval=True)

    assert cfg.runner.only_eval is True
    assert "actor" not in cfg
    assert cfg.cluster.component_placement["env,rollout"] == "0-6"
    assert cfg.env.eval.total_num_envs == 7
    assert cfg.rollout.model.gate_epsilon == 0.0
    assert cfg.rollout.model.eval_routing_mode == "learned_threshold"
    assert cfg.rollout.model.gate.layer_taps.mode == "indices"
    assert cfg.rollout.model.gate.layer_taps.indices == [14, 15, 16, 17, 18, 19]
    assert cfg.rollout.model.builder_target.endswith(".build_pad_frozen_model")
    assert cfg.rollout.model.policy_target.endswith(".PadFrozenPolicy")
    assert cfg.pad_rv_implementation.rollout_worker_target.endswith(
        ".PadFrozenRolloutWorker"
    )
    assert cfg.pad_rv_implementation.env_worker_target.endswith(".PadFrozenEnvWorker")
    assert cfg.pad_rv_implementation.evaluation_runner_target.endswith(
        ".PadFrozenEvalRunner"
    )
    assert cfg.runner.evaluation_collector._target_.endswith(".PadFrozenEvalCollector")
    assert cfg.runner.evaluation_collector.ledger_shard_count == 7


def test_pad_eval_rejects_filtered_ordered_reset_ids_before_model_load(
    monkeypatch,
) -> None:
    cfg = _compose_pad_eval_config(monkeypatch)
    with open_dict(cfg.env.eval):
        cfg.env.eval.task_id_filter = [0]

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_pad_frozen_training_config(cfg, only_eval=True)


def test_pad_eval_collector_completes_only_its_round_robin_ledger_shard(
    tmp_path,
) -> None:
    import hashlib
    import json

    from rlinf.models.embodiment.wam_policy.pad_rv.eval_collector import (
        PadFrozenEvalCollector,
    )

    entries = []
    for index in range(7):
        identity = {
            "task_suite": "libero_10",
            "task_id": 0,
            "trial_id": index,
            "reset_state_id": index,
            "environment_seed": 42,
            "action_noise_seed": 42000 + index,
            "idm_video_noise_seed": 52000 + index,
            "max_primitive_steps": 700,
            "generation_horizon": 32,
            "execution_horizon": 10,
            "prediction_video_frames": 9,
            "reset_wait_steps": 30,
        }
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        entries.append(
            {
                **identity,
                "episode_index": index,
                "episode_identity": hashlib.sha256(encoded).hexdigest(),
            }
        )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema": "fastwam-libero-eval-ledger-v2",
                "kind": "validation",
                "task_suite": "libero_10",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    collector = PadFrozenEvalCollector(
        output_dir=str(tmp_path / "output"),
        ledger_path=str(ledger_path),
        run_id="pad-shard-test",
        rank=3,
        ledger_shard_count=7,
        routing_mode="learned_threshold",
        idm_threshold=0.5,
        random_idm_probability=None,
        random_lag1_autocorrelation=None,
        routing_seed=0,
        fixed_idm_cost=0.01,
        evaluation_runtime_identity={
            "environment": {
                "ordered_reset_state_ids": list(range(7)),
                "total_num_envs": 7,
            }
        },
    )

    assert collector.local_ledger_reset_state_ids == (3,)
    assert collector.is_complete is False
    collector._episodes.append({"episode_identity": entries[3]["episode_identity"]})
    assert collector.is_complete is True
    stop = collector.build_rollout_stop_control(logical_batch_size=7)
    assert stop.logical_batch_size == 1
    assert stop.completed_episode_count == stop.ledger_episode_count == 1
    assert stop.ledger_sha256 == hashlib.sha256(ledger_path.read_bytes()).hexdigest()


def _small_pad_eval_payload(model_cfg, *, finite: bool = True) -> dict:
    value = torch.tensor([1.0 if finite else float("nan")])
    return {
        "schema": PAD_FROZEN_CHECKPOINT_SCHEMA,
        "owner": "actor",
        "step": 5,
        "optimizer_steps": 50,
        "versions": {"actor": 5, "gate": 50, "critic": 50},
        "stage_contract": {
            "schema": "fastwam-gate-only-frozen-pair-contract-v1",
            "stage": "gate_only_frozen_pair",
            "routing_semantics": "current_step",
            "loss_type": "fastwam_gate_only_ppo",
            "artifact_identities": pad_frozen_artifact_identities(model_cfg),
            "execution": {"training_only": True},
        },
        "policy": {
            "schema": "pad-frozen-policy-v1",
            "actor_version": 5,
            "gate": {"weight": value.clone()},
            "value_head": {"weight": torch.ones(1)},
            "route_tracker": {"training_state": True},
        },
        "optimizer": {},
        "lr_scheduler": {},
        "grad_scaler": {},
        "rng": {},
    }


def test_pad_eval_checkpoint_validates_artifacts_and_finite_trainables(
    monkeypatch,
) -> None:
    cfg = _compose_pad_eval_config(monkeypatch)
    payload = _small_pad_eval_payload(cfg.rollout.model)

    assert validate_pad_frozen_eval_checkpoint(payload, cfg.rollout.model) is payload

    invalid = _small_pad_eval_payload(cfg.rollout.model, finite=False)
    with pytest.raises(ValueError, match="non-finite"):
        validate_pad_frozen_eval_checkpoint(invalid, cfg.rollout.model)


def test_pad_policy_eval_loads_gate_and_resets_training_route_state(
    monkeypatch,
) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.policy import PadFrozenPolicy

    cfg = _compose_pad_eval_config(monkeypatch)
    policy = object.__new__(PadFrozenPolicy)
    torch.nn.Module.__init__(policy)
    policy.gate = torch.nn.Linear(1, 1, bias=False)
    policy.critic = None
    policy.route_tracker = object()
    payload = _small_pad_eval_payload(cfg.rollout.model)
    payload["policy"]["gate"] = policy.gate.state_dict()

    version = policy.load_eval_checkpoint(
        payload,
        expected_parent_checkpoint_sha256="a" * 64,
    )

    assert version == 5
    assert policy.actor_version == 5
    assert isinstance(policy.route_tracker, CurrentStepRouteTracker)


def test_pad_policy_eval_never_requests_training_critic() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.policy import PadFrozenPolicy

    policy = object.__new__(PadFrozenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        training_rollout_microbatch_size=None,
        decision_telemetry_enabled=False,
    )
    observed = {}
    policy._routing_metadata = lambda _obs, batch, device: (
        torch.arange(batch, device=device),
        torch.ones(batch, dtype=torch.bool, device=device),
    )

    def predict_current_step(**kwargs):
        observed.update(kwargs)
        return torch.zeros(1, 1), {}

    policy._predict_current_step = predict_current_step

    policy.predict_action_batch(
        {"states": torch.zeros(1, 8)},
        mode="eval",
        compute_values=True,
    )

    assert observed["compute_values"] is False


def test_pad_eval_runner_initializes_rollout_ranks_serially() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.eval_runner import (
        PadFrozenEvalRunner,
    )

    events = []

    class Handle:
        def wait(self):
            events.append("wait")

    class RankTarget:
        def __init__(self, rank):
            self.rank = rank

        def init_worker(self):
            events.append(("rollout", self.rank))
            return Handle()

    class Rollout:
        worker_info_list = [SimpleNamespace(rank=0), SimpleNamespace(rank=1)]

        def execute_on(self, rank):
            return RankTarget(rank)

    class Env:
        def init_worker(self):
            events.append("env")
            return Handle()

    runner = object.__new__(PadFrozenEvalRunner)
    runner.cfg = SimpleNamespace(
        pad_rv_implementation=SimpleNamespace(rollout_init_mode="serial_rank")
    )
    runner.rollout = Rollout()
    runner.env = Env()
    runner.logger = SimpleNamespace(info=lambda *_args: None)

    runner.init_workers()

    assert events == [
        ("rollout", 0),
        "wait",
        ("rollout", 1),
        "wait",
        "env",
        "wait",
    ]


def test_pad_physical_egl_factory_preserves_ray_assignment(monkeypatch) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.egl import (
        instantiate_with_physical_egl,
    )

    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "4")
    observed = {}

    class FakeEnvironment:
        pass

    def factory(*, marker):
        observed["egl"] = __import__("os").environ["MUJOCO_EGL_DEVICE_ID"]
        observed["marker"] = marker
        return FakeEnvironment()

    environment = instantiate_with_physical_egl(factory, {"marker": "created"})

    assert observed == {"egl": "4", "marker": "created"}
    assert environment._rlinf_egl_device_mapping == {
        "applied": True,
        "backend": "egl",
        "mapping_mode": "ray_assigned_physical_device",
        "physical_visible_device": "4",
        "physical_mujoco_egl_device_id": "4",
        "remap_boundary": "libero_env_factory_before_renderer_construction",
    }


def test_pad_validator_rejects_default_logical_zero_egl_factory(monkeypatch) -> None:
    cfg = _compose_pad_config(monkeypatch)
    with open_dict(cfg):
        cfg.env.train.egl_instantiation_target = (
            "rlinf.envs.libero.egl.instantiate_with_isolated_egl"
        )

    with pytest.raises(ValueError, match="env.train.egl_instantiation_target"):
        validate_pad_frozen_training_config(cfg)


def test_pad_checkpoint_contract_owns_stage_and_frozen_artifact_identities(
    monkeypatch,
) -> None:
    cfg = _compose_pad_config(monkeypatch)

    contract = build_pad_frozen_checkpoint_contract(cfg, world_size=1)

    assert contract["stage"] == "gate_only_frozen_pair"
    assert contract["loss_type"] == "fastwam_gate_only_ppo"
    identities = contract["artifact_identities"]
    assert identities["idm_parent_checkpoint_sha256"] == "a" * 64
    assert identities["merged_warm_uncond"] == {
        "schema": "fastwam-frozen-uncond-action-v1",
        "checkpoint_sha256": "b" * 64,
        "source_lora_sidecar_sha256": "c" * 64,
    }
    assert identities["critic_parent_checkpoint_sha256"] == "d" * 64


def test_current_step_tracker_controls_same_chunk_without_forced_first_idm() -> None:
    tracker = CurrentStepRouteTracker()
    identity = tracker.prepare(
        env_ids=torch.tensor([4, 9]),
        reset_mask=torch.tensor([True, True]),
        actor_version=3,
    )
    route = tracker.commit(
        identity=identity,
        routes=torch.tensor([int(WAMRoute.UNCOND), int(WAMRoute.IDM)]),
    )
    assert torch.equal(route.chunk_ids, torch.tensor([0, 0]))
    assert torch.equal(route.route_source_chunk_ids, route.chunk_ids)
    assert not bool(route.route_was_forced.any())
    assert route.route_used.tolist() == [int(WAMRoute.UNCOND), int(WAMRoute.IDM)]


def test_pad_global_step_never_injects_delayed_route_semantics() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.policy import PadFrozenPolicy

    policy = object.__new__(PadFrozenPolicy)
    torch.nn.Module.__init__(policy)
    policy.actor_version = 0
    policy.route_tracker = CurrentStepRouteTracker()

    PadFrozenPolicy.set_global_step(policy, 3)

    assert policy.actor_version == 3
    identity = policy.route_tracker.prepare(
        env_ids=torch.tensor([0]),
        reset_mask=torch.tensor([False]),
        actor_version=3,
    )
    route = policy.route_tracker.commit(identity=identity, routes=torch.tensor([0]))
    assert not bool(route.route_was_forced.any())


def test_pad_microbatch_merge_omits_action_logprobs() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.policy import PadFrozenPolicy

    tracker = CurrentStepRouteTracker()
    results = []
    actions = []
    for env_id in (0, 1):
        identity = tracker.prepare(
            env_ids=torch.tensor([env_id]),
            reset_mask=torch.tensor([True]),
            actor_version=0,
        )
        route = tracker.commit(identity=identity, routes=torch.tensor([env_id]))
        results.append(
            {
                "prev_logprobs": None,
                "prev_values": torch.tensor([[float(env_id)]]),
                "forward_inputs": {"feature": torch.tensor([[float(env_id)]])},
                "route_info": route,
                "emitted_gate": _gate_record(route),
                "action_execution_trace": None,
            }
        )
        actions.append(torch.full((1, 7), float(env_id)))

    policy = object.__new__(PadFrozenPolicy)
    torch.nn.Module.__init__(policy)
    merged_actions, merged = policy._merge_training_microbatch_results(actions, results)

    assert merged_actions.shape == (2, 7)
    assert merged["prev_logprobs"] is None
    assert merged["prev_values"].shape == (2, 1)
    assert merged["forward_inputs"]["feature"].shape == (2, 1)


def test_current_step_alignment_has_no_flow_policy_samples() -> None:
    tracker = CurrentStepRouteTracker()
    identity = tracker.prepare(
        env_ids=torch.tensor([0, 1]),
        reset_mask=torch.tensor([True, True]),
        actor_version=0,
    )
    route = tracker.commit(identity=identity, routes=torch.tensor([0, 1]))
    flat = ChunkRouteRecord.cat([route, route])
    route = ChunkRouteRecord(
        route_used=flat.route_used.reshape(2, 2),
        route_was_forced=flat.route_was_forced.reshape(2, 2),
        chunk_ids=flat.chunk_ids.reshape(2, 2),
        episode_ids=flat.episode_ids.reshape(2, 2),
        route_source_chunk_ids=flat.route_source_chunk_ids.reshape(2, 2),
        actor_versions=flat.actor_versions.reshape(2, 2),
    )
    emitted = _gate_record(route)
    advantages = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    alignment = align_current_step_advantages(
        advantages=advantages,
        route=route,
        emitted=emitted,
        loss_mask=torch.ones(2, 2, 1, dtype=torch.bool),
    )
    assert torch.equal(alignment.gate_advantages, advantages[..., 0])
    assert bool(alignment.gate_valid_mask.all())
    assert not bool(alignment.flow_valid_mask.any())
    assert torch.count_nonzero(alignment.flow_advantages) == 0


def test_pad_counterfactual_audit_uses_current_step_alignment_and_std_floor() -> None:
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
    emitted = _gate_record(route)
    environment_rewards = torch.zeros(2, 2, 1, dtype=torch.float32)
    values = torch.zeros(3, 2, 1)
    dones = torch.zeros(3, 2, 1, dtype=torch.bool)
    valid_mask = torch.ones(2, 2, 1, dtype=torch.bool)
    configured_cost = 0.015
    configured_rewards = apply_fastwam_chunk_cost(
        environment_rewards=environment_rewards,
        route_used=route.route_used,
        idm_cost=configured_cost,
        valid_mask=valid_mask,
    ).rewards
    configured_advantages, _ = compute_gae_advantages_and_returns(
        rewards=configured_rewards[..., 0],
        values=values[..., 0],
        dones=dones[..., 0],
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        loss_mask=valid_mask[..., 0],
        normalization_std_floor=0.15,
    )
    configured_alignment = align_current_step_advantages(
        advantages=configured_advantages.unsqueeze(-1),
        route=route,
        emitted=emitted,
        loss_mask=valid_mask,
    )
    actor = SimpleNamespace(
        cfg=SimpleNamespace(algorithm={"advantage_normalization_std_floor": 0.15}),
        _align_fastwam_training_advantages=lambda **kwargs: (
            align_current_step_advantages(
                advantages=kwargs["advantages"],
                route=kwargs["route"],
                emitted=kwargs["emitted"],
                loss_mask=kwargs.get("loss_mask"),
            )
        ),
    )

    audit = PadFrozenFSDPActor._summarize_fastwam_counterfactual_costs(
        actor,
        environment_rewards=environment_rewards,
        route=route,
        emitted=emitted,
        dones=dones,
        values=values,
        valid_mask=valid_mask,
        idm_costs=(0.0, configured_cost, 0.025),
        configured_idm_cost=configured_cost,
        configured_gate_advantages=configured_alignment.gate_advantages,
        gamma=0.99,
        gae_lambda=0.95,
        rollout_epoch=1,
        carry_pending_across_epochs=False,
    )

    assert audit.configured_alignment_max_abs_error == 0.0
    assert audit.eligible_gate_decision_count == 4
    assert audit.eligible_idm_decision_count == 2
    assert audit.eligible_uncond_decision_count == 2


def test_pad_rollout_audit_declares_condition_replay_and_zero_action_kv() -> None:
    tracker = CurrentStepRouteTracker()
    identity = tracker.prepare(
        env_ids=torch.tensor([0, 1]),
        reset_mask=torch.tensor([True, True]),
        actor_version=0,
    )
    one_step = tracker.commit(identity=identity, routes=torch.tensor([0, 1]))
    flat = ChunkRouteRecord.cat([one_step, one_step])
    route = ChunkRouteRecord(
        route_used=flat.route_used.reshape(2, 2),
        route_was_forced=flat.route_was_forced.reshape(2, 2),
        chunk_ids=flat.chunk_ids.reshape(2, 2),
        episode_ids=flat.episode_ids.reshape(2, 2),
        route_source_chunk_ids=flat.route_source_chunk_ids.reshape(2, 2),
        actor_versions=flat.actor_versions.reshape(2, 2),
    )
    emitted = _gate_record(route)

    audit = summarize_pad_frozen_rollout_state(
        route=route,
        emitted=emitted,
        eligible_gate_mask=torch.ones(2, 2, dtype=torch.bool),
        valid_mask=torch.ones(2, 2, 1, dtype=torch.bool),
        kv_replay_backend="condition",
        max_bytes_per_sample=268435456,
    )

    artifact = audit.to_artifact()
    assert artifact["kv_replay_backend"] == "condition"
    assert artifact["kv_storage_dtype"] == "none"
    assert artifact["kv_layer_indices"] == []
    assert artifact["kv_denoise_tap_count"] == 0
    assert artifact["kv_all_emitted"] == {
        "sample_count": 0,
        "nonzero_sample_count": 0,
        "total_bytes": 0,
        "maximum_bytes_per_sample": 0,
    }
    assert artifact["valid_idm_chunk_count"] == 2
    assert artifact["valid_uncond_chunk_count"] == 2


def test_pad_absent_flow_metrics_reconcile_executed_uncond_without_flow_loss() -> None:
    reference = torch.tensor(3.0, requires_grad=True)
    metrics = absent_uncond_flow_metrics(
        route_used=torch.tensor(
            [WAMRoute.UNCOND, WAMRoute.IDM, WAMRoute.UNCOND, WAMRoute.UNCOND]
        ),
        valid_chunk_mask=torch.tensor([True, True, False, True]),
        reference=reference,
    )

    assert metrics["uncond_flow/sample_count"].item() == 2.0
    assert metrics["uncond_flow/ratio"].item() == 1.0
    for suffix in (
        "policy_loss",
        "total_loss",
        "ratio_abs",
        "log_ratio_max_abs",
        "approx_kl",
        "clip_fraction",
        "entropy",
    ):
        assert metrics[f"uncond_flow/{suffix}"].item() == 0.0
        assert not metrics[f"uncond_flow/{suffix}"].requires_grad


def test_pad_actor_unwraps_exact_policy_through_forwarding_fsdp_wrapper() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.actor import PadFrozenFSDPActor
    from rlinf.models.embodiment.wam_policy.pad_rv.policy import PadFrozenPolicy

    policy = PadFrozenPolicy.__new__(PadFrozenPolicy)
    torch.nn.Module.__init__(policy)

    class ForwardingFSDPWrapper:
        def __init__(self, wrapped):
            self._fsdp_wrapped_module = wrapped

        def __getattr__(self, name):
            return getattr(self._fsdp_wrapped_module, name)

    actor = PadFrozenFSDPActor.__new__(PadFrozenFSDPActor)
    actor.model = ForwardingFSDPWrapper(policy)
    assert actor._fastwam_policy_module() is policy


def test_pad_replay_uses_pi05_pooling_interface_before_value_head() -> None:
    from fastwam.models.wan22.condition_kv import ConditionLayerKV
    from fastwam.models.wan22.kv_tap import KeyValueBank, KVSource

    from rlinf.models.embodiment.wam_policy.critic import FastWAMValueFeatures
    from rlinf.models.embodiment.wam_policy.pad_rv.gate import (
        serialize_condition_features,
    )
    from rlinf.models.embodiment.wam_policy.pad_rv.policy import PadFrozenPolicy

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(layer_indices=(14,))

        def forward(self, features):
            return torch.zeros(
                features.layers[0].current_frame_video.key.shape[0],
                device=self.anchor.device,
            )

    class RejectDirectValueHead(torch.nn.Module):
        def forward(self, _features):
            raise AssertionError("PAD replay bypassed pi0.5 prefix pooling")

    class Critic(torch.nn.Module):
        kind = "pi0_5_value_after_vlm"
        replay_feature_key = "critic_prefix"

        def __init__(self):
            super().__init__()
            self.value_head = RejectDirectValueHead()
            self.pooling_calls = 0

        def value_from_features(self, features):
            assert features.shape == (1, 3, 4)
            self.pooling_calls += 1
            return torch.tensor([0.25])

    def bank(source):
        return KeyValueBank(
            source=source,
            key=torch.zeros(1, 1, 1),
            value=torch.zeros(1, 1, 1),
            valid_mask=torch.ones(1, 1, dtype=torch.bool),
        )

    features = FastWAMValueFeatures(
        (
            ConditionLayerKV(
                layer_index=14,
                current_frame_video=bank(KVSource.CURRENT_FRAME_VIDEO),
                context=bank(KVSource.TEXT_STATE_CONTEXT),
            ),
        )
    )
    route = ChunkRouteRecord(
        route_used=torch.tensor([1]),
        route_was_forced=torch.tensor([False]),
        chunk_ids=torch.tensor([0]),
        episode_ids=torch.tensor([0]),
        route_source_chunk_ids=torch.tensor([0]),
        actor_versions=torch.tensor([0]),
    )
    policy = PadFrozenPolicy.__new__(PadFrozenPolicy)
    torch.nn.Module.__init__(policy)
    policy.gate = Gate()
    policy.critic = Critic()
    forward_inputs = serialize_condition_features(features, prefix="route_condition")
    forward_inputs["critic_prefix"] = torch.zeros(1, 3, 4)

    result = policy.default_forward(
        forward_inputs,
        route_info=route,
        emitted_gate=_gate_record(route),
    )

    assert result["values"].shape == (1, 1)
    assert result["values"].item() == pytest.approx(0.25)
    assert policy.critic.pooling_calls == 1


def test_pad_policy_loss_contains_gate_only() -> None:
    loss, metrics = compute_pad_frozen_policy_loss(
        gate_logprobs=torch.log(torch.tensor([0.6, 0.4])),
        gate_old_logprobs=torch.log(torch.tensor([0.6, 0.4])),
        gate_advantages=torch.tensor([1.0, -1.0]),
        gate_valid_mask=torch.tensor([True, True]),
        gate_clip_ratio_low=0.2,
        gate_clip_ratio_high=0.2,
        gate_base_probabilities=torch.tensor([0.6, 0.4]),
        gate_behavior_probabilities=torch.tensor([0.6, 0.4]),
        gate_entropy_coefficient=0.0,
        gate_loss_coefficient=1.0,
        selected_loss_scale=0.5,
    )
    assert torch.isfinite(loss)
    assert "pad_frozen/total_policy_loss" in metrics
    assert not any("flow" in key for key in metrics)


class _OwnershipModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, 2).requires_grad_(False)
        self.uncond_action_expert = torch.nn.Linear(2, 2).requires_grad_(False)
        self.gate = torch.nn.Linear(2, 1).float()
        self.value_head = torch.nn.Linear(2, 1).float()


def test_pad_optimizer_owns_exactly_gate_and_value() -> None:
    model = _OwnershipModule()
    groups = partition_pad_frozen_parameters(model.named_parameters())
    assert set(groups) == {"gate", "value_head"}
    owned = {
        id(parameter) for parameters in groups.values() for parameter in parameters
    }
    assert owned == {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("gate.") or name.startswith("value_head.")
    }
    assert all(not parameter.requires_grad for parameter in model.actor.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.uncond_action_expert.parameters()
    )


def test_pad_update_resolution_audits_uninitialized_adam_state() -> None:
    model = _OwnershipModule()
    groups = partition_pad_frozen_parameters(model.named_parameters())
    for parameter in (*groups["gate"], *groups["value_head"]):
        parameter.grad = torch.full_like(parameter, 0.1)
    optimizer = torch.optim.AdamW(
        [
            {"name": "gate", "params": groups["gate"], "lr": 3e-5},
            {
                "name": "value_head",
                "params": groups["value_head"],
                "lr": 1e-4,
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )

    report = assert_pad_frozen_update_resolution(optimizer, minimum_half_ulp_ratio=1.0)

    assert set(report) == {"gate", "value_head"}
    assert all(details["observed_half_ulp_ratio"] > 1 for details in report.values())
    assert all(
        details["active_update_fraction"] == pytest.approx(1.0)
        for details in report.values()
    )
    assert not optimizer.state


def test_pad_update_resolution_accepts_sparse_active_relu_updates() -> None:
    model = _OwnershipModule()
    groups = partition_pad_frozen_parameters(model.named_parameters())
    for parameter in groups["gate"]:
        parameter.grad = torch.full_like(parameter, 0.1)
    for parameter in groups["value_head"]:
        parameter.grad = torch.zeros_like(parameter)
    groups["value_head"][0].grad.reshape(-1)[0] = 0.1
    optimizer = torch.optim.AdamW(
        [
            {"name": "gate", "params": groups["gate"], "lr": 3e-5},
            {
                "name": "value_head",
                "params": groups["value_head"],
                "lr": 1e-4,
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )

    report = assert_pad_frozen_update_resolution(optimizer, minimum_half_ulp_ratio=1.0)

    assert report["value_head"]["active_update_value_count"] == 1
    assert report["value_head"]["active_update_fraction"] == pytest.approx(1 / 3)
    assert report["value_head"]["observed_half_ulp_ratio"] > 1


def test_pad_update_resolution_rejects_all_zero_value_updates() -> None:
    model = _OwnershipModule()
    groups = partition_pad_frozen_parameters(model.named_parameters())
    for parameter in groups["gate"]:
        parameter.grad = torch.full_like(parameter, 0.1)
    for parameter in groups["value_head"]:
        parameter.grad = torch.zeros_like(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"name": "gate", "params": groups["gate"], "lr": 3e-5},
            {
                "name": "value_head",
                "params": groups["value_head"],
                "lr": 1e-4,
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )

    with pytest.raises(RuntimeError, match="no nonzero proposed updates.*value_head"):
        assert_pad_frozen_update_resolution(optimizer, minimum_half_ulp_ratio=1.0)


def test_pad_ownership_audit_separates_moving_and_frozen_families() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.actor import PadFrozenFSDPActor

    before = {
        "gate.weight": torch.zeros(2),
        "value_head.weight": torch.zeros(3),
    }
    after = {
        "gate.weight": torch.tensor([0.0, 0.25]),
        "value_head.weight": torch.tensor([0.5, 0.0, 0.0]),
    }
    gate = PadFrozenFSDPActor._movement_summary(before, after, prefix="gate.")
    value = PadFrozenFSDPActor._movement_summary(before, after, prefix="value_head.")
    frozen = torch.nn.Linear(2, 2).requires_grad_(False)
    frozen_hash = PadFrozenFSDPActor._module_parameter_sha256(frozen)

    assert gate["changed_value_count"] == 1
    assert value["changed_value_count"] == 1
    assert frozen_hash == PadFrozenFSDPActor._module_parameter_sha256(frozen)
    with torch.no_grad():
        frozen.weight.add_(1.0)
    assert frozen_hash != PadFrozenFSDPActor._module_parameter_sha256(frozen)


def test_pad_actor_uses_versions_as_batch_reference() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.actor import PadFrozenFSDPActor
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    pad_actor = object.__new__(PadFrozenFSDPActor)
    versions = torch.zeros(10, 6, 1)
    assert pad_actor._training_batch_reference({"versions": versions}) is versions
    assert pad_actor._allows_absent_action_logprobs()

    legacy_actor = object.__new__(EmbodiedFSDPActor)
    logprobs = torch.zeros(10, 6, 4)
    assert (
        legacy_actor._training_batch_reference({"prev_logprobs": logprobs}) is logprobs
    )
    assert not legacy_actor._allows_absent_action_logprobs()


def test_model_builder_target_is_opt_in(monkeypatch) -> None:
    import hydra.utils

    from rlinf.models import get_model

    sentinel = SimpleNamespace(to=lambda *args, **kwargs: sentinel)
    called = []

    def builder(cfg, dtype):
        called.append((cfg.model_type, dtype))
        return sentinel

    monkeypatch.setattr(hydra.utils, "get_method", lambda target: builder)
    cfg = OmegaConf.create(
        {
            "model_type": "fastwam_adaptive",
            "precision": "bf16",
            "is_lora": False,
            "load_to_device": False,
            "builder_target": "tests.pad.builder",
        }
    )
    assert get_model(cfg) is sentinel
    assert called and called[0][0] == "fastwam_adaptive"


def test_pad_runner_initializes_rollout_ranks_serially() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.runner import PadFrozenRunner

    events = []

    class _Handle:
        def __init__(self, label):
            self.label = label

        def wait(self):
            events.append(("done", self.label))
            return None

    class _Rollout:
        worker_info_list = [SimpleNamespace(rank=0), SimpleNamespace(rank=1)]
        selected_rank = None

        def execute_on(self, rank):
            self.selected_rank = rank
            return self

        def init_worker(self):
            events.append(("start", f"rollout-{self.selected_rank}"))
            return _Handle(f"rollout-{self.selected_rank}")

    class _SingleGroup:
        def __init__(self, label):
            self.label = label

        def init_worker(self):
            events.append(("start", self.label))
            return _Handle(self.label)

    runner = object.__new__(PadFrozenRunner)
    runner.cfg = OmegaConf.create(
        {
            "pad_rv_implementation": {"rollout_init_mode": "serial_rank"},
            "runner": {"resume_dir": None},
        }
    )
    runner.rollout = _Rollout()
    runner.env = _SingleGroup("env")
    runner.actor = _SingleGroup("actor")
    runner.reward = None
    runner.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    runner.init_workers()

    assert events == [
        ("start", "rollout-0"),
        ("done", "rollout-0"),
        ("start", "rollout-1"),
        ("done", "rollout-1"),
        ("start", "env"),
        ("done", "env"),
        ("start", "actor"),
        ("done", "actor"),
    ]


def test_pad_rollout_versions_do_not_depend_on_action_logprobs() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.rollout import (
        PadFrozenRolloutWorker,
    )

    worker = object.__new__(PadFrozenRolloutWorker)
    worker.collect_prev_infos = True
    worker.model_cfg = SimpleNamespace(num_action_chunks=10)
    worker.version = 4
    worker.get_bootstrap_values = lambda _final_obs: None
    actions = torch.zeros(2, 70)
    rollout = worker._build_rollout_result(
        actions,
        {
            "prev_logprobs": None,
            "prev_values": torch.zeros(2, 1),
            "forward_inputs": {"feature": torch.zeros(2, 3)},
        },
    )

    assert rollout.prev_logprobs is None
    assert rollout.versions.shape == (2, 1)
    assert rollout.versions.tolist() == [[4.0], [4.0]]
    assert all(dimension > 0 for dimension in rollout.versions.shape)


class _CheckpointStateful:
    def __init__(self, value: float) -> None:
        self.value = torch.tensor([value])

    def state_dict(self):
        return {"value": self.value.clone()}

    def load_state_dict(self, state) -> None:
        self.value.copy_(state["value"])


class _TinyPadCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value_head = torch.nn.Linear(2, 1)


def _tiny_pad_policy():
    from rlinf.models.embodiment.wam_policy.pad_rv.policy import PadFrozenPolicy

    policy = object.__new__(PadFrozenPolicy)
    torch.nn.Module.__init__(policy)
    policy.gate = torch.nn.Linear(2, 2)
    policy.critic = _TinyPadCritic()
    policy.route_tracker = CurrentStepRouteTracker()
    policy.actor_version = 0
    return policy


def test_pad_actor_checkpoint_is_stage_owned_and_omits_frozen_experts(
    tmp_path,
    monkeypatch,
) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv import actor as actor_module
    from rlinf.models.embodiment.wam_policy.pad_rv.actor import PadFrozenFSDPActor
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    class _ActorWorker:
        _checkpoint_cpu_clone = staticmethod(EmbodiedFSDPActor._checkpoint_cpu_clone)
        _fastwam_policy_module = EmbodiedFSDPActor._fastwam_policy_module
        save_checkpoint = PadFrozenFSDPActor.save_checkpoint
        load_checkpoint = PadFrozenFSDPActor.load_checkpoint

    monkeypatch.setattr(
        actor_module,
        "build_pad_frozen_checkpoint_contract",
        lambda *_args, **_kwargs: {"kind": "pad-unit"},
    )
    monkeypatch.setattr(
        actor_module,
        "validate_pad_frozen_checkpoint_contract",
        lambda contract, *_args, **_kwargs: contract,
    )
    rng_state = {"cpu": torch.tensor([7], dtype=torch.uint8)}
    monkeypatch.setattr(actor_module, "get_rng_state", lambda: rng_state)
    monkeypatch.setattr(actor_module, "set_rng_state", lambda _state: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    worker = _ActorWorker()
    worker.model = _tiny_pad_policy()
    worker.cfg = SimpleNamespace()
    worker.pad_config = SimpleNamespace(stage=PadRVStage.FROZEN)
    worker.optimizer = _CheckpointStateful(2.0)
    worker.lr_scheduler = _CheckpointStateful(3.0)
    worker.grad_scaler = _CheckpointStateful(4.0)
    worker.optimizer_steps = 10
    worker.version = 0
    worker._rank = 0
    worker._world_size = 1
    worker.device = torch.device("cpu")
    worker.is_weight_offloaded = False
    worker.is_optimizer_offloaded = False
    checkpoint_dir = tmp_path / "actor"
    original_gate = {
        name: value.clone() for name, value in worker.model.gate.state_dict().items()
    }

    worker.save_checkpoint(str(checkpoint_dir), step=1)

    payload = torch.load(
        checkpoint_dir / "rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["schema"] == PAD_FROZEN_CHECKPOINT_SCHEMA
    assert payload["owner"] == "actor"
    assert payload["versions"] == {"actor": 1, "gate": 10, "critic": 10}
    assert set(payload["policy"]) == {
        "schema",
        "actor_version",
        "gate",
        "value_head",
        "route_tracker",
    }
    assert "uncond_action_expert" not in payload
    assert "action_optimizer" not in payload

    with torch.no_grad():
        for parameter in worker.model.gate.parameters():
            parameter.add_(9.0)
    worker.optimizer.value.fill_(9.0)
    assert worker.load_checkpoint(str(checkpoint_dir)) == 1
    assert worker.version == 1
    assert worker.optimizer_steps == 10
    assert worker.optimizer.value.item() == pytest.approx(2.0)
    for name, value in worker.model.gate.state_dict().items():
        assert torch.equal(value, original_gate[name])


def test_pad_rollout_checkpoint_rejects_legacy_schema(tmp_path, monkeypatch) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv import rollout as rollout_module
    from rlinf.models.embodiment.wam_policy.pad_rv.rollout import (
        PadFrozenRolloutWorker,
    )

    monkeypatch.setattr(
        rollout_module,
        "build_pad_frozen_checkpoint_contract",
        lambda *_args, **_kwargs: {"kind": "pad-unit"},
    )
    monkeypatch.setattr(
        rollout_module,
        "validate_pad_frozen_checkpoint_contract",
        lambda contract, *_args, **_kwargs: contract,
    )
    rng_state = {"cpu": torch.tensor([8], dtype=torch.uint8)}
    monkeypatch.setattr(rollout_module, "get_rng_state", lambda: rng_state)
    monkeypatch.setattr(rollout_module, "set_rng_state", lambda _state: None)

    worker = object.__new__(PadFrozenRolloutWorker)
    worker.hf_model = _tiny_pad_policy()
    worker.hf_model.set_global_step(1)
    worker.cfg = SimpleNamespace()
    worker._rank = 0
    worker._world_size = 1
    worker.version = 1
    checkpoint_dir = tmp_path / "rollout"
    worker.save_checkpoint(str(checkpoint_dir), step=1)

    payload = torch.load(
        checkpoint_dir / "rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["schema"] == PAD_FROZEN_CHECKPOINT_SCHEMA
    assert payload["owner"] == "rollout"
    payload["schema"] = "fastwam-adaptive-rollout-runtime-v1"
    torch.save(payload, checkpoint_dir / "rank_0.pt")

    with pytest.raises(ValueError, match="Unsupported PAD-Frozen"):
        worker.load_checkpoint(str(checkpoint_dir))


def test_pad_runtime_targets_subclass_legacy_interfaces() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.env import PadFrozenEnvWorker
    from rlinf.models.embodiment.wam_policy.pad_rv.rollout import (
        PadFrozenRolloutWorker,
    )
    from rlinf.models.embodiment.wam_policy.pad_rv.runner import PadFrozenRunner
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    assert issubclass(PadFrozenRunner, EmbodiedRunner)
    assert issubclass(PadFrozenRolloutWorker, MultiStepRolloutWorker)
    assert issubclass(PadFrozenEnvWorker, EnvWorker)


def test_pad_actor_seeds_fresh_genesis_before_model_construction(monkeypatch) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv import actor as actor_module
    from rlinf.models.embodiment.wam_policy.pad_rv.actor import PadFrozenFSDPActor
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    events = []

    def fake_seed(seed):
        events.append(("seed", seed))
        return seed

    def fake_init_worker(_self):
        events.append(("model", None))

    monkeypatch.setattr(actor_module, "seed_everything", fake_seed)
    monkeypatch.setattr(EmbodiedFSDPActor, "init_worker", fake_init_worker)
    worker = object.__new__(PadFrozenFSDPActor)
    worker.cfg = SimpleNamespace(actor=SimpleNamespace(seed=11))
    worker._rank = 0

    worker.init_worker()

    assert events == [("seed", 11), ("model", None)]


def test_pad_env_releases_host_pages_only_after_trajectory_send(monkeypatch) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv import env as env_module
    from rlinf.models.embodiment.wam_policy.pad_rv.env import PadFrozenEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker

    events = []

    async def fake_send(_self, rollout_result, channel, *, stage_id):
        events.append(("send", rollout_result, channel, stage_id))

    def fake_release(**kwargs):
        events.append(("release", kwargs))
        return {"status": "PASS"}

    monkeypatch.setattr(EnvWorker, "send_rollout_trajectories", fake_send)
    monkeypatch.setattr(env_module, "release_pad_host_memory", fake_release)
    worker = object.__new__(PadFrozenEnvWorker)
    worker.cfg = SimpleNamespace(
        pad_rv_implementation=SimpleNamespace(
            trajectory_send_mode="concurrent",
            release_host_memory_after_trajectory_send=True,
        )
    )
    worker._rank = 3

    asyncio.run(worker.send_rollout_trajectories("trajectory", "channel", stage_id=0))

    assert events[0] == ("send", "trajectory", "channel", 0)
    assert events[1] == (
        "release",
        {
            "schema": "pad-env-trajectory-host-memory-release-v1",
            "rank": 3,
            "phase": "post_trajectory_send",
        },
    )


def test_pad_env_serializes_large_trajectory_sends(tmp_path, monkeypatch) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv import env as env_module
    from rlinf.models.embodiment.wam_policy.pad_rv.env import PadFrozenEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker

    active_sends = 0
    maximum_active_sends = 0
    sent_ranks = []

    async def fake_send(worker, _rollout_result, _channel, *, stage_id):
        nonlocal active_sends, maximum_active_sends
        active_sends += 1
        maximum_active_sends = max(maximum_active_sends, active_sends)
        await asyncio.sleep(0.02)
        sent_ranks.append((worker._rank, stage_id))
        active_sends -= 1

    monkeypatch.setattr(EnvWorker, "send_rollout_trajectories", fake_send)
    monkeypatch.setattr(
        env_module,
        "release_pad_host_memory",
        lambda **_kwargs: {"status": "PASS"},
    )

    workers = []
    for rank in range(2):
        worker = object.__new__(PadFrozenEnvWorker)
        worker.cfg = SimpleNamespace(
            pad_rv_implementation=SimpleNamespace(
                trajectory_send_mode="serialized",
                release_host_memory_after_trajectory_send=True,
            ),
            runner=SimpleNamespace(
                logger=SimpleNamespace(log_path=str(tmp_path)),
            ),
        )
        worker._rank = rank
        workers.append(worker)

    async def run_sends() -> None:
        await asyncio.gather(
            *(
                worker.send_rollout_trajectories(
                    f"trajectory-{worker._rank}",
                    "channel",
                    stage_id=0,
                )
                for worker in workers
            )
        )

    asyncio.run(run_sends())

    assert maximum_active_sends == 1
    assert sorted(sent_ranks) == [(0, 0), (1, 0)]


def test_pad_actor_releases_host_pages_after_batch_assembly(monkeypatch) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv import actor as actor_module
    from rlinf.models.embodiment.wam_policy.pad_rv.actor import PadFrozenFSDPActor
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    events = []

    async def fake_receive(_self, input_channel):
        events.append(("receive", input_channel))

    def fake_release(**kwargs):
        events.append(("release", kwargs))
        return {"status": "PASS"}

    monkeypatch.setattr(EmbodiedFSDPActor, "recv_rollout_trajectories", fake_receive)
    monkeypatch.setattr(actor_module, "release_pad_host_memory", fake_release)
    worker = object.__new__(PadFrozenFSDPActor)
    worker.cfg = SimpleNamespace(
        pad_rv_implementation=SimpleNamespace(
            release_host_memory_after_trajectory_receive=True
        )
    )
    worker._rank = 0

    asyncio.run(worker.recv_rollout_trajectories("actor-channel"))

    assert events == [
        ("receive", "actor-channel"),
        (
            "release",
            {
                "schema": "pad-actor-trajectory-host-memory-release-v1",
                "rank": 0,
                "phase": "post_trajectory_receive",
            },
        ),
    ]


def test_pad_actor_releases_consumed_batch_before_next_receive(monkeypatch) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv import actor as actor_module
    from rlinf.models.embodiment.wam_policy.pad_rv.actor import PadFrozenFSDPActor

    events = []

    def fake_release(**kwargs):
        events.append(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(actor_module, "release_pad_host_memory", fake_release)
    worker = object.__new__(PadFrozenFSDPActor)
    worker.cfg = SimpleNamespace(
        pad_rv_implementation=SimpleNamespace(
            release_host_memory_after_trajectory_receive=True
        )
    )
    worker._rank = 0
    worker.rollout_batch = {"large": torch.ones(1)}

    worker._release_consumed_rollout_batch_before_receive()

    assert worker.rollout_batch is None
    assert events == [
        {
            "schema": "pad-actor-consumed-batch-host-memory-release-v1",
            "rank": 0,
            "phase": "pre_trajectory_receive",
        }
    ]


def test_legacy_actor_consumed_batch_release_hook_is_noop() -> None:
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    worker = object.__new__(EmbodiedFSDPActor)
    sentinel = {"legacy": torch.ones(1)}
    worker.rollout_batch = sentinel

    worker._release_consumed_rollout_batch_before_receive()

    assert worker.rollout_batch is sentinel


def test_legacy_env_policy_metadata_hook_preserves_flow_audit() -> None:
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker.model_cfg = OmegaConf.create(
        {
            "runtime": {"num_inference_steps": 10},
            "flow_sde": {"ignore_last_transition": False},
        }
    )
    streams = []
    result = SimpleNamespace(forward_inputs={"denoise_indices": torch.tensor([0, 1])})

    worker._record_fastwam_training_policy_metadata(result, streams)
    report = worker._build_fastwam_training_policy_metadata_audit(
        streams=streams,
        traces=[object()],
        environment_count=2,
        global_environment_offset=4,
    )

    assert [
        item["global_environment_index"]
        for item in report["denoise_index_stream_sha256_by_global_environment"]
    ] == [4, 5]
    assert report["flow_sde_denoise_indices"]["selected_count"] == 2


def test_pad_env_policy_metadata_hook_requires_zero_flow() -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.env import PadFrozenEnvWorker

    worker = object.__new__(PadFrozenEnvWorker)
    streams = []
    worker._record_fastwam_training_policy_metadata(
        SimpleNamespace(forward_inputs={}), streams
    )
    report = worker._build_fastwam_training_policy_metadata_audit(
        streams=streams,
        traces=[object()],
        environment_count=2,
        global_environment_offset=0,
    )

    assert report == {
        "flow_sde_enabled": False,
        "denoise_index_stream_sha256_by_global_environment": [],
        "flow_sde_denoise_indices": None,
    }
    with pytest.raises(ValueError, match="cannot collect Flow-SDE"):
        worker._record_fastwam_training_policy_metadata(
            SimpleNamespace(forward_inputs={"denoise_indices": torch.tensor([0])}),
            streams,
        )


def test_pad_env_failure_audit_omits_only_flow_index_requirement(monkeypatch) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv import env as pad_env_module
    from rlinf.models.embodiment.wam_policy.pad_rv.env import PadFrozenEnvWorker
    from rlinf.workers.env import env_worker as env_worker_module
    from rlinf.workers.env.env_worker import EnvWorker

    monkeypatch.setattr(
        env_worker_module,
        "build_fastwam_action_failure_audit",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        pad_env_module,
        "build_fastwam_action_failure_audit",
        lambda **kwargs: kwargs,
    )

    legacy_report = object.__new__(
        EnvWorker
    )._build_fastwam_training_action_failure_audit(policy="legacy-flow")
    pad_report = object.__new__(
        PadFrozenEnvWorker
    )._build_fastwam_training_action_failure_audit(policy="static-merged-u")

    assert legacy_report == {
        "policy": "legacy-flow",
        "require_uncond_denoise_index": True,
    }
    assert pad_report == {
        "policy": "static-merged-u",
        "require_uncond_denoise_index": False,
    }


def test_pad_text_cache_preflight_covers_selected_task(monkeypatch, tmp_path) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.preflight import (
        validate_pad_text_cache_coverage,
    )

    prompt_template = "robot instruction: {task}"
    instruction = "put the mug on the plate"
    digest = (
        __import__("hashlib")
        .sha256(prompt_template.format(task=instruction).encode("utf-8"))
        .hexdigest()
    )
    (tmp_path / f"{digest}.t5_len128.wan22ti2v5b.pt").touch()

    class _Suite:
        n_tasks = 2

        def get_task(self, task_id):
            assert task_id == 1
            return SimpleNamespace(language=instruction)

    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_10": _Suite})
    libero_package = types.ModuleType("libero")
    libero_submodule = types.ModuleType("libero.libero")
    libero_submodule.benchmark = benchmark
    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_submodule)
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": False, "val_check_interval": -1},
            "env": {
                "train": {
                    "task_suite_name": "libero_10",
                    "task_id_filter": [1],
                }
            },
            "actor": {
                "model": {
                    "runtime": {
                        "prompt_template": prompt_template,
                        "text_embedding_cache_dir": str(tmp_path),
                        "text_embedding_context_len": 128,
                    }
                }
            },
        }
    )

    report = validate_pad_text_cache_coverage(cfg)

    assert report["status"] == "PASS"
    assert report["required_prompt_count"] == 1
    assert report["prompt_sha256"] == [digest]


def test_pad_text_cache_preflight_uses_rollout_for_eval(monkeypatch, tmp_path) -> None:
    from rlinf.models.embodiment.wam_policy.pad_rv.preflight import (
        validate_pad_text_cache_coverage,
    )

    prompt_template = "robot instruction: {task}"
    instruction = "pick up the bowl"
    digest = (
        __import__("hashlib")
        .sha256(prompt_template.format(task=instruction).encode("utf-8"))
        .hexdigest()
    )
    (tmp_path / f"{digest}.t5_len128.wan22ti2v5b.pt").touch()

    class _Suite:
        n_tasks = 1

        def get_task(self, task_id):
            assert task_id == 0
            return SimpleNamespace(language=instruction)

    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_10": _Suite})
    libero_package = types.ModuleType("libero")
    libero_submodule = types.ModuleType("libero.libero")
    libero_submodule.benchmark = benchmark
    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_submodule)
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": True, "val_check_interval": -1},
            "env": {
                "eval": {
                    "task_suite_name": "libero_10",
                    "task_id_filter": [0],
                }
            },
            "rollout": {
                "model": {
                    "runtime": {
                        "prompt_template": prompt_template,
                        "text_embedding_cache_dir": str(tmp_path),
                        "text_embedding_context_len": 128,
                    }
                }
            },
        }
    )

    report = validate_pad_text_cache_coverage(cfg)

    assert report["status"] == "PASS"
    assert report["required_prompt_count"] == 1
    assert report["prompt_sha256"] == [digest]
