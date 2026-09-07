# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Configuration boundary for route-neutral Gate + trainable UNCOND."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf, open_dict

from rlinf.config_contracts import validate_libero_terminal_reward_config
from rlinf.models.embodiment.wam_policy.evaluation import EvaluationRoutingConfig
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import OnlineIDMBCConfig
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_budget import (
    PAD_WARMUP_DAMPED_CONTROLLER_TYPE,
    PadCriticWarmupReversalDampedController,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_contracts import (
    PadCriticWarmupConfig,
    RouteNeutralGateInputContract,
)

BUILDER_TARGET = (
    "rlinf.models.embodiment.wam_policy.route_neutral_online.builder."
    "build_route_neutral_online_idm_bc_model"
)
RUNTIME_TARGET = (
    "rlinf.models.embodiment.wam_policy.route_neutral_online.runtime."
    "RouteNeutralOnlineIDMTeacherLiberoRuntime"
)
POLICY_TARGET = (
    "rlinf.models.embodiment.wam_policy.route_neutral_online.policy."
    "RouteNeutralOnlineIDMBCFastWAMPolicy"
)
ACTOR_TARGET = (
    "rlinf.models.embodiment.wam_policy.route_neutral_online.actor."
    "RouteNeutralOnlineIDMBCFSDPActor"
)
ROLLOUT_WORKER_TARGET = (
    "rlinf.models.embodiment.wam_policy.route_neutral_online.lifecycle."
    "RouteNeutralOnlineRolloutWorker"
)
ENV_WORKER_TARGET = (
    "rlinf.models.embodiment.wam_policy.route_neutral_online.lifecycle."
    "RouteNeutralOnlineEnvWorker"
)
RUNNER_TARGET = (
    "rlinf.models.embodiment.wam_policy.route_neutral_online.lifecycle."
    "RouteNeutralOnlineRunner"
)


def validate_route_neutral_online_idm_bc_training_config(
    cfg: Any,
    *,
    only_eval: bool = False,
) -> OnlineIDMBCConfig:
    """Validate the additive profile before allocating pretrained parents."""

    if only_eval:
        model = OmegaConf.select(cfg, "rollout.model")
        expected = {
            "builder_target": BUILDER_TARGET,
            "policy_target": POLICY_TARGET,
            "runtime._target_": RUNTIME_TARGET,
        }
        for field, value in expected.items():
            actual = str(OmegaConf.select(model, field))
            if actual != value:
                raise ValueError(
                    f"Route-neutral rollout.model.{field} must be {value}, "
                    f"got {actual}."
                )
        profile = OmegaConf.to_container(
            OmegaConf.select(model, "route_neutral_online"), resolve=True
        )
        if not isinstance(profile, Mapping):
            raise TypeError("Route-neutral model profile must resolve to a mapping.")
        RouteNeutralGateInputContract.from_mapping(
            profile["input_contract"],
            state_dim=int(OmegaConf.select(model, "fastwam.proprio_dim")),
        )
        visual = profile["visual"]
        if tuple(visual.get("sources", ())) != ("current_frame_video",):
            raise ValueError("Route-neutral visual sources changed.")
        layer_indices = tuple(int(v) for v in visual.get("layer_indices", ()))
        if layer_indices != (14, 15, 16, 17, 18, 19):
            raise ValueError("Route-neutral Gate must use MoT layers 15--20.")
        if bool(model.gate.current_mode_embedding) or bool(
            model.gate.denoise_timestep_embedding
        ):
            raise ValueError("Route-neutral Gate excludes mode/timestep embeddings.")
        if bool(model.get("decision_telemetry_enabled", False)):
            raise ValueError("Route-neutral Gate disables legacy mode-flip telemetry.")
        if str(model.kv_replay.backend) != "recompute":
            raise ValueError(
                "Route-neutral orchestration requires no K/V handle store."
            )
        if model.kv_replay.get("gate_kv_sample_budget") is not None:
            raise ValueError("Route-neutral replay cannot sample Action K/V.")
        if model.get("uncond_lora") is None:
            raise ValueError("Trainable UNCOND requires a dynamic LoRA adapter.")

        random_probability = model.get("eval_random_idm_probability", None)
        random_autocorrelation = model.get("eval_random_lag1_autocorrelation", None)
        EvaluationRoutingConfig(
            mode=str(model.get("eval_routing_mode", "learned_threshold")),
            idm_threshold=float(model.get("eval_idm_threshold", 0.5)),
            random_idm_probability=(
                None if random_probability is None else float(random_probability)
            ),
            random_lag1_autocorrelation=(
                None
                if random_autocorrelation is None
                else float(random_autocorrelation)
            ),
            periodic_period=model.get("eval_period", None),
            periodic_on_count=model.get("eval_periodic_on_count", None),
            periodic_phase=model.get("eval_periodic_phase", None),
            routing_seed=model.get("eval_routing_seed", 0),
        )
        from rlinf.runners.fastwam_budget_calibration import (
            validate_fastwam_budget_evaluation_config,
        )

        validate_fastwam_budget_evaluation_config(cfg)
        for split_name in ("train", "eval"):
            split_cfg = cfg.env.get(split_name, None)
            if split_cfg is not None:
                validate_libero_terminal_reward_config(
                    ignore_terminations=bool(
                        split_cfg.get("ignore_terminations", False)
                    ),
                    use_rel_reward=bool(split_cfg.get("use_rel_reward", True)),
                )
        load_for_eval = bool(model.critic.get("load_for_eval", False))
        if load_for_eval:
            raise ValueError(
                "Route-neutral Plus evaluation does not load the training critic."
            )
        with open_dict(model):
            model.eval_without_critic = True
        return OnlineIDMBCConfig.from_mapping(profile["online_idm_bc"])
    online = OnlineIDMBCConfig.from_mapping(
        OmegaConf.select(cfg, "algorithm.uncond_idm_bc")
    )
    if not online.enabled:
        raise ValueError("Route-neutral trainable UNCOND requires online BC enabled.")
    expected = {
        "route_neutral_online_implementation.actor_target": ACTOR_TARGET,
        "route_neutral_online_implementation.policy_target": POLICY_TARGET,
        "route_neutral_online_implementation.rollout_worker_target": (
            ROLLOUT_WORKER_TARGET
        ),
        "route_neutral_online_implementation.env_worker_target": ENV_WORKER_TARGET,
        "route_neutral_online_implementation.runner_target": RUNNER_TARGET,
        "actor.model.builder_target": BUILDER_TARGET,
        "rollout.model.builder_target": BUILDER_TARGET,
        "actor.model.runtime._target_": RUNTIME_TARGET,
        "rollout.model.runtime._target_": RUNTIME_TARGET,
    }
    for field, value in expected.items():
        actual = str(OmegaConf.select(cfg, field))
        if actual != value:
            raise ValueError(f"Route-neutral {field} must be {value}, got {actual}.")

    actor_profile = OmegaConf.select(cfg, "actor.model.route_neutral_online")
    rollout_profile = OmegaConf.select(cfg, "rollout.model.route_neutral_online")
    actor_resolved = OmegaConf.to_container(actor_profile, resolve=True)
    rollout_resolved = OmegaConf.to_container(rollout_profile, resolve=True)
    if actor_resolved != rollout_resolved:
        raise ValueError("Actor and rollout route-neutral profiles differ.")
    if not isinstance(actor_resolved, Mapping):
        raise TypeError("Route-neutral model profile must resolve to a mapping.")
    state_dim = int(OmegaConf.select(cfg, "actor.model.fastwam.proprio_dim"))
    RouteNeutralGateInputContract.from_mapping(
        actor_resolved["input_contract"],
        state_dim=state_dim,
    )
    warmup = PadCriticWarmupConfig.from_mapping(actor_resolved["critic_warmup"])
    model_online = OnlineIDMBCConfig.from_mapping(actor_resolved["online_idm_bc"])
    if model_online != online:
        raise ValueError("Model and algorithm online-BC contracts differ.")

    visual = actor_resolved["visual"]
    if tuple(visual.get("sources", ())) != ("current_frame_video",):
        raise ValueError("Route-neutral visual sources changed.")
    layer_indices = tuple(int(v) for v in visual.get("layer_indices", ()))
    if layer_indices != (14, 15, 16, 17, 18, 19):
        raise ValueError("Route-neutral Gate must use MoT layers 15--20.")
    for owner in ("actor", "rollout"):
        model = OmegaConf.select(cfg, f"{owner}.model")
        if bool(model.gate.current_mode_embedding) or bool(
            model.gate.denoise_timestep_embedding
        ):
            raise ValueError("Route-neutral Gate excludes mode/timestep embeddings.")
        if bool(model.get("decision_telemetry_enabled", False)):
            raise ValueError("Route-neutral Gate disables legacy mode-flip telemetry.")
        if str(model.kv_replay.backend) != "recompute":
            raise ValueError(
                "Route-neutral orchestration requires no K/V handle store."
            )
        if model.kv_replay.get("gate_kv_sample_budget") is not None:
            raise ValueError("Route-neutral replay cannot sample Action K/V.")
        if not bool(model.flow_sde.enabled) or float(model.flow_sde.noise_level) <= 0:
            raise ValueError("Trainable UNCOND requires Flow-SDE replay.")
        if model.get("uncond_lora") is None:
            raise ValueError("Trainable UNCOND requires a dynamic LoRA adapter.")

    if str(OmegaConf.select(cfg, "algorithm.loss_type")) != "fastwam_dual_ppo":
        raise ValueError("Route-neutral trainable UNCOND requires dual PPO.")
    if float(OmegaConf.select(cfg, "algorithm.uncond_flow_ppo.loss_weight")) <= 0:
        raise ValueError("UNCOND Flow-PPO must remain enabled.")
    if int(OmegaConf.select(cfg, "actor.optim.critic_warmup_steps")) != 0:
        raise ValueError("Use runner-update warm-up, not optimizer reconstruction.")
    controller = OmegaConf.select(cfg, "algorithm.fixed_branch_cost.controller")
    if not isinstance(controller, Mapping) or str(controller.get("type")) != (
        PAD_WARMUP_DAMPED_CONTROLLER_TYPE
    ):
        raise ValueError("Route-neutral profile requires warm-up damped band control.")
    controller_warmup = PadCriticWarmupConfig.from_mapping(
        controller.get("critic_warmup")
    )
    if controller_warmup != warmup:
        raise ValueError("Policy and branch-controller warm-up contracts differ.")
    resolved_controller = OmegaConf.to_container(controller, resolve=True)
    if not isinstance(resolved_controller, Mapping):
        raise TypeError("Branch controller must resolve to a mapping.")
    PadCriticWarmupReversalDampedController(resolved_controller)

    for field in (
        "runner.bootstrap_uncond_lora_sidecar",
        "runner.bootstrap_uncond_lora_sidecar_sha256",
    ):
        if not str(OmegaConf.select(cfg, field, default="") or "").strip():
            raise ValueError(f"BC-initialized UNCOND requires {field}.")
    lifecycle = OmegaConf.select(cfg, "route_neutral_online_implementation")
    if str(lifecycle.get("rollout_init_mode")) != "serial_rank":
        raise ValueError("Route-neutral rollout initialization must be serial_rank.")
    if str(lifecycle.get("trajectory_send_mode")) not in {
        "concurrent",
        "serialized",
    }:
        raise ValueError("Route-neutral trajectory send mode is invalid.")
    for field in (
        "release_host_memory_after_rollout_init",
        "release_host_memory_after_trajectory_send",
        "release_host_memory_after_trajectory_receive",
        "release_host_memory_after_train_preparation",
        "consume_rollout_batch_during_train_preparation",
    ):
        if not bool(lifecycle.get(field, False)):
            raise ValueError(f"Route-neutral lifecycle requires {field}=true.")
    placement = OmegaConf.to_container(
        OmegaConf.select(cfg, "cluster.component_placement"), resolve=True
    )
    supported_placements = (
        {"actor": "0-0", "env": "1-5", "rollout": "1-5"},
        {"actor": "0-0", "env": "1-6", "rollout": "1-6"},
        {"actor": "0-0", "env": "1-7", "rollout": "1-7"},
    )
    shared_rank = lifecycle.get("shared_gpu_rollout_rank")
    shared_placement = {"actor": "0-0", "env": "0-3,5-7", "rollout": "0-3,5-7"}
    balanced_tasks = cfg.env.train.get("task_sampling") == "global_balanced"
    if cfg.algorithm.gate_ppo.get("entropy_loss_source", "behavior") not in {
        "base",
        "behavior",
    }:
        raise ValueError("Gate entropy loss source must be base or behavior.")
    balanced_placement = {"actor": "0-0", "env": "0-6", "rollout": "0-6"}
    total_envs = int(cfg.env.train.total_num_envs)
    batch_size = int(cfg.actor.global_batch_size)
    balanced_geometry = (
        balanced_tasks
        and placement == balanced_placement
        and total_envs >= 42
        and total_envs % 42 == 0
        and batch_size == 196 * (total_envs // 42)
        and int(cfg.actor.micro_batch_size) == 4
    )
    if balanced_tasks:
        if not balanced_geometry or shared_rank != 0:
            raise ValueError(
                "Balanced ten-task training requires shared GPU0, env=rollout=0-6, "
                "N a positive multiple of 42, global batch 196*N/42 and microbatch 4."
            )
        train = cfg.env.train
        if (
            str(train.task_suite_name) != "libero_10"
            or list(train.task_id_filter) != list(range(10))
            or int(train.group_size) != 1
            or bool(train.auto_reset)
            or bool(train.ignore_terminations)
            or not bool(train.use_fixed_reset_state_ids)
            or bool(train.use_ordered_reset_state_ids)
            or train.get("specific_reset_id") is not None
            or bool(train.get("stage_invariant_fixed_reset_ids", False))
            or int(train.max_episode_steps) != 700
            or int(train.max_steps_per_rollout_epoch) != 700
            or int(train.reset_wait_steps) != 30
            or int(train.seed) != 42
            or int(cfg.algorithm.rollout_epoch) != 1
            or int(cfg.algorithm.update_epoch) != 1
        ):
            raise ValueError(
                "Balanced ten-task sampling requires the fixed episode protocol."
            )
    if shared_rank is not None:
        if (
            isinstance(shared_rank, bool)
            or shared_rank != 0
            or (placement != shared_placement and not balanced_geometry)
            or not bool(cfg.actor.enable_offload)
            or bool(cfg.rollout.enable_offload)
            or bool(cfg.runner.get("use_training_pipeline", False))
            or bool(cfg.runner.get("overlap_env_bootstrap", False))
            or int(cfg.rollout.pipeline_stage_num) != 1
            or (not balanced_geometry and (total_envs != 28 or batch_size != 196))
        ):
            raise ValueError(
                "Shared-GPU route-neutral training requires actor=0-0, "
                "env=rollout=0-3,5-7, shared rollout rank 0, actor offload, "
                "rank-local rollout offload, synchronous 28-env/global-batch-196 "
                "training without environment bootstrap overlap."
            )
    elif placement not in supported_placements:
        raise ValueError(
            "Route-neutral training placement must be one of "
            f"{supported_placements}, got {placement}."
        )
    total_num_envs = int(OmegaConf.select(cfg, "env.train.total_num_envs"))
    rollout_steps = int(OmegaConf.select(cfg, "env.train.max_steps_per_rollout_epoch"))
    execution_horizon = int(
        OmegaConf.select(cfg, "actor.model.runtime.execution_horizon")
    )
    if rollout_steps % execution_horizon != 0:
        raise ValueError(
            "Route-neutral rollout steps must be divisible by the execution horizon."
        )
    rollout_size = total_num_envs * (rollout_steps // execution_horizon)
    global_batch_size = int(OmegaConf.select(cfg, "actor.global_batch_size"))
    micro_batch_size = int(OmegaConf.select(cfg, "actor.micro_batch_size"))
    if micro_batch_size not in {1, 4}:
        raise ValueError("Route-neutral actor microbatch must be either 1 or 4.")
    if global_batch_size % micro_batch_size != 0:
        raise ValueError(
            "Route-neutral global batch size must be divisible by actor microbatch."
        )
    if rollout_size % global_batch_size != 0:
        raise ValueError(
            f"Route-neutral rollout size {rollout_size} must be divisible by "
            f"actor global batch size {global_batch_size}."
        )
    if micro_batch_size == 4:
        base_kl = OmegaConf.select(
            cfg,
            "algorithm.regularization.base_uncond_kl",
        )
        if (
            bool(base_kl.get("enabled", False))
            or float(base_kl.get("coefficient", 0.0)) != 0.0
            or bool(base_kl.get("log_metric", False))
        ):
            raise ValueError(
                "Optimized route-neutral training requires disabled base UNCOND KL "
                "loss and logging."
            )
    return online


def validate_shared_gpu_device_plan(
    cfg: Any, cluster: Any, placement: Any
) -> dict | None:
    """Check actual physical worker assignments before launching model workers."""

    if cfg.route_neutral_online_implementation.get("shared_gpu_rollout_rank") is None:
        return None
    healthy_devices = (
        list(range(7))
        if str(cfg.cluster.component_placement.env) == "0-6"
        else [0, 1, 2, 3, 5, 6, 7]
    )
    report = {"schema": "route-neutral-shared-gpu-device-plan-v1", "status": "PASS"}
    for component in ("actor", "rollout", "env"):
        planned = placement.get_strategy(component).get_placement(cluster)
        expected = [0] if component == "actor" else healthy_devices
        actual = [
            (item.rank, item.cluster_node_rank, item.visible_accelerators)
            for item in planned
        ]
        required = [(rank, 0, [str(device)]) for rank, device in enumerate(expected)]
        if actual != required:
            raise ValueError(
                f"Shared-GPU physical {component} placement changed: {actual}."
            )
        report[component] = actual
    return report


__all__ = [
    "ACTOR_TARGET",
    "BUILDER_TARGET",
    "POLICY_TARGET",
    "ENV_WORKER_TARGET",
    "ROLLOUT_WORKER_TARGET",
    "RUNNER_TARGET",
    "RUNTIME_TARGET",
    "validate_route_neutral_online_idm_bc_training_config",
]
