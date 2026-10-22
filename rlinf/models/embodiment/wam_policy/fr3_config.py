# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FR3 model and single-owner training contracts, without LIBERO defaults."""

from __future__ import annotations

import math
from typing import Any

from omegaconf import OmegaConf

from .pad_rv.route_neutral_contracts import (
    PadCriticWarmupConfig,
    RouteNeutralGateInputContract,
)
from .route_neutral_online.task_budget import validate_fr3_task_budget_config

RUNTIME = "rlinf.models.embodiment.wam_policy.fr3_runtime.Fr3RouteNeutralRuntime"
BUILDER = "rlinf.models.embodiment.wam_policy.fr3_policy.build_fr3_mixed_model"


def validate_fr3_model_config(model: Any) -> None:
    """Check fixed scientific settings before loading the task parent."""
    expected = {
        "builder_target": BUILDER,
        "action_dim": 8,
        "num_action_chunks": 16,
        "fastwam.proprio_dim": 8,
        "fastwam.action_dit_config.action_dim": 8,
        "fastwam.text_padding": "masked",
        "runtime._target_": RUNTIME,
        "runtime.generation_horizon": 16,
        "runtime.execution_horizon": 16,
        "runtime.num_video_frames": 5,
        "runtime.reset_wait_steps": 0,
        "runtime.binarize_gripper": False,
        "gate.current_mode_embedding": False,
        "gate.denoise_timestep_embedding": False,
        "kv_replay.backend": "recompute",
    }
    for key, value in expected.items():
        if OmegaConf.select(model, key) != value:
            raise ValueError(f"FR3 {key} must equal {value!r}.")
    for branch in ("uncond_lora", "video_lora"):
        for key, value in {
            "rank": 128,
            "alpha": 128,
            "dropout": 0,
            "freeze_base": True,
        }.items():
            if OmegaConf.select(model, f"{branch}.{key}") != value:
                raise ValueError(f"FR3 {branch}.{key} must equal {value!r}.")
    profile = model.route_neutral_online
    RouteNeutralGateInputContract.from_mapping(profile.input_contract, state_dim=8)
    if list(profile.visual.layer_indices) != [14, 15, 16, 17, 18, 19]:
        raise ValueError("FR3 Gate must use canonical parent layers 15 through 20.")
    if list(profile.visual.sources) != ["current_frame_video"]:
        raise ValueError("FR3 Gate must use current-frame parent video features.")
    if (
        model.runtime.task_name not in {"drawer", "insert_tubes"}
        or not model.runtime.task_prompt
    ):
        raise ValueError("FR3 model must bind a supported task and full prompt.")


def validate_fr3_training_config(cfg: Any, *, only_eval: bool = False) -> None:
    """Require confirmed assets, finite run limits and calibrated task budgets."""
    model = cfg.actor.model
    validate_fr3_model_config(model)
    fr3 = cfg.fr3
    identities = {"drawer": 0, "insert_tubes": 1}
    if isinstance(fr3.task_id, bool) or identities.get(fr3.task_name) != fr3.task_id:
        raise ValueError("FR3 task_id must bind drawer=0 or insert_tubes=1.")
    if not fr3.get("data_contract_confirmed", False) or not fr3.get("ready", False):
        raise ValueError(
            "FR3 pipeline is NOT_READY: confirm data contract and bind BC/budget first."
        )
    bindings = {
        "actor_checkpoint": fr3.parent_checkpoint,
        "actor_checkpoint_sha256": fr3.parent_checkpoint_sha256,
        "runtime.processor_stats_path": fr3.statistics,
        "runtime.training_config": fr3.training_config,
        "runtime.task_name": fr3.task_name,
        "runtime.task_prompt": fr3.prompt,
        "runtime.action_dt": fr3.action_dt,
    }
    for key, value in bindings.items():
        if OmegaConf.select(model, key) != value:
            raise ValueError(f"FR3 model {key} differs from its task contract.")
    if cfg.env.train.task_name != fr3.task_name or cfg.env.train.num_envs != 1:
        raise ValueError("FR3 environment must match the single task-bound policy.")
    if only_eval:
        return
    for key in (
        "updates",
        "episodes_per_update",
        "max_episode_actions",
        "checkpoint_every",
    ):
        value = fr3.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"FR3 {key} must be an explicit positive integer.")
    for key in (
        "max_episode_seconds",
        "request_timeout_s",
        "poll_interval_s",
        "action_dt",
    ):
        value = fr3.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"FR3 {key} must be positive and finite.")
    warmup = PadCriticWarmupConfig.from_mapping(
        model.route_neutral_online.critic_warmup
    )
    if warmup.runner_updates != fr3.warmup_updates:
        raise ValueError("FR3 runner and policy critic warm-up counts differ.")
    if (
        cfg.algorithm.fixed_branch_cost.controller.critic_warmup.runner_updates
        != warmup.runner_updates
    ):
        raise ValueError("FR3 controller and policy critic warm-up counts differ.")
    if not model.route_neutral_online.online_idm_bc.enabled:
        raise ValueError("FR3 mixed RL requires online IDM-to-UNCOND BC.")
    online = model.route_neutral_online.online_idm_bc
    if (
        cfg.algorithm.uncond_idm_bc.enabled != online.enabled
        or cfg.algorithm.uncond_idm_bc.loss_weight != online.loss_weight
    ):
        raise ValueError("FR3 algorithm and model online BC settings differ.")
    if (
        cfg.algorithm.gate_ppo.entropy_metric_source != "base"
        or cfg.algorithm.gate_ppo.entropy_loss_source != "base"
    ):
        raise ValueError("FR3 Gate PPO requires base entropy.")
    epoch = cfg.algorithm.update_epoch
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("FR3 update_epoch must be a positive integer.")
    seed = model.get("formal_training_sampling_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            "FR3 formal_training_sampling_seed must be a nonnegative integer."
        )
    validate_fr3_task_budget_config(cfg)
