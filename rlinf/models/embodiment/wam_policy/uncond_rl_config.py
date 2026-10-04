# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Hydra contract for the isolated LIBERO-Long UNCOND PPO experiment."""

from __future__ import annotations

import math
from typing import Any

from omegaconf import OmegaConf

from . import _validate_exact_pi05_critic_config, _validate_flow_sde_config


def uncond_training_helper_ranks(cfg: Any) -> tuple[int, ...]:
    """Select P1 or the six native rollout ranks without changing topology."""
    ranks = tuple(cfg.actor.get("uncond_rl_execution", {}).get("helper_ranks", ()))
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks):
        raise ValueError("UNCOND helper ranks must be integers.")
    if ranks not in ((), tuple(range(6))):
        raise ValueError("UNCOND helper ranks must be [] or [0,1,2,3,4,5].")
    return ranks


def validate_uncond_rl_config(cfg: Any, *, only_eval: bool = False) -> None:
    """Reject mixed-policy/teacher settings before allocating model weights."""

    model = cfg.rollout.model if only_eval else cfg.actor.model
    expected = {
        "model_type": "fastwam_adaptive",
        "fastwam._target_": "fastwam.runtime.create_fastwam",
        "runtime._target_": "rlinf.models.embodiment.wam_policy.uncond_rl_runtime.UncondRLLiberoRuntime",
        "critic.kind": "pi0_5_value_after_vlm",
        "kv_replay.backend": "recompute",
        "eval_routing_mode": "forced_uncond",
    }
    for path, value in expected.items():
        if OmegaConf.select(model, path) != value:
            raise ValueError(f"UNCOND RL requires model.{path}={value!r}.")
    for field in ("gate", "route_neutral_online", "online_idm_bc"):
        if model.get(field) is not None:
            raise ValueError(f"UNCOND RL excludes model.{field}.")
    _validate_flow_sde_config(model.flow_sde)
    _validate_exact_pi05_critic_config(model.critic)
    for field in ("uncond_lora", "video_lora"):
        adapter = model.get(field)
        if adapter is not None and (
            float(adapter.dropout) != 0.0 or not bool(adapter.freeze_base)
        ):
            raise ValueError(
                "UNCOND PPO requires frozen base weights and LoRA dropout=0."
            )
    warmup = model.uncond_rl.critic_warmup_updates
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("critic_warmup_updates must be a non-negative integer.")
    for split in ("eval",) if only_eval else ("train", "eval"):
        env = cfg.env[split]
        if env.env_type != "libero" or env.task_suite_name != "libero_10":
            raise ValueError(
                "This UNCOND experiment is restricted to LIBERO-Long (libero_10)."
            )
        if not only_eval and list(env.task_id_filter) != list(range(10)):
            raise ValueError("LIBERO-Long UNCOND training uses all ten tasks.")
        if bool(env.get("use_step_penalty", False)) or not bool(env.use_rel_reward):
            raise ValueError("UNCOND RL requires sparse terminal-success rewards.")
    if only_eval:
        if not bool(model.get("eval_without_critic", False)) or bool(
            model.critic.load_for_eval
        ):
            raise ValueError("Pure UNCOND evaluation does not load a critic.")
        return
    helpers = uncond_training_helper_ranks(cfg)
    if helpers and (bool(cfg.actor.enable_offload) or bool(cfg.rollout.enable_offload)):
        raise ValueError(
            "UNCOND training helpers require resident actor/rollout models."
        )
    if OmegaConf.to_container(cfg.actor.model, resolve=True) != OmegaConf.to_container(
        cfg.rollout.model, resolve=True
    ):
        raise ValueError("UNCOND actor and rollout model configurations differ.")
    for field in (
        "gate_ppo",
        "uncond_idm_bc",
        "online_idm_bc",
        "fixed_branch_cost",
        "regularization",
    ):
        if cfg.algorithm.get(field) is not None:
            raise ValueError(f"UNCOND RL excludes algorithm.{field}.")
    required = {
        "algorithm.loss_type": "fastwam_uncond_ppo",
        "algorithm.adv_type": "gae",
        "algorithm.reward_type": "chunk_level",
        "algorithm.logprob_type": "chunk_level",
        "runner.use_training_pipeline": False,
        "env.train.total_num_envs": 84,
        "env.eval.total_num_envs": 84,
        "env.train.task_sampling": "global_balanced",
        "env.train.auto_reset": False,
        "env.train.ignore_terminations": False,
        "rollout.pipeline_stage_num": 1,
        "rollout.recompute_logprobs": False,
        "actor.global_batch_size": 392,
        "actor.micro_batch_size": 4,
        "actor.optim.critic_warmup_steps": 0,
        "actor.fsdp_config.use_orig_params": True,
        "actor.fsdp_config.ignore_frozen_parameters": True,
        "actor.fsdp_config.mixed_precision.param_dtype": "fp32",
        "actor.fsdp_config.mixed_precision.reduce_dtype": "fp32",
        "actor.fsdp_config.mixed_precision.cast_root_forward_inputs": False,
    }
    for path, value in required.items():
        if OmegaConf.select(cfg, path) != value:
            raise ValueError(f"UNCOND RL requires {path}={value!r}.")
    if bool(cfg.actor.get("enable_sft_co_train", False)):
        raise ValueError("UNCOND RL has no online BC or SFT loss.")
    if (
        cfg.runner.get("ckpt_path") is not None
        or cfg.runner.get("bootstrap_project_checkpoint_dir") is not None
    ):
        raise ValueError(
            "Initialize UNCOND directly from its BC sidecar, or use resume_dir."
        )
    if cfg.runner.get("resume_dir") is None:
        for field in (
            "bootstrap_uncond_lora_sidecar",
            "bootstrap_uncond_lora_sidecar_sha256",
        ):
            if not cfg.runner.get(field):
                raise ValueError(f"Fresh UNCOND RL requires runner.{field}.")
    for path in (
        "actor.optim.lora_lr",
        "actor.optim.value_lr",
        "algorithm.uncond_flow_ppo.loss_weight",
        "algorithm.critic_loss.loss_weight",
    ):
        value = float(OmegaConf.select(cfg, path))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"UNCOND RL requires positive finite {path}.")
    if bool(cfg.runner.get("fastwam_training_guard", {}).get("enabled", False)):
        raise ValueError(
            "The mixed-policy Gate/cost guard does not apply to UNCOND RL."
        )
