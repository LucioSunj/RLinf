# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Standalone single-robot validation; old LIBERO validators stay strict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from rlinf.envs.pad_realworld.action_codec import ActionCodec
from rlinf.envs.pad_realworld.contracts import RealActionProtocol
from rlinf.envs.pad_realworld.observation import ObservationAdapter


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a complete static Hydra/OmegaConf profile, listing missing live fields."""
    config = OmegaConf.load(path)
    missing = sorted(OmegaConf.missing_keys(config))
    if missing:
        raise ValueError("Missing required configuration: " + ", ".join(missing))
    result = OmegaConf.to_container(config, resolve=True)
    validate_config(result)
    return result


def action_protocol(cfg: dict[str, Any]) -> RealActionProtocol:
    execution = cfg["execution"]
    return RealActionProtocol(
        generation_horizon=execution["generation_horizon"],
        execution_horizon=execution["execution_horizon"],
        prediction_video_frames=execution["prediction_video_frames"],
        sample_period=1.0 / execution["command_hz"],
        video_offsets=tuple(execution["video_offsets"]),
    )


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate scientific invariants and I/O without constructing a model/driver."""
    required = {
        "profile": "real_robot_pad",
        "execution.mode": "synchronous",
        "execution.auto_reset": False,
        "execution.generation_horizon": 32,
        "execution.execution_horizon": 10,
        "execution.prediction_video_frames": 9,
        "collection.physical_num_envs": 1,
        "collection.complete_episodes": True,
        "auxiliary_inference.teacher_timing": "after_rollout_batch",
        "auxiliary_inference.critic_timing": "after_rollout_batch",
        "training.critic_warmup_runner_updates": 5,
        "training.ppo_epochs": 1,
        "training.gamma": 0.99,
        "training.gae_lambda": 0.95,
        "training.ppo_clip": 0.2,
        "training.online_bc_weight": 0.2,
        "training.gate_entropy_weight": 0.01,
        "training.budget": "b50_rate_limited_reset0",
        "model.lora_rank": 16,
        "model.lora_alpha": 16,
        "model.critic_kind": "pi0_5_value_after_vlm",
        "action.translation": "meters_in_base_frame",
        "action.rotation": "Euler_xyz_increment_left",
        "action.gripper": "open_fraction",
        "action.anchor": "measured_before_each_send",
        "reward": "manual_terminal_v1",
        "evaluation.learned_mode": "learned_behavior_sampling",
    }
    for path, expected in required.items():
        actual = cfg
        for key in path.split("."):
            actual = actual[key]
        if actual != expected:
            raise ValueError(f"Real PAD requires {path}={expected!r}, got {actual!r}.")
    if cfg["backend"] not in {"mock", "franka"}:
        raise ValueError("Select mock or the explicit existing Franka binding.")
    if cfg["model"]["kind"] not in {"tiny", "fastwam"}:
        raise ValueError("Select the tiny CPU fixture or FastWAM real assets.")
    if cfg["execution"]["command_hz"] <= 0:
        raise ValueError("command_hz must be positive.")
    for key in ("optimizer_batch_size", "micro_batch_size", "total_runner_updates"):
        if not isinstance(cfg["training"][key], int) or cfg["training"][key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    if (
        cfg["collection"]["episodes_per_task_per_update"] < 1
        or not cfg["collection"]["task_ids"]
    ):
        raise ValueError("At least one complete sequential episode is required.")
    if not 0 <= cfg["evaluation"]["random_idm_probability"] <= 1:
        raise ValueError("Random routing probability must be in [0,1].")
    protocol = action_protocol(cfg)
    if protocol.video_offsets != (0, 4, 8, 12, 16, 20, 24, 28, 32):
        raise ValueError(
            "This FastWAM data profile uses action/video frequency ratio four."
        )
    observation = ObservationAdapter.from_config(cfg["observation"])
    if any(c.resize[0] % 16 or c.resize[1] % 16 for c in observation.cameras):
        raise ValueError("Model camera dimensions must be divisible by 16.")
    ActionCodec(**cfg["limits"])
    if cfg["backend"] == "franka":
        hardware = cfg["hardware"]
        for key in (
            "robot_model",
            "driver",
            "robot_ip",
            "controller_node_rank",
            "gripper_type",
            "gripper_connection",
            "camera_type",
            "camera_timeout_seconds",
            "camera_serials",
            "sdk_version",
            "contact_limits",
            "rpc_wait_semantics",
            "gripper_wait_seconds",
            "stop_capability",
            "calibration_id",
        ):
            if hardware.get(key) in (None, "", "???"):
                raise ValueError(f"Missing hardware.{key}.")
        if hardware["rpc_wait_semantics"] != "blocking_submission_no_timeout_or_cancel":
            raise ValueError("Existing Franka worker wait has no timeout/cancel API.")
        if cfg["model"]["kind"] != "fastwam":
            raise ValueError("Tiny-model fixtures cannot drive live hardware.")
    if cfg["model"]["kind"] == "fastwam":
        for key in (
            "parent_checkpoint",
            "uncond_bc_sidecar",
            "stats_path",
            "model_config",
            "critic_config",
            "text_cache",
        ):
            path = cfg["model"].get(key)
            if not path or not Path(path).exists():
                raise ValueError(f"REAL-ASSET-NOT-RUN: missing model.{key}: {path!r}.")


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_config(cfg)
    return {
        "status": "PASS",
        "backend": cfg["backend"],
        "physical_num_envs": 1,
        "model": cfg["model"]["kind"],
        "real_assets": "REAL-ASSET-NOT-RUN",
        "hardware": "HARDWARE-NOT-RUN",
        "generation_horizon": 32,
        "execution_horizon": 10,
        "teacher_horizon": 32,
        "motion_started": False,
    }
