# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Inference-only FR3 sessions using the shared sampler and canonical Gate."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from fastwam.adapters import sha256_file
from fastwam.uncond_bc_checkpoint import _validate_adapter_payload
from omegaconf import OmegaConf

from .fr3_policy import build_fr3_mixed_model
from .fr3_runtime import fr3_observation_batch
from .pad_rv.route_neutral_policy import RouteNeutralRoutingState

DEPLOY_SCHEMA = "fr3-fastwam-deployment-v1"
CONTRACT_KEYS = (
    "task_name",
    "task_id",
    "prompt",
    "parent_checkpoint_sha256",
    "statistics_sha256",
    "action_dt",
    "action_horizon",
    "action_dim",
)
MODES = {"idm", "uncond_bc", "rl_mixed", "forced_idm", "forced_uncond"}


def deployment_contract(fr3: Any) -> dict[str, Any]:
    """Return immutable task/parent/processing bindings used by deployment."""
    return {key: fr3[key] for key in CONTRACT_KEYS}


def load_deployment_state(policy: Any, payload: dict, contract: dict) -> None:
    """Restore only Gate and dual adapters, rejecting crossed task assets."""
    if (
        set(payload) != {"schema", "kind", "contract", "policy_version", "state"}
        or payload["schema"] != DEPLOY_SCHEMA
    ):
        raise ValueError("Unsupported FR3 deployment artifact schema.")
    if payload["contract"] != contract:
        raise ValueError("FR3 deployment task, parent, statistics or timing differs.")
    state = payload["state"]
    if payload["kind"] == "rl":
        if set(state) != {"gate", "lora", "video_lora"}:
            raise ValueError("RL deployment requires exactly Gate and dual LoRA state.")
        policy.gate.load_state_dict(state["gate"], strict=True)
        policy.lora_adapter.load_lora_state_dict(state["lora"], strict=True)
        policy.video_lora_adapter.load_lora_state_dict(state["video_lora"], strict=True)
    elif payload["kind"] == "bc":
        if set(state) != {"action", "video"}:
            raise ValueError("BC deployment requires both adapter sidecars.")
        for branch, adapter in (
            ("action", policy.lora_adapter),
            ("video", policy.video_lora_adapter),
        ):
            _validate_adapter_payload(
                adapter,
                state[branch],
                expected_parent_checkpoint_sha256=contract["parent_checkpoint_sha256"],
            )
    elif payload["kind"] != "idm" or state:
        raise ValueError("Unsupported FR3 deployment state kind.")
    version = payload["policy_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("Deployment policy version must be a nonnegative integer.")
    policy.set_global_step(version)


class Fr3InferenceSession:
    """One task/model session; episode changes discard routing and text caches."""

    def __init__(self, policy: Any, contract: dict, *, mode: str, artifact_kind: str):
        if mode not in MODES:
            raise ValueError(f"Unsupported FR3 deployment mode {mode!r}.")
        if mode == "rl_mixed" and artifact_kind != "rl":
            raise ValueError("Learned mixed routing requires an RL Gate checkpoint.")
        if mode in {"uncond_bc", "forced_uncond"} and artifact_kind == "idm":
            raise ValueError("UNCOND inference requires trained dual adapters.")
        if mode == "uncond_bc" and artifact_kind != "bc":
            raise ValueError("UNCOND BC comparison requires a BC sidecar.")
        if policy.critic is not None:
            raise ValueError("FR3 deployment must be constructed without a critic.")
        self.policy, self.contract, self.mode = policy, dict(contract), mode
        route_mode = (
            "learned_threshold"
            if mode == "rl_mixed"
            else "forced_idm"
            if mode in {"idm", "forced_idm"}
            else "forced_uncond"
        )
        policy.config = replace(policy.config, eval_routing_mode=route_mode)
        policy.eval().requires_grad_(False)
        self._episode = None
        self._next_chunk = 0
        self.metadata = {
            "protocol": "fastwam_fr3_v1",
            "model_variant": "fastwam_idm",
            "policy_type": "fastwam_fr3_mixed",
            "policy_mode": mode,
            "policy_version": policy.actor_version,
            "task_name": contract["task_name"],
            "camera_roles": ["left", "wrist"],
            "state_dim": 8,
            "action_dim": 8,
            "action_horizon": 16,
            "num_video_frames": 5,
            "action_dt": contract["action_dt"],
            "action_space": "absolute_joint_position",
            "gripper_encoding": "closed_0_1",
            "preview_video": False,
        }

    def reset(self) -> None:
        """Clear all derived session state when a connection/model is replaced."""
        self.policy.runtime.clear_session()
        self.policy.route_tracker = RouteNeutralRoutingState(
            physical_history=self.policy.runtime.physical_history
        )
        self.policy._inference_engine = None
        self.policy._replay_text_ids.clear()
        self.policy._inactive_rollout_template = None
        self._episode, self._next_chunk = None, 0

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Generate one chunk with deterministic evaluation and no teacher/replay."""
        started = time.monotonic()
        episode, chunk = observation.get("episode_id"), observation.get("chunk_id")
        if (
            not isinstance(episode, str)
            or not episode
            or isinstance(chunk, bool)
            or not isinstance(chunk, int)
            or chunk < 0
        ):
            raise ValueError(
                "FR3 requires a nonempty episode_id and nonnegative chunk_id."
            )
        if observation.get("task") != self.contract["prompt"]:
            raise ValueError("FR3 inference prompt differs from the task-bound parent.")
        if (
            "policy_version" in observation
            and observation["policy_version"] != self.policy.actor_version
        ):
            raise ValueError(
                "FR3 request policy version differs from deployed weights."
            )
        new_episode = episode != self._episode
        if new_episode:
            if chunk != 0:
                raise ValueError("A new FR3 episode must start at chunk zero.")
            self.reset()
            self._episode = episode
        if chunk != self._next_chunk:
            raise ValueError(
                f"Expected FR3 chunk {self._next_chunk}, received {chunk}."
            )
        obs = fr3_observation_batch(observation)
        obs["_fastwam_env_ids"] = torch.tensor([0])
        obs["_fastwam_reset_mask"] = torch.tensor([new_episode])
        actions, info = self.policy.predict_action_batch(
            obs, mode="eval", compute_values=False
        )
        if info["forward_inputs"]:
            raise RuntimeError("Deployment unexpectedly produced training replay.")
        self._next_chunk += 1
        return {
            "actions": actions[0].detach().float().cpu().numpy(),
            "episode_id": episode,
            "chunk_id": chunk,
            "policy_version": self.policy.actor_version,
            "policy_mode": self.mode,
            "action_space": "absolute_joint_position",
            "route": "idm" if int(info["route_info"].route_used[0]) == 1 else "uncond",
            "gate_probability": float(info["emitted_gate"].base_probability[0]),
            "timing": {"inference_s": time.monotonic() - started},
        }


def build_fr3_inference(config: Any) -> Fr3InferenceSession:
    """Load a deployment bundle without optimizer, critic, teacher or replay."""
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    contract = deployment_contract(cfg.fr3)
    model = cfg.model
    if model.actor_checkpoint_sha256 != contract["parent_checkpoint_sha256"]:
        raise ValueError("Deployment model has a different parent checkpoint.")
    if sha256_file(model.runtime.processor_stats_path) != contract["statistics_sha256"]:
        raise ValueError("Deployment normalization statistics have changed.")
    if (
        model.runtime.task_name != contract["task_name"]
        or model.runtime.task_prompt != contract["prompt"]
        or model.runtime.action_dt != contract["action_dt"]
    ):
        raise ValueError(
            "Deployment model task/prompt/timing differs from its contract."
        )
    model.eval_without_critic = True
    payload = torch.load(Path(cfg.artifact), map_location="cpu", weights_only=True)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        str(model.precision)
    ]
    policy = build_fr3_mixed_model(model, dtype).to(torch.device(str(cfg.device)))
    load_deployment_state(policy, payload, contract)
    return Fr3InferenceSession(
        policy, contract, mode=str(cfg.policy_mode), artifact_kind=payload["kind"]
    )
