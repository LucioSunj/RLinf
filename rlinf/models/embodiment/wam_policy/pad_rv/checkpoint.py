# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Stage-owned checkpoint contract for the config-selected PAD path."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.config_contracts import build_fastwam_checkpoint_contract
from rlinf.models.embodiment.wam_policy.critic import (
    critic_parent_checkpoint_sha256,
)

from .config import PAD_ROUTE_NEUTRAL_POLICY_TARGET, PadRVStage

PAD_FROZEN_CHECKPOINT_SCHEMA = "fastwam-gate-only-frozen-pair-v1"
PAD_FROZEN_CONTRACT_SCHEMA = "fastwam-gate-only-frozen-pair-contract-v1"


def _checkpoint_sha256(value: Any, *, name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"PAD-Frozen {name} must be a SHA-256 identity.")
    return result


def pad_frozen_artifact_identities(model_cfg: Any) -> dict[str, Any]:
    """Identify the external frozen experts that are rebuilt on resume."""

    experts = model_cfg.route_action_experts
    if str(experts.get("idm_source", "")) != "parent_checkpoint":
        raise ValueError("PAD-Frozen checkpoint requires the parent IDM expert.")
    return {
        "idm_parent_checkpoint_sha256": _checkpoint_sha256(
            model_cfg.actor_checkpoint_sha256,
            name="IDM parent",
        ),
        "merged_warm_uncond": {
            "schema": "fastwam-frozen-uncond-action-v1",
            "checkpoint_sha256": _checkpoint_sha256(
                experts.get("uncond_merged_checkpoint_sha256"),
                name="merged Warm UNCOND artifact",
            ),
            "source_lora_sidecar_sha256": _checkpoint_sha256(
                experts.get("source_lora_sidecar_sha256"),
                name="merged Warm UNCOND source sidecar",
            ),
        },
        "critic_parent_checkpoint_sha256": critic_parent_checkpoint_sha256(
            model_cfg.critic
        ),
    }


def build_pad_frozen_checkpoint_contract(
    cfg: Any,
    *,
    world_size: int,
) -> dict[str, Any]:
    """Bind Stage 1 ownership and the inherited execution continuation state."""

    return {
        "schema": PAD_FROZEN_CONTRACT_SCHEMA,
        "stage": PadRVStage.FROZEN.value,
        "routing_semantics": "current_step",
        "loss_type": "fastwam_gate_only_ppo",
        "artifact_identities": pad_frozen_artifact_identities(cfg.actor.model),
        "execution": build_fastwam_checkpoint_contract(
            cfg,
            world_size=int(world_size),
        ),
    }


def validate_pad_frozen_checkpoint_contract(
    checkpoint_contract: Any,
    cfg: Any,
    *,
    world_size: int,
) -> dict[str, Any]:
    """Accept only an exact Stage 1 continuation contract."""

    expected = build_pad_frozen_checkpoint_contract(cfg, world_size=world_size)
    if checkpoint_contract != expected:
        raise ValueError(
            "PAD-Frozen checkpoint contract differs from the live Stage 1 config."
        )
    return expected


def validate_pad_frozen_eval_checkpoint(
    payload: Any,
    model_cfg: Any,
) -> dict[str, Any]:
    """Validate a Stage 1 actor checkpoint for inference-only restoration."""

    expected_keys = {
        "schema",
        "owner",
        "step",
        "optimizer_steps",
        "versions",
        "stage_contract",
        "policy",
        "optimizer",
        "lr_scheduler",
        "grad_scaler",
        "rng",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        actual = (
            sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        )
        raise ValueError(f"PAD-Frozen evaluation checkpoint keys changed: {actual}.")
    if (
        payload.get("schema") != PAD_FROZEN_CHECKPOINT_SCHEMA
        or payload.get("owner") != "actor"
    ):
        raise ValueError("Unsupported PAD-Frozen evaluation checkpoint.")
    step = int(payload.get("step", -1))
    optimizer_steps = int(payload.get("optimizer_steps", -1))
    if step < 1 or optimizer_steps < 1:
        raise ValueError("PAD-Frozen evaluation requires a trained checkpoint.")
    versions = payload.get("versions")
    route_neutral = str(model_cfg.get("policy_target", "")) == (
        PAD_ROUTE_NEUTRAL_POLICY_TARGET
    )
    if route_neutral:
        if not isinstance(versions, dict) or set(versions) != {
            "actor",
            "gate",
            "critic",
        }:
            raise ValueError("Route-neutral evaluation versions are malformed.")
        versions = {name: int(value) for name, value in versions.items()}
        if (
            versions["actor"] != step
            or versions["critic"] != optimizer_steps
            or not 0 < versions["gate"] <= versions["critic"]
        ):
            raise ValueError("Route-neutral evaluation checkpoint versions disagree.")
    elif versions != build_pad_frozen_versions(
        step=step,
        optimizer_steps=optimizer_steps,
    ):
        raise ValueError("PAD-Frozen evaluation checkpoint versions disagree.")
    contract = payload.get("stage_contract")
    if not isinstance(contract, dict):
        raise ValueError("PAD-Frozen evaluation checkpoint has no stage contract.")
    if (
        contract.get("schema") != PAD_FROZEN_CONTRACT_SCHEMA
        or contract.get("stage") != PadRVStage.FROZEN.value
        or contract.get("routing_semantics") != "current_step"
        or contract.get("loss_type") != "fastwam_gate_only_ppo"
    ):
        raise ValueError("PAD-Frozen evaluation stage contract changed.")
    expected_artifacts = pad_frozen_artifact_identities(model_cfg)
    if contract.get("artifact_identities") != expected_artifacts:
        raise ValueError("PAD-Frozen evaluation frozen-artifact identities differ.")
    policy = payload.get("policy")
    expected_policy_keys = {
        "schema",
        "actor_version",
        "gate",
        "value_head",
        "route_tracker",
    }
    if not isinstance(policy, dict) or set(policy) != expected_policy_keys:
        raise ValueError("PAD-Frozen evaluation policy payload is malformed.")
    if (
        policy.get("schema") != "pad-frozen-policy-v1"
        or int(policy.get("actor_version", -1)) != step
    ):
        raise ValueError("PAD-Frozen evaluation policy version differs from its step.")
    tensor_count = 0
    for owner in ("gate", "value_head"):
        state = policy.get(owner)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"PAD-Frozen evaluation {owner} state is empty.")
        for value in state.values():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"PAD-Frozen evaluation {owner} state is not tensor-only."
                )
            if value.is_floating_point() and not bool(
                torch.isfinite(value).all().item()
            ):
                raise ValueError(f"PAD-Frozen evaluation {owner} state is non-finite.")
            tensor_count += 1
    if tensor_count < 1:
        raise ValueError("PAD-Frozen evaluation checkpoint has no tensors.")
    return payload


def build_pad_frozen_versions(*, step: int, optimizer_steps: int) -> dict[str, int]:
    """Record the jointly updated Gate and critic versions explicitly."""

    step = int(step)
    optimizer_steps = int(optimizer_steps)
    if step < 0 or optimizer_steps < 0:
        raise ValueError("PAD-Frozen checkpoint versions must be non-negative.")
    return {
        "actor": step,
        "gate": optimizer_steps,
        "critic": optimizer_steps,
    }
