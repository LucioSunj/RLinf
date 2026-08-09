# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Default-off construction contract for P8 DINO-guided Wan-current K/V."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from fastwam.models.wan22.visual_contracts import validate_sha256
from fastwam.runtime import (
    WAN_CURRENT_REFINEMENT_SIDECAR_TYPE,
    create_wan_current_refinement_sidecar,
    validate_wan_current_refinement_config,
)

from .p8_visual_replay import P8VisualReplayConfig

P8_SIDECAR_TYPE = WAN_CURRENT_REFINEMENT_SIDECAR_TYPE


@dataclass(frozen=True)
class P8SidecarBuild:
    """Resolved enabled components shared by actor policy and LIBERO runtime."""

    encoder: Any
    refiner: Any
    replay: P8VisualReplayConfig
    camera_ids: tuple[str, ...]
    camera_input_contract_sha256: str
    license_record_sha256: str
    fixed_cost_profile_sha256: str


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            # Keep disabled TO_BE_PINNED placeholders inert. Enabled validation
            # below resolves every required primitive through its strict parser.
            value = OmegaConf.to_container(value, resolve=False)
    except ImportError:
        pass
    if not isinstance(value, Mapping):
        raise TypeError(f"P8 `{name}` must resolve to a mapping.")
    return dict(value)


def validate_p8_sidecar_config(config: Any) -> dict[str, Any]:
    """Validate the lightweight config without loading DINO or FastWAM assets."""

    payload = _mapping(config or {"enabled": False}, name="sidecar")
    enabled = payload.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("P8 `enabled` must be a boolean.")
    sidecar_type = str(payload.get("type", P8_SIDECAR_TYPE))
    if sidecar_type != P8_SIDECAR_TYPE:
        raise ValueError(f"Unsupported P8 sidecar type {sidecar_type!r}.")
    if not enabled:
        # Crucially, disabled validation does not resolve or inspect asset fields.
        return {"enabled": False, "type": sidecar_type}

    required = {
        "type",
        "enabled",
        "compile",
        "enabled_regimes",
        "dino",
        "refiner",
        "replay",
        "camera_ids",
        "camera_input_contract_sha256",
        "license_record_sha256",
        "fixed_cost_profile_sha256",
    }
    if set(payload) != required:
        raise ValueError(
            "Invalid enabled P8 sidecar fields; "
            f"missing={sorted(required - set(payload))}, "
            f"unknown={sorted(set(payload) - required)}."
        )
    core_keys = required - {"replay", "fixed_cost_profile_sha256"}
    core = validate_wan_current_refinement_config(
        {name: payload[name] for name in core_keys}
    )
    payload.update(core)
    payload["fixed_cost_profile_sha256"] = validate_sha256(
        payload["fixed_cost_profile_sha256"],
        label="P8 fixed_cost_profile_sha256",
    )
    payload["replay"] = _mapping(payload["replay"], name="replay")
    P8VisualReplayConfig.from_mapping(payload["replay"])
    return payload


def build_p8_sidecar(
    config: Any,
    *,
    actor,
    device: torch.device | str,
    dtype: torch.dtype,
) -> P8SidecarBuild | None:
    """Build enabled P8 assets; disabled mode performs no asset access or load."""

    payload = validate_p8_sidecar_config(config)
    if not payload["enabled"]:
        return None
    core_keys = {
        "type",
        "enabled",
        "compile",
        "enabled_regimes",
        "dino",
        "refiner",
        "camera_ids",
        "camera_input_contract_sha256",
        "license_record_sha256",
    }
    fastwam_build = create_wan_current_refinement_sidecar(
        {name: payload[name] for name in core_keys},
        actor=actor,
        device=device,
        dtype=dtype,
    )
    if fastwam_build is None:  # pragma: no cover - guarded by enabled validation.
        raise RuntimeError("Enabled P8 config unexpectedly produced no sidecar.")
    return P8SidecarBuild(
        encoder=fastwam_build.encoder,
        refiner=fastwam_build.refiner,
        replay=P8VisualReplayConfig.from_mapping(payload["replay"]),
        camera_ids=fastwam_build.camera_ids,
        camera_input_contract_sha256=(fastwam_build.camera_input_contract_sha256),
        license_record_sha256=fastwam_build.license_record_sha256,
        fixed_cost_profile_sha256=payload["fixed_cost_profile_sha256"],
    )
