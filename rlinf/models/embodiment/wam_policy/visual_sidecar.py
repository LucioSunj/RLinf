# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construction and validation for the optional P6 UNCOND visual sidecar."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .visual_replay import VisualReplayConfig

P6_SIDECAR_TYPE = "dinov3_router_wan_value"


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"P6 `{name}` must be a mapping.")
    return dict(value)


def _exact_fields(
    payload: Mapping[str, Any],
    required: set[str],
    *,
    name: str,
) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        raise ValueError(
            f"Invalid P6 {name} fields; missing={missing}, unknown={unknown}."
        )


def _require(value: Any, expected: Any, *, path: str) -> None:
    if value != expected:
        raise ValueError(f"P6 requires `{path}: {expected}`, got {value!r}.")


def _positive_float(value: Any, *, path: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"P6 `{path}` must be finite and positive.")
    return result


@dataclass(frozen=True)
class BuiltUncondVisualSidecar:
    """Modules and immutable contracts injected into policy/runtime ownership."""

    encoder: nn.Module
    reader: nn.Module
    spatial_metadata: Any
    transport: Any
    asset: Any
    camera_input_contract_sha256: str
    replay: VisualReplayConfig
    optimizer: Mapping[str, Any]


@dataclass(frozen=True)
class PreflightUncondVisualSidecar:
    """Asset and geometry objects validated before large-model allocation."""

    asset: Any
    spatial_metadata: Any
    transport: Any
    memory_contract_sha256: str
    replay: VisualReplayConfig


def preflight_uncond_visual_sidecar(
    payload: Mapping[str, Any],
) -> PreflightUncondVisualSidecar | None:
    """Resolve every static P6 contract without loading weights or allocating DINO."""

    if not validate_uncond_visual_sidecar_config(payload):
        return None
    config = dict(payload)
    dino = dict(config["dino"])
    transport_config = dict(config["transport"])
    spatial = _mapping(
        dict(config["wan_value"])["spatial_metadata"],
        name="spatial_metadata",
    )
    from fastwam.models.wan22.dinov3_memory import (
        DinoV3AssetSpec,
        native_memory_contract_sha256,
    )
    from fastwam.models.wan22.visual_contracts import (
        WanValueSpatialMetadata,
        build_area_overlap_dino_wan_transport,
    )

    asset = DinoV3AssetSpec.from_mapping(
        {
            key: dino[key]
            for key in (
                "source_root",
                "source_revision",
                "model_name",
                "weights_path",
                "weights_sha256",
                "preprocess_sha256",
                "output_contract_sha256",
                "compute_dtype",
                "license_id",
            )
        }
    )
    spatial_metadata = WanValueSpatialMetadata(**spatial)
    transport = build_area_overlap_dino_wan_transport(spatial_metadata)
    expected_transport_hash = str(transport_config["contract_sha256"]).lower()
    if transport.transport_sha256 != expected_transport_hash:
        raise ValueError(
            "P6 deterministic transport hash mismatch: "
            f"expected {expected_transport_hash}, got {transport.transport_sha256}."
        )
    camera_ids = tuple(spatial_metadata.camera_order)
    memory_hash = native_memory_contract_sha256(
        asset,
        camera_ids=camera_ids,
        input_contract_sha256=dino["camera_input_contract_sha256"],
    )
    return PreflightUncondVisualSidecar(
        asset=asset,
        spatial_metadata=spatial_metadata,
        transport=transport,
        memory_contract_sha256=memory_hash,
        replay=VisualReplayConfig(**dict(config["replay"])),
    )


def validate_uncond_visual_sidecar_config(payload: Mapping[str, Any]) -> bool:
    """Validate the lightweight contract and return whether P6 is enabled.

    The disabled path intentionally does not inspect or resolve asset fields;
    this is what makes the v0 preset safe when DINO assets are unavailable.
    """

    config = _mapping(payload, name="uncond_visual_sidecar")
    if "enabled" not in config or not isinstance(config["enabled"], bool):
        raise ValueError("P6 `enabled` must be an explicit boolean.")
    if not config["enabled"]:
        return False
    required = {
        "enabled",
        "type",
        "enabled_regimes",
        "dispatch_before_encoder",
        "dino",
        "router",
        "transport",
        "optimizer",
        "wan_value",
        "injection",
        "replay",
    }
    _exact_fields(config, required, name="top-level")
    _require(config["type"], P6_SIDECAR_TYPE, path="type")
    _require(list(config["enabled_regimes"]), ["uncond"], path="enabled_regimes")
    _require(
        config["dispatch_before_encoder"],
        True,
        path="dispatch_before_encoder",
    )

    dino = _mapping(config["dino"], name="dino")
    _exact_fields(
        dino,
        {
            "source_root",
            "source_revision",
            "model_name",
            "weights_path",
            "weights_sha256",
            "preprocess_sha256",
            "output_contract_sha256",
            "compute_dtype",
            "license_id",
            "camera_input_contract_sha256",
            "token_kind",
            "per_camera",
            "frozen",
            "stop_gradient",
        },
        name="dino",
    )
    for key in (
        "source_root",
        "source_revision",
        "model_name",
        "weights_path",
        "weights_sha256",
        "preprocess_sha256",
        "output_contract_sha256",
        "compute_dtype",
        "license_id",
        "camera_input_contract_sha256",
    ):
        if (
            dino[key] is None
            or not str(dino[key]).strip()
            or str(dino[key]).startswith("TO_BE_")
        ):
            raise ValueError(f"P6 requires a resolved `dino.{key}`.")
    for key in (
        "weights_sha256",
        "preprocess_sha256",
        "output_contract_sha256",
        "camera_input_contract_sha256",
    ):
        digest = str(dino[key]).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"P6 `dino.{key}` must be a hexadecimal SHA256.")
    _require(dino["token_kind"], "x_norm_patchtokens", path="dino.token_kind")
    for key in ("per_camera", "frozen", "stop_gradient"):
        _require(dino[key], True, path=f"dino.{key}")

    router = _mapping(config["router"], name="router")
    _exact_fields(
        router,
        {
            "query_source",
            "query_projection",
            "query_rank",
            "temperature",
            "score_dtype",
            "per_camera_softmax",
            "camera_mass",
            "camera_mass_values",
            "invalid_camera_policy",
        },
        name="router",
    )
    _require(
        router["query_source"],
        "base_modulated_self_attn_input",
        path="router.query_source",
    )
    if router["query_projection"] not in {"low_rank", "full_linear"}:
        raise ValueError("P6 router projection must be low_rank or full_linear.")
    if router["query_projection"] == "low_rank":
        rank = router["query_rank"]
        if isinstance(rank, bool) or rank is None or int(rank) < 1:
            raise ValueError("P6 low-rank router requires positive `query_rank`.")
    elif router["query_rank"] is not None:
        raise ValueError("P6 full-linear router requires null `query_rank`.")
    _positive_float(router["temperature"], path="router.temperature")
    _require(router["score_dtype"], "float32", path="router.score_dtype")
    _require(router["per_camera_softmax"], True, path="router.per_camera_softmax")
    _require(router["camera_mass"], "fixed", path="router.camera_mass")
    camera_mass = tuple(float(item) for item in router["camera_mass_values"])
    if not camera_mass or any(
        not math.isfinite(item) or item <= 0 for item in camera_mass
    ):
        raise ValueError("P6 camera masses must be finite and positive.")
    _require(
        router["invalid_camera_policy"],
        "renormalize_active_or_fail_closed",
        path="router.invalid_camera_policy",
    )

    transport = _mapping(config["transport"], name="transport")
    _exact_fields(
        transport,
        {"mode", "contract_sha256", "preserve_row_mass", "fail_closed"},
        name="transport",
    )
    _require(
        transport["mode"],
        "deterministic_area_overlap",
        path="transport.mode",
    )
    for key in ("preserve_row_mass", "fail_closed"):
        _require(transport[key], True, path=f"transport.{key}")
    transport_hash = str(transport["contract_sha256"] or "").lower()
    if len(transport_hash) != 64 or any(
        character not in "0123456789abcdef" for character in transport_hash
    ):
        raise ValueError("P6 transport contract hash must be a hexadecimal SHA256.")

    optimizer = _mapping(config["optimizer"], name="optimizer")
    _exact_fields(
        optimizer,
        {
            "family",
            "lr",
            "weight_decay",
            "scheduler",
            "parameter_allowlist",
            "fail_on_empty_or_overlap",
        },
        name="optimizer",
    )
    _require(optimizer["family"], "visual_router", path="optimizer.family")
    _positive_float(optimizer["lr"], path="optimizer.lr")
    weight_decay = float(optimizer["weight_decay"])
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("P6 optimizer weight decay must be finite and non-negative.")
    if not str(optimizer["scheduler"]).strip():
        raise ValueError("P6 optimizer scheduler must be explicit.")
    _require(
        list(optimizer["parameter_allowlist"]),
        ["routers.*.query_projection.*", "branches.*.*.raw_beta"],
        path="optimizer.parameter_allowlist",
    )
    _require(
        optimizer["fail_on_empty_or_overlap"],
        True,
        path="optimizer.fail_on_empty_or_overlap",
    )

    wan = _mapping(config["wan_value"], name="wan_value")
    _exact_fields(
        wan,
        {
            "source",
            "flatten_order",
            "output_projection",
            "output_bias",
            "reuse_base_gate_msa",
            "spatial_metadata",
        },
        name="wan_value",
    )
    _require(wan["source"], "video_cache_current_prefix", path="wan_value.source")
    _require(wan["flatten_order"], "t_h_w_row_major", path="wan_value.flatten_order")
    _require(
        wan["output_projection"],
        "frozen_action_self_attn_o_weight_only",
        path="wan_value.output_projection",
    )
    _require(wan["output_bias"], False, path="wan_value.output_bias")
    _require(wan["reuse_base_gate_msa"], True, path="wan_value.reuse_base_gate_msa")
    spatial = _mapping(wan["spatial_metadata"], name="spatial_metadata")
    spatial_fields = {
        "wan_grid_f",
        "wan_grid_h",
        "wan_grid_w",
        "current_frame_video_tokens",
        "wan_flatten_order",
        "vae_model_type",
        "vae_weights_sha256",
        "vae_spatial_downsample_factor",
        "video_dit_weights_sha256",
        "video_dit_patch_size",
        "video_attention_num_heads",
        "video_attention_head_dim",
        "video_value_layout",
        "video_value_rope_applied",
        "camera_concat_mode",
        "camera_order",
        "per_camera_post_crop_hw",
        "per_camera_combined_rgb_box",
        "per_camera_wan_grid_support",
        "dino_patch_grid",
        "dino_preprocess_sha256",
        "invalid_mask_policy",
        "query_source",
        "query_timing",
        "residual_timing",
        "spatial_transport_contract_sha256",
    }
    _exact_fields(spatial, spatial_fields, name="spatial metadata")
    unresolved_spatial = sorted(
        key
        for key in spatial_fields
        if spatial[key] is None
        or (isinstance(spatial[key], str) and not spatial[key].strip())
    )
    if unresolved_spatial:
        raise ValueError(
            f"P6 spatial metadata contains unresolved fields: {unresolved_spatial}."
        )
    for key in (
        "vae_weights_sha256",
        "video_dit_weights_sha256",
        "dino_preprocess_sha256",
        "spatial_transport_contract_sha256",
    ):
        digest = str(spatial[key]).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"P6 spatial `{key}` must be a hexadecimal SHA256.")
    _require(
        spatial["wan_flatten_order"],
        wan["flatten_order"],
        path="wan_value.spatial_metadata.wan_flatten_order",
    )
    _require(
        spatial["dino_preprocess_sha256"],
        dino["preprocess_sha256"],
        path="wan_value.spatial_metadata.dino_preprocess_sha256",
    )
    if len(tuple(spatial["camera_order"])) != len(camera_mass):
        raise ValueError("P6 camera masses must align exactly with camera order.")

    injection = _mapping(config["injection"], name="injection")
    _exact_fields(
        injection,
        {
            "query_timing",
            "residual_timing",
            "layer_indices",
            "beta_parameterization",
            "beta_max",
            "zero_init",
            "modify_base_attention",
        },
        name="injection",
    )
    _require(injection["query_timing"], "pre_block", path="injection.query_timing")
    _require(
        injection["residual_timing"], "post_block", path="injection.residual_timing"
    )
    _require(
        spatial["query_source"],
        router["query_source"],
        path="wan_value.spatial_metadata.query_source",
    )
    _require(
        spatial["query_timing"],
        injection["query_timing"],
        path="wan_value.spatial_metadata.query_timing",
    )
    _require(
        spatial["residual_timing"],
        injection["residual_timing"],
        path="wan_value.spatial_metadata.residual_timing",
    )
    layers = tuple(int(item) for item in injection["layer_indices"])
    if not layers or layers != tuple(sorted(set(layers))) or layers[0] < 0:
        raise ValueError("P6 injection layers must be non-empty, sorted, and unique.")
    _require(
        injection["beta_parameterization"],
        "bounded_tanh",
        path="injection.beta_parameterization",
    )
    _positive_float(injection["beta_max"], path="injection.beta_max")
    _require(injection["zero_init"], "beta", path="injection.zero_init")
    _require(
        injection["modify_base_attention"],
        False,
        path="injection.modify_base_attention",
    )
    replay = _mapping(config["replay"], name="replay")
    VisualReplayConfig(**replay)
    return True


def build_uncond_visual_sidecar(
    payload: Mapping[str, Any],
    *,
    actor: nn.Module,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    encoder_factory: Callable[..., nn.Module] | None = None,
    preflight: PreflightUncondVisualSidecar | None = None,
) -> BuiltUncondVisualSidecar | None:
    """Build P6 after lightweight validation; return before any DINO load if off."""

    if not validate_uncond_visual_sidecar_config(payload):
        return None
    config = dict(payload)
    dino = dict(config["dino"])
    router = dict(config["router"])
    injection = dict(config["injection"])
    preflight = preflight or preflight_uncond_visual_sidecar(payload)
    if preflight is None:
        raise RuntimeError("Enabled P6 produced no preflight contract.")

    from fastwam.models.wan22.dinov3_memory import (
        FrozenDinoV3Encoder,
    )
    from fastwam.models.wan22.visual_sidecar import (
        DinoWanValueReaderConfig,
        ProjectionSpec,
        build_dino_wan_value_reader,
    )

    asset = preflight.asset
    spatial_metadata = preflight.spatial_metadata
    transport = preflight.transport
    runtime = getattr(actor, "video_expert", None)
    action = getattr(actor, "action_expert", None)
    mot = getattr(actor, "mot", None)
    if runtime is None or action is None or mot is None:
        raise TypeError("P6 requires FastWAM video/action experts and MoT.")
    configured_action_hidden = int(getattr(action, "hidden_dim", -1))
    if configured_action_hidden < 1:
        raise ValueError("P6 could not audit the ActionDiT hidden width.")
    if (
        int(getattr(mot, "num_heads", -1)) != spatial_metadata.video_attention_num_heads
        or int(getattr(mot, "attn_head_dim", -1))
        != spatial_metadata.video_attention_head_dim
    ):
        raise ValueError("P6 spatial value-head contract differs from the actor MoT.")
    layer_indices = tuple(int(item) for item in injection["layer_indices"])
    if layer_indices[-1] >= int(getattr(mot, "num_layers", 0)):
        raise ValueError("P6 injection layer lies outside the actor MoT.")
    camera_ids = tuple(spatial_metadata.camera_order)
    projection = ProjectionSpec(
        kind=(
            "low_rank" if router["query_projection"] == "low_rank" else "full_linear"
        ),
        rank=router["query_rank"],
    )
    reader_config = DinoWanValueReaderConfig(
        action_hidden_dim=configured_action_hidden,
        camera_ids=camera_ids,
        layer_indices=layer_indices,
        temperature=float(router["temperature"]),
        beta_max=float(injection["beta_max"]),
        query_projection=projection,
        memory_contract_sha256=preflight.memory_contract_sha256,
        spatial_metadata=spatial_metadata,
        transport=transport,
        camera_mass=tuple(float(item) for item in router["camera_mass_values"]),
    )
    reader = build_dino_wan_value_reader(reader_config)
    reader = (
        reader.to(device=device)
        if dtype is None
        else reader.to(device=device, dtype=dtype)
    )
    factory = encoder_factory or FrozenDinoV3Encoder.from_local_asset
    encoder = factory(asset, device=device)
    encoder.requires_grad_(False)
    encoder.eval()
    return BuiltUncondVisualSidecar(
        encoder=encoder,
        reader=reader,
        spatial_metadata=spatial_metadata,
        transport=transport,
        asset=asset,
        camera_input_contract_sha256=dino["camera_input_contract_sha256"],
        replay=preflight.replay,
        optimizer=dict(config["optimizer"]),
    )
