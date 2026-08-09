from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

OUTER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OUTER / "FastWAM/src"))

from fastwam.models.wan22.dinov3_memory import (  # noqa: E402
    DINO_V3_OUTPUT_CONTRACT_SHA256,
    DINO_V3_PREPROCESS_SHA256,
    PINNED_DINOV3_MODEL_NAME,
    PINNED_DINOV3_SOURCE_REVISION,
)
from fastwam.models.wan22.visual_contracts import (  # noqa: E402
    WAN_FLATTEN_ORDER,
    WAN_VIDEO_VALUE_LAYOUT,
    NativePatchMemory,
    PreparedCameraBatch,
    WanValueSpatialMetadata,
    build_area_overlap_dino_wan_transport,
)


def _load_modules():
    repo = Path(__file__).resolve().parents[2]
    package_name = "fastwam_p6_sidecar_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(repo / "rlinf/models/embodiment/wam_policy")]
    sys.modules[package_name] = package
    for name in ("visual_replay", "visual_sidecar"):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name,
            repo / f"rlinf/models/embodiment/wam_policy/{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return (
        sys.modules[f"{package_name}.visual_replay"],
        sys.modules[f"{package_name}.visual_sidecar"],
    )


_replay, _sidecar = _load_modules()


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _metadata() -> WanValueSpatialMetadata:
    return WanValueSpatialMetadata(
        wan_grid_f=1,
        wan_grid_h=2,
        wan_grid_w=4,
        current_frame_video_tokens=8,
        wan_flatten_order=WAN_FLATTEN_ORDER,
        vae_model_type="WanVideoVAE38",
        vae_weights_sha256=_hash("vae"),
        vae_spatial_downsample_factor=16,
        video_dit_weights_sha256=_hash("video"),
        video_dit_patch_size=(1, 2, 2),
        video_attention_num_heads=2,
        video_attention_head_dim=4,
        video_value_layout=WAN_VIDEO_VALUE_LAYOUT,
        video_value_rope_applied=False,
        camera_concat_mode="horizontal",
        camera_order=("main", "wrist"),
        per_camera_post_crop_hw=((224, 224), (224, 224)),
        per_camera_combined_rgb_box=((0, 0, 224, 224), (0, 224, 224, 448)),
        per_camera_wan_grid_support=((0, 2, 0, 2), (0, 2, 2, 4)),
        dino_patch_grid=(14, 14),
        dino_preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        invalid_mask_policy="renormalize_active_or_fail_closed",
    )


def _camera_batch(batch: int = 1) -> PreparedCameraBatch:
    return PreparedCameraBatch(
        pixels=torch.randint(0, 256, (batch, 2, 3, 224, 224), dtype=torch.uint8),
        camera_ids=("main", "wrist"),
        camera_valid_mask=torch.ones(batch, 2, dtype=torch.bool),
        input_contract_sha256=_hash("camera"),
    )


def _memory(batch: int = 1) -> NativePatchMemory:
    return NativePatchMemory(
        tokens=torch.randn(batch, 2, 196, 384).detach(),
        patch_valid_mask=torch.ones(batch, 2, 196, dtype=torch.bool),
        camera_valid_mask=torch.ones(batch, 2, dtype=torch.bool),
        camera_ids=("main", "wrist"),
        grid=(14, 14),
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        weights_sha256=_hash("weights"),
        input_contract_sha256=_hash("camera"),
        preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        output_contract_sha256=DINO_V3_OUTPUT_CONTRACT_SHA256,
        memory_contract_sha256=_hash("memory"),
    )


def _config(backend="stored_native", *, per_sample=1 << 22, aggregate=1 << 24):
    return _replay.VisualReplayConfig(
        backend=backend,
        storage_dtype="bfloat16",
        pin_memory=True,
        max_bytes_per_sample=per_sample,
        max_aggregate_bytes=aggregate,
    )


def test_stored_native_replay_scatter_roundtrip_and_hash_rejection() -> None:
    metadata = _metadata()
    transport = build_area_overlap_dino_wan_transport(metadata)
    memory = _memory()
    packed = _replay.pack_visual_replay(
        config=_config(),
        transport=transport,
        camera_batch=_camera_batch(),
        memory=memory,
        sample_indices=torch.tensor([1]),
        full_batch_size=3,
    )

    assert packed["visual_route_mask"].tolist() == [False, True, False]
    restored = _replay.unpack_stored_native_memory(
        {name: value[1:2] for name, value in packed.items()},
        camera_ids=memory.camera_ids,
        patch_grid=memory.grid,
        source_revision=memory.source_revision,
        weights_sha256=memory.weights_sha256,
        input_contract_sha256=memory.input_contract_sha256,
        preprocess_sha256=memory.preprocess_sha256,
        output_contract_sha256=memory.output_contract_sha256,
        memory_contract_sha256=memory.memory_contract_sha256,
        transport=transport,
        device="cpu",
    )
    torch.testing.assert_close(restored.tokens, memory.tokens.to(torch.bfloat16))

    tampered = {name: value.clone() for name, value in packed.items()}
    tampered["visual_effective_transport_sha256"][1, 0] ^= 1
    with pytest.raises(ValueError, match="effective transport hash"):
        _replay.unpack_stored_native_memory(
            {name: value[1:2] for name, value in tampered.items()},
            camera_ids=memory.camera_ids,
            patch_grid=memory.grid,
            source_revision=memory.source_revision,
            weights_sha256=memory.weights_sha256,
            input_contract_sha256=memory.input_contract_sha256,
            preprocess_sha256=memory.preprocess_sha256,
            output_contract_sha256=memory.output_contract_sha256,
            memory_contract_sha256=memory.memory_contract_sha256,
            transport=transport,
            device="cpu",
        )


def test_visual_replay_caps_and_recompute_camera_contract() -> None:
    metadata = _metadata()
    transport = build_area_overlap_dino_wan_transport(metadata)
    camera_batch = _camera_batch()
    memory = _memory()
    with pytest.raises(MemoryError, match="max_bytes_per_sample"):
        _replay.pack_visual_replay(
            config=_config(per_sample=16, aggregate=32),
            transport=transport,
            camera_batch=camera_batch,
            memory=memory,
        )

    packed = _replay.pack_visual_replay(
        config=_config("recompute_native"),
        transport=transport,
        camera_batch=camera_batch,
        memory=memory,
    )
    restored = _replay.unpack_recompute_camera_batch(
        packed,
        camera_ids=camera_batch.camera_ids,
        input_contract_sha256=camera_batch.input_contract_sha256,
    )
    assert torch.equal(restored.pixels, camera_batch.pixels)
    _replay.validate_recomputed_effective_hash(
        packed,
        memory=memory,
        transport=transport,
    )


def test_cuda_host_never_silently_falls_back_from_pinned_replay(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_pin(_tensor):
        raise RuntimeError("synthetic pinned allocator failure")

    monkeypatch.setattr(torch.Tensor, "pin_memory", fail_pin)
    with pytest.raises(RuntimeError, match="CUDA-capable host"):
        _replay._pin_cpu(torch.zeros(1))


def _enabled_payload() -> dict:
    metadata = _metadata()
    transport = build_area_overlap_dino_wan_transport(metadata)
    return {
        "enabled": True,
        "type": "dinov3_router_wan_value",
        "enabled_regimes": ["uncond"],
        "dispatch_before_encoder": True,
        "dino": {
            "source_root": "/not-loaded/dinov3",
            "source_revision": PINNED_DINOV3_SOURCE_REVISION,
            "model_name": PINNED_DINOV3_MODEL_NAME,
            "weights_path": "/not-loaded/dinov3.pth",
            "weights_sha256": _hash("weights"),
            "preprocess_sha256": DINO_V3_PREPROCESS_SHA256,
            "output_contract_sha256": DINO_V3_OUTPUT_CONTRACT_SHA256,
            "compute_dtype": "bfloat16",
            "license_id": "DINOv3-License",
            "camera_input_contract_sha256": _hash("camera"),
            "token_kind": "x_norm_patchtokens",
            "per_camera": True,
            "frozen": True,
            "stop_gradient": True,
        },
        "router": {
            "query_source": "base_modulated_self_attn_input",
            "query_projection": "low_rank",
            "query_rank": 3,
            "temperature": 0.2,
            "score_dtype": "float32",
            "per_camera_softmax": True,
            "camera_mass": "fixed",
            "camera_mass_values": [0.5, 0.5],
            "invalid_camera_policy": "renormalize_active_or_fail_closed",
        },
        "transport": {
            "mode": "deterministic_area_overlap",
            "contract_sha256": transport.transport_sha256,
            "preserve_row_mass": True,
            "fail_closed": True,
        },
        "optimizer": {
            "family": "visual_router",
            "lr": 1e-4,
            "weight_decay": 0.0,
            "scheduler": "cosine",
            "parameter_allowlist": [
                "routers.*.query_projection.*",
                "branches.*.*.raw_beta",
            ],
            "fail_on_empty_or_overlap": True,
        },
        "wan_value": {
            "source": "video_cache_current_prefix",
            "flatten_order": "t_h_w_row_major",
            "output_projection": "frozen_action_self_attn_o_weight_only",
            "output_bias": False,
            "reuse_base_gate_msa": True,
            "spatial_metadata": metadata.__dict__,
        },
        "injection": {
            "query_timing": "pre_block",
            "residual_timing": "post_block",
            "layer_indices": [0],
            "beta_parameterization": "bounded_tanh",
            "beta_max": 0.5,
            "zero_init": "beta",
            "modify_base_attention": False,
        },
        "replay": {
            "backend": "stored_native",
            "storage_dtype": "bfloat16",
            "pin_memory": True,
            "max_bytes_per_sample": 1 << 22,
            "max_aggregate_bytes": 1 << 24,
        },
    }


def test_default_off_returns_before_encoder_and_enabled_builder_is_owned() -> None:
    calls = []

    def factory(asset, *, device):
        calls.append((asset, device))
        return nn.Linear(1, 1, bias=False)

    assert (
        _sidecar.build_uncond_visual_sidecar(
            {"enabled": False},
            actor=object(),
            device="cpu",
            encoder_factory=factory,
        )
        is None
    )
    assert calls == []

    actor = SimpleNamespace(
        action_expert=SimpleNamespace(hidden_dim=6),
        video_expert=SimpleNamespace(patch_size=(1, 2, 2)),
        mot=SimpleNamespace(num_heads=2, attn_head_dim=4, num_layers=1),
    )
    built = _sidecar.build_uncond_visual_sidecar(
        _enabled_payload(),
        actor=actor,
        device="cpu",
        encoder_factory=factory,
    )
    assert len(calls) == 1
    assert built.reader.parameter_family == "visual_router"
    assert set(built.reader.trainable_parameter_manifest()) == {"visual_router"}
    assert all(not parameter.requires_grad for parameter in built.encoder.parameters())


def test_enabled_config_rejects_unresolved_assets_before_build() -> None:
    payload = _enabled_payload()
    payload["dino"]["weights_path"] = None
    with pytest.raises(ValueError, match="resolved `dino.weights_path`"):
        _sidecar.validate_uncond_visual_sidecar_config(payload)


def test_enabled_config_requires_explicit_spatial_contract_hash() -> None:
    payload = _enabled_payload()
    payload["wan_value"]["spatial_metadata"]["spatial_transport_contract_sha256"] = None
    with pytest.raises(ValueError, match="unresolved fields"):
        _sidecar.validate_uncond_visual_sidecar_config(payload)
