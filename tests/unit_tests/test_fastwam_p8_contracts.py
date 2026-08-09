# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.adaptive_action import CachedActionCondition
from fastwam.models.wan22.visual_contracts import NativePatchMemory
from fastwam.models.wan22.wan_current_refiner import (
    ActionVideoKVView,
    WanCurrentKVRefiner,
    WanCurrentLayerSource,
    WanCurrentRefinerConfig,
)

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    _resolve_fastwam_refiner_manifest,
)
from rlinf.models.embodiment.wam_policy.adaptive_policy import (
    FastWAMAdaptivePolicy,
    FastWAMAdaptivePolicyConfig,
)
from rlinf.models.embodiment.wam_policy.libero_runtime import LiberoFastWAMRuntime
from rlinf.models.embodiment.wam_policy.optimizer import (
    partition_fastwam_trainable_parameters,
)
from rlinf.models.embodiment.wam_policy.p8_sidecar import (
    build_p8_sidecar,
    validate_p8_sidecar_config,
)
from rlinf.models.embodiment.wam_policy.p8_visual_replay import (
    P8FrozenVisualSource,
    P8VisualReplayConfig,
    P8VisualReplaySpec,
    PackedP8VisualReplay,
    pack_p8_visual_sources,
    validate_p8_replay_bytes,
)
from rlinf.workers.actor.fastwam_selective_sync import capture_fastwam_sync_tensors
from rlinf.workers.actor.fsdp_actor_worker import _fastwam_actor_checkpoint_schema
from rlinf.workers.rollout.hf.huggingface_worker import (
    _fastwam_rollout_runtime_schema,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _memory() -> NativePatchMemory:
    tokens = torch.randn(1, 2, 196, 384, dtype=torch.bfloat16)
    valid = torch.ones(1, 2, dtype=torch.bool)
    return NativePatchMemory(
        tokens=tokens,
        patch_valid_mask=valid.unsqueeze(-1).expand(-1, -1, 196),
        camera_valid_mask=valid,
        camera_ids=("main", "wrist"),
        grid=(14, 14),
        source_revision="abc123",
        weights_sha256=_hash("weights"),
        input_contract_sha256=_hash("input"),
        preprocess_sha256=_hash("preprocess"),
        output_contract_sha256=_hash("output"),
        memory_contract_sha256=_hash("memory"),
    )


def _source(actor_version: int = 3) -> P8FrozenVisualSource:
    layer = WanCurrentLayerSource(
        layer_index=0,
        hidden_current=torch.randn(1, 4, 8, dtype=torch.bfloat16),
        attention_input_current=torch.randn(1, 4, 8, dtype=torch.bfloat16),
        key_pre_norm_current=torch.randn(1, 4, 8, dtype=torch.bfloat16),
        base_key_current=torch.randn(1, 4, 8, dtype=torch.bfloat16),
        base_value_current=torch.randn(1, 4, 8, dtype=torch.bfloat16),
        rope_freqs_current=torch.ones(4, 1, 2, dtype=torch.complex64),
        camera_index_current=torch.tensor([[0, 0, 1, 1]]),
        current_frame_video_tokens=4,
        source_contract_sha256=_hash("source"),
    )
    return P8FrozenVisualSource(
        memory=_memory(),
        layers=(layer,),
        actor_version=actor_version,
    )


def _spec() -> P8VisualReplaySpec:
    memory = _memory()
    return P8VisualReplaySpec(
        layer_indices=(0,),
        camera_ids=memory.camera_ids,
        current_frame_video_tokens=4,
        wan_hidden_dim=8,
        kv_dim=8,
        rope_shape=(4, 1, 2),
        memory_contract_sha256=memory.memory_contract_sha256,
        source_contract_sha256=_hash("source"),
        native_source_revision=memory.source_revision,
        native_weights_sha256=memory.weights_sha256,
        native_input_contract_sha256=memory.input_contract_sha256,
        native_preprocess_sha256=memory.preprocess_sha256,
        native_output_contract_sha256=memory.output_contract_sha256,
    )


def _replay_config(**overrides) -> P8VisualReplayConfig:
    values = {
        "backend": "stored_native",
        "storage_dtype": "bfloat16",
        "pin_memory": True,
        "max_bytes_per_sample": 10_000_000,
        "max_aggregate_bytes": 20_000_000,
        "max_combined_gate_plus_p8_bytes_per_sample": 11_000_000,
        "max_combined_gate_plus_p8_aggregate_bytes": 22_000_000,
        "fail_closed": True,
    }
    values.update(overrides)
    return P8VisualReplayConfig.from_mapping(values)


def test_stored_visual_replay_roundtrips_real_native_provenance_and_idm_slot() -> None:
    source = _source()
    spec = _spec()
    packed = pack_p8_visual_sources((source, None), spec=spec)
    restored = packed.materialize_sample(
        0,
        device="cpu",
        expected_actor_version=3,
    )

    assert restored.memory.source_revision == source.memory.source_revision
    assert restored.memory.weights_sha256 == source.memory.weights_sha256
    assert restored.memory.input_contract_sha256 == source.memory.input_contract_sha256
    assert restored.memory.preprocess_sha256 == source.memory.preprocess_sha256
    assert (
        restored.memory.output_contract_sha256 == source.memory.output_contract_sha256
    )
    assert torch.equal(restored.memory.tokens, source.memory.tokens)
    assert torch.equal(
        restored.layers[0].hidden_current, source.layers[0].hidden_current
    )
    assert not bool(packed.present[1])
    with pytest.raises(ValueError, match="IDM replay slots"):
        packed.materialize_sample(1, device="cpu", expected_actor_version=3)

    reconstructed = PackedP8VisualReplay.from_forward_inputs(
        packed.as_forward_inputs(),
        spec=spec,
    )
    assert torch.equal(reconstructed.bytes_per_sample(), packed.bytes_per_sample())


def test_visual_replay_caps_and_unimplemented_recompute_fail_closed() -> None:
    p8_bytes = torch.tensor([100, 200])
    gate_bytes = torch.tensor([10, 20])
    validate_p8_replay_bytes(
        p8_bytes_per_sample=p8_bytes,
        gate_bytes_per_sample=gate_bytes,
        config=_replay_config(
            max_bytes_per_sample=200,
            max_aggregate_bytes=300,
            max_combined_gate_plus_p8_bytes_per_sample=220,
            max_combined_gate_plus_p8_aggregate_bytes=330,
        ),
    )
    with pytest.raises(MemoryError, match=r"combined Gate\+P8 aggregate"):
        validate_p8_replay_bytes(
            p8_bytes_per_sample=p8_bytes,
            gate_bytes_per_sample=gate_bytes,
            config=_replay_config(
                max_bytes_per_sample=200,
                max_aggregate_bytes=300,
                max_combined_gate_plus_p8_bytes_per_sample=220,
                max_combined_gate_plus_p8_aggregate_bytes=329,
            ),
        )
    with pytest.raises(NotImplementedError, match="recompute_native"):
        _replay_config(backend="recompute_native")


def test_default_off_never_resolves_assets_and_enabled_compile_rejects() -> None:
    disabled = {
        "enabled": False,
        "type": "dinov3_guided_wan_current_refinement",
        "dino": object(),
    }
    assert validate_p8_sidecar_config(disabled) == {
        "enabled": False,
        "type": "dinov3_guided_wan_current_refinement",
    }
    assert (
        build_p8_sidecar(
            disabled,
            actor=object(),
            device="cpu",
            dtype=torch.bfloat16,
        )
        is None
    )

    enabled = {
        "type": "dinov3_guided_wan_current_refinement",
        "enabled": True,
        "compile": True,
        "enabled_regimes": ["uncond"],
        "dino": {},
        "refiner": {},
        "replay": {},
        "camera_ids": ["main", "wrist"],
        "camera_input_contract_sha256": _hash("input"),
        "license_record_sha256": _hash("license"),
        "fixed_cost_profile_sha256": _hash("cost"),
    }
    with pytest.raises(ValueError, match="compiled execution is not implemented"):
        validate_p8_sidecar_config(enabled)


def test_p8_outer_checkpoint_schemas_share_versioned_suffix() -> None:
    disabled = SimpleNamespace()
    enabled = SimpleNamespace(uncond_visual_sidecar=SimpleNamespace(enabled=True))
    assert _fastwam_actor_checkpoint_schema(disabled).endswith("checkpoint-v1")
    assert _fastwam_rollout_runtime_schema(disabled).endswith("runtime-v1")
    assert _fastwam_actor_checkpoint_schema(enabled).endswith("v2-p8-a0-kv")
    assert _fastwam_rollout_runtime_schema(enabled).endswith("v2-p8-a0-kv")


class _CountingEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def prepare_memory(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("IDM called DINO")


class _RecordingRefiner:
    def __init__(self) -> None:
        self.grad_flags = []

    def build_action_view(
        self,
        *,
        base_video_kv_cache,
        actor_version,
        **_kwargs,
    ):
        self.grad_flags.append(torch.is_grad_enabled())
        return ActionVideoKVView.base_alias(
            base_video_kv_cache,
            actor_version=actor_version,
        )


def test_runtime_idm_no_call_and_live_shadow_is_grad_only_during_replay() -> None:
    runtime = object.__new__(LiberoFastWAMRuntime)
    runtime.p8_encoder = _CountingEncoder()
    runtime.p8_refiner = _RecordingRefiner()
    runtime.actor = SimpleNamespace(
        mot=SimpleNamespace(mixtures={"video": SimpleNamespace(blocks=[])})
    )

    assert (
        runtime._prepare_p8_native_memory(
            regime=PolicyRegime.IDM,
            camera_batch=None,
        )
        is None
    )
    assert runtime.p8_encoder.calls == 0

    base = [{"k": torch.randn(1, 4, 8), "v": torch.randn(1, 4, 8)}]
    condition = CachedActionCondition(
        context=torch.randn(1, 2, 8),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        video_kv_cache=base,
        attention_mask=torch.ones(7, 7, dtype=torch.bool),
        video_seq_len=4,
        current_frame_video_tokens=4,
    )
    source = SimpleNamespace(
        actor_version=9,
        layers=(),
        memory=None,
    )
    attached = runtime._attach_p8_shadow(
        condition,
        source,
        actor_version=9,
        allow_no_grad=False,
    )
    assert attached.action_video_kv_view is not None
    assert runtime.p8_refiner.grad_flags == [True]

    behavior = runtime._attach_p8_behavior_shadow(
        condition,
        source,
        actor_version=9,
    )
    assert behavior.action_video_kv_view is not None
    assert runtime.p8_refiner.grad_flags == [True, False]

    with (
        torch.no_grad(),
        pytest.raises(RuntimeError, match="must be built with gradients"),
    ):
        runtime._attach_p8_shadow(
            condition,
            source,
            actor_version=9,
            allow_no_grad=False,
        )


class _LoRA:
    def __init__(self) -> None:
        self.parameter = nn.Parameter(torch.zeros(1))

    def lora_parameters(self):
        yield self.parameter

    def lora_state_dict(self):
        return {"p": self.parameter.detach().clone()}

    def load_lora_state_dict(self, state, strict=True):
        del strict
        self.parameter.data.copy_(state["p"])


class _Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value_head = nn.Linear(1, 1)


def _refiner() -> WanCurrentKVRefiner:
    return WanCurrentKVRefiner(
        WanCurrentRefinerConfig(
            wan_hidden_dim=8,
            native_dim=384,
            layer_indices=(0,),
            query_rank=2,
            output_rank=2,
            temperature=0.2,
            alpha=1.0,
            memory_contract_sha256=_hash("memory"),
            source_contract_sha256=_hash("source"),
        )
    )


def test_p8_policy_optimizer_checkpoint_and_selective_sync_are_versioned() -> None:
    refiner = _refiner()
    policy = FastWAMAdaptivePolicy(
        actor=nn.Linear(1, 1),
        runtime=object(),
        lora_adapter=_LoRA(),
        gate=nn.Linear(1, 1),
        critic=_Critic(),
        config=FastWAMAdaptivePolicyConfig(p8_visual_replay=_replay_config()),
        wan_current_refiner=refiner,
        p8_checkpoint_contract={"variant": "p8-a0-kv"},
    )
    groups = policy.optimizer_parameter_groups(
        gate_lr=1e-4,
        lora_lr=2e-4,
        value_lr=3e-4,
        refiner_lr=4e-4,
    )
    assert [group["name"] for group in groups] == [
        "gate",
        "uncond_lora",
        "value_head",
        "wan_current_refiner",
    ]
    payload = policy.trainable_state_dict()
    assert payload["schema"] == "fastwam-adaptive-policy-v2-p8-a0-kv"
    assert payload["p8"]["checkpoint_contract"] == {"variant": "p8-a0-kv"}
    old_payload = dict(payload)
    old_payload.pop("p8")
    old_payload["schema"] = "fastwam-adaptive-policy-v1"
    with pytest.raises(ValueError, match="checkpoint keys changed"):
        policy.load_trainable_state_dict(old_payload)

    captured = capture_fastwam_sync_tensors(policy)
    refiner_names = [
        name for name in captured if name.startswith("wan_current_refiner.")
    ]
    assert refiner_names
    assert all(captured[name].is_parameter for name in refiner_names)

    refiner_manifest = policy.refiner_parameter_manifest()
    assert refiner_manifest is not None
    wrapper = nn.Module()
    wrapper.add_module("module", policy)
    unwrapped_manifest = _resolve_fastwam_refiner_manifest(wrapper)
    assert unwrapped_manifest is not None
    assert unwrapped_manifest.parameter_ids == refiner_manifest.parameter_ids
    named = [
        ("gate.weight", nn.Parameter(torch.zeros(1))),
        ("actor.q.lora_A", nn.Parameter(torch.zeros(1))),
        ("critic.value_head.weight", nn.Parameter(torch.zeros(1))),
        *[
            (f"_fsdp_flat_refiner_{index}", parameter)
            for index, parameter in enumerate(refiner_manifest.parameters)
        ],
    ]
    partitioned = partition_fastwam_trainable_parameters(
        named,
        require_refiner=True,
        refiner_manifest=refiner_manifest,
    )
    assert set(partitioned) == {
        "gate",
        "uncond_lora",
        "value_head",
        "wan_current_refiner",
    }
