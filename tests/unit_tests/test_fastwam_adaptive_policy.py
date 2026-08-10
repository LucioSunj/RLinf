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

import hashlib
import importlib.util
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

OUTER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OUTER / "FastWAM/src"))

from fastwam.adapters import PolicyRegime  # noqa: E402
from fastwam.models.wan22.dinov3_memory import (  # noqa: E402
    DINO_V3_OUTPUT_CONTRACT_SHA256,
    DINO_V3_PREPROCESS_SHA256,
    PINNED_DINOV3_SOURCE_REVISION,
)
from fastwam.models.wan22.kv_tap import (  # noqa: E402
    GateKVSnapshot,
    GateLayerKV,
    KeyValueBank,
    KVSource,
)
from fastwam.models.wan22.visual_contracts import (  # noqa: E402
    WAN_FLATTEN_ORDER,
    WAN_VIDEO_VALUE_LAYOUT,
    NativePatchMemory,
    PreparedCameraBatch,
    WanValueSpatialMetadata,
    build_area_overlap_dino_wan_transport,
)


def _load_policy_package():
    repo = Path(__file__).resolve().parents[2]
    base_policy = ModuleType("rlinf.models.embodiment.base_policy")
    base_policy.ForwardType = Enum("ForwardType", {"DEFAULT": "default"})

    class BasePolicy:
        def forward(self, forward_type=base_policy.ForwardType.DEFAULT, **kwargs):
            if forward_type is base_policy.ForwardType.DEFAULT:
                return self.default_forward(**kwargs)
            raise NotImplementedError

    base_policy.BasePolicy = BasePolicy
    sys.modules[base_policy.__name__] = base_policy

    package_name = "fastwam_policy_composite_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(repo / "rlinf/models/embodiment/wam_policy")]
    sys.modules[package_name] = package
    for name in (
        "contracts",
        "kv_replay",
        "routing_state",
        "evaluation",
        "visual_replay",
        "adaptive_policy",
        "libero_runtime",
    ):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name,
            repo / f"rlinf/models/embodiment/wam_policy/{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.adaptive_policy"]


_policy = _load_policy_package()
_runtime_module = sys.modules["fastwam_policy_composite_under_test.libero_runtime"]
_visual_replay_module = sys.modules["fastwam_policy_composite_under_test.visual_replay"]
FastWAMAdaptivePolicy = _policy.FastWAMAdaptivePolicy
FastWAMAdaptivePolicyConfig = _policy.FastWAMAdaptivePolicyConfig
FastWAMChunkSample = _policy.FastWAMChunkSample


def _bank(source, value, batch=2):
    tensor = torch.full((batch, 1, 2), value)
    return KeyValueBank(
        source=source,
        key=tensor,
        value=tensor + 1,
        valid_mask=torch.ones(batch, 1, dtype=torch.bool),
    )


def _snapshots(routes):
    batch = len(routes)
    modes = tuple(
        PolicyRegime.IDM if int(route) else PolicyRegime.UNCOND for route in routes
    )
    result = []
    for timestep, action_value in ((900.0, 3.0), (100.0, 4.0)):
        result.append(
            GateKVSnapshot(
                (
                    GateLayerKV(
                        layer_index=0,
                        denoise_timestep=torch.full((batch,), timestep),
                        current_mode=modes,
                        current_frame_video=_bank(
                            KVSource.CURRENT_FRAME_VIDEO, 1.0, batch=batch
                        ),
                        action=_bank(KVSource.ACTION, action_value, batch=batch),
                        context=_bank(KVSource.TEXT_STATE_CONTEXT, 2.0, batch=batch),
                        actor_version=5,
                    ),
                )
            )
        )
    return tuple(result)


class _Runtime:
    def __init__(self):
        self.recompute_calls = 0
        self.sample_batch_sizes = []
        self.collect_replay_flags = []
        self.grad_enabled_flags = []

    def sample_action_batch(
        self,
        *,
        env_obs,
        routes,
        mode,
        actor_version,
        collect_replay=True,
    ):
        del mode
        batch = routes.shape[0]
        self.sample_batch_sizes.append(batch)
        self.collect_replay_flags.append(collect_replay)
        self.grad_enabled_flags.append(torch.is_grad_enabled())
        forward_inputs = {"critic_states": env_obs["states"].clone()}
        if hasattr(self, "visual_sidecar"):
            forward_inputs.update(
                _visual_replay_module.empty_visual_replay(
                    config=self.visual_replay,
                    batch_size=batch,
                    camera_count=2,
                    patch_grid=(14, 14),
                    camera_hw=(224, 224),
                    actor_version=actor_version,
                    static_contract_sha256="a" * 64,
                )
            )
        return FastWAMChunkSample(
            actions=torch.zeros(batch, 2, 3),
            old_flow_logprobs=torch.zeros(batch, 2, 3),
            flow_chains=torch.zeros(batch, 3, 2, 3),
            denoise_indices=torch.zeros(batch, dtype=torch.long),
            gate_snapshots=_snapshots(routes),
            forward_inputs=forward_inputs,
        )

    def replay_action_batch(self, *, forward_inputs, route_info):
        del route_info
        batch = forward_inputs["critic_states"].shape[0]
        return {
            "flow_logprobs": torch.zeros(batch, 2, 3, dtype=torch.float32),
            "flow_entropy": torch.ones(batch, 1, dtype=torch.float32),
        }

    def critic_observation(self, *, env_obs=None, forward_inputs=None):
        if env_obs is not None:
            return {"states": env_obs["states"]}
        return {"states": forward_inputs["critic_states"]}

    def recompute_gate_snapshots(self, *, forward_inputs, route_info):
        del forward_inputs
        self.recompute_calls += 1
        return _snapshots(route_info.route_used)


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(-1.0))

    def forward(self, snapshots):
        return self.bias.expand(snapshots[0].batch_size)


class _ThirtyBlockGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(1, 1, bias=False) for _ in range(30))

    def forward(self, snapshots):
        return self.blocks[0].weight.reshape(()).expand(snapshots[0].batch_size)


class _Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = nn.Linear(1, 1)
        self.predict_calls = 0

    def predict_value_batch(self, obs, *, return_prefix=False):
        self.predict_calls += 1
        prefix = obs["states"][:, None, :1]
        values = self.value_head(prefix.mean(dim=1)).squeeze(-1)
        return (values, prefix) if return_prefix else values

    def value_from_prefix(self, prefix):
        return self.value_head(prefix.mean(dim=1)).squeeze(-1)


class _LoRA:
    def __init__(self):
        self.parameter = nn.Parameter(torch.zeros(1))
        self.replay_reference_version = None

    def lora_parameters(self):
        yield self.parameter

    def lora_state_dict(self):
        return {"p": self.parameter.detach().clone()}

    def load_lora_state_dict(self, state, strict=True):
        del strict
        self.parameter.data.copy_(state["p"])

    def capture_replay_reference(self, *, actor_version):
        self.replay_reference_version = actor_version


class _VisualReader(nn.Module):
    reader_kind = "test-p6-reader"
    reader_contract_sha256 = "1" * 64
    memory_contract_sha256 = "2" * 64
    parameter_family = "visual_router"

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(1, 1, bias=False)
        self.replay_reference_version = None
        self.replay_reference_weight = None

    def trainable_parameter_manifest(self):
        return {"visual_router": ("router.weight",)}

    def export_trainable_state(self):
        return {
            "schema": "test-reader-v1",
            "state": {"router.weight": self.router.weight.detach().clone()},
        }

    def load_trainable_state(self, payload):
        if set(payload) != {"schema", "state"} or payload["schema"] != (
            "test-reader-v1"
        ):
            raise ValueError("test visual reader state mismatch")
        self.router.weight.data.copy_(payload["state"]["router.weight"])

    def capture_replay_reference(self, *, actor_version):
        self.replay_reference_version = actor_version
        self.replay_reference_weight = self.router.weight.detach().clone()

    @contextmanager
    def use_replay_reference(self, *, actor_version):
        if actor_version != self.replay_reference_version:
            raise ValueError("test visual replay version mismatch")
        current = self.router.weight.detach().clone()
        try:
            self.router.weight.data.copy_(self.replay_reference_weight)
            yield
        finally:
            self.router.weight.data.copy_(current)


def _visual_sidecar(reader):
    return SimpleNamespace(
        reader=reader,
        replay=_visual_replay_module.VisualReplayConfig(
            backend="stored_native",
            storage_dtype="bfloat16",
            pin_memory=True,
            max_bytes_per_sample=1 << 22,
            max_aggregate_bytes=1 << 24,
            max_combined_gate_plus_visual_bytes_per_sample=1 << 23,
            max_combined_gate_plus_visual_aggregate_bytes=1 << 25,
        ),
        asset=SimpleNamespace(
            source_revision="3" * 40,
            weights_sha256="4" * 64,
            preprocess_sha256="5" * 64,
            output_contract_sha256="6" * 64,
        ),
        camera_input_contract_sha256="7" * 64,
        spatial_metadata=SimpleNamespace(
            spatial_transport_contract_sha256="8" * 64,
            camera_order=("main", "wrist"),
            dino_patch_grid=(14, 14),
            wan_grid_f=1,
            wan_grid_h=7,
            wan_grid_w=14,
        ),
        transport=SimpleNamespace(transport_sha256="9" * 64),
    )


def _make_policy(
    backend="stored",
    *,
    with_critic=True,
    eval_routing_mode="learned_threshold",
    eval_random_idm_probability=None,
    eval_routing_seed=0,
    eval_timing_cuda_synchronize=False,
    with_visual=False,
    gate_trainable=True,
    training_route_override="none",
):
    runtime = _Runtime()
    reader = _VisualReader() if with_visual else None
    if reader is not None:
        runtime.visual_sidecar = _visual_sidecar(reader)
        runtime.visual_replay = runtime.visual_sidecar.replay
    return FastWAMAdaptivePolicy(
        actor=nn.Linear(1, 1),
        runtime=runtime,
        lora_adapter=_LoRA(),
        gate=_Gate(),
        critic=_Critic() if with_critic else None,
        visual_encoder=nn.Linear(1, 1, bias=False) if with_visual else None,
        visual_reader=reader,
        config=FastWAMAdaptivePolicyConfig(
            gate_epsilon=0.0,
            eval_idm_threshold=0.5,
            eval_routing_mode=eval_routing_mode,
            eval_random_idm_probability=eval_random_idm_probability,
            eval_routing_seed=eval_routing_seed,
            eval_timing_cuda_synchronize=eval_timing_cuda_synchronize,
            gate_trainable=gate_trainable,
            training_route_override=training_route_override,
            kv_replay=_policy.GateKVReplayConfig(
                backend=backend,
                pin_memory=False,
            ),
        ),
    )


def test_policy_forwards_and_deduplicates_critic_backbone_no_split_metadata():
    class Backbone(nn.Module):
        _no_split_modules = [
            "GemmaRMSNorm",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
        ]
        _no_split_names = [
            "action_in_proj",
            "lm_head",
            "action_in_proj",
        ]

    policy = _make_policy()
    policy.critic.backbone = Backbone()

    assert policy._no_split_modules == [
        "GemmaRMSNorm",
        "SiglipVisionEmbeddings",
    ]
    assert policy._no_split_names == ["action_in_proj", "lm_head"]

    # Callers receive a fresh list and cannot mutate backbone metadata.
    policy._no_split_modules.append("Unexpected")
    assert policy._no_split_modules == [
        "GemmaRMSNorm",
        "SiglipVisionEmbeddings",
    ]


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(_make_policy(with_critic=False), id="no-critic"),
        pytest.param(_make_policy(), id="critic-without-backbone"),
    ],
)
def test_policy_no_split_metadata_is_empty_without_nested_backbone(policy):
    assert policy._no_split_modules == []
    assert policy._no_split_names == []


def test_policy_no_split_metadata_is_empty_when_backbone_attributes_are_missing():
    policy = _make_policy()
    policy.critic.backbone = nn.Identity()

    assert policy._no_split_modules == []
    assert policy._no_split_names == []


def test_policy_forces_first_idm_and_applies_gate_to_next_chunk():
    policy = _make_policy()
    obs = {
        "states": torch.ones(2, 3),
        "_fastwam_env_ids": torch.tensor([11, 22]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }
    _, first = policy.predict_action_batch(obs, mode="eval")
    assert first["route_info"].route_used.tolist() == [1, 1]
    assert first["emitted_gate"].next_route.tolist() == [0, 0]
    assert first["forward_inputs"] == {}
    assert first["emitted_gate"].kv_metadata is None
    assert policy.runtime.sample_batch_sizes == [1, 1]
    assert policy.runtime.collect_replay_flags == [False, False]
    assert policy.runtime.grad_enabled_flags == [False, False]
    assert policy.critic.predict_calls == 0

    obs["_fastwam_reset_mask"] = torch.tensor([False, False])
    _, second = policy.predict_action_batch(obs, mode="eval")
    assert second["route_info"].route_used.tolist() == [0, 0]
    assert second["route_info"].route_source_chunk_ids.tolist() == [0, 0]


def test_policy_eval_gate_timing_is_explicit_and_finite() -> None:
    obs = {
        "states": torch.ones(2, 3),
        "_fastwam_env_ids": torch.tensor([11, 22]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }
    disabled = _make_policy()
    _, disabled_result = disabled.predict_action_batch(obs, mode="eval")
    assert disabled_result["gate_latency_seconds"] is None
    assert disabled_result["gate_h2d_seconds"] is None

    enabled = _make_policy(eval_timing_cuda_synchronize=True)
    _, result = enabled.predict_action_batch(obs, mode="eval")

    gate_latency = result["gate_latency_seconds"]
    gate_h2d = result["gate_h2d_seconds"]
    assert gate_latency.shape == (2,)
    assert gate_latency.dtype == torch.float64
    assert torch.isfinite(gate_latency).all()
    assert (gate_latency > 0).all()
    assert gate_h2d.shape == (2,)
    assert torch.equal(gate_h2d, torch.zeros(2, dtype=torch.float64))
    assert enabled.runtime.sample_batch_sizes == [1, 1]


@pytest.mark.parametrize(
    ("mode", "random_probability", "expected_next"),
    [
        ("learned_threshold", None, [0, 0]),
        ("forced_idm", None, [1, 1]),
        ("forced_uncond", None, [0, 0]),
        ("matched_random", 1.0, [1, 1]),
    ],
)
def test_policy_eval_uses_explicit_route_control_after_forced_first_chunk(
    mode,
    random_probability,
    expected_next,
):
    policy = _make_policy(
        eval_routing_mode=mode,
        eval_random_idm_probability=random_probability,
        eval_routing_seed=41,
    )
    obs = {
        "states": torch.ones(2, 3),
        "_fastwam_env_ids": torch.tensor([11, 22]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }

    _, first = policy.predict_action_batch(obs, mode="eval")

    assert first["route_info"].route_used.tolist() == [1, 1]
    assert first["route_info"].route_was_forced.tolist() == [True, True]
    assert first["emitted_gate"].next_route.tolist() == expected_next
    assert first["emitted_gate"].epsilon.tolist() == [0.0, 0.0]
    assert torch.allclose(
        first["emitted_gate"].base_probability,
        torch.full((2,), torch.sigmoid(torch.tensor(-1.0))),
    )
    selection = first["evaluation_selection"]
    assert selection.mode.value == mode
    assert selection.effective_next_route.tolist() == expected_next
    assert selection.counterfactual_next_route.tolist() == [0, 0]
    assert (selection.random_draws is not None) == (mode == "matched_random")
    assert policy.critic.predict_calls == 0

    obs["_fastwam_reset_mask"] = torch.tensor([False, False])
    _, second = policy.predict_action_batch(obs, mode="eval")
    assert second["route_info"].route_used.tolist() == expected_next
    assert second["route_info"].route_was_forced.tolist() == [False, False]


def test_training_gate_sampling_does_not_call_evaluation_selector(monkeypatch):
    policy = _make_policy()
    monkeypatch.setattr(
        _policy,
        "select_evaluation_routes",
        lambda *args, **kwargs: pytest.fail("training called evaluation selector"),
    )
    obs = {
        "states": torch.ones(2, 3),
        "_fastwam_env_ids": torch.tensor([1, 2]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }
    policy.predict_action_batch(obs, mode="train", compute_values=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_policy_gate_record_normalizes_cpu_route_metadata_to_cuda(mode):
    policy = _make_policy().to("cuda")
    obs = {
        "states": torch.ones(1, 3),
        "_fastwam_env_ids": torch.tensor([11]),
        "_fastwam_reset_mask": torch.tensor([True]),
    }

    _, first = policy.predict_action_batch(
        obs,
        mode=mode,
        compute_values=False,
    )

    assert first["route_info"].chunk_ids.device.type == "cpu"
    for field in (
        "next_route",
        "base_probability",
        "behavior_probability",
        "old_logprob",
        "epsilon",
        "temperature",
        "valid",
        "source_chunk_ids",
        "episode_ids",
        "actor_versions",
    ):
        assert getattr(first["emitted_gate"], field).device.type == "cuda"

    obs["_fastwam_reset_mask"] = torch.tensor([False])
    _, second = policy.predict_action_batch(
        obs,
        mode=mode,
        compute_values=False,
    )
    assert torch.equal(
        second["route_info"].route_used,
        first["emitted_gate"].next_route.cpu(),
    )


def test_libero_critic_observation_canonicalizes_optional_camera_keys():
    main_images = torch.zeros(1, 8, 8, 3)
    states = torch.ones(1, 8)
    raw = {
        "main_images": main_images,
        "states": states,
        "task_descriptions": ["test task"],
        "_fastwam_env_ids": torch.tensor([3]),
    }

    canonical = _runtime_module.LiberoFastWAMRuntime.critic_observation(
        object(),
        env_obs=raw,
    )

    assert canonical["main_images"] is main_images
    assert canonical["states"] is states
    assert canonical["wrist_images"] is None
    assert canonical["extra_view_images"] is None
    assert "_fastwam_env_ids" not in canonical
    assert "wrist_images" not in raw
    assert "extra_view_images" not in raw

    extra_view_images = torch.ones(1, 2, 8, 8, 3)
    explicit = _runtime_module.LiberoFastWAMRuntime.critic_observation(
        object(),
        env_obs={**raw, "extra_view_images": extra_view_images},
    )
    assert explicit["extra_view_images"] is extra_view_images


def test_p6_runtime_calls_dino_once_for_uncond_subset_and_never_for_idm() -> None:
    metadata = WanValueSpatialMetadata(
        wan_grid_f=1,
        wan_grid_h=2,
        wan_grid_w=4,
        current_frame_video_tokens=8,
        wan_flatten_order=WAN_FLATTEN_ORDER,
        vae_model_type="WanVideoVAE38",
        vae_weights_sha256="1" * 64,
        vae_spatial_downsample_factor=16,
        video_dit_weights_sha256="2" * 64,
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
    camera_batch = PreparedCameraBatch(
        pixels=torch.zeros(1, 2, 3, 224, 224, dtype=torch.uint8),
        camera_ids=("main", "wrist"),
        camera_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        input_contract_sha256="3" * 64,
    )
    memory = NativePatchMemory(
        tokens=torch.randn(1, 2, 196, 384).detach(),
        patch_valid_mask=torch.ones(1, 2, 196, dtype=torch.bool),
        camera_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        camera_ids=("main", "wrist"),
        grid=(14, 14),
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        weights_sha256="4" * 64,
        input_contract_sha256="3" * 64,
        preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        output_contract_sha256=DINO_V3_OUTPUT_CONTRACT_SHA256,
        memory_contract_sha256="5" * 64,
    )

    class Encoder:
        calls = 0

        def prepare_memory(self, regime, prepared):
            assert regime is PolicyRegime.UNCOND
            assert prepared is camera_batch
            self.calls += 1
            return memory

    replay_module = sys.modules["fastwam_policy_composite_under_test.visual_replay"]
    runtime = _runtime_module.LiberoFastWAMRuntime.__new__(
        _runtime_module.LiberoFastWAMRuntime
    )
    runtime.visual_sidecar = object()
    runtime.visual_encoder = Encoder()
    runtime.visual_spatial_metadata = metadata
    runtime.visual_transport = build_area_overlap_dino_wan_transport(metadata)
    runtime.visual_replay = replay_module.VisualReplayConfig(
        backend="stored_native",
        storage_dtype="bfloat16",
        pin_memory=True,
        max_bytes_per_sample=1 << 22,
        max_aggregate_bytes=1 << 24,
    )
    runtime.visual_replay_static_contract_sha256 = "6" * 64
    runtime.camera_height = 224
    runtime.camera_width = 224
    runtime._visual_camera_batch = lambda *_args, **_kwargs: camera_batch
    env_obs = {"states": torch.zeros(2, 8)}

    memories, replay = runtime._prepare_visual_rollout(
        env_obs=env_obs,
        routes=torch.tensor([1, 1]),
        collect_replay=True,
    )
    assert memories == {}
    assert runtime.visual_encoder.calls == 0
    assert replay["visual_route_mask"].tolist() == [False, False]

    memories, replay = runtime._prepare_visual_rollout(
        env_obs=env_obs,
        routes=torch.tensor([1, 0]),
        collect_replay=True,
    )
    assert set(memories) == {1}
    assert runtime.visual_encoder.calls == 1
    assert replay["visual_route_mask"].tolist() == [False, True]
    runtime._validate_visual_replay_route_alignment(
        replay,
        SimpleNamespace(
            route_used=torch.tensor([1, 0]),
            actor_versions=torch.zeros(2, dtype=torch.long),
        ),
    )
    tampered_route = {
        **replay,
        "visual_route_mask": torch.tensor([True, True]),
    }
    with pytest.raises(ValueError, match="content SHA256"):
        runtime._validate_visual_replay_route_alignment(
            tampered_route,
            SimpleNamespace(
                route_used=torch.tensor([1, 0]),
                actor_versions=torch.zeros(2, dtype=torch.long),
            ),
        )
    tampered_route["visual_content_sha256"] = replay_module._content_sha256(
        tampered_route
    )
    with pytest.raises(ValueError, match="consumed UNCOND routes"):
        runtime._validate_visual_replay_route_alignment(
            tampered_route,
            SimpleNamespace(
                route_used=torch.tensor([1, 0]),
                actor_versions=torch.zeros(2, dtype=torch.long),
            ),
        )


def test_standalone_eval_runs_and_restores_gate_lora_without_critic():
    source = _make_policy()
    source.set_global_step(3)
    with torch.no_grad():
        source.gate.bias.fill_(2.0)
        source.lora_adapter.parameter.fill_(4.0)
        source.critic.value_head.weight.fill_(6.0)
    parent_sha256 = "a" * 64
    payload = {
        "schema": "fastwam-adaptive-rl-checkpoint-v1",
        "step": 3,
        "parent_checkpoint_sha256": parent_sha256,
        "contract": {
            "model": {"actor_checkpoint_sha256": parent_sha256},
        },
        "policy": source.trainable_state_dict(),
    }

    policy = _make_policy(with_critic=False)
    restored_step = policy.load_eval_checkpoint(
        payload,
        expected_parent_checkpoint_sha256=parent_sha256,
    )
    obs = {
        "states": torch.ones(1, 3),
        "_fastwam_env_ids": torch.tensor([11]),
        "_fastwam_reset_mask": torch.tensor([True]),
    }
    actions, rollout = policy.predict_action_batch(
        obs,
        mode="eval",
        compute_values=False,
    )

    assert policy.critic is None
    assert restored_step == 3
    assert policy.actor_version == 3
    assert torch.equal(policy.gate.bias, source.gate.bias)
    assert torch.equal(policy.lora_adapter.parameter, source.lora_adapter.parameter)
    assert payload["policy"]["value_head"]
    assert actions.shape == (1, 2, 3)
    assert rollout["prev_values"].shape == (1, 1)
    with pytest.raises(RuntimeError, match="critic is intentionally absent"):
        policy.predict_action_batch(obs, mode="train")

    incompatible_policy = dict(payload["policy"])
    incompatible_policy["gate"] = {
        "blocks.0.weight": torch.zeros(1),
    }
    with pytest.raises(ValueError, match="Gate architecture mismatch"):
        _make_policy(with_critic=False).load_eval_checkpoint(
            {**payload, "policy": incompatible_policy},
            expected_parent_checkpoint_sha256=parent_sha256,
        )

    with pytest.raises(ValueError, match="parent hash mismatch"):
        _make_policy(with_critic=False).load_eval_checkpoint(
            payload,
            expected_parent_checkpoint_sha256="b" * 64,
        )

    critic_parent_sha256 = "c" * 64
    payload["critic_parent_checkpoint_sha256"] = critic_parent_sha256
    payload["contract"]["model"]["critic"] = {
        "backbone_checkpoint_sha256": critic_parent_sha256,
    }
    with pytest.raises(ValueError, match="pi0.5 evaluation checkpoint parent"):
        _make_policy().load_eval_checkpoint(
            payload,
            expected_parent_checkpoint_sha256=parent_sha256,
            expected_critic_parent_checkpoint_sha256="d" * 64,
        )

    assert (
        _make_policy().load_eval_checkpoint(
            payload,
            expected_parent_checkpoint_sha256=parent_sha256,
            expected_critic_parent_checkpoint_sha256=critic_parent_sha256,
        )
        == 3
    )


def test_policy_update_invalidates_pending_route_and_forces_idm_boundary():
    policy = _make_policy()
    obs = {
        "states": torch.ones(1, 3),
        "_fastwam_env_ids": torch.tensor([11]),
        "_fastwam_reset_mask": torch.tensor([True]),
    }
    _, rollout = policy.predict_action_batch(obs, mode="eval")
    assert rollout["emitted_gate"].kv_metadata is None
    policy.set_global_step(1)
    obs["_fastwam_reset_mask"] = torch.tensor([False])

    _, boundary = policy.predict_action_batch(obs, mode="eval")

    assert boundary["route_info"].route_used.item() == 1
    assert boundary["route_info"].route_was_forced.item()
    assert boundary["route_info"].route_source_chunk_ids.item() == -1
    assert boundary["route_info"].actor_versions.item() == 1


def test_policy_replay_exposes_separate_gate_and_flow_outputs():
    policy = _make_policy()
    obs = {
        "states": torch.ones(2, 3),
        "_fastwam_env_ids": torch.tensor([1, 2]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }
    _, rollout = policy.predict_action_batch(obs, mode="train")
    replay = policy.default_forward(
        rollout["forward_inputs"],
        route_info=rollout["route_info"],
        emitted_gate=rollout["emitted_gate"],
    )
    assert replay["gate_logprobs"].shape == (2,)
    assert replay["gate_behavior_probabilities"].shape == (2,)
    assert replay["flow_logprobs"].shape == (2, 2, 3)
    assert rollout["prev_values"].shape == (2, 1)
    assert replay["values"].shape == (2, 1)


def test_nn_module_forward_dispatches_and_actor_stays_in_eval_mode():
    policy = _make_policy()
    policy.train()
    assert policy.training
    assert policy.gate.training
    assert policy.critic.value_head.training
    assert not policy.actor.training

    obs = {
        "states": torch.ones(2, 3),
        "_fastwam_env_ids": torch.tensor([1, 2]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }
    _, rollout = policy.predict_action_batch(obs, mode="train")
    replay = policy(
        forward_inputs=rollout["forward_inputs"],
        route_info=rollout["route_info"],
        emitted_gate=rollout["emitted_gate"],
    )
    assert replay["values"].shape == rollout["prev_values"].shape


def test_recompute_backend_omits_stored_kv_and_rebuilds_gate_inputs():
    policy = _make_policy(backend="recompute")
    obs = {
        "states": torch.ones(2, 3),
        "_fastwam_env_ids": torch.tensor([1, 2]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }
    _, rollout = policy.predict_action_batch(obs, mode="train")
    assert not any(key.startswith("gate_kv_") for key in rollout["forward_inputs"])
    assert rollout["emitted_gate"].kv_metadata.total_bytes.tolist() == [0, 0]
    policy.capture_gate_recompute_reference()
    assert policy.lora_adapter.replay_reference_version == 0

    replay = policy.default_forward(
        rollout["forward_inputs"],
        route_info=rollout["route_info"],
        emitted_gate=rollout["emitted_gate"],
    )
    assert policy.runtime.recompute_calls == 1
    assert replay["gate_logprobs"].shape == (2,)


def test_trainable_checkpoint_excludes_frozen_actor_and_round_trips_version():
    policy = _make_policy()
    policy.set_global_step(3)
    payload = policy.trainable_state_dict()

    assert set(payload) == {
        "schema",
        "actor_version",
        "gate",
        "lora",
        "value_head",
        "route_tracker",
    }
    assert "actor" not in payload

    restored = _make_policy()
    restored.load_trainable_state_dict(payload)
    assert restored.actor_version == 3
    assert torch.equal(restored.gate.bias, policy.gate.bias)


def test_p6_checkpoint_optimizer_and_rollout_contract_are_strict() -> None:
    policy = _make_policy(with_visual=True)
    policy.set_global_step(4)
    with torch.no_grad():
        policy.visual_reader.router.weight.fill_(7.0)
    payload = policy.trainable_state_dict()

    assert policy.project_checkpoint_schema == ("fastwam-adaptive-rl-checkpoint-v2-p6")
    assert payload["schema"] == "fastwam-adaptive-policy-v2-p6"
    assert "visual_reader" in payload
    assert [
        group["name"]
        for group in policy.optimizer_parameter_groups(
            gate_lr=1e-4,
            lora_lr=1e-5,
            value_lr=1e-4,
            visual_router_lr=3e-5,
        )
    ] == ["gate", "uncond_lora", "value_head", "visual_router"]

    restored = _make_policy(with_visual=True)
    restored.load_trainable_state_dict(payload)
    assert restored.actor_version == 4
    assert torch.equal(
        restored.visual_reader.router.weight,
        policy.visual_reader.router.weight,
    )

    old = dict(payload)
    old.pop("visual_reader")
    old["schema"] = "fastwam-adaptive-policy-v1"
    with pytest.raises(ValueError, match="checkpoint keys changed"):
        _make_policy(with_visual=True).load_trainable_state_dict(old)

    runtime_state = policy.rollout_runtime_state_dict()
    assert runtime_state["schema"] == ("fastwam-adaptive-rollout-policy-runtime-v2-p6")
    restored.load_rollout_runtime_state_dict(runtime_state)
    tampered = {
        **runtime_state,
        "visual_contract": {
            **runtime_state["visual_contract"],
            "transport_sha256": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="reader/transport/replay contract"):
        restored.load_rollout_runtime_state_dict(tampered)


def test_p6_frozen_gate_stays_synced_but_has_three_optimizer_families() -> None:
    policy = _make_policy(
        with_visual=True,
        gate_trainable=False,
        training_route_override="forced_uncond_after_initial",
    )

    assert not policy.gate.training
    assert all(not parameter.requires_grad for parameter in policy.gate.parameters())
    assert policy.additional_rollout_sync_parameter_names() == ("gate.bias",)
    assert [
        group["name"]
        for group in policy.optimizer_parameter_groups(
            gate_lr=0.0,
            lora_lr=1e-5,
            value_lr=1e-4,
            visual_router_lr=1e-5,
        )
    ] == ["uncond_lora", "value_head", "visual_router"]

    logits = torch.tensor([-4.0, 4.0])
    route, *_ = policy._training_gate_decision(logits=logits)
    assert torch.equal(route, torch.zeros(2, dtype=torch.long))

    policy.train()
    assert not policy.gate.training
    assert policy.visual_reader.training


def test_p6_behavior_reference_and_outer_v1_rejection() -> None:
    policy = _make_policy(backend="recompute", with_visual=True, with_critic=False)
    policy.capture_gate_recompute_reference()
    assert policy.lora_adapter.replay_reference_version == 0
    assert policy.visual_reader.replay_reference_version == 0

    parent_sha256 = "a" * 64
    with pytest.raises(ValueError, match="Unsupported FastWAM adaptive evaluation"):
        policy.load_eval_checkpoint(
            {
                "schema": "fastwam-adaptive-rl-checkpoint-v1",
                "step": 0,
                "parent_checkpoint_sha256": parent_sha256,
                "contract": {
                    "model": {"actor_checkpoint_sha256": parent_sha256},
                },
                "policy": {},
            },
            expected_parent_checkpoint_sha256=parent_sha256,
        )


def test_p6_gate_recompute_uses_behavior_reader_but_flow_uses_live_grad() -> None:
    policy = _make_policy(backend="recompute", with_visual=True)

    class ReaderAwareRuntime(_Runtime):
        def __init__(self, reader):
            super().__init__()
            self.reader = reader
            self.behavior_weight_seen = None
            self.visual_sidecar = _visual_sidecar(reader)
            self.visual_replay = self.visual_sidecar.replay

        def recompute_gate_snapshots(self, *, forward_inputs, route_info):
            del forward_inputs
            with self.reader.use_replay_reference(
                actor_version=int(route_info.actor_versions[0])
            ):
                self.behavior_weight_seen = float(self.reader.router.weight.item())
                return _snapshots(route_info.route_used)

        def replay_action_batch(self, *, forward_inputs, route_info):
            del route_info
            batch = forward_inputs["critic_states"].shape[0]
            flow = self.reader.router.weight.reshape(1, 1, 1).expand(batch, 2, 3)
            return {
                "flow_logprobs": flow,
                "flow_entropy": torch.ones_like(flow),
            }

    policy.runtime = ReaderAwareRuntime(policy.visual_reader)
    with torch.no_grad():
        policy.visual_reader.router.weight.fill_(1.0)
    obs = {
        "states": torch.ones(2, 3),
        "_fastwam_env_ids": torch.tensor([1, 2]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }
    _, rollout = policy.predict_action_batch(obs, mode="train")
    policy.capture_gate_recompute_reference()
    with torch.no_grad():
        policy.visual_reader.router.weight.fill_(2.0)

    replay = policy.default_forward(
        rollout["forward_inputs"],
        route_info=rollout["route_info"],
        emitted_gate=rollout["emitted_gate"],
        compute_values=False,
    )
    replay["flow_logprobs"].sum().backward()

    assert policy.runtime.behavior_weight_seen == 1.0
    assert policy.visual_reader.router.weight.item() == 2.0
    assert policy.visual_reader.router.weight.grad.item() == 12.0


def test_native_all_layer_policy_payload_round_trips_without_frozen_actor() -> None:
    policy = _make_policy()
    policy.gate = _ThirtyBlockGate()
    policy.set_global_step(7)
    with torch.no_grad():
        for index, block in enumerate(policy.gate.blocks):
            block.weight.fill_(index + 1)
        policy.lora_adapter.parameter.fill_(31.0)
        policy.critic.value_head.weight.fill_(32.0)
        policy.critic.value_head.bias.fill_(33.0)
    policy.predict_action_batch(
        {
            "states": torch.ones(1, 3),
            "_fastwam_env_ids": torch.tensor([19]),
            "_fastwam_reset_mask": torch.tensor([True]),
        },
        mode="eval",
    )

    payload = policy.trainable_state_dict()

    assert set(payload["gate"]) == {f"blocks.{index}.weight" for index in range(30)}
    assert "actor" not in payload
    restored = _make_policy()
    restored.gate = _ThirtyBlockGate()
    restored.load_trainable_state_dict(payload)
    assert restored.actor_version == 7
    assert restored.route_tracker.state_dict() == policy.route_tracker.state_dict()
    assert torch.equal(
        restored.lora_adapter.parameter,
        policy.lora_adapter.parameter,
    )
    for name, expected in policy.gate.state_dict().items():
        assert torch.equal(restored.gate.state_dict()[name], expected)
    for name, expected in policy.critic.value_head.state_dict().items():
        assert torch.equal(restored.critic.value_head.state_dict()[name], expected)


def test_zero_flow_sde_noise_is_rejected_only_for_training_uncond() -> None:
    with pytest.raises(ValueError, match="noise_level > 0"):
        _runtime_module._validate_flow_sde_sampling(
            mode="train",
            routes=torch.tensor([0, 1]),
            noise_level=0.0,
        )
    _runtime_module._validate_flow_sde_sampling(
        mode="train",
        routes=torch.tensor([1, 1]),
        noise_level=0.0,
    )
    with pytest.raises(ValueError, match="finite"):
        _runtime_module._validate_flow_sde_sampling(
            mode="eval",
            routes=torch.tensor([1, 1]),
            noise_level=float("nan"),
        )


def test_fastwam_prompt_format_matches_training_template() -> None:
    prompts = _runtime_module._format_fastwam_prompts(
        ["pick up the mug", "open the drawer"],
        prompt_template=_runtime_module.DEFAULT_FASTWAM_PROMPT_TEMPLATE,
    )

    assert prompts == [
        "A video recorded from a robot's point of view executing the following instruction: pick up the mug",
        "A video recorded from a robot's point of view executing the following instruction: open the drawer",
    ]
    with pytest.raises(ValueError, match="must contain"):
        _runtime_module._format_fastwam_prompts(
            "pick up the mug",
            prompt_template="static prompt",
        )
    _runtime_module._validate_flow_sde_sampling(
        mode="eval",
        routes=torch.tensor([0, 0]),
        noise_level=0.0,
    )


def test_cached_eval_text_context_matches_fastwam_padding_and_fails_closed(
    tmp_path: Path,
) -> None:
    prompts = ["first prompt", "second prompt"]
    expected_contexts = []
    for index, prompt in enumerate(prompts):
        context = torch.full((3, 4), float(index + 1), dtype=torch.bfloat16)
        expected_context = context.clone()
        expected_context[-1] = 0
        expected_contexts.append(expected_context)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        torch.save(
            {"context": context, "mask": torch.tensor([True, True, False])},
            tmp_path / f"{digest}.t5_len3.wan22ti2v5b.pt",
        )

    context, mask = _runtime_module._load_cached_text_contexts(
        prompts,
        cache_dir=tmp_path,
        context_len=3,
        expected_dim=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert torch.equal(context, torch.stack(expected_contexts))
    assert torch.equal(
        mask,
        torch.ones((2, 3), dtype=torch.bool),
    )
    with pytest.raises(FileNotFoundError, match="prompt hash"):
        _runtime_module._load_cached_text_contexts(
            ["not precomputed"],
            cache_dir=tmp_path,
            context_len=3,
            expected_dim=4,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

    broken_prompt = "broken"
    broken_digest = hashlib.sha256(broken_prompt.encode("utf-8")).hexdigest()
    torch.save(
        {
            "context": torch.zeros(2, 4, dtype=torch.bfloat16),
            "mask": torch.ones(2, dtype=torch.bool),
        },
        tmp_path / f"{broken_digest}.t5_len3.wan22ti2v5b.pt",
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        _runtime_module._load_cached_text_contexts(
            [broken_prompt],
            cache_dir=tmp_path,
            context_len=3,
            expected_dim=4,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )


def test_action_schedule_stays_fp32_for_a_bfloat16_actor() -> None:
    class _Scheduler:
        requested_dtype = None
        num_train_timesteps = 1000

        def build_inference_schedule(
            self,
            *,
            num_inference_steps,
            device,
            dtype,
            shift_override,
        ):
            del shift_override
            self.requested_dtype = dtype
            return (
                torch.ones(num_inference_steps, device=device, dtype=dtype),
                -torch.ones(num_inference_steps, device=device, dtype=dtype),
            )

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.infer_action_scheduler = _Scheduler()

    runtime = object.__new__(_runtime_module.LiberoFastWAMRuntime)
    runtime.actor = _Actor()
    runtime.num_inference_steps = 20
    runtime.sigma_shift = None

    timesteps, deltas = runtime._action_schedule()

    assert runtime.actor.infer_action_scheduler.requested_dtype is torch.float32
    assert timesteps.dtype is torch.float32
    assert deltas.dtype is torch.float32


def test_runtime_aligns_plain_normalizer_tensors_and_converts_gripper():
    class _Normalizer:
        scale = torch.ones(3, dtype=torch.float64)
        offset = torch.zeros(3, dtype=torch.float64)

    normalizer = _Normalizer()
    reference = torch.zeros(2, 3, dtype=torch.float32)
    _runtime_module._align_linear_normalizer(normalizer, reference)
    assert normalizer.scale.dtype == torch.float32
    assert normalizer.offset.dtype == torch.float32

    actions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    converted = _runtime_module._convert_fastwam_gripper_to_libero(
        actions,
        binarize=False,
    )
    assert torch.equal(converted[:, -1], torch.tensor([1.0, -1.0]))
    assert torch.equal(actions[:, -1], torch.tensor([0.0, 1.0]))


def test_optimizer_groups_are_disjoint():
    policy = _make_policy()
    groups = policy.optimizer_parameter_groups(
        gate_lr=1e-4,
        lora_lr=2e-4,
        value_lr=3e-4,
    )
    assert [group["name"] for group in groups] == [
        "gate",
        "uncond_lora",
        "value_head",
    ]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids))
