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
from enum import Enum
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn

OUTER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OUTER / "FastWAM/src"))

from fastwam.adapters import PolicyRegime  # noqa: E402
from fastwam.models.wan22.kv_tap import (  # noqa: E402
    GateKVSnapshot,
    GateLayerKV,
    KeyValueBank,
    KVSource,
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
        del mode, actor_version
        batch = routes.shape[0]
        self.sample_batch_sizes.append(batch)
        self.collect_replay_flags.append(collect_replay)
        self.grad_enabled_flags.append(torch.is_grad_enabled())
        return FastWAMChunkSample(
            actions=torch.zeros(batch, 2, 3),
            old_flow_logprobs=torch.zeros(batch, 2, 3),
            flow_chains=torch.zeros(batch, 3, 2, 3),
            denoise_indices=torch.zeros(batch, dtype=torch.long),
            gate_snapshots=_snapshots(routes),
            forward_inputs={"critic_states": env_obs["states"].clone()},
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


def _make_policy(
    backend="stored",
    *,
    with_critic=True,
    eval_routing_mode="learned_threshold",
    eval_random_idm_probability=None,
    eval_routing_seed=0,
    eval_timing_cuda_synchronize=False,
    gate_trainable=True,
    training_route_override="none",
):
    return FastWAMAdaptivePolicy(
        actor=nn.Linear(1, 1),
        runtime=_Runtime(),
        lora_adapter=_LoRA(),
        gate=_Gate(),
        critic=_Critic() if with_critic else None,
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

    assert payload["schema"] == "fastwam-adaptive-policy-v1"
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


def test_p8_readiness_frozen_gate_has_no_optimizer_but_remains_checkpointed():
    policy = _make_policy(gate_trainable=False)

    policy.train()
    groups = policy.optimizer_parameter_groups(
        gate_lr=0.0,
        lora_lr=2e-4,
        value_lr=3e-4,
    )

    assert [group["name"] for group in groups] == ["uncond_lora", "value_head"]
    assert not policy.gate.training
    assert all(not parameter.requires_grad for parameter in policy.gate.parameters())
    assert policy.trainable_state_dict()["gate"]
    assert policy.additional_rollout_sync_parameter_names()
    with pytest.raises(ValueError, match="exactly zero"):
        policy.optimizer_parameter_groups(
            gate_lr=1e-4,
            lora_lr=2e-4,
            value_lr=3e-4,
        )


def test_p8_readiness_forces_uncond_after_initial_idm_chunk():
    policy = _make_policy(
        gate_trainable=False,
        training_route_override="forced_uncond_after_initial",
    )
    obs = {
        "states": torch.ones(1, 3),
        "_fastwam_env_ids": torch.tensor([0]),
        "_fastwam_reset_mask": torch.tensor([True]),
    }

    _, first = policy.predict_action_batch(obs, compute_values=False)
    obs["_fastwam_reset_mask"] = torch.tensor([False])
    _, second = policy.predict_action_batch(obs, compute_values=False)

    assert first["route_info"].route_used.tolist() == [1]
    assert first["emitted_gate"].next_route.tolist() == [0]
    assert second["route_info"].route_used.tolist() == [0]
