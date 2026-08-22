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
from types import ModuleType, SimpleNamespace

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


def _load_dual_ppo_module():
    repo = Path(__file__).resolve().parents[2]
    module_name = "fastwam_dual_ppo_sparse_parity_under_test"
    contracts_name = "rlinf.models.embodiment.wam_policy.contracts"
    previous_contracts = sys.modules.get(contracts_name)
    sys.modules[contracts_name] = sys.modules[
        "fastwam_policy_composite_under_test.contracts"
    ]
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            repo / "rlinf/algorithms/fastwam_dual_ppo.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_contracts is None:
            sys.modules.pop(contracts_name, None)
        else:
            sys.modules[contracts_name] = previous_contracts


_dual_ppo = _load_dual_ppo_module()
_runtime_module = sys.modules["fastwam_policy_composite_under_test.libero_runtime"]
FastWAMAdaptivePolicy = _policy.FastWAMAdaptivePolicy
FastWAMAdaptivePolicyConfig = _policy.FastWAMAdaptivePolicyConfig
FastWAMChunkSample = _policy.FastWAMChunkSample
FastWAMCurrentFrameValueCritic = sys.modules[
    "fastwam_policy_composite_under_test.critic"
].FastWAMCurrentFrameValueCritic
FastWAMCurrentFrameFeatureConfig = sys.modules[
    "fastwam_policy_composite_under_test.critic"
].FastWAMCurrentFrameFeatureConfig


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
        self.action_noise_seeds = []
        self.idm_noise_seeds = []
        self.critic_feature_calls = 0

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
        self.action_noise_seeds.append(
            env_obs.get("_fastwam_action_noise_seeds", torch.empty(0, dtype=torch.long))
            .detach()
            .cpu()
            .clone()
        )
        self.idm_noise_seeds.append(
            env_obs.get("_fastwam_idm_noise_seeds", torch.empty(0, dtype=torch.long))
            .detach()
            .cpu()
            .clone()
        )
        return FastWAMChunkSample(
            actions=torch.zeros(batch, 2, 3),
            old_flow_logprobs=torch.zeros(batch, 2, 3),
            flow_chains=torch.zeros(batch, 3, 2, 3),
            denoise_indices=torch.zeros(batch, dtype=torch.long),
            gate_snapshots=_snapshots(routes),
            forward_inputs={"critic_states": env_obs["states"].clone()},
            critic_features=env_obs["states"][:, :1].detach().float(),
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

    def critic_features(self, *, env_obs):
        self.critic_feature_calls += 1
        return env_obs["states"][:, :1].detach().float()

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
    training_rollout_microbatch_size=None,
    formal_training_sampling_seed=None,
    critic_kind="pi05",
):
    if not with_critic:
        critic = None
    elif critic_kind == "fastwam":
        critic = FastWAMCurrentFrameValueCritic(
            input_dim=1,
            hidden_sizes=(2,),
        )
    else:
        critic = _Critic()
    return FastWAMAdaptivePolicy(
        actor=nn.Linear(1, 1),
        runtime=_Runtime(),
        lora_adapter=_LoRA(),
        gate=_Gate(),
        critic=critic,
        config=FastWAMAdaptivePolicyConfig(
            gate_epsilon=0.0,
            eval_idm_threshold=0.5,
            eval_routing_mode=eval_routing_mode,
            eval_random_idm_probability=eval_random_idm_probability,
            eval_routing_seed=eval_routing_seed,
            eval_timing_cuda_synchronize=eval_timing_cuda_synchronize,
            training_rollout_microbatch_size=training_rollout_microbatch_size,
            formal_training_sampling_seed=formal_training_sampling_seed,
            kv_replay=_policy.GateKVReplayConfig(
                backend=backend,
                pin_memory=False,
            ),
        ),
    )


def _assert_tensor_record_equal(left, right) -> None:
    assert type(left) is type(right)
    for name in left.__dataclass_fields__:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if isinstance(left_value, torch.Tensor):
            assert torch.equal(left_value, right_value), name
        elif hasattr(left_value, "__dataclass_fields__"):
            _assert_tensor_record_equal(left_value, right_value)
        else:
            assert left_value == right_value, name


def test_training_rollout_microbatch_is_exact_across_stage_shards() -> None:
    torch.manual_seed(123)
    baseline = _make_policy(training_rollout_microbatch_size=1)
    torch.manual_seed(123)
    staged = _make_policy(training_rollout_microbatch_size=1)
    obs = {
        "states": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "_fastwam_env_ids": torch.tensor([0, 1, 2, 3]),
        "_fastwam_reset_mask": torch.tensor([True, True, True, True]),
    }

    torch.manual_seed(999)
    baseline_actions, baseline_result = baseline.predict_action_batch(
        obs,
        mode="train",
    )
    torch.manual_seed(999)
    staged_actions = []
    staged_results = []
    for start, end in ((0, 2), (2, 4)):
        action, result = staged.predict_action_batch(
            staged._slice_env_obs(obs, start=start, end=end, batch_size=4),
            mode="train",
        )
        staged_actions.append(action)
        staged_results.append(result)
    merged_actions, merged_result = staged._merge_training_microbatch_results(
        staged_actions,
        staged_results,
    )

    assert torch.equal(merged_actions, baseline_actions)
    assert set(merged_result["forward_inputs"]) == set(
        baseline_result["forward_inputs"]
    )
    for key, value in baseline_result["forward_inputs"].items():
        assert torch.equal(merged_result["forward_inputs"][key], value), key
    for key in ("prev_logprobs", "prev_values"):
        assert torch.equal(merged_result[key], baseline_result[key]), key
    for key in ("route_info", "emitted_gate"):
        _assert_tensor_record_equal(merged_result[key], baseline_result[key])
    assert baseline.runtime.sample_batch_sizes == staged.runtime.sample_batch_sizes
    assert baseline.runtime.sample_batch_sizes == [1, 1, 1, 1]
    assert baseline.route_tracker.state_dict() == staged.route_tracker.state_dict()


def test_formal_training_sampling_is_exact_across_stage_calls_and_rng_noise() -> None:
    torch.manual_seed(123)
    baseline = _make_policy(
        training_rollout_microbatch_size=1,
        formal_training_sampling_seed=42,
    )
    torch.manual_seed(123)
    staged = _make_policy(
        training_rollout_microbatch_size=1,
        formal_training_sampling_seed=42,
    )
    observations = {
        "states": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "_fastwam_env_ids": torch.tensor([0, 1, 2, 3]),
        "_fastwam_reset_mask": torch.tensor([True, True, True, True]),
    }

    for chunk_index in range(5):
        baseline_actions, baseline_result = baseline.predict_action_batch(
            observations,
            mode="train",
        )
        staged_actions = []
        staged_results = []
        for start, end in ((0, 2), (2, 4)):
            torch.rand(17 + chunk_index)
            action, result = staged.predict_action_batch(
                staged._slice_env_obs(
                    observations,
                    start=start,
                    end=end,
                    batch_size=4,
                ),
                mode="train",
            )
            staged_actions.append(action)
            staged_results.append(result)
        merged_actions, merged_result = staged._merge_training_microbatch_results(
            staged_actions,
            staged_results,
        )

        assert torch.equal(merged_actions, baseline_actions)
        for key in ("route_info", "emitted_gate"):
            _assert_tensor_record_equal(merged_result[key], baseline_result[key])
        observations["_fastwam_reset_mask"] = torch.zeros(4, dtype=torch.bool)

    assert baseline.route_tracker.state_dict() == staged.route_tracker.state_dict()
    assert torch.equal(
        torch.cat(baseline.runtime.action_noise_seeds),
        torch.cat(staged.runtime.action_noise_seeds),
    )
    assert torch.equal(
        torch.cat(baseline.runtime.idm_noise_seeds),
        torch.cat(staged.runtime.idm_noise_seeds),
    )


@pytest.mark.parametrize("invalid", [True, 0, 1.5])
def test_training_rollout_microbatch_rejects_invalid_values(invalid) -> None:
    with pytest.raises(
        (TypeError, ValueError), match="training_rollout_microbatch_size"
    ):
        FastWAMAdaptivePolicyConfig(training_rollout_microbatch_size=invalid)


@pytest.mark.parametrize("invalid", [True, -1, 1.5])
def test_formal_training_sampling_seed_rejects_invalid_values(invalid) -> None:
    with pytest.raises((TypeError, ValueError), match="formal_training_sampling_seed"):
        FastWAMAdaptivePolicyConfig(formal_training_sampling_seed=invalid)


def test_formal_training_sampling_refuses_caller_seed_collision() -> None:
    policy = _make_policy(formal_training_sampling_seed=42)
    observations = {
        "states": torch.ones(1, 3),
        "_fastwam_env_ids": torch.tensor([0]),
        "_fastwam_reset_mask": torch.tensor([True]),
        "_fastwam_action_noise_seeds": torch.tensor([7]),
    }

    with pytest.raises(ValueError, match="caller-supplied sampling seeds"):
        policy.predict_action_batch(observations, mode="train")


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


def test_libero_current_frame_bootstrap_encodes_only_uncond_conditions():
    runtime = object.__new__(_runtime_module.LiberoFastWAMRuntime)
    runtime.critic_feature_config = FastWAMCurrentFrameFeatureConfig(
        input_dim=2,
        layer_index=1,
        pooling="mean_token",
    )
    runtime._encode_condition = lambda env_obs: (
        torch.zeros(2, 3, 4, 4),
        torch.zeros(2, 5, 2),
        torch.ones(2, 5, dtype=torch.bool),
    )
    regimes = []

    def prepare_condition(**kwargs):
        regimes.append(kwargs["regime"])
        sample_index = len(regimes) - 1
        values = torch.tensor([[[1.0 + sample_index, 2.0], [3.0 + sample_index, 4.0]]])
        return (
            SimpleNamespace(
                video_kv_cache=[{"v": torch.zeros_like(values)}, {"v": values}],
                current_frame_video_tokens=2,
            ),
            None,
        )

    runtime._prepare_action_condition = prepare_condition

    features = runtime.critic_features(env_obs={"unused": True})

    assert regimes == [PolicyRegime.UNCOND, PolicyRegime.UNCOND]
    assert torch.equal(
        features,
        torch.tensor([[2.0, 3.0], [3.0, 3.0]]),
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
    with pytest.raises(ValueError, match="evaluation critic parent"):
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


def test_current_frame_critic_eval_restore_has_no_external_parent():
    source = _make_policy(critic_kind="fastwam")
    source.set_global_step(3)
    parent_sha256 = "a" * 64
    payload = {
        "schema": "fastwam-adaptive-rl-checkpoint-v1",
        "step": 3,
        "parent_checkpoint_sha256": parent_sha256,
        "critic_parent_checkpoint_sha256": None,
        "contract": {
            "model": {
                "actor_checkpoint_sha256": parent_sha256,
                "critic": {
                    "kind": "fastwam_current_frame_value",
                    "backbone_checkpoint_sha256": None,
                },
            }
        },
        "policy": source.trainable_state_dict(),
    }

    target = _make_policy(critic_kind="fastwam")
    assert (
        target.load_eval_checkpoint(
            payload,
            expected_parent_checkpoint_sha256=parent_sha256,
            expected_critic_parent_checkpoint_sha256=None,
        )
        == 3
    )
    assert target.actor_version == 3
    for expected, actual in zip(
        source.critic.value_head.parameters(),
        target.critic.value_head.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match="Expected pi0.5 critic parent"):
        _make_policy().load_eval_checkpoint(
            payload,
            expected_parent_checkpoint_sha256=parent_sha256,
            expected_critic_parent_checkpoint_sha256=None,
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


def test_current_frame_critic_reuses_rollout_features_and_bootstrap_is_route_pure():
    torch.manual_seed(41)
    policy = _make_policy(critic_kind="fastwam")
    obs = {
        "states": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "_fastwam_env_ids": torch.tensor([1, 2]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }

    _, rollout = policy.predict_action_batch(obs, mode="train")
    assert policy.runtime.critic_feature_calls == 0
    assert "fastwam_critic_features" in rollout["forward_inputs"]
    assert "critic_prefix" not in rollout["forward_inputs"]
    replay = policy.default_forward(
        rollout["forward_inputs"],
        route_info=rollout["route_info"],
        emitted_gate=rollout["emitted_gate"],
    )
    torch.testing.assert_close(replay["values"], rollout["prev_values"])

    replay["values"].sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in policy.critic.value_head.parameters()
    )
    assert all(parameter.grad is None for parameter in policy.actor.parameters())
    assert policy.gate.bias.grad is None
    assert policy.lora_adapter.parameter.grad is None

    route_state = policy.route_tracker.state_dict()
    bootstrap = policy.predict_value_batch(obs)
    assert bootstrap.shape == (2, 1)
    assert policy.runtime.critic_feature_calls == 1
    assert policy.route_tracker.state_dict() == route_state

    groups = policy.optimizer_parameter_groups(
        gate_lr=1e-4,
        lora_lr=2e-4,
        value_lr=3e-4,
    )
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert [group["name"] for group in groups] == [
        "gate",
        "uncond_lora",
        "value_head",
    ]
    assert len(parameter_ids) == len(set(parameter_ids))


def test_policy_sparse_handle_replay_preserves_eligible_gate_values():
    from rlinf.models.embodiment.wam_policy.tiered_kv_store import (
        GATE_KV_BATCH_INDICES,
    )

    policy = _make_policy()
    obs = {
        "states": torch.ones(2, 3),
        "_fastwam_env_ids": torch.tensor([1, 2]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }
    _, rollout = policy.predict_action_batch(obs, mode="train")
    full = policy.default_forward(
        rollout["forward_inputs"],
        route_info=rollout["route_info"],
        emitted_gate=rollout["emitted_gate"],
    )
    sparse_inputs = {}
    for key, value in rollout["forward_inputs"].items():
        sparse_inputs[key] = value[1:2] if key.startswith("gate_kv_") else value
    sparse_inputs[GATE_KV_BATCH_INDICES] = torch.tensor([1])

    sparse = policy.default_forward(
        sparse_inputs,
        route_info=rollout["route_info"],
        emitted_gate=rollout["emitted_gate"],
    )

    for key in (
        "gate_logprobs",
        "gate_entropy",
        "gate_base_probabilities",
        "gate_behavior_probabilities",
    ):
        torch.testing.assert_close(sparse[key][1], full[key][1])
        assert sparse[key][0].item() == 0
    torch.testing.assert_close(sparse["flow_logprobs"], full["flow_logprobs"])
    torch.testing.assert_close(sparse["values"], full["values"])


def test_sparse_handle_replay_matches_full_loss_gradients_and_optimizer_delta():
    from rlinf.models.embodiment.wam_policy.tiered_kv_store import (
        GATE_KV_BATCH_INDICES,
    )

    torch.manual_seed(2026)
    full_policy = _make_policy()
    torch.manual_seed(2026)
    sparse_policy = _make_policy()

    def bind_differentiable_flow(policy):
        flow_parameter = policy.lora_adapter.parameter

        def replay_action_batch(*, forward_inputs, route_info):
            del route_info
            batch_size = forward_inputs["critic_states"].shape[0]
            return {
                "flow_logprobs": flow_parameter.expand(batch_size, 2, 3),
                "flow_entropy": torch.ones(batch_size, 1, dtype=torch.float32),
            }

        policy.runtime.replay_action_batch = replay_action_batch

    bind_differentiable_flow(full_policy)
    bind_differentiable_flow(sparse_policy)
    observations = {
        "states": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "_fastwam_env_ids": torch.tensor([1, 2]),
        "_fastwam_reset_mask": torch.tensor([True, True]),
    }
    _, rollout = full_policy.predict_action_batch(observations, mode="train")

    full_replay = full_policy.default_forward(
        rollout["forward_inputs"],
        route_info=rollout["route_info"],
        emitted_gate=rollout["emitted_gate"],
    )
    sparse_inputs = {
        key: value[1:2] if key.startswith("gate_kv_") else value
        for key, value in rollout["forward_inputs"].items()
    }
    sparse_inputs[GATE_KV_BATCH_INDICES] = torch.tensor([1])
    sparse_replay = sparse_policy.default_forward(
        sparse_inputs,
        route_info=rollout["route_info"],
        emitted_gate=rollout["emitted_gate"],
    )

    for key in (
        "gate_logprobs",
        "gate_entropy",
        "gate_base_probabilities",
        "gate_behavior_probabilities",
    ):
        torch.testing.assert_close(sparse_replay[key][1], full_replay[key][1])
        assert sparse_replay[key][0].item() == 0
    torch.testing.assert_close(
        sparse_replay["flow_logprobs"], full_replay["flow_logprobs"]
    )
    torch.testing.assert_close(sparse_replay["values"], full_replay["values"])

    gate_old_logprobs = full_replay["gate_logprobs"].detach().clone()
    flow_old_logprobs = full_replay["flow_logprobs"].detach().clone()
    gate_advantages = torch.tensor([0.25, -0.75], dtype=torch.float32)
    flow_advantages = torch.tensor([0.5, -1.25], dtype=torch.float32)
    gate_valid_mask = torch.tensor([False, True])
    flow_valid_mask = torch.tensor([True, True])
    route_used = torch.tensor([0, 0], dtype=torch.long)
    value_targets = torch.tensor([[0.25], [-0.5]], dtype=torch.float32)

    def losses(replay):
        policy_loss, metrics = _dual_ppo.compute_fastwam_dual_ppo_loss(
            gate_logprobs=replay["gate_logprobs"],
            gate_old_logprobs=gate_old_logprobs,
            gate_advantages=gate_advantages,
            gate_valid_mask=gate_valid_mask,
            gate_clip_ratio_low=0.2,
            gate_clip_ratio_high=0.2,
            flow_logprobs=replay["flow_logprobs"],
            flow_old_logprobs=flow_old_logprobs,
            flow_advantages=flow_advantages,
            route_used=route_used,
            flow_clip_ratio_low=0.2,
            flow_clip_ratio_high=0.2,
            flow_valid_mask=flow_valid_mask,
            gate_behavior_probabilities=replay["gate_behavior_probabilities"],
            flow_entropy=replay["flow_entropy"],
        )
        value_loss = torch.nn.functional.mse_loss(replay["values"], value_targets)
        return policy_loss, value_loss, policy_loss + value_loss, metrics

    full_losses = losses(full_replay)
    sparse_losses = losses(sparse_replay)
    for full_loss, sparse_loss in zip(full_losses[:3], sparse_losses[:3], strict=True):
        torch.testing.assert_close(sparse_loss, full_loss)
    assert set(full_losses[3]) == set(sparse_losses[3])
    for key in full_losses[3]:
        torch.testing.assert_close(sparse_losses[3][key], full_losses[3][key])

    optimizer_kwargs = {
        "gate_lr": 3e-3,
        "lora_lr": 1e-3,
        "value_lr": 1e-2,
    }
    full_optimizer = torch.optim.SGD(
        full_policy.optimizer_parameter_groups(**optimizer_kwargs)
    )
    sparse_optimizer = torch.optim.SGD(
        sparse_policy.optimizer_parameter_groups(**optimizer_kwargs)
    )
    full_before = [
        parameter.detach().clone()
        for group in full_optimizer.param_groups
        for parameter in group["params"]
    ]
    sparse_before = [
        parameter.detach().clone()
        for group in sparse_optimizer.param_groups
        for parameter in group["params"]
    ]
    full_losses[2].backward()
    sparse_losses[2].backward()

    for full_group, sparse_group in zip(
        full_optimizer.param_groups,
        sparse_optimizer.param_groups,
        strict=True,
    ):
        assert full_group["name"] == sparse_group["name"]
        for full_parameter, sparse_parameter in zip(
            full_group["params"], sparse_group["params"], strict=True
        ):
            torch.testing.assert_close(sparse_parameter.grad, full_parameter.grad)

    full_optimizer.step()
    sparse_optimizer.step()
    full_after = [
        parameter.detach().clone()
        for group in full_optimizer.param_groups
        for parameter in group["params"]
    ]
    sparse_after = [
        parameter.detach().clone()
        for group in sparse_optimizer.param_groups
        for parameter in group["params"]
    ]
    for full_start, sparse_start, full_end, sparse_end in zip(
        full_before,
        sparse_before,
        full_after,
        sparse_after,
        strict=True,
    ):
        torch.testing.assert_close(sparse_start, full_start)
        torch.testing.assert_close(sparse_end, full_end)
        torch.testing.assert_close(sparse_end - sparse_start, full_end - full_start)
    assert all(
        not torch.equal(start, end) for start, end in zip(full_before, full_after)
    )


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
