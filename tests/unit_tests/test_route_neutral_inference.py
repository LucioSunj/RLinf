# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exercise inference reuse with real small VideoDiT, MoT, Gate and LoRA layers."""

from __future__ import annotations

import copy
import importlib.util
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from fastwam.adapters import (
    PolicyRegime,
    RegimeLoRAConfig,
    RegimeLoRALinear,
    inject_action_dit_lora,
    inject_video_bc_dit_lora,
)
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.batch_linear import (
    BatchInvariantLinear,
    install_batch_invariant_linears,
)
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.fastwam_idm import FastWAMIDM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fastwam.models.wan22.wan_video_dit import WanVideoDiT

from rlinf.models.embodiment.wam_policy.adaptive_policy import (
    FastWAMAdaptivePolicyConfig,
)
from rlinf.models.embodiment.wam_policy.contracts import ChunkRouteRecord
from rlinf.models.embodiment.wam_policy.critic import FastWAMValueTransformerConfig
from rlinf.models.embodiment.wam_policy.libero_runtime import LiberoFastWAMRuntime
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import OnlineIDMBCConfig
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_contracts import (
    RouteNeutralGateInputContract,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PadRouteNeutralCurrentStepGate,
    PadRouteNeutralGateConfig,
    PhysicalStateHistoryTracker,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.policy import (
    RouteNeutralOnlineIDMBCFastWAMPolicy,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.runtime import (
    RouteNeutralOnlineIDMTeacherLiberoRuntime,
)


class _TinyActor(torch.nn.Module):
    _append_proprio_to_context = FastWAM._append_proprio_to_context
    _build_mot_attention_mask = FastWAMIDM._build_mot_attention_mask
    _video_denoise_step_compiled = FastWAMIDM._video_denoise_step_compiled

    def __init__(self):
        super().__init__()
        common = {
            "hidden_dim": 16,
            "ffn_dim": 32,
            "text_dim": 8,
            "freq_dim": 8,
            "eps": 1e-6,
            "num_heads": 2,
            "attn_head_dim": 8,
            "num_layers": 2,
        }
        self.action_expert = ActionDiT(action_dim=7, **common)
        self.video_expert = WanVideoDiT(
            in_dim=2,
            out_dim=2,
            patch_size=(1, 1, 1),
            has_image_input=False,
            seperated_timestep=True,
            video_attention_mask_mode="first_frame_causal",
            **common,
        )
        self.mot = MoT(
            mixtures={"video": self.video_expert, "action": self.action_expert},
            mot_checkpoint_mixed_attn=False,
        )
        self.proprio_dim = 2
        self.text_dim = 8
        self.proprio_encoder = torch.nn.Linear(2, 8)
        self.vae = SimpleNamespace(
            model=SimpleNamespace(z_dim=2),
            upsampling_factor=2,
            temporal_downsample_factor=4,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
        self.calls = Counter()

    @property
    def device(self):
        return next(self.parameters()).device

    def _encode_input_image_latents_tensor(self, image, *, tiled):
        self.calls["vae"] += 1
        return image[:, :2, ::2, ::2].unsqueeze(2).contiguous()

    def encode_prompt(self, prompts):
        self.calls["text"] += 1
        context = (
            torch.tensor(
                [len(prompt) / 100 for prompt in prompts],
                device=self.device,
                dtype=next(self.parameters()).dtype,
            )[:, None, None]
            .expand(-1, 4, self.text_dim)
            .clone()
        )
        return context, torch.ones(context.shape[:2], dtype=torch.bool)


def _make_policy(monkeypatch, *, dual=False, checkpoint=False):
    torch.manual_seed(41)
    actor = _TinyActor()
    adapter = inject_action_dit_lora(
        actor.action_expert,
        RegimeLoRAConfig(rank=128 if dual else 2, alpha=128 if dual else 2),
    )
    video_adapter = (
        inject_video_bc_dit_lora(
            actor.video_expert,
            RegimeLoRAConfig(rank=128, alpha=128),
            regime_context=adapter.regime_context,
        )
        if dual
        else None
    )
    actor.video_expert.use_gradient_checkpointing = checkpoint
    actor.mot.mot_checkpoint_mixed_attn = checkpoint
    # Exercise a trained, nonzero UNCOND adapter rather than the zero-LoRA endpoint.
    with torch.no_grad():
        for parameter in adapter.lora_parameters():
            parameter.normal_(std=0.05)
        if video_adapter is not None:
            for parameter in video_adapter.lora_parameters():
                parameter.normal_(std=0.05)
    runtime = object.__new__(RouteNeutralOnlineIDMTeacherLiberoRuntime)
    LiberoFastWAMRuntime.__init__(
        runtime,
        actor=actor,
        lora_adapter=adapter,
        generation_horizon=4,
        execution_horizon=2,
        num_video_frames=9,
        num_inference_steps=3,
        gate_layer_indices=(0, 1),
        binarize_gripper=True,
    )
    runtime.video_lora_adapter = video_adapter
    runtime.batch_linear_context = install_batch_invariant_linears(actor)
    runtime.route_neutral_input = RouteNeutralGateInputContract(3, 2)
    runtime.route_neutral_visual = FastWAMValueTransformerConfig(
        num_mot_layers=2,
        source_num_heads=2,
        source_head_dim=8,
        layer_indices=(0, 1),
        sources=("current_frame_video",),
        hidden_dim=16,
        num_query_tokens=2,
    )
    runtime.physical_history = PhysicalStateHistoryTracker(runtime.route_neutral_input)
    runtime._evaluation_text_context = None
    monkeypatch.setattr(runtime, "_model_images", lambda obs: obs["image"])
    monkeypatch.setattr(runtime, "_normalized_proprio", lambda state: state)

    prefill = actor.mot.prefill_video_cache
    forward_action = actor.mot.forward_action_with_video_cache

    def counted_prefill(**kwargs):
        actor.calls["prefill"] += 1
        return prefill(**kwargs)

    def counted_action(**kwargs):
        actor.calls["action_steps"] += 1
        actor.calls["action_taps"] += int(kwargs.get("kv_tap") is not None)
        return forward_action(**kwargs)

    monkeypatch.setattr(actor.mot, "prefill_video_cache", counted_prefill)
    monkeypatch.setattr(actor.mot, "forward_action_with_video_cache", counted_action)
    gate = PadRouteNeutralCurrentStepGate(
        PadRouteNeutralGateConfig(
            visual=runtime.route_neutral_visual,
            language_dim=8,
            state_dim=2,
            history_length_chunks=3,
        )
    )
    return RouteNeutralOnlineIDMBCFastWAMPolicy(
        actor=actor,
        runtime=runtime,
        lora_adapter=adapter,
        video_lora_adapter=video_adapter,
        gate=gate,
        critic=None,
        config=FastWAMAdaptivePolicyConfig(),
        online_idm_bc_config=OnlineIDMBCConfig(enabled=True, loss_weight=0.2),
        critic_warmup={
            "runner_updates": 10,
            "route_behavior": "independent_random",
            "idm_probability": 0.5,
            "freeze_gate": True,
            "freeze_cost_controller": True,
        },
    ).eval()


@pytest.fixture
def policy(monkeypatch):
    return _make_policy(monkeypatch)


def _observation():
    return {
        "image": torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 48,
        "states": torch.tensor([[0.2, -0.3]]),
        "task_descriptions": ["open the drawer"],
        "_fastwam_env_ids": torch.tensor([0]),
        "_fastwam_reset_mask": torch.tensor([True]),
    }


def _dual_training_sample(policy):
    observation = _observation()
    observation = {
        key: value.repeat(2, *([1] * (value.ndim - 1)))
        if isinstance(value, torch.Tensor)
        else value * 2
        for key, value in observation.items()
    }
    observation.update(
        _fastwam_env_ids=torch.tensor([0, 1]),
        _fastwam_action_noise_seeds=torch.tensor([41, 42]),
        _fastwam_idm_noise_seeds=torch.tensor([51, 52]),
    )
    routes = torch.tensor([0, 1])
    route_info = ChunkRouteRecord(
        route_used=routes,
        route_was_forced=torch.zeros(2, dtype=torch.bool),
        chunk_ids=torch.zeros(2, dtype=torch.long),
        episode_ids=torch.zeros(2, dtype=torch.long),
        route_source_chunk_ids=torch.zeros(2, dtype=torch.long),
        actor_versions=torch.full((2,), 10, dtype=torch.long),
    )
    prepared = policy.runtime.prepare_route_neutral_step(env_obs=observation)
    sample = policy.runtime.sample_routed_action_batch(
        env_obs=observation,
        routes=routes,
        actor_version=10,
        mode="train",
        collect_replay=True,
        prepared=prepared,
    )
    sample.forward_inputs.update(
        flow_chains=sample.flow_chains,
        denoise_indices=sample.denoise_indices,
        online_idm_bc_flow_valid=routes == 0,
    )
    return prepared, sample, route_info


@pytest.mark.parametrize("checkpoint", [False, True])
def test_dual_lora_flow_replay_and_bc_reach_both_branches(monkeypatch, checkpoint):
    from rlinf.models.embodiment.wam_policy.online_idm_bc.actor import (
        audit_online_idm_bc_backward_gradient_ownership,
        audit_online_idm_bc_gradient_ownership,
    )

    policy = _make_policy(monkeypatch, dual=True, checkpoint=checkpoint).train()
    _, sample, routes = _dual_training_sample(policy)
    replay = policy.runtime.replay_action_batch(
        forward_inputs=sample.forward_inputs,
        route_info=routes,
    )
    torch.testing.assert_close(
        replay["flow_logprobs"], sample.old_flow_logprobs, atol=1e-6, rtol=1e-6
    )
    flow_metrics = audit_online_idm_bc_gradient_ownership(
        bc_loss=replay["flow_logprobs"].sum(), policy=policy
    )
    assert flow_metrics["online_idm_bc/gradient_audit_video_lora_nonzero_count"] > 0
    assert flow_metrics["online_idm_bc/gradient_audit_action_lora_nonzero_count"] > 0
    bc = policy.runtime.compute_online_idm_bc_loss(
        forward_inputs=sample.forward_inputs, route_info=routes
    )
    bc_metrics = audit_online_idm_bc_gradient_ownership(
        bc_loss=bc.loss_sum, policy=policy
    )
    assert bc_metrics["online_idm_bc/gradient_audit_video_lora_nonzero_count"] > 0
    assert bc_metrics["online_idm_bc/gradient_audit_action_lora_nonzero_count"] > 0
    policy.critic = torch.nn.Module()
    policy.critic.value_head = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        policy.optimizer_parameter_groups(gate_lr=3e-5, lora_lr=1e-5, value_lr=1e-4)
    )
    before = {
        name: adapter.lora_state_dict()
        for name, adapter in policy.lora_adapters.items()
    }
    # The production FSDP audit builds a fresh isolated BC graph for backward.
    bc = policy.runtime.compute_online_idm_bc_loss(
        forward_inputs=sample.forward_inputs, route_info=routes
    )
    bc.loss_sum.backward()
    audit_online_idm_bc_backward_gradient_ownership(optimizer=optimizer, policy=policy)
    optimizer.step()
    for name, adapter in policy.lora_adapters.items():
        assert any(
            not torch.equal(before[name][key], value)
            for key, value in adapter.lora_state_dict().items()
        )
    assert policy.lora_adapter.regime_context.current is PolicyRegime.IDM
    assert not policy.actor.mot.training
    assert not policy.actor.video_expert.training


def test_dual_video_changes_uncond_but_not_parent_gate_critic_or_idm(monkeypatch):
    policy = _make_policy(monkeypatch, dual=True)
    policy.runtime.critic_feature_config = policy.runtime.route_neutral_visual
    before, before_sample, _ = _dual_training_sample(policy)
    with torch.no_grad():
        for parameter in policy.video_lora_adapter.lora_parameters():
            parameter.add_(0.1)
    after, after_sample, _ = _dual_training_sample(policy)
    _assert_equal(asdict(before.gate_features), asdict(after.gate_features))
    _assert_equal(asdict(before.critic_features), asdict(after.critic_features))
    assert torch.equal(before_sample.actions[1], after_sample.actions[1])
    assert not torch.equal(before_sample.actions[0], after_sample.actions[0])
    assert torch.equal(
        before_sample.forward_inputs["online_idm_bc_teacher_actions"],
        after_sample.forward_inputs["online_idm_bc_teacher_actions"],
    )


def test_dual_native_checkpoint_and_optimizer_own_both_adapters(monkeypatch):
    from rlinf.workers.actor.fastwam_selective_sync import capture_fastwam_sync_tensors

    policy = _make_policy(monkeypatch, dual=True)
    policy.critic = torch.nn.Module()
    policy.critic.value_head = torch.nn.Linear(2, 1)
    original = copy.deepcopy(policy.trainable_state_dict())
    assert original["schema"] == "fastwam-adaptive-policy-dual-lora-v1"
    groups = policy.optimizer_parameter_groups(
        gate_lr=1e-4, lora_lr=1e-4, value_lr=1e-4
    )
    lora_group = next(group for group in groups if group["name"] == "uncond_lora")
    assert {id(p) for p in lora_group["params"]} == {
        id(p)
        for adapter in policy.lora_adapters.values()
        for p in adapter.lora_parameters()
    }
    synchronized = capture_fastwam_sync_tensors(policy)
    assert {id(p) for p in lora_group["params"]} <= {
        id(entry.tensor) for entry in synchronized.values()
    }
    with torch.no_grad():
        for parameter in policy.parameters():
            if parameter.requires_grad:
                parameter.add_(1.0)
    policy.load_trainable_state_dict(original)
    _assert_equal(original, policy.trainable_state_dict())
    action_only = _make_policy(monkeypatch)
    with pytest.raises(ValueError, match="checkpoint keys changed"):
        action_only.load_trainable_state_dict(original)


def test_dual_lora_rejects_action_only_inference_merge(monkeypatch):
    policy = _make_policy(monkeypatch, dual=True)
    policy.enable_inference_acceleration(compile=False)
    with pytest.raises(ValueError, match="requires eager inference"):
        policy.predict_action_batch(_observation(), mode="eval")


def test_continuation_ledger_keeps_stochastic_gate_draws_and_actions(policy):
    policy.config = replace(
        policy.config, eval_routing_mode="stochastic_keyed", eval_routing_seed=43
    )
    initial = copy.deepcopy(policy.route_tracker.state_dict())
    continued = copy.deepcopy(initial)
    continued["current_step"]["next_episode_ids"] = {0: 86}
    policy.route_tracker.load_state_dict(continued)
    observation = {
        **_observation(),
        "_fastwam_action_noise_seeds": torch.tensor([604]),
        "_fastwam_idm_noise_seeds": torch.tensor([605]),
    }
    expected_actions, expected = policy.predict_action_batch(observation, mode="eval")

    policy.route_tracker.load_state_dict(initial)
    resumed_actions, resumed = policy.predict_action_batch(
        {**observation, "_fastwam_evaluation_episode_ids": torch.tensor([86])},
        mode="eval",
    )

    assert resumed["route_info"].episode_ids.tolist() == [86]
    assert resumed["route_info"].chunk_ids.tolist() == [0]
    assert torch.equal(
        expected["evaluation_selection"].random_draws,
        resumed["evaluation_selection"].random_draws,
    )
    assert torch.equal(
        expected["route_info"].route_used, resumed["route_info"].route_used
    )
    assert torch.equal(expected_actions, resumed_actions)


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _assert_equal(lhs, rhs)
    else:
        assert left == right


def _calibrated_atol(first, repeated, floor):
    repeat_error = float((first.float() - repeated.float()).abs().max())
    return max(floor, 1.25 * repeat_error)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "routing", ["forced_idm", "forced_uncond", "learned_threshold"]
)
def test_merged_inference_preserves_base_gate_rng_and_bounded_actions(
    policy, monkeypatch, dtype, routing
):
    policy.config = replace(policy.config, eval_routing_mode=routing)
    policy.actor.to(dtype=dtype)
    for parameter in policy.lora_adapter.lora_parameters():
        parameter.data = parameter.data.float()
    obs = _observation()
    obs["image"] = obs["image"].to(dtype)
    rng = torch.get_rng_state().clone()
    initial_history = policy.route_tracker.state_dict()
    weights = {key: value.clone() for key, value in policy.state_dict().items()}
    parameter_ids = [id(parameter) for parameter in policy.parameters()]
    original_action_freqs = policy.actor.action_expert.freqs
    original_video_freqs = policy.actor.video_expert.freqs
    futures = []
    prefill = LiberoFastWAMRuntime._prefill_video_condition

    def capture_future(self, **kwargs):
        futures.append(kwargs["video_latents"].clone())
        return prefill(self, **kwargs)

    monkeypatch.setattr(
        LiberoFastWAMRuntime, "_prefill_video_condition", capture_future
    )
    actions, result = policy.predict_action_batch(obs, mode="eval")
    baseline_rng = torch.get_rng_state().clone()
    baseline_history = policy.route_tracker.state_dict()
    baseline_futures = list(futures)
    futures.clear()
    torch.set_rng_state(rng)
    policy.route_tracker.load_state_dict(initial_history)
    policy.enable_inference_acceleration(compile=False)
    merged_actions, merged_result = policy.predict_action_batch(obs, mode="eval")
    engine = policy._inference_engine
    assert engine is not None
    assert policy.actor.action_expert.freqs is original_action_freqs
    assert policy.actor.video_expert.freqs is original_video_freqs
    assert engine.uncond_action_expert.freqs is engine.idm_action_expert.freqs
    assert engine.runtime.actor.mot.mixtures["action"] is engine.idm_action_expert
    assert (
        engine.runtime.actor.mot.mixtures["video"] is engine.runtime.actor.video_expert
    )
    for expert in (engine.idm_action_expert, engine.uncond_action_expert):
        assert not any(
            isinstance(layer, (RegimeLoRALinear, BatchInvariantLinear))
            for layer in expert.modules()
        )
    for name, layer in policy.lora_adapter.iter_adapted_linears():
        plain = engine.idm_action_expert.get_submodule(name)
        merged = engine.uncond_action_expert.get_submodule(name)
        assert plain.weight is layer.weight
        expected = (
            layer.weight.float()
            + (layer.lora_B.float() @ layer.lora_A.float()) * layer.scaling
        )
        assert torch.equal(merged.weight, expected.to(dtype))
        assert merged.weight.data_ptr() != layer.weight.data_ptr()
    assert engine.merged_projection_count == len(policy.lora_adapter.target_names)
    assert engine.additional_weight_bytes > 0
    # FP32 merging changes operation order; BF16 additionally rounds the merged
    # weight once. Declare a two-BF16-ULP absolute floor for normalized actions.
    atol = 1e-5 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
    if routing == "forced_idm":
        assert torch.equal(actions, merged_actions)
    else:
        torch.testing.assert_close(actions, merged_actions, rtol=0, atol=atol)
    assert torch.equal(
        result["emitted_gate"].base_probability,
        merged_result["emitted_gate"].base_probability,
    )
    assert torch.equal(
        result["route_info"].route_used, merged_result["route_info"].route_used
    )
    _assert_equal(futures, baseline_futures)
    _assert_equal(policy.route_tracker.state_dict(), baseline_history)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    _assert_equal(policy.state_dict(), weights)
    assert parameter_ids == [id(parameter) for parameter in policy.parameters()]


@pytest.mark.parametrize("mutation", ["load_state", "device", "version", "train"])
def test_inference_view_is_invalidated_before_model_changes(policy, mutation):
    policy.enable_inference_acceleration(compile=False)
    policy.predict_action_batch(_observation(), mode="eval")
    old_engine = policy._inference_engine
    assert old_engine is not None
    if mutation == "load_state":
        state = {key: value.clone() for key, value in policy.state_dict().items()}
        lora_key = next(key for key in state if key.endswith("lora_B"))
        state[lora_key].add_(0.1)
        policy.load_state_dict(state)
    elif mutation == "device":
        policy.to("cpu")
    elif mutation == "version":
        policy.set_global_step(3)
    else:
        policy.train()
        policy.eval()
    assert policy._inference_engine is None
    policy.predict_action_batch(_observation(), mode="eval")
    assert policy._inference_engine is not old_engine


def test_native_eval_reload_remerges_new_lora_and_resets_history(policy):
    policy.enable_inference_acceleration(compile=False)
    policy.predict_action_batch(_observation(), mode="eval")
    old_engine = policy._inference_engine
    lora = policy.lora_adapter.lora_state_dict()
    changed_name = next(name for name in lora if name.endswith("lora_B"))
    lora[changed_name].add_(0.1)
    parent = "a" * 64
    payload = {
        "schema": "fastwam-adaptive-rl-checkpoint-v1",
        "parent_checkpoint_sha256": parent,
        "contract": {"model": {"actor_checkpoint_sha256": parent}},
        "step": 12,
        "policy": {
            "schema": "fastwam-adaptive-policy-v1",
            "actor_version": 12,
            "gate": policy.gate.state_dict(),
            "lora": lora,
            "value_head": {},
            "route_tracker": policy.route_tracker.state_dict(),
        },
    }
    assert (
        policy.load_eval_checkpoint(payload, expected_parent_checkpoint_sha256=parent)
        == 12
    )
    assert policy._inference_engine is None
    assert policy.runtime.physical_history.state_dict()["states"] == {}
    policy.predict_action_batch(_observation(), mode="eval")
    assert (
        policy._inference_engine.runtime.physical_history
        is policy.runtime.physical_history
    )
    layer_name = changed_name.removesuffix(".lora_B")
    old = old_engine.uncond_action_expert.get_submodule(layer_name).weight
    new = policy._inference_engine.uncond_action_expert.get_submodule(layer_name).weight
    assert not torch.equal(old, new)
    # A caller loading the existing native endpoint needs no new config field
    # to select accelerated standalone inference. Explicit options above survive.
    policy.inference_acceleration = None
    policy.load_eval_checkpoint(payload, expected_parent_checkpoint_sha256=parent)
    assert policy.inference_acceleration.merge_lora
    assert policy.inference_acceleration.compile
    assert policy._inference_engine is None


def test_acceleration_is_eval_only(policy, monkeypatch):
    policy.enable_torch_compile()

    def training_prepare(**kwargs):
        assert kwargs["mode"] == "train"
        assert policy._inference_engine is None
        raise RuntimeError("Reached original training runtime")

    monkeypatch.setattr(policy.runtime, "prepare_route_neutral_step", training_prepare)
    with pytest.raises(RuntimeError, match="Reached original training runtime"):
        policy.predict_action_batch(_observation(), mode="train", compute_values=False)


def test_warmup_covers_both_routes_and_preserves_control_state(policy):
    config = policy.config
    rng = torch.get_rng_state().clone()
    history = policy.route_tracker.state_dict()
    policy.enable_inference_acceleration(compile=False)
    times = policy.warmup_inference(_observation(), repetitions=1)
    assert set(times) == {"forced_uncond", "forced_idm"}
    assert all(len(values) == 1 and values[0] > 0 for values in times.values())
    assert policy._inference_engine is not None
    engine = policy._inference_engine
    policy.set_global_step(policy.actor_version)
    assert policy._inference_engine is engine
    assert policy.config is config
    _assert_equal(policy.route_tracker.state_dict(), history)
    _assert_equal(torch.get_rng_state(), rng)


def test_eval_contract_allows_acceleration_but_retains_scientific_fields():
    from rlinf.config_contracts import validate_fastwam_eval_model_contract

    source = {"lora": {"rank": 16}, "num_inference_steps": 10}
    live = {
        **source,
        "inference_acceleration": {"merge_lora": True, "compile": True},
    }
    validate_fastwam_eval_model_contract(source, live, load_critic=False)
    live["num_inference_steps"] = 9
    with pytest.raises(ValueError, match="num_inference_steps"):
        validate_fastwam_eval_model_contract(source, live, load_critic=False)


@pytest.mark.parametrize("routing", ["forced_idm", "forced_uncond"])
def test_torch_compile_captures_actual_inference_kernels(policy, monkeypatch, routing):
    # Use the real Dynamo frontend on every production boundary. A recording
    # backend executes FX graphs on CPU; Inductor lowering is checked separately.
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    from torch._dynamo.backends.registry import register_backend

    name = f"route_neutral_test_{routing}"
    register_backend(backend, name=name)
    # Counter instrumentation belongs outside compiled tensor functions.
    monkeypatch.setattr(
        policy.actor,
        "_encode_input_image_latents_tensor",
        lambda image, *, tiled: image[:, :2, ::2, ::2].unsqueeze(2).contiguous(),
    )
    policy.config = replace(policy.config, eval_routing_mode=routing)
    policy.enable_inference_acceleration(compile=False)
    rng = torch.get_rng_state().clone()
    obs = _observation()
    obs["_fastwam_action_noise_seeds"] = torch.tensor([13])
    obs["_fastwam_idm_noise_seeds"] = torch.tensor([19])
    expected, expected_result = policy.predict_action_batch(obs, mode="eval")
    repeated, repeated_result = policy.predict_action_batch(obs, mode="eval")
    action_atol = _calibrated_atol(expected, repeated, 1e-5)
    gate_atol = _calibrated_atol(
        expected_result["emitted_gate"].base_probability,
        repeated_result["emitted_gate"].base_probability,
        1e-6,
    )
    policy.enable_inference_acceleration(backend=name)
    actual, result = policy.predict_action_batch(obs, mode="eval")
    assert graphs
    torch.testing.assert_close(actual, expected, atol=action_atol, rtol=0)
    torch.testing.assert_close(
        result["emitted_gate"].base_probability,
        expected_result["emitted_gate"].base_probability,
        atol=gate_atol,
        rtol=0,
    )
    count = len(graphs)
    # Changes in tensor values/history must not cause per-chunk recompilation.
    obs["image"] = obs["image"] + 0.01
    obs["states"] = obs["states"] + 0.02
    policy.predict_action_batch(obs, mode="eval")
    assert len(graphs) == count
    assert torch.equal(torch.get_rng_state(), rng)


def test_inductor_matches_merged_eager_with_real_vae(policy, monkeypatch):
    from fastwam.models.wan22.wan_video_vae import VideoVAE_, WanVideoVAE

    # Exercise the production VAE encoder and temporal-cache implementation at
    # small dimensions, not the counting substitute used by reuse tests.
    vae = WanVideoVAE.__new__(WanVideoVAE)
    torch.nn.Module.__init__(vae)
    vae.model = VideoVAE_(
        dim=4,
        z_dim=2,
        dim_mult=[1, 1],
        num_res_blocks=1,
        temperal_downsample=[False],
    ).eval()
    vae.scale = [torch.zeros(2), torch.ones(2)]
    vae.upsampling_factor = 2
    vae.temporal_downsample_factor = 4
    policy.actor.vae = vae
    monkeypatch.setattr(
        _TinyActor,
        "_encode_input_image_latents_tensor",
        FastWAM._encode_input_image_latents_tensor,
    )
    obs = _observation()
    obs["_fastwam_action_noise_seeds"] = torch.tensor([13])
    obs["_fastwam_idm_noise_seeds"] = torch.tensor([19])
    futures = []
    prefill = LiberoFastWAMRuntime._prefill_video_condition

    def capture_future(self, **kwargs):
        futures.append(kwargs["video_latents"].clone())
        return prefill(self, **kwargs)

    monkeypatch.setattr(
        LiberoFastWAMRuntime, "_prefill_video_condition", capture_future
    )
    expected = {}
    policy.enable_inference_acceleration(compile=False)
    for routing in ("forced_idm", "forced_uncond"):
        policy.config = replace(policy.config, eval_routing_mode=routing)
        futures.clear()
        actions, result = policy.predict_action_batch(obs, mode="eval")
        probability = result["emitted_gate"].base_probability
        videos = list(futures)
        futures.clear()
        repeated, repeated_result = policy.predict_action_batch(obs, mode="eval")
        expected[routing] = (
            actions,
            probability,
            videos,
            _calibrated_atol(actions, repeated, 1e-5),
            _calibrated_atol(
                probability, repeated_result["emitted_gate"].base_probability, 1e-6
            ),
            [
                _calibrated_atol(first, second, 1e-5)
                for first, second in zip(videos, futures, strict=True)
            ],
        )
    policy.enable_inference_acceleration(backend="inductor")
    for routing in ("forced_idm", "forced_uncond"):
        policy.config = replace(policy.config, eval_routing_mode=routing)
        futures.clear()
        actual, result = policy.predict_action_batch(obs, mode="eval")
        actions, probability, videos, action_atol, gate_atol, video_atols = expected[
            routing
        ]
        torch.testing.assert_close(actual, actions, atol=action_atol, rtol=0)
        torch.testing.assert_close(
            result["emitted_gate"].base_probability, probability, atol=gate_atol, rtol=0
        )
        for lhs, rhs, atol in zip(futures, videos, video_atols, strict=True):
            torch.testing.assert_close(lhs, rhs, atol=atol, rtol=0)
    assert policy._inference_engine.config.compile
    assert policy._inference_engine.config.backend == "inductor"
    old_frame = futures[0].clone()
    obs["image"] = obs["image"] + 0.3
    futures.clear()
    actual, _ = policy.predict_action_batch(obs, mode="eval")
    assert not torch.equal(futures[0], old_frame)
    # The native encoder clears its feature caches for the changed image too.
    policy.inference_acceleration = None
    eager, _ = policy.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(actual, eager, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "routing", ["forced_uncond", "forced_idm", "learned_threshold"]
)
@pytest.mark.parametrize("noise", ["seeds", "injected", "global_rng"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_eval_matches_legacy_actions_gate_future_and_rng(
    policy, monkeypatch, routing, noise, dtype
):
    policy.config = replace(policy.config, eval_routing_mode=routing)
    policy.actor.to(dtype=dtype)
    for parameter in policy.lora_adapter.lora_parameters():
        parameter.data = parameter.data.float()
    obs = _observation()
    obs["image"] = obs["image"].to(dtype=dtype)
    if noise == "seeds":
        obs.update(
            _fastwam_action_noise_seeds=torch.tensor([101]),
            _fastwam_idm_noise_seeds=torch.tensor([203]),
        )
    elif noise == "injected":
        obs.update(
            _fastwam_action_initial_noise=torch.randn(1, 4, 7),
            _fastwam_idm_initial_latents=torch.randn(1, 2, 3, 2, 2),
        )
    initial_rng = torch.get_rng_state().clone()
    initial_state = policy.route_tracker.state_dict()
    parameters = {name: value.clone() for name, value in policy.state_dict().items()}
    runtime = policy.runtime
    prepare = runtime.prepare_route_neutral_step
    sample = runtime.sample_routed_action_batch
    video_inputs = []
    original_pre_dit = policy.actor.video_expert.pre_dit

    def capture_video(**kwargs):
        if kwargs["x"].shape[2] > 1:
            video_inputs.append(kwargs["x"].clone())
        return original_pre_dit(**kwargs)

    monkeypatch.setattr(policy.actor.video_expert, "pre_dit", capture_video)
    with monkeypatch.context() as baseline:
        baseline.setattr(
            runtime,
            "prepare_route_neutral_step",
            lambda **kw: prepare(**{**kw, "mode": "train"}),
        )
        baseline.setattr(
            runtime,
            "sample_routed_action_batch",
            lambda **kw: sample(**{**kw, "prepared": None}),
        )
        old_actions, old_result = policy.predict_action_batch(obs, mode="eval")
    old_rng = torch.get_rng_state().clone()
    old_state = policy.route_tracker.state_dict()
    old_video_inputs = list(video_inputs)
    assert policy.actor.calls == {
        "text": 2,
        "vae": 2,
        "prefill": 2,
        "action_steps": 3,
        "action_taps": 3,
    }

    policy.route_tracker.load_state_dict(initial_state)
    torch.set_rng_state(initial_rng)
    policy.actor.calls.clear()
    video_inputs.clear()
    new_actions, new_result = policy.predict_action_batch(obs, mode="eval")
    _assert_equal(old_actions, new_actions)
    _assert_equal(
        asdict(old_result["emitted_gate"]), asdict(new_result["emitted_gate"])
    )
    _assert_equal(asdict(old_result["route_info"]), asdict(new_result["route_info"]))
    _assert_equal(old_video_inputs, video_inputs)
    _assert_equal(old_rng, torch.get_rng_state())
    _assert_equal(old_state, policy.route_tracker.state_dict())
    _assert_equal(parameters, policy.state_dict())
    assert not new_actions.requires_grad
    assert not new_result["emitted_gate"].base_probability.requires_grad
    assert new_result["forward_inputs"] == {}
    assert new_result["prev_logprobs"].numel() == 0
    assert policy.actor.calls == {
        "text": 1,
        "vae": 1,
        "prefill": 2 if int(new_result["route_info"].route_used[0]) else 1,
        "action_steps": 3,
        "action_taps": 0,
    }


def test_resident_text_refreshes_proprio_images_and_prompt(policy):
    runtime = policy.runtime
    obs = _observation()
    first = runtime.prepare_route_neutral_step(
        env_obs=obs, mode="eval", include_critic_features=False
    )
    cached_text = runtime._evaluation_text_context[1][0].clone()
    obs["states"] = obs["states"] + 1
    obs["image"] = obs["image"] + 0.5
    obs["_fastwam_reset_mask"] = torch.tensor([False])
    second = runtime.prepare_route_neutral_step(
        env_obs=obs, mode="eval", include_critic_features=False
    )
    assert policy.actor.calls["text"] == 1
    assert policy.actor.calls["vae"] == 2
    assert torch.equal(first.context[:, :-1], second.context[:, :-1])
    assert not torch.equal(first.context[:, -1], second.context[:, -1])
    assert not torch.equal(first.first_frame_latents, second.first_frame_latents)
    assert torch.equal(runtime._evaluation_text_context[1][0], cached_text)
    assert torch.equal(
        second.gate_features.physical_history[:, -1], first.gate_features.state
    )
    obs["task_descriptions"] = ["close the much longer drawer"]
    third = runtime.prepare_route_neutral_step(
        env_obs=obs, mode="eval", include_critic_features=False
    )
    assert policy.actor.calls["text"] == 2
    assert not torch.equal(second.context[:, :-1], third.context[:, :-1])


@pytest.mark.parametrize(
    "field,seed_field",
    [
        ("_fastwam_action_initial_noise", "_fastwam_action_noise_seeds"),
        ("_fastwam_idm_initial_latents", "_fastwam_idm_noise_seeds"),
    ],
)
def test_eval_preserves_noise_override_exclusivity(policy, field, seed_field):
    obs = _observation()
    obs[field] = (
        torch.zeros(1, 4, 7) if "action" in field else torch.zeros(1, 2, 3, 2, 2)
    )
    obs[seed_field] = torch.tensor([1])
    with pytest.raises(ValueError, match="not both"):
        policy.predict_action_batch(obs, mode="eval")


def test_multi_environment_eval_keeps_serial_action_path(policy, monkeypatch):
    obs = _observation()
    for key, value in list(obs.items()):
        obs[key] = torch.cat((value, value)) if torch.is_tensor(value) else value * 2
    obs["_fastwam_env_ids"] = torch.tensor([0, 1])
    policy.config = replace(policy.config, eval_routing_mode="forced_uncond")
    monkeypatch.setattr(
        policy.runtime,
        "_sample_prepared_evaluation",
        lambda **_: pytest.fail("B1 reuse reached B2"),
    )
    actions, result = policy.predict_action_batch(obs, mode="eval")
    assert actions.shape == (2, 2, 7)
    assert result["forward_inputs"] == {}
    assert policy.actor.calls["vae"] == 3
    assert policy.actor.calls["prefill"] == 3


def test_direct_eval_disables_autograd_without_changing_train_context(
    policy, monkeypatch
):
    grad_modes = []

    def predict(_self, **kwargs):
        grad_modes.append(torch.is_grad_enabled())
        return torch.empty(1), {}

    monkeypatch.setattr(policy, "_predict_current_step", MethodType(predict, policy))
    with torch.enable_grad():
        policy.predict_action_batch(_observation(), mode="eval")
        policy.predict_action_batch(_observation(), mode="train")
        assert torch.is_grad_enabled()
    assert grad_modes == [False, True]


@pytest.mark.parametrize("merge", [False, True])
def test_benchmark_runs_complete_policy_and_restores_state(policy, merge):
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts/adaptive_gate/benchmark_route_neutral_inference.py"
    )
    spec = importlib.util.spec_from_file_location(
        "route_neutral_inference_benchmark", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial_config = policy.config
    initial_rng = torch.get_rng_state().clone()
    initial_state = policy.route_tracker.state_dict()
    acceleration = {"merge_lora": True, "compile": False} if merge else None
    report = module.benchmark_policy(
        policy,
        _observation(),
        warmups=1,
        repetitions=3,
        atol=None if merge else 1e-6,
        acceleration=acceleration,
    )
    assert report["status"] == "PASS"
    assert policy.config is initial_config
    assert policy.inference_acceleration is None
    assert policy._inference_engine is None
    assert policy.runtime._evaluation_text_context is None
    _assert_equal(initial_rng, torch.get_rng_state())
    _assert_equal(initial_state, policy.route_tracker.state_dict())
    for routing, arm in report["routes"].items():
        assert all(
            value <= arm["parity_bounds"][key]
            for key, value in arm["parity_max_abs"].items()
        )
        assert "generated_actions" in arm["parity_max_abs"]
        assert "gate_probability" in arm["parity_max_abs"]
        if routing == "forced_idm":
            assert "future_latents" in arm["parity_max_abs"]
        assert all(len(values) == 3 for values in arm["seconds"].values())
        assert all(value > 0 for value in arm["median_seconds"].values())
    assert policy.actor.calls["action_taps"] > 0
    assert policy.actor.calls["vae"] > policy.actor.calls["text"]


def test_benchmark_retains_failed_measurements_and_restores_state(policy, monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts/adaptive_gate/benchmark_route_neutral_inference.py"
    )
    spec = importlib.util.spec_from_file_location("inference_failure_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    max_abs = module._max_abs
    comparisons = 0

    def force_failure(first, second):
        nonlocal comparisons
        differences = max_abs(first, second)
        comparisons += 1
        if comparisons == 2:
            differences["generated_actions"] = 1.0
        return differences

    monkeypatch.setattr(module, "_max_abs", force_failure)
    config = policy.config
    rng = torch.get_rng_state().clone()
    history = policy.route_tracker.state_dict()
    with pytest.raises(module.InferenceParityError) as caught:
        module.benchmark_policy(
            policy,
            _observation(),
            warmups=1,
            repetitions=3,
            acceleration={"merge_lora": True, "compile": False},
        )
    report = caught.value.report
    assert report["status"] == "FAIL_PARITY"
    assert report["routes"]["forced_uncond"]["parity_max_abs"]["generated_actions"] == 1
    assert "speedup" not in report["routes"]["forced_uncond"]
    assert policy.config is config
    assert policy.inference_acceleration is None
    assert policy._inference_engine is None
    _assert_equal(torch.get_rng_state(), rng)
    _assert_equal(policy.route_tracker.state_dict(), history)
