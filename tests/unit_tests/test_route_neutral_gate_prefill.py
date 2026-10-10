# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Check Gate-only prefill against complete real Transformer computations."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, replace

import pytest
import test_route_neutral_inference as inference_fixtures
import torch
from fastwam.adapters import PolicyRegime

from rlinf.models.embodiment.wam_policy.critic import extract_fastwam_value_features
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    RouteNeutralVisualFeatures,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.inference import (
    InferenceAccelerationConfig,
    RouteNeutralInference,
)


class _FourLayerActor(inference_fixtures._TinyActor):
    """Keep the fixture's two Gate taps and add two unused later layers."""

    def __init__(self):
        super().__init__()
        for expert in (self.video_expert, self.action_expert):
            expert.blocks.extend(copy.deepcopy(list(expert.blocks)))
        self.mot.num_layers = 4


def _make_policy(monkeypatch, *, dual=True):
    monkeypatch.setattr(inference_fixtures, "_TinyActor", _FourLayerActor)
    return inference_fixtures._make_policy(monkeypatch, dual=dual)


def _observation(batch_size=1):
    observation = inference_fixtures._observation()
    if batch_size > 1:
        observation = {
            name: value.repeat(batch_size, *([1] * (value.ndim - 1)))
            if isinstance(value, torch.Tensor)
            else value * batch_size
            for name, value in observation.items()
        }
        observation["_fastwam_env_ids"] = torch.arange(batch_size)
    return observation


def _legacy_visual_features(runtime, observation):
    images, context, mask = runtime._encode_condition(observation)
    condition, _ = runtime._prepare_parent_current_condition(
        image=images, context=context, context_mask=mask
    )
    features = extract_fastwam_value_features(
        condition,
        mot=runtime.actor.mot,
        action_expert=runtime.actor.action_expert,
        config=runtime.route_neutral_visual,
    )
    return condition, RouteNeutralVisualFeatures.from_value_features(features)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("merged", [False, True])
def test_gate_prefix_matches_full_cache_and_skips_unused_work(
    monkeypatch, dtype, merged
):
    policy = _make_policy(monkeypatch)
    policy.actor.to(dtype=dtype)
    for adapter in policy.lora_adapters.values():
        for parameter in adapter.lora_parameters():
            parameter.data = parameter.data.float()
    runtime = policy.runtime
    if merged:
        runtime = RouteNeutralInference(
            runtime, policy.gate, InferenceAccelerationConfig(compile=False)
        ).runtime
    observation = _observation()
    observation["image"] = observation["image"].to(dtype)
    _, expected = _legacy_visual_features(runtime, observation)
    calls = Counter()
    handles = []
    for index, block in enumerate(runtime.actor.video_expert.blocks):
        for name, module in (("key", block.self_attn.k), ("ffn", block.ffn)):
            handles.append(
                module.register_forward_hook(
                    lambda *_, key=(name, index): calls.update([key])
                )
            )

    def unused_projection(*args, **kwargs):
        raise AssertionError("Gate-only features must not project Action context.")

    monkeypatch.setattr(
        runtime.actor.action_expert.text_embedding, "forward", unused_projection
    )
    monkeypatch.setattr(runtime.actor.mot, "_project_context_bank", unused_projection)
    try:
        prepared = runtime.prepare_route_neutral_step(
            env_obs=observation, mode="eval", include_critic_features=False
        )
    finally:
        for handle in handles:
            handle.remove()
    inference_fixtures._assert_equal(
        asdict(prepared.gate_features.visual), asdict(expected)
    )
    assert calls == {("key", 0): 1, ("ffn", 0): 1, ("key", 1): 1}
    assert len(prepared.current_condition.video_kv_cache) == 2
    assert runtime.actor.mot.num_layers == 4
    assert len(runtime.actor.video_expert.blocks) == 4
    assert len(runtime.actor.action_expert.blocks) == 4
    assert runtime.lora_adapter.regime_context.current is PolicyRegime.IDM
    reference = replace(prepared.gate_features, visual=expected)
    with torch.no_grad():
        assert torch.equal(policy.gate(prepared.gate_features), policy.gate(reference))


@pytest.mark.parametrize(
    ("dual", "mode", "batch_size"),
    [(False, "eval", 1), (True, "train", 1), (True, "train", 2), (True, "eval", 2)],
)
def test_other_paths_keep_complete_action_cache(monkeypatch, dual, mode, batch_size):
    policy = _make_policy(monkeypatch, dual=dual)
    runtime = policy.runtime

    def unexpected_prefix(*args, **kwargs):
        raise AssertionError("This path requires the complete current-frame cache.")

    monkeypatch.setattr(runtime, "_prefill_parent_gate_kv", unexpected_prefix)
    prepared = runtime.prepare_route_neutral_step(
        env_obs=_observation(batch_size), mode=mode, include_critic_features=False
    )
    assert len(prepared.current_condition.video_kv_cache) == 4
    assert prepared.gate_features.batch_size == batch_size
    if not dual:
        assert (
            runtime._uncond_condition_from_prepared(prepared)
            is prepared.current_condition
        )


def test_requested_critic_features_keep_complete_parent_cache(monkeypatch):
    policy = _make_policy(monkeypatch)
    runtime = policy.runtime
    runtime.critic_feature_config = replace(
        runtime.route_neutral_visual, num_mot_layers=4, layer_indices=(0, 3)
    )
    prepared = runtime.prepare_route_neutral_step(
        env_obs=_observation(), mode="eval", include_critic_features=True
    )
    assert len(prepared.current_condition.video_kv_cache) == 4
    assert prepared.critic_features.layer_indices == (0, 3)


@pytest.mark.parametrize("route", [0, 1])
def test_dual_actions_build_full_cache_after_gate_prefix(monkeypatch, route):
    policy = _make_policy(monkeypatch)
    runtime = policy.runtime
    observation = _observation()
    observation["_fastwam_action_noise_seeds"] = torch.tensor([71])
    observation["_fastwam_idm_noise_seeds"] = torch.tensor([83])
    prepared = runtime.prepare_route_neutral_step(
        env_obs=observation, mode="eval", include_critic_features=False
    )
    seen = []
    velocity = runtime._velocity

    def checked_velocity(condition, **kwargs):
        assert condition is not prepared.current_condition
        assert len(condition.video_kv_cache) == 4
        seen.append((kwargs["regime"], condition.video_seq_len))
        return velocity(condition, **kwargs)

    monkeypatch.setattr(runtime, "_velocity", checked_velocity)
    sample = runtime._sample_prepared_evaluation(
        env_obs=observation,
        routes=torch.tensor([route]),
        actor_version=0,
        prepared=prepared,
    )
    assert sample.actions.shape == (1, 2, 7)
    assert torch.isfinite(sample.actions).all()
    expected = (PolicyRegime.IDM, 12) if route else (PolicyRegime.UNCOND, 4)
    assert seen == [expected]
    assert runtime.actor.mot.num_layers == 4


def test_gate_prefix_is_fullgraph_compilable(monkeypatch):
    torch._dynamo.reset()
    policy = _make_policy(monkeypatch)
    runtime = RouteNeutralInference(
        policy.runtime, policy.gate, InferenceAccelerationConfig(compile=False)
    ).runtime
    observation = _observation()
    expected = runtime.prepare_route_neutral_step(
        env_obs=observation, mode="eval", include_critic_features=False
    ).gate_features
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    runtime._prefill_parent_gate_kv = torch.compile(
        runtime._prefill_parent_gate_kv, backend=backend, fullgraph=True, dynamic=False
    )
    try:
        for _ in range(2):
            actual = runtime.prepare_route_neutral_step(
                env_obs=observation, mode="eval", include_critic_features=False
            ).gate_features
            inference_fixtures._assert_equal(asdict(actual), asdict(expected))
        assert len(graphs) == 1
    finally:
        torch._dynamo.reset()
