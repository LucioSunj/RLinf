# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FR3 native processing, actual tiny MoT replay, and executed-prefix PPO."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastwam.adapters import (
    RegimeLoRAConfig,
    inject_action_dit_lora,
    inject_video_bc_dit_lora,
)
from fastwam.models.wan22.gate_transformer import epsilon_mixture_bernoulli
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.fastwam_dual_ppo import compute_fastwam_dual_ppo_loss
from rlinf.envs.action_protocol import (
    select_executed_action_prefix,
    select_executed_flow_statistics,
)
from rlinf.models.embodiment.wam_policy.adaptive_policy import (
    FastWAMAdaptivePolicyConfig,
)
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    WAMRoute,
)
from rlinf.models.embodiment.wam_policy.critic import (
    FastWAMCurrentFrameValueCritic,
    FastWAMValueTransformerConfig,
)
from rlinf.models.embodiment.wam_policy.fr3_critic import Fr3Pi05Inputs
from rlinf.models.embodiment.wam_policy.fr3_runtime import (
    Fr3ActionProtocol,
    Fr3RouteNeutralRuntime,
    fr3_observation_batch,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import OnlineIDMBCConfig
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PadRouteNeutralCurrentStepGate,
    PadRouteNeutralGateConfig,
    serialize_route_neutral_features,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.policy import (
    RouteNeutralOnlineIDMBCFastWAMPolicy,
)

ROOT = Path(__file__).resolve().parents[3]


class _TinySpatialEncoder(nn.Module):
    """Real convolution keeps native camera inputs tractable for CPU tiny MoT."""

    upsampling_factor = 32
    temporal_downsample_factor = 4

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(z_dim=2)
        self.encoder = nn.Conv3d(3, 2, kernel_size=(1, 32, 32), stride=(4, 32, 32))

    def encode(self, videos, **_kwargs):
        return self.encoder(torch.stack(videos) if isinstance(videos, list) else videos)


def _statistics(path):
    stats = {
        key: {
            "default": {
                "global_mean": [0.1] * 7 + [0.5],
                "global_std": [0.2] * 7 + [0.25],
            }
        }
        for key in ("state", "action")
    }
    path.write_text(json.dumps(stats))
    return stats


def _observation(task, *, batch_size=2):
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    left[..., 0] = np.arange(640, dtype=np.uint16)[None, :] % 256
    wrist = np.zeros_like(left)
    wrist[..., 2] = np.arange(480, dtype=np.uint16)[:, None] % 256
    state = np.array([1.7, -1.2, 0.3, -0.4, 0.5, -0.6, 0.7, 1.0], dtype=np.float32)
    row = fr3_observation_batch(
        {
            "observation.state": state,
            "observation.images.left": left,
            "observation.images.wrist": wrist,
            "task": task,
        }
    )
    result = {
        key: value.repeat(batch_size, *([1] * (value.ndim - 1)))
        if isinstance(value, torch.Tensor)
        else value * batch_size
        for key, value in row.items()
        if key in ("states", "main_images", "wrist_images", "task_descriptions")
    }
    if batch_size > 1:
        result["states"][1, :7] -= 0.2
    result.update(
        _fastwam_env_ids=torch.arange(batch_size),
        _fastwam_reset_mask=torch.ones(batch_size, dtype=torch.bool),
        _fastwam_action_noise_seeds=torch.arange(41, 41 + batch_size),
        _fastwam_idm_noise_seeds=torch.arange(51, 51 + batch_size),
    )
    return result


@pytest.fixture(params=["drawer", "insert_tubes"])
def policy(tmp_path, request):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    spec = importlib.util.spec_from_file_location(
        "fr3_bc_helpers", ROOT / "tests/test_fr3_bc_pipeline.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    actor = helpers._tiny_model()
    actor.vae = _TinySpatialEncoder()
    action = inject_action_dit_lora(
        actor.action_expert, RegimeLoRAConfig(rank=128, alpha=128)
    )
    video = inject_video_bc_dit_lora(
        actor.video_expert,
        RegimeLoRAConfig(rank=128, alpha=128),
        regime_context=action.regime_context,
    )
    with torch.no_grad():
        for adapter in (action, video):
            for parameter in adapter.lora_parameters():
                parameter.normal_(std=0.01)
    task = helpers.TASKS[request.param][1]
    stats = tmp_path / "stats.json"
    _statistics(stats)
    training = OmegaConf.load(
        ROOT / f"deployments/fr3_wam/config_assets/{request.param}/config.yaml"
    )
    source = OmegaConf.load(
        ROOT
        / "RLinf/examples/embodiment/config/model/fastwam_route_neutral_online.yaml"
    )
    visual = FastWAMValueTransformerConfig(
        num_mot_layers=2,
        source_num_heads=2,
        source_head_dim=8,
        layer_indices=(0, 1),
        sources=("current_frame_video",),
        hidden_dim=16,
        num_query_tokens=2,
    )
    critic_config = replace(
        visual, sources=("current_frame_video", "text_state_context")
    )
    cache = tmp_path / "text"
    cache.mkdir()
    prompt = helpers.PROMPT_TEMPLATE.format(task=task)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    torch.save(
        {
            "context": torch.randn(4, 8).to(torch.bfloat16),
            "mask": torch.tensor([True, True, False, False]),
        },
        cache / f"{digest}.text_len4.wan22ti2v5b.pt",
    )
    runtime = Fr3RouteNeutralRuntime(
        actor=actor,
        lora_adapter=action,
        video_lora_adapter=video,
        training_config=training,
        easywam_root=ROOT / "EasyWAM",
        processor_stats_path=stats,
        task_name=request.param,
        task_prompt=task,
        action_dt=0.1,
        route_neutral_input=OmegaConf.to_container(
            source.route_neutral_online.input_contract
        ),
        route_neutral_visual=visual,
        critic_feature_config=critic_config,
        gate_layer_indices=(0, 1),
        num_inference_steps=3,
        max_episode_steps=35,
        text_embedding_cache_dir=str(cache),
        text_embedding_context_len=4,
    )
    gate = PadRouteNeutralCurrentStepGate(
        PadRouteNeutralGateConfig(
            visual=visual, language_dim=8, state_dim=8, history_length_chunks=4
        )
    )
    critic = FastWAMCurrentFrameValueCritic(config=critic_config, hidden_sizes=(16,))
    return RouteNeutralOnlineIDMBCFastWAMPolicy(
        actor=actor,
        runtime=runtime,
        lora_adapter=action,
        video_lora_adapter=video,
        gate=gate,
        critic=critic,
        config=FastWAMAdaptivePolicyConfig(),
        online_idm_bc_config=OnlineIDMBCConfig(enabled=True, loss_weight=0.2),
        critic_warmup={
            "runner_updates": 1,
            "route_behavior": "independent_random",
            "idm_probability": 0.5,
            "freeze_gate": True,
            "freeze_cost_controller": True,
        },
    ).train()


def test_fr3_runtime_native_image_state_action_and_session_contract(policy):
    runtime = policy.runtime
    obs = _observation(runtime.task_prompt)
    rows = runtime.fr3_processor.prepare_batch(runtime._rows(obs))
    torch.testing.assert_close(
        runtime._model_images(obs), rows["input_image"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        runtime._normalized_proprio(obs["states"]),
        rows["proprio"][:, 0],
        atol=0,
        rtol=0,
    )
    assert rows["input_image"].shape == (2, 3, 224, 448)
    assert rows["proprio"][0, 0, 0] == 5
    normalized = torch.zeros((2, 16, 8))
    normalized[..., 7] = 2
    decoded, trace = runtime._denormalize_action_stages(normalized, env_obs=obs)
    torch.testing.assert_close(
        decoded[..., :7], obs["states"][:, None, :7].expand(-1, 16, -1) + 0.1
    )
    torch.testing.assert_close(decoded[..., 7], torch.ones((2, 16)))
    assert trace is None
    assert runtime.action_protocol.max_episode_steps == 35
    assert runtime.action_protocol.reset_wait_steps == 0
    critic_observation = runtime.critic_observation(env_obs=obs)
    assert critic_observation["extra_view_images"] is None
    assert torch.equal(critic_observation["states"], obs["states"])
    assert not any(key.startswith("_fastwam_") for key in critic_observation)
    prepared = runtime.prepare_route_neutral_step(env_obs=obs)
    assert prepared.gate_features.state.shape == (2, 8)
    assert runtime.physical_history.state_dict()["states"]
    assert runtime._text_context_cache
    runtime._evaluation_text_context = ("cached", None)
    runtime.clear_session()
    assert not runtime.physical_history.state_dict()["states"]
    assert runtime._evaluation_text_context is None and not runtime._text_context_cache
    bad = copy.deepcopy(obs)
    bad["task_descriptions"][0] = "wrong task prompt"
    with pytest.raises(ValueError, match="task-bound"):
        runtime.prepare_route_neutral_step(env_obs=bad)


def test_fr3_actual_mixed_rollout_replay_gradient_and_parent_gate_isolation(policy):
    runtime = policy.runtime
    obs = _observation(runtime.task_prompt)
    routes = torch.tensor([int(WAMRoute.UNCOND), int(WAMRoute.IDM)])
    route = ChunkRouteRecord(
        route_used=routes,
        route_was_forced=torch.zeros(2, dtype=torch.bool),
        chunk_ids=torch.zeros(2, dtype=torch.long),
        episode_ids=torch.zeros(2, dtype=torch.long),
        route_source_chunk_ids=torch.zeros(2, dtype=torch.long),
        actor_versions=torch.ones(2, dtype=torch.long),
    )
    with torch.no_grad():
        prepared = runtime.prepare_route_neutral_step(env_obs=obs)
        logits = policy.gate(prepared.gate_features)
        behavior = epsilon_mixture_bernoulli(logits, epsilon=0.1, temperature=1.0)
        sample = runtime.sample_routed_action_batch(
            env_obs=obs,
            routes=routes,
            actor_version=1,
            mode="train",
            collect_replay=True,
            prepared=prepared,
        )
    assert sample.actions.shape == (2, 16, 8)
    assert sample.old_flow_logprobs.shape == (2, 16, 8)
    assert sample.forward_inputs["online_idm_bc_teacher_actions"].shape == (2, 16, 8)
    assert torch.equal(
        prepared.gate_features.language_mask,
        torch.tensor([[True, True, False, False]]).expand(2, -1),
    )
    sample.forward_inputs.update(
        serialize_route_neutral_features(prepared.gate_features)
    )
    sample.forward_inputs.update(
        flow_chains=sample.flow_chains,
        denoise_indices=sample.denoise_indices,
        online_idm_bc_flow_valid=routes == int(WAMRoute.UNCOND),
    )
    emitted = GateDecisionRecord(
        next_route=routes,
        base_probability=behavior.base_idm_probability,
        behavior_probability=behavior.behavior_idm_probability,
        old_logprob=behavior.log_prob(routes),
        epsilon=torch.full_like(logits, 0.1),
        temperature=torch.ones_like(logits),
        valid=torch.ones(2, dtype=torch.bool),
        source_chunk_ids=route.chunk_ids,
        episode_ids=route.episode_ids,
        actor_versions=route.actor_versions,
    )
    output = policy.default_forward(
        sample.forward_inputs, route_info=route, emitted_gate=emitted
    )
    torch.testing.assert_close(
        output["flow_logprobs"], sample.old_flow_logprobs, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        output["gate_logprobs"], emitted.old_logprob, atol=1e-6, rtol=1e-6
    )
    execution_mask = torch.tensor([[True] * 3 + [False] * 13, [True] * 16])
    loss, metrics = compute_fastwam_dual_ppo_loss(
        gate_logprobs=output["gate_logprobs"],
        gate_old_logprobs=emitted.old_logprob,
        gate_advantages=torch.tensor([1.0, -1.0]),
        gate_valid_mask=torch.ones(2, dtype=torch.bool),
        gate_clip_ratio_low=0.2,
        gate_clip_ratio_high=0.2,
        flow_logprobs=output["flow_logprobs"],
        flow_old_logprobs=sample.old_flow_logprobs,
        flow_advantages=torch.tensor([1.0, -1.0]),
        flow_valid_mask=torch.ones(2, dtype=torch.bool),
        route_used=routes,
        flow_clip_ratio_low=0.2,
        flow_clip_ratio_high=0.2,
        flow_execution_mask=execution_mask,
    )
    assert metrics["gate/sample_count"] == 2
    assert metrics["uncond_flow/sample_count"] == 1
    frozen = {
        name: parameter.clone()
        for name, parameter in policy.actor.named_parameters()
        if not parameter.requires_grad
    }
    trainable = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    initial_adapters = {
        key: copy.deepcopy(adapter.lora_state_dict())
        for key, adapter in (
            ("action", policy.lora_adapter),
            ("video", policy.video_lora_adapter),
        )
    }
    total = (
        loss
        + 0.2 * output["online_idm_bc_raw_loss"]
        + (output["values"] - 1).square().mean()
    )
    total.backward()
    for adapter in (policy.lora_adapter, policy.video_lora_adapter):
        assert any(
            p.grad is not None and torch.count_nonzero(p.grad)
            for p in adapter.lora_parameters()
        )
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad)
        for p in policy.gate.parameters()
    )
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad)
        for p in policy.critic.parameters()
    )
    optimizer.step()
    for key, adapter in (
        ("action", policy.lora_adapter),
        ("video", policy.video_lora_adapter),
    ):
        assert any(
            not torch.equal(value, initial_adapters[key][name])
            for name, value in adapter.lora_state_dict().items()
        )
    for name, parameter in policy.actor.named_parameters():
        if name in frozen:
            assert parameter.grad is None and torch.equal(parameter, frozen[name])
    # Reset gives the same physical history; adapter updates cannot change Gate's
    # frozen current-frame banks or the frozen same-state IDM teacher.
    with torch.no_grad():
        after = runtime.prepare_route_neutral_step(env_obs=obs)
        after_sample = runtime.sample_routed_action_batch(
            env_obs=obs,
            routes=routes,
            actor_version=2,
            mode="train",
            collect_replay=True,
            prepared=after,
        )
    for before_layer, after_layer in zip(
        prepared.gate_features.visual.layers,
        after.gate_features.visual.layers,
        strict=True,
    ):
        torch.testing.assert_close(
            before_layer.current_frame_video.key,
            after_layer.current_frame_video.key,
            atol=0,
            rtol=0,
        )
    torch.testing.assert_close(
        after_sample.actions[1], sample.actions[1], atol=0, rtol=0
    )
    torch.testing.assert_close(
        after_sample.forward_inputs["online_idm_bc_teacher_actions"],
        sample.forward_inputs["online_idm_bc_teacher_actions"],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("executed", [0, 1, 7, 16])
def test_executed_prefix_ppo_masks_tail_before_ratio_without_masking_gate(executed):
    mask = torch.arange(16)[None] < executed
    gate = torch.zeros(1, requires_grad=True)
    new = torch.zeros((1, 16, 8), requires_grad=True)
    old = torch.zeros_like(new)
    with torch.no_grad():
        new[:, executed:] = 50
    loss, metrics = compute_fastwam_dual_ppo_loss(
        gate_logprobs=gate,
        gate_old_logprobs=torch.zeros(1),
        gate_advantages=torch.ones(1),
        gate_valid_mask=torch.tensor([True]),
        gate_clip_ratio_low=0.2,
        gate_clip_ratio_high=0.2,
        flow_logprobs=new,
        flow_old_logprobs=old,
        flow_advantages=torch.ones(1),
        flow_valid_mask=torch.tensor([True]),
        route_used=torch.tensor([int(WAMRoute.UNCOND)]),
        flow_clip_ratio_low=0.2,
        flow_clip_ratio_high=0.2,
        flow_execution_mask=mask,
    )
    loss.backward()
    assert metrics["gate/sample_count"] == 1
    assert metrics["uncond_flow/sample_count"] == int(executed > 0)
    assert torch.isfinite(loss)
    assert torch.count_nonzero(new.grad[:, executed:]) == 0
    if executed:
        assert torch.count_nonzero(new.grad[:, :executed]) == executed * 8
    assert gate.grad.item() == -1


def test_fr3_prefix_protocol_has_no_libero_divisibility_or_reset_constraint():
    protocol = Fr3ActionProtocol(max_episode_steps=35)
    actions = torch.randn(1, 16, 8)
    assert select_executed_action_prefix(actions, protocol=protocol).shape == (1, 16, 8)
    empty = torch.empty(1, 0, 8)
    assert select_executed_flow_statistics(empty, protocol=protocol) is empty
    with pytest.raises(ValueError, match="16"):
        Fr3ActionProtocol(execution_horizon=10)
    with pytest.raises(ValueError, match="reset"):
        Fr3ActionProtocol(reset_wait_steps=30)


def test_pi05_fr3_raw_rgb_joint_statistics_and_missing_camera_mask(tmp_path):
    path = tmp_path / "statistics.json"
    stats = _statistics(path)["state"]["default"]
    adapter = Fr3Pi05Inputs(path)
    row = _observation("insert the tubes", batch_size=1)
    result = adapter(
        {
            "observation/state": row["states"][0].numpy(),
            "observation/image": row["main_images"][0].numpy(),
            "observation/wrist_image": row["wrist_images"][0].numpy(),
            "prompt": "insert the tubes",
        }
    )
    np.testing.assert_allclose(
        result["state"],
        (row["states"][0].numpy() - stats["global_mean"]) / stats["global_std"],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        result["image"]["base_0_rgb"], row["main_images"][0].numpy()
    )
    np.testing.assert_array_equal(
        result["image"]["left_wrist_0_rgb"], row["wrist_images"][0].numpy()
    )
    assert result["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    }
    assert np.count_nonzero(result["image"]["right_wrist_0_rgb"]) == 0
    assert result["prompt"] == "insert the tubes"
