# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Seeded batched UNCOND actions and fixed-axis terminated-row coverage."""

import copy

import pytest
import torch
from fastwam.models.wan22.adaptive_sampler import sample_action_flow_sde
from test_fastwam_uncond_rl import _obs, _policy

from rlinf.envs.libero.action_protocol import (
    select_executed_action_prefix,
    select_executed_flow_statistics,
)
from rlinf.models.embodiment.wam_policy.libero_runtime import (
    _domain_separated_noise_seed,
    _seeded_randn,
)


def _serial_reference(runtime, obs, seeds):
    """Retain the original row-wise generator/sampler call as an independent oracle."""

    images, context, mask = runtime._encode_condition(obs)
    times, deltas = runtime._action_schedule()
    rollouts = []
    with torch.no_grad():
        for index in range(images.shape[0]):
            latents = runtime.actor._encode_input_image_latents_tensor(
                images[index : index + 1], tiled=runtime.tiled_vae
            )
            condition = runtime._current_condition(
                latents, context[index : index + 1], mask[index : index + 1]
            )
            shape = (1, runtime.action_protocol.generation_horizon, 7)
            generator = None
            if seeds is None:
                noise = torch.randn(shape, device=runtime.device).to(runtime.dtype)
            else:
                seed = int(seeds[index])
                noise = _seeded_randn(
                    seed,
                    shape,
                    device=runtime.device,
                    dtype=runtime.dtype,
                    rand_device=runtime.seeded_noise_device,
                )
                generator = torch.Generator(device=runtime.device).manual_seed(
                    _domain_separated_noise_seed(seed, domain="flow-sde")
                )
            rollouts.append(
                sample_action_flow_sde(
                    noise,
                    velocity_fn=runtime._uncond_velocity(condition, 10),
                    timesteps=times,
                    scheduler_deltas=deltas,
                    num_train_timesteps=1000,
                    noise_level=runtime.flow_sde_noise_level,
                    generator=generator,
                    ignore_last_transition=True,
                    stochastic=True,
                )
            )
    return {
        "actions": select_executed_action_prefix(
            torch.cat([item.actions for item in rollouts]),
            protocol=runtime.action_protocol,
        ),
        "chains": torch.cat([item.chains for item in rollouts]),
        "indices": torch.cat([item.denoise_indices for item in rollouts]),
        "logprobs": select_executed_flow_statistics(
            torch.cat([item.old_log_probs for item in rollouts]),
            protocol=runtime.action_protocol,
        ),
    }


@pytest.mark.parametrize("seeded", [False, True])
def test_batched_sampling_preserves_serial_rng_and_flow_replay(monkeypatch, seeded):
    torch.set_num_threads(1)
    runtime = _policy(monkeypatch).runtime
    obs = _obs()
    seeds = torch.tensor([101, 320]) if seeded else None
    torch.manual_seed(19)
    expected = _serial_reference(runtime, obs, seeds)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    initial_rng = torch.get_rng_state()
    prefill_sizes = []
    current_condition = runtime._current_condition

    def record_prefill(latents, context, context_mask):
        prefill_sizes.append(latents.shape[0])
        return current_condition(latents, context, context_mask)

    monkeypatch.setattr(runtime, "_current_condition", record_prefill)
    actions, result = runtime.sample_uncond_actions(
        obs, mode="train", actor_version=10, action_seeds=seeds
    )
    inputs = result["forward_inputs"]
    assert prefill_sizes == [2]
    assert torch.equal(inputs["denoise_indices"], expected["indices"])
    assert torch.equal(inputs["flow_chains"][:, 0], expected["chains"][:, 0])
    torch.testing.assert_close(actions, expected["actions"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        inputs["flow_chains"], expected["chains"], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        result["prev_logprobs"], expected["logprobs"], rtol=1e-5, atol=1e-5
    )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    if seeded:
        assert torch.equal(torch.get_rng_state(), initial_rng)
    output = runtime.replay_uncond_actions(inputs, actor_version=10)
    torch.testing.assert_close(
        output["flow_logprobs"], result["prev_logprobs"], rtol=1e-5, atol=1e-5
    )
    output["flow_logprobs"].sum().backward()
    for adapter in (runtime.lora_adapter, runtime.video_lora_adapter):
        assert (
            sum(
                parameter.grad.abs().sum()
                for parameter in adapter.lora_parameters()
                if parameter.grad is not None
            )
            > 0
        )


def _four_obs():
    obs = _obs()
    obs = {key: torch.cat([value, value]) for key, value in obs.items()}
    obs["_fastwam_env_ids"] = torch.tensor([1, 7, 11, 19])
    return obs


def test_sparse_rows_keep_critic_geometry_rng_and_all_identities(monkeypatch):
    torch.set_num_threads(1)
    policy = _policy(monkeypatch)
    policy.set_global_step(10)
    obs = _four_obs()
    initial_state = copy.deepcopy(policy.rollout_runtime_state_dict())
    expected_actions, expected = policy.predict_action_batch(obs)
    expected_state = policy.rollout_runtime_state_dict()
    policy.load_rollout_runtime_state_dict(initial_state)
    obs["_uncond_rl_active"] = torch.tensor([False, True, False, True])
    prefill_sizes, critic_sizes = [], []
    prefill = policy.runtime._current_condition
    predict_value = policy.critic.predict_value_batch

    def record_prefill(latents, context, context_mask):
        prefill_sizes.append(latents.shape[0])
        return prefill(latents, context, context_mask)

    def record_value(observations, **kwargs):
        critic_sizes.append(observations["states"].shape[0])
        return predict_value(observations, **kwargs)

    monkeypatch.setattr(policy.runtime, "_current_condition", record_prefill)
    monkeypatch.setattr(policy.critic, "predict_value_batch", record_value)
    actions, result = policy.predict_action_batch(obs)
    assert prefill_sizes == [2]
    assert critic_sizes == [4]
    active = obs["_uncond_rl_active"]
    torch.testing.assert_close(actions[active], expected_actions[active])
    assert not actions[~active].any()
    for key, value in result["forward_inputs"].items():
        if key == policy.critic.replay_feature_key:
            assert torch.equal(value, expected["forward_inputs"][key])
        else:
            torch.testing.assert_close(
                value[active], expected["forward_inputs"][key][active]
            )
            assert not value[~active].any()
    assert torch.equal(result["prev_values"], expected["prev_values"])
    torch.testing.assert_close(
        result["prev_logprobs"][active], expected["prev_logprobs"][active]
    )
    assert not result["prev_logprobs"][~active].any()
    assert torch.equal(result["route_info"].chunk_ids, expected["route_info"].chunk_ids)
    assert torch.equal(
        result["route_info"].episode_ids, expected["route_info"].episode_ids
    )
    assert policy.rollout_runtime_state_dict() == expected_state
    assert not result["route_info"].route_used.any()
    assert not result["emitted_gate"].next_route.any()
    assert not result["emitted_gate"].valid.any()


def test_finished_rows_skip_action_compute_but_keep_full_critic_and_replay(monkeypatch):
    torch.set_num_threads(1)
    policy = _policy(monkeypatch)
    obs = _obs()
    policy.predict_action_batch(obs)
    obs["_fastwam_reset_mask"].zero_()
    obs["_uncond_rl_active"] = torch.zeros(2, dtype=torch.bool)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "All-finished sampling executed heavy action preprocessing/inference"
        )

    monkeypatch.setattr(policy.runtime, "_encode_condition", forbidden)
    monkeypatch.setattr(policy.actor, "_encode_input_image_latents_tensor", forbidden)
    actions, result = policy.predict_action_batch(obs)
    assert not actions.any()
    assert result["route_info"].chunk_ids.tolist() == [1, 1]
    assert not result["route_info"].route_used.any()
    assert not result["emitted_gate"].valid.any()
    assert not result["forward_inputs"]["denoise_indices"].any()
    assert torch.equal(result["forward_inputs"]["critic_prefix"], obs["states"])
    output = policy.default_forward(result["forward_inputs"])
    assert all(torch.isfinite(value).all() for value in output.values())
    assert torch.equal(result["prev_values"], output["values"])


def test_all_finished_requires_an_initial_active_rollout(monkeypatch):
    policy = _policy(monkeypatch)
    obs = _obs()
    obs["_uncond_rl_active"] = torch.zeros(2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="start with active environments"):
        policy.predict_action_batch(obs)


def test_eval_keeps_serial_prefill_geometry_and_ignores_training_active_mask(
    monkeypatch,
):
    torch.set_num_threads(1)
    policy = _policy(monkeypatch)
    obs = _obs()
    obs["_uncond_rl_active"] = torch.zeros(2, dtype=torch.bool)
    obs["_fastwam_action_noise_seeds"] = torch.tensor([17, 21])
    prefill_sizes = []
    prefill = policy.runtime._current_condition

    def record_prefill(latents, context, context_mask):
        prefill_sizes.append(latents.shape[0])
        return prefill(latents, context, context_mask)

    monkeypatch.setattr(policy.runtime, "_current_condition", record_prefill)
    actions, sample = policy.predict_action_batch(
        obs, mode="eval", compute_values=False
    )
    assert prefill_sizes == [1, 1]
    assert actions.abs().sum() > 0
    assert sample["forward_inputs"] == {}
