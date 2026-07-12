# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch.distributions.categorical import Categorical

from _gate_test_imports import load_gate_modules


mods = load_gate_modules()
GatePolicy = mods.gate.GatePolicy
obs_prep = mods.obs
reward_mod = mods.reward

WF_DIM, P_DIM, TEXT_DIM = 8, 5, 4


@dataclass
class _EncodedState:
    world_feat: torch.Tensor
    first_frame_latents: torch.Tensor


class _StubAdapter:
    world_feat_dim = WF_DIM

    def __init__(self, ta=4, action_dim=7):
        self.ta = ta
        self.action_dim = action_dim
        self.encoded_states = []
        self.acted_states = []
        self.acted_modes = []

    def encode_world_state(self, input_image):
        state = _EncodedState(
            world_feat=input_image.detach().float().reshape(-1)[:WF_DIM].clone(),
            first_frame_latents=torch.ones(1),
        )
        self.encoded_states.append(state)
        return state

    def act(self, *, mode, encoded_state, **kwargs):
        self.acted_modes.append(int(mode))
        self.acted_states.append(encoded_state)
        return {
            "action_chunk": torch.zeros(self.ta, self.action_dim),
            "cost": 0.1 if int(mode) == 0 else 1.0,
        }


def _policy(*, add_value_head=True, adapter=True, **kwargs):
    return GatePolicy(
        world_feat_dim=WF_DIM,
        proprio_dim=P_DIM,
        text_feat_dim=TEXT_DIM,
        num_modes=2,
        hidden_sizes=(16, 16),
        add_value_head=add_value_head,
        wam_adapter=_StubAdapter() if adapter else None,
        **kwargs,
    )


def _obs(batch=2, context_dim=8):
    return {
        "input_image": torch.rand(batch, 3, 16, 16),
        "proprio": torch.randn(batch, P_DIM),
        "context": torch.randn(batch, 4, context_dim),
        "context_mask": torch.ones(batch, 4, dtype=torch.bool),
    }


def test_two_mode_contract_and_gate_input_dimensions():
    policy = _policy()
    assert policy.gate_input_dim == WF_DIM + P_DIM + TEXT_DIM
    assert policy.action_dim == 1 and policy.num_action_chunks == 1
    with pytest.raises(ValueError, match="exactly two"):
        GatePolicy(WF_DIM, P_DIM, text_feat_dim=TEXT_DIM, num_modes=3)
    for invalid in (True, 0.9, 1.9, "0.9"):
        with pytest.raises(ValueError, match="force_mode"):
            _policy(force_mode=invalid)


def test_wam_dtype_parser_accepts_only_real_float_dtypes():
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/gate_policy/__init__.py"
    )
    spec = importlib.util.spec_from_file_location("_gate_factory_dtype_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._resolve_wam_dtype("float32") is torch.float32
    assert module._resolve_wam_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="wam.dtype"):
        module._resolve_wam_dtype("int8")


def test_predict_reuses_encoded_state_and_caches_minimal_replay_input():
    policy = _policy()
    actions, result = policy.predict_action_batch(_obs(batch=2), mode="train")
    assert actions.shape == (2, 4, 7)
    assert actions.device.type == "cpu"
    assert actions.dtype == torch.float32
    assert result["prev_logprobs"].shape == (2, 1)
    assert result["prev_logprobs"].dtype == torch.float32
    assert result["prev_values"].shape == (2, 1)
    assert result["mode_logits"].shape == (2, 2)
    forward = result["forward_inputs"]
    assert forward["gate_input"].shape == (2, WF_DIM + P_DIM + TEXT_DIM)
    assert forward["action"].shape == (2, 1)
    assert forward["mode_entropy"].shape == (2, 1)
    assert forward["explore_eps"].shape == (2, 1)
    assert "world_feat" not in forward and "text_feat" not in forward
    assert policy.wam_adapter.acted_states == policy.wam_adapter.encoded_states
    assert all(mode in (0, 1) for mode in policy.wam_adapter.acted_modes)


def test_default_forward_uses_fp32_categorical_under_bfloat16():
    policy = _policy(add_value_head=False).to(dtype=torch.bfloat16)
    gate_input = torch.randn(3, policy.gate_input_dim)
    action = torch.tensor([[0], [1], [0]])
    out = policy.default_forward(
        {"gate_input": gate_input, "action": action},
        compute_values=False,
    )
    assert out["logprobs"].dtype == torch.float32
    assert out["entropy"].dtype == torch.float32
    reference = Categorical(logits=policy._logits(gate_input).float())
    assert torch.allclose(
        out["logprobs"], reference.log_prob(action[:, 0]).unsqueeze(-1)
    )


def test_eval_is_argmax_and_train_mixture_logprob_is_exact():
    policy = _policy(explore_eps=0.6)
    with torch.no_grad():
        policy.logits_head.weight.zero_()
        policy.logits_head.bias.copy_(torch.tensor([8.0, -8.0]))
    obs = _obs(batch=32)
    _, train = policy.predict_action_batch(obs, mode="train")
    stored_gate_input = train["forward_inputs"]["gate_input"]
    gate_input = torch.empty_like(stored_gate_input)
    gate_input.copy_(stored_gate_input)
    pi = Categorical(logits=policy._logits(gate_input).float()).probs
    mix = 0.4 * pi + 0.6 / 2.0
    expected = torch.log(mix.gather(-1, train["mode"].clone().unsqueeze(-1)))
    assert torch.allclose(train["prev_logprobs"], expected, atol=1e-6)
    # Before any optimizer update actor replay must recompute the same policy
    # that collected the sample, including the uniform exploration floor.
    replay_inputs = dict(train["forward_inputs"])
    for key in ("gate_input", "action", "explore_eps"):
        stored = train["forward_inputs"][key]
        replay_inputs[key] = torch.empty_like(stored)
        replay_inputs[key].copy_(stored)
    replay = policy.default_forward(replay_inputs, compute_values=False)
    assert torch.allclose(replay["logprobs"], train["prev_logprobs"], atol=1e-6)
    expected_entropy = Categorical(probs=mix).entropy().unsqueeze(-1)
    assert torch.allclose(replay["entropy"], expected_entropy, atol=1e-6)
    assert torch.allclose(
        train["forward_inputs"]["mode_entropy"], expected_entropy, atol=1e-6
    )

    _, evaluated = policy.predict_action_batch(obs, mode="eval")
    assert torch.equal(evaluated["mode"], torch.zeros(32, dtype=torch.long))


def test_uniform_exploration_and_validation():
    policy = _policy(explore_eps=1.0)
    _, result = policy.predict_action_batch(_obs(batch=16), mode="train")
    assert torch.allclose(
        result["prev_logprobs"],
        torch.full_like(result["prev_logprobs"], torch.log(torch.tensor(0.5))),
    )
    policy.set_explore_eps(0.1)
    assert policy.explore_eps == 0.1
    with pytest.raises(ValueError):
        policy.set_explore_eps(1.1)
    with pytest.raises(ValueError, match="explore_eps"):
        policy.default_forward(
            {
                "gate_input": torch.randn(2, policy.gate_input_dim),
                "action": torch.zeros(2, 1, dtype=torch.long),
                "explore_eps": torch.tensor([[0.1], [float("nan")]]),
            },
            compute_values=False,
        )


@pytest.mark.parametrize("forced", [0, 1])
def test_force_mode_is_eval_only_for_end_to_end_smoke(forced):
    policy = _policy(force_mode=forced)
    _, result = policy.predict_action_batch(_obs(batch=5), mode="eval")
    assert torch.equal(result["mode"], torch.full((5,), forced, dtype=torch.long))
    assert policy.wam_adapter.acted_modes == [forced] * 5
    with pytest.raises(RuntimeError, match="evaluation-only"):
        policy.predict_action_batch(_obs(batch=1), mode="train")


def test_actor_side_kl_is_differentiable_and_uses_global_step_schedule():
    policy = _policy(
        add_value_head=False,
        adapter=False,
        kl_prior_beta=0.4,
        kl_prior_beta_end=0.0,
        kl_prior_decay_steps=100,
    )
    policy.attach_bc_prior(policy.state_dict())
    with torch.no_grad():
        policy.logits_head.bias.add_(torch.tensor([1.5, -1.5]))
    forward = {
        "gate_input": torch.randn(12, policy.gate_input_dim),
        "action": torch.randint(0, 2, (12, 1)),
    }
    policy.set_global_step(50)
    out = policy.default_forward(forward, compute_values=False)
    assert out["kl_beta"].item() == pytest.approx(0.2)
    assert out["aux_loss"].requires_grad
    assert (out["kl_to_prior"] > 0).all()
    out["aux_loss"].mean().backward()
    assert policy.logits_head.bias.grad is not None
    assert policy.logits_head.bias.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in policy.bc_prior.parameters())
    policy.set_global_step(100)
    assert policy.current_kl_beta() == pytest.approx(0.0)


def test_text_pool_is_mask_aware_and_deterministic():
    context = torch.tensor([[[1.0] * 8, [3.0] * 8, [99.0] * 8]])
    mask = torch.tensor([[True, True, False]])
    got = obs_prep.pool_text_context(context, mask, output_dim=TEXT_DIM)
    assert got.shape == (1, TEXT_DIM)
    assert torch.allclose(got, torch.full_like(got, 2.0))
    assert torch.equal(got, obs_prep.pool_text_context(context, mask, TEXT_DIM))
    assert torch.equal(
        got,
        obs_prep._fallback_pool_text_context(
            context, mask, output_dim=TEXT_DIM
        ),
    )


def test_gate_resize_tracks_official_pil_bilinear_downsampling():
    image_module = pytest.importorskip("PIL.Image")
    generator = torch.Generator().manual_seed(7)
    image = torch.randint(
        0, 256, (1, 3, 48, 64), generator=generator, dtype=torch.uint8
    )
    resized = obs_prep._resize(image.float(), (25, 31))[0].permute(1, 2, 0)
    pil_image = image[0].permute(1, 2, 0).numpy()
    reference = torch.from_numpy(
        __import__("numpy").asarray(
            image_module.fromarray(pil_image).resize(
                (31, 25), resample=image_module.Resampling.BILINEAR
            )
        ).copy()
    ).float()
    assert torch.mean(torch.abs(resized - reference)).item() < 1.0


def test_preprocessor_reads_fastwam_text_cache_without_online_encoder(tmp_path):
    class _NoOnlineEncoder:
        def encode_prompt(self, prompt):
            raise AssertionError("online encoder must not be called")

    prep = obs_prep.GateObsPreprocessor(
        _NoOnlineEncoder(),
        "libero",
        text_feat_dim=TEXT_DIM,
        text_embedding_cache_dir=tmp_path,
        context_len=4,
    )
    prompt = prep._format_prompt("pick block")
    name = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    torch.save(
        {
            "context": torch.tensor(
                [[1.0] * 8, [3.0] * 8, [50.0] * 8, [60.0] * 8]
            ),
            "mask": torch.tensor([True, True, False, False]),
        },
        tmp_path / f"{name}.t5_len4.wan22ti2v5b.pt",
    )
    result = prep(
        {
            "main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
            "wrist_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
            "states": torch.zeros(1, P_DIM),
            "task_descriptions": ["pick block"],
        }
    )
    assert result["context"].shape == (1, 4, 8)
    assert torch.count_nonzero(result["context"][:, 2:]) == 0
    assert result["context_mask"].all()
    # FastWAM training zeroes padding then uses an all-true mask: (1+3+0+0)/4.
    assert torch.allclose(result["text_feat"], torch.full((1, TEXT_DIM), 1.0))


def test_missing_cache_fails_instead_of_silently_loading_text_encoder(tmp_path):
    prep = obs_prep.GateObsPreprocessor(
        object(), "libero", text_embedding_cache_dir=tmp_path, context_len=4
    )
    with pytest.raises(FileNotFoundError, match="precompute_text_embeds"):
        prep._encode_text(["missing task"])


def test_robotwin_requires_two_wrist_cameras_with_clear_error():
    prep = obs_prep.GateObsPreprocessor(object(), "robotwin")
    with pytest.raises(ValueError, match="collect_wrist_camera=true"):
        prep._image_fn(
            {
                "main_images": torch.zeros(1, 8, 8, 3),
                "wrist_images": None,
            }
        )
    with pytest.raises(ValueError, match=r"\[B,2,H,W,3\]"):
        prep._image_fn(
            {
                "main_images": torch.zeros(1, 8, 8, 3),
                "wrist_images": torch.zeros(1, 8, 8, 3),
            }
        )


def test_libero_action_postprocess_and_existing_image_range():
    prep = obs_prep.GateObsPreprocessor(
        object(), "libero", binarize_libero_gripper=True
    )
    actions = torch.zeros(1, 2, 7)
    actions[0, :, -1] = torch.tensor([0.25, 0.75])
    processed = prep.process_actions(actions)
    assert torch.equal(processed[0, :, -1], torch.tensor([1.0, -1.0]))
    image = torch.tensor([[[[-1.0, 1.0]]]])
    assert torch.equal(obs_prep._normalize(image), image)


def test_compute_cost_penalty_is_counted_once_per_chunk():
    components = reward_mod.apply_gate_reward(
        rewards=torch.zeros(2, 10),
        mode_cost=torch.tensor([[1.0], [0.2]]),
        step=10,
        lambda_cost=0.2,
        lambda_warmup_steps=0,
    )
    assert torch.allclose(
        components["total"].sum(dim=-1), torch.tensor([-0.2, -0.04])
    )
