# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the adaptive-prediction GatePolicy.

The gate's core (3-way categorical head, value head, default_forward /
predict_action_batch contracts) is exercised with a STUB WAMModeAdapter and a
passthrough obs, so no fastwam weights / GPU are needed. Importing the gate policy
pulls rlinf.models.__init__ (registry); skipped if the package can't import here.

Run on the server:
    cd RLinf
    pytest tests/unit_tests/test_gate_policy.py -v
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gp = pytest.importorskip(
    "rlinf.models.embodiment.gate_policy.gate_policy",
    reason="rlinf not importable in this environment",
)
GatePolicy = gp.GatePolicy
obs_prep = pytest.importorskip(
    "rlinf.models.embodiment.gate_policy.obs_preprocessor",
    reason="gate obs preprocessor not importable in this environment",
)
reward_mod = pytest.importorskip(
    "rlinf.models.embodiment.gate_policy.reward",
    reason="gate reward helpers not importable in this environment",
)
from torch.distributions.categorical import Categorical  # noqa: E402


class _StubAdapter:
    """Stand-in for fastwam's WAMModeAdapter (no weights)."""

    world_feat_dim = 8

    def __init__(self, ta=4, a_robot=7):
        self.ta = ta
        self.a_robot = a_robot
        self.acted_modes = []

    def encode_world_feat(self, input_image):
        # deterministic given the image so eval (argmax) is reproducible
        flat = input_image.detach().float().reshape(-1)
        return flat[: self.world_feat_dim].clone()

    def act(self, *, input_image, mode, proprio=None, context=None, context_mask=None, world_feat=None):
        self.acted_modes.append(int(mode))
        return {
            "action_chunk": torch.zeros(self.ta, self.a_robot),
            "world_feat": world_feat,
            "cost": 0.1 * (int(mode) + 1),
            "aux": {"mode": int(mode)},
        }


def _policy(proprio_dim=5, add_value_head=True, **kw):
    return GatePolicy(
        world_feat_dim=_StubAdapter.world_feat_dim,
        proprio_dim=proprio_dim,
        num_modes=3,
        hidden_sizes=(16, 16),
        add_value_head=add_value_head,
        wam_adapter=_StubAdapter(),
        **kw,
    )


def _obs(batch=2, proprio_dim=5, h=16, w=16, ctx_len=4):
    return {
        "input_image": torch.rand(batch, 3, h, w),
        "proprio": torch.randn(batch, proprio_dim),
        "context": torch.randn(batch, ctx_len, 4096),
        "context_mask": torch.ones(batch, ctx_len, dtype=torch.bool),
    }


def test_gate_input_dim():
    p = _policy(proprio_dim=5)
    assert p.gate_input_dim == 8 + 5
    assert p.action_dim == 1 and p.num_action_chunks == 1


def test_predict_action_batch_shapes_and_contract():
    p = _policy()
    out_actions, result = p.predict_action_batch(_obs(batch=2), mode="train")
    # robot action chunk for the simulator
    assert out_actions.shape == (2, 4, 7)
    # policy-gradient quantities are per chunk-step, [B,1]
    assert result["prev_logprobs"].shape == (2, 1)
    assert result["prev_values"].shape == (2, 1)
    fi = result["forward_inputs"]
    assert fi["gate_input"].shape == (2, 13)
    assert fi["action"].shape == (2, 1)          # trained action = mode index
    assert result["mode"].shape == (2,)
    assert result["mode_cost"].shape == (2,)
    assert result["mode_logits"].shape == (2, 3)
    # the frozen WAM was invoked once per env, with the chosen modes
    assert p.wam_adapter.acted_modes == [int(m) for m in result["mode"].tolist()]
    assert torch.all((result["mode"] >= 0) & (result["mode"] < 3))


def test_default_forward_matches_categorical():
    p = _policy()
    obs = _obs(batch=3)
    _, result = p.predict_action_batch(obs, mode="train")
    # predict_action_batch runs under inference_mode; in the real pipeline the
    # buffer crosses a worker channel (fresh tensors). Mimic that here, else the
    # grad-enabled value-head forward rejects inference tensors.
    fi = {k: v.clone() for k, v in result["forward_inputs"].items()}
    out = p.default_forward(fi, compute_logprobs=True, compute_entropy=True, compute_values=True)
    assert out["logprobs"].shape == (3, 1)
    assert out["entropy"].shape == (3, 1)
    assert out["values"].shape == (3, 1)
    # logprob must equal a fresh Categorical recompute on the same gate_input/action
    dist = Categorical(logits=p._logits(fi["gate_input"]))
    ref = dist.log_prob(fi["action"].reshape(-1).long()).unsqueeze(-1)
    assert torch.allclose(out["logprobs"], ref, atol=1e-5)


def test_eval_mode_is_argmax():
    p = _policy()
    obs = _obs(batch=2)
    _, result = p.predict_action_batch(obs, mode="eval")
    # recompute logits deterministically (stub world_feat is a function of the image)
    wf = torch.stack([p.wam_adapter.encode_world_feat(obs["input_image"][i:i+1]) for i in range(2)], 0)
    gate_input = p._build_gate_input(wf, obs["proprio"])
    expected = torch.argmax(p._logits(gate_input), dim=-1)
    assert torch.equal(result["mode"], expected)


def test_value_head_required_for_values():
    p = _policy(add_value_head=False)
    obs = _obs(batch=1)
    _, result = p.predict_action_batch(obs, mode="train", calculate_values=True)
    # with no value head, prev_values falls back to zeros (not an error)
    assert torch.allclose(result["prev_values"], torch.zeros_like(result["prev_values"]))
    fi = {k: v.clone() for k, v in result["forward_inputs"].items()}  # see note above
    with pytest.raises(NotImplementedError):
        p.default_forward(fi, compute_values=True)


def test_robotwin_preprocessor_accepts_two_wrist_cameras_and_uses_prompt_template():
    prompts = []

    class _TextWAM:
        def encode_prompt(self, prompt):
            prompts.append(prompt)
            return torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)

    prep = obs_prep.GateObsPreprocessor(_TextWAM(), "robotwin")
    env_obs = {
        "main_images": torch.zeros(2, 480, 640, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 2, 480, 640, 3, dtype=torch.uint8),
        "states": torch.randn(2, 14),
        "task_descriptions": ["pick block", "place cup"],
    }
    out = prep(env_obs)
    assert out["input_image"].shape == (2, 3, 384, 320)
    assert out["proprio"].shape == (2, 14)
    assert out["context"].shape == (2, 4, 8)
    assert prompts[0].startswith("A video recorded from a robot's point of view")
    assert "pick block" in prompts[0]


def test_gate_reward_cost_is_not_multiplied_by_action_horizon():
    rewards = torch.zeros(2, 32)
    mode_cost = torch.tensor([[1.0], [0.5]])
    components = reward_mod.apply_gate_reward(
        rewards=rewards,
        mode_cost=mode_cost,
        step=10,
        lambda_cost=0.2,
        lambda_warmup_steps=0,
    )
    chunk_total = components["total"].sum(dim=-1)
    assert torch.allclose(chunk_total, torch.tensor([-0.2, -0.1]))
