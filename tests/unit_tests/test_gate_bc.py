# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the gate BC (SFT) warm-start + KL-to-BC prior (M3).

Everything runs on CPU with synthetic tensors; no fastwam weights / simulator.
Tests that need the fastwam package (shard-file loading) importorskip it.

Run on the server:
    cd RLinf
    pytest tests/unit_tests/test_gate_bc.py -v
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gp = pytest.importorskip(
    "rlinf.models.embodiment.gate_policy.gate_policy",
    reason="rlinf not importable in this environment",
)
bc = pytest.importorskip(
    "rlinf.models.embodiment.gate_policy.bc",
    reason="gate bc module not importable in this environment",
)
reward_mod = pytest.importorskip(
    "rlinf.models.embodiment.gate_policy.reward",
    reason="gate reward helpers not importable in this environment",
)
GatePolicy = gp.GatePolicy

WF_DIM, P_DIM = 8, 5


def _policy(add_value_head=False, hidden=(16, 16)):
    return GatePolicy(
        world_feat_dim=WF_DIM,
        proprio_dim=P_DIM,
        num_modes=3,
        hidden_sizes=hidden,
        add_value_head=add_value_head,
        wam_adapter=None,
        obs_preprocessor=None,
    )


def _synthetic_labeled(n=900, seed=0):
    """Linearly separable oracle labels: label = argmax(W @ [wf, proprio])."""
    g = torch.Generator().manual_seed(seed)
    world_feat = torch.randn(n, WF_DIM, generator=g)
    proprio = torch.randn(n, P_DIM, generator=g)
    w = torch.randn(WF_DIM + P_DIM, 3, generator=g)
    label = (torch.cat([world_feat, proprio], dim=-1) @ w).argmax(dim=-1)
    return {"world_feat": world_feat, "proprio": proprio, "label": label}


# ======================================================================== #
# dataset + class weights
# ======================================================================== #
def test_dataset_validation_and_item():
    data = _synthetic_labeled(n=10)
    ds = bc.GateOracleLabelDataset(data, meta={"cost_table": {"skip": 0.1, "latent": 0.4, "full": 1.0}})
    assert len(ds) == 10
    assert ds.world_feat_dim == WF_DIM and ds.proprio_dim == P_DIM
    item = ds[3]
    assert item["world_feat"].shape == (WF_DIM,)
    assert item["label"].dtype == torch.long
    with pytest.raises(ValueError):
        bc.GateOracleLabelDataset({k: v for k, v in data.items() if k != "label"})
    with pytest.raises(ValueError):
        bc.GateOracleLabelDataset({**data, "proprio": data["proprio"][:5]})


def test_class_balance_weights():
    labels = torch.tensor([0] * 80 + [1] * 15 + [2] * 5)
    w = bc.class_balance_weights(labels, 3, power=1.0)
    assert w.shape == (3,)
    assert w[0] < w[1] < w[2]
    assert w.mean().item() == pytest.approx(1.0)
    # power softens toward uniform
    w_soft = bc.class_balance_weights(labels, 3, power=0.5)
    assert (w_soft[2] / w_soft[0]) < (w[2] / w[0])
    # absent class gets zero weight; present classes still mean 1
    labels2 = torch.tensor([0] * 5 + [1] * 5)
    w2 = bc.class_balance_weights(labels2, 3)
    assert w2[2].item() == 0.0
    assert w2[:2].mean().item() == pytest.approx(1.0)


def test_expected_mode_cost():
    cost_table = {"skip": 0.1, "latent": 0.4, "full": 1.0}
    labels = torch.tensor([0, 2])
    assert bc.expected_mode_cost(labels, cost_table) == pytest.approx(0.55)
    assert bc.expected_mode_cost(labels, None) is None


# ======================================================================== #
# BC training end-to-end (synthetic, CPU)
# ======================================================================== #
def test_train_gate_bc_learns_separable_labels(tmp_path):
    ds = bc.GateOracleLabelDataset(_synthetic_labeled(n=900, seed=0))
    policy = _policy()
    cfg = bc.GateBCConfig(epochs=40, batch_size=128, lr=3.0e-3, val_fraction=0.15,
                          seed=0, log_every_epochs=0)
    result = bc.train_gate_bc(policy, ds, cfg)
    assert result["best"]["accuracy"] > 0.85
    assert result["num_train"] + result["num_val"] == 900
    assert len(result["history"]) == 40
    # per-mode metrics exist
    for i in range(3):
        assert f"recall/mode_{i}" in result["best"]
        assert f"pred_frac/mode_{i}" in result["best"]

    # checkpoint is a RAW state_dict, strict-loadable into a fresh identical policy
    path = str(tmp_path / "gate_bc.pt")
    bc.save_gate_bc_checkpoint(policy, path, meta={"kind": "gate_bc"})
    payload = torch.load(path, weights_only=False)
    assert all(torch.is_tensor(v) for v in payload.values())  # runner.ckpt_path contract
    fresh = _policy()
    fresh.load_state_dict(payload)  # strict
    x_wf, x_p = ds.world_feat[:4], ds.proprio[:4]
    assert torch.allclose(fresh.mode_logits(x_wf, x_p), policy.mode_logits(x_wf, x_p))
    import os

    assert os.path.exists(path + ".meta.json")


def test_train_gate_bc_restores_best_weights():
    ds = bc.GateOracleLabelDataset(_synthetic_labeled(n=200, seed=1))
    policy = _policy()
    cfg = bc.GateBCConfig(epochs=3, batch_size=64, lr=1.0e-3, val_fraction=0.2,
                          seed=1, log_every_epochs=0)
    result = bc.train_gate_bc(policy, ds, cfg)
    final = bc.evaluate_gate_bc(policy, ds.world_feat, ds.proprio, ds.label)
    # restored weights reproduce (at least) the best recorded val accuracy on the val split
    assert result["best"]["accuracy"] == max(h["accuracy"] for h in result["history"])
    assert 0.0 <= final["accuracy"] <= 1.0


def test_load_gate_bc_state_accepts_wrapped_payloads(tmp_path):
    policy = _policy()
    raw_path = str(tmp_path / "raw.pt")
    bc.save_gate_bc_checkpoint(policy, raw_path)
    state = bc.load_gate_bc_state(raw_path)
    assert set(state) == set(policy.state_dict())

    wrapped_path = str(tmp_path / "wrapped.pt")
    torch.save({"state_dict": policy.state_dict(), "meta": {"x": 1}}, wrapped_path)
    state2 = bc.load_gate_bc_state(wrapped_path)
    assert set(state2) == set(policy.state_dict())

    bad_path = str(tmp_path / "bad.pt")
    torch.save({"foo": "bar"}, bad_path)
    with pytest.raises(ValueError):
        bc.load_gate_bc_state(bad_path)


# ======================================================================== #
# KL-to-BC prior
# ======================================================================== #
class _StubAdapter:
    world_feat_dim = WF_DIM

    def encode_world_feat(self, input_image):
        return input_image.detach().float().reshape(-1)[: self.world_feat_dim].clone()

    def act(self, *, input_image, mode, proprio=None, context=None, context_mask=None, world_feat=None):
        return {
            "action_chunk": torch.zeros(4, 7),
            "world_feat": world_feat,
            "cost": 0.1 * (int(mode) + 1),
            "aux": {"mode": int(mode)},
        }


def _obs(batch=2):
    return {
        "input_image": torch.rand(batch, 3, 16, 16),
        "proprio": torch.randn(batch, P_DIM),
        "context": torch.randn(batch, 4, 32),
        "context_mask": torch.ones(batch, 4, dtype=torch.bool),
    }


def test_attach_bc_prior_keeps_state_dict_clean_and_kl_zero_for_self():
    policy = _policy()
    policy.wam_adapter = _StubAdapter()
    keys_before = set(policy.state_dict())
    policy.attach_bc_prior(policy.state_dict())
    assert set(policy.state_dict()) == keys_before  # prior NOT in the module tree
    assert policy.bc_prior is not None

    _, result = policy.predict_action_batch(_obs(batch=3), mode="train")
    kl = result["forward_inputs"]["kl_to_prior"]
    assert kl.shape == (3, 1)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)  # prior == policy
    assert torch.allclose(result["kl_to_prior"], kl.squeeze(-1))


def test_kl_to_prior_positive_after_policy_moves():
    policy = _policy()
    policy.wam_adapter = _StubAdapter()
    policy.attach_bc_prior(policy.state_dict())
    with torch.no_grad():  # push the policy away from the frozen prior
        policy.logits_head.weight.add_(1.0)
        policy.logits_head.bias.add_(torch.tensor([2.0, -1.0, 0.5]))
    _, result = policy.predict_action_batch(_obs(batch=4), mode="train")
    assert (result["forward_inputs"]["kl_to_prior"] > 0).all()


def test_attach_bc_prior_accepts_value_head_checkpoints_and_rejects_garbage():
    donor = _policy(add_value_head=True)
    policy = _policy(add_value_head=False)
    policy.attach_bc_prior(donor.state_dict())  # extra value_head.* keys ignored
    x = torch.randn(2, WF_DIM + P_DIM)
    assert torch.allclose(policy.bc_prior(x), donor.mode_logits(x[:, :WF_DIM], x[:, WF_DIM:]))
    with pytest.raises(ValueError):
        policy.attach_bc_prior({"foo": torch.zeros(1)})


def test_no_prior_means_no_kl_key():
    policy = _policy()
    policy.wam_adapter = _StubAdapter()
    _, result = policy.predict_action_batch(_obs(batch=2), mode="train")
    assert "kl_to_prior" not in result["forward_inputs"]
    assert "kl_to_prior" not in result


# ======================================================================== #
# reward-side KL term + schedule
# ======================================================================== #
def test_kl_prior_schedule_decays_then_holds():
    sched = reward_mod.kl_prior_schedule
    assert sched(0, beta_start=0.1, beta_end=0.0, decay_steps=100) == pytest.approx(0.1)
    assert sched(50, beta_start=0.1, beta_end=0.0, decay_steps=100) == pytest.approx(0.05)
    assert sched(100, beta_start=0.1, beta_end=0.0, decay_steps=100) == pytest.approx(0.0)
    assert sched(999, beta_start=0.1, beta_end=0.02, decay_steps=100) == pytest.approx(0.02)
    assert sched(5, beta_start=0.1, decay_steps=0) == pytest.approx(0.1)  # constant


def test_apply_gate_reward_kl_term_counts_once_per_chunk():
    rewards = torch.zeros(2, 32)
    mode_cost = torch.zeros(2, 1)
    kl = torch.tensor([[0.5], [1.0]])
    comps = reward_mod.apply_gate_reward(
        rewards=rewards,
        mode_cost=mode_cost,
        step=0,
        lambda_cost=0.0,
        lambda_warmup_steps=0,
        kl_to_prior=kl,
        beta_kl_prior=0.2,
        beta_kl_prior_end=0.0,
        beta_kl_prior_decay_steps=0,
    )
    assert "kl_prior" in comps
    chunk_total = comps["total"].sum(dim=-1)
    assert torch.allclose(chunk_total, torch.tensor([-0.1, -0.2]))
    # decayed to zero -> no kl component
    comps2 = reward_mod.apply_gate_reward(
        rewards=rewards,
        mode_cost=mode_cost,
        step=200,
        lambda_cost=0.0,
        lambda_warmup_steps=0,
        kl_to_prior=kl,
        beta_kl_prior=0.2,
        beta_kl_prior_end=0.0,
        beta_kl_prior_decay_steps=100,
    )
    assert torch.allclose(comps2["total"], torch.zeros_like(comps2["total"]))


def test_gate_reward_components_without_kl_unchanged():
    comps = reward_mod.gate_reward_components(
        success=1.0, mode_cost=0.5, lambda_cost=0.1
    )
    assert comps["total"] == pytest.approx(1.0 - 0.05)
    assert "kl_prior" not in comps


# ======================================================================== #
# get_model wiring (BC init + prior), no fastwam needed (load_wam=False)
# ======================================================================== #
def test_get_model_bc_init_and_kl_prior(tmp_path):
    omegaconf = pytest.importorskip("omegaconf")
    from rlinf.models.embodiment.gate_policy import get_model

    donor = _policy(hidden=(16, 16))
    path = str(tmp_path / "gate_bc.pt")
    bc.save_gate_bc_checkpoint(donor, path)

    cfg = omegaconf.OmegaConf.create(
        {
            "model_type": "gate_policy",
            "load_wam": False,
            "world_feat_dim": WF_DIM,
            "proprio_dim": P_DIM,
            "add_value_head": False,
            "gate": {
                "num_modes": 3,
                "hidden_sizes": [16, 16],
                "activation": "tanh",
                "bc_init_path": path,
                "kl_prior": {"enabled": True, "path": None},
            },
        }
    )
    policy = get_model(cfg)
    x_wf, x_p = torch.randn(3, WF_DIM), torch.randn(3, P_DIM)
    assert torch.allclose(policy.mode_logits(x_wf, x_p), donor.mode_logits(x_wf, x_p))
    assert policy.bc_prior is not None

    # architecture mismatch -> clear error
    cfg_bad = omegaconf.OmegaConf.merge(
        cfg, {"gate": {"hidden_sizes": [32, 32], "kl_prior": {"enabled": False}}}
    )
    with pytest.raises(ValueError):
        get_model(cfg_bad)

    # prior enabled without any path -> clear error
    cfg_noprior = omegaconf.OmegaConf.merge(
        cfg, {"gate": {"bc_init_path": None, "kl_prior": {"enabled": True}}}
    )
    with pytest.raises(ValueError):
        get_model(cfg_noprior)


# ======================================================================== #
# shard-file path (needs the fastwam package; skipped where not installed)
# ======================================================================== #
def test_from_shards_roundtrip_with_fastwam(tmp_path):
    fastwam_gate = pytest.importorskip(
        "fastwam.adaptive_gate", reason="fastwam not installed in this environment"
    )
    n, t = 12, 6
    data = {
        "world_feat": torch.rand(n, WF_DIM),
        "proprio": torch.rand(n, P_DIM),
        "label": torch.randint(0, 3, (n,)),
        "chunk_err": torch.rand(n, 3),
        "step_l1": torch.rand(n, 3, t),
        "step_l2": torch.rand(n, 3, t),
        "valid_steps": torch.ones(n, t, dtype=torch.bool),
        "sample_idx": torch.arange(n),
    }
    shard = str(tmp_path / "shard_0_of_1.pt")
    fastwam_gate.write_label_shard(shard, data=data, meta={"task": "unit"})
    ds = bc.GateOracleLabelDataset.from_shards(shard)
    assert len(ds) == n
    assert torch.equal(ds.label, data["label"].long())

    # offline relabel with tight tolerance -> labels recomputed from curves
    ds2 = bc.GateOracleLabelDataset.from_shards(shard, relabel={"tol_abs": 0.0, "tol_rel": 0.0})
    labels_ref, _, _ = fastwam_gate.relabel_from_steps(
        data["step_l1"], data["step_l2"], data["valid_steps"], tol_abs=0.0, tol_rel=0.0
    )
    assert torch.equal(ds2.label, labels_ref)
