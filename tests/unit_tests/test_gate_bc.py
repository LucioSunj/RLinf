# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import os
import sys
import types

import pytest
import torch

from _gate_test_imports import load_gate_modules


mods = load_gate_modules()
GatePolicy = mods.gate.GatePolicy
bc = mods.bc

WF_DIM, P_DIM, TEXT_DIM = 8, 5, 4


def _policy(add_value_head=False, hidden=(16, 16)):
    return GatePolicy(
        world_feat_dim=WF_DIM,
        proprio_dim=P_DIM,
        text_feat_dim=TEXT_DIM,
        num_modes=2,
        hidden_sizes=hidden,
        add_value_head=add_value_head,
    )


def _synthetic_labeled(n=800, seed=0, *, with_groups=True):
    generator = torch.Generator().manual_seed(seed)
    world_feat = torch.randn(n, WF_DIM, generator=generator)
    proprio = torch.randn(n, P_DIM, generator=generator)
    text_feat = torch.randn(n, TEXT_DIM, generator=generator)
    joined = torch.cat([world_feat, proprio, text_feat], dim=-1)
    weights = torch.randn(joined.shape[-1], generator=generator)
    labels = (joined @ weights > 0).long()
    data = {
        "world_feat": world_feat,
        "proprio": proprio,
        "text_feat": text_feat,
        "label": labels,
    }
    if with_groups:
        data["episode_id"] = torch.arange(n) // 20
    return data


def test_dataset_validates_two_mode_text_contract():
    data = _synthetic_labeled(n=20)
    dataset = bc.GateOracleLabelDataset(
        data, meta={"cost_table": {"uncond": 0.1, "idm": 1.0}}
    )
    assert dataset.world_feat_dim == WF_DIM
    assert dataset.proprio_dim == P_DIM
    assert dataset.text_feat_dim == TEXT_DIM
    assert dataset[3]["text_feat"].shape == (TEXT_DIM,)
    with pytest.raises(ValueError, match="text_feat"):
        bc.GateOracleLabelDataset({k: v for k, v in data.items() if k != "text_feat"})
    with pytest.raises(ValueError, match="UNCOND=0 or IDM=1"):
        bc.GateOracleLabelDataset({**data, "label": torch.full((20,), 2)})


def test_class_balance_is_opt_in_and_expected_cost_is_two_mode():
    assert bc.GateBCConfig().class_weight_power == 0.0
    labels = torch.tensor([0] * 80 + [1] * 20)
    weights = bc.class_balance_weights(labels, 2, power=1.0)
    assert weights[0] < weights[1]
    assert weights.mean().item() == pytest.approx(1.0)
    assert bc.expected_mode_cost(
        torch.tensor([0, 1]), {"uncond": 0.2, "idm": 1.0}
    ) == pytest.approx(0.6)


def test_train_gate_bc_learns_and_splits_by_episode(tmp_path):
    dataset = bc.GateOracleLabelDataset(_synthetic_labeled(n=800, seed=0))
    policy = _policy()
    result = bc.train_gate_bc(
        policy,
        dataset,
        bc.GateBCConfig(
            epochs=30,
            batch_size=128,
            lr=3.0e-3,
            val_fraction=0.2,
            seed=0,
            log_every_epochs=0,
        ),
    )
    assert result["best"]["accuracy"] > 0.85
    assert result["split_kind"] == "group:episode_id"
    assert result["num_train"] + result["num_val"] == len(dataset)
    for mode in range(2):
        assert f"recall/mode_{mode}" in result["best"]

    path = str(tmp_path / "gate_bc.pt")
    bc.save_gate_bc_checkpoint(policy, path, meta={"num_modes": 2})
    fresh = _policy()
    fresh.load_state_dict(bc.load_gate_bc_state(path))
    assert torch.allclose(
        fresh.mode_logits(
            dataset.world_feat[:4], dataset.proprio[:4], dataset.text_feat[:4]
        ),
        policy.mode_logits(
            dataset.world_feat[:4], dataset.proprio[:4], dataset.text_feat[:4]
        ),
    )
    assert os.path.exists(path + ".meta.json")


def test_row_split_emits_leakage_warning():
    dataset = bc.GateOracleLabelDataset(
        _synthetic_labeled(n=80, seed=1, with_groups=False)
    )
    with pytest.warns(UserWarning, match="may leak adjacent states"):
        result = bc.train_gate_bc(
            _policy(),
            dataset,
            bc.GateBCConfig(
                epochs=1,
                batch_size=32,
                val_fraction=0.2,
                log_every_epochs=0,
            ),
        )
    assert result["split_kind"] == "row"


def test_partial_unknown_group_ids_disable_group_split():
    data = _synthetic_labeled(n=80, seed=5)
    data["episode_id"][0] = -1
    with pytest.warns(UserWarning, match="unknown negative ids"):
        dataset = bc.GateOracleLabelDataset(data)
    assert dataset.group_id is None
    with pytest.warns(UserWarning, match="row-wise split"):
        result = bc.train_gate_bc(
            _policy(),
            dataset,
            bc.GateBCConfig(
                epochs=1,
                batch_size=32,
                val_fraction=0.2,
                log_every_epochs=0,
            ),
        )
    assert result["split_kind"] == "row"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"lr": float("nan")}, "lr"),
        ({"lr": 0.0}, "lr"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"weight_decay": float("inf")}, "weight_decay"),
        ({"val_fraction": -0.1}, "val_fraction"),
        ({"val_fraction": 1.0}, "val_fraction"),
        ({"class_weight_power": -1.0}, "class_weight_power"),
        ({"label_smoothing": 1.0}, "label_smoothing"),
        ({"log_every_epochs": -1}, "log_every_epochs"),
    ],
)
def test_gate_bc_config_rejects_invalid_hyperparameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        bc.GateBCConfig(**kwargs)


def test_evaluate_gate_bc_accepts_task_text_feature():
    data = _synthetic_labeled(n=40, seed=2)
    policy = _policy()
    metrics = bc.evaluate_gate_bc(
        policy,
        data["world_feat"],
        data["proprio"],
        data["text_feat"],
        data["label"],
        cost_table={"uncond": 0.1, "idm": 1.0},
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "mean_cost/pred" in metrics


def test_load_gate_bc_state_accepts_wrapped_payload_and_rejects_bad(tmp_path):
    policy = _policy()
    wrapped = tmp_path / "wrapped.pt"
    torch.save({"state_dict": policy.state_dict(), "meta": {"x": 1}}, wrapped)
    assert set(bc.load_gate_bc_state(str(wrapped))) == set(policy.state_dict())
    bad = tmp_path / "bad.pt"
    torch.save({"foo": "bar"}, bad)
    with pytest.raises(ValueError):
        bc.load_gate_bc_state(str(bad))


def test_bc_init_rejects_non_value_head_missing_and_unexpected_keys():
    policy = _policy()
    state = policy.state_dict()
    state["stale_router.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected=.*stale_router"):
        policy.load_bc_init(state)

    missing = dict(policy.state_dict())
    missing.pop("logits_head.bias")
    with pytest.raises(ValueError, match="missing=.*logits_head.bias"):
        policy.load_bc_init(missing)

    donor_with_value = _policy(add_value_head=True)
    no_value = _policy(add_value_head=False)
    _, unexpected = no_value.load_bc_init(donor_with_value.state_dict())
    assert unexpected and all(key.startswith("value_head.") for key in unexpected)


def test_from_shards_reports_and_filters_absolute_oracle_quality(monkeypatch):
    data = _synthetic_labeled(n=4, seed=3)
    data.update(
        {
            "best_err": torch.tensor([0.1, 0.2, 0.8, float("inf")]),
            "idm_err": torch.tensor([0.2, 0.3, 0.9, float("inf")]),
        }
    )
    adaptive = types.ModuleType("fastwam.adaptive_gate")
    adaptive.MODE_ORDER = ("uncond", "idm")
    adaptive.WORLD_FEAT_LAYOUT = "world_layout_v1"
    adaptive.TEXT_FEAT_LAYOUT = "text_layout_v1"
    shard_meta = {
        "task": "unit",
        "mode_order": ["uncond", "idm"],
        "world_feat_layout": adaptive.WORLD_FEAT_LAYOUT,
        "text_feat_layout": adaptive.TEXT_FEAT_LAYOUT,
        "world_feat_dim": WF_DIM,
        "proprio_dim": P_DIM,
        "text_feat_dim": TEXT_DIM,
    }
    adaptive.load_label_shards = lambda shards: (data, shard_meta)
    adaptive.relabel_from_steps = lambda *args, **kwargs: None
    adaptive.quality_metadata = lambda chunk: {}
    fastwam = types.ModuleType("fastwam")
    fastwam.__path__ = []
    monkeypatch.setitem(sys.modules, "fastwam", fastwam)
    monkeypatch.setitem(sys.modules, "fastwam.adaptive_gate", adaptive)

    with pytest.warns(UserWarning, match="no absolute-quality filter"):
        unfiltered = bc.GateOracleLabelDataset.from_shards("ignored")
    assert len(unfiltered) == 4
    assert unfiltered.meta["quality_report"]["low_quality_fraction"] == pytest.approx(0.5)

    filtered = bc.GateOracleLabelDataset.from_shards(
        "ignored", max_best_err=0.5, max_idm_err=0.5
    )
    assert len(filtered) == 2
    assert filtered.meta["num_samples_before_quality_filter"] == 4


def test_relabel_inherits_and_records_effective_exec_horizon(monkeypatch):
    n, horizon = 3, 32
    data = _synthetic_labeled(n=n, seed=31)
    data.update(
        {
            "step_l1": torch.zeros(n, 2, horizon),
            "step_l2": torch.zeros(n, 2, horizon),
            "valid_steps": torch.ones(n, horizon, dtype=torch.bool),
            "best_err": torch.zeros(n),
            "idm_err": torch.zeros(n),
        }
    )
    adaptive = types.ModuleType("fastwam.adaptive_gate")
    adaptive.MODE_ORDER = ("uncond", "idm")
    adaptive.WORLD_FEAT_LAYOUT = "world_layout_v1"
    adaptive.TEXT_FEAT_LAYOUT = "text_layout_v1"
    meta = {
        "mode_order": ["uncond", "idm"],
        "world_feat_layout": adaptive.WORLD_FEAT_LAYOUT,
        "text_feat_layout": adaptive.TEXT_FEAT_LAYOUT,
        "world_feat_dim": WF_DIM,
        "proprio_dim": P_DIM,
        "text_feat_dim": TEXT_DIM,
        "exec_horizon": 10,
    }
    captured = {}

    def _relabel(*args, **kwargs):
        captured.update(kwargs)
        return (
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n, 2),
            torch.ones(n, dtype=torch.bool),
        )

    adaptive.load_label_shards = lambda shards: (data, meta)
    adaptive.relabel_from_steps = _relabel
    adaptive.quality_metadata = lambda chunk: {
        "best_err": chunk.min(dim=-1).values,
        "idm_err": chunk[:, 1],
        "idm_regret": chunk[:, 1] - chunk.min(dim=-1).values,
    }
    fastwam = types.ModuleType("fastwam")
    fastwam.__path__ = []
    monkeypatch.setitem(sys.modules, "fastwam", fastwam)
    monkeypatch.setitem(sys.modules, "fastwam.adaptive_gate", adaptive)

    dataset = bc.GateOracleLabelDataset.from_shards(
        "ignored",
        relabel={
            "metric": "l1",
            "exec_horizon": None,
            "tol_abs": 0.02,
            "tol_rel": 0.1,
        },
    )
    assert captured["exec_horizon"] == 10
    assert dataset.meta["source_exec_horizon"] == 10
    assert dataset.meta["exec_horizon"] == 10
    assert dataset.meta["relabel"]["exec_horizon"] == 10


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("world_feat_layout", "same_dim_wrong_semantics", "world_feat_layout"),
        ("text_feat_layout", "same_dim_wrong_semantics", "text_feat_layout"),
        ("mode_order", ["idm", "uncond"], "mode_order"),
        ("world_feat_dim", WF_DIM + 1, "world_feat_dim"),
        ("proprio_dim", P_DIM + 1, "proprio_dim"),
        ("text_feat_dim", TEXT_DIM + 1, "text_feat_dim"),
    ],
)
def test_from_shards_rejects_semantically_incompatible_features(
    monkeypatch, field, bad_value, match
):
    data = _synthetic_labeled(n=4, seed=4)
    data.update(
        {
            "best_err": torch.full((4,), 0.1),
            "idm_err": torch.full((4,), 0.2),
        }
    )
    adaptive = types.ModuleType("fastwam.adaptive_gate")
    adaptive.MODE_ORDER = ("uncond", "idm")
    adaptive.WORLD_FEAT_LAYOUT = "world_layout_v1"
    adaptive.TEXT_FEAT_LAYOUT = "text_layout_v1"
    meta = {
        "mode_order": ["uncond", "idm"],
        "world_feat_layout": adaptive.WORLD_FEAT_LAYOUT,
        "text_feat_layout": adaptive.TEXT_FEAT_LAYOUT,
        "world_feat_dim": WF_DIM,
        "proprio_dim": P_DIM,
        "text_feat_dim": TEXT_DIM,
    }
    meta[field] = bad_value
    adaptive.load_label_shards = lambda shards: (data, meta)
    adaptive.relabel_from_steps = lambda *args, **kwargs: None
    adaptive.quality_metadata = lambda chunk: {}
    fastwam = types.ModuleType("fastwam")
    fastwam.__path__ = []
    monkeypatch.setitem(sys.modules, "fastwam", fastwam)
    monkeypatch.setitem(sys.modules, "fastwam.adaptive_gate", adaptive)

    with pytest.raises(ValueError, match=match):
        bc.GateOracleLabelDataset.from_shards("ignored")


def _install_sidecar_fastwam_stub(monkeypatch):
    adaptive = types.ModuleType("fastwam.adaptive_gate")
    adaptive.MODE_ORDER = ("uncond", "idm")
    adaptive.WORLD_FEAT_LAYOUT = "spatial_2x2_plus_channel_std_v1"
    adaptive.TEXT_FEAT_LAYOUT = "masked_mean_adaptive_avg_pool_v1"
    fastwam = types.ModuleType("fastwam")
    fastwam.__path__ = []
    monkeypatch.setitem(sys.modules, "fastwam", fastwam)
    monkeypatch.setitem(sys.modules, "fastwam.adaptive_gate", adaptive)
    return adaptive


def _training_cfg(*, checkpoint=None):
    gate = {
        "bc_init_path": checkpoint,
        "explore_eps": 0.1,
        "kl_prior": {
            "enabled": checkpoint is not None,
            "path": checkpoint,
            "beta": 0.05,
            "beta_end": 0.0,
            "decay_steps": 200,
        },
    }
    return {
        "actor": {"seed": 13, "model": {"gate": gate}},
        "runner": {"ckpt_path": checkpoint},
        "algorithm": {
            "adv_type": "grpo",
            "loss_type": "actor",
            "reward_type": "chunk_level",
            "group_size": 8,
            "normalize_advantages": True,
            "update_epoch": 2,
            "entropy_bonus": 0.01,
        },
        "gate_reward": {
            "w_success": 1.0,
            "lambda_start": 0.0,
            "lambda_cost": 0.05,
            "lambda_warmup_steps": 200,
        },
        "gate_diagnostics": {
            "collapse": {"enabled": True, "target_idm_usage": 0.5}
        },
    }


def test_gate_training_provenance_binds_init_objective_and_budget(tmp_path):
    scratch_cfg = _training_cfg()
    scratch_cfg["gate_diagnostics"]["evidence_run_id"] = "a" * 64
    scratch = bc.build_gate_training_provenance(scratch_cfg)
    assert scratch["seed"] == 13
    assert scratch["target_idm_usage"] == 0.5
    assert scratch["evidence_run_id"] == "a" * 64
    assert scratch["collapse"]["enabled"] is True
    assert scratch["initialization"]["kind"] == "scratch"
    assert scratch["objective"]["group_size"] == 8
    assert scratch["reward"]["lambda_warmup_steps"] == 200

    checkpoint = tmp_path / "uplift.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "uplift.pt.meta.json").write_text(
        '{"kind":"gate_uplift","schema_version":1}', encoding="utf-8"
    )
    provenance = bc.build_gate_training_provenance(
        _training_cfg(checkpoint=str(checkpoint))
    )
    assert provenance["initialization"]["kind"] == "gate_uplift"
    assert len(provenance["initialization"]["checkpoint_sha256"]) == 64
    assert provenance["kl_prior"]["checkpoint_sha256"] == (
        provenance["initialization"]["checkpoint_sha256"]
    )

    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    bad = _training_cfg(checkpoint=str(checkpoint))
    bad["runner"]["ckpt_path"] = str(other)
    with pytest.raises(ValueError, match="same Gate initialization"):
        bc.build_gate_training_provenance(bad)

    invalid_run_id = _training_cfg()
    invalid_run_id["gate_diagnostics"]["evidence_run_id"] = "not-a-sha"
    with pytest.raises(ValueError, match="evidence_run_id"):
        bc.build_gate_training_provenance(invalid_run_id)


def test_gate_rl_resume_metadata_rejects_changed_training_contract():
    expected = {
        "kind": "gate_rl",
        "schema_version": 1,
        "step": 20,
        "gate": {"hidden_sizes": [16, 16]},
        "wam_provenance": {"ckpt_fingerprint": "w" * 64},
        "training": {"seed": 3, "target_idm_usage": 0.5},
    }
    bc.validate_gate_rl_resume_metadata(dict(expected), expected)
    bc.validate_gate_rl_resume_metadata(
        dict(expected), expected, expected_step=20
    )
    changed = {
        **expected,
        "training": {"seed": 4, "target_idm_usage": 0.5},
    }
    with pytest.raises(ValueError, match="training contract"):
        bc.validate_gate_rl_resume_metadata(changed, expected)
    with pytest.raises(ValueError, match="JSON object"):
        bc.validate_gate_rl_resume_metadata([], expected)
    with pytest.raises(ValueError, match="sidecar step"):
        bc.validate_gate_rl_resume_metadata(
            dict(expected), expected, expected_step=19
        )


def _valid_bc_metadata(policy, adaptive):
    policy.bc_expected_provenance = {
        "task": "unit_task",
        "backbone_kind": "idm",
        "ckpt_fingerprint": "unit_ckpt_sha256",
        "dataset_stats_fingerprint": "unit_stats_sha256",
        "num_video_frames": 9,
        "inference_steps": 20,
        "solver_fingerprint": "unit_solver_sha256",
        "context_len": 128,
        "model_dtype": "torch.bfloat16",
        "exec_horizon": 10,
        "action_horizon": 32,
        "cost_table_path": None,
    }
    return {
        "kind": "gate_bc",
        "shard_meta": {
            "mode_order": list(adaptive.MODE_ORDER),
            "world_feat_layout": adaptive.WORLD_FEAT_LAYOUT,
            "text_feat_layout": adaptive.TEXT_FEAT_LAYOUT,
            "world_feat_dim": policy.world_feat_dim,
            "proprio_dim": policy.proprio_dim,
            "text_feat_dim": policy.text_feat_dim,
            "task": "unit_task",
            "backbone_kind": "idm",
            "ckpt_fingerprint": "unit_ckpt_sha256",
            "dataset_stats_fingerprint": "unit_stats_sha256",
            "num_video_frames": 9,
            "inference_steps": 20,
            "solver_fingerprint": "unit_solver_sha256",
            "context_len": 128,
            "model_dtype": "torch.bfloat16",
            "exec_horizon": 10,
            "action_horizon": 32,
        },
        "gate": {
            "world_feat_dim": policy.world_feat_dim,
            "proprio_dim": policy.proprio_dim,
            "text_feat_dim": policy.text_feat_dim,
            "num_modes": policy.num_modes,
            "hidden_sizes": list(policy.hidden_sizes),
            "activation": policy.activation,
            "add_value_head": policy.add_value_head,
        },
    }


def test_bc_sidecar_validates_architecture_and_feature_provenance(
    tmp_path, monkeypatch
):
    adaptive = _install_sidecar_fastwam_stub(monkeypatch)
    policy = _policy()
    path = str(tmp_path / "gate_bc.pt")
    metadata = _valid_bc_metadata(policy, adaptive)
    bc.save_gate_bc_checkpoint(policy, path, meta=metadata)
    loaded = bc.load_gate_bc_state(path, expected_policy=policy)
    assert set(loaded) == set(policy.state_dict())

    rl_path = str(tmp_path / "gate_rl.pt")
    torch.save(policy.state_dict(), rl_path)
    rl_meta = bc.build_gate_policy_sidecar_metadata(policy, step=12)
    bc.save_gate_policy_sidecar(rl_path, rl_meta)
    assert rl_meta["kind"] == "gate_rl" and rl_meta["step"] == 12
    assert set(bc.load_gate_bc_state(rl_path, expected_policy=policy)) == set(
        policy.state_dict()
    )

    metadata["gate"]["hidden_sizes"] = [999]
    with open(path + ".meta.json", "w") as handle:
        import json

        json.dump(metadata, handle)
    with pytest.raises(ValueError, match="architecture metadata"):
        bc.load_gate_bc_state(path, expected_policy=policy)

    metadata = _valid_bc_metadata(policy, adaptive)
    metadata["shard_meta"]["world_feat_layout"] = "wrong_layout_same_dim"
    with open(path + ".meta.json", "w") as handle:
        import json

        json.dump(metadata, handle)
    with pytest.raises(ValueError, match="feature provenance"):
        bc.load_gate_bc_state(path, expected_policy=policy)


def test_bc_sidecar_resolves_expected_wam_from_cost_profile(tmp_path, monkeypatch):
    adaptive = _install_sidecar_fastwam_stub(monkeypatch)
    policy = _policy()
    path = str(tmp_path / "gate_bc.pt")
    metadata = _valid_bc_metadata(policy, adaptive)
    bc.save_gate_bc_checkpoint(policy, path, meta=metadata)

    cost_path = tmp_path / "cost.yaml"
    with open(cost_path, "w") as handle:
        import json

        json.dump(
            {
                "meta": {
                    "task": "unit_task",
                    "backbone_kind": "idm",
                    "ckpt_fingerprint": "unit_ckpt_sha256",
                }
            },
            handle,
        )
    policy.bc_expected_provenance = {
        "task": "unit_task",
        "backbone_kind": "idm",
        "ckpt_fingerprint": None,
        "dataset_stats_fingerprint": "unit_stats_sha256",
        "num_video_frames": 9,
        "inference_steps": 20,
        "solver_fingerprint": "unit_solver_sha256",
        "context_len": 128,
        "model_dtype": "torch.bfloat16",
        "exec_horizon": 10,
        "action_horizon": 32,
        "cost_table_path": str(cost_path),
    }
    assert set(bc.load_gate_bc_state(path, expected_policy=policy)) == set(
        policy.state_dict()
    )

    policy.bc_expected_provenance["cost_table_path"] = str(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="measured cost profile"):
        bc.load_gate_bc_state(path, expected_policy=policy)

    metadata = _valid_bc_metadata(policy, adaptive)
    metadata["shard_meta"]["task"] = "different_same_shape_task"
    with open(path + ".meta.json", "w") as handle:
        import json

        json.dump(metadata, handle)
    with pytest.raises(ValueError, match="feature provenance"):
        bc.load_gate_bc_state(path, expected_policy=policy)

    metadata = _valid_bc_metadata(policy, adaptive)
    metadata["shard_meta"]["dataset_stats_fingerprint"] = "wrong_stats"
    with open(path + ".meta.json", "w") as handle:
        import json

        json.dump(metadata, handle)
    with pytest.raises(ValueError, match="feature provenance"):
        bc.load_gate_bc_state(path, expected_policy=policy)

    metadata = _valid_bc_metadata(policy, adaptive)
    metadata["shard_meta"]["exec_horizon"] = 24
    with open(path + ".meta.json", "w") as handle:
        import json

        json.dump(metadata, handle)
    with pytest.raises(ValueError, match="feature provenance"):
        bc.load_gate_bc_state(path, expected_policy=policy)

    metadata = _valid_bc_metadata(policy, adaptive)
    metadata["shard_meta"]["ckpt_fingerprint"] = "another_wam"
    with open(path + ".meta.json", "w") as handle:
        import json

        json.dump(metadata, handle)
    with pytest.raises(ValueError, match="feature provenance"):
        bc.load_gate_bc_state(path, expected_policy=policy)


def test_bc_checkpoint_without_sidecar_warns_as_legacy(tmp_path):
    policy = _policy()
    path = tmp_path / "legacy.pt"
    torch.save(policy.state_dict(), path)
    with pytest.raises(ValueError, match="no provenance sidecar"):
        bc.load_gate_bc_state(str(path), expected_policy=policy)
    policy.allow_legacy_gate_checkpoint = True
    with pytest.warns(UserWarning, match="legacy checkpoint"):
        loaded = bc.load_gate_bc_state(str(path), expected_policy=policy)
    assert set(loaded) == set(policy.state_dict())
