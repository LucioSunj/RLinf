from __future__ import annotations

import copy
import enum
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from _gate_test_imports import load_gate_modules


mods = load_gate_modules()
paired = mods.paired
benefit = mods.benefit
collector_mod = mods.collector
GatePolicy = mods.gate.GatePolicy


def _meta(**updates):
    value = {
        "task": "libero_dual_regime_fused_2cam224_1e-4",
        "backbone_kind": "idm",
        "ckpt_fingerprint": "adaptive-v1:abc",
        "ckpt_file_sha256": "c" * 64,
        "dataset_stats_fingerprint": "stats-sha",
        "num_video_frames": 9,
        "inference_steps": 20,
        "solver_fingerprint": "f" * 64,
        "context_len": 128,
        "model_dtype": "torch.bfloat16",
        "exec_horizon": 10,
        "action_horizon": 32,
        "world_feat_layout": "world-layout-v1",
        "text_feat_layout": "text-layout-v1",
        "mode_order": ["uncond", "idm"],
        "world_feat_dim": 2,
        "proprio_dim": 1,
        "text_feat_dim": 1,
        "snapshot_schema": "libero-gate-snapshot-v1",
        "episode_manifest_sha256": "d" * 64,
        "heldout_test_manifest_sha256": "e" * 64,
        "libero_plus_commit": "a" * 40,
        "manifest_split": "train",
        "collector_seed": 7,
        "continuation_mode": "uncond",
        "max_reference_decisions": 70,
        "max_branch_decisions": 70,
        "sensitivity_fraction": 0.2,
        "reference_policy_mix": ["uncond", "idm", "random_0.5"],
        "reference_policy_assignment": "balanced_shuffled_v1",
        "reference_assignment_manifest_sha256": "d" * 64,
        "reference_policy_episode_assignments": {"episode-0": "uncond"},
    }
    value.update(updates)
    value["reference_assignment_sha256"] = paired.reference_assignment_sha256(
        value["reference_policy_episode_assignments"]
    )
    return value


def _payload(episode_uid="episode-0", trajectory_id=0, rows=2, **meta_updates):
    assignments = dict(
        meta_updates.pop(
            "reference_policy_episode_assignments", {episode_uid: "random_0.5"}
        )
    )
    helpful = torch.tensor([(index % 2) == 0 for index in range(rows)])
    data = {
        "world_feat": torch.stack(
            [torch.tensor([1.0, 0.0]) if flag else torch.tensor([-1.0, 0.0]) for flag in helpful]
        ),
        "proprio": torch.zeros(rows, 1),
        "text_feat": torch.zeros(rows, 1),
        "trajectory_id": torch.full((rows,), trajectory_id, dtype=torch.int64),
        "decision_index": torch.arange(rows),
        "task_id": torch.zeros(rows, dtype=torch.int64),
        "source_mode": torch.arange(rows) % 2,
        "branch_seed": torch.arange(rows) + 10,
        "success_uncond": ~helpful,
        "success_idm": torch.ones(rows, dtype=torch.bool),
        "progress_1_uncond": (~helpful).float(),
        "progress_1_idm": torch.ones(rows),
        "progress_3_uncond": (~helpful).float(),
        "progress_3_idm": torch.ones(rows),
        "sensitivity_mask": torch.zeros(rows, dtype=torch.bool),
        "sensitivity_success_uncond": torch.zeros(rows, dtype=torch.bool),
        "sensitivity_success_idm": torch.zeros(rows, dtype=torch.bool),
        "sensitivity_progress_3_uncond": torch.zeros(rows),
        "sensitivity_progress_3_idm": torch.zeros(rows),
    }
    records = [
        {
            "state_id": f"{episode_uid}-state-{index}",
            "episode_uid": episode_uid,
            "base_task": "pick_mug",
            "task_suite_name": "libero_10",
            "task_description": "pick up the mug",
            "trial_id": 0,
            "reset_state_id": 0,
            "env_seed": 11,
            "factor": "layout",
            "level": "L1",
            "perturbation_id": "layout-001",
            "phase": "unknown",
            "phase_reliable": False,
            "snapshot_path": f"snapshot-{index}.pt",
            "snapshot_sha256": f"{index:064x}",
            "asset_ids": ["asset-a"],
        }
        for index in range(rows)
    ]
    return {
        "schema": paired.PAIRED_SCHEMA,
        "meta": _meta(
            reference_policy_episode_assignments=assignments, **meta_updates
        ),
        "data": data,
        "records": records,
    }


def test_paired_shards_merge_by_complete_episode_and_bind_provenance(tmp_path):
    assignments = {"episode-a": "random_0.5", "episode-b": "random_0.5"}
    first = _payload(
        "episode-a",
        trajectory_id=0,
        reference_policy_episode_assignments=assignments,
    )
    second = _payload(
        "episode-b",
        trajectory_id=0,
        reference_policy_episode_assignments=assignments,
    )
    first_path = tmp_path / "paired-0.pt"
    second_path = tmp_path / "paired-1.pt"
    paired.save_paired_shard(first_path, data=first["data"], records=first["records"], meta=first["meta"])
    paired.save_paired_shard(second_path, data=second["data"], records=second["records"], meta=second["meta"])

    data, records, meta = paired.load_paired_shards([str(first_path), str(second_path)])
    assert len(records) == 4
    assert torch.unique(data["trajectory_id"]).numel() == 2
    assert meta["num_trajectories"] == 2
    assert len(meta["paired_dataset_fingerprint"]) == 64

    bad = _payload(
        "episode-a",
        inference_steps=19,
        reference_policy_episode_assignments=assignments,
    )
    bad_path = tmp_path / "bad.pt"
    paired.save_paired_shard(bad_path, data=bad["data"], records=bad["records"], meta=bad["meta"])
    with pytest.raises(ValueError, match="provenance differs"):
        paired.load_paired_shards([str(first_path), str(bad_path)])


def test_paired_payload_enforces_parent_reference_assignment():
    payload = _payload(
        "episode-a",
        rows=1,
        reference_policy_episode_assignments={"episode-a": "uncond"},
    )
    payload["data"]["source_mode"][0] = 1
    with pytest.raises(ValueError, match="always-UNCOND trajectory used IDM"):
        paired.validate_paired_payload(payload)


def test_paired_directory_contract_verifies_snapshots_outcomes_and_folds(tmp_path):
    payloads = [
        _payload(f"episode-{group}", trajectory_id=group, rows=1)
        for group in range(5)
    ]
    data = {
        key: torch.cat([payload["data"][key] for payload in payloads])
        for key in payloads[0]["data"]
    }
    records = []
    snapshot_dir = tmp_path / "snapshots-source"
    snapshot_dir.mkdir()
    for group, payload in enumerate(payloads):
        record = payload["records"][0]
        snapshot = snapshot_dir / f"snapshot-{group}.pt"
        snapshot.write_bytes(f"snapshot-{group}".encode())
        record["snapshot_path"] = str(snapshot)
        record["snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        records.append(record)
    root = tmp_path / "paired-dataset"
    assert paired.write_paired_dataset(
        root,
        data=data,
        records=records,
        meta=_meta(
            reference_policy_episode_assignments={
                f"episode-{group}": "random_0.5" for group in range(5)
            }
        ),
    ) == str(root.resolve())
    summary = paired.validate_paired_dataset(root)
    assert summary["num_states"] == 5
    assert summary["num_outcomes"] == 10
    assert summary["num_folds"] == 5
    assert summary["snapshots_verified"] == 5

    Path(records[0]["snapshot_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="snapshot SHA256"):
        paired.validate_paired_dataset(root)


def _write_suite_dataset(
    root: Path,
    episode_uids,
    *,
    suite: str,
    assignments,
    parent_sha: str,
    solver_fingerprint: str = "f" * 64,
):
    payloads = []
    snapshot_dir = root / "snapshots"
    snapshot_dir.mkdir(parents=True)
    for local_index, episode_uid in enumerate(episode_uids):
        payload = _payload(
            episode_uid,
            trajectory_id=local_index,
            rows=1,
            reference_policy_episode_assignments=assignments,
            episode_manifest_sha256=parent_sha,
            reference_assignment_manifest_sha256=parent_sha,
            solver_fingerprint=solver_fingerprint,
        )
        assignment = assignments[episode_uid]
        payload["data"]["source_mode"][0] = 1 if assignment == "idm" else 0
        helpful = int(episode_uid.rsplit("-", 1)[-1]) % 2 == 0
        payload["data"]["success_uncond"][0] = not helpful
        payload["data"]["progress_1_uncond"][0] = float(not helpful)
        payload["data"]["progress_3_uncond"][0] = float(not helpful)
        payload["records"][0]["task_suite_name"] = suite
        snapshot = snapshot_dir / f"{episode_uid}.pt"
        snapshot.write_bytes(f"snapshot:{episode_uid}".encode())
        payload["records"][0]["snapshot_path"] = str(snapshot)
        payload["records"][0]["snapshot_sha256"] = hashlib.sha256(
            snapshot.read_bytes()
        ).hexdigest()
        payloads.append(payload)
    data = {
        key: torch.cat([payload["data"][key] for payload in payloads])
        for key in payloads[0]["data"]
    }
    records = [payload["records"][0] for payload in payloads]
    meta = _meta(
        reference_policy_episode_assignments=assignments,
        episode_manifest_sha256=parent_sha,
        reference_assignment_manifest_sha256=parent_sha,
        solver_fingerprint=solver_fingerprint,
    )
    paired.write_paired_dataset(root, data=data, records=records, meta=meta)


def test_logical_multisuite_merge_has_global_folds_and_composite_provenance(tmp_path):
    suite_episodes = {
        "libero_10": [f"suite-a-{index}" for index in range(5)],
        "libero_goal": [f"suite-b-{index}" for index in range(7)],
    }
    all_uids = [uid for values in suite_episodes.values() for uid in values]
    assignments = {
        uid: ("uncond", "idm", "random_0.5")[index % 3]
        for index, uid in enumerate(all_uids)
    }
    commit = "a" * 40
    parent_path = tmp_path / "parent.json"
    parent_payload = {
        "schema": "libero-plus-episode-manifest-v1",
        "libero_plus_commit": commit,
        "split": "train",
        "episodes": [
            {
                "episode_id": uid,
                "task_suite_name": suite,
                "base_task": "pick_mug",
                "task_id": 0,
                "factor": "layout",
                "level": "L1",
                "trial_id": 0,
                "reset_state_id": 0,
                "env_seed": 11,
                "perturbation_id": "layout-001",
                "asset_ids": ["asset-a"],
            }
            for suite, uids in suite_episodes.items()
            for uid in uids
        ],
    }
    parent_path.write_text(json.dumps(parent_payload) + "\n", encoding="utf-8")
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    episodes = tuple(
        types.SimpleNamespace(
            episode_id=uid,
            base_task="pick_mug",
            task_suite_name=suite,
            task_id=0,
            factor="layout",
            level="L1",
            trial_id=0,
            reset_state_id=0,
            env_seed=11,
            perturbation_id="layout-001",
            asset_ids=("asset-a",),
        )
        for suite, uids in suite_episodes.items()
        for uid in uids
    )
    manifest = types.SimpleNamespace(
        path=str(parent_path),
        sha256=parent_sha,
        parent_manifest_path=None,
        episodes=episodes,
        libero_plus_commit=commit,
        split="train",
    )
    sources = {}
    for suite, uids in suite_episodes.items():
        source = tmp_path / f"source-{suite}"
        _write_suite_dataset(
            source,
            uids,
            suite=suite,
            assignments=assignments,
            parent_sha=parent_sha,
        )
        sources[suite] = source

    bad_second = tmp_path / "source-libero_goal-bad-solver"
    _write_suite_dataset(
        bad_second,
        suite_episodes["libero_goal"],
        suite="libero_goal",
        assignments=assignments,
        parent_sha=parent_sha,
        solver_fingerprint="2" * 64,
    )
    with pytest.raises(ValueError, match="provenance differs.*solver_fingerprint"):
        paired.merge_paired_suite_datasets(
            manifest,
            {"libero_10": sources["libero_10"], "libero_goal": bad_second},
            tmp_path / "bad-logical-paired",
        )

    output = tmp_path / "logical-paired"
    summary = paired.merge_paired_suite_datasets(manifest, sources, output)
    assert summary["num_trajectories"] == 12
    assert summary["logical_suite_count"] == 2
    assert summary["logical_parent_manifest_sha256"] == parent_sha
    assert len(summary["composite_source_fingerprint"]) == 64
    data, records, meta = paired.load_paired_shards(output)
    assert torch.unique(data["fold_id"]).tolist() == [0, 1, 2, 3, 4]
    assert torch.unique(data["trajectory_id"]).numel() == 12
    assert {record["task_suite_name"] for record in records} == set(suite_episodes)
    assert len(meta["logical_merge"]["components"]) == 2
    logical_dataset = benefit.GateBenefitDataset.from_shards(output)
    assert torch.unique(logical_dataset.label).tolist() == [0.0, 1.0]
    assert logical_dataset.fold_id is not None

    missing_output = tmp_path / "missing-output"
    with pytest.raises(ValueError, match="does not match logical manifest"):
        paired.merge_paired_suite_datasets(
            manifest, {"libero_10": sources["libero_10"]}, missing_output
        )
    (output / "provenance" / "parent_manifest.json").write_text("{}\n")
    with pytest.raises(ValueError, match="parent manifest SHA mismatch"):
        paired.validate_paired_dataset(output)


def test_logical_multisuite_merge_rejects_solver_contract_mismatch(tmp_path):
    # The lower-level shard loader must reject this before any logical artifact
    # is written, even when all feature dimensions happen to agree.
    first = _payload("episode-a", solver_fingerprint="1" * 64)
    second = _payload("episode-b", solver_fingerprint="2" * 64)
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    paired.save_paired_shard(
        first_path, data=first["data"], records=first["records"], meta=first["meta"]
    )
    paired.save_paired_shard(
        second_path,
        data=second["data"],
        records=second["records"],
        meta=second["meta"],
    )
    with pytest.raises(ValueError, match="provenance differs.*solver_fingerprint"):
        paired.load_paired_shards([first_path, second_path])


def test_benefit_metrics_and_feature_masks():
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    metrics = benefit.benefit_metrics(labels, scores, labels.float())
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    payload = _payload(rows=4)
    dataset = benefit.GateBenefitDataset(
        payload["data"],
        payload["records"],
        {**payload["meta"], "paired_dataset_fingerprint": "data-sha"},
        enabled_features=["proprio"],
    )
    world, proprio, text = dataset.features(torch.arange(4))
    assert not bool(world.any())
    assert not bool(text.any())
    assert torch.equal(proprio, payload["data"]["proprio"])


def _synthetic_dataset():
    payloads = [_payload(f"episode-{group}", trajectory_id=group, rows=4) for group in range(10)]
    data = {
        key: torch.cat([payload["data"][key] for payload in payloads])
        for key in payloads[0]["data"]
    }
    data["fold_id"] = data["trajectory_id"] % 5
    records = [record for payload in payloads for record in payload["records"]]
    meta = {
        **_meta(
            reference_policy_episode_assignments={
                f"episode-{group}": "random_0.5" for group in range(10)
            }
        ),
        "paired_dataset_fingerprint": "synthetic-sha",
        "splits_sha256": "split-sha",
    }
    return benefit.GateBenefitDataset(data, records, meta)


def test_cross_fit_is_grouped_deterministic_and_learns_helpful_states():
    torch.manual_seed(3)
    dataset = _synthetic_dataset()
    policy = GatePolicy(
        world_feat_dim=2,
        proprio_dim=1,
        text_feat_dim=1,
        hidden_sizes=(8,),
        add_value_head=False,
        activation="tanh",
    )
    cfg = benefit.GateBenefitConfig(
        folds=5,
        epochs=12,
        batch_size=16,
        lr=0.03,
        weight_decay=0.0,
        device="cpu",
        seed=9,
    )
    final_policy, result = benefit.cross_fit_gate_benefit(policy, dataset, cfg)
    assert result["metrics"]["auroc"] > 0.95
    assert result["metrics"]["auprc"] > 0.95
    assert torch.equal(torch.sort(result["fold_id"]).values, torch.arange(5).repeat_interleave(8))
    for fold in result["folds"]:
        assert not set(fold["train_groups"]) & set(fold["validation_groups"])
    assert final_policy.mode_logits(
        torch.tensor([[1.0, 0.0]]), torch.zeros(1, 1), torch.zeros(1, 1)
    )[0, 1] > final_policy.mode_logits(
        torch.tensor([[-1.0, 0.0]]), torch.zeros(1, 1), torch.zeros(1, 1)
    )[0, 1]


class _Mode(enum.Enum):
    UNCOND = "uncond"
    IDM = "idm"


def _install_fastwam_stub(monkeypatch):
    package = types.ModuleType("fastwam")
    adaptive = types.ModuleType("fastwam.adaptive_gate")
    adaptive.MODE_ORDER = (_Mode.UNCOND, _Mode.IDM)
    adaptive.WORLD_FEAT_LAYOUT = "world-layout-v1"
    adaptive.TEXT_FEAT_LAYOUT = "text-layout-v1"
    package.adaptive_gate = adaptive
    monkeypatch.setitem(sys.modules, "fastwam", package)
    monkeypatch.setitem(sys.modules, "fastwam.adaptive_gate", adaptive)


def test_uplift_sidecar_is_accepted_for_online_warmstart(tmp_path, monkeypatch):
    _install_fastwam_stub(monkeypatch)
    dataset = _synthetic_dataset()
    policy = GatePolicy(
        world_feat_dim=2,
        proprio_dim=1,
        text_feat_dim=1,
        hidden_sizes=(8,),
        add_value_head=False,
    )
    policy.bc_expected_provenance = {
        "task": _meta()["task"],
        "backbone_kind": "idm",
        "ckpt_fingerprint": "adaptive-v1:abc",
        "dataset_stats_fingerprint": "stats-sha",
        "num_video_frames": 9,
        "inference_steps": 20,
        "solver_fingerprint": "f" * 64,
        "context_len": 128,
        "model_dtype": "torch.bfloat16",
        "exec_horizon": 10,
        "action_horizon": 32,
    }
    policy.wam_checkpoint_sha256 = "c" * 64
    fake_result = {
        "metrics": {"auroc": 0.8},
        "config": {"folds": 5},
    }
    sidecar = benefit.build_gate_benefit_sidecar(policy, dataset, fake_result)
    assert sidecar["wam_provenance"]["solver_fingerprint"] == "f" * 64
    assert "logical_composite_source_fingerprint" in sidecar["paired_provenance"]
    path = str(tmp_path / "uplift.pt")
    benefit.save_gate_benefit_checkpoint(policy, path, sidecar=sidecar)
    loaded = mods.bc.load_gate_bc_state(path, expected_policy=policy)
    assert set(loaded) == set(policy.state_dict())
    with open(path + ".meta.json") as handle:
        broken = json.load(handle)
    broken["paired_provenance"]["target"] = "difficulty"
    with open(path + ".meta.json", "w") as handle:
        json.dump(broken, handle)
    with pytest.raises(ValueError, match="not deployable"):
        mods.bc.load_gate_bc_state(path, expected_policy=policy)


class _FakeCollectorDriver:
    def __init__(self):
        self.state = 0
        self.episode_uid = "episode-fake"
        self.actions = []
        self.paired_metadata = _meta()

    def reset_episode(self, episode):
        self.state = 0
        self.episode_uid = str(episode.get("episode_id", "episode-fake"))
        return {"state": self.state}

    def capture_snapshot(self):
        return {"schema": "libero-gate-snapshot-v1", "state": self.state}

    def restore_snapshot(self, snapshot):
        self.state = snapshot["state"]
        return {"state": self.state}

    def context(self, observation):
        return {
            "episode_uid": self.episode_uid,
            "decision_index": self.state,
            "task_id": 0,
            "base_task": "pick_mug",
            "task_suite_name": "libero_10",
            "task_description": "pick up the mug",
            "trial_id": 0,
            "reset_state_id": 0,
            "env_seed": 11,
            "factor": "layout",
            "level": "L1",
            "perturbation_id": "layout-001",
            "phase": "unknown",
            "phase_reliable": False,
            "asset_ids": ["asset-a"],
        }

    def features(self, observation):
        return {
            "world_feat": torch.tensor([1.0, 2.0]),
            "proprio": torch.tensor([0.0]),
            "text_feat": torch.tensor([0.0]),
        }

    def action(self, observation, *, mode, seed):
        self.actions.append((mode, seed))
        return mode

    def step_chunk(self, action):
        self.state += 1
        return {
            "observation": {"state": self.state},
            "done": True,
            "success": bool(action),
            "progress": float(bool(action)),
        }


def test_collector_uses_common_branch_seed_and_emits_valid_paired_payload():
    driver = _FakeCollectorDriver()
    collector = collector_mod.PairedStateCollector(
        driver,
        collector_seed=5,
        max_reference_decisions=1,
        max_branch_decisions=1,
        sensitivity_fraction=0.0,
    )
    result = collector.collect([{"episode_id": "episode-fake"}])
    assert driver.actions[0][0] == 0
    assert driver.actions[1][0] == 1
    assert driver.actions[0][1] == driver.actions[1][1]
    payload = paired.validate_paired_payload(
        {"schema": paired.PAIRED_SCHEMA, **result}
    )
    assert bool(payload["data"]["success_idm"][0])
    assert not bool(payload["data"]["success_uncond"][0])


def test_collector_reference_policy_is_balanced_shuffled_and_reproducible():
    first = collector_mod.PairedStateCollector(
        _FakeCollectorDriver(),
        collector_seed=23,
        max_reference_decisions=1,
        max_branch_decisions=1,
    )
    second = collector_mod.PairedStateCollector(
        _FakeCollectorDriver(),
        collector_seed=23,
        max_reference_decisions=1,
        max_branch_decisions=1,
    )
    assignments = first._reference_assignments(12)
    assert assignments == second._reference_assignments(12)
    assert assignments.count(collector_mod.UNCOND) == 4
    assert assignments.count(collector_mod.IDM) == 4
    assert assignments.count(None) == 4
    assert assignments != [0, 1, None] * 4


def test_partitioned_collector_uses_parent_assignment_without_local_thirds():
    driver = _FakeCollectorDriver()
    collector = collector_mod.PairedStateCollector(
        driver,
        collector_seed=31,
        max_reference_decisions=1,
        max_branch_decisions=1,
        sensitivity_fraction=0.0,
    )
    parent_episodes = [
        {"episode_id": f"episode-{index}"} for index in range(12)
    ]
    assignments = collector.reference_assignment_map(parent_episodes)
    result = collector.collect(
        parent_episodes[:5],
        reference_assignments=assignments,
        reference_assignment_manifest_sha256="d" * 64,
    )
    assert len(result["records"]) == 5
    assert result["meta"]["reference_policy_episode_assignments"] == assignments
    assert result["meta"]["reference_assignment_sha256"] == (
        paired.reference_assignment_sha256(assignments)
    )
    paired.validate_paired_payload({"schema": paired.PAIRED_SCHEMA, **result})
