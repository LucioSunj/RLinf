# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from _gate_test_imports import load_gate_modules


mods = load_gate_modules()
control = mods.control
GatePolicy = mods.gate.GatePolicy
trace = mods.trace


def _profile_payload(*, no_read_matched=True, extra_matched=True):
    entries = {}
    for index, name in enumerate(sorted(control.CONTROL_KINDS), start=1):
        entries[name] = {
            "flops": float(index * 100),
            "latency_ms": float(index * 10),
            "action_steps": 20 if name != "extra_compute" else 47,
        }
    entries["no_read"]["compute_matched"] = no_read_matched
    entries["extra_compute"]["compute_matched"] = extra_matched
    return {
        "schema_version": 1,
        "kind": "fastwam_control_profile",
        "meta": {
            "task": "task-a",
            "ckpt_fingerprint": "wam-ckpt",
            "dataset_stats_fingerprint": "stats-sha",
            "inference_steps": 20,
            "solver_fingerprint": "solver-sha",
            "height": 16,
            "width": 24,
            "num_video_frames": 9,
            "action_horizon": 32,
            "context_len": 4,
            "model_dtype": "torch.float32",
            "device_name": "cpu",
        },
        "controls": entries,
    }


def _write_profile(tmp_path: Path, **kwargs) -> Path:
    path = tmp_path / "controls.yaml"
    path.write_text(yaml.safe_dump(_profile_payload(**kwargs)), encoding="utf-8")
    return path


class _FactoryAdapter:
    task = "task-a"
    dataset_stats_fingerprint = "stats-sha"
    inference_steps = 20
    solver_fingerprint = "solver-sha"
    num_video_frames = 9
    generation_horizon = 32
    context_len = 4
    _cost_meta = {"height": 16, "width": 24}
    model = SimpleNamespace(
        _loaded_checkpoint_fingerprint="wam-ckpt",
        torch_dtype=torch.float32,
        device="cpu",
    )

    @staticmethod
    def _device_name():
        return "cpu"


@dataclass
class _Encoded:
    world_feat: torch.Tensor


class _ControlAdapter:
    world_feat_dim = 8

    def __init__(self):
        self.calls = []

    def encode_world_state(self, image):
        return _Encoded(image.float().reshape(-1)[:8])

    def act_control(self, **kwargs):
        self.calls.append(kwargs)
        result = {
            "action_chunk": torch.zeros(3, 7),
            "cost": -1.0,
            "aux": {"adapter": True},
        }
        if kwargs.get("return_video_latents"):
            result["video_latents"] = torch.arange(48).reshape(1, 2, 3, 2, 4)
        return result

    def act(self, **kwargs):
        self.calls.append({"production_act": True, **kwargs})
        return {
            "action_chunk": torch.zeros(3, 7),
            "cost": -1.0,
            "aux": {},
        }


def _loaded_profile(tmp_path: Path):
    path = _write_profile(tmp_path)
    return control.load_control_profile(
        path,
        expected_metadata={"task": "task-a", "ckpt_fingerprint": "wam-ckpt"},
    )


def _context(batch=1):
    return {
        "episode_uid": [f"episode-{index}" for index in range(batch)],
        "decision_index": torch.arange(batch),
        "base_task": ["pick mug"] * batch,
        "task_id": list(range(batch)),
        "trial_id": list(range(batch)),
        "reset_state_id": torch.arange(batch) + 10,
        "env_seed": torch.arange(batch) + 100,
        "factor": ["camera"] * batch,
        "level": ["L3"] * batch,
        "phase": ["contact_alignment"] * batch,
        "phase_reliable": torch.ones(batch, dtype=torch.bool),
        "perturbation_id": ["camera-yaw"] * batch,
        "episode_manifest_sha256": ["manifest-sha"] * batch,
    }


def test_control_profile_and_builder_fail_closed_on_provenance_and_matching(tmp_path):
    path = _write_profile(tmp_path, no_read_matched=False)
    with pytest.raises(ValueError, match="provenance mismatch"):
        control.load_control_profile(
            path, expected_metadata={"ckpt_fingerprint": "wrong"}
        )
    with pytest.raises(ValueError, match="not compute-matched"):
        control.build_eval_control_runtime(
            {
                "kind": "no_read",
                "profile_path": str(path),
                "cost_metric": "latency_ms",
            },
            adapter=_FactoryAdapter(),
            fastwam_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("kind", "branch_mode", "expected_steps"),
    [
        ("valid_idm", 1, None),
        ("no_read", 1, None),
        ("repeat_current", 1, None),
        ("extra_compute", 0, 47),
    ],
)
def test_runtime_dispatches_controls_and_uses_separate_profile_cost(
    tmp_path, kind, branch_mode, expected_steps
):
    profile = _loaded_profile(tmp_path)
    entry = profile.entry(kind)
    runtime = control.EvalControlRuntime(
        kind=kind,
        profile=profile,
        cost_metric="latency_ms",
        cost=float(entry["latency_ms"]),
        action_steps=int(entry["action_steps"]),
        wam_seed=7,
        donor_seed=11,
    )
    adapter = _ControlAdapter()
    result = runtime.act(
        adapter,
        input_image=torch.zeros(1, 3, 16, 24),
        proprio=torch.zeros(1, 5),
        context=torch.zeros(1, 4, 6),
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        encoded_state=_Encoded(torch.zeros(8)),
        gate_context=_context(),
        batch_index=0,
    )
    assert runtime.branch_mode == branch_mode
    assert result["cost"] == pytest.approx(float(entry["latency_ms"]))
    assert adapter.calls[0]["control"] == kind
    assert adapter.calls[0]["seed"] == 7
    assert adapter.calls[0]["extra_action_steps"] == expected_steps
    assert adapter.calls[0]["return_video_latents"] is False


def test_valid_idm_capture_writes_bank_ready_artifact_and_hash(tmp_path):
    profile = _loaded_profile(tmp_path)
    capture_dir = tmp_path / "donors"
    runtime = control.EvalControlRuntime(
        kind="valid_idm",
        profile=profile,
        cost_metric="flops",
        cost=123.0,
        action_steps=20,
        wam_seed=13,
        donor_seed=0,
        capture_donor_dir=str(capture_dir),
    )
    adapter = _ControlAdapter()
    result = runtime.act(
        adapter,
        input_image=torch.zeros(1, 3, 16, 24),
        proprio=torch.zeros(1, 5),
        context=torch.zeros(1, 4, 6),
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        encoded_state=_Encoded(torch.zeros(8)),
        gate_context=_context(),
        batch_index=0,
    )
    artifact = result["aux"]["donor_artifact"]
    assert artifact["sha256"] == control._sha256_file(artifact["path"])
    payload = torch.load(artifact["path"], map_location="cpu", weights_only=False)
    assert payload["state_id"] == "episode-0:000"
    assert payload["video_latents"].shape == (1, 2, 3, 2, 4)
    assert payload["metadata"]["task"] == "pick mug"
    assert payload["metadata"]["factor"] == "camera"
    assert payload["metadata"]["wam_seed"] == 13
    assert payload["metadata"]["ckpt_fingerprint"] == "wam-ckpt"
    with pytest.raises(FileExistsError, match="already exists"):
        runtime.act(
            adapter,
            input_image=torch.zeros(1, 3, 16, 24),
            proprio=torch.zeros(1, 5),
            context=torch.zeros(1, 4, 6),
            context_mask=torch.ones(1, 4, dtype=torch.bool),
            encoded_state=_Encoded(torch.zeros(8)),
            gate_context=_context(),
            batch_index=0,
        )


def test_donor_capture_skips_absorbed_reference_slots(tmp_path):
    profile = _loaded_profile(tmp_path)
    runtime = control.EvalControlRuntime(
        kind="valid_idm",
        profile=profile,
        cost_metric="flops",
        cost=123.0,
        action_steps=20,
        wam_seed=13,
        donor_seed=0,
        capture_donor_dir=str(tmp_path / "donors"),
    )
    adapter = _ControlAdapter()
    context = {**_context(), "_active_mask": torch.tensor([False])}
    result = runtime.act(
        adapter,
        input_image=torch.zeros(1, 3, 16, 24),
        proprio=torch.zeros(1, 5),
        context=torch.zeros(1, 4, 6),
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        encoded_state=_Encoded(torch.zeros(8)),
        gate_context=context,
        batch_index=0,
    )
    assert adapter.calls[0]["production_act"] is True
    assert result["aux"]["control_skipped_after_absorption"] is True
    assert result["aux"]["donor_artifact"] is None
    assert list((tmp_path / "donors").glob("donor_*.pt")) == []


class _Bank:
    def __init__(self, metadata):
        self.metadata = metadata
        self.calls = []

    def select(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(metadata={**self.metadata, **kwargs["recipient_metadata"]})


def test_shuffled_runtime_matches_cell_and_binds_donor_seed(tmp_path):
    profile = _loaded_profile(tmp_path)
    bank = _Bank(
        {
            "ckpt_fingerprint": "wam-ckpt",
            "dataset_stats_fingerprint": "stats-sha",
            "solver_steps": 20,
            "num_video_frames": 9,
            "action_horizon": 32,
            "wam_seed": 13,
        }
    )
    runtime = control.EvalControlRuntime(
        kind="shuffled",
        profile=profile,
        cost_metric="flops",
        cost=500.0,
        action_steps=20,
        wam_seed=7,
        donor_seed=19,
        donor_bank=bank,
        expected_donor_wam_seed=13,
    )
    adapter = _ControlAdapter()
    runtime.act(
        adapter,
        input_image=torch.zeros(1, 3, 16, 24),
        proprio=torch.zeros(1, 5),
        context=torch.zeros(1, 4, 6),
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        encoded_state=_Encoded(torch.zeros(8)),
        gate_context=_context(),
        batch_index=0,
    )
    assert bank.calls == [
        {
            "recipient_state_id": "episode-0:000",
            "recipient_metadata": {
                "task": "pick mug",
                "factor": "camera",
                "level": "L3",
                "phase": "contact_alignment",
            },
            "seed": 19,
        }
    ]
    expected = adapter.calls[0]["expected_donor_metadata"]
    assert expected["wam_seed"] == 13
    assert expected["phase"] == "contact_alignment"


def test_shuffled_builder_rejects_donor_bank_seed_mismatch(tmp_path):
    profile_path = _write_profile(tmp_path)
    bank_path = tmp_path / "bank.pt"
    torch.save({"placeholder": True}, bank_path)
    bad_bank = _Bank(
        {
            "ckpt_fingerprint": "wam-ckpt",
            "dataset_stats_fingerprint": "stats-sha",
            "solver_steps": 20,
            "num_video_frames": 9,
            "action_horizon": 32,
            "wam_seed": 99,
        }
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        control.build_eval_control_runtime(
            {
                "kind": "shuffled",
                "profile_path": str(profile_path),
                "donor_bank_path": str(bank_path),
                "cost_metric": "flops",
                "wam_seed": 7,
                "expected_donor_wam_seed": 13,
            },
            adapter=_FactoryAdapter(),
            fastwam_root=tmp_path,
            donor_bank_loader=lambda payload: bad_bank,
        )


def _policy_obs(batch=2):
    result = {
        "input_image": torch.rand(batch, 3, 16, 24),
        "proprio": torch.randn(batch, 5),
        "context": torch.randn(batch, 4, 6),
        "context_mask": torch.ones(batch, 4, dtype=torch.bool),
        "gate_context": _context(batch),
    }
    return result


def test_gate_policy_forces_control_branch_but_rejects_training(tmp_path):
    profile = _loaded_profile(tmp_path)
    runtime = control.EvalControlRuntime(
        kind="no_read",
        profile=profile,
        cost_metric="latency_ms",
        cost=40.0,
        action_steps=20,
        wam_seed=0,
        donor_seed=0,
    )
    adapter = _ControlAdapter()
    policy = GatePolicy(
        world_feat_dim=8,
        proprio_dim=5,
        text_feat_dim=4,
        hidden_sizes=(8,),
        add_value_head=False,
        eval_policy={"kind": "learned", "max_decisions": 70},
        eval_control={"kind": "no_read"},
        eval_control_runtime=runtime,
        wam_adapter=adapter,
    )
    _, result = policy.predict_action_batch(_policy_obs(), mode="eval")
    assert result["eval_policy_method"] == "control:no_read"
    assert result["mode"].tolist() == [1, 1]
    assert result["reserved_modes"].shape == (2, 70)
    assert result["reserved_modes"].sum().item() == 140
    assert result["mode_cost"].tolist() == pytest.approx([40.0, 40.0])
    assert [call["control"] for call in adapter.calls] == ["no_read", "no_read"]
    with pytest.raises(RuntimeError, match="evaluation-only"):
        policy.predict_action_batch(_policy_obs(batch=1), mode="train")

    no_wam = GatePolicy(
        world_feat_dim=8,
        proprio_dim=5,
        text_feat_dim=4,
        eval_control={"kind": "extra_compute"},
    )
    with pytest.raises(RuntimeError, match="evaluation-only"):
        no_wam.predict_action_batch(_policy_obs(batch=1), mode="train")


def test_eval_control_rejects_nonlearned_selector_and_force_mode():
    with pytest.raises(ValueError, match="non-learned"):
        GatePolicy(
            8,
            5,
            eval_policy={"kind": "forced", "mode": 1, "max_decisions": 70},
            eval_control={"kind": "valid_idm"},
        )
    with pytest.raises(ValueError, match="force_mode"):
        GatePolicy(8, 5, force_mode=1, eval_control={"kind": "valid_idm"})


def test_canonical_trace_records_donor_artifact_sidecar():
    builder = trace.RolloutGateTraceBuilder(
        method="control:valid_idm",
        max_decisions=70,
        selector_provenance={"control_profile_sha256": "p" * 64},
        gate_checkpoint_sha256="g" * 64,
        wam_checkpoint_sha256="w" * 64,
    )
    context = _context()
    artifact = {"path": "/tmp/donor.pt", "sha256": "d" * 64}
    builder.add_batch(
        context=context,
        modes=torch.ones(1, dtype=torch.long),
        costs=torch.tensor([100.0]),
        active_mask=torch.tensor([True]),
        reserved_modes=torch.ones(1, 70, dtype=torch.long),
        control_artifacts=[artifact],
    )
    decision = builder.records()[0]["decisions"][0]
    assert decision["donor_artifact"] == artifact
