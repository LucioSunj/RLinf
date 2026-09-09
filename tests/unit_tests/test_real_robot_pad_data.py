# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native LeRobot windows, train-only calibration and offline initialization."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from fastwam.real_robot_adaptation import _tiny_text_cache, run_tiny_adaptation
from fastwam.real_robot_data import _group_splits, write_real_robot_dataset
from fastwam.real_robot_tiny import TinyFastWAM
from hydra.utils import instantiate
from omegaconf import OmegaConf

from rlinf.envs.pad_realworld.mock_backend import MockRobotBackend
from rlinf.models.embodiment.wam_policy.real_robot.builder import (
    build_real_robot_policy,
)
from rlinf.models.embodiment.wam_policy.real_robot.config import load_config
from rlinf.models.embodiment.wam_policy.real_robot.data import (
    collect_demos,
    load_sessions,
    prepare_dataset,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("pad_data")
    cfg = load_config(ROOT / "configs/real_robot/pad_mock.yaml")
    cfg["observation"]["cameras"].append(
        {"name": "wrist", "source_color": "BGR", "crop": None, "resize": [32, 32]}
    )
    collect_demos(cfg, root / "demos")
    report = prepare_dataset(root / "demos", root / "dataset")
    torch.manual_seed(42)
    _tiny_text_cache(root / "dataset", TinyFastWAM())
    return cfg, root, report


def test_dataset_roundtrip_matches_runtime_stats_images_and_command_anchors(converted):
    cfg, root, report = converted
    sessions = load_sessions(root / "demos")
    data_cfg = OmegaConf.load(root / "dataset/data_idm.yaml")
    train = instantiate(data_cfg.train)
    sample = train[0]
    assert sample["video"].shape == (3, 9, 32, 64)
    assert sample["action"].shape == (32, 7)
    assert sample["proprio"].shape == (32, 8)
    assert not sample["action_is_pad"].any()
    assert len(train) == sum(
        s["full_windows"] for s in report["segments"] if s["split"] == "train"
    )
    assert report["video_offsets"] == list(range(0, 33, 4))
    segment = next(s for s in report["segments"] if s["split"] == "train")
    session = next(s for s in sessions if s["episode_id"] == segment["episode_id"])
    stats = json.loads((root / "dataset/dataset_stats.json").read_text())
    actions = np.concatenate(
        [s["actions"] for s in sessions if report["splits"][s["episode_id"]] == "train"]
    )
    np.testing.assert_allclose(stats["action"]["default"]["global_min"], actions.min(0))
    np.testing.assert_allclose(stats["action"]["default"]["global_max"], actions.max(0))
    processor = train.lerobot_dataset.processor
    normalizer = processor.normalizer.normalizers["action"]["default"]
    np.testing.assert_allclose(
        normalizer.backward(sample["action"]).numpy(),
        session["actions"][:32],
        atol=1e-6,
    )
    # Recorded commands lead the lagging measured plant; they are not state deltas.
    assert not np.allclose(
        session["actions"][:32, :3], np.diff(session["states"][:33, :3], axis=0)
    )
    expected = np.concatenate(
        [session["images"]["main"][0], session["images"]["wrist"][0]], axis=1
    )
    np.testing.assert_allclose(
        sample["video"][:, 0].permute(1, 2, 0).numpy(), expected / 127.5 - 1, atol=1e-6
    )
    # Shared inference adapter uses identical training stats with no LIBERO gripper flip.
    policy = build_real_robot_policy(cfg, initialize=False)
    policy.runtime.processor = processor
    state = torch.from_numpy(session["states"][:1])
    torch.testing.assert_close(
        policy.runtime._normalized_proprio(state)[0], sample["proprio"][0]
    )
    recovered, _ = policy.runtime._denormalize_action_stages(
        sample["action"].unsqueeze(0), env_obs={}
    )
    np.testing.assert_allclose(recovered[0].numpy(), session["actions"][:32], atol=1e-6)
    bc = instantiate(OmegaConf.load(root / "dataset/data_uncond_bc.yaml").train)
    assert bc[0]["video"].shape == (3, 1, 32, 64)
    torch.testing.assert_close(bc[0]["action"], sample["action"])


def test_session_layout_splits_are_connected_not_frame_randomized(converted):
    _, root, _ = converted
    sessions = load_sessions(root / "demos")
    sessions[1]["session_id"] = sessions[0]["session_id"]
    sessions[2]["layout_id"] = sessions[1]["layout_id"]
    splits = _group_splits(sessions)
    assert len({splits[s["episode_id"]] for s in sessions[:3]}) == 1
    assert set(splits.values()) == {"train", "validation", "test"}


def test_elapsed_gripper_pause_splits_windows_without_time_compression(
    converted, tmp_path
):
    _, root, _ = converted
    sessions = load_sessions(root / "demos")
    sessions[0]["timestamps"][13:] += 0.6
    report = write_real_robot_dataset(sessions, tmp_path / "gap_dataset")
    assert report["dropped_gap_actions"] == 1
    segment = next(
        s for s in report["segments"] if s["episode_id"] == sessions[0]["episode_id"]
    )
    assert segment["start"] == 13 and segment["full_windows"] == 1
    assert segment["elapsed_timestamps"][0] == pytest.approx(
        sessions[0]["timestamps"][13]
    )


def test_missing_anchor_and_camera_skew_are_not_silently_guessed(converted, tmp_path):
    _, root, _ = converted
    imported = tmp_path / "sessions"
    shutil.copytree(root / "demos", imported)
    path = next(imported.glob("*/session.json"))
    session = json.loads(path.read_text())
    saved = deepcopy(session)
    del session["commands"][0]["anchor_pose"]
    path.write_text(json.dumps(session))
    with pytest.raises(ValueError, match="target/anchor/timing"):
        load_sessions(imported)
    saved["capture_times"][0]["wrist"] -= 0.1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="synchronization skew"):
        load_sessions(imported)


def test_converted_data_offline_parent_bc_and_online_builder(converted, tmp_path):
    cfg, root, _ = converted
    report = run_tiny_adaptation(root / "dataset", tmp_path / "tiny", steps=2)
    assert len(report["parent_losses"]) == len(report["bc_losses"]) == 2
    assert all(np.isfinite(report["parent_losses"] + report["bc_losses"]))
    cfg = deepcopy(cfg)
    cfg["model"].update(
        tiny_initialization_dir=str(tmp_path / "tiny"),
        stats_path=report["stats"],
        model_config=report["processor_config"],
        text_cache=report["text_cache"],
    )
    policy = build_real_robot_policy(cfg)
    parent = torch.load(report["parent"], weights_only=True)["state_dict"]
    for name, value in parent.items():
        assert torch.equal(policy.actor.state_dict()[name], value)
    sidecar = torch.load(report["sidecar"], weights_only=True)["state_dict"]
    for name, value in policy.lora_adapter.lora_state_dict().items():
        assert torch.equal(value, sidecar[name])
    assert policy.actor_version == 0
    backend = MockRobotBackend(camera_names=("main", "wrist"))
    backend.begin_episode("new", "new")
    snapshot = backend.observe(
        run_id="dataset-test",
        episode_id="new",
        chunk_id=0,
        task_id="fixture",
        instruction="Move the gripper to the target.",
        layout_id="new",
        calibration_id="mock_only",
    )
    policy.begin_episode()
    assert policy.propose(snapshot).normalized_actions.shape == (32, 7)


def test_native_idm_configuration_keeps_validation_split_and_missing_assets(converted):
    _, root, _ = converted
    cfg = OmegaConf.load(root / "dataset/idm_train.yaml")
    assert cfg.data.val.dataset_dirs[0].endswith("/validation")
    assert cfg.model._target_ == "fastwam.runtime.create_fastwam_idm"
    missing = OmegaConf.missing_keys(cfg)
    assert {"resume", "learning_rate", "max_steps", "model.model_id"} <= missing
