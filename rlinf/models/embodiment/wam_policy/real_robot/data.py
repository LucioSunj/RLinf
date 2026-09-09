# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Recorded demonstration sessions with explicit command anchors and timestamps."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rlinf.envs.pad_realworld.action_codec import (
    ActionCodec,
    canonical_delta_from_target,
)
from rlinf.envs.pad_realworld.mock_backend import MockRobotBackend
from rlinf.envs.pad_realworld.observation import ObservationAdapter


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def collect_demos(cfg: dict, output: Path, *, import_dir: Path | None = None) -> dict:
    """Record a mock demonstration or import a complete existing session directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new demonstration output directory: {output}")
    if import_dir is not None:
        # Validate command interpretation before copying any imported data.
        load_sessions(import_dir)
        shutil.copytree(import_dir, output)
        return {"status": "IMPORTED", "output": str(output)}
    if cfg["backend"] != "mock":
        raise PermissionError(
            "Live teleoperation recording requires an explicitly bound site recorder."
        )
    output.mkdir(parents=True)
    adapter = ObservationAdapter.from_config(cfg["observation"])
    codec = ActionCodec(**cfg["limits"])
    backend = MockRobotBackend(
        camera_names=[c.name for c in adapter.cameras],
        gripper_wait_seconds=cfg["mock"]["gripper_wait_seconds"],
    )
    period = 1.0 / cfg["execution"]["command_hz"]
    count = cfg["mock"]["demo_episodes"]
    for episode_index in range(count):
        episode_id = f"demo_{episode_index:04d}"
        path = output / episode_id
        path.mkdir()
        identity = {
            "run_id": "mock_demos",
            "episode_id": episode_id,
            "task_id": "mock_reach",
            "instruction": "Move the gripper to the target.",
            "layout_id": f"layout_{episode_index}",
            "calibration_id": "mock_only",
        }
        backend.begin_episode(episode_id, identity["layout_id"])
        snapshots, commands = [], []
        for step in range(cfg["mock"]["demo_episode_steps"]):
            snapshot = backend.observe(**identity, chunk_id=step)
            snapshots.append(snapshot)
            # A nonconstant synthetic expert commands poses; measured motion lags it.
            phase = (step + episode_index * 3) / 9
            action = np.array(
                [
                    0.002 * np.cos(phase),
                    0.003 * np.sin(phase),
                    0.001 * np.cos(phase * 0.7),
                    0.002,
                    -0.001,
                    0.003,
                    1.0,
                ]
            )
            anchor = backend.read_state()
            command = codec.command(action, adapter.canonical_pose(anchor))
            sent = backend.clock.now()
            assert backend.send(command), (
                "Mock demonstration command must be acknowledged."
            )
            ack = backend.clock.now()
            commands.append(
                {
                    "anchor_pose": command.anchor_pose,
                    "target_pose": command.target_pose,
                    "gripper_open_fraction": command.limited_action[-1],
                    "requested_action": command.requested_action,
                    "actual_action": command.limited_action,
                    "send_time": sent,
                    "acknowledge_time": ack,
                    "state_after": backend.read_state(),
                }
            )
            backend.clock.sleep(max(0, period - (ack - sent)))
        snapshots.append(backend.observe(**identity, chunk_id=len(commands)))
        arrays = {
            camera.name: np.stack([s.images[camera.name] for s in snapshots])
            for camera in adapter.cameras
        }
        np.savez(path / "images.npz", **arrays)
        payload = {
            "schema": "pad-demonstration-session-v1",
            "source": "mock",
            "session_id": f"mock_session_{episode_index}",
            **identity,
            "observation": cfg["observation"],
            "action_spec": cfg["action"],
            "sample_period": period,
            "states": [s.state for s in snapshots],
            "capture_times": [s.capture_times for s in snapshots],
            "receive_times": [s.receive_times for s in snapshots],
            "commands": commands,
            "outcome": asdict(backend.poll_outcome(episode_id)),
        }
        (path / "session.json").write_text(json.dumps(_json(payload), indent=2))
    return {"status": "MOCK-RECORDED", "episodes": count, "output": str(output)}


def load_sessions(source: Path) -> list[dict]:
    """Interpret recorded target commands using their saved anchors and view order."""
    sessions = []
    for path in sorted(Path(source).glob("*/session.json")):
        payload = json.loads(path.read_text())
        if payload["schema"] != "pad-demonstration-session-v1":
            raise ValueError(f"Unsupported demonstration session: {path}")
        required_action = {
            "translation": "meters_in_base_frame",
            "rotation": "Euler_xyz_increment_left",
            "gripper": "open_fraction",
            "anchor": "measured_before_each_send",
        }
        if payload["action_spec"] != required_action:
            raise ValueError(f"Uninterpretable recorded action spec in {path}")
        adapter = ObservationAdapter.from_config(payload["observation"])
        n = len(payload["commands"])
        if n < 1 or len(payload["states"]) != n + 1:
            raise ValueError(
                "Sessions need one pre-action snapshot per command and one final snapshot."
            )
        from rlinf.envs.pad_realworld.contracts import ObservationSnapshot

        raw_images = np.load(path.with_name("images.npz"), allow_pickle=False)
        states, images, timestamps = [], [], []
        for index in range(n + 1):
            received = payload["receive_times"][index]
            snap = ObservationSnapshot(
                run_id=payload["run_id"],
                episode_id=payload["episode_id"],
                chunk_id=index,
                task_id=payload["task_id"],
                instruction=payload["instruction"],
                layout_id=payload["layout_id"],
                calibration_id=payload["calibration_id"],
                images={c.name: raw_images[c.name][index] for c in adapter.cameras},
                state=payload["states"][index],
                capture_times=payload["capture_times"][index],
                receive_times=received,
                receive_time=max(received.values()),
                clock_domain=payload["observation"]["freshness"]["clock_domain"],
            )
            adapter.validate_freshness(snap, snap.receive_time)
            states.append(adapter.canonical_state(snap.state))
            images.append(adapter.images(snap))
            timestamps.append(snap.receive_time)
        raw_images.close()
        actions = []
        for index, command in enumerate(payload["commands"]):
            if (
                not {
                    "anchor_pose",
                    "target_pose",
                    "gripper_open_fraction",
                    "send_time",
                    "acknowledge_time",
                    "state_after",
                }
                <= command.keys()
            ):
                raise ValueError(
                    "Recorded command lacks target/anchor/timing; measured-state differences are not expert actions."
                )
            actions.append(
                canonical_delta_from_target(
                    np.asarray(command["target_pose"]),
                    np.asarray(command["anchor_pose"]),
                    command["gripper_open_fraction"],
                )
            )
            if (
                not timestamps[index]
                <= command["send_time"]
                <= command["acknowledge_time"]
                <= timestamps[index + 1]
            ):
                raise ValueError(
                    "A command is not aligned to its current and post-action snapshots."
                )
        sessions.append(
            {
                **payload,
                "states": np.asarray(states),
                "actions": np.asarray(actions, dtype=np.float32),
                "images": {
                    c.name: np.stack([item[i] for item in images])
                    for i, c in enumerate(adapter.cameras)
                },
                "timestamps": np.asarray(timestamps),
                "source_path": str(path),
            }
        )
    if not sessions:
        raise ValueError(f"No recorded sessions found in {source}")
    if len({s["episode_id"] for s in sessions}) != len(sessions):
        raise ValueError("Imported episode IDs must be unique.")
    return sessions


def prepare_dataset(source: Path, output: Path) -> dict:
    """Write the project's pinned LeRobot reader format, plus train-only stats."""
    from fastwam.real_robot_data import write_real_robot_dataset

    return write_real_robot_dataset(load_sessions(source), output)
