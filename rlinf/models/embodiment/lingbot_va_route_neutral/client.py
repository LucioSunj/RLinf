# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Human resets/labels, SpaceMouse collection and synchronous WebSocket use."""

from __future__ import annotations

import json
import select
import sys
from pathlib import Path
from time import monotonic, sleep, time

import numpy as np
from scipy.spatial.transform import Rotation

from .data import TASKS, write_episode
from .robot import ChunkExecutor


class OperatorLabels:
    """Terminal input: s=success, f=failure, t=takeover, p <fraction>=progress."""

    def __init__(self):
        self.value = {}

    def reset(self):
        self.value = {}

    def poll(self):
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip().lower()
            if not line:
                break
            if line in {"s", "f", "t"}:
                for flag in ("success", "failure", "takeover"):
                    self.value.pop(flag, None)
                self.value[{"s": "success", "f": "failure", "t": "takeover"}[line]] = (
                    True
                )
            elif line.startswith("p "):
                self.value["stage_progress"] = float(line.split()[1])
        return self.value.copy()


def run_robot_client(remote, driver, *, task, output, schedule=None, endpoint="idm"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels, executor = OperatorLabels(), ChunkExecutor(driver)
    trials = [r for r in schedule["trials"] if r["task"] == task] if schedule else []
    while True:
        status = remote.infer({"operation": "status"})
        if status["task"] != task:
            raise ValueError("The robot task and policy server task differ.")
        if status["complete"]:
            return
        if schedule and not trials:
            return
        trial = trials.pop(0) if schedule else None
        if trial and (output / f"{trial['scene_id']}_{trial['method']}.json").exists():
            continue
        print(
            f"Task {task}; update {status['step'] + 1}; trial {trial['trial_id'] if trial else 'training/deployment'}",
            flush=True,
        )
        if trial:
            print(
                f"Reset reference: {trial['reset_photo']}\n{trial['reset_description']}",
                flush=True,
            )
        reset_note = input(
            "Reset the scene, record the reset photo path/note, then press Enter (q to stop): "
        )
        if reset_note.strip().lower() == "q":
            return
        labels.reset()
        initial = driver.reset()
        started = time()
        request = {
            "operation": "reset",
            "observation": initial,
            "episode_started": started,
            "reset_metadata": {"note": reset_note, "scene": trial},
        }
        if trial:
            request["trial_id"] = trial["trial_id"]
        remote.infer(request)
        observations, actions, action_times, commands = [initial], [], [], []
        step = 0
        print(
            "During execution: s + Enter success; f failure; t takeover; p 0.5 progress.",
            flush=True,
        )
        try:
            while True:
                observation = driver.observe()
                response = remote.infer(
                    {
                        "operation": "decide",
                        "observation": observation,
                        "route": endpoint,
                    }
                )
                feedback = executor.execute(
                    response["actions"],
                    remaining_steps=TASKS[task]["max_steps"] - step,
                    stop_status=labels.poll,
                )
                step += len(feedback.executed_actions)
                observations.extend(feedback.observed_frames)
                actions.extend(feedback.executed_actions)
                action_times.extend(feedback.action_timestamps)
                commands.extend(feedback.submitted_commands)
                result = remote.infer(
                    {
                        "operation": "commit",
                        "feedback": feedback.wire(),
                        "episode_finished": time(),
                    }
                )
                if feedback.terminated:
                    episode_result = result["episode_result"]
                    identity = (
                        f"{trial['scene_id']}_{trial['method']}"
                        if trial
                        else f"episode_{int(started * 1000)}"
                    )
                    (output / f"{identity}.json").write_text(
                        json.dumps(episode_result, indent=2) + "\n"
                    )
                    if actions:
                        write_episode(
                            output / "recordings",
                            task,
                            int(started * 1000),
                            observations,
                            actions,
                            action_times,
                            success=feedback.success and feedback.autonomous,
                            reset_metadata=request["reset_metadata"],
                            submitted_commands=commands,
                        )
                    print(json.dumps(result, indent=2), flush=True)
                    break
        except BaseException as exc:
            remote.infer(
                {"operation": "abort", "reason": f"{type(exc).__name__}: {exc}"}
            )
            raise


def collect_spacemouse(driver, output, task, *, count=100):
    """Collect commanded TCP targets and measured state as distinct quantities."""
    from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import (
        SpaceMouseExpert,
    )

    mouse, labels = SpaceMouseExpert(), OperatorLabels()
    root = Path(output) / task
    metadata = list(root.glob("episode_*/metadata.json"))
    successful = sum(json.loads(p.read_text())["success"] for p in metadata)
    episode_id = (
        max(
            [int(json.loads(p.read_text())["episode_id"]) for p in metadata], default=-1
        )
        + 1
    )
    while successful < count:
        note = input(
            f"{task}: {successful}/{count} successful demos. Reset scene; enter photo path/note (q to stop): "
        )
        if note.strip().lower() == "q":
            return
        initial = driver.reset()
        labels.reset()
        observations, actions, timestamps, commands = [initial], [], [], []
        gripper = (
            driver.gripper_open
            if initial["state"][-1] > (driver.gripper_open + driver.gripper_closed) / 2
            else driver.gripper_closed
        )
        deadline = monotonic()
        print(
            "SpaceMouse controls motion; left closes grip, right opens. s/f + Enter labels episode.",
            flush=True,
        )
        while True:
            status = labels.poll()
            if status.get("success") or status.get("failure") or status.get("takeover"):
                break
            movement, buttons = mouse.get_action()
            if buttons[0]:
                gripper = driver.gripper_closed
            elif buttons[1]:
                gripper = driver.gripper_open
            measured = driver.env.get_tcp_pose()
            scales = driver.env.get_action_scale()
            target = np.r_[
                measured[:3] + movement[:3] * scales[0],
                (
                    Rotation.from_euler("xyz", movement[3:6] * scales[1])
                    * Rotation.from_quat(measured[3:])
                ).as_quat(),
                gripper,
            ]
            submitted_at = time()
            try:
                observation, submitted, command, _info = driver.submit(target)
            except Exception as exc:
                status = {"failure": True}
                print(f"Collection stopped: {type(exc).__name__}: {exc}", flush=True)
                break
            timestamps.append(submitted_at)
            observations.append(observation)
            actions.append(submitted)
            commands.append(command)
            deadline += 0.05
            sleep(max(0, deadline - monotonic()))
        if actions:
            success = bool(status.get("success") and not status.get("takeover"))
            write_episode(
                output,
                task,
                episode_id,
                observations,
                actions,
                timestamps,
                success=success,
                reset_metadata={"note": note},
                submitted_commands=commands,
            )
            successful += int(success)
            episode_id += 1
