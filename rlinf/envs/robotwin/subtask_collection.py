# Copyright 2025 The RLinf Authors.
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

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from rlinf.data.embodied_io_struct import Trajectory, get_model_weights_id
from rlinf.utils.nested_dict_process import stack_list_of_dict_tensor


DEFAULT_STACK_THREE_PROMPT = (
    "Move red block, green block, and blue block to the center. Stack green "
    "block on red block and blue block on green block."
)


@dataclass(frozen=True)
class SubtaskSegment:
    episode_idx: int
    subtask_id: int
    name: str
    arm: str
    start_step: int
    end_step: int
    action_slice: tuple[int, int]
    num_steps: int
    hdf5_file: str


@dataclass
class CollectedChunk:
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    dones: torch.Tensor
    curr_obs: dict[str, Any]
    next_obs: dict[str, Any]
    prev_logprobs: torch.Tensor | None = None
    prev_values: torch.Tensor | None = None
    versions: torch.Tensor | None = None
    forward_inputs: dict[str, Any] = field(default_factory=dict)


def _plain_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_list(value: Any) -> list:
    value = _plain_container(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def load_seed_list(seed_path: str | Path) -> list[int]:
    seed_path = Path(seed_path)
    seeds = [int(token) for token in seed_path.read_text().split()]
    if not seeds:
        raise ValueError(f"No seeds found in {seed_path}")
    return seeds


def load_subtask_segments(
    manifest_path: str | Path,
    episode_ids: list[int] | None = None,
    subtask_ids: list[int] | None = None,
) -> list[SubtaskSegment]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    episodes = manifest.get("episodes", {})
    if not isinstance(episodes, dict):
        raise ValueError(f"Expected dict episodes in {manifest_path}")

    selected_episodes = set(episode_ids) if episode_ids is not None else None
    selected_subtasks = set(subtask_ids) if subtask_ids is not None else None
    segments: list[SubtaskSegment] = []
    for episode_key in sorted(episodes.keys(), key=_episode_sort_key):
        episode_idx = _episode_sort_key(episode_key)
        if selected_episodes is not None and episode_idx not in selected_episodes:
            continue
        episode = episodes[episode_key]
        if not episode.get("valid", False):
            continue
        for subtask in episode.get("subtasks", []):
            subtask_id = int(subtask["subtask_id"])
            if selected_subtasks is not None and subtask_id not in selected_subtasks:
                continue
            action_slice = tuple(int(x) for x in subtask["action_slice"])
            segments.append(
                SubtaskSegment(
                    episode_idx=episode_idx,
                    subtask_id=subtask_id,
                    name=str(subtask["name"]),
                    arm=str(subtask.get("arm", "")),
                    start_step=int(subtask["start_step"]),
                    end_step=int(subtask["end_step"]),
                    action_slice=(action_slice[0], action_slice[1]),
                    num_steps=int(subtask["num_steps"]),
                    hdf5_file=str(episode["file"]),
                )
            )
    return segments


def _episode_sort_key(episode_key: str) -> int:
    match = re.search(r"(\d+)$", str(episode_key))
    if match is None:
        raise ValueError(f"Cannot parse episode index from key {episode_key!r}")
    return int(match.group(1))


def load_episode_prompt(
    dataset_root: str | Path,
    episode_idx: int,
    split: str = "seen",
    prompt_index: int = 0,
) -> str:
    instruction_path = Path(dataset_root) / "instructions" / f"episode{episode_idx}.json"
    if not instruction_path.exists():
        return DEFAULT_STACK_THREE_PROMPT

    with instruction_path.open("r") as f:
        instructions = json.load(f)

    prompts = instructions.get(split) or instructions.get("seen") or instructions.get("unseen")
    if not prompts:
        return DEFAULT_STACK_THREE_PROMPT
    return str(prompts[prompt_index % len(prompts)])


def make_model_weights_id(model_path: str | Path, model_tag: str | None = None) -> str:
    if model_tag:
        return model_tag
    payload = str(model_path).encode("utf-8")
    return get_model_weights_id(torch.tensor(list(payload), dtype=torch.uint8))


class RoboTwinStackThreeSubtaskEnv:
    """Single-env Robotwin wrapper that resets to expert-defined subtask starts."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        manifest_path: str | Path,
        seed_path: str | Path,
        task_config: dict[str, Any] | DictConfig,
        assets_path: str | Path,
        robotwin_python_path: str | Path | None = None,
        center_crop: bool = False,
        prompt_split: str = "seen",
        prompt_index: int = 0,
        action_chunk_size: int = 50,
        replay_chunk_size: int = 50,
        max_subtask_steps: int | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.seed_path = Path(seed_path)
        self.assets_path = str(assets_path)
        self.robotwin_python_path = (
            str(robotwin_python_path) if robotwin_python_path is not None else None
        )
        self.center_crop = center_crop
        self.prompt_split = prompt_split
        self.prompt_index = prompt_index
        self.action_chunk_size = int(action_chunk_size)
        self.replay_chunk_size = int(replay_chunk_size)
        self.max_subtask_steps = max_subtask_steps

        self.seed_list = load_seed_list(self.seed_path)
        self.task_config = _plain_container(task_config)
        self._setup_robotwin_imports()
        self.task_args = self._prepare_task_args(self.task_config)

        self.task = None
        self.current_segment: SubtaskSegment | None = None
        self.current_prompt: str = DEFAULT_STACK_THREE_PROMPT
        self.current_seed: int | None = None
        self.expert_vectors: np.ndarray | None = None
        self.elapsed_steps = 0
        self.max_steps = 0
        self._class_decorator = None

    def _setup_robotwin_imports(self):
        os.environ["ASSETS_PATH"] = self.assets_path
        if self.robotwin_python_path and self.robotwin_python_path not in sys.path:
            sys.path.insert(0, self.robotwin_python_path)

    def _prepare_task_args(self, task_config: dict[str, Any]) -> dict[str, Any]:
        from envs._GLOBAL_CONFIGS import CONFIGS_PATH

        args = dict(task_config)
        args["planner_backend"] = args.get("planner_backend", "mplib")
        camera_cfg = args.get("camera", {})
        head_camera_type = camera_cfg.get("head_camera_type", "D435")

        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r") as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        def get_embodiment_file(embodiment_type: str) -> str:
            robot_file = embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise ValueError(f"No embodiment file for {embodiment_type}")
            return robot_file

        def get_embodiment_config(robot_file: str):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)

        embodiment_type = list(args.get("embodiment", ["aloha-agilex"]))
        if len(embodiment_type) == 1:
            args["left_robot_file"] = os.path.join(
                self.assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                self.assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["dual_arm_embodied"] = True
            args["embodiment_name"] = str(embodiment_type[0])
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = os.path.join(
                self.assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                self.assets_path, get_embodiment_file(embodiment_type[1])
            )
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
            args["embodiment_name"] = f"{embodiment_type[0]}_{embodiment_type[1]}"
        else:
            raise ValueError("embodiment must contain either 1 or 3 items")

        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
        args["rdt_step"] = args.get("rdt_step", 10)
        args["save_path"] = os.path.join(
            str(args.get("save_path", "./data")), f"{args['task_name']}_reward"
        )
        args["n_envs"] = 1
        args["action_dim"] = 14
        args["eval_mode"] = True
        args["eval_video_log"] = False
        args["render_freq"] = 0
        args.setdefault("step_lim", 800)
        return args

    def _get_class_decorator(self):
        if self._class_decorator is None:
            try:
                from robotwin.envs.vector_env import class_decorator
            except ModuleNotFoundError:
                from script.collect_data import class_decorator

            self._class_decorator = class_decorator
        return self._class_decorator

    def reset(self, segment: SubtaskSegment) -> dict[str, Any]:
        self.close(clear_cache=False)
        if segment.episode_idx >= len(self.seed_list):
            raise IndexError(
                f"Episode {segment.episode_idx} has no seed in {self.seed_path}"
            )

        self.current_segment = segment
        self.current_seed = self.seed_list[segment.episode_idx]
        self.current_prompt = load_episode_prompt(
            self.dataset_root, segment.episode_idx, self.prompt_split, self.prompt_index
        )
        self.expert_vectors = self._load_expert_vectors(segment)
        self.max_steps = self._resolve_max_steps(segment)
        self.elapsed_steps = 0

        class_decorator = self._get_class_decorator()
        self.task = class_decorator(self.task_args["task_name"])
        setup_step_lim = max(
            int(self.task_args.get("step_lim", 800)),
            segment.start_step + self.max_steps + self.action_chunk_size,
        )
        setup_args = dict(self.task_args)
        setup_args["step_lim"] = setup_step_lim
        setup_args["instruction"] = self.current_prompt
        self.task.setup_demo(
            now_ep_num=segment.episode_idx,
            seed=self.current_seed,
            **setup_args,
        )
        self.task.set_instruction(self.current_prompt)

        self._replay_expert_prefix(segment)
        self.task.take_action_cnt = 0
        self.task.eval_success = False
        self.task.step_lim = self.max_steps
        self.task.run_steps = 0
        self.task.reward_step = 0

        return self.get_obs()

    def _resolve_max_steps(self, segment: SubtaskSegment) -> int:
        if self.max_subtask_steps is None:
            return int(segment.num_steps)
        return min(int(segment.num_steps), int(self.max_subtask_steps))

    def _load_expert_vectors(self, segment: SubtaskSegment) -> np.ndarray:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to load Robotwin expert HDF5 files") from exc

        hdf5_path = self.dataset_root / segment.hdf5_file
        with h5py.File(hdf5_path, "r") as f:
            return np.asarray(f["joint_action/vector"], dtype=np.float64)

    def _replay_expert_prefix(self, segment: SubtaskSegment):
        if segment.start_step <= 0:
            return
        assert self.expert_vectors is not None
        prefix_actions = self.expert_vectors[1 : segment.start_step + 1]
        executed_total = 0
        for start in range(0, len(prefix_actions), self.replay_chunk_size):
            chunk = prefix_actions[start : start + self.replay_chunk_size]
            if len(chunk) == 0:
                continue
            executed = self._execute_qpos_actions(
                chunk, stop_on_subtask_success=False
            )
            executed_total += executed
            if executed < len(chunk):
                raise RuntimeError(
                    "Failed to replay Robotwin expert prefix completely: "
                    f"executed {executed_total}/{len(prefix_actions)} actions "
                    f"before subtask start_step={segment.start_step}."
                )

    def _execute_qpos_actions(
        self, actions: np.ndarray, *, stop_on_subtask_success: bool
    ) -> int:
        if self.task is None:
            raise RuntimeError("reset must be called before executing actions")

        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[-1] != 14:
            raise ValueError(f"Expected qpos actions [T, 14], got {actions.shape}")

        executed = 0
        for action in actions:
            before_count = int(getattr(self.task, "take_action_cnt", 0))
            self.task.take_action(action, action_type="qpos")
            after_count = int(getattr(self.task, "take_action_cnt", before_count))
            if after_count <= before_count:
                break
            executed += 1
            if stop_on_subtask_success and self.check_subtask_success():
                break
        return executed

    def get_expert_action_chunk(self) -> torch.Tensor:
        if self.current_segment is None or self.expert_vectors is None:
            raise RuntimeError("reset must be called before get_expert_action_chunk")
        action_start = self.current_segment.action_slice[0] + self.elapsed_steps
        action_end = min(
            self.current_segment.action_slice[1],
            action_start + self.action_chunk_size,
        )
        chunk = self.expert_vectors[action_start:action_end]
        if len(chunk) == 0:
            current_state = self._current_joint_state()
            chunk = np.repeat(current_state[None], self.action_chunk_size, axis=0)
        elif len(chunk) < self.action_chunk_size:
            pad = np.repeat(chunk[-1][None], self.action_chunk_size - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return torch.from_numpy(chunk[None].astype(np.float32))

    def step_chunk(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.task is None or self.current_segment is None:
            raise RuntimeError("reset must be called before step_chunk")

        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)
        if actions_np.ndim == 3:
            if actions_np.shape[0] != 1:
                raise ValueError("RoboTwinStackThreeSubtaskEnv only supports B=1")
            actions_np = actions_np[0]
        if actions_np.ndim != 2 or actions_np.shape[-1] != 14:
            raise ValueError(f"Expected chunk actions [T, 14], got {actions_np.shape}")

        chunk_len = int(actions_np.shape[0])
        remaining = max(self.max_steps - self.elapsed_steps, 0)
        exec_len = min(chunk_len, remaining)

        if exec_len > 0:
            executed_len = self._execute_qpos_actions(
                actions_np[:exec_len], stop_on_subtask_success=True
            )
            self.elapsed_steps += executed_len
        else:
            executed_len = 0

        success = self.check_subtask_success()
        truncation = (self.elapsed_steps >= self.max_steps) and not success
        terminal = success or truncation
        terminal_idx = max(executed_len - 1, 0)

        rewards = torch.zeros((1, chunk_len), dtype=torch.float32)
        terminations = torch.zeros((1, chunk_len), dtype=torch.bool)
        truncations = torch.zeros((1, chunk_len), dtype=torch.bool)
        if success:
            rewards[0, terminal_idx] = 1.0
            terminations[0, terminal_idx] = True
        elif truncation:
            truncations[0, terminal_idx] = True
        dones = torch.logical_or(terminations, truncations)

        obs = self.get_obs()
        info = self.get_diagnostics(
            success=bool(success),
            truncated=bool(truncation),
            terminal=bool(terminal),
        )
        return obs, rewards, terminations, truncations, info

    def get_obs(self) -> dict[str, Any]:
        if self.task is None:
            raise RuntimeError("reset must be called before get_obs")
        raw_obs = self.task.get_obs()
        observation = raw_obs["observation"]

        main_image = self._prepare_image(observation["head_camera"]["rgb"])
        wrist_images = []
        left_image = observation.get("left_camera", {}).get("rgb")
        right_image = observation.get("right_camera", {}).get("rgb")
        if left_image is not None:
            wrist_images.append(torch.from_numpy(self._prepare_image(left_image)))
        if right_image is not None:
            wrist_images.append(torch.from_numpy(self._prepare_image(right_image)))

        wrist_tensor = (
            torch.stack(wrist_images, dim=0).unsqueeze(0).contiguous()
            if wrist_images
            else None
        )
        state = np.asarray(raw_obs["joint_action"]["vector"], dtype=np.float32)
        return {
            "main_images": torch.from_numpy(main_image).unsqueeze(0).contiguous(),
            "wrist_images": wrist_tensor,
            "extra_view_images": None,
            "states": torch.from_numpy(state).unsqueeze(0).contiguous(),
            "task_descriptions": [self.current_prompt],
        }

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if self.center_crop:
            from rlinf.envs.utils import center_crop_image

            image = np.asarray(center_crop_image(image))
        return np.ascontiguousarray(image)

    def check_subtask_success(self) -> bool:
        if self.task is None or self.current_segment is None:
            return False
        subtask_id = self.current_segment.subtask_id
        if subtask_id == 2:
            return bool(self.task.check_success())

        both_grippers_open = (
            self.task.is_left_gripper_open() and self.task.is_right_gripper_open()
        )
        if not both_grippers_open:
            return False

        if subtask_id == 0:
            target = self._target_for_subtask0()
            err = np.abs(self.task.block1.get_pose().p - target)
            return bool(np.all(err < np.array([0.04, 0.04, 0.03])))
        if subtask_id == 1:
            target = self._target_for_subtask1()
            err = np.abs(self.task.block2.get_pose().p - target)
            return bool(np.all(err < np.array([0.035, 0.035, 0.025])))
        raise ValueError(f"Unsupported subtask_id={subtask_id}")

    def get_diagnostics(self, *, success: bool, truncated: bool, terminal: bool) -> dict[str, Any]:
        assert self.current_segment is not None
        diagnostics = {
            "episode_idx": self.current_segment.episode_idx,
            "sim_seed": self.current_seed,
            "subtask_id": self.current_segment.subtask_id,
            "subtask_name": self.current_segment.name,
            "start_step": self.current_segment.start_step,
            "end_step": self.current_segment.end_step,
            "elapsed_steps": self.elapsed_steps,
            "max_steps": self.max_steps,
            "success": success,
            "truncated": truncated,
            "terminal": terminal,
        }
        diagnostics.update(self._pose_diagnostics())
        diagnostics["joint_l2_to_expert_end"] = self._joint_l2_to_expert_end()
        return diagnostics

    def _pose_diagnostics(self) -> dict[str, Any]:
        assert self.current_segment is not None
        block1 = self.task.block1.get_pose().p
        block2 = self.task.block2.get_pose().p
        block3 = self.task.block3.get_pose().p
        target = self._target_for_current_subtask()
        current = [block1, block2, block3][self.current_segment.subtask_id]
        return {
            "block1_pose": block1.tolist(),
            "block2_pose": block2.tolist(),
            "block3_pose": block3.tolist(),
            "target_pose": target.tolist(),
            "target_l2": float(np.linalg.norm(current - target)),
            "target_linf": float(np.max(np.abs(current - target))),
        }

    def _target_for_current_subtask(self) -> np.ndarray:
        assert self.current_segment is not None
        if self.current_segment.subtask_id == 0:
            return self._target_for_subtask0()
        if self.current_segment.subtask_id == 1:
            return self._target_for_subtask1()
        block2 = self.task.block2.get_pose().p
        return np.array([block2[0], block2[1], block2[2] + 0.05])

    def _target_for_subtask0(self) -> np.ndarray:
        if hasattr(self.task, "block1_target_pose"):
            return np.asarray(self.task.block1_target_pose[:3], dtype=np.float64)
        return np.asarray([0.0, -0.13, 0.75 + self.task.table_z_bias], dtype=np.float64)

    def _target_for_subtask1(self) -> np.ndarray:
        block1 = self.task.block1.get_pose().p
        return np.asarray([block1[0], block1[1], block1[2] + 0.05], dtype=np.float64)

    def _current_joint_state(self) -> np.ndarray:
        left = self.task.robot.get_left_arm_jointState()
        right = self.task.robot.get_right_arm_jointState()
        return np.asarray(left + right, dtype=np.float64)

    def _joint_l2_to_expert_end(self) -> float | None:
        if self.current_segment is None or self.expert_vectors is None:
            return None
        end_step = min(self.current_segment.end_step, len(self.expert_vectors) - 1)
        return float(np.linalg.norm(self._current_joint_state() - self.expert_vectors[end_step]))

    def close(self, clear_cache: bool = True):
        if self.task is not None:
            try:
                self.task.close_env(clear_cache=clear_cache)
            finally:
                self.task = None


def obs_for_storage(obs: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: value.cpu().contiguous()
        for key, value in obs.items()
        if key != "task_descriptions" and torch.is_tensor(value)
    }


def pad_or_trim_actions(actions: torch.Tensor, action_chunk_size: int, action_dim: int) -> torch.Tensor:
    if actions.dim() == 2:
        actions = actions[:, None, :]
    if actions.dim() != 3:
        raise ValueError(f"Expected actions [B, T, A], got {actions.shape}")
    if actions.shape[1] == action_chunk_size and actions.shape[2] == action_dim:
        return actions
    if actions.shape[2] != action_dim:
        actions = actions[..., :action_dim]
    if actions.shape[1] > action_chunk_size:
        return actions[:, :action_chunk_size, :]
    pad = actions[:, -1:, :].repeat(1, action_chunk_size - actions.shape[1], 1)
    return torch.cat([actions, pad], dim=1)


def build_trajectory(
    chunks: list[CollectedChunk],
    *,
    max_episode_length: int,
    model_weights_id: str,
) -> Trajectory:
    if not chunks:
        raise ValueError("Cannot build trajectory from zero chunks")

    trajectory = Trajectory(max_episode_length=max_episode_length)
    trajectory.model_weights_id = model_weights_id
    trajectory.actions = torch.stack([c.actions for c in chunks], dim=0).cpu().contiguous()
    trajectory.rewards = torch.stack([c.rewards for c in chunks], dim=0).cpu().contiguous()
    trajectory.terminations = (
        torch.stack([c.terminations for c in chunks], dim=0).cpu().contiguous()
    )
    trajectory.truncations = (
        torch.stack([c.truncations for c in chunks], dim=0).cpu().contiguous()
    )
    trajectory.dones = torch.stack([c.dones for c in chunks], dim=0).cpu().contiguous()

    if all(c.prev_logprobs is not None for c in chunks):
        trajectory.prev_logprobs = (
            torch.stack([c.prev_logprobs for c in chunks], dim=0).cpu().contiguous()
        )
    if all(c.prev_values is not None for c in chunks):
        trajectory.prev_values = (
            torch.stack([c.prev_values for c in chunks], dim=0).cpu().contiguous()
        )
    if all(c.versions is not None for c in chunks):
        trajectory.versions = (
            torch.stack([c.versions for c in chunks], dim=0).cpu().contiguous()
        )
    if any(c.forward_inputs for c in chunks):
        trajectory.forward_inputs = stack_list_of_dict_tensor(
            [c.forward_inputs for c in chunks]
        )
    trajectory.curr_obs = stack_list_of_dict_tensor([c.curr_obs for c in chunks])
    trajectory.next_obs = stack_list_of_dict_tensor([c.next_obs for c in chunks])
    return trajectory


def set_collection_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))
