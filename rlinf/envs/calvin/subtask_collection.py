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

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory, get_model_weights_id
from rlinf.utils.nested_dict_process import stack_list_of_dict_tensor


DEFAULT_CALVIN_PROMPT = "Complete the requested CALVIN manipulation subtask."
CALVIN_SCENES = {
    "calvin_scene_A",
    "calvin_scene_B",
    "calvin_scene_C",
    "calvin_scene_D",
}


@dataclass(frozen=True)
class CalvinSubtaskSegment:
    split: str
    segment_index: int
    owner_long_episode: int
    start: int
    end: int
    prompt_text: str
    canonical_task: str
    target_window: tuple[int, ...]
    next_subtask_start_window: tuple[int, ...] = ()
    scene: str | None = None

    @property
    def length(self) -> int:
        return int(self.end - self.start + 1)


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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            converted = OmegaConf.to_container(value, resolve=True)
            return _as_list(converted)
    except ImportError:
        pass
    return [value]


def _normalize_spans(spans: Any) -> list[tuple[int, int]]:
    if isinstance(spans, np.ndarray) and spans.ndim == 2 and spans.shape[1] == 2:
        return [(int(start), int(end)) for start, end in spans.tolist()]
    normalized: list[tuple[int, int]] = []
    for span in list(spans):
        start, end = span
        normalized.append((int(start), int(end)))
    return normalized


def load_long_episode_spans(split_dir: str | Path) -> list[tuple[int, int]]:
    split_dir = Path(split_dir)
    spans_path = split_dir / "ep_start_end_ids.npy"
    if not spans_path.exists():
        return []
    return _normalize_spans(np.load(spans_path, allow_pickle=True))


def load_scene_ranges(split_dir: str | Path) -> dict[str, tuple[int, int]]:
    scene_info_path = Path(split_dir) / "scene_info.npy"
    if not scene_info_path.exists():
        return {}
    scene_info = np.load(scene_info_path, allow_pickle=True).item()
    ranges: dict[str, tuple[int, int]] = {}
    for scene, span in scene_info.items():
        start, end = span
        ranges[str(scene)] = (int(start), int(end))
    return ranges


def scene_for_frame(
    frame_index: int,
    scene_ranges: dict[str, tuple[int, int]],
) -> str | None:
    for scene, (start, end) in scene_ranges.items():
        if int(start) <= int(frame_index) <= int(end):
            return scene
    return None


def _find_owner_long_episode(
    start: int,
    end: int,
    long_spans: list[tuple[int, int]],
) -> int:
    for index, (span_start, span_end) in enumerate(long_spans):
        if span_start <= start and end <= span_end:
            return index
    return -1


def _window_ending_at(start: int, end: int, size: int) -> tuple[int, ...]:
    size = max(int(size), 1)
    window_start = max(int(start), int(end) - size + 1)
    return tuple(range(window_start, int(end) + 1))


def _window_starting_at(start: int, end: int, size: int) -> tuple[int, ...]:
    size = max(int(size), 1)
    window_end = min(int(end), int(start) + size - 1)
    return tuple(range(int(start), window_end + 1))


def load_calvin_subtask_segments(
    data_root: str | Path,
    *,
    split: str = "training",
    canonical_tasks: list[str] | tuple[str, ...] | None = None,
    segment_indices: list[int] | tuple[int, ...] | None = None,
    max_segments_per_task: int | None = None,
    max_total_segments: int | None = None,
    target_window_size: int = 8,
    next_window_size: int = 8,
    resolve_scene_from_scene_info: bool = False,
) -> list[CalvinSubtaskSegment]:
    """Load CALVIN language segments keyed by canonical task, not prompt text."""

    data_root = Path(data_root)
    split_dir = data_root / split
    annotations_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing CALVIN language annotations: {annotations_path}")

    lang_obj = np.load(annotations_path, allow_pickle=True).item()
    spans = _normalize_spans(lang_obj["info"]["indx"])
    prompts = list(lang_obj["language"]["ann"])
    tasks = list(lang_obj["language"]["task"])
    if not (len(spans) == len(prompts) == len(tasks)):
        raise ValueError(
            "CALVIN annotation lengths differ: "
            f"spans={len(spans)}, prompts={len(prompts)}, tasks={len(tasks)}"
        )

    selected_tasks = {str(task) for task in _as_list(canonical_tasks)}
    selected_indices = {int(index) for index in _as_list(segment_indices)}
    long_spans = load_long_episode_spans(split_dir)
    scene_ranges = load_scene_ranges(split_dir) if resolve_scene_from_scene_info else {}

    records: list[CalvinSubtaskSegment] = []
    for index, ((start, end), prompt, task) in enumerate(zip(spans, prompts, tasks)):
        canonical_task = str(task)
        if selected_tasks and canonical_task not in selected_tasks:
            continue
        if selected_indices and index not in selected_indices:
            continue
        owner = _find_owner_long_episode(start, end, long_spans)
        records.append(
            CalvinSubtaskSegment(
                split=str(split),
                segment_index=index,
                owner_long_episode=owner,
                start=int(start),
                end=int(end),
                prompt_text=str(prompt),
                canonical_task=canonical_task,
                target_window=_window_ending_at(start, end, target_window_size),
                scene=scene_for_frame(start, scene_ranges),
            )
        )

    records.sort(key=lambda item: (item.owner_long_episode, item.start, item.end, item.segment_index))

    next_windows_by_index: dict[int, tuple[int, ...]] = {}
    for pos, segment in enumerate(records):
        next_segment = records[pos + 1] if pos + 1 < len(records) else None
        same_long_episode = (
            next_segment is not None
            and segment.owner_long_episode >= 0
            and next_segment.owner_long_episode == segment.owner_long_episode
        )
        next_windows_by_index[segment.segment_index] = (
            _window_starting_at(next_segment.start, next_segment.end, next_window_size)
            if same_long_episode
            else ()
        )

    counts_by_task: dict[str, int] = {}
    filtered: list[CalvinSubtaskSegment] = []
    for segment in records:
        if max_segments_per_task is not None:
            count = counts_by_task.get(segment.canonical_task, 0)
            if count >= int(max_segments_per_task):
                continue
            counts_by_task[segment.canonical_task] = count + 1
        filtered.append(
            CalvinSubtaskSegment(
                split=segment.split,
                segment_index=segment.segment_index,
                owner_long_episode=segment.owner_long_episode,
                start=segment.start,
                end=segment.end,
                prompt_text=segment.prompt_text,
                canonical_task=segment.canonical_task,
                target_window=segment.target_window,
                next_subtask_start_window=next_windows_by_index[segment.segment_index],
                scene=segment.scene,
            )
        )
        if max_total_segments is not None and len(filtered) >= int(max_total_segments):
            break
    return filtered


def _frame_path(split_dir: str | Path, frame_index: int) -> Path:
    return Path(split_dir) / f"episode_{int(frame_index):07d}.npz"


def load_calvin_frame(split_dir: str | Path, frame_index: int) -> dict[str, np.ndarray]:
    frame_path = _frame_path(split_dir, frame_index)
    with np.load(frame_path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _to_batched_tensor(array: Any, *, dtype: np.dtype | None = None) -> torch.Tensor:
    arr = np.asarray(array)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).contiguous()


def obs_for_storage(obs: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: value.cpu().contiguous()
        for key, value in obs.items()
        if key != "task_descriptions" and torch.is_tensor(value)
    }


def pad_or_trim_actions(
    actions: torch.Tensor,
    action_chunk_size: int,
    action_dim: int,
) -> torch.Tensor:
    if actions.dim() == 2:
        actions = actions.unsqueeze(0)
    if actions.dim() != 3:
        raise ValueError(f"Expected actions [B, T, A], got {actions.shape}")
    if actions.shape[2] != action_dim:
        actions = actions[..., :action_dim]
    if actions.shape[1] == action_chunk_size:
        return actions
    if actions.shape[1] > action_chunk_size:
        return actions[:, :action_chunk_size, :]
    if actions.shape[1] == 0:
        raise ValueError("Cannot pad an empty action sequence")
    pad = actions[:, -1:, :].repeat(1, action_chunk_size - actions.shape[1], 1)
    return torch.cat([actions, pad], dim=1)


def _robot_obs(frame: dict[str, Any]) -> np.ndarray:
    return np.asarray(frame["robot_obs"], dtype=np.float32)


def _rgb_image(frame: dict[str, Any], key: str) -> np.ndarray | None:
    if "rgb_obs" in frame and key in frame["rgb_obs"]:
        return np.asarray(frame["rgb_obs"][key])
    if key in frame:
        return np.asarray(frame[key])
    return None


def _state_distance(current_frame: dict[str, Any], target_frames: list[dict[str, Any]]) -> float:
    current = _robot_obs(current_frame)[:7]
    distances = [
        float(np.linalg.norm(current - _robot_obs(target)[:7]) / math.sqrt(len(current)))
        for target in target_frames
    ]
    return min(distances) if distances else float("inf")


def _visual_distance(current_frame: dict[str, Any], target_frames: list[dict[str, Any]]) -> float:
    distances: list[float] = []
    for target in target_frames:
        parts: list[float] = []
        for key in ("rgb_static", "rgb_gripper"):
            current_image = _rgb_image(current_frame, key)
            target_image = _rgb_image(target, key)
            if current_image is None or target_image is None:
                continue
            if current_image.shape != target_image.shape:
                continue
            delta = current_image.astype(np.float32) / 255.0 - target_image.astype(np.float32) / 255.0
            parts.append(float(np.mean(delta * delta)))
        if parts:
            distances.append(float(np.mean(parts)))
    return min(distances) if distances else 0.0


def compute_dense_handoff_reward(
    current_frame: dict[str, Any],
    target_frames: list[dict[str, Any]],
    *,
    beta: float,
    state_weight: float = 1.0,
    visual_weight: float = 0.1,
) -> float:
    if not target_frames:
        return 0.0
    distance = 0.0
    if state_weight:
        distance += float(state_weight) * _state_distance(current_frame, target_frames)
    if visual_weight:
        distance += float(visual_weight) * _visual_distance(current_frame, target_frames)
    return float(math.exp(-distance / max(float(beta), 1e-8)))


def make_model_weights_id(model_path: str | Path, model_tag: str | None = None) -> str:
    if model_tag:
        return model_tag
    payload = str(model_path).encode("utf-8")
    return get_model_weights_id(torch.tensor(list(payload), dtype=torch.uint8))


class CalvinSubtaskEnv:
    """Single CALVIN env wrapper that resets to raw language-segment starts."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        split: str = "training",
        task_suite_name: str = "calvin_abcd",
        scene: str | None = "auto",
        action_chunk_size: int = 5,
        max_subtask_steps: int = 480,
        action_key: str = "rel_actions",
        target_window_size: int = 8,
        include_next_window_in_dense: bool = True,
        dense_beta: float = 0.2,
        dense_state_weight: float = 1.0,
        dense_visual_weight: float = 0.1,
        oracle_success_bonus: float = 1.0,
    ):
        self.data_root = Path(data_root)
        self.split = str(split)
        self.split_dir = self.data_root / self.split
        self.task_suite_name = str(task_suite_name)
        self.scene = scene
        self.action_chunk_size = int(action_chunk_size)
        self.max_subtask_steps = int(max_subtask_steps)
        self.action_key = str(action_key)
        self.target_window_size = int(target_window_size)
        self.include_next_window_in_dense = bool(include_next_window_in_dense)
        self.dense_beta = float(dense_beta)
        self.dense_state_weight = float(dense_state_weight)
        self.dense_visual_weight = float(dense_visual_weight)
        self.oracle_success_bonus = float(oracle_success_bonus)

        self.current_segment: CalvinSubtaskSegment | None = None
        self.elapsed_steps = 0
        self.max_steps = 0
        self.start_info: dict[str, Any] | None = None
        self.env = None
        self.current_scene: str | None = None
        self.target_frames: list[dict[str, Any]] = []

        from rlinf.envs.calvin import _get_calvin_tasks_and_reward

        _task_definitions, self.task_annotations, self.task_oracle = (
            _get_calvin_tasks_and_reward(1, self.task_suite_name)
        )

    def reset(self, segment: CalvinSubtaskSegment) -> dict[str, Any]:
        self.current_segment = segment
        scene = self._resolve_scene(segment)
        self._ensure_env(scene)
        start_frame = load_calvin_frame(self.split_dir, segment.start)
        self.env.reset(robot_obs=start_frame["robot_obs"], scene_obs=start_frame["scene_obs"])
        self.start_info = self.env.get_info()
        self.elapsed_steps = 0
        self.max_steps = max(1, self.max_subtask_steps)
        self.target_frames = self._load_reward_target_frames(segment)
        return self.get_obs()

    def _resolve_scene(self, segment: CalvinSubtaskSegment) -> str | None:
        if self.scene == "auto":
            if segment.scene:
                return segment.scene
            if self.task_suite_name == "calvin_d":
                return "calvin_scene_D"
            return None
        if self.scene in ("", "none", None):
            return None
        return str(self.scene)

    def _ensure_env(self, scene: str | None):
        if self.env is not None and scene == self.current_scene:
            return
        self.close()
        from rlinf.envs.calvin import make_env

        kwargs = {"scene": scene} if scene in CALVIN_SCENES else {}
        self.env = make_env(**kwargs)
        self.current_scene = scene

    def _load_reward_target_frames(self, segment: CalvinSubtaskSegment) -> list[dict[str, Any]]:
        target_indices = list(segment.target_window)
        if self.include_next_window_in_dense:
            target_indices.extend(segment.next_subtask_start_window)
        if not target_indices:
            target_indices = list(_window_ending_at(segment.start, segment.end, self.target_window_size))
        return [load_calvin_frame(self.split_dir, index) for index in target_indices]

    def get_obs(self) -> dict[str, Any]:
        if self.env is None or self.current_segment is None:
            raise RuntimeError("reset must be called before get_obs")
        raw_obs = self.env.get_obs()
        return {
            "main_images": _to_batched_tensor(raw_obs["rgb_obs"]["rgb_static"], dtype=np.uint8),
            "wrist_images": _to_batched_tensor(raw_obs["rgb_obs"]["rgb_gripper"], dtype=np.uint8),
            "extra_view_images": None,
            "states": _to_batched_tensor(raw_obs["robot_obs"][:7], dtype=np.float32),
            "robot_obs": _to_batched_tensor(raw_obs["robot_obs"], dtype=np.float32),
            "scene_obs": _to_batched_tensor(raw_obs["scene_obs"], dtype=np.float32),
            "task_descriptions": [self.current_segment.prompt_text],
        }

    def get_expert_action_chunk(self) -> torch.Tensor:
        if self.current_segment is None:
            raise RuntimeError("reset must be called before get_expert_action_chunk")
        action_start = self.current_segment.start + self.elapsed_steps
        action_end = min(self.current_segment.end + 1, action_start + self.action_chunk_size)
        actions: list[np.ndarray] = []
        for frame_index in range(action_start, action_end):
            frame = load_calvin_frame(self.split_dir, frame_index)
            if self.action_key not in frame:
                raise KeyError(f"Missing action key {self.action_key!r} in frame {frame_index}")
            actions.append(np.asarray(frame[self.action_key], dtype=np.float32))
        if not actions:
            frame = load_calvin_frame(self.split_dir, self.current_segment.end)
            actions.append(np.asarray(frame[self.action_key], dtype=np.float32))
        while len(actions) < self.action_chunk_size:
            actions.append(actions[-1].copy())
        chunk = np.stack(actions[: self.action_chunk_size], axis=0)
        return torch.from_numpy(chunk[None].astype(np.float32))

    def step_chunk(
        self,
        actions: torch.Tensor | np.ndarray,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.env is None or self.current_segment is None or self.start_info is None:
            raise RuntimeError("reset must be called before step_chunk")
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)
        if actions_np.ndim == 3:
            if actions_np.shape[0] != 1:
                raise ValueError("CalvinSubtaskEnv only supports B=1")
            actions_np = actions_np[0]
        if actions_np.ndim != 2:
            raise ValueError(f"Expected actions [T, A], got {actions_np.shape}")

        chunk_len = int(actions_np.shape[0])
        remaining = max(self.max_steps - self.elapsed_steps, 0)
        exec_len = min(chunk_len, remaining)
        rewards = torch.zeros((1, chunk_len), dtype=torch.float32)
        terminations = torch.zeros((1, chunk_len), dtype=torch.bool)
        truncations = torch.zeros((1, chunk_len), dtype=torch.bool)

        success = False
        current_info = self.env.get_info()
        executed_len = 0
        for index in range(exec_len):
            action = np.asarray(actions_np[index], dtype=np.float32).copy()
            if action.shape[-1] < 7:
                raise ValueError(f"Expected CALVIN action dim >= 7, got {action.shape}")
            action = action[:7]
            action[-1] = 1.0 if action[-1] > 0 else -1.0
            raw_obs, _reward, _done, current_info = self.env.step(action)
            dense_reward = compute_dense_handoff_reward(
                raw_obs,
                self.target_frames,
                beta=self.dense_beta,
                state_weight=self.dense_state_weight,
                visual_weight=self.dense_visual_weight,
            )
            rewards[0, index] = dense_reward
            executed_len += 1
            self.elapsed_steps += 1
            success = self.check_subtask_success(current_info)
            if success:
                rewards[0, index] += self.oracle_success_bonus
                terminations[0, index] = True
                break
            if self.elapsed_steps >= self.max_steps:
                truncations[0, index] = True
                break

        terminal_idx = max(executed_len - 1, 0)
        if exec_len == 0:
            truncations[0, terminal_idx] = True

        truncated = bool(truncations.any().item()) and not success
        terminal = bool(success or truncated)
        obs = self.get_obs()
        info = self.get_diagnostics(
            success=bool(success),
            truncated=truncated,
            terminal=terminal,
            current_info=current_info,
            chunk_reward_sum=float(rewards.sum().item()),
        )
        return obs, rewards, terminations, truncations, info

    def check_subtask_success(self, current_info: dict[str, Any] | None = None) -> bool:
        if self.current_segment is None or self.start_info is None:
            return False
        if current_info is None:
            current_info = self.env.get_info()
        current_task_info = self.task_oracle.get_task_info_for_set(
            self.start_info,
            current_info,
            {self.current_segment.canonical_task},
        )
        return len(current_task_info) > 0

    def get_diagnostics(
        self,
        *,
        success: bool,
        truncated: bool,
        terminal: bool,
        current_info: dict[str, Any] | None,
        chunk_reward_sum: float,
    ) -> dict[str, Any]:
        assert self.current_segment is not None
        diagnostics = {
            "split": self.current_segment.split,
            "segment_index": self.current_segment.segment_index,
            "owner_long_episode": self.current_segment.owner_long_episode,
            "canonical_task": self.current_segment.canonical_task,
            "prompt_text": self.current_segment.prompt_text,
            "start_frame": self.current_segment.start,
            "end_frame": self.current_segment.end,
            "target_window": list(self.current_segment.target_window),
            "next_subtask_start_window": list(
                self.current_segment.next_subtask_start_window
            ),
            "scene": self.current_scene,
            "elapsed_steps": self.elapsed_steps,
            "max_steps": self.max_steps,
            "success": success,
            "truncated": truncated,
            "terminal": terminal,
            "chunk_reward_sum": chunk_reward_sum,
        }
        if current_info is not None and self.target_frames:
            raw_obs = self.env.get_obs()
            diagnostics["target_state_distance"] = _state_distance(raw_obs, self.target_frames)
            diagnostics["target_visual_distance"] = _visual_distance(raw_obs, self.target_frames)
        return diagnostics

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
                self.current_scene = None


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
