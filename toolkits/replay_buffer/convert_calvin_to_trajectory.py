#!/usr/bin/env python
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
"""Convert CALVIN raw episodes into RLinf Trajectory files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.data.embodied_io_struct import Trajectory


@dataclass(frozen=True)
class CalvinSegment:
    split_name: str
    segment_index: int
    start: int
    end: int
    prompt: str | None = None
    canonical_task: str | None = None


@dataclass(frozen=True)
class ConversionConfig:
    input_dir: Path
    output_dir: Path
    split_name: str
    segment_mode: str
    action_key: str
    state_source: str
    step_reward: float
    terminal_reward: float
    num_action_chunks: int
    include_rgb_static: bool
    include_rgb_gripper: bool
    include_scene_obs: bool
    include_robot_obs_full: bool
    overwrite: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CALVIN raw frames into RLinf trajectory files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--segment-mode",
        choices=("auto", "long_episode", "language"),
        default="auto",
    )
    parser.add_argument(
        "--action-key",
        choices=("rel_actions", "actions"),
        default="rel_actions",
    )
    parser.add_argument(
        "--state-source",
        choices=("robot_obs_7", "robot_obs", "robot_scene"),
        default="robot_obs_7",
    )
    parser.add_argument("--step-reward", type=float, default=0.0)
    parser.add_argument("--terminal-reward", type=float, default=1.0)
    parser.add_argument("--num-action-chunks", type=int, default=1)
    parser.add_argument(
        "--include-rgb-static",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-rgb-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-scene-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-robot-obs-full",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _to_tensor(array: np.ndarray, *, dtype: np.dtype | None = None) -> torch.Tensor:
    arr = np.asarray(array)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr)


def _to_batched_tensor(
    array: np.ndarray,
    *,
    dtype: np.dtype | None = None,
) -> torch.Tensor:
    return _to_tensor(array, dtype=dtype).unsqueeze(0)


def _empty_extra_view_placeholder() -> torch.Tensor:
    return torch.empty((1, 0, 0, 3), dtype=torch.uint8)


def _frame_path(split_dir: Path, frame_id: int) -> Path:
    return split_dir / f"episode_{frame_id:07d}.npz"


def _load_frame(frame_path: Path) -> dict[str, np.ndarray]:
    with np.load(frame_path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _normalize_spans(spans: np.ndarray) -> list[tuple[int, int]]:
    spans = np.asarray(spans)
    if spans.ndim == 1:
        spans = spans.reshape(1, 2)
    return [(int(start), int(end)) for start, end in spans]


def _load_long_episode_segments(split_dir: Path, split_name: str) -> list[CalvinSegment]:
    spans = _normalize_spans(np.load(split_dir / "ep_start_end_ids.npy", allow_pickle=True))
    return [
        CalvinSegment(
            split_name=split_name,
            segment_index=index,
            start=start,
            end=end,
        )
        for index, (start, end) in enumerate(spans)
    ]


def _load_language_segments(split_dir: Path, split_name: str) -> list[CalvinSegment]:
    lang_obj = np.load(
        split_dir / "lang_annotations" / "auto_lang_ann.npy",
        allow_pickle=True,
    ).item()
    spans = _normalize_spans(lang_obj["info"]["indx"])
    prompts = list(lang_obj["language"]["ann"])
    tasks = list(lang_obj["language"].get("task", [None] * len(spans)))
    return [
        CalvinSegment(
            split_name=split_name,
            segment_index=index,
            start=start,
            end=end,
            prompt=str(prompt),
            canonical_task=None if task is None else str(task),
        )
        for index, ((start, end), prompt, task) in enumerate(zip(spans, prompts, tasks))
    ]


def resolve_segments(split_dir: Path, split_name: str, mode: str) -> tuple[str, list[CalvinSegment]]:
    if mode == "long_episode":
        return mode, _load_long_episode_segments(split_dir, split_name)
    if mode == "language":
        return mode, _load_language_segments(split_dir, split_name)
    if (split_dir / "lang_annotations" / "auto_lang_ann.npy").exists():
        return "language", _load_language_segments(split_dir, split_name)
    return "long_episode", _load_long_episode_segments(split_dir, split_name)


def _build_states(frame: dict[str, np.ndarray], state_source: str) -> np.ndarray:
    robot_obs = np.asarray(frame["robot_obs"], dtype=np.float32)
    if state_source == "robot_obs_7":
        return robot_obs[:7]
    if state_source == "robot_obs":
        return robot_obs
    if state_source == "robot_scene":
        scene_obs = np.asarray(frame["scene_obs"], dtype=np.float32)
        return np.concatenate([robot_obs, scene_obs], axis=0)
    raise ValueError(f"Unsupported state_source: {state_source}")


def _extract_obs(
    frame: dict[str, np.ndarray],
    config: ConversionConfig,
    prompt: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "states": _to_batched_tensor(
            _build_states(frame, config.state_source), dtype=np.float32
        ),
        "task_descriptions": [prompt],
        "extra_view_images": _empty_extra_view_placeholder(),
    }
    if config.include_robot_obs_full:
        obs["robot_obs"] = _to_batched_tensor(frame["robot_obs"], dtype=np.float32)
    if config.include_scene_obs:
        obs["scene_obs"] = _to_batched_tensor(frame["scene_obs"], dtype=np.float32)
    if config.include_rgb_static:
        obs["main_images"] = _to_batched_tensor(frame["rgb_static"])
    if config.include_rgb_gripper:
        obs["wrist_images"] = _to_batched_tensor(frame["rgb_gripper"])
    return obs


def _stack_step_dicts(step_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not step_dicts:
        return {}

    keys = step_dicts[0].keys()
    stacked: dict[str, Any] = {}
    for key in keys:
        values = [step_dict[key] for step_dict in step_dicts]
        if all(isinstance(value, torch.Tensor) for value in values):
            stacked[key] = torch.stack(values, dim=0).contiguous()
        elif all(isinstance(value, (list, tuple)) for value in values):
            stacked[key] = [list(value) for value in values]
        else:
            stacked[key] = values
    return stacked


def _build_action_chunk(
    split_dir: Path,
    frame_id: int,
    *,
    action_key: str,
    num_action_chunks: int,
    last_valid_frame_id: int,
) -> torch.Tensor:
    action_chunks = []
    last_action = None
    for offset in range(num_action_chunks):
        action_frame_id = min(frame_id + offset, last_valid_frame_id)
        action_frame = _load_frame(_frame_path(split_dir, action_frame_id))
        action = np.asarray(action_frame[action_key], dtype=np.float32)
        last_action = action
        action_chunks.append(action)
    if not action_chunks:
        action_chunks.append(last_action)
    return _to_tensor(np.stack(action_chunks, axis=0), dtype=np.float32).unsqueeze(0)


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def convert_calvin_split(config: ConversionConfig) -> dict[str, Any]:
    split_dir = config.input_dir
    _prepare_output_dir(config.output_dir, config.overwrite)
    resolved_mode, segments = resolve_segments(
        split_dir=split_dir,
        split_name=config.split_name,
        mode=config.segment_mode,
    )

    manifest_path = config.output_dir / "manifest.jsonl"
    num_written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for segment in segments:
            if segment.end <= segment.start:
                continue

            prompt = segment.prompt or segment.canonical_task or config.split_name
            curr_obs_steps: list[dict[str, Any]] = []
            next_obs_steps: list[dict[str, Any]] = []
            actions: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            terminations: list[torch.Tensor] = []
            truncations: list[torch.Tensor] = []
            dones: list[torch.Tensor] = []

            current_frame = _load_frame(_frame_path(split_dir, segment.start))
            for frame_id in range(segment.start, segment.end):
                next_frame = _load_frame(_frame_path(split_dir, frame_id + 1))
                curr_obs_steps.append(_extract_obs(current_frame, config, prompt))
                next_obs_steps.append(_extract_obs(next_frame, config, prompt))
                actions.append(
                    _build_action_chunk(
                        split_dir,
                        frame_id,
                        action_key=config.action_key,
                        num_action_chunks=config.num_action_chunks,
                        last_valid_frame_id=segment.end - 1,
                    )
                )

                is_terminal = frame_id == segment.end - 1
                reward_value = (
                    config.terminal_reward if is_terminal else config.step_reward
                )
                rewards.append(torch.tensor([[reward_value]], dtype=torch.float32))
                terminations.append(torch.tensor([[is_terminal]], dtype=torch.bool))
                truncations.append(torch.tensor([[False]], dtype=torch.bool))
                dones.append(torch.tensor([[is_terminal]], dtype=torch.bool))
                current_frame = next_frame

            trajectory_name = f"trajectory_{segment.segment_index:06d}"
            trajectory = Trajectory(
                max_episode_length=len(actions),
                model_weights_id=trajectory_name,
                actions=torch.stack(actions, dim=0).contiguous(),
                rewards=torch.stack(rewards, dim=0).contiguous(),
                terminations=torch.stack(terminations, dim=0).contiguous(),
                truncations=torch.stack(truncations, dim=0).contiguous(),
                dones=torch.stack(dones, dim=0).contiguous(),
                curr_obs=_stack_step_dicts(curr_obs_steps),
                next_obs=_stack_step_dicts(next_obs_steps),
            )
            output_path = config.output_dir / f"{trajectory_name}.pt"
            torch.save(trajectory, output_path)
            manifest_file.write(
                json.dumps(
                    {
                        "trajectory_file": output_path.name,
                        "segment": asdict(segment),
                        "num_transitions": len(actions),
                        "num_action_chunks": config.num_action_chunks,
                        "action_key": config.action_key,
                        "state_source": config.state_source,
                        "resolved_mode": resolved_mode,
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            num_written += 1

    summary = {
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "split_name": config.split_name,
        "resolved_segment_mode": resolved_mode,
        "num_segments": len(segments),
        "num_written": num_written,
        "num_action_chunks": config.num_action_chunks,
    }
    (config.output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    config = ConversionConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_name=args.input_dir.name,
        segment_mode=args.segment_mode,
        action_key=args.action_key,
        state_source=args.state_source,
        step_reward=args.step_reward,
        terminal_reward=args.terminal_reward,
        num_action_chunks=max(1, int(args.num_action_chunks)),
        include_rgb_static=bool(args.include_rgb_static),
        include_rgb_gripper=bool(args.include_rgb_gripper),
        include_scene_obs=bool(args.include_scene_obs),
        include_robot_obs_full=bool(args.include_robot_obs_full),
        overwrite=bool(args.overwrite),
    )
    summary = convert_calvin_split(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
