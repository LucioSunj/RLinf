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
"""Convert Robotwin LeRobot v2.1 parquet episodes into RLinf Trajectory files."""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.data.embodied_io_struct import Trajectory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Robotwin LeRobot parquet episodes into RLinf trajectory files."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-camera-key", default="observation.images.cam_high")
    parser.add_argument(
        "--wrist-camera-keys",
        nargs="*",
        default=[
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
    )
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--step-reward", type=float, default=0.0)
    parser.add_argument("--terminal-reward", type=float, default=1.0)
    parser.add_argument("--num-action-chunks", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_pyarrow() -> Any:
    import pyarrow.parquet as pq

    return pq


def _require_pillow() -> Any:
    from PIL import Image

    return Image


def _resolve_parquet_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    default_data_dir = dataset_path / "data"
    if default_data_dir.exists():
        return sorted(default_data_dir.glob("chunk-*/episode_*.parquet"))
    return sorted(dataset_path.glob("**/*.parquet"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_task_map(meta_dir: Path) -> dict[int, str]:
    return {
        int(row["task_index"]): row["task"]
        for row in _load_jsonl(meta_dir / "tasks.jsonl")
        if "task_index" in row and "task" in row
    }


def _infer_episode_index(parquet_path: Path, rows: list[dict[str, Any]]) -> int:
    if rows and "episode_index" in rows[0]:
        return int(rows[0]["episode_index"])
    match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unable to infer episode index from {parquet_path}")


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _decode_image(image_struct: dict[str, Any], image_cls: Any) -> np.ndarray:
    raw_bytes = image_struct.get("bytes")
    if isinstance(raw_bytes, memoryview):
        raw_bytes = raw_bytes.tobytes()
    with image_cls.open(io.BytesIO(raw_bytes)) as image:
        return np.asarray(image.convert("RGB"))


def _to_batched_tensor(
    value: Any,
    *,
    dtype: np.dtype | None = None,
) -> torch.Tensor:
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def _extract_prompt(
    row: dict[str, Any],
    prompt_key: str,
    task_map: dict[int, str],
) -> str:
    prompt = row.get(prompt_key, None)
    if prompt is not None and str(prompt).strip():
        return str(prompt)
    task_index = row.get("task_index", None)
    if task_index is not None and int(task_index) in task_map:
        return task_map[int(task_index)]
    return "robotwin task"


def _extract_obs(
    row: dict[str, Any],
    *,
    image_cls: Any,
    main_camera_key: str,
    wrist_camera_keys: list[str],
    state_key: str,
    prompt: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "states": _to_batched_tensor(row[state_key], dtype=np.float32),
        "task_descriptions": [prompt],
    }
    obs["main_images"] = _to_batched_tensor(_decode_image(row[main_camera_key], image_cls))

    wrist_images = []
    for wrist_key in wrist_camera_keys:
        if wrist_key in row and row[wrist_key] is not None:
            wrist_images.append(_decode_image(row[wrist_key], image_cls))
    if wrist_images:
        obs["wrist_images"] = torch.from_numpy(
            np.ascontiguousarray(np.stack(wrist_images, axis=0))
        ).unsqueeze(0)
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
    rows: list[dict[str, Any]],
    frame_id: int,
    *,
    action_key: str,
    num_action_chunks: int,
) -> torch.Tensor:
    last_idx = len(rows) - 2
    chunks = []
    for offset in range(num_action_chunks):
        idx = min(frame_id + offset, last_idx)
        chunks.append(np.asarray(rows[idx][action_key], dtype=np.float32))
    return torch.from_numpy(np.ascontiguousarray(np.stack(chunks, axis=0))).unsqueeze(0)


def convert_robotwin_dataset(
    *,
    dataset_path: Path,
    output_dir: Path,
    main_camera_key: str,
    wrist_camera_keys: list[str],
    state_key: str,
    action_key: str,
    prompt_key: str,
    step_reward: float,
    terminal_reward: float,
    num_action_chunks: int,
    overwrite: bool,
) -> dict[str, Any]:
    pq = _require_pyarrow()
    image_cls = _require_pillow()
    parquet_files = _resolve_parquet_files(dataset_path)
    _prepare_output_dir(output_dir, overwrite)
    task_map = _build_task_map(dataset_path / "meta")

    manifest_path = output_dir / "manifest.jsonl"
    num_written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for parquet_path in parquet_files:
            rows = pq.read_table(parquet_path).to_pylist()
            if len(rows) < 2:
                continue
            episode_index = _infer_episode_index(parquet_path, rows)
            curr_obs_steps: list[dict[str, Any]] = []
            next_obs_steps: list[dict[str, Any]] = []
            actions: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            terminations: list[torch.Tensor] = []
            truncations: list[torch.Tensor] = []
            dones: list[torch.Tensor] = []

            for frame_id in range(len(rows) - 1):
                prompt = _extract_prompt(rows[frame_id], prompt_key, task_map)
                curr_obs_steps.append(
                    _extract_obs(
                        rows[frame_id],
                        image_cls=image_cls,
                        main_camera_key=main_camera_key,
                        wrist_camera_keys=wrist_camera_keys,
                        state_key=state_key,
                        prompt=prompt,
                    )
                )
                next_obs_steps.append(
                    _extract_obs(
                        rows[frame_id + 1],
                        image_cls=image_cls,
                        main_camera_key=main_camera_key,
                        wrist_camera_keys=wrist_camera_keys,
                        state_key=state_key,
                        prompt=prompt,
                    )
                )
                actions.append(
                    _build_action_chunk(
                        rows,
                        frame_id,
                        action_key=action_key,
                        num_action_chunks=max(1, num_action_chunks),
                    )
                )

                is_terminal = frame_id == len(rows) - 2
                reward_value = terminal_reward if is_terminal else step_reward
                rewards.append(torch.tensor([[reward_value]], dtype=torch.float32))
                terminations.append(torch.tensor([[is_terminal]], dtype=torch.bool))
                truncations.append(torch.tensor([[False]], dtype=torch.bool))
                dones.append(torch.tensor([[is_terminal]], dtype=torch.bool))

            trajectory_name = f"trajectory_{episode_index:06d}"
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
            output_path = output_dir / f"{trajectory_name}.pt"
            torch.save(trajectory, output_path)
            manifest_file.write(
                json.dumps(
                    {
                        "trajectory_file": output_path.name,
                        "episode_index": episode_index,
                        "source_parquet": str(parquet_path),
                        "num_transitions": len(actions),
                        "num_action_chunks": num_action_chunks,
                        "main_camera_key": main_camera_key,
                        "wrist_camera_keys": wrist_camera_keys,
                        "state_key": state_key,
                        "action_key": action_key,
                        "prompt_key": prompt_key,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            num_written += 1

    summary = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "num_episodes": len(parquet_files),
        "num_written": num_written,
        "num_action_chunks": num_action_chunks,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    summary = convert_robotwin_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        main_camera_key=args.main_camera_key,
        wrist_camera_keys=list(args.wrist_camera_keys),
        state_key=args.state_key,
        action_key=args.action_key,
        prompt_key=args.prompt_key,
        step_reward=args.step_reward,
        terminal_reward=args.terminal_reward,
        num_action_chunks=max(1, int(args.num_action_chunks)),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
