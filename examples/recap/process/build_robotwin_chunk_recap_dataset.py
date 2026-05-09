#!/usr/bin/env python3
"""Build chunk-level RoboTwin RECAP datasets from expert HDF5 and RLinf rollouts.

Each output LeRobot row is one OpenPI action chunk:

    observation = chunk start observation
    action      = [action_chunk_size, action_dim]
    reward      = chunk-level DINO handoff reward plus terminal task reward

The generated ``meta/returns_{tag}.parquet`` sidecar is aligned by
``(episode_index, frame_index)`` with the chunk-level LeRobot rows, which is the
granularity consumed by RECAP value and CFG training.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm


RLINF_ROOT = Path(__file__).resolve().parents[3]
REWARD_DIR = RLINF_ROOT / "examples" / "reward"
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))
if str(REWARD_DIR) not in sys.path:
    sys.path.insert(0, str(REWARD_DIR))

from dino_reward_calculator import DEFAULT_DINOV3_CHECKPOINT, DINORewardCalculator
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rlinf.envs.robotwin.subtask_collection import (
    DEFAULT_STACK_THREE_PROMPT,
    SubtaskSegment,
    load_episode_prompt,
    load_subtask_segments,
)


DEFAULT_DATASET_ROOT = Path(
    "/home/lsk/long-horizon-manipulation/dataset/robotwin/stack_block_three/"
    "aloha-agilex_clean_50"
)
DEFAULT_ROLLOUT_DIR = (
    DEFAULT_DATASET_ROOT
    / "subtask_rollouts"
    / "pi05_19000_rlinf_all_episodes_n1_20260509"
)
DEFAULT_TAG = "dino_handoff_chunk"


@dataclass
class ChunkRecord:
    frame: dict[str, Any]
    curr_head: Image.Image
    next_head: Image.Image
    source: str
    original_episode_idx: int
    subtask_id: int
    subtask_name: str
    chunk_index: int
    terminal: bool
    success: bool
    truncated: bool
    rollout_index: int | None = None
    real_action_steps: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--expert-output", type=Path, default=None)
    parser.add_argument("--rollout-output", type=Path, default=None)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--limit-rollouts", type=int, default=None)

    parser.add_argument("--dino-checkpoint", type=Path, default=Path(DEFAULT_DINOV3_CHECKPOINT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reference-window", type=int, default=1)
    parser.add_argument("--handoff-scale", type=float, default=5.0)
    parser.add_argument("--step-penalty", type=float, default=-1.0)
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--failure-reward", type=float, default=-10.0)
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument("--action-chunk-size", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=0)
    parser.add_argument("--prompt-split", default="seen")
    parser.add_argument("--prompt-index", type=int, default=0)
    return parser.parse_args()


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_root = args.output_root
    if output_root is None:
        output_root = args.rollout_dir / "recap_chunk_datasets" / args.tag
    expert_output = args.expert_output or output_root / "expert"
    rollout_output = args.rollout_output or output_root / "rollout"
    return output_root, expert_output, rollout_output


def maybe_remove_output(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    shutil.rmtree(path)


def create_dataset(
    output_path: Path,
    *,
    fps: int,
    image_shape: tuple[int, int, int],
    action_chunk_size: int,
    action_dim: int,
    image_writer_processes: int,
    image_writer_threads: int,
) -> LeRobotDataset:
    features = {
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_chunk_size, action_dim),
            "names": ["chunk_step", "action"],
        },
        "done": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["done"],
        },
        "is_success": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_success"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=output_path.name,
        root=output_path,
        robot_type="aloha",
        fps=fps,
        features=features,
        use_videos=False,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def close_dataset(dataset: LeRobotDataset) -> None:
    if getattr(dataset, "image_writer", None) is not None:
        dataset.stop_image_writer()


def load_segments_by_episode(dataset_root: Path) -> dict[int, list[SubtaskSegment]]:
    segments = load_subtask_segments(dataset_root / "subtask_segments.json")
    result: dict[int, list[SubtaskSegment]] = {}
    for segment in segments:
        result.setdefault(segment.episode_idx, []).append(segment)
    for episode_segments in result.values():
        episode_segments.sort(key=lambda item: item.subtask_id)
    return result


def decode_hdf5_rgb(value: np.bytes_ | bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(bytes(value))).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_hdf5_frame(h5: h5py.File, frame_idx: int, action_dim: int) -> dict[str, np.ndarray]:
    def image(camera: str) -> np.ndarray:
        key = f"observation/{camera}/rgb"
        max_frame = len(h5[key]) - 1
        return decode_hdf5_rgb(h5[key][min(frame_idx, max_frame)])

    state_key = "joint_action/vector"
    max_state_frame = len(h5[state_key]) - 1
    return {
        "head": image("head_camera"),
        "left": image("left_camera"),
        "right": image("right_camera"),
        "state": np.asarray(
            h5[state_key][min(frame_idx, max_state_frame)], dtype=np.float32
        )[:action_dim],
    }


def as_pil_rgb(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


def pad_action_chunk(
    actions: np.ndarray,
    *,
    action_chunk_size: int,
    action_dim: int,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        actions = actions.reshape(-1, action_dim)
    actions = actions[:, :action_dim]
    if len(actions) == 0:
        actions = np.zeros((1, action_dim), dtype=np.float32)
    if len(actions) > action_chunk_size:
        return actions[:action_chunk_size].astype(np.float32, copy=False)
    if len(actions) < action_chunk_size:
        pad = np.repeat(actions[-1][None], action_chunk_size - len(actions), axis=0)
        actions = np.concatenate([actions, pad], axis=0)
    return actions.astype(np.float32, copy=False)


def build_expert_chunks(
    dataset_root: Path,
    segment: SubtaskSegment,
    *,
    action_chunk_size: int,
    action_dim: int,
    prompt_split: str,
    prompt_index: int,
) -> list[ChunkRecord]:
    h5_path = dataset_root / segment.hdf5_file
    prompt = load_episode_prompt(dataset_root, segment.episode_idx, prompt_split, prompt_index)
    chunks: list[ChunkRecord] = []
    with h5py.File(h5_path, "r") as h5:
        expert_actions = np.asarray(h5["joint_action/vector"], dtype=np.float32)
        for chunk_index, offset in enumerate(range(0, segment.num_steps, action_chunk_size)):
            real_steps = min(action_chunk_size, segment.num_steps - offset)
            curr_frame_idx = segment.start_step + offset
            next_frame_idx = segment.start_step + offset + real_steps
            curr = load_hdf5_frame(h5, curr_frame_idx, action_dim)
            nxt = load_hdf5_frame(h5, next_frame_idx, action_dim)
            action_start = segment.action_slice[0] + offset
            action_end = action_start + real_steps
            actions = pad_action_chunk(
                expert_actions[action_start:action_end],
                action_chunk_size=action_chunk_size,
                action_dim=action_dim,
            )
            terminal = offset + real_steps >= segment.num_steps
            frame = {
                "observation.images.cam_high": curr["head"],
                "observation.images.cam_left_wrist": curr["left"],
                "observation.images.cam_right_wrist": curr["right"],
                "observation.state": curr["state"],
                "action": actions,
                "done": np.asarray([terminal], dtype=np.bool_),
                "is_success": np.asarray([True], dtype=np.bool_),
                "task": prompt,
            }
            chunks.append(
                ChunkRecord(
                    frame=frame,
                    curr_head=as_pil_rgb(curr["head"]),
                    next_head=as_pil_rgb(nxt["head"]),
                    source="expert",
                    original_episode_idx=segment.episode_idx,
                    subtask_id=segment.subtask_id,
                    subtask_name=segment.name,
                    chunk_index=chunk_index,
                    terminal=terminal,
                    success=True,
                    truncated=False,
                    real_action_steps=real_steps,
                )
            )
    return chunks


def load_rollout_manifest(rollout_dir: Path) -> list[dict[str, Any]]:
    manifest_path = rollout_dir / "subtask_rollout_manifest.jsonl"
    with manifest_path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def rollout_path(rollout_dir: Path, rollout_index: int) -> Path:
    candidates = sorted(
        (rollout_dir / "replay_buffer").glob(f"trajectory_{rollout_index}_*.pt")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one replay trajectory for rollout {rollout_index}, got {len(candidates)}"
        )
    return candidates[0]


def tensor_image(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.uint8, copy=False)


def build_rollout_chunks(
    dataset_root: Path,
    rollout_dir: Path,
    entry: dict[str, Any],
    *,
    action_chunk_size: int,
    action_dim: int,
    prompt_split: str,
    prompt_index: int,
) -> list[ChunkRecord]:
    path = rollout_path(rollout_dir, int(entry["rollout_index"]))
    payload = torch.load(path, map_location="cpu")
    curr_obs = payload["curr_obs"]
    next_obs = payload["next_obs"]
    actions = payload["actions"]
    num_chunks = int(actions.shape[0])
    prompt = load_episode_prompt(
        dataset_root, int(entry["episode_idx"]), prompt_split, prompt_index
    )
    success = bool(entry.get("success", False))
    truncated = bool(entry.get("truncated", False))
    chunks: list[ChunkRecord] = []
    for chunk_index in range(num_chunks):
        curr_head = tensor_image(curr_obs["main_images"][chunk_index, 0])
        left = tensor_image(curr_obs["wrist_images"][chunk_index, 0, 0])
        right = tensor_image(curr_obs["wrist_images"][chunk_index, 0, 1])
        state = curr_obs["states"][chunk_index, 0].detach().cpu().numpy().astype(np.float32)
        next_head = tensor_image(next_obs["main_images"][chunk_index, 0])
        action_arr = actions[chunk_index, 0].detach().cpu().numpy().astype(np.float32)
        action_arr = pad_action_chunk(
            action_arr.reshape(-1, action_dim),
            action_chunk_size=action_chunk_size,
            action_dim=action_dim,
        )
        terminal = chunk_index == num_chunks - 1
        frame = {
            "observation.images.cam_high": curr_head,
            "observation.images.cam_left_wrist": left,
            "observation.images.cam_right_wrist": right,
            "observation.state": state[:action_dim],
            "action": action_arr,
            "done": np.asarray([terminal], dtype=np.bool_),
            "is_success": np.asarray([success], dtype=np.bool_),
            "task": prompt,
        }
        chunks.append(
            ChunkRecord(
                frame=frame,
                curr_head=as_pil_rgb(curr_head),
                next_head=as_pil_rgb(next_head),
                source="rollout",
                original_episode_idx=int(entry["episode_idx"]),
                subtask_id=int(entry["subtask_id"]),
                subtask_name=str(entry.get("subtask_name", "")),
                chunk_index=chunk_index,
                terminal=terminal,
                success=success,
                truncated=truncated,
                rollout_index=int(entry["rollout_index"]),
                real_action_steps=action_chunk_size,
            )
        )
    return chunks


def build_reference_embeddings(
    calculator: DINORewardCalculator,
    dataset_root: Path,
    segments_by_episode: dict[int, list[SubtaskSegment]],
    *,
    reference_window: int,
    batch_size: int,
    action_dim: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    refs: dict[tuple[int, int], dict[str, Any]] = {}
    for episode_idx, segments in tqdm(
        sorted(segments_by_episode.items()), desc="DINO references"
    ):
        h5_path = dataset_root / segments[0].hdf5_file
        with h5py.File(h5_path, "r") as h5:
            for segment in segments:
                if segment.subtask_id >= len(segments) - 1:
                    refs[(episode_idx, segment.subtask_id)] = {
                        "embedding": None,
                        "ref_frame_start": None,
                        "ref_frame_end": None,
                    }
                    continue
                next_start = int(segments[segment.subtask_id + 1].start_step)
                images = []
                for offset in range(reference_window):
                    frame = load_hdf5_frame(h5, next_start + offset, action_dim)
                    images.append(as_pil_rgb(frame["head"]))
                embedding = calculator.get_embeddings_batched(
                    images, batch_size=batch_size
                ).mean(dim=0, keepdim=True)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                refs[(episode_idx, segment.subtask_id)] = {
                    "embedding": embedding.cpu(),
                    "ref_frame_start": next_start,
                    "ref_frame_end": next_start + reference_window,
                }
    return refs


def score_chunks(
    calculator: DINORewardCalculator,
    chunks: list[ChunkRecord],
    ref_info: dict[str, Any],
    *,
    batch_size: int,
    temperature: float,
    handoff_scale: float,
    step_penalty: float,
    success_reward: float,
    failure_reward: float,
    gamma: float,
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    rewards: list[float] = []
    details: list[dict[str, Any]] = []
    ref_embedding = ref_info.get("embedding")

    reach_curr_values: list[float | None] = [None] * len(chunks)
    reach_next_values: list[float | None] = [None] * len(chunks)
    dist_curr_values: list[float | None] = [None] * len(chunks)
    dist_next_values: list[float | None] = [None] * len(chunks)
    if ref_embedding is not None and chunks:
        images: list[Image.Image] = []
        for chunk in chunks:
            images.append(chunk.curr_head)
            images.append(chunk.next_head)
        embeddings = calculator.get_embeddings_batched(images, batch_size=batch_size)
        curr_embeddings = embeddings[0::2]
        next_embeddings = embeddings[1::2]
        ref_embedding = ref_embedding.to(curr_embeddings.device)
        curr_dist = ((curr_embeddings - ref_embedding) ** 2).sum(dim=1).cpu().numpy()
        next_dist = ((next_embeddings - ref_embedding) ** 2).sum(dim=1).cpu().numpy()
        reach_curr = np.exp(-curr_dist / float(temperature))
        reach_next = np.exp(-next_dist / float(temperature))
        reach_curr_values = [float(v) for v in reach_curr]
        reach_next_values = [float(v) for v in reach_next]
        dist_curr_values = [float(v) for v in curr_dist]
        dist_next_values = [float(v) for v in next_dist]

    for idx, chunk in enumerate(chunks):
        task_reward = 0.0
        if chunk.terminal:
            task_reward = success_reward if chunk.success else failure_reward
        handoff_reward = 0.0
        if reach_curr_values[idx] is not None and reach_next_values[idx] is not None:
            handoff_reward = handoff_scale * (
                reach_next_values[idx] - reach_curr_values[idx]
            )
        reward = float(step_penalty + handoff_reward + task_reward)
        rewards.append(reward)
        details.append(
            {
                "source": chunk.source,
                "original_episode_idx": chunk.original_episode_idx,
                "subtask_id": chunk.subtask_id,
                "subtask_name": chunk.subtask_name,
                "rollout_index": chunk.rollout_index,
                "chunk_index": chunk.chunk_index,
                "terminal": chunk.terminal,
                "success": chunk.success,
                "truncated": chunk.truncated,
                "real_action_steps": chunk.real_action_steps,
                "has_handoff_reference": ref_embedding is not None,
                "ref_frame_start": ref_info.get("ref_frame_start"),
                "ref_frame_end": ref_info.get("ref_frame_end"),
                "reach_curr": reach_curr_values[idx],
                "reach_next": reach_next_values[idx],
                "dino_l2_sq_curr": dist_curr_values[idx],
                "dino_l2_sq_next": dist_next_values[idx],
                "step_penalty": float(step_penalty),
                "handoff_reward": float(handoff_reward),
                "task_reward": float(task_reward),
                "reward": reward,
            }
        )

    returns = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = float(running)
    return rewards, returns, details


def write_episode(
    dataset: LeRobotDataset,
    chunks: list[ChunkRecord],
    *,
    dataset_episode_index: int,
    rewards: list[float],
    returns: list[float],
    details: list[dict[str, Any]],
    returns_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
) -> None:
    for frame_index, chunk in enumerate(chunks):
        dataset.add_frame(chunk.frame)
        returns_rows.append(
            {
                "episode_index": dataset_episode_index,
                "frame_index": frame_index,
                "return": returns[frame_index],
                "reward": rewards[frame_index],
                "prompt": str(chunk.frame.get("task", DEFAULT_STACK_THREE_PROMPT)),
            }
        )
        detail = {
            **details[frame_index],
            "episode_index": dataset_episode_index,
            "frame_index": frame_index,
            "return": returns[frame_index],
        }
        detail_rows.append(detail)
    dataset.save_episode()


def write_sidecars(
    dataset_path: Path,
    *,
    tag: str,
    returns_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = meta_dir / f"returns_{tag}.parquet"
    table = pa.table(
        {
            "episode_index": pa.array([int(row["episode_index"]) for row in returns_rows]),
            "frame_index": pa.array([int(row["frame_index"]) for row in returns_rows]),
            "return": pa.array(
                [float(row["return"]) for row in returns_rows], type=pa.float32()
            ),
            "reward": pa.array(
                [float(row["reward"]) for row in returns_rows], type=pa.float32()
            ),
            "prompt": pa.array([str(row["prompt"]) for row in returns_rows], type=pa.string()),
        }
    )
    pq.write_table(table, sidecar_path)

    details_path = meta_dir / f"chunk_rewards_{tag}.jsonl"
    with details_path.open("w") as f:
        for row in detail_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    returns_arr = np.asarray([row["return"] for row in returns_rows], dtype=np.float32)
    rewards_arr = np.asarray([row["reward"] for row in returns_rows], dtype=np.float32)
    stats = {
        "return": {
            "mean": float(returns_arr.mean()) if len(returns_arr) else 0.0,
            "std": float(returns_arr.std()) if len(returns_arr) else 0.0,
            "min": float(returns_arr.min()) if len(returns_arr) else 0.0,
            "max": float(returns_arr.max()) if len(returns_arr) else 0.0,
        },
        "reward": {
            "mean": float(rewards_arr.mean()) if len(rewards_arr) else 0.0,
            "std": float(rewards_arr.std()) if len(rewards_arr) else 0.0,
            "min": float(rewards_arr.min()) if len(rewards_arr) else 0.0,
            "max": float(rewards_arr.max()) if len(rewards_arr) else 0.0,
        },
    }

    stats_path = meta_dir / "stats.json"
    existing_stats = {}
    if stats_path.exists():
        with stats_path.open("r") as f:
            existing_stats = json.load(f)
    existing_stats["return"] = stats["return"]
    existing_stats["reward"] = stats["reward"]
    with stats_path.open("w") as f:
        json.dump(existing_stats, f, indent=2, sort_keys=True)

    info_path = meta_dir / "info.json"
    if info_path.exists():
        with info_path.open("r") as f:
            info = json.load(f)
        info.setdefault("features", {})
        info["features"]["return"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        info["features"]["reward"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        info["features"]["prompt"] = {
            "dtype": "string",
            "shape": [1],
            "names": None,
        }
        with info_path.open("w") as f:
            json.dump(info, f, indent=2)

    return {
        "dataset_path": str(dataset_path),
        "returns_sidecar": str(sidecar_path),
        "details_jsonl": str(details_path),
        "num_rows": len(returns_rows),
        "num_episodes": len({row["episode_index"] for row in returns_rows}),
        "stats": stats,
    }


def selected_segments(
    segments_by_episode: dict[int, list[SubtaskSegment]],
    limit_episodes: int | None,
) -> list[SubtaskSegment]:
    episode_ids = sorted(segments_by_episode)
    if limit_episodes is not None:
        episode_ids = episode_ids[:limit_episodes]
    segments: list[SubtaskSegment] = []
    for episode_idx in episode_ids:
        segments.extend(segments_by_episode[episode_idx])
    return segments


def process_expert_dataset(
    args: argparse.Namespace,
    dataset: LeRobotDataset,
    calculator: DINORewardCalculator,
    refs: dict[tuple[int, int], dict[str, Any]],
    segments: list[SubtaskSegment],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    returns_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    dataset_episode_index = 0
    for segment in tqdm(segments, desc="Expert chunks"):
        chunks = build_expert_chunks(
            args.dataset_root,
            segment,
            action_chunk_size=args.action_chunk_size,
            action_dim=args.action_dim,
            prompt_split=args.prompt_split,
            prompt_index=args.prompt_index,
        )
        rewards, returns, details = score_chunks(
            calculator,
            chunks,
            refs[(segment.episode_idx, segment.subtask_id)],
            batch_size=args.batch_size,
            temperature=args.temperature,
            handoff_scale=args.handoff_scale,
            step_penalty=args.step_penalty,
            success_reward=args.success_reward,
            failure_reward=args.failure_reward,
            gamma=args.gamma,
        )
        write_episode(
            dataset,
            chunks,
            dataset_episode_index=dataset_episode_index,
            rewards=rewards,
            returns=returns,
            details=details,
            returns_rows=returns_rows,
            detail_rows=detail_rows,
        )
        dataset_episode_index += 1
    return returns_rows, detail_rows


def process_rollout_dataset(
    args: argparse.Namespace,
    dataset: LeRobotDataset,
    calculator: DINORewardCalculator,
    refs: dict[tuple[int, int], dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    returns_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    dataset_episode_index = 0
    for entry in tqdm(manifest_rows, desc="Rollout chunks"):
        chunks = build_rollout_chunks(
            args.dataset_root,
            args.rollout_dir,
            entry,
            action_chunk_size=args.action_chunk_size,
            action_dim=args.action_dim,
            prompt_split=args.prompt_split,
            prompt_index=args.prompt_index,
        )
        key = (int(entry["episode_idx"]), int(entry["subtask_id"]))
        rewards, returns, details = score_chunks(
            calculator,
            chunks,
            refs[key],
            batch_size=args.batch_size,
            temperature=args.temperature,
            handoff_scale=args.handoff_scale,
            step_penalty=args.step_penalty,
            success_reward=args.success_reward,
            failure_reward=args.failure_reward,
            gamma=args.gamma,
        )
        write_episode(
            dataset,
            chunks,
            dataset_episode_index=dataset_episode_index,
            rewards=rewards,
            returns=returns,
            details=details,
            returns_rows=returns_rows,
            detail_rows=detail_rows,
        )
        dataset_episode_index += 1
    return returns_rows, detail_rows


def main() -> None:
    args = parse_args()
    output_root, expert_output, rollout_output = resolve_outputs(args)
    image_shape = (args.image_height, args.image_width, 3)

    maybe_remove_output(expert_output, args.overwrite)
    maybe_remove_output(rollout_output, args.overwrite)
    output_root.mkdir(parents=True, exist_ok=True)

    segments_by_episode = load_segments_by_episode(args.dataset_root)
    segments = selected_segments(segments_by_episode, args.limit_episodes)
    allowed_episodes = {segment.episode_idx for segment in segments}
    selected_segments_by_episode = {
        episode_idx: segments_by_episode[episode_idx]
        for episode_idx in sorted(allowed_episodes)
    }

    manifest_rows = load_rollout_manifest(args.rollout_dir)
    if args.limit_episodes is not None:
        manifest_rows = [
            row for row in manifest_rows if int(row["episode_idx"]) in allowed_episodes
        ]
    if args.limit_rollouts is not None:
        manifest_rows = manifest_rows[: args.limit_rollouts]

    calculator = DINORewardCalculator(
        device=args.device,
        model_name=str(args.dino_checkpoint),
        local_files_only=True,
    )
    refs = build_reference_embeddings(
        calculator,
        args.dataset_root,
        selected_segments_by_episode,
        reference_window=args.reference_window,
        batch_size=args.batch_size,
        action_dim=args.action_dim,
    )

    expert_dataset = create_dataset(
        expert_output,
        fps=args.fps,
        image_shape=image_shape,
        action_chunk_size=args.action_chunk_size,
        action_dim=args.action_dim,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )
    rollout_dataset = create_dataset(
        rollout_output,
        fps=args.fps,
        image_shape=image_shape,
        action_chunk_size=args.action_chunk_size,
        action_dim=args.action_dim,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    try:
        expert_returns, expert_details = process_expert_dataset(
            args, expert_dataset, calculator, refs, segments
        )
        rollout_returns, rollout_details = process_rollout_dataset(
            args, rollout_dataset, calculator, refs, manifest_rows
        )
    finally:
        close_dataset(expert_dataset)
        close_dataset(rollout_dataset)

    summary = {
        "tag": args.tag,
        "dataset_root": str(args.dataset_root),
        "rollout_dir": str(args.rollout_dir),
        "expert": write_sidecars(
            expert_output,
            tag=args.tag,
            returns_rows=expert_returns,
            detail_rows=expert_details,
        ),
        "rollout": write_sidecars(
            rollout_output,
            tag=args.tag,
            returns_rows=rollout_returns,
            detail_rows=rollout_details,
        ),
        "reward_config": {
            "temperature": args.temperature,
            "reference_window": args.reference_window,
            "handoff_scale": args.handoff_scale,
            "step_penalty": args.step_penalty,
            "success_reward": args.success_reward,
            "failure_reward": args.failure_reward,
            "gamma": args.gamma,
        },
    }
    summary_path = output_root / f"summary_{args.tag}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
