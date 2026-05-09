#!/usr/bin/env python3
"""Compute DINOv3 handoff reward sidecars for RoboTwin stack-three data.

The reference for subtask i is the same expert episode's next-subtask start
observation. Rollout rows use their manifest episode_idx/subtask_id to pair with
that expert reference.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dino_reward_calculator import DEFAULT_DINOV3_CHECKPOINT, DINORewardCalculator


DEFAULT_DATASET_ROOT = Path(
    "/home/lsk/long-horizon-manipulation/dataset/robotwin/stack_block_three/"
    "aloha-agilex_clean_50"
)
DEFAULT_ROLLOUT_DIR = (
    DEFAULT_DATASET_ROOT
    / "subtask_rollouts"
    / "pi05_19000_rlinf_all_episodes_n1_20260509"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--dino-checkpoint", type=Path, default=Path(DEFAULT_DINOV3_CHECKPOINT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--handoff-scale", type=float, default=1.0)
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--failure-reward", type=float, default=0.0)
    parser.add_argument("--reference-window", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cameras", nargs="+", default=["head_camera"])
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--limit-rollouts", type=int, default=None)
    return parser.parse_args()


def load_segments(dataset_root: Path) -> dict[int, list[dict[str, Any]]]:
    manifest_path = dataset_root / "subtask_segments.json"
    with manifest_path.open("r") as f:
        payload = json.load(f)

    result: dict[int, list[dict[str, Any]]] = {}
    for episode_key, episode in payload["episodes"].items():
        if not episode.get("valid", False):
            continue
        match = re.search(r"(\d+)$", str(episode_key))
        if match is None:
            raise ValueError(f"Cannot parse episode index from key {episode_key!r}")
        episode_idx = int(match.group(1))
        result[episode_idx] = sorted(
            episode["subtasks"], key=lambda item: int(item["subtask_id"])
        )
    return result


def decode_hdf5_rgb(value: np.bytes_ | bytes) -> Image.Image:
    raw = bytes(value)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def load_expert_images(
    dataset_root: Path,
    episode_idx: int,
    frame_idx: int,
    cameras: list[str],
) -> list[Image.Image]:
    h5_path = dataset_root / "data" / f"episode{episode_idx}.hdf5"
    images: list[Image.Image] = []
    with h5py.File(h5_path, "r") as h5:
        for camera in cameras:
            key = f"observation/{camera}/rgb"
            if key not in h5:
                raise KeyError(f"{key} not found in {h5_path}")
            max_frame = len(h5[key]) - 1
            images.append(decode_hdf5_rgb(h5[key][min(frame_idx, max_frame)]))
    return images


def load_rollout_terminal_images(path: Path, cameras: list[str]) -> list[Image.Image]:
    payload = torch.load(path, map_location="cpu")
    next_obs = payload["next_obs"]
    images: list[Image.Image] = []
    for camera in cameras:
        if camera == "head_camera":
            arr = next_obs["main_images"][-1, 0].numpy()
        elif camera == "left_camera":
            arr = next_obs["wrist_images"][-1, 0, 0].numpy()
        elif camera == "right_camera":
            arr = next_obs["wrist_images"][-1, 0, 1].numpy()
        else:
            raise ValueError(
                f"Rollout replay only stores head_camera/left_camera/right_camera, got {camera!r}"
            )
        images.append(Image.fromarray(arr.astype(np.uint8)).convert("RGB"))
    return images


def prototype_embedding(
    calculator: DINORewardCalculator,
    images: list[Image.Image],
    batch_size: int,
) -> torch.Tensor:
    embeddings = calculator.get_embeddings_batched(images, batch_size=batch_size)
    prototype = embeddings.mean(dim=0, keepdim=True)
    return F.normalize(prototype, p=2, dim=1)


def reward_to_ref(
    cur_embedding: torch.Tensor,
    ref_embedding: torch.Tensor | None,
    temperature: float,
) -> tuple[float | None, float | None]:
    if ref_embedding is None:
        return None, None
    dist = float(((cur_embedding - ref_embedding) ** 2).sum().item())
    reward = float(np.exp(-dist / temperature))
    return reward, dist


def build_reference_embeddings(
    calculator: DINORewardCalculator,
    dataset_root: Path,
    segments: dict[int, list[dict[str, Any]]],
    cameras: list[str],
    reference_window: int,
    batch_size: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    refs: dict[tuple[int, int], dict[str, Any]] = {}
    for episode_idx, subtasks in segments.items():
        for subtask in subtasks:
            subtask_id = int(subtask["subtask_id"])
            if subtask_id >= len(subtasks) - 1:
                refs[(episode_idx, subtask_id)] = {
                    "embedding": None,
                    "ref_frame_start": None,
                    "ref_frame_end": None,
                }
                continue

            next_start = int(subtasks[subtask_id + 1]["start_step"])
            images: list[Image.Image] = []
            for offset in range(reference_window):
                images.extend(
                    load_expert_images(
                        dataset_root,
                        episode_idx,
                        next_start + offset,
                        cameras,
                    )
                )
            refs[(episode_idx, subtask_id)] = {
                "embedding": prototype_embedding(calculator, images, batch_size),
                "ref_frame_start": next_start,
                "ref_frame_end": next_start + reference_window,
            }
    return refs


def make_row(
    *,
    source: str,
    episode_idx: int,
    subtask_id: int,
    subtask_name: str,
    terminal_frame: int | None,
    rollout_index: int | None,
    success: bool,
    truncated: bool,
    dino_reward: float | None,
    dino_l2_sq: float | None,
    task_reward: float,
    handoff_scale: float,
    ref_info: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    handoff_reward = 0.0 if dino_reward is None else float(dino_reward)
    total_reward = float(task_reward + handoff_scale * handoff_reward)
    row = {
        "source": source,
        "episode_idx": int(episode_idx),
        "subtask_id": int(subtask_id),
        "subtask_name": subtask_name,
        "terminal_frame": None if terminal_frame is None else int(terminal_frame),
        "rollout_index": rollout_index,
        "success": bool(success),
        "truncated": bool(truncated),
        "has_handoff_reference": dino_reward is not None,
        "ref_frame_start": ref_info.get("ref_frame_start"),
        "ref_frame_end": ref_info.get("ref_frame_end"),
        "dino_terminal_reward": dino_reward,
        "dino_l2_sq": dino_l2_sq,
        "task_reward": float(task_reward),
        "handoff_reward": handoff_reward,
        "total_reward": total_reward,
    }
    if extra:
        row.update(extra)
    return row


def compute_expert_rows(
    calculator: DINORewardCalculator,
    dataset_root: Path,
    segments: dict[int, list[dict[str, Any]]],
    refs: dict[tuple[int, int], dict[str, Any]],
    cameras: list[str],
    batch_size: int,
    temperature: float,
    success_reward: float,
    handoff_scale: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_idx, subtasks in segments.items():
        for subtask in subtasks:
            subtask_id = int(subtask["subtask_id"])
            terminal_frame = int(subtask["end_step"])
            images = load_expert_images(dataset_root, episode_idx, terminal_frame, cameras)
            cur_embedding = prototype_embedding(calculator, images, batch_size)
            ref_info = refs[(episode_idx, subtask_id)]
            dino_reward, dino_l2_sq = reward_to_ref(
                cur_embedding, ref_info["embedding"], temperature
            )
            rows.append(
                make_row(
                    source="expert",
                    episode_idx=episode_idx,
                    subtask_id=subtask_id,
                    subtask_name=str(subtask["name"]),
                    terminal_frame=terminal_frame,
                    rollout_index=None,
                    success=True,
                    truncated=False,
                    dino_reward=dino_reward,
                    dino_l2_sq=dino_l2_sq,
                    task_reward=success_reward,
                    handoff_scale=handoff_scale,
                    ref_info=ref_info,
                )
            )
    return rows


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


def compute_rollout_rows(
    calculator: DINORewardCalculator,
    rollout_dir: Path,
    manifest_rows: list[dict[str, Any]],
    refs: dict[tuple[int, int], dict[str, Any]],
    cameras: list[str],
    batch_size: int,
    temperature: float,
    success_reward: float,
    failure_reward: float,
    handoff_scale: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in manifest_rows:
        episode_idx = int(entry["episode_idx"])
        subtask_id = int(entry["subtask_id"])
        path = rollout_path(rollout_dir, int(entry["rollout_index"]))
        images = load_rollout_terminal_images(path, cameras)
        cur_embedding = prototype_embedding(calculator, images, batch_size)
        ref_info = refs[(episode_idx, subtask_id)]
        dino_reward, dino_l2_sq = reward_to_ref(
            cur_embedding, ref_info["embedding"], temperature
        )
        success = bool(entry.get("success", False))
        task_reward = success_reward if success else failure_reward
        rows.append(
            make_row(
                source="rollout",
                episode_idx=episode_idx,
                subtask_id=subtask_id,
                subtask_name=str(entry.get("subtask_name", "")),
                terminal_frame=None,
                rollout_index=int(entry["rollout_index"]),
                success=success,
                truncated=bool(entry.get("truncated", False)),
                dino_reward=dino_reward,
                dino_l2_sq=dino_l2_sq,
                task_reward=task_reward,
                handoff_scale=handoff_scale,
                ref_info=ref_info,
                extra={
                    "elapsed_steps": int(entry.get("elapsed_steps", 0)),
                    "max_steps": int(entry.get("max_steps", 0)),
                    "start_step": int(entry.get("start_step", 0)),
                    "end_step": int(entry.get("end_step", 0)),
                },
            )
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_rows": len(rows)}
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        rewards = [
            row["dino_terminal_reward"]
            for row in source_rows
            if row["dino_terminal_reward"] is not None
        ]
        totals = [row["total_reward"] for row in source_rows]
        summary[source] = {
            "num_rows": len(source_rows),
            "num_handoff_rows": len(rewards),
            "num_success": int(sum(bool(row["success"]) for row in source_rows)),
            "dino_reward_mean": None if not rewards else float(np.mean(rewards)),
            "dino_reward_min": None if not rewards else float(np.min(rewards)),
            "dino_reward_max": None if not rewards else float(np.max(rewards)),
            "total_reward_mean": float(np.mean(totals)) if totals else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    output_jsonl = args.output_jsonl
    if output_jsonl is None:
        reward_dir = args.rollout_dir / "rewards"
        output_jsonl = reward_dir / "dino_handoff_terminal_rewards.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    segments = load_segments(args.dataset_root)
    if args.limit_episodes is not None:
        selected = sorted(segments)[: args.limit_episodes]
        segments = {episode_idx: segments[episode_idx] for episode_idx in selected}

    manifest_rows = load_rollout_manifest(args.rollout_dir)
    if args.limit_episodes is not None:
        allowed = set(segments)
        manifest_rows = [
            row for row in manifest_rows if int(row["episode_idx"]) in allowed
        ]
    if args.limit_rollouts is not None:
        manifest_rows = manifest_rows[: args.limit_rollouts]

    calculator = DINORewardCalculator(
        device=args.device,
        model_name=str(args.dino_checkpoint),
        local_files_only=True,
    )

    refs = build_reference_embeddings(
        calculator=calculator,
        dataset_root=args.dataset_root,
        segments=segments,
        cameras=list(args.cameras),
        reference_window=int(args.reference_window),
        batch_size=int(args.batch_size),
    )
    rows = compute_expert_rows(
        calculator=calculator,
        dataset_root=args.dataset_root,
        segments=segments,
        refs=refs,
        cameras=list(args.cameras),
        batch_size=int(args.batch_size),
        temperature=float(args.temperature),
        success_reward=float(args.success_reward),
        handoff_scale=float(args.handoff_scale),
    )
    rows.extend(
        compute_rollout_rows(
            calculator=calculator,
            rollout_dir=args.rollout_dir,
            manifest_rows=manifest_rows,
            refs=refs,
            cameras=list(args.cameras),
            batch_size=int(args.batch_size),
            temperature=float(args.temperature),
            success_reward=float(args.success_reward),
            failure_reward=float(args.failure_reward),
            handoff_scale=float(args.handoff_scale),
        )
    )

    with output_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize(rows)
    summary_path = output_jsonl.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps({"output": str(output_jsonl), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
