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

"""Successful demonstrations, episode splits and one shared temporal index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .contracts import (
    ActionNormalizer,
    action_tensor,
    canonical_quaternions,
    valid_action_mask,
)
from .runtime import ContextSnapshot, HistoryBlock

TASKS = {
    "grasp_place": {
        "instruction": "Pick up the object and place it in the target container.",
        "max_steps": 320,
    },
    "occluded_drawer": {
        "instruction": "Open the drawer, retrieve the occluded object, and place it in the target container.",
        "max_steps": 640,
    },
    "peg_insertion": {
        "instruction": "Grasp the peg and insert it into the fixed hole.",
        "max_steps": 320,
    },
}


def write_episode(
    directory,
    task,
    episode_id,
    observations,
    actions,
    action_timestamps,
    *,
    success,
    reset_metadata,
    submitted_commands=None,
):
    """Store the initial frame and one feedback frame per submitted command."""
    if task not in TASKS or len(observations) != len(actions) + 1:
        raise ValueError(
            "An episode needs a known task and T+1 observations for T actions."
        )
    path = Path(directory) / task / f"episode_{episode_id:06d}"
    path.mkdir(parents=True, exist_ok=False)
    values = {
        "actions": canonical_quaternions(np.asarray(actions)),
        "states": np.stack([o["state"] for o in observations]),
        "observation_timestamps": np.array(
            [o["timestamp"] for o in observations], dtype=np.float64
        ),
        "action_timestamps": np.asarray(action_timestamps, dtype=np.float64),
        **{
            key: np.stack([o["images"][key] for o in observations])
            for key in ("external", "wrist")
        },
    }
    if submitted_commands is not None:
        values["submitted_commands"] = np.asarray(submitted_commands, dtype=np.float32)
    np.savez_compressed(path / "episode.npz", **values)
    metadata = {
        "task": task,
        "episode_id": episode_id,
        "success": bool(success),
        "instruction": TASKS[task]["instruction"],
        "reset": reset_metadata,
        "action_hz": 20,
        "action_format": "absolute_base_tcp_xyzw_gripper_metres",
        "image_format": "RGB",
        "state_format": "joint7_tcp_xyzw7_gripper1",
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def align_episode(path, hz=20):
    """Sample both cameras and feedback with the same nearest observed-frame IDs."""
    with np.load(Path(path) / "episode.npz", allow_pickle=False) as handle:
        raw = {k: handle[k] for k in handle.files}
    times = raw["observation_timestamps"]
    if len(times) != len(raw["actions"]) + 1 or len(raw["action_timestamps"]) != len(
        raw["actions"]
    ):
        raise ValueError("Demonstration timestamps do not match the submitted actions.")
    if len(times) < 17 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError(
            "A demonstration needs increasing timestamps and at least 16 actions."
        )
    if raw["states"].shape != (len(times), 15) or not np.isfinite(raw["states"]).all():
        raise ValueError("Demonstration measured states must have 15 finite channels.")
    for key in ("external", "wrist"):
        if (
            raw[key].dtype != np.uint8
            or len(raw[key]) != len(times)
            or raw[key].shape[-1] != 3
        ):
            raise ValueError(
                "The two camera streams must contain matched RGB uint8 frames."
            )
    count = int(np.floor((times[-1] - times[0]) * hz + 1e-6))
    count -= count % 4
    grid = times[0] + np.arange(count + 1) / hz
    right = np.searchsorted(times, grid).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    ids = np.where(
        np.abs(times[left] - grid) <= np.abs(times[right] - grid), left, right
    )
    ids[0] = 0
    if np.any(ids[1:] == 0) or count < 16:
        raise ValueError(
            "Recording does not cover an aligned 16-action training window."
        )
    result = {key: raw[key][ids] for key in ("external", "wrist", "states")}
    result["actions"] = canonical_quaternions(raw["actions"][ids[1:] - 1])
    result["states"][:, 7:14] = canonical_quaternions(result["states"][:, 7:14])
    result.update(frame_ids=ids, timestamps=grid, recorded_timestamps=times[ids])
    return result


def prepare_split(raw_root, output, *, per_task=100, validation_count=20, seed=42):
    """Split whole successful episodes, then fit every statistic on training only."""
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(
            "A prepared demonstration split is immutable; use a new output directory."
        )
    generator = np.random.default_rng(seed)
    rows, train_actions, train_states = [], [], []
    for task in TASKS:
        paths = sorted((Path(raw_root) / task).glob("episode_*/metadata.json"))
        successful = [
            (p.parent, json.loads(p.read_text()))
            for p in paths
            if json.loads(p.read_text())["success"]
        ]
        if len(successful) != per_task:
            raise ValueError(
                f"{task} needs exactly {per_task} successful demonstrations; found {len(successful)}."
            )
        validation = set(generator.permutation(per_task)[:validation_count].tolist())
        for i, (path, metadata) in enumerate(successful):
            aligned = align_episode(path)
            split = "validation" if i in validation else "train"
            if split == "train":
                train_actions.append(torch.from_numpy(aligned["actions"]))
                train_states.append(torch.from_numpy(aligned["states"]).float())
            rows.append(
                {
                    "task": task,
                    "episode_id": metadata["episode_id"],
                    "split": split,
                    "raw_path": str(path.resolve()),
                    "instruction": metadata["instruction"],
                }
            )
    actions, states = torch.cat(train_actions), torch.cat(train_states)
    q01, q99 = torch.zeros(30), torch.zeros(30)
    q01[[0, 1, 2, 3, 4, 5, 6, 28]] = torch.quantile(actions, 0.01, dim=0)
    q99[[0, 1, 2, 3, 4, 5, 6, 28]] = torch.quantile(actions, 0.99, dim=0)
    normalizer = ActionNormalizer(
        q01,
        q99,
        torch.quantile(states, 0.01, dim=0),
        torch.quantile(states, 0.99, dim=0),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "normalizer.json").write_text(
        json.dumps(normalizer.to_dict(), indent=2) + "\n"
    )
    # The frozen Pi0.5 prefix uses the new 15D state statistics through its
    # existing image/language preprocessing; no LIBERO normalization is loaded.
    statistics = {}
    for key, value in (("state", states), ("actions", actions)):
        statistics[key] = {
            "mean": value.mean(0).tolist(),
            "std": value.std(0, unbiased=False).tolist(),
            "q01": torch.quantile(value, 0.01, dim=0).tolist(),
            "q99": torch.quantile(value, 0.99, dim=0).tolist(),
        }
    (output / "critic_norm").mkdir(exist_ok=True)
    (output / "critic_norm/norm_stats.json").write_text(
        json.dumps({"norm_stats": statistics}, indent=2) + "\n"
    )
    manifest = {
        "model_type": "lingbot_va_route_neutral",
        "seed": seed,
        "hz": 20,
        "per_task": per_task,
        "validation_count": validation_count,
        "episodes": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_lerobot(prepared, output):
    """Write the native loader's LeRobot 2.1 episodes and action_config fields."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    prepared, output = Path(prepared), Path(output)
    manifest = json.loads((prepared / "manifest.json").read_text())
    features = {
        f"observation.images.{key}": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        }
        for key in ("external", "wrist")
    }
    features.update(
        action={
            "dtype": "float32",
            "shape": (8,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        },
        **{
            "observation.state": {
                "dtype": "float32",
                "shape": (15,),
                "names": [f"state_{i}" for i in range(15)],
            },
            "observation.recorded_timestamp": {
                "dtype": "float64",
                "shape": (1,),
                "names": None,
            },
        },
    )
    dataset = LeRobotDataset.create(
        repo_id="local/lingbot_va_franka",
        fps=20,
        root=output,
        robot_type="franka",
        features=features,
        use_videos=True,
        video_backend="pyav",
    )
    import cv2

    metadata_rows = []
    for row in manifest["episodes"]:
        episode = align_episode(row["raw_path"])
        for index in range(len(episode["states"])):
            frame = {
                f"observation.images.{key}": cv2.resize(episode[key][index], (256, 256))
                for key in ("external", "wrist")
            }
            frame.update(
                action=episode["actions"][index]
                if index < len(episode["actions"])
                else np.zeros(8, dtype=np.float32),
                **{
                    "observation.state": episode["states"][index],
                    "observation.recorded_timestamp": np.array(
                        [episode["recorded_timestamps"][index]]
                    ),
                },
            )
            dataset.add_frame(frame, task=row["instruction"], timestamp=index / 20)
        dataset.save_episode()
        metadata_rows.append(
            {
                "split": row["split"],
                "source_task": row["task"],
                "source_episode_id": row["episode_id"],
                "action_config": [
                    {"start_frame": 0, "end_frame": len(episode["states"]) - 1}
                ],
            }
        )
    path = output / "meta/episodes.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(
        "".join(
            json.dumps({**line, **extra}) + "\n"
            for line, extra in zip(lines, metadata_rows)
        )
    )
    return output


@torch.no_grad()
def precompute(prepared, encoder, normalizer):
    """Frozen streaming VAE and T5; cache all cameras/actions with one index grid."""
    prepared = Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    for row in manifest["episodes"]:
        episode = align_episode(row["raw_path"])
        encoder.reset()
        observations = [
            {"images": {key: episode[key][i] for key in ("external", "wrist")}}
            for i in range(len(episode["states"]))
        ]
        latent_blocks = [encoder.images(observations[:1]).cpu()]
        for start in range(1, len(observations), 4):
            latent_blocks.append(encoder.images(observations[start : start + 4]).cpu())
        latents = torch.cat(latent_blocks, dim=2)
        normalized = action_tensor(
            normalizer.encode(torch.from_numpy(episode["actions"]))
        )
        actions = torch.cat((torch.zeros_like(normalized[:, :, :1]), normalized), dim=2)
        if actions.shape[2] != latents.shape[2]:
            raise ValueError(
                "The VAE temporal compression does not match four actions/frame."
            )
        text, text_mask = encoder.text(row["instruction"])
        negative, _ = encoder.text("")
        path = (
            prepared / "latents" / row["task"] / f"episode_{row['episode_id']:06d}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latents": latents,
                "actions": actions,
                "action_mask": valid_action_mask(actions, True),
                "text": text,
                "text_mask": text_mask,
                "negative_text": negative,
                "frame_ids": torch.from_numpy(episode["frame_ids"][::4]),
                "source": row,
                "base_model_path": getattr(encoder, "base_model_path", ""),
            },
            path,
        )


class DemonstrationDataset(torch.utils.data.Dataset):
    """Episode-level windows; BC uses the same executed eight-action prefix grid."""

    def __init__(self, prepared, split):
        self.root = Path(prepared)
        self.rows = [
            r
            for r in json.loads((self.root / "manifest.json").read_text())["episodes"]
            if r["split"] == split
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (
            self.root / "latents" / row["task"] / f"episode_{row['episode_id']:06d}.pt"
        )
        return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def sft_window(episode, seed, max_frames=64):
        frames = episode["latents"].shape[2]
        generator = torch.Generator().manual_seed(seed)
        start = int(
            torch.randint(max(1, frames - max_frames + 1), (), generator=generator)
        )
        end = min(start + max_frames, frames)
        return {
            **{
                key: episode[key][:, :, start:end]
                for key in ("latents", "actions", "action_mask")
            },
            "text": episode["text"],
            "negative_text": episode["negative_text"],
            "frame_ids": episode["frame_ids"][start:end],
            "start": start,
        }

    @staticmethod
    def bc_window(episode, seed):
        frames = episode["latents"].shape[2]
        starts = [0] + list(range(3, frames - 3, 2))
        start = starts[
            int(
                torch.randint(
                    len(starts), (), generator=torch.Generator().manual_seed(seed)
                )
            )
        ]
        blocks = []
        if start:
            for block_start in [0] + list(range(3, start, 2)):
                end = 3 if block_start == 0 else block_start + 2
                blocks.append(
                    HistoryBlock(
                        episode["latents"][:, :, block_start:end],
                        episode["actions"][:, :, block_start:end],
                        block_start,
                    )
                )
        context = ContextSnapshot(
            episode["text"],
            episode["negative_text"],
            episode["latents"][:, :, :1],
            tuple(blocks),
            start,
        )
        return context, episode["actions"][:, :, start : start + 4]
