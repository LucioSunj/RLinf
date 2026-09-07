# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Opt-in chunk-aligned RGB capture alongside the unchanged eval collector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.models.embodiment.wam_policy.evaluation_video import (
    observation_rgb,
    recording_chunk_dir,
)
from rlinf.runners.fastwam_libero_eval_collector import (
    EvaluationIdentityBatch,
    FastWAMLiberoEvalCollector,
)


class FastWAMLiberoVideoCollector(FastWAMLiberoEvalCollector):
    """Save pre-decision and actual post-step frames, including terminal frames."""

    def __init__(self, *, video_recording_dir: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.resume:
            raise ValueError("Video recording uses a fresh evaluation identity.")
        self.video_recording_dir = Path(video_recording_dir)
        self._video_snapshots: dict[int, EvaluationIdentityBatch] = {}

    def augment_rollout_input(
        self, data: dict[str, Any], snapshot: EvaluationIdentityBatch
    ) -> dict[str, Any]:
        """Save current observation without changing policy inputs or seeds."""

        result = super().augment_rollout_input(data, snapshot)
        if len(snapshot.slots) != 1 or snapshot.slots[0].entry is None:
            raise ValueError("Video recording requires one active ledger slot.")
        slot = snapshot.slots[0]
        entry = slot.entry
        self._video_snapshots[slot.stage_id] = snapshot
        destination = recording_chunk_dir(
            self.video_recording_dir,
            task_id=int(entry["task_id"]),
            trial_id=int(entry["trial_id"]),
            chunk_id=slot.chunk_id,
        )
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "observation.npz").open("xb") as stream:
            np.savez_compressed(stream, frame=observation_rgb(result["obs"], 0))
        for name in ("task_id", "trial_id", "reset_state_id"):
            result["obs"][f"_fastwam_{name}s"] = torch.tensor(
                [int(entry[name])], dtype=torch.long
            )
        result["obs"]["_fastwam_recording_chunk_ids"] = torch.tensor(
            [slot.chunk_id], dtype=torch.long
        )
        return result

    def record_step_observations(self, stage_id: int, chunk_result: tuple) -> None:
        """Capture every submitted primitive frame, not a post-reset substitute."""

        slot = self._video_snapshots[stage_id].slots[0]
        entry = slot.entry
        observations, rewards, terminations, truncations, infos = chunk_result
        frames = []
        for index, observation in enumerate(observations):
            info = infos[index]
            # LIBERO auto-reset replaces the final list element; restore its
            # terminal observation for recording without changing the result.
            if isinstance(info, dict):
                final = info.get("final_observation", info.get("final_obs"))
                if final is not None:
                    observation = final
            frames.append(observation_rgb(observation, slot.local_env_index))
        destination = recording_chunk_dir(
            self.video_recording_dir,
            task_id=int(entry["task_id"]),
            trial_id=int(entry["trial_id"]),
            chunk_id=slot.chunk_id,
        )
        with (destination / "execution.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                frames=np.stack(frames),
                rewards=torch.as_tensor(rewards).cpu().numpy(),
                terminations=torch.as_tensor(terminations).cpu().numpy(),
                truncations=torch.as_tensor(truncations).cpu().numpy(),
            )
