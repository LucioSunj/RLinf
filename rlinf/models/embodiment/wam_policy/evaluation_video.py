# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Read-only recording of the actual IDM prediction used by an eval chunk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.envs.libero.image_preprocessing import (
    OFFICIAL_LIBERO_CAMERA_RESIZE_MODE,
    prepare_libero_camera_batch,
)


def recording_chunk_dir(
    root: Path, *, task_id: int, trial_id: int, chunk_id: int
) -> Path:
    """Locate one task/trial/current-chunk recording without synthetic IDs."""

    return root / f"task-{task_id:04d}-trial-{trial_id:02d}" / f"chunk-{chunk_id:03d}"


def observation_rgb(obs: dict[str, Any], index: int) -> np.ndarray:
    """Use the policy's official 224-pixel crops for both observation cameras."""

    cameras = []
    for name in ("main_images", "wrist_images"):
        camera = prepare_libero_camera_batch(
            obs[name][index : index + 1],
            height=224,
            width=224,
            resize_mode=OFFICIAL_LIBERO_CAMERA_RESIZE_MODE,
        )
        cameras.append(camera[0].permute(1, 2, 0).numpy())
    return np.concatenate(cameras, axis=1)


class EvaluationPredictionRecorder:
    """Decode a detached copy after action sampling; never regenerate a future."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.latents: torch.Tensor | None = None

    def begin(self, *, env_obs: dict[str, Any], routes: torch.Tensor) -> None:
        """Bind capture to the one-environment recorded evaluation contract."""

        if routes.numel() != 1:
            raise ValueError("Recorded route-neutral evaluation requires batch one.")
        self.route = "idm" if int(routes.item()) == 1 else "uncond"
        self.identity = {
            name: int(torch.as_tensor(env_obs[field]).reshape(-1)[0])
            for name, field in {
                "task_id": "_fastwam_task_ids",
                "trial_id": "_fastwam_trial_ids",
                "reset_state_id": "_fastwam_reset_state_ids",
                "chunk_id": "_fastwam_recording_chunk_ids",
                "action_noise_seed": "_fastwam_action_noise_seeds",
                "idm_video_noise_seed": "_fastwam_idm_noise_seeds",
            }.items()
        }
        self.latents = None

    def observe(self, latents: torch.Tensor) -> None:
        """Copy the final conditioned prediction, without modifying the source."""

        if self.route != "idm" or self.latents is not None:
            raise RuntimeError("Expected exactly one prediction for an IDM chunk.")
        self.latents = latents.detach().clone()

    @torch.no_grad()
    def finish(self, *, actor: Any, actor_version: int, tiled: bool) -> None:
        """Export decoded RGB after the original action sampler has returned."""

        if (self.latents is not None) != (self.route == "idm"):
            raise RuntimeError("Recorded prediction and executed route disagree.")
        destination = recording_chunk_dir(
            self.root,
            task_id=self.identity["task_id"],
            trial_id=self.identity["trial_id"],
            chunk_id=self.identity["chunk_id"],
        )
        destination.mkdir(parents=True, exist_ok=True)
        frame_count = 0
        latent_shape = None
        if self.latents is not None:
            latent_shape = list(self.latents.shape)
            # The actor's native deterministic VAE decoder is the only extra
            # model work. It happens after action generation and draws no noise.
            frames = np.stack(
                [
                    np.asarray(frame)
                    for frame in actor._decode_latents(self.latents, tiled=tiled)
                ]
            )
            frame_count = int(frames.shape[0])
            if frames.dtype != np.uint8 or frames.shape != (9, 224, 448, 3):
                raise ValueError(f"Unexpected native prediction RGB: {frames.shape}.")
            with (destination / "prediction.npz").open("xb") as stream:
                np.savez_compressed(stream, frames=frames)
        self.latents = None
        metadata = {
            "schema": "fastwam-executed-prediction-video-v1",
            **self.identity,
            "actor_version": int(actor_version),
            "route": self.route,
            "prediction_frame_count": frame_count,
            "prediction_latent_shape": latent_shape,
            "prediction_source": (
                "copy_of_final_idm_latents_used_for_action_condition"
                if self.route == "idm"
                else "not_called"
            ),
            "frame_zero": "conditioned_current_frame_reconstruction",
            "future_preview": "frames_1_through_8_not_primitive_step_alignment",
        }
        with (destination / "inference.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")


__all__ = ["EvaluationPredictionRecorder", "observation_rgb", "recording_chunk_dir"]
