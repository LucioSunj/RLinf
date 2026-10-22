# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FR3 observation/action boundary around the shared route-neutral sampler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from .fr3_processing import Fr3Processor
from .route_neutral_online.runtime import RouteNeutralOnlineIDMTeacherLiberoRuntime

FR3_TASK_IDS = {"drawer": 0, "insert_tubes": 1}


@dataclass(frozen=True)
class Fr3ActionProtocol:
    """FR3 generates sixteen joint targets and predicts five video frames."""

    generation_horizon: int = 16
    execution_horizon: int = 16
    prediction_video_frames: int = 5
    reset_wait_steps: int = 0
    max_episode_steps: int = 1

    def __post_init__(self) -> None:
        if (
            self.generation_horizon,
            self.execution_horizon,
            self.prediction_video_frames,
        ) != (16, 16, 5):
            raise ValueError(
                "FR3 requires 16 generated/executed steps and 5 IDM frames."
            )
        if self.reset_wait_steps != 0:
            raise ValueError(
                "FR3 reset is an explicit operator operation, not simulator wait steps."
            )
        if (
            isinstance(self.max_episode_steps, bool)
            or int(self.max_episode_steps) != self.max_episode_steps
            or self.max_episode_steps < 1
        ):
            raise ValueError("FR3 max_episode_steps must be a positive integer.")


def fr3_observation_batch(observation: dict[str, Any]) -> dict[str, Any]:
    """Map a raw bridge observation to the shared policy's explicit batch keys."""
    if "states" in observation:
        result = dict(observation)
    else:
        result = {
            **observation,
            "states": torch.tensor(observation["observation.state"])[None],
            "main_images": torch.tensor(observation["observation.images.left"])[None],
            "wrist_images": torch.tensor(observation["observation.images.wrist"])[None],
            "task_descriptions": [observation["task"]],
        }
    states = torch.as_tensor(result["states"])
    if states.ndim != 2 or states.shape[1] != 8:
        raise ValueError("FR3 raw states must have shape [B,8] in joint/gripper order.")
    result["states"] = states
    result.setdefault("extra_view_images", None)
    return result


class Fr3RouteNeutralRuntime(RouteNeutralOnlineIDMTeacherLiberoRuntime):
    """Reuse Gate/Flow/teacher/replay while preserving native FR3 processing."""

    def __init__(
        self,
        *,
        training_config: str | Path | Any,
        easywam_root: str | Path,
        processor_stats_path: str | Path,
        task_name: str,
        task_prompt: str,
        action_dt: float = 0.1,
        generation_horizon: int = 16,
        execution_horizon: int = 16,
        num_video_frames: int = 5,
        reset_wait_steps: int = 0,
        processor: Any = None,
        **kwargs: Any,
    ) -> None:
        if processor is not None:
            raise ValueError(
                "FR3 processor is constructed from its native training config."
            )
        if task_name not in FR3_TASK_IDS or not task_prompt.strip():
            raise ValueError(
                "FR3 runtime requires a supported task and its complete prompt."
            )
        if not math.isfinite(action_dt) or action_dt <= 0:
            raise ValueError("FR3 action_dt must be positive and finite.")
        config = (
            OmegaConf.load(training_config)
            if isinstance(training_config, (str, Path))
            else training_config
        )
        self.fr3_processor = Fr3Processor(config, processor_stats_path, easywam_root)
        self.task_name, self.task_prompt = task_name, task_prompt
        self.task_id, self.action_dt = FR3_TASK_IDS[task_name], float(action_dt)
        super().__init__(
            processor=None,
            processor_stats_path=None,
            generation_horizon=generation_horizon,
            execution_horizon=execution_horizon,
            num_video_frames=num_video_frames,
            reset_wait_steps=reset_wait_steps,
            **kwargs,
        )
        self.processor_stats_path = str(processor_stats_path)
        if (
            int(self.actor.action_expert.action_dim) != 8
            or int(self.actor.proprio_dim) != 8
        ):
            raise ValueError("FR3 requires action_dim=8 and proprio_dim=8.")
        if getattr(self.actor, "text_padding", None) != "masked":
            raise ValueError(
                "FR3 requires masked text context from the corresponding parent."
            )

    @staticmethod
    def _make_action_protocol(**kwargs: Any) -> Fr3ActionProtocol:
        return Fr3ActionProtocol(**kwargs)

    def _validate_camera_contract(self) -> None:
        if self.camera_concat != "horizontal":
            raise ValueError("FR3 requires left/wrist horizontal camera concatenation.")

    def _rows(self, env_obs: dict[str, Any]) -> list[dict[str, Any]]:
        states = torch.as_tensor(env_obs["states"]).detach().cpu().numpy()
        main = torch.as_tensor(env_obs["main_images"]).detach().cpu().numpy()
        wrist = torch.as_tensor(env_obs["wrist_images"]).detach().cpu().numpy()
        prompts = env_obs["task_descriptions"]
        if len(prompts) != len(states) or any(
            prompt != self.task_prompt for prompt in prompts
        ):
            raise ValueError(
                "FR3 observations must match the task-bound complete prompt."
            )
        return [
            {
                "observation.state": state,
                "observation.images.left": left,
                "observation.images.wrist": right,
                "task": prompt,
            }
            for state, left, right, prompt in zip(
                states, main, wrist, prompts, strict=True
            )
        ]

    def _model_images(self, env_obs: dict[str, Any]) -> torch.Tensor:
        prepared = self.fr3_processor.prepare_batch(self._rows(env_obs))
        return prepared["input_image"].to(device=self.device, dtype=self.dtype)

    def _normalized_proprio(self, states: torch.Tensor) -> torch.Tensor:
        raw = torch.as_tensor(states).detach().to(device="cpu", dtype=torch.float32)
        if raw.ndim != 2 or raw.shape[1] != 8 or not torch.isfinite(raw).all():
            raise ValueError("FR3 proprio requires finite raw states [B,8].")
        if bool(((raw[:, 7] < 0) | (raw[:, 7] > 1)).any()):
            raise ValueError("FR3 gripper state must be an absolute label in [0,1].")
        processor = self.fr3_processor.processor
        batch = processor.action_state_transform({"state": {"default": raw.clone()}})
        batch = processor.normalizer.forward(batch)
        return batch["state"]["default"].to(device=self.device, dtype=self.dtype)

    def _denormalize_action_stages(
        self, actions: torch.Tensor, *, env_obs: dict[str, Any]
    ) -> tuple[torch.Tensor, None]:
        raw_state = torch.as_tensor(env_obs["states"]).detach().cpu().float()[:, None]
        decoded = self.fr3_processor.decode_batch(actions, raw_state)
        return torch.from_numpy(np.asarray(decoded)).to(device=actions.device), None

    def clear_session(self) -> None:
        """Discard derived history and language cache at session/model boundaries."""
        state = self.physical_history.state_dict()
        state["states"] = {}
        self.physical_history.load_state_dict(state)
        self._evaluation_text_context = None
        self._text_context_cache.clear()
