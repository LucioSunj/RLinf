# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Physical I/O and deferred inference over the existing FastWAM sampler."""

from __future__ import annotations

import time

import numpy as np
import torch

from rlinf.envs.pad_realworld.observation import ObservationAdapter
from rlinf.models.embodiment.wam_policy.libero_runtime import (
    LiberoFastWAMRuntime,
    _align_linear_normalizer,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.runtime import (
    RouteNeutralOnlineIDMTeacherLiberoRuntime,
    RouteNeutralTrainableChunkSample,
)


class RealRobotFastWAMRuntime(RouteNeutralOnlineIDMTeacherLiberoRuntime):
    """Keep model math while replacing simulator observations and execution units."""

    capture_action_gate_snapshots = False

    def __init__(self, *, observation: dict, action_protocol, **kwargs) -> None:
        super().__init__(action_protocol=action_protocol, **kwargs)
        self.observation_adapter = ObservationAdapter.from_config(observation)
        self._encoded = None
        self.last_sample = None
        self.last_seeds = None
        self.timings = {}
        self._preparing_gate = False
        self.calls = dict.fromkeys(("feature", "teacher", "action", "video"), 0)

    def prepare_snapshot(self, snapshot) -> dict:
        """Preprocess the saved observation once, preserving explicit view order."""
        started = time.perf_counter()
        images = np.concatenate(self.observation_adapter.images(snapshot), axis=1)
        state = self.observation_adapter.canonical_state(snapshot.state)
        self._encoded = None
        self.timings = {}
        env_obs = {
            "real_images": torch.from_numpy(images).unsqueeze(0),
            "states": torch.from_numpy(state).unsqueeze(0).to(self.device),
            "task_descriptions": [snapshot.instruction],
        }
        self._encode_condition(env_obs)
        self.synchronize()
        self.timings["image_preprocess"] = time.perf_counter() - started
        return env_obs

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _model_images(self, env_obs: dict) -> torch.Tensor:
        # The data recorder and runtime use the identical RGB/crop/resize adapter.
        return (
            env_obs["real_images"]
            .to(device=self.device, dtype=self.dtype)
            .permute(0, 3, 1, 2)
            .unsqueeze(2)
            / 127.5
            - 1.0
        )

    def _normalized_proprio(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(device=self.device, dtype=torch.float32)
        if self.processor is not None:
            key = self.processor.shape_meta["state"][0]["key"]
            normalizer = self.processor.normalizer.normalizers["state"][key]
            _align_linear_normalizer(normalizer, states)
            states = normalizer.forward(states)
        return states.to(dtype=self.dtype)

    def _denormalize_action_stages(self, actions, *, env_obs):
        actions = actions.float()
        if self.processor is not None:
            key = self.processor.shape_meta["action"][0]["key"]
            normalizer = self.processor.normalizer.normalizers["action"][key]
            _align_linear_normalizer(normalizer, actions)
            actions = normalizer.backward(actions)
        # No LIBERO gripper flip or conversion: positive means more open.
        return actions, None

    def _encode_condition(self, env_obs):
        if self._encoded is None:
            self._encoded = super()._encode_condition(env_obs)
        return self._encoded

    def prepare_route_neutral_gate_features(self, *, env_obs):
        self.calls["feature"] += 1
        self.synchronize()
        started = time.perf_counter()
        self._preparing_gate = True
        try:
            result = super().prepare_route_neutral_gate_features(env_obs=env_obs)
        finally:
            self._preparing_gate = False
        self.synchronize()
        self.timings["current_only_feature"] = time.perf_counter() - started
        return result

    def _prepare_action_condition(self, **kwargs):
        from fastwam.adapters import PolicyRegime

        self.synchronize()
        started = time.perf_counter()
        result = super()._prepare_action_condition(**kwargs)
        self.synchronize()
        if kwargs["regime"] is PolicyRegime.IDM:
            self.calls["video"] += 1
            self.timings["selected_video_condition"] = (
                self.timings.get("selected_video_condition", 0.0)
                + time.perf_counter()
                - started
            )
        elif not self._preparing_gate:
            self.timings["selected_current_condition"] = (
                self.timings.get("selected_current_condition", 0.0)
                + time.perf_counter()
                - started
            )
        return result

    def _velocity(self, *args, **kwargs):
        velocity = super()._velocity(*args, **kwargs)

        def measured(*values):
            self.synchronize()
            started = time.perf_counter()
            result = velocity(*values)
            self.synchronize()
            self.timings["action_denoise"] = (
                self.timings.get("action_denoise", 0.0) + time.perf_counter() - started
            )
            return result

        return measured

    def _make_chunk_sample(self, *, gate_snapshots, **kwargs):
        return RouteNeutralTrainableChunkSample(**kwargs)

    def sample_routed_action_batch(self, **kwargs):
        """Record requests only; teacher inference belongs to the batch boundary."""
        self.calls["action"] += 1
        env_obs = kwargs["env_obs"]
        self.last_seeds = {
            key: env_obs[key].detach().clone()
            for key in ("_fastwam_action_noise_seeds", "_fastwam_idm_noise_seeds")
        }
        self.last_sample = LiberoFastWAMRuntime.sample_action_batch(self, **kwargs)
        return self.last_sample

    def complete_teacher(self, *, sample, seeds, route, actor_version):
        """Materialize a saved request without refreshing sensors or physical history."""
        self.calls["teacher"] += int(route.eq(0).sum())
        return self.materialize_idm_teacher(
            sample=sample, env_obs=seeds, routes=route, actor_version=actor_version
        )

    def critic_observation(self, *, env_obs=None, forward_inputs=None):
        if env_obs is None:
            raise ValueError("Real critic replay requires its saved detached prefix.")
        images = env_obs["real_images"]
        # Use the same processed RGB views; split the configured concatenation.
        widths = [camera.resize[1] for camera in self.observation_adapter.cameras]
        views = images.split(widths, dim=2)
        return {
            "images": views[0],
            "wrist_images": views[1] if len(views) > 1 else None,
            "extra_view_images": views[2] if len(views) > 2 else None,
            "states": env_obs["states"],
            "task_descriptions": env_obs["task_descriptions"],
        }
