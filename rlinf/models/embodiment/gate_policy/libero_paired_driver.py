# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Concrete one-env LIBERO-Plus/FastWAM driver for paired-v1 collection."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _load_callable(spec: str, *, flag: str):
    if not spec or ":" not in spec:
        raise ValueError(f"{flag} must have the form python.module:function")
    module_name, function_name = spec.rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"{flag}={spec!r} is not callable")
    return function


class LiberoFastWAMPairedDriver:
    """A strict batch-one adapter around RLinf ``LiberoEnv`` and frozen WAM."""

    def __init__(self, *, cfg, manifest, progress_fn):
        from rlinf.envs.libero.episode_manifest import FrozenEpisode
        from rlinf.envs.libero.libero_env import (
            LIBERO_GATE_SNAPSHOT_SCHEMA,
            LiberoEnv,
        )
        from rlinf.models.embodiment.gate_policy import get_model

        if os.environ.get("LIBERO_TYPE", "").lower() != "plus":
            raise ValueError("concrete paired collection requires LIBERO_TYPE=plus")
        self.FrozenEpisode = FrozenEpisode
        self.manifest = manifest
        self.progress_fn = progress_fn
        self.policy = get_model(cfg.rollout.model)
        self.device = torch.device(str(cfg.rollout.model.wam.device))
        self.policy.to(self.device)
        self.policy.eval()
        if self.policy.wam_adapter is None or self.policy.obs_preprocessor is None:
            raise RuntimeError("paired driver failed to construct frozen WAM/preprocessor")
        self.adapter = self.policy.wam_adapter
        self.exec_horizon = int(cfg.rollout.model.wam.exec_horizon)
        if self.exec_horizon <= 0:
            raise ValueError("rollout.model.wam.exec_horizon must be positive")
        self.env = LiberoEnv(
            cfg=cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        provenance = dict(self.policy.bc_expected_provenance or {})
        try:
            from fastwam.adaptive_gate import TEXT_FEAT_LAYOUT, WORLD_FEAT_LAYOUT
        except ImportError as exc:
            raise ImportError("paired collection requires the FastWAM checkout") from exc
        self.paired_metadata = {
            "task": provenance["task"],
            "backbone_kind": provenance["backbone_kind"],
            "ckpt_fingerprint": provenance["ckpt_fingerprint"],
            "ckpt_file_sha256": self.policy.wam_checkpoint_sha256,
            "dataset_stats_fingerprint": provenance[
                "dataset_stats_fingerprint"
            ],
            "num_video_frames": provenance["num_video_frames"],
            "inference_steps": provenance["inference_steps"],
            "solver_fingerprint": provenance["solver_fingerprint"],
            "context_len": provenance["context_len"],
            "model_dtype": provenance["model_dtype"],
            "exec_horizon": provenance["exec_horizon"],
            "action_horizon": provenance["action_horizon"],
            "world_feat_layout": WORLD_FEAT_LAYOUT,
            "text_feat_layout": TEXT_FEAT_LAYOUT,
            "mode_order": ["uncond", "idm"],
            "world_feat_dim": int(self.policy.world_feat_dim),
            "proprio_dim": int(self.policy.proprio_dim),
            "text_feat_dim": int(self.policy.text_feat_dim),
            "snapshot_schema": LIBERO_GATE_SNAPSHOT_SCHEMA,
            "episode_manifest_sha256": manifest.sha256,
            "heldout_test_manifest_sha256": (
                self.env.test_episode_manifest.sha256
                if self.env.test_episode_manifest is not None
                else manifest.sha256
            ),
            "libero_plus_commit": manifest.libero_plus_commit,
            "manifest_split": manifest.split,
        }
        if not self.paired_metadata["ckpt_file_sha256"]:
            raise RuntimeError("paired collection cannot resolve the WAM file SHA256")
        self._started = False
        self._last_observation = None

    def _episode(self, value: Mapping[str, Any]):
        fields = {
            key: value[key]
            for key in (
                "episode_id",
                "base_task",
                "task_suite_name",
                "task_id",
                "factor",
                "level",
                "bddl_path",
                "bddl_sha256",
                "reset_state_id",
                "trial_id",
                "env_seed",
                "perturbation_id",
                "asset_ids",
            )
        }
        fields["asset_ids"] = tuple(fields["asset_ids"])
        return self.FrozenEpisode(**fields)

    def reset_episode(self, episode: Mapping[str, Any]):
        entry = self._episode(episode)
        # Train manifests are shuffled by LiberoEnv for GRPO. Paired collection,
        # however, walks the caller's balanced episode order, so every reset,
        # including the first, must bind the explicitly requested frozen entry.
        observation, _ = self.env.reset(
            env_idx=np.asarray([0]), manifest_entries=[entry]
        )
        self._started = True
        self._last_observation = observation
        return observation

    def capture_snapshot(self):
        return self.env.capture_gate_snapshot(env_idx=np.asarray([0]))

    def restore_snapshot(self, snapshot):
        observation = self.env.restore_gate_snapshot(dict(snapshot))
        self._last_observation = observation
        return observation

    def context(self, observation):
        raw = observation.get("gate_context")
        if not isinstance(raw, dict):
            raise RuntimeError("LiberoEnv observation has no gate_context")
        context = {}
        for key, value in raw.items():
            if isinstance(value, torch.Tensor):
                if value.shape[0] != 1:
                    raise RuntimeError("paired driver requires batch-one gate_context")
                context[key] = value.reshape(-1)[0].item()
            elif isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise RuntimeError("paired driver requires batch-one gate_context")
                context[key] = value[0]
            else:
                context[key] = value
        return context

    def _wam_inputs(self, observation):
        inputs = self.policy.preprocess_env_obs(observation)
        input_image = inputs["input_image"].to(self.device)
        proprio = inputs["proprio"].to(self.device)
        context = inputs["context"].to(self.device)
        context_mask = inputs["context_mask"].to(self.device)
        text_feat = inputs["text_feat"].to(
            device=self.device, dtype=torch.float32
        )
        if input_image.shape[0] != 1:
            raise RuntimeError("concrete paired driver supports exactly one environment")
        encoded = self.adapter.encode_world_state(input_image)
        return inputs, input_image, proprio, context, context_mask, text_feat, encoded

    def features(self, observation):
        _, _, proprio, _, _, text_feat, encoded = self._wam_inputs(observation)
        return {
            "world_feat": encoded.world_feat.detach().float().cpu(),
            "proprio": proprio[0].detach().float().cpu(),
            "text_feat": text_feat[0].detach().float().cpu(),
        }

    def action(self, observation, *, mode: int, seed: int):
        (
            _,
            input_image,
            proprio,
            context,
            context_mask,
            _,
            encoded,
        ) = self._wam_inputs(observation)
        output = self.adapter.act(
            input_image=input_image,
            mode=int(mode),
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            encoded_state=encoded,
            seed=int(seed),
        )
        actions = output["action_chunk"].detach().cpu().float().unsqueeze(0)
        if hasattr(self.policy.obs_preprocessor, "process_actions"):
            actions = self.policy.obs_preprocessor.process_actions(actions)
        elif hasattr(self.policy.obs_preprocessor, "denormalize_actions"):
            actions = self.policy.obs_preprocessor.denormalize_actions(actions)
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise RuntimeError(
                f"FastWAM paired action must be [1,T,A], got {tuple(actions.shape)}"
            )
        return actions[:, : self.exec_horizon]

    def step_chunk(self, action):
        obs_list, _, terminations, truncations, infos = self.env.chunk_step(action)
        observation = obs_list[-1]
        success = bool(self.env.success_once[0]) or bool(terminations[0].any())
        done = success or bool(truncations[0].any())
        progress = float(
            self.progress_fn(
                env=self.env,
                observation=observation,
                infos=infos,
            )
        )
        if not np.isfinite(progress):
            raise ValueError("task-specific progress function returned non-finite value")
        self._last_observation = observation
        return {
            "observation": observation,
            "done": done,
            "success": success,
            "progress": progress,
        }

    def close(self):
        self.env.env.close()


def build_libero_fastwam_driver(*, args, manifest):
    """Factory directly usable by ``collect_gate_paired_states.py --driver``."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    config_dir_value = getattr(args, "rlinf_config_dir", None)
    config_name = getattr(args, "rlinf_config_name", None)
    progress_spec = getattr(args, "progress_fn", None)
    if not config_dir_value or not config_name or not progress_spec:
        raise ValueError(
            "the concrete LIBERO/FastWAM driver requires --rlinf-config-dir, "
            "--rlinf-config-name and --progress-fn"
        )
    config_dir = Path(config_dir_value).expanduser().resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"RLinf config directory not found: {config_dir}")
    os.environ.setdefault("EMBODIED_PATH", str(config_dir.parent))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name=str(config_name).removesuffix(".yaml"),
            overrides=list(getattr(args, "config_override", None) or []),
        )
    configured_suite = str(cfg.env.eval.task_suite_name)
    manifest_suites = {entry.task_suite_name for entry in manifest.episodes}
    if manifest_suites != {configured_suite}:
        raise ValueError(
            "the direct LiberoEnv paired driver runs exactly one benchmark suite "
            f"per process: config={configured_suite!r}, manifest={sorted(manifest_suites)!r}. "
            "Partition the logical manifest with "
            "scripts/adaptive_gate/plus_suite_manifest.py and collect one paired-v1 "
            "dataset per suite; cross-fit inputs may include all validated suite datasets."
        )
    heldout_test_manifest = getattr(args, "heldout_test_manifest", None)
    if manifest.split in {"train", "validation"} and not heldout_test_manifest:
        raise ValueError(
            f"a split={manifest.split} paired collection requires "
            "--heldout-test-manifest "
            "for the mandatory primary/test disjointness audit"
        )
    if manifest.split == "test" and heldout_test_manifest:
        raise ValueError(
            "--heldout-test-manifest must not point a split=test headline "
            "manifest back to itself"
        )
    with open_dict(cfg.env.eval):
        cfg.env.eval.total_num_envs = 1
        cfg.env.eval.group_size = 1
        cfg.env.eval.auto_reset = False
        cfg.env.eval.ignore_terminations = False
        cfg.env.eval.is_eval = manifest.split != "train"
        cfg.env.eval.use_fixed_reset_state_ids = True
        cfg.env.eval.use_ordered_reset_state_ids = True
        cfg.env.eval.episode_manifest_path = str(manifest.path)
        cfg.env.eval.test_episode_manifest_path = heldout_test_manifest
        cfg.env.eval.gate_exec_horizon = int(cfg.rollout.model.wam.exec_horizon)
        cfg.env.eval.video_cfg.save_video = False
    progress_fn = _load_callable(progress_spec, flag="--progress-fn")
    return LiberoFastWAMPairedDriver(
        cfg=cfg, manifest=manifest, progress_fn=progress_fn
    )
