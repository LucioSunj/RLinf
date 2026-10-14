# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Standalone EasyWAM-parent inference through the registered RLinf policy.

This module owns artifact selection and per-session environment identities. The
registered policy retains image processing, masked text, denoising, routing,
physical history, normalization and executable-action conversion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config_contracts import validate_fastwam_eval_checkpoint_contract

_ROUTES = {"learned_threshold", "forced_idm", "forced_uncond"}


def load_inference_config(pipeline_dir: str | Path) -> DictConfig:
    """Compose the prepared model without resolving training or critic assets.

    This is a CPU-only configuration operation. It also works before BC has
    produced a sidecar, when only ``mixed.template.yaml`` exists.
    """
    pipeline = Path(pipeline_dir).expanduser().resolve()
    name = "mixed" if (pipeline / "mixed.yaml").is_file() else "mixed.template"
    with initialize_config_dir(version_base="1.3", config_dir=str(pipeline)):
        cfg = compose(config_name=name)
    with open_dict(cfg.actor.model):
        cfg.actor.model.critic = {
            "kind": "pi0_5_value_after_vlm",
            "load_for_eval": False,
        }
        cfg.actor.model.eval_without_critic = True
    return OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))


def _project_payload(checkpoint: str | Path) -> dict[str, Any]:
    from . import resolve_fastwam_adaptive_eval_checkpoint

    rank_file = resolve_fastwam_adaptive_eval_checkpoint(str(checkpoint), rank=0)
    return torch.load(rank_file, map_location="cpu", weights_only=False, mmap=True)


class EasyWAMInferenceSession:
    """Predict executable LIBERO chunks from a parent, BC, or mixed RL policy.

    Pass batched ``main_images``/``wrist_images`` in uint8 BHWC layout,
    unnormalized ``states`` in Bx8 layout, and ``task_descriptions``. Optional
    ``_fastwam_env_ids`` keep identities stable when batch rows move. Call
    :meth:`reset` after each environment reset, or supply the existing RLinf
    ``_fastwam_reset_mask``. Returned actions already contain only the execution
    horizon; no simulator is launched and no action is clamped here.
    """

    def __init__(self, policy: Any, *, device: str | torch.device, seed: int = 42):
        self.policy = policy.to(device=device).eval()
        self.policy.requires_grad_(False)
        self.device = torch.device(device)
        self._seen: set[int] = set()
        self._pending_reset: set[int] = set()
        self._noise = torch.Generator(device="cpu").manual_seed(seed)

    @classmethod
    def from_pipeline(
        cls,
        pipeline_dir: str | Path,
        *,
        mode: str,
        device: str | torch.device = "cuda:0",
        sidecar: str | Path | None = None,
        project_checkpoint: str | Path | None = None,
        routing_mode: str | None = None,
        threshold: float = 0.5,
        seed: int = 42,
    ) -> EasyWAMInferenceSession:
        """Load existing artifacts without a critic, optimizer, Ray, or environment.

        ``idm`` uses the frozen parent. ``bc`` strictly loads both UNCOND LoRA
        branches. ``mixed`` restores the native checkpoint's Gate and adapters,
        including its model configuration, and starts fresh episode history.
        All modes use eager inference; no compilation or warmup is launched.
        """
        from fastwam.uncond_bc_checkpoint import load_uncond_bc_sidecar

        from rlinf.models import get_model

        if mode not in {"idm", "bc", "mixed"}:
            raise ValueError("mode must be idm, bc, or mixed.")
        route = (
            routing_mode
            or {
                "idm": "forced_idm",
                "bc": "forced_uncond",
                "mixed": "learned_threshold",
            }[mode]
        )
        if route not in _ROUTES or (
            mode != "mixed"
            and route != f"forced_{'idm' if mode == 'idm' else 'uncond'}"
        ):
            raise ValueError(
                "Endpoint modes require their fixed route; mixed supports learned/forced routes."
            )
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1].")
        if (mode == "mixed") != (project_checkpoint is not None):
            raise ValueError("Only mixed mode requires a native project checkpoint.")
        if sidecar is not None and mode != "bc":
            raise ValueError("sidecar is only used by bc mode.")

        pipeline = Path(pipeline_dir).expanduser().resolve()
        if mode == "bc":
            if sidecar is None:
                bound_path = pipeline / "mixed.yaml"
                if not bound_path.is_file():
                    raise ValueError(
                        "BC inference needs --sidecar or a bind-bc mixed.yaml."
                    )
                bound = OmegaConf.load(bound_path)
                from rlinf.runners.fastwam_checkpoint_export import (
                    _resolve_uncond_lora_bootstrap,
                )

                bootstrap = _resolve_uncond_lora_bootstrap(bound)
                if bootstrap is None:
                    raise ValueError("mixed.yaml has no bound BC sidecar.")
                sidecar, _ = bootstrap
            sidecar = Path(sidecar).expanduser().resolve()
            if not sidecar.is_file():
                raise FileNotFoundError(sidecar)
        model_cfg = load_inference_config(pipeline)
        bc = OmegaConf.load(pipeline / "bc.yaml")
        expected_parent = str(bc.parent.checkpoint_sha256)
        payload = None
        if mode == "mixed":
            payload = _project_payload(project_checkpoint)
            model_cfg = OmegaConf.create(payload["contract"]["model"])
            with open_dict(model_cfg):
                model_cfg.actor_checkpoint = str(bc.parent.checkpoint)
                model_cfg.model_path = str(bc.parent.checkpoint)
                model_cfg.runtime.text_embedding_cache_dir = str(
                    bc.provenance.text_cache_path
                )
                model_cfg.critic = {
                    "kind": "pi0_5_value_after_vlm",
                    "load_for_eval": False,
                }
                model_cfg.eval_without_critic = True
            if model_cfg.runtime.processor_stats_path != bc.parent.statistics:
                raise ValueError(
                    "Project checkpoint uses different pipeline statistics."
                )
        if model_cfg.fastwam.get("text_padding") != "masked":
            raise ValueError(
                "EasyWAM inference requires the masked-text parent configuration."
            )
        with open_dict(model_cfg):
            model_cfg.init_device = "cpu"
            model_cfg.gate_epsilon = 0.0
            model_cfg.eval_routing_mode = route
            model_cfg.eval_idm_threshold = threshold
            model_cfg.decision_telemetry_enabled = False
            model_cfg.inference_acceleration = None
        if payload is not None:
            validate_fastwam_eval_checkpoint_contract(
                payload,
                model_cfg,
                expected_parent_checkpoint_sha256=expected_parent,
                load_critic=False,
            )
        # Keep placement local to this session rather than RLinf's Worker device.
        with open_dict(model_cfg):
            model_cfg.load_to_device = False
        policy = get_model(model_cfg)
        if mode == "bc":
            metadata = load_uncond_bc_sidecar(
                sidecar,
                adapter=policy.lora_adapter,
                video_adapter=policy.video_lora_adapter,
                expected_parent_checkpoint_sha256=expected_parent,
            )
            if (
                metadata["extra"].get("statistics_sha256")
                != bc.parent.statistics_sha256
            ):
                raise ValueError("BC sidecar used different normalization statistics.")
        elif payload is not None:
            policy.load_eval_checkpoint(
                payload, expected_parent_checkpoint_sha256=expected_parent
            )
            # The native action-only loader can opt into compilation; this
            # standalone entrypoint keeps eager behavior for every profile.
            policy.inference_acceleration = None
        return cls(policy, device=device, seed=seed)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Start new episodes on the next chunk for all known or selected IDs."""
        self._pending_reset.update(self._seen if env_ids is None else env_ids)

    @torch.no_grad()
    def predict(self, observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Return actions, route (0=UNCOND/1=IDM), and episode/chunk identities.

        The input is not mutated. Consecutive calls continue physical history.
        Explicit action/IDM noise tensors or per-row seeds are honored; otherwise
        this session generates independent seeds from its deterministic local RNG.
        IDM probability is the trained Gate probability for learned routing,
        and the fixed 0/1 route for forced endpoints.
        """
        obs = dict(observation)
        for name in ("states", "main_images", "wrist_images"):
            if name in obs:
                obs[name] = torch.as_tensor(obs[name], device=self.device)
        batch = int(obs["states"].shape[0])
        ids = torch.as_tensor(
            obs.get("_fastwam_env_ids", torch.arange(batch)),
            dtype=torch.long,
            device=self.device,
        )
        if ids.shape != (batch,) or ids.unique().numel() != batch:
            raise ValueError(
                "Each observation row requires one distinct environment ID."
            )
        id_list = ids.tolist()
        reset = torch.as_tensor(
            obs.get("_fastwam_reset_mask", torch.zeros(batch)),
            dtype=torch.bool,
            device=self.device,
        )
        if reset.shape != (batch,):
            raise ValueError("Reset mask must contain one value per observation row.")
        reset = reset | torch.tensor(
            [
                env_id not in self._seen or env_id in self._pending_reset
                for env_id in id_list
            ],
            dtype=torch.bool,
            device=self.device,
        )
        obs.update(_fastwam_env_ids=ids, _fastwam_reset_mask=reset)
        for seed_field, noise_field in (
            ("_fastwam_action_noise_seeds", "_fastwam_action_initial_noise"),
            ("_fastwam_idm_noise_seeds", "_fastwam_idm_initial_latents"),
        ):
            if obs.get(seed_field) is None and obs.get(noise_field) is None:
                obs[seed_field] = torch.randint(
                    0, 2**62, (batch,), generator=self._noise
                )
        actions, result = self.policy.predict_action_batch(
            obs, mode="eval", compute_values=False
        )
        if not torch.isfinite(actions).all():
            raise ValueError("The policy produced non-finite executable actions.")
        self._seen.update(id_list)
        self._pending_reset.difference_update(id_list)
        routes = result["route_info"]
        return {
            "actions": actions,
            "routes": routes.route_used,
            "idm_probability": (
                result["emitted_gate"].base_probability
                if self.policy.config.eval_routing_mode == "learned_threshold"
                else routes.route_used.float()
            ),
            "episode_ids": routes.episode_ids,
            "chunk_ids": routes.chunk_ids,
        }
