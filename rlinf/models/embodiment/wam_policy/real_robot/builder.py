# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fresh real PAD initialization without LIBERO geometry or released U aliases."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from fastwam.adapters import RegimeLoRAConfig, inject_action_dit_lora, sha256_file
from torch import nn

from rlinf.models.embodiment.wam_policy.adaptive_policy import (
    FastWAMAdaptivePolicyConfig,
)
from rlinf.models.embodiment.wam_policy.critic import FastWAMValueTransformerConfig
from rlinf.models.embodiment.wam_policy.kv_replay import GateKVReplayConfig
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import OnlineIDMBCConfig
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PadRouteNeutralCurrentStepGate,
    PadRouteNeutralGateConfig,
)
from rlinf.models.embodiment.wam_policy.pi05_critic import Pi05ValueAfterVLMCritic

from .config import action_protocol, validate_config
from .policy import RealRobotPADPolicy
from .runtime import RealRobotFastWAMRuntime


def route_neutral_profile():
    """Use the checked-in current trainable route-neutral contract verbatim."""
    path = (
        Path(__file__).parents[5]
        / "examples/embodiment/config/model/fastwam_route_neutral_online.yaml"
    )
    return yaml.safe_load(path.read_text())["route_neutral_online"]


class TinyRealRuntime(RealRobotFastWAMRuntime):
    """CPU backbone fixture; production sampler, teacher and feature producer."""

    @torch.no_grad()
    def _prepare_action_condition(
        self,
        *,
        image,
        context,
        context_mask,
        regime,
        idm_initial_latents=None,
        idm_noise_seed=None,
    ):
        from fastwam.adapters import PolicyRegime

        started = time.perf_counter()
        noise = None
        if regime is PolicyRegime.IDM:
            self.calls["video"] += 1
            generator = torch.Generator().manual_seed(int(idm_noise_seed or 0))
            noise = torch.randn(len(image), 1, 16, generator=generator)
        condition = self.actor.make_condition(
            image, context, context_mask, future_noise=noise
        )
        if not self._preparing_gate:
            key = (
                "selected_video_condition"
                if noise is not None
                else "selected_current_condition"
            )
            self.timings[key] = (
                self.timings.get(key, 0.0) + time.perf_counter() - started
            )
        return condition, noise

    def _denormalize_action_stages(self, actions, *, env_obs):
        if self.processor is not None:
            return super()._denormalize_action_stages(actions, env_obs=env_obs)
        # Declared synthetic calibration for the mock dynamics only.
        scale = actions.new_tensor([0.015, 0.015, 0.015, 0.03, 0.03, 0.03, 0.5])
        return actions.float() * scale + actions.new_tensor(
            [0, 0, 0, 0, 0, 0, 0.5]
        ), None


class TinyPrefixBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(config_name="pi05_tiny_prefix_fixture")
        self.prefix_encoder = nn.Linear(11, 16)

    def get_value_from_vlm(self, prefix):
        return self.value_head(prefix.mean(1))


class TinyPrefixCritic(Pi05ValueAfterVLMCritic):
    """Exercise the real value head/detach wrapper with a CPU prefix fixture."""

    def __init__(self):
        super().__init__(TinyPrefixBackbone(), input_dim=16, hidden_sizes=(32, 16))
        self.calls = 0

    def predict_value_batch(self, env_obs, *, return_prefix=False):
        self.calls += 1
        with torch.no_grad():
            image = env_obs["images"].float().mean((1, 2)) / 255
            prefix = torch.tanh(
                self.backbone.prefix_encoder(
                    torch.cat([image, env_obs["states"].float()], -1)
                )
            )[:, None]
        value = self.value_from_prefix(prefix)
        return (value, prefix) if return_prefix else value


def build_real_robot_policy(cfg: dict, *, evaluation_method=None, initialize=True):
    """Construct only owners required by the selected training/evaluation path."""
    validate_config(cfg)
    profile = route_neutral_profile()
    need_gate = evaluation_method in (None, "learned")
    torch.manual_seed(cfg["seed"])
    model_cfg = cfg["model"]
    processor = None
    if model_cfg["kind"] == "tiny":
        from fastwam.real_robot_tiny import TinyFastWAM, adapt_tiny_fixture

        actor = TinyFastWAM(model_cfg["sigma_shift"])
        tiny_init = model_cfg.get("tiny_initialization_dir")
        if initialize and tiny_init:
            import json

            initialization = json.loads(
                (Path(tiny_init) / "initialization.json").read_text()
            )
            parent_state = torch.load(
                initialization["parent"], map_location="cpu", weights_only=True
            )["state_dict"]
            actor.load_state_dict(parent_state)
            adapter = inject_action_dit_lora(actor.action_expert, RegimeLoRAConfig())
            adapter.load_sidecar(
                initialization["sidecar"],
                expected_parent_checkpoint_sha256=sha256_file(initialization["parent"]),
            )
        elif initialize:
            adapter, initialization, parent_state = adapt_tiny_fixture(actor)
        else:
            adapter = inject_action_dit_lora(actor.action_expert, RegimeLoRAConfig())
            initialization, parent_state = {}, None
        visual = FastWAMValueTransformerConfig(
            num_mot_layers=20,
            source_num_heads=1,
            source_head_dim=16,
            layer_indices=tuple(range(14, 20)),
            sources=("current_frame_video",),
        )
        critic = TinyPrefixCritic() if evaluation_method is None else None
        runtime_type = TinyRealRuntime
        if model_cfg["stats_path"] is not None:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf

            processor = instantiate(OmegaConf.load(model_cfg["model_config"]).processor)
    else:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from rlinf.models.embodiment.wam_policy import _load_strict_fastwam_parent

        assets = OmegaConf.load(model_cfg["model_config"])
        dtype = getattr(torch, assets.precision)
        actor = instantiate(assets.fastwam, model_dtype=dtype, device=assets.device)
        _load_strict_fastwam_parent(actor, model_cfg["parent_checkpoint"])
        adapter = inject_action_dit_lora(
            actor.action_expert, RegimeLoRAConfig(rank=16, alpha=16)
        )
        adapter.load_sidecar(
            model_cfg["uncond_bc_sidecar"],
            expected_parent_checkpoint_sha256=sha256_file(
                model_cfg["parent_checkpoint"]
            ),
        )
        processor = instantiate(assets.processor)
        visual = FastWAMValueTransformerConfig(
            num_mot_layers=actor.mot.num_layers,
            source_num_heads=actor.mot.num_heads,
            source_head_dim=actor.mot.attn_head_dim,
            layer_indices=tuple(range(14, 20)),
            sources=("current_frame_video",),
        )
        critic = None
        if evaluation_method is None:
            from rlinf.models.embodiment.openpi import get_model as get_openpi_model

            critic_cfg = OmegaConf.load(model_cfg["critic_config"])
            critic = Pi05ValueAfterVLMCritic(
                get_openpi_model(critic_cfg.backbone, dtype),
                input_dim=critic_cfg.input_dim,
            )
        initialization = {
            "parent": model_cfg["parent_checkpoint"],
            "bc": model_cfg["uncond_bc_sidecar"],
        }
        parent_state = None
        runtime_type = RealRobotFastWAMRuntime
    runtime = runtime_type(
        actor=actor,
        lora_adapter=adapter,
        observation=cfg["observation"],
        action_protocol=action_protocol(cfg),
        route_neutral_input=profile["input_contract"],
        route_neutral_visual=visual,
        processor=processor,
        processor_stats_path=model_cfg["stats_path"],
        num_inference_steps=model_cfg["flow_steps"],
        sigma_shift=model_cfg["sigma_shift"],
        flow_sde_noise_level=model_cfg["flow_noise"],
        flow_sde_ignore_last_transition=model_cfg["flow_ignore_last"],
        text_embedding_cache_dir=model_cfg["text_cache"],
    )
    gate = (
        PadRouteNeutralCurrentStepGate(
            PadRouteNeutralGateConfig(
                visual=visual,
                language_dim=actor.text_dim,
                state_dim=8,
                history_length_chunks=4,
            )
        )
        if need_gate
        else nn.Identity()
    )
    gate.to(device=runtime.device, dtype=torch.float32)
    if critic is not None:
        critic.to(device=runtime.device)
        critic.value_head.to(dtype=torch.float32)
    policy = RealRobotPADPolicy(
        seed=cfg["seed"],
        actor=actor,
        runtime=runtime,
        lora_adapter=adapter,
        gate=gate,
        critic=critic,
        config=FastWAMAdaptivePolicyConfig(
            gate_epsilon=cfg["training"]["gate_epsilon"],
            gate_temperature=cfg["training"]["gate_temperature"],
            kv_replay=GateKVReplayConfig(backend="recompute"),
        ),
        online_idm_bc_config=OnlineIDMBCConfig(enabled=True, loss_weight=0.2),
        critic_warmup=profile["critic_warmup"],
    )
    policy.initialization = initialization
    policy.initial_parent_state = parent_state
    return policy
