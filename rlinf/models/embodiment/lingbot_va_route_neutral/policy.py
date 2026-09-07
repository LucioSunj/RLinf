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

"""Registered single-robot policy, current-step Gate and native model builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rlinf.algorithms.fastwam_dual_ppo import epsilon_mixture_bernoulli
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.wam_policy.critic import FastWAMValueTransformerConfig
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PadRouteNeutralCurrentStepGate,
    PadRouteNeutralGateConfig,
)

from .contracts import ActionNormalizer, Route, RoutingConfig
from .encoder import ObservationEncoder
from .functional import FunctionalVA
from .runtime import ChunkNoise, ChunkSample, LingBotVARuntime, clock_sync


@dataclass
class Decision:
    sample: ChunkSample
    gate_features: object | None
    critic_features: torch.Tensor | None
    log_prob: torch.Tensor
    probability: float
    value: float
    epsilon: float
    actor_version: int
    autonomous: bool = True
    reward: float = 0.0
    executed_steps: int = 0
    terminal: bool = False


class LingBotVARouteNeutralPolicy(BasePolicy, nn.Module):
    """One synchronous episode; actor and rollout share parameters between rounds."""

    def __init__(
        self,
        runtime,
        gate,
        critic=None,
        *,
        parent_path="",
        seed=42,
        asset_references=None,
    ):
        super().__init__()
        self.core = runtime.model
        self.encoder = runtime.encoder
        self.gate, self.critic = gate.float(), critic
        self.runtime, self.config = runtime, runtime.config
        self.parent_path = str(Path(parent_path).resolve()) if parent_path else ""
        self.asset_references = asset_references or {}
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.actor_version = 0
        self.pending_decision = None

    def train(self, mode=True):
        super().train(mode)
        self.core.parent.eval()
        if isinstance(self.encoder, nn.Module):
            self.encoder.eval()
        return self

    def reset_episode(self, instruction, initial_observation):
        self.instruction = instruction
        self.runtime.reset_episode(instruction, initial_observation)
        self.pending_decision = None

    def prepare_gate_features(self, current_observation):
        return self.runtime.prepare_gate_features(current_observation)

    def critic_observation(self, observation):
        keys = self.config.camera_keys
        return {
            "main_images": torch.from_numpy(
                np.asarray(observation["images"][keys[0]]).copy()
            )[None],
            "wrist_images": torch.from_numpy(
                np.asarray(observation["images"][keys[1]]).copy()
            )[None],
            "extra_view_images": None,
            "states": torch.as_tensor(observation["state"], dtype=torch.float32)[None],
            "task_descriptions": [self.instruction],
        }

    @torch.no_grad()
    def decide(
        self,
        observation,
        *,
        training=False,
        warmup=False,
        route=None,
        routing_rate=None,
        gate_framework=True,
    ):
        """Evaluate the Gate before selecting this same chunk's branch."""
        if self.pending_decision is not None:
            raise RuntimeError("Commit execution feedback before another decision.")
        if training and (
            route is not None or routing_rate is not None or not gate_framework
        ):
            raise ValueError(
                "Training uses only the declared warm-up/learned distribution."
            )
        reference = self.runtime.reference
        inference_started = clock_sync(reference)
        timings = {"gate_features_s": 0.0, "gate_s": 0.0, "critic_s": 0.0}
        features = None
        epsilon = 1.0 if warmup else (self.config.epsilon if training else 0.0)
        if gate_framework:
            begin = clock_sync(reference)
            features = self.prepare_gate_features(observation)
            timings["gate_features_s"] = clock_sync(reference) - begin
            begin = clock_sync(reference)
            logits = self.gate(features).cpu()
            distribution = epsilon_mixture_bernoulli(
                logits,
                epsilon=epsilon,
                temperature=self.config.temperature,
                # Forced endpoints do not draw route randomness.
                route=None
                if route is None and routing_rate is None
                else torch.tensor([int(route == Route.IDM)]),
                generator=self.rng,
            )
            probability = float(distribution.behavior_probability.item())
            if route is None and routing_rate is None:
                route = Route.IDM if distribution.route.item() else Route.UNCOND
            log_prob = distribution.logprob.cpu()
            timings["gate_s"] = clock_sync(reference) - begin
        else:
            self.runtime.observation = observation
            probability, log_prob = float(route == Route.IDM), torch.zeros(1)
        if routing_rate is not None:
            if not 0 <= routing_rate <= 1:
                raise ValueError("Routing rate must lie in [0,1].")
            route = (
                Route.IDM
                if torch.rand((), generator=self.rng).item() < routing_rate
                else Route.UNCOND
            )
            probability = routing_rate
        if route is None:
            raise ValueError("Standalone deployment requires an explicit endpoint.")
        critic_features, value = None, 0.0
        if training:
            if self.critic is None:
                raise ValueError("Online RL requires the frozen Pi0.5 critic prefix.")
            begin = clock_sync(reference)
            critic_features = self.critic.encode_features(
                self.critic_observation(observation)
            )
            value = float(self.critic.value_from_features(critic_features).item())
            critic_features = critic_features.detach().cpu()
            timings["critic_s"] = clock_sync(reference) - begin
        seeds = torch.randint(0, 2**62, (3,), generator=self.rng).tolist()
        sample = self.sample_chunk(
            route, ChunkNoise(*seeds), training=training and not warmup
        )
        timings["inference_s"] = clock_sync(reference) - inference_started
        sample.timings.update(timings)
        decision = Decision(
            sample,
            None if features is None else features.detached().to(device="cpu"),
            critic_features,
            log_prob,
            probability,
            value,
            epsilon,
            self.actor_version,
        )
        self.pending_decision = decision
        return decision

    def sample_chunk(self, route, noise, training=False):
        return self.runtime.sample_chunk(route, noise, training)

    def commit_execution(
        self,
        observed_frames,
        executed_actions,
        *,
        terminated=False,
        success=False,
        autonomous=True,
    ):
        decision = self.pending_decision
        if decision is None:
            raise RuntimeError("No pending decision to commit.")
        if success and not terminated:
            raise ValueError("Success is a terminal reward, paid once.")
        decision.autonomous = bool(autonomous)
        decision.executed_steps = len(executed_actions)
        decision.terminal = bool(terminated)
        decision.reward = float(success and autonomous)
        result = self.runtime.commit_execution(
            observed_frames, executed_actions, terminated=terminated
        )
        decision.sample.timings.update(result)
        self.pending_decision = None
        return decision

    def replay_uncond_transition(self, replay_sample, **kwargs):
        return self.runtime.replay_uncond_transition(replay_sample, **kwargs)

    def default_forward(self, *, decision, history=None):
        route = torch.tensor(
            [int(decision.sample.route is Route.IDM)],
            device=next(self.gate.parameters()).device,
        )
        distribution = epsilon_mixture_bernoulli(
            self.gate(decision.gate_features),
            epsilon=decision.epsilon,
            temperature=self.config.temperature,
            route=route,
        )
        value = self.critic.value_from_features(
            decision.critic_features.to(route.device)
        ).reshape(1)
        flow = (
            value.new_zeros(1)
            if decision.sample.flow is None
            else self.replay_uncond_transition(
                decision.sample.flow, history=history
            ).reshape(1)
        )
        return {"gate": distribution, "value": value, "flow_log_prob": flow}

    def predict_action_batch(self, env_obs, **kwargs):
        decision = self.decide(env_obs, **kwargs)
        return decision.sample.actions.numpy()[None], {"decision": decision}


def make_gate(parent, config):
    visual = FastWAMValueTransformerConfig(
        num_mot_layers=len(parent.blocks),
        source_num_heads=parent.config.num_attention_heads,
        source_head_dim=parent.config.attention_head_dim,
        layer_indices=config.gate_layers,
        sources=("current_frame_video",),
        hidden_dim=config.gate_hidden,
        num_query_tokens=config.gate_queries,
    )
    return (
        PadRouteNeutralCurrentStepGate(
            PadRouteNeutralGateConfig(
                visual,
                parent.config.text_dim,
                config.state_dim,
                config.physical_history,
            )
        )
        .float()
        .to(next(parent.parameters()).device)
    )


def get_model(cfg, torch_dtype=None):
    """Load a new real-world parent and independent adapters; no FastWAM state."""
    from wan_va.modules.utils import (
        load_text_encoder,
        load_tokenizer,
        load_transformer,
        load_vae,
    )

    dtype = torch.bfloat16 if torch_dtype is None else torch_dtype
    config = RoutingConfig(**dict(cfg.get("routing", {})))
    base, parent_path = Path(cfg.base_model_path), Path(cfg.parent_path)
    device = cfg.get("device", "cuda:0")
    parent = load_transformer(
        str(parent_path / "transformer"),
        torch_dtype=dtype,
        torch_device=device,
        attn_mode="torch",
    )
    core = FunctionalVA(parent, rank=config.lora_rank, alpha=config.lora_alpha)
    encoder = ObservationEncoder(
        load_vae(str(base / "vae"), torch_dtype=dtype, torch_device=device),
        load_text_encoder(
            str(base / "text_encoder"),
            torch_dtype=dtype,
            torch_device=cfg.get("text_device", "cpu"),
        ),
        load_tokenizer(str(base / "tokenizer")),
        config,
    )
    normalizer = ActionNormalizer.load(cfg.normalizer_path)
    runtime = LingBotVARuntime(core, encoder, normalizer, config)
    critic = None
    if cfg.get("load_critic", False):
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model
        from rlinf.models.embodiment.wam_policy.pi05_critic import (
            Pi05ValueAfterVLMCritic,
        )

        critic = Pi05ValueAfterVLMCritic(
            get_openpi_model(cfg.critic.backbone, dtype)
        ).to(device)
        critic.value_head.float()
    policy = LingBotVARouteNeutralPolicy(
        runtime,
        make_gate(parent, config),
        critic,
        parent_path=str(parent_path),
        seed=int(cfg.get("seed", 42)),
        asset_references={
            "base_model_path": str(base.resolve()),
            "bc_path": str(Path(cfg.bc_path).resolve()) if cfg.get("bc_path") else "",
            "critic_path": str(Path(cfg.critic.backbone.model_path).resolve())
            if critic is not None
            else "",
        },
    )
    if cfg.get("bc_path"):
        payload = torch.load(cfg.bc_path, map_location="cpu", weights_only=False)
        if (
            payload["model_type"] != "lingbot_va_route_neutral"
            or payload["stage"] != "bc"
        ):
            raise ValueError("Expected the new LingBot-VA BC adapter checkpoint.")
        if (
            payload["parent_path"] != policy.parent_path
            or payload["normalizer"] != normalizer.to_dict()
        ):
            raise ValueError("BC parent or demonstration normalization does not match.")
        core.adapters.load_state_dict(payload["lora"], strict=True)
    return policy
