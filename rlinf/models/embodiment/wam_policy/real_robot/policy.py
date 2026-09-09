# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Current-step PAD with a policy-neutral physical proposal/receipt boundary."""

from __future__ import annotations

import time

import torch
from fastwam.models.wan22.gate_transformer import epsilon_mixture_bernoulli

from rlinf.envs.pad_realworld.contracts import ActionProposal
from rlinf.models.embodiment.wam_policy.libero_runtime import (
    _domain_separated_noise_seed,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.policy import (
    RouteNeutralOnlineIDMBCFastWAMPolicy,
)


class RealRobotPADPolicy(RouteNeutralOnlineIDMBCFastWAMPolicy):
    """Reuse current-step routing/replay; omit unused inference in physical phases."""

    def __init__(self, *, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.streams = {
            name: torch.Generator().manual_seed(seed + offset)
            for name, offset in (("gate", 11), ("action", 29), ("idm", 47))
        }
        self._first_chunk = True
        self.gate_calls = 0
        self.gate.register_forward_pre_hook(self._start_gate_timer)
        self.gate.register_forward_hook(self._finish_gate_timer)

    def _start_gate_timer(self, module, inputs):
        self.runtime.synchronize()
        self._gate_started = time.perf_counter()

    def _finish_gate_timer(self, module, inputs, result):
        self.runtime.synchronize()
        self.runtime.timings["gate"] = time.perf_counter() - self._gate_started

    def begin_episode(self):
        self._first_chunk = True
        self.runtime.observation_adapter.reset()
        self.runtime._encoded = None

    def _formal_training_sampling_seeds(self, **kwargs):
        self.last_sampling_seeds = {
            name: torch.randint(0, 2**60, (1,), generator=generator)
            for name, generator in self.streams.items()
        }
        return self.last_sampling_seeds

    def _compute_online_bc_for_batch(self, route_info):
        return bool(
            (route_info.actor_versions >= self.critic_warmup.runner_updates).all()
        )

    def default_forward(self, forward_inputs, *, route_info, emitted_gate, **kwargs):
        if "critic_prefix" not in forward_inputs:
            raise RuntimeError(
                "Deferred old critic/prefix must finish before replay or GAE."
            )
        if bool((route_info.actor_versions < self.critic_warmup.runner_updates).all()):
            return {
                "values": self.critic.value_from_features(
                    forward_inputs["critic_prefix"]
                ).reshape(-1, 1)
            }
        return super().default_forward(
            forward_inputs, route_info=route_info, emitted_gate=emitted_gate, **kwargs
        )

    @torch.no_grad()
    def propose(
        self, snapshot, *, mode="train", method="learned", random_idm_probability=0.5
    ):
        """Generate one full sample, retaining original replay before execution limits."""
        started = time.perf_counter()
        env_obs = self.runtime.prepare_snapshot(snapshot)
        env_obs.update(
            {
                "_fastwam_env_ids": torch.tensor([0]),
                "_fastwam_reset_mask": torch.tensor([self._first_chunk]),
            }
        )
        if mode == "train":
            _actions, replay = self.predict_action_batch(
                env_obs, mode="train", compute_values=False
            )
            self.gate_calls += 1
            route = int(replay["route_info"].route_used.item())
            replay["old_values_complete"] = False
            replay["saved_env_obs"] = env_obs
        else:
            if method not in {"learned", "random", "always_idm", "always_uncond"}:
                raise ValueError(f"Unknown evaluation route: {method}")
            seeds = self._formal_training_sampling_seeds()
            env_obs.update(
                {
                    "_fastwam_action_noise_seeds": seeds["action"],
                    "_fastwam_idm_noise_seeds": seeds["idm"],
                }
            )
            if method == "learned":
                features = self.runtime.prepare_route_neutral_gate_features(
                    env_obs=env_obs
                )
                logits = self.gate(features)
                self.gate_calls += 1
                probability = epsilon_mixture_bernoulli(
                    logits,
                    temperature=self.config.gate_temperature,
                    epsilon=self.config.gate_epsilon,
                ).behavior_idm_probability
            else:
                probability = torch.tensor(
                    [
                        1.0
                        if method == "always_idm"
                        else 0.0
                        if method == "always_uncond"
                        else random_idm_probability
                    ]
                )
            generator = torch.Generator(device=probability.device).manual_seed(
                int(seeds["gate"].item())
            )
            routes = torch.bernoulli(probability, generator=generator).long()
            route = int(routes.item())
            sample = self.runtime.sample_routed_action_batch(
                env_obs=env_obs,
                routes=routes,
                mode="eval",
                actor_version=self.actor_version,
                collect_replay=False,
            )
            replay = {"behavior_probability": float(probability.item())}
        sample = self.runtime.last_sample
        if mode == "train":
            replay["teacher_request"] = {
                "sample": sample,
                "seeds": self.runtime.last_seeds,
                "actor_version": self.actor_version,
            }
        normalized = sample.normalized_actions
        post_started = time.perf_counter()
        canonical, _ = self.runtime._denormalize_action_stages(
            normalized, env_obs=env_obs
        )
        self.runtime.synchronize()
        self.runtime.timings["postprocess"] = time.perf_counter() - post_started
        timings = dict(self.runtime.timings)
        timings["prediction_total"] = time.perf_counter() - started
        self._first_chunk = False
        self.runtime._encoded = None
        return ActionProposal(
            proposal_id=f"{snapshot.episode_id}:{snapshot.chunk_id}",
            snapshot=snapshot,
            actor_version=self.actor_version,
            route=route,
            normalized_actions=normalized[0].cpu().numpy(),
            canonical_actions=canonical[0].cpu().numpy(),
            replay=replay,
            timings=timings,
            noise={
                "route": int(self.last_sampling_seeds["gate"].item()),
                "action": int(self.last_sampling_seeds["action"].item()),
                "video": int(self.last_sampling_seeds["idm"].item()),
                "sde": _domain_separated_noise_seed(
                    int(self.last_sampling_seeds["action"].item()), domain="flow-sde"
                ),
            },
        )
