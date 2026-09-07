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

"""Synchronous LingBot-VA routing and exact, causal replay conditions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from fastwam.models.wan22.flow_sde import flow_sde_mean_std, gaussian_log_prob
from fastwam.models.wan22.kv_tap import KeyValueBank, KVSource

from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    RouteNeutralGateFeatures,
    RouteNeutralVisualFeatures,
    RouteNeutralVisualLayer,
)

from .cache import Cache, append_cache, positive_cache
from .contracts import Route, action_tensor, flatten_actions, valid_action_mask


def cpu_noise(shape, seed, reference):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(reference)


def clock_sync(reference):
    if reference.device.type == "cuda":
        torch.cuda.synchronize(reference.device)
    return perf_counter()


@dataclass(frozen=True)
class ChunkNoise:
    action: int
    video: int
    flow: int


@dataclass(frozen=True)
class HistoryBlock:
    video: torch.Tensor
    actions: torch.Tensor
    start: int


@dataclass(frozen=True)
class ContextSnapshot:
    text: torch.Tensor
    negative_text: torch.Tensor
    initial_latent: torch.Tensor
    blocks: tuple[HistoryBlock, ...]
    start: int


@dataclass
class FlowReplay:
    context: ContextSnapshot
    sample: torch.Tensor
    next_sample: torch.Tensor
    time: float
    next_time: float
    mask: torch.Tensor
    old_log_prob: torch.Tensor


@dataclass
class ChunkSample:
    route: Route
    actions: torch.Tensor  # physical [8,8], offered to the client
    normalized_plan: torch.Tensor
    context: ContextSnapshot
    noise: ChunkNoise
    flow: FlowReplay | None
    timings: dict[str, float]
    predicted_video: torch.Tensor | None = None


class LingBotVARuntime:
    def __init__(self, model, encoder, normalizer, config):
        from wan_va.utils.scheduler import FlowMatchScheduler

        self.model, self.encoder, self.normalizer, self.config = (
            model,
            encoder,
            normalizer,
            config,
        )
        self.video_scheduler = FlowMatchScheduler(
            shift=config.video_shift, sigma_min=0, extra_one_step=True
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_shift, sigma_min=0, extra_one_step=True
        )
        self.bc_scheduler = FlowMatchScheduler(
            shift=config.action_shift, sigma_min=0, extra_one_step=True
        )
        self.bc_scheduler.set_timesteps(1000, training=True)
        self.history: Cache = ()
        self.blocks: tuple[HistoryBlock, ...] = ()
        self.physical_states = []
        self.pending = None
        self.active = False

    @property
    def reference(self):
        return next(self.model.parent.parameters())

    @torch.no_grad()
    def reset_episode(self, instruction, initial_observation):
        self.encoder.reset()
        self.text, self.text_mask = self.encoder.text(instruction)
        self.negative_text, _ = self.encoder.text("")
        self.initial_latent = self.encoder.images([initial_observation]).detach().cpu()
        self.history, self.blocks = (), ()
        self.physical_states = []
        self.start = 0
        self.pending = None
        self.observation = initial_observation
        self.active = True

    def snapshot(self):
        return ContextSnapshot(
            self.text, self.negative_text, self.initial_latent, self.blocks, self.start
        )

    def _texts(self, context, cfg):
        values = (
            torch.cat((context.text, context.negative_text)) if cfg else context.text
        )
        return values.to(self.reference)

    def _forward(
        self, x, timestep, context, *, history=(), action=False, uncond=False, cfg=False
    ):
        if cfg:
            x = x.repeat(2, 1, 1, 1, 1)
        else:
            history = positive_cache(history)
        return self.model(
            x,
            timestep,
            self._texts(context, cfg),
            action=action,
            uncond=uncond,
            history=history,
            start=context.start,
        )

    @torch.no_grad()
    def prepare_gate_features(self, current_observation):
        if not self.active:
            raise RuntimeError("Reset the episode before preparing a decision.")
        self.observation = current_observation
        state = torch.as_tensor(
            current_observation["state"],
            dtype=torch.float32,
            device=self.reference.device,
        )[None]
        if state.shape != (1, self.config.state_dim) or not torch.isfinite(state).all():
            raise ValueError("Measured state must be joint7 + TCP7 + gripper1.")
        state = self.normalizer.state(state)
        # No episode VAE history, action tokens, main cache or absolute clock.
        latent = self.encoder.images([current_observation], current_only=True).to(
            self.reference
        )
        _, current = self.model(latent, 0, self.text.to(self.reference), start=0)
        layers = []
        for index in self.config.gate_layers:
            kv = current[index]
            bank = KeyValueBank(
                source=KVSource.CURRENT_FRAME_VIDEO,
                key=kv.key.flatten(2).detach(),
                value=kv.value.flatten(2).detach(),
                valid_mask=torch.ones(
                    kv.key.shape[:2], dtype=torch.bool, device=kv.key.device
                ),
                contains_generated_future_video=False,
            )
            layers.append(RouteNeutralVisualLayer(index, bank))
        previous = self.physical_states[-self.config.physical_history :]
        previous = (
            ([previous[0]] * (self.config.physical_history - len(previous)) + previous)
            if previous
            else [state[0].cpu()] * self.config.physical_history
        )
        return RouteNeutralGateFeatures(
            visual=RouteNeutralVisualFeatures(tuple(layers)),
            language=self.text.to(self.reference),
            language_mask=self.text_mask.to(self.reference.device),
            state=state,
            physical_history=torch.stack(previous)[None].to(state),
        )

    @torch.no_grad()
    def _prefill(self, history, block, context):
        cfg = self.config.guidance_scale > 1 or self.config.action_guidance_scale > 1
        block_context = ContextSnapshot(
            context.text, context.negative_text, context.initial_latent, (), block.start
        )
        _, video = self._forward(
            block.video.to(self.reference), 0, block_context, history=history, cfg=cfg
        )
        with_video = append_cache(history, video)
        _, actions = self._forward(
            block.actions.to(self.reference),
            0,
            block_context,
            history=with_video,
            action=True,
            cfg=cfg,
        )
        return append_cache(with_video, actions, max_frames=self.config.history_frames)

    @torch.no_grad()
    def rebuild_history(self, context):
        # Replay the complete causal prefix: retained high-layer keys may themselves
        # contain information from older, since-evicted blocks.
        history = ()
        for block in context.blocks:
            history = self._prefill(history, block, context)
        return history

    @torch.no_grad()
    def uncond_condition(self, context, history):
        if context.start != 0:
            return history
        # The native IDM anchors its first video chunk internally. Only the first
        # UNCOND call needs this separate current-image prefill.
        cfg = self.config.guidance_scale > 1 or self.config.action_guidance_scale > 1
        _, anchor = self._forward(
            context.initial_latent.to(self.reference), 0, context, cfg=cfg
        )
        return append_cache(history, anchor)

    def action_velocity(self, sample, time, context, history, *, uncond):
        cfg = self.config.action_guidance_scale > 1
        timestep = torch.as_tensor(time, device=sample.device) * 1000
        if context.start == 0:
            timestep = timestep.expand(1, sample.shape[2]).clone()
            timestep[:, 0] = 0
        if cfg and torch.is_tensor(timestep) and timestep.ndim == 2:
            timestep = timestep.repeat(2, 1)
        velocity, _ = self._forward(
            sample,
            timestep,
            context,
            history=history,
            action=True,
            uncond=uncond,
            cfg=cfg,
        )
        if cfg:
            return velocity[1:] + self.config.action_guidance_scale * (
                velocity[:1] - velocity[1:]
            )
        return velocity

    @torch.no_grad()
    def sample_from_context(
        self, route, noise, context, *, training=False, history=None
    ):
        route = Route(route)
        history = self.rebuild_history(context) if history is None else history
        reference = self.reference
        timings = {"video_denoise_s": 0.0, "action_denoise_s": 0.0, "video_forwards": 0}
        first = context.start == 0
        predicted = None
        if route is Route.IDM:
            begin = clock_sync(reference)
            shape = (
                1,
                context.initial_latent.shape[1],
                self.config.frame_chunk_size,
                *context.initial_latent.shape[-2:],
            )
            predicted = cpu_noise(shape, noise.video, reference)
            cfg = (
                self.config.guidance_scale > 1 or self.config.action_guidance_scale > 1
            )
            self.video_scheduler.set_timesteps(self.config.video_steps)
            times = torch.cat((self.video_scheduler.timesteps, torch.zeros(1)))
            for index, t in enumerate(times):
                time = (
                    t.expand(2 if cfg else 1, self.config.frame_chunk_size)
                    .to(reference.device)
                    .clone()
                )
                if first:
                    predicted[:, :, :1] = context.initial_latent.to(predicted)
                    time[:, 0] = 0
                velocity, current = self._forward(
                    predicted, time, context, history=history, cfg=cfg
                )
                timings["video_forwards"] += 1
                if index != len(times) - 1:
                    velocity = (
                        velocity[1:]
                        + self.config.guidance_scale * (velocity[:1] - velocity[1:])
                        if self.config.guidance_scale > 1
                        else velocity[:1]
                    )
                    predicted = self.video_scheduler.step(velocity, t, predicted)
                else:
                    condition = append_cache(history, current)
            timings["video_denoise_s"] = clock_sync(reference) - begin
        else:
            condition = self.uncond_condition(context, history)
        begin = clock_sync(reference)
        sample = cpu_noise(
            (1, 30, self.config.frame_chunk_size, self.config.action_per_frame, 1),
            noise.action,
            reference,
        )
        mask = valid_action_mask(sample, first)
        sample *= mask
        self.action_scheduler.set_timesteps(self.config.action_steps)
        times = torch.cat((self.action_scheduler.sigmas, torch.zeros(1)))
        generator = torch.Generator().manual_seed(int(noise.flow))
        selected = (
            int(torch.randint(self.config.action_steps - 1, (), generator=generator))
            if training and route is Route.UNCOND
            else -1
        )
        replay = None
        for index in range(self.config.action_steps):
            time, next_time = float(times[index]), float(times[index + 1])
            velocity = self.action_velocity(
                sample, time, context, condition, uncond=route is Route.UNCOND
            )
            if index == selected:
                mean, std = flow_sde_mean_std(
                    sample,
                    velocity,
                    time=torch.tensor(time, device=sample.device),
                    next_time=torch.tensor(next_time, device=sample.device),
                    noise_level=self.config.flow_noise,
                )
                sampled = (
                    (mean + cpu_noise(sample.shape, noise.flow + 1, sample) * std)
                    * mask
                ).to(reference)
                log_prob = gaussian_log_prob(sampled, mean, std)[mask].sum()
                replay = FlowReplay(
                    context,
                    sample.cpu(),
                    sampled.cpu(),
                    time,
                    next_time,
                    mask.cpu(),
                    log_prob.cpu(),
                )
                sample = sampled * mask
            else:
                sample = (sample + (next_time - time) * velocity) * mask
        timings["action_denoise_s"] = clock_sync(reference) - begin
        plan = flatten_actions(sample)
        start = self.config.action_per_frame if first else 0
        actions = self.normalizer.decode(
            plan[start : start + self.config.execution_horizon]
        ).cpu()
        return ChunkSample(
            route,
            actions,
            sample.cpu(),
            context,
            noise,
            replay,
            timings,
            None if predicted is None else predicted.cpu(),
        )

    def sample_chunk(self, route, noise, training=False):
        if not self.active or self.pending is not None:
            raise RuntimeError(
                "Reset or commit the preceding execution before sampling."
            )
        self.pending = self.sample_from_context(
            route, noise, self.snapshot(), training=training, history=self.history
        )
        state = self.normalizer.state(
            torch.as_tensor(self.observation["state"], dtype=torch.float32)
        )
        self.physical_states = (self.physical_states + [state])[
            -self.config.physical_history :
        ]
        return self.pending

    @torch.no_grad()
    def commit_execution(self, observed_frames, executed_actions, *, terminated=False):
        if self.pending is None:
            raise RuntimeError("No sampled action chunk is awaiting execution.")
        count = len(executed_actions)
        if (
            count != len(observed_frames)
            or not 0 <= count <= self.config.execution_horizon
        ):
            raise ValueError(
                "Commit must contain exactly the feedback for executed actions."
            )
        if not terminated and count != self.config.execution_horizon:
            raise ValueError(
                "A nonterminal execution must complete the eight-action prefix."
            )
        begin = clock_sync(self.reference)
        if terminated:
            self.pending = None
            self.active = False
            return {"executed_steps": count, "history_update_s": 0.0}
        video = self.encoder.images(observed_frames).to(self.reference)
        actions = action_tensor(
            self.normalizer.encode(
                torch.as_tensor(executed_actions, dtype=torch.float32)
            )
        ).to(self.reference)
        if self.start == 0:
            video = torch.cat((self.initial_latent.to(video), video), dim=2)
            actions = torch.cat((torch.zeros_like(actions[:, :, :1]), actions), dim=2)
        if video.shape[2] != actions.shape[2]:
            raise ValueError("VAE frames and executed action groups are not aligned.")
        block = HistoryBlock(video.detach().cpu(), actions.detach().cpu(), self.start)
        self.history = self._prefill(self.history, block, self.snapshot())
        self.blocks += (block,)
        self.start += video.shape[2]
        self.observation = observed_frames[-1]
        self.pending = None
        return {
            "executed_steps": count,
            "history_update_s": clock_sync(self.reference) - begin,
        }

    def replay_uncond_transition(self, replay, *, history=None):
        context = replay.context
        history = self.rebuild_history(context) if history is None else history
        condition = self.uncond_condition(context, history)
        sample = replay.sample.to(self.reference)
        velocity = self.action_velocity(
            sample, replay.time, context, condition, uncond=True
        )
        mean, std = flow_sde_mean_std(
            sample,
            velocity,
            time=torch.tensor(replay.time, device=sample.device),
            next_time=torch.tensor(replay.next_time, device=sample.device),
            noise_level=self.config.flow_noise,
        )
        return gaussian_log_prob(replay.next_sample.to(sample), mean, std)[
            replay.mask.to(sample.device)
        ].sum()

    def bc_loss(self, context, target, seed, *, history=None):
        history = self.rebuild_history(context) if history is None else history
        history = self.uncond_condition(context, history)
        target = target.to(self.reference)
        mask = valid_action_mask(target, context.start == 0)
        generator = torch.Generator().manual_seed(int(seed))
        index = int(torch.randint(1000, (), generator=generator))
        t = self.bc_scheduler.timesteps[index].to(target.device)
        sigma = float(self.bc_scheduler.sigmas[index])
        noise = cpu_noise(target.shape, seed + 1, target)
        sample = ((1 - sigma) * target + sigma * noise) * mask
        velocity = self.action_velocity(sample, sigma, context, history, uncond=True)
        loss = (velocity.float() - (noise - target).float()).square()[mask].mean()
        return loss * self.bc_scheduler.training_weight(t.reshape(1))[0]
