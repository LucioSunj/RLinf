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

"""Single-robot round collection, independent PPO losses and durable resume."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.fastwam_dual_ppo import compute_fastwam_dual_ppo_loss

from .contracts import Route
from .runtime import ChunkNoise, clock_sync


@dataclass(frozen=True)
class TrainingConfig:
    updates: int = 30
    episodes_per_update: int = 4
    warmup_updates: int = 5
    save_interval: int = 5
    minibatch_size: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    gate_lr: float = 3e-5
    lora_lr: float = 1e-5
    critic_lr: float = 1e-4
    gate_entropy: float = 0.01
    flow_weight: float = 1.0
    bc_weight: float = 0.2
    value_weight: float = 0.5
    max_grad_norm: float = 1.0


@dataclass
class B50Controller:
    target: float = 0.50
    half_width: float = 0.03
    gain: float = 0.0025
    max_increment: float = 0.005
    max_price: float = 0.1
    price: float = 0.0
    last_side: int = 0
    reversals: int = 0

    def update(self, behavior_probability):
        p = float(behavior_probability)
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Controller feedback must be a finite probability.")
        error = p - self.target
        side = 1 if error > self.half_width else (-1 if error < -self.half_width else 0)
        if side == 0:
            return self.price
        if self.last_side and side != self.last_side:
            self.price = 0.0
            self.reversals += 1
        # Bound the NEW feedback, after clearing history. Do not bound the reset.
        increment = np.clip(self.gain * error, -self.max_increment, self.max_increment)
        self.price = float(
            np.clip(self.price + increment, -self.max_price, self.max_price)
        )
        self.last_side = side
        return self.price

    def cost(self, route):
        return max(0.0, self.price) if route is Route.IDM else max(0.0, -self.price)


def padded_minibatches(size, batch_size, generator):
    """The last batch has explicit invalid slots, never duplicate training data."""
    order = torch.randperm(size, generator=generator).tolist()
    for start in range(0, size, batch_size):
        selected = order[start : start + batch_size]
        mask = torch.arange(batch_size) < len(selected)
        yield selected + [-1] * (batch_size - len(selected)), mask


def round_advantages(episodes, controller, config, warmup):
    decisions, advantages, returns = [], [], []
    for episode in episodes:
        valid = []
        for decision in episode:
            if not decision.autonomous or not decision.executed_steps:
                break
            valid.append(decision)
        if not valid:
            continue
        rewards = torch.tensor(
            [
                [d.reward - (0.0 if warmup else controller.cost(d.sample.route))]
                for d in valid
            ]
        )
        values = torch.tensor([[d.value] for d in valid] + [[0.0]])
        dones = torch.zeros(len(valid) + 1, 1, dtype=torch.bool)
        dones[-1] = True
        adv, ret = compute_gae_advantages_and_returns(
            rewards,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            values=values,
            dones=dones,
            normalize_advantages=False,
        )
        decisions.extend(valid)
        advantages.append(adv.flatten())
        returns.append(ret.flatten())
    if not decisions:
        raise ValueError(
            "This round contains no autonomous chunks; no update is committed."
        )
    adv, ret = torch.cat(advantages), torch.cat(returns)
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
    return decisions, adv, ret


class SingleRobotTrainer:
    """Collocated actor/rollout: parameters change only after four full episodes."""

    def __init__(self, policy, config=TrainingConfig(), *, task="grasp_place"):
        if policy.critic is None:
            raise ValueError("Training requires a Pi0.5 critic.")
        self.policy, self.config, self.task = policy, config, task
        self.controller = B50Controller()
        self.step = 0
        self.optimizers = {
            "gate": torch.optim.AdamW(
                policy.gate.parameters(), lr=config.gate_lr, weight_decay=0
            ),
            "lora": torch.optim.AdamW(
                policy.core.adapters.parameters(), lr=config.lora_lr, weight_decay=0
            ),
            "critic": torch.optim.AdamW(
                policy.critic.value_head.parameters(),
                lr=config.critic_lr,
                weight_decay=0,
            ),
        }
        self.schedulers = {
            k: torch.optim.lr_scheduler.LambdaLR(v, lambda _: 1.0)
            for k, v in self.optimizers.items()
        }
        self.shuffle_rng = torch.Generator().manual_seed(42)
        self.teacher_rng = torch.Generator().manual_seed(424242)

    @property
    def warmup(self):
        return self.step < self.config.warmup_updates

    def update(self, episodes):
        if len(episodes) != self.config.episodes_per_update:
            raise ValueError(
                "Collect four complete episodes before updating the actor."
            )
        if self.policy.pending_decision is not None or self.policy.runtime.active:
            raise ValueError(
                "Robot sampling must finish before teacher queries or training."
            )
        if self.step >= self.config.updates:
            raise ValueError("The configured terminal update is already complete.")
        if any(not e or not e[-1].terminal for e in episodes):
            raise ValueError(
                "Each collected episode must have a terminal execution record."
            )
        if any(d.actor_version != self.step for e in episodes for d in e):
            raise ValueError("The entire round must use one fixed actor version.")
        warmup = self.warmup
        decisions, advantages, returns = round_advantages(
            episodes, self.controller, self.config, warmup
        )
        device = self.policy.runtime.reference.device
        advantages, returns = advantages.to(device), returns.to(device)
        begin = clock_sync(self.policy.runtime.reference)
        teachers = {}
        if not warmup:
            teacher_history, previous_context = (), None
            for index, d in enumerate(decisions):
                if d.sample.route is Route.UNCOND:
                    if d.sample.flow is None:
                        raise ValueError(
                            "A joint-update UNCOND chunk is missing its sampled Flow transition."
                        )
                    video_seed = int(
                        torch.randint(0, 2**62, (), generator=self.teacher_rng)
                    )
                    noise = ChunkNoise(
                        d.sample.noise.action, video_seed, d.sample.noise.flow
                    )
                    context = d.sample.context
                    same_episode = (
                        previous_context is not None
                        and context.initial_latent is previous_context.initial_latent
                    )
                    offset = len(previous_context.blocks) if same_episode else 0
                    if not same_episode:
                        teacher_history = ()
                    for block in context.blocks[offset:]:
                        teacher_history = self.policy.runtime._prefill(
                            teacher_history, block, context
                        )
                    previous_context = context
                    teachers[index] = self.policy.runtime.sample_from_context(
                        Route.IDM, noise, context, history=teacher_history
                    ).normalized_plan
            del teacher_history, previous_context
        metrics = {
            "teacher_s": clock_sync(self.policy.runtime.reference) - begin,
            "teacher_samples": len(teachers),
            "warmup": int(warmup),
            "price_used": self.controller.price,
        }
        begin = clock_sync(self.policy.runtime.reference)
        losses = []
        for indices, valid_mask in padded_minibatches(
            len(decisions), self.config.minibatch_size, self.shuffle_rng
        ):
            active = [i for i, valid in zip(indices, valid_mask) if valid]
            uncond_count = sum(i in teachers for i in active)
            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            for i in active:
                d = decisions[i]
                if warmup:
                    value = self.policy.critic.value_from_features(
                        d.critic_features.to(device)
                    ).reshape(1)
                else:
                    history = (
                        self.policy.runtime.rebuild_history(d.sample.context)
                        if i in teachers
                        else None
                    )
                    result = self.policy(decision=d, history=history)
                    value = result["value"]
                value_loss = (
                    (value - returns[i]).square().mean()
                    * self.config.value_weight
                    / len(active)
                )
                loss = value_loss
                if not warmup:
                    gate, flow = result["gate"], result["flow_log_prob"]
                    old_flow = (
                        flow.detach()
                        if d.sample.flow is None
                        else d.sample.flow.old_log_prob.to(device).reshape(1)
                    )
                    policy_loss, _ = compute_fastwam_dual_ppo_loss(
                        gate_logprobs=gate.logprob,
                        gate_old_logprobs=d.log_prob.to(device),
                        gate_advantages=advantages[i : i + 1],
                        gate_valid_mask=torch.ones(1, dtype=torch.bool, device=device),
                        gate_clip_ratio_low=self.config.clip,
                        gate_clip_ratio_high=self.config.clip,
                        flow_logprobs=flow,
                        flow_old_logprobs=old_flow,
                        flow_advantages=advantages[i : i + 1],
                        route_used=torch.tensor(
                            [int(d.sample.route is Route.IDM)], device=device
                        ),
                        flow_clip_ratio_low=self.config.clip,
                        flow_clip_ratio_high=self.config.clip,
                        gate_base_probabilities=gate.base_probability,
                        gate_behavior_probabilities=gate.behavior_probability,
                        gate_entropy_coefficient=self.config.gate_entropy,
                        flow_loss_coefficient=self.config.flow_weight,
                        gate_selected_loss_scale=1 / len(active),
                        flow_selected_loss_scale=1 / max(1, uncond_count),
                    )
                    loss = loss + policy_loss
                    if i in teachers:
                        loss = (
                            loss
                            + self.config.bc_weight
                            * self.policy.runtime.bc_loss(
                                d.sample.context,
                                teachers[i],
                                d.sample.noise.flow + self.step,
                                history=history,
                            )
                            / uncond_count
                        )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Nonfinite update loss; the runner step was not committed."
                    )
                loss.backward()
                losses.append(float(loss.detach()))
            for name in ("critic",) if warmup else ("gate", "lora", "critic"):
                parameters = [
                    p
                    for group in self.optimizers[name].param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizers[name].step()
                self.schedulers[name].step()
        if not warmup:
            self.controller.update(np.mean([d.probability for d in decisions]))
        self.step += 1
        self.policy.actor_version = self.step
        metrics.update(
            update=self.step,
            chunks=len(decisions),
            update_s=clock_sync(self.policy.runtime.reference) - begin,
            price_next=self.controller.price,
            reversals=self.controller.reversals,
            mean_behavior_probability=float(
                np.mean([d.probability for d in decisions])
            ),
            idm_rate=float(np.mean([d.sample.route is Route.IDM for d in decisions])),
            successful_episodes=sum(e[-1].reward == 1 for e in episodes),
            loss=sum(losses),
        )
        return metrics

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        policy = self.policy
        payload = {
            "model_type": "lingbot_va_route_neutral",
            "stage": "rl",
            "task": self.task,
            "step": self.step,
            "actor_version": policy.actor_version,
            "parent_path": policy.parent_path,
            "routing_config": policy.config.to_dict(),
            "training_config": asdict(self.config),
            "asset_references": policy.asset_references,
            "normalizer": policy.runtime.normalizer.to_dict(),
            "gate": policy.gate.state_dict(),
            "lora": policy.core.adapters.state_dict(),
            "critic": policy.critic.value_head.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "schedulers": {k: v.state_dict() for k, v in self.schedulers.items()},
            "controller": asdict(self.controller),
            "policy_rng": policy.rng.get_state(),
            "shuffle_rng": self.shuffle_rng.get_state(),
            "teacher_rng": self.teacher_rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state(policy.runtime.reference.device)
            if policy.runtime.reference.is_cuda
            else None,
        }
        temporary = path / "training.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(path / "training.pt")
        return path / "training.pt"

    def load(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "model_type": "lingbot_va_route_neutral",
            "stage": "rl",
            "task": self.task,
            "parent_path": self.policy.parent_path,
            "routing_config": self.policy.config.to_dict(),
            "asset_references": self.policy.asset_references,
            "training_config": asdict(self.config),
            "normalizer": self.policy.runtime.normalizer.to_dict(),
        }
        for key, value in expected.items():
            if payload[key] != value:
                raise ValueError(f"Resume changes the experiment's {key}.")
        self.policy.gate.load_state_dict(payload["gate"], strict=True)
        self.policy.core.adapters.load_state_dict(payload["lora"], strict=True)
        self.policy.critic.value_head.load_state_dict(payload["critic"], strict=True)
        for k, optimizer in self.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][k])
            self.schedulers[k].load_state_dict(payload["schedulers"][k])
        self.controller = B50Controller(**payload["controller"])
        self.step = int(payload["step"])
        self.policy.actor_version = int(payload["actor_version"])
        self.policy.rng.set_state(payload["policy_rng"])
        self.shuffle_rng.set_state(payload["shuffle_rng"])
        self.teacher_rng.set_state(payload["teacher_rng"])
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        if payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state(
                payload["cuda_rng"], self.policy.runtime.reference.device
            )
        runtime = self.policy.runtime
        runtime.history, runtime.blocks, runtime.physical_states = (), (), []
        runtime.pending, runtime.active, runtime.start = None, False, 0
        runtime.encoder.reset()
        self.policy.pending_decision = None
