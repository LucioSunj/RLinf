# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Complete episodes collected sequentially from exactly one physical executor."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from rlinf.algorithms.advantages import (
    apply_fastwam_chunk_cost,
    compute_gae_advantages_and_returns,
)
from rlinf.envs.pad_realworld.contracts import (
    ActionProposal,
    EpisodeOutcome,
    ExecutionReceipt,
)
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import (
    ONLINE_IDM_BC_FLOW_VALID,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.actor import (
    align_current_step_trainable_advantages,
)


@dataclass
class EpisodeRollout:
    episode_id: str
    task_id: str
    layout_id: str
    proposals: list[ActionProposal] = field(default_factory=list)
    receipts: list[ExecutionReceipt] = field(default_factory=list)
    outcome: EpisodeOutcome | None = None
    wallclock_seconds: float = 0.0
    manual_reset_seconds: float | None = None

    @property
    def trainable(self):
        # Unknown/intervened attempts have no trustworthy bootstrap in v1.
        return (
            self.outcome is not None
            and self.outcome.trainable
            and all(r.chunk_valid for r in self.receipts)
        )


class SequentialRobotCollector:
    """Use operator-ready events and whole attempts, never synthetic env workers."""

    physical_num_envs = 1

    def __init__(self, env, policy, *, wait_for_outcome=None):
        self.env = env
        self.policy = policy
        self.wait_for_outcome = wait_for_outcome

    def collect(
        self,
        attempts,
        *,
        mode="train",
        method="learned",
        random_idm_probability=0.5,
        delay_seconds=0.0,
    ):
        episodes = []
        version = self.policy.actor_version
        for attempt in attempts:
            attempt = dict(attempt)
            reset_seconds = attempt.pop(
                "manual_reset_seconds", 0.0 if self.env.backend.is_mock else None
            )
            self.env.begin_episode(**attempt)
            self.policy.begin_episode()
            episode = EpisodeRollout(
                attempt["episode_id"],
                attempt["task_id"],
                attempt["layout_id"],
                manual_reset_seconds=reset_seconds,
            )
            started = self.env.backend.clock.now()
            while self.env.outcome is None:
                if self.policy.actor_version != version:
                    raise RuntimeError(
                        "Policy version changed while an attempt was active."
                    )
                try:
                    snapshot = self.env.observe(len(episode.proposals))
                except (ConnectionError, TimeoutError, ValueError) as error:
                    self.env.confirm_outcome(
                        EpisodeOutcome(
                            episode.episode_id, "censored", detail=str(error)
                        )
                    )
                    break
                if delay_seconds:
                    self.env.backend.clock.sleep(delay_seconds)
                proposal = self.policy.propose(
                    snapshot,
                    mode=mode,
                    method=method,
                    random_idm_probability=random_idm_probability,
                )
                if self.env.backend.is_mock:
                    self.env.backend.clock.sleep(proposal.timings["prediction_total"])
                receipt = self.env.execute_prefix(proposal)
                episode.proposals.append(proposal)
                episode.receipts.append(receipt)
                if receipt.stopped_reason is not None and self.env.outcome is None:
                    if self.wait_for_outcome is None:
                        raise RuntimeError(
                            "Policy stopped; terminal scoring is pending for "
                            + episode.episode_id
                        )
                    self.env.confirm_outcome(self.wait_for_outcome(episode.episode_id))
                    episode.receipts[-1] = replace(
                        receipt,
                        outcome=self.env.outcome,
                        stopped_reason=self.env.outcome.kind,
                    )
            episode.outcome = self.env.finish_episode()
            episode.wallclock_seconds = self.env.backend.clock.now() - started
            episodes.append(episode)
        return episodes


@torch.no_grad()
def complete_auxiliary_inference(episodes, policy):
    """Fill old values and required U teachers only after all motion is finished."""
    started = time.perf_counter()
    teacher_seconds = 0.0
    critic_seconds = 0.0
    for episode in episodes:
        if not episode.trainable:
            continue
        for proposal in episode.proposals:
            replay = proposal.replay
            if proposal.actor_version != policy.actor_version:
                raise RuntimeError(
                    "Deferred inference requires the unchanged rollout policy version."
                )
            if replay["old_values_complete"]:
                raise RuntimeError(
                    "Auxiliary inference was already completed for this batch."
                )
            policy.runtime.synchronize()
            critic_started = time.perf_counter()
            values, prefix = policy.critic.predict_value_batch(
                policy.runtime.critic_observation(env_obs=replay["saved_env_obs"]),
                return_prefix=True,
            )
            policy.runtime.synchronize()
            critic_seconds += time.perf_counter() - critic_started
            if not torch.isfinite(values).all() or not torch.isfinite(prefix).all():
                raise FloatingPointError(
                    "Old critic produced non-finite values/prefix."
                )
            replay["prev_values"] = values.detach().reshape(1, 1)
            replay["forward_inputs"]["critic_prefix"] = prefix.detach()
            request = replay["teacher_request"]
            if proposal.actor_version >= policy.critic_warmup.runner_updates:
                policy.runtime.synchronize()
                teacher_started = time.perf_counter()
                completed = policy.runtime.complete_teacher(
                    sample=request["sample"],
                    seeds=request["seeds"],
                    route=replay["route_info"].route_used,
                    actor_version=request["actor_version"],
                )
                policy.runtime.synchronize()
                replay["forward_inputs"].update(completed.forward_inputs)
                teacher_seconds += time.perf_counter() - teacher_started
            replay["forward_inputs"][ONLINE_IDM_BC_FLOW_VALID] = torch.tensor([True])
            replay["old_values_complete"] = True
    return {
        "teacher_seconds": teacher_seconds,
        "critic_seconds": critic_seconds,
        "auxiliary_seconds": time.perf_counter() - started,
    }


def prepare_training_batch(episodes, *, decision, training):
    """Concatenate variable-length attempts as [T,1] with true terminal breaks."""
    rows, rewards, terminals, masks = [], [], [], []
    for episode in episodes:
        if not episode.trainable:
            continue
        for index, (proposal, receipt) in enumerate(
            zip(episode.proposals, episode.receipts, strict=True)
        ):
            if not proposal.replay.get("old_values_complete"):
                raise RuntimeError(
                    "GAE cannot consume collection-time zero value placeholders."
                )
            terminal = index == len(episode.proposals) - 1
            rows.append(proposal)
            rewards.append(float(terminal and episode.outcome.kind == "success"))
            terminals.append(terminal)
            masks.append(receipt.executed_prefix_mask)
    if not rows:
        raise RuntimeError(
            "No complete autonomous episode is eligible for PPO; batch retained without update."
        )
    route = ChunkRouteRecord.stack([row.replay["route_info"] for row in rows])
    emitted = GateDecisionRecord.stack([row.replay["emitted_gate"] for row in rows])
    device = rows[0].replay["prev_values"].device
    chunk_valid = torch.ones(len(rows), 1, dtype=torch.bool, device=device)
    rewards = torch.tensor(rewards, device=device).reshape(-1, 1, 1)
    cost = apply_fastwam_chunk_cost(
        environment_rewards=rewards,
        route_used=route.route_used,
        idm_cost=decision.idm_cost,
        uncond_cost=decision.uncond_cost,
        valid_mask=chunk_valid,
    )
    values = torch.cat(
        [row.replay["prev_values"] for row in rows] + [torch.zeros(1, 1, device=device)]
    )
    dones = torch.tensor([False] + terminals, dtype=torch.bool, device=device).reshape(
        -1, 1
    )
    advantages, returns = compute_gae_advantages_and_returns(
        rewards=cost.rewards[..., 0],
        values=values,
        dones=dones,
        gamma=training["gamma"],
        gae_lambda=training["gae_lambda"],
        loss_mask=chunk_valid,
        normalize_advantages=True,
        normalization_std_floor=training.get("advantage_normalization_std_floor", 1e-4),
    )
    alignment = align_current_step_trainable_advantages(
        advantages=advantages[..., None],
        route=route,
        emitted=emitted,
        loss_mask=chunk_valid[..., None],
    )
    return {
        "rows": rows,
        "rewards": cost.rewards,
        "costs": cost.costs,
        "returns": returns,
        "advantages": advantages,
        "prev_values": values,
        "dones": dones,
        "chunk_valid": chunk_valid,
        "route_info": route,
        "emitted_gate": emitted,
        "executed_prefix_mask": torch.as_tensor(np.stack(masks))[:, None],
        "alignment": alignment,
    }


def collate_microbatch(batch, indices):
    """Flatten time only for the learner; no synthetic physical parallelism."""
    rows = [batch["rows"][index] for index in indices]
    inputs = {
        key: torch.cat([row.replay["forward_inputs"][key] for row in rows], 0)
        for key in rows[0].replay["forward_inputs"]
    }
    return {
        "forward_inputs": inputs,
        "route_info": ChunkRouteRecord.cat([r.replay["route_info"] for r in rows]),
        "emitted_gate": GateDecisionRecord.cat(
            [r.replay["emitted_gate"] for r in rows]
        ),
        "prev_logprobs": torch.cat([r.replay["prev_logprobs"] for r in rows]),
        "prev_values": torch.cat([r.replay["prev_values"] for r in rows]),
        "returns": batch["returns"][indices],
        "flow_advantages": batch["alignment"].flow_advantages[indices, 0],
        "gate_advantages": batch["alignment"].gate_advantages[indices, 0],
        "flow_valid_mask": batch["alignment"].flow_valid_mask[indices, 0],
        "gate_valid_mask": batch["alignment"].gate_valid_mask[indices, 0],
        "loss_mask": batch["chunk_valid"][indices],
    }
