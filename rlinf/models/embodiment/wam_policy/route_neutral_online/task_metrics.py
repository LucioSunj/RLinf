# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Detached task statistics, separate from model features and loss reduction."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.algorithms.fastwam_dual_ppo import (
    compute_gate_ppo_loss,
    compute_uncond_flow_ppo_loss,
)
from rlinf.models.embodiment.wam_policy.contracts import WAMRoute


@torch.no_grad()
def summarize_task_rollout(batch: dict[str, Any]) -> dict[str, float]:
    """Summarize actual episode starts and valid chunks before compaction."""

    forward = batch["forward_inputs"]
    task_ids = forward["multitask_task_id"].long()
    route = batch["route_info"]
    if task_ids.shape != route.route_used.shape or not torch.equal(
        task_ids, task_ids[:1].expand_as(task_ids)
    ):
        raise ValueError("Task identity changed inside a non-auto-reset rollout.")
    valid = batch["loss_mask"].bool().reshape(*task_ids.shape, -1).all(dim=-1)
    gate_valid = batch["gate_valid_mask"].bool().reshape_as(valid)
    uncond = valid & (route.route_used == int(WAMRoute.UNCOND))
    teacher = forward["online_idm_bc_teacher_present"].bool() & valid
    if not torch.equal(teacher, uncond):
        raise ValueError("Valid task UNCOND chunks and teacher targets do not align.")
    q = batch["emitted_gate"].behavior_probability
    raw_advantage = (batch["returns"] - batch["prev_values"][:-1]).reshape_as(task_ids)
    episode_ids = forward["multitask_episode_slot_id"]
    if torch.unique(episode_ids[0]).numel() != task_ids.shape[1]:
        raise ValueError("Duplicate episode slot identities after trajectory merge.")
    metrics: dict[str, float] = {}
    success_rates = []
    idm_rates = []
    for task in range(10):
        prefix = f"task/{task}"
        episodes = task_ids[0] == task
        selected = valid & (task_ids == task)
        eligible = gate_valid & (task_ids == task)
        n = int(selected.sum())
        episode_count = int(episodes.sum())
        if not episode_count or not n:
            raise ValueError(f"Task {task} lost all episode starts or valid chunks.")
        success = int(forward["multitask_episode_success"][0][episodes].sum())
        idm = int((selected & ~uncond).sum())
        advantage = raw_advantage[selected].float()
        metrics.update(
            {
                f"{prefix}/episode_slots": float(episode_count),
                f"{prefix}/successes": float(success),
                f"{prefix}/failures": float(
                    forward["multitask_episode_failed"][0][episodes].sum()
                ),
                f"{prefix}/truncations": float(
                    forward["multitask_episode_truncated"][0][episodes].sum()
                ),
                f"{prefix}/training_success": success / episode_count,
                f"{prefix}/valid_chunks": float(n),
                f"{prefix}/uncond_chunks": float((selected & uncond).sum()),
                f"{prefix}/teacher_chunks": float((selected & teacher).sum()),
                f"{prefix}/gate_chunks": float(eligible.sum()),
                f"{prefix}/mean_behavior_probability": float(
                    q[eligible].float().mean()
                ),
                f"{prefix}/realized_idm_fraction": idm / n,
                f"{prefix}/critic_preupdate_mse": float(advantage.square().mean()),
                f"{prefix}/raw_advantage_mean": float(advantage.mean()),
                f"{prefix}/raw_advantage_std": float(advantage.std(unbiased=False)),
            }
        )
        success_rates.append(success / episode_count)
        idm_rates.append(idm / n)
    metrics["task_macro/training_success"] = sum(success_rates) / 10
    metrics["task_macro/idm_fraction"] = sum(idm_rates) / 10
    metrics["task_global/chunk_weighted_idm_fraction"] = float(
        (valid & ~uncond).sum()
    ) / int(valid.sum())
    metrics["task_global/episode_slots"] = float(task_ids.shape[1])
    metrics["task_global/valid_chunks"] = float(valid.sum())
    return metrics


@torch.no_grad()
def accumulate_task_losses(
    totals: dict[str, float],
    batch: dict[str, Any],
    output: dict[str, torch.Tensor],
    cfg: Any,
) -> None:
    """Reuse the actual PPO loss functions on detached, task-masked outputs."""

    def cpu(value):
        return value.detach().cpu()

    task_ids = cpu(batch["forward_inputs"]["multitask_task_id"]).long().reshape(-1)
    route = cpu(batch["route_info"].route_used)
    outputs = {
        key: cpu(output[key]).float()
        for key in (
            "gate_logprobs",
            "gate_base_probabilities",
            "gate_behavior_probabilities",
            "flow_logprobs",
            "online_idm_bc_per_sample_loss",
        )
    }
    for task in task_ids.unique().tolist():
        selected = task_ids == task
        gate_mask = cpu(batch["gate_valid_mask"]).bool().reshape(-1) & selected
        flow_mask = cpu(batch["flow_valid_mask"]).bool().reshape(-1) & selected
        _, gate = compute_gate_ppo_loss(
            logprobs=outputs["gate_logprobs"],
            old_logprobs=cpu(batch["emitted_gate"].old_logprob).float(),
            advantages=cpu(batch["gate_advantages"]).float().reshape(-1),
            valid_mask=gate_mask,
            clip_ratio_low=float(cfg.algorithm.gate_ppo.clip_ratio_low),
            clip_ratio_high=float(cfg.algorithm.gate_ppo.clip_ratio_high),
            base_probabilities=outputs["gate_base_probabilities"],
            behavior_probabilities=outputs["gate_behavior_probabilities"],
        )
        _, flow = compute_uncond_flow_ppo_loss(
            logprobs=outputs["flow_logprobs"],
            old_logprobs=cpu(batch["prev_logprobs"]).float(),
            advantages=cpu(batch["flow_advantages"]).float(),
            route_used=route,
            valid_mask=flow_mask,
            clip_ratio_low=float(cfg.algorithm.uncond_flow_ppo.clip_ratio_low),
            clip_ratio_high=float(cfg.algorithm.uncond_flow_ppo.clip_ratio_high),
        )
        for owner, values, original in (
            ("gate", gate, "gate"),
            ("flow", flow, "uncond_flow"),
        ):
            count = float(values[f"{original}/sample_count"])
            for name, value in (
                ("count", count),
                ("loss_sum", float(values[f"{original}/policy_loss"]) * count),
            ):
                key = f"task/{task}/{owner}_{name}"
                totals[key] = totals.get(key, 0.0) + value
        bc_mask = flow_mask & (route == int(WAMRoute.UNCOND))
        for name, value in (
            ("count", float(bc_mask.sum())),
            (
                "loss_sum",
                float(outputs["online_idm_bc_per_sample_loss"][bc_mask].sum()),
            ),
        ):
            key = f"task/{task}/bc_{name}"
            totals[key] = totals.get(key, 0.0) + value


def finalize_task_losses(totals: dict[str, float]) -> dict[str, float]:
    """Convert additive counts and loss numerators into task-conditioned means."""

    metrics = {}
    for task in range(10):
        for owner in ("gate", "flow", "bc"):
            prefix = f"task/{task}/{owner}"
            count = totals.get(f"{prefix}_count", 0.0)
            metrics[f"{prefix}_sample_count"] = count
            metrics[f"{prefix}_raw_loss"] = totals.get(f"{prefix}_loss_sum", 0.0) / max(
                count, 1.0
            )
    return metrics
