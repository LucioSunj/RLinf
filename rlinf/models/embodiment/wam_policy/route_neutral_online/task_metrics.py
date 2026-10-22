# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Detached task statistics, separate from model features and loss reduction."""

from __future__ import annotations

from typing import Any

import torch

from rlinf.algorithms.fastwam_dual_ppo import (
    clipped_ppo_objective,
    mask_executed_flow_statistics,
)
from rlinf.models.embodiment.wam_policy.contracts import WAMRoute


@torch.no_grad()
def summarize_task_rollout(
    batch: dict[str, Any], *, active_task_ids: tuple[int, ...] | None = None
) -> dict[str, float]:
    """Summarize actual episode starts and valid chunks before compaction."""

    forward = batch["forward_inputs"]
    task_ids = forward["multitask_task_id"].long()
    route = batch["route_info"]
    if task_ids.shape != route.route_used.shape or not torch.equal(
        task_ids, task_ids[:1].expand_as(task_ids)
    ):
        raise ValueError("Task identity changed inside a non-auto-reset rollout.")
    valid = (
        batch.get("chunk_valid", batch["loss_mask"])
        .bool()
        .reshape(*task_ids.shape, -1)
        .all(dim=-1)
    )
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
    tasks = tuple(range(10)) if active_task_ids is None else active_task_ids
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Active task IDs must be nonempty and unique.")
    if bool(
        (
            valid & ~torch.isin(task_ids, torch.tensor(tasks, device=task_ids.device))
        ).any()
    ):
        raise ValueError("Valid rollout contains a task outside the active task table.")
    for task in tasks:
        prefix = f"task/{task}"
        episodes = task_ids[0] == task
        selected = valid & (task_ids == task)
        eligible = gate_valid & selected
        n = int(selected.sum())
        episode_count = int(episodes.sum())
        if active_task_ids is None and (not episode_count or not n):
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
                f"{prefix}/training_success": success / max(episode_count, 1),
                f"{prefix}/valid_chunks": float(n),
                f"{prefix}/idm_chunks": float(idm),
                f"{prefix}/eligible_idm_chunks": float((eligible & ~uncond).sum()),
                f"{prefix}/forced_chunks": float(
                    (selected & route.route_was_forced).sum()
                ),
                f"{prefix}/behavior_probability_sum": float(q[eligible].double().sum()),
                f"{prefix}/uncond_chunks": float((selected & uncond).sum()),
                f"{prefix}/teacher_chunks": float((selected & teacher).sum()),
                f"{prefix}/gate_chunks": float(eligible.sum()),
                f"{prefix}/mean_behavior_probability": float(q[eligible].float().mean())
                if bool(eligible.any())
                else 0.0,
                f"{prefix}/realized_idm_fraction": idm / max(n, 1),
                f"{prefix}/critic_preupdate_mse": float(advantage.square().mean())
                if n
                else 0.0,
                f"{prefix}/raw_advantage_mean": float(advantage.mean()) if n else 0.0,
                f"{prefix}/raw_advantage_std": float(advantage.std(unbiased=False))
                if n
                else 0.0,
            }
        )
        if episode_count:
            success_rates.append(success / episode_count)
        if n:
            idm_rates.append(idm / n)
    metrics["task_macro/training_success"] = sum(success_rates) / max(
        len(success_rates), 1
    )
    metrics["task_macro/idm_fraction"] = sum(idm_rates) / max(len(idm_rates), 1)
    metrics["task_global/chunk_weighted_idm_fraction"] = float(
        (valid & ~uncond).sum()
    ) / max(int(valid.sum()), 1)
    metrics["task_global/episode_slots"] = float(task_ids.shape[1])
    metrics["task_global/valid_chunks"] = float(valid.sum())
    return metrics


@torch.no_grad()
def accumulate_task_losses(
    totals: dict[str, torch.Tensor],
    batch: dict[str, Any],
    output: dict[str, torch.Tensor],
    cfg: Any,
    *,
    active_task_ids: tuple[int, ...] | None = None,
) -> None:
    """Accumulate detached PPO/BC numerators for the active tasks on-device."""

    task_ids = batch["forward_inputs"]["multitask_task_id"].reshape(-1).long()
    route = batch["route_info"].route_used
    tasks = tuple(range(10)) if active_task_ids is None else active_task_ids
    by_task = task_ids[None, :] == torch.tensor(tasks, device=task_ids.device)[:, None]
    gate_mask = batch["gate_valid_mask"].bool().reshape(-1)
    flow_mask = batch["flow_valid_mask"].bool().reshape(-1) & (
        route.reshape(-1) == int(WAMRoute.UNCOND)
    )
    gate_logprobs = output["gate_logprobs"].detach().float().reshape(-1)
    gate_old = batch["emitted_gate"].old_logprob.detach().float().reshape(-1)
    flow_logprobs = output["flow_logprobs"].detach().float()
    flow_old = batch["prev_logprobs"].detach().float()
    execution_mask = batch.get("flow_execution_mask")
    if execution_mask is not None:
        flow_logprobs = mask_executed_flow_statistics(flow_logprobs, execution_mask)
        flow_old = mask_executed_flow_statistics(flow_old, execution_mask)
        flow_mask = flow_mask & execution_mask.any(dim=-1).reshape(-1)
    reduction_dims = tuple(range(route.ndim, flow_logprobs.ndim))
    if reduction_dims:
        flow_logprobs = flow_logprobs.sum(dim=reduction_dims)
        flow_old = flow_old.sum(dim=reduction_dims)
    objectives = {}
    for owner, log_ratio, advantage, clip_cfg in (
        (
            "gate",
            gate_logprobs - gate_old,
            batch["gate_advantages"],
            cfg.algorithm.gate_ppo,
        ),
        (
            "flow",
            (flow_logprobs - flow_old).reshape(-1),
            batch["flow_advantages"],
            cfg.algorithm.uncond_flow_ppo,
        ),
    ):
        objectives[owner] = clipped_ppo_objective(
            log_ratio.exp(),
            advantage.detach().float().reshape(-1),
            clip_ratio_low=float(clip_cfg.clip_ratio_low),
            clip_ratio_high=float(clip_cfg.clip_ratio_high),
        )
    objectives["bc"] = (
        output["online_idm_bc_per_sample_loss"].detach().float().reshape(-1)
    )
    for owner, valid in (("gate", gate_mask), ("flow", flow_mask), ("bc", flow_mask)):
        selected = by_task & valid[None, :]
        for suffix, value in (
            ("count", selected.sum(dim=1).double()),
            (
                "loss_sum",
                torch.where(selected, objectives[owner][None, :], 0)
                .double()
                .sum(dim=1),
            ),
        ):
            key = f"{owner}_{suffix}"
            if key in totals:
                totals[key].add_(value)
            else:
                totals[key] = value


def finalize_task_losses(
    totals: dict[str, torch.Tensor], *, active_task_ids: tuple[int, ...] | None = None
) -> dict[str, float]:
    """Transfer additive statistics once and recover the original task means."""

    tasks = tuple(range(10)) if active_task_ids is None else active_task_ids
    keys = [
        (owner, suffix)
        for owner in ("gate", "flow", "bc")
        for suffix in ("count", "loss_sum")
    ]
    values = (
        torch.stack([totals[f"{owner}_{suffix}"] for owner, suffix in keys])
        .cpu()
        .tolist()
        if totals
        else [[0.0] * len(tasks) for _ in keys]
    )
    accumulated = dict(zip(keys, values, strict=True))
    metrics = {}
    for index, task in enumerate(tasks):
        for owner in ("gate", "flow", "bc"):
            prefix = f"task/{task}/{owner}"
            count = accumulated[owner, "count"][index]
            metrics[f"{prefix}_sample_count"] = count
            metrics[f"{prefix}_raw_loss"] = accumulated[owner, "loss_sum"][index] / max(
                count, 1.0
            )
    return metrics
