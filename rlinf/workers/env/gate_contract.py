# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pure helpers for the gate's decision-vs-environment-step contract."""

from __future__ import annotations

from typing import Any

import torch


def validate_gate_horizons(*, generation_horizon: int, exec_horizon: int) -> int:
    generation_horizon = int(generation_horizon)
    exec_horizon = int(exec_horizon)
    if generation_horizon <= 0 or exec_horizon <= 0:
        raise ValueError(
            "generation_horizon and exec_horizon must be positive, got "
            f"{generation_horizon} and {exec_horizon}."
        )
    if exec_horizon > generation_horizon:
        raise ValueError(
            f"exec_horizon={exec_horizon} exceeds "
            f"generation_horizon={generation_horizon}."
        )
    return exec_horizon


def slice_gate_action_chunk(
    actions,
    *,
    generation_horizon: int,
    exec_horizon: int,
):
    """Return the executed `[B, exec_horizon, A]` prefix."""
    if actions.ndim != 3:
        raise ValueError(
            "gate_policy must return robot chunks shaped [B, T, A], "
            f"got {tuple(actions.shape)}."
        )
    exec_horizon = int(exec_horizon)
    generation_horizon = int(generation_horizon)
    if actions.shape[1] != generation_horizon:
        raise ValueError(
            f"WAM returned {actions.shape[1]} steps, expected configured "
            f"generation_horizon={generation_horizon}."
        )
    if actions.shape[1] < exec_horizon:
        raise ValueError(
            f"Generated action chunk has {actions.shape[1]} steps, fewer than "
            f"exec_horizon={exec_horizon}."
        )
    return actions[:, :exec_horizon]


def limit_gate_action_chunk_to_episode(
    actions,
    *,
    elapsed_steps,
    max_episode_steps: int,
):
    """Do not execute past the shortest remaining episode budget in the batch.

    Environment adapters require a common chunk length for the whole vectorized
    batch. Gate GRPO also requires each reset group to share one episode clock, so
    heterogeneous remaining budgets fail closed rather than silently mixing
    different decision horizons. The caller pads returned reward/done tensors back
    to the fixed rollout width.
    """
    if actions.ndim != 3:
        raise ValueError(f"gate actions must be [B,T,A], got {tuple(actions.shape)}")
    elapsed = torch.as_tensor(elapsed_steps).detach().reshape(-1).cpu().long()
    if elapsed.numel() != int(actions.shape[0]):
        raise ValueError(
            f"elapsed_steps has {elapsed.numel()} entries for action batch "
            f"{actions.shape[0]}."
        )
    max_episode_steps = int(max_episode_steps)
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    remaining = max_episode_steps - elapsed
    if bool((remaining <= 0).any()):
        raise RuntimeError(
            "gate action requested for an environment whose episode budget is "
            "already exhausted; reset it before requesting another action."
        )
    if not bool((remaining == remaining[0]).all()):
        raise RuntimeError(
            "gate vector-environment slots have different remaining episode "
            "budgets; asynchronous episode clocks would invalidate the shared "
            "fixed-width decision contract."
        )
    effective_horizon = min(int(actions.shape[1]), int(remaining[0].item()))
    return actions[:, :effective_horizon]


def pad_gate_env_chunk(
    values: torch.Tensor,
    *,
    target_horizon: int,
    move_final_flag: bool = False,
) -> torch.Tensor:
    """Right-pad a partial env chunk while preserving its terminal event."""
    if values.ndim < 2:
        raise ValueError(f"environment chunk must be at least 2D, got {values.shape}")
    current_horizon = int(values.shape[-1])
    target_horizon = int(target_horizon)
    if current_horizon <= 0 or target_horizon <= 0:
        raise ValueError("chunk horizons must be positive")
    if current_horizon > target_horizon:
        raise ValueError(
            f"environment chunk width {current_horizon} exceeds target {target_horizon}."
        )
    if current_horizon == target_horizon:
        return values

    padded = values.new_zeros((*values.shape[:-1], target_horizon))
    padded[..., :current_horizon] = values
    if move_final_flag:
        final_flag = values[..., -1].clone()
        padded[..., current_horizon - 1] = False
        padded[..., -1] = final_flag
    return padded


def make_bootstrap_dones(*, batch_size: int, exec_horizon: int) -> torch.Tensor:
    """Initial dones must have the same final dimension as environment chunks."""
    return torch.zeros((int(batch_size), int(exec_horizon)), dtype=torch.bool)


def max_gate_decisions(*, max_episode_steps: int, exec_horizon: int) -> int:
    """Return ceil(max_episode_steps / exec_horizon)."""
    max_episode_steps = int(max_episode_steps)
    exec_horizon = int(exec_horizon)
    if max_episode_steps <= 0 or exec_horizon <= 0:
        raise ValueError("max_episode_steps and exec_horizon must be positive")
    return (max_episode_steps + exec_horizon - 1) // exec_horizon


def normalize_episode_mode_cost(
    mode_cost,
    *,
    max_episode_steps: int,
    exec_horizon: int,
):
    """Scale one decision cost so a full all-IDM episode sums to one."""
    return mode_cost / max_gate_decisions(
        max_episode_steps=max_episode_steps,
        exec_horizon=exec_horizon,
    )


class PendingGateDecisions:
    """Track the decision whose robot chunk is currently awaiting its reward."""

    def __init__(self, num_stages: int):
        self._by_stage: list[dict[str, Any] | None] = [None] * int(num_stages)

    def mark_executed(self, stage_id: int, forward_inputs: dict[str, Any]) -> None:
        if self._by_stage[int(stage_id)] is not None:
            raise RuntimeError(
                f"stage {stage_id} still has an unconsumed gate decision"
            )
        self._by_stage[int(stage_id)] = forward_inputs

    def consume_reward(self, stage_id: int) -> dict[str, Any] | None:
        stage_id = int(stage_id)
        decision = self._by_stage[stage_id]
        self._by_stage[stage_id] = None
        return decision


def rollout_decision_count(
    *,
    max_env_steps: int,
    exec_horizon: int,
    rollout_epoch: int,
    total_num_envs: int,
) -> int:
    """Number of gate samples produced by one synchronous training rollout."""
    max_env_steps = int(max_env_steps)
    exec_horizon = int(exec_horizon)
    rollout_epoch = int(rollout_epoch)
    total_num_envs = int(total_num_envs)
    if min(max_env_steps, exec_horizon, rollout_epoch, total_num_envs) <= 0:
        raise ValueError("rollout sizes and horizons must all be positive")
    if max_env_steps % exec_horizon != 0:
        raise ValueError(
            f"max_env_steps={max_env_steps} is not divisible by "
            f"exec_horizon={exec_horizon}; a partial action chunk would be dropped."
        )
    return max_env_steps // exec_horizon * rollout_epoch * total_num_envs


def validate_gate_distribution_contract(
    *,
    algorithm_group_size: int,
    env_group_size: int | None,
    train_total_envs: int | None,
    eval_total_envs: int | None,
    env_world_size: int,
    rollout_world_size: int,
    actor_world_size: int,
    stage_num: int,
    actor_split_num: int,
    use_training_pipeline: bool,
) -> None:
    """Fail early when distributed sharding would split a GRPO reset group."""
    sizes = {
        "algorithm_group_size": algorithm_group_size,
        "env_world_size": env_world_size,
        "rollout_world_size": rollout_world_size,
        "actor_world_size": actor_world_size,
        "stage_num": stage_num,
        "actor_split_num": actor_split_num,
    }
    sizes = {name: int(value) for name, value in sizes.items()}
    if any(value <= 0 for value in sizes.values()):
        raise ValueError(f"gate distributed sizes must be positive: {sizes}")
    group_size = sizes["algorithm_group_size"]
    logical_env_world = sizes["env_world_size"] * sizes["stage_num"]

    def _require_divisible(name: str, total: int, divisor_name: str, divisor: int):
        if total % divisor != 0:
            raise ValueError(
                f"{name}={total} must be divisible by {divisor_name}={divisor}."
            )

    if train_total_envs is not None:
        if env_group_size is None:
            raise ValueError("gate training requires env.train.group_size")
        env_group_size = int(env_group_size)
        if env_group_size <= 0:
            raise ValueError("env.train.group_size must be positive")
        if env_group_size != group_size:
            raise ValueError(
                "Gate GRPO requires env.train.group_size == algorithm.group_size; "
                f"got {env_group_size} and {group_size}."
            )
        train_total_envs = int(train_total_envs)
        if train_total_envs <= 0:
            raise ValueError("env.train.total_num_envs must be positive")
        for name, size in (
            ("logical_env_world_size", logical_env_world),
            ("rollout_world_size", sizes["rollout_world_size"]),
            ("actor_world_size", sizes["actor_world_size"]),
        ):
            _require_divisible("env.train.total_num_envs", train_total_envs, name, size)
        local_envs = train_total_envs // logical_env_world
        actor_envs = train_total_envs // sizes["actor_world_size"]
        _require_divisible("local train env batch", local_envs, "group_size", group_size)
        _require_divisible("per-actor train env batch", actor_envs, "group_size", group_size)
        if not use_training_pipeline:
            split_num = sizes["actor_split_num"]
            _require_divisible("local train env batch", local_envs, "actor_split_num", split_num)
            _require_divisible(
                "non-pipeline actor trajectory env batch",
                local_envs // split_num,
                "group_size",
                group_size,
            )

    if eval_total_envs is not None:
        eval_total_envs = int(eval_total_envs)
        if eval_total_envs <= 0:
            raise ValueError("env.eval.total_num_envs must be positive")
        for name, size in (
            ("logical_env_world_size", logical_env_world),
            ("rollout_world_size", sizes["rollout_world_size"]),
        ):
            _require_divisible("env.eval.total_num_envs", eval_total_envs, name, size)


def update_gate_eval_active_mask(
    active: torch.Tensor,
    *,
    success_once: torch.Tensor | None,
    dones: torch.Tensor | None,
) -> torch.Tensor:
    """Exclude decisions after success/done for one evaluation episode per slot."""
    active = active.detach().reshape(-1).bool().cpu()
    next_active = active.clone()
    if success_once is not None:
        success = success_once.detach().reshape(-1).bool().cpu()
        if success.shape != active.shape:
            raise ValueError("success_once does not match gate eval active mask")
        next_active &= ~success
    if dones is not None:
        done = dones.detach().bool().cpu()
        if done.ndim > 1:
            done = done[:, -1]
        done = done.reshape(-1)
        if done.shape != active.shape:
            raise ValueError("dones does not match gate eval active mask")
        next_active &= ~done
    return next_active
