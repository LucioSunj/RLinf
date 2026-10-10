# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fixed offline task targets with the original lagged band-price controller."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.advantages import (
    FastWAMChunkCost,
    _chunk_mask,
    apply_fastwam_chunk_cost,
    summarize_fastwam_chunk_cost,
)
from rlinf.models.embodiment.wam_policy.contracts import WAMRoute
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_budget import (
    PadCriticWarmupReversalDampedController,
)
from rlinf.runners.fastwam_idm_cost_control import (
    FastWAMIDMCostControlRuntime,
    FastWAMIDMCostObservation,
    _canonical_sha256,
)


def validate_task_budget_config(cfg: Any) -> dict[str, Any]:
    """Resolve the fixed ten-task table under the existing Long training profile."""
    value = OmegaConf.to_container(
        cfg.algorithm.fixed_branch_cost.task_budget, resolve=True
    )
    if not isinstance(value, dict):
        raise TypeError("Task budget must resolve to a mapping.")
    if (
        value.get("suite") != cfg.env.train.task_suite_name
        or value["suite"] != "libero_10"
    ):
        raise ValueError("Offline task budgets must match the LIBERO-Long suite.")
    targets = value.get("targets", {})
    if set(targets) != {str(i) for i in range(10)}:
        raise ValueError("Offline task budgets require exactly task IDs 0 through 9.")
    if (
        cfg.env.train.task_sampling != "global_balanced"
        or list(cfg.env.train.task_id_filter) != list(range(10))
        or cfg.cluster.component_placement.actor != "0-0"
    ):
        raise ValueError(
            "Task budgets require the balanced ten-task single-Actor profile."
        )
    guard = cfg.runner.fastwam_training_guard
    if (
        not guard.enabled
        or not guard.cost_audit.enabled
        or guard.cost_audit.break_even_guard_enabled
    ):
        raise ValueError(
            "Task budgets require actual cost audits and no scalar break-even guard."
        )
    if cfg.actor.model.decision_telemetry_enabled:
        raise ValueError(
            "Task budgets retain the original disabled scalar decision telemetry."
        )
    controller = OmegaConf.to_container(
        cfg.algorithm.fixed_branch_cost.controller, resolve=True
    )
    if (
        controller["type"] != "pad_critic_warmup_reversal_damped"
        or controller["rate"]["scope"] != "eligible_gate_decisions"
        or controller["rate"]["feedback"] != "expected_behavior_probability"
        or controller["charge_scope"] != "eligible_nonforced"
    ):
        raise ValueError(
            "Task budgets must retain the original controller and charge scope."
        )
    for target in targets.values():
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(target)
        ):
            raise ValueError("Every task target must be a finite number.")
        task_config = copy.deepcopy(controller)
        task_config["rate"]["target_idm_fraction"] = target
        PadCriticWarmupReversalDampedController(task_config)
    return value


class TaskBudgetController:
    """Keep independent prices and original update equations for each task."""

    controller_type = "offline_task_pad_warmup_reversal_damped"
    enabled = True
    requires_rollout_feedback = True
    requires_break_even_audit = False

    def __init__(self, controller_config: Mapping, table: Mapping) -> None:
        self.config = copy.deepcopy(dict(controller_config))
        self.table = copy.deepcopy(dict(table))
        self.controllers = {}
        for task in range(10):
            config = copy.deepcopy(self.config)
            config["rate"]["target_idm_fraction"] = self.table["targets"][str(task)]
            self.controllers[task] = PadCriticWarmupReversalDampedController(config)

    @property
    def observed_runner_steps(self) -> int:
        """Return the common completed rollout count."""
        steps = {
            controller.observed_runner_steps for controller in self.controllers.values()
        }
        if len(steps) != 1:
            raise RuntimeError("Task controllers have diverging runner steps.")
        return steps.pop()

    def decisions(self, runner_step: int) -> dict[str, Any]:
        """Freeze all costs before sampling; current feedback affects the next rollout."""
        return {
            "runner_step": runner_step,
            "tasks": {
                str(task): controller.decision_for_step(runner_step).to_artifact()
                for task, controller in self.controllers.items()
            },
        }

    def observe(self, runner_step: int, metrics: Mapping) -> dict[str, Any]:
        """Consume unique raw-rollout counts before any PPO replay/compaction."""
        records = {}
        for task, controller in self.controllers.items():
            prefix = f"task/{task}"
            eligible = int(metrics[f"{prefix}/gate_chunks"])
            valid = int(metrics[f"{prefix}/valid_chunks"])
            idm = int(metrics[f"{prefix}/idm_chunks"])
            eligible_idm = int(metrics[f"{prefix}/eligible_idm_chunks"])
            if eligible == 0:
                if (
                    controller._pending is None
                    or controller._pending.runner_step != runner_step
                ):
                    raise RuntimeError(
                        "Missing-feedback hold has no pending task decision."
                    )
                # No fake zero-rate observation: retain price/EMA, advance lifecycle.
                records[str(task)] = {
                    "status": "HOLD_NO_ELIGIBLE_FEEDBACK",
                    "applied": controller._pending.to_artifact(),
                }
                controller._pending = None
                controller.observed_runner_steps += 1
                continue
            observation = FastWAMIDMCostObservation(
                runner_step=runner_step,
                eligible_gate_decision_count=eligible,
                eligible_idm_decision_count=eligible_idm,
                eligible_realized_fraction=eligible_idm / eligible,
                eligible_expected_fraction=float(
                    metrics[f"{prefix}/behavior_probability_sum"]
                )
                / eligible,
                valid_chunk_count=valid,
                valid_idm_chunk_count=idm,
                executed_realized_fraction=idm / valid,
                forced_fraction=float(metrics[f"{prefix}/forced_chunks"]) / valid,
                break_even_idm_cost=None,
                configured_idm_cost=None,
            )
            records[str(task)] = controller.observe_rollout(observation)
        return {
            "schema": "offline-task-budget-observation-v1",
            "runner_step": runner_step,
            "tasks": records,
        }

    def state_dict(self) -> dict[str, Any]:
        """Use native controller state; checkpoints are taken after feedback."""
        return {
            "config": copy.deepcopy(self.config),
            "table": copy.deepcopy(self.table),
            "observed_runner_steps": self.observed_runner_steps,
            "controllers": {
                str(task): c.state_dict() for task, c in self.controllers.items()
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore each price/EMA/counter only under an identical static table."""
        if state["config"] != self.config or state["table"] != self.table:
            raise ValueError(
                "Task targets or controller configuration changed on resume."
            )
        if set(state["controllers"]) != {str(i) for i in self.controllers}:
            raise ValueError("Checkpoint task controller set differs.")
        for task, controller in self.controllers.items():
            controller.load_state_dict(state["controllers"][str(task)])
        if self.observed_runner_steps != state["observed_runner_steps"]:
            raise ValueError("Task controller checkpoint step differs.")


class TaskBudgetRuntime(FastWAMIDMCostControlRuntime):
    """Publish task costs and checkpoint their state through the existing v3 runtime."""

    @classmethod
    def from_config(cls, cfg: Any) -> TaskBudgetRuntime:
        table = validate_task_budget_config(cfg)
        config = OmegaConf.to_container(
            cfg.algorithm.fixed_branch_cost.controller, resolve=True
        )
        return cls(
            controller=TaskBudgetController(config, table),
            explicit=True,
            # Existing native resume identity binds the actual table as well.
            config_sha256=_canonical_sha256(
                {"controller": config, "task_budget": table}
            ),
            audit_root=Path(cfg.runner.logger.log_path)
            / cfg.runner.logger.experiment_name
            / "audits",
        )

    def before_rollout(self, *, actor: Any, runner_step: int) -> dict[str, Any]:
        decision = self.controller.decisions(runner_step)
        actor.set_fastwam_task_branch_costs(decision).wait()
        return decision

    def after_rollout(
        self, *, runner_step: int, actor_rollout_metrics: list[dict], guard_result: dict
    ) -> dict[str, Any]:
        if (
            guard_result.get("status") != "PASS"
            or guard_result.get("task_cost_identity") != "PASS"
        ):
            raise RuntimeError(
                "Task feedback requires the accepted actual task-cost audit."
            )
        if len(actor_rollout_metrics) != 1:
            raise ValueError(
                "The selected N84 profile has one Actor receiving the complete rollout."
            )
        metrics = actor_rollout_metrics[0]
        record = self.controller.observe(runner_step, metrics)
        for task, controller in self.controller.controllers.items():
            prefix = f"task/{task}/budget"
            metrics[f"{prefix}/target"] = controller.target_fraction
            metrics[f"{prefix}/next_signed_price"] = controller.signed_price
            item = record["tasks"][str(task)]
            metrics[f"{prefix}/applied_idm_cost"] = item["applied"]["idm_cost"]
            metrics[f"{prefix}/applied_uncond_cost"] = item["applied"]["uncond_cost"]
            metrics[f"{prefix}/feedback_present"] = float(
                item.get("status") != "HOLD_NO_ELIGIBLE_FEEDBACK"
            )
        self.audit_root.mkdir(parents=True, exist_ok=True)
        with (self.audit_root / "task_budget_control.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return record


@dataclass(frozen=True)
class TaskCostAudit:
    """Actual per-task prices, costs and reward identity, without a scalar substitute."""

    artifact: dict[str, Any]

    def to_artifact(self) -> dict[str, Any]:
        return self.artifact

    def to_metrics(self) -> dict[str, float]:
        return {
            "fastwam/task_budget/enabled": 1.0,
            **{
                f"fastwam/task_budget/{key}": float(self.artifact[key])
                for key in (
                    "cost_identity_max_abs_error",
                    "cost_tensor_max_abs_error",
                    "expected_cost_sum",
                    "actual_cost_sum",
                )
            },
        }


def apply_task_chunk_costs(
    actor: Any, raw_rewards: torch.Tensor, charge_mask: torch.Tensor
) -> tuple[FastWAMChunkCost, TaskCostAudit]:
    """Reuse the scalar kernel on disjoint task masks, charging each chunk once."""
    batch = actor.rollout_batch
    if "fastwam_branch_costs" in batch:
        raise RuntimeError("Task branch costs have already been applied.")
    decision = actor._fastwam_task_cost_decision
    if decision["runner_step"] != int(actor.version):
        raise ValueError("Task costs do not belong to this rollout version.")
    route = batch["route_info"]
    ids = batch["forward_inputs"]["multitask_task_id"]
    if ids.shape != route.route_used.shape or ids.dtype not in (
        torch.int64,
        torch.int32,
    ):
        raise ValueError(
            "Task costs require integral task identities on the original T/B rollout."
        )
    valid_mask = batch.get("loss_mask")
    valid = _chunk_mask(
        valid_mask, shape=route.shape, name="loss_mask", device=ids.device
    )
    costs = torch.zeros_like(raw_rewards.sum(dim=-1, keepdim=True))
    covered = torch.zeros_like(valid)
    tasks = {}
    for task in range(10):
        selected = valid & (ids == task)
        covered |= selected
        price = decision["tasks"][str(task)]
        task_charge = charge_mask & selected
        part = apply_fastwam_chunk_cost(
            environment_rewards=raw_rewards,
            route_used=route.route_used,
            idm_cost=price["idm_cost"],
            uncond_cost=price["uncond_cost"],
            valid_mask=valid_mask,
            charge_mask=task_charge,
        )
        costs.add_(part.costs)
        task_mask = selected.unsqueeze(-1)
        if valid_mask is not None:
            task_mask = (
                selected & valid_mask
                if valid_mask.ndim == 2
                else task_mask & valid_mask
            )
        tasks[str(task)] = summarize_fastwam_chunk_cost(
            environment_rewards=raw_rewards,
            route=route,
            cost_result=part,
            idm_cost=price["idm_cost"],
            uncond_cost=price["uncond_cost"],
            valid_mask=task_mask,
            charge_mask=task_charge,
            charge_scope="eligible_nonforced",
        ).to_artifact()
    if bool((valid & ~covered).any()):
        raise ValueError("Valid rollout has a task with no fixed budget.")
    result = FastWAMChunkCost(
        rewards=raw_rewards.sum(dim=-1, keepdim=True) - costs, costs=costs
    )
    batch["rewards"], batch["fastwam_branch_costs"] = result.rewards, result.costs
    # Independent lookup reconciles the composed tensor to the published prices.
    lookup = torch.tensor(
        [
            [
                decision["tasks"][str(i)]["uncond_cost"],
                decision["tasks"][str(i)]["idm_cost"],
            ]
            for i in range(10)
        ],
        dtype=costs.dtype,
        device=costs.device,
    )
    expected = torch.zeros_like(costs)
    selected = charge_mask & valid
    expected[..., 0][selected] = lookup[
        ids[selected].long(), (route.route_used[selected] == int(WAMRoute.IDM)).long()
    ]
    audit = TaskCostAudit(
        {
            "schema": "offline-task-cost-audit-v1",
            "runner_step": int(actor.version),
            "tasks": tasks,
            "expected_cost_sum": sum(t["expected_cost_sum"] for t in tasks.values()),
            "actual_cost_sum": float(costs.double().sum()),
            "cost_identity_max_abs_error": float(
                (batch["rewards"] - (raw_rewards.sum(dim=-1, keepdim=True) - expected))
                .abs()
                .max()
            ),
            "cost_tensor_max_abs_error": float(
                (batch["fastwam_branch_costs"] - expected).abs().max()
            ),
            "scalar_counterfactual": "NOT_APPLICABLE_TASK_INDEXED_COSTS",
        }
    )
    return result, audit


def validate_task_cost_metrics(metrics_list: list[Mapping]) -> None:
    """Reject missing, nonfinite or inconsistent heterogeneous cost audits."""
    for metrics in metrics_list:
        if metrics.get("fastwam/task_budget/enabled") != 1.0:
            raise ValueError("Actor task-cost audit is missing.")
        prefix = "fastwam/task_budget/"
        for field in ("cost_identity_max_abs_error", "cost_tensor_max_abs_error"):
            value = float(metrics[prefix + field])
            if not math.isfinite(value) or not 0 <= value <= 1e-6:
                raise ValueError("Task cost/reward tensor identity failed.")
        expected, actual = (
            float(metrics[prefix + field])
            for field in ("expected_cost_sum", "actual_cost_sum")
        )
        if (
            not math.isfinite(actual)
            or not math.isfinite(expected)
            or not math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ValueError("Task cost totals do not reconcile.")
