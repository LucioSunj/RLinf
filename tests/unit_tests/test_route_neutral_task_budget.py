# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Task budgets preserve warmup, independent feedback, native resume and costs."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.advantages import apply_fastwam_chunk_cost
from rlinf.models.embodiment.wam_policy.contracts import WAMRoute
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_budget import (
    PadCriticWarmupReversalDampedController,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.task_budget import (
    TaskBudgetController,
    TaskBudgetRuntime,
    apply_task_chunk_costs,
    validate_task_cost_metrics,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.task_metrics import (
    summarize_task_rollout,
)
from rlinf.runners.fastwam_idm_cost_control import (
    FastWAMIDMCostControlRuntime,
    FastWAMIDMCostObservation,
)
from rlinf.runners.fastwam_training_guard import FastWAMTrainingGuard


def controller_config(warmup=2):
    return {
        "type": "pad_critic_warmup_reversal_damped",
        "constraint": "two_sided_band",
        "charge_scope": "eligible_nonforced",
        "rate": {
            "scope": "eligible_gate_decisions",
            "feedback": "expected_behavior_probability",
            "target_idm_fraction": 0.5,
            "half_width": 0.03,
        },
        "signed_price": {
            "initial_value": 0.0,
            "learning_rate": 0.0025,
            "ema_beta": 0.0,
            "update_interval": 1,
            "max_abs_value": 0.1,
            "max_delta_per_update": 0.005,
            "reversal": {"mode": "opposing_decay", "factor": 0.0},
        },
        "critic_warmup": {
            "runner_updates": warmup,
            "route_behavior": "independent_random",
            "idm_probability": 0.5,
            "freeze_gate": True,
            "freeze_cost_controller": True,
        },
    }


def table():
    return {
        "suite": "libero_10",
        "targets": {
            str(i): 0.3 if i == 0 else 0.7 if i == 1 else 0.5 for i in range(10)
        },
    }


def feedback(q=0.5):
    return {
        f"task/{i}/{key}": value
        for i in range(10)
        for key, value in {
            "gate_chunks": 10,
            "eligible_idm_chunks": 5,
            "valid_chunks": 10,
            "idm_chunks": 5,
            "forced_chunks": 0,
            "behavior_probability_sum": q * 10,
        }.items()
    }


def config(tmp_path):
    return OmegaConf.create(
        {
            "env": {
                "train": {
                    "task_suite_name": "libero_10",
                    "task_sampling": "global_balanced",
                    "task_id_filter": list(range(10)),
                }
            },
            "cluster": {"component_placement": {"actor": "0-0"}},
            "algorithm": {
                "fixed_branch_cost": {
                    "controller": controller_config(),
                    "task_budget": table(),
                }
            },
            "actor": {"model": {"decision_telemetry_enabled": False}},
            "runner": {
                "logger": {"log_path": str(tmp_path), "experiment_name": "test"},
                "fastwam_training_guard": {
                    "enabled": True,
                    "zero_success_patience": 201,
                    "cost_audit": {"enabled": True, "break_even_guard_enabled": False},
                },
            },
        }
    )


def test_independent_targets_start_only_after_warmup():
    bank = TaskBudgetController(controller_config(), table())
    for step in range(3):
        decisions = bank.decisions(step)
        assert all(
            p["idm_cost"] == p["uncond_cost"] == 0 for p in decisions["tasks"].values()
        )
        bank.observe(step, feedback())
    assert bank.controllers[0].signed_price > 0
    assert bank.controllers[1].signed_price < 0
    assert all(bank.controllers[i].signed_price == 0 for i in range(2, 10))
    next_costs = bank.decisions(3)["tasks"]
    assert next_costs["0"]["idm_cost"] > 0 and next_costs["0"]["uncond_cost"] == 0
    assert next_costs["1"]["idm_cost"] == 0 and next_costs["1"]["uncond_cost"] > 0


def test_missing_feedback_holds_state_without_zero_rate_sample():
    bank = TaskBudgetController(controller_config(warmup=1), table())
    bank.decisions(0)
    bank.observe(0, feedback(0.9))
    bank.decisions(1)
    bank.observe(1, feedback(0.9))
    previous = bank.controllers[0].state_dict()
    bank.decisions(2)
    metrics = feedback()
    metrics.update(
        {
            "task/0/gate_chunks": 0,
            "task/0/eligible_idm_chunks": 0,
            "task/0/forced_chunks": 10,
            "task/0/behavior_probability_sum": 0,
        }
    )
    assert (
        bank.observe(2, metrics)["tasks"]["0"]["status"] == "HOLD_NO_ELIGIBLE_FEEDBACK"
    )
    current = bank.controllers[0].state_dict()
    assert current == {**previous, "observed_runner_steps": 3}


def test_uniform_targets_reproduce_original_controller_exactly():
    uniform = table()
    uniform["targets"] = dict.fromkeys(uniform["targets"], 0.5)
    bank = TaskBudgetController(controller_config(), uniform)
    original = PadCriticWarmupReversalDampedController(controller_config())
    for step, q in enumerate((0.9, 0.1, 0.8, 0.5, 0.1, 0.9)):
        decision = original.decision_for_step(step).to_artifact()
        assert all(item == decision for item in bank.decisions(step)["tasks"].values())
        original.observe_rollout(
            FastWAMIDMCostObservation(
                runner_step=step,
                eligible_gate_decision_count=10,
                eligible_idm_decision_count=5,
                eligible_realized_fraction=0.5,
                eligible_expected_fraction=q,
                valid_chunk_count=10,
                valid_idm_chunk_count=5,
                executed_realized_fraction=0.5,
                forced_fraction=0,
                break_even_idm_cost=None,
                configured_idm_cost=None,
            )
        )
        bank.observe(step, feedback(q))
        assert all(
            c.state_dict() == original.state_dict() for c in bank.controllers.values()
        )


def test_native_runtime_roundtrip_and_changed_table_rejection(tmp_path):
    cfg = config(tmp_path)
    runtime = FastWAMIDMCostControlRuntime.from_config(cfg)
    assert isinstance(runtime, TaskBudgetRuntime)
    published = []
    actor = SimpleNamespace(
        set_fastwam_task_branch_costs=lambda value: (
            published.append(value) or SimpleNamespace(wait=lambda: None)
        )
    )
    for step in range(3):
        runtime.before_rollout(actor=actor, runner_step=step)
        runtime.after_rollout(
            runner_step=step,
            actor_rollout_metrics=[feedback()],
            guard_result={"status": "PASS", "task_cost_identity": "PASS"},
        )
    state = runtime.state_dict()
    restored = TaskBudgetRuntime.from_config(cfg)
    restored.load_state_dict(copy.deepcopy(state), global_step=3)
    assert restored.state_dict() == state
    assert restored.before_rollout(
        actor=actor, runner_step=3
    ) == runtime.before_rollout(actor=actor, runner_step=3)
    cfg.algorithm.fixed_branch_cost.task_budget.targets["0"] = 0.4
    with pytest.raises(ValueError, match="config hash"):
        TaskBudgetRuntime.from_config(cfg).load_state_dict(state, global_step=3)
    assert len(published) == 5


def cost_fixture():
    ids = torch.arange(10).expand(3, 10)
    used = torch.full((3, 10), int(WAMRoute.IDM), dtype=torch.long)
    used[1] = int(WAMRoute.UNCOND)
    forced = torch.zeros((3, 10), dtype=torch.bool)
    forced[0, 0] = True
    valid = torch.ones((3, 10, 1), dtype=torch.bool)
    valid[2, 1] = False
    route = SimpleNamespace(route_used=used, route_was_forced=forced, shape=used.shape)
    raw = torch.zeros((3, 10, 10))
    raw[2, :, 0] = 1
    decision = {
        "runner_step": 2,
        "tasks": {
            str(i): {
                "runner_step": 2,
                "idm_cost": (i + 1) * 0.001,
                "uncond_cost": (10 - i) * 0.002,
            }
            for i in range(10)
        },
    }
    batch = {
        "rewards": raw,
        "route_info": route,
        "loss_mask": valid,
        "forward_inputs": {"multitask_task_id": ids},
    }
    actor = SimpleNamespace(
        rollout_batch=batch, version=2, _fastwam_task_cost_decision=decision
    )
    charge = valid[..., 0] & ~forced
    return actor, raw, charge


def test_task_costs_charge_once_and_exclude_forced_and_padding():
    actor, raw, charge = cost_fixture()
    result, audit = apply_task_chunk_costs(actor, raw, charge)
    expected = torch.zeros((3, 10, 1))
    for i in range(10):
        expected[0, i, 0] = expected[2, i, 0] = (i + 1) * 0.001
        expected[1, i, 0] = (10 - i) * 0.002
    expected[0, 0, 0] = expected[2, 1, 0] = 0
    torch.testing.assert_close(result.costs, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        result.rewards, raw.sum(-1, keepdim=True) - expected, atol=0, rtol=0
    )
    validate_task_cost_metrics([audit.to_metrics()])
    with pytest.raises(RuntimeError, match="already"):
        apply_task_chunk_costs(actor, raw, charge)
    broken = audit.to_metrics()
    broken["fastwam/task_budget/cost_tensor_max_abs_error"] = 0.01
    with pytest.raises(ValueError, match="identity"):
        validate_task_cost_metrics([broken])


def test_task_cost_kernel_matches_scalar_when_prices_match():
    actor, raw, charge = cost_fixture()
    for value in actor._fastwam_task_cost_decision["tasks"].values():
        value.update(idm_cost=0.012, uncond_cost=0.0)
    original = apply_fastwam_chunk_cost(
        environment_rewards=raw,
        route_used=actor.rollout_batch["route_info"].route_used,
        idm_cost=0.012,
        uncond_cost=0.0,
        valid_mask=actor.rollout_batch["loss_mask"],
        charge_mask=charge,
    )
    result, _ = apply_task_chunk_costs(actor, raw, charge)
    assert torch.equal(result.rewards, original.rewards)
    assert torch.equal(result.costs, original.costs)


def test_raw_task_feedback_excludes_padding_and_forced_chunks():
    actor, _, charge = cost_fixture()
    batch = actor.rollout_batch
    valid = batch["loss_mask"][..., 0]
    teacher = valid & (batch["route_info"].route_used == int(WAMRoute.UNCOND))
    batch.update(
        gate_valid_mask=charge,
        emitted_gate=SimpleNamespace(behavior_probability=torch.full((3, 10), 0.6)),
        returns=torch.zeros((3, 10, 1)),
        prev_values=torch.zeros((4, 10, 1)),
    )
    batch["forward_inputs"].update(
        online_idm_bc_teacher_present=teacher,
        multitask_episode_slot_id=torch.arange(10).expand(3, 10),
        multitask_episode_success=torch.ones((3, 10)),
        multitask_episode_failed=torch.zeros((3, 10)),
        multitask_episode_truncated=torch.zeros((3, 10)),
    )
    metrics = summarize_task_rollout(batch)
    assert metrics["task/0/gate_chunks"] == 2
    assert metrics["task/0/eligible_idm_chunks"] == 1
    assert metrics["task/0/forced_chunks"] == 1
    assert metrics["task/1/valid_chunks"] == 2
    assert metrics["task/1/behavior_probability_sum"] == pytest.approx(1.2)


def test_scientific_guard_accepts_task_identity_without_fake_scalar_cost(tmp_path):
    actor, raw, charge = cost_fixture()
    _, audit = apply_task_chunk_costs(actor, raw, charge)
    guard = FastWAMTrainingGuard(config(tmp_path).runner.fastwam_training_guard)
    metrics = dict.fromkeys(guard._ROLLOUT_KEYS, 0.0)
    metrics.update(
        {
            "fastwam/raw_positive_success_signal_count": 1.0,
            "fastwam/successful_trajectory_count": 1.0,
            "fastwam/eligible_idm_fraction": 0.5,
            "fastwam/eligible_gate_decision_count": 10.0,
            "fastwam/eligible_idm_decision_count": 5.0,
            "fastwam/valid_uncond_chunk_count": 5.0,
            **audit.to_metrics(),
        }
    )
    result = guard.observe_rollout([metrics])
    assert result["task_cost_identity"] == "PASS"
    assert result["configured_idm_cost"] is result["break_even_idm_cost"] is None
    damaged = {**metrics, "fastwam/task_budget/actual_cost_sum": 100.0}
    with pytest.raises(ValueError, match="totals"):
        FastWAMTrainingGuard(
            config(tmp_path).runner.fastwam_training_guard
        ).observe_rollout([damaged])
