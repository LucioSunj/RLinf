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


@pytest.mark.parametrize(
    "suite", ("libero_10", "libero_spatial", "libero_object", "libero_goal")
)
def test_native_runtime_roundtrip_and_changed_table_rejection(tmp_path, suite):
    cfg = config(tmp_path)
    cfg.env.train.task_suite_name = suite
    cfg.algorithm.fixed_branch_cost.task_budget.suite = suite
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
    cfg.algorithm.fixed_branch_cost.task_budget.targets["0"] = 0.3
    other_suite = "libero_goal" if suite != "libero_goal" else "libero_10"
    cfg.env.train.task_suite_name = other_suite
    cfg.algorithm.fixed_branch_cost.task_budget.suite = other_suite
    with pytest.raises(ValueError, match="config hash"):
        TaskBudgetRuntime.from_config(cfg).load_state_dict(state, global_step=3)
    assert len(published) == 5


@pytest.mark.parametrize(
    "training_suite,budget_suite",
    (
        ("libero_spatial", "libero_10"),
        ("libero_goal", "libero_object"),
        ("libero_90", "libero_90"),
    ),
)
def test_budget_suite_must_match_supported_training_suite(
    tmp_path, training_suite, budget_suite
):
    cfg = config(tmp_path)
    cfg.env.train.task_suite_name = training_suite
    cfg.algorithm.fixed_branch_cost.task_budget.suite = budget_suite
    with pytest.raises(ValueError, match="supported ten-task LIBERO training suite"):
        FastWAMIDMCostControlRuntime.from_config(cfg)


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


def fr3_table(task_ids=(0, 1)):
    names = {0: "drawer", 1: "insert_tubes"}
    return {
        "profile": "fr3",
        "suite": "fr3",
        "targets": {str(task): 0.3 if task == 0 else 0.7 for task in task_ids},
        "task_names": {str(task): names[task] for task in task_ids},
        "calibration": {
            "status": "COMPLETE",
            "source": "fixture/budget_targets.json",
            "data_version": "CPU fixture, not physical calibration",
            "parameters": {
                "awe_threshold": 0.005,
                "sail_radius_m": 0.04,
                "sail_min_samples": 3,
                "execution_horizon": 16,
            },
        },
    }


@pytest.mark.parametrize("active", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("radius", [0.025, 0.03, 0.04])
def test_fr3_sparse_task_runtime_resume_and_calibration_contract(
    tmp_path, active, radius
):
    cfg = config(tmp_path)
    cfg.algorithm.fixed_branch_cost.task_budget = fr3_table(active)
    cfg.algorithm.fixed_branch_cost.task_budget.calibration.parameters.sail_radius_m = (
        radius
    )
    cfg.env.train.task_suite_name = "fr3"
    cfg.env.train.task_id_filter = list(active)
    cfg.env.train.task_sampling = "sequential_episodes"
    cfg.env.train.num_envs = 1
    cfg.cluster.component_placement.actor = "learner:0"
    runtime = TaskBudgetRuntime.from_config(cfg)
    assert set(runtime.controller.controllers) == set(active)
    for step in range(3):
        decision = runtime.controller.decisions(step)
        assert set(decision["tasks"]) == {str(task) for task in active}
        metrics = {
            key: value
            for key, value in feedback().items()
            if int(key.split("/")[1]) in active
        }
        runtime.controller.observe(step, metrics)
    state = runtime.state_dict()
    restored = TaskBudgetRuntime.from_config(cfg)
    restored.load_state_dict(copy.deepcopy(state), global_step=3)
    assert restored.state_dict() == state
    changed = copy.deepcopy(cfg)
    changed.algorithm.fixed_branch_cost.task_budget.calibration.parameters.sail_radius_m = (
        0.04 if radius != 0.04 else 0.025
    )
    with pytest.raises(ValueError, match="config hash"):
        TaskBudgetRuntime.from_config(changed).load_state_dict(state, global_step=3)
    assert all(
        (runtime.controller.controllers[task].signed_price > 0) == (task == 0)
        for task in active
    )
    cfg.algorithm.fixed_branch_cost.task_budget.calibration.data_version = (
        "different demonstrations"
    )
    with pytest.raises(ValueError, match="config hash"):
        TaskBudgetRuntime.from_config(cfg).load_state_dict(state, global_step=3)
    cfg.algorithm.fixed_branch_cost.task_budget.calibration.status = "NOT_READY"
    with pytest.raises(ValueError, match="COMPLETE"):
        TaskBudgetRuntime.from_config(cfg)


def test_fr3_no_feedback_hold_and_duplicate_rollout_rejection():
    bank = TaskBudgetController(controller_config(warmup=1), fr3_table())
    for step in (0, 1):
        bank.decisions(step)
        bank.observe(step, feedback(0.9))
    old = bank.controllers[0].state_dict()
    bank.decisions(2)
    metrics = feedback(0.1)
    for name in (
        "gate_chunks",
        "eligible_idm_chunks",
        "valid_chunks",
        "idm_chunks",
        "forced_chunks",
        "behavior_probability_sum",
    ):
        metrics[f"task/0/{name}"] = 0
    result = bank.observe(2, metrics)
    assert result["tasks"]["0"]["status"] == "HOLD_NO_ELIGIBLE_FEEDBACK"
    assert bank.controllers[0].state_dict() == {**old, "observed_runner_steps": 3}
    with pytest.raises(RuntimeError, match="pending"):
        bank.observe(2, metrics)


def fr3_cost_fixture():
    actor, raw, charge = cost_fixture()
    batch = actor.rollout_batch
    raw = raw[:, 1:2]
    charge = charge[:, 1:2]
    route = batch["route_info"]
    batch["route_info"] = SimpleNamespace(
        route_used=route.route_used[:, 1:2],
        route_was_forced=route.route_was_forced[:, 1:2],
        shape=(3, 1),
    )
    batch["loss_mask"] = batch["loss_mask"][:, 1:2]
    batch["chunk_valid"] = batch["loss_mask"].clone()
    batch["forward_inputs"]["multitask_task_id"] = batch["forward_inputs"][
        "multitask_task_id"
    ][:, 1:2]
    batch["rewards"] = raw
    actor._fastwam_task_cost_decision["tasks"] = {
        "1": actor._fastwam_task_cost_decision["tasks"]["1"]
    }
    return actor, raw, charge


def test_fr3_cost_audit_sparse_id1_keeps_partial_chunk_decision():
    actor, raw, charge = fr3_cost_fixture()
    actor.rollout_batch["loss_mask"] = torch.zeros((3, 1, 16), dtype=torch.bool)
    actor.rollout_batch["loss_mask"][:2, :, 0] = True
    result, audit = apply_task_chunk_costs(actor, raw, charge)
    torch.testing.assert_close(result.costs[:, 0, 0], torch.tensor([0.002, 0.018, 0]))
    assert set(audit.artifact["tasks"]) == {"1"}
    validate_task_cost_metrics([audit.to_metrics()])


def test_fr3_feedback_active_tasks_no_padded_fake_episodes():
    actor, _, charge = fr3_cost_fixture()
    batch = actor.rollout_batch
    valid = batch["chunk_valid"][..., 0]
    teacher = valid & (batch["route_info"].route_used == int(WAMRoute.UNCOND))
    batch.update(
        gate_valid_mask=charge,
        emitted_gate=SimpleNamespace(behavior_probability=torch.full((3, 1), 0.6)),
        returns=torch.zeros((3, 1, 1)),
        prev_values=torch.zeros((4, 1, 1)),
    )
    batch["forward_inputs"].update(
        online_idm_bc_teacher_present=teacher,
        multitask_episode_slot_id=torch.full((3, 1), 7),
        multitask_episode_success=torch.ones((3, 1)),
        multitask_episode_failed=torch.zeros((3, 1)),
        multitask_episode_truncated=torch.zeros((3, 1)),
    )
    metrics = summarize_task_rollout(batch, active_task_ids=(0, 1))
    assert metrics["task/0/gate_chunks"] == 0
    assert metrics["task/0/valid_chunks"] == 0
    assert metrics["task/0/episode_slots"] == 0
    assert metrics["task/1/gate_chunks"] == 2
    assert metrics["task_global/episode_slots"] == 1
    assert metrics["task_macro/training_success"] == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_fr3_task_flow_metrics_mask_unexecuted_tail_before_sum():
    from rlinf.models.embodiment.wam_policy.route_neutral_online.task_metrics import (
        accumulate_task_losses,
        finalize_task_losses,
    )

    batch = {
        "forward_inputs": {"multitask_task_id": torch.tensor([1])},
        "route_info": SimpleNamespace(route_used=torch.tensor([int(WAMRoute.UNCOND)])),
        "gate_valid_mask": torch.tensor([True]),
        "flow_valid_mask": torch.tensor([True]),
        "gate_advantages": torch.tensor([1.0]),
        "flow_advantages": torch.tensor([-1.0]),
        "emitted_gate": SimpleNamespace(old_logprob=torch.tensor([0.0])),
        "prev_logprobs": torch.zeros((1, 16, 8)),
        "flow_execution_mask": torch.tensor([[True] + [False] * 15]),
    }
    output = {
        "gate_logprobs": torch.tensor([0.0]),
        "flow_logprobs": torch.full((1, 16, 8), 100.0),
        "online_idm_bc_per_sample_loss": torch.tensor([0.25]),
    }
    output["flow_logprobs"][:, 0] = 0
    cfg = OmegaConf.create(
        {
            "algorithm": {
                owner: {"clip_ratio_low": 0.2, "clip_ratio_high": 0.2}
                for owner in ("gate_ppo", "uncond_flow_ppo")
            }
        }
    )
    totals = {}
    accumulate_task_losses(totals, batch, output, cfg, active_task_ids=(1,))
    metrics = finalize_task_losses(totals, active_task_ids=(1,))
    assert metrics["task/1/flow_sample_count"] == 1
    assert metrics["task/1/flow_raw_loss"] == 1
    assert metrics["task/1/gate_sample_count"] == 1
    assert metrics["task/1/bc_raw_loss"] == 0.25
    assert not any(key.startswith("task/0/") for key in metrics)
