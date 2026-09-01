# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fake-worker runner and checkpoint coverage for IDM cost control."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.runners.fastwam_idm_cost_control import (
    FastWAMIDMCostControlRuntime,
    validate_fastwam_idm_cost_control_config,
)
from rlinf.runners.fastwam_training_guard import FastWAMTrainingGuard

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "examples/embodiment/config"


class _Handle:
    def __init__(self, value=None) -> None:
        self.value = value

    def wait(self):
        return self.value


class _Actor:
    def __init__(self) -> None:
        self.calls = []

    def set_fastwam_idm_cost(self, cost: float, runner_step: int) -> _Handle:
        self.calls.append((cost, runner_step))
        return _Handle()

    def set_fastwam_branch_costs(
        self, idm_cost: float, uncond_cost: float, runner_step: int
    ) -> _Handle:
        assert uncond_cost == 0.0
        return self.set_fastwam_idm_cost(idm_cost, runner_step)


class _BranchActor:
    def __init__(self) -> None:
        self.calls = []

    def set_fastwam_branch_costs(
        self, idm_cost: float, uncond_cost: float, runner_step: int
    ) -> _Handle:
        self.calls.append((idm_cost, uncond_cost, runner_step))
        return _Handle()


def _controller_config(*, target: float = 0.5) -> dict:
    return {
        "type": "budget_dual",
        "constraint": "upper_bound",
        "rate": {
            "scope": "eligible_gate_decisions",
            "feedback": "expected_behavior_probability",
            "target_idm_fraction": target,
        },
        "charge_scope": "eligible_nonforced_idm",
        "initializer": {"type": "constant", "idm_cost": 0.01},
        "dual": {
            "learning_rate": 0.1,
            "ema_beta": 0.0,
            "deadband": 0.0,
            "update_interval": 1,
            "min_idm_cost": 0.0,
            "max_idm_cost": 0.2,
            "max_delta_per_update": 0.05,
        },
    }


def _cfg(tmp_path: Path, *, target: float = 0.5):
    return OmegaConf.create(
        {
            "actor": {"model": {"gate_epsilon": 0.1}},
            "rollout": {"model": {"gate_epsilon": 0.1}},
            "algorithm": {
                "fixed_branch_cost": {
                    "enabled": True,
                    "idm_cost": 0.015,
                    "uncond_cost": 0.0,
                    "fair_cost": {"enabled": False},
                    "controller": _controller_config(target=target),
                }
            },
            "runner": {
                "use_training_pipeline": False,
                "weight_sync_interval": 1,
                "fastwam_training_guard": {"enabled": True},
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "dual-test",
                },
            },
        }
    )


def _band_cfg(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.routing_objective = {"type": "train_band_target"}
    cfg.algorithm.fixed_branch_cost.idm_cost = 0.0
    cfg.algorithm.fixed_branch_cost.controller = {
        "type": "band_price",
        "constraint": "two_sided_band",
        "rate": {
            "scope": "eligible_gate_decisions",
            "feedback": "expected_behavior_probability",
            "target_idm_fraction": 0.5,
            "half_width": 0.03,
        },
        "charge_scope": "eligible_nonforced",
        "signed_price": {
            "initial_value": 0.0,
            "learning_rate": 1.0,
            "ema_beta": 0.0,
            "update_interval": 1,
            "max_abs_value": 0.2,
            "max_delta_per_update": 0.2,
        },
    }
    return cfg


def _worker_metrics(*, probability: float = 0.8) -> list[dict]:
    return [
        {
            "fastwam/eligible_gate_decision_count": 100.0,
            "fastwam/eligible_idm_decision_count": 80.0,
            "fastwam/gate/behavior_idm_probability_mean": probability,
            "fastwam/route/valid_chunk_count": 110.0,
            "fastwam/route/valid_idm_chunk_count": 85.0,
            "fastwam/route/forced_count": 10.0,
        }
    ]


def _guard_result() -> dict:
    return {
        "status": "PASS",
        "eligible_gate_decision_count": 100,
        "eligible_idm_decision_count": 80,
        "eligible_idm_fraction": 0.8,
        "valid_chunk_count": 110,
        "valid_idm_chunk_count": 85,
        "forced_route_count": 10,
        "break_even_idm_cost": None,
        "configured_idm_cost": 0.01,
    }


def test_runtime_calls_actor_before_observation_and_applies_feedback_next_step(
    tmp_path: Path,
) -> None:
    runtime = FastWAMIDMCostControlRuntime.from_config(_cfg(tmp_path))
    actor = _Actor()

    decision0 = runtime.before_rollout(actor=actor, runner_step=0)
    record0 = runtime.after_rollout(
        runner_step=0,
        actor_rollout_metrics=_worker_metrics(),
        guard_result=_guard_result(),
    )
    decision1 = runtime.before_rollout(actor=actor, runner_step=1)

    assert decision0.applied_idm_cost == pytest.approx(0.01)
    assert record0["applied"]["applied_idm_cost"] == pytest.approx(0.01)
    assert record0["next"]["applied_idm_cost"] == pytest.approx(0.04)
    assert decision1.applied_idm_cost == pytest.approx(0.04)
    assert actor.calls[0] == (0.01, 0)
    assert actor.calls[1][0] == pytest.approx(0.04)
    assert actor.calls[1][1] == 1
    artifact = tmp_path / "dual-test/audits/idm_cost_control.jsonl"
    assert json.loads(artifact.read_text())["runner_step"] == 0


def test_runtime_publishes_both_band_costs_with_one_rollout_lag(
    tmp_path: Path,
) -> None:
    cfg = _band_cfg(tmp_path)
    validate_fastwam_idm_cost_control_config(cfg)
    runtime = FastWAMIDMCostControlRuntime.from_config(cfg)
    actor = _BranchActor()
    first = runtime.before_rollout(actor=actor, runner_step=0)
    record = runtime.after_rollout(
        runner_step=0,
        actor_rollout_metrics=_worker_metrics(probability=0.8),
        guard_result=_guard_result(),
    )
    second = runtime.before_rollout(actor=actor, runner_step=1)
    assert first.idm_cost == first.uncond_cost == 0.0
    assert second.idm_cost == pytest.approx(record["next"]["idm_cost"])
    assert second.idm_cost > 0.0
    assert second.uncond_cost == 0.0
    assert actor.calls == [
        (0.0, 0.0, 0),
        (pytest.approx(second.idm_cost), 0.0, 1),
    ]
    artifact = tmp_path / "dual-test/audits/branch_cost_control.jsonl"
    assert json.loads(artifact.read_text())["controller_type"] == "band_price"


def _bare_runner(cfg, runtime, *, global_step: int) -> EmbodiedRunner:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = cfg
    runner.global_step = global_step
    runner.fastwam_idm_cost_control = runtime
    runner.fastwam_training_guard = FastWAMTrainingGuard({"enabled": False})
    return runner


def test_v3_checkpoint_round_trip_preserves_next_decision(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    source_runtime = FastWAMIDMCostControlRuntime.from_config(cfg)
    source_runtime.before_rollout(actor=_Actor(), runner_step=0)
    source_runtime.after_rollout(
        runner_step=0,
        actor_rollout_metrics=_worker_metrics(),
        guard_result=_guard_result(),
    )
    source = _bare_runner(cfg, source_runtime, global_step=1)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source._save_fastwam_training_guard(str(checkpoint))

    payload = json.loads((checkpoint / "training_guard.json").read_text())
    assert payload["schema"] == "fastwam-training-guard-checkpoint-v3"
    assert payload["idm_cost_controller"]["controller_type"] == "budget_dual"

    resumed_runtime = FastWAMIDMCostControlRuntime.from_config(cfg)
    resumed = _bare_runner(cfg, resumed_runtime, global_step=1)
    resumed._load_fastwam_training_guard(str(checkpoint))
    source_actor = _Actor()
    resumed_actor = _Actor()
    assert source_runtime.before_rollout(
        actor=source_actor, runner_step=1
    ) == resumed_runtime.before_rollout(actor=resumed_actor, runner_step=1)
    assert source_actor.calls == resumed_actor.calls

    changed = _bare_runner(
        _cfg(tmp_path, target=0.6),
        FastWAMIDMCostControlRuntime.from_config(_cfg(tmp_path, target=0.6)),
        global_step=1,
    )
    with pytest.raises(ValueError, match="config hash mismatch"):
        changed._load_fastwam_training_guard(str(checkpoint))


def test_band_price_v3_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = _band_cfg(tmp_path)
    source_runtime = FastWAMIDMCostControlRuntime.from_config(cfg)
    source_runtime.before_rollout(actor=_BranchActor(), runner_step=0)
    source_runtime.after_rollout(
        runner_step=0,
        actor_rollout_metrics=_worker_metrics(probability=0.8),
        guard_result=_guard_result(),
    )
    source = _bare_runner(cfg, source_runtime, global_step=1)
    checkpoint = tmp_path / "band-checkpoint"
    checkpoint.mkdir()
    source._save_fastwam_training_guard(str(checkpoint))

    restored_runtime = FastWAMIDMCostControlRuntime.from_config(cfg)
    restored = _bare_runner(cfg, restored_runtime, global_step=1)
    restored._load_fastwam_training_guard(str(checkpoint))
    assert source_runtime.before_rollout(
        actor=_BranchActor(), runner_step=1
    ) == restored_runtime.before_rollout(actor=_BranchActor(), runner_step=1)


def test_config_validation_rejects_pipeline_infeasible_target_and_scope_mismatch(
    tmp_path: Path,
) -> None:
    valid = _cfg(tmp_path)
    validate_fastwam_idm_cost_control_config(valid)

    pipeline = copy.deepcopy(valid)
    pipeline.runner.use_training_pipeline = True
    with pytest.raises(ValueError, match="use_training_pipeline=false"):
        validate_fastwam_idm_cost_control_config(pipeline)

    stale_weights = copy.deepcopy(valid)
    stale_weights.runner.weight_sync_interval = 2
    with pytest.raises(ValueError, match="weight_sync_interval=1"):
        validate_fastwam_idm_cost_control_config(stale_weights)

    infeasible = copy.deepcopy(valid)
    infeasible.actor.model.gate_epsilon = 0.4
    infeasible.rollout.model.gate_epsilon = 0.4
    infeasible.algorithm.fixed_branch_cost.controller.rate.target_idm_fraction = 0.1
    with pytest.raises(ValueError, match="reachable interval"):
        validate_fastwam_idm_cost_control_config(infeasible)

    mismatch = copy.deepcopy(valid)
    mismatch.algorithm.fixed_branch_cost.controller.charge_scope = "all_valid_idm"
    with pytest.raises(ValueError, match="rate scope and charge scope"):
        validate_fastwam_idm_cost_control_config(mismatch)

    ambiguous = copy.deepcopy(valid)
    ambiguous.algorithm.fixed_branch_cost.fair_cost.enabled = True
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_fastwam_idm_cost_control_config(ambiguous)

    missing_fair_audit = copy.deepcopy(valid)
    missing_fair_audit.algorithm.fixed_branch_cost.controller.initializer = {
        "type": "break_even_median",
        "bootstrap_idm_cost": 0.015,
        "window_size": 5,
        "warmup_rollouts": 5,
        "minimum_valid_observations": 3,
        "insufficient_data_policy": "keep_bootstrap",
        "monitor_after_warmup": True,
    }
    with pytest.raises(ValueError, match="fair warm-start requires"):
        validate_fastwam_idm_cost_control_config(missing_fair_audit)

    band = _band_cfg(tmp_path)
    validate_fastwam_idm_cost_control_config(band)
    wrong_objective = copy.deepcopy(band)
    wrong_objective.routing_objective.type = "upper_bound"
    with pytest.raises(ValueError, match="train_band_target"):
        validate_fastwam_idm_cost_control_config(wrong_objective)

    unreachable_band = copy.deepcopy(band)
    unreachable_band.algorithm.fixed_branch_cost.controller.rate.target_idm_fraction = (
        0.94
    )
    with pytest.raises(ValueError, match="reachable interval"):
        validate_fastwam_idm_cost_control_config(unreachable_band)

    zero_init = copy.deepcopy(valid)
    zero_init.routing_objective = {"type": "upper_bound"}
    zero_init.algorithm.fixed_branch_cost.controller.profile = "upper_bound_zero_init"
    zero_init.algorithm.fixed_branch_cost.controller.initializer.idm_cost = 0.01
    with pytest.raises(ValueError, match="constant zero initialization"):
        validate_fastwam_idm_cost_control_config(zero_init)


@pytest.mark.parametrize(
    ("name", "controller_type", "target"),
    (
        ("fixed", "fixed", None),
        ("legacy_fair", "legacy_fair", None),
        ("legacy_fair_pi", "legacy_fair_pi", None),
        ("budget_dual_constant", "budget_dual", 0.5),
        ("budget_dual_fair_warmstart_b25", "budget_dual", 0.25),
        ("budget_dual_fair_warmstart_b50", "budget_dual", 0.5),
        ("budget_dual_fair_warmstart_b75", "budget_dual", 0.75),
        ("upper_bound_zero_init_b25", "budget_dual", 0.25),
        ("upper_bound_zero_init_b50", "budget_dual", 0.5),
        ("upper_bound_zero_init_b75", "budget_dual", 0.75),
        ("band_price_b25", "band_price", 0.25),
        ("band_price_b50", "band_price", 0.5),
        ("band_price_b75", "band_price", 0.75),
    ),
)
def test_hydra_cost_control_group_composes_single_explicit_source(
    name: str,
    controller_type: str,
    target: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_ROOT.parent))
    monkeypatch.setenv("FASTWAM_CALIBRATION_LEDGER", "/calibration-ledger.json")
    monkeypatch.setenv("FASTWAM_TEST_LEDGER", "/test-ledger.json")
    monkeypatch.setenv("FASTWAM_BUDGET_EVAL_OUTPUT_DIR", "/budget-eval")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(
            config_name="libero_10_ppo_fastwam_adaptive",
            overrides=[f"+fastwam_idm_cost_control={name}"],
        )

    controller = cfg.algorithm.fixed_branch_cost.controller
    assert controller.type == controller_type
    assert not cfg.algorithm.fixed_branch_cost.fair_cost.enabled
    if target is not None:
        assert controller.rate.target_idm_fraction == pytest.approx(target)
    validate_fastwam_idm_cost_control_config(cfg)


@pytest.mark.parametrize(
    ("name", "target"),
    (
        ("rate_matched_b25", 0.25),
        ("rate_matched_b50", 0.5),
        ("rate_matched_b75", 0.75),
    ),
)
def test_hydra_budget_evaluation_profiles_compose(
    name: str,
    target: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_ROOT.parent))
    monkeypatch.setenv("FASTWAM_CALIBRATION_LEDGER", "/calibration-ledger.json")
    monkeypatch.setenv("FASTWAM_TEST_LEDGER", "/test-ledger.json")
    monkeypatch.setenv("FASTWAM_BUDGET_EVAL_OUTPUT_DIR", "/budget-eval")
    monkeypatch.setenv("FASTWAM_STEP0_CHECKPOINT", "/step0")
    monkeypatch.setenv("FASTWAM_STEP5_CHECKPOINT", "/step5")
    monkeypatch.setenv("FASTWAM_STEP10_CHECKPOINT", "/step10")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(
            config_name="libero_10_ppo_fastwam_adaptive",
            overrides=[f"+fastwam_budget_evaluation={name}"],
        )
    budget = cfg.evaluation.budget_matching
    assert cfg.routing_objective.type == "eval_calibrated_target"
    assert budget.target_idm_fraction == pytest.approx(target)
    assert budget.calibration.selection.success_blind
    assert budget.calibration.reset_ledger_path != budget.test.reset_ledger_path
    from rlinf.runners.fastwam_budget_calibration import (
        validate_fastwam_budget_evaluation_config,
    )

    validate_fastwam_budget_evaluation_config(cfg)


@pytest.mark.parametrize(
    "prefix",
    ("upper_bound_zero_init", "band_price", "rate_matched"),
)
def test_budget_overlays_only_change_target(
    prefix: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_ROOT.parent))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/checkpoint")
    monkeypatch.setenv("FASTWAM_CALIBRATION_LEDGER", "/calibration-ledger.json")
    monkeypatch.setenv("FASTWAM_TEST_LEDGER", "/test-ledger.json")
    monkeypatch.setenv("FASTWAM_STEP0_CHECKPOINT", "/step0")
    monkeypatch.setenv("FASTWAM_STEP5_CHECKPOINT", "/step5")
    monkeypatch.setenv("FASTWAM_STEP10_CHECKPOINT", "/step10")
    group = (
        "fastwam_budget_evaluation"
        if prefix == "rate_matched"
        else "fastwam_idm_cost_control"
    )
    resolved = []
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        for suffix in ("b25", "b50", "b75"):
            cfg = compose(
                config_name="libero_10_ppo_fastwam_adaptive",
                overrides=[f"+{group}={prefix}_{suffix}"],
            )
            if prefix == "rate_matched":
                container = OmegaConf.to_container(
                    cfg.evaluation.budget_matching, resolve=True
                )
                del container["target_idm_fraction"]
            else:
                container = OmegaConf.to_container(
                    cfg.algorithm.fixed_branch_cost.controller, resolve=True
                )
                del container["rate"]["target_idm_fraction"]
            resolved.append(container)
    assert resolved[0] == resolved[1] == resolved[2]
