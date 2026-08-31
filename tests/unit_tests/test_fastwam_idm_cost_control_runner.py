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
    ),
)
def test_hydra_cost_control_group_composes_single_explicit_source(
    name: str,
    controller_type: str,
    target: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_ROOT.parent))
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
