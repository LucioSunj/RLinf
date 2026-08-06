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

"""Focused tests for the fail-closed FastWAM scientific-training guard."""

from __future__ import annotations

import math

import pytest
from omegaconf import OmegaConf

from rlinf.runners.fastwam_training_guard import FastWAMTrainingGuard


def _config(*, patience: int = 3):
    return OmegaConf.create(
        {
            "enabled": True,
            "zero_success_patience": patience,
            "window_size": 3,
            "eligible_idm_fraction_min": 0.05,
            "eligible_idm_fraction_max": 0.95,
            "gate_entropy_min": 0.1,
            "gate_kl_median_max": 0.05,
            "gate_kl_single_max": 0.1,
            "gate_clip_median_max": 0.6,
            "gate_clip_single_max": 0.8,
        }
    )


def _rollout_metrics(*, successes: int, idm_fraction: float = 0.5):
    return [
        {
            "fastwam/raw_positive_success_signal_count": float(successes),
            "fastwam/successful_trajectory_count": float(successes > 0),
            "fastwam/eligible_idm_fraction": idm_fraction,
            "fastwam/eligible_gate_decision_count": 100.0,
            "fastwam/eligible_idm_decision_count": 100.0 * idm_fraction,
            "fastwam/valid_uncond_chunk_count": 8.0,
            "rewards": 0.1,
            "advantages_max": 1.0,
            "advantages_mean": 0.0,
            "advantages_min": -1.0,
            "returns_max": 1.0,
            "returns_mean": 0.5,
            "returns_min": 0.0,
            "values_max": 0.5,
            "values_mean": 0.25,
            "values_min": 0.0,
        }
    ]


def _training_metrics(
    *,
    gate_kl: float = 0.02,
    gate_clip: float = 0.2,
    gate_entropy: float = 0.5,
    uncond_samples: float = 8.0,
):
    return [
        {
            "actor/grad_norm": 1.0,
            "critic/explained_variance": -0.2,
            "critic/value_clip_ratio": 0.0,
            "critic/value_loss": 0.4,
            "gate/approx_kl": gate_kl,
            "gate/clip_fraction": gate_clip,
            "gate/entropy": gate_entropy,
            "gate/ratio": 1.0,
            "gate/sample_count": 100.0,
            "uncond_flow/approx_kl": 0.001,
            "uncond_flow/clip_fraction": 0.0,
            "uncond_flow/entropy": 14.0,
            "uncond_flow/ratio": 1.0,
            "uncond_flow/sample_count": uncond_samples,
        }
    ]


def test_positive_cost_guard_stops_first_zero_success_batch() -> None:
    guard = FastWAMTrainingGuard(_config(patience=1))

    with pytest.raises(RuntimeError, match="zero sparse-success"):
        guard.observe_rollout(_rollout_metrics(successes=0))


def test_zero_cost_guard_state_round_trip_preserves_patience() -> None:
    first = FastWAMTrainingGuard(_config(patience=3))
    first.observe_rollout(_rollout_metrics(successes=0))
    first.observe_rollout(_rollout_metrics(successes=0))
    state = first.state_dict()

    resumed = FastWAMTrainingGuard(_config(patience=3))
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state
    with pytest.raises(RuntimeError, match="3 consecutive"):
        resumed.observe_rollout(_rollout_metrics(successes=0))


def test_success_resets_zero_success_streak() -> None:
    guard = FastWAMTrainingGuard(_config(patience=3))
    guard.observe_rollout(_rollout_metrics(successes=0))
    result = guard.observe_rollout(_rollout_metrics(successes=1))
    assert result["consecutive_zero_success_batches"] == 0


def test_training_guard_applies_single_and_rolling_limits() -> None:
    guard = FastWAMTrainingGuard(_config())
    for _ in range(3):
        guard.observe_rollout(_rollout_metrics(successes=1))
        result = guard.observe_training(_training_metrics())
    assert result["status"] == "PASS"
    assert result["history_count"] == 3

    guard.observe_rollout(_rollout_metrics(successes=1))
    with pytest.raises(RuntimeError, match="single-step Gate KL"):
        guard.observe_training(_training_metrics(gate_kl=0.11))


def test_single_all_idm_batch_does_not_preempt_the_rolling_collapse_rule() -> None:
    guard = FastWAMTrainingGuard(_config())
    rollout = _rollout_metrics(successes=1, idm_fraction=1.0)
    rollout[0]["fastwam/valid_uncond_chunk_count"] = 0.0
    guard.observe_rollout(rollout)
    result = guard.observe_training(_training_metrics(uncond_samples=0.0))
    assert result["status"] == "PASS"


def test_training_guard_reconciles_route_and_ppo_sample_counts() -> None:
    guard = FastWAMTrainingGuard(_config())
    guard.observe_rollout(_rollout_metrics(successes=1))
    mismatched = _training_metrics()
    mismatched[0]["gate/sample_count"] = 99.0
    with pytest.raises(RuntimeError, match="does not reconcile"):
        guard.observe_training(mismatched)


def test_training_guard_fails_closed_on_missing_or_nonfinite_metrics() -> None:
    guard = FastWAMTrainingGuard(_config())
    guard.observe_rollout(_rollout_metrics(successes=1))
    missing = _training_metrics()
    del missing[0]["gate/entropy"]
    with pytest.raises(ValueError, match="missing required metrics"):
        guard.observe_training(missing)

    guard = FastWAMTrainingGuard(_config())
    guard.observe_rollout(_rollout_metrics(successes=1))
    nonfinite = _training_metrics()
    nonfinite[0]["critic/value_loss"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        guard.observe_training(nonfinite)


def test_training_guard_rejects_route_collapse_after_full_window() -> None:
    guard = FastWAMTrainingGuard(_config())
    for _ in range(2):
        guard.observe_rollout(_rollout_metrics(successes=1, idm_fraction=0.01))
        guard.observe_training(_training_metrics())
    guard.observe_rollout(_rollout_metrics(successes=1, idm_fraction=0.01))
    with pytest.raises(RuntimeError, match="IDM fraction"):
        guard.observe_training(_training_metrics())


def test_guard_state_rejects_config_mismatch() -> None:
    source = FastWAMTrainingGuard(_config(patience=3))
    state = source.state_dict()
    target = FastWAMTrainingGuard(_config(patience=1))

    with pytest.raises(ValueError, match="config hash"):
        target.load_state_dict(state)
