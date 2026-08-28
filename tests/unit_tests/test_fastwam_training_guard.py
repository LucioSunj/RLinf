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

import json
import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.algorithms.advantages import (
    FastWAMCounterfactualCostAudit,
    FastWAMCounterfactualCostEntry,
    FastWAMScalarAudit,
)
from rlinf.runners.fastwam_training_guard import (
    FastWAMTrainingGuard,
    append_fastwam_counterfactual_cost_audit_jsonl,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def _config(*, patience: int = 3, cost_audit: bool = False):
    return OmegaConf.create(
        {
            "enabled": True,
            "zero_success_patience": patience,
            "break_even_patience": 3,
            "window_size": 3,
            "eligible_idm_fraction_min": 0.05,
            "eligible_idm_fraction_max": 0.95,
            "gate_entropy_min": 0.1,
            "gate_kl_median_max": 0.05,
            "gate_kl_single_max": 0.1,
            "gate_clip_median_max": 0.6,
            "gate_clip_single_max": 0.8,
            "cost_audit": {"enabled": cost_audit},
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
    gate_samples: float = 100.0,
    uncond_samples: float = 8.0,
    effective_gate_samples: float | None = None,
    full_eligible_gate_samples: float | None = None,
):
    metrics = {
        "actor/grad_norm": 1.0,
        "critic/explained_variance": -0.2,
        "critic/value_clip_ratio": 0.0,
        "critic/value_loss": 0.4,
        "gate/approx_kl": gate_kl,
        "gate/clip_fraction": gate_clip,
        "gate/entropy": gate_entropy,
        "gate/ratio": 1.0,
        "gate/sample_count": gate_samples,
        "uncond_flow/approx_kl": 0.001,
        "uncond_flow/clip_fraction": 0.0,
        "uncond_flow/entropy": 14.0,
        "uncond_flow/ratio": 1.0,
        "uncond_flow/sample_count": uncond_samples,
    }
    if effective_gate_samples is not None:
        metrics["kv_cache/effective_gate_gradient_count"] = effective_gate_samples
    if full_eligible_gate_samples is not None:
        metrics["kv_cache/full_eligible_gate_samples"] = full_eligible_gate_samples
    return [metrics]


def _scalar(total: float, count: int) -> FastWAMScalarAudit:
    return FastWAMScalarAudit(
        count=count,
        finite_count=count,
        nonfinite_count=0,
        minimum=None,
        maximum=None,
        total=total,
    )


def _counterfactual_audit_from_v9_record(
    record: dict[str, object],
) -> FastWAMCounterfactualCostAudit:
    placeholder = _scalar(0.0, 1)
    entries = []
    for key, idm_cost in (("zero", 0.0), ("max", record["max"]["idm_cost"])):
        values = record[key]
        entries.append(
            FastWAMCounterfactualCostEntry(
                idm_cost=float(idm_cost),
                expected_cost_sum=0.0,
                unnormalized_gate_advantage=placeholder,
                normalized_gate_advantage=placeholder,
                unnormalized_idm_gate_advantage=_scalar(
                    float(values["idm_sum"]), int(values["idm_count"])
                ),
                normalized_idm_gate_advantage=placeholder,
                unnormalized_uncond_gate_advantage=_scalar(
                    float(values["uncond_sum"]), int(values["uncond_count"])
                ),
                normalized_uncond_gate_advantage=placeholder,
                unnormalized_idm_delta_from_zero=placeholder,
                normalized_idm_delta_from_zero=placeholder,
            )
        )
    zero = record["zero"]
    return FastWAMCounterfactualCostAudit(
        configured_idm_cost=float(record["configured_idm_cost"]),
        configured_alignment_max_abs_error=0.0,
        eligible_gate_decision_count=int(zero["idm_count"]) + int(zero["uncond_count"]),
        eligible_idm_decision_count=int(zero["idm_count"]),
        eligible_uncond_decision_count=int(zero["uncond_count"]),
        entries=tuple(entries),
    )


def test_positive_cost_guard_stops_first_zero_success_batch() -> None:
    guard = FastWAMTrainingGuard(_config(patience=1))

    with pytest.raises(RuntimeError, match="zero sparse-success"):
        guard.observe_rollout(_rollout_metrics(successes=0))


def test_guard_rejects_inverted_eligible_idm_fraction_bounds() -> None:
    invalid = _config()
    invalid.eligible_idm_fraction_min = 0.9
    invalid.eligible_idm_fraction_max = 0.5

    with pytest.raises(ValueError, match="fraction bounds"):
        FastWAMTrainingGuard(invalid)


def test_break_even_guard_rejects_insufficient_route_history() -> None:
    invalid = _config(cost_audit=True)
    invalid.break_even_patience = 4
    invalid.window_size = 2

    with pytest.raises(ValueError, match="route-history window"):
        FastWAMTrainingGuard(invalid)


def test_v9_counterfactual_replay_triggers_break_even_guard_at_update_7() -> None:
    fixture = json.loads(
        (FIXTURE_ROOT / "fastwam_v9_counterfactual_break_even.json").read_text(
            encoding="utf-8"
        )
    )
    records = fixture["records"]
    assert len(records) == 21
    assert [record["actor_version"] for record in records] == list(range(21))
    audits = [_counterfactual_audit_from_v9_record(record) for record in records]
    break_evens = [audit.break_even_idm_cost for audit in audits]
    assert break_evens[0] == pytest.approx(0.022167795441826472)
    assert break_evens[1] == pytest.approx(0.04514449758500185)
    assert break_evens[3] is None
    assert break_evens[5] == pytest.approx(0.02461735974379635)
    assert break_evens[6] == pytest.approx(0.013035805397631172)
    assert break_evens[7] == pytest.approx(0.005439436564658173)
    assert break_evens[20] == pytest.approx(0.0037622571305554143)

    guard = FastWAMTrainingGuard(_config(cost_audit=True))
    trigger_version = None
    for record, audit in zip(records, audits, strict=True):
        idm_count = int(record["zero"]["idm_count"])
        uncond_count = int(record["zero"]["uncond_count"])
        eligible_count = idm_count + uncond_count
        rollout = _rollout_metrics(
            successes=1,
            idm_fraction=idm_count / eligible_count,
        )
        rollout[0]["fastwam/eligible_gate_decision_count"] = float(eligible_count)
        rollout[0]["fastwam/eligible_idm_decision_count"] = float(idm_count)
        rollout[0]["fastwam/valid_uncond_chunk_count"] = float(uncond_count)
        rollout[0].update(audit.to_metrics())
        try:
            guard.observe_rollout(rollout)
        except RuntimeError as error:
            assert "break-even IDM cost" in str(error)
            assert "declined monotonically" in str(error)
            trigger_version = record["actor_version"]
            break
        guard.observe_training(
            _training_metrics(
                gate_samples=float(eligible_count),
                uncond_samples=float(uncond_count),
            )
        )
        if record["actor_version"] == 6:
            state = guard.state_dict()
            resumed = FastWAMTrainingGuard(_config(cost_audit=True))
            resumed.load_state_dict(state)
            guard = resumed

    assert trigger_version == 7


def test_v10_stable_route_does_not_trip_break_even_guard() -> None:
    guard = FastWAMTrainingGuard(_config(cost_audit=True))
    v10_updates_21_to_23 = (
        (512, 464, 48),
        (383, 347, 36),
        (424, 386, 38),
    )

    for eligible_count, idm_count, uncond_count in v10_updates_21_to_23:
        rollout = _rollout_metrics(
            successes=1,
            idm_fraction=idm_count / eligible_count,
        )
        rollout[0]["fastwam/eligible_gate_decision_count"] = float(eligible_count)
        rollout[0]["fastwam/eligible_idm_decision_count"] = float(idm_count)
        rollout[0]["fastwam/valid_uncond_chunk_count"] = float(uncond_count)
        rollout[0]["fastwam/counterfactual/break_even_idm_cost"] = None
        rollout[0]["fastwam/counterfactual/configured_idm_cost"] = 0.01
        result = guard.observe_rollout(rollout)
        guard.observe_training(
            _training_metrics(
                gate_samples=float(eligible_count),
                uncond_samples=float(uncond_count),
            )
        )

    assert result["consecutive_break_even_below_cost"] == 3
    assert result["break_even_route_window"] == pytest.approx(
        [464 / 512, 347 / 383, 386 / 424]
    )
    assert result["break_even_route_monotonic_decline"] is False


def test_counterfactual_cost_audit_appends_full_jsonl_tables(tmp_path: Path) -> None:
    path = tmp_path / "run/audits/counterfactual_cost_audit.jsonl"
    scalar = {
        "count": 2,
        "finite_count": 2,
        "nonfinite_count": 0,
        "minimum": -1.0,
        "maximum": 1.0,
        "sum": 0.0,
        "sum_of_squares": 2.0,
        "quantiles": {
            "p10": -0.8,
            "p25": -0.5,
            "p50": 0.0,
            "p75": 0.5,
            "p90": 0.8,
        },
    }
    groups = {
        name: {"unnormalized": scalar, "normalized": scalar}
        for name in (
            "gate_advantage",
            "idm_destination_gate_advantage",
            "uncond_destination_gate_advantage",
            "idm_destination_delta_from_zero",
        )
    }
    artifact = {
        "schema": "fastwam-counterfactual-cost-audit-v1",
        "configured_idm_cost": 0.01,
        "break_even_idm_cost": 0.0275,
        "entries": [
            {"idm_cost": 0.0, **groups},
            {"idm_cost": 0.1, **groups},
        ],
    }

    append_fastwam_counterfactual_cost_audit_jsonl(
        path, runner_step=0, artifact=artifact
    )
    append_fastwam_counterfactual_cost_audit_jsonl(
        path, runner_step=1, artifact=artifact
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["runner_step"] for row in rows] == [0, 1]
    assert all(row["entries"] == artifact["entries"] for row in rows)
    assert all(row["break_even_idm_cost"] == 0.0275 for row in rows)


def test_counterfactual_cost_audit_jsonl_requires_variance_fields(
    tmp_path: Path,
) -> None:
    artifact = {
        "schema": "fastwam-counterfactual-cost-audit-v1",
        "entries": [
            {
                "gate_advantage": {
                    "unnormalized": {},
                    "normalized": {},
                }
            }
        ],
    }

    with pytest.raises(ValueError, match="sum_of_squares"):
        append_fastwam_counterfactual_cost_audit_jsonl(
            tmp_path / "audit.jsonl", runner_step=0, artifact=artifact
        )


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


def test_training_guard_reconciles_sampled_gate_ppo_with_effective_count() -> None:
    guard = FastWAMTrainingGuard(_config())
    guard.observe_rollout(_rollout_metrics(successes=1))

    result = guard.observe_training(
        _training_metrics(
            gate_samples=40.0,
            effective_gate_samples=40.0,
            full_eligible_gate_samples=100.0,
        )
    )

    assert result["gate_sample_count_basis"] == "sampled_effective"
    assert result["expected_gate_sample_count"] == 40.0
    assert result["observed_gate_sample_count"] == 40.0


def test_training_guard_rejects_sampled_gate_count_disagreement() -> None:
    guard = FastWAMTrainingGuard(_config())
    guard.observe_rollout(_rollout_metrics(successes=1))

    with pytest.raises(RuntimeError, match="Gate PPO sample count"):
        guard.observe_training(
            _training_metrics(
                gate_samples=39.0,
                effective_gate_samples=40.0,
                full_eligible_gate_samples=100.0,
            )
        )


def test_training_guard_rejects_sampled_full_eligible_disagreement() -> None:
    guard = FastWAMTrainingGuard(_config())
    guard.observe_rollout(_rollout_metrics(successes=1))

    with pytest.raises(RuntimeError, match="full eligible Gate count"):
        guard.observe_training(
            _training_metrics(
                gate_samples=40.0,
                effective_gate_samples=40.0,
                full_eligible_gate_samples=99.0,
            )
        )


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
