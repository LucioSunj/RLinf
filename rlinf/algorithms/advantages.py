# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.models.embodiment.wam_policy.contracts import (
    ChunkRouteRecord,
    GateDecisionRecord,
    WAMRoute,
)
from rlinf.utils.checkpoint_state import checkpoint_state_sha256
from rlinf.utils.utils import masked_mean

FASTWAM_REWARD_AUDIT_SCHEMA = "fastwam-environment-reward-audit-v1"
FASTWAM_REWARD_AUDIT_SENTINEL = "FASTWAM_SHORT_RL_REWARD_AUDIT"
FASTWAM_ROLLOUT_STATE_AUDIT_SCHEMA = "fastwam-rollout-state-audit-v1"
FASTWAM_ROLLOUT_STATE_AUDIT_SENTINEL = "FASTWAM_SHORT_RL_ROLLOUT_STATE_AUDIT"
FASTWAM_CHUNK_COST_AUDIT_SCHEMA = "fastwam-chunk-cost-audit-v1"
FASTWAM_CHUNK_COST_AUDIT_SENTINEL = "FASTWAM_TRAINING_COST_AUDIT"
FASTWAM_COUNTERFACTUAL_COST_AUDIT_SCHEMA = "fastwam-counterfactual-cost-audit-v1"
FASTWAM_COUNTERFACTUAL_COST_AUDIT_SENTINEL = (
    "FASTWAM_TRAINING_COUNTERFACTUAL_COST_AUDIT"
)
FASTWAM_GATE_UPDATE_AUDIT_SCHEMA = "fastwam-gate-update-audit-v1"
FASTWAM_GATE_UPDATE_AUDIT_SENTINEL = "FASTWAM_TRAINING_GATE_UPDATE_AUDIT"


def compute_fastwam_break_even_idm_cost(
    points: Sequence[tuple[float, float, int, float, int]],
) -> float | None:
    """Compute the zero-gap IDM cost from counterfactual destination summaries."""

    if len(points) < 2:
        return None
    lower = min(points, key=lambda point: point[0])
    upper = max(points, key=lambda point: point[0])
    cost_span = upper[0] - lower[0]
    if not math.isclose(lower[0], 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Counterfactual break-even requires a zero-cost entry.")
    if cost_span <= 0.0:
        return None

    def destination_gap(point: tuple[float, float, int, float, int]) -> float:
        _, idm_total, idm_count, uncond_total, uncond_count = point
        if idm_count < 1 or uncond_count < 1:
            raise ValueError(
                "Counterfactual break-even requires finite IDM and UNCOND samples."
            )
        return idm_total / idm_count - uncond_total / uncond_count

    gap_at_zero = destination_gap(lower)
    slope = (destination_gap(upper) - gap_at_zero) / cost_span
    if gap_at_zero <= 0.0 or slope >= 0.0:
        return None
    break_even = gap_at_zero / -slope
    return break_even if math.isfinite(break_even) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMChunkCost:
    """Chunk rewards with exactly one fixed cost per executed route."""

    rewards: torch.Tensor
    costs: torch.Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMScalarAudit:
    """Finite scalar aggregates that never retain the source tensor."""

    count: int
    finite_count: int
    nonfinite_count: int
    minimum: float | None
    maximum: float | None
    total: float
    sum_of_squares: float = 0.0
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None

    def to_artifact(self) -> dict[str, object]:
        """Return a JSON-safe aggregate."""

        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "sum": self.total,
            "sum_of_squares": self.sum_of_squares,
            "quantiles": {
                "p10": self.p10,
                "p25": self.p25,
                "p50": self.p50,
                "p75": self.p75,
                "p90": self.p90,
            },
        }

    def to_metrics(self, *, prefix: str) -> dict[str, float]:
        """Return finite scalar summaries suitable for metric backends."""

        metrics = {
            f"{prefix}/count": float(self.count),
            f"{prefix}/finite_count": float(self.finite_count),
            f"{prefix}/nonfinite_count": float(self.nonfinite_count),
            f"{prefix}/sum": self.total,
            f"{prefix}/sum_of_squares": self.sum_of_squares,
        }
        if self.finite_count:
            metrics[f"{prefix}/mean"] = self.total / self.finite_count
        if self.minimum is not None:
            metrics[f"{prefix}/min"] = self.minimum
        if self.maximum is not None:
            metrics[f"{prefix}/max"] = self.maximum
        for name in ("p10", "p25", "p50", "p75", "p90"):
            value = getattr(self, name)
            if value is not None:
                metrics[f"{prefix}/{name}"] = value
        return metrics


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMChunkCostAudit:
    """Exact production branch-cost accounting for one rollout batch."""

    reward_shape: tuple[int, ...]
    reward_dtype: str
    idm_cost: float
    uncond_cost: float
    valid_chunk_count: int
    valid_idm_chunk_count: int
    forced_idm_chunk_count: int
    eligible_idm_chunk_count: int
    charge_scope: str
    charged_idm_chunk_count: int
    uncharged_forced_idm_chunk_count: int
    valid_uncond_chunk_count: int
    expected_cost_sum: float
    raw_primitive_rewards: FastWAMScalarAudit
    aggregated_raw_rewards: FastWAMScalarAudit
    actual_branch_costs: FastWAMScalarAudit
    shaped_rewards: FastWAMScalarAudit
    shaped_reward_identity_max_abs_error: float

    def to_artifact(self) -> dict[str, object]:
        """Return compact reward/cost evidence without per-step values."""

        return {
            "schema": FASTWAM_CHUNK_COST_AUDIT_SCHEMA,
            "reward_shape": list(self.reward_shape),
            "reward_dtype": self.reward_dtype,
            "idm_cost": self.idm_cost,
            "uncond_cost": self.uncond_cost,
            "valid_chunk_count": self.valid_chunk_count,
            "valid_idm_chunk_count": self.valid_idm_chunk_count,
            "forced_idm_chunk_count": self.forced_idm_chunk_count,
            "eligible_idm_chunk_count": self.eligible_idm_chunk_count,
            "charge_scope": self.charge_scope,
            "charged_idm_chunk_count": self.charged_idm_chunk_count,
            "uncharged_forced_idm_chunk_count": (self.uncharged_forced_idm_chunk_count),
            "valid_uncond_chunk_count": self.valid_uncond_chunk_count,
            "expected_cost_sum": self.expected_cost_sum,
            "raw_primitive_rewards": self.raw_primitive_rewards.to_artifact(),
            "aggregated_raw_rewards": self.aggregated_raw_rewards.to_artifact(),
            "actual_branch_costs": self.actual_branch_costs.to_artifact(),
            "shaped_rewards": self.shaped_rewards.to_artifact(),
            "shaped_reward_identity_max_abs_error": (
                self.shaped_reward_identity_max_abs_error
            ),
        }

    def to_metrics(self) -> dict[str, float]:
        """Return reward/cost decomposition for live training diagnostics."""

        metrics = {
            "fastwam/cost/idm_per_chunk": self.idm_cost,
            "fastwam/cost/uncond_per_chunk": self.uncond_cost,
            "fastwam/cost/expected_sum": self.expected_cost_sum,
            "fastwam/cost/identity_max_abs_error": (
                self.shaped_reward_identity_max_abs_error
            ),
            "fastwam/cost/forced_idm_chunk_count": float(self.forced_idm_chunk_count),
            "fastwam/cost/eligible_idm_chunk_count": float(
                self.eligible_idm_chunk_count
            ),
            "fastwam/cost/charged_idm_chunk_count": float(self.charged_idm_chunk_count),
            "fastwam/cost/uncharged_forced_idm_chunk_count": float(
                self.uncharged_forced_idm_chunk_count
            ),
            # Preserve the established flat tags used by training guards and
            # historical validators while adding grouped TensorBoard views.
            "fastwam/branch_cost_sum": self.actual_branch_costs.total,
            "fastwam/shaped_reward_sum": self.shaped_rewards.total,
            "fastwam/cost_identity_max_abs_error": (
                self.shaped_reward_identity_max_abs_error
            ),
        }
        summaries = {
            "fastwam/reward/raw_primitive": self.raw_primitive_rewards,
            "fastwam/reward/raw_chunk": self.aggregated_raw_rewards,
            "fastwam/reward/shaped_chunk": self.shaped_rewards,
            "fastwam/cost/actual_chunk": self.actual_branch_costs,
        }
        for prefix, summary in summaries.items():
            metrics.update(summary.to_metrics(prefix=prefix))
        return metrics


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMCounterfactualCostEntry:
    """Read-only same-batch advantage response to one hypothetical IDM cost."""

    idm_cost: float
    expected_cost_sum: float
    unnormalized_gate_advantage: FastWAMScalarAudit
    normalized_gate_advantage: FastWAMScalarAudit
    unnormalized_idm_gate_advantage: FastWAMScalarAudit
    normalized_idm_gate_advantage: FastWAMScalarAudit
    unnormalized_uncond_gate_advantage: FastWAMScalarAudit
    normalized_uncond_gate_advantage: FastWAMScalarAudit
    unnormalized_idm_delta_from_zero: FastWAMScalarAudit
    normalized_idm_delta_from_zero: FastWAMScalarAudit

    def to_artifact(self) -> dict[str, object]:
        """Return compact same-batch counterfactual evidence."""

        return {
            "idm_cost": self.idm_cost,
            "expected_cost_sum": self.expected_cost_sum,
            "gate_advantage": {
                "unnormalized": self.unnormalized_gate_advantage.to_artifact(),
                "normalized": self.normalized_gate_advantage.to_artifact(),
            },
            "idm_destination_gate_advantage": {
                "unnormalized": (self.unnormalized_idm_gate_advantage.to_artifact()),
                "normalized": self.normalized_idm_gate_advantage.to_artifact(),
            },
            "uncond_destination_gate_advantage": {
                "unnormalized": (self.unnormalized_uncond_gate_advantage.to_artifact()),
                "normalized": self.normalized_uncond_gate_advantage.to_artifact(),
            },
            "idm_destination_delta_from_zero": {
                "unnormalized": (self.unnormalized_idm_delta_from_zero.to_artifact()),
                "normalized": self.normalized_idm_delta_from_zero.to_artifact(),
            },
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMCounterfactualCostAudit:
    """Same-rollout causal branch-cost audit with no optimization side effects."""

    configured_idm_cost: float
    configured_alignment_max_abs_error: float
    eligible_gate_decision_count: int
    eligible_idm_decision_count: int
    eligible_uncond_decision_count: int
    entries: tuple[FastWAMCounterfactualCostEntry, ...]

    @property
    def break_even_idm_cost(self) -> float | None:
        """Return the interpolated cost where the unnormalized route gap is zero."""

        points = tuple(
            (
                entry.idm_cost,
                entry.unnormalized_idm_gate_advantage.total,
                entry.unnormalized_idm_gate_advantage.finite_count,
                entry.unnormalized_uncond_gate_advantage.total,
                entry.unnormalized_uncond_gate_advantage.finite_count,
            )
            for entry in self.entries
        )
        return compute_fastwam_break_even_idm_cost(points)

    def to_artifact(self) -> dict[str, object]:
        """Return compact counterfactual evidence."""

        return {
            "schema": FASTWAM_COUNTERFACTUAL_COST_AUDIT_SCHEMA,
            "configured_idm_cost": self.configured_idm_cost,
            "break_even_idm_cost": self.break_even_idm_cost,
            "configured_alignment_max_abs_error": (
                self.configured_alignment_max_abs_error
            ),
            "eligible_gate_decision_count": self.eligible_gate_decision_count,
            "eligible_idm_decision_count": self.eligible_idm_decision_count,
            "eligible_uncond_decision_count": self.eligible_uncond_decision_count,
            "entries": [entry.to_artifact() for entry in self.entries],
        }

    def to_metrics(self) -> dict[str, float]:
        """Return configured-cost counterfactual diagnostics for live curves."""

        metrics = {
            "fastwam/counterfactual/configured_idm_cost": self.configured_idm_cost,
            "fastwam/counterfactual/alignment_max_abs_error": (
                self.configured_alignment_max_abs_error
            ),
            "fastwam/counterfactual/eligible_gate_decision_count": float(
                self.eligible_gate_decision_count
            ),
            "fastwam/counterfactual/eligible_idm_decision_count": float(
                self.eligible_idm_decision_count
            ),
            "fastwam/counterfactual/eligible_uncond_decision_count": float(
                self.eligible_uncond_decision_count
            ),
            # Preserve the established validator-facing tag.
            "fastwam/counterfactual_alignment_max_abs_error": (
                self.configured_alignment_max_abs_error
            ),
        }
        break_even = self.break_even_idm_cost
        if break_even is not None:
            metrics["fastwam/counterfactual/break_even_idm_cost"] = break_even
        configured = next(
            (
                entry
                for entry in self.entries
                if math.isclose(
                    entry.idm_cost,
                    self.configured_idm_cost,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            None,
        )
        if configured is None:
            return metrics
        summaries = {
            "fastwam/counterfactual/gate_advantage_unnormalized": (
                configured.unnormalized_gate_advantage
            ),
            "fastwam/counterfactual/gate_advantage_normalized": (
                configured.normalized_gate_advantage
            ),
            "fastwam/counterfactual/idm_gate_advantage_unnormalized": (
                configured.unnormalized_idm_gate_advantage
            ),
            "fastwam/counterfactual/idm_gate_advantage_normalized": (
                configured.normalized_idm_gate_advantage
            ),
            "fastwam/counterfactual/uncond_gate_advantage_unnormalized": (
                configured.unnormalized_uncond_gate_advantage
            ),
            "fastwam/counterfactual/uncond_gate_advantage_normalized": (
                configured.normalized_uncond_gate_advantage
            ),
            "fastwam/counterfactual/idm_delta_from_zero_unnormalized": (
                configured.unnormalized_idm_delta_from_zero
            ),
            "fastwam/counterfactual/idm_delta_from_zero_normalized": (
                configured.normalized_idm_delta_from_zero
            ),
        }
        for prefix, summary in summaries.items():
            metrics.update(summary.to_metrics(prefix=prefix))
        return metrics


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMGateUpdateAudit:
    """One runner step's aggregate Gate parameter update."""

    optimizer_steps_before: int
    optimizer_steps_after: int
    tensor_count: int
    parameter_count: int
    before_sha256: str
    after_sha256: str
    before_l2_norm: float
    update_l2_norm: float
    update_max_abs: float
    relative_update_l2_norm: float
    finite_update_count: int
    nonfinite_update_count: int

    def to_artifact(self) -> dict[str, object]:
        """Return compact parameter-update evidence without model tensors."""

        return {
            "schema": FASTWAM_GATE_UPDATE_AUDIT_SCHEMA,
            "optimizer_steps_before": self.optimizer_steps_before,
            "optimizer_steps_after": self.optimizer_steps_after,
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "before_l2_norm": self.before_l2_norm,
            "update_l2_norm": self.update_l2_norm,
            "update_max_abs": self.update_max_abs,
            "relative_update_l2_norm": self.relative_update_l2_norm,
            "finite_update_count": self.finite_update_count,
            "nonfinite_update_count": self.nonfinite_update_count,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMEnvironmentRewardAudit:
    """Compact raw-environment reward evidence with no trajectory payload."""

    reward_shape: tuple[int, ...]
    reward_dtype: str
    total_value_count: int
    valid_value_count: int
    finite_value_count: int
    nonfinite_value_count: int
    positive_success_signal_count: int
    successful_trajectory_count: int
    total_chunk_count: int
    valid_chunk_count: int
    valid_idm_chunk_count: int
    valid_uncond_chunk_count: int
    valid_reward_min: float | None
    valid_reward_max: float | None
    valid_reward_sum: float

    def to_artifact(self) -> dict[str, object]:
        """Return a JSON-safe aggregate without individual rewards."""

        return {
            "schema": FASTWAM_REWARD_AUDIT_SCHEMA,
            "reward_shape": list(self.reward_shape),
            "reward_dtype": self.reward_dtype,
            "total_value_count": self.total_value_count,
            "valid_value_count": self.valid_value_count,
            "finite_value_count": self.finite_value_count,
            "nonfinite_value_count": self.nonfinite_value_count,
            "positive_success_signal_count": self.positive_success_signal_count,
            "successful_trajectory_count": self.successful_trajectory_count,
            "total_chunk_count": self.total_chunk_count,
            "valid_chunk_count": self.valid_chunk_count,
            "valid_idm_chunk_count": self.valid_idm_chunk_count,
            "valid_uncond_chunk_count": self.valid_uncond_chunk_count,
            "valid_reward_min": self.valid_reward_min,
            "valid_reward_max": self.valid_reward_max,
            "valid_reward_sum": self.valid_reward_sum,
        }

    def to_metrics(self) -> dict[str, float]:
        """Return raw sparse-reward health metrics for live monitoring."""

        metrics = {
            "fastwam/reward/raw_positive_success_signal_count": float(
                self.positive_success_signal_count
            ),
            "fastwam/reward/successful_trajectory_count": float(
                self.successful_trajectory_count
            ),
            "fastwam/reward/valid_value_count": float(self.valid_value_count),
            "fastwam/reward/nonfinite_value_count": float(self.nonfinite_value_count),
            "fastwam/reward/raw_sum": self.valid_reward_sum,
            # Preserve the established flat tags used by the guard.
            "fastwam/raw_positive_success_signal_count": float(
                self.positive_success_signal_count
            ),
            "fastwam/successful_trajectory_count": float(
                self.successful_trajectory_count
            ),
        }
        if self.valid_reward_min is not None:
            metrics["fastwam/reward/raw_min"] = self.valid_reward_min
        if self.valid_reward_max is not None:
            metrics["fastwam/reward/raw_max"] = self.valid_reward_max
        if self.valid_value_count:
            metrics["fastwam/reward/raw_mean"] = (
                self.valid_reward_sum / self.valid_value_count
            )
        return metrics

    def require_success_signal(self) -> None:
        """Fail before optimization when the audited rollout is unsafe to train."""

        if self.nonfinite_value_count:
            raise RuntimeError(
                "FastWAM short-RL raw environment rewards contain non-finite "
                f"values ({self.nonfinite_value_count})."
            )
        if self.positive_success_signal_count < 1:
            raise RuntimeError(
                "FastWAM short-RL rollout contains zero positive sparse-success "
                "signals; refusing to optimize Gate/LoRA/value parameters."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMRolloutStateAudit:
    """Compact route-probability and stored-K/V evidence without payloads."""

    decision_shape: tuple[int, ...]
    total_decision_count: int
    valid_chunk_count: int
    valid_idm_chunk_count: int
    valid_uncond_chunk_count: int
    forced_route_count: int
    emitted_decision_count: int
    eligible_gate_decision_count: int
    eligible_idm_decision_count: int
    unused_emitted_decision_count: int
    route_decision_sha256: str
    base_probability_min: float
    base_probability_max: float
    base_probability_mean: float
    base_probability_p10: float
    base_probability_p50: float
    base_probability_p90: float
    base_probability_bimodality_score: float
    base_probability_outside_0p2_0p8_fraction: float
    behavior_probability_min: float
    behavior_probability_max: float
    behavior_probability_mean: float
    kv_replay_backend: str
    kv_storage_dtype: str
    kv_layer_indices: tuple[int, ...]
    kv_denoise_tap_count: int
    kv_configured_max_bytes_per_sample: int | None
    kv_all_emitted_sample_count: int
    kv_all_emitted_nonzero_sample_count: int
    kv_all_emitted_total_bytes: int
    kv_all_emitted_maximum_bytes_per_sample: int
    kv_eligible_sample_count: int
    kv_eligible_nonzero_sample_count: int
    kv_eligible_total_bytes: int
    kv_eligible_maximum_bytes_per_sample: int

    def to_artifact(self) -> dict[str, object]:
        """Return JSON-safe aggregates without K/V tensors or trajectories."""

        probability_count = self.eligible_gate_decision_count
        return {
            "schema": FASTWAM_ROLLOUT_STATE_AUDIT_SCHEMA,
            "decision_shape": list(self.decision_shape),
            "total_decision_count": self.total_decision_count,
            "valid_chunk_count": self.valid_chunk_count,
            "valid_idm_chunk_count": self.valid_idm_chunk_count,
            "valid_uncond_chunk_count": self.valid_uncond_chunk_count,
            "forced_route_count": self.forced_route_count,
            "executed_idm_fraction": (
                self.valid_idm_chunk_count / self.valid_chunk_count
            ),
            "emitted_decision_count": self.emitted_decision_count,
            "eligible_gate_decision_count": self.eligible_gate_decision_count,
            "eligible_idm_decision_count": self.eligible_idm_decision_count,
            "eligible_idm_fraction": (
                self.eligible_idm_decision_count / self.eligible_gate_decision_count
            ),
            "unused_emitted_decision_count": self.unused_emitted_decision_count,
            "route_decision_sha256": self.route_decision_sha256,
            "base_probability": {
                "count": probability_count,
                "minimum": self.base_probability_min,
                "maximum": self.base_probability_max,
                "mean": self.base_probability_mean,
                "p10": self.base_probability_p10,
                "p50": self.base_probability_p50,
                "p90": self.base_probability_p90,
                "bimodality_score": self.base_probability_bimodality_score,
                "outside_0p2_0p8_fraction": (
                    self.base_probability_outside_0p2_0p8_fraction
                ),
            },
            "behavior_probability": {
                "count": probability_count,
                "minimum": self.behavior_probability_min,
                "maximum": self.behavior_probability_max,
                "mean": self.behavior_probability_mean,
            },
            "kv_replay_backend": self.kv_replay_backend,
            "kv_storage_dtype": self.kv_storage_dtype,
            "kv_layer_indices": list(self.kv_layer_indices),
            "kv_denoise_tap_count": self.kv_denoise_tap_count,
            "kv_configured_max_bytes_per_sample": (
                self.kv_configured_max_bytes_per_sample
            ),
            "kv_all_emitted": {
                "sample_count": self.kv_all_emitted_sample_count,
                "nonzero_sample_count": self.kv_all_emitted_nonzero_sample_count,
                "total_bytes": self.kv_all_emitted_total_bytes,
                "maximum_bytes_per_sample": (
                    self.kv_all_emitted_maximum_bytes_per_sample
                ),
            },
            "kv_eligible": {
                "sample_count": self.kv_eligible_sample_count,
                "nonzero_sample_count": self.kv_eligible_nonzero_sample_count,
                "total_bytes": self.kv_eligible_total_bytes,
                "maximum_bytes_per_sample": (self.kv_eligible_maximum_bytes_per_sample),
            },
        }

    def to_metrics(self) -> dict[str, float]:
        """Return Gate output, route, and K/V summaries for live monitoring."""

        executed_idm_fraction = self.valid_idm_chunk_count / self.valid_chunk_count
        eligible_idm_fraction = (
            self.eligible_idm_decision_count / self.eligible_gate_decision_count
        )
        forced_fraction = self.forced_route_count / self.valid_chunk_count
        metrics = {
            "fastwam/route/valid_chunk_count": float(self.valid_chunk_count),
            "fastwam/route/valid_idm_chunk_count": float(self.valid_idm_chunk_count),
            "fastwam/route/valid_uncond_chunk_count": float(
                self.valid_uncond_chunk_count
            ),
            "fastwam/route/forced_count": float(self.forced_route_count),
            "fastwam/route/forced_fraction": forced_fraction,
            "fastwam/route/executed_idm_fraction": executed_idm_fraction,
            "fastwam/route/emitted_decision_count": float(self.emitted_decision_count),
            "fastwam/route/eligible_gate_decision_count": float(
                self.eligible_gate_decision_count
            ),
            "fastwam/route/eligible_idm_decision_count": float(
                self.eligible_idm_decision_count
            ),
            "fastwam/route/eligible_idm_fraction": eligible_idm_fraction,
            "fastwam/route/unused_emitted_decision_count": float(
                self.unused_emitted_decision_count
            ),
            "fastwam/gate/base_idm_probability_min": self.base_probability_min,
            "fastwam/gate/base_idm_probability_max": self.base_probability_max,
            "fastwam/gate/base_idm_probability_mean": self.base_probability_mean,
            "fastwam/gate/base_idm_probability_p10": self.base_probability_p10,
            "fastwam/gate/base_idm_probability_p50": self.base_probability_p50,
            "fastwam/gate/base_idm_probability_p90": self.base_probability_p90,
            "fastwam/gate/base_idm_probability_bimodality_score": (
                self.base_probability_bimodality_score
            ),
            "fastwam/gate/base_idm_probability_outside_0p2_0p8_fraction": (
                self.base_probability_outside_0p2_0p8_fraction
            ),
            "fastwam/gate/behavior_idm_probability_min": (
                self.behavior_probability_min
            ),
            "fastwam/gate/behavior_idm_probability_max": (
                self.behavior_probability_max
            ),
            "fastwam/gate/behavior_idm_probability_mean": (
                self.behavior_probability_mean
            ),
            "fastwam/kv/all_emitted_sample_count": float(
                self.kv_all_emitted_sample_count
            ),
            "fastwam/kv/all_emitted_nonzero_sample_count": float(
                self.kv_all_emitted_nonzero_sample_count
            ),
            "fastwam/kv/all_emitted_total_bytes": float(
                self.kv_all_emitted_total_bytes
            ),
            "fastwam/kv/all_emitted_max_bytes_per_sample": float(
                self.kv_all_emitted_maximum_bytes_per_sample
            ),
            "fastwam/kv/eligible_sample_count": float(self.kv_eligible_sample_count),
            "fastwam/kv/eligible_nonzero_sample_count": float(
                self.kv_eligible_nonzero_sample_count
            ),
            "fastwam/kv/eligible_total_bytes": float(self.kv_eligible_total_bytes),
            "fastwam/kv/eligible_max_bytes_per_sample": float(
                self.kv_eligible_maximum_bytes_per_sample
            ),
            # Preserve the established guard-facing tags.
            "fastwam/eligible_idm_fraction": eligible_idm_fraction,
            "fastwam/eligible_gate_decision_count": float(
                self.eligible_gate_decision_count
            ),
            "fastwam/eligible_idm_decision_count": float(
                self.eligible_idm_decision_count
            ),
            "fastwam/valid_uncond_chunk_count": float(self.valid_uncond_chunk_count),
        }
        if self.kv_configured_max_bytes_per_sample is not None:
            metrics["fastwam/kv/configured_max_bytes_per_sample"] = float(
                self.kv_configured_max_bytes_per_sample
            )
        return metrics


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMPolicyAlignment:
    """Source-aligned policy fields for delayed Gate and Flow-SDE PPO."""

    flow_advantages: torch.Tensor
    flow_valid_mask: torch.Tensor
    gate_advantages: torch.Tensor
    gate_valid_mask: torch.Tensor


def _raise_first_pair(
    mask: torch.Tensor,
    *,
    message: str,
    source_times: torch.Tensor,
    source_columns: torch.Tensor,
    destination_times: torch.Tensor,
    destination_columns: torch.Tensor,
) -> None:
    if not bool(mask.any().item()):
        return
    index = int(mask.nonzero(as_tuple=False)[0].item())
    source = (int(source_times[index]), int(source_columns[index]))
    destination = (
        int(destination_times[index]),
        int(destination_columns[index]),
    )
    raise ValueError(
        f"{message} First source/destination pair is {source} -> {destination}."
    )


def _chunk_mask(
    value: torch.Tensor | None,
    *,
    shape: torch.Size,
    name: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool, got {value.dtype}.")
    if value.shape[: len(shape)] != shape:
        raise ValueError(
            f"{name} must start with shape {tuple(shape)}, got {tuple(value.shape)}."
        )
    if value.ndim == len(shape):
        return value
    return value.reshape(*shape, -1).any(dim=-1)


def _primitive_reward_mask(
    value: torch.Tensor | None,
    *,
    reward_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.ones(reward_shape, dtype=torch.bool, device=device)
    if value.dtype != torch.bool:
        raise TypeError(f"valid_mask must use torch.bool, got {value.dtype}.")
    if value.shape[:2] != reward_shape[:2]:
        raise ValueError(
            "valid_mask must match reward [time, batch] dimensions; got "
            f"{tuple(value.shape)} and {tuple(reward_shape)}."
        )
    if value.ndim == 2:
        value = value.unsqueeze(-1)
    elif value.ndim != 3:
        raise ValueError(
            "valid_mask must have shape [time, batch] or "
            "[time, batch, 1|action_chunks]."
        )
    if value.shape[-1] not in {1, reward_shape[-1]}:
        raise ValueError(
            "valid_mask trailing dimension must be one or match action chunks; "
            f"got {value.shape[-1]} and {reward_shape[-1]}."
        )
    return value.expand(reward_shape)


def _summarize_selected_scalars(
    values: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> FastWAMScalarAudit:
    """Reduce selected values to finite JSON-safe aggregates."""

    selected = values.reshape(-1) if mask is None else values[mask]
    selected = selected.detach()
    finite = torch.isfinite(selected)
    finite_values = selected[finite].to(torch.float64)
    count = int(selected.numel())
    finite_count = int(finite.sum().item())
    quantiles = (
        torch.quantile(
            finite_values,
            torch.tensor(
                [0.10, 0.25, 0.50, 0.75, 0.90],
                dtype=torch.float64,
                device=finite_values.device,
            ),
        )
        if finite_count
        else None
    )
    return FastWAMScalarAudit(
        count=count,
        finite_count=finite_count,
        nonfinite_count=count - finite_count,
        minimum=(float(finite_values.min().item()) if finite_count else None),
        maximum=(float(finite_values.max().item()) if finite_count else None),
        total=(float(finite_values.sum().item()) if finite_count else 0.0),
        sum_of_squares=(
            float(finite_values.square().sum().item()) if finite_count else 0.0
        ),
        p10=(float(quantiles[0].item()) if quantiles is not None else None),
        p25=(float(quantiles[1].item()) if quantiles is not None else None),
        p50=(float(quantiles[2].item()) if quantiles is not None else None),
        p75=(float(quantiles[3].item()) if quantiles is not None else None),
        p90=(float(quantiles[4].item()) if quantiles is not None else None),
    )


def summarize_fastwam_chunk_cost(
    *,
    environment_rewards: torch.Tensor,
    route: ChunkRouteRecord,
    cost_result: FastWAMChunkCost,
    idm_cost: float,
    uncond_cost: float = 0.0,
    valid_mask: torch.Tensor | None = None,
    charge_mask: torch.Tensor | None = None,
    charge_scope: str = "all_valid_idm",
) -> FastWAMChunkCostAudit:
    """Audit the exact production reward aggregation and route cost."""

    if environment_rewards.ndim != 3 or route.shape != environment_rewards.shape[:2]:
        raise ValueError(
            "FastWAM cost audit requires rewards [time,batch,actions] and "
            "matching route metadata."
        )
    expected_shape = (*route.shape, 1)
    if cost_result.rewards.shape != expected_shape:
        raise ValueError("Shaped FastWAM rewards do not have chunk-level shape.")
    if cost_result.costs.shape != expected_shape:
        raise ValueError("FastWAM branch costs do not have chunk-level shape.")
    if not environment_rewards.is_floating_point():
        raise TypeError("FastWAM cost audit rewards must use a floating dtype.")
    if (
        not cost_result.rewards.is_floating_point()
        or not cost_result.costs.is_floating_point()
    ):
        raise TypeError("FastWAM shaped rewards and costs must use floating dtypes.")
    idm_cost = float(idm_cost)
    uncond_cost = float(uncond_cost)
    if not torch.isfinite(torch.tensor([idm_cost, uncond_cost])).all() or (
        idm_cost < 0 or uncond_cost < 0
    ):
        raise ValueError("FastWAM audited route costs must be finite and non-negative.")

    chunk_mask = _chunk_mask(
        valid_mask,
        shape=route.shape,
        name="valid_mask",
        device=environment_rewards.device,
    )
    charged_chunk_mask = (
        chunk_mask
        if charge_mask is None
        else _chunk_mask(
            charge_mask,
            shape=route.shape,
            name="charge_mask",
            device=environment_rewards.device,
        )
    )
    if bool((charged_chunk_mask & ~chunk_mask).any().item()):
        raise ValueError("charge_mask must be a subset of valid_mask.")
    primitive_mask = _primitive_reward_mask(
        valid_mask,
        reward_shape=environment_rewards.shape,
        device=environment_rewards.device,
    )
    idm_mask = chunk_mask & (route.route_used == int(WAMRoute.IDM))
    uncond_mask = chunk_mask & (route.route_used == int(WAMRoute.UNCOND))
    forced_idm_mask = idm_mask & route.route_was_forced
    eligible_idm_mask = idm_mask & ~route.route_was_forced
    charged_idm_mask = idm_mask & charged_chunk_mask
    charged_uncond_mask = uncond_mask & charged_chunk_mask
    uncharged_forced_idm_mask = forced_idm_mask & ~charged_chunk_mask
    aggregated = environment_rewards.sum(dim=-1, keepdim=True)
    identity_error = (cost_result.rewards - (aggregated - cost_result.costs)).abs()
    identity_max = float(identity_error.max().item()) if identity_error.numel() else 0.0
    expected_cost_sum = (
        float(charged_idm_mask.sum().item()) * idm_cost
        + float(charged_uncond_mask.sum().item()) * uncond_cost
    )

    return FastWAMChunkCostAudit(
        reward_shape=tuple(int(item) for item in environment_rewards.shape),
        reward_dtype=str(environment_rewards.dtype),
        idm_cost=idm_cost,
        uncond_cost=uncond_cost,
        valid_chunk_count=int(chunk_mask.sum().item()),
        valid_idm_chunk_count=int(idm_mask.sum().item()),
        forced_idm_chunk_count=int(forced_idm_mask.sum().item()),
        eligible_idm_chunk_count=int(eligible_idm_mask.sum().item()),
        charge_scope=str(charge_scope),
        charged_idm_chunk_count=int(charged_idm_mask.sum().item()),
        uncharged_forced_idm_chunk_count=int(uncharged_forced_idm_mask.sum().item()),
        valid_uncond_chunk_count=int(uncond_mask.sum().item()),
        expected_cost_sum=expected_cost_sum,
        raw_primitive_rewards=_summarize_selected_scalars(
            environment_rewards,
            mask=primitive_mask,
        ),
        aggregated_raw_rewards=_summarize_selected_scalars(
            aggregated[..., 0],
            mask=chunk_mask,
        ),
        actual_branch_costs=_summarize_selected_scalars(
            cost_result.costs[..., 0],
            mask=chunk_mask,
        ),
        shaped_rewards=_summarize_selected_scalars(
            cost_result.rewards[..., 0],
            mask=chunk_mask,
        ),
        shaped_reward_identity_max_abs_error=identity_max,
    )


def summarize_fastwam_environment_rewards(
    *,
    environment_rewards: torch.Tensor,
    route_used: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> FastWAMEnvironmentRewardAudit:
    """Summarize raw primitive rewards before applying any branch cost.

    Positive finite raw LIBERO rewards are the sparse task-success signal. The
    audit deliberately applies the actor loss mask first, so rewards in padded
    or post-terminal chunks cannot unlock training. It retains only aggregates.
    """

    if not environment_rewards.is_floating_point():
        raise TypeError("environment_rewards must use a floating dtype.")
    if environment_rewards.ndim != 3:
        raise ValueError(
            "FastWAM environment rewards must have shape "
            "[time, batch, action_chunks], "
            f"got {tuple(environment_rewards.shape)}."
        )
    if route_used.shape != environment_rewards.shape[:2]:
        raise ValueError(
            "route_used must match the reward [time, batch] dimensions; got "
            f"{tuple(route_used.shape)} and {tuple(environment_rewards.shape)}."
        )
    if route_used.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("route_used must use an integer dtype.")
    invalid_route = (route_used != int(WAMRoute.UNCOND)) & (
        route_used != int(WAMRoute.IDM)
    )
    if bool(invalid_route.any().item()):
        raise ValueError("route_used contains a value outside WAMRoute.")

    primitive_mask = _primitive_reward_mask(
        valid_mask,
        reward_shape=environment_rewards.shape,
        device=environment_rewards.device,
    )
    chunk_mask = primitive_mask.any(dim=-1)
    finite_mask = torch.isfinite(environment_rewards)
    valid_finite_mask = primitive_mask & finite_mask
    positive_mask = valid_finite_mask & (environment_rewards > 0)
    valid_finite_rewards = environment_rewards[valid_finite_mask]
    if valid_finite_rewards.numel():
        reward_min = float(valid_finite_rewards.min().item())
        reward_max = float(valid_finite_rewards.max().item())
        reward_sum = float(valid_finite_rewards.to(torch.float64).sum().item())
    else:
        reward_min = None
        reward_max = None
        reward_sum = 0.0

    return FastWAMEnvironmentRewardAudit(
        reward_shape=tuple(int(size) for size in environment_rewards.shape),
        reward_dtype=str(environment_rewards.dtype),
        total_value_count=int(environment_rewards.numel()),
        valid_value_count=int(primitive_mask.sum().item()),
        finite_value_count=int(finite_mask.sum().item()),
        nonfinite_value_count=int((~finite_mask).sum().item()),
        positive_success_signal_count=int(positive_mask.sum().item()),
        successful_trajectory_count=int(positive_mask.any(dim=(0, 2)).sum().item()),
        total_chunk_count=int(route_used.numel()),
        valid_chunk_count=int(chunk_mask.sum().item()),
        valid_idm_chunk_count=int(
            (chunk_mask & (route_used == int(WAMRoute.IDM))).sum().item()
        ),
        valid_uncond_chunk_count=int(
            (chunk_mask & (route_used == int(WAMRoute.UNCOND))).sum().item()
        ),
        valid_reward_min=reward_min,
        valid_reward_max=reward_max,
        valid_reward_sum=reward_sum,
    )


def summarize_fastwam_rollout_state(
    *,
    route: ChunkRouteRecord,
    emitted: GateDecisionRecord,
    eligible_gate_mask: torch.Tensor,
    valid_mask: torch.Tensor | None,
    kv_replay_backend: str,
    max_bytes_per_sample: int | None,
) -> FastWAMRolloutStateAudit:
    """Summarize routed chunks, Gate probabilities, and actual K/V byte volume."""

    if len(route.shape) != 2 or emitted.shape != route.shape:
        raise ValueError(
            "FastWAM rollout-state audit requires matching [time, batch] "
            "route and emitted decision records."
        )
    if eligible_gate_mask.dtype != torch.bool:
        raise TypeError("eligible_gate_mask must use torch.bool.")
    if eligible_gate_mask.shape != route.shape:
        raise ValueError("eligible_gate_mask must match the route shape.")
    chunk_mask = _chunk_mask(
        valid_mask,
        shape=route.shape,
        name="valid_mask",
        device=route.route_used.device,
    )
    if bool((eligible_gate_mask & ~emitted.valid).any().item()):
        raise ValueError("Eligible Gate decisions must be valid emitted decisions.")
    if bool((eligible_gate_mask & ~chunk_mask).any().item()):
        raise ValueError("Eligible Gate decisions must belong to valid chunks.")

    valid_chunk_count = int(chunk_mask.sum().item())
    if valid_chunk_count < 1:
        raise ValueError("FastWAM rollout-state audit has no valid chunks.")
    emitted_count = int(emitted.valid.sum().item())
    eligible_count = int(eligible_gate_mask.sum().item())
    if eligible_count < 1:
        raise ValueError("FastWAM rollout-state audit has no eligible Gate decisions.")

    metadata = emitted.kv_metadata
    backend = str(kv_replay_backend).strip().lower()
    if backend not in {"stored", "recompute"}:
        raise ValueError(f"Unsupported FastWAM K/V replay backend {backend!r}.")
    if metadata is None:
        raise ValueError(f"FastWAM {backend} K/V metadata is missing.")
    byte_limit = max_bytes_per_sample
    if byte_limit is not None:
        if isinstance(byte_limit, bool) or int(byte_limit) < 1:
            raise ValueError("K/V max_bytes_per_sample must be a positive integer.")
        byte_limit = int(byte_limit)

    byte_values = metadata.total_bytes.to(emitted.valid.device)
    emitted_bytes = byte_values[emitted.valid]
    eligible_bytes = byte_values[eligible_gate_mask]
    if backend == "stored":
        if emitted_bytes.numel() == 0 or bool((emitted_bytes <= 0).any().item()):
            raise ValueError(
                "FastWAM stored K/V bytes must be positive for every emitted decision."
            )
        if byte_limit is None:
            raise ValueError("FastWAM stored K/V requires max_bytes_per_sample.")
        if int(emitted_bytes.max().item()) > byte_limit:
            raise ValueError("FastWAM stored K/V exceeds max_bytes_per_sample.")
    elif bool((emitted_bytes != 0).any().item()):
        raise ValueError(
            "FastWAM recompute metadata must report zero stored K/V bytes."
        )

    def probability_summary(value: torch.Tensor) -> dict[str, float]:
        selected = value[eligible_gate_mask].to(torch.float64)
        if not bool(torch.isfinite(selected).all().item()):
            raise ValueError("Eligible Gate probability contains non-finite values.")
        quantiles = torch.quantile(
            selected,
            torch.tensor(
                [0.10, 0.50, 0.90],
                dtype=torch.float64,
                device=selected.device,
            ),
        )
        centered = selected - selected.mean()
        second_moment = centered.square().mean()
        if float(second_moment.item()) == 0.0:
            bimodality_score = 0.0
        else:
            third_moment = centered.pow(3).mean()
            fourth_moment = centered.pow(4).mean()
            skewness = third_moment / second_moment.pow(1.5)
            kurtosis = fourth_moment / second_moment.square()
            pearson_coefficient = (skewness.square() + 1.0) / kurtosis
            bimodality_score = float(
                (pearson_coefficient * (quantiles[2] - quantiles[0])).item()
            )
        return {
            "minimum": float(selected.min().item()),
            "maximum": float(selected.max().item()),
            "mean": float(selected.mean().item()),
            "p10": float(quantiles[0].item()),
            "p50": float(quantiles[1].item()),
            "p90": float(quantiles[2].item()),
            # Pearson's coefficient alone is scale invariant and can call two
            # nearby peaks strongly bimodal. Weighting it by the interdecile
            # range makes the score reflect materially separated routing modes.
            "bimodality_score": bimodality_score,
            "outside_0p2_0p8_fraction": float(
                ((selected < 0.2) | (selected > 0.8)).to(torch.float64).mean().item()
            ),
        }

    def byte_summary(values: torch.Tensor) -> tuple[int, int, int, int]:
        sample_count = int(values.numel())
        nonzero_count = int((values > 0).sum().item())
        total_bytes = int(values.to(torch.int64).sum().item())
        maximum = int(values.max().item()) if sample_count else 0
        return sample_count, nonzero_count, total_bytes, maximum

    base_summary = probability_summary(emitted.base_probability)
    behavior_summary = probability_summary(emitted.behavior_probability)
    all_count, all_nonzero, all_total, all_max = byte_summary(emitted_bytes)
    eligible_byte_count, eligible_nonzero, eligible_total, eligible_max = byte_summary(
        eligible_bytes
    )
    valid_idm_count = int(
        (chunk_mask & (route.route_used == int(WAMRoute.IDM))).sum().item()
    )
    valid_uncond_count = int(
        (chunk_mask & (route.route_used == int(WAMRoute.UNCOND))).sum().item()
    )
    eligible_idm_count = int(
        (eligible_gate_mask & (emitted.next_route == int(WAMRoute.IDM))).sum().item()
    )
    forced_route_count = int((chunk_mask & route.route_was_forced).sum().item())
    return FastWAMRolloutStateAudit(
        decision_shape=tuple(int(size) for size in route.shape),
        total_decision_count=int(route.route_used.numel()),
        valid_chunk_count=valid_chunk_count,
        valid_idm_chunk_count=valid_idm_count,
        valid_uncond_chunk_count=valid_uncond_count,
        forced_route_count=forced_route_count,
        emitted_decision_count=emitted_count,
        eligible_gate_decision_count=eligible_count,
        eligible_idm_decision_count=eligible_idm_count,
        unused_emitted_decision_count=emitted_count - eligible_count,
        route_decision_sha256=checkpoint_state_sha256(
            {
                "route_used": route.route_used,
                "route_was_forced": route.route_was_forced,
                "chunk_ids": route.chunk_ids,
                "episode_ids": route.episode_ids,
                "route_source_chunk_ids": route.route_source_chunk_ids,
                "actor_versions": route.actor_versions,
                "emitted_next_route": emitted.next_route,
                "emitted_valid": emitted.valid,
                "emitted_source_chunk_ids": emitted.source_chunk_ids,
            }
        ),
        base_probability_min=base_summary["minimum"],
        base_probability_max=base_summary["maximum"],
        base_probability_mean=base_summary["mean"],
        base_probability_p10=base_summary["p10"],
        base_probability_p50=base_summary["p50"],
        base_probability_p90=base_summary["p90"],
        base_probability_bimodality_score=base_summary["bimodality_score"],
        base_probability_outside_0p2_0p8_fraction=base_summary[
            "outside_0p2_0p8_fraction"
        ],
        behavior_probability_min=behavior_summary["minimum"],
        behavior_probability_max=behavior_summary["maximum"],
        behavior_probability_mean=behavior_summary["mean"],
        kv_replay_backend=backend,
        kv_storage_dtype=metadata.storage_dtype,
        kv_layer_indices=metadata.layer_indices,
        kv_denoise_tap_count=int(metadata.denoise_timesteps.shape[-1]),
        kv_configured_max_bytes_per_sample=byte_limit,
        kv_all_emitted_sample_count=all_count,
        kv_all_emitted_nonzero_sample_count=all_nonzero,
        kv_all_emitted_total_bytes=all_total,
        kv_all_emitted_maximum_bytes_per_sample=all_max,
        kv_eligible_sample_count=eligible_byte_count,
        kv_eligible_nonzero_sample_count=eligible_nonzero,
        kv_eligible_total_bytes=eligible_total,
        kv_eligible_maximum_bytes_per_sample=eligible_max,
    )


def apply_fastwam_chunk_cost(
    *,
    environment_rewards: torch.Tensor,
    route_used: torch.Tensor,
    idm_cost: float,
    uncond_cost: float = 0.0,
    valid_mask: torch.Tensor | None = None,
    charge_mask: torch.Tensor | None = None,
) -> FastWAMChunkCost:
    """Aggregate primitive rewards, then subtract one actual-route cost.

    FastWAM routes are chosen once per action chunk while the environment keeps
    one reward entry per primitive action. This helper deliberately aggregates
    the primitive rewards before applying the route cost, preventing accidental
    multiplication by ``num_action_chunks``.
    """

    if not environment_rewards.is_floating_point():
        raise TypeError("environment_rewards must use a floating dtype.")
    if environment_rewards.ndim != 3:
        raise ValueError(
            "FastWAM chunk rewards must have shape [time, batch, action_chunks], "
            f"got {tuple(environment_rewards.shape)}."
        )
    if route_used.shape != environment_rewards.shape[:2]:
        raise ValueError(
            "route_used must match the reward [time, batch] dimensions; got "
            f"{tuple(route_used.shape)} and {tuple(environment_rewards.shape)}."
        )
    if route_used.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("route_used must use an integer dtype.")
    if idm_cost < 0 or uncond_cost < 0:
        raise ValueError("FastWAM branch costs must be non-negative.")
    invalid_route = (route_used != int(WAMRoute.UNCOND)) & (
        route_used != int(WAMRoute.IDM)
    )
    if bool(invalid_route.any().item()):
        raise ValueError("route_used contains a value outside WAMRoute.")

    rewards = environment_rewards.sum(dim=-1, keepdim=True)
    costs = torch.where(
        route_used == int(WAMRoute.IDM),
        torch.as_tensor(idm_cost, dtype=rewards.dtype, device=rewards.device),
        torch.as_tensor(uncond_cost, dtype=rewards.dtype, device=rewards.device),
    ).unsqueeze(-1)
    valid_chunk_mask = _chunk_mask(
        valid_mask,
        shape=route_used.shape,
        name="valid_mask",
        device=rewards.device,
    )
    charged_chunk_mask = (
        valid_chunk_mask
        if charge_mask is None
        else _chunk_mask(
            charge_mask,
            shape=route_used.shape,
            name="charge_mask",
            device=rewards.device,
        )
    )
    if bool((charged_chunk_mask & ~valid_chunk_mask).any().item()):
        raise ValueError("charge_mask must be a subset of valid_mask.")
    if valid_mask is not None or charge_mask is not None:
        costs = torch.where(
            charged_chunk_mask.unsqueeze(-1),
            costs,
            torch.zeros_like(costs),
        )
    return FastWAMChunkCost(rewards=rewards - costs, costs=costs)


def align_fastwam_policy_advantages(
    *,
    advantages: torch.Tensor,
    route: ChunkRouteRecord,
    emitted: GateDecisionRecord,
    dones: torch.Tensor,
    rollout_epoch: int,
    carry_pending_across_epochs: bool,
    loss_mask: torch.Tensor | None = None,
) -> FastWAMPolicyAlignment:
    """Pair chunk-``t`` Gate replay with chunk-``t+1`` advantage.

    The Gate record and its K/V remain at their source chunk. Only the scalar
    destination advantage is copied back to that source position. When rollout
    epochs were folded into the batch dimension, an auto-reset rollout may pair
    the last source in epoch ``e`` with the first destination in epoch ``e+1``.
    Forced/reset destinations and the final unused decision receive no Gate loss.

    Route, episode, chunk-id, and actor-version mismatches fail closed rather
    than silently changing the training mask.
    """

    if len(route.shape) != 2:
        raise ValueError(
            "FastWAM policy alignment requires route records shaped [time, batch], "
            f"got {tuple(route.shape)}."
        )
    if emitted.shape != route.shape:
        raise ValueError(
            "Route and emitted Gate records must have identical shapes, got "
            f"{tuple(route.shape)} and {tuple(emitted.shape)}."
        )
    if rollout_epoch < 1:
        raise ValueError("rollout_epoch must be positive.")
    time_steps, folded_batch = route.shape
    if folded_batch % rollout_epoch != 0:
        raise ValueError(
            f"Folded batch {folded_batch} is not divisible by rollout_epoch "
            f"{rollout_epoch}."
        )
    if advantages.shape != (*route.shape, 1):
        raise ValueError(
            "FastWAM chunk-level advantages must have shape [time, batch, 1], "
            f"got {tuple(advantages.shape)} for routes {tuple(route.shape)}."
        )
    if not advantages.is_floating_point():
        raise TypeError("advantages must use a floating dtype.")
    expected_done_prefix = (time_steps + 1, folded_batch)
    if dones.shape[:2] != expected_done_prefix:
        raise ValueError(
            "dones must have one bootstrap timestep and begin with shape "
            f"{expected_done_prefix}, got {tuple(dones.shape)}."
        )
    if dones.dtype != torch.bool:
        raise TypeError("dones must use torch.bool.")

    source_metadata_mismatch = emitted.valid & (
        (emitted.source_chunk_ids != route.chunk_ids)
        | (emitted.episode_ids != route.episode_ids)
        | (emitted.actor_versions != route.actor_versions)
    )
    if bool(source_metadata_mismatch.any().item()):
        index = tuple(
            int(item)
            for item in source_metadata_mismatch.nonzero(as_tuple=False)[0].tolist()
        )
        raise ValueError(
            "A valid Gate decision does not match its source chunk metadata. "
            f"First mismatch at {index}."
        )

    flow_valid_mask = _chunk_mask(
        loss_mask,
        shape=route.shape,
        name="loss_mask",
        device=route.route_used.device,
    )
    done_mask = _chunk_mask(dones, shape=torch.Size(expected_done_prefix), name="dones")
    destination_times = torch.full(
        route.shape, -1, dtype=torch.long, device=route.route_used.device
    )
    destination_columns = torch.full_like(destination_times, -1)
    columns = torch.arange(
        folded_batch, dtype=torch.long, device=route.route_used.device
    )
    if time_steps > 1:
        destination_times[:-1] = torch.arange(
            1, time_steps, dtype=torch.long, device=route.route_used.device
        )[:, None]
        destination_columns[:-1] = columns[None, :]
    base_batch = folded_batch // rollout_epoch
    if carry_pending_across_epochs and rollout_epoch > 1:
        cross_epoch_columns = columns < folded_batch - base_batch
        destination_times[-1, cross_epoch_columns] = 0
        destination_columns[-1, cross_epoch_columns] = (
            columns[cross_epoch_columns] + base_batch
        )

    candidate_mask = destination_times >= 0
    source_times, source_columns = candidate_mask.nonzero(as_tuple=True)
    target_times = destination_times[source_times, source_columns]
    target_columns = destination_columns[source_times, source_columns]
    target_is_valid = flow_valid_mask[target_times, target_columns]
    if not bool(target_is_valid.any().item()):
        return FastWAMPolicyAlignment(
            flow_advantages=advantages,
            flow_valid_mask=flow_valid_mask,
            gate_advantages=torch.zeros_like(advantages[..., 0]),
            gate_valid_mask=torch.zeros_like(flow_valid_mask),
        )

    source_episode = route.episode_ids[source_times, source_columns]
    target_episode = route.episode_ids[target_times, target_columns]
    episode_changed = target_episode != source_episode
    source_done = done_mask[source_times + 1, source_columns]
    target_forced = route.route_was_forced[target_times, target_columns]
    target_is_idm = route.route_used[target_times, target_columns] == int(WAMRoute.IDM)
    target_is_first_chunk = route.chunk_ids[target_times, target_columns] == 0

    _raise_first_pair(
        target_is_valid & source_done & ~episode_changed,
        message="A valid chunk followed a terminal source without an episode reset.",
        source_times=source_times,
        source_columns=source_columns,
        destination_times=target_times,
        destination_columns=target_columns,
    )
    _raise_first_pair(
        target_is_valid & episode_changed & ~source_done,
        message="An episode reset occurred without a terminal source chunk.",
        source_times=source_times,
        source_columns=source_columns,
        destination_times=target_times,
        destination_columns=target_columns,
    )
    invalid_reset = episode_changed & (
        (~target_forced) | (~target_is_idm) | (~target_is_first_chunk)
    )
    _raise_first_pair(
        target_is_valid & invalid_reset,
        message="The first valid chunk after reset must be forced IDM chunk zero.",
        source_times=source_times,
        source_columns=source_columns,
        destination_times=target_times,
        destination_columns=target_columns,
    )

    consumed = target_is_valid & ~episode_changed & ~target_forced
    source_decision_valid = emitted.valid[source_times, source_columns]
    _raise_first_pair(
        consumed & ~source_decision_valid,
        message="A non-forced route has no valid preceding Gate decision.",
        source_times=source_times,
        source_columns=source_columns,
        destination_times=target_times,
        destination_columns=target_columns,
    )
    _raise_first_pair(
        consumed
        & (
            route.route_source_chunk_ids[target_times, target_columns]
            != emitted.source_chunk_ids[source_times, source_columns]
        ),
        message="A route references the wrong Gate source chunk.",
        source_times=source_times,
        source_columns=source_columns,
        destination_times=target_times,
        destination_columns=target_columns,
    )
    _raise_first_pair(
        consumed
        & (
            route.chunk_ids[target_times, target_columns]
            != emitted.source_chunk_ids[source_times, source_columns] + 1
        ),
        message="A Gate decision did not control the immediately following chunk.",
        source_times=source_times,
        source_columns=source_columns,
        destination_times=target_times,
        destination_columns=target_columns,
    )
    _raise_first_pair(
        consumed
        & (
            route.route_used[target_times, target_columns]
            != emitted.next_route[source_times, source_columns]
        ),
        message="The executed route differs from the Gate decision.",
        source_times=source_times,
        source_columns=source_columns,
        destination_times=target_times,
        destination_columns=target_columns,
    )
    _raise_first_pair(
        consumed
        & (
            route.actor_versions[target_times, target_columns]
            != emitted.actor_versions[source_times, source_columns]
        ),
        message="A Gate decision crossed an actor-version boundary.",
        source_times=source_times,
        source_columns=source_columns,
        destination_times=target_times,
        destination_columns=target_columns,
    )

    gate_advantages = torch.zeros_like(advantages[..., 0])
    gate_valid_mask = torch.zeros_like(flow_valid_mask)
    valid_source_times = source_times[consumed]
    valid_source_columns = source_columns[consumed]
    valid_target_times = target_times[consumed]
    valid_target_columns = target_columns[consumed]
    gate_advantages[valid_source_times, valid_source_columns] = advantages[
        valid_target_times, valid_target_columns, 0
    ]
    gate_valid_mask[valid_source_times, valid_source_columns] = True
    return FastWAMPolicyAlignment(
        flow_advantages=advantages,
        flow_valid_mask=flow_valid_mask,
        gate_advantages=gate_advantages,
        gate_valid_mask=gate_valid_mask,
    )


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    normalization_std_floor: float = 0.0,
    normalization_statistics: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(
            advantages,
            loss_mask=loss_mask,
            std_floor=normalization_std_floor,
            statistics=normalization_statistics,
        )
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


def compute_fastwam_unnormalized_gate_alignment(
    *,
    rewards: torch.Tensor,
    route: ChunkRouteRecord,
    emitted: GateDecisionRecord,
    dones: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor | None,
    gamma: float,
    gae_lambda: float,
    rollout_epoch: int,
    carry_pending_across_epochs: bool,
) -> FastWAMPolicyAlignment:
    """Return configured-cost destination advantages before normalization."""

    if rewards.shape != (*route.shape, 1):
        raise ValueError("FastWAM configured rewards must have shape [time, batch, 1].")
    expected_values_shape = (route.shape[0] + 1, route.shape[1], 1)
    if values.shape != expected_values_shape:
        raise ValueError(
            "FastWAM critic values must include one bootstrap timestep; "
            f"expected {expected_values_shape}, got {tuple(values.shape)}."
        )
    if dones.shape[:2] != expected_values_shape[:2] or dones.dtype != torch.bool:
        raise ValueError(
            "FastWAM dones must be boolean and include one bootstrap timestep."
        )
    chunk_mask = _chunk_mask(
        valid_mask,
        shape=route.shape,
        name="valid_mask",
        device=route.route_used.device,
    )
    gae_dones = dones.reshape(*dones.shape[:2], -1).any(dim=-1)
    unnormalized, _ = compute_gae_advantages_and_returns(
        rewards=rewards[..., 0],
        gamma=gamma,
        gae_lambda=gae_lambda,
        values=values[..., 0],
        normalize_advantages=False,
        loss_mask=chunk_mask,
        dones=gae_dones,
    )
    return align_fastwam_policy_advantages(
        advantages=unnormalized.unsqueeze(-1),
        route=route,
        emitted=emitted,
        dones=dones,
        rollout_epoch=rollout_epoch,
        carry_pending_across_epochs=carry_pending_across_epochs,
        loss_mask=valid_mask,
    )


def summarize_fastwam_counterfactual_costs(
    *,
    environment_rewards: torch.Tensor,
    route: ChunkRouteRecord,
    emitted: GateDecisionRecord,
    dones: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor | None,
    charge_mask: torch.Tensor | None = None,
    idm_costs: Sequence[float],
    configured_idm_cost: float,
    configured_gate_advantages: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    rollout_epoch: int,
    carry_pending_across_epochs: bool,
    normalization_std_floor: float = 0.0,
    alignment_fn: Callable[..., FastWAMPolicyAlignment] | None = None,
) -> FastWAMCounterfactualCostAudit:
    """Evaluate several IDM costs on one immutable rollout batch.

    This diagnostic deliberately recomputes GAE and Gate alignment without
    autograd or optimizer work. The production rollout tensors are never
    mutated, so every candidate observes identical rewards, values, routes,
    dones, masks, and behavior-policy decisions. Legacy callers retain delayed
    alignment; config-selected actor subclasses may supply their route-contract
    alignment and configured normalization floor through the optional hooks.
    """

    normalized_costs = tuple(float(item) for item in idm_costs)
    if (
        not normalized_costs
        or normalized_costs != tuple(sorted(set(normalized_costs)))
        or normalized_costs[0] != 0.0
        or any(not math.isfinite(item) or item < 0 for item in normalized_costs)
    ):
        raise ValueError(
            "Counterfactual IDM costs must be unique, sorted, finite, and begin at zero."
        )
    configured_idm_cost = float(configured_idm_cost)
    if configured_idm_cost not in normalized_costs:
        raise ValueError("Configured IDM cost is absent from the counterfactual grid.")
    if route.shape != environment_rewards.shape[:2] or emitted.shape != route.shape:
        raise ValueError("Counterfactual rewards, routes, and Gate records disagree.")
    expected_values_shape = (route.shape[0] + 1, route.shape[1], 1)
    if values.shape != expected_values_shape:
        raise ValueError(
            "Counterfactual critic values must include one bootstrap timestep; "
            f"expected {expected_values_shape}, got {tuple(values.shape)}."
        )
    if dones.shape[:2] != expected_values_shape[:2] or dones.dtype != torch.bool:
        raise ValueError(
            "Counterfactual dones must be boolean and include one bootstrap timestep."
        )
    if configured_gate_advantages.shape != route.shape:
        raise ValueError("Configured Gate advantages must match the route shape.")

    chunk_mask = _chunk_mask(
        valid_mask,
        shape=route.shape,
        name="valid_mask",
        device=route.route_used.device,
    )
    gae_dones = dones.reshape(*dones.shape[:2], -1).any(dim=-1)
    gae_values = values[..., 0]
    align = alignment_fn or align_fastwam_policy_advantages
    results: list[
        tuple[
            float,
            FastWAMPolicyAlignment,
            FastWAMPolicyAlignment,
            float,
        ]
    ] = []
    with torch.no_grad():
        for idm_cost in normalized_costs:
            cost_result = apply_fastwam_chunk_cost(
                environment_rewards=environment_rewards,
                route_used=route.route_used,
                idm_cost=idm_cost,
                uncond_cost=0.0,
                valid_mask=valid_mask,
                charge_mask=charge_mask,
            )
            unnormalized, _ = compute_gae_advantages_and_returns(
                rewards=cost_result.rewards[..., 0],
                gamma=gamma,
                gae_lambda=gae_lambda,
                values=gae_values,
                normalize_advantages=False,
                loss_mask=chunk_mask,
                dones=gae_dones,
            )
            normalized, _ = compute_gae_advantages_and_returns(
                rewards=cost_result.rewards[..., 0],
                gamma=gamma,
                gae_lambda=gae_lambda,
                values=gae_values,
                normalize_advantages=True,
                loss_mask=chunk_mask,
                dones=gae_dones,
                normalization_std_floor=normalization_std_floor,
            )
            unnormalized_alignment = align(
                advantages=unnormalized.unsqueeze(-1),
                route=route,
                emitted=emitted,
                dones=dones,
                rollout_epoch=rollout_epoch,
                carry_pending_across_epochs=carry_pending_across_epochs,
                loss_mask=valid_mask,
            )
            normalized_alignment = align(
                advantages=normalized.unsqueeze(-1),
                route=route,
                emitted=emitted,
                dones=dones,
                rollout_epoch=rollout_epoch,
                carry_pending_across_epochs=carry_pending_across_epochs,
                loss_mask=valid_mask,
            )
            expected_cost_sum = float(
                cost_result.costs[..., 0][chunk_mask].to(torch.float64).sum().item()
            )
            results.append(
                (
                    idm_cost,
                    unnormalized_alignment,
                    normalized_alignment,
                    expected_cost_sum,
                )
            )

    baseline_unnormalized = results[0][1].gate_advantages
    baseline_normalized = results[0][2].gate_advantages
    eligible_mask = results[0][1].gate_valid_mask
    if any(
        not torch.equal(item[1].gate_valid_mask, eligible_mask)
        or not torch.equal(item[2].gate_valid_mask, eligible_mask)
        for item in results[1:]
    ):
        raise ValueError("Counterfactual costs changed Gate eligibility.")
    idm_mask = eligible_mask & (emitted.next_route == int(WAMRoute.IDM))
    uncond_mask = eligible_mask & (emitted.next_route == int(WAMRoute.UNCOND))
    if not bool(idm_mask.any().item()) or not bool(uncond_mask.any().item()):
        raise ValueError(
            "Counterfactual cost audit requires eligible IDM and UNCOND decisions."
        )

    entries = []
    configured_alignment_error = None
    for idm_cost, unnormalized, normalized, expected_cost_sum in results:
        unnormalized_delta = unnormalized.gate_advantages - baseline_unnormalized
        normalized_delta = normalized.gate_advantages - baseline_normalized
        if idm_cost == configured_idm_cost:
            difference = (
                normalized.gate_advantages - configured_gate_advantages
            ).abs()[eligible_mask]
            configured_alignment_error = (
                float(difference.max().item()) if difference.numel() else 0.0
            )
        entries.append(
            FastWAMCounterfactualCostEntry(
                idm_cost=idm_cost,
                expected_cost_sum=expected_cost_sum,
                unnormalized_gate_advantage=_summarize_selected_scalars(
                    unnormalized.gate_advantages,
                    mask=eligible_mask,
                ),
                normalized_gate_advantage=_summarize_selected_scalars(
                    normalized.gate_advantages,
                    mask=eligible_mask,
                ),
                unnormalized_idm_gate_advantage=_summarize_selected_scalars(
                    unnormalized.gate_advantages,
                    mask=idm_mask,
                ),
                normalized_idm_gate_advantage=_summarize_selected_scalars(
                    normalized.gate_advantages,
                    mask=idm_mask,
                ),
                unnormalized_uncond_gate_advantage=_summarize_selected_scalars(
                    unnormalized.gate_advantages,
                    mask=uncond_mask,
                ),
                normalized_uncond_gate_advantage=_summarize_selected_scalars(
                    normalized.gate_advantages,
                    mask=uncond_mask,
                ),
                unnormalized_idm_delta_from_zero=_summarize_selected_scalars(
                    unnormalized_delta,
                    mask=idm_mask,
                ),
                normalized_idm_delta_from_zero=_summarize_selected_scalars(
                    normalized_delta,
                    mask=idm_mask,
                ),
            )
        )
    if configured_alignment_error is None:
        raise AssertionError("Configured counterfactual candidate was not evaluated.")
    return FastWAMCounterfactualCostAudit(
        configured_idm_cost=configured_idm_cost,
        configured_alignment_max_abs_error=configured_alignment_error,
        eligible_gate_decision_count=int(eligible_mask.sum().item()),
        eligible_idm_decision_count=int(idm_mask.sum().item()),
        eligible_uncond_decision_count=int(uncond_mask.sum().item()),
        entries=tuple(entries),
    )


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",  # "trajectory" or "turn"
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx (e.g., [0,0,1,1,2,2,3,3,4,4,...,15,15])
    - Trajectories 0-3 belong to question 0, 4-7 to question 1, etc.
    - We must compute GRPO separately for each question's group_size trajectories

    Two advantage computation modes:
    1. "trajectory": Trajectory-level GRPO (Method 1)
       - Compute mean/std over group_size trajectory rewards per question
       - Broadcast same advantage to all turns in a trajectory
       - Example: Q0 has 4 trajs with 1,2,3,4 turns. Compute GRPO over 4 traj rewards,
                  then assign traj0_adv to its 1 turn, traj1_adv to its 2 turns, etc.

    2. "turn": Turn-level GRPO (Method 2)
       - Compute mean/std over all turns within each question
       - Example: Q0 has 4 trajs with 1,2,3,4 turns = 10 turns total.
                  Compute GRPO over these 10 turn rewards (currently all same within traj).
       - Future-proof: works when turns have different rewards within same trajectory

    Args:
        rewards: Shape [num_sequence, 1] after preprocessing (num_sequence = total turns)
        loss_mask: Shape [seq_len, num_sequence] after preprocessing
        group_size: Number of trajectories per question (e.g., 4)
        idx_to_traj: List mapping turn_idx -> global_traj_idx
        advantage_mode: "trajectory" or "turn"

    Returns:
        advantages: Shape [seq_len, num_sequence]
    """
    num_sequence = len(idx_to_traj)

    rewards_flat = rewards.squeeze(-1)

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )

    if advantage_mode == "trajectory":
        # Aggregate turn rewards into per-trajectory rewards first.
        trajectory_rewards = torch.zeros(
            num_trajectories, dtype=rewards.dtype, device=rewards.device
        )
        trajectory_counts = torch.zeros(
            num_trajectories, dtype=torch.long, device=rewards.device
        )

        for turn_idx, traj_idx in enumerate(idx_to_traj):
            trajectory_rewards[traj_idx] += rewards_flat[turn_idx]
            trajectory_counts[traj_idx] += 1

        # Step 1: Average rewards per trajectory.
        trajectory_rewards = trajectory_rewards / trajectory_counts.clamp(min=1).float()

        # Step 2: reshape to [num_questions, group_size] for per-question GRPO.
        trajectory_rewards_grouped = trajectory_rewards.view(num_questions, group_size)

        # Step 3: compute per-question mean and std.
        per_question_mean = trajectory_rewards_grouped.mean(
            dim=-1, keepdim=True
        )  # [num_questions, 1]
        per_question_std = trajectory_rewards_grouped.std(
            dim=-1, keepdim=True
        )  # [num_questions, 1]

        # Step 4: normalize within each question group.
        normalized_trajectory_rewards = (
            trajectory_rewards_grouped - per_question_mean
        ) / (per_question_std + 1e-6)  # [num_questions, group_size]

        # Step 5: flatten back to [num_trajectories].
        normalized_trajectory_rewards = normalized_trajectory_rewards.view(-1)

        # Step 6: broadcast trajectory advantages to all turns in that trajectory.
        for turn_idx, traj_idx in enumerate(idx_to_traj):
            turn_advantages[turn_idx] = normalized_trajectory_rewards[traj_idx]

    elif advantage_mode == "turn":
        # Step 1: map each turn to its owning question.
        turn_to_question = torch.tensor(
            [idx_to_traj[i] // group_size for i in range(num_sequence)],
            dtype=torch.long,
            device=rewards.device,
        )

        # Step 2: normalize turn rewards within each question group.
        for question_idx in range(num_questions):
            question_mask = turn_to_question == question_idx
            question_turn_rewards = rewards_flat[question_mask]

            # Step 3: compute mean and std for all turns in this question.
            question_mean = question_turn_rewards.mean()
            question_std = question_turn_rewards.std()

            # Step 4: normalize turn rewards within the question.
            normalized_question_rewards = (question_turn_rewards - question_mean) / (
                question_std + 1e-6
            )

            # Step 5: write normalized turn-level advantages back.
            turn_advantages[question_mask] = normalized_question_rewards

    else:
        raise ValueError(
            f"Invalid advantage_mode: {advantage_mode}. Must be 'trajectory' or 'turn'"
        )

    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("opd")
def compute_opd_advantages(
    prev_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    normalize_advantages: bool = False,
    **kwargs,
):
    """Compute OPD advantages from frozen teacher token log-probabilities."""
    assert teacher_logprobs is not None, (
        "OPD advantage computation requires post-rollout teacher_logprobs."
    )
    assert prev_logprobs is not None, (
        "OPD advantage computation requires prev_logprobs from student rollout."
    )
    assert teacher_logprobs.shape == prev_logprobs.shape, (
        f"teacher_logprobs shape {teacher_logprobs.shape} must match "
        f"prev_logprobs shape {prev_logprobs.shape}."
    )
    assert not normalize_advantages, (
        "VLA-OPD uses raw reverse-KL rewards; set normalize_advantages to False."
    )
    num_action_chunks = kwargs.get("num_action_chunks", None)
    assert num_action_chunks is not None, (
        "OPD advantage computation requires num_action_chunks."
    )
    advantages = teacher_logprobs.float() - prev_logprobs.float()
    assert advantages.shape[-1] % num_action_chunks == 0, (
        f"OPD token count {advantages.shape[-1]} must be divisible by "
        f"num_action_chunks {num_action_chunks}."
    )
    advantages = advantages.reshape(*advantages.shape[:-1], num_action_chunks, -1)
    if loss_mask is not None:
        target_steps = loss_mask.shape[0]
        assert advantages.shape[0] in {target_steps, target_steps + 1}, (
            f"OPD advantages time dimension {advantages.shape[0]} must match "
            f"loss_mask time dimension {target_steps} or include one bootstrap step."
        )
        advantages = advantages[:target_steps]

    return advantages, None


@register_advantage("raw")
def compute_raw_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_advantages: bool = False,
    **kwargs,
):
    """
    Return raw rewards or normalized rewards.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        normalize_advantages (bool): Whether to normalize advantages.

    Returns:
        torch.Tensor: advantages
    """
    if rewards.ndim == 2:
        rewards = rewards.reshape(-1)
    advantages = rewards.unsqueeze(0).expand_as(loss_mask) * loss_mask

    # Simple baseline subtraction (mean of valid advantages)
    if normalize_advantages:
        valid = advantages[loss_mask.bool()]
        if valid.numel() > 0:
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-5)

    return advantages, None
