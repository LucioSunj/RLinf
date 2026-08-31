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

"""Fail-closed runtime guards for bounded FastWAM scientific training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FASTWAM_TRAINING_GUARD_STATE_SCHEMA = "fastwam-training-guard-state-v2"
FASTWAM_BREAK_EVEN_METRIC = "fastwam/counterfactual/break_even_idm_cost"
FASTWAM_CONFIGURED_COST_METRIC = "fastwam/counterfactual/configured_idm_cost"
FASTWAM_EFFECTIVE_GATE_COUNT_METRIC = "kv_cache/effective_gate_gradient_count"
FASTWAM_FULL_ELIGIBLE_GATE_COUNT_METRIC = "kv_cache/full_eligible_gate_samples"


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {"enabled": False}
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    raise TypeError("FastWAM training guard config must be a mapping.")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite.")
    return result


def append_fastwam_counterfactual_cost_audit_jsonl(
    path: str | Path,
    *,
    runner_step: int,
    artifact: Mapping[str, Any],
) -> None:
    """Append one full counterfactual-cost table to a run-scoped JSONL file."""

    if runner_step < 0:
        raise ValueError("Counterfactual audit runner_step must be non-negative.")
    payload = {"runner_step": int(runner_step), **dict(artifact)}
    if payload.get("schema") != "fastwam-counterfactual-cost-audit-v1":
        raise ValueError("Counterfactual audit schema mismatch.")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Counterfactual audit entries must be a non-empty list.")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Counterfactual audit entries must be mappings.")
        for group in (
            "gate_advantage",
            "idm_destination_gate_advantage",
            "uncond_destination_gate_advantage",
            "idm_destination_delta_from_zero",
        ):
            grouped = entry.get(group)
            if not isinstance(grouped, Mapping):
                raise ValueError(f"Counterfactual audit {group} group is missing.")
            for normalization in ("unnormalized", "normalized"):
                summary = grouped.get(normalization)
                if not isinstance(summary, Mapping):
                    raise ValueError(
                        f"Counterfactual audit {group}/{normalization} is missing."
                    )
                if "sum_of_squares" not in summary:
                    raise ValueError(
                        f"Counterfactual audit {group}/{normalization} "
                        "sum_of_squares is missing."
                    )
                quantiles = summary.get("quantiles")
                if not isinstance(quantiles, Mapping) or any(
                    key not in quantiles for key in ("p10", "p25", "p50", "p75", "p90")
                ):
                    raise ValueError(
                        f"Counterfactual audit {group}/{normalization} "
                        "quantiles are missing."
                    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class FastWAMTrainingGuard:
    """Track sparse-success and PPO stability across runner-step resumes."""

    _ROLLOUT_KEYS = (
        "fastwam/raw_positive_success_signal_count",
        "fastwam/successful_trajectory_count",
        "fastwam/eligible_idm_fraction",
        "fastwam/eligible_gate_decision_count",
        "fastwam/eligible_idm_decision_count",
        "fastwam/valid_uncond_chunk_count",
        "rewards",
        "advantages_max",
        "advantages_mean",
        "advantages_min",
        "returns_max",
        "returns_mean",
        "returns_min",
        "values_max",
        "values_mean",
        "values_min",
    )
    _TRAINING_KEYS = (
        "actor/grad_norm",
        "critic/explained_variance",
        "critic/value_clip_ratio",
        "critic/value_loss",
        "gate/approx_kl",
        "gate/clip_fraction",
        "gate/entropy",
        "gate/ratio",
        "gate/sample_count",
        "uncond_flow/approx_kl",
        "uncond_flow/clip_fraction",
        "uncond_flow/entropy",
        "uncond_flow/ratio",
        "uncond_flow/sample_count",
    )

    def __init__(self, config: Any):
        raw = _plain_mapping(config)
        self.enabled = bool(raw.get("enabled", False))
        self.zero_success_patience = int(raw.get("zero_success_patience", 1))
        self.break_even_patience = int(raw.get("break_even_patience", 3))
        cost_audit = _plain_mapping(raw.get("cost_audit", None))
        self.cost_audit_enabled = self.enabled and bool(
            cost_audit.get("enabled", False)
        )
        self.break_even_guard_enabled = self.cost_audit_enabled and bool(
            cost_audit.get("break_even_guard_enabled", True)
        )
        self.window_size = int(raw.get("window_size", 3))
        self.eligible_idm_fraction_min = float(
            raw.get("eligible_idm_fraction_min", 0.05)
        )
        self.eligible_idm_fraction_max = float(
            raw.get("eligible_idm_fraction_max", 0.95)
        )
        self.gate_entropy_min = float(raw.get("gate_entropy_min", 0.1))
        self.gate_kl_median_max = float(raw.get("gate_kl_median_max", 0.05))
        self.gate_kl_single_max = float(raw.get("gate_kl_single_max", 0.1))
        self.gate_clip_median_max = float(raw.get("gate_clip_median_max", 0.6))
        self.gate_clip_single_max = float(raw.get("gate_clip_single_max", 0.8))
        self._validate_config()
        self._config = {
            "enabled": self.enabled,
            "zero_success_patience": self.zero_success_patience,
            "break_even_patience": self.break_even_patience,
            "break_even_guard_enabled": self.break_even_guard_enabled,
            "window_size": self.window_size,
            "eligible_idm_fraction_min": self.eligible_idm_fraction_min,
            "eligible_idm_fraction_max": self.eligible_idm_fraction_max,
            "gate_entropy_min": self.gate_entropy_min,
            "gate_kl_median_max": self.gate_kl_median_max,
            "gate_kl_single_max": self.gate_kl_single_max,
            "gate_clip_median_max": self.gate_clip_median_max,
            "gate_clip_single_max": self.gate_clip_single_max,
        }
        self._config_sha256 = _canonical_sha256(self._config)
        self._consecutive_zero_success_batches = 0
        self._consecutive_break_even_below_cost = 0
        self._observed_runner_steps = 0
        self._pending_idm_fraction: float | None = None
        self._pending_gate_sample_count: int | None = None
        self._pending_uncond_sample_count: int | None = None
        self._history: dict[str, list[float]] = {
            "eligible_idm_fraction": [],
            "gate_entropy": [],
            "gate_approx_kl": [],
            "gate_clip_fraction": [],
        }

    def _validate_config(self) -> None:
        if self.zero_success_patience < 1:
            raise ValueError("zero_success_patience must be positive.")
        if self.break_even_patience < 1:
            raise ValueError("break_even_patience must be positive.")
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")
        if (
            self.break_even_guard_enabled
            and self.break_even_patience > self.window_size + 1
        ):
            raise ValueError(
                "FastWAM break-even patience exceeds the retained route-history window."
            )
        finite = {
            "eligible_idm_fraction_min": self.eligible_idm_fraction_min,
            "eligible_idm_fraction_max": self.eligible_idm_fraction_max,
            "gate_entropy_min": self.gate_entropy_min,
            "gate_kl_median_max": self.gate_kl_median_max,
            "gate_kl_single_max": self.gate_kl_single_max,
            "gate_clip_median_max": self.gate_clip_median_max,
            "gate_clip_single_max": self.gate_clip_single_max,
        }
        if any(not math.isfinite(value) for value in finite.values()):
            raise ValueError("FastWAM training guard thresholds must be finite.")
        if not (
            0.0
            <= self.eligible_idm_fraction_min
            < self.eligible_idm_fraction_max
            <= 1.0
        ):
            raise ValueError("Eligible IDM fraction bounds must lie inside [0, 1].")
        if self.gate_entropy_min < 0.0:
            raise ValueError("gate_entropy_min must be non-negative.")
        if not 0.0 <= self.gate_kl_median_max <= self.gate_kl_single_max:
            raise ValueError("Gate KL median/single limits are inconsistent.")
        if not 0.0 <= self.gate_clip_median_max <= self.gate_clip_single_max <= 1.0:
            raise ValueError("Gate clip median/single limits are inconsistent.")

    @staticmethod
    def _aggregate(
        metrics_list: Sequence[Mapping[str, Any]],
        keys: Sequence[str],
        *,
        summed: frozenset[str] = frozenset(),
    ) -> dict[str, float]:
        if not metrics_list:
            raise ValueError("FastWAM training guard received no worker metrics.")
        missing = sorted(
            key for key in keys if any(key not in metrics for metrics in metrics_list)
        )
        if missing:
            raise ValueError(
                "FastWAM training guard is missing required metrics: "
                + ", ".join(missing)
            )
        result: dict[str, float] = {}
        for key in keys:
            values = [
                _finite_float(metrics[key], name=f"FastWAM metric {key}")
                for metrics in metrics_list
            ]
            result[key] = sum(values) if key in summed else sum(values) / len(values)
        return result

    def observe_rollout(
        self, metrics_list: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Check raw success before optimization and stage route history."""

        if not self.enabled:
            return {"status": "DISABLED"}
        metrics = self._aggregate(
            metrics_list,
            self._ROLLOUT_KEYS,
            summed=frozenset(
                (
                    *self._ROLLOUT_KEYS[:2],
                    "fastwam/eligible_gate_decision_count",
                    "fastwam/eligible_idm_decision_count",
                    "fastwam/valid_uncond_chunk_count",
                )
            ),
        )
        successes = int(metrics[self._ROLLOUT_KEYS[0]])
        eligible_count = int(metrics[self._ROLLOUT_KEYS[3]])
        eligible_idm_count = int(metrics[self._ROLLOUT_KEYS[4]])
        uncond_count = int(metrics[self._ROLLOUT_KEYS[5]])
        if successes < 0:
            raise ValueError("FastWAM success count must be non-negative.")
        if eligible_count < 1 or not 0 <= eligible_idm_count <= eligible_count:
            raise ValueError("FastWAM eligible Gate/IDM counts are invalid.")
        if uncond_count < 0:
            raise ValueError("FastWAM valid UNCOND count is negative.")
        idm_fraction = eligible_idm_count / eligible_count
        for worker, worker_metrics in enumerate(metrics_list):
            local_count = _finite_float(
                worker_metrics["fastwam/eligible_gate_decision_count"],
                name=f"worker {worker} eligible Gate count",
            )
            local_idm_count = _finite_float(
                worker_metrics["fastwam/eligible_idm_decision_count"],
                name=f"worker {worker} eligible IDM count",
            )
            local_fraction = _finite_float(
                worker_metrics["fastwam/eligible_idm_fraction"],
                name=f"worker {worker} eligible IDM fraction",
            )
            if (
                not local_count.is_integer()
                or not local_idm_count.is_integer()
                or local_count < 1
                or not 0 <= local_idm_count <= local_count
                or not math.isclose(
                    local_fraction,
                    local_idm_count / local_count,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    "FastWAM worker eligible IDM count/fraction values do not "
                    "reconcile."
                )
        if successes:
            self._consecutive_zero_success_batches = 0
        else:
            self._consecutive_zero_success_batches += 1
        self._pending_idm_fraction = idm_fraction
        self._pending_gate_sample_count = eligible_count
        self._pending_uncond_sample_count = uncond_count
        break_even = None
        configured_idm_cost = None
        break_even_route_window: list[float] = []
        break_even_route_monotonic_decline = False
        if self.cost_audit_enabled:
            missing_cost = [
                index
                for index, worker_metrics in enumerate(metrics_list)
                if FASTWAM_CONFIGURED_COST_METRIC not in worker_metrics
            ]
            if missing_cost:
                raise ValueError(
                    "FastWAM counterfactual cost audit is missing configured-cost metrics "
                    f"from workers {missing_cost}."
                )
            configured_costs = [
                _finite_float(
                    worker_metrics[FASTWAM_CONFIGURED_COST_METRIC],
                    name="FastWAM configured IDM cost",
                )
                for worker_metrics in metrics_list
            ]
            configured_idm_cost = configured_costs[0]
            if configured_idm_cost < 0.0 or any(
                not math.isclose(
                    value,
                    configured_idm_cost,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for value in configured_costs[1:]
            ):
                raise ValueError(
                    "FastWAM configured IDM costs are invalid or disagree across "
                    "actor workers."
                )
            break_even_values: list[float] = []
            break_even_defined = True
            for worker_metrics in metrics_list:
                value = worker_metrics.get(FASTWAM_BREAK_EVEN_METRIC)
                if value is None:
                    break_even_defined = False
                    continue
                break_even_values.append(
                    _finite_float(value, name="FastWAM break-even IDM cost")
                )
            if break_even_defined and break_even_values:
                break_even = min(break_even_values)
                if break_even < 0.0:
                    raise ValueError("FastWAM break-even IDM cost is negative.")

        if self.break_even_guard_enabled:
            below_cost = configured_idm_cost > 0.0 and (
                break_even is None or break_even < configured_idm_cost
            )
            if below_cost:
                self._consecutive_break_even_below_cost += 1
                previous_count = min(
                    self._consecutive_break_even_below_cost - 1,
                    self.break_even_patience - 1,
                )
                previous_fractions = (
                    self._history["eligible_idm_fraction"][-previous_count:]
                    if previous_count
                    else []
                )
                break_even_route_window = [*previous_fractions, idm_fraction]
                break_even_route_monotonic_decline = len(
                    break_even_route_window
                ) == self.break_even_patience and all(
                    break_even_route_window[index] < break_even_route_window[index - 1]
                    for index in range(1, len(break_even_route_window))
                )
            else:
                self._consecutive_break_even_below_cost = 0
            if (
                self._consecutive_break_even_below_cost >= self.break_even_patience
                and break_even_route_monotonic_decline
            ):
                rendered = "undefined" if break_even is None else str(break_even)
                raise RuntimeError(
                    "FastWAM break-even IDM cost stayed below the configured cost "
                    "while the eligible IDM fraction declined monotonically for "
                    f"{self.break_even_patience} rollouts: {rendered} < "
                    f"{configured_idm_cost}; route window "
                    f"{break_even_route_window}."
                )
        if self._consecutive_zero_success_batches >= self.zero_success_patience:
            if self.zero_success_patience == 1:
                raise RuntimeError(
                    "FastWAM training guard observed a zero sparse-success rollout "
                    "batch; refusing to optimize this positive-cost stage."
                )
            raise RuntimeError(
                "FastWAM training guard observed "
                f"{self._consecutive_zero_success_batches} consecutive "
                "zero sparse-success rollout batches."
            )
        route_count_keys = {
            "valid_chunk_count": "fastwam/route/valid_chunk_count",
            "valid_idm_chunk_count": "fastwam/route/valid_idm_chunk_count",
            "forced_route_count": "fastwam/route/forced_count",
        }
        route_counts: dict[str, int] = {}
        present = {
            source: [source in worker for worker in metrics_list]
            for source in route_count_keys.values()
        }
        if any(any(values) and not all(values) for values in present.values()):
            raise ValueError(
                "FastWAM route count metrics are present for only some workers."
            )
        for destination, source in route_count_keys.items():
            if all(present[source]):
                values = [
                    _finite_float(
                        worker[source],
                        name=f"FastWAM route count {source}",
                    )
                    for worker in metrics_list
                ]
                if any(value < 0 or not value.is_integer() for value in values):
                    raise ValueError("FastWAM route count metrics are invalid.")
                route_counts[destination] = int(sum(values))
        return {
            "status": "PASS",
            "positive_success_signal_count": successes,
            "eligible_idm_fraction": idm_fraction,
            "eligible_gate_decision_count": eligible_count,
            "eligible_idm_decision_count": eligible_idm_count,
            "valid_uncond_chunk_count": uncond_count,
            "consecutive_zero_success_batches": (
                self._consecutive_zero_success_batches
            ),
            "break_even_idm_cost": break_even,
            "configured_idm_cost": configured_idm_cost,
            "consecutive_break_even_below_cost": (
                self._consecutive_break_even_below_cost
            ),
            **route_counts,
            "break_even_route_window": break_even_route_window,
            "break_even_route_monotonic_decline": (break_even_route_monotonic_decline),
        }

    def observe_training(
        self, metrics_list: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Check one completed update group and rolling PPO stability."""

        if not self.enabled:
            return {"status": "DISABLED"}
        if (
            self._pending_idm_fraction is None
            or self._pending_gate_sample_count is None
            or self._pending_uncond_sample_count is None
        ):
            raise RuntimeError(
                "Training metrics arrived without a rollout observation."
            )
        metrics = self._aggregate(metrics_list, self._TRAINING_KEYS)
        gate_sample_count_basis = "full_eligible_rollout"
        expected_gate_sample_count = float(self._pending_gate_sample_count)
        sampling_count_keys = (
            FASTWAM_EFFECTIVE_GATE_COUNT_METRIC,
            FASTWAM_FULL_ELIGIBLE_GATE_COUNT_METRIC,
        )
        if any(
            key in worker_metrics
            for worker_metrics in metrics_list
            for key in sampling_count_keys
        ):
            sampling_counts = self._aggregate(metrics_list, sampling_count_keys)
            full_eligible_count = sampling_counts[
                FASTWAM_FULL_ELIGIBLE_GATE_COUNT_METRIC
            ]
            if not math.isclose(
                full_eligible_count,
                float(self._pending_gate_sample_count),
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "FastWAM full eligible Gate count does not reconcile with "
                    "the rollout."
                )
            expected_gate_sample_count = sampling_counts[
                FASTWAM_EFFECTIVE_GATE_COUNT_METRIC
            ]
            if not 0.0 <= expected_gate_sample_count <= full_eligible_count:
                raise RuntimeError(
                    "FastWAM effective Gate gradient count is outside the full "
                    "eligible count."
                )
            gate_sample_count_basis = "sampled_effective"
        gate_kl = metrics["gate/approx_kl"]
        gate_clip = metrics["gate/clip_fraction"]
        gate_entropy = metrics["gate/entropy"]
        if gate_kl > self.gate_kl_single_max:
            raise RuntimeError(
                "FastWAM single-step Gate KL exceeds the configured limit: "
                f"{gate_kl} > {self.gate_kl_single_max}."
            )
        if gate_clip > self.gate_clip_single_max:
            raise RuntimeError(
                "FastWAM single-step Gate clip fraction exceeds the configured "
                f"limit: {gate_clip} > {self.gate_clip_single_max}."
            )
        if not math.isclose(
            metrics["gate/sample_count"],
            expected_gate_sample_count,
            abs_tol=1e-6,
        ):
            raise RuntimeError("FastWAM Gate PPO sample count does not reconcile.")
        if not math.isclose(
            metrics["uncond_flow/sample_count"],
            float(self._pending_uncond_sample_count),
            abs_tol=1e-6,
        ):
            raise RuntimeError("FastWAM UNCOND Flow sample count does not reconcile.")

        additions = {
            "eligible_idm_fraction": self._pending_idm_fraction,
            "gate_entropy": gate_entropy,
            "gate_approx_kl": gate_kl,
            "gate_clip_fraction": gate_clip,
        }
        for key, value in additions.items():
            history = self._history[key]
            history.append(value)
            del history[: max(0, len(history) - self.window_size)]
        self._pending_idm_fraction = None
        self._pending_gate_sample_count = None
        self._pending_uncond_sample_count = None
        self._observed_runner_steps += 1

        if len(self._history["gate_approx_kl"]) == self.window_size:
            medians = {
                key: statistics.median(values) for key, values in self._history.items()
            }
            if (
                not self.eligible_idm_fraction_min
                <= medians["eligible_idm_fraction"]
                <= self.eligible_idm_fraction_max
            ):
                raise RuntimeError(
                    "FastWAM rolling eligible IDM fraction is outside the "
                    "configured bounds."
                )
            if medians["gate_entropy"] <= self.gate_entropy_min:
                raise RuntimeError("FastWAM rolling Gate entropy is too low.")
            if medians["gate_approx_kl"] > self.gate_kl_median_max:
                raise RuntimeError("FastWAM rolling Gate KL exceeds its limit.")
            if medians["gate_clip_fraction"] > self.gate_clip_median_max:
                raise RuntimeError(
                    "FastWAM rolling Gate clip fraction exceeds its limit."
                )
        else:
            medians = {
                key: statistics.median(values) for key, values in self._history.items()
            }
        return {
            "status": "PASS",
            "observed_runner_steps": self._observed_runner_steps,
            "history_count": len(self._history["gate_approx_kl"]),
            "rolling_medians": medians,
            "consecutive_zero_success_batches": (
                self._consecutive_zero_success_batches
            ),
            "consecutive_break_even_below_cost": (
                self._consecutive_break_even_below_cost
            ),
            "gate_sample_count_basis": gate_sample_count_basis,
            "expected_gate_sample_count": expected_gate_sample_count,
            "observed_gate_sample_count": metrics["gate/sample_count"],
        }

    def state_dict(self) -> dict[str, Any]:
        """Return compact checkpointable guard state without trajectory data."""

        return {
            "schema": FASTWAM_TRAINING_GUARD_STATE_SCHEMA,
            "config_sha256": self._config_sha256,
            "consecutive_zero_success_batches": (
                self._consecutive_zero_success_batches
            ),
            "consecutive_break_even_below_cost": (
                self._consecutive_break_even_below_cost
            ),
            "observed_runner_steps": self._observed_runner_steps,
            "pending_idm_fraction": self._pending_idm_fraction,
            "pending_gate_sample_count": self._pending_gate_sample_count,
            "pending_uncond_sample_count": self._pending_uncond_sample_count,
            "history": {key: list(values) for key, values in self._history.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Strictly restore guard state for a paired training resume."""

        if state.get("schema") != FASTWAM_TRAINING_GUARD_STATE_SCHEMA:
            raise ValueError("FastWAM training guard state schema mismatch.")
        if state.get("config_sha256") != self._config_sha256:
            raise ValueError("FastWAM training guard config hash mismatch.")
        streak = int(state.get("consecutive_zero_success_batches", -1))
        break_even_streak = int(state.get("consecutive_break_even_below_cost", -1))
        observed = int(state.get("observed_runner_steps", -1))
        if (
            streak < 0
            or break_even_streak < 0
            or break_even_streak >= self.break_even_patience
            or observed < 0
        ):
            raise ValueError("FastWAM training guard counters are invalid.")
        history = state.get("history")
        if not isinstance(history, Mapping) or set(history) != set(self._history):
            raise ValueError("FastWAM training guard history keys mismatch.")
        restored: dict[str, list[float]] = {}
        for key in self._history:
            values = list(history[key])
            if len(values) > self.window_size:
                raise ValueError("FastWAM training guard history exceeds window size.")
            restored[key] = [
                _finite_float(value, name=f"FastWAM guard history {key}")
                for value in values
            ]
        lengths = {len(values) for values in restored.values()}
        if len(lengths) != 1:
            raise ValueError("FastWAM training guard histories are misaligned.")
        pending = state.get("pending_idm_fraction")
        if pending is not None:
            pending = _finite_float(pending, name="pending IDM fraction")
            if not 0.0 <= pending <= 1.0:
                raise ValueError("Pending IDM fraction must lie in [0, 1].")
        pending_gate = state.get("pending_gate_sample_count")
        pending_uncond = state.get("pending_uncond_sample_count")
        if (pending is None) != (pending_gate is None) or (pending is None) != (
            pending_uncond is None
        ):
            raise ValueError("Pending FastWAM guard fields are incomplete.")
        if pending is not None:
            pending_gate = int(pending_gate)
            pending_uncond = int(pending_uncond)
            if pending_gate < 1 or pending_uncond < 0:
                raise ValueError("Pending FastWAM route sample counts are invalid.")
        self._consecutive_zero_success_batches = streak
        self._consecutive_break_even_below_cost = break_even_streak
        self._observed_runner_steps = observed
        self._pending_idm_fraction = pending
        self._pending_gate_sample_count = pending_gate
        self._pending_uncond_sample_count = pending_uncond
        self._history = restored
