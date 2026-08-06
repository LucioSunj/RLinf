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
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

FASTWAM_TRAINING_GUARD_STATE_SCHEMA = "fastwam-training-guard-state-v1"


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
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")
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
        if (
            not 0.0
            <= self.eligible_idm_fraction_min
            < (self.eligible_idm_fraction_max <= 1.0)
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
            summed=frozenset(self._ROLLOUT_KEYS[:2]),
        )
        successes = int(metrics[self._ROLLOUT_KEYS[0]])
        idm_fraction = metrics[self._ROLLOUT_KEYS[2]]
        eligible_count = int(metrics[self._ROLLOUT_KEYS[3]])
        eligible_idm_count = int(metrics[self._ROLLOUT_KEYS[4]])
        uncond_count = int(metrics[self._ROLLOUT_KEYS[5]])
        if successes < 0:
            raise ValueError("FastWAM success count must be non-negative.")
        if not 0.0 <= idm_fraction <= 1.0:
            raise ValueError("FastWAM eligible IDM fraction must lie in [0, 1].")
        if eligible_count < 1 or not 0 <= eligible_idm_count <= eligible_count:
            raise ValueError("FastWAM eligible Gate/IDM counts are invalid.")
        if uncond_count < 0:
            raise ValueError("FastWAM valid UNCOND count is negative.")
        expected_fraction = eligible_idm_count / eligible_count
        if not math.isclose(idm_fraction, expected_fraction, abs_tol=1e-12):
            raise ValueError("FastWAM eligible IDM count/fraction do not reconcile.")
        if successes:
            self._consecutive_zero_success_batches = 0
        else:
            self._consecutive_zero_success_batches += 1
        self._pending_idm_fraction = idm_fraction
        self._pending_gate_sample_count = eligible_count
        self._pending_uncond_sample_count = uncond_count
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
            float(self._pending_gate_sample_count),
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
        }

    def state_dict(self) -> dict[str, Any]:
        """Return compact checkpointable guard state without trajectory data."""

        return {
            "schema": FASTWAM_TRAINING_GUARD_STATE_SCHEMA,
            "config_sha256": self._config_sha256,
            "consecutive_zero_success_batches": (
                self._consecutive_zero_success_batches
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
        observed = int(state.get("observed_runner_steps", -1))
        if streak < 0 or observed < 0:
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
        self._observed_runner_steps = observed
        self._pending_idm_fraction = pending
        self._pending_gate_sample_count = pending_gate
        self._pending_uncond_sample_count = pending_uncond
        self._history = restored
