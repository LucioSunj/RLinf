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

"""Closed-loop, success-blind FastWAM routing-budget calibration."""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from omegaconf import OmegaConf

from rlinf.runners.fastwam_routing_arms import (
    FastWAMEvalRoutingArm,
    RoutingParameter,
    build_fastwam_eval_routing_arm,
)

CALIBRATION_SELECTION_SCHEMA = "fastwam-routing-budget-selection-v1"
CALIBRATION_RESULT_SCHEMA = "fastwam-routing-budget-calibration-v1"
RATE_MATCH_TOLERANCE = 0.03


def _finite_fraction(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    """Create an artifact once, accepting only byte-equivalent reruns."""

    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise ValueError(f"Immutable FastWAM artifact differs: {path}.")
        return
    _write_json(path, payload)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"JSONL record at {path} is not a mapping.")
                records.append(payload)
    return records


def _ledger_reset_ids(path: str | Path) -> set[int]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("FastWAM evaluation ledger is missing entries.")
    ids = {int(entry["reset_state_id"]) for entry in entries}
    if len(ids) != len(entries):
        raise ValueError("FastWAM evaluation ledger has duplicate reset IDs.")
    return ids


def validate_calibration_test_ledgers(
    calibration_ledger: str | Path,
    test_ledger: str | Path,
) -> None:
    """Reject path or reset-ID leakage between calibration and test."""

    calibration = Path(calibration_ledger).expanduser().resolve()
    test = Path(test_ledger).expanduser().resolve()
    if calibration == test:
        raise ValueError("Calibration and test ledger paths must differ.")
    overlap = _ledger_reset_ids(calibration) & _ledger_reset_ids(test)
    if overlap:
        raise ValueError("Calibration and test reset IDs must not overlap.")


def validate_fastwam_budget_evaluation_config(cfg: Any) -> None:
    """Validate target-rate evaluation before any worker or backend starts."""

    value = OmegaConf.select(cfg, "evaluation.budget_matching", default=None)
    if value is None:
        return
    resolved = OmegaConf.to_container(value, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("FastWAM budget evaluation config must be a mapping.")
    config = dict(resolved)
    if not bool(config.get("enabled", False)):
        return
    if str(config.get("protocol", "")) != "closed_loop_calibrated_threshold":
        raise ValueError("Unsupported FastWAM budget evaluation protocol.")
    if not str(config.get("output_dir", "")):
        raise ValueError("FastWAM budget evaluation output_dir is required.")
    backend = dict(config.get("backend", {}))
    if backend.get("type") != "existing_hydra" or not str(
        backend.get("config_name", "")
    ):
        raise ValueError("FastWAM budget evaluation backend config is invalid.")
    if str(config.get("budget_semantics", "")) not in {
        "target_rate",
        "upper_bound",
    }:
        raise ValueError("Unsupported FastWAM evaluation budget semantics.")
    _finite_fraction(config.get("target_idm_fraction"), name="target rate")
    rate_tolerance = _finite_fraction(
        config.get("rate_tolerance", RATE_MATCH_TOLERANCE),
        name="rate tolerance",
    )
    if not math.isclose(
        rate_tolerance,
        RATE_MATCH_TOLERANCE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("FastWAM formal rate-match tolerance must equal 0.03.")
    if str(config.get("rate_scope", "")) != "eligible_gate_decisions":
        raise ValueError("FastWAM v1 budget calibration requires eligible Gate scope.")
    objective = str(OmegaConf.select(cfg, "routing_objective.type", default="")).lower()
    if objective != "eval_calibrated_target":
        raise ValueError(
            "Enabled budget calibration requires eval_calibrated_target objective."
        )
    calibration = dict(config.get("calibration", {}))
    test = dict(config.get("test", {}))
    search = dict(calibration.get("search", {}))
    selection = dict(calibration.get("selection", {}))
    if selection.get("metric") != "eligible_realized_fraction" or not bool(
        selection.get("success_blind", False)
    ):
        raise ValueError("FastWAM calibration selection must be success-blind rate.")
    if not bool(test.get("run_only_after_calibration_pass", True)):
        raise ValueError("FastWAM formal test must follow frozen calibration.")
    calibration_ledger = str(calibration.get("reset_ledger_path", ""))
    test_ledger = str(test.get("reset_ledger_path", ""))
    if not calibration_ledger or not test_ledger:
        raise ValueError("FastWAM budget evaluation requires both ledger paths.")
    if (
        Path(calibration_ledger).expanduser().resolve()
        == Path(test_ledger).expanduser().resolve()
    ):
        raise ValueError("Calibration and test ledger paths must differ.")
    if Path(calibration_ledger).is_file() and Path(test_ledger).is_file():
        validate_calibration_test_ledgers(calibration_ledger, test_ledger)
    checkpoints = list(config.get("checkpoints", ()))
    names = [str(item.get("name", "")) for item in checkpoints]
    paths = [str(item.get("path", "")) for item in checkpoints]
    if not checkpoints or any(not value for value in names + paths):
        raise ValueError("FastWAM budget evaluation checkpoints are incomplete.")
    if len(names) != len(set(names)):
        raise ValueError("FastWAM budget evaluation checkpoint names must be unique.")
    if (
        int(search.get("refine_top_k", 0)) < 1
        or int(search.get("refine_points_per_interval", -1)) < 0
        or int(search.get("maximum_candidate_runs", 0)) < 1
    ):
        raise ValueError("FastWAM budget calibration search settings are invalid.")
    default_candidates = search.get("coarse_candidates", ())
    arms = dict(config.get("arms", {}))
    if "learned" not in arms or not bool(arms["learned"].get("enabled", True)):
        raise ValueError("FastWAM budget evaluation requires a learned arm.")
    for arm_config in arms.values():
        if bool(arm_config.get("enabled", True)):
            arm = build_fastwam_eval_routing_arm(arm_config)
            domain = arm.parameter_domain(
                {"candidates": arm_config.get("candidates", default_candidates)}
            )
            seed = arm_config.get("routing_seed", calibration.get("routing_seed", 0))
            test_seed = arm_config.get(
                "test_routing_seed",
                arm_config.get("routing_seed", test.get("routing_seed", 0)),
            )
            for candidate in domain:
                arm.hydra_overrides(parameter=candidate, routing_seed=seed)
            arm.hydra_overrides(parameter=domain[0], routing_seed=test_seed)


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMEvalRequest:
    """One complete closed-loop evaluation request for a fixed route parameter."""

    checkpoint_path: str
    checkpoint_name: str
    ledger_path: str
    arm_name: str
    parameter: RoutingParameter
    routing_seed: int
    output_dir: str
    hydra_overrides: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class FastWAMEvalResult:
    """Count-reconciled result from one full closed-loop request."""

    request: FastWAMEvalRequest
    episode_count: int
    success_count: int
    eligible_decision_count: int
    eligible_idm_count: int
    eligible_realized_fraction: float
    executed_chunk_count: int
    executed_idm_count: int
    executed_realized_fraction: float
    forced_chunk_count: int
    status: str
    artifact_path: str
    gate_score_quantiles: Mapping[str, float] | None = None
    route_lag1_autocorrelation: float | None = None

    def __post_init__(self) -> None:
        counts = (
            self.episode_count,
            self.success_count,
            self.eligible_decision_count,
            self.eligible_idm_count,
            self.executed_chunk_count,
            self.executed_idm_count,
            self.forced_chunk_count,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("FastWAM evaluation counts must be non-negative.")
        if self.success_count > self.episode_count:
            raise ValueError("FastWAM success count exceeds episode count.")
        if self.eligible_idm_count > self.eligible_decision_count:
            raise ValueError("FastWAM eligible IDM count exceeds eligible count.")
        if self.executed_idm_count > self.executed_chunk_count:
            raise ValueError("FastWAM executed IDM count exceeds executed count.")
        for name in ("eligible_realized_fraction", "executed_realized_fraction"):
            _finite_fraction(getattr(self, name), name=name)
        if self.eligible_decision_count and not math.isclose(
            self.eligible_realized_fraction,
            self.eligible_idm_count / self.eligible_decision_count,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("FastWAM eligible rate does not reconcile with counts.")
        if self.executed_chunk_count and not math.isclose(
            self.executed_realized_fraction,
            self.executed_idm_count / self.executed_chunk_count,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("FastWAM executed rate does not reconcile with counts.")
        if self.route_lag1_autocorrelation is not None and (
            not math.isfinite(float(self.route_lag1_autocorrelation))
            or not -1.0 <= float(self.route_lag1_autocorrelation) <= 1.0
        ):
            raise ValueError("FastWAM route autocorrelation must lie in [-1, 1].")

    def to_artifact(self) -> dict[str, Any]:
        artifact = asdict(self)
        artifact["request"] = asdict(self.request)
        return artifact


class RoutingEvaluationBackend(Protocol):
    """Execute one closed-loop request without exposing orchestration details."""

    def run(self, request: FastWAMEvalRequest) -> FastWAMEvalResult: ...


class FakeRoutingEvaluationBackend:
    """Pure-CPU deterministic backend for calibrator and suite tests."""

    def __init__(
        self,
        rate_function: Callable[[FastWAMEvalRequest], float],
        *,
        success_function: Callable[[FastWAMEvalRequest], float] | None = None,
        decision_count: int = 10_000,
    ) -> None:
        if decision_count < 1:
            raise ValueError("Fake backend decision_count must be positive.")
        self.rate_function = rate_function
        self.success_function = success_function or (lambda request: 0.5)
        self.decision_count = decision_count
        self.calls: list[FastWAMEvalRequest] = []

    def run(self, request: FastWAMEvalRequest) -> FastWAMEvalResult:
        self.calls.append(request)
        rate = _finite_fraction(self.rate_function(request), name="fake eligible rate")
        success = _finite_fraction(
            self.success_function(request), name="fake success rate"
        )
        eligible_idm = round(rate * self.decision_count)
        realized = eligible_idm / self.decision_count
        episode_count = 100
        return FastWAMEvalResult(
            request=request,
            episode_count=episode_count,
            success_count=round(success * episode_count),
            eligible_decision_count=self.decision_count,
            eligible_idm_count=eligible_idm,
            eligible_realized_fraction=realized,
            executed_chunk_count=self.decision_count + episode_count,
            executed_idm_count=eligible_idm + episode_count,
            executed_realized_fraction=(eligible_idm + episode_count)
            / (self.decision_count + episode_count),
            forced_chunk_count=episode_count,
            status="COMPLETE",
            artifact_path=request.output_dir,
            gate_score_quantiles={"p10": 0.1, "p50": 0.5, "p90": 0.9},
            route_lag1_autocorrelation=None,
        )


class ExistingHydraEvaluationBackend:
    """Invoke the existing Hydra evaluator and aggregate its collector shards."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        config_name: str = "libero_10_fastwam_adaptive_eval",
        command_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.config_name = str(config_name)
        self.command_runner = command_runner

    def run(self, request: FastWAMEvalRequest) -> FastWAMEvalResult:
        output_dir = Path(request.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "FASTWAM_EVAL_OUTPUT_DIR": str(output_dir),
                "FASTWAM_EVAL_LEDGER": str(
                    Path(request.ledger_path).expanduser().resolve()
                ),
                "FASTWAM_EVAL_RUN_ID": (
                    f"{request.checkpoint_name}-{request.arm_name}"
                ),
                "FASTWAM_PROJECT_CHECKPOINT": str(
                    Path(request.checkpoint_path).expanduser().resolve()
                ),
            }
        )
        command = [
            "bash",
            str(self.repo_root / "evaluations/run_eval.sh"),
            "libero",
            self.config_name,
            *request.hydra_overrides,
        ]
        self.command_runner(
            command,
            cwd=self.repo_root,
            env=env,
            check=True,
        )
        return self._aggregate_existing_artifacts(request)

    @staticmethod
    def _aggregate_existing_artifacts(
        request: FastWAMEvalRequest,
    ) -> FastWAMEvalResult:
        root = Path(request.output_dir).expanduser().resolve()
        episodes = [
            record
            for path in sorted(root.rglob("episodes.rank-*.jsonl"))
            for record in _read_jsonl(path)
        ]
        chunks = [
            record
            for path in sorted(root.rglob("chunks.rank-*.jsonl"))
            for record in _read_jsonl(path)
        ]
        ledger_count = len(_ledger_reset_ids(request.ledger_path))
        eligible = sum(int(item["eligible_chunk_count"]) for item in episodes)
        eligible_idm = sum(int(item["eligible_idm_count"]) for item in episodes)
        executed = sum(int(item["executed_chunk_count"]) for item in episodes)
        executed_idm = sum(int(item["idm_chunk_count_total"]) for item in episodes)
        forced = sum(int(item["forced_initial_idm_count"]) for item in episodes)
        scores = sorted(
            float(item["gate_idm_probability"])
            for item in chunks
            if item.get("gate_idm_probability") is not None
        )
        quantiles = None
        if scores:
            quantiles = {
                "p10": scores[round(0.10 * (len(scores) - 1))],
                "p50": statistics.median(scores),
                "p90": scores[round(0.90 * (len(scores) - 1))],
            }
        route_pairs = []
        by_episode: dict[str, list[dict[str, Any]]] = {}
        for item in chunks:
            if not bool(item.get("route_was_forced", False)):
                by_episode.setdefault(str(item["episode_identity"]), []).append(item)
        for episode_chunks in by_episode.values():
            ordered = sorted(episode_chunks, key=lambda item: int(item["chunk_id"]))
            values = [float(item["route"] == "idm") for item in ordered]
            route_pairs.extend(zip(values[:-1], values[1:]))
        route_autocorrelation = None
        if route_pairs:
            left = [pair[0] for pair in route_pairs]
            right = [pair[1] for pair in route_pairs]
            left_mean = statistics.fmean(left)
            right_mean = statistics.fmean(right)
            left_variance = statistics.fmean((value - left_mean) ** 2 for value in left)
            right_variance = statistics.fmean(
                (value - right_mean) ** 2 for value in right
            )
            denominator = math.sqrt(left_variance * right_variance)
            if denominator > 0.0:
                covariance = statistics.fmean(
                    (left_value - left_mean) * (right_value - right_mean)
                    for left_value, right_value in route_pairs
                )
                route_autocorrelation = covariance / denominator
        return FastWAMEvalResult(
            request=request,
            episode_count=len(episodes),
            success_count=sum(bool(item["success"]) for item in episodes),
            eligible_decision_count=eligible,
            eligible_idm_count=eligible_idm,
            eligible_realized_fraction=eligible_idm / eligible if eligible else 0.0,
            executed_chunk_count=executed,
            executed_idm_count=executed_idm,
            executed_realized_fraction=executed_idm / executed if executed else 0.0,
            forced_chunk_count=forced,
            status="COMPLETE" if len(episodes) == ledger_count else "PARTIAL",
            artifact_path=str(root),
            gate_score_quantiles=quantiles,
            route_lag1_autocorrelation=route_autocorrelation,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutingCalibrationResult:
    """Success-blind selection from all completed observed candidates."""

    checkpoint_name: str
    checkpoint_path: str
    arm_name: str
    parameter_name: str
    target_fraction: float
    selected_parameter: RoutingParameter
    calibration_realized_fraction: float
    absolute_rate_error: float
    status: str
    tolerance: float
    success_used_for_selection: bool
    candidate_results: tuple[FastWAMEvalResult, ...]
    selected_gate_score_quantiles: Mapping[str, float] | None = None
    selected_route_lag1_autocorrelation: float | None = None
    autocorrelation_error: float | None = None

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATION_RESULT_SCHEMA,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_path": self.checkpoint_path,
            "arm_name": self.arm_name,
            "parameter_name": self.parameter_name,
            "target_fraction": self.target_fraction,
            "selected_parameter": self.selected_parameter,
            "calibration_realized_fraction": self.calibration_realized_fraction,
            "absolute_rate_error": self.absolute_rate_error,
            "status": self.status,
            "tolerance": self.tolerance,
            "success_used_for_selection": self.success_used_for_selection,
            "candidate_count": len(self.candidate_results),
            "selected_gate_score_quantiles": self.selected_gate_score_quantiles,
            "selected_route_lag1_autocorrelation": (
                self.selected_route_lag1_autocorrelation
            ),
            "autocorrelation_error": self.autocorrelation_error,
        }


class RoutingBudgetCalibrator(Protocol):
    """Protocol for rate-only closed-loop routing parameter selection."""

    def calibrate(
        self,
        *,
        target_fraction: float,
        arm: FastWAMEvalRoutingArm,
        backend: RoutingEvaluationBackend,
        base_request: FastWAMEvalRequest,
    ) -> RoutingCalibrationResult: ...


class ClosedLoopCoarseToFineCalibrator:
    """Search observed closed-loop candidates without monotonicity assumptions."""

    def __init__(
        self,
        *,
        coarse_candidates: Sequence[RoutingParameter],
        refine_top_k: int = 2,
        refine_points_per_interval: int = 4,
        maximum_candidate_runs: int = 25,
        rate_tolerance: float = 0.03,
        rate_scope: str = "eligible_gate_decisions",
        budget_semantics: str = "target_rate",
        candidate_results_path: str | Path,
    ) -> None:
        if not coarse_candidates:
            raise ValueError("Coarse calibration candidates cannot be empty.")
        if refine_top_k < 1 or refine_points_per_interval < 0:
            raise ValueError("Coarse-to-fine refinement settings are invalid.")
        if maximum_candidate_runs < 1:
            raise ValueError("maximum_candidate_runs must be positive.")
        self.coarse_candidates = tuple(coarse_candidates)
        self.refine_top_k = int(refine_top_k)
        self.refine_points_per_interval = int(refine_points_per_interval)
        self.maximum_candidate_runs = int(maximum_candidate_runs)
        self.rate_tolerance = _finite_fraction(rate_tolerance, name="rate tolerance")
        self.rate_scope = str(rate_scope)
        self.budget_semantics = str(budget_semantics)
        if self.budget_semantics not in {"target_rate", "upper_bound"}:
            raise ValueError("Unsupported calibration budget semantics.")
        self.candidate_results_path = Path(candidate_results_path)

    @staticmethod
    def _candidate_key(request: FastWAMEvalRequest) -> tuple[Any, ...]:
        return (
            request.checkpoint_path,
            request.checkpoint_name,
            request.ledger_path,
            request.arm_name,
            json.dumps(request.parameter, sort_keys=True),
            request.routing_seed,
            request.hydra_overrides,
        )

    def _completed_results(self) -> dict[tuple[Any, ...], FastWAMEvalResult]:
        completed = {}
        for record in _read_jsonl(self.candidate_results_path):
            result = _result_from_artifact(record)
            if result.status == "COMPLETE":
                completed[self._candidate_key(result.request)] = result
        return completed

    def _run_candidate(
        self,
        *,
        candidate: RoutingParameter,
        order: int,
        arm: FastWAMEvalRoutingArm,
        backend: RoutingEvaluationBackend,
        base_request: FastWAMEvalRequest,
        completed: dict[tuple[Any, ...], FastWAMEvalResult],
    ) -> FastWAMEvalResult:
        request_identity = replace(
            base_request,
            parameter=candidate,
            hydra_overrides=(
                *base_request.hydra_overrides,
                *arm.hydra_overrides(
                    parameter=candidate,
                    routing_seed=base_request.routing_seed,
                ),
            ),
        )
        key = self._candidate_key(request_identity)
        if key in completed:
            return completed[key]
        prior_attempts = sum(
            self._candidate_key(_result_from_artifact(record).request) == key
            for record in _read_jsonl(self.candidate_results_path)
        )
        request = replace(
            request_identity,
            output_dir=str(
                Path(base_request.output_dir)
                / base_request.arm_name
                / f"candidate-{order:03d}"
                / f"attempt-{prior_attempts:03d}"
            ),
        )
        result = backend.run(request)
        _append_jsonl(self.candidate_results_path, result.to_artifact())
        if result.status == "COMPLETE":
            completed[key] = result
        return result

    def _selection_key(
        self,
        result: FastWAMEvalResult,
        *,
        target: float,
        order: int,
        arm: FastWAMEvalRoutingArm,
    ) -> tuple[float, int, int]:
        rate = arm.extract_rate(result, rate_scope=self.rate_scope)
        over_budget = (
            int(rate > target) if self.budget_semantics == "upper_bound" else 0
        )
        return abs(rate - target), over_budget, order

    def _refinement_candidates(
        self,
        observed: Sequence[tuple[RoutingParameter, FastWAMEvalResult]],
        *,
        target: float,
        arm: FastWAMEvalRoutingArm,
    ) -> tuple[RoutingParameter, ...]:
        numeric = sorted(
            {
                float(candidate)
                for candidate, _ in observed
                if isinstance(candidate, (int, float))
            }
        )
        if len(numeric) < 2 or self.refine_points_per_interval == 0:
            return ()
        order_by_candidate = {
            candidate: index for index, (candidate, _) in enumerate(observed)
        }
        top = sorted(
            observed,
            key=lambda item: self._selection_key(
                item[1],
                target=target,
                order=order_by_candidate[item[0]],
                arm=arm,
            ),
        )[: self.refine_top_k]
        refinements: list[float] = []
        for candidate, _ in top:
            if not isinstance(candidate, (int, float)):
                continue
            value = float(candidate)
            index = numeric.index(value)
            for neighbor_index in (index - 1, index + 1):
                if not 0 <= neighbor_index < len(numeric):
                    continue
                low, high = sorted((value, numeric[neighbor_index]))
                step = (high - low) / (self.refine_points_per_interval + 1)
                refinements.extend(
                    low + step * point
                    for point in range(1, self.refine_points_per_interval + 1)
                )
        existing = {float(value) for value in numeric}
        return tuple(
            value for value in dict.fromkeys(refinements) if value not in existing
        )

    def calibrate(
        self,
        *,
        target_fraction: float,
        arm: FastWAMEvalRoutingArm,
        backend: RoutingEvaluationBackend,
        base_request: FastWAMEvalRequest,
    ) -> RoutingCalibrationResult:
        target = _finite_fraction(target_fraction, name="target IDM fraction")
        candidates = tuple(arm.parameter_domain({"candidates": self.coarse_candidates}))
        completed = self._completed_results()
        observed: list[tuple[RoutingParameter, FastWAMEvalResult]] = []
        attempted = 0
        incomplete = False
        for order, candidate in enumerate(candidates[: self.maximum_candidate_runs]):
            result = self._run_candidate(
                candidate=candidate,
                order=order,
                arm=arm,
                backend=backend,
                base_request=base_request,
                completed=completed,
            )
            attempted += 1
            if result.status == "COMPLETE":
                observed.append((candidate, result))
            else:
                incomplete = True
        refinements = self._refinement_candidates(observed, target=target, arm=arm)
        for refinement_index, candidate in enumerate(refinements):
            if attempted >= self.maximum_candidate_runs:
                break
            result = self._run_candidate(
                candidate=candidate,
                order=len(candidates) + refinement_index,
                arm=arm,
                backend=backend,
                base_request=base_request,
                completed=completed,
            )
            attempted += 1
            if result.status == "COMPLETE":
                observed.append((candidate, result))
            else:
                incomplete = True
        if incomplete:
            raise RuntimeError(
                "FastWAM calibration has partial candidates; resume before selection."
            )
        if not observed:
            raise RuntimeError("No FastWAM calibration candidate completed.")
        selected_index, (selected_parameter, selected_result) = min(
            enumerate(observed),
            key=lambda item: self._selection_key(
                item[1][1],
                target=target,
                order=item[0],
                arm=arm,
            ),
        )
        del selected_index
        selected_rate = arm.extract_rate(selected_result, rate_scope=self.rate_scope)
        error = abs(selected_rate - target)
        target_autocorrelation = getattr(arm, "lag1_autocorrelation", None)
        autocorrelation_error = None
        if (
            target_autocorrelation is not None
            and selected_result.route_lag1_autocorrelation is not None
        ):
            autocorrelation_error = abs(
                selected_result.route_lag1_autocorrelation
                - float(target_autocorrelation)
            )
        return RoutingCalibrationResult(
            checkpoint_name=base_request.checkpoint_name,
            checkpoint_path=base_request.checkpoint_path,
            arm_name=base_request.arm_name,
            parameter_name=arm.parameter_name,
            target_fraction=target,
            selected_parameter=selected_parameter,
            calibration_realized_fraction=selected_rate,
            absolute_rate_error=error,
            status="PASS" if error <= self.rate_tolerance else "FAIL_RATE_TOLERANCE",
            tolerance=self.rate_tolerance,
            success_used_for_selection=False,
            candidate_results=tuple(result for _, result in observed),
            selected_gate_score_quantiles=selected_result.gate_score_quantiles,
            selected_route_lag1_autocorrelation=(
                selected_result.route_lag1_autocorrelation
            ),
            autocorrelation_error=autocorrelation_error,
        )


def _request_from_artifact(payload: Mapping[str, Any]) -> FastWAMEvalRequest:
    return FastWAMEvalRequest(
        checkpoint_path=str(payload["checkpoint_path"]),
        checkpoint_name=str(payload["checkpoint_name"]),
        ledger_path=str(payload["ledger_path"]),
        arm_name=str(payload["arm_name"]),
        parameter=payload.get("parameter"),
        routing_seed=int(payload["routing_seed"]),
        output_dir=str(payload["output_dir"]),
        hydra_overrides=tuple(payload.get("hydra_overrides", ())),
    )


def _result_from_artifact(payload: Mapping[str, Any]) -> FastWAMEvalResult:
    request = payload.get("request")
    if not isinstance(request, Mapping):
        raise TypeError("FastWAM evaluation result request is malformed.")
    return FastWAMEvalResult(
        request=_request_from_artifact(request),
        episode_count=int(payload["episode_count"]),
        success_count=int(payload["success_count"]),
        eligible_decision_count=int(payload["eligible_decision_count"]),
        eligible_idm_count=int(payload["eligible_idm_count"]),
        eligible_realized_fraction=float(payload["eligible_realized_fraction"]),
        executed_chunk_count=int(payload["executed_chunk_count"]),
        executed_idm_count=int(payload["executed_idm_count"]),
        executed_realized_fraction=float(payload["executed_realized_fraction"]),
        forced_chunk_count=int(payload["forced_chunk_count"]),
        status=str(payload["status"]),
        artifact_path=str(payload["artifact_path"]),
        gate_score_quantiles=payload.get("gate_score_quantiles"),
        route_lag1_autocorrelation=payload.get("route_lag1_autocorrelation"),
    )


class FastWAMRoutingBudgetSuite:
    """Calibrate and test multiple arms/checkpoints with immutable selection."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backend: RoutingEvaluationBackend,
        output_dir: str | Path,
    ) -> None:
        resolved = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError("FastWAM budget evaluation config must be a mapping.")
        self.config = resolved
        self.backend = backend
        self.output_dir = Path(output_dir).expanduser().resolve()

    def run(self) -> dict[str, Any]:
        cfg = self.config
        calibration_cfg = dict(cfg["calibration"])
        test_cfg = dict(cfg["test"])
        calibration_ledger = str(calibration_cfg["reset_ledger_path"])
        test_ledger = str(test_cfg["reset_ledger_path"])
        validate_calibration_test_ledgers(calibration_ledger, test_ledger)
        target = _finite_fraction(cfg["target_idm_fraction"], name="target rate")
        tolerance = _finite_fraction(cfg.get("rate_tolerance", 0.03), name="tolerance")
        rate_scope = str(cfg.get("rate_scope", "eligible_gate_decisions"))
        budget_semantics = str(cfg.get("budget_semantics", "target_rate"))
        checkpoints = list(cfg.get("checkpoints", ()))
        arms_cfg = dict(cfg.get("arms", {}))
        if not checkpoints or not arms_cfg:
            raise ValueError("Budget evaluation requires checkpoints and arms.")
        enabled_arms = {
            name: dict(arm_cfg)
            for name, arm_cfg in arms_cfg.items()
            if bool(arm_cfg.get("enabled", True))
        }
        if "learned" not in enabled_arms:
            raise ValueError("Budget evaluation requires a learned arm.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = self.output_dir / "resolved_config.yaml"
        resolved_config = OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True)
        if resolved_config_path.is_file():
            if resolved_config_path.read_text(encoding="utf-8") != resolved_config:
                raise ValueError(
                    "FastWAM budget evaluation output belongs to a different "
                    "resolved config."
                )
        else:
            resolved_config_path.write_text(resolved_config, encoding="utf-8")
        _write_json(
            self.output_dir / "calibration_manifest.json",
            {
                "schema": "fastwam-routing-calibration-manifest-v1",
                "target_fraction": target,
                "rate_scope": rate_scope,
                "calibration_ledger": calibration_ledger,
                "test_ledger": test_ledger,
                "success_blind_selection": True,
                "checkpoint_count": len(checkpoints),
                "arm_count": len(enabled_arms),
            },
        )
        search = dict(calibration_cfg["search"])
        results: dict[str, dict[str, RoutingCalibrationResult]] = {}
        for checkpoint in checkpoints:
            checkpoint_name = str(checkpoint["name"])
            checkpoint_path = str(checkpoint["path"])
            checkpoint_results = {}
            for arm_name, arm_cfg in enabled_arms.items():
                arm = build_fastwam_eval_routing_arm(arm_cfg)
                calibrator = ClosedLoopCoarseToFineCalibrator(
                    coarse_candidates=arm_cfg.get(
                        "candidates", search["coarse_candidates"]
                    ),
                    refine_top_k=int(search.get("refine_top_k", 2)),
                    refine_points_per_interval=int(
                        search.get("refine_points_per_interval", 4)
                    ),
                    maximum_candidate_runs=int(
                        search.get("maximum_candidate_runs", 25)
                    ),
                    rate_tolerance=tolerance,
                    rate_scope=rate_scope,
                    budget_semantics=budget_semantics,
                    candidate_results_path=(
                        self.output_dir / "candidate_results.jsonl"
                    ),
                )
                checkpoint_results[arm_name] = calibrator.calibrate(
                    target_fraction=target,
                    arm=arm,
                    backend=self.backend,
                    base_request=FastWAMEvalRequest(
                        checkpoint_path=checkpoint_path,
                        checkpoint_name=checkpoint_name,
                        ledger_path=calibration_ledger,
                        arm_name=arm_name,
                        parameter=None,
                        routing_seed=int(
                            arm_cfg.get(
                                "routing_seed",
                                calibration_cfg.get("routing_seed", 0),
                            )
                        ),
                        output_dir=str(
                            self.output_dir / "calibration" / checkpoint_name
                        ),
                    ),
                )
            results[checkpoint_name] = checkpoint_results
        selected_payload = {
            "schema": CALIBRATION_SELECTION_SCHEMA,
            "target_fraction": target,
            "rate_scope": rate_scope,
            "tolerance": tolerance,
            "calibration_ledger": calibration_ledger,
            "success_used_for_selection": False,
            "checkpoints": {
                checkpoint_name: {
                    arm_name: result.to_artifact()
                    for arm_name, result in checkpoint_results.items()
                }
                for checkpoint_name, checkpoint_results in results.items()
            },
        }
        selected_path = self.output_dir / "selected_parameters.json"
        _write_json_immutable(selected_path, selected_payload)
        _write_json(
            self.output_dir / "calibration_report.json",
            {
                "schema": "fastwam-routing-calibration-report-v1",
                "all_rate_matched": all(
                    result.status == "PASS"
                    for checkpoint_results in results.values()
                    for result in checkpoint_results.values()
                ),
                "closest_candidates": selected_payload["checkpoints"],
            },
        )
        allow_nearest = bool(
            calibration_cfg.get("allow_nearest_outside_tolerance", False)
        )
        calibration_passed = all(
            result.status == "PASS"
            for checkpoint_results in results.values()
            for result in checkpoint_results.values()
        )
        if not calibration_passed and not allow_nearest:
            _write_json(
                self.output_dir / "test_manifest.json",
                {
                    "schema": "fastwam-routing-test-manifest-v1",
                    "status": "NOT_RUN_CALIBRATION_FAILED",
                    "test_ledger": test_ledger,
                    "selection_path": str(selected_path),
                },
            )
            (self.output_dir / "test_arm_results.jsonl").touch()
            _write_json(
                self.output_dir / "rate_match_report.json",
                {
                    "schema": "fastwam-routing-rate-match-report-v1",
                    "status": "NOT_RUN_CALIBRATION_FAILED",
                    "target_fraction": target,
                    "tolerance": tolerance,
                    "success_blind_selection": True,
                    "pairwise": [],
                },
            )
            return {
                "status": "FAIL_RATE_TOLERANCE",
                "selected_parameters": selected_payload,
                "test_results": (),
            }

        selected_bytes = selected_path.read_bytes()
        _write_json(
            self.output_dir / "test_manifest.json",
            {
                "schema": "fastwam-routing-test-manifest-v1",
                "test_ledger": test_ledger,
                "selection_path": str(selected_path),
                "selection_frozen": True,
            },
        )
        test_results: dict[str, dict[str, FastWAMEvalResult]] = {}
        test_results_path = self.output_dir / "test_arm_results.jsonl"
        for checkpoint in checkpoints:
            checkpoint_name = str(checkpoint["name"])
            checkpoint_path = str(checkpoint["path"])
            checkpoint_results = {}
            for arm_name, arm_cfg in enabled_arms.items():
                selected = results[checkpoint_name][arm_name]
                arm = build_fastwam_eval_routing_arm(arm_cfg)
                request = FastWAMEvalRequest(
                    checkpoint_path=checkpoint_path,
                    checkpoint_name=checkpoint_name,
                    ledger_path=test_ledger,
                    arm_name=arm_name,
                    parameter=selected.selected_parameter,
                    routing_seed=int(
                        arm_cfg.get(
                            "test_routing_seed",
                            arm_cfg.get(
                                "routing_seed", test_cfg.get("routing_seed", 0)
                            ),
                        )
                    ),
                    output_dir=str(
                        self.output_dir / "test" / checkpoint_name / arm_name
                    ),
                    hydra_overrides=arm.hydra_overrides(
                        parameter=selected.selected_parameter,
                        routing_seed=int(
                            arm_cfg.get(
                                "test_routing_seed",
                                arm_cfg.get(
                                    "routing_seed", test_cfg.get("routing_seed", 0)
                                ),
                            )
                        ),
                    ),
                )
                test_result = self.backend.run(request)
                _append_jsonl(test_results_path, test_result.to_artifact())
                checkpoint_results[arm_name] = test_result
            test_results[checkpoint_name] = checkpoint_results
        if selected_path.read_bytes() != selected_bytes:
            raise RuntimeError("Test evaluation modified the calibration selection.")
        all_tests_complete = all(
            result.status == "COMPLETE"
            for checkpoint_results in test_results.values()
            for result in checkpoint_results.values()
        )
        pairwise = []
        for checkpoint_name, checkpoint_results in test_results.items():
            learned = checkpoint_results["learned"]
            for arm_name, baseline in checkpoint_results.items():
                if arm_name == "learned":
                    continue
                learned_rate = float(learned.eligible_realized_fraction)
                baseline_rate = float(baseline.eligible_realized_fraction)
                mismatch = abs(learned_rate - baseline_rate)
                pair_complete = (
                    learned.status == "COMPLETE" and baseline.status == "COMPLETE"
                )
                rate_matched = pair_complete and mismatch <= tolerance
                pairwise.append(
                    {
                        "checkpoint": checkpoint_name,
                        "baseline": arm_name,
                        "learned_test_fraction": learned_rate,
                        "baseline_test_fraction": baseline_rate,
                        "absolute_rate_mismatch": mismatch,
                        "rate_matched": rate_matched,
                        "status": (
                            "RATE_MATCHED" if rate_matched else "NOT_RATE_MATCHED"
                        ),
                        "learned_success": (
                            learned.success_count / learned.episode_count
                            if learned.episode_count
                            else 0.0
                        ),
                        "baseline_success": (
                            baseline.success_count / baseline.episode_count
                            if baseline.episode_count
                            else 0.0
                        ),
                        "learned_episode_count": learned.episode_count,
                        "baseline_episode_count": baseline.episode_count,
                        "learned_gate_score_quantiles": learned.gate_score_quantiles,
                        "baseline_gate_score_quantiles": baseline.gate_score_quantiles,
                    }
                )
        report = {
            "schema": "fastwam-routing-rate-match-report-v1",
            "target_fraction": target,
            "tolerance": tolerance,
            "success_blind_selection": True,
            "calibration_passed": calibration_passed,
            "all_tests_complete": all_tests_complete,
            "pairwise": pairwise,
        }
        _write_json(self.output_dir / "rate_match_report.json", report)
        _write_json(
            self.output_dir / "test_manifest.json",
            {
                "schema": "fastwam-routing-test-manifest-v1",
                "status": "COMPLETE" if all_tests_complete else "PARTIAL",
                "test_ledger": test_ledger,
                "selection_path": str(selected_path),
                "selection_frozen": True,
            },
        )
        all_pairwise_matched = bool(pairwise) and all(
            item["rate_matched"] for item in pairwise
        )
        return {
            "status": (
                "RATE_MATCHED"
                if calibration_passed and all_pairwise_matched
                else "NOT_RATE_MATCHED"
            ),
            "selected_parameters": selected_payload,
            "test_results": test_results,
            "rate_match_report": report,
        }


def run_fastwam_routing_budget_suite(
    cfg: Any,
    *,
    backend: RoutingEvaluationBackend | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a Hydra profile and execute its calibration/test state machine."""

    validate_fastwam_budget_evaluation_config(cfg)
    value = OmegaConf.select(cfg, "evaluation.budget_matching")
    resolved = OmegaConf.to_container(value, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("FastWAM budget evaluation config must be a mapping.")
    if backend is None:
        if repo_root is None:
            raise ValueError("Existing Hydra backend requires the RLinf repo root.")
        backend_config = dict(resolved["backend"])
        backend = ExistingHydraEvaluationBackend(
            repo_root=repo_root,
            config_name=str(backend_config["config_name"]),
        )
    return FastWAMRoutingBudgetSuite(
        resolved,
        backend=backend,
        output_dir=str(resolved["output_dir"]),
    ).run()


__all__ = [
    "ClosedLoopCoarseToFineCalibrator",
    "ExistingHydraEvaluationBackend",
    "FakeRoutingEvaluationBackend",
    "FastWAMEvalRequest",
    "FastWAMEvalResult",
    "FastWAMRoutingBudgetSuite",
    "RoutingBudgetCalibrator",
    "RoutingCalibrationResult",
    "RoutingEvaluationBackend",
    "run_fastwam_routing_budget_suite",
    "validate_calibration_test_ledgers",
    "validate_fastwam_budget_evaluation_config",
]
