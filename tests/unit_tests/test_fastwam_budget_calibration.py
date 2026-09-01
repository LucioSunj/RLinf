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

"""Pure-CPU tests for closed-loop FastWAM routing budget calibration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.runners.fastwam_budget_calibration import (
    ClosedLoopCoarseToFineCalibrator,
    ExistingHydraEvaluationBackend,
    FakeRoutingEvaluationBackend,
    FastWAMEvalRequest,
    FastWAMRoutingBudgetSuite,
    run_fastwam_routing_budget_suite,
    validate_calibration_test_ledgers,
    validate_fastwam_budget_evaluation_config,
)
from rlinf.runners.fastwam_routing_arms import (
    LearnedThresholdArmAdapter,
    build_fastwam_eval_routing_arm,
)


def _write_ledger(path: Path, reset_ids: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "fastwam-libero-eval-ledger-v1",
                "entries": [
                    {
                        "episode_identity": f"episode-{reset_id}",
                        "reset_state_id": reset_id,
                    }
                    for reset_id in reset_ids
                ],
            }
        ),
        encoding="utf-8",
    )


def _base_request(tmp_path: Path, *, checkpoint_name: str = "step0"):
    return FastWAMEvalRequest(
        checkpoint_path=f"/{checkpoint_name}",
        checkpoint_name=checkpoint_name,
        ledger_path="/calibration.json",
        arm_name="learned",
        parameter=None,
        routing_seed=1,
        output_dir=str(tmp_path / checkpoint_name),
    )


def _calibrator(tmp_path: Path, candidates, **kwargs):
    return ClosedLoopCoarseToFineCalibrator(
        coarse_candidates=candidates,
        refine_top_k=kwargs.get("refine_top_k", 2),
        refine_points_per_interval=kwargs.get("refine_points_per_interval", 2),
        maximum_candidate_runs=kwargs.get("maximum_candidate_runs", 20),
        rate_tolerance=kwargs.get("rate_tolerance", 0.03),
        rate_scope="eligible_gate_decisions",
        budget_semantics=kwargs.get("budget_semantics", "target_rate"),
        candidate_results_path=tmp_path / "candidate_results.jsonl",
    )


def test_coarse_to_fine_handles_monotonic_and_nonmonotonic_closed_loop_rates(
    tmp_path: Path,
) -> None:
    arm = LearnedThresholdArmAdapter({})
    monotonic = FakeRoutingEvaluationBackend(
        lambda request: 1.0 - float(request.parameter)
    )
    result = _calibrator(
        tmp_path / "monotonic",
        (0.0, 0.4, 0.8, 1.0),
        refine_points_per_interval=3,
    ).calibrate(
        target_fraction=0.5,
        arm=arm,
        backend=monotonic,
        base_request=_base_request(tmp_path / "monotonic"),
    )
    assert result.selected_parameter == pytest.approx(0.5)
    assert result.status == "PASS"

    observed_rates = {0.0: 0.1, 0.25: 0.49, 0.5: 0.9, 0.75: 0.51, 1.0: 0.2}
    nonmonotonic = FakeRoutingEvaluationBackend(
        lambda request: observed_rates[float(request.parameter)]
    )
    result = _calibrator(
        tmp_path / "nonmonotonic",
        tuple(observed_rates),
        refine_points_per_interval=0,
    ).calibrate(
        target_fraction=0.5,
        arm=arm,
        backend=nonmonotonic,
        base_request=_base_request(tmp_path / "nonmonotonic"),
    )
    assert result.selected_parameter == 0.25
    assert result.calibration_realized_fraction == pytest.approx(0.49)


def test_selection_tie_break_is_candidate_order_and_never_success(
    tmp_path: Path,
) -> None:
    backend = FakeRoutingEvaluationBackend(
        lambda request: 0.4 if request.parameter == 0.2 else 0.6,
        success_function=lambda request: 0.0 if request.parameter == 0.2 else 1.0,
    )
    result = _calibrator(
        tmp_path,
        (0.2, 0.8),
        refine_points_per_interval=0,
    ).calibrate(
        target_fraction=0.5,
        arm=LearnedThresholdArmAdapter({}),
        backend=backend,
        base_request=_base_request(tmp_path),
    )
    assert result.selected_parameter == 0.2
    assert not result.success_used_for_selection


def test_candidate_cap_and_completed_candidate_resume(tmp_path: Path) -> None:
    arm = LearnedThresholdArmAdapter({})
    calibrator = _calibrator(
        tmp_path,
        (0.0, 0.25, 0.5, 0.75, 1.0),
        refine_points_per_interval=4,
        maximum_candidate_runs=3,
    )
    first_backend = FakeRoutingEvaluationBackend(
        lambda request: 1.0 - float(request.parameter)
    )
    calibrator.calibrate(
        target_fraction=0.5,
        arm=arm,
        backend=first_backend,
        base_request=_base_request(tmp_path),
    )
    assert len(first_backend.calls) == 3

    resumed_backend = FakeRoutingEvaluationBackend(
        lambda request: 1.0 - float(request.parameter)
    )
    calibrator.calibrate(
        target_fraction=0.5,
        arm=arm,
        backend=resumed_backend,
        base_request=_base_request(tmp_path),
    )
    assert resumed_backend.calls == []


def test_partial_candidate_is_rerun_before_selection(tmp_path: Path) -> None:
    delegate = FakeRoutingEvaluationBackend(
        lambda request: 1.0 - float(request.parameter)
    )

    class PartialThenComplete:
        def __init__(self) -> None:
            self.partial_emitted = False
            self.calls = 0

        def run(self, request):
            self.calls += 1
            result = delegate.run(request)
            if not self.partial_emitted:
                self.partial_emitted = True
                return replace(result, status="PARTIAL")
            return result

    backend = PartialThenComplete()
    calibrator = _calibrator(
        tmp_path,
        (0.0, 0.5, 1.0),
        refine_points_per_interval=0,
    )
    with pytest.raises(RuntimeError, match="partial candidates"):
        calibrator.calibrate(
            target_fraction=0.5,
            arm=LearnedThresholdArmAdapter({}),
            backend=backend,
            base_request=_base_request(tmp_path),
        )
    assert backend.calls == 3
    assert not (tmp_path / "selected_parameters.json").exists()
    result = calibrator.calibrate(
        target_fraction=0.5,
        arm=LearnedThresholdArmAdapter({}),
        backend=backend,
        base_request=_base_request(tmp_path),
    )
    assert backend.calls == 4
    assert delegate.calls[0].output_dir.endswith("attempt-000")
    assert delegate.calls[3].output_dir.endswith("attempt-001")
    assert result.status == "PASS"


def test_calibration_test_ledger_paths_and_reset_ids_must_not_leak(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration.json"
    test = tmp_path / "test.json"
    _write_ledger(calibration, [0, 1])
    _write_ledger(test, [2, 3])
    validate_calibration_test_ledgers(calibration, test)
    with pytest.raises(ValueError, match="paths must differ"):
        validate_calibration_test_ledgers(calibration, calibration)
    _write_ledger(test, [1, 2])
    with pytest.raises(ValueError, match="must not overlap"):
        validate_calibration_test_ledgers(calibration, test)


@pytest.mark.parametrize(
    ("config", "parameter", "expected_mode"),
    (
        ({"type": "learned_threshold"}, 0.4, "learned_threshold"),
        ({"type": "independent_random"}, 0.4, "matched_random"),
        (
            {
                "type": "autocorrelation_matched_random",
                "lag1_autocorrelation": -0.2,
            },
            0.4,
            "autocorrelation_matched_random",
        ),
        ({"type": "periodic"}, "4:2:1", "periodic"),
    ),
)
def test_routing_arm_adapters_only_emit_existing_evaluator_overrides(
    config: dict,
    parameter,
    expected_mode: str,
) -> None:
    arm = build_fastwam_eval_routing_arm(config)
    overrides = arm.hydra_overrides(parameter=parameter, routing_seed=17)
    assert f"rollout.model.eval_routing_mode={expected_mode}" in overrides
    assert "rollout.model.eval_routing_seed=17" in overrides


def test_existing_hydra_backend_invokes_existing_eval_and_aggregates_shards(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [0])
    output = tmp_path / "output"
    calls = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        output.mkdir(parents=True, exist_ok=True)
        (output / "episodes.rank-0.jsonl").write_text(
            json.dumps(
                {
                    "success": True,
                    "eligible_chunk_count": 2,
                    "eligible_idm_count": 1,
                    "executed_chunk_count": 3,
                    "idm_chunk_count_total": 2,
                    "forced_initial_idm_count": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "chunks.rank-0.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "episode_identity": "episode-0",
                        "chunk_id": index,
                        "route": route,
                        "route_was_forced": index == 0,
                        "gate_idm_probability": probability,
                    }
                )
                for index, (route, probability) in enumerate(
                    (("idm", 0.8), ("idm", 0.6), ("uncond", 0.2))
                )
            )
            + "\n",
            encoding="utf-8",
        )

    request = FastWAMEvalRequest(
        checkpoint_path="/checkpoint",
        checkpoint_name="step0",
        ledger_path=str(ledger),
        arm_name="learned",
        parameter=0.5,
        routing_seed=1,
        output_dir=str(output),
        hydra_overrides=("rollout.model.eval_idm_threshold=0.5",),
    )
    result = ExistingHydraEvaluationBackend(
        repo_root=tmp_path,
        command_runner=command_runner,
    ).run(request)
    assert calls[0][0][:2] == ["bash", str(tmp_path / "evaluations/run_eval.sh")]
    assert result.status == "COMPLETE"
    assert result.eligible_realized_fraction == 0.5
    assert result.executed_realized_fraction == pytest.approx(2 / 3)
    assert result.gate_score_quantiles["p50"] == 0.6


def _suite_config(calibration_ledger: Path, test_ledger: Path) -> dict:
    return {
        "enabled": True,
        "protocol": "closed_loop_calibrated_threshold",
        "output_dir": str(calibration_ledger.parent / "suite-output"),
        "backend": {
            "type": "existing_hydra",
            "config_name": "libero_10_fastwam_adaptive_eval",
        },
        "budget_semantics": "target_rate",
        "target_idm_fraction": 0.5,
        "rate_tolerance": 0.03,
        "rate_scope": "eligible_gate_decisions",
        "checkpoints": [
            {"name": "step0", "path": "/step0"},
            {"name": "step5", "path": "/step5"},
            {"name": "step10", "path": "/step10"},
        ],
        "calibration": {
            "reset_ledger_path": str(calibration_ledger),
            "routing_seed": 1,
            "allow_nearest_outside_tolerance": False,
            "search": {
                "coarse_candidates": [0.0, 0.25, 0.5, 0.75, 1.0],
                "refine_top_k": 1,
                "refine_points_per_interval": 0,
                "maximum_candidate_runs": 5,
            },
            "selection": {
                "metric": "eligible_realized_fraction",
                "success_blind": True,
            },
        },
        "test": {
            "reset_ledger_path": str(test_ledger),
            "routing_seed": 2,
            "run_only_after_calibration_pass": True,
        },
        "arms": {
            "learned": {"type": "learned_threshold", "enabled": True},
            "independent_random": {
                "type": "independent_random",
                "enabled": True,
                "candidates": [0.25, 0.5, 0.75],
            },
        },
    }


def test_budget_evaluation_config_fails_fast_on_leakage_and_success_selection(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration.json"
    test = tmp_path / "test.json"
    _write_ledger(calibration, [0, 1])
    _write_ledger(test, [2, 3])
    budget = _suite_config(calibration, test)
    cfg = OmegaConf.create(
        {
            "routing_objective": {"type": "eval_calibrated_target"},
            "evaluation": {"budget_matching": budget},
        }
    )
    validate_fastwam_budget_evaluation_config(cfg)

    leaked = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    leaked.evaluation.budget_matching.test.reset_ledger_path = str(calibration)
    with pytest.raises(ValueError, match="paths must differ"):
        validate_fastwam_budget_evaluation_config(leaked)

    success_selected = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    success_selected.evaluation.budget_matching.calibration.selection.success_blind = (
        False
    )
    with pytest.raises(ValueError, match="success-blind"):
        validate_fastwam_budget_evaluation_config(success_selected)


def test_hydra_resolved_suite_factory_runs_with_fake_backend(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    test = tmp_path / "test.json"
    _write_ledger(calibration, [0, 1])
    _write_ledger(test, [2, 3])
    budget = _suite_config(calibration, test)
    budget["checkpoints"] = [{"name": "step0", "path": "/step0"}]
    cfg = OmegaConf.create(
        {
            "routing_objective": {"type": "eval_calibrated_target"},
            "evaluation": {"budget_matching": budget},
        }
    )
    result = run_fastwam_routing_budget_suite(
        cfg,
        backend=FakeRoutingEvaluationBackend(lambda request: 0.5),
    )
    assert result["status"] == "RATE_MATCHED"


@pytest.mark.parametrize(
    ("baseline_test_rate", "expected_matched"),
    ((0.52, True), (0.55, False)),
)
def test_suite_selects_step0_5_10_independently_and_uses_test_actual_rates(
    tmp_path: Path,
    baseline_test_rate: float,
    expected_matched: bool,
) -> None:
    calibration_ledger = tmp_path / "calibration.json"
    test_ledger = tmp_path / "test.json"
    _write_ledger(calibration_ledger, [0, 1, 2])
    _write_ledger(test_ledger, [10, 11, 12])
    shifts = {"step0": -0.25, "step5": 0.0, "step10": 0.25}

    def rate(request: FastWAMEvalRequest) -> float:
        if request.ledger_path == str(test_ledger):
            return 0.5 if request.arm_name == "learned" else baseline_test_rate
        if request.arm_name == "learned":
            return max(
                0.0,
                min(
                    1.0,
                    1.0 - float(request.parameter) + shifts[request.checkpoint_name],
                ),
            )
        return float(request.parameter)

    backend = FakeRoutingEvaluationBackend(
        rate,
        success_function=lambda request: 1.0 - float(request.parameter),
    )
    output = tmp_path / "suite"
    result = FastWAMRoutingBudgetSuite(
        _suite_config(calibration_ledger, test_ledger),
        backend=backend,
        output_dir=output,
    ).run()
    selected = result["selected_parameters"]["checkpoints"]
    assert selected["step0"]["learned"]["selected_parameter"] == 0.25
    assert selected["step5"]["learned"]["selected_parameter"] == 0.5
    assert selected["step10"]["learned"]["selected_parameter"] == 0.75
    assert (
        result["rate_match_report"]["pairwise"][0]["rate_matched"] is expected_matched
    )
    assert result["rate_match_report"]["pairwise"][0]["status"] == (
        "RATE_MATCHED" if expected_matched else "NOT_RATE_MATCHED"
    )
    assert result["status"] == (
        "RATE_MATCHED" if expected_matched else "NOT_RATE_MATCHED"
    )
    assert (
        json.loads((output / "selected_parameters.json").read_text())[
            "success_used_for_selection"
        ]
        is False
    )
    for name in (
        "calibration_manifest.json",
        "candidate_results.jsonl",
        "selected_parameters.json",
        "calibration_report.json",
        "test_manifest.json",
        "test_arm_results.jsonl",
        "rate_match_report.json",
        "resolved_config.yaml",
    ):
        assert (output / name).is_file()
    assert selected["step0"]["learned"]["selected_gate_score_quantiles"] == {
        "p10": 0.1,
        "p50": 0.5,
        "p90": 0.9,
    }

    changed = _suite_config(calibration_ledger, test_ledger)
    changed["checkpoints"][0]["path"] = "/different-step0"
    with pytest.raises(ValueError, match="different resolved config"):
        FastWAMRoutingBudgetSuite(
            changed,
            backend=backend,
            output_dir=output,
        ).run()


def test_calibration_failure_saves_closest_and_does_not_enter_test(
    tmp_path: Path,
) -> None:
    calibration_ledger = tmp_path / "calibration.json"
    test_ledger = tmp_path / "test.json"
    _write_ledger(calibration_ledger, [0, 1])
    _write_ledger(test_ledger, [10, 11])
    config = _suite_config(calibration_ledger, test_ledger)
    config["checkpoints"] = [{"name": "step0", "path": "/step0"}]
    backend = FakeRoutingEvaluationBackend(lambda request: 0.0)
    output = tmp_path / "failed-suite"
    result = FastWAMRoutingBudgetSuite(
        config,
        backend=backend,
        output_dir=output,
    ).run()
    assert result["status"] == "FAIL_RATE_TOLERANCE"
    assert result["test_results"] == ()
    assert json.loads((output / "test_manifest.json").read_text())["status"] == (
        "NOT_RUN_CALIBRATION_FAILED"
    )
    assert json.loads((output / "rate_match_report.json").read_text())["pairwise"] == []
    assert (output / "test_arm_results.jsonl").read_text() == ""
