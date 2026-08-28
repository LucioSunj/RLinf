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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_wam_policy_modules():
    repo = Path(__file__).resolve().parents[2]
    package_name = "fastwam_evaluation_routing_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(repo / "rlinf/models/embodiment/wam_policy")]
    sys.modules[package_name] = package
    for name in ("contracts", "routing_state", "evaluation"):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name,
            repo / f"rlinf/models/embodiment/wam_policy/{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return (
        sys.modules[f"{package_name}.contracts"],
        sys.modules[f"{package_name}.routing_state"],
        sys.modules[f"{package_name}.evaluation"],
    )


_contracts, _routing_state, _evaluation = _load_wam_policy_modules()
EvaluationRouteSelection = _evaluation.EvaluationRouteSelection
EvaluationRoutingConfig = _evaluation.EvaluationRoutingConfig
EvaluationRoutingMode = _evaluation.EvaluationRoutingMode
autocorrelated_transition_probabilities = (
    _evaluation.autocorrelated_transition_probabilities
)
PendingRouteTracker = _routing_state.PendingRouteTracker
WAMRoute = _contracts.WAMRoute
select_evaluation_routes = _evaluation.select_evaluation_routes


def _inputs():
    return {
        "gate_idm_probabilities": torch.tensor([0.2, 0.5, 0.8]),
        "env_ids": torch.tensor([12, 3, 91]),
        "episode_ids": torch.tensor([7, 8, 9]),
        "source_chunk_ids": torch.tensor([0, 4, 13]),
    }


def test_routing_modes_are_exact_and_defaults_are_learned_threshold() -> None:
    assert [mode.value for mode in EvaluationRoutingMode] == [
        "learned_threshold",
        "forced_idm",
        "forced_uncond",
        "matched_random",
        "autocorrelation_matched_random",
    ]
    config = EvaluationRoutingConfig()
    assert config.mode is EvaluationRoutingMode.LEARNED_THRESHOLD
    assert config.idm_threshold == 0.5
    assert config.random_idm_probability is None
    assert config.random_lag1_autocorrelation is None
    assert config.routing_seed == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "threshold_zero_endpoint"}, "eval_routing_mode"),
        ({"idm_threshold": -0.01}, "eval_idm_threshold"),
        ({"idm_threshold": 1.01}, "eval_idm_threshold"),
        ({"idm_threshold": float("nan")}, "eval_idm_threshold"),
        ({"idm_threshold": float("inf")}, "eval_idm_threshold"),
        ({"random_idm_probability": -0.01}, "eval_random_idm_probability"),
        ({"random_idm_probability": 1.01}, "eval_random_idm_probability"),
        (
            {"random_idm_probability": float("nan")},
            "eval_random_idm_probability",
        ),
        (
            {"mode": "matched_random", "random_idm_probability": None},
            "requires eval_random_idm_probability",
        ),
        (
            {
                "mode": "autocorrelation_matched_random",
                "random_idm_probability": 0.52,
            },
            "requires eval_random_lag1_autocorrelation",
        ),
        (
            {"random_lag1_autocorrelation": -0.2},
            "only valid for autocorrelation_matched_random",
        ),
        (
            {
                "mode": "autocorrelation_matched_random",
                "random_idm_probability": 0.25,
                "random_lag1_autocorrelation": -1.0,
            },
            "invalid transition probabilities",
        ),
        (
            {
                "mode": "autocorrelation_matched_random",
                "random_idm_probability": 0.52,
                "random_lag1_autocorrelation": float("nan"),
            },
            "eval_random_lag1_autocorrelation",
        ),
        ({"routing_seed": -1}, "eval_routing_seed"),
        ({"routing_seed": 0.5}, "eval_routing_seed"),
    ],
)
def test_routing_config_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        EvaluationRoutingConfig(**kwargs)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_routing_config_accepts_probability_boundaries(probability) -> None:
    config = EvaluationRoutingConfig(
        mode="matched_random",
        idm_threshold=probability,
        random_idm_probability=probability,
    )
    assert config.idm_threshold == probability
    assert config.random_idm_probability == probability


def test_autocorrelated_transition_formula_matches_preregistered_self_check() -> None:
    after_idm, after_uncond = autocorrelated_transition_probabilities(0.52, -0.20)

    assert after_idm == pytest.approx(0.424)
    assert after_uncond == pytest.approx(0.624)


def test_autocorrelation_matched_random_uses_current_route_conditionals() -> None:
    sample_count = 20_000
    half = sample_count // 2
    inputs = {
        "gate_idm_probabilities": torch.full((sample_count,), 0.5),
        "env_ids": torch.arange(sample_count),
        "episode_ids": torch.zeros(sample_count, dtype=torch.long),
        "source_chunk_ids": torch.zeros(sample_count, dtype=torch.long),
        "current_routes": torch.cat(
            (
                torch.full((half,), int(WAMRoute.IDM)),
                torch.full((half,), int(WAMRoute.UNCOND)),
            )
        ),
    }
    selection = select_evaluation_routes(
        EvaluationRoutingConfig(
            mode="autocorrelation_matched_random",
            random_idm_probability=0.52,
            random_lag1_autocorrelation=-0.20,
            routing_seed=17,
        ),
        **inputs,
    )

    measured_after_idm = selection.effective_next_route[:half].float().mean().item()
    measured_after_uncond = selection.effective_next_route[half:].float().mean().item()
    assert measured_after_idm == pytest.approx(0.424, abs=0.015)
    assert measured_after_uncond == pytest.approx(0.624, abs=0.015)
    assert selection.random_draws is not None


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("learned_threshold", [0, 1, 1]),
        ("forced_idm", [1, 1, 1]),
        ("forced_uncond", [0, 0, 0]),
    ],
)
def test_explicit_nonrandom_modes_preserve_counterfactual_gate_route(
    mode,
    expected,
) -> None:
    selection = select_evaluation_routes(
        EvaluationRoutingConfig(mode=mode),
        **_inputs(),
    )

    assert selection.mode.value == mode
    assert selection.effective_next_route.tolist() == expected
    assert selection.counterfactual_next_route.tolist() == [0, 1, 1]
    assert selection.random_draws is None
    assert selection.effective_next_route.dtype == torch.long


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, [0, 0, 0]), (1.0, [1, 1, 1])],
)
def test_matched_random_obeys_probability_boundaries(probability, expected) -> None:
    selection = select_evaluation_routes(
        EvaluationRoutingConfig(
            mode="matched_random",
            random_idm_probability=probability,
            routing_seed=17,
        ),
        **_inputs(),
    )

    assert selection.effective_next_route.tolist() == expected
    assert selection.counterfactual_next_route.tolist() == [0, 1, 1]
    assert selection.random_draws is not None
    assert bool((selection.random_draws >= 0).all())
    assert bool((selection.random_draws < 1).all())


def test_matched_random_is_stateless_and_batch_order_invariant() -> None:
    config = EvaluationRoutingConfig(
        mode="matched_random",
        random_idm_probability=0.47,
        routing_seed=20260801,
    )
    inputs = _inputs()
    baseline = select_evaluation_routes(config, **inputs)
    repeated = select_evaluation_routes(config, **inputs)
    permutation = torch.tensor([2, 0, 1])
    permuted = select_evaluation_routes(
        config,
        **{name: value[permutation] for name, value in inputs.items()},
    )
    inverse = torch.argsort(permutation)

    assert torch.equal(baseline.effective_next_route, repeated.effective_next_route)
    assert torch.equal(baseline.random_draws, repeated.random_draws)
    assert torch.equal(
        baseline.effective_next_route,
        permuted.effective_next_route[inverse],
    )
    assert torch.equal(baseline.random_draws, permuted.random_draws[inverse])

    for index in range(3):
        single = select_evaluation_routes(
            config,
            **{name: value[index : index + 1] for name, value in inputs.items()},
        )
        assert (
            single.effective_next_route.item()
            == baseline.effective_next_route[index].item()
        )
        assert single.random_draws.item() == baseline.random_draws[index].item()


def test_matched_random_key_uses_every_declared_identity_field() -> None:
    config = EvaluationRoutingConfig(
        mode="matched_random",
        random_idm_probability=0.5,
        routing_seed=5,
    )
    inputs = {
        "gate_idm_probabilities": torch.tensor([0.5]),
        "env_ids": torch.tensor([11]),
        "episode_ids": torch.tensor([13]),
        "source_chunk_ids": torch.tensor([17]),
    }
    baseline = select_evaluation_routes(config, **inputs).random_draws.item()
    changed_draws = []
    for field in ("env_ids", "episode_ids", "source_chunk_ids"):
        changed = dict(inputs)
        changed[field] = changed[field] + 1
        changed_draws.append(
            select_evaluation_routes(config, **changed).random_draws.item()
        )
    changed_seed = select_evaluation_routes(
        EvaluationRoutingConfig(
            mode="matched_random",
            random_idm_probability=0.5,
            routing_seed=6,
        ),
        **inputs,
    ).random_draws.item()

    assert all(draw != baseline for draw in [*changed_draws, changed_seed])


def test_route_selection_record_round_trips_cpu_cat_and_chunks() -> None:
    selection = select_evaluation_routes(
        EvaluationRoutingConfig(
            mode="matched_random",
            random_idm_probability=0.5,
        ),
        **_inputs(),
    )
    pieces = selection.chunk(2)
    restored = EvaluationRouteSelection.cat(pieces)

    assert restored.mode is EvaluationRoutingMode.MATCHED_RANDOM
    assert torch.equal(restored.effective_next_route, selection.effective_next_route)
    assert torch.equal(
        restored.counterfactual_next_route,
        selection.counterfactual_next_route,
    )
    assert torch.equal(restored.random_draws, selection.random_draws)
    assert restored.cpu().effective_next_route.device.type == "cpu"


@pytest.mark.parametrize(
    "mode",
    [
        "learned_threshold",
        "forced_idm",
        "forced_uncond",
        "matched_random",
        "autocorrelation_matched_random",
    ],
)
def test_pending_tracker_forces_first_chunk_before_selected_route(mode) -> None:
    config = EvaluationRoutingConfig(
        mode=mode,
        random_idm_probability=(
            0.5
            if mode == "autocorrelation_matched_random"
            else 1.0
            if mode == "matched_random"
            else None
        ),
        random_lag1_autocorrelation=(
            -0.2 if mode == "autocorrelation_matched_random" else None
        ),
    )
    tracker = PendingRouteTracker()
    env_ids = torch.tensor([5, 9])
    first = tracker.consume(
        env_ids=env_ids,
        reset_mask=torch.tensor([True, True]),
        actor_version=0,
    )
    assert first.route_used.tolist() == [int(WAMRoute.IDM)] * 2
    assert first.route_was_forced.tolist() == [True, True]

    selected = select_evaluation_routes(
        config,
        gate_idm_probabilities=torch.tensor([0.25, 0.75]),
        env_ids=env_ids,
        episode_ids=first.episode_ids,
        source_chunk_ids=first.chunk_ids,
        current_routes=first.route_used,
    )
    tracker.emit(
        env_ids=env_ids,
        routes=selected.effective_next_route,
        source_chunk_ids=first.chunk_ids,
        episode_ids=first.episode_ids,
        actor_version=0,
    )
    second = tracker.consume(
        env_ids=env_ids,
        reset_mask=torch.tensor([False, False]),
        actor_version=0,
    )

    assert torch.equal(second.route_used, selected.effective_next_route.cpu())
    assert second.route_was_forced.tolist() == [False, False]


@pytest.mark.parametrize(
    "field",
    ["gate_idm_probabilities", "env_ids", "episode_ids", "source_chunk_ids"],
)
def test_selector_rejects_misaligned_or_invalid_inputs(field) -> None:
    inputs = _inputs()
    inputs[field] = inputs[field][:-1]
    with pytest.raises(ValueError, match="shape"):
        select_evaluation_routes(EvaluationRoutingConfig(), **inputs)

    inputs = _inputs()
    inputs["gate_idm_probabilities"] = torch.tensor([0.2, float("nan"), 0.8])
    with pytest.raises(ValueError, match="finite"):
        select_evaluation_routes(EvaluationRoutingConfig(), **inputs)

    inputs = _inputs()
    inputs["source_chunk_ids"] = torch.tensor([0, -1, 2])
    with pytest.raises(ValueError, match="non-negative"):
        select_evaluation_routes(EvaluationRoutingConfig(), **inputs)

    autocorrelated = EvaluationRoutingConfig(
        mode="autocorrelation_matched_random",
        random_idm_probability=0.5,
        random_lag1_autocorrelation=-0.2,
    )
    with pytest.raises(ValueError, match="requires current_routes"):
        select_evaluation_routes(autocorrelated, **_inputs())
    with pytest.raises(ValueError, match="invalid route"):
        select_evaluation_routes(
            autocorrelated,
            **_inputs(),
            current_routes=torch.tensor([0, 1, 2]),
        )


def _only_eval_config(**model_overrides):
    from omegaconf import OmegaConf

    model = {
        "model_type": "fastwam_adaptive",
        "eval_routing_mode": "learned_threshold",
        "eval_idm_threshold": 0.5,
        "eval_random_idm_probability": None,
        "eval_random_lag1_autocorrelation": None,
        "eval_routing_seed": 0,
        "critic": {"load_for_eval": False},
    }
    model.update(model_overrides)
    return OmegaConf.create(
        {
            "rollout": {"model": model},
            "env": {},
        }
    )


def test_rlinf_only_eval_config_validates_routing_before_critic_return() -> None:
    from rlinf.config import _validate_fastwam_adaptive_cfg

    config = _only_eval_config(
        eval_routing_mode="matched_random",
        eval_random_idm_probability=0.375,
        eval_routing_seed=19,
    )
    _validate_fastwam_adaptive_cfg(config, only_eval=True)
    assert config.rollout.model.eval_without_critic is True

    autocorrelated = _only_eval_config(
        eval_routing_mode="autocorrelation_matched_random",
        eval_random_idm_probability=0.52,
        eval_random_lag1_autocorrelation=-0.2,
    )
    _validate_fastwam_adaptive_cfg(autocorrelated, only_eval=True)

    invalid_cases = [
        (
            {"eval_routing_mode": "threshold_endpoint"},
            "eval_routing_mode",
        ),
        (
            {
                "eval_routing_mode": "matched_random",
                "eval_random_idm_probability": None,
            },
            "requires eval_random_idm_probability",
        ),
        (
            {"eval_random_idm_probability": float("inf")},
            "eval_random_idm_probability",
        ),
        ({"eval_routing_seed": -1}, "eval_routing_seed"),
    ]
    for overrides, message in invalid_cases:
        with pytest.raises((TypeError, ValueError), match=message):
            _validate_fastwam_adaptive_cfg(
                _only_eval_config(**overrides),
                only_eval=True,
            )
