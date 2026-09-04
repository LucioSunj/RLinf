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

"""Pure, reproducible route selection for formal FastWAM evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import torch

from .contracts import WAMRoute


class EvaluationRoutingMode(str, Enum):
    """Supported formal-evaluation route controls."""

    LEARNED_THRESHOLD = "learned_threshold"
    STOCHASTIC_KEYED = "stochastic_keyed"
    FORCED_IDM = "forced_idm"
    FORCED_UNCOND = "forced_uncond"
    MATCHED_RANDOM = "matched_random"
    AUTOCORRELATION_MATCHED_RANDOM = "autocorrelation_matched_random"
    PERIODIC = "periodic"


def _finite_probability(value: object, *, field_name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"`{field_name}` must be a finite probability.") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"`{field_name}` must be finite and lie in [0, 1].")
    return probability


def autocorrelated_transition_probabilities(
    idm_probability: object,
    lag1_autocorrelation: object,
) -> tuple[float, float]:
    """Return P(next IDM | current IDM/UNCOND) for a stationary binary chain."""

    probability = _finite_probability(
        idm_probability,
        field_name="eval_random_idm_probability",
    )
    try:
        autocorrelation = float(lag1_autocorrelation)
    except (TypeError, ValueError) as exc:
        raise TypeError("`eval_random_lag1_autocorrelation` must be finite.") from exc
    if not math.isfinite(autocorrelation):
        raise ValueError("`eval_random_lag1_autocorrelation` must be finite.")

    probability_after_idm = probability + (1.0 - probability) * autocorrelation
    probability_after_uncond = probability - probability * autocorrelation
    if not (
        0.0 <= probability_after_idm <= 1.0 and 0.0 <= probability_after_uncond <= 1.0
    ):
        raise ValueError(
            "`eval_random_lag1_autocorrelation` yields invalid transition "
            "probabilities for eval_random_idm_probability."
        )
    return probability_after_idm, probability_after_uncond


@dataclass(frozen=True, slots=True)
class EvaluationRoutingConfig:
    """Validated evaluation-only routing configuration."""

    mode: EvaluationRoutingMode | str = EvaluationRoutingMode.LEARNED_THRESHOLD
    idm_threshold: float = 0.5
    random_idm_probability: float | None = None
    random_lag1_autocorrelation: float | None = None
    periodic_period: int | None = None
    periodic_on_count: int | None = None
    periodic_phase: int | None = None
    routing_seed: int = 0

    def __post_init__(self) -> None:
        try:
            mode = EvaluationRoutingMode(self.mode)
        except (TypeError, ValueError) as exc:
            supported = ", ".join(item.value for item in EvaluationRoutingMode)
            raise ValueError(
                f"`eval_routing_mode` must be one of {supported}; got {self.mode!r}."
            ) from exc
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "idm_threshold",
            _finite_probability(
                self.idm_threshold,
                field_name="eval_idm_threshold",
            ),
        )

        random_probability = self.random_idm_probability
        if random_probability is not None:
            random_probability = _finite_probability(
                random_probability,
                field_name="eval_random_idm_probability",
            )
            object.__setattr__(
                self,
                "random_idm_probability",
                random_probability,
            )
        random_modes_requiring_fixed_probability = {
            EvaluationRoutingMode.MATCHED_RANDOM,
            EvaluationRoutingMode.AUTOCORRELATION_MATCHED_RANDOM,
        }
        if (
            mode in random_modes_requiring_fixed_probability
            and random_probability is None
        ):
            raise ValueError(f"`{mode.value}` requires eval_random_idm_probability.")

        autocorrelation = self.random_lag1_autocorrelation
        if mode is EvaluationRoutingMode.AUTOCORRELATION_MATCHED_RANDOM:
            if autocorrelation is None:
                raise ValueError(
                    "`autocorrelation_matched_random` requires "
                    "eval_random_lag1_autocorrelation."
                )
            _, _ = autocorrelated_transition_probabilities(
                random_probability,
                autocorrelation,
            )
            object.__setattr__(
                self,
                "random_lag1_autocorrelation",
                float(autocorrelation),
            )
        elif autocorrelation is not None:
            raise ValueError(
                "`eval_random_lag1_autocorrelation` is only valid for "
                "autocorrelation_matched_random."
            )

        periodic_values = (
            self.periodic_period,
            self.periodic_on_count,
            self.periodic_phase,
        )
        if mode is EvaluationRoutingMode.PERIODIC:
            if any(value is None for value in periodic_values):
                raise ValueError(
                    "`periodic` requires eval_period, eval_periodic_on_count, "
                    "and eval_periodic_phase."
                )
            period, on_count, phase = (int(value) for value in periodic_values)
            if period < 1 or not 0 <= on_count <= period or not 0 <= phase < period:
                raise ValueError("Periodic evaluation routing pattern is invalid.")
            object.__setattr__(self, "periodic_period", period)
            object.__setattr__(self, "periodic_on_count", on_count)
            object.__setattr__(self, "periodic_phase", phase)
        elif any(value is not None for value in periodic_values):
            raise ValueError("Periodic routing fields are only valid for `periodic`.")

        if isinstance(self.routing_seed, bool) or not isinstance(
            self.routing_seed,
            Integral,
        ):
            raise TypeError("`eval_routing_seed` must be a non-negative integer.")
        routing_seed = int(self.routing_seed)
        if routing_seed < 0:
            raise ValueError("`eval_routing_seed` must be non-negative.")
        object.__setattr__(self, "routing_seed", routing_seed)


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_route_tensor(name: str, value: torch.Tensor) -> None:
    if value.ndim != 1:
        raise ValueError(f"`{name}` must have shape [B], got {tuple(value.shape)}.")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"`{name}` must use an integer dtype, got {value.dtype}.")
    if bool((value < 0).any().item()):
        raise ValueError(f"`{name}` must contain only non-negative values.")


def _stateless_uniform(
    *,
    routing_seed: int,
    env_id: int,
    episode_id: int,
    source_chunk_id: int,
) -> float:
    """Map one stable routing identity to an exact 53-bit uniform variate."""

    payload = b"\0".join(
        (
            b"fastwam-eval-routing-v1",
            str(routing_seed).encode("ascii"),
            str(env_id).encode("ascii"),
            str(episode_id).encode("ascii"),
            str(source_chunk_id).encode("ascii"),
        )
    )
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 11
    return integer / float(1 << 53)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRouteSelection:
    """Compact typed metadata for one batch of emitted evaluation routes."""

    mode: EvaluationRoutingMode | str
    effective_next_route: torch.Tensor
    counterfactual_next_route: torch.Tensor
    random_draws: torch.Tensor | None = None

    def __post_init__(self) -> None:
        try:
            mode = EvaluationRoutingMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown evaluation routing mode {self.mode!r}.") from exc
        object.__setattr__(self, "mode", mode)

        shape = self.effective_next_route.shape
        if len(shape) != 1:
            raise ValueError(
                f"Evaluation routes must have shape [B], got {tuple(shape)}."
            )
        for name, value in (
            ("effective_next_route", self.effective_next_route),
            ("counterfactual_next_route", self.counterfactual_next_route),
        ):
            if value.shape != shape:
                raise ValueError(f"`{name}` must have shape {tuple(shape)}.")
            if value.dtype not in _INTEGER_DTYPES:
                raise TypeError(f"`{name}` must use an integer dtype.")
            invalid = (value != int(WAMRoute.UNCOND)) & (value != int(WAMRoute.IDM))
            if bool(invalid.any().item()):
                raise ValueError(f"`{name}` contains an invalid route.")

        random_modes = {
            EvaluationRoutingMode.STOCHASTIC_KEYED,
            EvaluationRoutingMode.MATCHED_RANDOM,
            EvaluationRoutingMode.AUTOCORRELATION_MATCHED_RANDOM,
        }
        if mode in random_modes:
            if self.random_draws is None:
                raise ValueError("Random route selection requires random draws.")
            if self.random_draws.shape != shape:
                raise ValueError(f"`random_draws` must have shape {tuple(shape)}.")
            if not self.random_draws.is_floating_point():
                raise TypeError("`random_draws` must use a floating dtype.")
            invalid_draw = (
                ~torch.isfinite(self.random_draws)
                | (self.random_draws < 0)
                | (self.random_draws >= 1)
            )
            if bool(invalid_draw.any().item()):
                raise ValueError("`random_draws` must be finite and lie in [0, 1).")
        elif self.random_draws is not None:
            raise ValueError("Only random route selections may carry random draws.")

    def cpu(self) -> EvaluationRouteSelection:
        """Return a contiguous CPU copy suitable for worker transport."""

        return EvaluationRouteSelection(
            mode=self.mode,
            effective_next_route=self.effective_next_route.cpu().contiguous(),
            counterfactual_next_route=(
                self.counterfactual_next_route.cpu().contiguous()
            ),
            random_draws=(
                self.random_draws.cpu().contiguous()
                if self.random_draws is not None
                else None
            ),
        )

    @classmethod
    def cat(
        cls,
        selections: Iterable[EvaluationRouteSelection],
        dim: int = 0,
    ) -> EvaluationRouteSelection:
        """Concatenate selections from compatible evaluation shards."""

        items = tuple(selections)
        if not items:
            raise ValueError("At least one evaluation route selection is required.")
        mode = items[0].mode
        if any(item.mode is not mode for item in items[1:]):
            raise ValueError("Cannot combine different evaluation routing modes.")
        first_has_draws = items[0].random_draws is not None
        if any(
            (item.random_draws is not None) != first_has_draws for item in items[1:]
        ):
            raise ValueError("Cannot combine inconsistent random-draw metadata.")
        return cls(
            mode=mode,
            effective_next_route=torch.cat(
                [item.effective_next_route for item in items],
                dim=dim,
            ),
            counterfactual_next_route=torch.cat(
                [item.counterfactual_next_route for item in items],
                dim=dim,
            ),
            random_draws=(
                torch.cat(
                    [
                        item.random_draws
                        for item in items
                        if item.random_draws is not None
                    ],
                    dim=dim,
                )
                if items[0].random_draws is not None
                else None
            ),
        )

    def chunk(
        self,
        chunks: int,
        dim: int = 0,
    ) -> tuple[EvaluationRouteSelection, ...]:
        """Split evaluation selection metadata along a batch dimension."""

        effective_chunks = torch.chunk(self.effective_next_route, chunks, dim=dim)
        counterfactual_chunks = torch.chunk(
            self.counterfactual_next_route,
            chunks,
            dim=dim,
        )
        random_chunks = (
            torch.chunk(self.random_draws, chunks, dim=dim)
            if self.random_draws is not None
            else (None,) * len(effective_chunks)
        )
        return tuple(
            EvaluationRouteSelection(
                mode=self.mode,
                effective_next_route=effective,
                counterfactual_next_route=counterfactual,
                random_draws=random_draw,
            )
            for effective, counterfactual, random_draw in zip(
                effective_chunks,
                counterfactual_chunks,
                random_chunks,
            )
        )

    def split(
        self,
        split_sizes: list[int],
        dim: int = 0,
    ) -> tuple[EvaluationRouteSelection, ...]:
        """Split selection metadata with explicit batch sizes."""

        effective_splits = torch.split(
            self.effective_next_route,
            split_sizes,
            dim=dim,
        )
        counterfactual_splits = torch.split(
            self.counterfactual_next_route,
            split_sizes,
            dim=dim,
        )
        random_splits = (
            torch.split(self.random_draws, split_sizes, dim=dim)
            if self.random_draws is not None
            else (None,) * len(effective_splits)
        )
        return tuple(
            EvaluationRouteSelection(
                mode=self.mode,
                effective_next_route=effective,
                counterfactual_next_route=counterfactual,
                random_draws=random_draw,
            )
            for effective, counterfactual, random_draw in zip(
                effective_splits,
                counterfactual_splits,
                random_splits,
            )
        )


def select_evaluation_routes(
    config: EvaluationRoutingConfig,
    *,
    gate_idm_probabilities: torch.Tensor,
    env_ids: torch.Tensor,
    episode_ids: torch.Tensor,
    source_chunk_ids: torch.Tensor,
    current_routes: torch.Tensor | None = None,
) -> EvaluationRouteSelection:
    """Select effective next routes without mutable or process-global RNG state."""

    if gate_idm_probabilities.ndim != 1:
        raise ValueError(
            "`gate_idm_probabilities` must have shape [B], got "
            f"{tuple(gate_idm_probabilities.shape)}."
        )
    if not gate_idm_probabilities.is_floating_point():
        raise TypeError("`gate_idm_probabilities` must use a floating dtype.")
    invalid_probability = (
        ~torch.isfinite(gate_idm_probabilities)
        | (gate_idm_probabilities < 0)
        | (gate_idm_probabilities > 1)
    )
    if bool(invalid_probability.any().item()):
        raise ValueError("`gate_idm_probabilities` must be finite and lie in [0, 1].")

    shape = gate_idm_probabilities.shape
    for name, value in (
        ("env_ids", env_ids),
        ("episode_ids", episode_ids),
        ("source_chunk_ids", source_chunk_ids),
    ):
        _validate_route_tensor(name, value)
        if value.shape != shape:
            raise ValueError(
                f"`{name}` must have shape {tuple(shape)}, got {tuple(value.shape)}."
            )

    if current_routes is not None:
        _validate_route_tensor("current_routes", current_routes)
        if current_routes.shape != shape:
            raise ValueError(
                "`current_routes` must have shape "
                f"{tuple(shape)}, got {tuple(current_routes.shape)}."
            )
        invalid_route = (current_routes != int(WAMRoute.UNCOND)) & (
            current_routes != int(WAMRoute.IDM)
        )
        if bool(invalid_route.any().item()):
            raise ValueError("`current_routes` contains an invalid route.")
    if (
        config.mode is EvaluationRoutingMode.AUTOCORRELATION_MATCHED_RANDOM
        and current_routes is None
    ):
        raise ValueError("`autocorrelation_matched_random` requires current_routes.")

    counterfactual = (gate_idm_probabilities >= config.idm_threshold).to(
        dtype=torch.long
    )
    random_draws = None
    if config.mode is EvaluationRoutingMode.LEARNED_THRESHOLD:
        effective = counterfactual.clone()
    elif config.mode is EvaluationRoutingMode.FORCED_IDM:
        effective = torch.full_like(counterfactual, int(WAMRoute.IDM))
    elif config.mode is EvaluationRoutingMode.FORCED_UNCOND:
        effective = torch.full_like(counterfactual, int(WAMRoute.UNCOND))
    elif config.mode in {
        EvaluationRoutingMode.STOCHASTIC_KEYED,
        EvaluationRoutingMode.MATCHED_RANDOM,
        EvaluationRoutingMode.AUTOCORRELATION_MATCHED_RANDOM,
    }:
        key_values = zip(
            env_ids.detach().cpu().tolist(),
            episode_ids.detach().cpu().tolist(),
            source_chunk_ids.detach().cpu().tolist(),
        )
        random_draws = torch.tensor(
            [
                _stateless_uniform(
                    routing_seed=config.routing_seed,
                    env_id=int(env_id),
                    episode_id=int(episode_id),
                    source_chunk_id=int(source_chunk_id),
                )
                for env_id, episode_id, source_chunk_id in key_values
            ],
            dtype=torch.float64,
            device=gate_idm_probabilities.device,
        )
        if config.mode is EvaluationRoutingMode.STOCHASTIC_KEYED:
            idm_thresholds = gate_idm_probabilities.to(dtype=torch.float64)
        elif config.mode is EvaluationRoutingMode.MATCHED_RANDOM:
            idm_thresholds: float | torch.Tensor = float(config.random_idm_probability)
        else:
            probability_after_idm, probability_after_uncond = (
                autocorrelated_transition_probabilities(
                    config.random_idm_probability,
                    config.random_lag1_autocorrelation,
                )
            )
            assert current_routes is not None
            idm_thresholds = torch.where(
                current_routes.to(device=gate_idm_probabilities.device)
                == int(WAMRoute.IDM),
                probability_after_idm,
                probability_after_uncond,
            )
        effective = (random_draws < idm_thresholds).to(dtype=torch.long)
    elif config.mode is EvaluationRoutingMode.PERIODIC:
        assert config.periodic_period is not None
        assert config.periodic_on_count is not None
        assert config.periodic_phase is not None
        effective = (
            (source_chunk_ids + config.periodic_phase) % config.periodic_period
            < config.periodic_on_count
        ).to(dtype=torch.long)
    else:  # pragma: no cover - enum validation makes this unreachable.
        raise AssertionError(f"Unhandled evaluation routing mode {config.mode}.")

    return EvaluationRouteSelection(
        mode=config.mode,
        effective_next_route=effective,
        counterfactual_next_route=counterfactual,
        random_draws=random_draws,
    )


__all__ = [
    "EvaluationRouteSelection",
    "EvaluationRoutingConfig",
    "EvaluationRoutingMode",
    "autocorrelated_transition_probabilities",
    "select_evaluation_routes",
]
