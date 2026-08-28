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

from __future__ import annotations

import math

import pytest
import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.utils import safe_normalize


def test_unfloored_normalization_is_bitwise_identical_to_legacy_expression() -> None:
    values = torch.tensor([-2.0, -0.5, 0.25, 3.0], dtype=torch.float32)
    mask = torch.ones_like(values, dtype=torch.bool)
    valid = values[mask]
    legacy = (values - valid.mean()) / (valid.std() + 1e-5)

    actual = safe_normalize(values, mask, std_floor=0.15)

    assert valid.std().item() > 0.15
    assert torch.equal(actual, legacy)


def test_low_variance_normalization_uses_exact_configured_floor() -> None:
    half_span = 0.01 / math.sqrt(2.0)
    values = torch.tensor([-half_span, half_span], dtype=torch.float32)
    mask = torch.ones_like(values, dtype=torch.bool)
    statistics = {}

    actual = safe_normalize(
        values,
        mask,
        std_floor=0.15,
        statistics=statistics,
    )

    expected = (values - values.mean()) / 0.15
    assert values.std().item() == pytest.approx(0.01)
    assert torch.equal(actual, expected)
    assert statistics["effective_standard_deviation"] == 0.15
    assert statistics["floor_hit_fraction"] == 1.0


def test_single_sample_normalization_uses_floor_instead_of_nan() -> None:
    values = torch.tensor([2.5], dtype=torch.float32)
    mask = torch.ones_like(values, dtype=torch.bool)
    statistics = {}

    actual = safe_normalize(
        values,
        mask,
        std_floor=0.15,
        statistics=statistics,
    )

    assert torch.equal(actual, torch.zeros_like(values))
    assert torch.isfinite(actual).all()
    assert statistics["effective_standard_deviation"] == 0.15
    assert statistics["floor_hit_fraction"] == 1.0


def test_gae_reports_normalization_floor_hit_for_training_telemetry() -> None:
    statistics = {}
    advantages, _returns = compute_gae_advantages_and_returns(
        rewards=torch.tensor([[0.0], [0.01]], dtype=torch.float32),
        values=torch.zeros(3, 1, dtype=torch.float32),
        dones=torch.zeros(3, 1, dtype=torch.bool),
        gamma=0.0,
        gae_lambda=0.0,
        normalize_advantages=True,
        loss_mask=torch.ones(2, 1, dtype=torch.bool),
        normalization_std_floor=0.15,
        normalization_statistics=statistics,
    )

    assert torch.isfinite(advantages).all()
    assert statistics["floor_hit_fraction"] == 1.0
    assert statistics["valid_count"] == 2
