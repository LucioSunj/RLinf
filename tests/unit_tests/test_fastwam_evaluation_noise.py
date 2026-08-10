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

import pytest
import torch

from rlinf.models.embodiment.wam_policy.libero_runtime import (
    _domain_separated_noise_seed,
    _seeded_randn,
    _validate_noise_seeds,
)


def test_seeded_noise_is_reproducible_batch_order_invariant_and_local() -> None:
    seeds = _validate_noise_seeds(
        torch.tensor([101, 202], dtype=torch.long),
        batch_size=2,
        name="action",
    )
    before = torch.random.get_rng_state()
    first = torch.cat(
        [
            _seeded_randn(
                int(seed),
                (1, 4, 7),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for seed in seeds
        ]
    )
    after = torch.random.get_rng_state()
    reversed_result = torch.cat(
        [
            _seeded_randn(
                int(seed),
                (1, 4, 7),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for seed in reversed(seeds.tolist())
        ]
    )

    assert torch.equal(before, after)
    assert torch.equal(first[0], reversed_result[1])
    assert torch.equal(first[1], reversed_result[0])
    assert not torch.equal(first[0], first[1])


def test_domain_separated_flow_seed_is_stable_and_distinct() -> None:
    first = _domain_separated_noise_seed(101, domain="flow-sde")

    assert first == _domain_separated_noise_seed(101, domain="flow-sde")
    assert first != _domain_separated_noise_seed(102, domain="flow-sde")
    assert first != _domain_separated_noise_seed(101, domain="other")
    assert 0 <= first < 1 << 63


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.tensor([[1, 2]]), "one-dimensional"),
        (torch.tensor([1]), "batch"),
        (torch.tensor([1, -1]), "non-negative"),
        (torch.tensor([True, False]), "integer"),
    ],
)
def test_noise_seed_validation_fails_closed(value: torch.Tensor, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _validate_noise_seeds(value, batch_size=2, name="action")
