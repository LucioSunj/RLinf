# Copyright 2026 The RLinf Authors.
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


"""Reward helpers shared by LIBERO chunk execution paths."""

from __future__ import annotations

import torch


def mask_rewards_after_first_done(
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    """Keep rewards through the first terminal primitive and zero the rest."""

    if rewards.shape != dones.shape:
        raise ValueError(
            "LIBERO chunk rewards and dones must have identical shapes, got "
            f"{tuple(rewards.shape)} and {tuple(dones.shape)}."
        )
    if rewards.ndim != 2:
        raise ValueError("LIBERO chunk rewards must have shape [batch, actions].")
    if dones.dtype != torch.bool:
        raise TypeError("LIBERO chunk dones must use torch.bool.")
    prior_done = torch.zeros_like(dones)
    if dones.shape[1] > 1:
        prior_done[:, 1:] = dones[:, :-1].to(torch.int64).cumsum(dim=1) > 0
    return torch.where(prior_done, torch.zeros_like(rewards), rewards)
