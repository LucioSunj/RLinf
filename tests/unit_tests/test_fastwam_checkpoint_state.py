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

"""Focused tests for compact checkpoint-state and value-metric audits."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlinf.scheduler import Worker
from rlinf.utils.checkpoint_state import checkpoint_state_sha256
from rlinf.utils.metric_utils import compute_rollout_metrics


def test_checkpoint_state_digest_is_deterministic_and_value_sensitive() -> None:
    first = {
        "tensor": torch.tensor([[1, 2]], dtype=torch.int64),
        "numpy": np.asarray([3, 4], dtype=np.uint32),
        "nested": ("state", [True, None, 0.5]),
    }
    reordered = {
        "nested": ("state", [True, None, 0.5]),
        "numpy": np.asarray([3, 4], dtype=np.uint32),
        "tensor": torch.tensor([[1, 2]], dtype=torch.int64),
    }

    assert checkpoint_state_sha256(first) == checkpoint_state_sha256(reordered)
    reordered["tensor"][0, 1] = 9
    assert checkpoint_state_sha256(first) != checkpoint_state_sha256(reordered)


def test_checkpoint_state_digest_rejects_unsupported_payload() -> None:
    with pytest.raises(TypeError, match="Unsupported checkpoint-state"):
        checkpoint_state_sha256(object())


def test_rollout_metrics_record_valid_value_estimates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Worker.torch_platform,
        "current_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _value, op: None)
    metrics = compute_rollout_metrics(
        {
            "prev_values": torch.tensor([[[1.0]], [[3.0]], [[99.0]]]),
            "loss_mask": torch.ones(2, 1, 1, dtype=torch.bool),
        }
    )

    assert metrics["values_mean"] == pytest.approx(2.0)
    assert metrics["values_min"] == pytest.approx(1.0)
    assert metrics["values_max"] == pytest.approx(3.0)


def test_rollout_metrics_record_reward_and_branch_advantage_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Worker.torch_platform,
        "current_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _value, op: None)
    metrics = compute_rollout_metrics(
        {
            "rewards": torch.tensor([[[1.0]], [[-0.5]], [[99.0]]]),
            "gate_advantages": torch.tensor([[[2.0]], [[-1.0]], [[7.0]]]),
            "gate_valid_mask": torch.tensor([[[True]], [[True]], [[False]]]),
            "flow_advantages": torch.tensor([[[3.0]], [[5.0]], [[11.0]]]),
            "flow_valid_mask": torch.tensor([[[True]], [[False]], [[True]]]),
            "loss_mask": torch.tensor([[[True]], [[True]], [[False]]]),
        }
    )

    assert metrics["rewards"] == pytest.approx(0.25)
    assert metrics["rewards_mean"] == pytest.approx(0.25)
    assert metrics["rewards_min"] == pytest.approx(-0.5)
    assert metrics["rewards_max"] == pytest.approx(1.0)
    assert metrics["gate_advantages_mean"] == pytest.approx(0.5)
    assert metrics["gate_advantages_min"] == pytest.approx(-1.0)
    assert metrics["gate_advantages_max"] == pytest.approx(2.0)
    assert metrics["flow_advantages_mean"] == pytest.approx(7.0)
    assert metrics["flow_advantages_min"] == pytest.approx(3.0)
    assert metrics["flow_advantages_max"] == pytest.approx(11.0)
