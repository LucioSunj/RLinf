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
