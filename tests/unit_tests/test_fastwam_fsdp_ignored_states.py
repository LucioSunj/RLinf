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

"""CPU contracts for FastWAM frozen-parameter FSDP ownership."""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.hybrid_engines.fsdp.strategy.fsdp import (
    _frozen_parameters_to_ignore,
)
from rlinf.workers.actor.fastwam_selective_sync import (
    prepare_fastwam_sync_tensors,
)


class _MixedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
        self.trainable = nn.Parameter(torch.ones(3), requires_grad=True)


def _config(*, enabled: bool, use_orig_params: bool = True):
    return OmegaConf.create(
        {
            "ignore_frozen_parameters": enabled,
            "use_orig_params": use_orig_params,
        }
    )


def test_frozen_parameter_selection_is_exact_and_opt_in() -> None:
    module = _MixedModule()

    assert (
        _frozen_parameters_to_ignore(
            module,
            _config(enabled=False),
        )
        is None
    )
    ignored = _frozen_parameters_to_ignore(
        module,
        _config(enabled=True),
    )

    assert ignored == (module.frozen,)
    assert all(parameter is not module.trainable for parameter in ignored)


def test_frozen_parameter_selection_fails_closed() -> None:
    with pytest.raises(ValueError, match="use_orig_params"):
        _frozen_parameters_to_ignore(
            _MixedModule(),
            _config(enabled=True, use_orig_params=False),
        )

    all_trainable = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="no frozen"):
        _frozen_parameters_to_ignore(
            all_trainable,
            _config(enabled=True),
        )

    all_frozen = nn.Linear(2, 2)
    all_frozen.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable"):
        _frozen_parameters_to_ignore(
            all_frozen,
            _config(enabled=True),
        )


class _ReplacingBufferModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer("persistent", torch.zeros(1), persistent=True)
        self.to_called = False

    def to(self, *args, **kwargs):
        self.to_called = True
        self._buffers["persistent"] = self.persistent.clone()
        return self


def test_sync_references_are_captured_after_module_move() -> None:
    module = _ReplacingBufferModule()
    original_buffer = module.persistent

    captured = prepare_fastwam_sync_tensors(module, device="cpu")

    assert module.to_called is True
    assert module.persistent is not original_buffer
    assert captured["persistent"].tensor is module.persistent
    assert captured["weight"].tensor is module.weight


def test_sync_preparation_rejects_tensors_left_off_device() -> None:
    module = _ReplacingBufferModule()

    with pytest.raises(RuntimeError, match="remain off"):
        prepare_fastwam_sync_tensors(module, device="meta")
