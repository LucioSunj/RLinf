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

import importlib.util
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "fastwam_optimizer_under_test",
    _REPO_ROOT / "rlinf/models/embodiment/wam_policy/optimizer.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
partition_fastwam_trainable_parameters = _MODULE.partition_fastwam_trainable_parameters
assert_fastwam_optimizer_update_resolution = (
    _MODULE.assert_fastwam_optimizer_update_resolution
)


def _parameter(dtype: torch.dtype = torch.float32) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros(1, dtype=dtype))


def test_partition_fastwam_trainable_parameters_is_disjoint() -> None:
    gate = _parameter()
    lora_a = _parameter()
    lora_b = _parameter()
    value = _parameter()

    groups = partition_fastwam_trainable_parameters(
        [
            ("gate.layers.0.weight", gate),
            ("actor.action_expert.q.lora_A", lora_a),
            ("actor.action_expert.q.lora_B", lora_b),
            ("critic.backbone.value_head.0.weight", value),
        ]
    )

    assert groups == {
        "gate": [gate],
        "uncond_lora": [lora_a, lora_b],
        "value_head": [value],
    }
    assert len({id(item) for values in groups.values() for item in values}) == 4


def test_partition_fastwam_trainable_parameters_rejects_base_weight() -> None:
    with pytest.raises(RuntimeError, match="outside Gate"):
        partition_fastwam_trainable_parameters(
            [
                ("gate.output.weight", _parameter()),
                ("actor.action_expert.base.weight", _parameter()),
                ("actor.action_expert.q.lora_A", _parameter()),
                ("critic.value_head.weight", _parameter()),
            ]
        )


def test_partition_fastwam_trainable_parameters_requires_all_groups() -> None:
    with pytest.raises(RuntimeError, match="uncond_lora"):
        partition_fastwam_trainable_parameters(
            [
                ("gate.output.weight", _parameter()),
                ("critic.value_head.weight", _parameter()),
            ]
        )


def test_partition_fastwam_trainable_parameters_rejects_reduced_precision() -> None:
    """A BF16 trainable silently discards sub-ULP optimizer steps."""

    with pytest.raises(RuntimeError, match="master weights"):
        partition_fastwam_trainable_parameters(
            [
                ("gate.output.weight", _parameter(dtype=torch.bfloat16)),
                ("actor.action_expert.q.lora_A", _parameter()),
                ("critic.value_head.weight", _parameter()),
            ]
        )


def test_partition_fastwam_trainable_parameters_accepts_fp32_groups() -> None:
    groups = partition_fastwam_trainable_parameters(
        [
            ("gate.output.weight", _parameter()),
            ("actor.action_expert.q.lora_A", _parameter()),
            ("critic.value_head.weight", _parameter()),
        ]
    )

    assert sorted(groups) == ["gate", "uncond_lora", "value_head"]


def _adamw_with_named_groups(dtype: torch.dtype) -> torch.optim.AdamW:
    parameters = {
        name: torch.nn.Parameter(torch.full((5,), 0.02, dtype=dtype))
        for name in ("gate", "uncond_lora", "value_head")
    }
    for parameter in parameters.values():
        parameter.grad = torch.full_like(parameter, 0.1)
    return torch.optim.AdamW(
        [
            {"name": name, "params": [parameter], "lr": 1e-5}
            for name, parameter in parameters.items()
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )


def test_update_resolution_accepts_representable_fp32_adam_step() -> None:
    report = assert_fastwam_optimizer_update_resolution(
        _adamw_with_named_groups(torch.float32),
        minimum_half_ulp_ratio=1.0,
    )

    assert set(report) == {"gate", "uncond_lora", "value_head"}
    assert all(
        details["median_update"] > details["half_ulp"] for details in report.values()
    )


def test_update_resolution_rejects_sub_ulp_bf16_adam_step() -> None:
    with pytest.raises(RuntimeError, match="not representable.*gate"):
        assert_fastwam_optimizer_update_resolution(
            _adamw_with_named_groups(torch.bfloat16),
            minimum_half_ulp_ratio=1.0,
        )


def test_update_resolution_counts_zero_updates_in_group_median() -> None:
    optimizer = _adamw_with_named_groups(torch.float32)
    optimizer.param_groups[0]["params"][0].grad[:3].zero_()

    with pytest.raises(RuntimeError, match="not representable.*gate"):
        assert_fastwam_optimizer_update_resolution(
            optimizer,
            minimum_half_ulp_ratio=1.0,
        )
