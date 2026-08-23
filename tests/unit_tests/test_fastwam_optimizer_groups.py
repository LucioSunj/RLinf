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
partition_fastwam_trainable_parameters = (
    _MODULE.partition_fastwam_trainable_parameters
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
