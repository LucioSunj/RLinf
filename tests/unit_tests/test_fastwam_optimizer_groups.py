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


def _parameter() -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros(1))


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


def test_partition_fastwam_trainable_parameters_adds_explicit_visual_family() -> None:
    gate = _parameter()
    lora = _parameter()
    value = _parameter()
    visual_query = _parameter()
    visual_beta = _parameter()

    groups = partition_fastwam_trainable_parameters(
        [
            ("gate.output.weight", gate),
            ("actor.action_expert.q.lora_A", lora),
            ("critic.value_head.weight", value),
            ("visual_reader.routers.0.query_projection.weight", visual_query),
            ("visual_reader.branches.0.0.raw_beta", visual_beta),
        ],
        visual_router_parameter_ids={id(visual_query), id(visual_beta)},
    )

    assert groups["visual_router"] == [visual_query, visual_beta]
    assert len({id(item) for values in groups.values() for item in values}) == 5


def test_partition_fastwam_visual_ownership_comes_from_manifest_ids() -> None:
    gate = _parameter()
    lora = _parameter()
    value = _parameter()
    visual = _parameter()

    groups = partition_fastwam_trainable_parameters(
        [
            ("gate.output.weight", gate),
            ("actor.action_expert.q.lora_A", lora),
            ("critic.value_head.weight", value),
            ("fsdp_flat_parameter_7", visual),
        ],
        visual_router_parameter_ids={id(visual)},
    )

    assert groups["visual_router"] == [visual]


def test_partition_fastwam_rejects_unmanifested_visual_name() -> None:
    with pytest.raises(RuntimeError, match="not present in visual-router manifest"):
        partition_fastwam_trainable_parameters(
            [
                ("gate.output.weight", _parameter()),
                ("actor.action_expert.q.lora_A", _parameter()),
                ("critic.value_head.weight", _parameter()),
                ("visual_reader.routers.0.weight", _parameter()),
            ]
        )


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
