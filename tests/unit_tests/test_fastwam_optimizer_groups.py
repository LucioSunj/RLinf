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
FastWAMRefinerParameterManifest = _MODULE.FastWAMRefinerParameterManifest


def _parameter() -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros(1))


def _base_named_parameters():
    return [
        ("gate.layers.0.weight", _parameter()),
        ("actor.action_expert.q.lora_A", _parameter()),
        ("actor.action_expert.q.lora_B", _parameter()),
        ("critic.backbone.value_head.0.weight", _parameter()),
    ]


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


def test_partition_frozen_gate_endpoint_omits_gate_group() -> None:
    named = _base_named_parameters()
    named[0][1].requires_grad_(False)

    groups = partition_fastwam_trainable_parameters(named, require_gate=False)

    assert set(groups) == {"uncond_lora", "value_head"}

    named[0][1].requires_grad_(True)
    with pytest.raises(RuntimeError, match="Frozen Gate endpoint"):
        partition_fastwam_trainable_parameters(named, require_gate=False)


def test_p8_refiner_partition_uses_manifest_identity_with_flat_name() -> None:
    refiner = _parameter()
    manifest = FastWAMRefinerParameterManifest(
        entries=(("layers.0.output_up.weight", refiner),)
    )
    groups = partition_fastwam_trainable_parameters(
        [*_base_named_parameters(), ("_fsdp_wrapped_module._flat_param", refiner)],
        require_refiner=True,
        refiner_manifest=manifest,
    )

    assert groups["wan_current_refiner"] == [refiner]


def test_p8_refiner_partition_rejects_spoofed_name() -> None:
    refiner = _parameter()
    spoofed = _parameter()
    manifest = FastWAMRefinerParameterManifest(entries=(("real", refiner),))

    with pytest.raises(RuntimeError, match="outside Gate"):
        partition_fastwam_trainable_parameters(
            [
                *_base_named_parameters(),
                ("flat.real", refiner),
                ("wan_current_refiner.spoofed", spoofed),
            ],
            require_refiner=True,
            refiner_manifest=manifest,
        )


def test_p8_refiner_partition_rejects_missing_manifest_identity() -> None:
    refiner = _parameter()
    manifest = FastWAMRefinerParameterManifest(entries=(("real", refiner),))

    with pytest.raises(RuntimeError, match="missing from the optimizer model"):
        partition_fastwam_trainable_parameters(
            _base_named_parameters(),
            require_refiner=True,
            refiner_manifest=manifest,
        )


def test_p8_refiner_partition_rejects_manifest_when_disabled() -> None:
    refiner = _parameter()
    manifest = FastWAMRefinerParameterManifest(entries=(("real", refiner),))

    with pytest.raises(RuntimeError, match="optimizer contract is disabled"):
        partition_fastwam_trainable_parameters(
            [*_base_named_parameters(), ("flat.real", refiner)],
            require_refiner=False,
            refiner_manifest=manifest,
        )
