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
fastwam_visual_reader_parameter_ids = _MODULE.fastwam_visual_reader_parameter_ids


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


def test_p7_partition_uses_typed_ids_with_fsdp_flat_names() -> None:
    gate = _parameter()
    lora = _parameter()
    value = _parameter()
    reader = _parameter()

    groups = partition_fastwam_trainable_parameters(
        [
            ("gate.output.weight", gate),
            ("actor.q.lora_A", lora),
            ("critic.value_head.weight", value),
            ("_fsdp_wrapped_module._flat_param_3", reader),
        ],
        visual_reader_parameter_ids={id(reader)},
    )

    assert groups["dual_visual_reader"] == [reader]


def test_p7_partition_rejects_spoofed_name_and_missing_manifest_id() -> None:
    gate = _parameter()
    lora = _parameter()
    value = _parameter()
    spoofed = _parameter()
    with pytest.raises(RuntimeError, match="outside Gate"):
        partition_fastwam_trainable_parameters(
            [
                ("gate.output.weight", gate),
                ("actor.q.lora_A", lora),
                ("critic.value_head.weight", value),
                ("visual_reader.spoofed.weight", spoofed),
            ]
        )

    missing = _parameter()
    with pytest.raises(RuntimeError, match="did not observe every typed"):
        partition_fastwam_trainable_parameters(
            [
                ("gate.output.weight", gate),
                ("actor.q.lora_A", lora),
                ("critic.value_head.weight", value),
            ],
            visual_reader_parameter_ids={id(missing)},
        )


def test_p7_manifest_ids_unwrap_policy_shell() -> None:
    class _Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = _parameter()

        def trainable_parameter_manifest(self):
            return {"dual_visual_reader": ("weight",)}

    class _Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_reader = _Reader()

    class _Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._fsdp_wrapped_module = module

    policy = _Policy()

    assert fastwam_visual_reader_parameter_ids(_Wrapper(policy)) == {
        id(policy.visual_reader.weight)
    }
