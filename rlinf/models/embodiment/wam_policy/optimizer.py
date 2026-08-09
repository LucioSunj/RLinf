"""Optimizer ownership checks for the adaptive FastWAM policy."""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


def fastwam_visual_reader_parameter_ids(model: nn.Module) -> frozenset[int]:
    """Resolve the typed P7 reader manifest through an optional FSDP wrapper."""

    current = model
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "visual_reader"):
            reader = current.visual_reader
            if reader is None:
                return frozenset()
            manifest = reader.trainable_parameter_manifest()
            if tuple(manifest) != ("dual_visual_reader",):
                raise RuntimeError(
                    "P7 reader must expose exactly the dual_visual_reader family."
                )
            parameter_names = tuple(manifest["dual_visual_reader"])
            named = dict(reader.named_parameters())
            if any(name not in named for name in parameter_names):
                raise RuntimeError(
                    "P7 reader manifest names are missing from the module."
                )
            actual_trainable = tuple(
                name for name, parameter in named.items() if parameter.requires_grad
            )
            if actual_trainable != parameter_names:
                raise RuntimeError(
                    "P7 reader manifest does not exactly cover its trainable tensors."
                )
            parameter_ids = frozenset(id(named[name]) for name in parameter_names)
            if len(parameter_ids) != len(parameter_names):
                raise RuntimeError("P7 reader manifest aliases a trainable parameter.")
            return parameter_ids
        next_model = getattr(current, "module", None)
        if next_model is None:
            next_model = getattr(current, "_fsdp_wrapped_module", None)
        if next_model is None:
            break
        current = next_model
    raise TypeError("Could not unwrap the FastWAM policy for P7 ownership audit.")


def partition_fastwam_trainable_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    visual_reader_parameter_ids: Iterable[int] = (),
) -> dict[str, list[nn.Parameter]]:
    """Fail closed unless every trainable tensor has one intended owner."""

    groups: dict[str, list[nn.Parameter]] = {
        "gate": [],
        "uncond_lora": [],
        "value_head": [],
    }
    expected_visual_ids = frozenset(int(value) for value in visual_reader_parameter_ids)
    if expected_visual_ids:
        groups["dual_visual_reader"] = []
    observed_visual_ids: set[int] = set()
    observed_trainable_ids: set[int] = set()
    unexpected: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in observed_trainable_ids:
            raise RuntimeError(
                f"FastWAM optimizer encountered aliased trainable parameter {name!r}."
            )
        observed_trainable_ids.add(parameter_id)
        visual_name = name.startswith("visual_reader.") or ".visual_reader." in name
        if parameter_id in expected_visual_ids:
            groups["dual_visual_reader"].append(parameter)
            observed_visual_ids.add(parameter_id)
        elif visual_name:
            unexpected.append(name)
        elif "value_head" in name:
            groups["value_head"].append(parameter)
        elif name.endswith(".lora_A") or name.endswith(".lora_B"):
            groups["uncond_lora"].append(parameter)
        elif name.startswith("gate.") or ".gate." in name:
            groups["gate"].append(parameter)
        else:
            unexpected.append(name)
    missing_visual_ids = expected_visual_ids - observed_visual_ids
    if missing_visual_ids:
        raise RuntimeError(
            "FastWAM optimizer did not observe every typed P7 reader parameter ID: "
            f"missing={len(missing_visual_ids)}."
        )
    if unexpected:
        raise RuntimeError(
            "FastWAM adaptive training found trainable parameters outside Gate, "
            f"UNCOND LoRA, value head, and optional P7 reader: {unexpected}"
        )
    missing = [name for name, parameters in groups.items() if not parameters]
    if missing:
        raise RuntimeError(
            f"FastWAM adaptive optimizer is missing parameter groups: {missing}"
        )
    return groups
