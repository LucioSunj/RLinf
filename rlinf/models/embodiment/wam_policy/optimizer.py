"""Optimizer ownership checks for the adaptive FastWAM policy."""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


def partition_fastwam_trainable_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    visual_router_parameter_ids: set[int] | frozenset[int] | None = None,
) -> dict[str, list[nn.Parameter]]:
    """Fail closed unless every trainable tensor has one intended owner.

    P6 visual-router ownership is supplied from the reader's typed parameter
    manifest. Names remain an audit signal, but never establish ownership by
    themselves.
    """

    groups: dict[str, list[nn.Parameter]] = {
        "gate": [],
        "uncond_lora": [],
        "value_head": [],
        "visual_router": [],
    }
    unexpected: list[str] = []
    observed_visual_ids: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        visual_name = name.startswith("visual_reader.") or ".visual_reader." in name
        if (
            visual_router_parameter_ids is not None
            and parameter_id in visual_router_parameter_ids
        ):
            groups["visual_router"].append(parameter)
            observed_visual_ids.add(parameter_id)
        elif visual_name:
            unexpected.append(f"{name} (not present in visual-router manifest)")
        elif "value_head" in name:
            groups["value_head"].append(parameter)
        elif name.endswith(".lora_A") or name.endswith(".lora_B"):
            groups["uncond_lora"].append(parameter)
        elif name.startswith("gate.") or ".gate." in name:
            groups["gate"].append(parameter)
        else:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(
            "FastWAM adaptive training found trainable parameters outside Gate, "
            f"UNCOND LoRA, value head, and visual router: {unexpected}"
        )
    if visual_router_parameter_ids is not None:
        missing_visual_ids = set(visual_router_parameter_ids) - observed_visual_ids
        if missing_visual_ids:
            raise RuntimeError(
                "FastWAM visual-router manifest contains parameters that are not "
                "trainable members of the wrapped policy."
            )
    if not groups["visual_router"]:
        groups.pop("visual_router")
    missing = [name for name, parameters in groups.items() if not parameters]
    if missing:
        raise RuntimeError(
            f"FastWAM adaptive optimizer is missing parameter groups: {missing}"
        )
    return groups
