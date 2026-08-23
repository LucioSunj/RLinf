"""Optimizer ownership checks for the adaptive FastWAM policy."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

# Optimizer updates for these groups are far smaller than a BF16
# unit-in-last-place, so a reduced-precision parameter would silently discard
# every step. See `docs/BF16_PARAMETER_UPDATE_LOSS.md`.
_REQUIRED_TRAINABLE_DTYPE = torch.float32


def partition_fastwam_trainable_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> dict[str, list[nn.Parameter]]:
    """Fail closed unless every trainable tensor has one intended owner."""

    groups: dict[str, list[nn.Parameter]] = {
        "gate": [],
        "uncond_lora": [],
        "value_head": [],
    }
    unexpected: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if "value_head" in name:
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
            f"UNCOND LoRA, and value head: {unexpected}"
        )
    missing = [name for name, parameters in groups.items() if not parameters]
    if missing:
        raise RuntimeError(
            f"FastWAM adaptive optimizer is missing parameter groups: {missing}"
        )
    reduced_precision = sorted(
        {
            f"{group}:{parameter.dtype}"
            for group, parameters in groups.items()
            for parameter in parameters
            if parameter.dtype is not _REQUIRED_TRAINABLE_DTYPE
        }
    )
    if reduced_precision:
        raise RuntimeError(
            "FastWAM adaptive trainable parameters must be "
            f"{_REQUIRED_TRAINABLE_DTYPE} master weights; an optimizer step "
            "smaller than half the stored dtype's unit-in-last-place is "
            "discarded by round-to-nearest and the parameter never moves. "
            f"Offending groups: {reduced_precision}"
        )
    return groups
