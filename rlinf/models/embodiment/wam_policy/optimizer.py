"""Optimizer ownership checks for the adaptive FastWAM policy."""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


class FastWAMRefinerParameterManifest:
    """Identity-bound trainable P8 parameters exported by the typed policy."""

    __slots__ = ("entries",)

    def __init__(self, entries: tuple[tuple[str, nn.Parameter], ...]) -> None:
        self.entries = tuple(entries)
        names = tuple(name for name, _parameter in self.entries)
        parameters = tuple(parameter for _name, parameter in self.entries)
        if (
            not names
            or len(names) != len(set(names))
            or len(parameters) != len({id(parameter) for parameter in parameters})
            or any(not isinstance(parameter, nn.Parameter) for parameter in parameters)
            or any(not parameter.requires_grad for parameter in parameters)
        ):
            raise ValueError(
                "P8 refiner manifest requires unique names and trainable parameters."
            )

    @property
    def parameter_ids(self) -> frozenset[int]:
        """Return immutable object identities used after FSDP unwrapping."""

        return frozenset(id(parameter) for _name, parameter in self.entries)

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        """Return manifest parameters in deterministic module order."""

        return tuple(parameter for _name, parameter in self.entries)


def partition_fastwam_trainable_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    require_refiner: bool = False,
    refiner_manifest: FastWAMRefinerParameterManifest | None = None,
) -> dict[str, list[nn.Parameter]]:
    """Fail closed unless every trainable tensor has one intended owner."""

    manifest_ids = (
        frozenset() if refiner_manifest is None else refiner_manifest.parameter_ids
    )
    if require_refiner and not manifest_ids:
        raise RuntimeError("Enabled P8 optimizer requires a typed refiner manifest.")
    if not require_refiner and manifest_ids:
        raise RuntimeError(
            "P8 refiner manifest is non-empty while its optimizer contract is disabled."
        )
    groups: dict[str, list[nn.Parameter]] = {
        "gate": [],
        "uncond_lora": [],
        "value_head": [],
        "wan_current_refiner": [],
    }
    unexpected: list[str] = []
    seen_refiner_ids: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        refiner_named = (
            name.startswith("wan_current_refiner.") or ".wan_current_refiner." in name
        )
        if parameter_id in manifest_ids:
            groups["wan_current_refiner"].append(parameter)
            seen_refiner_ids.add(parameter_id)
        elif refiner_named:
            unexpected.append(name)
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
            f"UNCOND LoRA, Wan-current refiner, and value head: {unexpected}"
        )
    missing_manifest_ids = manifest_ids - seen_refiner_ids
    if missing_manifest_ids:
        raise RuntimeError(
            "P8 refiner manifest parameters are missing from the optimizer model: "
            f"{len(missing_manifest_ids)}."
        )
    required = {"gate", "uncond_lora", "value_head"}
    if require_refiner:
        required.add("wan_current_refiner")
    missing = [name for name in sorted(required) if not groups[name]]
    if missing:
        raise RuntimeError(
            f"FastWAM adaptive optimizer is missing parameter groups: {missing}"
        )
    if not require_refiner:
        groups.pop("wan_current_refiner")
    return groups
