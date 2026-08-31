# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Exact Gate/value ownership for PAD-Frozen."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

PAD_FROZEN_GROUP_NAMES = ("gate", "value_head")


def partition_pad_frozen_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> dict[str, list[nn.Parameter]]:
    """Partition every trainable parameter into one Stage 1 owner."""

    groups = {name: [] for name in PAD_FROZEN_GROUP_NAMES}
    unexpected: list[str] = []
    seen: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if "value_head" in name:
            groups["value_head"].append(parameter)
        elif name.startswith("gate.") or ".gate." in name:
            groups["gate"].append(parameter)
        else:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(
            f"PAD-Frozen has unexpected trainable tensors: {unexpected}."
        )
    empty = [name for name, parameters in groups.items() if not parameters]
    if empty:
        raise RuntimeError(f"PAD-Frozen optimizer groups are empty: {empty}.")
    for name, parameters in groups.items():
        wrong_dtype = [
            str(parameter.dtype)
            for parameter in parameters
            if parameter.dtype != torch.float32
        ]
        if wrong_dtype:
            raise TypeError(
                f"PAD-Frozen {name} parameters must be FP32, got {wrong_dtype[:4]}."
            )
    return groups


def pad_frozen_gradient_norms(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Measure finite per-owner gradient norms after distributed unscale."""

    groups = {str(group.get("name", "")): group for group in optimizer.param_groups}
    if set(groups) != set(PAD_FROZEN_GROUP_NAMES):
        raise RuntimeError(f"PAD optimizer groups changed: {sorted(groups)}.")
    result: dict[str, float] = {}
    for name in PAD_FROZEN_GROUP_NAMES:
        squared = 0.0
        for parameter in groups[name]["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError(f"PAD {name} gradient is non-finite.")
            squared += float(gradient.square().sum(dtype=torch.float64).item())
        result[name] = math.sqrt(squared)
    return result


def _step_value(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("PAD optimizer step must be scalar.")
        value = value.item()
    return int(value)


def _exact_group_median(
    parameters: list[nn.Parameter],
    value_fn,
) -> float:
    value_count = sum(parameter.numel() for parameter in parameters)
    if value_count < 1:
        raise RuntimeError("PAD update-resolution group is empty.")
    flattened = torch.empty(value_count, dtype=torch.float32, device="cpu")
    offset = 0
    for parameter in parameters:
        values = value_fn(parameter)
        count = parameter.numel()
        if values is None:
            flattened[offset : offset + count].zero_()
        else:
            if values.shape != parameter.shape:
                raise RuntimeError("PAD update-resolution tensor shape changed.")
            flattened[offset : offset + count].copy_(
                values.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            )
        offset += count
    return float(flattened.median().item())


def _exact_active_group_median(
    parameters: list[nn.Parameter],
    value_fn,
) -> tuple[float, int, int]:
    value_count = sum(parameter.numel() for parameter in parameters)
    if value_count < 1:
        raise RuntimeError("PAD update-resolution group is empty.")
    flattened = torch.empty(value_count, dtype=torch.float32, device="cpu")
    offset = 0
    for parameter in parameters:
        values = value_fn(parameter)
        count = parameter.numel()
        if values is None:
            flattened[offset : offset + count].zero_()
        else:
            if values.shape != parameter.shape:
                raise RuntimeError("PAD update-resolution tensor shape changed.")
            flattened[offset : offset + count].copy_(
                values.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            )
        offset += count
    if not bool(torch.isfinite(flattened).all()):
        raise FloatingPointError("PAD proposed optimizer update is non-finite.")
    active = flattened > 0
    active_count = int(torch.count_nonzero(active).item())
    if active_count == 0:
        return 0.0, 0, value_count
    return float(flattened[active].median().item()), active_count, value_count


def _parameter_ulp(value: float, *, dtype: torch.dtype) -> float:
    stored = torch.tensor(value, dtype=dtype, device="cpu")
    successor = torch.nextafter(
        stored, torch.tensor(math.inf, dtype=dtype, device="cpu")
    )
    ulp = float((successor - stored).item())
    if not math.isfinite(ulp) or ulp <= 0:
        raise RuntimeError(
            f"PAD could not determine a finite parameter ULP at {value:.9g}."
        )
    return ulp


def assert_pad_frozen_update_resolution(
    optimizer: torch.optim.Optimizer,
    *,
    minimum_half_ulp_ratio: float,
) -> dict[str, dict[str, float | int | str]]:
    """Audit the next AdamW step before it mutates parameters or state."""

    minimum_ratio = float(minimum_half_ulp_ratio)
    if not math.isfinite(minimum_ratio) or minimum_ratio < 1.0:
        raise ValueError("PAD minimum_half_ulp_ratio must be finite and at least 1.")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("PAD update-resolution guard requires AdamW.")

    groups = {str(group.get("name", "")): group for group in optimizer.param_groups}
    if set(groups) != set(PAD_FROZEN_GROUP_NAMES):
        raise RuntimeError(f"PAD optimizer groups changed: {sorted(groups)}.")
    report: dict[str, dict[str, float | int | str]] = {}
    for name in PAD_FROZEN_GROUP_NAMES:
        group = groups[name]
        parameters = list(group["params"])
        if not parameters:
            raise RuntimeError(f"PAD optimizer group {name!r} is empty.")
        dtypes = {parameter.dtype for parameter in parameters}
        if dtypes != {torch.float32}:
            raise TypeError(
                f"PAD optimizer group {name!r} must be FP32, got "
                f"{sorted(map(str, dtypes))}."
            )
        beta1, beta2 = (float(value) for value in group["betas"])
        learning_rate = float(group["lr"])
        epsilon = float(group["eps"])
        amsgrad = bool(group.get("amsgrad", False))

        def update_magnitude(parameter: nn.Parameter) -> torch.Tensor | None:
            gradient = parameter.grad
            if gradient is None:
                return None
            if gradient.is_sparse:
                raise TypeError("PAD AdamW resolution audit rejects sparse gradients.")
            state = optimizer.state.get(parameter, {})
            step = _step_value(state.get("step", 0)) + 1
            gradient_fp32 = gradient.detach().float()
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            next_exp_avg = (
                gradient_fp32.mul(1.0 - beta1)
                if exp_avg is None
                else exp_avg.detach()
                .float()
                .mul(beta1)
                .add(gradient_fp32, alpha=1.0 - beta1)
            )
            next_exp_avg_sq = (
                gradient_fp32.square().mul(1.0 - beta2)
                if exp_avg_sq is None
                else exp_avg_sq.detach()
                .float()
                .mul(beta2)
                .addcmul(gradient_fp32, gradient_fp32, value=1.0 - beta2)
            )
            if amsgrad and state.get("max_exp_avg_sq") is not None:
                next_exp_avg_sq = torch.maximum(
                    state["max_exp_avg_sq"].detach().float(), next_exp_avg_sq
                )
            m_hat = next_exp_avg / (1.0 - beta1**step)
            v_hat = next_exp_avg_sq / (1.0 - beta2**step)
            return learning_rate * m_hat.abs() / (v_hat.sqrt() + epsilon)

        median_update, active_update_count, value_count = _exact_active_group_median(
            parameters, update_magnitude
        )
        if active_update_count == 0:
            raise RuntimeError(
                f"PAD optimizer group has no nonzero proposed updates: {name!r}."
            )
        median_abs_weight = _exact_group_median(
            parameters, lambda parameter: parameter.detach().abs()
        )
        half_ulp = 0.5 * _parameter_ulp(median_abs_weight, dtype=torch.float32)
        required = minimum_ratio * half_ulp
        if not math.isfinite(median_update) or median_update <= required:
            raise RuntimeError(
                "PAD optimizer update is not representable for group "
                f"{name!r}: median_update={median_update:.9g}, "
                f"half_ulp={half_ulp:.9g}, required={required:.9g}."
            )
        report[name] = {
            "parameter_count": len(parameters),
            "value_count": value_count,
            "active_update_value_count": active_update_count,
            "active_update_fraction": active_update_count / value_count,
            "dtype": "torch.float32",
            "learning_rate": learning_rate,
            "median_update": median_update,
            "median_abs_weight": median_abs_weight,
            "half_ulp": half_ulp,
            "minimum_half_ulp_ratio": minimum_ratio,
            "observed_half_ulp_ratio": median_update / half_ulp,
        }
    return report
