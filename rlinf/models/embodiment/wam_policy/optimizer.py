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

"""Optimizer ownership checks for the adaptive FastWAM policy."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn

# Optimizer updates for these groups are far smaller than a BF16
# unit-in-last-place, so a reduced-precision parameter would silently discard
# every step. See `docs/BF16_PARAMETER_UPDATE_LOSS.md`.
_REQUIRED_TRAINABLE_DTYPE = torch.float32
_FASTWAM_GROUP_NAMES = ("gate", "uncond_lora", "value_head")


def fastwam_optimizer_gradient_norms(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Measure global post-clip gradient norms for each adaptive family."""

    indexed_groups: dict[str, Mapping[str, Any]] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name in indexed_groups:
            raise RuntimeError(
                f"FastWAM optimizer has duplicate parameter group {name!r}."
            )
        indexed_groups[name] = group
    if set(indexed_groups) != set(_FASTWAM_GROUP_NAMES):
        raise RuntimeError(
            "FastWAM gradient-norm groups differ from Gate, UNCOND LoRA, "
            f"and value head: {sorted(indexed_groups)}."
        )

    parameters_by_group = [
        list(indexed_groups[name]["params"]) for name in _FASTWAM_GROUP_NAMES
    ]
    if any(not parameters for parameters in parameters_by_group):
        raise RuntimeError("FastWAM gradient-norm group is empty.")
    devices = {
        parameter.device
        for parameters in parameters_by_group
        for parameter in parameters
    }
    if len(devices) != 1:
        raise RuntimeError(
            "FastWAM gradient-norm parameters must share one device, got "
            f"{sorted(map(str, devices))}."
        )
    squared_norms = torch.zeros(
        len(_FASTWAM_GROUP_NAMES),
        dtype=torch.float64,
        device=next(iter(devices)),
    )
    with torch.no_grad():
        for index, parameters in enumerate(parameters_by_group):
            for parameter in parameters:
                gradient = parameter.grad
                if gradient is None:
                    continue
                values = (
                    gradient.coalesce().values() if gradient.is_sparse else gradient
                )
                squared_norms[index].add_(
                    values.detach().float().square().sum(dtype=torch.float64)
                )
    norms = squared_norms.sqrt().cpu().tolist()
    return {name: float(norm) for name, norm in zip(_FASTWAM_GROUP_NAMES, norms)}


def _optimizer_step_value(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _exact_group_median(
    parameters: list[nn.Parameter],
    value_fn: Callable[[nn.Parameter], torch.Tensor | None],
) -> float:
    """Compute one exact group median without concatenating on the GPU."""

    value_count = sum(parameter.numel() for parameter in parameters)
    if value_count < 1:
        raise RuntimeError("FastWAM update-resolution group is empty.")
    flattened = torch.empty(value_count, dtype=torch.float32, device="cpu")
    offset = 0
    for parameter in parameters:
        values = value_fn(parameter)
        count = parameter.numel()
        if values is None:
            flattened[offset : offset + count].zero_()
        else:
            if values.shape != parameter.shape:
                raise RuntimeError(
                    "FastWAM update-resolution tensor shape does not match its "
                    "parameter."
                )
            flattened[offset : offset + count].copy_(
                values.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            )
        offset += count
    return float(flattened.median().item())


def _parameter_ulp(value: float, *, dtype: torch.dtype) -> float:
    if not dtype.is_floating_point:
        raise TypeError(
            "FastWAM update resolution requires floating-point parameters, "
            f"got {dtype}."
        )
    stored = torch.tensor(value, dtype=dtype, device="cpu")
    successor = torch.nextafter(stored, torch.tensor(math.inf, dtype=dtype))
    ulp = float((successor - stored).item())
    if not math.isfinite(ulp) or ulp <= 0:
        raise RuntimeError(
            "FastWAM could not determine a finite positive parameter ULP at "
            f"|w|={value:.9g} for {dtype}."
        )
    return ulp


def assert_fastwam_optimizer_update_resolution(
    optimizer: torch.optim.Optimizer,
    *,
    minimum_half_ulp_ratio: float,
) -> dict[str, dict[str, float | int | str]]:
    """Require the first AdamW update to be representable in every group.

    The check consumes unscaled, clipped gradients immediately before the first
    real optimizer step. It evaluates the next bias-corrected Adam moments but
    does not mutate parameters or optimizer state.
    """

    minimum_ratio = float(minimum_half_ulp_ratio)
    if not math.isfinite(minimum_ratio) or minimum_ratio < 1.0:
        raise ValueError(
            "FastWAM minimum_half_ulp_ratio must be finite and at least 1.0."
        )
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("FastWAM update-resolution guard requires AdamW.")

    indexed_groups: dict[str, Mapping[str, Any]] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name in indexed_groups:
            raise RuntimeError(
                f"FastWAM optimizer has duplicate parameter group {name!r}."
            )
        indexed_groups[name] = group
    if set(indexed_groups) != set(_FASTWAM_GROUP_NAMES):
        raise RuntimeError(
            "FastWAM update-resolution groups differ from Gate, UNCOND LoRA, "
            f"and value head: {sorted(indexed_groups)}."
        )

    report: dict[str, dict[str, float | int | str]] = {}
    for name in _FASTWAM_GROUP_NAMES:
        group = indexed_groups[name]
        parameters = list(group["params"])
        if not parameters:
            raise RuntimeError(f"FastWAM optimizer group {name!r} has no parameters.")
        dtypes = {parameter.dtype for parameter in parameters}
        if len(dtypes) != 1:
            raise RuntimeError(
                f"FastWAM optimizer group {name!r} mixes parameter dtypes: "
                f"{sorted(map(str, dtypes))}."
            )
        parameter_dtype = next(iter(dtypes))
        beta1, beta2 = (float(value) for value in group["betas"])
        learning_rate = float(group["lr"])
        epsilon = float(group["eps"])
        amsgrad = bool(group.get("amsgrad", False))

        def update_magnitude(parameter: nn.Parameter) -> torch.Tensor | None:
            gradient = parameter.grad
            if gradient is None:
                return None
            if gradient.is_sparse:
                raise TypeError(
                    "FastWAM AdamW update-resolution guard does not support "
                    "sparse gradients."
                )
            state = optimizer.state.get(parameter, {})
            step = _optimizer_step_value(state.get("step", 0)) + 1
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            gradient_fp32 = gradient.detach().float()
            if exp_avg is None:
                next_exp_avg = gradient_fp32.mul(1.0 - beta1)
            else:
                next_exp_avg = (
                    exp_avg.detach()
                    .float()
                    .mul(beta1)
                    .add(
                        gradient_fp32,
                        alpha=1.0 - beta1,
                    )
                )
            if exp_avg_sq is None:
                next_exp_avg_sq = gradient_fp32.square().mul(1.0 - beta2)
            else:
                next_exp_avg_sq = (
                    exp_avg_sq.detach()
                    .float()
                    .mul(beta2)
                    .addcmul(
                        gradient_fp32,
                        gradient_fp32,
                        value=1.0 - beta2,
                    )
                )
            if amsgrad:
                max_exp_avg_sq = state.get("max_exp_avg_sq")
                if max_exp_avg_sq is not None:
                    next_exp_avg_sq = torch.maximum(
                        max_exp_avg_sq.detach().float(),
                        next_exp_avg_sq,
                    )
            m_hat = next_exp_avg / (1.0 - beta1**step)
            v_hat = next_exp_avg_sq / (1.0 - beta2**step)
            return learning_rate * m_hat.abs() / (v_hat.sqrt() + epsilon)

        median_update = _exact_group_median(parameters, update_magnitude)
        median_abs_weight = _exact_group_median(
            parameters,
            lambda parameter: parameter.detach().abs(),
        )
        ulp = _parameter_ulp(median_abs_weight, dtype=parameter_dtype)
        required_update = minimum_ratio * 0.5 * ulp
        if not math.isfinite(median_update) or median_update <= required_update:
            raise RuntimeError(
                "FastWAM optimizer update is not representable at the first "
                f"step for group {name!r}: median_update={median_update:.9g}, "
                f"median_abs_weight={median_abs_weight:.9g}, dtype="
                f"{parameter_dtype}, half_ulp={0.5 * ulp:.9g}, required="
                f"{required_update:.9g}."
            )
        report[name] = {
            "parameter_count": len(parameters),
            "value_count": sum(parameter.numel() for parameter in parameters),
            "dtype": str(parameter_dtype),
            "learning_rate": learning_rate,
            "median_update": median_update,
            "median_abs_weight": median_abs_weight,
            "half_ulp": 0.5 * ulp,
            "minimum_half_ulp_ratio": minimum_ratio,
            "observed_half_ulp_ratio": median_update / (0.5 * ulp),
        }
    return report


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
