# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""One real residency check before the seven-device continuation samples."""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from rlinf.models.embodiment.wam_policy.pad_rv.memory import release_pad_host_memory
from rlinf.utils.utils import get_rng_state


def cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def assert_same_state(before: Any, after: Any, path: str = "state") -> None:
    """Compare values and dtypes directly, without digesting model contents."""

    if isinstance(before, torch.Tensor):
        same = (
            isinstance(after, torch.Tensor)
            and before.dtype == after.dtype
            and before.shape == after.shape
            and torch.equal(before, after.detach().cpu())
        )
    elif isinstance(before, np.ndarray):
        same = isinstance(after, np.ndarray) and np.array_equal(before, after)
    elif isinstance(before, dict):
        same = isinstance(after, dict) and before.keys() == after.keys()
        if same:
            for key in before:
                assert_same_state(before[key], after[key], f"{path}.{key}")
    elif isinstance(before, (list, tuple)):
        same = type(before) is type(after) and len(before) == len(after)
        if same:
            for index, (left, right) in enumerate(zip(before, after)):
                assert_same_state(left, right, f"{path}[{index}]")
    else:
        same = before == after
    if not same:
        raise RuntimeError(f"Shared-GPU residency changed {path}.")


def audit_residency_roundtrip(
    *,
    model: torch.nn.Module,
    onload: Callable[[], None],
    offload: Callable[[], None],
    device: torch.device,
    state: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Require exact CPU→device→CPU weights, optimizer state and RNG."""

    def tensors() -> dict[str, torch.Tensor]:
        return {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }

    initial = tensors()
    if any(value.device.type != "cpu" for value in initial.values()):
        raise RuntimeError("Shared-GPU roundtrip must start with an offloaded model.")
    parameter_ids = {key: id(value) for key, value in model.named_parameters()}
    weights = cpu_clone(initial)
    saved_state = cpu_clone(state())
    rng = cpu_clone(get_rng_state())
    del initial
    for move, expected_device in (
        (onload, torch.device(device)),
        (offload, torch.device("cpu")),
    ):
        move()
        current = tensors()
        if any(value.device != expected_device for value in current.values()):
            raise RuntimeError(
                f"Shared-GPU model did not move fully to {expected_device}."
            )
        if parameter_ids != {key: id(value) for key, value in model.named_parameters()}:
            raise RuntimeError("Shared-GPU move replaced Parameter objects.")
        assert_same_state(weights, current, "model")
        assert_same_state(saved_state, state())
        assert_same_state(rng, get_rng_state(), "rng")
        del current
    report = {
        "schema": "route-neutral-shared-gpu-roundtrip-v1",
        "status": "PASS",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_tensors": len(weights),
        "model_values": sum(value.numel() for value in weights.values()),
        "weights_dtypes_state_rng_unchanged": True,
        "final_model_device": "cpu",
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
    }
    del weights, saved_state, rng
    release_pad_host_memory(
        schema="route-neutral-shared-gpu-audit-release-v1",
        rank=0,
        phase="post_residency_audit",
    )
    return report
