# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD-RV LIBERO factory for hosts whose EGL indices remain physical."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, TypeVar

_EnvironmentT = TypeVar("_EnvironmentT")


def configure_physical_egl_device(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Keep Ray's one-GPU physical EGL assignment through construction."""

    env = os.environ if environ is None else environ
    configured_backends = {
        str(value).strip().lower()
        for value in (env.get("MUJOCO_GL"), env.get("PYOPENGL_PLATFORM"))
        if str(value or "").strip()
    }
    if len(configured_backends) > 1:
        raise RuntimeError(
            "Conflicting MuJoCo/OpenGL backends prevent an auditable physical "
            f"EGL mapping: {sorted(configured_backends)}."
        )
    if configured_backends != {"egl"}:
        return {
            "applied": False,
            "backend": next(iter(configured_backends), None),
            "mapping_mode": "ray_assigned_physical_device",
            "remap_boundary": "libero_env_factory_before_renderer_construction",
        }

    visible = [
        item.strip()
        for item in str(env.get("CUDA_VISIBLE_DEVICES", "")).split(",")
        if item.strip()
    ]
    if len(visible) != 1:
        raise RuntimeError(
            "PAD-RV EGL workers require exactly one Ray-assigned physical GPU; "
            f"got {visible}."
        )
    physical_device = visible[0]
    previous = env.get("MUJOCO_EGL_DEVICE_ID")
    if previous != physical_device:
        raise RuntimeError(
            "PAD-RV requires MUJOCO_EGL_DEVICE_ID to equal the Ray-assigned "
            "physical GPU through renderer construction; got "
            f"visible={physical_device!r}, egl={previous!r}."
        )
    return {
        "applied": True,
        "backend": "egl",
        "mapping_mode": "ray_assigned_physical_device",
        "physical_visible_device": physical_device,
        "physical_mujoco_egl_device_id": previous,
        "remap_boundary": "libero_env_factory_before_renderer_construction",
    }


def instantiate_with_physical_egl(
    factory: Callable[..., _EnvironmentT],
    kwargs: Mapping[str, Any],
) -> _EnvironmentT:
    """Instantiate a PAD-RV worker without rewriting physical EGL to zero."""

    evidence = configure_physical_egl_device()
    environment = factory(**dict(kwargs))
    environment._rlinf_egl_device_mapping = evidence
    return environment


__all__ = ["configure_physical_egl_device", "instantiate_with_physical_egl"]
