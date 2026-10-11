# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MuJoCo EGL mapping for Ray-isolated LIBERO environment processes."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, TypeVar

_EnvironmentT = TypeVar("_EnvironmentT")


def configure_isolated_egl_device(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Map one Ray-isolated physical GPU to MuJoCo's logical EGL device zero.

    Ray's worker setup exposes the assigned physical GPU in
    ``CUDA_VISIBLE_DEVICES`` and ``MUJOCO_EGL_DEVICE_ID``. After isolation,
    MuJoCo enumerates that sole visible GPU as logical device zero. This
    function is intentionally called only after the LIBERO/robosuite renderer
    class has been imported and immediately before it is instantiated.

    Args:
        environ: Mutable environment mapping. Defaults to ``os.environ``.

    Returns:
        Machine-readable evidence describing whether and how remapping occurred.

    Raises:
        RuntimeError: If EGL is configured but the worker is not exactly
            one-GPU isolated, or the physical device provenance is inconsistent.
    """

    env = os.environ if environ is None else environ
    configured_backends = {
        str(value).strip().lower()
        for value in (env.get("MUJOCO_GL"), env.get("PYOPENGL_PLATFORM"))
        if str(value or "").strip()
    }
    if len(configured_backends) > 1:
        raise RuntimeError(
            "Conflicting MuJoCo/OpenGL backends prevent an auditable EGL mapping: "
            f"{sorted(configured_backends)}."
        )
    if configured_backends != {"egl"}:
        return {
            "applied": False,
            "backend": next(iter(configured_backends), None),
            "remap_boundary": "libero_env_factory_before_renderer_construction",
        }

    visible = [
        item.strip()
        for item in str(env.get("CUDA_VISIBLE_DEVICES", "")).split(",")
        if item.strip()
    ]
    if len(visible) != 1:
        raise RuntimeError(
            "EGL LIBERO workers require exactly one Ray-isolated visible GPU; "
            f"got {visible}."
        )
    physical_device = visible[0]
    previous = env.get("MUJOCO_EGL_DEVICE_ID")
    if previous not in {physical_device, "0"}:
        raise RuntimeError(
            "MUJOCO_EGL_DEVICE_ID must retain the Ray-assigned physical GPU until "
            "LIBERO/robosuite import, or already be logical zero; got "
            f"visible={physical_device!r}, egl={previous!r}."
        )
    env["MUJOCO_EGL_DEVICE_ID"] = "0"
    return {
        "applied": True,
        "backend": "egl",
        "physical_visible_device": physical_device,
        "previous_mujoco_egl_device_id": previous,
        "logical_mujoco_egl_device_id": "0",
        "remap_boundary": "libero_env_factory_before_renderer_construction",
    }


def instantiate_with_isolated_egl(
    factory: Callable[..., _EnvironmentT],
    kwargs: Mapping[str, Any],
) -> _EnvironmentT:
    """Instantiate a LIBERO worker after the renderer-time logical EGL remap."""

    evidence = configure_isolated_egl_device()
    environment = factory(**dict(kwargs))
    environment._rlinf_egl_device_mapping = evidence
    return environment


__all__ = ["configure_isolated_egl_device", "instantiate_with_isolated_egl"]
