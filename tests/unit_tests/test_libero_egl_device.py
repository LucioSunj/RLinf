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

"""Tests for physical-to-logical EGL mapping in isolated LIBERO workers."""

from __future__ import annotations

import os

import pytest

from rlinf.envs.libero.egl import (
    configure_isolated_egl_device,
    instantiate_with_isolated_egl,
)


def test_renderer_factory_sees_logical_zero_after_physical_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    observed: dict[str, str] = {}

    class FakeEnvironment:
        pass

    def factory(*, marker: str) -> FakeEnvironment:
        observed["egl"] = os.environ["MUJOCO_EGL_DEVICE_ID"]
        observed["marker"] = marker
        return FakeEnvironment()

    environment = instantiate_with_isolated_egl(factory, {"marker": "created"})

    assert observed == {"egl": "0", "marker": "created"}
    assert environment._rlinf_egl_device_mapping == {
        "applied": True,
        "backend": "egl",
        "physical_visible_device": "1",
        "previous_mujoco_egl_device_id": "1",
        "logical_mujoco_egl_device_id": "0",
        "remap_boundary": "libero_env_factory_before_renderer_construction",
    }


def test_repeated_logical_zero_mapping_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")

    evidence = configure_isolated_egl_device()

    assert evidence["previous_mujoco_egl_device_id"] == "0"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "0"


@pytest.mark.parametrize(
    ("visible", "egl_device", "match"),
    [
        ("", "0", "exactly one"),
        ("0,1", "0", "exactly one"),
        ("1", "2", "physical GPU"),
    ],
)
def test_egl_mapping_fails_closed_on_ambiguous_provenance(
    monkeypatch: pytest.MonkeyPatch,
    visible: str,
    egl_device: str,
    match: str,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", egl_device)

    with pytest.raises(RuntimeError, match=match):
        configure_isolated_egl_device()


def test_non_egl_backend_is_not_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")

    evidence = configure_isolated_egl_device()

    assert evidence["applied"] is False
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"
