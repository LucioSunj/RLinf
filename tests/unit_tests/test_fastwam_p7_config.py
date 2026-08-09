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

from __future__ import annotations

import hashlib

import pytest

from rlinf.models.embodiment.wam_policy.visual_config import (
    validate_p7_combination_config,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _disabled() -> dict:
    return {
        "enabled": False,
        "scientific_status": "NOT-RUN",
        "combination_training_authorized": False,
        "prerequisites": {
            "p7_0_assets_and_contracts": {
                "passed": False,
                "evidence_sha256": None,
            },
            "p7_1_p1_endpoint": {
                "passed": False,
                "evidence_sha256": None,
            },
            "p7_2_p6_endpoint": {
                "passed": False,
                "evidence_sha256": None,
            },
            "p7_3_combination_authorization": {
                "passed": False,
                "evidence_sha256": None,
            },
        },
        "dinov3_asset": None,
        "transport_asset": None,
        "reader": None,
        "replay": None,
    }


def test_p7_default_off_keeps_assets_null_and_status_not_run() -> None:
    config = validate_p7_combination_config(_disabled())

    assert not config.enabled
    assert config.scientific_status == "NOT-RUN"
    assert config.dinov3_asset is None
    assert config.transport_asset is None


@pytest.mark.parametrize(
    "failed_gate",
    [
        "p7_0_assets_and_contracts",
        "p7_1_p1_endpoint",
        "p7_2_p6_endpoint",
        "p7_3_combination_authorization",
    ],
)
def test_p7_enabled_rejects_each_missing_prerequisite(failed_gate: str) -> None:
    payload = _disabled()
    payload["enabled"] = True
    payload["combination_training_authorized"] = True
    payload.update(
        dinov3_asset={"asset": "test"},
        transport_asset={"asset": "test"},
        reader={"reader": "test"},
        replay={"backend": "test"},
    )
    for name, gate in payload["prerequisites"].items():
        gate["passed"] = name != failed_gate
        gate["evidence_sha256"] = None if name == failed_gate else _hash(name)

    with pytest.raises(PermissionError, match=failed_gate):
        validate_p7_combination_config(payload)


def test_p7_passed_gates_still_need_explicit_combination_authorization() -> None:
    payload = _disabled()
    payload["enabled"] = True
    payload.update(
        dinov3_asset={"asset": "test"},
        transport_asset={"asset": "test"},
        reader={"reader": "test"},
        replay={"backend": "test"},
    )
    for name, gate in payload["prerequisites"].items():
        gate["passed"] = True
        gate["evidence_sha256"] = _hash(name)

    with pytest.raises(PermissionError, match="explicitly true"):
        validate_p7_combination_config(payload)


def test_p7_disabled_cannot_smuggle_asset_paths_or_claim_results() -> None:
    payload = _disabled()
    payload["dinov3_asset"] = {"weights_path": "/should/not/be-read"}
    with pytest.raises(ValueError, match="must leave all"):
        validate_p7_combination_config(payload)

    payload = _disabled()
    payload["scientific_status"] = "PASS"
    with pytest.raises(ValueError, match="must remain NOT-RUN"):
        validate_p7_combination_config(payload)
