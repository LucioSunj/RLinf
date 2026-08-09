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

"""P7 combination authorization and lightweight build-time validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
P7_SCIENTIFIC_STATUS = "NOT-RUN"


@dataclass(frozen=True)
class P7Prerequisite:
    """One immutable P7 gate and its evidence identity."""

    name: str
    passed: bool
    evidence_sha256: str | None

    @classmethod
    def from_mapping(cls, name: str, payload: Mapping[str, Any]) -> P7Prerequisite:
        if set(payload) != {"passed", "evidence_sha256"}:
            raise ValueError(f"{name} must contain `passed` and `evidence_sha256`.")
        passed = payload["passed"]
        if not isinstance(passed, bool):
            raise TypeError(f"{name}.passed must be a boolean.")
        evidence = payload["evidence_sha256"]
        if evidence is not None:
            evidence = str(evidence).strip().lower()
            if not _SHA256.fullmatch(evidence):
                raise ValueError(f"{name}.evidence_sha256 is invalid.")
        if passed and evidence is None:
            raise ValueError(f"{name} PASS requires an evidence SHA256.")
        if not passed and evidence is not None:
            raise ValueError(f"{name} cannot bind evidence before PASS.")
        return cls(name=name, passed=passed, evidence_sha256=evidence)


@dataclass(frozen=True)
class P7CombinationConfig:
    """Resolved P7 top-level config after gates are checked."""

    enabled: bool
    scientific_status: str
    combination_training_authorized: bool
    prerequisites: tuple[P7Prerequisite, ...]
    dinov3_asset: Mapping[str, Any] | None
    transport_asset: Mapping[str, Any] | None
    reader: Mapping[str, Any] | None
    replay: Mapping[str, Any] | None


_PREREQUISITES = (
    "p7_0_assets_and_contracts",
    "p7_1_p1_endpoint",
    "p7_2_p6_endpoint",
    "p7_3_combination_authorization",
)


def validate_p7_combination_config(
    payload: Mapping[str, Any] | None,
) -> P7CombinationConfig:
    """Validate P7 before importing DINO code or touching local assets."""

    if payload is None:
        payload = {
            "enabled": False,
            "scientific_status": P7_SCIENTIFIC_STATUS,
            "combination_training_authorized": False,
            "prerequisites": {
                name: {"passed": False, "evidence_sha256": None}
                for name in _PREREQUISITES
            },
            "dinov3_asset": None,
            "transport_asset": None,
            "reader": None,
            "replay": None,
        }
    required = {
        "enabled",
        "scientific_status",
        "combination_training_authorized",
        "prerequisites",
        "dinov3_asset",
        "transport_asset",
        "reader",
        "replay",
    }
    if set(payload) != required:
        raise ValueError(
            "Invalid P7 top-level fields; "
            f"missing={sorted(required - set(payload))}, "
            f"unknown={sorted(set(payload) - required)}."
        )
    enabled = payload["enabled"]
    authorization = payload["combination_training_authorized"]
    if not isinstance(enabled, bool) or not isinstance(authorization, bool):
        raise TypeError("P7 enable/authorization fields must be booleans.")
    scientific_status = str(payload["scientific_status"])
    if scientific_status != P7_SCIENTIFIC_STATUS:
        raise ValueError(
            "Candidate P7 scientific status must remain NOT-RUN until evidence exists."
        )
    prerequisite_payload = payload["prerequisites"]
    if not isinstance(prerequisite_payload, Mapping) or set(
        prerequisite_payload
    ) != set(_PREREQUISITES):
        raise ValueError(
            "P7 prerequisites must explicitly include P7-0, P7-1, P7-2, and P7-3."
        )
    prerequisites = tuple(
        P7Prerequisite.from_mapping(name, prerequisite_payload[name])
        for name in _PREREQUISITES
    )
    artifact_fields = ("dinov3_asset", "transport_asset", "reader", "replay")
    artifacts = tuple(payload[name] for name in artifact_fields)
    if not enabled:
        if authorization:
            raise ValueError("Disabled P7 cannot authorize combination training.")
        if any(value is not None for value in artifacts):
            raise ValueError(
                "Disabled P7 must leave all asset/reader/replay payloads null so "
                "the baseline cannot inspect or load DINO."
            )
        return P7CombinationConfig(
            enabled=False,
            scientific_status=scientific_status,
            combination_training_authorized=False,
            prerequisites=prerequisites,
            dinov3_asset=None,
            transport_asset=None,
            reader=None,
            replay=None,
        )

    failed = [gate.name for gate in prerequisites if not gate.passed]
    if failed:
        raise PermissionError(
            f"P7 remains fail-closed because prerequisite gates are not PASS: {failed}."
        )
    if not authorization:
        raise PermissionError(
            "P7-3 evidence does not itself authorize training; "
            "`combination_training_authorized` must be explicitly true."
        )
    for name, value in zip(artifact_fields, artifacts, strict=True):
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"Enabled P7 requires a non-empty `{name}` mapping.")
    return P7CombinationConfig(
        enabled=True,
        scientific_status=scientific_status,
        combination_training_authorized=True,
        prerequisites=prerequisites,
        dinov3_asset=payload["dinov3_asset"],
        transport_asset=payload["transport_asset"],
        reader=payload["reader"],
        replay=payload["replay"],
    )
