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

"""Deterministic, compact digests for checkpoint-owned runtime state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

FASTWAM_RESUME_AUDIT_SCHEMA = "fastwam-checkpoint-load-audit-v1"
FASTWAM_ACTOR_RESUME_AUDIT_SENTINEL = "FASTWAM_ACTOR_RESUME_AUDIT"
FASTWAM_ROLLOUT_RESUME_AUDIT_SENTINEL = "FASTWAM_ROLLOUT_RESUME_AUDIT"


def checkpoint_state_sha256(value: Any) -> str:
    """Hash nested checkpoint-compatible state without retaining its payload."""

    digest = hashlib.sha256()

    def update_token(kind: str, payload: bytes = b"") -> None:
        encoded_kind = kind.encode("utf-8")
        digest.update(len(encoded_kind).to_bytes(4, "big"))
        digest.update(encoded_kind)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            metadata = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            update_token("torch.Tensor.metadata", metadata)
            update_token(
                "torch.Tensor.bytes",
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
            )
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            metadata = json.dumps(
                {"dtype": str(array.dtype), "shape": list(array.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            update_token("numpy.ndarray.metadata", metadata)
            update_token("numpy.ndarray.bytes", array.tobytes())
            return
        if isinstance(item, Mapping):
            update_token("mapping.start", str(len(item)).encode("ascii"))
            for key in sorted(
                item, key=lambda candidate: (type(candidate).__name__, repr(candidate))
            ):
                visit(key)
                visit(item[key])
            update_token("mapping.end")
            return
        if isinstance(item, tuple):
            update_token("tuple.start", str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
            update_token("tuple.end")
            return
        if isinstance(item, list):
            update_token("list.start", str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
            update_token("list.end")
            return
        if isinstance(item, np.generic):
            update_token(
                f"numpy.scalar.{item.dtype}",
                np.asarray(item).tobytes(),
            )
            return
        if item is None or isinstance(item, (bool, int, float, str, bytes)):
            payload = item if isinstance(item, bytes) else repr(item).encode("utf-8")
            update_token(type(item).__name__, payload)
            return
        raise TypeError(
            "Unsupported checkpoint-state value for deterministic hashing: "
            f"{type(item).__name__}."
        )

    visit(value)
    return digest.hexdigest()
