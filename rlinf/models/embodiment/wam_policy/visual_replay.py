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

"""Typed replay payloads for the optional P6 native-DINO visual sidecar."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import torch
from fastwam.models.wan22.visual_contracts import (
    DINO_V3_NATIVE_DIM,
    DinoWanSpatialTransport,
    NativePatchMemory,
    PreparedCameraBatch,
)


class VisualReplayBackend(str, Enum):
    """Supported P6 replay inputs.

    Neither backend stores routed weights. They are always reconstructed from
    the live visual-router parameters during PPO replay.
    """

    STORED_NATIVE = "stored_native"
    RECOMPUTE_NATIVE = "recompute_native"


@dataclass(frozen=True)
class VisualReplayConfig:
    """Fail-closed native-DINO replay storage contract."""

    backend: VisualReplayBackend | str
    storage_dtype: str
    pin_memory: bool
    max_bytes_per_sample: int
    max_aggregate_bytes: int

    def __post_init__(self) -> None:
        backend = VisualReplayBackend(self.backend)
        if self.storage_dtype != "bfloat16":
            raise ValueError("P6 visual replay requires `storage_dtype: bfloat16`.")
        if self.pin_memory is not True:
            raise ValueError("P6 visual replay requires pinned CPU storage.")
        for name in ("max_bytes_per_sample", "max_aggregate_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"P6 `{name}` must be a positive integer.")
            object.__setattr__(self, name, int(value))
        if self.max_aggregate_bytes < self.max_bytes_per_sample:
            raise ValueError(
                "P6 aggregate replay cap cannot be below the per-sample cap."
            )
        object.__setattr__(self, "backend", backend)


def _pin_cpu(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().contiguous().cpu()
    try:
        return tensor.pin_memory()
    except RuntimeError as error:
        # CPU-only PyTorch builds have no pinned allocator. The requested
        # contract remains explicit; a CUDA production host must never fall
        # back to pageable replay without an error.
        if torch.cuda.is_available():
            raise RuntimeError(
                "P6 requested pinned replay storage but pinning failed on a "
                "CUDA-capable host."
            ) from error
        return tensor


def pin_visual_replay_forward_inputs(
    forward_inputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Re-pin P6 replay tensors after trajectory stack/shuffle copies."""

    pinned = (
        forward_inputs if isinstance(forward_inputs, dict) else dict(forward_inputs)
    )
    for name in list(pinned):
        value = pinned[name]
        if name.startswith("visual_") and value.device.type == "cpu":
            pinned[name] = _pin_cpu(value)
    return pinned


def _sha256_bytes(values: tuple[str, ...]) -> torch.Tensor:
    return torch.tensor(
        [list(bytes.fromhex(value)) for value in values],
        dtype=torch.uint8,
        device="cpu",
    )


def _decode_sha256_bytes(values: torch.Tensor) -> tuple[str, ...]:
    if values.ndim != 2 or values.shape[1] != 32 or values.dtype is not torch.uint8:
        raise ValueError("P6 effective transport hashes must have shape [B,32].")
    return tuple(bytes(row.tolist()).hex() for row in values.cpu())


def _batched_tensor_bytes(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    batch_sizes = {
        int(tensor.shape[0]) for tensor in payload.values() if tensor.ndim > 0
    }
    if len(batch_sizes) != 1:
        raise ValueError("P6 replay tensors must share one batch dimension.")
    batch_size = batch_sizes.pop()
    result = torch.zeros(batch_size, dtype=torch.long)
    for tensor in payload.values():
        result += tensor[0].numel() * tensor.element_size()
    return result


def _enforce_caps(
    payload: Mapping[str, torch.Tensor],
    config: VisualReplayConfig,
) -> None:
    per_sample = _batched_tensor_bytes(payload)
    if bool((per_sample > config.max_bytes_per_sample).any().item()):
        raise MemoryError(
            "P6 replay exceeds `max_bytes_per_sample`: "
            f"max={int(per_sample.max())}, limit={config.max_bytes_per_sample}."
        )
    aggregate = int(per_sample.sum())
    if aggregate > config.max_aggregate_bytes:
        raise MemoryError(
            "P6 replay exceeds `max_aggregate_bytes`: "
            f"actual={aggregate}, limit={config.max_aggregate_bytes}."
        )


def pack_visual_replay(
    *,
    config: VisualReplayConfig,
    transport: DinoWanSpatialTransport,
    camera_batch: PreparedCameraBatch,
    memory: NativePatchMemory,
    sample_indices: torch.Tensor | None = None,
    full_batch_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pack only the selected native replay source plus typed behavior masks."""

    if camera_batch.camera_ids != transport.camera_ids:
        raise ValueError("P6 replay camera order differs from its transport.")
    if memory.camera_ids != transport.camera_ids:
        raise ValueError("P6 native memory camera order differs from its transport.")
    if not torch.equal(
        camera_batch.camera_valid_mask.to(memory.camera_valid_mask.device),
        memory.camera_valid_mask,
    ):
        raise ValueError("P6 camera validity changed between pixels and memory.")
    effective_hashes = transport.effective_sha256(
        camera_valid_mask=memory.camera_valid_mask,
        patch_valid_mask=memory.patch_valid_mask,
    )
    common = {
        "visual_camera_valid_mask": _pin_cpu(memory.camera_valid_mask),
        "visual_effective_transport_sha256": _pin_cpu(_sha256_bytes(effective_hashes)),
    }
    if config.backend is VisualReplayBackend.STORED_NATIVE:
        payload = {
            **common,
            "visual_native_tokens": _pin_cpu(memory.tokens.to(torch.bfloat16)),
            "visual_patch_valid_mask": _pin_cpu(memory.patch_valid_mask),
        }
    else:
        payload = {
            **common,
            "visual_camera_pixels": _pin_cpu(camera_batch.pixels),
        }
    if (sample_indices is None) != (full_batch_size is None):
        raise ValueError(
            "P6 replay scatter requires both sample indices and full batch size."
        )
    if sample_indices is not None:
        indices = torch.as_tensor(sample_indices, dtype=torch.long, device="cpu")
        full_batch_size = int(full_batch_size)
        if indices.ndim != 1 or indices.numel() != memory.tokens.shape[0]:
            raise ValueError("P6 replay sample indices do not match native memory.")
        if (
            full_batch_size < 1
            or bool((indices < 0).any().item())
            or bool((indices >= full_batch_size).any().item())
            or torch.unique(indices).numel() != indices.numel()
        ):
            raise ValueError("P6 replay sample indices are invalid or duplicated.")
        scattered = {}
        for name, tensor in payload.items():
            target = torch.zeros(
                (full_batch_size, *tensor.shape[1:]),
                dtype=tensor.dtype,
                device="cpu",
            )
            target[indices] = tensor
            scattered[name] = _pin_cpu(target)
        route_mask = torch.zeros(full_batch_size, dtype=torch.bool)
        route_mask[indices] = True
        scattered["visual_route_mask"] = _pin_cpu(route_mask)
        payload = scattered
    _enforce_caps(payload, config)
    return payload


def empty_visual_replay(
    *,
    config: VisualReplayConfig,
    batch_size: int,
    camera_count: int,
    patch_grid: tuple[int, int],
    camera_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Create a schema-stable all-IDM payload without invoking DINO."""

    if batch_size < 1 or camera_count < 1:
        raise ValueError("P6 empty replay dimensions must be positive.")
    patches = int(patch_grid[0]) * int(patch_grid[1])
    common = {
        "visual_camera_valid_mask": torch.zeros(
            batch_size, camera_count, dtype=torch.bool
        ),
        "visual_effective_transport_sha256": torch.zeros(
            batch_size, 32, dtype=torch.uint8
        ),
        "visual_route_mask": torch.zeros(batch_size, dtype=torch.bool),
    }
    if config.backend is VisualReplayBackend.STORED_NATIVE:
        payload = {
            **common,
            "visual_native_tokens": torch.zeros(
                batch_size,
                camera_count,
                patches,
                DINO_V3_NATIVE_DIM,
                dtype=torch.bfloat16,
            ),
            "visual_patch_valid_mask": torch.zeros(
                batch_size,
                camera_count,
                patches,
                dtype=torch.bool,
            ),
        }
    else:
        payload = {
            **common,
            "visual_camera_pixels": torch.zeros(
                batch_size,
                camera_count,
                3,
                int(camera_hw[0]),
                int(camera_hw[1]),
                dtype=torch.uint8,
            ),
        }
    payload = {name: _pin_cpu(tensor) for name, tensor in payload.items()}
    _enforce_caps(payload, config)
    return payload


def unpack_stored_native_memory(
    payload: Mapping[str, torch.Tensor],
    *,
    camera_ids: tuple[str, ...],
    patch_grid: tuple[int, int],
    source_revision: str,
    weights_sha256: str,
    input_contract_sha256: str,
    preprocess_sha256: str,
    output_contract_sha256: str,
    memory_contract_sha256: str,
    transport: DinoWanSpatialTransport,
    device: torch.device | str,
) -> NativePatchMemory:
    """Restore stored native tokens while revalidating all static/effective hashes."""

    required = {
        "visual_native_tokens",
        "visual_patch_valid_mask",
        "visual_camera_valid_mask",
        "visual_effective_transport_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"P6 stored-native replay is missing inputs: {missing}.")
    memory = NativePatchMemory(
        tokens=payload["visual_native_tokens"].to(device=device).detach(),
        patch_valid_mask=payload["visual_patch_valid_mask"].to(device=device),
        camera_valid_mask=payload["visual_camera_valid_mask"].to(device=device),
        camera_ids=camera_ids,
        grid=patch_grid,
        source_revision=source_revision,
        weights_sha256=weights_sha256,
        input_contract_sha256=input_contract_sha256,
        preprocess_sha256=preprocess_sha256,
        output_contract_sha256=output_contract_sha256,
        memory_contract_sha256=memory_contract_sha256,
    )
    expected = transport.effective_sha256(
        camera_valid_mask=memory.camera_valid_mask,
        patch_valid_mask=memory.patch_valid_mask,
    )
    actual = _decode_sha256_bytes(payload["visual_effective_transport_sha256"])
    if actual != expected:
        raise ValueError("P6 effective transport hash changed during replay.")
    return memory


def unpack_recompute_camera_batch(
    payload: Mapping[str, torch.Tensor],
    *,
    camera_ids: tuple[str, ...],
    input_contract_sha256: str,
) -> PreparedCameraBatch:
    """Restore official-crop uint8 views for frozen-DINO recomputation."""

    required = {
        "visual_camera_pixels",
        "visual_camera_valid_mask",
        "visual_effective_transport_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"P6 recompute-native replay is missing inputs: {missing}.")
    return PreparedCameraBatch(
        pixels=payload["visual_camera_pixels"].cpu(),
        camera_ids=camera_ids,
        camera_valid_mask=payload["visual_camera_valid_mask"].cpu(),
        input_contract_sha256=input_contract_sha256,
    )


def validate_recomputed_effective_hash(
    payload: Mapping[str, torch.Tensor],
    *,
    memory: NativePatchMemory,
    transport: DinoWanSpatialTransport,
) -> None:
    """Reject recomputed native memory whose sample-specific masks changed."""

    expected = transport.effective_sha256(
        camera_valid_mask=memory.camera_valid_mask,
        patch_valid_mask=memory.patch_valid_mask,
    )
    actual = _decode_sha256_bytes(payload["visual_effective_transport_sha256"])
    if actual != expected:
        raise ValueError("P6 recomputed effective transport hash changed.")
