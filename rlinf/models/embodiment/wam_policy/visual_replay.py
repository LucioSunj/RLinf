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

import hashlib
import json
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
    max_combined_gate_plus_visual_bytes_per_sample: int | None = None
    max_combined_gate_plus_visual_aggregate_bytes: int | None = None

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
        combined_names = (
            "max_combined_gate_plus_visual_bytes_per_sample",
            "max_combined_gate_plus_visual_aggregate_bytes",
        )
        combined_values = tuple(getattr(self, name) for name in combined_names)
        if any(value is None for value in combined_values) and not all(
            value is None for value in combined_values
        ):
            raise ValueError(
                "P6 combined Gate+visual replay caps must be set together."
            )
        if all(value is not None for value in combined_values):
            for name, value in zip(combined_names, combined_values):
                if isinstance(value, bool) or int(value) < 1:
                    raise ValueError(f"P6 `{name}` must be a positive integer.")
                object.__setattr__(self, name, int(value))
            if (
                self.max_combined_gate_plus_visual_aggregate_bytes
                < self.max_combined_gate_plus_visual_bytes_per_sample
            ):
                raise ValueError(
                    "P6 combined Gate+visual aggregate cap cannot be below its "
                    "per-sample cap."
                )
        object.__setattr__(self, "backend", backend)


def visual_replay_static_contract_sha256(
    *,
    config: VisualReplayConfig,
    transport: DinoWanSpatialTransport,
    source_revision: str,
    weights_sha256: str,
    input_contract_sha256: str,
    preprocess_sha256: str,
    output_contract_sha256: str,
    memory_contract_sha256: str,
) -> str:
    """Bind P6 replay rows to the frozen source, geometry, and storage mode."""

    payload = {
        "schema": "fastwam-p6-visual-replay-static-contract-v2",
        "backend": config.backend.value,
        "storage_dtype": config.storage_dtype,
        "camera_ids": list(transport.camera_ids),
        "source_revision": str(source_revision),
        "weights_sha256": str(weights_sha256).lower(),
        "input_contract_sha256": str(input_contract_sha256).lower(),
        "preprocess_sha256": str(preprocess_sha256).lower(),
        "output_contract_sha256": str(output_contract_sha256).lower(),
        "memory_contract_sha256": str(memory_contract_sha256).lower(),
        "spatial_contract_sha256": transport.spatial_contract_sha256,
        "transport_sha256": transport.transport_sha256,
        "max_bytes_per_sample": config.max_bytes_per_sample,
        "max_aggregate_bytes": config.max_aggregate_bytes,
        "max_combined_gate_plus_visual_bytes_per_sample": (
            config.max_combined_gate_plus_visual_bytes_per_sample
        ),
        "max_combined_gate_plus_visual_aggregate_bytes": (
            config.max_combined_gate_plus_visual_aggregate_bytes
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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

    validate_visual_replay_integrity(forward_inputs)
    pinned = (
        forward_inputs if isinstance(forward_inputs, dict) else dict(forward_inputs)
    )
    for name in list(pinned):
        value = pinned[name]
        if name.startswith("visual_") and value.device.type == "cpu":
            pinned[name] = _pin_cpu(value)
    validate_visual_replay_integrity(pinned)
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


def _content_sha256(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    names = sorted(
        name
        for name in payload
        if name.startswith("visual_") and name != "visual_content_sha256"
    )
    if not names:
        raise ValueError("P6 visual replay has no hash-bound tensors.")
    tensors = {name: payload[name] for name in names}
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError("P6 visual replay values must be tensors.")
    batch_sizes = {
        int(tensor.shape[0]) for tensor in tensors.values() if tensor.ndim > 0
    }
    if len(batch_sizes) != 1 or any(tensor.ndim == 0 for tensor in tensors.values()):
        raise ValueError("P6 visual replay tensors must share one batch dimension.")
    batch_size = batch_sizes.pop()
    rows: list[list[int]] = []
    for index in range(batch_size):
        digest = hashlib.sha256()
        digest.update(b"fastwam-p6-visual-replay-content-v1\0")
        for name in names:
            tensor = tensors[name]
            row = tensor[index].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(b"\0")
            digest.update(json.dumps(list(row.shape), separators=(",", ":")).encode())
            digest.update(b"\0")
            digest.update(row.reshape(-1).view(torch.uint8).numpy().tobytes())
            digest.update(b"\0")
        rows.append(list(digest.digest()))
    return torch.tensor(rows, dtype=torch.uint8, device="cpu")


def validate_visual_replay_integrity(
    payload: Mapping[str, torch.Tensor],
) -> None:
    """Reject missing or modified P6 replay content before device transfer."""

    if not any(name.startswith("visual_") for name in payload):
        return
    actual = payload.get("visual_content_sha256")
    if actual is None:
        raise KeyError("P6 visual replay is missing `visual_content_sha256`.")
    expected = _content_sha256(payload)
    if (
        actual.dtype is not torch.uint8
        or actual.shape != expected.shape
        or not torch.equal(actual.detach().cpu(), expected)
    ):
        raise ValueError("P6 visual replay content SHA256 mismatch.")


def _seal_visual_replay(
    payload: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if "visual_content_sha256" in payload:
        raise ValueError("P6 visual replay content hash must be generated internally.")
    payload["visual_content_sha256"] = _pin_cpu(_content_sha256(payload))
    return payload


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


def replay_bytes_by_prefix(
    forward_inputs: Mapping[str, torch.Tensor],
    *,
    prefix: str,
) -> torch.Tensor:
    """Measure row-wise replay bytes after stack/cat/shuffle transport."""

    tensors = {
        name: tensor
        for name, tensor in forward_inputs.items()
        if name.startswith(prefix) and isinstance(tensor, torch.Tensor)
    }
    if not tensors:
        raise KeyError(f"P6 replay has no tensors with prefix {prefix!r}.")
    return _batched_tensor_bytes(tensors)


def validate_visual_forward_input_budget(
    forward_inputs: Mapping[str, torch.Tensor],
    *,
    config: VisualReplayConfig,
    gate_bytes_per_sample: torch.Tensor | None = None,
) -> None:
    """Enforce visual and optional combined Gate+visual replay caps."""

    validate_visual_replay_integrity(forward_inputs)
    visual_bytes = replay_bytes_by_prefix(forward_inputs, prefix="visual_")
    checks: list[tuple[int, int, str]] = [
        (
            int(visual_bytes.max().item()),
            config.max_bytes_per_sample,
            "P6 visual `max_bytes_per_sample`",
        ),
        (
            int(visual_bytes.sum().item()),
            config.max_aggregate_bytes,
            "P6 visual `max_aggregate_bytes`",
        ),
    ]
    if config.max_combined_gate_plus_visual_bytes_per_sample is not None:
        if gate_bytes_per_sample is None:
            try:
                gate_bytes_per_sample = replay_bytes_by_prefix(
                    forward_inputs,
                    prefix="gate_kv_",
                )
            except KeyError:
                gate_bytes_per_sample = torch.zeros_like(visual_bytes)
        gate_bytes = gate_bytes_per_sample.to(device="cpu", dtype=torch.long)
        if gate_bytes.shape != visual_bytes.shape:
            raise ValueError("P6 Gate and visual replay bytes must align as [B].")
        combined = visual_bytes + gate_bytes
        checks.extend(
            [
                (
                    int(combined.max().item()),
                    config.max_combined_gate_plus_visual_bytes_per_sample,
                    "P6 combined Gate+visual per-sample",
                ),
                (
                    int(combined.sum().item()),
                    config.max_combined_gate_plus_visual_aggregate_bytes,
                    "P6 combined Gate+visual aggregate",
                ),
            ]
        )
    for actual, limit, label in checks:
        if actual > int(limit):
            raise MemoryError(f"{label} replay bytes exceed cap: {actual} > {limit}.")


def _enforce_caps(
    payload: Mapping[str, torch.Tensor],
    config: VisualReplayConfig,
) -> None:
    validate_visual_forward_input_budget(payload, config=config)


def pack_visual_replay(
    *,
    config: VisualReplayConfig,
    transport: DinoWanSpatialTransport,
    camera_batch: PreparedCameraBatch,
    memory: NativePatchMemory,
    actor_version: int = 0,
    static_contract_sha256: str | None = None,
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
    static_contract_sha256 = (
        static_contract_sha256
        or visual_replay_static_contract_sha256(
            config=config,
            transport=transport,
            source_revision=memory.source_revision,
            weights_sha256=memory.weights_sha256,
            input_contract_sha256=memory.input_contract_sha256,
            preprocess_sha256=memory.preprocess_sha256,
            output_contract_sha256=memory.output_contract_sha256,
            memory_contract_sha256=memory.memory_contract_sha256,
        )
    )
    common = {
        "visual_camera_valid_mask": _pin_cpu(memory.camera_valid_mask),
        "visual_effective_transport_sha256": _pin_cpu(_sha256_bytes(effective_hashes)),
        "visual_actor_versions": _pin_cpu(
            torch.full(
                (memory.tokens.shape[0],),
                int(actor_version),
                dtype=torch.long,
            )
        ),
        "visual_static_contract_sha256": _pin_cpu(
            _sha256_bytes((static_contract_sha256,) * memory.tokens.shape[0])
        ),
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
        # Actor/static provenance describes the whole consumed route batch,
        # including IDM rows whose visual payload is intentionally absent.
        # Keep this identical to the all-IDM schema and avoid a latent failure
        # as soon as actor_version becomes non-zero.
        scattered["visual_actor_versions"] = _pin_cpu(
            torch.full(
                (full_batch_size,),
                int(actor_version),
                dtype=torch.long,
            )
        )
        scattered["visual_static_contract_sha256"] = _pin_cpu(
            _sha256_bytes((static_contract_sha256,) * full_batch_size)
        )
        route_mask = torch.zeros(full_batch_size, dtype=torch.bool)
        route_mask[indices] = True
        scattered["visual_route_mask"] = _pin_cpu(route_mask)
        payload = scattered
    elif "visual_route_mask" not in payload:
        payload["visual_route_mask"] = _pin_cpu(
            torch.ones(memory.tokens.shape[0], dtype=torch.bool)
        )
    payload = _seal_visual_replay(payload)
    _enforce_caps(payload, config)
    return payload


def empty_visual_replay(
    *,
    config: VisualReplayConfig,
    batch_size: int,
    camera_count: int,
    patch_grid: tuple[int, int],
    camera_hw: tuple[int, int],
    actor_version: int = 0,
    static_contract_sha256: str,
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
        "visual_actor_versions": torch.full(
            (batch_size,), int(actor_version), dtype=torch.long
        ),
        "visual_static_contract_sha256": _sha256_bytes(
            (static_contract_sha256,) * batch_size
        ),
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
    payload = _seal_visual_replay(payload)
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

    validate_visual_replay_integrity(payload)
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

    validate_visual_replay_integrity(payload)
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

    validate_visual_replay_integrity(payload)
    expected = transport.effective_sha256(
        camera_valid_mask=memory.camera_valid_mask,
        patch_valid_mask=memory.patch_valid_mask,
    )
    actual = _decode_sha256_bytes(payload["visual_effective_transport_sha256"])
    if actual != expected:
        raise ValueError("P6 recomputed effective transport hash changed.")
