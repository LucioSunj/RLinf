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

"""Typed, fail-closed replay records for the P7 dual visual reader."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import torch
from fastwam.models.wan22.visual_contracts import (
    DINO_V3_NATIVE_DIM,
    DINO_V3_PATCH_COUNT,
    DINO_V3_PATCH_GRID,
    NativePatchMemory,
    PreparedCameraBatch,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PREFIX = "p7_visual"


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return normalized


def _sha_tensor(digest: str, batch_size: int) -> torch.Tensor:
    value = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8)
    return value[None].expand(batch_size, -1).clone()


def _decode_batch_sha(values: torch.Tensor, *, name: str) -> str:
    if values.ndim != 2 or values.shape[1] != 32 or values.dtype != torch.uint8:
        raise ValueError(f"{name} must have shape [B,32] and uint8 dtype.")
    rows = values.detach().cpu()
    first = bytes(rows[0].tolist()).hex()
    if any(bytes(row.tolist()).hex() != first for row in rows[1:]):
        raise ValueError(f"One replay batch cannot mix {name} values.")
    return _validate_sha256(first, name=name)


def _pin(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise ValueError("P7 replay tensors must be on CPU before pinning.")
    try:
        return tensor.pin_memory()
    except RuntimeError:
        if torch.cuda.is_available():
            raise
        # CPU-only PyTorch builds have no pinned allocator; unit tests may
        # exercise the schema without claiming production pinning evidence.
        return tensor


def pin_dual_visual_forward_inputs(
    forward_inputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Re-pin P7 replay fields after trajectory stack/shuffle copies."""

    pinned = (
        forward_inputs if isinstance(forward_inputs, dict) else dict(forward_inputs)
    )
    for name in list(pinned):
        value = pinned[name]
        if (
            name.startswith(f"{_PREFIX}_")
            and torch.is_tensor(value)
            and value.device.type == "cpu"
        ):
            pinned[name] = _pin(value.contiguous())
    return pinned


def validate_dual_visual_aggregate_bytes(
    forward_inputs: Mapping[str, torch.Tensor],
    *,
    max_bytes_aggregate: int,
) -> int:
    """Validate the complete actor-side P7 replay-buffer allocation."""

    if isinstance(max_bytes_aggregate, bool) or int(max_bytes_aggregate) < 1:
        raise ValueError("P7 aggregate replay cap must be a positive integer.")
    total = sum(
        int(value.numel() * value.element_size())
        for name, value in forward_inputs.items()
        if (name.startswith(f"{_PREFIX}_") or name == "fastwam_p7_visual_proprio")
        and torch.is_tensor(value)
    )
    if total > int(max_bytes_aggregate):
        raise MemoryError(
            "P7 actor replay exceeds max_bytes_aggregate: "
            f"actual={total}, limit={int(max_bytes_aggregate)}."
        )
    return total


class DualVisualReplayBackend(str, Enum):
    """Allowed native-memory replay strategies."""

    RECOMPUTE_NATIVE = "recompute_native"
    STORED_NATIVE = "stored_native"
    STORED_NATIVE_AND_WAN_V = "stored_native_and_wan_v"


@dataclass(frozen=True)
class DualVisualReplayConfig:
    """P7 replay storage/capacity contract with no permissive production default."""

    backend: DualVisualReplayBackend | str
    storage_dtype: str
    pin_memory: bool
    max_bytes_per_sample: int
    max_bytes_aggregate: int
    fail_closed: bool
    wan_v_capacity_passed: bool = False
    wan_v_capacity_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        backend = DualVisualReplayBackend(self.backend)
        if self.storage_dtype != "bfloat16":
            raise ValueError("P7 stored native replay requires BF16 storage.")
        if not self.pin_memory:
            raise ValueError("P7 replay requires pinned CPU storage.")
        if not self.fail_closed:
            raise ValueError("P7 replay must fail closed above its capacity caps.")
        for name in ("max_bytes_per_sample", "max_bytes_aggregate"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"`{name}` must be a positive integer.")
            object.__setattr__(self, name, int(value))
        if self.max_bytes_aggregate < self.max_bytes_per_sample:
            raise ValueError("P7 aggregate cap cannot be smaller than its sample cap.")
        if backend is DualVisualReplayBackend.STORED_NATIVE_AND_WAN_V:
            if not self.wan_v_capacity_passed:
                raise ValueError(
                    "stored_native_and_wan_v requires an explicit capacity PASS."
                )
            if self.wan_v_capacity_artifact_sha256 is None:
                raise ValueError("Wan-V capacity PASS requires its artifact SHA256.")
            _validate_sha256(
                self.wan_v_capacity_artifact_sha256,
                name="Wan-V capacity artifact SHA256",
            )
        elif self.wan_v_capacity_passed or self.wan_v_capacity_artifact_sha256:
            raise ValueError("Wan-V capacity fields are valid only for its backend.")
        object.__setattr__(self, "backend", backend)


@dataclass(frozen=True)
class NativeMemoryIdentity:
    """Static frozen-DINO identity needed to reconstruct stored native memory."""

    camera_ids: tuple[str, ...]
    source_revision: str
    weights_sha256: str
    input_contract_sha256: str
    preprocess_sha256: str
    output_contract_sha256: str
    memory_contract_sha256: str

    def __post_init__(self) -> None:
        camera_ids = tuple(str(item) for item in self.camera_ids)
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Native-memory camera IDs must be non-empty and unique.")
        for field_name in (
            "weights_sha256",
            "input_contract_sha256",
            "preprocess_sha256",
            "output_contract_sha256",
            "memory_contract_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_sha256(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(self, "camera_ids", camera_ids)


def effective_transport_sha256(
    *,
    transport_sha256: str,
    camera_valid_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one 32-byte provenance digest per replay sample."""

    transport_hash = _validate_sha256(transport_sha256, name="Transport SHA256")
    if camera_valid_mask.ndim != 2 or camera_valid_mask.dtype != torch.bool:
        raise ValueError("Camera validity must have shape [B,V] and bool dtype.")
    if (
        target_valid_mask.ndim != 3
        or target_valid_mask.dtype != torch.bool
        or target_valid_mask.shape[:2] != camera_valid_mask.shape
    ):
        raise ValueError("Target validity must have shape [B,V,N_wan] and bool dtype.")
    result = []
    for camera_mask, target_mask in zip(
        camera_valid_mask.detach().cpu(),
        target_valid_mask.detach().cpu(),
        strict=True,
    ):
        digest = hashlib.sha256()
        digest.update(bytes.fromhex(transport_hash))
        digest.update(bytes(camera_mask.to(torch.uint8).tolist()))
        digest.update(bytes(target_mask.to(torch.uint8).flatten().tolist()))
        result.append(torch.tensor(list(digest.digest()), dtype=torch.uint8))
    return torch.stack(result)


@dataclass(frozen=True)
class PackedDualVisualReplay:
    """Batch-first replay payload containing no trainable reader outputs."""

    backend: DualVisualReplayBackend
    present_mask: torch.Tensor
    camera_valid_mask: torch.Tensor
    target_valid_mask: torch.Tensor
    active_camera_count: torch.Tensor
    actor_versions: torch.Tensor
    effective_transport_hash: torch.Tensor
    memory_contract_hash: torch.Tensor
    transport_hash: torch.Tensor
    camera_pixels: torch.Tensor | None = None
    native_tokens: torch.Tensor | None = None
    patch_valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        backend = DualVisualReplayBackend(self.backend)
        if self.present_mask.ndim != 1 or self.present_mask.dtype != torch.bool:
            raise ValueError("P7 present mask must have shape [B] and bool dtype.")
        batch = self.present_mask.shape[0]
        if batch < 1:
            raise ValueError("P7 replay batches must be non-empty.")
        if self.camera_valid_mask.ndim != 2 or self.camera_valid_mask.shape[0] != batch:
            raise ValueError("P7 camera mask must have shape [B,V].")
        if self.camera_valid_mask.dtype != torch.bool:
            raise TypeError("P7 camera mask must use bool dtype.")
        views = self.camera_valid_mask.shape[1]
        if self.target_valid_mask.ndim != 3 or self.target_valid_mask.shape[:2] != (
            batch,
            views,
        ):
            raise ValueError("P7 target mask must have shape [B,V,N_wan].")
        if self.target_valid_mask.dtype != torch.bool:
            raise TypeError("P7 target mask must use bool dtype.")
        if self.active_camera_count.shape != (batch,):
            raise ValueError("P7 active-camera count must have shape [B].")
        expected_count = self.camera_valid_mask.sum(dim=-1).to(torch.long)
        if not torch.equal(self.active_camera_count.to(torch.long), expected_count):
            raise ValueError("P7 active-camera count disagrees with camera validity.")
        if (
            self.actor_versions.shape != (batch,)
            or self.actor_versions.dtype != torch.long
        ):
            raise ValueError("P7 actor versions must have shape [B] and long dtype.")
        if bool((self.actor_versions < 0).any().item()):
            raise ValueError("P7 actor versions must be non-negative.")
        if torch.unique(self.actor_versions).numel() != 1:
            raise ValueError("One P7 replay batch cannot mix behavior actor versions.")
        if bool((expected_count[self.present_mask] < 1).any().item()):
            raise ValueError("Every consumed UNCOND row needs an active P7 camera.")
        for name, value in (
            ("effective transport hash", self.effective_transport_hash),
            ("memory contract hash", self.memory_contract_hash),
            ("transport hash", self.transport_hash),
        ):
            if value.shape != (batch, 32) or value.dtype != torch.uint8:
                raise ValueError(f"P7 {name} must have shape [B,32] and uint8 dtype.")
        if backend is DualVisualReplayBackend.RECOMPUTE_NATIVE:
            if self.camera_pixels is None or any(
                item is not None for item in (self.native_tokens, self.patch_valid_mask)
            ):
                raise ValueError("Recompute-native replay stores only camera pixels.")
            if self.camera_pixels.shape != (batch, views, 3, 224, 224):
                raise ValueError("P7 camera pixels must have shape [B,V,3,224,224].")
            if self.camera_pixels.dtype != torch.uint8:
                raise TypeError("P7 camera pixels must use uint8 dtype.")
        elif backend is DualVisualReplayBackend.STORED_NATIVE:
            if self.camera_pixels is not None:
                raise ValueError(
                    "Stored-native replay must not duplicate camera pixels."
                )
            if self.native_tokens is None or self.patch_valid_mask is None:
                raise ValueError(
                    "Stored-native replay requires tokens and patch masks."
                )
            if self.native_tokens.shape != (
                batch,
                views,
                DINO_V3_PATCH_COUNT,
                DINO_V3_NATIVE_DIM,
            ):
                raise ValueError("Stored native tokens have the wrong shape.")
            if self.native_tokens.dtype is not torch.bfloat16:
                raise TypeError("Stored native tokens must use BF16 dtype.")
            if (
                self.patch_valid_mask.shape
                != (
                    batch,
                    views,
                    DINO_V3_PATCH_COUNT,
                )
                or self.patch_valid_mask.dtype is not torch.bool
            ):
                raise ValueError("Stored native patch mask has the wrong contract.")
            expected_patch_cameras = self.camera_valid_mask & self.present_mask[:, None]
            if not torch.equal(
                self.patch_valid_mask.any(dim=-1), expected_patch_cameras
            ):
                raise ValueError(
                    "Stored patch validity disagrees with route/camera validity."
                )
            absent = ~self.present_mask
            if bool((self.native_tokens[absent] != 0).any().item()):
                raise ValueError("IDM replay rows must not contain DINO native tokens.")
            if bool(self.patch_valid_mask[absent].any().item()):
                raise ValueError(
                    "IDM replay rows must not contain DINO patch validity."
                )
        else:
            raise ValueError(
                "Wan-V replay records are unavailable until their capacity gate passes."
            )
        tensors = self.as_forward_inputs()
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) != 1:
            raise ValueError("P7 replay payload tensors must share one device.")
        object.__setattr__(self, "backend", backend)

    @property
    def batch_size(self) -> int:
        return int(self.present_mask.shape[0])

    def as_forward_inputs(self) -> dict[str, torch.Tensor]:
        result = {
            f"{_PREFIX}_backend": torch.full(
                (self.present_mask.shape[0],),
                list(DualVisualReplayBackend).index(self.backend),
                dtype=torch.uint8,
                device=self.present_mask.device,
            ),
            f"{_PREFIX}_present_mask": self.present_mask,
            f"{_PREFIX}_camera_valid_mask": self.camera_valid_mask,
            f"{_PREFIX}_target_valid_mask": self.target_valid_mask,
            f"{_PREFIX}_active_camera_count": self.active_camera_count,
            f"{_PREFIX}_actor_versions": self.actor_versions,
            f"{_PREFIX}_effective_transport_hash": self.effective_transport_hash,
            f"{_PREFIX}_memory_contract_hash": self.memory_contract_hash,
            f"{_PREFIX}_transport_hash": self.transport_hash,
        }
        if self.camera_pixels is not None:
            result[f"{_PREFIX}_camera_pixels"] = self.camera_pixels
        if self.native_tokens is not None:
            result[f"{_PREFIX}_native_tokens"] = self.native_tokens
        if self.patch_valid_mask is not None:
            result[f"{_PREFIX}_patch_valid_mask"] = self.patch_valid_mask
        return result

    @classmethod
    def from_forward_inputs(
        cls, forward_inputs: Mapping[str, torch.Tensor]
    ) -> PackedDualVisualReplay:
        required = {
            "backend",
            "present_mask",
            "camera_valid_mask",
            "target_valid_mask",
            "active_camera_count",
            "actor_versions",
            "effective_transport_hash",
            "memory_contract_hash",
            "transport_hash",
        }
        values = {
            name.removeprefix(f"{_PREFIX}_"): tensor
            for name, tensor in forward_inputs.items()
            if name.startswith(f"{_PREFIX}_")
        }
        missing = required - set(values)
        if missing:
            raise KeyError(f"P7 replay inputs are missing: {sorted(missing)}.")
        backend_codes = values.pop("backend")
        if backend_codes.ndim != 1 or backend_codes.dtype != torch.uint8:
            raise ValueError("P7 backend codes must have shape [B] and uint8 dtype.")
        if torch.unique(backend_codes).numel() != 1:
            raise ValueError("One P7 replay batch cannot mix backends.")
        code = int(backend_codes[0].item())
        backends = list(DualVisualReplayBackend)
        if not 0 <= code < len(backends):
            raise ValueError("Unknown P7 replay backend code.")
        return cls(backend=backends[code], **values)

    def bytes_per_sample(self) -> torch.Tensor:
        result = torch.zeros(self.batch_size, dtype=torch.long)
        for tensor in self.as_forward_inputs().values():
            if tensor.shape[0] != self.batch_size:
                raise ValueError("Every P7 replay tensor must be batch first.")
            result += tensor[0].numel() * tensor.element_size()
        return result

    def validate_contract(
        self,
        *,
        backend: DualVisualReplayBackend | str,
        memory_contract_sha256: str,
        transport_sha256: str,
    ) -> None:
        """Bind a packed batch to the live reader/replay configuration."""

        # Tensor fields remain mutable even on a frozen dataclass. Re-run all
        # structural/mask invariants before checking their hash-bound identity.
        self.__post_init__()
        if self.backend is not DualVisualReplayBackend(backend):
            raise ValueError("Packed P7 replay backend differs from live config.")
        actual_memory = _decode_batch_sha(
            self.memory_contract_hash, name="memory contract hash"
        )
        if actual_memory != _validate_sha256(
            memory_contract_sha256, name="Expected memory contract SHA256"
        ):
            raise ValueError("Packed P7 native-memory contract mismatch.")
        actual_transport = _decode_batch_sha(self.transport_hash, name="transport hash")
        if actual_transport != _validate_sha256(
            transport_sha256, name="Expected transport SHA256"
        ):
            raise ValueError("Packed P7 transport contract mismatch.")
        expected_effective = effective_transport_sha256(
            transport_sha256=actual_transport,
            camera_valid_mask=self.camera_valid_mask,
            target_valid_mask=self.target_valid_mask,
        )
        if not torch.equal(
            self.effective_transport_hash.detach().cpu(), expected_effective
        ):
            raise ValueError("Packed P7 effective transport provenance mismatch.")

    def native_memory(
        self,
        index: int,
        *,
        identity: NativeMemoryIdentity,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> NativePatchMemory:
        if self.backend is not DualVisualReplayBackend.STORED_NATIVE:
            raise ValueError("Only stored-native replay can materialize native memory.")
        if not bool(self.present_mask[index].item()):
            raise ValueError("IDM rows have no P7 native memory.")
        memory_hash = _decode_batch_sha(
            self.memory_contract_hash, name="memory contract hash"
        )
        if memory_hash != identity.memory_contract_sha256:
            raise ValueError("Stored native memory contract hash mismatch.")
        tokens = self.native_tokens[index : index + 1].to(device=device, dtype=dtype)
        patch_mask = self.patch_valid_mask[index : index + 1].to(device=device)
        camera_mask = self.camera_valid_mask[index : index + 1].to(device=device)
        return NativePatchMemory(
            tokens=tokens.detach(),
            patch_valid_mask=patch_mask,
            camera_valid_mask=camera_mask,
            camera_ids=identity.camera_ids,
            grid=DINO_V3_PATCH_GRID,
            source_revision=identity.source_revision,
            weights_sha256=identity.weights_sha256,
            input_contract_sha256=identity.input_contract_sha256,
            preprocess_sha256=identity.preprocess_sha256,
            output_contract_sha256=identity.output_contract_sha256,
            memory_contract_sha256=identity.memory_contract_sha256,
        )


def pack_dual_visual_replay(
    *,
    config: DualVisualReplayConfig,
    cameras: PreparedCameraBatch,
    present_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
    memory_contract_sha256: str,
    transport_sha256: str,
    actor_version: int,
    native_memories: tuple[NativePatchMemory | None, ...] | None = None,
    auxiliary_bytes_per_sample: int = 0,
) -> PackedDualVisualReplay:
    """Pack pixels or frozen native tokens and enforce both storage caps."""

    batch, views = cameras.camera_valid_mask.shape
    if (
        isinstance(auxiliary_bytes_per_sample, bool)
        or int(auxiliary_bytes_per_sample) < 0
    ):
        raise ValueError("P7 auxiliary replay bytes must be a nonnegative integer.")
    if isinstance(actor_version, bool) or int(actor_version) < 0:
        raise ValueError("P7 replay actor version must be non-negative.")
    present = present_mask.detach().to(device="cpu", dtype=torch.bool)
    if present.shape != (batch,):
        raise ValueError("P7 present mask must align with the camera batch.")
    camera_mask = cameras.camera_valid_mask.detach().cpu()
    target_mask = target_valid_mask.detach().to(device="cpu")
    if target_mask.ndim == 2:
        target_mask = target_mask[None].expand(batch, -1, -1).clone()
    if target_mask.ndim != 3 or target_mask.shape[:2] != (batch, views):
        raise ValueError("P7 target mask must have shape [B,V,N_wan].")
    if target_mask.dtype is not torch.bool:
        raise TypeError("P7 target mask must use bool dtype.")
    memory_hash = _validate_sha256(
        memory_contract_sha256, name="Memory contract SHA256"
    )
    transport_hash = _validate_sha256(transport_sha256, name="Transport SHA256")
    common = {
        "present_mask": present,
        "camera_valid_mask": camera_mask,
        "target_valid_mask": target_mask,
        "active_camera_count": camera_mask.sum(dim=-1).to(torch.long),
        "actor_versions": torch.full((batch,), int(actor_version), dtype=torch.long),
        "effective_transport_hash": effective_transport_sha256(
            transport_sha256=transport_hash,
            camera_valid_mask=camera_mask,
            target_valid_mask=target_mask,
        ),
        "memory_contract_hash": _sha_tensor(memory_hash, batch),
        "transport_hash": _sha_tensor(transport_hash, batch),
    }
    if config.backend is DualVisualReplayBackend.RECOMPUTE_NATIVE:
        if native_memories is not None:
            raise ValueError("Recompute-native replay must not receive native tokens.")
        packed = PackedDualVisualReplay(
            backend=config.backend,
            camera_pixels=cameras.pixels.detach().cpu(),
            **common,
        )
    elif config.backend is DualVisualReplayBackend.STORED_NATIVE:
        if native_memories is None or len(native_memories) != batch:
            raise ValueError("Stored-native replay needs one optional memory per row.")
        tokens = torch.zeros(
            batch,
            views,
            DINO_V3_PATCH_COUNT,
            DINO_V3_NATIVE_DIM,
            dtype=torch.bfloat16,
        )
        patch_mask = torch.zeros(batch, views, DINO_V3_PATCH_COUNT, dtype=torch.bool)
        for index, memory in enumerate(native_memories):
            if bool(present[index].item()) != (memory is not None):
                raise ValueError("P7 route presence and stored memory disagree.")
            if memory is None:
                continue
            if memory.memory_contract_sha256 != memory_hash:
                raise ValueError("P7 stored memory contract hash mismatch.")
            tokens[index].copy_(memory.tokens[0].detach().cpu().to(torch.bfloat16))
            patch_mask[index].copy_(memory.patch_valid_mask[0].detach().cpu())
        packed = PackedDualVisualReplay(
            backend=config.backend,
            native_tokens=tokens,
            patch_valid_mask=patch_mask,
            **common,
        )
    else:
        raise NotImplementedError(
            "Wan-V replay remains blocked until measured capacity is authorized."
        )
    if config.pin_memory:
        packed = PackedDualVisualReplay.from_forward_inputs(
            {
                name: _pin(tensor.contiguous())
                for name, tensor in packed.as_forward_inputs().items()
            }
        )
    sample_bytes = packed.bytes_per_sample() + int(auxiliary_bytes_per_sample)
    if bool((sample_bytes > config.max_bytes_per_sample).any().item()):
        raise MemoryError(
            "P7 replay exceeds max_bytes_per_sample: "
            f"max={int(sample_bytes.max())}, limit={config.max_bytes_per_sample}."
        )
    aggregate = int(sample_bytes.sum().item())
    if aggregate > config.max_bytes_aggregate:
        raise MemoryError(
            "P7 replay exceeds max_bytes_aggregate: "
            f"actual={aggregate}, limit={config.max_bytes_aggregate}."
        )
    return packed
