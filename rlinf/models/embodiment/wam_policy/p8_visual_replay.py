# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Strict P8 frozen-source replay and aggregate byte-budget contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from fastwam.models.wan22.visual_contracts import NativePatchMemory, validate_sha256
from fastwam.models.wan22.wan_current_refiner import WanCurrentLayerSource


class P8VisualReplayBackend(str, Enum):
    """Allowed P8 source replay implementations."""

    STORED_NATIVE = "stored_native"
    RECOMPUTE_NATIVE = "recompute_native"


@dataclass(frozen=True)
class P8VisualReplayConfig:
    """Independent P8 and combined Gate+P8 fail-closed storage limits."""

    backend: P8VisualReplayBackend = P8VisualReplayBackend.STORED_NATIVE
    storage_dtype: str = "bfloat16"
    pin_memory: bool = True
    max_bytes_per_sample: int | None = None
    max_aggregate_bytes: int | None = None
    max_combined_gate_plus_p8_bytes_per_sample: int | None = None
    max_combined_gate_plus_p8_aggregate_bytes: int | None = None
    fail_closed: bool = True

    def __post_init__(self) -> None:
        backend = P8VisualReplayBackend(self.backend)
        object.__setattr__(self, "backend", backend)
        if backend is P8VisualReplayBackend.RECOMPUTE_NATIVE:
            raise NotImplementedError(
                "P8 recompute_native replay is not implemented in the A0/KV MVP; "
                "use stored_native until recompute parity is validated."
            )
        if self.storage_dtype != "bfloat16":
            raise ValueError("P8 MVP visual source storage must use bfloat16.")
        if self.fail_closed is not True:
            raise ValueError("P8 visual replay must remain fail closed.")
        for name in (
            "max_bytes_per_sample",
            "max_aggregate_bytes",
            "max_combined_gate_plus_p8_bytes_per_sample",
            "max_combined_gate_plus_p8_aggregate_bytes",
        ):
            value = getattr(self, name)
            if value is None or isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"Enabled P8 replay requires positive `{name}`.")
        if not self.pin_memory:
            raise ValueError("P8 MVP visual replay requires pinned CPU storage.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> P8VisualReplayConfig:
        """Parse the enabled replay contract without hidden or unknown fields."""

        required = {
            "backend",
            "storage_dtype",
            "pin_memory",
            "max_bytes_per_sample",
            "max_aggregate_bytes",
            "max_combined_gate_plus_p8_bytes_per_sample",
            "max_combined_gate_plus_p8_aggregate_bytes",
            "fail_closed",
        }
        if set(payload) != required:
            raise ValueError(
                "Invalid P8 visual replay fields; "
                f"missing={sorted(required - set(payload))}, "
                f"unknown={sorted(set(payload) - required)}."
            )
        return cls(**dict(payload))


@dataclass(frozen=True)
class P8VisualReplaySpec:
    """Static shape/provenance needed for route-aligned empty IDM slots."""

    layer_indices: tuple[int, ...]
    camera_ids: tuple[str, ...]
    current_frame_video_tokens: int
    wan_hidden_dim: int
    kv_dim: int
    rope_shape: tuple[int, ...]
    rope_complex_dtype: str
    memory_contract_sha256: str
    source_contract_sha256: str
    native_source_revision: str
    native_weights_sha256: str
    native_input_contract_sha256: str
    native_preprocess_sha256: str
    native_output_contract_sha256: str

    def __post_init__(self) -> None:
        layers = tuple(int(index) for index in self.layer_indices)
        cameras = tuple(str(item) for item in self.camera_ids)
        if not layers or len(layers) > 2 or tuple(sorted(set(layers))) != layers:
            raise ValueError("P8 replay requires one or two sorted selected layers.")
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("P8 replay camera IDs must be non-empty and unique.")
        for name in ("current_frame_video_tokens", "wan_hidden_dim", "kv_dim"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"P8 replay `{name}` must be positive.")
        if not self.rope_shape or self.rope_shape[0] != self.current_frame_video_tokens:
            raise ValueError("P8 replay RoPE shape must cover the current prefix.")
        rope_complex_dtype = str(self.rope_complex_dtype).removeprefix("torch.")
        if rope_complex_dtype not in {"complex64", "complex128"}:
            raise ValueError(
                "P8 replay RoPE source dtype must be complex64 or complex128."
            )
        object.__setattr__(self, "layer_indices", layers)
        object.__setattr__(self, "camera_ids", cameras)
        object.__setattr__(self, "rope_complex_dtype", rope_complex_dtype)
        object.__setattr__(
            self,
            "memory_contract_sha256",
            validate_sha256(
                self.memory_contract_sha256,
                label="P8 replay native-memory contract SHA256",
            ),
        )
        revision = str(self.native_source_revision).strip().lower()
        if not revision:
            raise ValueError("P8 replay native source revision cannot be empty.")
        object.__setattr__(self, "native_source_revision", revision)
        for name, label in (
            ("native_weights_sha256", "P8 native weights SHA256"),
            ("native_input_contract_sha256", "P8 native input contract SHA256"),
            ("native_preprocess_sha256", "P8 native preprocess SHA256"),
            ("native_output_contract_sha256", "P8 native output contract SHA256"),
        ):
            object.__setattr__(
                self,
                name,
                validate_sha256(getattr(self, name), label=label),
            )
        object.__setattr__(
            self,
            "source_contract_sha256",
            validate_sha256(
                self.source_contract_sha256,
                label="P8 replay Wan-source contract SHA256",
            ),
        )


@dataclass(frozen=True)
class P8FrozenVisualSource:
    """Canonical BF16 source used to construct rollout or replay shadows."""

    memory: NativePatchMemory
    layers: tuple[WanCurrentLayerSource, ...]
    actor_version: int

    def __post_init__(self) -> None:
        if self.actor_version < 0:
            raise ValueError("P8 source actor version must be non-negative.")
        if not self.layers:
            raise ValueError("P8 source requires selected Wan layers.")
        if self.memory.tokens.dtype is not torch.bfloat16:
            raise TypeError("P8 native memory must be canonicalized to BF16.")
        for layer in self.layers:
            tensors = (
                layer.hidden_current,
                layer.attention_input_current,
                layer.key_pre_norm_current,
                layer.base_key_current,
                layer.base_value_current,
            )
            if any(tensor.dtype is not torch.bfloat16 for tensor in tensors):
                raise TypeError("P8 Wan sources must be canonicalized to BF16.")
            if not layer.rope_freqs_current.is_complex():
                raise TypeError("P8 Wan RoPE sources must use a complex dtype.")


def canonicalize_p8_source_bf16(
    *,
    memory: NativePatchMemory,
    layers: Sequence[WanCurrentLayerSource],
    actor_version: int,
) -> P8FrozenVisualSource:
    """Canonicalize at source creation, before rollout retrieval and storage."""

    tokens = memory.tokens.detach().to(dtype=torch.bfloat16)
    canonical_memory = NativePatchMemory(
        tokens=tokens,
        patch_valid_mask=memory.patch_valid_mask.detach(),
        camera_valid_mask=memory.camera_valid_mask.detach(),
        camera_ids=memory.camera_ids,
        grid=memory.grid,
        source_revision=memory.source_revision,
        weights_sha256=memory.weights_sha256,
        input_contract_sha256=memory.input_contract_sha256,
        preprocess_sha256=memory.preprocess_sha256,
        output_contract_sha256=memory.output_contract_sha256,
        memory_contract_sha256=memory.memory_contract_sha256,
    )
    canonical_layers = tuple(
        WanCurrentLayerSource(
            layer_index=layer.layer_index,
            hidden_current=layer.hidden_current.detach().to(dtype=torch.bfloat16),
            attention_input_current=(
                layer.attention_input_current.detach().to(dtype=torch.bfloat16)
            ),
            key_pre_norm_current=(
                layer.key_pre_norm_current.detach().to(dtype=torch.bfloat16)
            ),
            base_key_current=layer.base_key_current.detach().to(dtype=torch.bfloat16),
            base_value_current=(
                layer.base_value_current.detach().to(dtype=torch.bfloat16)
            ),
            rope_freqs_current=layer.rope_freqs_current.detach(),
            camera_index_current=layer.camera_index_current.detach(),
            current_frame_video_tokens=layer.current_frame_video_tokens,
            source_contract_sha256=layer.source_contract_sha256,
        )
        for layer in layers
    )
    return P8FrozenVisualSource(
        memory=canonical_memory,
        layers=canonical_layers,
        actor_version=actor_version,
    )


_CONTENT_TENSOR_NAMES = (
    "present",
    "actor_versions",
    "native_tokens",
    "patch_valid_mask",
    "camera_valid_mask",
    "hidden_current",
    "attention_input_current",
    "key_pre_norm_current",
    "base_key_current",
    "base_value_current",
    "rope_freqs_real_current",
    "rope_freqs_imag_current",
    "camera_index_current",
)
_HASH_BYTES = hashlib.sha256().digest_size
_TENSOR_NAMES = (
    *_CONTENT_TENSOR_NAMES,
    "contract_sha256",
    "integrity_sha256",
)
_INTEGRITY_SCHEMA = "fastwam-p8-stored-native-replay-integrity-v2-bf16-rope"


def _replay_contract_payload(spec: P8VisualReplaySpec) -> dict[str, Any]:
    """Return the canonical metadata/provenance bound to every replay row."""

    return {
        "schema": _INTEGRITY_SCHEMA,
        "layer_indices": list(spec.layer_indices),
        "camera_ids": list(spec.camera_ids),
        "current_frame_video_tokens": spec.current_frame_video_tokens,
        "wan_hidden_dim": spec.wan_hidden_dim,
        "kv_dim": spec.kv_dim,
        "rope_shape": list(spec.rope_shape),
        "rope_complex_dtype": spec.rope_complex_dtype,
        "rope_storage_encoding": "separate_real_imag_bfloat16",
        "native_grid": [14, 14],
        "native_patch_count": 196,
        "native_width": 384,
        "memory_contract_sha256": spec.memory_contract_sha256,
        "source_contract_sha256": spec.source_contract_sha256,
        "native_source_revision": spec.native_source_revision,
        "native_weights_sha256": spec.native_weights_sha256,
        "native_input_contract_sha256": spec.native_input_contract_sha256,
        "native_preprocess_sha256": spec.native_preprocess_sha256,
        "native_output_contract_sha256": spec.native_output_contract_sha256,
        "content_tensor_order": list(_CONTENT_TENSOR_NAMES),
    }


def _replay_contract_digest(spec: P8VisualReplaySpec) -> bytes:
    payload = json.dumps(
        _replay_contract_payload(spec),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _update_digest_with_tensor(
    digest: Any,
    *,
    name: str,
    tensor: torch.Tensor,
) -> None:
    """Hash one tensor without depending on its source device or strides."""

    header = json.dumps(
        {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
    digest.update(header)
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    if raw.device.type != "cpu":
        raw = raw.cpu()
    for chunk in raw.split(1024 * 1024):
        digest.update(chunk.numpy().tobytes())


def _validate_integrity_payload(
    payload: Mapping[str, torch.Tensor],
) -> tuple[int, torch.device]:
    missing = [name for name in _TENSOR_NAMES if name not in payload]
    if missing:
        raise KeyError(f"P8 visual replay is missing integrity tensors: {missing}.")
    if any(not torch.is_tensor(payload[name]) for name in _TENSOR_NAMES):
        raise TypeError("Every P8 visual replay integrity field must be a tensor.")
    present = payload["present"]
    if present.ndim != 1 or present.shape[0] < 1:
        raise ValueError("P8 replay integrity requires a non-empty batch dimension.")
    batch = int(present.shape[0])
    for name in _TENSOR_NAMES:
        tensor = payload[name]
        if tensor.ndim < 1 or tensor.shape[0] != batch:
            raise ValueError(
                f"P8 replay integrity tensor `{name}` must be batch first with B={batch}."
            )
    for name in ("contract_sha256", "integrity_sha256"):
        tensor = payload[name]
        if tensor.shape != (batch, _HASH_BYTES) or tensor.dtype is not torch.uint8:
            raise ValueError(
                f"P8 replay `{name}` must have shape [B,{_HASH_BYTES}] and uint8 dtype."
            )
    devices = {payload[name].device for name in _TENSOR_NAMES}
    if len(devices) != 1:
        raise ValueError("P8 replay payload tensors must share one device.")
    return batch, devices.pop()


def _content_integrity_tensor(
    payload: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Recompute one content digest per row from stored contract and raw tensors."""

    batch, _device = _validate_integrity_payload(payload)
    contract_hashes = payload["contract_sha256"].detach().cpu()
    result = []
    for index in range(batch):
        digest = hashlib.sha256()
        digest.update(_INTEGRITY_SCHEMA.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes(contract_hashes[index].tolist()))
        for name in _CONTENT_TENSOR_NAMES:
            _update_digest_with_tensor(
                digest,
                name=name,
                tensor=payload[name][index],
            )
        result.append(torch.tensor(list(digest.digest()), dtype=torch.uint8))
    return torch.stack(result)


def _verify_content_integrity(payload: Mapping[str, torch.Tensor]) -> None:
    """Fail closed when transported P8 content differs from its stored digest."""

    expected = _content_integrity_tensor(payload)
    actual = payload["integrity_sha256"].detach().cpu()
    if not torch.equal(actual, expected):
        raise ValueError("P8 replay content integrity SHA256 mismatch.")


def _verify_contract_integrity(
    spec: P8VisualReplaySpec,
    payload: Mapping[str, torch.Tensor],
) -> None:
    """Bind the transported content digest to the live typed replay spec."""

    batch, _device = _validate_integrity_payload(payload)
    expected = torch.tensor(
        list(_replay_contract_digest(spec)),
        dtype=torch.uint8,
    ).expand(batch, -1)
    actual = payload["contract_sha256"].detach().cpu()
    if not torch.equal(actual, expected):
        raise ValueError("P8 replay metadata/provenance contract SHA256 mismatch.")
    _verify_content_integrity(payload)


def _build_integrity_hashes(
    *,
    spec: P8VisualReplaySpec,
    payload: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Create compact contract/content hashes on the payload's source device."""

    present = payload.get("present")
    if not torch.is_tensor(present) or present.ndim != 1 or present.shape[0] < 1:
        raise ValueError("P8 replay hashing requires a non-empty presence tensor.")
    batch = int(present.shape[0])
    device = present.device
    contract = (
        torch.tensor(
            list(_replay_contract_digest(spec)),
            dtype=torch.uint8,
        )
        .expand(batch, -1)
        .clone()
    )
    hash_payload = {
        **payload,
        "contract_sha256": contract.to(device=device),
        "integrity_sha256": torch.zeros(
            batch,
            _HASH_BYTES,
            dtype=torch.uint8,
            device=device,
        ),
    }
    integrity = _content_integrity_tensor(hash_payload)
    return {
        "contract_sha256": hash_payload["contract_sha256"],
        "integrity_sha256": integrity.to(device=device),
    }


@dataclass(frozen=True)
class PackedP8VisualReplay:
    """Batch-first tensor payload compatible with RLinf trajectory sharding."""

    spec: P8VisualReplaySpec
    present: torch.Tensor
    actor_versions: torch.Tensor
    native_tokens: torch.Tensor
    patch_valid_mask: torch.Tensor
    camera_valid_mask: torch.Tensor
    hidden_current: torch.Tensor
    attention_input_current: torch.Tensor
    key_pre_norm_current: torch.Tensor
    base_key_current: torch.Tensor
    base_value_current: torch.Tensor
    rope_freqs_real_current: torch.Tensor
    rope_freqs_imag_current: torch.Tensor
    camera_index_current: torch.Tensor
    contract_sha256: torch.Tensor
    integrity_sha256: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.present.shape[0]
        if batch < 1:
            raise ValueError("P8 replay batches must be non-empty.")
        layers = len(self.spec.layer_indices)
        current = self.spec.current_frame_video_tokens
        views = len(self.spec.camera_ids)
        if self.present.dtype is not torch.bool or self.present.ndim != 1:
            raise TypeError("P8 replay presence must be a one-dimensional bool tensor.")
        if (
            self.actor_versions.shape != (batch,)
            or self.actor_versions.dtype != torch.long
        ):
            raise ValueError("P8 replay actor versions must be int64 [B].")
        expected = {
            "native_tokens": (batch, views, 196, 384),
            "patch_valid_mask": (batch, views, 196),
            "camera_valid_mask": (batch, views),
            "hidden_current": (batch, layers, current, self.spec.wan_hidden_dim),
            "attention_input_current": (
                batch,
                layers,
                current,
                self.spec.wan_hidden_dim,
            ),
            "key_pre_norm_current": (batch, layers, current, self.spec.kv_dim),
            "base_key_current": (batch, layers, current, self.spec.kv_dim),
            "base_value_current": (batch, layers, current, self.spec.kv_dim),
            "rope_freqs_real_current": (batch, *self.spec.rope_shape),
            "rope_freqs_imag_current": (batch, *self.spec.rope_shape),
            "camera_index_current": (batch, current),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(
                    f"P8 replay `{name}` has shape {tuple(getattr(self, name).shape)}, "
                    f"expected {shape}."
                )
        for name in (
            "native_tokens",
            "hidden_current",
            "attention_input_current",
            "key_pre_norm_current",
            "base_key_current",
            "base_value_current",
            "rope_freqs_real_current",
            "rope_freqs_imag_current",
        ):
            if getattr(self, name).dtype is not torch.bfloat16:
                raise TypeError(f"P8 replay `{name}` must use bfloat16.")
        if self.patch_valid_mask.dtype is not torch.bool:
            raise TypeError("P8 replay patch masks must use bool dtype.")
        if self.camera_valid_mask.dtype is not torch.bool:
            raise TypeError("P8 replay camera masks must use bool dtype.")
        if self.camera_index_current.dtype is not torch.long:
            raise TypeError("P8 replay camera indices must use int64 dtype.")
        for name in ("contract_sha256", "integrity_sha256"):
            tensor = getattr(self, name)
            if tensor.shape != (batch, _HASH_BYTES) or tensor.dtype is not torch.uint8:
                raise ValueError(
                    f"P8 replay `{name}` must have shape [B,{_HASH_BYTES}] "
                    "and uint8 dtype."
                )
        if bool((self.present & (self.actor_versions < 0)).any().item()):
            raise ValueError("Present P8 sources require non-negative actor versions.")
        expected_patch_cameras = self.camera_valid_mask & self.present[:, None]
        if not torch.equal(
            self.patch_valid_mask.any(dim=-1),
            expected_patch_cameras,
        ):
            raise ValueError(
                "P8 replay patch validity disagrees with route/camera validity."
            )
        absent = ~self.present
        if bool(absent.any().item()):
            for name in (
                "native_tokens",
                "patch_valid_mask",
                "camera_valid_mask",
                "hidden_current",
                "attention_input_current",
                "key_pre_norm_current",
                "base_key_current",
                "base_value_current",
                "rope_freqs_real_current",
                "rope_freqs_imag_current",
                "camera_index_current",
            ):
                if bool((getattr(self, name)[absent] != 0).any().item()):
                    raise ValueError(
                        "Absent IDM P8 replay slots must contain exact zeros."
                    )
        _verify_contract_integrity(
            self.spec,
            {name: getattr(self, name) for name in _TENSOR_NAMES},
        )

    @property
    def batch_size(self) -> int:
        return int(self.present.shape[0])

    def as_forward_inputs(self) -> dict[str, torch.Tensor]:
        return {f"p8_visual_{name}": getattr(self, name) for name in _TENSOR_NAMES}

    def validate_integrity(self) -> None:
        """Revalidate mutable tensor contents before live replay materialization."""

        self.__post_init__()

    @classmethod
    def from_forward_inputs(
        cls,
        forward_inputs: Mapping[str, torch.Tensor],
        *,
        spec: P8VisualReplaySpec,
    ) -> PackedP8VisualReplay:
        prefixed = {
            name.removeprefix("p8_visual_")
            for name in forward_inputs
            if name.startswith("p8_visual_")
        }
        unknown = sorted(prefixed - set(_TENSOR_NAMES))
        if unknown:
            raise KeyError(f"P8 visual replay has unknown tensors: {unknown}.")
        missing = [
            name for name in _TENSOR_NAMES if f"p8_visual_{name}" not in forward_inputs
        ]
        if missing:
            raise KeyError(f"P8 visual replay is missing tensors: {missing}.")
        return cls(
            spec=spec,
            **{name: forward_inputs[f"p8_visual_{name}"] for name in _TENSOR_NAMES},
        )

    def bytes_per_sample(self) -> torch.Tensor:
        result = torch.zeros(self.batch_size, dtype=torch.long)
        for name in _TENSOR_NAMES:
            tensor = getattr(self, name)
            if tensor.ndim and tensor.shape[0] == self.batch_size:
                result += tensor[0].numel() * tensor.element_size()
        return result

    def materialize_sample(
        self,
        index: int,
        *,
        device: torch.device | str,
        expected_actor_version: int,
    ) -> P8FrozenVisualSource:
        # Tensor contents remain mutable even on a frozen dataclass. Validate
        # before the first `.to(target)` so corrupt replay never reaches a GPU
        # or any frozen runtime asset.
        self.validate_integrity()
        if not 0 <= index < self.batch_size:
            raise IndexError(index)
        if not bool(self.present[index].item()):
            raise ValueError("IDM replay slots do not contain P8 visual sources.")
        actor_version = int(self.actor_versions[index].item())
        if actor_version != expected_actor_version:
            raise ValueError("P8 replay source actor version mismatch.")
        target = torch.device(device)
        component_dtype = {
            "complex64": torch.float32,
            "complex128": torch.float64,
        }[self.spec.rope_complex_dtype]
        rope_freqs_current = torch.complex(
            self.rope_freqs_real_current[index].to(
                device=target,
                dtype=component_dtype,
            ),
            self.rope_freqs_imag_current[index].to(
                device=target,
                dtype=component_dtype,
            ),
        )
        expected_rope_dtype = {
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }[self.spec.rope_complex_dtype]
        if (
            rope_freqs_current.dtype is not expected_rope_dtype
            or tuple(rope_freqs_current.shape) != self.spec.rope_shape
        ):
            raise RuntimeError("P8 replay failed to reconstruct the typed Wan RoPE.")
        memory = NativePatchMemory(
            tokens=self.native_tokens[index : index + 1].to(target),
            patch_valid_mask=self.patch_valid_mask[index : index + 1].to(target),
            camera_valid_mask=self.camera_valid_mask[index : index + 1].to(target),
            camera_ids=self.spec.camera_ids,
            grid=(14, 14),
            source_revision=self.spec.native_source_revision,
            weights_sha256=self.spec.native_weights_sha256,
            input_contract_sha256=self.spec.native_input_contract_sha256,
            preprocess_sha256=self.spec.native_preprocess_sha256,
            output_contract_sha256=self.spec.native_output_contract_sha256,
            memory_contract_sha256=self.spec.memory_contract_sha256,
        )
        layers = tuple(
            WanCurrentLayerSource(
                layer_index=layer_index,
                hidden_current=self.hidden_current[index : index + 1, offset].to(
                    target
                ),
                attention_input_current=(
                    self.attention_input_current[index : index + 1, offset].to(target)
                ),
                key_pre_norm_current=(
                    self.key_pre_norm_current[index : index + 1, offset].to(target)
                ),
                base_key_current=(
                    self.base_key_current[index : index + 1, offset].to(target)
                ),
                base_value_current=(
                    self.base_value_current[index : index + 1, offset].to(target)
                ),
                rope_freqs_current=rope_freqs_current,
                camera_index_current=(
                    self.camera_index_current[index : index + 1].to(target)
                ),
                current_frame_video_tokens=self.spec.current_frame_video_tokens,
                source_contract_sha256=self.spec.source_contract_sha256,
            )
            for offset, layer_index in enumerate(self.spec.layer_indices)
        )
        return P8FrozenVisualSource(
            memory=memory,
            layers=layers,
            actor_version=actor_version,
        )


def pack_p8_visual_sources(
    sources: Sequence[P8FrozenVisualSource | None],
    *,
    spec: P8VisualReplaySpec,
) -> PackedP8VisualReplay:
    """Pack route-aligned sources; IDM slots remain zero and never materialize."""

    if not sources:
        raise ValueError("P8 replay packing requires a non-empty batch.")
    real = next((source for source in sources if source is not None), None)
    if real is not None:
        example_device = real.memory.tokens.device
    else:
        example_device = torch.device("cpu")
    batch = len(sources)
    layers = len(spec.layer_indices)
    current = spec.current_frame_video_tokens
    views = len(spec.camera_ids)
    payload: dict[str, torch.Tensor] = {
        "present": torch.zeros(batch, dtype=torch.bool, device=example_device),
        "actor_versions": torch.full(
            (batch,), -1, dtype=torch.long, device=example_device
        ),
        "native_tokens": torch.zeros(
            batch, views, 196, 384, dtype=torch.bfloat16, device=example_device
        ),
        "patch_valid_mask": torch.zeros(
            batch, views, 196, dtype=torch.bool, device=example_device
        ),
        "camera_valid_mask": torch.zeros(
            batch, views, dtype=torch.bool, device=example_device
        ),
        "hidden_current": torch.zeros(
            batch,
            layers,
            current,
            spec.wan_hidden_dim,
            dtype=torch.bfloat16,
            device=example_device,
        ),
        "attention_input_current": torch.zeros(
            batch,
            layers,
            current,
            spec.wan_hidden_dim,
            dtype=torch.bfloat16,
            device=example_device,
        ),
        "key_pre_norm_current": torch.zeros(
            batch,
            layers,
            current,
            spec.kv_dim,
            dtype=torch.bfloat16,
            device=example_device,
        ),
        "base_key_current": torch.zeros(
            batch,
            layers,
            current,
            spec.kv_dim,
            dtype=torch.bfloat16,
            device=example_device,
        ),
        "base_value_current": torch.zeros(
            batch,
            layers,
            current,
            spec.kv_dim,
            dtype=torch.bfloat16,
            device=example_device,
        ),
        "rope_freqs_real_current": torch.zeros(
            (batch, *spec.rope_shape), dtype=torch.bfloat16, device=example_device
        ),
        "rope_freqs_imag_current": torch.zeros(
            (batch, *spec.rope_shape), dtype=torch.bfloat16, device=example_device
        ),
        "camera_index_current": torch.zeros(
            batch, current, dtype=torch.long, device=example_device
        ),
    }
    for batch_index, source in enumerate(sources):
        if source is None:
            continue
        if source.memory.camera_ids != spec.camera_ids:
            raise ValueError("P8 source camera order differs from replay spec.")
        if source.memory.memory_contract_sha256 != spec.memory_contract_sha256:
            raise ValueError("P8 source native-memory hash differs from replay spec.")
        identity = (
            source.memory.source_revision,
            source.memory.weights_sha256,
            source.memory.input_contract_sha256,
            source.memory.preprocess_sha256,
            source.memory.output_contract_sha256,
        )
        expected_identity = (
            spec.native_source_revision,
            spec.native_weights_sha256,
            spec.native_input_contract_sha256,
            spec.native_preprocess_sha256,
            spec.native_output_contract_sha256,
        )
        if identity != expected_identity:
            raise ValueError(
                "P8 source native-memory provenance differs from replay spec."
            )
        if tuple(layer.layer_index for layer in source.layers) != spec.layer_indices:
            raise ValueError("P8 source layer order differs from replay spec.")
        payload["present"][batch_index] = True
        payload["actor_versions"][batch_index] = source.actor_version
        payload["native_tokens"][batch_index] = source.memory.tokens[0]
        payload["patch_valid_mask"][batch_index] = source.memory.patch_valid_mask[0]
        payload["camera_valid_mask"][batch_index] = source.memory.camera_valid_mask[0]
        payload["camera_index_current"][batch_index] = source.layers[
            0
        ].camera_index_current[0]
        source_rope = source.layers[0].rope_freqs_current
        if (
            not source_rope.is_complex()
            or tuple(source_rope.shape) != spec.rope_shape
            or str(source_rope.dtype).removeprefix("torch.") != spec.rope_complex_dtype
        ):
            raise TypeError("P8 source RoPE dtype/shape differs from replay spec.")
        payload["rope_freqs_real_current"][batch_index] = source_rope.real.to(
            dtype=torch.bfloat16
        )
        payload["rope_freqs_imag_current"][batch_index] = source_rope.imag.to(
            dtype=torch.bfloat16
        )
        for layer_offset, layer in enumerate(source.layers):
            if layer.source_contract_sha256 != spec.source_contract_sha256:
                raise ValueError("P8 source Wan hash differs from replay spec.")
            if not torch.equal(
                layer.camera_index_current,
                source.layers[0].camera_index_current,
            ) or not torch.equal(
                layer.rope_freqs_current,
                source.layers[0].rope_freqs_current,
            ):
                raise ValueError("P8 layer-invariant camera/RoPE payload diverged.")
            for name in (
                "hidden_current",
                "attention_input_current",
                "key_pre_norm_current",
                "base_key_current",
                "base_value_current",
            ):
                payload[name][batch_index, layer_offset] = getattr(layer, name)[0]
    payload.update(_build_integrity_hashes(spec=spec, payload=payload))
    return PackedP8VisualReplay(spec=spec, **payload)


def validate_p8_replay_bytes(
    *,
    p8_bytes_per_sample: torch.Tensor,
    gate_bytes_per_sample: torch.Tensor,
    config: P8VisualReplayConfig,
) -> None:
    """Enforce all independent and combined sample/batch aggregate caps."""

    p8 = p8_bytes_per_sample.to(device="cpu", dtype=torch.long)
    gate = gate_bytes_per_sample.to(device="cpu", dtype=torch.long)
    if p8.ndim != 1 or gate.shape != p8.shape:
        raise ValueError("Gate and P8 replay bytes must be aligned [B] tensors.")
    combined = p8 + gate
    checks = (
        (int(p8.max()), config.max_bytes_per_sample, "P8 per-sample"),
        (int(p8.sum()), config.max_aggregate_bytes, "P8 aggregate"),
        (
            int(combined.max()),
            config.max_combined_gate_plus_p8_bytes_per_sample,
            "combined Gate+P8 per-sample",
        ),
        (
            int(combined.sum()),
            config.max_combined_gate_plus_p8_aggregate_bytes,
            "combined Gate+P8 aggregate",
        ),
    )
    for actual, limit, label in checks:
        if actual > int(limit):
            raise MemoryError(f"{label} replay bytes exceed cap: {actual} > {limit}.")


def replay_bytes_by_prefix(
    forward_inputs: Mapping[str, torch.Tensor],
    *,
    prefix: str,
) -> torch.Tensor:
    """Measure a flattened trajectory payload after stack/cat/shuffle."""

    tensors = [
        tensor
        for name, tensor in forward_inputs.items()
        if name.startswith(prefix) and torch.is_tensor(tensor) and tensor.ndim > 0
    ]
    if not tensors:
        raise KeyError(f"Replay payload has no tensors with prefix {prefix!r}.")
    batch_sizes = {int(tensor.shape[0]) for tensor in tensors}
    if len(batch_sizes) != 1:
        raise ValueError(f"Replay tensors with prefix {prefix!r} have mixed batches.")
    batch = batch_sizes.pop()
    result = torch.zeros(batch, dtype=torch.long)
    for tensor in tensors:
        result += tensor[0].numel() * tensor.element_size()
    return result


def validate_p8_forward_input_integrity(
    forward_inputs: Mapping[str, torch.Tensor],
) -> None:
    """Validate transported content before actor-side device transfer.

    This transport-level check uses the stored contract digest, so it survives
    trajectory stack/cat/shuffle without needing to construct frozen assets.
    Live replay additionally binds that digest to its typed
    :class:`P8VisualReplaySpec` before materializing a sample.
    """

    values = {
        name.removeprefix("p8_visual_"): tensor
        for name, tensor in forward_inputs.items()
        if name.startswith("p8_visual_")
    }
    if not values:
        return
    unknown = sorted(set(values) - set(_TENSOR_NAMES))
    if unknown:
        raise KeyError(f"P8 visual replay has unknown tensors: {unknown}.")
    missing = sorted(set(_TENSOR_NAMES) - set(values))
    if missing:
        raise KeyError(f"P8 visual replay is missing tensors: {missing}.")
    _verify_content_integrity(values)


def validate_p8_forward_input_budget(
    forward_inputs: Mapping[str, torch.Tensor],
    *,
    config: P8VisualReplayConfig,
) -> None:
    """Enforce aggregate limits on the complete actor-side rollout payload."""

    validate_p8_forward_input_integrity(forward_inputs)
    p8_bytes = replay_bytes_by_prefix(forward_inputs, prefix="p8_visual_")
    try:
        gate_bytes = replay_bytes_by_prefix(forward_inputs, prefix="gate_kv_")
    except KeyError:
        gate_bytes = torch.zeros_like(p8_bytes)
    validate_p8_replay_bytes(
        p8_bytes_per_sample=p8_bytes,
        gate_bytes_per_sample=gate_bytes,
        config=config,
    )


def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        return tensor
    try:
        return tensor.contiguous().pin_memory()
    except RuntimeError:
        if torch.cuda.is_available():
            raise RuntimeError(
                "P8 pinned replay allocation failed while CUDA is available."
            )
        return tensor.contiguous()


def pin_p8_visual_forward_inputs(
    forward_inputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Re-pin P8 payload after stack/cat/shuffle copies."""

    validate_p8_forward_input_integrity(forward_inputs)
    result = (
        forward_inputs if isinstance(forward_inputs, dict) else dict(forward_inputs)
    )
    for name in list(result):
        value = result[name]
        if name.startswith("p8_visual_") and torch.is_tensor(value):
            result[name] = _pin_tensor(value)
    return result
