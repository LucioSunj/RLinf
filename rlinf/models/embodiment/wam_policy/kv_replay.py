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

"""Stored-K/V and opt-in recompute backends for Gate PPO replay."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from fastwam.models.wan22.kv_tap import (
    GateKVSnapshot,
    GateLayerKV,
    KeyValueBank,
)


class GateKVReplayBackend(str, Enum):
    """Available Gate replay strategies."""

    STORED = "stored"
    RECOMPUTE = "recompute"


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class GateKVReplayConfig:
    """Storage policy for exact, handle-based Gate K/V replay."""

    backend: GateKVReplayBackend = GateKVReplayBackend.STORED
    storage_dtype: str = "bfloat16"
    pin_memory: bool = True
    deduplicate_static_banks: bool = True
    max_bytes_per_sample: int | None = None
    gate_kv_sample_budget: int | None = None
    gate_kv_sample_seed: int = 0
    hot_capacity_bytes_per_rollout_rank: int = 25 * 1024**3 // 2
    cold_capacity_bytes_per_rollout_rank: int = 24 * 1024**3
    nvme_capacity_bytes_per_rollout_rank: int = 0
    nvme_path: str | None = None
    hot_min_free_bytes: int = 4 * 1024**3
    prefetch_depth: int = 3
    transport: str = "host_staging"

    def __post_init__(self) -> None:
        backend = GateKVReplayBackend(self.backend)
        if self.storage_dtype not in _DTYPES:
            raise ValueError(
                f"Unsupported Gate K/V dtype {self.storage_dtype!r}; "
                f"expected one of {tuple(_DTYPES)}."
            )
        object.__setattr__(self, "backend", backend)
        if self.max_bytes_per_sample is not None and self.max_bytes_per_sample <= 0:
            raise ValueError("`max_bytes_per_sample` must be positive when set.")
        if self.gate_kv_sample_budget is not None and (
            isinstance(self.gate_kv_sample_budget, bool)
            or not isinstance(self.gate_kv_sample_budget, int)
            or self.gate_kv_sample_budget < 1
        ):
            raise ValueError(
                "`gate_kv_sample_budget` must be a positive integer or null."
            )
        if (
            isinstance(self.gate_kv_sample_seed, bool)
            or not isinstance(self.gate_kv_sample_seed, int)
            or self.gate_kv_sample_seed < 0
        ):
            raise ValueError("`gate_kv_sample_seed` must be a non-negative integer.")
        for name in (
            "hot_capacity_bytes_per_rollout_rank",
            "cold_capacity_bytes_per_rollout_rank",
            "nvme_capacity_bytes_per_rollout_rank",
            "hot_min_free_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        if self.nvme_capacity_bytes_per_rollout_rank and not self.nvme_path:
            raise ValueError("`nvme_path` is required when the NVMe tier is enabled.")
        if (
            isinstance(self.prefetch_depth, bool)
            or not isinstance(self.prefetch_depth, int)
            or self.prefetch_depth < 1
            or self.prefetch_depth > 16
        ):
            raise ValueError("`prefetch_depth` must be an integer in [1, 16].")
        if self.transport not in {"host_staging", "cuda_direct"}:
            raise ValueError("`transport` must be `host_staging` or `cuda_direct`.")
        if backend is GateKVReplayBackend.STORED and not self.deduplicate_static_banks:
            raise ValueError(
                "Packed stored-K/V replay always stores video/context once per layer; "
                "`deduplicate_static_banks` must remain true."
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.storage_dtype]


def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise ValueError("Only CPU Gate K/V tensors can be pinned.")
    try:
        return tensor.pin_memory()
    except RuntimeError:
        # CPU-only PyTorch builds do not provide a pinned-memory allocator.
        return tensor


def pin_gate_kv_forward_inputs(
    forward_inputs: Mapping[str, torch.Tensor],
    *,
    prefix: str = "gate_kv",
) -> dict[str, torch.Tensor]:
    """Re-pin packed Gate K/V after trajectory stack/cat/shuffle copies."""

    if isinstance(forward_inputs, dict):
        pinned = forward_inputs
    else:
        pinned = dict(forward_inputs)
    key_prefix = f"{prefix}_"
    # Keep only the keys while replacing values in place. Holding an
    # ``items()`` snapshot would retain every pageable K/V tensor until the
    # entire packed payload had been duplicated in pinned memory.
    for name in list(pinned):
        value = pinned[name]
        if (
            name.startswith(key_prefix)
            and torch.is_tensor(value)
            and value.device.type == "cpu"
        ):
            pinned[name] = _pin_tensor(value.contiguous())
    return pinned


def _map_bank(
    bank: KeyValueBank,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> KeyValueBank:
    return KeyValueBank(
        source=bank.source,
        key=transform(bank.key),
        value=transform(bank.value),
        valid_mask=transform(bank.valid_mask),
        contains_generated_future_video=bank.contains_generated_future_video,
    )


def _map_layer(
    layer: GateLayerKV,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> GateLayerKV:
    return GateLayerKV(
        layer_index=layer.layer_index,
        denoise_timestep=transform(layer.denoise_timestep),
        current_mode=layer.current_mode,
        current_frame_video=_map_bank(layer.current_frame_video, transform),
        action=_map_bank(layer.action, transform),
        context=_map_bank(layer.context, transform),
        actor_version=layer.actor_version,
    )


def _pin_snapshot(snapshot: GateKVSnapshot) -> GateKVSnapshot:
    return GateKVSnapshot(
        tuple(_map_layer(layer, _pin_tensor) for layer in snapshot.layers)
    )


def _banks_equal(left: KeyValueBank, right: KeyValueBank) -> bool:
    return (
        left.source is right.source
        and left.contains_generated_future_video
        == right.contains_generated_future_video
        and torch.equal(left.key, right.key)
        and torch.equal(left.value, right.value)
        and torch.equal(left.valid_mask, right.valid_mask)
    )


def _deduplicate_static_banks(
    snapshots: tuple[GateKVSnapshot, ...],
) -> tuple[tuple[GateKVSnapshot, ...], int]:
    """Share exact video/context tensors across denoising taps."""

    canonical: dict[tuple[int, str], KeyValueBank] = {}
    replacements = 0
    result: list[GateKVSnapshot] = []
    for snapshot in snapshots:
        layers: list[GateLayerKV] = []
        for layer in snapshot.layers:
            banks: dict[str, KeyValueBank] = {}
            for name, bank in (
                ("video", layer.current_frame_video),
                ("context", layer.context),
            ):
                key = (layer.layer_index, name)
                previous = canonical.get(key)
                if previous is not None and _banks_equal(previous, bank):
                    banks[name] = previous
                    replacements += 1
                else:
                    canonical[key] = bank
                    banks[name] = bank
            layers.append(
                GateLayerKV(
                    layer_index=layer.layer_index,
                    denoise_timestep=layer.denoise_timestep,
                    current_mode=layer.current_mode,
                    current_frame_video=banks["video"],
                    action=layer.action,
                    context=banks["context"],
                    actor_version=layer.actor_version,
                )
            )
        result.append(GateKVSnapshot(tuple(layers)))
    return tuple(result), replacements


def _snapshot_tensors(snapshot: GateKVSnapshot):
    for layer in snapshot.layers:
        yield layer.denoise_timestep
        for bank in (layer.current_frame_video, layer.action, layer.context):
            yield bank.key
            yield bank.value
            yield bank.valid_mask


def unique_tensor_bytes(snapshots: tuple[GateKVSnapshot, ...]) -> int:
    """Count tensor objects once so deduplicated banks are measured correctly."""

    seen: set[int] = set()
    total = 0
    for snapshot in snapshots:
        for tensor in _snapshot_tensors(snapshot):
            identity = id(tensor)
            if identity in seen:
                continue
            seen.add(identity)
            total += tensor.numel() * tensor.element_size()
    return total


@dataclass(frozen=True)
class StoredGateKVTaps:
    """Detached denoising snapshots held on CPU for exact Gate replay."""

    snapshots: tuple[GateKVSnapshot, ...]
    storage_dtype: torch.dtype
    requested_pin_memory: bool
    deduplicated_banks: int

    @property
    def total_bytes(self) -> int:
        return unique_tensor_bytes(self.snapshots)

    @property
    def is_pinned(self) -> bool:
        tensors = [
            tensor
            for snapshot in self.snapshots
            for tensor in _snapshot_tensors(snapshot)
        ]
        return bool(tensors) and all(tensor.is_pinned() for tensor in tensors)

    def materialize(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        non_blocking: bool = True,
    ) -> tuple[GateKVSnapshot, ...]:
        """Transfer only this replay record to the Gate compute device."""

        return tuple(
            snapshot.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking and self.is_pinned,
            )
            for snapshot in self.snapshots
        )


def offload_gate_kv(
    snapshots: tuple[GateKVSnapshot, ...] | list[GateKVSnapshot],
    config: GateKVReplayConfig,
) -> StoredGateKVTaps:
    """Detach and offload Gate K/V under the configured stored backend."""

    if config.backend is not GateKVReplayBackend.STORED:
        raise ValueError("`offload_gate_kv` requires the stored replay backend.")
    if not snapshots:
        raise ValueError("At least one Gate K/V snapshot is required.")

    stored = tuple(
        snapshot.detached().to(device="cpu", dtype=config.torch_dtype)
        for snapshot in snapshots
    )
    deduplicated = 0
    if config.deduplicate_static_banks:
        stored, deduplicated = _deduplicate_static_banks(stored)
    if config.pin_memory:
        stored = tuple(_pin_snapshot(snapshot) for snapshot in stored)

    return StoredGateKVTaps(
        snapshots=stored,
        storage_dtype=config.torch_dtype,
        requested_pin_memory=config.pin_memory,
        deduplicated_banks=deduplicated,
    )


@dataclass(frozen=True)
class GateKVReplayRecord:
    """Union record for default stored K/V and opt-in training recomputation."""

    backend: GateKVReplayBackend
    stored: StoredGateKVTaps | None = None
    recompute_inputs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        backend = GateKVReplayBackend(self.backend)
        object.__setattr__(self, "backend", backend)
        if backend is GateKVReplayBackend.STORED:
            if self.stored is None or self.recompute_inputs is not None:
                raise ValueError("Stored Gate replay requires only `stored` payload.")
        elif self.recompute_inputs is None or self.stored is not None:
            raise ValueError("Recompute Gate replay requires only `recompute_inputs`.")

    def materialize(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        recompute_fn: Callable[[Mapping[str, Any]], tuple[GateKVSnapshot, ...]]
        | None = None,
    ) -> tuple[GateKVSnapshot, ...]:
        if self.backend is GateKVReplayBackend.STORED:
            return self.stored.materialize(device=device, dtype=dtype)
        if recompute_fn is None:
            raise ValueError("Recompute Gate replay requires `recompute_fn`.")
        snapshots = recompute_fn(self.recompute_inputs)
        return tuple(
            snapshot.detached().to(device=device, dtype=dtype) for snapshot in snapshots
        )


@dataclass(frozen=True)
class PackedGateKVTaps:
    """Batch-first tensor representation accepted by RLinf trajectory utilities.

    Current-frame video and context banks are stored once per layer. Action K/V
    remains per denoising tap because it changes with the sampled action state.
    """

    layer_indices: torch.Tensor
    denoise_timesteps: torch.Tensor
    current_modes: torch.Tensor
    actor_versions: torch.Tensor
    video_key: torch.Tensor
    video_value: torch.Tensor
    video_mask: torch.Tensor
    action_key: torch.Tensor
    action_value: torch.Tensor
    action_mask: torch.Tensor
    context_key: torch.Tensor
    context_value: torch.Tensor
    context_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.layer_indices.ndim != 1:
            raise ValueError("`layer_indices` must be one-dimensional.")
        batch_size, num_taps = self.denoise_timesteps.shape
        num_layers = self.layer_indices.numel()
        if self.current_modes.shape != (batch_size,):
            raise ValueError("`current_modes` must have shape [B].")
        if self.actor_versions.shape != (batch_size,):
            raise ValueError("`actor_versions` must have shape [B].")
        if bool(((self.current_modes != 0) & (self.current_modes != 1)).any()):
            raise ValueError("`current_modes` must contain only UNCOND=0 or IDM=1.")
        if batch_size and bool((self.actor_versions != self.actor_versions[0]).any()):
            raise ValueError(
                "One packed Gate K/V batch must contain exactly one actor version."
            )
        if self.video_key.shape[:2] != (batch_size, num_layers):
            raise ValueError("`video_key` must have leading shape [B, L].")
        if self.context_key.shape[:2] != (batch_size, num_layers):
            raise ValueError("`context_key` must have leading shape [B, L].")
        if self.action_key.shape[:3] != (batch_size, num_taps, num_layers):
            raise ValueError("`action_key` must have leading shape [B, N, L].")
        pairs = (
            (self.video_key, self.video_value, self.video_mask),
            (self.context_key, self.context_value, self.context_mask),
            (self.action_key, self.action_value, self.action_mask),
        )
        for key, value, mask in pairs:
            if key.shape != value.shape or key.shape[:-1] != mask.shape:
                raise ValueError("Packed Gate K/V values and masks must match keys.")
            if mask.dtype != torch.bool:
                raise TypeError("Packed Gate K/V masks must use bool dtype.")

    @property
    def batch_size(self) -> int:
        return int(self.denoise_timesteps.shape[0])

    @property
    def num_taps(self) -> int:
        return int(self.denoise_timesteps.shape[1])

    @property
    def total_bytes(self) -> int:
        tensors = self.as_forward_inputs().values()
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def as_forward_inputs(self, prefix: str = "gate_kv") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}_{name}": value
            for name, value in (
                ("layer_indices", self.layer_indices),
                ("denoise_timesteps", self.denoise_timesteps),
                ("current_modes", self.current_modes),
                ("actor_versions", self.actor_versions),
                ("video_key", self.video_key),
                ("video_value", self.video_value),
                ("video_mask", self.video_mask),
                ("action_key", self.action_key),
                ("action_value", self.action_value),
                ("action_mask", self.action_mask),
                ("context_key", self.context_key),
                ("context_value", self.context_value),
                ("context_mask", self.context_mask),
            )
        }

    @classmethod
    def from_forward_inputs(
        cls,
        forward_inputs: Mapping[str, torch.Tensor],
        prefix: str = "gate_kv",
    ) -> PackedGateKVTaps:
        names = (
            "layer_indices",
            "denoise_timesteps",
            "current_modes",
            "actor_versions",
            "video_key",
            "video_value",
            "video_mask",
            "action_key",
            "action_value",
            "action_mask",
            "context_key",
            "context_value",
            "context_mask",
        )
        missing = [name for name in names if f"{prefix}_{name}" not in forward_inputs]
        if missing:
            raise KeyError(f"Missing packed Gate K/V fields: {missing}.")
        return cls(**{name: forward_inputs[f"{prefix}_{name}"] for name in names})

    def materialize(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        non_blocking: bool = True,
    ) -> tuple[GateKVSnapshot, ...]:
        tensors = {
            name: value.to(
                device=device,
                dtype=(
                    dtype
                    if value.is_floating_point() and name != "denoise_timesteps"
                    else None
                ),
                non_blocking=non_blocking,
            )
            for name, value in (
                ("denoise_timesteps", self.denoise_timesteps),
                ("current_modes", self.current_modes),
                ("actor_versions", self.actor_versions),
                ("video_key", self.video_key),
                ("video_value", self.video_value),
                ("video_mask", self.video_mask),
                ("action_key", self.action_key),
                ("action_value", self.action_value),
                ("action_mask", self.action_mask),
                ("context_key", self.context_key),
                ("context_value", self.context_value),
                ("context_mask", self.context_mask),
            )
        }
        layers = [int(index) for index in self.layer_indices.tolist()]
        snapshots = []
        for tap_index in range(self.num_taps):
            tap_layers = []
            for layer_offset, layer_index in enumerate(layers):
                modes = tuple(
                    ("idm" if int(mode) == 1 else "uncond")
                    for mode in tensors["current_modes"].tolist()
                )
                tap_layers.append(
                    GateLayerKV(
                        layer_index=layer_index,
                        denoise_timestep=tensors["denoise_timesteps"][:, tap_index],
                        current_mode=modes,
                        current_frame_video=KeyValueBank(
                            source=_kv_source("current_frame_video"),
                            key=tensors["video_key"][:, layer_offset],
                            value=tensors["video_value"][:, layer_offset],
                            valid_mask=tensors["video_mask"][:, layer_offset],
                        ),
                        action=KeyValueBank(
                            source=_kv_source("action"),
                            key=tensors["action_key"][:, tap_index, layer_offset],
                            value=tensors["action_value"][:, tap_index, layer_offset],
                            valid_mask=tensors["action_mask"][
                                :, tap_index, layer_offset
                            ],
                        ),
                        context=KeyValueBank(
                            source=_kv_source("text_state_context"),
                            key=tensors["context_key"][:, layer_offset],
                            value=tensors["context_value"][:, layer_offset],
                            valid_mask=tensors["context_mask"][:, layer_offset],
                        ),
                        actor_version=int(tensors["actor_versions"][0].item()),
                    )
                )
            snapshots.append(GateKVSnapshot(tuple(tap_layers)))
        return tuple(snapshots)


def _kv_source(value: str):
    # Local import keeps this module compatible with older FastWAM checkouts
    # until the adaptive policy is selected.
    from fastwam.models.wan22.kv_tap import KVSource

    return KVSource(value)


def pack_gate_kv(
    snapshots: tuple[GateKVSnapshot, ...] | list[GateKVSnapshot],
    config: GateKVReplayConfig,
    *,
    storage_device: torch.device | str | None = None,
) -> PackedGateKVTaps:
    """Pack stored Gate K/V into batch-first tensors with static-bank dedup.

    ``storage_device=None`` preserves the producing device.  Production
    rollout workers use that path so the hot tier never takes an eager
    full-batch detour through host memory.  CPU callers retain the historical
    pinned-host behavior.
    """

    if config.backend is not GateKVReplayBackend.STORED:
        raise ValueError("Tensor packing is available only for stored Gate K/V.")
    if not snapshots:
        raise ValueError("At least one Gate K/V snapshot is required.")
    if storage_device is None:
        storage_device = snapshots[0].layers[0].action.key.device
    storage_device = torch.device(storage_device)
    snapshots = tuple(
        snapshot.detached().to(device=storage_device, dtype=config.torch_dtype)
        for snapshot in snapshots
    )
    first = snapshots[0]
    for snapshot in snapshots[1:]:
        if snapshot.layer_indices != first.layer_indices:
            raise ValueError("Every denoising tap must use identical Gate layers.")
        for layer_index in first.layer_indices:
            left = first.layer(layer_index)
            right = snapshot.layer(layer_index)
            if not _banks_equal(left.current_frame_video, right.current_frame_video):
                raise ValueError(
                    "Current-frame video K/V changed across denoising taps; "
                    "cannot apply static-bank deduplication."
                )
            if not _banks_equal(left.context, right.context):
                raise ValueError(
                    "Text/state context K/V changed across denoising taps; "
                    "cannot apply static-bank deduplication."
                )

    def stack_static(name: str, field: str) -> torch.Tensor:
        return torch.stack(
            [
                getattr(getattr(first.layer(index), name), field)
                for index in first.layer_indices
            ],
            dim=1,
        )

    def stack_action(field: str) -> torch.Tensor:
        return torch.stack(
            [
                torch.stack(
                    [
                        getattr(snapshot.layer(index).action, field)
                        for index in first.layer_indices
                    ],
                    dim=1,
                )
                for snapshot in snapshots
            ],
            dim=1,
        )

    packed = PackedGateKVTaps(
        layer_indices=torch.tensor(first.layer_indices, dtype=torch.long),
        denoise_timesteps=torch.stack(
            [snapshot.layers[0].denoise_timestep for snapshot in snapshots],
            dim=1,
        ),
        current_modes=torch.tensor(
            [1 if mode.value == "idm" else 0 for mode in first.layers[0].current_mode],
            dtype=torch.long,
        ),
        actor_versions=torch.full(
            (first.batch_size,),
            first.layers[0].actor_version,
            dtype=torch.long,
        ),
        video_key=stack_static("current_frame_video", "key"),
        video_value=stack_static("current_frame_video", "value"),
        video_mask=stack_static("current_frame_video", "valid_mask"),
        action_key=stack_action("key"),
        action_value=stack_action("value"),
        action_mask=stack_action("valid_mask"),
        context_key=stack_static("context", "key"),
        context_value=stack_static("context", "value"),
        context_mask=stack_static("context", "valid_mask"),
    )
    if not config.pin_memory or storage_device.type != "cpu":
        return packed
    return PackedGateKVTaps(
        **{
            name: _pin_tensor(value)
            for name, value in packed.__dict__.items()
            if torch.is_tensor(value)
        }
    )
