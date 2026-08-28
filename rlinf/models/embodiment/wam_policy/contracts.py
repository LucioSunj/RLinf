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

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Iterable

import torch


class WAMRoute(IntEnum):
    """Execution route used for one FastWAM action chunk."""

    UNCOND = 0
    IDM = 1


# Keep the model-facing name used by the FastWAM design while exposing a
# route-oriented name to the rollout code.
WAMMode = WAMRoute


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_shape(name: str, value: torch.Tensor, shape: torch.Size) -> None:
    if value.shape != shape:
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}."
        )


def _require_integer(name: str, value: torch.Tensor) -> None:
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype, got {value.dtype}.")


def _require_bool(name: str, value: torch.Tensor) -> None:
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool, got {value.dtype}.")


def _require_float(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype, got {value.dtype}.")


def _first_bad_index(mask: torch.Tensor) -> tuple[int, ...]:
    index = mask.nonzero(as_tuple=False)[0]
    return tuple(int(item) for item in index.tolist())


def _raise_where(mask: torch.Tensor, message: str) -> None:
    if bool(mask.any().item()):
        raise ValueError(f"{message} First mismatch at index {_first_bad_index(mask)}.")


def _validate_routes(name: str, value: torch.Tensor) -> None:
    _require_integer(name, value)
    invalid = (value != int(WAMRoute.UNCOND)) & (value != int(WAMRoute.IDM))
    _raise_where(invalid, f"{name} contains a value outside WAMRoute.")


def _combine_tensors(
    values: Iterable[torch.Tensor],
    *,
    operation: str,
    dim: int,
) -> torch.Tensor:
    tensors = tuple(values)
    if operation == "cat":
        return torch.cat(tensors, dim=dim)
    if operation == "stack":
        return torch.stack(tensors, dim=dim)
    raise ValueError(f"Unknown tensor combine operation: {operation}.")


def _same_static_metadata(
    records: tuple["GateKVMetadata", ...],
) -> "GateKVMetadata":
    reference = records[0]
    for record in records[1:]:
        if record.layer_indices != reference.layer_indices:
            raise ValueError(
                "Cannot combine K/V metadata with different layer indices."
            )
        if record.storage_dtype != reference.storage_dtype:
            raise ValueError(
                "Cannot combine K/V metadata with different storage dtypes."
            )
        if record.tensor_shapes != reference.tensor_shapes:
            raise ValueError(
                "Cannot combine K/V metadata with different tensor shapes."
            )
        if (record.payload_reference_ids is None) != (
            reference.payload_reference_ids is None
        ):
            raise ValueError(
                "Cannot combine K/V metadata with inconsistent payload references."
            )
    return reference


@dataclass(frozen=True, slots=True, kw_only=True)
class GateKVMetadata:
    """Metadata for the detached K/V payload used by a Gate decision.

    The leading dimensions of ``denoise_timesteps``, ``total_bytes``, and
    ``payload_reference_ids`` are the decision batch dimensions. Layer selection,
    storage dtype, and per-bank shapes are static schema metadata.
    """

    layer_indices: tuple[int, ...]
    denoise_timesteps: torch.Tensor
    total_bytes: torch.Tensor
    storage_dtype: str = "bfloat16"
    tensor_shapes: tuple[tuple[int, ...], ...] = ()
    payload_reference_ids: torch.Tensor | None = None

    _SUPPORTED_DTYPES: ClassVar[frozenset[str]] = frozenset(
        {"bfloat16", "float16", "float32"}
    )

    def __post_init__(self) -> None:
        if not self.layer_indices:
            raise ValueError("layer_indices must contain at least one MoT layer.")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must not contain duplicates.")
        if any(layer_index < 0 for layer_index in self.layer_indices):
            raise ValueError("layer_indices must be non-negative.")
        if self.storage_dtype not in self._SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported K/V storage dtype {self.storage_dtype!r}; "
                f"expected one of {sorted(self._SUPPORTED_DTYPES)}."
            )
        if any(
            any(dimension < 0 for dimension in shape) for shape in self.tensor_shapes
        ):
            raise ValueError("tensor_shapes must contain only non-negative dimensions.")

        _require_float("denoise_timesteps", self.denoise_timesteps)
        _require_integer("total_bytes", self.total_bytes)
        if self.denoise_timesteps.ndim != self.total_bytes.ndim + 1:
            raise ValueError(
                "denoise_timesteps must have exactly one trailing tap dimension "
                "after the total_bytes batch dimensions."
            )
        if self.denoise_timesteps.shape[:-1] != self.total_bytes.shape:
            raise ValueError(
                "denoise_timesteps batch dimensions must match total_bytes."
            )
        _raise_where(self.total_bytes < 0, "total_bytes must be non-negative.")

        if self.payload_reference_ids is not None:
            _require_integer("payload_reference_ids", self.payload_reference_ids)
            _require_shape(
                "payload_reference_ids",
                self.payload_reference_ids,
                self.total_bytes.shape,
            )

    @property
    def batch_shape(self) -> torch.Size:
        """Return the leading decision batch shape."""

        return self.total_bytes.shape

    def cpu(self) -> "GateKVMetadata":
        """Return a contiguous CPU copy of all tensor metadata."""

        return GateKVMetadata(
            layer_indices=self.layer_indices,
            denoise_timesteps=self.denoise_timesteps.cpu().contiguous(),
            total_bytes=self.total_bytes.cpu().contiguous(),
            storage_dtype=self.storage_dtype,
            tensor_shapes=self.tensor_shapes,
            payload_reference_ids=(
                self.payload_reference_ids.cpu().contiguous()
                if self.payload_reference_ids is not None
                else None
            ),
        )

    @classmethod
    def _combine(
        cls,
        records: Iterable["GateKVMetadata"],
        *,
        operation: str,
        dim: int,
    ) -> "GateKVMetadata":
        record_tuple = tuple(records)
        if not record_tuple:
            raise ValueError("At least one K/V metadata record is required.")
        reference = _same_static_metadata(record_tuple)
        return cls(
            layer_indices=reference.layer_indices,
            denoise_timesteps=_combine_tensors(
                (record.denoise_timesteps for record in record_tuple),
                operation=operation,
                dim=dim,
            ),
            total_bytes=_combine_tensors(
                (record.total_bytes for record in record_tuple),
                operation=operation,
                dim=dim,
            ),
            storage_dtype=reference.storage_dtype,
            tensor_shapes=reference.tensor_shapes,
            payload_reference_ids=(
                _combine_tensors(
                    (
                        record.payload_reference_ids
                        for record in record_tuple
                        if record.payload_reference_ids is not None
                    ),
                    operation=operation,
                    dim=dim,
                )
                if reference.payload_reference_ids is not None
                else None
            ),
        )

    @classmethod
    def cat(cls, records: Iterable["GateKVMetadata"], dim: int = 0) -> "GateKVMetadata":
        """Concatenate compatible K/V metadata records."""

        return cls._combine(records, operation="cat", dim=dim)

    @classmethod
    def stack(
        cls, records: Iterable["GateKVMetadata"], dim: int = 0
    ) -> "GateKVMetadata":
        """Stack compatible K/V metadata records."""

        return cls._combine(records, operation="stack", dim=dim)

    def chunk(self, chunks: int, dim: int = 0) -> tuple["GateKVMetadata", ...]:
        """Split metadata into ``chunks`` along a batch dimension."""

        timestep_chunks = torch.chunk(self.denoise_timesteps, chunks, dim=dim)
        byte_chunks = torch.chunk(self.total_bytes, chunks, dim=dim)
        reference_chunks = (
            torch.chunk(self.payload_reference_ids, chunks, dim=dim)
            if self.payload_reference_ids is not None
            else (None,) * len(byte_chunks)
        )
        return tuple(
            GateKVMetadata(
                layer_indices=self.layer_indices,
                denoise_timesteps=timesteps,
                total_bytes=total_bytes,
                storage_dtype=self.storage_dtype,
                tensor_shapes=self.tensor_shapes,
                payload_reference_ids=references,
            )
            for timesteps, total_bytes, references in zip(
                timestep_chunks, byte_chunks, reference_chunks
            )
        )

    def split(
        self, split_sizes: list[int], dim: int = 0
    ) -> tuple["GateKVMetadata", ...]:
        """Split metadata with explicit sizes along a batch dimension."""

        timestep_splits = torch.split(self.denoise_timesteps, split_sizes, dim=dim)
        byte_splits = torch.split(self.total_bytes, split_sizes, dim=dim)
        reference_splits = (
            torch.split(self.payload_reference_ids, split_sizes, dim=dim)
            if self.payload_reference_ids is not None
            else (None,) * len(byte_splits)
        )
        return tuple(
            GateKVMetadata(
                layer_indices=self.layer_indices,
                denoise_timesteps=timesteps,
                total_bytes=total_bytes,
                storage_dtype=self.storage_dtype,
                tensor_shapes=self.tensor_shapes,
                payload_reference_ids=references,
            )
            for timesteps, total_bytes, references in zip(
                timestep_splits, byte_splits, reference_splits
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkRouteRecord:
    """Route metadata for chunks that were actually executed."""

    route_used: torch.Tensor
    route_was_forced: torch.Tensor
    chunk_ids: torch.Tensor
    episode_ids: torch.Tensor
    route_source_chunk_ids: torch.Tensor
    actor_versions: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.route_used.shape
        _validate_routes("route_used", self.route_used)
        _require_bool("route_was_forced", self.route_was_forced)
        _require_integer("chunk_ids", self.chunk_ids)
        _require_integer("episode_ids", self.episode_ids)
        _require_integer("route_source_chunk_ids", self.route_source_chunk_ids)
        _require_integer("actor_versions", self.actor_versions)
        for name in (
            "route_was_forced",
            "chunk_ids",
            "episode_ids",
            "route_source_chunk_ids",
            "actor_versions",
        ):
            _require_shape(name, getattr(self, name), shape)

        _raise_where(self.chunk_ids < 0, "chunk_ids must be non-negative.")
        _raise_where(self.episode_ids < 0, "episode_ids must be non-negative.")
        _raise_where(self.actor_versions < 0, "actor_versions must be non-negative.")
        _raise_where(
            self.route_was_forced & (self.route_source_chunk_ids != -1),
            "Forced routes must use route_source_chunk_ids == -1.",
        )
        _raise_where(
            ~self.route_was_forced & (self.route_source_chunk_ids < 0),
            "Gate-selected routes must identify a non-negative source chunk.",
        )

    @property
    def shape(self) -> torch.Size:
        """Return the chunk batch shape."""

        return self.route_used.shape

    def cpu(self) -> "ChunkRouteRecord":
        """Return a contiguous CPU copy."""

        return ChunkRouteRecord(
            route_used=self.route_used.cpu().contiguous(),
            route_was_forced=self.route_was_forced.cpu().contiguous(),
            chunk_ids=self.chunk_ids.cpu().contiguous(),
            episode_ids=self.episode_ids.cpu().contiguous(),
            route_source_chunk_ids=self.route_source_chunk_ids.cpu().contiguous(),
            actor_versions=self.actor_versions.cpu().contiguous(),
        )

    @classmethod
    def _combine(
        cls,
        records: Iterable["ChunkRouteRecord"],
        *,
        operation: str,
        dim: int,
    ) -> "ChunkRouteRecord":
        record_tuple = tuple(records)
        if not record_tuple:
            raise ValueError("At least one route record is required.")
        return cls(
            **{
                field_name: _combine_tensors(
                    (getattr(record, field_name) for record in record_tuple),
                    operation=operation,
                    dim=dim,
                )
                for field_name in (
                    "route_used",
                    "route_was_forced",
                    "chunk_ids",
                    "episode_ids",
                    "route_source_chunk_ids",
                    "actor_versions",
                )
            }
        )

    @classmethod
    def cat(
        cls, records: Iterable["ChunkRouteRecord"], dim: int = 0
    ) -> "ChunkRouteRecord":
        """Concatenate route records."""

        return cls._combine(records, operation="cat", dim=dim)

    @classmethod
    def stack(
        cls, records: Iterable["ChunkRouteRecord"], dim: int = 0
    ) -> "ChunkRouteRecord":
        """Stack route records."""

        return cls._combine(records, operation="stack", dim=dim)

    def chunk(self, chunks: int, dim: int = 0) -> tuple["ChunkRouteRecord", ...]:
        """Split this record into ``chunks`` along one batch dimension."""

        field_chunks = {
            field_name: torch.chunk(getattr(self, field_name), chunks, dim=dim)
            for field_name in (
                "route_used",
                "route_was_forced",
                "chunk_ids",
                "episode_ids",
                "route_source_chunk_ids",
                "actor_versions",
            )
        }
        return tuple(
            ChunkRouteRecord(
                **{
                    field_name: values[index]
                    for field_name, values in field_chunks.items()
                }
            )
            for index in range(len(field_chunks["route_used"]))
        )

    def split(
        self, split_sizes: list[int], dim: int = 0
    ) -> tuple["ChunkRouteRecord", ...]:
        """Split this record with explicit sizes."""

        field_splits = {
            field_name: torch.split(getattr(self, field_name), split_sizes, dim=dim)
            for field_name in (
                "route_used",
                "route_was_forced",
                "chunk_ids",
                "episode_ids",
                "route_source_chunk_ids",
                "actor_versions",
            )
        }
        return tuple(
            ChunkRouteRecord(
                **{
                    field_name: values[index]
                    for field_name, values in field_splits.items()
                }
            )
            for index in range(len(field_splits["route_used"]))
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GateDecisionRecord:
    """Gate decision emitted while generating a source chunk.

    ``base_probability`` and ``behavior_probability`` both denote the
    probability of selecting :attr:`WAMRoute.IDM`. ``old_logprob`` is the
    behavior-distribution log-probability of ``next_route``.
    """

    next_route: torch.Tensor
    base_probability: torch.Tensor
    behavior_probability: torch.Tensor
    old_logprob: torch.Tensor
    epsilon: torch.Tensor
    temperature: torch.Tensor
    valid: torch.Tensor
    source_chunk_ids: torch.Tensor
    episode_ids: torch.Tensor
    actor_versions: torch.Tensor
    kv_metadata: GateKVMetadata | None = None
    exploration_forced: torch.Tensor | None = None
    mode_flip_delta: torch.Tensor | None = None
    environment_ids: torch.Tensor | None = None
    task_ids: torch.Tensor | None = None
    trial_ids: torch.Tensor | None = None
    reset_state_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        shape = self.next_route.shape
        _validate_routes("next_route", self.next_route)
        for name in (
            "base_probability",
            "behavior_probability",
            "old_logprob",
            "epsilon",
            "temperature",
        ):
            value = getattr(self, name)
            _require_float(name, value)
            _require_shape(name, value, shape)
        _require_bool("valid", self.valid)
        _require_shape("valid", self.valid, shape)
        for name in ("source_chunk_ids", "episode_ids", "actor_versions"):
            value = getattr(self, name)
            _require_integer(name, value)
            _require_shape(name, value, shape)
        if self.exploration_forced is not None:
            _require_bool("exploration_forced", self.exploration_forced)
            _require_shape("exploration_forced", self.exploration_forced, shape)
            _raise_where(
                self.exploration_forced & (self.epsilon <= 0),
                "Forced exploration requires a positive epsilon.",
            )
        if self.mode_flip_delta is not None:
            _require_float("mode_flip_delta", self.mode_flip_delta)
            _require_shape("mode_flip_delta", self.mode_flip_delta, shape)
            _raise_where(
                ~torch.isfinite(self.mode_flip_delta),
                "mode_flip_delta must be finite.",
            )
        for name in (
            "environment_ids",
            "task_ids",
            "trial_ids",
            "reset_state_ids",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            _require_integer(name, value)
            _require_shape(name, value, shape)
            _raise_where(
                self.valid & (value < 0),
                f"Valid Gate decisions require non-negative {name}.",
            )

        for name in ("base_probability", "behavior_probability", "epsilon"):
            value = getattr(self, name)
            _raise_where(
                (~torch.isfinite(value)) | (value < 0) | (value > 1),
                f"{name} must be finite and in [0, 1].",
            )
        _raise_where(
            self.valid
            & ((~torch.isfinite(self.temperature)) | (self.temperature <= 0)),
            "Valid Gate decisions require a finite positive temperature.",
        )
        _raise_where(
            self.valid & ~torch.isfinite(self.old_logprob),
            "Valid Gate decisions require a finite old_logprob.",
        )
        _raise_where(
            self.valid & (self.source_chunk_ids < 0),
            "Valid Gate decisions require non-negative source_chunk_ids.",
        )
        _raise_where(
            self.valid & (self.episode_ids < 0),
            "Valid Gate decisions require non-negative episode_ids.",
        )
        _raise_where(
            self.valid & (self.actor_versions < 0),
            "Valid Gate decisions require non-negative actor_versions.",
        )
        if self.kv_metadata is not None and self.kv_metadata.batch_shape != shape:
            raise ValueError(
                "K/V metadata batch shape must match next_route; got "
                f"{tuple(self.kv_metadata.batch_shape)} and {tuple(shape)}."
            )

    @property
    def shape(self) -> torch.Size:
        """Return the decision batch shape."""

        return self.next_route.shape

    def cpu(self) -> "GateDecisionRecord":
        """Return a contiguous CPU copy."""

        return GateDecisionRecord(
            next_route=self.next_route.cpu().contiguous(),
            base_probability=self.base_probability.cpu().contiguous(),
            behavior_probability=self.behavior_probability.cpu().contiguous(),
            old_logprob=self.old_logprob.cpu().contiguous(),
            epsilon=self.epsilon.cpu().contiguous(),
            temperature=self.temperature.cpu().contiguous(),
            valid=self.valid.cpu().contiguous(),
            source_chunk_ids=self.source_chunk_ids.cpu().contiguous(),
            episode_ids=self.episode_ids.cpu().contiguous(),
            actor_versions=self.actor_versions.cpu().contiguous(),
            kv_metadata=(
                self.kv_metadata.cpu() if self.kv_metadata is not None else None
            ),
            exploration_forced=(
                self.exploration_forced.cpu().contiguous()
                if self.exploration_forced is not None
                else None
            ),
            mode_flip_delta=(
                self.mode_flip_delta.cpu().contiguous()
                if self.mode_flip_delta is not None
                else None
            ),
            environment_ids=(
                self.environment_ids.cpu().contiguous()
                if self.environment_ids is not None
                else None
            ),
            task_ids=(
                self.task_ids.cpu().contiguous() if self.task_ids is not None else None
            ),
            trial_ids=(
                self.trial_ids.cpu().contiguous()
                if self.trial_ids is not None
                else None
            ),
            reset_state_ids=(
                self.reset_state_ids.cpu().contiguous()
                if self.reset_state_ids is not None
                else None
            ),
        )

    @classmethod
    def _combine(
        cls,
        records: Iterable["GateDecisionRecord"],
        *,
        operation: str,
        dim: int,
    ) -> "GateDecisionRecord":
        record_tuple = tuple(records)
        if not record_tuple:
            raise ValueError("At least one Gate decision record is required.")
        if any(
            (record.kv_metadata is None) != (record_tuple[0].kv_metadata is None)
            for record in record_tuple
        ):
            raise ValueError(
                "Cannot combine Gate decisions with inconsistent K/V metadata."
            )
        tensor_fields = (
            "next_route",
            "base_probability",
            "behavior_probability",
            "old_logprob",
            "epsilon",
            "temperature",
            "valid",
            "source_chunk_ids",
            "episode_ids",
            "actor_versions",
        )
        optional_tensor_fields = (
            "exploration_forced",
            "mode_flip_delta",
            "environment_ids",
            "task_ids",
            "trial_ids",
            "reset_state_ids",
        )
        combined_optional: dict[str, torch.Tensor | None] = {}
        for field_name in optional_tensor_fields:
            values = tuple(getattr(record, field_name) for record in record_tuple)
            if all(value is None for value in values):
                combined_optional[field_name] = None
            elif any(value is None for value in values):
                raise ValueError(
                    "Cannot combine Gate decisions with inconsistent "
                    f"{field_name} metadata."
                )
            else:
                combined_optional[field_name] = _combine_tensors(
                    (value for value in values if value is not None),
                    operation=operation,
                    dim=dim,
                )
        return cls(
            **{
                field_name: _combine_tensors(
                    (getattr(record, field_name) for record in record_tuple),
                    operation=operation,
                    dim=dim,
                )
                for field_name in tensor_fields
            },
            **combined_optional,
            kv_metadata=(
                GateKVMetadata._combine(
                    (
                        record.kv_metadata
                        for record in record_tuple
                        if record.kv_metadata is not None
                    ),
                    operation=operation,
                    dim=dim,
                )
                if record_tuple[0].kv_metadata is not None
                else None
            ),
        )

    @classmethod
    def cat(
        cls, records: Iterable["GateDecisionRecord"], dim: int = 0
    ) -> "GateDecisionRecord":
        """Concatenate compatible Gate decision records."""

        return cls._combine(records, operation="cat", dim=dim)

    @classmethod
    def stack(
        cls, records: Iterable["GateDecisionRecord"], dim: int = 0
    ) -> "GateDecisionRecord":
        """Stack compatible Gate decision records."""

        return cls._combine(records, operation="stack", dim=dim)

    def chunk(self, chunks: int, dim: int = 0) -> tuple["GateDecisionRecord", ...]:
        """Split this record into ``chunks`` along one batch dimension."""

        tensor_fields = (
            "next_route",
            "base_probability",
            "behavior_probability",
            "old_logprob",
            "epsilon",
            "temperature",
            "valid",
            "source_chunk_ids",
            "episode_ids",
            "actor_versions",
        )
        field_chunks = {
            field_name: torch.chunk(getattr(self, field_name), chunks, dim=dim)
            for field_name in tensor_fields
        }
        metadata_chunks = (
            self.kv_metadata.chunk(chunks, dim=dim)
            if self.kv_metadata is not None
            else (None,) * len(field_chunks["next_route"])
        )
        optional_chunks = {}
        for field_name in (
            "exploration_forced",
            "mode_flip_delta",
            "environment_ids",
            "task_ids",
            "trial_ids",
            "reset_state_ids",
        ):
            value = getattr(self, field_name)
            optional_chunks[field_name] = (
                torch.chunk(value, chunks, dim=dim)
                if value is not None
                else (None,) * len(field_chunks["next_route"])
            )
        return tuple(
            GateDecisionRecord(
                **{
                    field_name: values[index]
                    for field_name, values in field_chunks.items()
                },
                **{
                    field_name: values[index]
                    for field_name, values in optional_chunks.items()
                },
                kv_metadata=metadata_chunks[index],
            )
            for index in range(len(field_chunks["next_route"]))
        )

    def split(
        self, split_sizes: list[int], dim: int = 0
    ) -> tuple["GateDecisionRecord", ...]:
        """Split this record with explicit sizes."""

        tensor_fields = (
            "next_route",
            "base_probability",
            "behavior_probability",
            "old_logprob",
            "epsilon",
            "temperature",
            "valid",
            "source_chunk_ids",
            "episode_ids",
            "actor_versions",
        )
        field_splits = {
            field_name: torch.split(getattr(self, field_name), split_sizes, dim=dim)
            for field_name in tensor_fields
        }
        metadata_splits = (
            self.kv_metadata.split(split_sizes, dim=dim)
            if self.kv_metadata is not None
            else (None,) * len(field_splits["next_route"])
        )
        optional_splits = {}
        for field_name in (
            "exploration_forced",
            "mode_flip_delta",
            "environment_ids",
            "task_ids",
            "trial_ids",
            "reset_state_ids",
        ):
            value = getattr(self, field_name)
            optional_splits[field_name] = (
                torch.split(value, split_sizes, dim=dim)
                if value is not None
                else (None,) * len(field_splits["next_route"])
            )
        return tuple(
            GateDecisionRecord(
                **{
                    field_name: values[index]
                    for field_name, values in field_splits.items()
                },
                **{
                    field_name: values[index]
                    for field_name, values in optional_splits.items()
                },
                kv_metadata=metadata_splits[index],
            )
            for index in range(len(field_splits["next_route"]))
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignedGateDecisions:
    """Gate records shifted onto the chunks whose routes they controlled."""

    decisions: GateDecisionRecord
    destination_chunk_ids: torch.Tensor
    source_time_indices: torch.Tensor

    def __post_init__(self) -> None:
        _require_integer("destination_chunk_ids", self.destination_chunk_ids)
        _require_integer("source_time_indices", self.source_time_indices)
        _require_shape(
            "destination_chunk_ids",
            self.destination_chunk_ids,
            self.decisions.shape,
        )
        _require_shape(
            "source_time_indices", self.source_time_indices, self.decisions.shape
        )
        _raise_where(
            self.decisions.valid & (self.source_time_indices < 0),
            "Valid aligned Gate decisions require a source time index.",
        )
        _raise_where(
            ~self.decisions.valid & (self.source_time_indices != -1),
            "Invalid aligned Gate decisions must use source_time_indices == -1.",
        )

    @property
    def valid(self) -> torch.Tensor:
        """Return the shifted Gate-loss mask."""

        return self.decisions.valid


def _shift_time(
    value: torch.Tensor,
    *,
    valid: torch.Tensor,
    fill_value: float | int | bool,
) -> torch.Tensor:
    if value.ndim < 1:
        raise ValueError("Delayed-route tensors must have a leading time dimension.")
    shifted = torch.full_like(value, fill_value)
    shifted[1:] = value[:-1]
    expanded_valid = valid
    while expanded_valid.ndim < shifted.ndim:
        expanded_valid = expanded_valid.unsqueeze(-1)
    return torch.where(expanded_valid, shifted, fill_value)


def _validate_delayed_route_inputs(
    route: ChunkRouteRecord,
    emitted: GateDecisionRecord,
    dones: torch.Tensor,
    reset_mask: torch.Tensor,
) -> None:
    if route.shape != emitted.shape:
        raise ValueError(
            "Route and emitted Gate records must have identical [time, batch] "
            f"shapes, got {tuple(route.shape)} and {tuple(emitted.shape)}."
        )
    if len(route.shape) != 2:
        raise ValueError(
            "Delayed-route alignment currently requires [time, batch] records, "
            f"got shape {tuple(route.shape)}."
        )
    _require_bool("dones", dones)
    _require_bool("reset_mask", reset_mask)
    _require_shape("dones", dones, route.shape)
    _require_shape("reset_mask", reset_mask, route.shape)

    _raise_where(
        emitted.valid & (emitted.source_chunk_ids != route.chunk_ids),
        "A Gate decision source_chunk_id does not match its source chunk.",
    )
    _raise_where(
        emitted.valid & (emitted.episode_ids != route.episode_ids),
        "A Gate decision was tagged with a different source episode.",
    )
    _raise_where(
        emitted.valid & (emitted.actor_versions != route.actor_versions),
        "A Gate decision actor version does not match its source chunk.",
    )

    _raise_where(
        reset_mask & ~route.route_was_forced,
        "The first chunk after reset must use a forced route.",
    )
    _raise_where(
        reset_mask & (route.route_used != int(WAMRoute.IDM)),
        "The first chunk after reset must be forced to IDM.",
    )
    _raise_where(
        reset_mask & (route.route_source_chunk_ids != -1),
        "The first chunk after reset cannot consume an earlier Gate decision.",
    )

    if route.shape[0] < 2:
        return

    previous_episode = route.episode_ids[:-1]
    destination_episode = route.episode_ids[1:]
    destination_reset = reset_mask[1:]
    episode_changed = destination_episode != previous_episode
    _raise_where(
        episode_changed & ~destination_reset,
        "A route crossed an episode boundary without a reset marker.",
    )
    _raise_where(
        destination_reset & ~episode_changed,
        "A reset marker must advance the per-environment episode id.",
    )
    _raise_where(
        dones[:-1] & ~destination_reset,
        "A chunk after a terminal source must be marked as reset.",
    )

    consumed = ~destination_reset & ~route.route_was_forced[1:]
    previous_valid = emitted.valid[:-1]
    _raise_where(
        consumed & ~previous_valid,
        "A non-forced route has no valid Gate decision from the previous chunk.",
    )
    _raise_where(
        consumed & dones[:-1],
        "A terminal chunk emitted a Gate decision that was consumed.",
    )
    _raise_where(
        consumed & (emitted.episode_ids[:-1] != destination_episode),
        "A Gate decision was consumed by a different episode.",
    )
    _raise_where(
        consumed & (emitted.actor_versions[:-1] != route.actor_versions[1:]),
        "A Gate decision was consumed under a different actor version.",
    )
    _raise_where(
        consumed & (route.route_source_chunk_ids[1:] != emitted.source_chunk_ids[:-1]),
        "route_source_chunk_ids does not identify the consumed Gate decision.",
    )
    _raise_where(
        consumed & (route.chunk_ids[1:] != emitted.source_chunk_ids[:-1] + 1),
        "A consumed Gate decision must control the immediately following chunk.",
    )
    _raise_where(
        consumed & (route.route_used[1:] != emitted.next_route[:-1]),
        "The executed route does not match the consumed Gate decision.",
    )


def shift_emitted_gate_decisions(
    *,
    route: ChunkRouteRecord,
    emitted: GateDecisionRecord,
    dones: torch.Tensor,
    reset_mask: torch.Tensor,
) -> AlignedGateDecisions:
    """Shift chunk-``t`` Gate records onto the consumed chunk ``t + 1``.

    Terminal emissions, reset destinations, and explicitly forced destinations
    are excluded from the returned Gate-loss mask. Cross-episode, non-adjacent,
    route, and actor-version mismatches fail rather than becoming silent masks.

    Args:
        route: Metadata for each executed chunk, shaped ``[time, batch]``.
        emitted: Gate decisions emitted by those chunks, with the same shape.
        dones: Whether each source chunk terminated its episode.
        reset_mask: Whether each destination chunk is the first after reset.

    Returns:
        Shifted decisions and source/destination identifiers.
    """

    _validate_delayed_route_inputs(route, emitted, dones, reset_mask)
    aligned_valid = torch.zeros_like(emitted.valid)
    if route.shape[0] > 1:
        aligned_valid[1:] = (
            emitted.valid[:-1]
            & ~dones[:-1]
            & ~reset_mask[1:]
            & ~route.route_was_forced[1:]
        )

    shifted_metadata = None
    if emitted.kv_metadata is not None:
        metadata = emitted.kv_metadata
        shifted_metadata = GateKVMetadata(
            layer_indices=metadata.layer_indices,
            denoise_timesteps=_shift_time(
                metadata.denoise_timesteps,
                valid=aligned_valid,
                fill_value=0.0,
            ),
            total_bytes=_shift_time(
                metadata.total_bytes,
                valid=aligned_valid,
                fill_value=0,
            ),
            storage_dtype=metadata.storage_dtype,
            tensor_shapes=metadata.tensor_shapes,
            payload_reference_ids=(
                _shift_time(
                    metadata.payload_reference_ids,
                    valid=aligned_valid,
                    fill_value=-1,
                )
                if metadata.payload_reference_ids is not None
                else None
            ),
        )

    shifted = GateDecisionRecord(
        next_route=torch.where(
            aligned_valid,
            _shift_time(
                emitted.next_route, valid=aligned_valid, fill_value=int(WAMRoute.UNCOND)
            ),
            route.route_used,
        ),
        base_probability=_shift_time(
            emitted.base_probability, valid=aligned_valid, fill_value=0.0
        ),
        behavior_probability=_shift_time(
            emitted.behavior_probability, valid=aligned_valid, fill_value=0.0
        ),
        old_logprob=_shift_time(
            emitted.old_logprob, valid=aligned_valid, fill_value=0.0
        ),
        epsilon=_shift_time(emitted.epsilon, valid=aligned_valid, fill_value=0.0),
        temperature=_shift_time(
            emitted.temperature, valid=aligned_valid, fill_value=1.0
        ),
        valid=aligned_valid,
        source_chunk_ids=_shift_time(
            emitted.source_chunk_ids, valid=aligned_valid, fill_value=-1
        ),
        episode_ids=_shift_time(
            emitted.episode_ids, valid=aligned_valid, fill_value=-1
        ),
        actor_versions=_shift_time(
            emitted.actor_versions, valid=aligned_valid, fill_value=-1
        ),
        exploration_forced=(
            _shift_time(
                emitted.exploration_forced,
                valid=aligned_valid,
                fill_value=False,
            )
            if emitted.exploration_forced is not None
            else None
        ),
        mode_flip_delta=(
            _shift_time(
                emitted.mode_flip_delta,
                valid=aligned_valid,
                fill_value=0.0,
            )
            if emitted.mode_flip_delta is not None
            else None
        ),
        environment_ids=(
            _shift_time(
                emitted.environment_ids,
                valid=aligned_valid,
                fill_value=-1,
            )
            if emitted.environment_ids is not None
            else None
        ),
        task_ids=(
            _shift_time(emitted.task_ids, valid=aligned_valid, fill_value=-1)
            if emitted.task_ids is not None
            else None
        ),
        trial_ids=(
            _shift_time(emitted.trial_ids, valid=aligned_valid, fill_value=-1)
            if emitted.trial_ids is not None
            else None
        ),
        reset_state_ids=(
            _shift_time(
                emitted.reset_state_ids,
                valid=aligned_valid,
                fill_value=-1,
            )
            if emitted.reset_state_ids is not None
            else None
        ),
        kv_metadata=shifted_metadata,
    )
    source_time_indices = torch.full_like(route.chunk_ids, -1)
    if route.shape[0] > 1:
        previous_indices = torch.arange(
            route.shape[0] - 1,
            dtype=source_time_indices.dtype,
            device=source_time_indices.device,
        )
        source_time_indices[1:] = previous_indices[:, None].expand(-1, route.shape[1])
    source_time_indices = torch.where(
        aligned_valid, source_time_indices, torch.full_like(source_time_indices, -1)
    )
    return AlignedGateDecisions(
        decisions=shifted,
        destination_chunk_ids=route.chunk_ids.clone(),
        source_time_indices=source_time_indices,
    )
