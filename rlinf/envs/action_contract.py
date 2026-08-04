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

"""Typed, compact Action statistics that never retain raw trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

ACTION_TRACE_SCHEMA = "fastwam-action-trace-v1"
NORMALIZED_ACTION_STAGE = "normalized_action"
DENORMALIZED_ACTION_STAGE = "normalizer_backward"
GRIPPER_CONVERTED_ACTION_STAGE = "gripper_conversion"
PREPARED_LIBERO_ACTION_STAGE = "prepare_actions_for_libero"
SUBMITTED_LIBERO_ACTION_STAGE = "submitted_to_env_step"
FASTWAM_LIBERO_ACTION_STAGES = (
    NORMALIZED_ACTION_STAGE,
    DENORMALIZED_ACTION_STAGE,
    GRIPPER_CONVERTED_ACTION_STAGE,
    PREPARED_LIBERO_ACTION_STAGE,
    SUBMITTED_LIBERO_ACTION_STAGE,
)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True, eq=False)
class ActionStageStatistics:
    """Per-environment, per-dimension statistics for one Action pipeline stage."""

    stage: str
    minimum: torch.Tensor
    maximum: torch.Tensor
    finite_count: torch.Tensor
    below_low_count: torch.Tensor
    above_high_count: torch.Tensor
    total_value_count: torch.Tensor
    per_sample_shape: tuple[int, ...]
    dtype: str
    gripper_dimension_index: int
    action_contract_sha256: str

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("Action statistics stage must be non-empty.")
        tensors = (
            self.minimum,
            self.maximum,
            self.finite_count,
            self.below_low_count,
            self.above_high_count,
            self.total_value_count,
        )
        if any(not isinstance(item, torch.Tensor) for item in tensors):
            raise TypeError("Action statistics values must be tensors.")
        shape = self.minimum.shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ValueError("Action statistics tensors must have shape [B, D].")
        if any(item.shape != shape for item in tensors[1:]):
            raise ValueError("Action statistics tensors must share shape [B, D].")
        if not self.per_sample_shape or self.per_sample_shape[-1] != shape[1]:
            raise ValueError(
                "Action per-sample shape must end in the Action dimension."
            )
        if not self.dtype:
            raise ValueError("Action statistics dtype must be non-empty.")
        if not 0 <= self.gripper_dimension_index < shape[1]:
            raise ValueError("Gripper dimension index is outside the Action dimension.")
        if not _valid_sha256(self.action_contract_sha256):
            raise ValueError("Action contract SHA256 is invalid.")
        for name, counts in (
            ("finite", self.finite_count),
            ("below-low", self.below_low_count),
            ("above-high", self.above_high_count),
            ("total", self.total_value_count),
        ):
            if counts.dtype == torch.bool or counts.dtype.is_floating_point:
                raise TypeError(f"Action {name} counts must use an integer dtype.")
            if bool((counts < 0).any()):
                raise ValueError(f"Action {name} counts must be non-negative.")
        if bool((self.finite_count > self.total_value_count).any()):
            raise ValueError("Finite Action counts exceed total counts.")
        if bool((self.below_low_count > self.total_value_count).any()):
            raise ValueError("Below-low Action counts exceed total counts.")
        if bool((self.above_high_count > self.total_value_count).any()):
            raise ValueError("Above-high Action counts exceed total counts.")

    @property
    def shape(self) -> torch.Size:
        """Return the batch-by-dimension statistics shape."""

        return self.minimum.shape

    @property
    def batch_size(self) -> int:
        """Return the number of environments represented."""

        return int(self.shape[0])

    @property
    def action_dim(self) -> int:
        """Return the Action dimension."""

        return int(self.shape[1])

    @classmethod
    def from_values(
        cls,
        *,
        stage: str,
        values: Any,
        low: Sequence[float] | torch.Tensor,
        high: Sequence[float] | torch.Tensor,
        gripper_dimension_index: int,
        action_contract_sha256: str,
    ) -> "ActionStageStatistics":
        """Reduce raw values immediately to compact per-dimension statistics."""

        tensor = torch.as_tensor(values)
        if tensor.ndim < 2 or tensor.shape[0] < 1 or tensor.shape[-1] < 1:
            raise ValueError("Action values must have shape [B, ..., D].")
        batch_size = int(tensor.shape[0])
        action_dim = int(tensor.shape[-1])
        flattened = tensor.reshape(batch_size, -1, action_dim)
        low_tensor = torch.as_tensor(low, device=tensor.device, dtype=torch.float32)
        high_tensor = torch.as_tensor(high, device=tensor.device, dtype=torch.float32)
        if low_tensor.shape == (action_dim,):
            low_tensor = low_tensor.reshape(1, 1, action_dim)
            high_tensor = high_tensor.reshape(1, 1, action_dim)
        elif low_tensor.shape == (batch_size, action_dim):
            low_tensor = low_tensor[:, None, :]
            high_tensor = high_tensor[:, None, :]
        else:
            raise ValueError("Action bounds must have shape [D] or [B, D].")
        if high_tensor.shape != low_tensor.shape:
            raise ValueError("Action low/high bounds have different shapes.")
        if (
            not torch.isfinite(low_tensor).all()
            or not torch.isfinite(high_tensor).all()
        ):
            raise ValueError("Action bounds must be finite.")
        if bool((low_tensor >= high_tensor).any()):
            raise ValueError("Every Action low bound must be below its high bound.")

        finite = torch.isfinite(flattened)
        finite_count = finite.sum(dim=1, dtype=torch.long)
        positive_infinity = torch.full(
            (), float("inf"), device=tensor.device, dtype=torch.float32
        )
        negative_infinity = -positive_infinity
        flattened_float = flattened.float()
        minimum = flattened_float.masked_fill(~finite, positive_infinity).amin(dim=1)
        maximum = flattened_float.masked_fill(~finite, negative_infinity).amax(dim=1)
        no_finite = finite_count == 0
        minimum = minimum.masked_fill(no_finite, 0.0)
        maximum = maximum.masked_fill(no_finite, 0.0)
        total = torch.full_like(finite_count, int(flattened.shape[1]))
        return cls(
            stage=str(stage),
            minimum=minimum,
            maximum=maximum,
            finite_count=finite_count,
            below_low_count=(flattened_float < low_tensor).sum(dim=1, dtype=torch.long),
            above_high_count=(flattened_float > high_tensor).sum(
                dim=1, dtype=torch.long
            ),
            total_value_count=total,
            per_sample_shape=tuple(int(item) for item in tensor.shape[1:]),
            dtype=str(tensor.dtype).removeprefix("torch."),
            gripper_dimension_index=int(gripper_dimension_index),
            action_contract_sha256=str(action_contract_sha256),
        )

    def cpu(self) -> "ActionStageStatistics":
        """Return a contiguous CPU copy."""

        return ActionStageStatistics(
            stage=self.stage,
            minimum=self.minimum.cpu().contiguous(),
            maximum=self.maximum.cpu().contiguous(),
            finite_count=self.finite_count.cpu().contiguous(),
            below_low_count=self.below_low_count.cpu().contiguous(),
            above_high_count=self.above_high_count.cpu().contiguous(),
            total_value_count=self.total_value_count.cpu().contiguous(),
            per_sample_shape=self.per_sample_shape,
            dtype=self.dtype,
            gripper_dimension_index=self.gripper_dimension_index,
            action_contract_sha256=self.action_contract_sha256,
        )

    def split(
        self,
        sizes: Sequence[int],
        *,
        dim: int = 0,
    ) -> tuple["ActionStageStatistics", ...]:
        """Split a batch-aligned statistics record."""

        if dim != 0 or sum(int(item) for item in sizes) != self.batch_size:
            raise ValueError(
                "Action statistics may only be split exactly on batch dim 0."
            )
        fields = [
            torch.split(tensor, tuple(int(item) for item in sizes), dim=0)
            for tensor in (
                self.minimum,
                self.maximum,
                self.finite_count,
                self.below_low_count,
                self.above_high_count,
                self.total_value_count,
            )
        ]
        return tuple(
            ActionStageStatistics(
                stage=self.stage,
                minimum=fields[0][index],
                maximum=fields[1][index],
                finite_count=fields[2][index],
                below_low_count=fields[3][index],
                above_high_count=fields[4][index],
                total_value_count=fields[5][index],
                per_sample_shape=self.per_sample_shape,
                dtype=self.dtype,
                gripper_dimension_index=self.gripper_dimension_index,
                action_contract_sha256=self.action_contract_sha256,
            )
            for index in range(len(sizes))
        )

    @classmethod
    def cat(
        cls,
        items: Iterable["ActionStageStatistics"],
        *,
        dim: int = 0,
    ) -> "ActionStageStatistics":
        """Concatenate compatible records on the batch dimension."""

        records = tuple(items)
        if not records or dim != 0:
            raise ValueError("Action statistics require batch-dimension concatenation.")
        first = records[0]
        metadata = (
            first.stage,
            first.per_sample_shape,
            first.dtype,
            first.gripper_dimension_index,
            first.action_contract_sha256,
        )
        if any(
            (
                item.stage,
                item.per_sample_shape,
                item.dtype,
                item.gripper_dimension_index,
                item.action_contract_sha256,
            )
            != metadata
            for item in records[1:]
        ):
            raise ValueError(
                "Cannot concatenate Action statistics with different metadata."
            )
        names = (
            "minimum",
            "maximum",
            "finite_count",
            "below_low_count",
            "above_high_count",
            "total_value_count",
        )
        values = {
            name: torch.cat([getattr(item, name) for item in records], dim=0)
            for name in names
        }
        return cls(
            stage=first.stage,
            **values,
            per_sample_shape=first.per_sample_shape,
            dtype=first.dtype,
            gripper_dimension_index=first.gripper_dimension_index,
            action_contract_sha256=first.action_contract_sha256,
        )

    @classmethod
    def merge_time(
        cls,
        items: Iterable["ActionStageStatistics"],
    ) -> "ActionStageStatistics":
        """Merge per-step records while preserving their batch dimension."""

        records = tuple(items)
        if not records:
            raise ValueError("Cannot merge an empty Action statistics sequence.")
        first = records[0]
        if any(item.batch_size != first.batch_size for item in records):
            raise ValueError("Time-merged Action statistics have different batches.")
        if any(len(item.per_sample_shape) != 2 for item in records):
            raise ValueError(
                "Time-merged Action statistics require [T, D] sample shapes."
            )
        metadata = (
            first.stage,
            first.dtype,
            first.gripper_dimension_index,
            first.action_contract_sha256,
            first.action_dim,
        )
        if any(
            (
                item.stage,
                item.dtype,
                item.gripper_dimension_index,
                item.action_contract_sha256,
                item.action_dim,
            )
            != metadata
            for item in records[1:]
        ):
            raise ValueError(
                "Cannot time-merge Action statistics with different metadata."
            )
        finite_count = sum(
            (item.finite_count for item in records),
            torch.zeros_like(first.finite_count),
        )
        minimum = torch.stack(
            [
                item.minimum.masked_fill(item.finite_count == 0, float("inf"))
                for item in records
            ]
        ).amin(dim=0)
        maximum = torch.stack(
            [
                item.maximum.masked_fill(item.finite_count == 0, float("-inf"))
                for item in records
            ]
        ).amax(dim=0)
        minimum = minimum.masked_fill(finite_count == 0, 0.0)
        maximum = maximum.masked_fill(finite_count == 0, 0.0)
        return cls(
            stage=first.stage,
            minimum=minimum,
            maximum=maximum,
            finite_count=finite_count,
            below_low_count=sum(
                (item.below_low_count for item in records),
                torch.zeros_like(first.below_low_count),
            ),
            above_high_count=sum(
                (item.above_high_count for item in records),
                torch.zeros_like(first.above_high_count),
            ),
            total_value_count=sum(
                (item.total_value_count for item in records),
                torch.zeros_like(first.total_value_count),
            ),
            per_sample_shape=(
                sum(item.per_sample_shape[0] for item in records),
                first.action_dim,
            ),
            dtype=first.dtype,
            gripper_dimension_index=first.gripper_dimension_index,
            action_contract_sha256=first.action_contract_sha256,
        )

    def record_for_batch_index(self, index: int) -> dict[str, Any]:
        """Convert one environment's compact statistics to JSON-safe data."""

        if not 0 <= index < self.batch_size:
            raise IndexError(index)
        dimensions = []
        for dimension in range(self.action_dim):
            finite_count = int(self.finite_count[index, dimension])
            dimensions.append(
                {
                    "index": dimension,
                    "minimum": (
                        float(self.minimum[index, dimension]) if finite_count else None
                    ),
                    "maximum": (
                        float(self.maximum[index, dimension]) if finite_count else None
                    ),
                    "finite_count": finite_count,
                    "below_low_count": int(self.below_low_count[index, dimension]),
                    "above_high_count": int(self.above_high_count[index, dimension]),
                    "total_value_count": int(self.total_value_count[index, dimension]),
                }
            )
        return {
            "shape": list(self.per_sample_shape),
            "dtype": self.dtype,
            "dimensions": dimensions,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionStageStatistics):
            return NotImplemented
        return (
            self.stage == other.stage
            and self.per_sample_shape == other.per_sample_shape
            and self.dtype == other.dtype
            and self.gripper_dimension_index == other.gripper_dimension_index
            and self.action_contract_sha256 == other.action_contract_sha256
            and all(
                torch.equal(getattr(self, name), getattr(other, name))
                for name in (
                    "minimum",
                    "maximum",
                    "finite_count",
                    "below_low_count",
                    "above_high_count",
                    "total_value_count",
                )
            )
        )


@dataclass(frozen=True, slots=True, eq=False)
class ActionExecutionTrace:
    """Ordered typed statistics for one or more Action pipeline stages."""

    stages: tuple[ActionStageStatistics, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("Action execution trace must contain at least one stage.")
        names = tuple(item.stage for item in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("Action execution trace contains duplicate stages.")
        first = self.stages[0]
        if any(
            item.batch_size != first.batch_size
            or item.action_dim != first.action_dim
            or item.gripper_dimension_index != first.gripper_dimension_index
            or item.action_contract_sha256 != first.action_contract_sha256
            for item in self.stages[1:]
        ):
            raise ValueError("Action trace stages have incompatible batch metadata.")

    @property
    def batch_size(self) -> int:
        """Return the represented environment count."""

        return self.stages[0].batch_size

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return stage names in execution order."""

        return tuple(item.stage for item in self.stages)

    @property
    def action_contract_sha256(self) -> str:
        """Return the exact live contract hash used for all stages."""

        return self.stages[0].action_contract_sha256

    def cpu(self) -> "ActionExecutionTrace":
        """Return a contiguous CPU copy."""

        return ActionExecutionTrace(tuple(item.cpu() for item in self.stages))

    def split(
        self,
        sizes: Sequence[int],
        *,
        dim: int = 0,
    ) -> tuple["ActionExecutionTrace", ...]:
        """Split every stage along the batch dimension."""

        per_stage = [item.split(sizes, dim=dim) for item in self.stages]
        return tuple(
            ActionExecutionTrace(
                tuple(
                    per_stage[stage_index][split_index]
                    for stage_index in range(len(self.stages))
                )
            )
            for split_index in range(len(sizes))
        )

    @classmethod
    def cat(
        cls,
        items: Iterable["ActionExecutionTrace"],
        *,
        dim: int = 0,
    ) -> "ActionExecutionTrace":
        """Concatenate traces with identical stage layouts."""

        traces = tuple(items)
        if not traces:
            raise ValueError("Cannot concatenate an empty Action trace sequence.")
        names = traces[0].stage_names
        if any(item.stage_names != names for item in traces[1:]):
            raise ValueError("Cannot concatenate Action traces with different stages.")
        return cls(
            tuple(
                ActionStageStatistics.cat(
                    [trace.stages[index] for trace in traces],
                    dim=dim,
                )
                for index in range(len(names))
            )
        )

    @classmethod
    def combine(
        cls,
        *items: "ActionExecutionTrace",
    ) -> "ActionExecutionTrace":
        """Combine disjoint stages from the same executed batch."""

        traces = tuple(item for item in items if item is not None)
        if not traces:
            raise ValueError("Cannot combine an empty Action trace sequence.")
        return cls(tuple(stage for trace in traces for stage in trace.stages))

    def record_for_batch_index(self, index: int) -> dict[str, Any]:
        """Return one JSON-safe trace without any raw Action values."""

        first = self.stages[0]
        return {
            "schema": ACTION_TRACE_SCHEMA,
            "action_contract_sha256": self.action_contract_sha256,
            "gripper_dimension_index": first.gripper_dimension_index,
            "stages": {
                item.stage: item.record_for_batch_index(index) for item in self.stages
            },
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionExecutionTrace):
            return NotImplemented
        return self.stages == other.stages
