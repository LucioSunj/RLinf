# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compact route-neutral replay, materialized only for an actor microbatch."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
)

REPLAY_ROW = "route_neutral_replay_row"
TEXT_ID = "route_neutral_text_id"


@dataclass
class RouteNeutralReplayTrajectory:
    """Send light time-major data and a flat tensor bank through Channel.

    The flat dictionary is intentional: Channel transports dataclass tensor
    dictionaries directly, whereas nested replay objects would be pickled.
    """

    trajectory: Trajectory
    replay_tensors: dict[str, torch.Tensor]
    block_sizes: list[int]


@dataclass(kw_only=True)
class RouteNeutralRolloutResult(EmbodiedRolloutResult):
    """Keep the original trajectory axes and omit post-terminal heavy rows."""

    mask_after_terminal: bool
    replay_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    block_sizes: list[int] = field(default_factory=list)
    _finished: torch.Tensor | None = None
    _row_count: int = 0

    def append_step_result(self, result: ChunkStepResult) -> None:
        """Store one copy of each instruction and pack only eligible rows."""

        if not result.forward_inputs:
            super().append_step_result(result)
            # Each rollout epoch has a bootstrap-only final result. The next
            # epoch starts a new time axis in convert_trajectories_to_batch.
            self._finished = None
            return

        forward = dict(result.forward_inputs)
        text_ids = forward[TEXT_ID].cpu()
        batch_size = int(text_ids.numel())
        if self._finished is None:
            self._finished = torch.zeros(batch_size, dtype=torch.bool)
        if self.mask_after_terminal and result.dones is not None:
            # These dones belong to the preceding action chunk. Its terminal
            # transition remains in replay; only subsequent chunks are omitted.
            self._finished |= result.dones.cpu().bool().reshape(batch_size, -1).any(1)
        selected = (~self._finished).nonzero(as_tuple=False).reshape(-1)

        context = forward.pop("fastwam_context")
        context_mask = forward.pop("fastwam_context_mask")
        for row in selected.tolist():
            text_id = int(text_ids[row])
            key = f"text/{text_id}"
            if key not in self.replay_tensors:
                self.replay_tensors[key] = context[row, :-1].detach().cpu().clone()
                self.replay_tensors[f"text_mask/{text_id}"] = (
                    context_mask[row, :-1].detach().cpu().clone()
                )
        forward["route_neutral_proprio_context"] = context[:, -1:]
        forward["route_neutral_proprio_mask"] = context_mask[:, -1:]

        block = len(self.block_sizes)
        for name in list(forward):
            value = forward[name]
            if value.ndim > 1:
                del forward[name]
                if selected.numel():
                    self.replay_tensors[f"row/{block}/{name}"] = (
                        value.detach().index_select(0, selected.to(value.device)).cpu()
                    )
        if selected.numel():
            self.replay_tensors[f"row/{block}/{TEXT_ID}"] = text_ids.index_select(
                0, selected
            )
            self.block_sizes.append(int(selected.numel()))
        row_ids = torch.full((batch_size,), -1, dtype=torch.long)
        row_ids[selected] = torch.arange(
            self._row_count, self._row_count + selected.numel()
        )
        self._row_count += int(selected.numel())
        forward[REPLAY_ROW] = row_ids
        super().append_step_result(replace(result, forward_inputs=forward))

    def to_splited_trajectories_by_sizes(
        self, split_sizes: list[int], *, consume: bool = False
    ) -> list[RouteNeutralReplayTrajectory]:
        """Transfer the bank once to the profile's single actor rank."""

        if len(split_sizes) != 1:
            raise ValueError("Route-neutral compact replay requires one actor rank.")
        trajectories = super().to_splited_trajectories_by_sizes(
            split_sizes, consume=consume
        )
        result = RouteNeutralReplayTrajectory(
            trajectories[0], self.replay_tensors, self.block_sizes
        )
        if consume:
            self.replay_tensors = {}
            self.block_sizes = []
            self._row_count = 0
        return [result]

    def clear(self) -> None:
        """Release this collector's bank after its ownership has transferred."""

        super().clear()
        self.replay_tensors = {}
        self.block_sizes = []
        self._finished = None
        self._row_count = 0


class RouteNeutralReplayStore:
    """Index rank-local banks without concatenating or shuffling their tensors."""

    def __init__(self) -> None:
        self._blocks: list[dict[str, torch.Tensor]] = []
        self._texts: list[dict[str, torch.Tensor]] = []
        self._ends: list[int] = []

    def add(self, received: RouteNeutralReplayTrajectory) -> Trajectory:
        """Assign global row indices while preserving receive and time order."""

        offset = self._ends[-1] if self._ends else 0
        blocks = [{} for _ in received.block_sizes]
        texts = {}
        for key, value in received.replay_tensors.items():
            if key.startswith("row/"):
                _, index, name = key.split("/", 2)
                blocks[int(index)][name] = value
            else:
                texts[key] = value
        for block, size in zip(blocks, received.block_sizes, strict=True):
            self._blocks.append(block)
            self._texts.append(texts)
            self._ends.append((self._ends[-1] if self._ends else 0) + size)
        trajectory = received.trajectory
        row_ids = trajectory.forward_inputs[REPLAY_ROW]
        trajectory.forward_inputs[REPLAY_ROW] = torch.where(
            row_ids >= 0, row_ids + offset, row_ids
        )
        return trajectory

    def materialize(self, forward_inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Gather one microbatch, using a finite stored row for masked padding."""

        row_ids = forward_inputs[REPLAY_ROW].reshape(-1).tolist()
        rows: list[dict[str, torch.Tensor]] = []
        for row_id in row_ids:
            # Inactive slots retain their light route/mask/metric fields. Their
            # features have no loss contribution, but Gate/critic still need a
            # well-formed finite input for the existing padded microbatch.
            row_id = max(int(row_id), 0)
            block_index = bisect_right(self._ends, row_id)
            start = self._ends[block_index - 1] if block_index else 0
            block = self._blocks[block_index]
            local_row = row_id - start
            row = {key: value[local_row] for key, value in block.items()}
            texts = self._texts[block_index]
            text_id = int(row.pop(TEXT_ID))
            row["fastwam_context"] = torch.cat(
                (texts[f"text/{text_id}"], row.pop("route_neutral_proprio_context"))
            )
            row["fastwam_context_mask"] = torch.cat(
                (texts[f"text_mask/{text_id}"], row.pop("route_neutral_proprio_mask"))
            )
            rows.append(row)
        result = dict(forward_inputs)
        result.pop(REPLAY_ROW)
        result.update({key: torch.stack([row[key] for row in rows]) for key in rows[0]})
        return result
