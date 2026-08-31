# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Rank-local ledger completion for incremental PAD-Frozen evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from rlinf.runners.fastwam_libero_eval_collector import (
    EvaluationArtifactShard,
    FastWAMLiberoEvalCollector,
)


class PadFrozenEvalCollector(FastWAMLiberoEvalCollector):
    """Complete one round-robin ledger shard on each PAD environment rank."""

    def __init__(self, *, ledger_shard_count: int, **kwargs: Any) -> None:
        if isinstance(ledger_shard_count, bool) or int(ledger_shard_count) != (
            ledger_shard_count
        ):
            raise TypeError("PAD ledger_shard_count must be an integer.")
        self.ledger_shard_count = int(ledger_shard_count)
        if self.ledger_shard_count < 1:
            raise ValueError("PAD ledger_shard_count must be positive.")
        if bool(kwargs.get("resume", False)):
            raise ValueError("PAD sharded evaluation does not support resume.")
        super().__init__(**kwargs)
        if self.rank < 0 or self.rank >= self.ledger_shard_count:
            raise ValueError(
                f"PAD collector rank {self.rank} is outside its "
                f"{self.ledger_shard_count} ledger shards."
            )
        environment = (self.evaluation_runtime_identity or {}).get("environment", {})
        ordered_reset_ids = environment.get("ordered_reset_state_ids")
        ledger_reset_ids = [
            int(entry["reset_state_id"]) for entry in self.ledger["entries"]
        ]
        if (
            ordered_reset_ids is None
            or [int(item) for item in ordered_reset_ids] != ledger_reset_ids
        ):
            raise ValueError(
                "PAD sharded evaluation requires ledger-order reset-state IDs."
            )
        total_num_envs = int(environment.get("total_num_envs", 0))
        if (
            total_num_envs < self.ledger_shard_count
            or total_num_envs % self.ledger_shard_count
        ):
            raise ValueError(
                "PAD evaluation environments must divide evenly across ledger shards."
            )
        self._global_logical_batch_size = total_num_envs
        self._local_logical_batch_size = total_num_envs // self.ledger_shard_count
        self._local_ledger_entries = tuple(
            self.ledger["entries"][self.rank :: self.ledger_shard_count]
        )
        if not self._local_ledger_entries:
            raise ValueError(f"PAD collector rank {self.rank} owns no ledger entries.")

    @property
    def local_ledger_reset_state_ids(self) -> tuple[int, ...]:
        """Return the ordered reset identities owned by this environment rank."""

        return tuple(
            int(entry["reset_state_id"]) for entry in self._local_ledger_entries
        )

    @property
    def is_complete(self) -> bool:
        """Return whether this rank, not every rank, completed its ledger shard."""

        expected = {
            str(entry["episode_identity"]) for entry in self._local_ledger_entries
        }
        completed = {str(episode["episode_identity"]) for episode in self._episodes}
        return completed == expected

    @contextmanager
    def _rank_local_ledger(self) -> Iterator[None]:
        full_ledger = self.ledger
        self.ledger = {**full_ledger, "entries": list(self._local_ledger_entries)}
        try:
            yield
        finally:
            self.ledger = full_ledger

    def build_rollout_stop_control(self, *, logical_batch_size: int):
        """Stop this rank after its own shard while retaining the full ledger SHA."""

        if int(logical_batch_size) != self._global_logical_batch_size:
            raise ValueError(
                "PAD runner stop batch differs from configured total environments."
            )
        with self._rank_local_ledger():
            return super().build_rollout_stop_control(
                logical_batch_size=self._local_logical_batch_size
            )

    def finalize(self) -> EvaluationArtifactShard:
        """Finalize this rank's shard; the PAD driver reconciles all rank outputs."""

        with self._rank_local_ledger():
            return super().finalize()
