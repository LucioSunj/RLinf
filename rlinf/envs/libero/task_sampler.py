# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Runner-owned, resumable episode quotas for the ten LIBERO-Long tasks."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


class BalancedLiberoTaskSampler:
    """Allocate global task quotas before assigning episodes to workers.

    The two persistent random streams are independent of policy randomness.
    Reset draws use a common per-update seed and per-task substreams so that
    shared episode ordinals use the same reset at different candidate N.
    """

    def __init__(
        self, *, total_envs: int, reset_pool_sizes: list[int], seed: int = 42
    ) -> None:
        if total_envs < 10 or total_envs % 7:
            raise ValueError("Balanced LIBERO needs N >= 10 divisible by seven.")
        if len(reset_pool_sizes) != 10 or min(reset_pool_sizes) < 1:
            raise ValueError("Balanced LIBERO needs ten nonempty reset pools.")
        self.total_envs = int(total_envs)
        self.reset_pool_sizes = list(map(int, reset_pool_sizes))
        self.seed = int(seed)
        self.remainder = self.total_envs % 10
        self.cycle_length = 10 // math.gcd(self.remainder, 10)
        self.assignment_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.reset_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self.permutation: list[int] = []
        self.cycle_position = 0
        self.next_update = 0

    def next_plan(self, runner_step: int) -> dict[str, Any]:
        """Consume exactly the next zero-based global runner update."""

        if runner_step != self.next_update:
            raise ValueError(
                f"Sampler expects runner step {self.next_update}, got {runner_step}."
            )
        if self.cycle_position == 0:
            self.permutation = self.assignment_rng.permutation(10).tolist()
        quotas = [self.total_envs // 10] * 10
        for j in range(self.remainder):
            task = self.permutation[(self.cycle_position * self.remainder + j) % 10]
            quotas[task] += 1

        reset_seed = int(self.reset_rng.integers(0, 2**63))
        starts = np.cumsum([0, *self.reset_pool_sizes[:-1]])
        slots = []
        for task in self.assignment_rng.permutation(10).tolist():
            rng = np.random.default_rng(np.random.SeedSequence([reset_seed, task]))
            trials = rng.integers(0, self.reset_pool_sizes[task], size=quotas[task])
            for ordinal, trial in enumerate(trials.tolist()):
                slots.append(
                    {
                        "task_id": task,
                        "task_episode_ordinal": ordinal,
                        "trial_id": trial,
                        "reset_state_id": int(starts[task]) + trial,
                    }
                )
        ranks: list[list[dict[str, int]]] = [[] for _ in range(7)]
        rank_order = self.assignment_rng.permutation(7).tolist()
        for index, slot in enumerate(slots):
            ranks[rank_order[index % 7]].append(slot)
        for rank, rank_slots in enumerate(ranks):
            self.assignment_rng.shuffle(rank_slots)
            for local_slot, slot in enumerate(rank_slots):
                global_slot = rank * (self.total_envs // 7) + local_slot
                slot["episode_slot_id"] = runner_step * self.total_envs + global_slot
        plan = {
            "runner_step": runner_step,
            "total_envs": self.total_envs,
            "quotas": quotas,
            "cycle_permutation": list(self.permutation),
            "cycle_position": self.cycle_position,
            "cycle_length": self.cycle_length,
            "ranks": ranks,
        }
        self.cycle_position = (self.cycle_position + 1) % self.cycle_length
        self.next_update += 1
        return plan

    def state_dict(self) -> dict[str, Any]:
        """Return the minimal JSON-serializable state for the next update."""

        return copy.deepcopy(
            {
                "schema": "libero10-balanced-task-sampler-v1",
                "total_envs": self.total_envs,
                "reset_pool_sizes": self.reset_pool_sizes,
                "seed": self.seed,
                "permutation": self.permutation,
                "cycle_position": self.cycle_position,
                "next_update": self.next_update,
                "assignment_rng": self.assignment_rng.bit_generator.state,
                "reset_rng": self.reset_rng.bit_generator.state,
            }
        )

    def load_state_dict(self, state: dict[str, Any], *, runner_step: int) -> None:
        """Restore the next plan while requiring the same sampling geometry."""

        for name in ("schema", "total_envs", "reset_pool_sizes", "seed"):
            if state[name] != self.state_dict()[name]:
                raise ValueError(f"Sampler resume changed {name}.")
        if state["next_update"] != runner_step:
            raise ValueError("Sampler next update differs from the checkpoint step.")
        position = int(state["cycle_position"])
        permutation = list(state["permutation"])
        if position != runner_step % self.cycle_length or (
            runner_step > 0 and sorted(permutation) != list(range(10))
        ):
            raise ValueError("Sampler checkpoint cycle is inconsistent.")
        self.permutation = permutation
        self.cycle_position = position
        self.next_update = int(runner_step)
        self.assignment_rng.bit_generator.state = copy.deepcopy(state["assignment_rng"])
        self.reset_rng.bit_generator.state = copy.deepcopy(state["reset_rng"])
