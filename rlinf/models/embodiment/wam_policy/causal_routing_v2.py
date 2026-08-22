# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Same-chunk hard-budget accounting for causal v2 evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from fastwam.causal_prediction import CausalComputeMode, UpliftGateOutput


@dataclass
class EpisodeComputeBudgetV2:
    """Mutable episode budget denominated in measured C2-equivalent latency."""

    total_cost: float
    remaining_cost: float
    fastest_mode_index: int = 0

    def __post_init__(self) -> None:
        if self.total_cost < 0 or not 0 <= self.remaining_cost <= self.total_cost:
            raise ValueError("Episode compute budget is invalid.")

    def debit_overhead(self, *, proposal_cost: float, gate_cost: float) -> None:
        """Debit pre-route proposal and Gate critical-path overhead."""

        overhead = float(proposal_cost) + float(gate_cost)
        if overhead < 0:
            raise ValueError("Routing overhead must be non-negative.")
        self.remaining_cost = max(0.0, self.remaining_cost - overhead)

    def debit_desired_mode(
        self,
        *,
        desired_mode: CausalComputeMode,
        mode_costs: dict[CausalComputeMode, float],
    ) -> tuple[CausalComputeMode, float]:
        """Debit a named route or fall back to fastest C0 when unaffordable."""

        desired = CausalComputeMode.parse(desired_mode)
        costs = {
            CausalComputeMode.parse(key): float(value)
            for key, value in mode_costs.items()
        }
        fastest = CausalComputeMode.C0_CURRENT
        if fastest not in costs or desired not in costs:
            raise ValueError(
                "Budgeted routes require costs for desired and fastest modes."
            )
        if any(value < 0 for value in costs.values()):
            raise ValueError("Budgeted mode costs must be non-negative.")
        executed = desired if costs[desired] <= self.remaining_cost else fastest
        charged = costs[executed]
        if charged > self.remaining_cost:
            raise RuntimeError("Fastest causal mode is not affordable after overhead.")
        self.remaining_cost -= charged
        return executed, charged

    def select(
        self,
        output: UpliftGateOutput,
        *,
        beta: float,
        cost_weight: float,
    ) -> int:
        """Choose the current chunk's affordable mode and debit actual cost."""

        if output.q_values.shape[0] != 1:
            raise ValueError("Closed-loop causal routing currently requires batch one.")
        costs = output.normalized_cost[0]
        affordable = costs <= self.remaining_cost
        if not bool(affordable[self.fastest_mode_index]):
            return self.fastest_mode_index
        utility = output.utilities(beta=beta, cost_weight=cost_weight)[0]
        utility = utility.masked_fill(~affordable, -torch.inf)
        selected = int(torch.argmax(utility).item())
        self.remaining_cost -= float(costs[selected].item())
        return selected
