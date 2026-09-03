# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Critic-warm-up wrapper for the existing reversal-damped band price."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlinf.runners.fastwam_branch_cost_control import (
    FastWAMBranchCostDecision,
    ReversalDampedBandPriceController,
    register_fastwam_branch_cost_controller,
)

from .route_neutral_contracts import PadCriticWarmupConfig

PAD_WARMUP_DAMPED_CONTROLLER_TYPE = "pad_critic_warmup_reversal_damped"


class PadCriticWarmupReversalDampedController(ReversalDampedBandPriceController):
    """Publish zero branch cost during critic warm-up, then start damped control."""

    controller_type = PAD_WARMUP_DAMPED_CONTROLLER_TYPE

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.critic_warmup = PadCriticWarmupConfig.from_mapping(
            config.get("critic_warmup")
        )
        super().__init__(config)

    def _warmup_active(self, runner_step: int) -> bool:
        return int(runner_step) < self.critic_warmup.runner_updates

    def _build_decision(self, runner_step: int) -> FastWAMBranchCostDecision:
        if not self._warmup_active(runner_step):
            return super()._build_decision(runner_step)
        return FastWAMBranchCostDecision(
            runner_step=runner_step,
            controller_type=self.controller_type,
            phase="critic_warmup",
            idm_cost=0.0,
            uncond_cost=0.0,
            components={
                "signed_price": 0.0,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "warmup_idm_probability": self.critic_warmup.idm_probability,
                "warmup_updates_remaining": float(
                    self.critic_warmup.runner_updates - runner_step
                ),
            },
        )

    def _update_after_rollout(self, observation: Any) -> dict[str, Any]:
        if not self._warmup_active(observation.runner_step):
            return super()._update_after_rollout(observation)
        feedback_rate = self._feedback_rate(observation)
        return {
            "observed": {
                "target_fraction": self.target_fraction,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "feedback_rate": feedback_rate,
                "rate_ema": None,
            },
            "update": {
                "band_error": 0.0,
                "raw_delta": 0.0,
                "applied_delta": 0.0,
                "clipped": False,
                "inside_band": True,
                "critic_warmup_active": True,
                "opposing_decay_applied": False,
                "reversal_decay_factor": self.reversal_decay_factor,
                "pre_decay_signed_price": 0.0,
                "post_decay_signed_price": 0.0,
                "reversal_decay_delta": 0.0,
                "feedback_delta": 0.0,
            },
        }

    def record_metrics(self, record: Mapping[str, Any]) -> dict[str, float]:
        metrics = super().record_metrics(record)
        metrics["fastwam/critic_warmup/active"] = float(
            record["applied"]["phase"] == "critic_warmup"
        )
        metrics["fastwam/critic_warmup/random_idm_probability"] = (
            self.critic_warmup.idm_probability
        )
        return metrics


@register_fastwam_branch_cost_controller(PAD_WARMUP_DAMPED_CONTROLLER_TYPE)
def _build_pad_critic_warmup_reversal_damped(
    config: Mapping[str, Any],
) -> PadCriticWarmupReversalDampedController:
    return PadCriticWarmupReversalDampedController(config)


__all__ = [
    "PAD_WARMUP_DAMPED_CONTROLLER_TYPE",
    "PadCriticWarmupReversalDampedController",
]
