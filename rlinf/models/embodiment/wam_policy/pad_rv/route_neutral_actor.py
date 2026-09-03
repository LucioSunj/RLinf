# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FSDP actor for route-neutral Gate training with critic-only warm-up."""

from __future__ import annotations

import json
from typing import Any

import torch

from .actor import PadFrozenFSDPActor
from .loss import absent_uncond_flow_metrics, compute_pad_frozen_policy_loss
from .optimizer import assert_pad_frozen_update_resolution, pad_frozen_gradient_norms
from .route_neutral_contracts import PadCriticWarmupConfig


class PadRouteNeutralFSDPActor(PadFrozenFSDPActor):
    """Freeze Gate loss/updates while the fresh critic learns random routing."""

    def __init__(self, cfg) -> None:
        self.critic_warmup = PadCriticWarmupConfig.from_mapping(
            cfg.actor.model.critic_warmup
        )
        self.gate_optimizer_steps = 0
        self.critic_optimizer_steps = 0
        self._pad_resolution_groups_checked: set[str] = set()
        self._pad_critic_warmup_active = True
        super().__init__(cfg)

    def _warmup_batch(self, micro_batch: dict) -> bool:
        route_info = micro_batch.get("route_info")
        versions = getattr(route_info, "actor_versions", None)
        if not isinstance(versions, torch.Tensor) or versions.numel() < 1:
            raise KeyError("Route-neutral warm-up requires route actor versions.")
        warmup = versions < self.critic_warmup.runner_updates
        if bool(warmup.any().item()) != bool(warmup.all().item()):
            raise ValueError("A route-neutral training batch straddles warm-up.")
        return bool(warmup.all().item())

    def _compute_fastwam_loss(
        self,
        *,
        micro_batch: dict,
        output_dict: dict[str, torch.Tensor],
        selected_loss_scales: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        warmup = self._warmup_batch(micro_batch)
        self._pad_critic_warmup_active = warmup
        if not warmup:
            return super()._compute_fastwam_loss(
                micro_batch=micro_batch,
                output_dict=output_dict,
                selected_loss_scales=selected_loss_scales,
            )

        required = {
            "returns",
            "prev_values",
            "gate_advantages",
            "gate_valid_mask",
            "route_info",
            "emitted_gate",
        }
        missing = sorted(required - set(micro_batch))
        if missing:
            raise KeyError(f"Route-neutral warm-up batch is missing fields: {missing}.")
        gate_cfg = self.cfg.algorithm.gate_ppo
        emitted = micro_batch["emitted_gate"]
        scales = selected_loss_scales or {}
        _, metrics = compute_pad_frozen_policy_loss(
            gate_logprobs=output_dict["gate_logprobs"].float(),
            gate_old_logprobs=emitted.old_logprob.float(),
            gate_advantages=torch.zeros_like(micro_batch["gate_advantages"]).float(),
            gate_valid_mask=micro_batch["gate_valid_mask"].bool(),
            gate_clip_ratio_low=float(gate_cfg.clip_ratio_low),
            gate_clip_ratio_high=float(gate_cfg.clip_ratio_high),
            gate_base_probabilities=(
                output_dict["gate_base_probabilities"].float()
                if str(gate_cfg.get("entropy_metric_source", "behavior")).lower()
                == "base"
                else None
            ),
            gate_behavior_probabilities=output_dict[
                "gate_behavior_probabilities"
            ].float(),
            gate_entropy_coefficient=0.0,
            gate_loss_coefficient=0.0,
            selected_loss_scale=scales.get("gate"),
        )
        weighted_critic_loss, critic_metrics = self._compute_pad_critic_loss(
            micro_batch=micro_batch,
            output_dict=output_dict,
        )
        metrics.update(critic_metrics)
        zero = weighted_critic_loss.detach().new_zeros(())
        metrics.update(
            absent_uncond_flow_metrics(
                route_used=micro_batch["route_info"].route_used,
                valid_chunk_mask=micro_batch["gate_valid_mask"].bool(),
                reference=zero,
            )
        )
        metrics.update(
            {
                "pad_frozen/total_policy_loss": zero,
                "fastwam/regularized_policy_loss": zero,
                "fastwam/total_loss": weighted_critic_loss.detach(),
                "fastwam/critic_warmup/active": 1.0,
                "fastwam/critic_warmup/gate_loss_enabled": 0.0,
                "fastwam/critic_warmup/random_idm_probability": (
                    self.critic_warmup.idm_probability
                ),
                "fastwam/critic_warmup/gate_base_probability_mean": output_dict[
                    "gate_base_probabilities"
                ]
                .detach()
                .float()
                .mean(),
            }
        )
        return weighted_critic_loss, {
            key: value.detach().item() if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()
        }

    def optimizer_step(self) -> tuple[float, list[float]]:
        """Step only owners with gradients and audit each owner on first use."""

        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = self._strategy.clip_grad_norm_(model=self.model)
        self._fastwam_last_gradient_norms = pad_frozen_gradient_norms(self.optimizer)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            self._logger.warning(
                f"[FSDP] Non-finite route-neutral PAD grad norm {grad_norm}; "
                "skipping step."
            )
        else:
            active_groups = (
                ("value_head",)
                if self._pad_critic_warmup_active
                else ("gate", "value_head")
            )
            unchecked = tuple(
                name
                for name in active_groups
                if name not in self._pad_resolution_groups_checked
            )
            if unchecked:
                resolution = assert_pad_frozen_update_resolution(
                    self.optimizer,
                    minimum_half_ulp_ratio=float(
                        self._cfg.optim.update_resolution_min_half_ulp_ratio
                    ),
                    group_names=unchecked,
                )
                self._pad_resolution_groups_checked.update(unchecked)
                self._fastwam_update_resolution_checked = (
                    self._pad_resolution_groups_checked == {"gate", "value_head"}
                )
                self._logger.info(
                    "[FSDP] PAD owner-first update resolution: "
                    f"{json.dumps(resolution, sort_keys=True)}"
                )
            if self._pad_critic_warmup_active:
                if self._fastwam_last_gradient_norms["gate"] != 0.0:
                    raise RuntimeError(
                        "Route-neutral Gate received a warm-up gradient."
                    )
                gate_group = next(
                    group
                    for group in self.optimizer.param_groups
                    if str(group.get("name", "")) == "gate"
                )
                for parameter in gate_group["params"]:
                    parameter.grad = None
            self.grad_scaler.step(optimizer=self.optimizer)
            self.optimizer_steps += 1
            self.critic_optimizer_steps += 1
            if not self._pad_critic_warmup_active:
                self.gate_optimizer_steps += 1
        self.grad_scaler.update()
        return grad_norm, [group["lr"] for group in self.optimizer.param_groups]

    def _optimizer_metrics(
        self,
        grad_norm: float,
        lr_list: list[float],
    ) -> dict[str, float]:
        metrics = super()._optimizer_metrics(grad_norm, lr_list)
        if self._pad_critic_warmup_active:
            metrics["gate/lr"] = 0.0
        metrics["fastwam/critic_warmup/active"] = float(self._pad_critic_warmup_active)
        metrics["gate/optimizer_steps"] = float(self.gate_optimizer_steps)
        metrics["critic/optimizer_steps"] = float(self.critic_optimizer_steps)
        return metrics

    def _pad_checkpoint_versions(self, *, step: int) -> dict[str, int]:
        if self.critic_optimizer_steps != self.optimizer_steps:
            raise RuntimeError("Route-neutral critic/global optimizer steps disagree.")
        return {
            "actor": int(step),
            "gate": int(self.gate_optimizer_steps),
            "critic": int(self.critic_optimizer_steps),
        }

    def _restore_pad_checkpoint_versions(
        self,
        payload: Any,
        *,
        step: int,
        optimizer_steps: int,
    ) -> dict[str, int]:
        if not isinstance(payload, dict) or set(payload) != {
            "actor",
            "gate",
            "critic",
        }:
            raise ValueError("Route-neutral checkpoint versions are malformed.")
        versions = {name: int(value) for name, value in payload.items()}
        if (
            versions["actor"] != step
            or versions["critic"] != optimizer_steps
            or not 0 <= versions["gate"] <= versions["critic"]
        ):
            raise ValueError("Route-neutral checkpoint versions disagree.")
        self.gate_optimizer_steps = versions["gate"]
        self.critic_optimizer_steps = versions["critic"]
        return versions

    def _pad_ownership_audit_updates(self) -> set[int]:
        configured = super()._pad_ownership_audit_updates()
        first_gate_update = self.critic_warmup.runner_updates + 1
        return {update for update in configured if update >= first_gate_update} | {
            first_gate_update
        }


__all__ = ["PadRouteNeutralFSDPActor"]
