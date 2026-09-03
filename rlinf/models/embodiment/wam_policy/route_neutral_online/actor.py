# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FSDP actor for current-step Gate, critic warm-up, and trainable UNCOND."""

from __future__ import annotations

import json
from typing import Any

import torch

from rlinf.algorithms.advantages import (
    FastWAMPolicyAlignment,
    summarize_fastwam_counterfactual_costs,
)
from rlinf.algorithms.fastwam_dual_ppo import (
    compute_base_uncond_kl_loss,
    compute_fastwam_dual_ppo_loss,
)
from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.models.embodiment.wam_policy.online_idm_bc.actor import (
    OnlineIDMBCFSDPActor,
)
from rlinf.models.embodiment.wam_policy.optimizer import (
    assert_fastwam_optimizer_update_resolution,
    fastwam_optimizer_gradient_norms,
)
from rlinf.models.embodiment.wam_policy.pad_rv.audit import (
    summarize_pad_frozen_rollout_state,
)
from rlinf.models.embodiment.wam_policy.pad_rv.memory import release_pad_host_memory
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_contracts import (
    PadCriticWarmupConfig,
)
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    fastwam_effective_gate_kv_mask,
)

from .policy import RouteNeutralOnlineIDMBCFastWAMPolicy


def align_current_step_trainable_advantages(
    *,
    advantages: torch.Tensor,
    route,
    emitted,
    loss_mask: torch.Tensor | None,
) -> FastWAMPolicyAlignment:
    """Use the same chunk for Gate credit while retaining UNCOND Flow credit."""

    if route.route_used.shape != emitted.next_route.shape:
        raise ValueError("Current-step route and Gate records must share shape.")
    if advantages.shape != (*route.route_used.shape, 1):
        raise ValueError("Current-step advantages must have shape [T,B,1].")
    valid = torch.ones_like(route.route_used, dtype=torch.bool)
    if loss_mask is not None:
        if loss_mask.shape[:2] != valid.shape:
            raise ValueError("Current-step loss mask must begin with [T,B].")
        valid &= loss_mask.bool().reshape(*valid.shape, -1).all(dim=-1)
    mismatch = emitted.valid & (
        route.route_was_forced
        | (route.route_source_chunk_ids != route.chunk_ids)
        | (route.route_used != emitted.next_route)
        | (route.chunk_ids != emitted.source_chunk_ids)
        | (route.episode_ids != emitted.episode_ids)
        | (route.actor_versions != emitted.actor_versions)
    )
    if bool(mismatch.any().item()):
        index = tuple(int(v) for v in mismatch.nonzero()[0].tolist())
        raise ValueError(
            "Current-step Gate decision does not own its action chunk; "
            f"first mismatch at {index}."
        )
    gate_valid = valid & emitted.valid
    return FastWAMPolicyAlignment(
        flow_advantages=advantages,
        flow_valid_mask=valid,
        gate_advantages=torch.where(
            gate_valid,
            advantages[..., 0],
            torch.zeros_like(advantages[..., 0]),
        ),
        gate_valid_mask=gate_valid,
    )


class RouteNeutralOnlineIDMBCFSDPActor(OnlineIDMBCFSDPActor):
    """Freeze Gate/LoRA updates during warm-up, then run Gate + RL + BC."""

    def __init__(self, cfg) -> None:
        self.critic_warmup = PadCriticWarmupConfig.from_mapping(
            cfg.actor.model.route_neutral_online.critic_warmup
        )
        self._route_neutral_warmup_active = True
        super().__init__(cfg)

    def model_provider_func(self) -> RouteNeutralOnlineIDMBCFastWAMPolicy:
        """Accept the final policy returned by the config-selected builder."""

        model = EmbodiedFSDPActor.model_provider_func(self)
        if not isinstance(model, RouteNeutralOnlineIDMBCFastWAMPolicy):
            raise TypeError(
                "Route-neutral builder returned "
                f"{type(model).__name__}, expected the trainable policy."
            )
        return model

    def _uses_fastwam_handle_replay(self) -> bool:
        """Gate replay is serialized neutral condition data, never Action K/V."""

        return False

    def _consume_rollout_batch_during_train_preparation(self) -> bool:
        """Flatten route-neutral replay fieldwise without retaining two copies."""

        profile = self.cfg.route_neutral_online_implementation
        if not bool(profile.consume_rollout_batch_during_train_preparation):
            raise ValueError("Route-neutral consuming train preparation was disabled.")
        return True

    def _after_rollout_batch_train_preparation(self) -> None:
        """Return pages from consumed source tensors before microbatch replay."""

        profile = self.cfg.route_neutral_online_implementation
        if not bool(profile.release_host_memory_after_train_preparation):
            raise ValueError(
                "Route-neutral post-preparation host-memory release was disabled."
            )
        report = release_pad_host_memory(
            schema="route-neutral-online-actor-host-memory-release-v1",
            rank=int(self._rank),
            phase="post_train_preparation",
        )
        print(
            "ROUTE_NEUTRAL_ONLINE_ACTOR_TRAIN_PREPARATION_RELEASE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )

    async def recv_rollout_trajectories(self, input_channel) -> None:
        """Release rank-transfer temporaries after standard Flow replay assembly."""

        await super().recv_rollout_trajectories(input_channel)
        profile = self.cfg.route_neutral_online_implementation
        if not bool(profile.release_host_memory_after_trajectory_receive):
            raise ValueError("Route-neutral actor host-memory release was disabled.")
        report = release_pad_host_memory(
            schema="route-neutral-online-actor-host-memory-release-v1",
            rank=int(self._rank),
            phase="post_trajectory_receive",
        )
        print(
            "ROUTE_NEUTRAL_ONLINE_ACTOR_HOST_MEMORY_RELEASE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )

    def _release_consumed_rollout_batch_before_receive(self) -> None:
        """Drop the previous update's replay before receiving the next one."""

        profile = self.cfg.route_neutral_online_implementation
        if not bool(profile.release_host_memory_after_trajectory_receive):
            raise ValueError("Route-neutral actor host-memory release was disabled.")
        if getattr(self, "rollout_batch", None) is None:
            return
        self.rollout_batch = None
        report = release_pad_host_memory(
            schema="route-neutral-online-actor-host-memory-release-v1",
            rank=int(self._rank),
            phase="pre_trajectory_receive",
        )
        print(
            "ROUTE_NEUTRAL_ONLINE_ACTOR_CONSUMED_BATCH_RELEASE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )

    def _warmup_batch(self, micro_batch: dict[str, Any]) -> bool:
        route = micro_batch.get("route_info")
        versions = getattr(route, "actor_versions", None)
        if not isinstance(versions, torch.Tensor) or versions.numel() < 1:
            raise KeyError("Critic warm-up requires route actor versions.")
        active = versions < self.critic_warmup.runner_updates
        if bool(active.any().item()) != bool(active.all().item()):
            raise ValueError("One training batch straddles critic warm-up.")
        return bool(active.all().item())

    def _align_fastwam_training_advantages(self, **kwargs):
        return align_current_step_trainable_advantages(
            advantages=kwargs["advantages"],
            route=kwargs["route"],
            emitted=kwargs["emitted"],
            loss_mask=kwargs.get("loss_mask"),
        )

    def _summarize_fastwam_rollout_state(self, **kwargs):
        condition_kwargs = dict(kwargs)
        condition_kwargs["kv_replay_backend"] = "condition"
        condition_kwargs["max_bytes_per_sample"] = None
        return summarize_pad_frozen_rollout_state(**condition_kwargs)

    def _summarize_fastwam_counterfactual_costs(self, **kwargs):
        return summarize_fastwam_counterfactual_costs(
            alignment_fn=self._align_fastwam_training_advantages,
            normalization_std_floor=float(
                self.cfg.algorithm.get("advantage_normalization_std_floor", 0.0) or 0.0
            ),
            **kwargs,
        )

    def _compute_fastwam_loss(
        self,
        *,
        micro_batch: dict,
        output_dict: dict[str, torch.Tensor],
        selected_loss_scales: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        warmup = self._warmup_batch(micro_batch)
        self._route_neutral_warmup_active = warmup
        if not warmup:
            return super()._compute_fastwam_loss(
                micro_batch=micro_batch,
                output_dict=output_dict,
                selected_loss_scales=selected_loss_scales,
            )

        gate_cfg = self.cfg.algorithm.gate_ppo
        flow_cfg = self.cfg.algorithm.uncond_flow_ppo
        route = micro_batch["route_info"]
        emitted = micro_batch["emitted_gate"]
        scales = selected_loss_scales or {}
        gate_mask = fastwam_effective_gate_kv_mask(
            micro_batch["gate_valid_mask"],
            micro_batch.get("gate_kv_sample_mask"),
        )
        policy_zero, metrics = compute_fastwam_dual_ppo_loss(
            gate_logprobs=output_dict["gate_logprobs"].float(),
            gate_old_logprobs=emitted.old_logprob.float(),
            gate_advantages=micro_batch["gate_advantages"].float(),
            gate_valid_mask=gate_mask,
            gate_clip_ratio_low=float(gate_cfg.clip_ratio_low),
            gate_clip_ratio_high=float(gate_cfg.clip_ratio_high),
            gate_base_probabilities=output_dict["gate_base_probabilities"].float(),
            gate_behavior_probabilities=output_dict[
                "gate_behavior_probabilities"
            ].float(),
            gate_entropy_coefficient=0.0,
            gate_loss_coefficient=0.0,
            flow_logprobs=output_dict["flow_logprobs"].float(),
            flow_old_logprobs=micro_batch["prev_logprobs"].float(),
            flow_advantages=micro_batch["flow_advantages"].float(),
            route_used=route.route_used,
            flow_valid_mask=micro_batch["flow_valid_mask"].bool(),
            flow_clip_ratio_low=float(flow_cfg.clip_ratio_low),
            flow_clip_ratio_high=float(flow_cfg.clip_ratio_high),
            flow_entropy=output_dict.get("flow_entropy"),
            flow_entropy_coefficient=0.0,
            flow_loss_coefficient=0.0,
            gate_selected_loss_scale=scales.get("gate"),
            flow_selected_loss_scale=scales.get("flow"),
        )
        base_kl_cfg = self.cfg.algorithm.get("regularization", {}).get(
            "base_uncond_kl", {}
        )
        if bool(base_kl_cfg.get("enabled", False)) or bool(
            base_kl_cfg.get("log_metric", False)
        ):
            _, base_metrics = compute_base_uncond_kl_loss(
                kl_values=output_dict["base_uncond_kl"].float(),
                route_used=route.route_used,
                valid_mask=micro_batch["flow_valid_mask"].bool(),
                selected_loss_scale=scales.get("flow"),
            )
            metrics.update(base_metrics)
        critic_cfg = self.cfg.algorithm.critic_loss
        critic_loss, critic_metrics = compute_ppo_critic_loss(
            values=output_dict["values"].float(),
            returns=micro_batch["returns"].float(),
            prev_values=micro_batch["prev_values"].float(),
            value_clip=float(critic_cfg.value_clip),
            huber_delta=float(critic_cfg.huber_delta),
            loss_mask=micro_batch.get("loss_mask"),
            loss_mask_sum=micro_batch.get("loss_mask_sum"),
            max_episode_steps=self.cfg.env.train.max_episode_steps,
        )
        loss = policy_zero + float(critic_cfg.get("loss_weight", 1.0)) * critic_loss
        metrics.update(critic_metrics)
        metrics.update(
            {
                "fastwam/regularized_policy_loss": policy_zero.detach(),
                "fastwam/total_loss": loss.detach(),
                "fastwam/critic_warmup/active": 1.0,
                "fastwam/critic_warmup/gate_update_enabled": 0.0,
                "fastwam/critic_warmup/uncond_update_enabled": 0.0,
                "fastwam/critic_warmup/random_idm_probability": (
                    self.critic_warmup.idm_probability
                ),
            }
        )
        return loss, {
            key: value.detach().item() if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()
        }

    def optimizer_step(self) -> tuple[float, list[float]]:
        """Step critic alone in warm-up and all three owners afterwards."""

        self.optimizer_steps += 1
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = self._strategy.clip_grad_norm_(model=self.model)
        self._fastwam_last_gradient_norms = fastwam_optimizer_gradient_norms(
            self.optimizer
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            self._logger.warning(
                f"[FSDP] Non-finite route-neutral grad norm {grad_norm}; skipping."
            )
        else:
            if self._route_neutral_warmup_active:
                for name in ("gate", "uncond_lora"):
                    if self._fastwam_last_gradient_norms[name] != 0.0:
                        raise RuntimeError(
                            f"{name} received a critic-warm-up gradient."
                        )
                    group = next(
                        group
                        for group in self.optimizer.param_groups
                        if str(group.get("name", "")) == name
                    )
                    for parameter in group["params"]:
                        parameter.grad = None
            elif not self._fastwam_update_resolution_checked:
                resolution = assert_fastwam_optimizer_update_resolution(
                    self.optimizer,
                    minimum_half_ulp_ratio=float(
                        self._cfg.optim.update_resolution_min_half_ulp_ratio
                    ),
                )
                self._fastwam_update_resolution_checked = True
                self._logger.info(
                    "[FSDP] First joint route-neutral update resolution: "
                    f"{json.dumps(resolution, sort_keys=True)}"
                )
            self.grad_scaler.step(optimizer=self.optimizer)
        self.grad_scaler.update()

        if self._route_neutral_warmup_active:
            self._online_idm_bc_audit_micro_batch = None
        elif (
            not self._online_idm_bc_gradient_audit_complete
            and self._online_idm_bc_audit_micro_batch is not None
        ):
            self._online_idm_bc_audit_metrics = self._run_online_idm_bc_gradient_audit()
        return grad_norm, [group["lr"] for group in self.optimizer.param_groups]

    def _optimizer_metrics(
        self,
        grad_norm: float,
        lr_list: list[float],
    ) -> dict[str, float]:
        metrics = super()._optimizer_metrics(grad_norm, lr_list)
        if self._route_neutral_warmup_active:
            metrics["gate/lr"] = 0.0
            metrics["uncond_flow/lora_lr"] = 0.0
        metrics.update(
            {
                "fastwam/critic_warmup/active": float(
                    self._route_neutral_warmup_active
                ),
                "fastwam/critic_warmup/gate_update_enabled": float(
                    not self._route_neutral_warmup_active
                ),
                "fastwam/critic_warmup/uncond_update_enabled": float(
                    not self._route_neutral_warmup_active
                ),
            }
        )
        return metrics


__all__ = [
    "RouteNeutralOnlineIDMBCFSDPActor",
    "align_current_step_trainable_advantages",
]
