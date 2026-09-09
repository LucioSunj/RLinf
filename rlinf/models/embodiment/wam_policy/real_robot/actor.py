# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Local single-owner learner over RLinf's existing current-step PAD losses."""

from __future__ import annotations

import math

import torch

from rlinf.algorithms.fastwam_dual_ppo import compute_fastwam_dual_ppo_loss
from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler
from rlinf.models.embodiment.wam_policy.online_idm_bc.actor import (
    assemble_online_idm_bc_loss,
)
from rlinf.models.embodiment.wam_policy.optimizer import (
    partition_fastwam_trainable_parameters,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.actor import (
    RouteNeutralOnlineIDMBCFSDPActor,
)

from .collector import collate_microbatch


class RealRobotPADActor(RouteNeutralOnlineIDMBCFSDPActor):
    """Replace simulator batch geometry and dispatch with one local learner owner.

    This constructor deliberately does not initialize Worker/Ray/FSDP. All
    surrogate, value and BC mathematics remain the existing RLinf helpers.
    The initial physical profile colocates rollout/learner at batch boundaries.
    """

    def __init__(self, policy, training):
        self.model = policy
        self.training = training
        self.groups = partition_fastwam_trainable_parameters(policy.named_parameters())
        rates = {
            "gate": training["gate_lr"],
            "uncond_lora": training["uncond_lr"],
            "value_head": training["value_lr"],
        }
        self.optimizer = torch.optim.AdamW(
            [
                {"params": parameters, "lr": rates[name], "name": name}
                for name, parameters in self.groups.items()
            ],
            betas=tuple(training["adam_betas"]),
            eps=training["adam_eps"],
            weight_decay=training["weight_decay"],
        )
        self.lr_scheduler = get_lr_scheduler(
            training.get("lr_scheduler", "cosine"),
            self.optimizer,
            num_warmup_steps=training.get("lr_warmup_steps", 0),
            num_training_steps=training["scheduler_horizon"],
            min_lr_rate=training.get("min_lr_rate", 0.1),
        )
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        self.optimizer_steps = 0
        self._minibatch_has_uncond = False

    def _compute_fastwam_loss(
        self, *, micro_batch, output_dict, selected_loss_scales=None
    ):
        t = self.training
        scales = selected_loss_scales
        warmup = self.model.actor_version < 5
        value, metrics = compute_ppo_critic_loss(
            values=output_dict["values"].float(),
            returns=micro_batch["returns"].float(),
            prev_values=micro_batch["prev_values"].float(),
            value_clip=t["value_clip"],
            huber_delta=t["huber_delta"],
            loss_mask=micro_batch["loss_mask"],
        )
        loss = value * scales["value"]
        if not warmup:
            policy_loss, policy_metrics = compute_fastwam_dual_ppo_loss(
                gate_logprobs=output_dict["gate_logprobs"].float(),
                gate_old_logprobs=micro_batch["emitted_gate"].old_logprob.float(),
                gate_advantages=micro_batch["gate_advantages"].float(),
                gate_valid_mask=micro_batch["gate_valid_mask"],
                gate_clip_ratio_low=t["ppo_clip"],
                gate_clip_ratio_high=t["ppo_clip"],
                gate_base_probabilities=output_dict["gate_base_probabilities"],
                gate_behavior_probabilities=output_dict["gate_behavior_probabilities"],
                gate_entropy_coefficient=t["gate_entropy_weight"],
                gate_selected_loss_scale=scales["gate"],
                flow_logprobs=output_dict["flow_logprobs"].float(),
                flow_old_logprobs=micro_batch["prev_logprobs"].float(),
                flow_advantages=micro_batch["flow_advantages"].float(),
                route_used=micro_batch["route_info"].route_used,
                flow_valid_mask=micro_batch["flow_valid_mask"],
                flow_clip_ratio_low=t["ppo_clip"],
                flow_clip_ratio_high=t["ppo_clip"],
                flow_entropy=output_dict["flow_entropy"],
                flow_selected_loss_scale=scales["flow"],
            )
            loss, bc_metrics = assemble_online_idm_bc_loss(
                current_loss=loss + policy_loss,
                output_dict=output_dict,
                config=self.model.online_idm_bc_config,
                selected_loss_scale=scales["flow"],
                metric_scale_numerator=1.0,
                flow_metric_loss=float(policy_metrics["uncond_flow/policy_loss"]),
            )
            metrics.update(policy_metrics)
            metrics.update(bc_metrics)
        return loss, metrics

    def optimizer_step(self):
        """Suppress absent owners at the whole optimizer-minibatch boundary."""
        warmup = self.model.actor_version < 5
        disabled = (
            ["gate", "uncond_lora"]
            if warmup
            else ["uncond_lora"]
            if not self._minibatch_has_uncond
            else []
        )
        for name in disabled:
            for parameter in self.groups[name]:
                parameter.grad = None
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.groups.values() for p in group],
            self.training["max_grad_norm"],
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.optimizer_steps += 1
        return float(grad_norm), [group["lr"] for group in self.optimizer.param_groups]

    @torch.no_grad()
    def audit_preupdate_ratios(self, batch):
        """Compare stored and reconstructed behavior before any optimizer mutation."""
        maxima = {"gate": 0.0, "flow": 0.0}
        counts = {"gate": 0, "flow": 0}
        if self.model.actor_version < 5:
            return {"status": "WARMUP_VALUE_ONLY"}
        for index in range(len(batch["rows"])):
            micro = collate_microbatch(batch, [index])
            output = self.model.default_forward(
                micro["forward_inputs"],
                route_info=micro["route_info"],
                emitted_gate=micro["emitted_gate"],
            )
            gate = (
                (output["gate_logprobs"] - micro["emitted_gate"].old_logprob)
                .abs()
                .max()
                .item()
            )
            maxima["gate"] = max(maxima["gate"], gate)
            counts["gate"] += 1
            if int(micro["route_info"].route_used.item()) == 0:
                delta = (
                    (output["flow_logprobs"] - micro["prev_logprobs"])
                    .sum((-2, -1))
                    .abs()
                    .max()
                    .item()
                )
                maxima["flow"] = max(maxima["flow"], delta)
                counts["flow"] += 1
        if any(not math.isfinite(value) or value > 1e-4 for value in maxima.values()):
            raise RuntimeError(f"On-policy replay changed before update: {maxima}")
        return {"status": "PASS", "max_abs_log_ratio": maxima, "sample_counts": counts}

    def update(self, batch):
        """One PPO epoch, arbitrary final minibatch, shared denominators across micros."""
        self.model.train()
        before = {
            name: [p.detach().clone() for p in group]
            for name, group in self.groups.items()
        }
        audit = self.audit_preupdate_ratios(batch)
        records = []
        count = len(batch["rows"])
        batch_size = self.training["optimizer_batch_size"]
        micro_size = self.training["micro_batch_size"]
        for start in range(0, count, batch_size):
            indices = list(range(start, min(start + batch_size, count)))
            uncond = sum(batch["rows"][i].route == 0 for i in indices)
            self._minibatch_has_uncond = uncond > 0
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for offset in range(0, len(indices), micro_size):
                selected = indices[offset : offset + micro_size]
                micro = collate_microbatch(batch, selected)
                output = self.model.default_forward(
                    micro["forward_inputs"],
                    route_info=micro["route_info"],
                    emitted_gate=micro["emitted_gate"],
                )
                loss, _metrics = self._compute_fastwam_loss(
                    micro_batch=micro,
                    output_dict=output,
                    selected_loss_scales={
                        "gate": 1 / len(indices),
                        "flow": 1 / uncond if uncond else 0.0,
                        "value": len(selected) / len(indices),
                    },
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite learner loss; update is uncommitted."
                    )
                loss.backward()
                total += float(loss.detach())
            grad_norm, rates = self.optimizer_step()
            records.append(
                {
                    "chunks": len(indices),
                    "uncond_chunks": uncond,
                    "loss": total,
                    "grad_norm": grad_norm,
                    "learning_rates": rates,
                }
            )
        # Same shared scheduler semantics as the existing actor: once per runner update.
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        deltas = {
            name: float(
                torch.stack(
                    [
                        (p.detach() - old).float().square().sum()
                        for p, old in zip(group, before[name], strict=True)
                    ]
                )
                .sum()
                .sqrt()
            )
            for name, group in self.groups.items()
        }
        return {
            "preupdate_ratios": audit,
            "minibatches": records,
            "parameter_delta_norms": deltas,
            "optimizer_steps": self.optimizer_steps,
            "scheduler_steps": self.lr_scheduler.last_epoch,
            "warmup": self.model.actor_version < 5,
        }
