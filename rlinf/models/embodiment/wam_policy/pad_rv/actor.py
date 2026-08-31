# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""FSDP actor subclass for PAD-Frozen Gate/value training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

import torch
from torch.optim import Optimizer

from rlinf.algorithms.advantages import summarize_fastwam_counterfactual_costs
from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.models.embodiment.wam_policy.pad_rv.config import PadFrozenConfig
from rlinf.scheduler import Worker
from rlinf.utils.checkpoint_state import (
    FASTWAM_ACTOR_RESUME_AUDIT_SENTINEL,
    FASTWAM_RESUME_AUDIT_SCHEMA,
    checkpoint_state_sha256,
)
from rlinf.utils.utils import get_rng_state, seed_everything, set_rng_state
from rlinf.workers.actor.fastwam_selective_sync import prepare_fastwam_sync_tensors
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    _raise_fastwam_collective_checkpoint_error,
    _restore_fastwam_fsdp_lazy_root_state,
    _snapshot_fastwam_fsdp_lazy_root_state,
)

from .audit import summarize_pad_frozen_rollout_state
from .checkpoint import (
    PAD_FROZEN_CHECKPOINT_SCHEMA,
    build_pad_frozen_checkpoint_contract,
    build_pad_frozen_versions,
    validate_pad_frozen_checkpoint_contract,
)
from .loss import (
    absent_uncond_flow_metrics,
    align_current_step_advantages,
    compute_pad_frozen_policy_loss,
)
from .memory import release_pad_host_memory
from .optimizer import (
    assert_pad_frozen_update_resolution,
    pad_frozen_gradient_norms,
    partition_pad_frozen_parameters,
)
from .policy import PadFrozenPolicy


class PadFrozenFSDPActor(EmbodiedFSDPActor):
    """Replace only PAD-specific ownership and loss hooks."""

    def __init__(self, cfg) -> None:
        self.pad_config = PadFrozenConfig.from_mapping(cfg.algorithm.pad_rv)
        if not self.pad_config.enabled:
            raise ValueError("PAD-Frozen actor requires `enabled: true`.")
        super().__init__(cfg)

    def init_worker(self) -> None:
        """Bind fresh PAD Gate/value initialization to the declared actor seed."""

        seed = int(self.cfg.actor.seed) + int(self._rank)
        normalized_seed = seed_everything(seed)
        print(
            "PAD_FRESH_GENESIS_SEED_AUDIT "
            + json.dumps(
                {
                    "schema": "pad-fresh-genesis-seed-audit-v1",
                    "status": "PASS",
                    "actor_rank": int(self._rank),
                    "configured_actor_seed": int(self.cfg.actor.seed),
                    "model_initialization_seed": normalized_seed,
                    "phase": "before_model_construction",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        super().init_worker()

    def model_provider_func(self) -> PadFrozenPolicy:
        model = super().model_provider_func()
        if not isinstance(model, PadFrozenPolicy):
            raise TypeError(
                f"PAD builder returned {type(model).__name__}, expected PadFrozenPolicy."
            )
        local_device = torch.device(
            Worker.torch_device_type,
            int(os.environ.get("LOCAL_RANK", 0)),
        )
        self._fastwam_rollout_sync_tensors = prepare_fastwam_sync_tensors(
            model, device=local_device
        )
        return model

    async def recv_rollout_trajectories(self, input_channel) -> None:
        """Release consumed receive/concatenation temporaries after batch assembly."""

        await super().recv_rollout_trajectories(input_channel)
        if not bool(
            self.cfg.pad_rv_implementation.release_host_memory_after_trajectory_receive
        ):
            raise ValueError("PAD trajectory-receive host-memory release was disabled.")
        report = release_pad_host_memory(
            schema="pad-actor-trajectory-host-memory-release-v1",
            rank=int(self._rank),
            phase="post_trajectory_receive",
        )
        print(
            "PAD_ACTOR_TRAJECTORY_HOST_MEMORY_RELEASE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )

    def _release_consumed_rollout_batch_before_receive(self) -> None:
        """Drop the consumed PAD batch before env workers materialize its successor."""

        if not bool(
            self.cfg.pad_rv_implementation.release_host_memory_after_trajectory_receive
        ):
            raise ValueError("PAD trajectory-receive host-memory release was disabled.")
        if getattr(self, "rollout_batch", None) is None:
            return
        self.rollout_batch = None
        report = release_pad_host_memory(
            schema="pad-actor-consumed-batch-host-memory-release-v1",
            rank=int(self._rank),
            phase="pre_trajectory_receive",
        )
        print(
            "PAD_ACTOR_CONSUMED_BATCH_HOST_MEMORY_RELEASE="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )

    def _fastwam_policy_module(self) -> PadFrozenPolicy:
        """Unwrap FSDP by exact PAD type instead of forwarded methods."""

        model = self.model
        visited: set[int] = set()
        while id(model) not in visited:
            visited.add(id(model))
            if isinstance(model, PadFrozenPolicy):
                return model
            next_model = getattr(model, "module", None)
            if next_model is None or next_model is model:
                next_model = getattr(model, "_fsdp_wrapped_module", None)
            if next_model is None:
                break
            model = next_model
        raise TypeError("Could not unwrap PadFrozenPolicy from its FSDP wrapper.")

    def _align_fastwam_training_advantages(self, **kwargs):
        """Bind runner GAE to the route selected for that same action chunk."""

        return align_current_step_advantages(
            advantages=kwargs["advantages"],
            route=kwargs["route"],
            emitted=kwargs["emitted"],
            loss_mask=kwargs.get("loss_mask"),
        )

    def _training_batch_reference(
        self,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        versions = batch.get("versions")
        if not isinstance(versions, torch.Tensor) or versions.shape[-1:] != (1,):
            raise KeyError("PAD-Frozen training requires [batch, 1] versions.")
        return versions

    def _allows_absent_action_logprobs(self) -> bool:
        return True

    def _summarize_fastwam_rollout_state(self, **kwargs):
        """Audit condition replay without inventing Action K/V metadata."""

        return summarize_pad_frozen_rollout_state(**kwargs)

    def _summarize_fastwam_counterfactual_costs(self, **kwargs):
        """Evaluate fixed costs with PAD's same-chunk Gate ownership."""

        return summarize_fastwam_counterfactual_costs(
            alignment_fn=self._align_fastwam_training_advantages,
            normalization_std_floor=float(
                self.cfg.algorithm.get("advantage_normalization_std_floor", 0.0) or 0.0
            ),
            **kwargs,
        )

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """Save only Stage 1 trainables and their optimizer continuation state."""

        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded
        if restore_weight_offload:
            self.load_param_and_grad(self.device)
        if restore_optimizer_offload:
            self.load_optimizer(self.device)
        try:
            local_error: Exception | None = None
            try:
                policy = self._fastwam_policy_module()
                if not isinstance(policy, PadFrozenPolicy):
                    raise TypeError("PAD checkpoint requires PadFrozenPolicy.")
                policy.set_global_step(int(step))
                self.version = int(step)
                payload = {
                    "schema": PAD_FROZEN_CHECKPOINT_SCHEMA,
                    "owner": "actor",
                    "step": int(step),
                    "optimizer_steps": int(self.optimizer_steps),
                    "versions": build_pad_frozen_versions(
                        step=int(step),
                        optimizer_steps=int(self.optimizer_steps),
                    ),
                    "stage_contract": build_pad_frozen_checkpoint_contract(
                        self.cfg,
                        world_size=int(self._world_size),
                    ),
                    "policy": policy.trainable_state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "grad_scaler": self.grad_scaler.state_dict(),
                    "rng": get_rng_state(),
                }
                payload = self._checkpoint_cpu_clone(payload)
                os.makedirs(save_path, exist_ok=True)
                target = os.path.join(save_path, f"rank_{self._rank}.pt")
                temporary = f"{target}.tmp"
                try:
                    torch.save(payload, temporary)
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            except Exception as error:
                local_error = error
            _raise_fastwam_collective_checkpoint_error(
                local_error,
                context="PAD-Frozen actor checkpoint save",
            )
        finally:
            if restore_weight_offload:
                self.offload_param_and_grad()
            if restore_optimizer_offload:
                self.offload_optimizer()

    def load_checkpoint(self, load_path: str) -> int:
        """Restore an exact Gate-only frozen-pair continuation."""

        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded
        if restore_weight_offload:
            self.load_param_and_grad(self.device)
        if restore_optimizer_offload:
            self.load_optimizer(self.device)
        fsdp_lazy_root_state = _snapshot_fastwam_fsdp_lazy_root_state(self.model)
        try:
            local_error: Exception | None = None
            loaded_version: int | None = None
            try:
                checkpoint_path = os.path.join(
                    load_path,
                    f"rank_{self._rank}.pt",
                )
                payload = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                expected_keys = {
                    "schema",
                    "owner",
                    "step",
                    "optimizer_steps",
                    "versions",
                    "stage_contract",
                    "policy",
                    "optimizer",
                    "lr_scheduler",
                    "grad_scaler",
                    "rng",
                }
                if set(payload) != expected_keys:
                    raise ValueError(
                        f"PAD-Frozen actor checkpoint keys changed: {sorted(payload)}."
                    )
                if (
                    payload.get("schema") != PAD_FROZEN_CHECKPOINT_SCHEMA
                    or payload.get("owner") != "actor"
                ):
                    raise ValueError("Unsupported PAD-Frozen actor checkpoint.")
                validate_pad_frozen_checkpoint_contract(
                    payload.get("stage_contract"),
                    self.cfg,
                    world_size=int(self._world_size),
                )
                step = int(payload.get("step", -1))
                optimizer_steps = int(payload.get("optimizer_steps", -1))
                expected_versions = build_pad_frozen_versions(
                    step=step,
                    optimizer_steps=optimizer_steps,
                )
                if payload.get("versions") != expected_versions:
                    raise ValueError("PAD-Frozen actor version fields disagree.")

                policy = self._fastwam_policy_module()
                if not isinstance(policy, PadFrozenPolicy):
                    raise TypeError("PAD checkpoint requires PadFrozenPolicy.")
                saved_policy = payload.get("policy")
                if (
                    not isinstance(saved_policy, dict)
                    or "route_tracker" not in saved_policy
                ):
                    raise ValueError(
                        "PAD actor checkpoint omits current-step route state."
                    )
                saved_route_sha256 = checkpoint_state_sha256(
                    saved_policy["route_tracker"]
                )
                saved_rng_sha256 = checkpoint_state_sha256(payload["rng"])
                policy.load_trainable_state_dict(saved_policy)
                self.optimizer.load_state_dict(payload["optimizer"])
                self.lr_scheduler.load_state_dict(payload["lr_scheduler"])
                self.grad_scaler.load_state_dict(payload["grad_scaler"])
                self.optimizer_steps = optimizer_steps
                self.version = step
                if policy.actor_version != self.version:
                    raise ValueError(
                        "PAD checkpoint policy version does not match its step."
                    )
                restored_route_sha256 = checkpoint_state_sha256(
                    policy.route_tracker.state_dict()
                )
                if restored_route_sha256 != saved_route_sha256:
                    raise ValueError(
                        "PAD current-step route state changed during actor load."
                    )
                set_rng_state(payload["rng"])
                restored_rng_sha256 = checkpoint_state_sha256(get_rng_state())
                if restored_rng_sha256 != saved_rng_sha256:
                    raise ValueError("PAD actor RNG state changed during load.")
                print(
                    f"{FASTWAM_ACTOR_RESUME_AUDIT_SENTINEL} "
                    + json.dumps(
                        {
                            "schema": FASTWAM_RESUME_AUDIT_SCHEMA,
                            "checkpoint_schema": PAD_FROZEN_CHECKPOINT_SCHEMA,
                            "stage": self.pad_config.stage.value,
                            "owner": "actor",
                            "rank": int(self._rank),
                            "step": step,
                            "optimizer_steps": optimizer_steps,
                            "actor_version": int(policy.actor_version),
                            "route_state_sha256": restored_route_sha256,
                            "rng_sha256": restored_rng_sha256,
                            "status": "PASS",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                loaded_version = step
            except Exception as error:
                local_error = error
            _raise_fastwam_collective_checkpoint_error(
                local_error,
                context="PAD-Frozen actor checkpoint load",
            )
            if loaded_version is None:
                raise RuntimeError("PAD actor checkpoint load returned no version.")
            return loaded_version
        finally:
            _restore_fastwam_fsdp_lazy_root_state(fsdp_lazy_root_state)
            if restore_weight_offload:
                self.offload_param_and_grad()
            if restore_optimizer_offload:
                self.offload_optimizer()

    def build_optimizer(
        self,
        model,
        enable_critic_warmup: bool = False,
    ) -> Optimizer:
        if enable_critic_warmup:
            raise ValueError("PAD-Frozen jointly starts Gate and value heads.")
        groups = partition_pad_frozen_parameters(model.named_parameters())
        cfg = self._cfg.optim
        betas = (cfg.adam_beta1, cfg.adam_beta2)
        weight_decay = cfg.get("weight_decay", 1e-2)
        param_groups = [
            {
                "name": "gate",
                "params": groups["gate"],
                "lr": cfg.get("gate_lr", cfg.lr),
                "betas": betas,
                "weight_decay": cfg.get("gate_weight_decay", weight_decay),
            },
            {
                "name": "value_head",
                "params": groups["value_head"],
                "lr": cfg.value_lr,
                "betas": betas,
                "weight_decay": cfg.get("value_weight_decay", weight_decay),
            },
        ]
        return torch.optim.AdamW(
            param_groups,
            eps=cfg.get("adam_eps", 1e-8),
            weight_decay=weight_decay,
        )

    def optimizer_step(self) -> tuple[float, list[float]]:
        """Use the normal FSDP step with PAD's exact two-group audits."""

        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = self._strategy.clip_grad_norm_(model=self.model)
        self._fastwam_last_gradient_norms = pad_frozen_gradient_norms(self.optimizer)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            self._logger.warning(
                f"[FSDP] Non-finite PAD grad norm {grad_norm}; skipping step."
            )
        else:
            if not self._fastwam_update_resolution_checked:
                resolution = assert_pad_frozen_update_resolution(
                    self.optimizer,
                    minimum_half_ulp_ratio=float(
                        self._cfg.optim.update_resolution_min_half_ulp_ratio
                    ),
                )
                self._fastwam_update_resolution_checked = True
                self._logger.info(
                    "[FSDP] PAD first-step update resolution: "
                    f"{json.dumps(resolution, sort_keys=True)}"
                )
            self.grad_scaler.step(optimizer=self.optimizer)
            self.optimizer_steps += 1
        self.grad_scaler.update()
        return grad_norm, [group["lr"] for group in self.optimizer.param_groups]

    def _compute_fastwam_loss(
        self,
        *,
        micro_batch: dict,
        output_dict: dict[str, torch.Tensor],
        selected_loss_scales: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Gate PPO plus critic loss, never action-policy loss."""

        required = {
            "route_info",
            "emitted_gate",
            "gate_advantages",
            "gate_valid_mask",
            "returns",
            "prev_values",
        }
        missing = sorted(required - set(micro_batch))
        if missing:
            raise KeyError(f"PAD training batch is missing fields: {missing}.")
        cfg = self.cfg.algorithm.gate_ppo
        route_info = micro_batch["route_info"]
        emitted = micro_batch["emitted_gate"]
        scales = selected_loss_scales or {}
        policy_loss, metrics = compute_pad_frozen_policy_loss(
            gate_logprobs=output_dict["gate_logprobs"].float(),
            gate_old_logprobs=emitted.old_logprob.float(),
            gate_advantages=micro_batch["gate_advantages"].float(),
            gate_valid_mask=micro_batch["gate_valid_mask"].bool(),
            gate_clip_ratio_low=float(cfg.clip_ratio_low),
            gate_clip_ratio_high=float(cfg.clip_ratio_high),
            gate_base_probabilities=(
                output_dict["gate_base_probabilities"].float()
                if str(cfg.get("entropy_metric_source", "behavior")).lower() == "base"
                else None
            ),
            gate_behavior_probabilities=output_dict[
                "gate_behavior_probabilities"
            ].float(),
            gate_entropy_coefficient=float(cfg.entropy_coefficient),
            gate_loss_coefficient=float(cfg.get("loss_weight", 1.0)),
            selected_loss_scale=scales.get("gate"),
        )
        metrics.update(
            absent_uncond_flow_metrics(
                route_used=route_info.route_used,
                valid_chunk_mask=micro_batch["gate_valid_mask"].bool(),
                reference=policy_loss,
            )
        )
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
        total = policy_loss + float(critic_cfg.get("loss_weight", 1.0)) * critic_loss
        metrics.update(critic_metrics)
        metrics.update(
            {
                "fastwam/regularized_policy_loss": policy_loss.detach(),
                "fastwam/total_loss": total.detach(),
            }
        )
        return total, {
            key: value.detach().item() if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()
        }

    def _optimizer_metrics(
        self,
        grad_norm: float,
        lr_list: list[float],
    ) -> dict[str, float]:
        groups = {
            str(group.get("name", "")): group for group in self.optimizer.param_groups
        }
        if set(groups) != {"gate", "value_head"}:
            raise RuntimeError(f"PAD optimizer groups changed: {sorted(groups)}.")
        norms = self._fastwam_last_gradient_norms
        if set(norms) != {"gate", "value_head"}:
            raise RuntimeError(f"PAD gradient groups changed: {sorted(norms)}.")
        return {
            "actor/grad_norm": float(grad_norm),
            "gate/lr": float(groups["gate"]["lr"]),
            "critic/lr": float(groups["value_head"]["lr"]),
            "gate/grad_norm": float(norms["gate"]),
            "value_head/grad_norm": float(norms["value_head"]),
        }

    @staticmethod
    def _module_parameter_sha256(module: torch.nn.Module) -> str:
        """Stream one frozen expert state without retaining a second copy."""

        digest = hashlib.sha256()
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                raise RuntimeError(f"PAD frozen expert parameter is trainable: {name}.")
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _trainable_snapshot(policy: PadFrozenPolicy) -> dict[str, torch.Tensor]:
        state = {
            f"gate.{name}": parameter.detach().cpu().contiguous().clone()
            for name, parameter in policy.gate.named_parameters()
            if parameter.requires_grad
        }
        state.update(
            {
                f"value_head.{name}": parameter.detach().cpu().contiguous().clone()
                for name, parameter in policy.value_head.named_parameters()
                if parameter.requires_grad
            }
        )
        if not state:
            raise RuntimeError("PAD ownership audit found no Gate/value tensors.")
        return state

    @staticmethod
    def _movement_summary(
        before: dict[str, torch.Tensor],
        after: dict[str, torch.Tensor],
        *,
        prefix: str,
    ) -> dict[str, float | int]:
        names = [name for name in sorted(before) if name.startswith(prefix)]
        if not names or set(before) != set(after):
            raise RuntimeError("PAD ownership audit trainable tensor names changed.")
        square_sum = 0.0
        max_abs = 0.0
        changed = 0
        values = 0
        for name in names:
            left, right = before[name], after[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError(f"PAD ownership tensor metadata changed: {name}.")
            delta = right.float() - left.float()
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError(f"PAD ownership delta is non-finite: {name}.")
            square_sum += float(delta.square().sum(dtype=torch.float64).item())
            max_abs = max(max_abs, float(delta.abs().max().item()))
            changed += int(torch.count_nonzero(delta).item())
            values += int(delta.numel())
        l2 = math.sqrt(square_sum)
        if changed == 0 or not math.isfinite(l2) or l2 <= 0.0:
            raise RuntimeError(f"PAD {prefix.rstrip('.')} did not move in its canary.")
        return {
            "tensor_count": len(names),
            "value_count": values,
            "changed_value_count": changed,
            "update_l2_norm": l2,
            "update_max_abs": max_abs,
        }

    def run_training(
        self,
        kv_request_channel=None,
        kv_response_channel=None,
    ) -> None:
        """Wrap selected updates with exact Stage 1 ownership evidence."""

        update = int(self.version) + 1
        audited_updates = {
            int(item)
            for item in self.cfg.runner.get(
                "pad_rv_action_ownership_audit_updates", [1]
            )
        }
        if update not in audited_updates:
            return super().run_training(
                kv_request_channel=kv_request_channel,
                kv_response_channel=kv_response_channel,
            )
        policy = self._fastwam_policy_module()
        if not isinstance(policy, PadFrozenPolicy):
            raise TypeError("PAD ownership audit requires PadFrozenPolicy.")
        trainable_before = self._trainable_snapshot(policy)
        idm_before = self._module_parameter_sha256(policy.actor.action_expert)
        uncond_before = self._module_parameter_sha256(policy.uncond_action_expert)
        result = super().run_training(
            kv_request_channel=kv_request_channel,
            kv_response_channel=kv_response_channel,
        )
        trainable_after = self._trainable_snapshot(policy)
        idm_after = self._module_parameter_sha256(policy.actor.action_expert)
        uncond_after = self._module_parameter_sha256(policy.uncond_action_expert)
        if idm_before != idm_after or uncond_before != uncond_after:
            raise RuntimeError("PAD optimizer mutated a frozen ActionDiT.")
        audit = {
            "schema": "pad-frozen-ownership-audit-v1",
            "runner_update": update,
            "optimizer_steps_after": int(self.optimizer_steps),
            "gate": self._movement_summary(
                trainable_before, trainable_after, prefix="gate."
            ),
            "value_head": self._movement_summary(
                trainable_before, trainable_after, prefix="value_head."
            ),
            "idm_action_expert_unchanged": True,
            "uncond_action_expert_unchanged": True,
            "status": "PASS",
        }
        self._pad_last_ownership_audit = audit
        if self._rank == 0:
            print(
                "PAD_FROZEN_OWNERSHIP_AUDIT " + json.dumps(audit, sort_keys=True),
                flush=True,
            )
        return result
