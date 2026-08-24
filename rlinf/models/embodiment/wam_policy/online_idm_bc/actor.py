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

"""FSDP actor subclass that assembles online IDM BC with adaptive PPO."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from hydra.utils import get_class

from rlinf.models.embodiment.wam_policy.adaptive_policy import FastWAMAdaptivePolicy
from rlinf.scheduler import Worker
from rlinf.workers.actor.fastwam_selective_sync import prepare_fastwam_sync_tensors
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

from .config import ONLINE_IDM_BC_FLOW_VALID, OnlineIDMBCConfig
from .policy import OnlineIDMBCFastWAMPolicy

_ONLINE_BC_OUTPUT_KEYS = {
    "online_idm_bc_loss_sum",
    "online_idm_bc_raw_loss",
    "online_idm_bc_selected_count",
    "online_idm_bc_expected_count",
    "online_idm_bc_present_count",
    "online_idm_bc_mse_per_dimension",
    "online_idm_bc_mse_pose",
    "online_idm_bc_mse_gripper",
    "online_idm_bc_mse_by_timestep_bin",
    "online_idm_bc_timestep_bin_count",
    "online_idm_bc_valid_action_count",
    "online_idm_bc_full_action_mse",
    "online_idm_bc_executed_prefix_mse",
    "online_idm_bc_teacher_seconds_sum",
    "online_idm_bc_teacher_bytes_sum",
}


def assemble_online_idm_bc_loss(
    *,
    current_loss: torch.Tensor,
    output_dict: dict[str, torch.Tensor],
    config: OnlineIDMBCConfig,
    selected_loss_scale: float | None,
    metric_scale_numerator: float,
    flow_metric_loss: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply global Flow-selected normalization and build scalar diagnostics."""

    missing = sorted(_ONLINE_BC_OUTPUT_KEYS - set(output_dict))
    if missing:
        raise KeyError(f"Online IDM BC policy outputs are missing: {missing}.")
    scale = 0.0 if selected_loss_scale is None else float(selected_loss_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("Online IDM BC selected loss scale must be nonnegative.")
    if not math.isfinite(metric_scale_numerator) or metric_scale_numerator <= 0.0:
        raise ValueError("Online IDM BC metric scale numerator must be positive.")
    local_selected = float(output_dict["online_idm_bc_selected_count"].item())
    if scale == 0.0 and local_selected != 0.0:
        raise RuntimeError(
            "Online IDM BC selected samples received a zero global scale."
        )

    normalized_loss = output_dict["online_idm_bc_loss_sum"].float() * scale
    weighted_loss = float(config.loss_weight) * normalized_loss
    total_loss = current_loss + weighted_loss
    raw_loss = float(normalized_loss.detach().item())
    weighted_metric_loss = float(config.loss_weight) * raw_loss
    denominator = max(abs(float(flow_metric_loss)), 1.0e-12)
    present = float(output_dict["online_idm_bc_present_count"].item())
    teacher_seconds = float(output_dict["online_idm_bc_teacher_seconds_sum"].item())
    teacher_bytes = float(output_dict["online_idm_bc_teacher_bytes_sum"].item())
    metrics = {
        "online_idm_bc/raw_loss": raw_loss,
        "online_idm_bc/weighted_loss": weighted_metric_loss,
        "online_idm_bc/weighted_to_flow_loss_ratio": (
            abs(weighted_metric_loss) / denominator
        ),
        "online_idm_bc/loss_weight": float(config.loss_weight),
        "online_idm_bc/selected_loss_scale": scale,
        "online_idm_bc/expected_count": metric_scale_numerator
        * float(output_dict["online_idm_bc_expected_count"].item()),
        "online_idm_bc/teacher_call_count": metric_scale_numerator * present,
        "online_idm_bc/selected_count": metric_scale_numerator * local_selected,
        "online_idm_bc/teacher_seconds": metric_scale_numerator * teacher_seconds,
        "online_idm_bc/transported_bytes": metric_scale_numerator * teacher_bytes,
        "online_idm_bc/globally_normalized_count": (
            metric_scale_numerator / scale if scale > 0.0 else 0.0
        ),
    }
    if local_selected > 0.0:
        metrics.update(
            {
                "online_idm_bc/valid_action_count": float(
                    output_dict["online_idm_bc_valid_action_count"].item()
                ),
                "online_idm_bc/mse_pose": float(
                    output_dict["online_idm_bc_mse_pose"].item()
                ),
                "online_idm_bc/mse_gripper": float(
                    output_dict["online_idm_bc_mse_gripper"].item()
                ),
                "online_idm_bc/full_action_mse": float(
                    output_dict["online_idm_bc_full_action_mse"].item()
                ),
                "online_idm_bc/executed_prefix_mse": float(
                    output_dict["online_idm_bc_executed_prefix_mse"].item()
                ),
            }
        )
        for index, value in enumerate(
            output_dict["online_idm_bc_mse_per_dimension"].reshape(-1).tolist()
        ):
            metrics[f"online_idm_bc/mse_dimension_{index}"] = float(value)
        bin_values = output_dict["online_idm_bc_mse_by_timestep_bin"].reshape(-1)
        bin_counts = output_dict["online_idm_bc_timestep_bin_count"].reshape(-1)
        for index, count in enumerate(bin_counts.tolist()):
            if int(count) > 0:
                metrics[f"online_idm_bc/mse_timestep_bin_{index}"] = float(
                    bin_values[index].item()
                )
                metrics[f"online_idm_bc/timestep_bin_count_{index}"] = float(count)
    if present > 0.0:
        metrics["online_idm_bc/teacher_seconds_per_call"] = teacher_seconds / present
        metrics["online_idm_bc/teacher_bytes_per_call"] = teacher_bytes / present
    return total_loss, metrics


def audit_online_idm_bc_gradient_ownership(
    *,
    bc_loss: torch.Tensor,
    policy: OnlineIDMBCFastWAMPolicy,
) -> dict[str, float]:
    """Prove that the isolated BC graph reaches LoRA and no other owner."""

    lora_parameters = tuple(policy.lora_adapter.lora_parameters())
    lora_ids = {id(parameter) for parameter in lora_parameters}
    gate_parameters = tuple(
        parameter for parameter in policy.gate.parameters() if parameter.requires_grad
    )
    critic_parameters = (
        ()
        if policy.critic is None
        else tuple(
            parameter
            for parameter in policy.critic.parameters()
            if parameter.requires_grad
        )
    )
    base_trainable = tuple(
        parameter
        for parameter in policy.actor.parameters()
        if parameter.requires_grad and id(parameter) not in lora_ids
    )
    if base_trainable:
        raise RuntimeError(
            "Online IDM BC found trainable frozen/base actor parameters."
        )
    candidates = lora_parameters + gate_parameters + critic_parameters
    gradients = torch.autograd.grad(
        bc_loss,
        candidates,
        retain_graph=True,
        allow_unused=True,
    )
    lora_gradients = gradients[: len(lora_parameters)]
    other_gradients = gradients[len(lora_parameters) :]
    finite_lora = [
        gradient
        for gradient in lora_gradients
        if gradient is not None and bool(torch.isfinite(gradient).all().item())
    ]
    nonzero_lora = [
        gradient for gradient in finite_lora if bool((gradient != 0).any().item())
    ]
    if len(finite_lora) != sum(gradient is not None for gradient in lora_gradients):
        raise FloatingPointError("Online IDM BC produced a non-finite LoRA gradient.")
    if not nonzero_lora:
        raise RuntimeError("Online IDM BC produced no nonzero LoRA gradient.")
    if any(
        gradient is not None and bool((gradient != 0).any().item())
        for gradient in other_gradients
    ):
        raise RuntimeError("Online IDM BC gradient escaped into Gate or critic state.")
    return {
        "online_idm_bc/gradient_audit_pass": 1.0,
        "online_idm_bc/gradient_audit_lora_parameter_count": float(
            len(lora_parameters)
        ),
        "online_idm_bc/gradient_audit_lora_nonzero_count": float(len(nonzero_lora)),
        "online_idm_bc/gradient_audit_gate_nonzero_count": 0.0,
        "online_idm_bc/gradient_audit_critic_nonzero_count": 0.0,
        "online_idm_bc/gradient_audit_base_trainable_count": 0.0,
    }


class OnlineIDMBCFSDPActor(EmbodiedFSDPActor):
    """Select the online policy and add its BC numerator to the actor loss."""

    def __init__(self, cfg) -> None:
        self.online_idm_bc_config = OnlineIDMBCConfig.from_mapping(
            cfg.algorithm.uncond_idm_bc
        )
        if not self.online_idm_bc_config.enabled:
            raise ValueError("Online IDM BC actor requires `enabled: true`.")
        self._online_idm_bc_gradient_audit_complete = False
        super().__init__(cfg)

    def model_provider_func(self) -> OnlineIDMBCFastWAMPolicy:
        """Reuse the standard builder, then replace only the policy wrapper."""

        model = super().model_provider_func()
        if type(model) is not FastWAMAdaptivePolicy:
            raise TypeError(
                "Online IDM BC actor expected FastWAMAdaptivePolicy, got "
                f"{type(model).__name__}."
            )
        policy_cls = get_class(str(self.cfg.online_idm_bc_implementation.policy_target))
        if policy_cls is not OnlineIDMBCFastWAMPolicy:
            raise TypeError(
                "Online IDM BC policy target must resolve to OnlineIDMBCFastWAMPolicy."
            )
        wrapped = policy_cls.from_base_policy(
            model,
            config=self.online_idm_bc_config,
        )
        local_device = torch.device(
            Worker.torch_device_type,
            int(os.environ.get("LOCAL_RANK", 0)),
        )
        self._fastwam_rollout_sync_tensors = prepare_fastwam_sync_tensors(
            wrapped,
            device=local_device,
        )
        return wrapped

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
        selected_loss_scales: dict[str, float] | None = None,
    ) -> None:
        """Expose the already-authoritative Flow-valid mask to policy replay."""

        if "flow_valid_mask" not in micro_batch:
            raise KeyError("Online IDM BC requires the existing flow_valid_mask.")
        forward_inputs = dict(micro_batch.get("forward_inputs", {}))
        if ONLINE_IDM_BC_FLOW_VALID in forward_inputs:
            raise KeyError("Online IDM BC flow-valid replay field already exists.")
        forward_inputs[ONLINE_IDM_BC_FLOW_VALID] = micro_batch["flow_valid_mask"]
        online_batch: dict[str, Any] = dict(micro_batch)
        online_batch["forward_inputs"] = forward_inputs
        super().train_micro_batch(
            online_batch,
            metrics,
            is_last=is_last,
            selected_loss_scales=selected_loss_scales,
        )

    def _compute_fastwam_loss(
        self,
        *,
        micro_batch: dict,
        output_dict: dict[str, torch.Tensor],
        selected_loss_scales: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Preserve the current loss and append fixed-weight online BC."""

        current_loss, metrics = super()._compute_fastwam_loss(
            micro_batch=micro_batch,
            output_dict=output_dict,
            selected_loss_scales=selected_loss_scales,
        )
        scales = selected_loss_scales or {}
        total_loss, online_metrics = assemble_online_idm_bc_loss(
            current_loss=current_loss,
            output_dict=output_dict,
            config=self.online_idm_bc_config,
            selected_loss_scale=scales.get("flow"),
            metric_scale_numerator=float(self.gradient_accumulation * self._world_size),
            flow_metric_loss=float(metrics.get("uncond_flow/total_loss", 0.0)),
        )
        if (
            not self._online_idm_bc_gradient_audit_complete
            and float(output_dict["online_idm_bc_selected_count"].item()) > 0.0
        ):
            online_metrics.update(
                audit_online_idm_bc_gradient_ownership(
                    bc_loss=(
                        float(self.online_idm_bc_config.loss_weight)
                        * output_dict["online_idm_bc_loss_sum"].float()
                    ),
                    policy=self._fastwam_policy_module(),
                )
            )
            self._online_idm_bc_gradient_audit_complete = True
        metrics.update(online_metrics)
        metrics["fastwam/total_loss"] = float(total_loss.detach().item())
        return total_loss, metrics
