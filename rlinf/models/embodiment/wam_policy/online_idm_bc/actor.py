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
from rlinf.models.embodiment.wam_policy.contracts import WAMRoute
from rlinf.scheduler import Worker
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.actor.fastwam_selective_sync import prepare_fastwam_sync_tensors
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

from .config import (
    ONLINE_IDM_BC_FLOW_VALID,
    ONLINE_IDM_BC_TEACHER_PRESENT,
    OnlineIDMBCConfig,
)
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


def audit_online_idm_bc_backward_gradient_ownership(
    *,
    optimizer: torch.optim.Optimizer,
    policy: OnlineIDMBCFastWAMPolicy,
) -> dict[str, float]:
    """Audit one completed BC-only backward through FSDP optimizer groups."""

    groups = {str(group.get("name", "")): group for group in optimizer.param_groups}
    expected_groups = {"gate", "uncond_lora", "value_head"}
    if set(groups) != expected_groups:
        raise RuntimeError(
            "Online IDM BC gradient audit requires the existing Gate, UNCOND LoRA, "
            "and value-head optimizer groups."
        )
    group_parameters = {name: tuple(group["params"]) for name, group in groups.items()}
    lora_parameters = tuple(policy.lora_adapter.lora_parameters())
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
    expected_parameter_ids = {
        "uncond_lora": {id(parameter) for parameter in lora_parameters},
        "gate": {id(parameter) for parameter in gate_parameters},
        "value_head": {id(parameter) for parameter in critic_parameters},
    }
    for name, parameters in group_parameters.items():
        if {id(parameter) for parameter in parameters} != expected_parameter_ids[name]:
            raise RuntimeError(
                f"Online IDM BC optimizer ownership differs for group {name!r}."
            )

    lora_ids = expected_parameter_ids["uncond_lora"]
    for parameter in policy.actor.parameters():
        if id(parameter) in lora_ids:
            continue
        if parameter.requires_grad:
            raise RuntimeError(
                "Online IDM BC found trainable frozen/base actor parameters."
            )
        if parameter.grad is not None:
            raise RuntimeError("Online IDM BC wrote a frozen/base actor gradient.")

    nonzero_counts: dict[str, int] = {}
    for name, parameters in group_parameters.items():
        nonzero = 0
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError(
                    f"Online IDM BC produced a non-finite {name} gradient."
                )
            nonzero += int(bool((gradient != 0).any().item()))
        nonzero_counts[name] = nonzero
    if nonzero_counts["uncond_lora"] == 0:
        raise RuntimeError("Online IDM BC produced no nonzero LoRA gradient.")
    if nonzero_counts["gate"] or nonzero_counts["value_head"]:
        raise RuntimeError("Online IDM BC gradient escaped into Gate or critic state.")
    return {
        "online_idm_bc/gradient_audit_pass": 1.0,
        "online_idm_bc/gradient_audit_lora_parameter_count": float(
            len(group_parameters["uncond_lora"])
        ),
        "online_idm_bc/gradient_audit_lora_nonzero_count": float(
            nonzero_counts["uncond_lora"]
        ),
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
        self._online_idm_bc_audit_micro_batch: dict[str, Any] | None = None
        self._online_idm_bc_audit_metrics: dict[str, float] | None = None
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

    def _prepare_fastwam_gate_diagnostic_forward_inputs(
        self,
        *,
        micro_batch: dict,
        forward_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Preserve the online policy's existing replay-input contract."""

        if ONLINE_IDM_BC_FLOW_VALID in forward_inputs:
            raise KeyError("Online IDM BC flow-valid replay field already exists.")
        prepared = dict(forward_inputs)
        prepared[ONLINE_IDM_BC_FLOW_VALID] = micro_batch["flow_valid_mask"]
        return prepared

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
        if (
            not self._online_idm_bc_gradient_audit_complete
            and self._online_idm_bc_audit_micro_batch is None
        ):
            routes = online_batch["route_info"].route_used.reshape(-1)
            flow_valid = online_batch["flow_valid_mask"].bool().reshape(-1)
            teacher_present = (
                forward_inputs[ONLINE_IDM_BC_TEACHER_PRESENT].bool().reshape(-1)
            )
            selected = flow_valid & (routes == int(WAMRoute.UNCOND)) & teacher_present
            if bool(selected.any().item()):
                self._online_idm_bc_audit_micro_batch = online_batch
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
        metrics.update(online_metrics)
        metrics["fastwam/total_loss"] = float(total_loss.detach().item())
        return total_loss, metrics

    def _run_online_idm_bc_gradient_audit(self) -> dict[str, float]:
        """Run one side-effect-free BC-only backward after the real update."""

        pending = self._online_idm_bc_audit_micro_batch
        if pending is None:
            raise RuntimeError("Online IDM BC gradient audit has no eligible batch.")
        micro_batch = put_tensor_device(pending, self.device)
        forward_inputs = dict(micro_batch["forward_inputs"])
        emitted_gate = micro_batch["emitted_gate"]
        if "gate_kv_denoise_timesteps" in forward_inputs:
            if emitted_gate.kv_metadata is None:
                raise ValueError("Stored Gate K/V replay requires K/V metadata.")
            forward_inputs["gate_kv_layer_indices"] = torch.tensor(
                emitted_gate.kv_metadata.layer_indices,
                dtype=torch.long,
                device=self.device,
            )

        rng_state = get_rng_state()
        self.optimizer.zero_grad()
        try:
            with self.amp_context:
                output_dict = self.model(
                    forward_inputs=forward_inputs,
                    compute_logprobs=True,
                    compute_entropy=True,
                    compute_values=self.cfg.algorithm.adv_type == "gae",
                    use_cache=False,
                    route_info=micro_batch["route_info"],
                    emitted_gate=emitted_gate,
                    compute_base_logprobs=False,
                )
            if float(output_dict["online_idm_bc_selected_count"].item()) <= 0.0:
                raise RuntimeError(
                    "Online IDM BC audit batch lost its eligible UNCOND sample."
                )
            bc_loss = (
                float(self.online_idm_bc_config.loss_weight)
                * output_dict["online_idm_bc_loss_sum"].float()
            )
            bc_loss.backward()
            restored_handles = (
                self._restore_fastwam_fsdp_parameter_views_after_backward()
            )
            metrics = audit_online_idm_bc_backward_gradient_ownership(
                optimizer=self.optimizer,
                policy=self._fastwam_policy_module(),
            )
            metrics["online_idm_bc/gradient_audit_fsdp_view_restore_handles"] = float(
                restored_handles
            )
            self._online_idm_bc_gradient_audit_complete = True
            return metrics
        finally:
            self.optimizer.zero_grad()
            set_rng_state(rng_state)
            self._online_idm_bc_audit_micro_batch = None

    def optimizer_step(self) -> tuple[float, list[float]]:
        """Preserve the real step, then run the one-shot isolated BC audit."""

        result = super().optimizer_step()
        if (
            not self._online_idm_bc_gradient_audit_complete
            and self._online_idm_bc_audit_micro_batch is not None
        ):
            self._online_idm_bc_audit_metrics = self._run_online_idm_bc_gradient_audit()
        return result

    def _optimizer_metrics(
        self,
        grad_norm: float,
        lr_list: list[float],
    ) -> dict[str, float]:
        """Attach one-shot BC ownership evidence to normal optimizer metrics."""

        metrics = super()._optimizer_metrics(grad_norm, lr_list)
        if self._online_idm_bc_audit_metrics is not None:
            metrics.update(self._online_idm_bc_audit_metrics)
            self._online_idm_bc_audit_metrics = None
        return metrics
