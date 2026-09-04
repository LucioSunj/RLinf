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
from rlinf.models.embodiment.wam_policy.contracts import WAMRoute
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
from rlinf.utils.nested_dict_process import map_nested_tensors
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    fastwam_effective_gate_kv_mask,
)

from .policy import RouteNeutralOnlineIDMBCFastWAMPolicy

_COMPACTION_METRIC_PREFIX = "_route_neutral_compaction/"


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
        self._route_neutral_metric_global_batch = -1
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

    def load_checkpoint(self, load_path: str) -> int | None:
        """Do not repeat the native first-joint-update audit after a resume."""

        loaded_step = super().load_checkpoint(load_path)
        if (
            loaded_step is not None
            and int(loaded_step) > self.critic_warmup.runner_updates
            and int(self.optimizer_steps) > 0
        ):
            self._fastwam_update_resolution_checked = True
            self._logger.info(
                "[FSDP] Preserving completed first joint route-neutral update "
                f"resolution audit from resumed step {int(loaded_step)}."
            )
        return loaded_step

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

    @staticmethod
    def _active_rows(mask: torch.Tensor, *, batch_size: int, name: str) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor) or mask.shape[0] != batch_size:
            raise ValueError(
                f"Route-neutral {name} must begin with batch size {batch_size}."
            )
        return mask.bool().reshape(batch_size, -1).any(dim=1)

    def _prepare_train_global_batch_for_microbatches(
        self,
        train_global_batch: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Remove rows with zero contribution while preserving global divisors."""

        reference = self._training_batch_reference(train_global_batch)
        batch_size = int(reference.shape[0])
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        if micro_batch_size not in {1, 4}:
            raise ValueError("Route-neutral actor microbatch must be 1 or 4.")
        self._route_neutral_metric_global_batch = (
            int(getattr(self, "_route_neutral_metric_global_batch", -1)) + 1
        )
        route_info = train_global_batch.get("route_info")
        if route_info is None:
            raise KeyError("Route-neutral compaction requires route_info.")
        gate_rows = self._active_rows(
            fastwam_effective_gate_kv_mask(
                train_global_batch["gate_valid_mask"],
                train_global_batch.get("gate_kv_sample_mask"),
            ),
            batch_size=batch_size,
            name="effective Gate mask",
        )
        flow_rows = self._active_rows(
            train_global_batch["flow_valid_mask"].bool()
            & (route_info.route_used == int(WAMRoute.UNCOND)),
            batch_size=batch_size,
            name="effective Flow mask",
        )
        loss_mask = train_global_batch.get("loss_mask")
        if loss_mask is None:
            raise KeyError("Route-neutral compaction requires critic loss_mask.")
        critic_rows = self._active_rows(
            loss_mask,
            batch_size=batch_size,
            name="critic loss mask",
        )
        active = gate_rows | flow_rows | critic_rows
        active_indices = active.nonzero(as_tuple=False).reshape(-1)
        active_count = int(active_indices.numel())

        if micro_batch_size == 1:
            return train_global_batch, {
                "perf/actor_rows_original": float(batch_size),
                "perf/actor_rows_active": float(active_count),
                "perf/actor_rows_padded": 0.0,
                "perf/actor_rows_forwarded": float(batch_size),
                "perf/actor_rows_saved_fraction": 0.0,
                "perf/actor_microbatches_executed": float(batch_size),
            }

        if active_count == 0:
            selected_indices = torch.arange(
                micro_batch_size,
                device=active.device,
                dtype=torch.long,
            )
        else:
            padding = (-active_count) % micro_batch_size
            if padding:
                inactive_indices = (~active).nonzero(as_tuple=False).reshape(-1)
                selected_indices = torch.cat(
                    (active_indices, inactive_indices[:padding]),
                    dim=0,
                )
            else:
                selected_indices = active_indices
        forwarded_count = int(selected_indices.numel())
        padded_count = forwarded_count - active_count

        def _select_rows(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim < 1 or tensor.shape[0] != batch_size:
                raise ValueError(
                    "Route-neutral flattened replay tensors must all begin with "
                    f"batch size {batch_size}, got {tuple(tensor.shape)}."
                )
            return tensor.index_select(
                0,
                selected_indices.to(tensor.device),
            )

        compacted = map_nested_tensors(train_global_batch, _select_rows)
        loss_mask_sum = compacted.get("loss_mask_sum")
        if isinstance(loss_mask_sum, torch.Tensor):
            # ``masked_mean_ratio`` divides before applying the boolean mask.
            # A dummy or padding row therefore needs a finite, nonzero divisor
            # even though its masked contribution remains exactly zero.
            compacted["loss_mask_sum"] = loss_mask_sum.clamp_min(1)
        return compacted, {
            "perf/actor_rows_original": float(batch_size),
            "perf/actor_rows_active": float(active_count),
            "perf/actor_rows_padded": float(padded_count),
            "perf/actor_rows_forwarded": float(forwarded_count),
            "perf/actor_rows_saved_fraction": float(
                (batch_size - forwarded_count) / batch_size
            ),
            "perf/actor_microbatches_executed": float(
                forwarded_count // micro_batch_size
            ),
        }

    @staticmethod
    def _append_route_neutral_perf_metrics(
        metrics: dict[str, float],
        output_dict: dict[str, torch.Tensor],
    ) -> None:
        for name in (
            "perf/rollout_idm_batch_size",
            "perf/rollout_uncond_batch_size",
            "perf/teacher_batch_size",
        ):
            value = output_dict.get(name)
            if isinstance(value, torch.Tensor):
                metrics[name] = float(value.detach().item())
        if "critic/value_clip_ratio" in metrics:
            metrics["critic/value_clip_ratio_compacted"] = metrics[
                "critic/value_clip_ratio"
            ]

    def _append_route_neutral_metric_numerators(
        self,
        metrics: dict[str, float],
        output_dict: dict[str, torch.Tensor],
    ) -> None:
        """Retain additive metric state before skipped zero rows disappear."""

        def scalar(value: Any) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.detach().item())
            return float(value)

        def record(name: str, value: Any) -> None:
            metrics[f"{_COMPACTION_METRIC_PREFIX}{name}"] = scalar(value)

        record("global_batch", self._route_neutral_metric_global_batch)
        for name in (
            "critic/value_loss",
            "fastwam/regularized_policy_loss",
            "fastwam/total_loss",
        ):
            if name in metrics:
                record(f"loss/{name}", metrics[name])
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        for prefix in ("gate", "uncond_flow", "online_idm_bc"):
            key = f"{prefix}/selected_loss_scale"
            if key not in metrics:
                continue
            actual = scalar(metrics[key])
            record(f"scale/{prefix}", actual)
            metrics[f"{key}_compacted"] = actual
            metrics[key] = actual * micro_batch_size

        if "online_idm_bc/raw_loss" not in metrics:
            return
        selected = scalar(output_dict["online_idm_bc_selected_count"])
        record("online/selected", selected)
        for output_name, metric_name in (
            ("online_idm_bc_loss_sum", "loss_sum"),
            ("online_idm_bc_expected_count", "expected"),
            ("online_idm_bc_present_count", "present"),
            ("online_idm_bc_valid_action_count", "valid_action_count"),
            ("online_idm_bc_teacher_seconds_sum", "teacher_seconds"),
            ("online_idm_bc_teacher_bytes_sum", "teacher_bytes"),
        ):
            record(f"online/{metric_name}", output_dict[output_name])
        for output_name, metric_name in (
            ("online_idm_bc_mse_pose", "mse_pose"),
            ("online_idm_bc_mse_gripper", "mse_gripper"),
            ("online_idm_bc_full_action_mse", "full_action_mse"),
            ("online_idm_bc_executed_prefix_mse", "executed_prefix_mse"),
        ):
            record(
                f"online/{metric_name}_sum", scalar(output_dict[output_name]) * selected
            )
        for index, value in enumerate(
            output_dict["online_idm_bc_mse_per_dimension"].reshape(-1)
        ):
            record(f"online/mse_dimension_sum_{index}", scalar(value) * selected)
        bin_mse = output_dict["online_idm_bc_mse_by_timestep_bin"].reshape(-1)
        bin_counts = output_dict["online_idm_bc_timestep_bin_count"].reshape(-1)
        for index, (mse, count) in enumerate(zip(bin_mse, bin_counts, strict=True)):
            count_value = scalar(count)
            record(f"online/timestep_count_{index}", count_value)
            record(
                f"online/timestep_mse_sum_{index}",
                scalar(mse) * count_value,
            )

    def _finalize_train_metrics_before_reduction(
        self,
        metrics: dict[str, list[float]],
    ) -> None:
        """Restore MB1 metric denominators without forwarding inactive rows."""

        prefix = _COMPACTION_METRIC_PREFIX
        batch_ids = [int(value) for value in metrics.pop(f"{prefix}global_batch", [])]
        if not batch_ids:
            return
        groups = {
            batch_id: [
                index for index, value in enumerate(batch_ids) if value == batch_id
            ]
            for batch_id in sorted(set(batch_ids))
        }
        group_count = len(groups)
        gradient_accumulation = float(self.gradient_accumulation)
        original_rows = float(
            int(self.cfg.actor.global_batch_size) // int(self._world_size)
        )
        micro_batch_size = float(self.cfg.actor.micro_batch_size)

        def take(name: str) -> list[float]:
            values = [float(value) for value in metrics.pop(f"{prefix}{name}", [])]
            if values and len(values) != len(batch_ids):
                raise ValueError(f"Compaction metric {name!r} lost a microbatch value.")
            return values

        def group_sums(values: list[float]) -> list[float]:
            return [
                sum(values[index] for index in indices) for indices in groups.values()
            ]

        for metric_name in (
            "critic/value_loss",
            "fastwam/regularized_policy_loss",
            "fastwam/total_loss",
        ):
            values = take(f"loss/{metric_name}")
            if values:
                metrics[metric_name] = [
                    sum(group_sums(values)) / gradient_accumulation / group_count
                ]

        actor_total = [float(value) for value in metrics.get("actor/total_loss", [])]
        if actor_total:
            if len(actor_total) != len(batch_ids):
                raise ValueError("Actor total-loss metrics lost a microbatch value.")
            metrics["actor/total_loss"] = [
                sum(group_sums(actor_total)) / original_rows / group_count
            ]

        for owner in ("gate", "uncond_flow", "online_idm_bc"):
            values = take(f"scale/{owner}")
            if not values:
                continue
            compacted = []
            for indices in groups.values():
                group_values = {values[index] for index in indices}
                if len(group_values) != 1:
                    raise ValueError(
                        f"{owner} selected scale changed within one global batch."
                    )
                compacted.append(group_values.pop())
            metrics[f"{owner}/selected_loss_scale_compacted"] = [
                sum(compacted) / group_count
            ]
            metrics[f"{owner}/selected_loss_scale"] = [
                sum(value * micro_batch_size for value in compacted) / group_count
            ]

        selected_values = take("online/selected")
        if selected_values:
            self._finalize_online_bc_compaction_metrics(
                metrics=metrics,
                take=take,
                group_sums=group_sums,
                selected_values=selected_values,
                group_count=group_count,
            )
        leftovers = sorted(name for name in metrics if name.startswith(prefix))
        if leftovers:
            raise RuntimeError(f"Unconsumed compaction metrics: {leftovers}.")

    @staticmethod
    def _finalize_online_bc_compaction_metrics(
        *,
        metrics: dict[str, list[float]],
        take,
        group_sums,
        selected_values: list[float],
        group_count: int,
    ) -> None:
        """Recover the historical per-global-batch and selected-row means."""

        selected_by_group = group_sums(selected_values)
        loss_by_group = group_sums(take("online/loss_sum"))
        raw_loss = (
            sum(
                loss / selected if selected > 0.0 else 0.0
                for loss, selected in zip(loss_by_group, selected_by_group, strict=True)
            )
            / group_count
        )
        loss_weight = float(metrics.get("online_idm_bc/loss_weight", [0.0])[0])
        metrics["online_idm_bc/raw_loss"] = [raw_loss]
        metrics["online_idm_bc/weighted_loss"] = [loss_weight * raw_loss]

        expected_by_group = group_sums(take("online/expected"))
        present_values = take("online/present")
        present_by_group = group_sums(present_values)
        teacher_seconds = take("online/teacher_seconds")
        teacher_bytes = take("online/teacher_bytes")
        metrics["online_idm_bc/expected_count"] = [sum(expected_by_group) / group_count]
        metrics["online_idm_bc/selected_count"] = [sum(selected_by_group) / group_count]
        metrics["online_idm_bc/teacher_call_count"] = [
            sum(present_by_group) / group_count
        ]
        metrics["online_idm_bc/teacher_seconds"] = [
            sum(group_sums(teacher_seconds)) / group_count
        ]
        metrics["online_idm_bc/transported_bytes"] = [
            sum(group_sums(teacher_bytes)) / group_count
        ]
        metrics["online_idm_bc/globally_normalized_count"] = [
            sum(selected_by_group) / group_count
        ]

        total_selected = sum(selected_values)
        total_present = sum(present_values)
        if total_present > 0.0:
            metrics["online_idm_bc/teacher_seconds_per_call"] = [
                sum(teacher_seconds) / total_present
            ]
            metrics["online_idm_bc/teacher_bytes_per_call"] = [
                sum(teacher_bytes) / total_present
            ]
        detailed_values = {
            metric_name: take(f"online/{hidden_name}")
            for hidden_name, metric_name in (
                ("valid_action_count", "valid_action_count"),
                ("mse_pose_sum", "mse_pose"),
                ("mse_gripper_sum", "mse_gripper"),
                ("full_action_mse_sum", "full_action_mse"),
                ("executed_prefix_mse_sum", "executed_prefix_mse"),
            )
        }
        dimension_values = [
            take(f"online/mse_dimension_sum_{index}") for index in range(7)
        ]
        timestep_values = [
            (
                take(f"online/timestep_count_{index}"),
                take(f"online/timestep_mse_sum_{index}"),
            )
            for index in range(10)
        ]
        if total_selected <= 0.0:
            return
        for metric_name, values in detailed_values.items():
            metrics[f"online_idm_bc/{metric_name}"] = [sum(values) / total_selected]
        for index, values in enumerate(dimension_values):
            metrics[f"online_idm_bc/mse_dimension_{index}"] = [
                sum(values) / total_selected
            ]
        for index, (count_values, mse_values) in enumerate(timestep_values):
            count = sum(count_values)
            mse_sum = sum(mse_values)
            if count > 0.0:
                metrics[f"online_idm_bc/mse_timestep_bin_{index}"] = [mse_sum / count]
                metrics[f"online_idm_bc/timestep_bin_count_{index}"] = [1.0]
                metrics[
                    f"online_idm_bc/timestep_bin_selected_count_compacted_{index}"
                ] = [count / group_count]

    def _finalize_train_metrics_after_reduction(
        self,
        metrics: dict[str, float],
    ) -> None:
        """Recompute a ratio whose two final operands use restored denominators."""

        weighted = metrics.get("online_idm_bc/weighted_loss")
        flow = metrics.get("uncond_flow/total_loss")
        if weighted is not None and flow is not None:
            metrics["online_idm_bc/weighted_to_flow_loss_ratio"] = abs(
                float(weighted)
            ) / max(abs(float(flow)), 1.0e-12)

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
            loss, metrics = super()._compute_fastwam_loss(
                micro_batch=micro_batch,
                output_dict=output_dict,
                selected_loss_scales=selected_loss_scales,
            )
            self._append_route_neutral_perf_metrics(metrics, output_dict)
            self._append_route_neutral_metric_numerators(metrics, output_dict)
            return loss, metrics

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
        scalar_metrics = {
            key: value.detach().item() if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()
        }
        self._append_route_neutral_perf_metrics(scalar_metrics, output_dict)
        self._append_route_neutral_metric_numerators(
            scalar_metrics,
            output_dict,
        )
        return loss, scalar_metrics

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
