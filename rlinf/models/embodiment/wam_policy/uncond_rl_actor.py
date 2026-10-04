# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pure UNCOND PPO compaction with one authoritative FSDP optimizer."""

from __future__ import annotations

import json
from typing import Any

import torch

from rlinf.scheduler import Channel
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

from .pad_rv.memory import release_pad_host_memory
from .uncond_rl_helpers import (
    UncondTrainingOwnerMixin,
    cpu_to_device_prefetch,
    map_prepared_tensors,
)

_ORIGINAL_BATCHES = "_uncond_compaction/original_microbatches"
_FULL_CLIP_RATIO = "_uncond_compaction/full_value_clip_ratio"
_FLOW_SCALE = "_uncond_compaction/flow_scale"


def compact_uncond_global_batch(
    batch: dict[str, Any], *, micro_batch_size: int
) -> tuple[dict[str, Any], dict[str, float]]:
    """Keep contributing rows inside the original optimizer batch and divisor.

    One complete microbatch is retained even for an empty optimizer batch. Its
    zero-valued backward preserves real zero gradients and the original Adam
    opportunity. Padding selects original inactive rows, without duplication.
    """

    flow_mask = batch["flow_valid_mask"]
    batch_size = int(flow_mask.shape[0])
    if batch_size % micro_batch_size:
        raise ValueError("UNCOND optimizer batches must contain complete microbatches.")
    loss_mask = batch.get("loss_mask")
    if loss_mask is None:
        raise ValueError("UNCOND compaction requires the original critic loss mask.")
    flow_rows = flow_mask.bool().reshape(batch_size, -1).any(dim=1)
    critic_rows = loss_mask.bool().reshape(batch_size, -1).any(dim=1)
    active = flow_rows | critic_rows
    active_indices = active.nonzero(as_tuple=False).reshape(-1)
    active_count = int(active_indices.numel())
    forwarded_count = max(
        micro_batch_size,
        ((active_count + micro_batch_size - 1) // micro_batch_size) * micro_batch_size,
    )
    padding = forwarded_count - active_count
    inactive = (~active).nonzero(as_tuple=False).reshape(-1)
    indices = torch.cat((active_indices, inactive[:padding]))

    def select_rows(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 1 or tensor.shape[0] != batch_size:
            raise ValueError(
                "UNCOND flattened replay tensors must begin with the optimizer "
                f"batch size {batch_size}, got {tuple(tensor.shape)}."
            )
        return tensor.index_select(0, indices.to(tensor.device))

    compacted = map_prepared_tensors(batch, select_rows)
    # This is the original per-episode divisor, not a compacted sample count.
    # masked_mean_ratio divides before applying masks, including padding rows.
    if "loss_mask_sum" in compacted:
        compacted["loss_mask_sum"] = compacted["loss_mask_sum"].clamp_min(1)
    original_microbatches = batch_size // micro_batch_size
    flow_count = int(flow_mask.sum())
    metrics = {
        _ORIGINAL_BATCHES: float(original_microbatches),
        _FLOW_SCALE: original_microbatches / flow_count if flow_count else 0.0,
        "perf/actor_rows_original": float(batch_size),
        "perf/actor_rows_active": float(active_count),
        "perf/actor_rows_padded": float(padding),
        "perf/actor_rows_forwarded": float(forwarded_count),
        "perf/actor_rows_saved_fraction": (batch_size - forwarded_count) / batch_size,
        "perf/actor_microbatches_executed": forwarded_count / micro_batch_size,
    }
    return compacted, metrics


def finalize_uncond_compaction_metrics(metrics: dict[str, list[float]]) -> None:
    """Retain original zero-row denominators and unmasked critic diagnostics."""

    original_counts = metrics.pop(_ORIGINAL_BATCHES, [])
    if not original_counts:
        return
    denominator = sum(float(value) for value in original_counts)
    for name in ("critic/value_loss", "fastwam/total_loss", "actor/total_loss"):
        values = metrics.get(name)
        if values:
            metrics[name] = [sum(float(value) for value in values) / denominator]
    full_clip = metrics.pop(_FULL_CLIP_RATIO, [])
    if len(full_clip) != len(original_counts):
        raise ValueError("UNCOND compaction lost an original-batch critic diagnostic.")
    metrics["critic/value_clip_ratio"] = [
        sum(
            float(value) * float(count)
            for value, count in zip(full_clip, original_counts, strict=True)
        )
        / denominator
    ]
    flow_scales = metrics.pop(_FLOW_SCALE, [])
    if len(flow_scales) != len(original_counts):
        raise ValueError("UNCOND compaction lost an original Flow normalization scale.")
    metrics["uncond_flow/selected_loss_scale"] = [
        sum(
            float(value) * float(count)
            for value, count in zip(flow_scales, original_counts, strict=True)
        )
        / denominator
    ]


class UncondRLFSDPActor(UncondTrainingOwnerMixin, EmbodiedFSDPActor):
    """Compact pure Flow/value replay and optionally share gradient computation."""

    def _release_replay_host_memory(self, *, phase: str) -> None:
        """Return unused replay pages and record the remaining pinned allocations."""

        report = release_pad_host_memory(
            schema="uncond-actor-host-memory-release-v1",
            rank=int(self._rank),
            phase=phase,
        )
        report["actor_version"] = int(self.version)
        pinned = torch.cuda.memory.host_memory_stats()
        report["pinned_bytes"] = {
            name: int(pinned.get(name, 0))
            for name in ("allocated_bytes.current", "reserved_bytes.current")
        }
        self.log_info(f"UNCOND_ACTOR_HOST_MEMORY_RELEASE {json.dumps(report)}")

    def _release_consumed_rollout_batch_before_receive(self) -> None:
        """Return the completed update's replay before receiving its successor."""

        if getattr(self, "rollout_batch", None) is None:
            return
        self.rollout_batch = None
        self._release_replay_host_memory(phase="pre_trajectory_receive")

    def _consume_rollout_batch_during_train_preparation(self) -> bool:
        """Shuffle a one-way replay without retaining its time-major source."""

        return True

    def _after_rollout_batch_train_preparation(self) -> None:
        """Return pages from the consumed time-major source before training."""

        self._release_replay_host_memory(phase="post_train_preparation")

    def run_training(
        self,
        kv_request_channel: Channel | None = None,
        kv_response_channel: Channel | None = None,
    ) -> dict:
        """Release consumed replay after backward, before the next rollout."""

        metrics = super().run_training(kv_request_channel, kv_response_channel)
        # The base frame and its last microbatch views must be gone before trim.
        self.rollout_batch = None
        self._release_replay_host_memory(phase="post_training")
        return metrics

    def prepare_cpu_microbatch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Leave validated, already materialized UNCOND replay on the CPU."""

        return batch

    def _iter_training_microbatches(self, micro_batches: list):
        """Overlap two bounded replay transfers with the existing FSDP backward."""

        return cpu_to_device_prefetch(
            micro_batches, prepare=self.prepare_cpu_microbatch, device=self.device
        )

    def _prepare_train_global_batch_for_microbatches(
        self, train_global_batch: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Retain the cheap original critic inputs before compacting heavy replay."""

        critic = self._fastwam_policy_module()._require_critic()
        self._uncond_original_critic_inputs = (
            train_global_batch["forward_inputs"][critic.replay_feature_key],
            train_global_batch["prev_values"],
        )
        return compact_uncond_global_batch(
            train_global_batch, micro_batch_size=int(self.cfg.actor.micro_batch_size)
        )

    @torch.no_grad()
    def _original_critic_clip_ratio(self) -> float:
        """Evaluate the original unmasked MB4 diagnostic without Video/Action work."""

        features, previous = self._uncond_original_critic_inputs
        self._uncond_original_critic_inputs = None
        critic = self._fastwam_policy_module()._require_critic()
        clipped = torch.zeros((), device=self.device, dtype=torch.float32)
        count = 0
        micro_size = int(self.cfg.actor.micro_batch_size)
        for start in range(0, features.shape[0], micro_size):
            with self.amp_context:
                values = critic.value_from_features(
                    features[start : start + micro_size].to(self.device)
                )
            old = previous[start : start + micro_size].to(self.device).float()
            values = values.float().reshape_as(old)
            clipped += (
                (values - old).abs() > float(self.cfg.algorithm.critic_loss.value_clip)
            ).sum()
            count += old.numel()
        return float((clipped / count).item())

    def _execute_training_microbatches(
        self, train_micro_batches: list, *, metrics: dict, selected_loss_scales: dict
    ) -> None:
        """Finish all backwards before scoring the original critic geometry."""

        super()._execute_training_microbatches(
            train_micro_batches,
            metrics=metrics,
            selected_loss_scales=selected_loss_scales,
        )
        metrics.setdefault(_FULL_CLIP_RATIO, []).append(
            self._original_critic_clip_ratio()
        )

    def _finalize_train_metrics_before_reduction(
        self, metrics: dict[str, list[float]]
    ) -> None:
        """Keep Flow weighting and critic statistics, restore skipped zero rows."""

        finalize_uncond_compaction_metrics(metrics)
