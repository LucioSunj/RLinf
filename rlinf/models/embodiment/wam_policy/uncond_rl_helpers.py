# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Synchronous microbatch helpers with one authoritative Actor optimizer."""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

GRADIENT_BUCKET_BYTES = 128 * 1024 * 1024


@dataclass
class HelperMessage:
    """Keep heavy replay tensors in Channel's supported flat tensor field."""

    kind: str
    version: int
    optimizer_step: int
    index: int = -1
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    structure: Any = None


def pack_helper_message(
    kind: str,
    payload: Any,
    *,
    version: int,
    optimizer_step: int,
    index: int = -1,
) -> HelperMessage:
    """Flatten CPU tensor leaves without pickling a nested replay tensor bank."""

    tensors = {}

    def flatten(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Helper replay and metadata must be prepared on CPU.")
            key = str(len(tensors))
            tensors[key] = value.detach().contiguous()
            return ("tensor", key)
        if isinstance(value, dict):
            return ("dict", [(key, flatten(item)) for key, item in value.items()])
        if is_dataclass(value) and not isinstance(value, type):
            return (
                "dataclass",
                type(value),
                [
                    (entry.name, flatten(getattr(value, entry.name)))
                    for entry in fields(value)
                ],
            )
        if isinstance(value, (tuple, list)):
            return (
                "tuple" if isinstance(value, tuple) else "list",
                [flatten(item) for item in value],
            )
        return ("value", value)

    structure = flatten(payload)
    return HelperMessage(
        kind, int(version), int(optimizer_step), int(index), tensors, structure
    )


def unpack_helper_message(message: HelperMessage) -> Any:
    """Restore exactly the original replay containers and dataclass fields."""

    def restore(node):
        kind, *value = node
        if kind == "tensor":
            return message.tensors[value[0]]
        if kind == "dict":
            return {key: restore(item) for key, item in value[0]}
        if kind == "dataclass":
            cls, entries = value
            result = object.__new__(cls)
            for name, item in entries:
                object.__setattr__(result, name, restore(item))
            return result
        if kind in {"tuple", "list"}:
            result = [restore(item) for item in value[0]]
            return tuple(result) if kind == "tuple" else result
        if kind == "value":
            return value[0]
        raise ValueError(f"Unsupported helper packet node {kind!r}.")

    return restore(message.structure)


def map_prepared_tensors(value: Any, transform: Callable) -> Any:
    """Move an already validated replay without rerunning content validation."""

    return _map_prepared_tensor_paths(value, lambda tensor, path: transform(tensor))


def _map_prepared_tensor_paths(
    value: Any, transform: Callable, path: tuple = ()
) -> Any:
    if isinstance(value, torch.Tensor):
        return transform(value, path)
    if isinstance(value, dict):
        return {
            key: _map_prepared_tensor_paths(item, transform, (*path, key))
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        result = object.__new__(type(value))
        for entry in fields(value):
            object.__setattr__(
                result,
                entry.name,
                _map_prepared_tensor_paths(
                    getattr(value, entry.name), transform, (*path, entry.name)
                ),
            )
        return result
    if isinstance(value, (tuple, list)):
        result = [
            _map_prepared_tensor_paths(item, transform, (*path, index))
            for index, item in enumerate(value)
        ]
        return tuple(result) if isinstance(value, tuple) else result
    return value


class _ReplayTransferSlot:
    """One persistent pinned/device bank, safe to reuse after its backward."""

    def __init__(self, target, copy_stream):
        self.target = target
        self.copy_stream = copy_stream
        self.buffers = {}
        self.copy_done = torch.cuda.Event()
        self.compute_done = torch.cuda.Event()
        self.has_copy = False
        self.has_compute = False

    def submit(self, prepared):
        if self.has_copy:
            # The CPU may rewrite pinned storage only after its previous DMA.
            self.copy_done.synchronize()
        if self.has_compute:
            # The copy stream may overwrite GPU storage only after backward.
            self.copy_stream.wait_event(self.compute_done)

        def copy_tensor(tensor, path):
            if tensor.device.type != "cpu":
                raise ValueError("Microbatch preparation must retain CPU tensors.")
            key = (path, tensor.dtype)
            capacity = tensor.numel()
            bank = self.buffers.get(key)
            if bank is None or bank[0].numel() < capacity:
                if self.has_compute:
                    self.compute_done.synchronize()
                bank = (
                    torch.empty(capacity, dtype=tensor.dtype, pin_memory=True),
                    torch.empty(capacity, dtype=tensor.dtype, device=self.target),
                )
                self.buffers[key] = bank
            host, device = bank
            host_view = host[:capacity].view(tensor.shape)
            device_view = device[:capacity].view(tensor.shape)
            host_view.copy_(tensor)
            device_view.copy_(host_view, non_blocking=True)
            return device_view

        with (
            torch.profiler.record_function("uncond_rl/H2D"),
            torch.cuda.stream(self.copy_stream),
        ):
            moved = _map_prepared_tensor_paths(prepared, copy_tensor)
            self.copy_done.record(self.copy_stream)
        self.has_copy = True
        return moved

    def begin_compute(self):
        torch.cuda.current_stream(self.target).wait_event(self.copy_done)

    def end_compute(self):
        self.compute_done.record(torch.cuda.current_stream(self.target))
        self.has_compute = True

    def drain(self):
        if self.has_copy:
            self.copy_done.synchronize()
        if self.has_compute:
            self.compute_done.synchronize()


def cpu_to_device_prefetch(
    micro_batches: Iterable[dict],
    *,
    prepare: Callable[[dict], dict],
    device: torch.device | str,
) -> Iterator[tuple[int, dict]]:
    """Reuse exactly two transfer slots, draining each optimizer boundary.

    A yielded batch belongs to the caller until the iterator advances, after
    its backward has been enqueued. Pinned and device storage grows only when
    that tensor path exceeds the slot's previous maximum shape.
    """

    target = torch.device(device)
    source = enumerate(micro_batches)
    if target.type == "cpu":
        for index, batch in source:
            yield index, prepare(batch)
        return
    if target.type != "cuda":
        raise ValueError("UNCOND prefetch supports the CUDA training profile.")
    copy_stream = torch.cuda.Stream(device=target)
    slots = [_ReplayTransferSlot(target, copy_stream) for _ in range(2)]
    pending = deque()

    def submit(slot) -> bool:
        try:
            index, batch = next(source)
        except StopIteration:
            return False
        prepared = prepare(batch)
        moved = slot.submit(prepared)
        pending.append((index, moved, slot))
        return True

    active_slot = None
    try:
        for slot in slots:
            submit(slot)
        while pending:
            index, moved, slot = pending.popleft()
            slot.begin_compute()
            active_slot = slot
            yield index, moved
            slot.end_compute()
            active_slot = None
            del moved
            submit(slot)
    finally:
        if active_slot is not None:
            active_slot.end_compute()
        for slot in slots:
            slot.drain()
        pending.clear()


@dataclass(frozen=True)
class ParameterSegment:
    """One contiguous parameter slice in a bounded FP32 communication bucket."""

    parameter: int
    begin: int
    end: int


class TrainableParameterLayout:
    """Reuse unique live FP32 parameters; never construct another model copy."""

    def __init__(self, model: nn.Module, *, bucket_bytes: int = GRADIENT_BUCKET_BYTES):
        self.named_parameters = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters(remove_duplicate=True)
            if parameter.requires_grad
        )
        if not self.named_parameters:
            raise ValueError("Training helpers need trainable policy parameters.")
        if any(
            parameter.dtype != torch.float32 for _, parameter in self.named_parameters
        ):
            raise TypeError(
                "Helper trainable parameters must retain FP32 master storage."
            )
        self.parameters = tuple(parameter for _, parameter in self.named_parameters)
        # FSDP forwards the policy interface but prefixes root and nested
        # parameter names. Helpers use the same policy without those wrappers.
        self.manifest = tuple(
            (
                name.replace("_fsdp_wrapped_module.", ""),
                tuple(parameter.shape),
                str(parameter.dtype),
            )
            for name, parameter in self.named_parameters
        )
        capacity = int(bucket_bytes) // 4
        if capacity < 1:
            raise ValueError("A gradient bucket must hold at least one FP32 element.")
        buckets = []
        bucket = []
        available = capacity
        for index, parameter in enumerate(self.parameters):
            begin = 0
            while begin < parameter.numel():
                length = min(available, parameter.numel() - begin)
                bucket.append(ParameterSegment(index, begin, begin + length))
                begin += length
                available -= length
                if available == 0:
                    buckets.append(tuple(bucket))
                    bucket = []
                    available = capacity
        if bucket:
            buckets.append(tuple(bucket))
        self.buckets = tuple(buckets)

    def presence(self) -> tuple[bool, ...]:
        """Distinguish a real zero gradient from an absent gradient."""

        return tuple(parameter.grad is not None for parameter in self.parameters)

    def pack_bucket(
        self, bucket: tuple[ParameterSegment, ...], *, gradients: bool
    ) -> torch.Tensor:
        """Materialize one communication bucket instead of a full gradient clone."""

        reference = self.parameters[0]
        result = torch.empty(
            sum(part.end - part.begin for part in bucket),
            device=reference.device,
            dtype=torch.float32,
        )
        offset = 0
        for part in bucket:
            parameter = self.parameters[part.parameter]
            value = parameter.grad if gradients else parameter.detach()
            length = part.end - part.begin
            if value is None:
                result[offset : offset + length].zero_()
            else:
                result[offset : offset + length].copy_(
                    value.reshape(-1)[part.begin : part.end]
                )
            offset += length
        return result

    def add_gradient_bucket(
        self, bucket, values: torch.Tensor, presence: tuple[bool, ...]
    ) -> None:
        """SUM in source-rank order, preserving globally absent gradients."""

        offset = 0
        for part in bucket:
            length = part.end - part.begin
            if presence[part.parameter]:
                parameter = self.parameters[part.parameter]
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad.reshape(-1)[part.begin : part.end].add_(
                    values[offset : offset + length]
                )
            offset += length

    @torch.no_grad()
    def copy_weight_bucket(self, bucket, values: torch.Tensor) -> None:
        """Update existing Parameter objects without disturbing their aliases."""

        offset = 0
        for part in bucket:
            length = part.end - part.begin
            self.parameters[part.parameter].reshape(-1)[part.begin : part.end].copy_(
                values[offset : offset + length]
            )
            offset += length


def microbatch_assignments(
    count: int, helpers: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Distribute complete original microbatches with no duplicated padding."""

    size = 1 + len(helpers)
    return tuple(tuple(range(rank, count, size)) for rank in range(size))


def _cpu(value):
    tensors = []

    def collect(tensor):
        tensors.append(tensor.detach())
        return tensor

    map_prepared_tensors(value, collect)
    groups = {}
    for index, tensor in enumerate(tensors):
        if tensor.device.type != "cpu":
            groups.setdefault((tensor.device, tensor.dtype), []).append(index)
    for indices in groups.values():
        packed = torch.cat([tensors[index].reshape(-1) for index in indices]).cpu()
        offset = 0
        for index in indices:
            source = tensors[index]
            tensors[index] = packed[offset : offset + source.numel()].view(source.shape)
            offset += source.numel()
    leaves = iter(tensors)
    return map_prepared_tensors(value, lambda tensor: next(leaves))


def _same_runtime(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_same_runtime(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_same_runtime(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


class HelperLossContext:
    """Pure PPO configuration for optimizer-free replay on a rollout worker."""

    def __init__(self, cfg: Any, *, actor_version: int):
        self.cfg = cfg
        self.actor_version = int(actor_version)
        self.gradient_accumulation = int(cfg.actor.global_batch_size) // int(
            cfg.actor.micro_batch_size
        )


def backward_helper_microbatch(
    model, context, batch, *, selected_loss_scales, normalization_statistics
):
    """Accumulate the original Flow/value objective without a local optimizer."""

    from .uncond_rl import compute_uncond_rl_loss

    output = model(
        forward_inputs=batch["forward_inputs"],
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=context.cfg.algorithm.adv_type == "gae",
        use_cache=False,
    )
    loss, metrics = compute_uncond_rl_loss(
        cfg=context.cfg,
        actor_version=context.actor_version,
        micro_batch=batch,
        output_dict=output,
        selected_loss_scales=selected_loss_scales,
    )
    if normalization_statistics is not None:
        metrics["normalize/floor_hit_fraction"] = float(
            normalization_statistics["floor_hit_fraction"]
        )
    loss = loss / context.gradient_accumulation
    loss.backward()
    metrics["actor/total_loss"] = loss.detach()
    metrics["fastwam/fsdp_view_restore_handles"] = 0.0
    return metrics


def _communication_groups(worker, helpers):
    return [
        (str(worker.cfg.actor.group_name), [0]),
        (str(worker.cfg.rollout.group_name), list(helpers)),
    ]


def broadcast_helper_weights(worker, layout, helpers, *, owner: bool) -> None:
    """Broadcast every FP32 trainable after every Adam opportunity, including 15."""

    groups = _communication_groups(worker, helpers)
    for bucket in layout.buckets:
        payload = layout.pack_bucket(bucket, gradients=False) if owner else None
        received = worker.broadcast(
            object=payload,
            groups=groups,
            src=(str(worker.cfg.actor.group_name), 0),
            options=worker._sync_weight_comm_options,
        )
        if not owner:
            layout.copy_weight_bucket(bucket, received)
        del payload, received


class UncondTrainingOwnerMixin:
    """Keep the existing Actor's optimizer and checkpoint authority on GPU0."""

    def configure_training_helpers(
        self, *, helper_ranks, command_channel, result_channel
    ) -> None:
        """Connect bounded CPU queues without changing logical Actor world size."""

        if int(self._world_size) != 1:
            raise ValueError("UNCOND helpers require one logical Actor owner.")
        if self.grad_scaler.is_enabled() or bool(
            self.cfg.actor.fsdp_config.amp_autocast.enabled
        ):
            raise ValueError(
                "UNCOND helpers require the original unscaled FP32 gradients."
            )
        self._training_helper_ranks = tuple(int(rank) for rank in helper_ranks)
        if self._training_helper_ranks != tuple(
            sorted(set(self._training_helper_ranks))
        ):
            raise ValueError("UNCOND helper ranks must be unique and in rank order.")
        self._training_helper_commands = command_channel
        self._training_helper_results = result_channel
        self._training_helper_layout = TrainableParameterLayout(
            self._fastwam_policy_module()
        )

    def _execute_training_microbatches(
        self, train_micro_batches, *, metrics, selected_loss_scales
    ):
        helpers = getattr(self, "_training_helper_ranks", ())
        if not helpers:
            return super()._execute_training_microbatches(
                train_micro_batches,
                metrics=metrics,
                selected_loss_scales=selected_loss_scales,
            )
        from rlinf.utils.metric_utils import append_to_dict

        version = int(self.version)
        step = int(self.optimizer_steps)
        commands = self._training_helper_commands
        layout = self._training_helper_layout
        assignments = microbatch_assignments(len(train_micro_batches), helpers)
        payload = {
            "manifest": layout.manifest,
            "selected_loss_scales": selected_loss_scales,
            "normalization_statistics": getattr(
                self, "_fastwam_advantage_normalization_statistics", None
            ),
        }
        for helper in helpers:
            commands.put(
                pack_helper_message(
                    "begin", payload, version=version, optimizer_step=step
                ),
                key=helper,
            )
        sent = dict.fromkeys(helpers, 0)

        def send_next(position, helper):
            assigned = assignments[position]
            cursor = sent[helper]
            if cursor >= len(assigned):
                return
            index = assigned[cursor]
            prepared = self.prepare_cpu_microbatch(train_micro_batches[index])
            commands.put(
                pack_helper_message(
                    "microbatch",
                    prepared,
                    version=version,
                    optimizer_step=step,
                    index=index,
                ),
                key=helper,
            )
            sent[helper] += 1
            train_micro_batches[index] = None

        for position, helper in enumerate(helpers, 1):
            send_next(position, helper)
            send_next(position, helper)
        rows = {}
        local_batches = [train_micro_batches[index] for index in assignments[0]]
        for ordinal, batch in cpu_to_device_prefetch(
            local_batches, prepare=self.prepare_cpu_microbatch, device=self.device
        ):
            original_index = assignments[0][ordinal]
            local_metrics = {}
            self.train_micro_batch(
                micro_batch=batch,
                metrics=local_metrics,
                is_last=(ordinal + 1) == len(local_batches),
                selected_loss_scales=selected_loss_scales,
            )
            rows[original_index] = {
                name: values[0] for name, values in local_metrics.items()
            }
            train_micro_batches[original_index] = None
            local_batches[ordinal] = None
            del batch
            for position, helper in enumerate(helpers, 1):
                send_next(position, helper)
        for position, helper in enumerate(helpers, 1):
            if sent[helper] != len(assignments[position]):
                raise RuntimeError(
                    "An original optimizer microbatch was not dispatched."
                )
            commands.put(
                pack_helper_message(
                    "end_batch", None, version=version, optimizer_step=step
                ),
                key=helper,
            )
        for position, helper in enumerate(helpers, 1):
            message = self._training_helper_results.get(key=helper)
            if (message.kind, message.version, message.optimizer_step) != (
                "gradients",
                version,
                step,
            ):
                raise RuntimeError(
                    "Helper gradient result crossed an optimizer boundary."
                )
            result = unpack_helper_message(message)
            presence = tuple(result["presence"])
            if len(presence) != len(layout.parameters):
                raise ValueError(
                    "Helper gradient presence differs from the parameter manifest."
                )
            if tuple(index for index, _ in result["metrics"]) != assignments[position]:
                raise ValueError(
                    "Helper result omitted or duplicated an original microbatch."
                )
            for bucket in layout.buckets:
                if not any(presence[part.parameter] for part in bucket):
                    continue
                received = torch.empty(
                    sum(part.end - part.begin for part in bucket),
                    dtype=torch.float32,
                    device=layout.parameters[0].device,
                )
                self.recv_tensor(
                    received,
                    src_group_name=str(self.cfg.rollout.group_name),
                    src_rank=helper,
                    options=self._sync_weight_comm_options,
                )
                layout.add_gradient_bucket(bucket, received, presence)
                del received
            for index, values in result["metrics"]:
                rows[index] = values
        for index in range(len(train_micro_batches)):
            append_to_dict(metrics, rows[index])
        self._training_helper_pending_step = step

    def _after_training_optimizer_step(self) -> None:
        helpers = getattr(self, "_training_helper_ranks", ())
        if not helpers:
            return super()._after_training_optimizer_step()
        if int(self.optimizer_steps) != self._training_helper_pending_step + 1:
            raise RuntimeError("Helpers must observe exactly one owner optimizer step.")
        broadcast_helper_weights(
            self, self._training_helper_layout, helpers, owner=True
        )


def serve_actor_training(
    worker, *, helper_ranks, command_channel, result_channel, version: int
) -> dict[str, Any]:
    """Temporarily use one resident rollout policy as a gradient-only helper."""

    from rlinf.utils.utils import get_rng_state, set_rng_state

    model = worker.hf_model
    helpers = tuple(int(rank) for rank in helper_ranks)
    rank = int(worker._rank)
    if rank not in helpers or int(worker.version) != int(version):
        raise ValueError("Helper rank/version does not match this rollout boundary.")
    layout = TrainableParameterLayout(model)
    rng = get_rng_state()
    runtime = copy.deepcopy(_cpu(model.rollout_runtime_state_dict()))
    completed = 0
    previous_step = None
    model.train()
    try:
        while True:
            message = command_channel.get(key=rank)
            if message.kind == "finish_update":
                if message.version != version or completed != int(
                    unpack_helper_message(message)["optimizer_opportunities"]
                ):
                    raise RuntimeError(
                        "Helper update ended before all optimizer opportunities."
                    )
                break
            if message.kind != "begin" or message.version != version:
                raise RuntimeError("Helper expected an optimizer-batch begin command.")
            step = message.optimizer_step
            if previous_step is not None and step != previous_step + 1:
                raise RuntimeError("Helper optimizer opportunities are not contiguous.")
            begin = unpack_helper_message(message)
            if begin["manifest"] != layout.manifest:
                raise ValueError("Helper and owner FP32 parameter manifests differ.")
            context = HelperLossContext(worker.cfg, actor_version=version)
            model.zero_grad(set_to_none=True)
            indices = []

            def receive_batches():
                while True:
                    item = command_channel.get(key=rank)
                    if (item.version, item.optimizer_step) != (version, step):
                        raise RuntimeError(
                            "Replay microbatch crossed an optimizer boundary."
                        )
                    if item.kind == "end_batch":
                        return
                    if item.kind != "microbatch":
                        raise RuntimeError(
                            "Helper expected an original microbatch or batch end."
                        )
                    indices.append(item.index)
                    yield unpack_helper_message(item)

            micro_metrics = []
            for index, batch in cpu_to_device_prefetch(
                receive_batches(), prepare=lambda batch: batch, device=worker.device
            ):
                values = backward_helper_microbatch(
                    model,
                    context,
                    batch,
                    selected_loss_scales=begin["selected_loss_scales"],
                    normalization_statistics=begin["normalization_statistics"],
                )
                micro_metrics.append((indices[index], values))
                del batch
            presence = layout.presence()
            result = _cpu(
                {
                    "presence": presence,
                    "metrics": micro_metrics,
                }
            )
            result_channel.put(
                pack_helper_message(
                    "gradients", result, version=version, optimizer_step=step
                ),
                key=rank,
            )
            for bucket in layout.buckets:
                if any(presence[part.parameter] for part in bucket):
                    gradient = layout.pack_bucket(bucket, gradients=True)
                    worker.send_tensor(
                        gradient,
                        dst_group_name=str(worker.cfg.actor.group_name),
                        dst_rank=0,
                        options=worker._sync_weight_comm_options,
                    )
                    del gradient
            broadcast_helper_weights(worker, layout, helpers, owner=False)
            model.zero_grad(set_to_none=True)
            previous_step = step
            completed += 1
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()
        set_rng_state(rng)
    if not _same_runtime(runtime, _cpu(model.rollout_runtime_state_dict())):
        raise RuntimeError("Training helper changed rollout-owned episode/chunk state.")
    return {
        "rank": rank,
        "version": version,
        "optimizer_opportunities": completed,
        "status": "PASS",
    }
