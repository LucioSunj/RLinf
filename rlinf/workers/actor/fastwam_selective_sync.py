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

"""Selective FastWAM actor-to-rollout state materialization.

The adaptive policy keeps the pretrained FastWAM and pi0.5 parents frozen. A
rollout sync therefore needs only trainable Gate/LoRA/value-head parameters and
persistent buffers. Building a normal FSDP state dict first would clone the
entire frozen composite model before the patch syncer can filter it.

With ``use_orig_params=True``, classic FSDP preserves the original parameter
objects but exposes their local one-dimensional shards outside forward/backward
unshard windows. This module captures those objects before wrapping and
reconstructs only the selected parameters when they are actually sharded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch import nn

from rlinf.utils.utils import collect_param_names_need_sync


@dataclass(frozen=True)
class CapturedSyncTensor:
    """One pre-FSDP tensor selected for actor-to-rollout synchronization."""

    tensor: torch.Tensor
    original_shape: torch.Size
    is_parameter: bool

    @property
    def original_numel(self) -> int:
        """Return the full unsharded element count."""

        return self.original_shape.numel()


def capture_fastwam_sync_tensors(
    module: nn.Module,
) -> dict[str, CapturedSyncTensor]:
    """Capture exact selective-sync tensor references before FSDP wrapping.

    Args:
        module: Unwrapped adaptive FastWAM policy.

    Returns:
        An insertion-ordered mapping whose keys exactly match
        :func:`collect_param_names_need_sync`.

    Raises:
        RuntimeError: If the collector produces duplicate or unresolved names.
    """

    sync_names = collect_param_names_need_sync(module)
    if not sync_names:
        raise RuntimeError("FastWAM selective sync found no tensors to synchronize.")
    if len(sync_names) != len(set(sync_names)):
        raise RuntimeError("FastWAM selective sync names must be unique.")

    parameters = {
        name: parameter
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }
    buffers = dict(module.named_buffers(remove_duplicate=False))

    captured: dict[str, CapturedSyncTensor] = {}
    for name in sync_names:
        parameter = parameters.get(name)
        buffer = buffers.get(name)
        if parameter is not None and buffer is not None:
            raise RuntimeError(
                f"FastWAM selective sync name is both a parameter and buffer: {name}."
            )
        tensor = parameter if parameter is not None else buffer
        if tensor is None:
            raise RuntimeError(
                f"FastWAM selective sync could not resolve collected tensor: {name}."
            )
        captured[name] = CapturedSyncTensor(
            tensor=tensor,
            original_shape=torch.Size(tensor.shape),
            is_parameter=parameter is not None,
        )
    return captured


def prepare_fastwam_sync_tensors(
    module: nn.Module,
    *,
    device: torch.device | str,
) -> dict[str, CapturedSyncTensor]:
    """Move the full composite to one device before capturing sync tensors.

    FSDP does not move parameters passed through ignored_states. Moving the
    module here also ensures that persistent buffer references are captured
    after Module.to has replaced them.
    """

    expected_device = torch.device(device)
    module.to(device=expected_device)
    misplaced = sorted(
        name
        for name, tensor in (
            list(module.named_parameters(remove_duplicate=False))
            + list(module.named_buffers(remove_duplicate=False))
        )
        if tensor.device != expected_device
    )
    if misplaced:
        raise RuntimeError(
            "FastWAM tensors remain off the actor device before FSDP wrapping: "
            f"{misplaced[:8]}."
        )
    return capture_fastwam_sync_tensors(module)


def _all_gather_parameter_sizes(
    entries: Sequence[CapturedSyncTensor],
    *,
    process_group: Any = None,
) -> list[list[int]]:
    """Gather local FSDP shard sizes once for all selected parameters."""

    if not entries:
        return []
    world_size = dist.get_world_size(group=process_group)
    devices = {entry.tensor.device for entry in entries}
    if len(devices) != 1:
        raise RuntimeError(
            "FastWAM selected parameters must be on one device before sync; got "
            f"{sorted(str(device) for device in devices)}."
        )
    device = entries[0].tensor.device
    local_sizes = torch.tensor(
        [entry.tensor.numel() for entry in entries],
        dtype=torch.int64,
        device=device,
    )
    gathered_sizes = [torch.empty_like(local_sizes) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_sizes, group=process_group)
    by_rank = [sizes.cpu().tolist() for sizes in gathered_sizes]
    return [
        [int(by_rank[rank][index]) for rank in range(world_size)]
        for index in range(len(entries))
    ]


def _reconstruct_parameter(
    entry: CapturedSyncTensor,
    shard_sizes: Sequence[int],
    *,
    process_group: Any = None,
) -> torch.Tensor:
    """Return a detached full view or all-gather one selected FSDP parameter."""

    expected_numel = entry.original_numel
    local_tensor = entry.tensor.detach().reshape(-1)
    if all(size == expected_numel for size in shard_sizes):
        if local_tensor.numel() != expected_numel:
            raise RuntimeError(
                "FastWAM replicated parameter size disagrees with local tensor: "
                f"expected {expected_numel}, got {local_tensor.numel()}."
            )
        return local_tensor.reshape(entry.original_shape)

    if sum(shard_sizes) != expected_numel:
        raise RuntimeError(
            "FastWAM FSDP shard sizes do not reconstruct the original parameter: "
            f"expected {expected_numel}, got shards {list(shard_sizes)}."
        )
    rank = dist.get_rank(group=process_group)
    if local_tensor.numel() != shard_sizes[rank]:
        raise RuntimeError(
            "FastWAM local FSDP shard size disagrees with gathered metadata: "
            f"rank {rank}, local {local_tensor.numel()}, gathered {shard_sizes[rank]}."
        )

    max_shard_numel = max(shard_sizes, default=0)
    padded = torch.zeros(
        max_shard_numel,
        dtype=local_tensor.dtype,
        device=local_tensor.device,
    )
    padded[: local_tensor.numel()].copy_(local_tensor)
    gathered = [torch.empty_like(padded) for _ in shard_sizes]
    dist.all_gather(gathered, padded, group=process_group)
    full_tensor = torch.cat(
        [shard[:size] for shard, size in zip(gathered, shard_sizes, strict=True)]
    )
    if full_tensor.numel() != expected_numel:
        raise RuntimeError(
            "FastWAM selective all-gather produced the wrong element count: "
            f"expected {expected_numel}, got {full_tensor.numel()}."
        )
    return full_tensor.reshape(entry.original_shape)


@torch.no_grad()
def materialize_fastwam_sync_state(
    captured: dict[str, CapturedSyncTensor],
    expected_names: Sequence[str],
    *,
    process_group: Any = None,
) -> dict[str, torch.Tensor]:
    """Materialize only the exact FastWAM rollout-sync tensor set.

    Args:
        captured: References recorded by :func:`capture_fastwam_sync_tensors`.
        expected_names: The FSDP manager's authoritative sync-key ordering.
        process_group: Actor FSDP process group, or the default group when unset.

    Returns:
        Detached full tensors keyed exactly and in the order of
        ``expected_names``. Replicated tensors share storage; sharded parameters
        are selectively reconstructed with an all-gather.

    Raises:
        RuntimeError: If keys, shapes, devices, or shard metadata disagree.
    """

    expected_names = list(expected_names)
    if not expected_names:
        raise RuntimeError("FastWAM selective sync expected names must not be empty.")
    if expected_names != list(captured):
        raise RuntimeError(
            "FastWAM selective sync keys changed across FSDP wrapping: "
            f"expected {expected_names}, captured {list(captured)}."
        )

    parameter_names = [name for name in expected_names if captured[name].is_parameter]
    parameter_entries = [captured[name] for name in parameter_names]
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size(group=process_group) if distributed else 1
    if world_size > 1:
        parameter_sizes = _all_gather_parameter_sizes(
            parameter_entries, process_group=process_group
        )
    else:
        parameter_sizes = [[entry.tensor.numel()] for entry in parameter_entries]
    sizes_by_name = dict(zip(parameter_names, parameter_sizes, strict=True))

    state: dict[str, torch.Tensor] = {}
    for name in expected_names:
        entry = captured[name]
        if entry.is_parameter:
            state[name] = _reconstruct_parameter(
                entry,
                sizes_by_name[name],
                process_group=process_group,
            )
            continue
        tensor = entry.tensor.detach()
        if tensor.numel() != entry.original_numel:
            raise RuntimeError(
                "FastWAM persistent buffer shape changed across FSDP wrapping: "
                f"{name}, expected {tuple(entry.original_shape)}, "
                f"got {tuple(tensor.shape)}."
            )
        state[name] = tensor.reshape(entry.original_shape)
    return state
