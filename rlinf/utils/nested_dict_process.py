# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable

import numpy as np
import torch

# Keys that we have already warned about in concat_batch, so each missing key
# only produces a single warning per process (avoid log spam in the replay /
# demo batch pipeline).
_CONCAT_BATCH_WARNED_KEYS: set[str] = set()


def update_nested_cfg(base_cfg, override_cfg):
    for key, value in override_cfg.items():
        if (
            key in base_cfg
            and isinstance(base_cfg[key], dict)
            and isinstance(value, dict)
        ):
            update_nested_cfg(base_cfg[key], value)
        else:
            base_cfg[key] = value
    return base_cfg


def map_nested_tensors(value: Any, tensor_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Apply ``tensor_fn`` without dropping structured batch records.

    Frozen dataclasses such as FastWAM route and Gate records are rebuilt with
    their tensor fields transformed. Static schema fields remain unchanged.

    Args:
        value: Arbitrarily nested batch value.
        tensor_fn: Transformation applied to every tensor leaf.

    Returns:
        A value with the same nested structure and transformed tensor leaves.
    """

    if isinstance(value, torch.Tensor):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {key: map_nested_tensors(item, tensor_fn) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: map_nested_tensors(getattr(value, field.name), tensor_fn)
            for field in fields(value)
            if field.init
        }
        return replace(value, **updates)
    if isinstance(value, list):
        return [map_nested_tensors(item, tensor_fn) for item in value]
    if isinstance(value, tuple):
        return tuple(map_nested_tensors(item, tensor_fn) for item in value)
    return value


def merge_rollout_epoch_batch(value: Any, rollout_epoch: int):
    """Fold epoch-major ``[epoch*time, batch]`` fields into ``[time, epoch*batch]``."""

    if rollout_epoch < 1:
        raise ValueError("rollout_epoch must be positive.")

    def merge_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(
                "Rollout tensor fields must have leading [epoch*time, batch] "
                f"dimensions, got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] % rollout_epoch != 0:
            raise ValueError(
                f"Rollout tensor length {tensor.shape[0]} is not divisible by "
                f"rollout_epoch {rollout_epoch}."
            )
        merged = tensor.reshape(rollout_epoch, -1, *tensor.shape[1:])
        merged = merged.transpose(0, 1)
        return merged.reshape(merged.shape[0], -1, *merged.shape[3:]).contiguous()

    return map_nested_tensors(value, merge_tensor)


def flatten_time_batch(
    value: Any,
    shuffle_id: torch.Tensor,
    *,
    field_name: str = "batch",
):
    """Flatten structured ``[time, batch]`` fields and apply one shared shuffle."""

    if shuffle_id.ndim != 1:
        raise ValueError("shuffle_id must be one-dimensional.")

    def flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(
                f"Training field {field_name!r} must have leading [time, batch] "
                f"dimensions, got {tuple(tensor.shape)}."
            )
        flattened = tensor.reshape(-1, *tensor.shape[2:])
        if flattened.shape[0] != shuffle_id.numel():
            raise ValueError(
                f"Training field {field_name!r} flattened to {flattened.shape[0]} "
                f"items, expected {shuffle_id.numel()}."
            )
        return flattened[shuffle_id].contiguous()

    return map_nested_tensors(value, flatten_tensor)


def flatten_time_batch_consuming(
    value: Any,
    shuffle_id: torch.Tensor,
    *,
    field_name: str = "batch",
):
    """Flatten and shuffle while releasing mutable dictionary sources.

    Large stored-replay batches are dictionaries containing many independent
    tensor fields. A regular recursive mapping retains the complete source
    dictionary until every shuffled output has been built, temporarily keeping
    two full replay batches alive. This variant pops each dictionary field as
    soon as its shuffled replacement is materialized. Non-dictionary structured
    values keep the same behavior as :func:`flatten_time_batch`.

    Args:
        value: Mutable nested batch value that will not be reused.
        shuffle_id: Shared one-dimensional sample permutation.
        field_name: Name used in shape-validation errors.

    Returns:
        The flattened and shuffled value. Every mutable dictionary in ``value``
        is empty after a successful call.
    """

    if shuffle_id.ndim != 1:
        raise ValueError("shuffle_id must be one-dimensional.")

    def flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(
                f"Training field {field_name!r} must have leading [time, batch] "
                f"dimensions, got {tuple(tensor.shape)}."
            )
        flattened = tensor.reshape(-1, *tensor.shape[2:])
        if flattened.shape[0] != shuffle_id.numel():
            raise ValueError(
                f"Training field {field_name!r} flattened to {flattened.shape[0]} "
                f"items, expected {shuffle_id.numel()}."
            )
        return flattened[shuffle_id].contiguous()

    def transform(item: Any):
        if isinstance(item, torch.Tensor):
            return flatten_tensor(item)
        if isinstance(item, dict):
            transformed = {}
            for key in list(item):
                source = item.pop(key)
                transformed[key] = transform(source)
                del source
            return transformed
        if is_dataclass(item) and not isinstance(item, type):
            updates = {
                field.name: transform(getattr(item, field.name))
                for field in fields(item)
                if field.init
            }
            return replace(item, **updates)
        if isinstance(item, list):
            return [transform(nested) for nested in item]
        if isinstance(item, tuple):
            return tuple(transform(nested) for nested in item)
        return item

    return transform(value)


def copy_dict_tensor(next_extracted_obs: dict):
    """
    Recursively clones all torch tensors in a dict.
    """
    return map_nested_tensors(next_extracted_obs, torch.Tensor.clone)


def clone_nested_to_cpu(value: Any):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_nested_to_cpu(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: clone_nested_to_cpu(getattr(value, field.name))
                for field in fields(value)
                if field.init
            },
        )
    if isinstance(value, list):
        return [clone_nested_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested_to_cpu(item) for item in value)
    return value


def put_tensor_device(data_dict, device):
    """Move every tensor leaf, including dataclass fields, to ``device``."""

    def move_tensor(value: torch.Tensor) -> torch.Tensor:
        non_blocking = value.device.type == "cpu" and value.is_pinned()
        return value.to(device=device, non_blocking=non_blocking).contiguous()

    return map_nested_tensors(
        data_dict,
        move_tensor,
    )


def _chunk_batch_value(value: Any, chunks: int, dim: int) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return [chunk.contiguous() for chunk in torch.chunk(value, chunks, dim=dim)]
    if isinstance(value, list):
        assert dim == 0, f"List field only supports dim=0, got {dim}."
        return _split_list_by_sizes(value, chunks)
    if value is None:
        return [None for _ in range(chunks)]
    if isinstance(value, dict):
        return split_dict_to_chunk(value, chunks, dim)
    if is_dataclass(value) and not isinstance(value, type):
        chunk_method = getattr(value, "chunk", None)
        if not callable(chunk_method):
            raise TypeError(
                f"Dataclass batch field {type(value).__name__} must implement chunk()."
            )
        return list(chunk_method(chunks, dim=dim))
    raise TypeError(f"Batch field type {type(value)} is not supported.")


def _split_list_by_sizes(value: list, split_sizes: list[int] | int) -> list[list]:
    if isinstance(split_sizes, int):
        chunks = split_sizes
        k, m = divmod(len(value), chunks)
        split_sizes = [k + (1 if i < m else 0) for i in range(chunks)]
    out, i = [], 0
    for n in split_sizes:
        out.append(value[i : i + n])
        i += n
    return out


def split_dict_to_chunk(data: dict, split_size, dim=0):
    splited_list = [{} for _ in range(split_size)]
    for key, value in data.items():
        try:
            split_vs = _chunk_batch_value(value, split_size, dim)
        except TypeError as error:
            raise ValueError(f"{key=}, {type(value)} is not supported.") from error
        if len(split_vs) != split_size:
            raise ValueError(
                f"Field {key!r} produced {len(split_vs)} chunks, expected {split_size}."
            )
        for split_id in range(split_size):
            splited_list[split_id][key] = (
                split_vs[split_id].contiguous()
                if isinstance(split_vs[split_id], torch.Tensor)
                else split_vs[split_id]
            )
    return splited_list


def concat_batch(data1, data2):
    batch = {}
    for key, value in data1.items():
        if isinstance(value, torch.Tensor):
            if key not in data2:
                # NOTE: NO WARNING FOR THE CASE THAT DATA2 DOES NOT CONTAIN SOME KEYS IN DATA1
                continue
            batch[key] = torch.cat([data1[key], data2[key]], dim=0)
        elif isinstance(value, dict):
            # NOTE: added this for dealing with different keys in demo data.
            if key not in data2:
                if key not in _CONCAT_BATCH_WARNED_KEYS:
                    _CONCAT_BATCH_WARNED_KEYS.add(key)
                    # Lazy import to avoid pulling rlinf.scheduler.worker (and
                    # its heavy deps) at module import time. This only runs
                    # once per missing key, inside a worker where that import
                    # is essentially free.
                    from rlinf.utils.logging import get_logger

                    get_logger().warning(
                        "concat_batch: key '%s' not found in data2 (value type: %s), "
                        "skipping. This warning is only emitted once per key.",
                        key,
                        type(value).__name__,
                    )
                continue
            batch[key] = concat_batch(data1[key], data2[key])
        elif is_dataclass(value) and not isinstance(value, type):
            if key not in data2:
                continue
            cat_method = getattr(type(value), "cat", None)
            if not callable(cat_method):
                raise TypeError(
                    f"Dataclass batch field {type(value).__name__} must implement cat()."
                )
            batch[key] = cat_method((value, data2[key]), dim=0)
    return batch


def stack_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if isinstance(_v0, torch.Tensor):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = torch.stack(v_list, dim=dim)
        elif isinstance(_v0, dict):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = stack_list_of_dict_tensor(v_list, dim=dim)
        elif is_dataclass(_v0) and not isinstance(_v0, type):
            v_list = [d[key] for d in list_of_dict]
            stack_method = getattr(type(_v0), "stack", None)
            if not callable(stack_method):
                raise TypeError(
                    f"Dataclass batch field {type(_v0).__name__} must implement stack()."
                )
            ret[key] = stack_method(v_list, dim=dim)
        elif _v0 is None:
            pass
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")
    return ret


def stack_list_of_dict_tensor_consuming(list_of_dict: list, dim=0):
    """Stack nested tensor dictionaries while releasing source fields eagerly.

    This is intended for one-way trajectory handoff.  Each source dictionary is
    consumed as soon as its corresponding output field has been materialized,
    so large replay payloads do not retain a complete pre-stack copy until the
    whole nested dictionary has finished stacking.

    Args:
        list_of_dict: Mutable dictionaries that are no longer needed by the
            caller after this operation.
        dim: Dimension inserted by ``torch.stack``.

    Returns:
        The same stacked structure produced by ``stack_list_of_dict_tensor``.
    """

    if len(list_of_dict) == 0:
        return {}
    keys = list(list_of_dict[0].keys())

    ret = {}
    for key in keys:
        value = list_of_dict[0][key]
        values = [item.pop(key) for item in list_of_dict]
        if isinstance(value, torch.Tensor):
            ret[key] = torch.stack(values, dim=dim)
        elif isinstance(value, dict):
            ret[key] = stack_list_of_dict_tensor_consuming(values, dim=dim)
        elif is_dataclass(value) and not isinstance(value, type):
            stack_method = getattr(type(value), "stack", None)
            if not callable(stack_method):
                raise TypeError(
                    f"Dataclass batch field {type(value).__name__} must implement stack()."
                )
            ret[key] = stack_method(values, dim=dim)
        elif value is not None:
            raise ValueError(f"{key=}, {type(value)} is not supported!")
        del values
    return ret


def cat_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if _v0 is None:
            continue

        v_list = [d[key] for d in list_of_dict]

        if isinstance(_v0, torch.Tensor):
            ret[key] = torch.cat(v_list, dim=dim)
        elif isinstance(_v0, np.ndarray):
            ret[key] = np.concatenate([v for v in v_list if v is not None], axis=dim)
        elif isinstance(_v0, list):
            assert dim == 0, f"{key=} is list, dim !=0 is not supported!"
            ret[key] = [item for sub in v_list if sub is not None for item in sub]
        elif isinstance(_v0, dict):
            ret[key] = cat_list_of_dict_tensor(v_list, dim=dim)
        elif is_dataclass(_v0) and not isinstance(_v0, type):
            cat_method = getattr(type(_v0), "cat", None)
            if not callable(cat_method):
                raise TypeError(
                    f"Dataclass batch field {type(_v0).__name__} must implement cat()."
                )
            ret[key] = cat_method(v_list, dim=dim)
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")

    return ret


def split_dict(
    batch: dict[str, Any],
    split_sizes: list[int],
    dim: int = 0,
) -> list[dict[str, Any]]:
    """Split one batch dict into size-specified sub-batches.

    Tensor values are chunked on ``dim``; list values are sliced proportionally;
    nested dict values are split recursively.

    Args:
        batch: Dict.
        split_sizes: Batch sizes for each destination rank.
        dim: Tensor dimension to split. Defaults to 0.

    Returns:
        A list of splited batches, one item per destination rank.
    """
    count = len(split_sizes)
    total_size = sum(split_sizes)
    splitted_batches = [{} for _ in range(count)]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[dim] == total_size, (
                f"Tensor field '{key}' expected split dim size {total_size}, "
                f"got {value.shape[dim]} on dim {dim}."
            )
            splitted_values = torch.split(value, split_sizes, dim=dim)
            for i in range(count):
                splitted_batches[i][key] = splitted_values[i].contiguous()
        elif isinstance(value, list):
            assert dim == 0, f"List field '{key}' only supports dim=0, got {dim}."
            length = len(value)
            assert length == total_size, (
                f"List field '{key}' expected length {total_size}, got {length}."
            )
            begin = 0
            for i, size in enumerate(split_sizes):
                splitted_batches[i][key] = value[begin : begin + size]
                begin += size
        elif isinstance(value, dict):
            splitted_sub_batches = split_dict(value, split_sizes, dim=dim)
            for i in range(count):
                splitted_batches[i][key] = splitted_sub_batches[i]
        elif is_dataclass(value) and not isinstance(value, type):
            split_method = getattr(value, "split", None)
            if not callable(split_method):
                raise TypeError(
                    f"Dataclass batch field {type(value).__name__} must implement split()."
                )
            splitted_values = split_method(split_sizes, dim=dim)
            if len(splitted_values) != count:
                raise ValueError(
                    f"Field {key!r} produced {len(splitted_values)} splits, "
                    f"expected {count}."
                )
            for i in range(count):
                splitted_batches[i][key] = splitted_values[i]
        else:
            for i in range(count):
                splitted_batches[i][key] = value

    return splitted_batches
