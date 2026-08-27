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

"""Rollout-resident hot/cold storage for exact Gate K/V replay.

The trajectory carries only a globally unique integer reference in
``GateKVMetadata.payload_reference_ids``.  Tensor payloads stay in the rollout
worker until the actor asks for an eligible microbatch.
"""

from __future__ import annotations

import mmap
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, Mapping

import psutil
import torch

from .kv_replay import GateKVReplayConfig

GATE_KV_PREFIX = "gate_kv"
GATE_KV_LAYER_INDICES = f"{GATE_KV_PREFIX}_layer_indices"
GATE_KV_BATCH_FIELDS = (
    "denoise_timesteps",
    "current_modes",
    "actor_versions",
    "video_key",
    "video_value",
    "video_mask",
    "action_key",
    "action_value",
    "action_mask",
    "context_key",
    "context_value",
    "context_mask",
)
GATE_KV_FORWARD_KEYS = tuple(
    f"{GATE_KV_PREFIX}_{name}" for name in GATE_KV_BATCH_FIELDS
)
GATE_KV_BATCH_INDICES = f"{GATE_KV_PREFIX}_batch_indices"
GATE_KV_RESPONSE_HANDLES = "gate_kv_response_handles"

_SOURCE_SHIFT = 56
_GENERATION_SHIFT = 32
_SOURCE_MASK = (1 << 7) - 1
_GENERATION_MASK = (1 << 24) - 1
_LOCAL_MASK = (1 << 32) - 1


def encode_gate_kv_handle(*, source_rank: int, generation: int, local_id: int) -> int:
    """Pack store ownership into a positive signed int64 reference."""

    if not 0 <= source_rank <= _SOURCE_MASK:
        raise ValueError("Gate K/V source rank is outside the 7-bit handle range.")
    if not 0 <= generation <= _GENERATION_MASK:
        raise ValueError("Gate K/V generation is outside the 24-bit handle range.")
    if not 0 <= local_id <= _LOCAL_MASK:
        raise ValueError("Gate K/V local id is outside the 32-bit handle range.")
    return (
        (int(source_rank) << _SOURCE_SHIFT)
        | (int(generation) << _GENERATION_SHIFT)
        | int(local_id)
    )


def decode_gate_kv_handle(handle: int) -> tuple[int, int, int]:
    """Return ``(source_rank, generation, local_id)`` from a reference."""

    handle = int(handle)
    if handle < 0:
        raise ValueError("Gate K/V handles must be non-negative.")
    return (
        (handle >> _SOURCE_SHIFT) & _SOURCE_MASK,
        (handle >> _GENERATION_SHIFT) & _GENERATION_MASK,
        handle & _LOCAL_MASK,
    )


def gate_kv_request_key(source_rank: int) -> str:
    """Return the request-channel key owned by one rollout rank."""

    return f"gate_kv_request_rollout_{int(source_rank)}"


def gate_kv_response_key(*, actor_rank: int, request_id: int) -> str:
    """Return the unique response-channel key for one actor request."""

    return f"gate_kv_response_actor_{int(actor_rank)}_{int(request_id)}"


@dataclass(frozen=True, slots=True)
class GateKVStoreRequest:
    """Small control message sent from an actor to one rollout store."""

    command: Literal["retain", "fetch", "release", "stop"]
    actor_rank: int
    request_id: int
    handles: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.command not in {"retain", "fetch", "release", "stop"}:
            raise ValueError(f"Unsupported Gate K/V request {self.command!r}.")
        if self.actor_rank < 0 or self.request_id < 0:
            raise ValueError("Gate K/V request ids must be non-negative.")
        if len(set(self.handles)) != len(self.handles):
            raise ValueError("One Gate K/V request must not repeat a handle.")


@dataclass(slots=True)
class _StoredEntry:
    payload: dict[str, torch.Tensor] | None
    byte_count: int
    tier: Literal["gpu", "cpu", "nvme"]
    nvme_fields: dict[str, "_NVMeTensorSpec"] | None = None


@dataclass(frozen=True, slots=True)
class _NVMeTensorSpec:
    offset: int
    byte_count: int
    shape: tuple[int, ...]
    dtype: torch.dtype


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _payload_bytes(payload: Mapping[str, torch.Tensor]) -> int:
    return sum(_tensor_bytes(tensor) for tensor in payload.values())


def _pinned_cpu_copy(
    tensor: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Make one contiguous CPU copy, pinned when CUDA provides an allocator."""

    tensor = tensor.detach().contiguous()
    try:
        target = torch.empty_like(tensor, device="cpu", pin_memory=True)
    except RuntimeError:
        return tensor.cpu().contiguous()
    if tensor.device.type == "cuda":
        if stream is None:
            target.copy_(tensor, non_blocking=False)
        else:
            with torch.cuda.stream(stream):
                target.copy_(tensor, non_blocking=True)
    else:
        target.copy_(tensor, non_blocking=False)
    return target


class TieredGateKVStore:
    """Per-rollout-rank K/V owner with bounded GPU and pinned-CPU tiers."""

    def __init__(
        self,
        *,
        source_rank: int,
        device: torch.device | str,
        config: GateKVReplayConfig,
        expected_global_samples: int | None = None,
        expected_local_samples: int | None = None,
    ) -> None:
        if config.transport != "host_staging":
            raise ValueError(
                "This runtime requires calibrated `host_staging`; CUDA-direct "
                "transport must not be selected without a passing topology probe."
            )
        self.source_rank = int(source_rank)
        self.device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        self.config = config
        self.expected_global_samples = expected_global_samples
        self.expected_local_samples = expected_local_samples
        if config.gate_kv_sample_budget is not None:
            for name, value in (
                ("expected_global_samples", expected_global_samples),
                ("expected_local_samples", expected_local_samples),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`{name}` must be a positive integer when Gate K/V "
                        "sampling is enabled."
                    )
            assert expected_global_samples is not None
            assert expected_local_samples is not None
            if expected_global_samples % expected_local_samples != 0:
                raise ValueError(
                    "Global Gate K/V candidates must divide evenly across rollout ranks."
                )
            source_world_size = expected_global_samples // expected_local_samples
            if not 0 <= self.source_rank < source_world_size:
                raise ValueError("Gate K/V source rank is outside the sampling world.")
        self.generation = 0
        self._next_local_id = 0
        self._entries: dict[int, _StoredEntry] = {}
        self._gpu_bytes = 0
        self._cpu_bytes = 0
        self._nvme_bytes = 0
        self._peak_gpu_bytes = 0
        self._peak_cpu_bytes = 0
        self._peak_nvme_bytes = 0
        self._emitted_count = 0
        self._emitted_bytes = 0
        self._sampled_count = 0
        self._sampled_bytes = 0
        self._unsampled_count = 0
        self._unsampled_bytes = 0
        self._selected_local_positions: set[int] | None = None
        self._eligible_count = 0
        self._eligible_bytes = 0
        self._discarded_count = 0
        self._discarded_bytes = 0
        self._fetched_count = 0
        self._gpu_hit_count = 0
        self._transfer_bytes = 0
        self._transfer_seconds = 0.0
        self._d2h_bytes = 0
        self._d2h_seconds = 0.0
        self._nvme_read_bytes = 0
        self._nvme_read_seconds = 0.0
        self._nvme_write_bytes = 0
        self._nvme_write_seconds = 0.0
        self._min_mig_free_bytes: int | None = None
        self._peak_mig_used_bytes = 0
        self._min_node_available_bytes: int | None = None
        self._nvme_directory = (
            Path(config.nvme_path).expanduser().resolve()
            if config.nvme_path is not None
            else None
        )
        self._nvme_file: BinaryIO | None = None
        self._nvme_file_path: Path | None = None
        self._nvme_map: mmap.mmap | None = None
        self._copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def begin_generation(self, generation: int) -> None:
        """Start a runner update after the previous update was fully released."""

        generation = int(generation)
        if self._entries:
            raise RuntimeError(
                "Gate K/V store still owns payloads at the next runner update: "
                f"rank={self.source_rank}, entries={len(self._entries)}."
            )
        if generation < 0 or generation > _GENERATION_MASK:
            raise ValueError("Gate K/V generation is outside the handle range.")
        self.generation = generation
        self._next_local_id = 0
        self._reset_interval_metrics()
        self._prepare_sample_plan()

    def _prepare_sample_plan(self) -> None:
        budget = self.config.gate_kv_sample_budget
        if budget is None:
            self._selected_local_positions = None
            return
        assert self.expected_global_samples is not None
        assert self.expected_local_samples is not None
        selected_count = min(int(budget), int(self.expected_global_samples))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.gate_kv_sample_seed) + self.generation)
        selected = torch.randperm(
            int(self.expected_global_samples),
            generator=generator,
        )[:selected_count]
        local_start = self.source_rank * int(self.expected_local_samples)
        local_stop = local_start + int(self.expected_local_samples)
        local = selected[(selected >= local_start) & (selected < local_stop)]
        self._selected_local_positions = {
            int(value) - local_start for value in local.tolist()
        }

    def _reset_interval_metrics(self) -> None:
        self._peak_gpu_bytes = self._gpu_bytes
        self._peak_cpu_bytes = self._cpu_bytes
        self._peak_nvme_bytes = self._nvme_bytes
        self._emitted_count = 0
        self._emitted_bytes = 0
        self._sampled_count = 0
        self._sampled_bytes = 0
        self._unsampled_count = 0
        self._unsampled_bytes = 0
        self._eligible_count = 0
        self._eligible_bytes = 0
        self._discarded_count = 0
        self._discarded_bytes = 0
        self._fetched_count = 0
        self._gpu_hit_count = 0
        self._transfer_bytes = 0
        self._transfer_seconds = 0.0
        self._d2h_bytes = 0
        self._d2h_seconds = 0.0
        self._nvme_read_bytes = 0
        self._nvme_read_seconds = 0.0
        self._nvme_write_bytes = 0
        self._nvme_write_seconds = 0.0
        self._min_mig_free_bytes = None
        self._peak_mig_used_bytes = 0
        self._min_node_available_bytes = None
        self._observe_resources()

    def _observe_resources(self) -> None:
        available = int(psutil.virtual_memory().available)
        if self._min_node_available_bytes is None:
            self._min_node_available_bytes = available
        else:
            self._min_node_available_bytes = min(
                self._min_node_available_bytes,
                available,
            )
        if self.device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            free_bytes = int(free_bytes)
            total_bytes = int(total_bytes)
            if self._min_mig_free_bytes is None:
                self._min_mig_free_bytes = free_bytes
            else:
                self._min_mig_free_bytes = min(
                    self._min_mig_free_bytes,
                    free_bytes,
                )
            self._peak_mig_used_bytes = max(
                self._peak_mig_used_bytes,
                total_bytes - free_bytes,
            )

    def _can_store_hot(self, byte_count: int) -> bool:
        if self.device.type != "cuda":
            return False
        if self._gpu_bytes + byte_count > int(
            self.config.hot_capacity_bytes_per_rollout_rank
        ):
            return False
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        self._observe_resources()
        return int(free_bytes) - byte_count >= int(self.config.hot_min_free_bytes)

    def _ensure_nvme_file(self) -> BinaryIO:
        if self._nvme_directory is None:
            raise RuntimeError("The Gate K/V NVMe tier has no configured path.")
        if self._nvme_file is None:
            self._nvme_directory.mkdir(parents=True, exist_ok=True)
            file = tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=(f"gate_kv_rank{self.source_rank}_generation{self.generation}_"),
                suffix=".bin",
                dir=self._nvme_directory,
                delete=False,
            )
            self._nvme_file = file
            self._nvme_file_path = Path(file.name)
        return self._nvme_file

    def _write_nvme_payload(
        self,
        packed: Mapping[str, torch.Tensor],
        sample_index: int,
    ) -> dict[str, _NVMeTensorSpec]:
        file = self._ensure_nvme_file()
        payload = self._make_payload(packed, sample_index, hot=False)
        fields: dict[str, _NVMeTensorSpec] = {}
        start = time.perf_counter()
        for key in GATE_KV_FORWARD_KEYS:
            tensor = payload[key].detach().cpu().contiguous()
            raw = tensor.view(torch.uint8).reshape(-1)
            offset = file.tell()
            byte_count = _tensor_bytes(tensor)
            written = file.write(raw.numpy().tobytes())
            if written != byte_count:
                raise OSError("Gate K/V NVMe shard write was incomplete.")
            fields[key] = _NVMeTensorSpec(
                offset=offset,
                byte_count=byte_count,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
            )
        self._nvme_write_seconds += time.perf_counter() - start
        self._nvme_write_bytes += sum(item.byte_count for item in fields.values())
        return fields

    def _ensure_nvme_map(self) -> mmap.mmap:
        if self._nvme_map is None:
            if self._nvme_file is None:
                raise RuntimeError("The Gate K/V NVMe shard is not open.")
            self._nvme_file.flush()
            self._nvme_map = mmap.mmap(
                self._nvme_file.fileno(),
                length=0,
                access=mmap.ACCESS_COPY,
            )
        return self._nvme_map

    def _read_nvme_payload(
        self,
        fields: Mapping[str, _NVMeTensorSpec],
    ) -> dict[str, torch.Tensor]:
        mapping = self._ensure_nvme_map()
        payload = {}
        start = time.perf_counter()
        for key in GATE_KV_FORWARD_KEYS:
            spec = fields[key]
            raw = torch.frombuffer(
                mapping,
                dtype=torch.uint8,
                count=spec.byte_count,
                offset=spec.offset,
            )
            tensor = raw.view(spec.dtype).reshape(spec.shape)
            payload[key] = _pinned_cpu_copy(tensor)
        elapsed = time.perf_counter() - start
        byte_count = sum(item.byte_count for item in fields.values())
        self._nvme_read_seconds += elapsed
        self._nvme_read_bytes += byte_count
        return payload

    def _close_nvme_shard(self) -> None:
        if self._nvme_map is not None:
            self._nvme_map.close()
            self._nvme_map = None
        if self._nvme_file is not None:
            self._nvme_file.close()
            self._nvme_file = None
        if self._nvme_file_path is not None:
            self._nvme_file_path.unlink(missing_ok=True)
            self._nvme_file_path = None

    def _make_payload(
        self,
        packed: Mapping[str, torch.Tensor],
        sample_index: int,
        *,
        hot: bool,
    ) -> dict[str, torch.Tensor]:
        sample = {
            key: packed[key][sample_index : sample_index + 1].detach().contiguous()
            for key in GATE_KV_FORWARD_KEYS
        }
        if hot:
            return {
                key: tensor.to(device=self.device, non_blocking=False).clone()
                for key, tensor in sample.items()
            }
        copied = {
            key: _pinned_cpu_copy(tensor, stream=self._copy_stream)
            for key, tensor in sample.items()
        }
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        return copied

    def register_forward_inputs(
        self,
        forward_inputs: Mapping[str, Any],
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Move packed K/V into this store and return compact forward inputs."""

        missing = [key for key in GATE_KV_FORWARD_KEYS if key not in forward_inputs]
        if missing:
            raise KeyError(
                f"Stored Gate K/V registration is missing fields: {missing}."
            )
        packed = {key: forward_inputs[key] for key in GATE_KV_FORWARD_KEYS}
        batch_sizes = {int(tensor.shape[0]) for tensor in packed.values()}
        if len(batch_sizes) != 1:
            raise ValueError("Packed Gate K/V fields disagree on batch size.")
        batch_size = batch_sizes.pop()
        handles: list[int] = []
        for sample_index in range(batch_size):
            local_id = self._next_local_id
            self._next_local_id += 1
            byte_count = sum(
                _tensor_bytes(tensor[sample_index : sample_index + 1])
                for tensor in packed.values()
            )
            if (
                self.config.max_bytes_per_sample is not None
                and byte_count > self.config.max_bytes_per_sample
            ):
                raise MemoryError(
                    "Stored Gate K/V exceeds max_bytes_per_sample during "
                    f"registration: {byte_count} > "
                    f"{self.config.max_bytes_per_sample}."
                )
            self._emitted_count += 1
            self._emitted_bytes += byte_count
            selected = (
                self._selected_local_positions is None
                or local_id in self._selected_local_positions
            )
            if not selected:
                self._unsampled_count += 1
                self._unsampled_bytes += byte_count
                handles.append(-1)
                self._observe_resources()
                continue
            hot = self._can_store_hot(byte_count)
            cold = not hot and self._cpu_bytes + byte_count <= int(
                self.config.cold_capacity_bytes_per_rollout_rank
            )
            nvme = not hot and not cold
            if nvme and self._nvme_bytes + byte_count > int(
                self.config.nvme_capacity_bytes_per_rollout_rank
            ):
                raise MemoryError(
                    "Rollout Gate K/V bounded tiers were exhausted without "
                    "dropping a sample: "
                    f"rank={self.source_rank}, requested={byte_count}, "
                    f"cpu_used={self._cpu_bytes}, cpu_capacity="
                    f"{self.config.cold_capacity_bytes_per_rollout_rank}, "
                    f"nvme_used={self._nvme_bytes}, nvme_capacity="
                    f"{self.config.nvme_capacity_bytes_per_rollout_rank}."
                )
            handle = encode_gate_kv_handle(
                source_rank=self.source_rank,
                generation=self.generation,
                local_id=local_id,
            )
            payload = None
            nvme_fields = None
            if nvme:
                nvme_fields = self._write_nvme_payload(packed, sample_index)
                actual_bytes = sum(item.byte_count for item in nvme_fields.values())
            else:
                payload = self._make_payload(packed, sample_index, hot=hot)
                actual_bytes = _payload_bytes(payload)
            if actual_bytes != byte_count:
                raise RuntimeError("Gate K/V byte count changed during tier placement.")
            if hot:
                tier = "gpu"
            elif nvme:
                tier = "nvme"
            else:
                tier = "cpu"
            self._entries[handle] = _StoredEntry(
                payload,
                byte_count,
                tier,
                nvme_fields,
            )
            if hot:
                self._gpu_bytes += byte_count
            elif cold:
                self._cpu_bytes += byte_count
            else:
                self._nvme_bytes += byte_count
            self._peak_gpu_bytes = max(self._peak_gpu_bytes, self._gpu_bytes)
            self._peak_cpu_bytes = max(self._peak_cpu_bytes, self._cpu_bytes)
            self._peak_nvme_bytes = max(self._peak_nvme_bytes, self._nvme_bytes)
            self._sampled_count += 1
            self._sampled_bytes += byte_count
            handles.append(handle)
            self._observe_resources()

        compact = dict(forward_inputs)
        for key in GATE_KV_FORWARD_KEYS:
            compact.pop(key, None)
        compact.pop(GATE_KV_LAYER_INDICES, None)
        return compact, torch.tensor(handles, dtype=torch.long)

    def _validate_handles(self, handles: tuple[int, ...]) -> None:
        missing = [handle for handle in handles if handle not in self._entries]
        if missing:
            raise KeyError(
                f"Gate K/V store rank {self.source_rank} does not own handles "
                f"{missing[:8]}."
            )
        wrong_generation = [
            handle
            for handle in handles
            if decode_gate_kv_handle(handle)[:2] != (self.source_rank, self.generation)
        ]
        if wrong_generation:
            raise ValueError("Gate K/V request crossed rank or generation ownership.")

    def retain(self, handles: tuple[int, ...]) -> None:
        """Discard every ineligible payload after delayed-route audits pass."""

        if (
            self.expected_local_samples is not None
            and self._emitted_count != self.expected_local_samples
        ):
            raise RuntimeError(
                "Gate K/V store observed a different local candidate count: "
                f"{self._emitted_count} != {self.expected_local_samples}."
            )
        if self._sampled_count != len(self._entries):
            raise RuntimeError(
                "Gate K/V sampled count disagrees with resident entries."
            )
        self._validate_handles(handles)
        keep = set(handles)
        discard = tuple(handle for handle in self._entries if handle not in keep)
        self._eligible_count = len(handles)
        self._eligible_bytes = sum(
            self._entries[handle].byte_count for handle in handles
        )
        self._discarded_count = len(discard)
        self._discarded_bytes = sum(
            self._entries[handle].byte_count for handle in discard
        )
        self.release(discard)

    def fetch(self, handles: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Materialize an ordered handle set in bounded pinned host memory."""

        self._validate_handles(handles)
        start = time.perf_counter()
        payloads: dict[str, list[torch.Tensor]] = {
            key: [] for key in GATE_KV_FORWARD_KEYS
        }
        d2h_start = time.perf_counter()
        hot_seen = False
        for handle in handles:
            entry = self._entries[handle]
            hot_seen = hot_seen or entry.tier == "gpu"
            if entry.tier == "gpu":
                self._gpu_hit_count += 1
                self._d2h_bytes += entry.byte_count
            if entry.tier == "nvme":
                if entry.nvme_fields is None:
                    raise RuntimeError("NVMe Gate K/V entry has no field descriptors.")
                entry_payload = self._read_nvme_payload(entry.nvme_fields)
            else:
                if entry.payload is None:
                    raise RuntimeError("Resident Gate K/V entry has no tensor payload.")
                entry_payload = entry.payload
            for key in GATE_KV_FORWARD_KEYS:
                tensor = entry_payload[key]
                if entry.tier != "nvme":
                    tensor = _pinned_cpu_copy(tensor, stream=self._copy_stream)
                payloads[key].append(tensor)
            self._fetched_count += 1
            self._transfer_bytes += entry.byte_count
        if hot_seen and self._copy_stream is not None:
            self._copy_stream.synchronize()
            self._d2h_seconds += time.perf_counter() - d2h_start
        response = {
            key: torch.cat(tensors, dim=0) if tensors else torch.empty(0)
            for key, tensors in payloads.items()
        }
        response[GATE_KV_RESPONSE_HANDLES] = torch.tensor(handles, dtype=torch.long)
        self._transfer_seconds += time.perf_counter() - start
        self._observe_resources()
        return response

    def release(self, handles: tuple[int, ...]) -> None:
        """Release exact payload ownership after its last actor consumption."""

        for handle in handles:
            entry = self._entries.pop(handle, None)
            if entry is None:
                continue
            if entry.tier == "gpu":
                self._gpu_bytes -= entry.byte_count
            elif entry.tier == "cpu":
                self._cpu_bytes -= entry.byte_count
            else:
                self._nvme_bytes -= entry.byte_count
        if not self._entries:
            self._close_nvme_shard()
        self._observe_resources()

    def metrics(self) -> dict[str, float]:
        """Return interval metrics suitable for TensorBoard and W&B."""

        process = psutil.Process(os.getpid())
        memory = process.memory_full_info()
        free_gpu = peak_gpu = 0
        if self.device.type == "cuda":
            free_gpu, _ = torch.cuda.mem_get_info(self.device)
            peak_gpu = torch.cuda.max_memory_allocated(self.device)
        return {
            "gpu_bytes": float(self._gpu_bytes),
            "cpu_bytes": float(self._cpu_bytes),
            "nvme_bytes": float(self._nvme_bytes),
            "peak_gpu_bytes": float(self._peak_gpu_bytes),
            "peak_cpu_bytes": float(self._peak_cpu_bytes),
            "peak_nvme_bytes": float(self._peak_nvme_bytes),
            "emitted_samples": float(self._emitted_count),
            "emitted_bytes": float(self._emitted_bytes),
            "sampled_samples": float(self._sampled_count),
            "sampled_bytes": float(self._sampled_bytes),
            "unsampled_samples": float(self._unsampled_count),
            "unsampled_bytes": float(self._unsampled_bytes),
            "actual_sample_rate": (
                float(self._sampled_count / self._emitted_count)
                if self._emitted_count
                else 0.0
            ),
            "eligible_samples": float(self._eligible_count),
            "eligible_bytes": float(self._eligible_bytes),
            "discarded_samples": float(self._discarded_count),
            "discarded_bytes": float(self._discarded_bytes),
            "gpu_hit_fraction": (
                float(self._gpu_hit_count / self._fetched_count)
                if self._fetched_count
                else 0.0
            ),
            "fetched_samples": float(self._fetched_count),
            "gpu_hit_samples": float(self._gpu_hit_count),
            "transfer_bytes": float(self._transfer_bytes),
            "transfer_seconds": float(self._transfer_seconds),
            "d2h_bytes": float(self._d2h_bytes),
            "d2h_seconds": float(self._d2h_seconds),
            "nvme_read_bytes": float(self._nvme_read_bytes),
            "nvme_read_seconds": float(self._nvme_read_seconds),
            "nvme_write_bytes": float(self._nvme_write_bytes),
            "nvme_write_seconds": float(self._nvme_write_seconds),
            "mig_free_bytes": float(free_gpu),
            "mig_min_free_bytes": float(self._min_mig_free_bytes or free_gpu),
            "mig_peak_used_bytes": float(self._peak_mig_used_bytes),
            "mig_peak_allocated_bytes": float(peak_gpu),
            "node_available_bytes": float(psutil.virtual_memory().available),
            "node_min_available_bytes": float(
                self._min_node_available_bytes or psutil.virtual_memory().available
            ),
            "rollout_rss_bytes": float(memory.rss),
            "rollout_uss_bytes": float(getattr(memory, "uss", memory.rss)),
        }
