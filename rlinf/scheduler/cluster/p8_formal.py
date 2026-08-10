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

"""Fail-closed local-Ray and placement evidence for P8 formal Stage2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import ray
from omegaconf import OmegaConf

P8_FORMAL_RAY_AUDIT_SENTINEL = "FASTWAM_P8_FORMAL_RAY_AUDIT"
P8_FORMAL_RAY_AUDIT_SCHEMA = "fastwam-p8-formal-ray-audit-v1"
P8_FORMAL_WORKER_PLACEMENT_AUDIT_SENTINEL = "FASTWAM_P8_FORMAL_WORKER_PLACEMENT_AUDIT"
P8_FORMAL_WORKER_PLACEMENT_AUDIT_SCHEMA = "fastwam-p8-formal-worker-placement-audit-v1"

P8_FORMAL_RAY_ENABLED_ENV = "RLINF_P8_FORMAL_RAY_ENABLED"
P8_FORMAL_RAY_NAMESPACE_ENV = "RLINF_P8_FORMAL_RAY_NAMESPACE"
P8_FORMAL_RAY_SESSION_ROOT_ENV = "RLINF_P8_FORMAL_RAY_SESSION_ROOT"
P8_FORMAL_RAY_SESSION_DIR_ENV = "RLINF_P8_FORMAL_RAY_SESSION_DIR"
P8_FORMAL_RAY_PHASE_ENV = "RLINF_P8_FORMAL_RAY_PHASE"
P8_FORMAL_RAY_DRIVER_HOSTNAME_ENV = "RLINF_P8_FORMAL_RAY_DRIVER_HOSTNAME"
P8_FORMAL_RAY_NODE_ID_ENV = "RLINF_P8_FORMAL_RAY_NODE_ID"
P8_FORMAL_RAY_GPU_INVENTORY_ENV = "RLINF_P8_FORMAL_RAY_GPU_INVENTORY"

_P8_FORMAL_LOCAL_RAY_KEYS = {
    "enabled",
    "num_nodes",
    "logical_gpu_count",
    "cuda_visible_devices",
    "namespace_prefix",
    "session_root",
    "phase",
    "worker_placement_audit",
}
_P8_FORMAL_ROLES = {
    "actor": {0: 0, 1: 1},
    "rollout": {0: 2, 1: 3},
    "env": {0: 2, 1: 3},
}
_GPU_UUID_PATTERN = re.compile(r"^GPU-[0-9A-Fa-f-]+$")


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(config, name, default)


def _plain_mapping(config: Any) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        value = OmegaConf.to_container(config, resolve=True)
    else:
        value = config
    if not isinstance(value, Mapping):
        raise TypeError("P8 formal fresh-local-Ray config must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _query_nvidia_gpu_inventory() -> dict[int, str]:
    """Return the physical NVIDIA index-to-UUID inventory."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "P8 formal local-Ray could not query the physical NVIDIA inventory."
        ) from error

    inventory: dict[int, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(
                "P8 formal local-Ray received malformed nvidia-smi inventory."
            )
        try:
            physical_index = int(fields[0])
        except ValueError as error:
            raise RuntimeError(
                "P8 formal local-Ray received a non-integer NVIDIA index."
            ) from error
        gpu_uuid = fields[1]
        if physical_index in inventory or _GPU_UUID_PATTERN.fullmatch(gpu_uuid) is None:
            raise RuntimeError(
                "P8 formal local-Ray received duplicate or invalid NVIDIA identity."
            )
        inventory[physical_index] = gpu_uuid
    if not inventory or len(set(inventory.values())) != len(inventory):
        raise RuntimeError(
            "P8 formal local-Ray requires non-empty, unique NVIDIA UUIDs."
        )
    return inventory


def _inventory_json(inventory: Mapping[int, str]) -> str:
    return json.dumps(
        [
            {"physical_gpu_index": int(index), "physical_gpu_uuid": uuid}
            for index, uuid in sorted(inventory.items())
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def _inventory_from_json(raw: str) -> dict[int, str]:
    try:
        rows = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "P8 formal Ray GPU inventory environment is invalid."
        ) from error
    if not isinstance(rows, list):
        raise RuntimeError("P8 formal Ray GPU inventory must be a list.")
    inventory: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "physical_gpu_index",
            "physical_gpu_uuid",
        }:
            raise RuntimeError("P8 formal Ray GPU inventory row is malformed.")
        index = row["physical_gpu_index"]
        uuid = row["physical_gpu_uuid"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(uuid, str)
            or _GPU_UUID_PATTERN.fullmatch(uuid) is None
            or index in inventory
        ):
            raise RuntimeError("P8 formal Ray GPU inventory row is invalid.")
        inventory[index] = uuid
    if not inventory or len(set(inventory.values())) != len(inventory):
        raise RuntimeError("P8 formal Ray GPU inventory identities are not unique.")
    return inventory


@dataclass(frozen=True)
class P8FormalFreshLocalRayRuntime:
    """Resolved, exact runtime contract for one fresh local Ray invocation."""

    num_nodes: int
    logical_gpu_count: int
    cuda_visible_devices: tuple[int, ...]
    namespace_prefix: str
    session_root: Path
    phase: str
    worker_placement_audit: bool

    @classmethod
    def from_config(cls, config: Any) -> "P8FormalFreshLocalRayRuntime":
        """Resolve and reject any drift in the formal-only Ray mapping."""
        raw = _plain_mapping(config)
        if set(raw) != _P8_FORMAL_LOCAL_RAY_KEYS:
            missing = sorted(_P8_FORMAL_LOCAL_RAY_KEYS - set(raw))
            extra = sorted(set(raw) - _P8_FORMAL_LOCAL_RAY_KEYS)
            raise ValueError(
                "P8 formal fresh-local-Ray keys differ from the frozen contract: "
                f"missing={missing}, extra={extra}."
            )
        if raw["enabled"] is not True:
            raise ValueError("P8 formal fresh-local-Ray must be enabled explicitly.")
        if raw["worker_placement_audit"] is not True:
            raise ValueError(
                "P8 formal fresh-local-Ray worker placement audit must be enabled."
            )
        if isinstance(raw["num_nodes"], bool) or int(raw["num_nodes"]) != 1:
            raise ValueError("P8 formal fresh-local-Ray requires exactly one node.")
        if (
            isinstance(raw["logical_gpu_count"], bool)
            or int(raw["logical_gpu_count"]) != 4
        ):
            raise ValueError(
                "P8 formal fresh-local-Ray requires exactly four logical GPUs."
            )
        visible_raw = str(raw["cuda_visible_devices"])
        if visible_raw != "0,1,2,3":
            raise ValueError(
                "P8 formal fresh-local-Ray requires CUDA_VISIBLE_DEVICES=0,1,2,3."
            )
        namespace_prefix = str(raw["namespace_prefix"])
        if namespace_prefix != "FastWAMP8Formal":
            raise ValueError(
                "P8 formal fresh-local-Ray namespace prefix must be FastWAMP8Formal."
            )
        phase = str(raw["phase"])
        if phase not in {"step_zero_export", "training"}:
            raise ValueError(
                "P8 formal fresh-local-Ray phase must be step_zero_export or training."
            )
        session_root = Path(str(raw["session_root"]))
        if not session_root.is_absolute() or session_root.name != "ray":
            raise ValueError(
                "P8 formal fresh-local-Ray session_root must be an absolute ray directory."
            )
        return cls(
            num_nodes=1,
            logical_gpu_count=4,
            cuda_visible_devices=(0, 1, 2, 3),
            namespace_prefix=namespace_prefix,
            session_root=session_root,
            phase=phase,
            worker_placement_audit=True,
        )

    @property
    def phase_temp_root(self) -> Path:
        """Return the phase-exclusive Ray temporary root."""
        return self.session_root / self.phase

    @property
    def namespace(self) -> str:
        """Return a deterministic namespace unique to output root and phase."""
        identity = hashlib.sha256(
            f"{self.session_root}|{self.phase}".encode()
        ).hexdigest()[:24]
        return f"{self.namespace_prefix}_{identity}"

    def prepare_ray_init_kwargs(
        self,
        *,
        logging_level: str,
        runtime_env: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[int, str]]:
        """Fail closed before starting Ray and return explicit local kwargs."""
        if ray.is_initialized():
            raise RuntimeError(
                "P8 formal fresh-local-Ray refuses a preinitialized Ray runtime."
            )
        if str(os.environ.get("RAY_ADDRESS", "")).strip():
            raise RuntimeError(
                "P8 formal fresh-local-Ray refuses RAY_ADDRESS/shared-cluster input."
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
            raise RuntimeError(
                "P8 formal fresh-local-Ray requires the driver to expose physical "
                "GPUs 0,1,2,3 in that exact order."
            )
        output_root = self.session_root.parent
        if (
            not output_root.is_dir()
            or output_root.is_symlink()
            or output_root.resolve(strict=True) != output_root
        ):
            raise RuntimeError(
                "P8 formal fresh-local-Ray output root is not a stable real directory."
            )
        if self.session_root.is_symlink() or self.phase_temp_root.exists():
            raise RuntimeError(
                "P8 formal fresh-local-Ray refuses a reused or linked phase session root."
            )

        inventory = _query_nvidia_gpu_inventory()
        selected = {
            index: inventory[index]
            for index in self.cuda_visible_devices
            if index in inventory
        }
        if tuple(selected) != self.cuda_visible_devices:
            raise RuntimeError(
                "P8 formal fresh-local-Ray cannot bind every selected physical GPU."
            )

        os.environ[P8_FORMAL_RAY_ENABLED_ENV] = "1"
        os.environ[P8_FORMAL_RAY_NAMESPACE_ENV] = self.namespace
        os.environ[P8_FORMAL_RAY_SESSION_ROOT_ENV] = str(self.session_root)
        os.environ[P8_FORMAL_RAY_PHASE_ENV] = self.phase
        os.environ[P8_FORMAL_RAY_DRIVER_HOSTNAME_ENV] = socket.gethostname()
        os.environ[P8_FORMAL_RAY_GPU_INVENTORY_ENV] = _inventory_json(selected)

        kwargs: dict[str, Any] = {
            "address": "local",
            "include_dashboard": False,
            "logging_level": logging_level,
            "namespace": self.namespace,
            "num_gpus": self.logical_gpu_count,
            "_temp_dir": str(self.phase_temp_root),
        }
        if runtime_env is not None:
            kwargs["runtime_env"] = dict(runtime_env)
        return kwargs, selected

    def complete_ray_startup(
        self,
        *,
        ray_context: Any,
        alive_nodes: list[dict[str, Any]],
        inventory: Mapping[int, str],
    ) -> dict[str, Any]:
        """Verify the new session is a one-node, four-GPU local cluster."""
        if len(alive_nodes) != 1:
            raise RuntimeError(
                "P8 formal fresh-local-Ray requires exactly one alive Ray node."
            )
        node = alive_nodes[0]
        gpu_resource = float(node.get("Resources", {}).get("GPU", 0.0))
        if gpu_resource != float(self.logical_gpu_count):
            raise RuntimeError(
                "P8 formal fresh-local-Ray node does not expose exactly four GPUs."
            )
        runtime_context = ray.get_runtime_context()
        node_id = str(runtime_context.get_node_id())
        if str(node.get("NodeID", "")) != node_id:
            raise RuntimeError(
                "P8 formal fresh-local-Ray driver is not attached to its sole node."
            )
        if str(runtime_context.namespace) != self.namespace:
            raise RuntimeError(
                "P8 formal fresh-local-Ray namespace drifted at startup."
            )

        address_info = getattr(ray_context, "address_info", None)
        if not isinstance(address_info, Mapping):
            raise RuntimeError("P8 formal fresh-local-Ray did not expose session info.")
        session_dir = Path(str(address_info.get("session_dir", "")))
        try:
            session_dir.relative_to(self.phase_temp_root)
        except ValueError as error:
            raise RuntimeError(
                "P8 formal Ray session escaped its phase-exclusive temp root."
            ) from error
        if not session_dir.is_dir() or session_dir.is_symlink():
            raise RuntimeError(
                "P8 formal Ray session directory is not a real directory."
            )

        os.environ[P8_FORMAL_RAY_SESSION_DIR_ENV] = str(session_dir)
        os.environ[P8_FORMAL_RAY_NODE_ID_ENV] = node_id
        payload = {
            "schema": P8_FORMAL_RAY_AUDIT_SCHEMA,
            "status": "PASS",
            "address_mode": "local",
            "driver_pid": os.getpid(),
            "driver_hostname": socket.gethostname(),
            "phase": self.phase,
            "namespace": self.namespace,
            "session_root": str(self.session_root),
            "session_dir": str(session_dir),
            "node_id": node_id,
            "alive_node_count": 1,
            "logical_gpu_count": self.logical_gpu_count,
            "cuda_visible_devices": list(self.cuda_visible_devices),
            "physical_gpu_inventory": [
                {
                    "physical_gpu_index": index,
                    "physical_gpu_uuid": uuid,
                }
                for index, uuid in sorted(inventory.items())
            ],
        }
        print(
            f"{P8_FORMAL_RAY_AUDIT_SENTINEL} " + json.dumps(payload, sort_keys=True),
            flush=True,
        )
        return payload


def resolve_p8_formal_fresh_local_ray_config(cfg: Any) -> Any | None:
    """Return the formal local-Ray config while keeping all other paths unchanged."""
    runner = _config_value(cfg, "runner", {})
    cluster = _config_value(cfg, "cluster", {})
    formal_endpoint = bool(_config_value(runner, "p8_formal_stage2_endpoint", False))
    local_config = _config_value(cluster, "p8_formal_fresh_local_ray", None)
    enabled = bool(_config_value(local_config, "enabled", False))
    if formal_endpoint != enabled:
        raise ValueError(
            "P8 formal Stage2 and fresh local Ray must be enabled together."
        )
    if not formal_endpoint:
        return None
    P8FormalFreshLocalRayRuntime.from_config(local_config)
    return local_config


def emit_p8_formal_worker_placement_audit(
    cfg: Any,
    worker: Any,
    *,
    role: str,
) -> dict[str, Any] | None:
    """Verify and emit one typed physical placement row per formal worker."""
    local_config = resolve_p8_formal_fresh_local_ray_config(cfg)
    if local_config is None:
        return None
    runtime = P8FormalFreshLocalRayRuntime.from_config(local_config)
    if role not in _P8_FORMAL_ROLES:
        raise ValueError(f"Unsupported P8 formal worker role: {role!r}.")
    if runtime.phase == "step_zero_export" and role != "actor":
        raise RuntimeError("P8 formal step-zero Ray may launch only actor workers.")

    expected_group = str(_config_value(_config_value(cfg, role, {}), "group_name", ""))
    group_name = str(getattr(worker, "_group_name", ""))
    rank = int(getattr(worker, "_rank", -1))
    world_size = int(getattr(worker, "_world_size", -1))
    local_accelerator_rank = int(getattr(worker, "_local_accelerator_rank", -1))
    expected_accelerator_rank = _P8_FORMAL_ROLES[role].get(rank)
    if (
        group_name != expected_group
        or world_size != 2
        or expected_accelerator_rank is None
        or local_accelerator_rank != expected_accelerator_rank
        or int(getattr(worker, "_cluster_node_rank", -1)) != 0
        or int(getattr(worker, "_node_local_rank", -1)) != rank
    ):
        raise RuntimeError(
            "P8 formal worker group/rank/logical placement differs from its contract."
        )

    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = [item.strip() for item in visible_raw.split(",") if item.strip()]
    physical_index = runtime.cuda_visible_devices[local_accelerator_rank]
    if visible_devices != [str(physical_index)]:
        raise RuntimeError(
            "P8 formal worker CUDA visibility differs from its single physical GPU."
        )

    expected_inventory = _inventory_from_json(
        os.environ.get(P8_FORMAL_RAY_GPU_INVENTORY_ENV, "")
    )
    current_inventory = _query_nvidia_gpu_inventory()
    selected_current = {
        index: current_inventory.get(index) for index in runtime.cuda_visible_devices
    }
    if selected_current != expected_inventory:
        raise RuntimeError("P8 formal worker observed physical GPU UUID drift.")
    physical_uuid = expected_inventory.get(physical_index)
    if physical_uuid is None:
        raise RuntimeError("P8 formal worker physical GPU identity is missing.")

    runtime_context = ray.get_runtime_context()
    node_id = str(runtime_context.get_node_id())
    namespace = str(runtime_context.namespace)
    hostname = socket.gethostname()
    session_dir = os.environ.get(P8_FORMAL_RAY_SESSION_DIR_ENV, "")
    if (
        os.environ.get(P8_FORMAL_RAY_ENABLED_ENV) != "1"
        or os.environ.get(P8_FORMAL_RAY_PHASE_ENV) != runtime.phase
        or os.environ.get(P8_FORMAL_RAY_NAMESPACE_ENV) != runtime.namespace
        or namespace != runtime.namespace
        or os.environ.get(P8_FORMAL_RAY_NODE_ID_ENV) != node_id
        or os.environ.get(P8_FORMAL_RAY_DRIVER_HOSTNAME_ENV) != hostname
        or os.environ.get(P8_FORMAL_RAY_SESSION_ROOT_ENV) != str(runtime.session_root)
    ):
        raise RuntimeError("P8 formal worker escaped its fresh local Ray session.")
    try:
        Path(session_dir).relative_to(runtime.phase_temp_root)
    except ValueError as error:
        raise RuntimeError(
            "P8 formal worker session directory escaped its phase root."
        ) from error
    if not Path(session_dir).is_dir() or Path(session_dir).is_symlink():
        raise RuntimeError("P8 formal worker session directory is invalid.")

    payload = {
        "schema": P8_FORMAL_WORKER_PLACEMENT_AUDIT_SCHEMA,
        "status": "PASS",
        "phase": runtime.phase,
        "role": role,
        "group_name": group_name,
        "worker_rank": rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "hostname": hostname,
        "ray_node_id": node_id,
        "ray_namespace": namespace,
        "ray_session_dir": session_dir,
        "cluster_node_rank": 0,
        "node_local_rank": rank,
        "logical_accelerator_rank": local_accelerator_rank,
        "cuda_visible_devices": visible_devices,
        "physical_gpu_index": physical_index,
        "physical_gpu_uuid": physical_uuid,
    }
    print(
        f"{P8_FORMAL_WORKER_PLACEMENT_AUDIT_SENTINEL} "
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return payload
