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

"""CPU-only contracts for P8's isolated formal Ray runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.scheduler.cluster import p8_formal


def _local_ray_config(output_root: Path, *, phase: str = "training") -> dict:
    return {
        "enabled": True,
        "num_nodes": 1,
        "logical_gpu_count": 4,
        "cuda_visible_devices": "0,1,2,3",
        "namespace_prefix": "FastWAMP8Formal",
        "session_root": str(output_root / "ray"),
        "phase": phase,
        "worker_placement_audit": True,
    }


def _gpu_inventory() -> dict[int, str]:
    return {
        index: f"GPU-00000000-0000-0000-0000-00000000000{index}" for index in range(4)
    }


def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    environment = dict(os.environ)
    monkeypatch.setattr(p8_formal.os, "environ", environment)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    environment.pop("RAY_ADDRESS", None)
    return environment


def test_formal_ray_prepares_explicit_fresh_local_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "formal-output"
    output_root.mkdir()
    environment = _isolate_environment(monkeypatch)
    inventory = _gpu_inventory()
    monkeypatch.setattr(p8_formal.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(
        p8_formal,
        "_query_nvidia_gpu_inventory",
        lambda: inventory,
    )
    monkeypatch.setattr(p8_formal.socket, "gethostname", lambda: "formal-host")

    runtime = p8_formal.P8FormalFreshLocalRayRuntime.from_config(
        _local_ray_config(output_root)
    )
    kwargs, selected = runtime.prepare_ray_init_kwargs(
        logging_level="INFO",
        runtime_env={"py_modules": ["rlinf"]},
    )

    assert kwargs == {
        "address": "local",
        "include_dashboard": False,
        "logging_level": "INFO",
        "namespace": runtime.namespace,
        "num_gpus": 4,
        "_temp_dir": str(output_root / "ray/training"),
        "runtime_env": {"py_modules": ["rlinf"]},
    }
    assert selected == inventory
    assert not runtime.phase_temp_root.exists()
    assert environment[p8_formal.P8_FORMAL_RAY_NAMESPACE_ENV] == runtime.namespace

    session_dir = runtime.phase_temp_root / "session_2026"
    session_dir.mkdir(parents=True)
    runtime_context = SimpleNamespace(
        get_node_id=lambda: "node-id",
        namespace=runtime.namespace,
    )
    monkeypatch.setattr(
        p8_formal.ray,
        "get_runtime_context",
        lambda: runtime_context,
    )
    payload = runtime.complete_ray_startup(
        ray_context=SimpleNamespace(address_info={"session_dir": str(session_dir)}),
        alive_nodes=[{"Alive": True, "NodeID": "node-id", "Resources": {"GPU": 4.0}}],
        inventory=selected,
    )

    assert payload["status"] == "PASS"
    assert payload["address_mode"] == "local"
    assert payload["alive_node_count"] == 1
    assert payload["logical_gpu_count"] == 4
    assert environment[p8_formal.P8_FORMAL_RAY_NODE_ID_ENV] == "node-id"
    sentinel, encoded = capsys.readouterr().out.strip().split(" ", 1)
    assert sentinel == p8_formal.P8_FORMAL_RAY_AUDIT_SENTINEL
    assert json.loads(encoded) == payload


def test_formal_ray_refuses_shared_address_and_reused_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "formal-output"
    output_root.mkdir()
    environment = _isolate_environment(monkeypatch)
    monkeypatch.setattr(p8_formal.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(
        p8_formal,
        "_query_nvidia_gpu_inventory",
        _gpu_inventory,
    )
    runtime = p8_formal.P8FormalFreshLocalRayRuntime.from_config(
        _local_ray_config(output_root)
    )

    environment["RAY_ADDRESS"] = "auto"
    with pytest.raises(RuntimeError, match="RAY_ADDRESS/shared-cluster"):
        runtime.prepare_ray_init_kwargs(logging_level="INFO", runtime_env=None)

    environment.pop("RAY_ADDRESS")
    runtime.phase_temp_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="reused or linked"):
        runtime.prepare_ray_init_kwargs(logging_level="INFO", runtime_env=None)


def test_formal_ray_flag_is_exactly_bound_to_the_formal_endpoint(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "runner": {"p8_formal_stage2_endpoint": False},
            "cluster": {},
        }
    )
    assert p8_formal.resolve_p8_formal_fresh_local_ray_config(cfg) is None

    cfg.cluster.p8_formal_fresh_local_ray = _local_ray_config(tmp_path)
    with pytest.raises(ValueError, match="enabled together"):
        p8_formal.resolve_p8_formal_fresh_local_ray_config(cfg)

    cfg.runner.p8_formal_stage2_endpoint = True
    resolved = p8_formal.resolve_p8_formal_fresh_local_ray_config(cfg)
    assert resolved is cfg.cluster.p8_formal_fresh_local_ray


def test_formal_worker_emits_typed_physical_uuid_placement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "formal-output"
    session_dir = output_root / "ray/training/session_2026"
    session_dir.mkdir(parents=True)
    runtime_config = _local_ray_config(output_root)
    runtime = p8_formal.P8FormalFreshLocalRayRuntime.from_config(runtime_config)
    inventory = _gpu_inventory()
    environment = _isolate_environment(monkeypatch)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "2",
            p8_formal.P8_FORMAL_RAY_ENABLED_ENV: "1",
            p8_formal.P8_FORMAL_RAY_NAMESPACE_ENV: runtime.namespace,
            p8_formal.P8_FORMAL_RAY_SESSION_ROOT_ENV: str(runtime.session_root),
            p8_formal.P8_FORMAL_RAY_SESSION_DIR_ENV: str(session_dir),
            p8_formal.P8_FORMAL_RAY_PHASE_ENV: "training",
            p8_formal.P8_FORMAL_RAY_DRIVER_HOSTNAME_ENV: "formal-host",
            p8_formal.P8_FORMAL_RAY_NODE_ID_ENV: "node-id",
            p8_formal.P8_FORMAL_RAY_GPU_INVENTORY_ENV: (
                p8_formal._inventory_json(inventory)
            ),
        }
    )
    monkeypatch.setattr(
        p8_formal,
        "_query_nvidia_gpu_inventory",
        lambda: inventory,
    )
    monkeypatch.setattr(p8_formal.socket, "gethostname", lambda: "formal-host")
    monkeypatch.setattr(
        p8_formal.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            get_node_id=lambda: "node-id",
            namespace=runtime.namespace,
        ),
    )
    cfg = OmegaConf.create(
        {
            "runner": {"p8_formal_stage2_endpoint": True},
            "cluster": {"p8_formal_fresh_local_ray": runtime_config},
            "actor": {"group_name": "ActorGroup"},
            "rollout": {"group_name": "RolloutGroup"},
            "env": {"group_name": "EnvGroup"},
        }
    )
    worker = SimpleNamespace(
        _group_name="RolloutGroup",
        _rank=0,
        _world_size=2,
        _local_accelerator_rank=2,
        _cluster_node_rank=0,
        _node_local_rank=0,
    )

    payload = p8_formal.emit_p8_formal_worker_placement_audit(
        cfg,
        worker,
        role="rollout",
    )

    assert payload is not None
    assert payload["status"] == "PASS"
    assert payload["role"] == "rollout"
    assert payload["logical_accelerator_rank"] == 2
    assert payload["physical_gpu_index"] == 2
    assert payload["physical_gpu_uuid"] == inventory[2]
    sentinel, encoded = capsys.readouterr().out.strip().split(" ", 1)
    assert sentinel == p8_formal.P8_FORMAL_WORKER_PLACEMENT_AUDIT_SENTINEL
    assert json.loads(encoded) == payload

    environment["CUDA_VISIBLE_DEVICES"] = "3"
    with pytest.raises(RuntimeError, match="CUDA visibility"):
        p8_formal.emit_p8_formal_worker_placement_audit(
            cfg,
            worker,
            role="rollout",
        )
