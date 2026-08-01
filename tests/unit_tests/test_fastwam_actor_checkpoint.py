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

"""Focused tests for the adaptive FastWAM actor checkpoint I/O path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rlinf.workers.actor import fsdp_actor_worker as worker_module
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class _Policy:
    def __init__(self) -> None:
        self.actor_version = 0
        self.weight = torch.tensor([1.0])

    def set_global_step(self, step: int) -> None:
        self.actor_version = int(step)

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "fastwam-adaptive-policy-v1",
            "actor_version": self.actor_version,
            "weight": self.weight.clone(),
        }

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        self.actor_version = int(state["actor_version"])
        self.weight.copy_(state["weight"])


class _Stateful:
    def __init__(self, value: float) -> None:
        self.value = torch.tensor([value])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.value.copy_(state["value"])


_FAKE_MISSING = object()


class _FakeFSDP:
    def __init__(self, initial, *children) -> None:
        self.children = list(children)
        if initial is not _FAKE_MISSING:
            self._is_root = initial

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()


class _RootMutatingPolicy(_Policy):
    def __init__(self) -> None:
        super().__init__()
        self.missing = _FakeFSDP(_FAKE_MISSING)
        self.nested = _FakeFSDP(False, self.missing)
        self.root = _FakeFSDP(None, self.nested)

    def modules(self):
        yield self
        yield from self.root.modules()

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        super().load_trainable_state_dict(state)
        self.root._is_root = True
        self.nested._is_root = True
        self.missing._is_root = True


def _checkpoint_worker() -> Any:
    class CheckpointWorker:
        _checkpoint_cpu_clone = staticmethod(EmbodiedFSDPActor._checkpoint_cpu_clone)
        _fastwam_policy_module = EmbodiedFSDPActor._fastwam_policy_module
        save_checkpoint = EmbodiedFSDPActor.save_checkpoint
        load_checkpoint = EmbodiedFSDPActor.load_checkpoint

        def _fastwam_checkpoint_contract(self) -> dict[str, str]:
            return {"kind": "unit"}

    worker = CheckpointWorker()
    worker.model = _Policy()
    worker.cfg = SimpleNamespace(
        actor=SimpleNamespace(
            model=SimpleNamespace(
                model_type="fastwam_adaptive",
                actor_checkpoint_sha256="a" * 64,
                critic=SimpleNamespace(backbone_checkpoint_sha256="b" * 64),
            )
        )
    )
    worker.optimizer = _Stateful(2.0)
    worker.lr_scheduler = _Stateful(3.0)
    worker.grad_scaler = _Stateful(4.0)
    worker.optimizer_steps = 1
    worker.version = 0
    worker._rank = 0
    worker._world_size = 1
    worker.is_weight_offloaded = False
    worker.is_optimizer_offloaded = False
    return worker


def test_fastwam_actor_checkpoint_rank_file_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng_state = {"cpu": torch.tensor([7], dtype=torch.uint8)}
    restored_rng = []
    monkeypatch.setattr(worker_module, "get_rng_state", lambda: rng_state)
    monkeypatch.setattr(
        worker_module,
        "set_rng_state",
        lambda state: restored_rng.append(state),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    worker = _checkpoint_worker()
    checkpoint_dir = tmp_path / "actor"

    worker.save_checkpoint(str(checkpoint_dir), step=7)

    checkpoint_path = checkpoint_dir / "rank_0.pt"
    assert checkpoint_path.is_file()
    assert not (checkpoint_dir / "rank_0.pt.tmp").exists()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "fastwam-adaptive-rl-checkpoint-v1"
    assert payload["step"] == 7
    assert payload["policy"]["actor_version"] == 7
    assert payload["optimizer_steps"] == 1

    worker.model.weight.fill_(9.0)
    worker.model.set_global_step(99)
    worker.optimizer.value.fill_(9.0)
    worker.lr_scheduler.value.fill_(9.0)
    worker.grad_scaler.value.fill_(9.0)
    worker.optimizer_steps = 99

    assert worker.load_checkpoint(str(checkpoint_dir)) == 7
    assert worker.version == 7
    assert worker.model.actor_version == 7
    assert torch.equal(worker.model.weight, torch.tensor([1.0]))
    assert torch.equal(worker.optimizer.value, torch.tensor([2.0]))
    assert torch.equal(worker.lr_scheduler.value, torch.tensor([3.0]))
    assert torch.equal(worker.grad_scaler.value, torch.tensor([4.0]))
    assert worker.optimizer_steps == 1
    assert len(restored_rng) == 1
    assert torch.equal(restored_rng[0]["cpu"], rng_state["cpu"])


def test_fastwam_actor_checkpoint_load_restores_fsdp_lazy_root_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_rng_state",
        lambda: {"cpu": torch.tensor([7], dtype=torch.uint8)},
    )
    monkeypatch.setattr(worker_module, "set_rng_state", lambda _state: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    original_snapshot = worker_module._snapshot_fastwam_fsdp_lazy_root_state
    monkeypatch.setattr(
        worker_module,
        "_snapshot_fastwam_fsdp_lazy_root_state",
        lambda model: original_snapshot(model, fsdp_cls=_FakeFSDP),
    )
    worker = _checkpoint_worker()
    policy = _RootMutatingPolicy()
    worker.model = policy
    checkpoint_dir = tmp_path / "actor"
    worker.save_checkpoint(str(checkpoint_dir), step=7)

    assert worker.load_checkpoint(str(checkpoint_dir)) == 7

    assert policy.root._is_root is None
    assert policy.nested._is_root is False
    assert not hasattr(policy.missing, "_is_root")


def test_fastwam_actor_checkpoint_failed_save_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_rng_state",
        lambda: {"cpu": torch.tensor([7], dtype=torch.uint8)},
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    def fail_after_partial_write(_payload: Any, path: str) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(worker_module.torch, "save", fail_after_partial_write)
    worker = _checkpoint_worker()
    checkpoint_dir = tmp_path / "actor"

    with pytest.raises(RuntimeError, match="simulated save failure"):
        worker.save_checkpoint(str(checkpoint_dir), step=7)

    assert not (checkpoint_dir / "rank_0.pt").exists()
    assert not (checkpoint_dir / "rank_0.pt.tmp").exists()


def test_fastwam_actor_checkpoint_rejects_extra_frozen_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng_state = {"cpu": torch.tensor([7], dtype=torch.uint8)}
    monkeypatch.setattr(worker_module, "get_rng_state", lambda: rng_state)
    monkeypatch.setattr(worker_module, "set_rng_state", lambda _state: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    worker = _checkpoint_worker()
    checkpoint_dir = tmp_path / "actor"
    worker.save_checkpoint(str(checkpoint_dir), step=7)
    checkpoint_path = checkpoint_dir / "rank_0.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["frozen_backbone"] = {"weight": torch.ones(1)}
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="keys changed"):
        worker.load_checkpoint(str(checkpoint_dir))


def test_collective_checkpoint_failure_reaches_healthy_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather_failures(output, local_message) -> None:
        output[:] = [local_message, "ValueError: corrupt rank checkpoint"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_failures)
    with pytest.raises(RuntimeError, match="failed collectively.*corrupt rank"):
        worker_module._raise_fastwam_collective_checkpoint_error(
            None,
            context="actor checkpoint load",
        )


def test_collective_checkpoint_success_is_one_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    calls = []

    def gather_success(output, local_message) -> None:
        calls.append(local_message)
        output[:] = [None, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_success)
    worker_module._raise_fastwam_collective_checkpoint_error(
        None,
        context="actor checkpoint save",
    )
    assert calls == [None]
